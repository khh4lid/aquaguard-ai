"""
Model wrappers — the ONLY files that touch the models.

Backend is chosen PER MODEL in config (.env), so you can run the hot-path
person detector on the Hailo-8 (fast, every frame) while keeping the rare pool
segmenter on the CPU (.pt). That avoids decoding segmentation masks off a HEF
(painful) and still gives high FPS, because the per-frame model is on Hailo.

Return contracts (identical no matter the backend — the pipeline never cares):
  PoolSegmenter.detect(frame)  -> np.int32 polygon (N, 2) | None
  PersonDetector.detect(frame) -> list[(x1, y1, x2, y2, cls, conf)]
"""
import numpy as np
import cv2
from . import config


# ─────────────────────────────────────────────────────────────────────────────
# POOL SEGMENTER  (default backend: ultralytics .pt on CPU, runs every ~30 frames)
# ─────────────────────────────────────────────────────────────────────────────
class PoolSegmenter:
    def __init__(self):
        self.backend = config.SEG_BACKEND
        if self.backend == "hailo":
            # Seg-on-Hailo needs host-side mask decode — left as a hook. Recommended
            # default is 'ultralytics' (CPU) since this model runs only occasionally.
            raise NotImplementedError(
                "Segmentation on Hailo needs custom mask decode. Set SEG_BACKEND=ultralytics "
                "(recommended) — the pool model runs rarely, so CPU is fine.")
        from ultralytics import YOLO
        self._m = YOLO(config.SEG_WEIGHTS)

    def detect(self, frame):
        res = self._m(frame, conf=config.CONF_SEG, imgsz=config.IMG_SIZE, verbose=False)[0]
        if res.masks is None:
            return None
        best_poly, best_area = None, 0.0
        for mask_xy, cls in zip(res.masks.xy, res.boxes.cls):
            if int(cls) != 0:          # single 'pool' class = id 0
                continue
            poly = np.asarray(mask_xy, dtype=np.int32)
            if len(poly) < 3:
                continue
            area = cv2.contourArea(poly)
            if area > best_area:
                best_area, best_poly = area, poly
        return best_poly


# ─────────────────────────────────────────────────────────────────────────────
# PERSON DETECTOR  (default backend: hailo .hef on the Hailo-8, runs every frame)
# ─────────────────────────────────────────────────────────────────────────────
class PersonDetector:
    def __init__(self):
        self.backend = config.PERSON_BACKEND
        if self.backend == "hailo":
            self._impl = _HailoDetector(config.PERSON_HEF, n_classes=2,
                                        conf=config.CONF_PERSON)
        else:
            from ultralytics import YOLO
            self._impl = None
            self._y = YOLO(config.PERSON_WEIGHTS)

    def detect(self, frame):
        if self._impl is not None:
            return self._impl.detect(frame)
        res = self._y(frame, conf=config.CONF_PERSON, imgsz=config.IMG_SIZE, verbose=False)[0]
        out = []
        if res.boxes is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                out.append((x1, y1, x2, y2, int(b.cls), float(b.conf)))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# HAILO BACKEND
# ─────────────────────────────────────────────────────────────────────────────
class _HailoDetector:
    """
    Runs a DETECTION HEF on the Hailo-8 via HailoRT.

    ASSUMPTION: the HEF was compiled with NMS post-processing baked in (the Hailo
    Model Zoo default when you compile `yolov9c` with the shorthand). The output
    is then already-decoded detections: per class, a list of
    [y_min, x_min, y_max, x_max, score] in NORMALISED [0..1] coords.

    >>> VERIFY YOUR HEF ONCE:  hailortcli parse-hef child_yolov9c.hef
        If the output layout differs from the above, `_to_boxes()` is the ONLY
        method to adjust — send me the parse-hef output and I'll finalise it.
    """
    def __init__(self, hef_path, n_classes=2, conf=0.40):
        from hailo_platform import (
            HEF, VDevice, ConfigureParams, HailoStreamInterface,
            InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType,
        )
        self._InferVStreams = InferVStreams
        self.conf = conf
        self.n_classes = n_classes

        self.hef = HEF(hef_path)
        self.device = VDevice()
        cfg = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.ng = self.device.configure(self.hef, cfg)[0]
        self.ng_params = self.ng.create_params()

        self.in_info = self.hef.get_input_vstream_infos()[0]
        self.out_name = self.hef.get_output_vstream_infos()[0].name
        self.in_h, self.in_w = self.in_info.shape[0], self.in_info.shape[1]
        # UINT8 input lets the on-chip normalisation (added at compile) do the scaling.
        self.in_params = InputVStreamParams.make(self.ng, format_type=FormatType.UINT8)
        self.out_params = OutputVStreamParams.make(self.ng)
        print(f"[hailo] loaded {hef_path}  in={self.in_w}x{self.in_h}")

    def detect(self, frame):
        h0, w0 = frame.shape[:2]
        img = cv2.resize(frame, (self.in_w, self.in_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[None].astype(np.uint8)  # (1,H,W,3)
        with self._InferVStreams(self.ng, self.in_params, self.out_params) as pipe:
            with self.ng.activate(self.ng_params):
                results = pipe.infer({self.in_info.name: img})
        return self._to_boxes(results[self.out_name], w0, h0)

    def _to_boxes(self, raw, w0, h0):
        """raw[class_id] -> ndarray of [ymin, xmin, ymax, xmax, score] (normalised)."""
        out = []
        # raw may come wrapped in a batch dim; unwrap to the per-class list.
        per_class = raw[0] if isinstance(raw, (list, tuple)) or getattr(raw, "ndim", 0) else raw
        for cls_id in range(self.n_classes):
            dets = per_class[cls_id] if cls_id < len(per_class) else []
            for d in dets:
                ymin, xmin, ymax, xmax, score = float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])
                if score < self.conf:
                    continue
                out.append((int(xmin * w0), int(ymin * h0),
                            int(xmax * w0), int(ymax * h0), cls_id, score))
        return out
