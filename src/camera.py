"""
Unified camera source. Supports the Pi CSI camera (Picamera2, e.g. the NoIR
module), a USB webcam, or a video file — all behind one .read() interface so
main.py never cares where frames come from.
"""
import cv2
from . import config


class Camera:
    def __init__(self, source=None, width=None, height=None):
        self.source = source or config.CAMERA_SOURCE
        self.width = width or config.FRAME_W
        self.height = height or config.FRAME_H
        self._picam = None
        self._cap = None
        self._open()

    def _open(self):
        if self.source == "picamera":
            # Picamera2 is the only supported CSI stack on Pi 5 (legacy stack is gone).
            from picamera2 import Picamera2
            self._picam = Picamera2()
            cfg = self._picam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self._picam.configure(cfg)
            self._picam.start()
        else:
            # 'usb' -> index 0; otherwise treat source as a file path.
            src = 0 if self.source == "usb" else self.source
            self._cap = cv2.VideoCapture(src)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open camera source: {self.source!r}")

    def read(self):
        """Return (ok, frame_bgr)."""
        if self._picam is not None:
            frame = self._picam.capture_array()
            # Picamera2 'RGB888' delivers BGR-ordered bytes, which is exactly
            # what OpenCV and ultralytics expect — no colour conversion needed.
            return True, frame
        return self._cap.read()

    def release(self):
        if self._picam is not None:
            self._picam.stop()
        if self._cap is not None:
            self._cap.release()
