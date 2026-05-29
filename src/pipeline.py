"""
Two-tier alert state machine. Pure decision logic — no drawing, no network.

DANGER:    child in frame AND no adult AND pool visible.
EMERGENCY: child foot-point inside the pool polygon AND no adult, sustained
           over CONFIRM_FRAMES consecutive frames.

step() returns (alert, fire, details):
  alert  - current visual state for the HUD ('EMERGENCY' | 'DANGER' | None)
  fire   - alert to actually SEND this frame after cooldown ('...' | None)
           (separating these means the banner can stay on-screen while we
            avoid re-spamming the parent's WhatsApp)
"""
import time
import cv2
from . import config


def _point_in_poly(x, y, poly):
    if poly is None or len(poly) < 3:
        return False
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


class AlertPipeline:
    def __init__(self):
        self._emerg_streak = 0
        self._last_danger_t = 0.0
        self._last_emerg_t = 0.0

    def step(self, detections, pool_poly):
        pool_found = pool_poly is not None and len(pool_poly) >= 3
        adult_visible = False
        children_frame = 0
        children_in_pool = 0
        analysed = []

        for (x1, y1, x2, y2, cls, conf) in detections:
            is_child = cls == config.CHILD_IDX
            if not is_child:
                adult_visible = True
            in_pool = False
            if is_child:
                children_frame += 1
                # foot-point = bottom-centre of the box (where the child stands)
                foot_x, foot_y = (x1 + x2) // 2, y2
                in_pool = pool_found and _point_in_poly(foot_x, foot_y, pool_poly)
                if in_pool:
                    children_in_pool += 1
            analysed.append((x1, y1, x2, y2, cls, conf, in_pool))

        # Hysteresis: build the streak up, decay it down. A brief miss-detection
        # won't reset the alarm, but a genuine clear (child leaves) ramps it down.
        if children_in_pool > 0 and not adult_visible:
            self._emerg_streak += 1
        else:
            self._emerg_streak = max(0, self._emerg_streak - 1)

        now = time.time()
        alert = None
        fire = None

        if self._emerg_streak >= config.CONFIRM_FRAMES:
            alert = "EMERGENCY"
            if now - self._last_emerg_t > config.COOLDOWN_EMERGENCY:
                self._last_emerg_t = now
                fire = "EMERGENCY"
        elif children_frame > 0 and pool_found and not adult_visible:
            alert = "DANGER"
            if now - self._last_danger_t > config.COOLDOWN_DANGER:
                self._last_danger_t = now
                fire = "DANGER"

        details = {
            "pool_found": pool_found,
            "pool_poly": pool_poly,
            "children_frame": children_frame,
            "children_in_pool": children_in_pool,
            "adult_visible": adult_visible,
            "confirm": self._emerg_streak,
            "boxes": analysed,
        }
        return alert, fire, details
