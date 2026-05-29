#!/usr/bin/env python3
"""
AquaGuard AI — Raspberry Pi 5 entry point.

Run windowed (see the HUD):   HEADLESS=false python deploy/main.py
Run as a service (no display): handled by deploy/aquaguard.service

Keys (windowed mode):  q = quit   r = force re-detect pool zone
"""
import sys
import time
import signal
from pathlib import Path
from datetime import datetime

import cv2

# Make the repo root importable so `from src...` works no matter the CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.camera import Camera
from src.detectors import PoolSegmenter, PersonDetector
from src.pipeline import AlertPipeline
from src.alerts import AlertManager
from src.watchdog import Heartbeat
from src import hud


def main():
    print("AquaGuard AI starting…")
    cam = Camera()
    seg = PoolSegmenter()
    person = PersonDetector()
    pipeline = AlertPipeline()
    alerts = AlertManager()
    heartbeat = Heartbeat(alerts, config.HEARTBEAT_EVERY, config.HEARTBEAT_ENABLED)

    running = {"on": True}

    def _stop(*_):
        running["on"] = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)   # systemd stop

    pool_poly = None
    frame_idx = 0
    fps = 0.0
    t_prev = time.time()

    while running["on"]:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.05)
            continue
        frame_idx += 1
        if config.PROCESS_EVERY > 1 and frame_idx % config.PROCESS_EVERY != 0:
            continue

        # Refresh the pool polygon on frame 1, then every SEG_UPDATE_EVERY frames.
        if frame_idx == 1 or frame_idx % config.SEG_UPDATE_EVERY == 1:
            new_poly = seg.detect(frame)
            if new_poly is not None:
                pool_poly = new_poly

        detections = person.detect(frame)
        alert, fire, details = pipeline.step(detections, pool_poly)
        heartbeat.tick()

        # Send (cooldown already enforced inside the pipeline)
        if fire in ("EMERGENCY", "DANGER"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            evidence = str(config.EVIDENCE_DIR / f"{fire}_{ts}.jpg")
            cv2.imwrite(evidence, frame)
            (alerts.emergency if fire == "EMERGENCY" else alerts.danger)(evidence)
            print(f"[{fire}] @ {ts}  (evidence: {evidence})")

        # Smoothed FPS
        now = time.time()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            inst = 1.0 / dt
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

        if not config.HEADLESS:
            cv2.imshow("AquaGuard AI", hud.draw(frame, details, alert, fps))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                pp = seg.detect(frame)
                if pp is not None:
                    pool_poly = pp

    print("Shutting down…")
    cam.release()
    alerts.cleanup()
    if not config.HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
