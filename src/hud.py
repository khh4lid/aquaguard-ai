"""
OpenCV HUD overlay — used only when HEADLESS=false (i.e. a display is attached).
Draws the pool mask, person boxes, foot-points, the alert banner and a status bar.
"""
import cv2
from . import config

# BGR colours
C_POOL = (0, 229, 255)        # teal
C_CHILD = (0, 165, 255)       # orange
C_ADULT = (178, 145, 8)       # blue
C_CHILD_POOL = (0, 0, 220)    # red — child inside pool


def draw(frame, details, alert, fps):
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Pool polygon: translucent fill + solid border
    poly = details.get("pool_poly")
    if poly is not None and len(poly) >= 3:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly], (0, 100, 60))
        cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0, canvas)
        cv2.polylines(canvas, [poly], True, C_POOL, 2, cv2.LINE_AA)

    # Person boxes + foot-points
    for (x1, y1, x2, y2, cls, conf, in_pool) in details["boxes"]:
        color = C_CHILD_POOL if in_pool else (C_CHILD if cls == config.CHILD_IDX else C_ADULT)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{config.CLASS_NAMES[cls]} {conf:.2f}",
                    (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
        cv2.circle(canvas, ((x1 + x2) // 2, y2), 4, color, -1)  # foot-point

    # Top bar + FPS (colour-coded against an 8 FPS "usable" line)
    cv2.rectangle(canvas, (0, 0), (w, 40), (15, 15, 15), -1)
    cv2.putText(canvas, "AquaGuard AI", (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, C_POOL, 2, cv2.LINE_AA)
    fps_col = (0, 255, 100) if fps >= 8 else (0, 165, 255) if fps >= 3 else (0, 0, 220)
    cv2.putText(canvas, f"{fps:4.1f} FPS", (w - 130, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, fps_col, 2)

    # Alert banner
    if alert == "EMERGENCY":
        cv2.rectangle(canvas, (0, 42), (w, 92), (0, 0, 140), -1)
        cv2.putText(canvas, "EMERGENCY: CHILD IN POOL", (12, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    elif alert == "DANGER":
        cv2.rectangle(canvas, (0, 42), (w, 86), (0, 120, 200), -1)
        cv2.putText(canvas, "DANGER: child near pool, no adult", (12, 73),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Status bar
    cv2.rectangle(canvas, (0, h - 30), (w, h), (15, 15, 15), -1)
    pool_txt = "Pool: DETECTED" if details["pool_found"] else "Pool: searching"
    adult_txt = "Adult: YES" if details["adult_visible"] else "Adult: NO"
    status = (f"{pool_txt} | Child:{details['children_frame']} "
              f"(in pool:{details['children_in_pool']}) | {adult_txt} "
              f"| confirm {details['confirm']}/{config.CONFIRM_FRAMES}")
    cv2.putText(canvas, status, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas
