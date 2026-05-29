"""
Central configuration. All tunables live here and are overridable via .env,
so the device can be re-tuned in the field without editing code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # read .env in the repo root if present

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
EVIDENCE_DIR = BASE_DIR / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

# ── Model weights (populated by scripts/download_models.py) ───────────────
SEG_WEIGHTS = os.getenv("SEG_WEIGHTS", str(MODELS_DIR / "pool_yolov8m_seg.pt"))
PERSON_WEIGHTS = os.getenv("PERSON_WEIGHTS", str(MODELS_DIR / "child_yolov9c.pt"))
SEG_HEF = os.getenv("SEG_HEF", str(MODELS_DIR / "pool_yolov8m_seg.hef"))
PERSON_HEF = os.getenv("PERSON_HEF", str(MODELS_DIR / "child_yolov9c.hef"))

# ── Inference backend PER MODEL ────────────────────────────────────────────
# 'hailo'       -> run the .hef on the Hailo-8 accelerator (fast, per-frame model)
# 'ultralytics' -> run the .pt on the Pi CPU (fine for the rare pool segmenter)
# Recommended: person on Hailo (every frame), seg on CPU (every ~30 frames) —
# avoids the painful on-Hailo mask decode while keeping FPS high.
PERSON_BACKEND = os.getenv("PERSON_BACKEND", "hailo")
SEG_BACKEND = os.getenv("SEG_BACKEND", "ultralytics")

# ── Class indices — MUST match training (notebook 02): adult=0, child=1 ───
CLASS_NAMES = ["adult", "child"]
ADULT_IDX = 0
CHILD_IDX = 1

# ── Detection thresholds ──────────────────────────────────────────────────
CONF_SEG = float(os.getenv("CONF_SEG", 0.40))
CONF_PERSON = float(os.getenv("CONF_PERSON", 0.40))
# 640 matches training imgsz. Drop to 480 on CPU to claw back latency at a
# small accuracy cost — children are small targets, so don't go lower than 416.
IMG_SIZE = int(os.getenv("IMG_SIZE", 640))

# ── Pipeline behaviour ─────────────────────────────────────────────────────
# Segmentation is the heavy model; the pool doesn't move, so cache the polygon
# and only re-run seg every N frames. This is the main CPU-budget lever.
SEG_UPDATE_EVERY = int(os.getenv("SEG_UPDATE_EVERY", 30))
# On a slow CPU, process every Kth frame to keep alert latency bounded.
PROCESS_EVERY = int(os.getenv("PROCESS_EVERY", 1))
# Require this many consecutive in-pool frames before EMERGENCY — kills the
# single-frame foot-point false positive (a child stepping past the edge).
CONFIRM_FRAMES = int(os.getenv("CONFIRM_FRAMES", 5))

# ── Alert cooldowns (seconds) — stop one event flooding the parent's phone ─
COOLDOWN_DANGER = int(os.getenv("COOLDOWN_DANGER", 30))
COOLDOWN_EMERGENCY = int(os.getenv("COOLDOWN_EMERGENCY", 30))

# ── Twilio WhatsApp (values come from .env / secrets, never hardcoded) ─────
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
PARENT_WHATSAPP = os.getenv("PARENT_WHATSAPP")

# ── Hardware ───────────────────────────────────────────────────────────────
SIREN_PIN = int(os.getenv("SIREN_PIN", 18))      # BCM pin to relay/active buzzer
SIREN_SECONDS = int(os.getenv("SIREN_SECONDS", 5))
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "picamera")  # 'picamera' | 'usb' | /path/to.mp4
FRAME_W = int(os.getenv("FRAME_W", 1280))
FRAME_H = int(os.getenv("FRAME_H", 720))
# Headless=true for the systemd service (no display); false to see the HUD window.
HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"

# ── Watchdog / heartbeat ───────────────────────────────────────────────────
# A safety device that dies silently is worse than none. Send a periodic
# "system OK" so a dead/crashed Pi is noticed (pairs with systemd auto-restart).
HEARTBEAT_ENABLED = os.getenv("HEARTBEAT_ENABLED", "true").lower() == "true"
HEARTBEAT_EVERY = int(os.getenv("HEARTBEAT_EVERY", 300))   # seconds between pings
