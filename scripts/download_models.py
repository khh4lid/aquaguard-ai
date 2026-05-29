#!/usr/bin/env python3
"""
Download model weights from Google Drive into ./models/.
File IDs taken from the README Drive links.
Run once on the Pi after cloning:  python scripts/download_models.py
"""
from pathlib import Path
import gdown

MODELS = {
    "pool_yolov8m_seg.pt": "17aHZsdLLXvAG9C1nLzuUTPi9_rRorNDD",
    "child_yolov9c.pt":    "1IGT9uFrJFYFJZy5CHxt72paOoMVPwbc1",
}

out = Path(__file__).resolve().parent.parent / "models"
out.mkdir(exist_ok=True)

for name, fid in MODELS.items():
    dest = out / name
    if dest.exists():
        print(f"skip {name} (already present)")
        continue
    print(f"downloading {name} …")
    gdown.download(id=fid, output=str(dest), quiet=False)

print("done — weights in ./models/")
