import os
import io
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from shapely.geometry import Polygon, Point
from ultralytics import YOLO

# ============================
# CONFIG
# ============================

MODEL_PATH = "best.pt"              # trained YOLOv8-seg model
INPUT_EXCEL = "input.xlsx"          # sample_id, latitude, longitude
OUTPUT_DIR = "outputs"
IMG_SIZE = 512

BUFFERS = [
    {"sqft": 1200, "radius_m": 6},
    {"sqft": 2400, "radius_m": 8.5},
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# HELPERS
# ============================

def download_esri(lat, lon, size=512):
    url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    d = 0.0012

    params = {
        "bbox": f"{lon-d},{lat-d},{lon+d},{lat+d}",
        "bboxSR": "4326",
        "imageSR": "3857",
        "size": f"{size},{size}",
        "format": "png",
        "f": "image"
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None

    try:
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def pixel_to_meter(lat, deg=0.0012, size=512):
    return (111000 * deg) / size


def visualize(img, poly, buffer_circle, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)

    ax.set_xlim(0, img.size[0])
    ax.set_ylim(img.size[1], 0)

    if buffer_circle:
        bx, by = buffer_circle.exterior.xy
        ax.plot(bx, by, color="yellow", linewidth=2)

    if poly:
        px, py = poly.exterior.xy
        ax.plot(px, py, color="red", linewidth=2)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================
# LOAD MODEL & DATA
# ============================

model = YOLO(MODEL_PATH)
df = pd.read_excel(INPUT_EXCEL)

results = []

# ============================
# MAIN LOOP
# ============================

for _, r in df.iterrows():
    sid = r["sample_id"]
    lat = r["latitude"]
    lon = r["longitude"]

    print(f"Processing {sid}")

    tile = download_esri(lat, lon)
    if tile is None:
        results.append({
            "sample_id": sid,
            "lat": lat,
            "lon": lon,
            "has_solar": False,
            "buffer_radius_sqft": None,
            "pv_area_sqm_est": None
        })
        continue

    image_np = np.array(tile)

    yolo_results = model(image_np, conf=0.25, verbose=False)
    masks = yolo_results[0].masks.xy if yolo_results[0].masks else []

    pm = pixel_to_meter(lat)
    center = Point(IMG_SIZE // 2, IMG_SIZE // 2)

    polys = []
    for m in masks:
        if len(m) >= 3:
            polys.append(Polygon(m))

    chosen_poly = None
    chosen_buf = None

    for buf in BUFFERS:
        buffer_px = buf["radius_m"] / pm
        buffer_circle = center.buffer(buffer_px)

        inside = [p for p in polys if buffer_circle.intersection(p).area > 0]
        if inside:
            chosen_poly = max(inside, key=lambda x: x.area)
            chosen_buf = buf
            break

    # ===== OUTPUT =====

    if chosen_poly:
        area_m2 = chosen_poly.area * (pm ** 2)
        buffer_used = chosen_buf["sqft"]
        has_solar = True
    else:
        area_m2 = None
        buffer_used = BUFFERS[-1]["sqft"]
        has_solar = False
        buffer_circle = center.buffer(BUFFERS[-1]["radius_m"] / pm)

    vis_path = os.path.join(OUTPUT_DIR, f"{sid}.png")

    visualize(
        tile,
        chosen_poly,
        center.buffer((chosen_buf or BUFFERS[-1])["radius_m"] / pm),
        vis_path,
        f"{sid} | Buffer {buffer_used} sqft"
    )

    results.append({
        "sample_id": sid,
        "lat": lat,
        "lon": lon,
        "has_solar": has_solar,
        "buffer_radius_sqft": buffer_used,
        "pv_area_sqm_est": None if area_m2 is None else round(area_m2, 2),
        "visualization": vis_path
    })

# ============================
# SAVE OUTPUTS
# ============================

with open(os.path.join(OUTPUT_DIR, "predictions.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Inference complete.")
