import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Add src/ directory to Python path to allow module import
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from generate_dataset_funcs import *

# Define input directories relative to this script's location (assumes notebooks/ directory)
intensity_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Output'))
depth_dir = os.path.join(intensity_dir, 'EXR')


# Gather intensity image paths and map EXR depth files using their camera-center keys
intensity_paths = sorted(glob.glob(os.path.join(intensity_dir, "intensity_Sun_*_Camera *.png")))
depth_paths_map = {
    extract_camera_center_key(f): f
    for f in glob.glob(os.path.join(depth_dir, "depth_Sun_0_Camera *.exr"))
}

records = []

# Iterate over each intensity image and generate 128x128, 64x64, and 32x32 crops with metadata
for intensity_path in tqdm(intensity_paths, desc="Processing"):
    key = extract_camera_center_key(intensity_path)
    if key not in depth_paths_map:
        print(f"No depth match for Camera {key[0]} at ({key[1]}, {key[2]})")
        continue

    cx_orig, cy_orig = extract_center_from_filename(intensity_path)
    if cx_orig is None or cy_orig is None:
        continue

    img = Image.open(intensity_path)
    intensity = np.array(img, dtype=np.float32)
    if intensity.shape != (128, 128):
        continue

    try:
        depth = load_exr_depth(depth_paths_map[key])
        if depth.shape != (128, 128):
            depth = None
    except:
        depth = None

    # Store the full 128x128 crop (Level 1)
    l1, l2, l3 = get_multilevel_quadrants(cx_orig, cy_orig)
    records.append({
        "image": intensity, "depth": depth, "crop_size": 128,
        "level1_quadrant": l1, "level2_quadrant": 0, "level3_quadrant": 0,
        "center_x": cx_orig, "center_y": cy_orig
    })

    # Split into 64x64 crops (Level 2)
    l2_crops_intensity = split_into_quadrants(intensity, 64)
    l2_crops_depth = split_into_quadrants(depth, 64) if depth is not None else [None] * 4

    for i64, (row64, col64, img64) in enumerate(l2_crops_intensity):
        dep64 = l2_crops_depth[i64][2] if depth is not None else None
        center_x = cx_orig + col64 - 32
        center_y = cy_orig + row64 - 32
        l1, l2, l3 = get_multilevel_quadrants(center_x, center_y)

        records.append({
            "image": img64, "depth": dep64, "crop_size": 64,
            "level1_quadrant": l1, "level2_quadrant": l2, "level3_quadrant": l3,
            "center_x": center_x, "center_y": center_y
        })

        # Split into 32x32 crops (Level 3)
        l3_crops_intensity = split_into_quadrants(img64, 32)
        l3_crops_depth = split_into_quadrants(dep64, 32) if dep64 is not None else [None] * 4

        for j32, (row32, col32, img32) in enumerate(l3_crops_intensity):
            dep32 = l3_crops_depth[j32][2] if dep64 is not None else None
            center_x_32 = center_x - 32 + col32
            center_y_32 = center_y - 32 + row32
            l1, l2, l3 = get_multilevel_quadrants(center_x_32, center_y_32)

            records.append({
                "image": img32, "depth": dep32, "crop_size": 32,
                "level1_quadrant": l1, "level2_quadrant": l2, "level3_quadrant": l3,
                "center_x": center_x_32, "center_y": center_y_32
            })

# Save dataset to pickle file
df = pd.DataFrame(records)
df.to_pickle("labeled_dataset_A.pkl")
print(f"Saved {len(df)} labeled crops to 'labeled_dataset_A.pkl'")

# Create an 8x8 heatmap of crop center distribution in the central region (576x576)
grid = np.zeros((8, 8), dtype=int)
lb, ub = 96, 672
spacing = (ub - lb) / 8
for x, y in zip(df['center_x'], df['center_y']):
    if lb <= x < ub and lb <= y < ub:
        col = int((x - lb) // spacing)
        row = int((y - lb) // spacing)
        grid[row, col] += 1

plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar(label='Samples per 48x48 cell')
plt.title("Central 576x576 Distribution (8x8 cells)")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.gca().invert_yaxis()
plt.grid(False)
plt.show()

# Display metadata for one 32x32 crop
df_32 = df[df["crop_size"] == 32].reset_index(drop=True)
sample = df_32.iloc[0]
x, y = sample["center_x"], sample["center_y"]
l1, l2, l3 = sample["level1_quadrant"], sample["level2_quadrant"], sample["level3_quadrant"]

print(f"Sample 0: Center=({x:.1f}, {y:.1f}) | L1={l1}, L2={l2}, L3={l3}")

# Visualize sample center location on an 8x8 grid
grid = np.zeros((8, 8))
col = int((x - 96) // 72)
row = int((y - 96) // 72)
if 0 <= row < 8 and 0 <= col < 8:
    grid[row, col] = 1

plt.imshow(grid, cmap='Greens', interpolation='nearest')
plt.title("Sample 0 Grid Position")
plt.scatter([col], [row], c='red', label='Center')
plt.gca().invert_yaxis()
plt.legend()
plt.grid(False)
plt.show()
