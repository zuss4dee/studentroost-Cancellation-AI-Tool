# debug_ela.py — run from repo root
# Usage: python debug_ela.py path/to/your/test-id.jpg

import sys
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from scipy.ndimage import uniform_filter

img_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
import fitz  # PyMuPDF
if img_path.lower().endswith('.pdf'):
    pdf_doc = fitz.open(img_path)
    first_page = pdf_doc[0]
    pix = first_page.get_pixmap(matrix=fitz.Matrix(2, 2))
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    pdf_doc.close()
else:
    image = Image.open(img_path).convert('RGB')
original_np = np.array(image, dtype=np.float32)

print(f"\n📁 File: {img_path}")
print(f"📐 Size: {image.size}, Mode: {image.mode}")

# --- ELA signal test ---
print("\n--- LAYER 1: ELA ---")
for q in [60, 75, 90, 95]:
    buf = BytesIO()
    image.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    resaved = np.array(Image.open(buf).convert('RGB'), dtype=np.float32)
    diff = np.abs(original_np - resaved).mean(axis=2)
    print(f"  Q{q}: mean={diff.mean():.4f}, max={diff.max():.4f}, p99={np.percentile(diff,99):.4f}")

# --- Noise signal test ---
print("\n--- LAYER 2: NOISE ---")
gray = np.array(image.convert('L'), dtype=np.float32)
smoothed = uniform_filter(gray, size=3)
residual = np.abs(gray - smoothed)
local_mean = uniform_filter(residual, size=16)
local_sq   = uniform_filter(residual**2, size=16)
local_var  = np.clip(local_sq - local_mean**2, 0, None)
print(f"  local_var: mean={local_var.mean():.4f}, max={local_var.max():.4f}, p99={np.percentile(local_var,99):.4f}")

# --- DCT signal test ---
print("\n--- LAYER 3: DCT ---")
img_gray = np.array(image.convert('L'), dtype=np.float32)
h, w = img_gray.shape
energies = []
for y in range(0, h-8, 8):
    for x in range(0, w-8, 8):
        block = img_gray[y:y+8, x:x+8]
        dct_b = cv2.dct(block)
        energies.append(float(np.sum(np.abs(dct_b[4:, 4:]))))
print(f"  DCT energy: mean={np.mean(energies):.2f}, std={np.std(energies):.2f}, max={np.max(energies):.2f}")
print(f"  Blocks above 1.5 std: {sum(1 for e in energies if e > np.mean(energies)+1.5*np.std(energies))}")

# --- Save each layer as PNG so you can see them raw ---
# ELA
buf = BytesIO()
image.save(buf, format='JPEG', quality=75)
buf.seek(0)
resaved = np.array(Image.open(buf).convert('RGB'), dtype=np.float32)
ela_raw = np.abs(original_np - resaved).mean(axis=2)
ela_scaled = np.clip(ela_raw / (ela_raw.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
cv2.imwrite("debug_ela_raw.png", ela_scaled)

# Noise
noise_scaled = np.clip(local_var / (local_var.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
cv2.imwrite("debug_noise_raw.png", noise_scaled)

print("\n✅ Saved: debug_ela_raw.png and debug_noise_raw.png")
print("   Open these — if they're black, the signal truly isn't there at this layer.")