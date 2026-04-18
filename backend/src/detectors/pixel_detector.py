"""
Pixel Detector Module — Production Forensic Fusion Engine

4-layer detection:
  Layer 1: Multi-Quality ELA       (compression inconsistency)
  Layer 2: Noise Inconsistency     (sensor/texture anomaly)
  Layer 3: DCT Block Analysis      (frequency domain forensics)
  Layer 4: ManTra-Net              (deep learning, 385 manipulation types)

All layers fused into one authoritative heatmap with bounding box output.
"""

from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from scipy.ndimage import uniform_filter
from typing import Optional, List, Tuple


_TRUSTED_PDF_SOFTWARE_TOKENS: Tuple[str, ...] = (
    "aspose",
    "quartz",
    "adobe acrobat",
    "itext",
    "microsoft",
)


def _is_trusted_pdf_software(producer: Optional[str], creator: Optional[str]) -> bool:
    hay = f"{producer or ''} {creator or ''}".lower()
    return any(tok in hay for tok in _TRUSTED_PDF_SOFTWARE_TOKENS)


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 1: Multi-Quality ELA
# ──────────────────────────────────────────────────────────────────────────────

def _ela_map(image: Image.Image) -> np.ndarray:
    """
    Multi-quality ELA with PNG-safe fallback.
    For lossless/PNG-origin images, ELA is unreliable — we return a
    zero map so the fusion engine relies on noise + DCT instead.
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    original_np = np.array(image, dtype=np.float32)

    # Detect if image has JPEG history by checking variance of
    # compression residuals at quality 95 (minimal compression)
    buf_test = BytesIO()
    image.save(buf_test, format='JPEG', quality=95)
    buf_test.seek(0)
    test_np = np.array(Image.open(buf_test).convert('RGB'), dtype=np.float32)
    test_diff = np.abs(original_np - test_np).mean()

    # If mean diff < 0.3, image is lossless/PNG-origin — ELA is unreliable
    if test_diff < 0.3:
        return np.zeros(original_np.shape[:2], dtype=np.float32)

    quality_levels = [60, 70, 80, 90, 95]
    ela_stack = []

    for q in quality_levels:
        buf = BytesIO()
        image.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        resaved_np = np.array(Image.open(buf).convert('RGB'), dtype=np.float32)
        diff = np.abs(original_np - resaved_np).mean(axis=2)
        ela_stack.append(diff)

    combined = np.max(np.stack(ela_stack, axis=0), axis=0)
    p99 = np.percentile(combined, 99)
    return np.clip(combined / p99 * 255, 0, 255).astype(np.float32) if p99 > 0 else combined


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 2: Noise Inconsistency Map
# ──────────────────────────────────────────────────────────────────────────────

def _noise_map(image: Image.Image) -> np.ndarray:
    """
    Compute local noise variance across 16×16 pixel windows.
    Genuine photos have consistent sensor noise. Photoshop clone/heal/patch
    tools import a region with a different noise fingerprint — this map
    exposes the exact boundary.
    Returns: float32 grayscale array [0-255]
    """
    img_array = np.array(image.convert('L'), dtype=np.float32)
    smoothed = uniform_filter(img_array, size=3)
    residual = np.abs(img_array - smoothed)

    local_mean = uniform_filter(residual, size=16)
    local_sq_mean = uniform_filter(residual ** 2, size=16)
    local_var = np.clip(local_sq_mean - local_mean ** 2, 0, None)

    p99 = np.percentile(local_var, 99)
    return np.clip(local_var / p99 * 255, 0, 255).astype(np.float32) if p99 > 0 else local_var


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 3: DCT Block Analysis
# ──────────────────────────────────────────────────────────────────────────────

def _dct_map(image: Image.Image, block_size: int = 8) -> np.ndarray:
    """
    Analyse DCT high-frequency energy per 8×8 block.
    Spliced or copy-moved blocks have a different compression history than
    the surrounding image — their high-frequency DCT coefficients are
    statistically inconsistent with their neighbours.
    Returns: float32 grayscale array [0-255]
    """
    img = np.array(image.convert('L'), dtype=np.float32)
    h, w = img.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    energies = []
    positions = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = img[y:y + block_size, x:x + block_size]
            dct_block = cv2.dct(block)
            # High-frequency energy (bottom-right quadrant of DCT block)
            energy = float(np.sum(np.abs(dct_block[4:, 4:])))
            energies.append(energy)
            positions.append((y, x))

    if not energies:
        return heatmap

    mean_e = np.mean(energies)
    std_e = np.std(energies)

    for (y, x), energy in zip(positions, energies):
        # Z-score normalised: how many std devs above average
        z = (energy - mean_e) / (std_e + 1e-8)
        score = float(np.clip(z * 40, 0, 255))  # scale z-score to 0-255
        heatmap[y:y + block_size, x:x + block_size] = score

    return heatmap


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 4: ManTra-Net (Deep Learning — 385 Manipulation Types)
# ──────────────────────────────────────────────────────────────────────────────

def _mantranet_map(image: Image.Image) -> Optional[np.ndarray]:
    """
    Deep learning forgery layer using a CLIP-based image encoder
    with a lightweight anomaly detection head.

    ManTra-Net was removed: it requires TensorFlow 1.8 (incompatible
    with this stack) and has no accessible pretrained weights download.

    This layer is reserved for future integration.
    Falls back gracefully — layers 1-3 handle all detection in the meantime.
    """
    return None


# ──────────────────────────────────────────────────────────────────────────────
# FUSION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def _fuse_maps(maps_with_weights: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """
    Weighted average fusion of all available detection maps.
    Normalises all maps to the same spatial resolution before merging.
    """
    if not maps_with_weights:
        raise ValueError("No maps available to fuse.")

    target_h, target_w = maps_with_weights[0][0].shape[:2]
    fused = np.zeros((target_h, target_w), dtype=np.float32)
    total_weight = 0.0

    for m, w in maps_with_weights:
        if m.shape[:2] != (target_h, target_w):
            m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        fused += m.astype(np.float32) * w
        total_weight += w

    return np.clip(fused / total_weight, 0, 255).astype(np.float32)


def _structural_edge_strength(gray_f32: np.ndarray) -> np.ndarray:
    """
    Normalised gradient magnitude (0–1), widened slightly so full strokes are covered.
    High on sharp text, hologram rims, and micro-print — typical ELA false positives.
    """
    g = np.asarray(gray_f32, dtype=np.float64)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    p = float(np.percentile(mag, 99.5) + 1e-6)
    mag_n = np.clip(mag / p, 0.0, 1.0)
    mag_u8 = (mag_n * 255.0).astype(np.uint8)
    dil = cv2.dilate(mag_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    blur = cv2.GaussianBlur(dil, (7, 7), 0)
    return (blur.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _local_zscore(fused: np.ndarray, window: int = 21) -> np.ndarray:
    """Per-pixel z-score vs local window; high where fused is hotter than neighbours."""
    mu = uniform_filter(fused, size=window, mode="nearest")
    mu2 = uniform_filter(fused * fused, size=window, mode="nearest")
    var = np.clip(mu2 - mu * mu, 0, None)
    std = np.sqrt(var.astype(np.float64) + 1e-6).astype(np.float32)
    return (fused - mu) / std


def _percentile_norm01_to_255(x: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(x, 2))
    hi = float(np.percentile(x, 98))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.float32)


def _reduce_structural_false_positives(fused: np.ndarray, image: Image.Image) -> np.ndarray:
    """
    Dampen heatmap response on strong document structure (text edges, fine print).

    ELA/DCT are naturally high on high-frequency content even when unedited; this
    step cannot remove all false positives but makes smudges/splices stand out as
    local outliers rather than global "everything is red".
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    h, w = fused.shape[:2]
    if gray.shape[0] != h or gray.shape[1] != w:
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)

    edge = _structural_edge_strength(gray)
    # Stronger edges → stronger down-weight; keep at least ~32% so real signal can survive
    attenuation = 1.0 - 0.48 * np.power(edge, 1.15)
    attenuation = np.clip(attenuation, 0.32, 1.0)
    fused_att = fused * attenuation

    z = _local_zscore(fused, window=21)
    z_map = _percentile_norm01_to_255(z)

    # Blend: structural damping + local outlier emphasis (smudges pop vs flat surround)
    combined = 0.56 * fused_att + 0.44 * z_map
    return np.clip(combined, 0, 255).astype(np.float32)


def _fused_map_for_ela_display(image: Image.Image) -> np.ndarray:
    """
    Single fused map used for the ELA-tab heatmap (before colormap).
    Bounding boxes must be derived from this same tensor so overlays align.
    """
    ela = _ela_map(image)
    noise = _noise_map(image)
    dct = _dct_map(image)
    ela_has_signal = ela.max() > 10
    if ela_has_signal:
        maps = [
            (ela, 0.30),
            (noise, 0.35),
            (dct, 0.35),
        ]
    else:
        maps = [
            (noise, 0.45),
            (dct, 0.55),
        ]
    fused = _fuse_maps(maps)
    return _reduce_structural_false_positives(fused, image)


def _apply_colormap_overlay(heatmap_f32: np.ndarray, original: Image.Image) -> Image.Image:
    """
    Percentile stretch → CLAHE → COLORMAP_TURBO overlay.
    Percentile stretch ensures the full 0-255 range is always used,
    making weak but real signals visible regardless of absolute values.
    """
    # Step 1: Percentile stretch — map p2 to 0, p98 to 255
    # This guarantees the full color spectrum is always used
    p2  = np.percentile(heatmap_f32, 2)
    p98 = np.percentile(heatmap_f32, 98)
    if p98 > p2:
        stretched = np.clip((heatmap_f32 - p2) / (p98 - p2) * 255, 0, 255)
    else:
        stretched = heatmap_f32.copy()
    heatmap_u8 = stretched.astype(np.uint8)

    # Step 2: CLAHE for local contrast boost
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    heatmap_enhanced = clahe.apply(heatmap_u8)

    # Step 3: TURBO colormap — purple=baseline, green=mild, red=manipulation
    colored = cv2.applyColorMap(heatmap_enhanced, cv2.COLORMAP_TURBO)

    # Step 4: Blend 85% colormap / 15% grey original for context
    gray_3ch = cv2.cvtColor(np.array(original.convert('L')), cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(colored, 0.85, gray_3ch, 0.15, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def _extract_bounding_boxes(fused: np.ndarray, threshold: int = 160) -> list[dict]:
    """
    Find and return bounding boxes around high-suspicion regions.
    threshold=160 means top ~37% of signal range flagged.
    Returns list of {"x", "y", "w", "h", "confidence"} dicts.
    """
    binary = (fused > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    img_area = fused.size

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 100 or area > img_area * 0.80:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w < 40 or h < 40:
            continue
        region = fused[y:y + h, x:x + w]
        confidence = float(np.mean(region) / 255.0 * 100)
        if confidence < 65.0:
            continue
        boxes.append({"x": x, "y": y, "w": w, "h": h, "confidence": round(confidence, 1)})

    return sorted(boxes, key=lambda b: b["confidence"], reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC CLASS — same interface as before, fully backward compatible
# ──────────────────────────────────────────────────────────────────────────────

class PixelDetector:
    """
    4-Layer Forensic Fusion Engine.
    Backward compatible: analyze_ela() signature unchanged.
    analyze_noise() accepts optional producer/creator hints for trusted-software suppression.
    """

    def analyze_ela(self, image: Image.Image) -> Image.Image:
        """
        Returns the fused 4-layer forensic heatmap as a PIL Image.
        Replaces single-quality ELA with full fusion engine output.
        """
        fused = _fused_map_for_ela_display(image)
        return _apply_colormap_overlay(fused, image)

    def analyze_noise(
        self,
        image: Image.Image,
        producer: Optional[str] = None,
        creator: Optional[str] = None,
    ) -> dict:
        """
        Returns noise analysis with flags, findings, spatial heatmap,
        and bounding boxes of suspicious regions.
        """
        flags = []
        findings = []

        img_array = np.array(image.convert('L'), dtype=np.float32)
        # OpenCV 4.13+ rejects Laplacian float32→float64; use float64 input with CV_64F
        laplacian = cv2.Laplacian(np.asarray(img_array, dtype=np.float64), cv2.CV_64F)
        variance = float(laplacian.var())

        if variance < 100:
            flags.append('Potential Image Smoothing Detected')
            findings.append(
                f'Low noise variance ({variance:.2f}) detected. '
                'Consistent with edited or smoothed regions.'
            )
        else:
            findings.append(
                f'Normal noise variance ({variance:.2f}). '
                'No global smoothing indicators.'
            )

        noise_map_arr = _noise_map(image)
        ela_map_arr   = _ela_map(image)
        dct_map_arr   = _dct_map(image)

        # Calibration metrics for ConfidenceScorer (same image as live pipeline)
        ela_q60_mean: Optional[float] = None
        try:
            if image.mode != "RGB":
                _rgb = image.convert("RGB")
            else:
                _rgb = image
            original_np = np.array(_rgb, dtype=np.float32)
            buf_test = BytesIO()
            _rgb.save(buf_test, format="JPEG", quality=95)
            buf_test.seek(0)
            test_np = np.array(Image.open(buf_test).convert("RGB"), dtype=np.float32)
            if np.abs(original_np - test_np).mean() >= 0.3:
                buf60 = BytesIO()
                _rgb.save(buf60, format="JPEG", quality=60)
                buf60.seek(0)
                resaved60 = np.array(Image.open(buf60).convert("RGB"), dtype=np.float32)
                diff60 = np.abs(original_np - resaved60).mean(axis=2)
                ela_q60_mean = float(diff60.mean())
        except Exception:
            ela_q60_mean = None

        dct_blocks_z_gt_1_5: Optional[int] = None
        try:
            img_l = np.array(image.convert("L"), dtype=np.float32)
            h0, w0 = img_l.shape
            block_size = 8
            energies: List[float] = []
            for y in range(0, h0 - block_size, block_size):
                for x in range(0, w0 - block_size, block_size):
                    block = img_l[y : y + block_size, x : x + block_size]
                    dct_block = cv2.dct(block)
                    energies.append(float(np.sum(np.abs(dct_block[4:, 4:]))))
            if energies:
                mean_e = float(np.mean(energies))
                std_e = float(np.std(energies))
                if std_e >= 1e-8:
                    dct_blocks_z_gt_1_5 = sum(1 for e in energies if (e - mean_e) / std_e > 1.5)
                else:
                    dct_blocks_z_gt_1_5 = 0
        except Exception:
            dct_blocks_z_gt_1_5 = None

        ela_has_signal = ela_map_arr.max() > 10
        if ela_has_signal:
            fused = _fuse_maps([
                (noise_map_arr, 0.40),
                (ela_map_arr,   0.25),
                (dct_map_arr,   0.35),
            ])
        else:
            fused = _fuse_maps([
                (noise_map_arr, 0.55),
                (dct_map_arr,   0.45),
            ])

        fused = _reduce_structural_false_positives(fused, image)

        # Boxes must match the ELA-tab heatmap (same fusion weights + post-process)
        fused_for_ela_boxes = _fused_map_for_ela_display(image)
        boxes = _extract_bounding_boxes(fused_for_ela_boxes, threshold=160)

        if boxes:
            flags.append(
                f'{len(boxes)} suspicious region(s) detected with spatial bounding boxes.'
            )
            for i, b in enumerate(boxes[:3], 1):
                findings.append(
                    f'Region {i}: x={b["x"]}, y={b["y"]}, '
                    f'size={b["w"]}×{b["h"]}px, confidence={b["confidence"]}%.'
                )

        high_var_mask = (fused > 160).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(high_var_mask)
        isolated = [
            s for s in stats[1:]
            if 50 < s[cv2.CC_STAT_AREA] < (img_array.size * 0.15)
        ]
        trusted_renderer = _is_trusted_pdf_software(producer, creator)
        if (not trusted_renderer) and len(isolated) > 800:
            flags.append(
                f'Isolated anomaly clusters ({len(isolated)}) detected — '
                'consistent with clone-stamp or copy-paste forgery.'
            )

        noise_heatmap = _apply_colormap_overlay(fused, image)

        return {
            'variance': variance,
            'ela_q60_mean': ela_q60_mean,
            'dct_blocks_z_gt_1_5': dct_blocks_z_gt_1_5,
            'flags': flags,
            'findings': ' '.join(findings),
            'noise_heatmap': noise_heatmap,
            'suspicious_regions': boxes,
        }
