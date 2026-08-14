"""
Image quality checks for the waste classifier app.
Each check returns a (passed: bool, message: str) tuple.
"""
import numpy as np
from PIL import Image, ImageStat
from typing import Dict, List, Tuple


def check_resolution(img: Image.Image, min_size: int = 100) -> Tuple[bool, str]:
    """Check if the image is high enough resolution."""
    width, height = img.size
    min_dim = min(width, height)
    
    if min_dim < min_size:
        return False, (
            f"Image is too small ({width}×{height}). "
            f"Please use at least {min_size}×{min_size} pixels."
        )
    return True, f"Resolution: {width}×{height} pixels (OK)"


def check_brightness(img: Image.Image, dark_thresh: float = 30, 
                     bright_thresh: float = 240) -> Tuple[bool, str]:
    """Check if the image brightness is reasonable."""
    # Convert to grayscale for analysis
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean_brightness = stat.mean[0]
    
    if mean_brightness < dark_thresh:
        return False, (
            f"Image is too dark (brightness: {mean_brightness:.0f}/255). "
            f"Try better lighting or increase exposure."
        )
    if mean_brightness > bright_thresh:
        return False, (
            f"Image is overexposed (brightness: {mean_brightness:.0f}/255). "
            f"Reduce lighting or decrease exposure."
        )
    return True, f"Brightness: {mean_brightness:.0f}/255 (OK)"


def check_contrast(img: Image.Image, min_std: float = 15.0) -> Tuple[bool, str]:
    """Check if the image has enough contrast (not flat/uniform)."""
    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    std = stat.stddev[0]
    
    if std < min_std:
        return False, (
            f"Image has very low contrast (std: {std:.1f}). "
            f"This may be a flat image with no clear subject. "
            f"Try a photo with the object clearly visible."
        )
    return True, f"Contrast: std={std:.1f} (OK)"


def check_saturation(img: Image.Image, 
                     min_sat: float = 5.0,
                     max_sat: float = 250.0) -> Tuple[bool, str]:
    """Check if image has reasonable color saturation."""
    # Convert to HSV
    hsv = img.convert("HSV")
    stat = ImageStat.Stat(hsv)
    # S channel is index 1
    mean_sat = stat.mean[1]
    
    if mean_sat < min_sat:
        return False, (
            f"Image is nearly grayscale (saturation: {mean_sat:.1f}). "
            f"Waste materials often need color cues to distinguish. "
            f"Try a color photo."
        )
    if mean_sat > max_sat:
        return False, (
            f"Image is oversaturated (saturation: {mean_sat:.1f}). "
            f"Colors may be unrealistic. Try a more natural photo."
        )
    return True, f"Saturation: {mean_sat:.1f}/255 (OK)"


def check_blur(img: Image.Image, threshold: float = 100.0) -> Tuple[bool, str]:
    """Estimate image blur using Laplacian variance."""
    try:
        import cv2
        # Convert PIL to OpenCV
        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < threshold:
            return False, (
                f"Image appears blurry (variance: {laplacian_var:.1f}). "
                f"Hold the camera steady and ensure the object is in focus."
            )
        return True, f"Sharpness: var={laplacian_var:.1f} (OK)"
    except ImportError:
        # If OpenCV isn't available, skip this check
        return True, "Sharpness: (skipped — OpenCV not installed)"


def check_aspect_ratio(img: Image.Image, 
                       max_ratio: float = 10.0) -> Tuple[bool, str]:
    """Check if the aspect ratio is reasonable."""
    width, height = img.size
    ratio = max(width, height) / min(width, height)
    
    if ratio > max_ratio:
        return False, (
            f"Extreme aspect ratio ({ratio:.1f}:1). "
            f"Try a more square image."
        )
    return True, f"Aspect ratio: {ratio:.2f}:1 (OK)"


def check_edge_density(img: Image.Image, 
                       min_edges: float = 0.01,
                       max_edges: float = 0.5) -> Tuple[bool, str]:
    """
    Check if the image has enough edges (subject visibility).
    Too few edges = blank/uniform image.
    Too many edges = cluttered/noisy image.
    """
    try:
        import cv2
        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density < min_edges:
            return False, (
                f"Image has very few edges (density: {edge_density:.3f}). "
                f"The image may be blank or have no clear object. "
                f"Try a photo with a clear subject."
            )
        if edge_density > max_edges:
            return False, (
                f"Image is very cluttered (edge density: {edge_density:.3f}). "
                f"Try a cleaner photo with the object more prominent."
            )
        return True, f"Edge density: {edge_density:.3f} (OK)"
    except ImportError:
        return True, "Edge density: (skipped — OpenCV not installed)"


def check_color_diversity(img: Image.Image, 
                          min_unique_colors: int = 100) -> Tuple[bool, str]:
    """Check if the image has enough color diversity."""
    try:
        # Get colors and count unique ones
        arr = np.array(img.convert("RGB"))
        # Quantize to reduce sensitivity
        quantized = (arr // 16).astype(np.uint8)
        unique_colors = len(np.unique(
            quantized.reshape(-1, 3), axis=0
        ))
        
        if unique_colors < min_unique_colors:
            return False, (
                f"Image has very few colors ({unique_colors} unique). "
                f"Try a photo with more visual variety."
            )
        return True, f"Color diversity: {unique_colors} unique colors (OK)"
    except Exception:
        return True, "Color diversity: (skipped due to error)"


def check_is_grayscale(img: Image.Image) -> Tuple[bool, str]:
    """Check if the image is mistakenly grayscale."""
    if img.mode == "L":
        return False, "Image is grayscale. Use a color photo for better material recognition."
    if img.mode == "RGBA":
        # Check if alpha is uniform
        alpha = np.array(img)[:, :, 3]
        if (alpha == 255).all() or (alpha == 0).all():
            return False, "Image has uniform transparency. Try a standard photo."
    return True, "Color mode: OK"

# Add to quality_checks.py
def auto_correct_image(img: Image.Image) -> Tuple[Image.Image, List[str]]:
    """
    Attempt to auto-correct common image quality issues.
    Returns corrected image and list of corrections applied.
    """
    corrections = []
    corrected = img.copy()
    
    # 1. Auto-orient based on EXIF
    try:
        from PIL import ImageOps
        corrected = ImageOps.exif_transpose(corrected)
        corrections.append("Auto-oriented based on EXIF")
    except Exception:
        pass
    
    # 2. Auto-contrast if too flat
    gray = corrected.convert("L")
    stat = ImageStat.Stat(gray)
    if stat.stddev[0] < 15:
        from PIL import ImageOps
        corrected = ImageOps.autocontrast(corrected, cutoff=2)
        corrections.append("Auto-contrast enhanced")
    
    # 3. Auto-brightness if too dark
    if stat.mean[0] < 30:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(corrected)
        corrected = enhancer.enhance(2.0)
        corrections.append("Brightness boosted")
    
    # 4. Resize if too large
    if max(corrected.size) > 2048:
        corrected.thumbnail((2048, 2048), Image.LANCZOS)
        corrections.append("Resized to max 2048px")
    
    return corrected, corrections


def run_all_checks(img: Image.Image) -> Dict:
    """
    Run all quality checks and return aggregated results.
    
    Returns:
        Dictionary with:
          - 'all_passed': bool (True if all checks pass)
          - 'warnings': list of warning strings
          - 'info': list of info strings (passed checks)
          - 'should_proceed': bool (True if critical checks pass)
          - 'quality_score': float (0-100, higher is better)
    """
    # Critical checks (must pass to proceed)
    critical_checks = [
        check_resolution,
        check_brightness,
        check_aspect_ratio,
    ]
    
    # Non-critical checks (warnings only)
    advisory_checks = [
        check_contrast,
        check_saturation,
        check_blur,
        check_edge_density,
        check_color_diversity,
        check_is_grayscale,
    ]
    
    warnings = []
    info = []
    critical_passed = True
    
    # Run critical checks
    for check_fn in critical_checks:
        passed, message = check_fn(img)
        if not passed:
            critical_passed = False
            warnings.append(f"🚫 {message}")
        else:
            info.append(f"✓ {message}")
    
    # Run advisory checks
    for check_fn in advisory_checks:
        try:
            passed, message = check_fn(img)
            if not passed:
                warnings.append(f"⚠️ {message}")
            else:
                info.append(f"✓ {message}")
        except Exception as e:
            # Don't fail the whole check if one check fails
            info.append(f"⚠️ {check_fn.__name__}: skipped (error)")
    
    # Calculate quality score (0-100)
    total_checks = len(critical_checks) + len(advisory_checks)
    passed_count = len(info)
    quality_score = (passed_count / total_checks) * 100
    
    return {
        "all_passed": critical_passed and len(warnings) == 0,
        "critical_passed": critical_passed,
        "warnings": warnings,
        "info": info,
        "should_proceed": critical_passed,
        "quality_score": quality_score,
    }
