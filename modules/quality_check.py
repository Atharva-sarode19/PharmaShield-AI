import cv2
import numpy as np


def calculate_blur_score(image: np.ndarray) -> float:
    """
    Measures how sharp the image is.

    Higher value = sharper image
    Lower value = more blurry
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return float(blur_score)


def calculate_brightness(image: np.ndarray) -> float:
    """
    Calculates the average brightness of the image.

    Range is approximately 0 to 255.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    brightness = np.mean(gray)

    return float(brightness)


def calculate_contrast(image: np.ndarray) -> float:
    """
    Calculates image contrast.

    Higher value = more contrast.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    contrast = np.std(gray)

    return float(contrast)


def check_image_quality(image: np.ndarray) -> dict:
    """
    Performs basic quality checks on the uploaded image.

    Checks:
    1. Blur / sharpness
    2. Brightness
    3. Contrast
    """

    blur_score = calculate_blur_score(image)

    brightness = calculate_brightness(image)

    contrast = calculate_contrast(image)

    issues = []

    # -------------------------
    # Blur Check
    # -------------------------

    if blur_score < 50:
        issues.append("Image is too blurry")

    # -------------------------
    # Brightness Check
    # -------------------------

    if brightness < 45:
        issues.append("Image is too dark")

    elif brightness > 220:
        issues.append("Image is too bright")

    # -------------------------
    # Contrast Check
    # -------------------------

    if contrast < 20:
        issues.append("Image has very low contrast")

    # -------------------------
    # Overall Quality
    # -------------------------

    quality_ok = len(issues) == 0

    # Start with perfect score
    quality_score = 100.0

    # Penalize blur
    if blur_score < 50:
        quality_score -= 40

    elif blur_score < 100:
        quality_score -= 20

    # Penalize brightness problems
    if brightness < 45 or brightness > 220:
        quality_score -= 25

    # Penalize low contrast
    if contrast < 20:
        quality_score -= 20

    # Keep score between 0 and 100
    quality_score = max(
        0.0,
        min(100.0, quality_score)
    )

    return {
        "quality_ok": quality_ok,
        "quality_score": quality_score,
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
        "issues": issues
    }