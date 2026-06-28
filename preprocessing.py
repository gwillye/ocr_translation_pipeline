# -*- coding: utf-8 -*-
"""
preprocessing.py — OpenCV preprocessing for OCR (adaptive threshold + deskew).

This module is the "Adaptive thresholding / deskew (OpenCV)" extension. It is
designed for *photos and scans* where the original PIL pipeline
(`ocr_utils.preprocess_for_ocr`, a global threshold) struggles:

  * Uneven lighting / shadows  -> ``cv2.adaptiveThreshold`` (local, per-region
    binarization) instead of a single global cutoff at 128.
  * Tilted / skewed text       -> ``deskew`` estimates the dominant text angle
    (minAreaRect over foreground pixels, with a Hough-line fallback) and rotates
    the page back to horizontal.

All functions work on numpy/OpenCV (grayscale ``uint8``) arrays and there are
thin ``PIL.Image`` adapters so this drops into the existing pipeline. Nothing
here requires Tesseract or a network; it is pure image processing.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# PIL <-> OpenCV adapters
# --------------------------------------------------------------------------- #
def pil_to_cv(image: Image.Image) -> np.ndarray:
    """PIL.Image -> grayscale ``uint8`` numpy array (OpenCV convention)."""
    return np.array(image.convert("L"))


def cv_to_pil(array: np.ndarray) -> Image.Image:
    """Grayscale ``uint8`` numpy array -> PIL.Image (mode ``L``)."""
    return Image.fromarray(array)


# --------------------------------------------------------------------------- #
# Deskew
# --------------------------------------------------------------------------- #
def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate the page skew angle in degrees.

    Primary method: binarize (Otsu, inverted so text is foreground), collect the
    coordinates of all foreground pixels and fit a ``cv2.minAreaRect``; its angle
    is the dominant orientation of the text mass. A ``cv2.HoughLinesP`` pass is
    used as a fallback / sanity check when minAreaRect is degenerate.

    Returns a value in roughly ``(-45, 45]`` degrees. Positive = rotated
    counter-clockwise (needs a clockwise correction).
    """
    # Foreground = dark text on light background -> invert so text is white.
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 20:  # almost empty image, nothing to deskew
        return 0.0

    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    # OpenCV reports the angle in [-90, 0); normalize to a small correction.
    if angle < -45:
        angle = 90 + angle
    # minAreaRect on (row, col) coords is rotated 90deg vs image axes for wide
    # text blocks; keep the small-angle interpretation.
    if angle > 45:
        angle = angle - 90

    # Hough fallback if minAreaRect produced a suspiciously large tilt.
    if abs(angle) > 30:
        hough = _hough_skew_angle(binary)
        if hough is not None:
            angle = hough
    return float(angle)


def _hough_skew_angle(binary: np.ndarray) -> float | None:
    """Median angle of near-horizontal Hough line segments, or None."""
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=binary.shape[1] // 3, maxLineGap=20,
    )
    if lines is None:
        return None
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 < a < 45:  # keep roughly-horizontal text lines only
            angles.append(a)
    if not angles:
        return None
    return float(np.median(angles))


def deskew(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """Rotate ``gray`` so the dominant text orientation is horizontal.

    Returns ``(rotated_image, applied_angle_degrees)``. The border is filled
    with white (255) so it does not introduce spurious black blobs for OCR.
    """
    angle = estimate_skew_angle(gray)
    if abs(angle) < 0.1:
        return gray, 0.0
    h, w = gray.shape
    center = (w // 2, h // 2)
    # estimate_skew_angle returns the *current* tilt of the text; to make it
    # horizontal we rotate by the opposite sign (verified empirically: rotating
    # by +angle doubles the skew, -angle straightens it to ~0 residual).
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, -angle


# --------------------------------------------------------------------------- #
# Adaptive thresholding pipeline
# --------------------------------------------------------------------------- #
def adaptive_preprocess(
    gray: np.ndarray,
    do_deskew: bool = True,
    block_size: int = 31,
    c: int = 15,
) -> np.ndarray:
    """Full OpenCV preprocessing for photos / scans.

    Steps: bilateral denoise (edge-preserving) -> optional deskew ->
    ``cv2.adaptiveThreshold`` (Gaussian, local binarization). ``block_size``
    must be odd; ``c`` is subtracted from the local mean (higher = more
    aggressive at removing background).
    """
    # Edge-preserving denoise keeps glyph strokes sharp while killing speckle.
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    if do_deskew:
        denoised, _ = deskew(denoised)

    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c,
    )
    return binary


def adaptive_preprocess_pil(image: Image.Image, **kwargs) -> Image.Image:
    """PIL-in / PIL-out wrapper around :func:`adaptive_preprocess`."""
    gray = pil_to_cv(image)
    return cv_to_pil(adaptive_preprocess(gray, **kwargs))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        src = Image.open(sys.argv[1])
        out = adaptive_preprocess_pil(src)
        out_path = "preprocessed_out.png"
        out.save(out_path)
        ang = estimate_skew_angle(pil_to_cv(src))
        print(f"estimated skew: {ang:.2f} deg  ->  saved {out_path}")
    else:
        print("usage: python preprocessing.py <image>")
