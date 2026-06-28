# -*- coding: utf-8 -*-
"""
sample_data.py — generate a small, reproducible ground-truthed OCR sample.

We render known strings to single-line images with real system fonts so we have
*exact* ground truth (no manual transcription, no copyright issues, fully
reproducible from a fixed RNG seed). Three conditions are produced per line so
the benchmark can measure both engines and the OpenCV preprocessing effect:

  * ``clean``    — black text on white, light Gaussian blur (an easy baseline).
  * ``degraded`` — low contrast + Gaussian noise + uneven (gradient) lighting.
                   This is where a single global threshold struggles and OpenCV
                   adaptive thresholding is expected to help.
  * ``skewed``   — the degraded image rotated by a known angle, to exercise the
                   deskew step.

Everything is deterministic given ``seed`` so the numbers in the README can be
reproduced exactly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Fixed sentences (printed-text domain). Mix of words, digits and punctuation.
SENTENCES: List[str] = [
    "Invoice number 2026-0481 total 1530.00",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning models improve OCR accuracy",
    "Order shipped on March 14 to Sao Paulo Brazil",
    "Contact support at help@example.com today",
    "Temperature reached 37.5 degrees at noon",
    "Chapter 7 Neural Networks and Deep Learning",
    "Receipt 998877 paid by card ending 4242",
    "Adaptive thresholding handles uneven lighting",
    "Deskew corrects pages rotated by small angles",
    "Python 3.12 powers this analysis pipeline",
    "Gross revenue grew 18 percent year over year",
]

# A handful of real fonts so the model is not memorising one glyph set.
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
    r"C:\Windows\Fonts\georgia.ttf",
]


@dataclass
class Sample:
    text: str          # ground truth
    image: Image.Image  # rendered (grayscale 'L') image
    condition: str     # 'clean' | 'degraded' | 'skewed'
    angle: float       # applied skew angle (0 for non-skewed)


def _load_fonts(size: int = 32) -> List[ImageFont.FreeTypeFont]:
    fonts = []
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            fonts.append(ImageFont.truetype(path, size))
    if not fonts:  # pragma: no cover - bitmap fallback if no TTF fonts exist
        fonts.append(ImageFont.load_default())
    return fonts


def _render_line(text: str, font: ImageFont.FreeTypeFont, pad: int = 16) -> Image.Image:
    """Render one line of black text on white, sized to fit."""
    tmp = Image.new("L", (10, 10), 255)
    box = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    img = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
    ImageDraw.Draw(img).text((pad - box[0], pad - box[1]), text, fill=0, font=font)
    return img


def _degrade(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Low contrast + Gaussian noise + gradient (uneven) lighting."""
    arr = np.asarray(img).astype(np.float32)
    # Compress dynamic range: ink -> ~90, paper -> ~190 (low contrast).
    arr = 90 + (arr / 255.0) * (190 - 90)
    # Uneven lighting: horizontal brightness gradient (a soft shadow).
    h, w = arr.shape
    gradient = np.linspace(-35, 35, w)[None, :].repeat(h, axis=0)
    arr = arr + gradient
    # Sensor noise.
    arr = arr + rng.normal(0, 12, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def _skew(img: Image.Image, angle: float) -> Image.Image:
    """Rotate by ``angle`` degrees, white background."""
    return img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)


def build_dataset(seed: int = 0) -> List[Sample]:
    """Return the deterministic list of Samples (clean/degraded/skewed each)."""
    rng = np.random.default_rng(seed)
    fonts = _load_fonts()
    samples: List[Sample] = []
    for i, text in enumerate(SENTENCES):
        font = fonts[i % len(fonts)]
        base = _render_line(text, font)

        clean = base.filter(ImageFilter.GaussianBlur(0.4))
        samples.append(Sample(text, clean, "clean", 0.0))

        degraded = _degrade(base, rng)
        samples.append(Sample(text, degraded, "degraded", 0.0))

        angle = float(rng.uniform(-8, 8))
        skewed = _skew(degraded, angle)
        samples.append(Sample(text, skewed, "skewed", angle))
    return samples


def save_dataset(out_dir: str, seed: int = 0) -> List[Sample]:
    """Build and write the sample images to ``out_dir`` (for inspection)."""
    os.makedirs(out_dir, exist_ok=True)
    samples = build_dataset(seed)
    for idx, s in enumerate(samples):
        s.image.save(os.path.join(out_dir, f"{idx:02d}_{s.condition}.png"))
    with open(os.path.join(out_dir, "ground_truth.tsv"), "w", encoding="utf-8") as fh:
        fh.write("index\tcondition\tangle\ttext\n")
        for idx, s in enumerate(samples):
            fh.write(f"{idx:02d}\t{s.condition}\t{s.angle:.2f}\t{s.text}\n")
    return samples


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "sample_images"
    saved = save_dataset(out)
    print(f"wrote {len(saved)} images + ground_truth.tsv to {out}/")
