# OCR + Automatic Translation

Extracts text from images with **Tesseract OCR** and **translates it automatically**.
Includes an **image-preprocessing step that significantly improves OCR accuracy** on
noisy / low-contrast inputs.

## ✨ Improvement (measured)
On a noisy, low-contrast test image the raw OCR returns **nothing**; after preprocessing
(denoise → autocontrast → binarization) it reads the text **perfectly**:

| Path | OCR output | accuracy |
|---|---|---:|
| Raw | `''` | **0.00** |
| Preprocessed | `Analise de dados e OCR aplicado em 2026` | **1.00** |

Reproduce it: `python demo.py`.

## Files
- **`ocr_utils.py`** — clean, reusable API:
  - `preprocess_for_ocr(image)` — grayscale → median denoise → autocontrast → binarize.
  - `ocr_image(src, lang="eng", preprocess=True)` — Tesseract wrapper (path or PIL image).
  - `translate_text(text, target="en")` — translation via `deep-translator`.
  - `ocr_and_translate(src, ...)` — full OCR → translation pipeline.
- **`demo.py`** — reproducible raw-vs-preprocessed benchmark (the table above).
- **`ocr_translation.ipynb`** — original exploratory notebook (outputs cleared).

## Setup
1. Install **Tesseract** (Windows: UB-Mannheim installer → `C:\Program Files\Tesseract-OCR`; Linux: `apt install tesseract-ocr`). For Portuguese add the `por` language data.
2. `pip install -r requirements.txt`

## Usage
```python
from ocr_utils import ocr_and_translate
print(ocr_and_translate("invoice.png", ocr_lang="eng", target="en"))
```

## Possible extensions
- Compare Tesseract vs a transformer model (**TrOCR**) and measure CER/WER.
- Adaptive thresholding / deskew (OpenCV) for photos and scans.
- Wrap as a FastAPI service + small web demo.

> Portfolio project (applied AI). The preprocessing + tested utility module are the
> improvement over the original course notebook. Images/models are not versioned.
