# OCR + Automatic Translation

> *Academic / portfolio project — Applied AI.*

An **Optical Character Recognition** pipeline that extracts text from images and **translates it automatically**.

## Pipeline
1. **OCR** — extracts text from the image with **Tesseract** (`pytesseract`) — supports `por` / `eng`.
2. **(optional) Models** — **`transformers`** (HuggingFace) for vision / language tasks.
3. **Translation** — automatic translation of the extracted text with **`deep-translator`**.

## Setup
- Install Tesseract OCR on your system (Windows: UB-Mannheim installer; Linux: `apt install tesseract-ocr`) plus the `por` language pack.
- `pip install -r requirements.txt`

## How to run
```bash
jupyter notebook ocr_translation.ipynb
```
Point it at an input image and choose the OCR / translation languages.

## Roadmap (possible extensions)
- Image **pre-processing** (deskew / threshold / denoise with OpenCV) to improve OCR accuracy.
- Compare Tesseract vs a **TrOCR** (transformers) model and measure CER / WER.
- Package as an API (FastAPI) + demo.

## Stack
Python · pytesseract (Tesseract) · transformers · deep-translator
