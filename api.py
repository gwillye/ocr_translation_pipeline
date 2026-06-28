# -*- coding: utf-8 -*-
"""
api.py — FastAPI service wrapping the OCR pipeline.

Endpoints
---------
  GET  /            -> the static web demo (static/index.html)
  GET  /health      -> {"status": "ok", ...}
  POST /ocr         -> multipart image upload -> {"engine", "preprocess",
                       "text", "translation"?}

Query / form params for POST /ocr
  engine      : "tesseract" (default) | "trocr"
  preprocess  : "none" | "pil" | "opencv"  (Tesseract only; default "opencv")
  lang        : Tesseract language(s), default "eng"
  translate_to: optional target language; if set, also returns a translation

The TrOCR model is loaded lazily on first use so the server starts instantly and
a Tesseract-only deployment never needs torch. Run with:

    uvicorn api:app --reload --port 8000

then open http://127.0.0.1:8000/ .
"""
from __future__ import annotations

import io
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

import ocr_utils
import preprocessing

app = FastAPI(
    title="OCR + Preprocessing API",
    description="Tesseract / TrOCR OCR with OpenCV preprocessing.",
    version="1.0.0",
)


def _run_tesseract(image: Image.Image, preprocess: str, lang: str) -> str:
    if preprocess == "opencv":
        image = preprocessing.adaptive_preprocess_pil(image, do_deskew=True)
        return ocr_utils.ocr_image(image, lang=lang, preprocess=False)
    if preprocess == "pil":
        return ocr_utils.ocr_image(image, lang=lang, preprocess=True)
    if preprocess == "none":
        return ocr_utils.ocr_image(image, lang=lang, preprocess=False)
    raise HTTPException(400, f"unknown preprocess '{preprocess}'")


def _run_trocr(image: Image.Image) -> str:
    try:
        from trocr_engine import get_engine
    except ImportError as exc:
        raise HTTPException(
            503, f"TrOCR backend unavailable: {exc}"
        ) from exc
    return get_engine().read(image)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ocr", "version": app.version}


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    engine: str = Form("tesseract"),
    preprocess: str = Form("opencv"),
    lang: str = Form("eng"),
    translate_to: Optional[str] = Form(None),
) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty file")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(400, "uploaded file is not a readable image")

    engine = engine.lower()
    if engine == "tesseract":
        text = _run_tesseract(image, preprocess.lower(), lang)
    elif engine == "trocr":
        text = _run_trocr(image)
        preprocess = "n/a (model-internal)"
    else:
        raise HTTPException(400, f"unknown engine '{engine}'")

    result = {"engine": engine, "preprocess": preprocess, "text": text}
    if translate_to:
        try:
            result["translation"] = ocr_utils.translate_text(text, target=translate_to)
        except Exception as exc:  # network / deep-translator missing
            result["translation_error"] = str(exc)
    return result


# Static web demo. Mounting at the end so /ocr and /health take precedence.
import os  # noqa: E402

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
