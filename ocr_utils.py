# -*- coding: utf-8 -*-
"""
ocr_utils.py — OCR + image preprocessing + automatic translation.

Improvement over the original notebook: adds an **image-preprocessing step**
(grayscale → median denoise → autocontrast → binarization) that dramatically
raises OCR accuracy on noisy / low-contrast inputs. See `demo.py` for a
reproducible benchmark (raw vs preprocessed).

Requires the Tesseract binary installed on the system. On Windows it usually
lives at: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""
from __future__ import annotations
import os
from PIL import Image, ImageOps, ImageFilter
import pytesseract

# Aponta para o binário do Tesseract (ajuste conforme seu SO se necessário).
_WIN_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.exists(_WIN_DEFAULT):
    pytesseract.pytesseract.tesseract_cmd = _WIN_DEFAULT


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Grayscale → median denoise → autocontrast → binarização (threshold).

    Em imagens ruidosas/baixo contraste isso costuma levar o OCR de
    ilegível para texto perfeito (ver benchmark em demo.py)."""
    img = image.convert("L")                       # grayscale
    img = img.filter(ImageFilter.MedianFilter(3))  # remove ruído sal-e-pimenta
    img = ImageOps.autocontrast(img)               # normaliza contraste
    img = img.point(lambda x: 0 if x < 128 else 255)  # binariza
    return img


def ocr_image(src, lang: str = "eng", preprocess: bool = True) -> str:
    """Extrai texto de uma imagem (caminho ou PIL.Image).

    lang: idioma(s) do Tesseract, ex. 'eng', 'por', 'por+eng'
          (requer o language data instalado no tessdata)."""
    img = Image.open(src) if isinstance(src, (str, bytes, os.PathLike)) else src
    if preprocess:
        img = preprocess_for_ocr(img)
    return pytesseract.image_to_string(img, lang=lang).strip()


def translate_text(text: str, target: str = "en", source: str = "auto") -> str:
    """Traduz o texto via deep-translator (Google Translate).
    Import tardio p/ não exigir rede/lib quando só se usa OCR."""
    from deep_translator import GoogleTranslator
    if not text.strip():
        return ""
    return GoogleTranslator(source=source, target=target).translate(text)


def ocr_and_translate(src, ocr_lang: str = "eng", target: str = "en",
                      preprocess: bool = True) -> dict:
    """Pipeline completo: OCR → tradução. Retorna {'text', 'translation'}."""
    text = ocr_image(src, lang=ocr_lang, preprocess=preprocess)
    return {"text": text, "translation": translate_text(text, target=target)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(ocr_and_translate(sys.argv[1]))
    else:
        print("uso: python ocr_utils.py <imagem>")
