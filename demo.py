# -*- coding: utf-8 -*-
"""
demo.py — benchmark reproduzível: OCR SEM vs COM preprocessing numa imagem ruidosa.
Gera a imagem de teste, roda os dois caminhos e imprime a acurácia (SequenceMatcher).
Resultado típico: RAW ~0.00 (ilegível) → PREPROCESSED ~1.00.
"""
import difflib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ocr_utils


def _noisy_image(text: str) -> Image.Image:
    img = Image.new("L", (680, 110), 200)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except Exception:
        font = ImageFont.load_default()
    d.text((12, 38), text, fill=110, font=font)          # baixo contraste
    arr = np.array(img).astype(int) + np.random.default_rng(0).integers(-45, 45, (110, 680))
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def _acc(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()


def main():
    gt = "Analise de dados e OCR aplicado em 2026"
    img = _noisy_image(gt)
    raw = ocr_utils.ocr_image(img, lang="eng", preprocess=False)
    prep = ocr_utils.ocr_image(img, lang="eng", preprocess=True)
    print(f"Ground truth : {gt}")
    print(f"RAW OCR      : {raw!r}   acc={_acc(raw, gt):.2f}")
    print(f"PREPROCESSED : {prep!r}   acc={_acc(prep, gt):.2f}")
    print("\n-> O preprocessing (denoise + autocontraste + binarizacao) recupera o texto"
          " que o OCR cru nao consegue ler.")


if __name__ == "__main__":
    main()
