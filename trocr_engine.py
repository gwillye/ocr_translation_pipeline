# -*- coding: utf-8 -*-
"""
trocr_engine.py — transformer OCR backend (microsoft/trocr-base-printed).

This is the "Compare Tesseract vs a transformer model (TrOCR)" extension. It
wraps HuggingFace's ``VisionEncoderDecoderModel`` so the benchmark
(``benchmark.py``) and the FastAPI service can call a single ``read(image)``
function with the same signature as ``ocr_utils.ocr_image``.

TrOCR is a *single-line* recognizer: its training data is one text line per
image. It is NOT a page-layout/segmentation model. For multi-line pages you
must segment lines first and run TrOCR per line — for our single-line
ground-truth sample that distinction does not matter, and the README documents
it as a limitation.

The model (~1.4 GB) is downloaded from the HuggingFace hub on first use and
cached under ~/.cache/huggingface. The class is lazy: importing this module is
cheap and never touches the network; the model loads on first ``TrOCREngine``
instantiation. If transformers/torch are missing the import error is surfaced
with a clear message so the rest of the pipeline keeps working.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Union

from PIL import Image

MODEL_NAME = "microsoft/trocr-base-printed"


class TrOCREngine:
    """Thin wrapper around microsoft/trocr-base-printed.

    Parameters
    ----------
    model_name:
        HuggingFace model id. Defaults to the printed-text checkpoint.
    device:
        ``"cpu"`` or ``"cuda"``. Defaults to CUDA when available else CPU.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except Exception as exc:  # pragma: no cover - missing optional deps
            raise ImportError(
                "TrOCR requires 'torch' and 'transformers'. "
                "Install them with: pip install torch transformers"
            ) from exc

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def read(self, image: Union[str, Image.Image]) -> str:
        """Recognize the text in a single-line image. Returns the string."""
        return self.read_batch([image])[0]

    def read_batch(self, images: List[Union[str, Image.Image]]) -> List[str]:
        """Recognize a batch of single-line images (one string each)."""
        pil_images = [
            Image.open(im).convert("RGB") if isinstance(im, str) else im.convert("RGB")
            for im in images
        ]
        pixel_values = self.processor(
            images=pil_images, return_tensors="pt"
        ).pixel_values.to(self.device)
        with self._torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=64)
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [t.strip() for t in texts]


@lru_cache(maxsize=1)
def get_engine(model_name: str = MODEL_NAME) -> "TrOCREngine":
    """Process-wide cached engine so the 1.4 GB model loads only once."""
    return TrOCREngine(model_name=model_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(get_engine().read(sys.argv[1]))
    else:
        print("usage: python trocr_engine.py <line-image>")
