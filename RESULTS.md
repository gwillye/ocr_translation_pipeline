# OCR Benchmark Results

These numbers come from running `benchmark.py` over 36 ground-truthed single-line images. The set is built from 12 sentences rendered under 3 different conditions, which gives the 36 total. The metrics are computed with the jiwer backend, and both CER and WER are case-insensitive (there is a note at the bottom explaining why). For all of these, lower is better.

## Overall (all conditions)

This table averages every engine and preprocessing combination across all 36 images.

| Engine / preprocessing | N | CER | WER | latency (ms/img) |
|---|--:|--:|--:|--:|
| Tesseract (raw) | 36 | 0.457 | 0.477 | 250 |
| Tesseract (PIL global threshold) | 36 | 0.147 | 0.218 | 229 |
| Tesseract (OpenCV adaptive + deskew) | 36 | 0.031 | 0.095 | 252 |
| TrOCR (base-printed) | 36 | 0.014 | 0.069 | 8088 |

A couple of things stand out here. Running Tesseract on the raw images is by far the worst, with a CER of 0.457 and WER of 0.477. Adding a simple PIL global threshold cuts the CER down to 0.147, and switching to OpenCV adaptive thresholding plus deskew brings it all the way to 0.031. TrOCR is the most accurate at 0.014 CER, but notice the latency column: it takes 8088 ms per image, roughly 32 times slower than any of the Tesseract variants, which all sit around 229 to 252 ms.

## Per-condition CER (the OpenCV effect)

Splitting the CER out by condition (clean, degraded, skewed) makes it clear where the preprocessing actually earns its keep.

| Engine / preprocessing | clean | degraded | skewed |
|---|--:|--:|--:|
| Tesseract (raw) | 0.002 | 0.402 | 0.967 |
| Tesseract (PIL global threshold) | 0.000 | 0.018 | 0.422 |
| Tesseract (OpenCV adaptive + deskew) | 0.002 | 0.000 | 0.091 |
| TrOCR (base-printed) | 0.013 | 0.015 | 0.013 |

On clean images everything does fine (the worst is TrOCR at 0.013, and even raw Tesseract is at 0.002). The interesting part is the degraded and skewed columns. Raw Tesseract basically falls apart on skewed text, hitting 0.967 CER, which means it is getting almost everything wrong. The global threshold helps a lot on degraded images (0.402 down to 0.018) but still struggles with skew (0.422). The OpenCV adaptive plus deskew path is the one that handles both: 0.000 on degraded and 0.091 on skewed. TrOCR stays remarkably steady across all three conditions (0.013, 0.015, 0.013), which is the whole point of a learned model, but again you pay for that robustness in latency.

> CER and WER are computed case-insensitively on purpose. The `microsoft/trocr-base-printed` model emits ALL-CAPS text, so a case-sensitive metric would punish it for a difference that is purely cosmetic. Tesseract preserves the original case either way, so making the comparison case-insensitive keeps things fair.
