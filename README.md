# OCR + Preprocessing + Translation — Tesseract vs TrOCR

An applied-AI portfolio project that extracts text from images, **measures** OCR
quality with CER/WER, shows how **OpenCV preprocessing** rescues hard inputs, and
ships the whole thing as a **FastAPI service with a small web demo**.

It started as a course notebook (Tesseract + automatic translation). This repo
adds the three "possible extensions" from the original write-up, each with real,
reproducible numbers — **no hand-written metrics**.

---

## Problem

Off-the-shelf OCR (Tesseract) is fast and free but degrades sharply on real-world
images: low contrast, uneven lighting/shadows, sensor noise, and skew from a
hand-held photo or a crooked scan. Two questions:

1. **How much** does a classical-CV preprocessing step (adaptive thresholding +
   deskew) actually recover, and on which failure modes?
2. Is a transformer OCR model (**TrOCR**, `microsoft/trocr-base-printed`) worth
   its cost compared to a well-preprocessed Tesseract?

To answer them honestly we need ground truth and a metric, not eyeballing.

## Methods

### Ground-truthed sample (`sample_data.py`)
Known strings are rendered to single-line images with real system fonts, so the
ground truth is **exact and reproducible** (fixed RNG seed, no copyright issues).
Each of 12 sentences is produced in three conditions:

| Condition | What it simulates |
|---|---|
| `clean` | easy baseline (black on white, slight blur) |
| `degraded` | low contrast + Gaussian noise + a brightness gradient (uneven lighting) |
| `skewed` | the degraded image rotated by a known angle in ±8° |

→ 36 images total. `python sample_data.py` writes them + `ground_truth.tsv`.

### Engines / pipelines compared
- **Tesseract (raw)** — image as-is.
- **Tesseract (PIL global threshold)** — the original notebook pipeline:
  grayscale → median denoise → autocontrast → single global threshold at 128
  (`ocr_utils.preprocess_for_ocr`).
- **Tesseract (OpenCV adaptive + deskew)** — `preprocessing.py`: edge-preserving
  bilateral denoise → **deskew** (estimate text angle via `minAreaRect`, Hough
  fallback; rotate back) → **adaptive Gaussian thresholding** (local, per-region
  binarization instead of one global cutoff).
- **TrOCR (base-printed)** — `microsoft/trocr-base-printed`, a ViT encoder +
  transformer decoder, run on the raw image (it has its own learned
  normalisation). `trocr_engine.py`.

### Metric (`metrics.py`)
- **CER** = character error rate = edit-distance(chars) / len(reference chars).
- **WER** = word error rate = edit-distance(words) / len(reference words).
- Uses **`jiwer`** when installed; otherwise a self-contained Levenshtein
  implementation (identical numbers) so the benchmark runs anywhere.
- Computed **case-insensitively**: TrOCR-base-printed emits ALL-CAPS, so a
  case-sensitive metric would penalise it for a purely cosmetic difference.

`python benchmark.py` runs every combination and writes `RESULTS.md`.

---

## Results (real numbers)

Produced by running `benchmark.py` on this machine (Windows, **CPU-only torch**),
36 images, `jiwer` backend. Lower CER/WER is better. Full table: [`RESULTS.md`](RESULTS.md).

### Overall (all conditions)

| Engine / preprocessing | CER | WER | latency (ms/img) |
|---|--:|--:|--:|
| Tesseract (raw) | 0.457 | 0.477 | 250 |
| Tesseract (PIL global threshold) | 0.147 | 0.218 | 229 |
| **Tesseract (OpenCV adaptive + deskew)** | **0.031** | **0.095** | 252 |
| **TrOCR (base-printed)** | **0.014** | **0.069** | 8088 |

### Per-condition CER — the OpenCV effect

| Engine / preprocessing | clean | degraded | skewed |
|---|--:|--:|--:|
| Tesseract (raw) | 0.002 | 0.402 | 0.967 |
| Tesseract (PIL global threshold) | 0.000 | 0.018 | 0.422 |
| **Tesseract (OpenCV adaptive + deskew)** | 0.002 | **0.000** | **0.091** |
| TrOCR (base-printed) | 0.013 | 0.015 | 0.013 |

### Analysis
- **Preprocessing is the single biggest lever for Tesseract.** Raw Tesseract is
  almost useless on hard images (overall CER 0.457). The classical pipeline cuts
  that ~15×, to 0.031 — at essentially the same latency (~0.25 s/img).
- **The OpenCV step earns its keep exactly where the global threshold fails:**
  - *degraded* (uneven lighting): global threshold 0.018 → **adaptive 0.000**.
    A single cutoff at 128 can't cope with a brightness gradient; local
    thresholding can.
  - *skewed*: global threshold 0.422 → **deskew 0.091** (raw was 0.967). Deskew
    is what turns a near-unreadable tilted line into a readable one.
  - On *clean* images preprocessing is a no-op (≈0 either way) — it doesn't hurt.
- **TrOCR is the most accurate and the most robust** (CER ≈0.013 in *every*
  condition — it barely notices noise or skew). But on CPU it is **~32× slower**
  (8.1 s vs 0.25 s per image) and needs a 1.4 GB model.
- **Practical takeaway:** for batch/throughput on printed text, *Tesseract +
  OpenCV preprocessing* gets you to CER 0.031 cheaply; reach for TrOCR when
  accuracy on degraded/skewed input matters more than speed, or on a GPU.

> A real bug was found and fixed while building this: the original `deskew`
> rotated by `+angle` (doubling the skew). Verified empirically that `-angle`
> drives residual skew to ~0; that fix is what moved skewed CER from 0.50 → 0.09.

---

## How to run the demo

```bash
pip install -r requirements.txt
# Tesseract binary must be installed separately (see Setup below).

uvicorn api:app --reload --port 8000
# open http://127.0.0.1:8000/
```

The web page (`static/index.html`) lets you drag in an image, pick the engine
(Tesseract / TrOCR) and the Tesseract preprocessing mode (none / PIL / OpenCV),
and see the extracted text + latency.

API directly:
```bash
curl -X POST http://127.0.0.1:8000/ocr \
  -F file=@your_image.png -F engine=tesseract -F preprocess=opencv -F lang=eng
# -> {"engine":"tesseract","preprocess":"opencv","text":"..."}
```

Reproduce the benchmark:
```bash
python sample_data.py sample_images   # optional: write the images to disk
python benchmark.py --out RESULTS.md  # full run (downloads TrOCR ~1.4GB once)
python benchmark.py --no-trocr        # Tesseract + OpenCV only (no torch needed)
```

## Files
- **`ocr_utils.py`** — Tesseract wrapper, PIL preprocessing, translation, full pipeline (original).
- **`preprocessing.py`** — OpenCV adaptive threshold + deskew (extension 2).
- **`trocr_engine.py`** — `microsoft/trocr-base-printed` wrapper (extension 1).
- **`metrics.py`** — CER/WER (jiwer or built-in Levenshtein).
- **`sample_data.py`** — deterministic ground-truthed image generator.
- **`benchmark.py`** — runs all engines, writes `RESULTS.md`.
- **`api.py`** + **`static/index.html`** — FastAPI service + web demo (extension 3).
- **`demo.py`** — original raw-vs-preprocessed mini-benchmark.
- **`ocr_translation.ipynb`** — original exploratory notebook.

## Setup
1. Install **Tesseract** (Windows: UB-Mannheim installer →
   `C:\Program Files\Tesseract-OCR`; Linux: `apt install tesseract-ocr`).
   For Portuguese add the `por` language data. `ocr_utils.py` auto-points to the
   Windows default path.
2. `pip install -r requirements.txt` (Python 3.12).

## Limitations & honest notes
- **TrOCR is single-line.** `trocr-base-printed` recognises one text line per
  image; it is not a page-layout model. Our sample is single-line, so the
  comparison is fair, but for multi-line pages you must segment lines first and
  run TrOCR per line.
- **Synthetic ground truth.** The sample is rendered, not photographed. It
  isolates the failure modes (noise / lighting / skew) cleanly and is fully
  reproducible, but absolute CER on *your* photos will differ. The relative
  ordering (raw < PIL < OpenCV ≲ TrOCR) is the transferable result.
- **CPU latency.** TrOCR timings are CPU-only (`torch ... +cpu`); a GPU would cut
  the 8 s/img by an order of magnitude and change the speed trade-off.
- **Case-insensitive metric** is a deliberate choice documented above; switch it
  off via `metrics.cer(ref, hyp, case_sensitive=True)`.
- The translation step (`deep-translator`) calls Google Translate and needs
  network; it is optional and unrelated to the OCR comparison.

> Images and model weights are not versioned (`.gitignore`). 0 secrets in the repo.
