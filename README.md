# OCR + Automatic Translation

Pipeline de **Optical Character Recognition** que extrai texto de imagens e o **traduz automaticamente**.

## Pipeline
1. **OCR** — extrai texto da imagem com **Tesseract** (`pytesseract`) — suporta `por`/`eng`.
2. **(opcional) Modelos** — uso de **`transformers`** (HuggingFace) p/ tarefas de visão/linguagem.
3. **Tradução** — tradução automática do texto com **`deep-translator`**.

## Setup
- Instale o Tesseract OCR no sistema (Windows: instalador UB-Mannheim; Linux: `apt install tesseract-ocr`) + o idioma `por`.
- `pip install -r requirements.txt`

## Como rodar
```bash
jupyter notebook ocr_translation.ipynb
```
Aponte para uma imagem de entrada e escolha os idiomas de OCR/tradução.

## Possíveis extensões
- **Pré-processamento** da imagem (deskew/threshold/denoise com OpenCV) p/ melhorar a acurácia do OCR.
- Comparar Tesseract vs um modelo **TrOCR** (transformers) e medir CER/WER.
- Empacotar como API (FastAPI) + demo.

> Notebook acadêmico/portfólio (IA aplicada). Outputs limpos; imagens/modelos não versionados.
