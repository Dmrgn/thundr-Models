# thundr Models

**thundrmodels** combines:

1. A Python/TensorFlow backend for building and training two Tensorflow Keras models:
  - **textzap**: a 3.5M parameter text classification model trained from `data/text_pairs_dataset.jsonl` via Jupyter notebooks
    - Distilled from [Llama 3 8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B) over 6,000 samples for 88% accuracy.
  - **imagezap**: a 38M parameter image classifier trained on a manually sorted dataset under `data/sorted_images/…`
    - Trained on a manually sorted dataset of 4,000 samples for 81% accuracy.
2. A lightweight Bun+TypeScript web app (port 3000) for manually sorting unsorted images into category folders.


## Features

- Data preparation & model training in Jupyter notebooks (`notebooks/`)
- Trained models saved in `models/` (`textzap.keras`, `imagezap.keras`)
- Manual Image Sorter UI served at http://localhost:3000
- Simple HTTP API:
  - GET  `/images?sourceDir=…&destDir=…` → list images and destination subfolders
  - GET  `/image?sourceDir=…&filename=…` → fetch raw image bytes
  - POST `/move-image` → move a file into a chosen category folder
- GPU‑enabled TensorFlow support

## Repository Layout
```
.
├── README.md             # This overview
├── pyproject.toml        # Python project (TensorFlow, Pillow, ...)
├── package.json          # Bun/TypeScript project
├── bun.lock              # Bun lockfile
├── uv.lock               # Python lockfile
├── models/               # Exported Keras models
├── data/                 # Raw images, sorted images, text datasets
├── notebooks/            # Jupyter notebooks for prep & training
├── src/ts/               # Bun+TS HTTP server for image sorter
├── public/               # Front‑end assets (HTML, JS)
├── tests/                # Stubbed tests (python, ts)
└── docs/                 # Developer guide & contribution notes
```

## Prerequisites

- Python ≥ 3.11
- Bun ≥ 1.2+
- (Optional) GPU + CUDA for TensorFlow

## Installation

### Python Dependencies
```bash
uv sync
```

### TypeScript Dependencies
```bash
bun install
```

## Usage

### 1. Prepare & Train Models
```bash
# Launch JupyterLab and run the notebooks:
jupyter lab notebooks/
```

- `notebooks/text_prep.ipynb` → trains `models/textzap.keras`
- `notebooks/image_prep.ipynb` → trains `models/imagezap.keras`

### 2. Start the Manual Image Sorter UI
```bash
bun run src/ts/index.ts
```

Open your browser at http://localhost:3000, enter a source folder (raw images) and a destination root (sorted images), then step through and categorize each image.