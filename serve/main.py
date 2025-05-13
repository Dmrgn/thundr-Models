from __future__ import annotations

"""
* Shared `aiohttp.ClientSession` + semaphore‑capped concurrency
* Heavy TensorFlow work is run in a threadpool – event loop stays free
* Pure‑TF preprocessing on the hot path; Pillow only when absolutely required
* Vectorised text padding; compiled regex
* Structured logging
"""

import asyncio
import io
import json
import logging
import os
import re
from typing import Any, Final

import aiohttp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration (override via env vars)
# ---------------------------------------------------------------------------
MAX_SEQ_LEN: Final[int] = int(os.getenv("MAX_SEQ_LEN", "200"))
IMAGE_SIZE: Final[int] = int(os.getenv("IMAGE_SIZE", "400"))  # square
IMAGE_CHANNELS: Final[int] = 3
DOWNLOAD_CONCURRENCY: Final[int] = int(os.getenv("DOWNLOAD_CONCURRENCY", "32"))
DUPLICATE_THRESHOLD: Final[float] = float(os.getenv("DUPLICATE_THRESHOLD", "0.07"))
USE_GPU: Final[bool] = os.getenv("USE_GPU", "true").lower() in {"1", "true", "yes"}

if not USE_GPU:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("thundrmodels")

# ---------------------------------------------------------------------------
# Load models once at process start‑up
# ---------------------------------------------------------------------------
textzap_labels = [
    "company overview",
    "company services",
    "client review",
    "other",
    "contact info",
]
imagezap_labels = [
    "certification",
    "icons",
    "logo",
    "other",
    "people",
    "project image",
]

logger.info("Loading TensorFlow models …")
textzap = tf.keras.models.load_model("./models/textzap.keras")
imagezap = tf.keras.models.load_model("./models/imagezap.keras")

# warm‑up – avoids first‑request latency spike
_ = textzap.predict(np.zeros((1, MAX_SEQ_LEN), np.int32))
_ = imagezap.predict(tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS), tf.float32))
logger.info("Models loaded & warmed‑up ✅")

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
with open("./data/vocab.json", "r", encoding="utf-8") as f:
    tokenizer = json.load(f)
word_index: dict[str, int] = tokenizer.get("word_index", {})

_TOKEN_FILTER = re.compile(r"[!\"#$%&()*+,\-./:;<=?>@\[\\\]^_`{|}~\t\n]")

def text_to_sequence(text: str) -> list[int]:
    tokens = _TOKEN_FILTER.sub(" ", text.lower()).split()
    return [word_index.get(tok, 0) for tok in tokens][:MAX_SEQ_LEN]

def pad_sequences(seqs: list[list[int]], max_len: int) -> np.ndarray:
    batch = np.zeros((len(seqs), max_len), np.int32)
    for i, seq in enumerate(seqs):
        batch[i, : len(seq)] = seq
    return batch

# ---------------------------------------------------------------------------
# Image preprocessing – TF first, Pillow fallback for WebP + exotic formats
# ---------------------------------------------------------------------------
@tf.function(reduce_retracing=True)
def _tf_preprocess(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.convert_image_dtype(img, tf.float32)  # uint8 → float32 [0,1]
    return tf.image.resize_with_pad(img, IMAGE_SIZE, IMAGE_SIZE,
                                    method=tf.image.ResizeMethod.LANCZOS3)

def _decode_via_pillow(b: bytes) -> tf.Tensor:
    with Image.open(io.BytesIO(b)) as im:
        # for animated GIF/WebP just take first frame; Pillow does that by default
        im = im.convert("RGB")
        return tf.convert_to_tensor(np.asarray(im, dtype=np.uint8))

def preprocess_image_bytes(blob: bytes) -> tuple[tf.Tensor, int]:
    """Decode *blob* → `(tensor, area)`.

    Fast path: `tf.io.decode_image`. On failure (e.g., WebP) we fall back to Pillow.
    """
    try:
        decoded = tf.io.decode_image(blob, channels=3, expand_animations=True)
    except tf.errors.InvalidArgumentError:
        decoded = _decode_via_pillow(blob)
    # if animated (rank 4) → first frame
    if tf.rank(decoded) == 4:
        decoded = decoded[0]
    h, w = int(decoded.shape[0]), int(decoded.shape[1])
    return _tf_preprocess(decoded), w * h

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="thundrmodels‑server (optimised)")

class TextRequest(BaseModel):
    requests: list[str]

class ImageRequest(BaseModel):
    images: list[str]

# ---------------------------------------------------------------------------
# Startup / shutdown – shared HTTP session & semaphore
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    app.state.session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30), trust_env=True
    )
    app.state.sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    logger.info("HTTP session ready • max %d concurrent downloads", DOWNLOAD_CONCURRENCY)

@app.on_event("shutdown")
async def _shutdown() -> None:
    await app.state.session.close()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _fetch_image(
    url: str,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    *,
    max_retries: int = 4,
    backoff: float = 1.5,
) -> bytes | Exception:
    retry_statuses = {429, 502, 503, 504}
    async with sem:
        for attempt in range(max_retries + 1):
            try:
                async with session.get(url) as res:
                    if res.status == 200:
                        return await res.read()
                    if res.status in retry_statuses:
                        await asyncio.sleep(backoff * 2**attempt)
                        continue
                    return Exception(f"HTTP {res.status} – {url}")
            except aiohttp.ClientError as exc:
                err: Exception = exc
                await asyncio.sleep(backoff * 2**attempt)
        return err  # type: ignore[return-value]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def health() -> str:  # liveness probe
    return "thundrmodels server (optimised)"

@app.post("/textzap")
async def textzap_route(payload: TextRequest) -> list[dict[str, float]]:
    batch = pad_sequences([text_to_sequence(t) for t in payload.requests], MAX_SEQ_LEN)
    preds = await run_in_threadpool(textzap.predict, batch)
    return [
        {label: float(p[i]) for i, label in enumerate(textzap_labels)}
        for p in preds.tolist()
    ]

@app.post("/imagezap")
async def imagezap_route(request: Request, payload: ImageRequest) -> list[dict[str, float]]:
    urls = payload.images
    session: aiohttp.ClientSession = request.app.state.session
    sem: asyncio.Semaphore = request.app.state.sem

    # 1) download concurrently
    downloads = await asyncio.gather(
        *(_fetch_image(u, session, sem) for u in urls), return_exceptions=True
    )

    # 2) preprocess & collect metadata
    records: list[dict[str, Any]] = []
    for idx, data in enumerate(downloads):
        if isinstance(data, Exception):
            continue
        tensor, area = preprocess_image_bytes(data)
        records.append({"idx": idx, "tensor": tensor, "size": area})

    if not records:
        return [{label: 0.0 for label in imagezap_labels} for _ in urls]

    # 3) inference (thread‑offloaded)
    batch = tf.stack([r["tensor"] for r in records])
    preds = await run_in_threadpool(imagezap.predict, batch)
    for rec, p in zip(records, preds.tolist()):
        rec["pred"] = np.array(p, dtype=np.float32)

    # 4) duplicate suppression (largest area keeps scores)
    records.sort(key=lambda r: r["size"], reverse=True)
    assigned: set[int] = set()
    results = [{label: 0.0 for label in imagezap_labels} for _ in urls]

    for rec in records:
        i = rec["idx"]
        if i in assigned:
            continue
        results[i] = {
            label: float(score) for label, score in zip(imagezap_labels, rec["pred"])
        }
        assigned.add(i)
        for other in records:
            j = other["idx"]
            if j in assigned:
                continue
            if np.linalg.norm(rec["pred"] - other["pred"]) < DUPLICATE_THRESHOLD:
                assigned.add(j)

    return results

# ---------------------------------------------------------------------------
# ASGI entry‑point (uvicorn) – for standalone runs
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "thundrmodels_server_optimized:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        loop="uvloop",
        workers=1,
    )
