from pydantic import BaseModel
import numpy as np
import os
import json
import re
from fastapi import FastAPI
from pathlib import Path
import uuid
import io
import asyncio
import aiohttp

from typing_extensions import Annotated
from fastapi import FastAPI, File, UploadFile

from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import easyocr
# reader = easyocr.Reader(['en'])
import tensorflow as tf

textzap_labels = ['company overview', 'company services', 'client review', 'other', 'contact info']
imagezap_labels = ['certification', 'icons', 'logo', 'other', 'people', 'project image']
textzap = tf.keras.models.load_model("./models/textzap.keras")
imagezap = tf.keras.models.load_model("./models/imagezap.keras")

class TextRequest(BaseModel):
    requests: list[str]

class ImageRequest(BaseModel):
    images: list[str]

MAX_SEQ_LEN = 200
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
IMAGE_CHANNELS = 3

def create_text_processor(word_index, max_length):
    filters = r'[!"#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~\t\n]'
    def text_to_sequence(text):
        cleaned = re.sub(filters, ' ', text.lower())
        tokens = [w for w in cleaned.split() if w]
        seq = [word_index.get(w, 0) for w in tokens]
        seq = [i for i in seq if i != 0][:max_length]
        return seq
    return text_to_sequence
def pad_sequence(seq, max_length):
    padded = np.zeros(max_length, dtype=np.int32)
    length = min(len(seq), max_length)
    padded[:length] = seq[:length]
    return padded
with open('./data/vocab.json', 'r', encoding='utf-8') as f:
    tokenizer = json.load(f)
word_index = tokenizer.get('word_index', {})
text_to_sequence = create_text_processor(word_index, MAX_SEQ_LEN)

import aiohttp
import asyncio

async def download_image( url: str, max_retries: int = 4, backoff_factor: float = 1.5) -> bytes | str:
    retry_statuses = {429, 502, 503, 504}
    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        return image_bytes
                    if response.status in retry_statuses:
                        delay = backoff_factor * (2 ** attempt)
                        await asyncio.sleep(delay)
                        print(f"error occured, retrying {attempt}")
                        continue
                    # non-retryable HTTP error, they don't like us anymore :(
                    raise Exception
        except aiohttp.ClientError:
            # network error – treat as retryable
            delay = backoff_factor * (2 ** attempt)
            print(f"client error occured, retrying {attempt}")
            await asyncio.sleep(delay)
            continue
    print(f"error occured, unable to retry")
    # all retries exhausted
    raise Exception

def preprocess_image_bytes(image_bytes: bytes) -> tf.Tensor:
    # resize and set pink background
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGBA")
        bg = Image.new("RGB", (400, 400), (255, 39, 255))
        img.thumbnail((400, 400), Image.LANCZOS)
        left = (400 - img.width) // 2
        top  = (400 - img.height) // 2
        bg.paste(img, (left, top), img)
        buf = io.BytesIO()
        bg.save(buf, format="PNG")
        tensor = tf.image.decode_image(buf.getvalue(), channels=IMAGE_CHANNELS)
        tensor = tf.image.resize(tensor, [IMAGE_WIDTH, IMAGE_HEIGHT])
        return tf.cast(tensor, tf.float32) / 255.0

app = FastAPI()

@app.get("/")
def read_root():
    return "thundrmodels server"

@app.post("/textzap")
def textzap_route(text_request: TextRequest):
    processed = []
    for request in text_request.requests:
        seq = text_to_sequence(request)
        processed.append(pad_sequence(seq, MAX_SEQ_LEN))
    predictions = textzap.predict(np.array(processed)).tolist()
    results = []
    for prediction in predictions:
        obj = {}
        for i in range(len(prediction)):
            obj[textzap_labels[i]] = prediction[i]
        results.append(obj)
    return results

@app.post("/imagezap")
async def imagezap_route(image_request: ImageRequest):
    urls = image_request.images

    # 1) download in parallel
    coros     = [download_image(u) for u in urls]
    downloads = await asyncio.gather(*coros, return_exceptions=True)

    # 2) build a record for each successful download,
    #    storing index, raw bytes, resolution, and preprocessed tensor
    records = []
    for idx, result in enumerate(downloads):
        if isinstance(result, Exception):
            continue

        image_bytes = result
        # get original size
        with Image.open(io.BytesIO(image_bytes)) as img:
            w, h = img.size

        # preprocess to a TF tensor
        tensor = preprocess_image_bytes(image_bytes)

        records.append({
            "idx": idx,
            "bytes": image_bytes,
            "size": w * h,
            "tensor": tensor
        })

    # if nothing downloaded, return all-zero labels
    if not records:
        return [
            {label: 0.0 for label in imagezap_labels}
            for _ in urls
        ]

    # 3) batch-predict
    batch = tf.stack([r["tensor"] for r in records], axis=0)
    preds = imagezap.predict(batch).tolist()
    for rec, p in zip(records, preds):
        rec["pred"] = np.array(p)

    # 4) cluster by prediction distance, picking largest first
    threshold = 0.07  # tune this: max distance to call "duplicate"
    # sort desc by area
    records.sort(key=lambda r: r["size"], reverse=True)

    assigned = set()
    clusters = []  # each cluster is list of recs

    for rec in records:
        if rec["idx"] in assigned:
            continue

        # this becomes the cluster’s "rep"
        rep = rec
        cluster = [rep]
        assigned.add(rep["idx"])

        # any unassigned that are "close" to rep?
        for other in records:
            if other["idx"] in assigned:
                continue
            dist = np.linalg.norm(rep["pred"] - other["pred"])
            if dist < threshold:
                cluster.append(other)
                assigned.add(other["idx"])

        clusters.append(cluster)

    # 5) build final results: reps keep their scores, others zero out
    zero_scores = {label: 0.0 for label in imagezap_labels}
    results = [zero_scores.copy() for _ in urls]

    for cluster in clusters:
        # the first element is the largest
        rep = cluster[0]
        results[rep["idx"]] = {
            label: float(score)
            for label, score in zip(imagezap_labels, rep["pred"])
        }

    return results