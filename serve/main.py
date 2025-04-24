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

async def download_image(url, image_id):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            image_bytes = await response.read()
    return (image_id, image_bytes)

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
async def textzap_route(image_request: ImageRequest):
    # download images
    id_to_index = {}
    image_data = [None for x in range(len(image_request.images))]
    async with asyncio.TaskGroup() as tg:
        tasks = []
        for i, url in enumerate(image_request.images):
            image_id = uuid.uuid4()
            id_to_index[image_id] = i
            tasks.append(tg.create_task(download_image(url, image_id)))
        for downloaded_image in asyncio.as_completed(tasks):
            image_id, image_bytes = await downloaded_image
            image_data[id_to_index[image_id]] = image_bytes
    # print(image_data)
    # preprocess images
    image_tensors = []
    for image_bytes in image_data:
        with Image.open(io.BytesIO(image_bytes)) as image:
            # Ensure image is in RGBA for proper transparency handling
            image = image.convert("RGBA")
            # Create a new 400x400 background with pink color
            bg_color = (255, 39, 255)
            new_img = Image.new("RGB", (400, 400), bg_color)
            image.thumbnail((400, 400), Image.LANCZOS)
            # Calculate coordinates to center the image on the background
            left = (400 - image.width) // 2
            top = (400 - image.height) // 2
            # Paste the resized image onto the off-gray background using the image's alpha channel as mask
            new_img.paste(image, (left, top), image)
            byte_img = io.BytesIO()
            new_img.save(byte_img, format='PNG')
            byte_img = byte_img.getvalue()
            img_tensor = tf.constant(byte_img)
            decoded_img_tensor = tf.image.decode_image(img_tensor, channels=IMAGE_CHANNELS)
            decoded_img_tensor.set_shape([None, None, IMAGE_CHANNELS])
            decoded_img_tensor = tf.image.resize(decoded_img_tensor, [IMAGE_WIDTH, IMAGE_HEIGHT])
            decoded_img_tensor = tf.cast(decoded_img_tensor, tf.float32) / 255.0
            image_tensors.append(decoded_img_tensor)
    # make predictions
    predictions = imagezap.predict(np.array(image_tensors)).tolist()
    # format predictions
    results = []
    for prediction in predictions:
        obj = {}
        for j in range(len(prediction)):
            obj[imagezap_labels[j]] = prediction[j]
        results.append(obj)
    return results