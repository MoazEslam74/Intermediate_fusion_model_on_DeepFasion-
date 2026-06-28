import io
import json
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── load once at startup ─────────────────────────────────────────────────────
session   = ort.InferenceSession("model.onnx",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

CLASS_NAMES = [
    'Blouses_Shirts', 'Cardigans', 'Denim', 'Dresses', 'Graphic_Tees', 'Jackets_Coats', 'Jackets_Vests', 'Leggings', 'Pants', 'Rompers_Jumpsuits', 'Shirts_Polos', 'Shorts', 'Skirts', 'Suiting', 'Sweaters', 'Sweatshirts_Hoodies', 'Tees_Tanks'
]   # must match your class_to_idx order exactly

img_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


@app.post("/predict")
async def predict(image: UploadFile = File(...), caption: str = Form("")):
    # ── preprocess image ─────────────────────────────────────────────────
    img_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = img_transform(pil_image).unsqueeze(0).numpy()   # (1,3,224,224)

    # ── preprocess text ──────────────────────────────────────────────────
    enc = tokenizer(caption, padding="max_length", max_length=64,
                    truncation=True, return_tensors="np")

    # ── run ONNX inference ───────────────────────────────────────────────
    logits = session.run(
        ["logits"],
        {
            "image":          img_tensor.astype(np.float32),
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
    )[0][0]   # shape (num_classes,)

    probs      = softmax(logits)
    top3_idx   = probs.argsort()[::-1][:3]
    results    = [{"class": CLASS_NAMES[i], "confidence": round(float(probs[i]) * 100, 1)}
                  for i in top3_idx]

    return {"predictions": results}


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")