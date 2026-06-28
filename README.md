# Intermediate Fusion on DeepFashion Dataset

A multimodal deep learning system for fashion item classification that fuses **visual** and **textual** information using an intermediate fusion strategy. The model allows each modality to attend to the other through **bidirectional cross-modal attention** before making a final prediction.

Two model variants are implemented:

| Variant | Image Encoder | Status |
|---|---|---|
| **Baseline** | ResNet-50 (CNN) | Training & evaluation |
| **ViT Fusion** | ViT-B/16 (Vision Transformer) | Training, evaluation & ONNX deployment |

---

## Dataset

- **Source:** [DeepFashion Multimodal](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- **Size:** 12,000+ front-view fashion images, each paired with a text caption
- **Classes:** 17 fashion categories

```
Blouses_Shirts · Cardigans · Denim · Dresses · Graphic_Tees
Jackets_Coats · Jackets_Vests · Leggings · Pants · Rompers_Jumpsuits
Shirts_Polos · Shorts · Skirts · Sweaters · Sweatshirts_Hoodies
Tees_Tanks · Tops
```

- **Split:** 80% train / 10% validation / 10% test (stratified, random seed 42)
- Missing images are automatically filtered out at dataset load time

---

## Architecture

### Baseline — ResNet-50 + BERT

```
Image → ResNet-50 → Global Pool → Linear Projection ─┐
                                                       ├→ Cross-Modal Attention → Classifier
Text  → BERT-base → [CLS] token → Linear Projection ─┘
```

### ViT Fusion (Deployed Variant)

```
Image → ViT-B/16 → [CLS] + 196 patch tokens → Linear Projection (512d) ─┐
                                                                           ├→ Bidirectional Cross-Attention → Concat → MLP → 17 classes
Text  → BERT-base → token sequence          → Linear Projection (512d) ─┘
```

- Image CLS attends over all text tokens
- Text CLS attends over all 196 image patch tokens
- Both enriched representations are concatenated and passed to a 2-layer classifier

### Explainability

- **Grad-CAM** on the image branch — highlights spatially important regions
- **Integrated Gradients** on the text branch — scores per-token importance

---

## Project Structure

```
├── Intermediate_fusion.ipynb   # Training notebook (both variants)
├── export_onnx.py              # Export trained ViT model to ONNX
├── Deploy/
│   ├── app.py                  # FastAPI inference server
│   ├── model.onnx              # Exported ONNX model
│   └── static/
│       └── index.html          # Frontend UI
├── labels_front.csv            # Dataset CSV (image path + caption + label)
└── selected_images/            # Dataset images
```

---

## Requirements

Install all dependencies with:

```bash
pip install torch torchvision transformers fastapi uvicorn \
            onnxruntime pillow numpy scikit-learn pandas \
            tqdm onnx onnxscript python-multipart
```

> **Note:** If you have a CUDA GPU, replace `onnxruntime` with `onnxruntime-gpu` for faster inference.

```bash
pip install onnxruntime-gpu
```

---

## Running the Deployment Server

### 1. Export the model to ONNX (if not already done)

Make sure `best_model_ViT.pth` is in the project root, then run:

```bash
python export_onnx.py
```

This generates `model.onnx` in the same directory.

### 2. Move files to the Deploy folder

```bash
cp model.onnx Deploy/
```

### 3. Start the FastAPI server

```bash
cd Deploy
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the app

Navigate to [http://localhost:8000](http://localhost:8000) in your browser.

### 5. Use the `/predict` endpoint directly

You can also call the API directly with `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@/path/to/your/image.jpg" \
  -F "caption=a blue denim jacket with front pockets"
```

**Response format:**

```json
{
  "predictions": [
    {"class": "Denim",        "confidence": 87.3},
    {"class": "Jackets_Coats","confidence": 7.1},
    {"class": "Shorts",       "confidence": 2.4}
  ]
}
```

The endpoint returns the **top-3 predicted classes** with their confidence scores.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI |
| `POST` | `/predict` | Runs multimodal inference |

**`/predict` parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `image` | File | ✅ | Fashion image (JPG/PNG) |
| `caption` | String | ❌ | Optional text description of the item |

---

## Model Export Details

- **Format:** ONNX opset 18
- **Inputs:** `image (1,3,224,224)`, `input_ids (1,64)`, `attention_mask (1,64)`
- **Output:** `logits (1,17)`
- **Dynamic axes:** batch dimension is dynamic

---

## Acknowledgements

- [DeepFashion Dataset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) — MMLAB, CUHK
- [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) — HuggingFace
- [bert-base-uncased](https://huggingface.co/bert-base-uncased) — HuggingFace
