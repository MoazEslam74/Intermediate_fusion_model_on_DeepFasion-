import torch
from transformers import BertTokenizer, ViTModel, BertModel
import torch.nn as nn
import torch
from transformers import BertTokenizer
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.notebook import tqdm
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from transformers import BertModel,BertTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
#################################
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sample_text = ""
encoded_input = tokenizer(
    sample_text,
    padding='max_length',    # Padding using Zeros
    max_length=16,           # Max length of tokens
    truncation=True,         # Crop any tokens more than the max length
    return_tensors='pt'      # to tensors direct
)
####################################


class DeepFashionTransforms:

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, img_size=256, crop_size=224):
        self.img_size  = img_size
        self.crop_size = crop_size

    def train_transform(self):
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomResizedCrop(self.crop_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
        ])

    def val_transform(self):
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
        ])

    # alias so test uses the same pipeline as val
    def test_transform(self):
        return self.val_transform()


#########################################################

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
transforms_handler = DeepFashionTransforms(img_size=256, crop_size=224)

IMG_ROOT = 'selected_images'
CSV_PATH = 'labels_front.csv'



####################################################################

from torchvision import transforms

# ViT expects exactly 224x224 — no random crop needed since ViT handles spatial variation
vit_train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],   # ViT pretrained on [-1, 1]
                         std=[0.5, 0.5, 0.5])
])

vit_val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

###############################################################
# ── paste your model class here or import it ────────────────────────────────
# (ViTFusionModel or IntermediateFusionModel — whichever you saved)

NUM_CLASSES = 17   # change to your actual number
DEVICE      = "cpu"

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


import json
import os

class CrossModalAttention(nn.Module):
    """
    One direction of cross-attention:
      query  comes from modality A  (shape: B, 1, dim)  — the single global token
      key/value come from modality B  (shape: B, seq_len, dim) — the full sequence

    For image→text: image queries the full BERT token sequence
    For text→image: [CLS] token queries the image spatial feature (treated as seq of 1)
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True      # (B, seq, dim) convention
        )
        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights = None

    def forward(self, query, key_value, key_padding_mask=None):
        """
        query        : (B, 1,       dim)
        key_value    : (B, seq_len, dim)
        Returns      : (B, 1,       dim)  — query enriched with context from key_value
        """
        attended, attn_weights = self.attn(
            query, key_value, key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        
        # store for XAI
        self.last_attn_weights = attn_weights
        
        # residual + norm
        out = self.norm(query + self.dropout(attended))
        return out  
    
class TrainingLogger:
    def __init__(self, save_path="training_logs.json"):
        self.save_path = save_path
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   []
        }

    def log(self, train_loss, train_acc, val_loss, val_acc):
        self.history["train_loss"].append(round(train_loss, 6))
        self.history["train_acc"].append(round(train_acc,  6))
        self.history["val_loss"].append(round(val_loss,   6))
        self.history["val_acc"].append(round(val_acc,     6))
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load(self):
        if os.path.exists(self.save_path):
            with open(self.save_path) as f:
                self.history = json.load(f)
        return self.history
class ViTFusionModel(nn.Module):
    """
  
    """

    def __init__(self, num_classes, proj_dim=512, num_heads=8, dropout=0.3):
        super().__init__()

        # ── Image encoder: ViT-B/16 ──────────────────────────────────────
        # hidden size = 768, sequence length = 197 (1 CLS + 196 patches)
        self.img_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        vit_hidden_dim   = self.img_encoder.config.hidden_size   # 768

        self.img_proj = nn.Sequential(
            nn.Linear(vit_hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── Text encoder: BERT-base ───────────────────────────────────────
        self.txt_encoder = BertModel.from_pretrained("bert-base-uncased")
        bert_hidden_dim  = self.txt_encoder.config.hidden_size   # 768

        self.txt_proj = nn.Linear(bert_hidden_dim, proj_dim)
        self.txt_norm = nn.LayerNorm(proj_dim)

        # ── Bidirectional cross-attention ─────────────────────────────────
        # Direction 1: image CLS queries the full text token sequence
        self.img_attends_txt = CrossModalAttention(proj_dim, num_heads, dropout)

        # Direction 2: text CLS queries all 196 image patch tokens
        self.txt_attends_img = CrossModalAttention(proj_dim, num_heads, dropout)

        # ── Classifier ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        # ── Image branch (ViT) ────────────────────────────────────────────
        vit_out     = self.img_encoder(pixel_values=images)
        # last_hidden_state: (B, 197, 768)  — CLS token is index 0
        img_tokens  = self.img_proj(vit_out.last_hidden_state)   # (B, 197, 512)
        img_cls     = img_tokens[:, :1, :]                        # (B,   1, 512)
        img_patches = img_tokens[:, 1:, :]                        # (B, 196, 512)

        # ── Text branch (BERT) ────────────────────────────────────────────
        bert_out   = self.txt_encoder(input_ids=input_ids,
                                      attention_mask=attention_mask)
        txt_tokens = self.txt_norm(
            self.txt_proj(bert_out.last_hidden_state)             # (B, seq_len, 512)
        )
        txt_cls    = txt_tokens[:, :1, :]                         # (B, 1, 512)

        # Convert HuggingFace mask (1=keep, 0=pad) → PyTorch key_padding_mask (True=ignore)
        txt_pad_mask = (attention_mask == 0)                      # (B, seq_len)

        # ── Cross-attention (bidirectional) ───────────────────────────────
        # Image CLS attends to all text tokens
        img_enriched = self.img_attends_txt(
            query=img_cls,
            key_value=txt_tokens,
            key_padding_mask=txt_pad_mask
        ).squeeze(1)                                              # (B, 512)

        # Text CLS attends to all 196 image patch tokens
        txt_enriched = self.txt_attends_img(
            query=txt_cls,
            key_value=img_patches                                 # no padding mask needed
        ).squeeze(1)                                              # (B, 512)

        # ── Fusion + classify ─────────────────────────────────────────────
        fused  = torch.cat([img_enriched, txt_enriched], dim=1)  # (B, 1024)
        logits = self.classifier(fused)                          # (B, num_classes)
        return logits
    
vit_model  = ViTFusionModel(num_classes=17).to(DEVICE)
vit_logger = TrainingLogger(save_path="training_logs_vit.json")
criterion  = nn.CrossEntropyLoss()

# Freeze both ViT and BERT initially
for param in vit_model.img_encoder.parameters():
    param.requires_grad = False
for param in vit_model.txt_encoder.parameters():
    param.requires_grad = False

optimizer = AdamW(filter(lambda p: p.requires_grad, vit_model.parameters()),
                  lr=1e-4, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=15)

##########################################################
def run_epoch_vit(loader, train=True, epoch=0):
    vit_model.train() if train else vit_model.eval()
    total_loss, correct, total = 0.0, 0, 0
    phase = "Train" if train else "Val"
    pbar  = tqdm(loader, desc=f"Epoch {epoch:02d} [{phase}]", leave=False)

    with torch.set_grad_enabled(train):
        for images, input_ids, attention_mask, labels in pbar:
            images, input_ids      = images.to(DEVICE), input_ids.to(DEVICE)
            attention_mask, labels = attention_mask.to(DEVICE), labels.to(DEVICE)

            logits = vit_model(images, input_ids, attention_mask)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
            pbar.set_postfix({"loss": f"{total_loss/total:.4f}",
                              "acc":  f"{correct/total:.4f}"})

    return total_loss / total, correct / total
best_val_acc = 0.0
##############################################################


model = ViTFusionModel(num_classes=NUM_CLASSES)
checkpoint = torch.load("best_model_ViT.pth", map_location="cpu")
print(type(checkpoint))       # dict → full checkpoint, OrderedDict → weights only
print(checkpoint.keys())      # shows what's inside
checkpoint = torch.load("best_model_ViT.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ── create dummy inputs matching your forward() signature ───────────────────
dummy_image        = torch.randn(1, 3, 224, 224)
dummy_text         = tokenizer("a blue dress with short sleeves",
                               padding="max_length", max_length=64,
                               truncation=True, return_tensors="pt")
dummy_input_ids    = dummy_text["input_ids"]
dummy_attn_mask    = dummy_text["attention_mask"]

torch.onnx.export(
    model,
    (dummy_image, dummy_input_ids, dummy_attn_mask),
    "model.onnx",
    input_names  = ["image", "input_ids", "attention_mask"],
    output_names = ["logits"],
    dynamic_axes = {
        "image":          {0: "batch"},
        "input_ids":      {0: "batch"},
        "attention_mask": {0: "batch"},
        "logits":         {0: "batch"},
    },
    opset_version=14
)
print("Exported model.onnx")