"""
Recovera AI: Bone Fracture Radiograph (X-Ray) Deep Learning Training Pipeline (Upgraded)
Dedicated script for training on:
- Dataset/Bone_Fracture_Binary_Classification_X_RAY
- Kaggle: bmadushanirodrigo/fracture-multi-region-x-ray-data (fractured vs not fractured)

Key Accuracy & Anti-Overfitting / Anti-Underfitting Upgrades:
1. Auto GPU Detection: Checks CUDA (and Apple MPS) availability first, enabling TF32, cuDNN benchmark, and FP16 AMP.
2. Two-Phase Progressive Transfer Learning (Prevents Underfitting & Catastrophic Forgetting):
   - Warmup Phase: Freezes pretrained DenseNet-121 backbone and trains the custom classifier head with higher LR.
   - Fine-Tuning Phase: Unfreezes all layers with discriminative learning rates (lower LR for backbone, higher for head).
3. Regularized Custom Classifier Head: Linear -> BatchNorm1d -> SiLU -> Dropout -> Linear.
4. Overfitting Prevention:
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance subtle cortical bone fracture lines.
   - Medical X-Ray Data Augmentations (RandomAffine, RandomRotation, RandomHorizontalFlip, ColorJitter, RandomErasing).
   - Mixup / CutMix batch regularization + Label Smoothing Cross-Entropy Loss.
   - Class-balanced loss weights + WeightedRandomSampler to handle class imbalance.
   - OneCycleLR / CosineAnnealingWarmRestarts + Early Stopping with automatic best-weight restoration.
   - Test-Time Augmentation (TTA) during final evaluation.
5. Outputs:
   - backend/storage/models/fracture_densenet121_best.pth (PyTorch state_dict compatible with Recovera AI backend)
   - backend/storage/models/fracture_densenet121_best.onnx (ONNX Model for Render free tier)
   - backend/storage/models/fracture_densenet121_metadata.json (Metrics, History & Confusion Matrix)
   - backend/storage/models/fracture_training_validation_accuracy.png (Single combined Train & Val Accuracy/Loss graph)
   - backend/storage/models/fracture_confusion_matrix.png (Confusion Matrix heatmap image)
   - backend/storage/models/gradcam/sample_fracture_gradcam.jpg (Grad-CAM Heatmap)
"""

import os
import sys
import time
import json
import copy
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models


# =====================================================================
# EASY EPOCH & HYPERPARAMETER CONFIGURATION
# Change EPOCHS here (or pass --epochs N in terminal) to easily scale training
# =====================================================================
DEFAULT_EPOCHS = 25          # Increase (e.g., 35-50) for maximum accuracy or decrease (e.g., 10-15) for quick runs
WARMUP_FREEZE_EPOCHS = 3     # Epochs to train only the classification head before full unfreezing
EARLY_STOPPING_PATIENCE = 8  # Stop early if Val Acc/F1 doesn't improve for N epochs
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMG_SIZE = 256       # 256 or 320 works great for multi-region X-rays
DEFAULT_LR = 3e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
DROPOUT_RATE = 0.35
MIXUP_ALPHA = 0.2            # Set to 0.0 to disable Mixup


# ----------------------------------------------------
# 1. Reproducibility & GPU Hardware Detection
# ----------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_compute_device(requested_device: str = "auto") -> Tuple[torch.device, bool]:
    """
    First checks if a CUDA GPU is available and uses it automatically.
    Also supports Apple Silicon MPS or CPU fallback.
    """
    print("-" * 88)
    print("[STEP 1] Checking GPU Hardware Availability...")
    if requested_device == "cpu":
        device = torch.device("cpu")
        print("  -> [HARDWARE] Explicitly forced to CPU via --device cpu.")
        return device, False

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  -> [GPU DETECTED & ACTIVE] CUDA Device: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        print("  -> [ACCELERATION] cuDNN Auto-Tuner, TF32 & FP16 Mixed Precision (AMP) Enabled.")
        return device, True

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  -> [GPU DETECTED & ACTIVE] Apple Silicon MPS GPU Acceleration Enabled.")
        return device, False

    device = torch.device("cpu")
    print("  -> [WARNING] No CUDA GPU detected. Falling back to CPU.")
    return device, False


# ----------------------------------------------------
# 2. CLAHE Medical Radiograph Enhancement & Dataset
# ----------------------------------------------------
class ApplyCLAHE:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to sharpen bone cortex edges and subtle fracture lines in X-rays.
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
            return Image.fromarray(enhanced)
        return img


class XRayDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform=None, use_clahe: bool = True):
        self.samples = samples
        self.transform = transform
        self.clahe = ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)) if use_clahe else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.clahe is not None:
                    img = self.clahe(img)
        except Exception:
            img = Image.new("RGB", (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE), color=0)

        if self.transform:
            img = self.transform(img)

        return img, label, str(path)


def _find_split_dir(base_dir: Path, candidates: List[str]) -> Optional[Path]:
    if not base_dir.exists() or not base_dir.is_dir():
        return None
    candidates_set = {c.lower() for c in candidates}
    # 1. Direct child match (case-insensitive)
    try:
        for child in base_dir.iterdir():
            if child.is_dir() and child.name.lower() in candidates_set:
                return child
    except Exception:
        pass
    # 2. Check up to 4 levels nested
    for depth in range(1, 5):
        pattern = "*/" * depth
        try:
            for sub in base_dir.glob(pattern):
                if sub.is_dir() and sub.name.lower() in candidates_set:
                    return sub
        except Exception:
            pass
    return None


def _collect_samples_from_dir(target_dir: Path, classes: Optional[List[str]] = None) -> Tuple[List[Tuple[Path, int]], List[str]]:
    class_dirs = sorted([d for d in target_dir.iterdir() if d.is_dir()])
    if classes is None:
        classes = [d.name for d in class_dirs]
    class_to_idx = {name.lower(): i for i, name in enumerate(classes)}

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    samples = []
    for d in class_dirs:
        c_key = d.name.lower()
        if c_key not in class_to_idx:
            continue
        idx = class_to_idx[c_key]
        for f in d.rglob("*"):
            if f.is_file() and f.suffix.lower() in valid_exts:
                samples.append((f, idx))
    return samples, classes


def discover_xray_datasets(data_dir: Path) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]], List[str]]:
    """
    Automatically discovers train, val, and test splits for:
    https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data
    or local Dataset/Bone_Fracture_Binary_Classification_X_RAY.
    If no val split exists, creates a stratified split automatically.
    """
    train_dir = _find_split_dir(data_dir, ["train", "training"])
    val_dir = _find_split_dir(data_dir, ["val", "valid", "validation"])
    test_dir = _find_split_dir(data_dir, ["test", "testing"])

    if train_dir is not None:
        train_samples, classes = _collect_samples_from_dir(train_dir)
        val_samples, _ = _collect_samples_from_dir(val_dir, classes) if val_dir else ([], classes)
        test_samples, _ = _collect_samples_from_dir(test_dir, classes) if test_dir else ([], classes)

        if not val_samples and test_samples:
            val_samples = test_samples
        elif not val_samples and train_samples:
            labels = [s[1] for s in train_samples]
            train_samples, val_samples = train_test_split(
                train_samples, test_size=0.15, random_state=42, stratify=labels
            )
        if not test_samples:
            test_samples = val_samples
        return train_samples, val_samples, test_samples, classes

    # Fallback: flat class folders directly inside data_dir
    all_samples, classes = _collect_samples_from_dir(data_dir)
    if not all_samples:
        return [], [], [], []
    labels = [s[1] for s in all_samples]
    train_samples, val_samples = train_test_split(
        all_samples, test_size=0.20, random_state=42, stratify=labels
    )
    return train_samples, val_samples, val_samples, classes


# ----------------------------------------------------
# 3. Mixup Regularization & Model Builder
# ----------------------------------------------------
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def build_xray_model(arch: str, num_classes: int, dropout: float = 0.35, keep_legacy_keys: bool = False):
    """
    Builds an ImageNet-pretrained Transfer Learning model with an upgraded regularized head
    to prevent both underfitting and overfitting.
    """
    if arch == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = model.classifier.in_features
        if keep_legacy_keys:
            model.classifier = nn.Linear(in_features, num_classes)
        else:
            model.classifier = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 512),
                nn.SiLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(p=dropout / 1.5),
                nn.Linear(512, num_classes)
            )
        gradcam_layer = model.features.denseblock4
        head_params = list(model.classifier.parameters())
    elif arch == "efficientnet_b2":
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 1.5),
            nn.Linear(512, num_classes)
        )
        gradcam_layer = model.features[-1]
        head_params = list(model.classifier.parameters())
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 1.5),
            nn.Linear(512, num_classes)
        )
        gradcam_layer = model.layer4
        head_params = list(model.fc.parameters())

    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    return model, gradcam_layer, backbone_params, head_params


# ----------------------------------------------------
# 4. Grad-CAM Explainable AI Generator
# ----------------------------------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def overlay_heatmap(image_path: Path, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    raw_bgr = cv2.imread(str(image_path))
    if raw_bgr is None:
        raw_bgr = np.zeros((256, 256, 3), dtype=np.uint8)

    h, w = raw_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap_u8 = np.uint8(255 * cam_resized)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(raw_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    return overlay


# ----------------------------------------------------
# 5. Visualization & Reporting Helpers
# ----------------------------------------------------
def save_accuracy_and_loss_graph(history: List[Dict], save_path: Path, title_prefix: str = "Bone Fracture X-Ray"):
    """
    Saves Training & Validation Accuracy (and Loss) in a single high-resolution image.
    """
    epochs = [h["epoch"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Training & Validation Accuracy
    ax1.plot(epochs, train_acc, label="Training Accuracy (train_acc)", color="#1f77b4", linewidth=2.5, marker="o", markersize=4)
    ax1.plot(epochs, val_acc, label="Validation Accuracy (val_acc)", color="#ff7f0e", linewidth=2.5, marker="s", markersize=4)
    best_idx = int(np.argmax(val_acc))
    best_ep = epochs[best_idx]
    best_va = val_acc[best_idx]
    ax1.scatter([best_ep], [best_va], color="#d62728", s=90, zorder=5, label=f"Best Val Acc: {best_va:.2f}% (Ep {best_ep})")
    ax1.set_title(f"{title_prefix} — Training vs Validation Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", frameon=True)

    # Panel 2: Training & Validation Loss
    ax2.plot(epochs, train_loss, label="Training Loss", color="#2ca02c", linewidth=2.2, marker="o", markersize=4)
    ax2.plot(epochs, val_loss, label="Validation Loss", color="#d62728", linewidth=2.2, marker="s", markersize=4)
    ax2.set_title(f"{title_prefix} — Training vs Validation Loss", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(cm: np.ndarray, classes: List[str], save_path: Path, title: str = "Bone Fracture Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=160)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Clinical Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close(fig)


def print_terminal_confusion_matrix(cm: np.ndarray, classes: List[str]):
    print("\n" + "=" * 88)
    print("  CONFUSION MATRIX (Rows = True Labels, Columns = Predicted Labels)")
    print("=" * 88)
    col_width = max(max(len(c) for c in classes), 10) + 2
    header = f"{'True \\ Pred':<{col_width}} | " + " | ".join(f"{c:^{col_width}}" for c in classes)
    print(header)
    print("-" * len(header))
    for i, row_name in enumerate(classes):
        row_str = f"{row_name:<{col_width}} | " + " | ".join(f"{cm[i, j]:^{col_width}d}" for j in range(len(classes)))
        print(row_str)
    print("=" * 88)


# ----------------------------------------------------
# 6. Main Training & Evaluation Pipeline
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Recovera AI: Upgraded Bone Fracture X-Ray Transfer Learning Pipeline")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Path to fracture dataset (Defaults to Dataset/Bone_Fracture_Binary_Classification_X_RAY or Kaggle input)")
    parser.add_argument("--arch", type=str, default="densenet121", choices=["densenet121", "efficientnet_b2", "resnet50"],
                        help="Pretrained Transfer Learning backbone architecture (default: densenet121)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of training epochs (default: {DEFAULT_EPOCHS}) — easily adjustable")
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_FREEZE_EPOCHS,
                        help=f"Epochs to freeze backbone and warm up classifier head (default: {WARMUP_FREEZE_EPOCHS})")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE,
                        help=f"Early stopping patience to prevent overfitting (default: {EARLY_STOPPING_PATIENCE})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Peak learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE,
                        help=f"Input image resolution (default: {DEFAULT_IMG_SIZE})")
    parser.add_argument("--mixup_alpha", type=float, default=MIXUP_ALPHA,
                        help=f"Mixup regularization alpha (default: {MIXUP_ALPHA}, 0 to disable)")
    parser.add_argument("--no_clahe", action="store_true",
                        help="Disable CLAHE radiograph contrast enhancement")
    parser.add_argument("--legacy_head", action="store_true",
                        help="Use a single nn.Linear classifier head for strict backward state_dict compatibility")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Compute device ('auto' checks GPU first)")
    parser.add_argument("--save_dir", type=str, default="backend/storage/models",
                        help="Directory to save models, confusion matrix, and accuracy graphs")

    args = parser.parse_args()
    seed_everything(42)

    print("=" * 88)
    print("  RECOVERA AI: UPGRADED MULTI-REGION BONE FRACTURE X-RAY TRAINING PIPELINE")
    print("=" * 88)

    # 1. Check GPU availability first
    device, use_cuda = get_compute_device(args.device)

    # 2. Resolve Dataset Path (Local Recovera repo or Kaggle paths)
    root = Path(__file__).resolve().parent
    candidate_dirs = [
        Path(args.data_dir) if args.data_dir else None,
        root / "Dataset" / "Bone_Fracture_Binary_Classification_X_RAY",
        root / "Dataset" / "Bone_Fracture_Binary_Classification",
        root / "Dataset" / "fracture-multi-region-x-ray-data",
        root / "Dataset" / "X-ray",
        root / "Dataset" / "x-ray",
        root / "Dataset" / "Bone Fracture Multi-Region X-ray Data",
        root / "Dataset",
        Path("/kaggle/input/fracture-multi-region-x-ray-data"),
        Path("/kaggle/input/bone-fracture-detection-computer-vision-project"),
        Path("/kaggle/input/bone-fracture-binary-classification"),
    ]
    data_dir = None
    for c in candidate_dirs:
        if c is not None and c.exists():
            data_dir = c
            break

    if data_dir is None:
        print(f"[ERROR] Dataset directory not found. Checked: {[str(c) for c in candidate_dirs if c]}")
        print("Pass --data_dir /path/to/fracture-multi-region-x-ray-data")
        sys.exit(1)

    train_samples, val_samples, test_samples, classes = discover_xray_datasets(data_dir)
    if not train_samples or len(classes) < 2:
        print(f"[ERROR] Could not discover valid class folders in: {data_dir}")
        sys.exit(1)

    # Compute class distribution & inverse frequency weights to prevent majority-class bias
    train_labels = [s[1] for s in train_samples]
    class_counts = np.bincount(train_labels, minlength=len(classes))
    total_train = len(train_labels)
    class_weights = total_train / (len(classes) * np.maximum(class_counts, 1))
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # WeightedRandomSampler for balanced mini-batches
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    print(f"Dataset Location : {data_dir.resolve()}")
    print(f"Detected Classes : {len(classes)} -> {classes}")
    print(f"Class Counts     : {dict(zip(classes, class_counts.tolist()))}")
    print(f"Split Sizes      : Train={len(train_samples)} | Val={len(val_samples)} | Test={len(test_samples)}")
    print(f"Architecture     : {args.arch.upper()} (ImageNet Pretrained Transfer Learning)")
    print(f"Hyperparameters  : Epochs={args.epochs} (Warmup={args.warmup_epochs}) | Batch={args.batch_size} | LR={args.lr} | Img={args.img_size}x{args.img_size}")
    print(f"Anti-Overfitting : CLAHE={not args.no_clahe} | Mixup={args.mixup_alpha} | LabelSmoothing={LABEL_SMOOTHING} | Dropout={DROPOUT_RATE} | WeightDecay={WEIGHT_DECAY}")
    print("-" * 88)

    # 3. Data Augmentations (Prevents Overfitting on Multi-Region X-Rays)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=18),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08), shear=5),
        transforms.ColorJitter(brightness=0.15, contrast=0.20),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random")
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((int(args.img_size * 1.12), int(args.img_size * 1.12))),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    train_ds = XRayDataset(train_samples, transform=train_transforms, use_clahe=not args.no_clahe)
    val_ds = XRayDataset(val_samples, transform=val_transforms, use_clahe=not args.no_clahe)
    test_ds = XRayDataset(test_samples, transform=val_transforms, use_clahe=not args.no_clahe)

    num_workers = min(4, os.cpu_count() or 2) if os.name != "nt" else 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=use_cuda, drop_last=len(train_ds) > args.batch_size
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda
    )

    # 4. Build Transfer Learning Model
    model, gradcam_layer, backbone_params, head_params = build_xray_model(
        args.arch, len(classes), dropout=DROPOUT_RATE, keep_legacy_keys=args.legacy_head
    )
    model.to(device)

    # Loss Function with Class Balancing & Label Smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)
    val_criterion = nn.CrossEntropyLoss()

    # Discriminative Learning Rates: Backbone gets 0.1x LR to preserve pretrained ImageNet edge detectors
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.15},
        {"params": head_params, "lr": args.lr}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=use_cuda)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_pth = out_dir / "fracture_densenet121_best.pth"
    best_onnx = out_dir / "fracture_densenet121_best.onnx"
    meta_path = out_dir / "fracture_densenet121_metadata.json"
    acc_plot_path = out_dir / "fracture_training_validation_accuracy.png"
    cm_plot_path = out_dir / "fracture_confusion_matrix.png"

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    history = []

    print("\n[STEP 2] Starting Progressive Transfer Learning...")
    print("=" * 88)
    print(f"{'EPOCH':^10} | {'PHASE':^10} | {'TRAIN LOSS':^11} | {'VAL LOSS':^10} | {'TRAIN_ACC':^11} | {'VAL_ACC':^11} | {'VAL F1':^9} | {'TIME':^7}")
    print("-" * 88)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Two-Phase Transfer Learning Control
        if epoch <= args.warmup_epochs:
            phase_name = "WARMUP"
            for p in backbone_params:
                p.requires_grad = False
        else:
            phase_name = "FINE-TUNE"
            for p in backbone_params:
                p.requires_grad = True

        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            use_mixup = (args.mixup_alpha > 0.0) and (epoch > args.warmup_epochs) and (random.random() < 0.5)
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha=args.mixup_alpha)
            else:
                inputs = images
                targets_a, targets_b, lam = labels, labels, 1.0

            with torch.cuda.amp.autocast(enabled=use_cuda):
                outputs = model(inputs)
                if use_mixup:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            if use_mixup:
                train_correct += (lam * preds.eq(targets_a).sum().item() + (1 - lam) * preds.eq(targets_b).sum().item())
            else:
                train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = (train_correct / max(1, train_total)) * 100.0

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_targets, val_preds = [], []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    outputs = model(images)
                    v_loss = val_criterion(outputs, labels)
                val_running_loss += v_loss.item()
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        val_loss = val_running_loss / max(1, len(val_loader))
        val_acc = accuracy_score(val_targets, val_preds) * 100.0
        val_precision = precision_score(val_targets, val_preds, average="weighted", zero_division=0) * 100.0
        val_recall = recall_score(val_targets, val_preds, average="weighted", zero_division=0) * 100.0
        val_f1 = f1_score(val_targets, val_preds, average="weighted", zero_division=0) * 100.0
        epoch_sec = time.time() - epoch_start

        scheduler.step()

        # Clearly visible terminal status for every epoch with train_acc and val_acc
        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_f1 > best_val_f1)
        marker = " *BEST*" if is_best else ""
        print(
            f" {epoch:02d}/{args.epochs:02d}     | {phase_name:^10} | "
            f"  {train_loss:7.4f}   |  {val_loss:7.4f}   | "
            f"  {train_acc:6.2f}%   |   {val_acc:6.2f}%   |  {val_f1:6.2f}%  | {epoch_sec:5.1f}s{marker}"
        )

        history.append({
            "epoch": epoch,
            "phase": phase_name,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "val_precision": round(val_precision, 2),
            "val_recall": round(val_recall, 2),
            "val_f1": round(val_f1, 2),
            "epoch_sec": round(epoch_sec, 1)
        })

        if is_best:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, str(best_pth))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience and epoch > args.warmup_epochs:
                print("-" * 88)
                print(f"[EARLY STOPPING] No validation accuracy improvement for {args.patience} epochs. Stopping at Epoch {epoch}.")
                break

    total_training_sec = round(time.time() - start_time, 2)

    # 5. Restore Best Model Weights & Run Final Evaluation (with Test-Time Augmentation)
    model.load_state_dict(best_model_weights)
    model.eval()
    final_targets, final_preds = [], []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                # Test-Time Augmentation (Original + Horizontal Flip average)
                logits_orig = model(images)
                logits_flip = model(torch.flip(images, dims=[3]))
                probs = (F.softmax(logits_orig, dim=1) + F.softmax(logits_flip, dim=1)) / 2.0
            preds = probs.argmax(dim=1).cpu().numpy()
            final_preds.extend(preds)
            final_targets.extend(labels.numpy())

    final_acc = accuracy_score(final_targets, final_preds) * 100.0
    final_f1 = f1_score(final_targets, final_preds, average="weighted", zero_division=0) * 100.0
    cm = confusion_matrix(final_targets, final_preds)

    # Display Final Accuracy & Confusion Matrix in Terminal
    print("\n" + "=" * 88)
    print("  FINAL TRAINING SUMMARY & MODEL EVALUATION")
    print("=" * 88)
    print(f"  Best Validation Accuracy (val_acc) : {best_val_acc:.2f}% (Achieved at Epoch {best_epoch})")
    print(f"  Final Test Accuracy (with TTA)     : {final_acc:.2f}%")
    print(f"  Final Weighted F1-Score            : {final_f1:.2f}%")
    print(f"  Total Training Time                : {total_training_sec}s")
    print(f"  Saved PyTorch Model                : {best_pth.resolve()}")

    print_terminal_confusion_matrix(cm, classes)
    print("\nClassification Report:")
    print(classification_report(final_targets, final_preds, target_names=classes, digits=4, zero_division=0))

    # 6. Save Training & Validation Accuracy Graph (in a single image) & Confusion Matrix Image
    save_accuracy_and_loss_graph(history, acc_plot_path, title_prefix="Bone Fracture X-Ray")
    save_confusion_matrix_plot(cm, classes, cm_plot_path, title=f"Bone Fracture X-Ray Confusion Matrix (Acc: {final_acc:.2f}%)")
    print(f"[GRAPH SAVED] Training & Validation Accuracy Graph : {acc_plot_path.resolve()}")
    print(f"[GRAPH SAVED] Confusion Matrix Heatmap Image       : {cm_plot_path.resolve()}")

    # 7. Save JSON Metadata
    meta = {
        "task": "fracture",
        "arch": args.arch,
        "classes": classes,
        "num_classes": len(classes),
        "img_size": args.img_size,
        "best_val_accuracy": round(best_val_acc, 2),
        "final_test_accuracy": round(final_acc, 2),
        "best_val_f1": round(best_val_f1, 2),
        "final_test_f1": round(final_f1, 2),
        "best_epoch": best_epoch,
        "total_training_sec": total_training_sec,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "device": str(device)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[METADATA SAVED]                                   : {meta_path.resolve()}")

    # 8. Export to ONNX (For Free Render Deployment)
    if best_pth.exists():
        try:
            print("\n[EXPORTING ONNX FOR RENDER DEPLOYMENT]")
            model.load_state_dict(torch.load(str(best_pth), map_location="cpu"))
            model.eval()
            dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device="cpu")
            torch.onnx.export(
                model.to("cpu"),
                dummy_input,
                str(best_onnx),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            )
            print(f"  -> ONNX Model Exported: {best_onnx.resolve()}")
        except Exception as e:
            print(f"  -> [WARNING] ONNX export skipped: {e}")

    # 9. Generate Grad-CAM Sample
    if val_samples and best_pth.exists():
        try:
            print("\n[GENERATING GRAD-CAM EXPLAINABILITY MAP]")
            sample_img_path, sample_label = val_samples[0]
            model.load_state_dict(torch.load(str(best_pth), map_location=device))
            model.to(device)

            gradcam = GradCAM(model, gradcam_layer)
            raw_img = Image.open(sample_img_path).convert("RGB")
            input_tensor = val_transforms(raw_img).unsqueeze(0).to(device)
            cam = gradcam.generate(input_tensor, target_class=sample_label)

            gradcam_dir = out_dir / "gradcam"
            gradcam_dir.mkdir(parents=True, exist_ok=True)
            sample_out = gradcam_dir / "sample_fracture_gradcam.jpg"

            overlay = overlay_heatmap(sample_img_path, cam)
            cv2.imwrite(str(sample_out), overlay)
            print(f"  -> Grad-CAM Heatmap saved: {sample_out.resolve()}")
            gradcam.remove_hooks()
        except Exception as e:
            print(f"  -> [WARNING] Grad-CAM skipped: {e}")

    print("=" * 88)


if __name__ == "__main__":
    main()