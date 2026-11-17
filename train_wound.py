"""
Recovera AI: Surgical Wound & Dermatological Lesion Deep Learning Training Pipeline
Dedicated script for training on: Dataset/Wound Analysis (23 clinical classes)

Supports BOTH:
- CUDA GPU Acceleration (NVIDIA Tensor Cores + FP16 Mixed Precision)
- Normal CPU Training (Automatic Fallback or manual switch via --device cpu)

Architecture: ResNet-50 (ImageNet Pretrained Transfer Learning)
Outputs:
- backend/storage/models/wound_resnet50_best.pth (PyTorch Model)
- backend/storage/models/wound_resnet50_best.onnx (ONNX Model for Render free tier)
- backend/storage/models/wound_resnet50_metadata.json (Metrics & Confusion Matrix)
- backend/storage/models/gradcam/sample_wound_gradcam.jpg (Grad-CAM Heatmap)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ----------------------------------------------------
# 1. Device Selection (Auto CUDA GPU vs CPU)
# ----------------------------------------------------
def get_compute_device(requested_device: str = "auto") -> Tuple[torch.device, bool]:
    if requested_device == "cpu":
        device = torch.device("cpu")
        print("[HARDWARE] Explicitly running on CPU.")
        return device, False

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU ACCELERATION ACTIVE] Using CUDA: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        print("[OPTIMIZATION] Tensor Cores & FP16 Mixed Precision Enabled for Fast Training.")
        return device, True
    else:
        device = torch.device("cpu")
        print("[INFO] CUDA GPU not detected. Automatically falling back to CPU.")
        return device, False


# ----------------------------------------------------
# 2. Dataset Discovery & Loading
# ----------------------------------------------------
class WoundDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256), color=0)

        if self.transform:
            img = self.transform(img)

        return img, label, str(path)


def discover_wound_samples(data_dir: Path, split: str = "train") -> Tuple[List[Tuple[Path, int]], List[str]]:
    split_dir = data_dir / split
    target_dir = split_dir if split_dir.exists() and split_dir.is_dir() else data_dir

    class_dirs = sorted([d for d in target_dir.iterdir() if d.is_dir()])
    classes = [d.name for d in class_dirs]
    class_to_idx = {name: i for i, name in enumerate(classes)}

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples = []
    for c_name in classes:
        c_dir = target_dir / c_name
        idx = class_to_idx[c_name]
        for f in c_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_exts:
                samples.append((f, idx))

    return samples, classes


# ----------------------------------------------------
# 3. Grad-CAM Explainable AI Generator
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

        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
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
# 4. Main Training Pipeline
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Recovera AI: Train Surgical Wound & Lesion Model (ResNet-50)")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Path to wound dataset. Defaults to Dataset/Wound Analysis")
    parser.add_argument("--epochs", type=int, default=12,
                        help="Number of epochs (default: 12)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Input image resolution (default: 256)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to train on ('auto' detects CUDA GPU, or 'cpu')")
    parser.add_argument("--save_dir", type=str, default="backend/storage/models",
                        help="Target output directory")

    args = parser.parse_args()

    print("=" * 75)
    print("  Recovera AI: Surgical Wound & Lesion Transfer Learning Pipeline")
    print("=" * 75)

    device, use_cuda = get_compute_device(args.device)

    # Resolve Dataset Path
    root = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else (root / "Dataset" / "Wound Analysis")
    if not data_dir.exists():
        print(f"[ERROR] Dataset directory not found: {data_dir}")
        print("Please check that Dataset/Wound Analysis exists.")
        sys.exit(1)

    train_samples, classes = discover_wound_samples(data_dir, split="train")
    val_samples, _ = discover_wound_samples(data_dir, split="val")
    if not val_samples:
        val_samples, _ = discover_wound_samples(data_dir, split="test")

    if not train_samples:
        print(f"[ERROR] No training images found in: {data_dir}")
        sys.exit(1)

    print(f"Dataset Location: {data_dir.resolve()}")
    print(f"Clinical Classes: {len(classes)} categories")
    print(f"Train Images:     {len(train_samples)}")
    print(f"Val Images:       {len(val_samples)}")
    print(f"Architecture:     ResNet-50 (ImageNet Pretrained)")
    print(f"Batch Size:       {args.batch_size} | Epochs: {args.epochs} | Base LR: {args.lr}")
    print("-" * 75)

    # Preprocessing Transforms with Wound Color & Phototype Invariance
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    train_ds = WoundDataset(train_samples, transform=train_transforms)
    val_ds = WoundDataset(val_samples, transform=val_transforms)

    num_workers = 2 if os.name != "nt" else 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_cuda, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda
    )

    # Model Setup
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(classes))
    gradcam_layer = model.layer4
    model.to(device)

    # Optimizer, Loss & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed precision scaler (active when CUDA is available)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_pth = out_dir / "wound_resnet50_best.pth"
    best_onnx = out_dir / "wound_resnet50_best.onnx"
    meta_path = out_dir / "wound_resnet50_metadata.json"

    best_val_f1 = 0.0
    best_val_acc = 0.0
    history = []

    print("[STARTING TRAINING]")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        train_preds, train_targets = [], []

        for step, (images, labels, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().numpy())

            if (step + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"  Epoch [{epoch}/{args.epochs}] Step [{step+1}/{len(train_loader)}] — Batch Loss: {loss.item():.4f}")

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds) * 100.0

        # Validation Phase
        model.eval()
        val_targets, val_preds = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())

        val_acc = accuracy_score(val_targets, val_preds) * 100.0
        val_precision = precision_score(val_targets, val_preds, average="weighted", zero_division=0) * 100.0
        val_recall = recall_score(val_targets, val_preds, average="weighted", zero_division=0) * 100.0
        val_f1 = f1_score(val_targets, val_preds, average="weighted", zero_division=0) * 100.0
        epoch_sec = time.time() - epoch_start

        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({epoch_sec:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
            f"Val Acc: {val_acc:.1f}% | Val Recall: {val_recall:.1f}% | "
            f"Val F1: {val_f1:.1f}%"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "val_precision": round(val_precision, 2),
            "val_recall": round(val_recall, 2),
            "val_f1": round(val_f1, 2),
            "epoch_sec": round(epoch_sec, 1)
        })

        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(best_pth))
            print(f"  >>> Best model saved to: {best_pth.resolve()} (F1: {val_f1:.2f}%)")

    total_training_sec = round(time.time() - start_time, 2)
    print("=" * 75)
    print(f"TRAINING COMPLETE in {total_training_sec}s | Best Val Acc: {best_val_acc:.2f}% | Best F1: {best_val_f1:.2f}%")

    cm = confusion_matrix(val_targets, val_preds).tolist()
    meta = {
        "task": "wound",
        "arch": "resnet50",
        "classes": classes,
        "num_classes": len(classes),
        "img_size": args.img_size,
        "best_val_accuracy": round(best_val_acc, 2),
        "best_val_f1": round(best_val_f1, 2),
        "total_training_sec": total_training_sec,
        "confusion_matrix": cm,
        "history": history,
        "device": str(device)
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved Metadata: {meta_path.resolve()}")

    # Export to ONNX (For Free Render Deployment)
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
            print(f"ONNX Model Exported: {best_onnx.resolve()}")
            print(f"ONNX Size: {best_onnx.stat().st_size / (1024*1024):.2f} MB (Runs on Render free tier in ~45MB RAM)")
        except Exception as e:
            print(f"[WARNING] ONNX export: {e}")

    # Generate Grad-CAM Sample
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
            sample_out = gradcam_dir / "sample_wound_gradcam.jpg"

            overlay = overlay_heatmap(sample_img_path, cam)
            cv2.imwrite(str(sample_out), overlay)
            print(f"Grad-CAM Heatmap saved: {sample_out.resolve()}")
            gradcam.remove_hooks()
        except Exception as e:
            print(f"[WARNING] Grad-CAM: {e}")

    print("=" * 75)


if __name__ == "__main__":
    main()
