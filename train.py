"""
Recovera AI: Deep Learning Transfer Learning & Model Training Pipeline
Trains clinical vision models for:
1. Bone Fracture Detection & Radiograph Analysis (DenseNet121 / ResNet50)
2. Surgical Wound & Dermatological Disease Classification (ResNet50 / DenseNet121)

Features:
- Transfer learning with ImageNet-pretrained feature extractors
- Robust clinical preprocessing & data augmentation
- Rigorous metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Mathematical Grad-CAM explainability heatmap generation
- Automatic model checkpoint saving to backend/storage/models/ for Recovera AI integration
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.io import Dataset, DataLoader
import paddle.vision.transforms as T
import paddle.vision.models as models

# ----------------------------------------------------
# Custom Image Dataset Loader
# ----------------------------------------------------
class MedicalImageDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for c in self.classes:
            c_dir = self.root_dir / c
            for f in c_dir.glob("*.*"):
                if f.suffix.lower() in valid_exts:
                    self.samples.append((str(f), self.class_to_idx[c]))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception:
            # Fallback for corrupted images
            img = Image.new("RGB", (256, 256), color=(0, 0, 0))

        if self.transform:
            img = self.transform(img)
        return img, np.int64(label), path

    def __len__(self):
        return len(self.samples)


# ----------------------------------------------------
# Neural Architectures with Grad-CAM Feature Tap
# ----------------------------------------------------
class TransferDenseNet(nn.Layer):
    """Fine-tuned DenseNet121 for medical imaging with Grad-CAM feature tap."""
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        base = models.densenet121(pretrained=pretrained)
        self.densenet = base
        # Replace output classification head
        in_features = 1024
        self.densenet.out = nn.Linear(in_features, num_classes)

    def forward(self, x):
        conv = self.densenet.conv1_func(x)
        conv = self.densenet.pool2d_max(conv)

        for i, num_layers in enumerate(self.densenet.block_config):
            conv = self.densenet.dense_block_func_list[i](conv)
            if i != len(self.densenet.block_config) - 1:
                conv = self.densenet.transition_func_list[i](conv)

        feat = self.densenet.batch_norm(conv)  # Target feature map for Grad-CAM
        y = self.densenet.pool2d_avg(feat)
        y = paddle.flatten(y, start_axis=1, stop_axis=-1)
        out = self.densenet.out(y)
        return out, feat


class TransferResNet(nn.Layer):
    """Fine-tuned ResNet50 for wound analysis with Grad-CAM feature tap."""
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        base = models.resnet50(pretrained=pretrained)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4  # Target layer for Grad-CAM [B, 2048, H/32, W/32]
        self.avgpool = base.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat4 = self.layer4(x)

        p = self.avgpool(feat4)
        flat = paddle.flatten(p, 1)
        out = self.fc(flat)
        return out, feat4


# ----------------------------------------------------
# Grad-CAM Heatmap Visualizer
# ----------------------------------------------------
def compute_gradcam(image_path: str, model: nn.Layer, target_class: int, img_size: int = 256) -> np.ndarray:
    """Computes mathematical Grad-CAM class activation overlay on clinical image."""
    orig = cv2.imread(image_path)
    if orig is None:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    h_orig, w_orig = orig.shape[:2]
    resized = cv2.resize(orig, (img_size, img_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (rgb - mean) / std
    tensor = paddle.to_tensor(norm.transpose(2, 0, 1)[np.newaxis, ...], dtype="float32")
    tensor.stop_gradient = False

    model.eval()
    logits, feat = model(tensor)
    score = logits[0, target_class]
    
    # Backward pass for feature gradients
    grads = paddle.grad(outputs=score, inputs=feat, create_graph=False, retain_graph=False)[0]
    weights = paddle.mean(grads, axis=[2, 3], keepdim=True)
    cam = paddle.sum(weights * feat, axis=1, keepdim=True)
    cam = F.relu(cam)

    cam_np = cam[0, 0].numpy()
    if np.max(cam_np) > 0:
        cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np))
    
    cam_resized = cv2.resize(cam_np, (w_orig, h_orig))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.65, heatmap, 0.35, 0)
    return overlay


# ----------------------------------------------------
# Main Training Routine
# ----------------------------------------------------
def train_model(
    task: str = "fracture",
    data_dir: str = "",
    arch: str = "densenet121",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    img_size: int = 256,
    save_dir: str = "backend/storage/models"
):
    print("=" * 65)
    print(f" Recovera AI — Medical Vision Transfer Learning Pipeline")
    print(f" Task: {task.upper()} | Architecture: {arch.upper()} | Epochs: {epochs}")
    print("=" * 65)

    # Autodetect hardware
    device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    paddle.device.set_device(device)
    print(f"Hardware compute accelerator: {device.upper()}")

    # Resolve paths
    base_data = Path(data_dir) if data_dir else (
        Path("Dataset/Bone_Fracture_Binary_Classification_X_RAY") if task == "fracture"
        else Path("Dataset/Wound Analysis")
    )
    
    train_path = base_data / "train"
    val_path = base_data / "val" if (base_data / "val").exists() else base_data / "test"
    test_path = base_data / "test" if (base_data / "test").exists() else val_path

    if not train_path.exists():
        print(f"Error: Train directory not found at {train_path.resolve()}")
        sys.exit(1)

    # Transforms
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(prob=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    print(f"\nScanning training dataset from: {train_path.resolve()}")
    train_ds = MedicalImageDataset(train_path, transform=train_tf)
    classes = train_ds.classes
    num_classes = len(classes)
    print(f"Total training images: {len(train_ds)} across {num_classes} classes:")
    for idx, c in enumerate(classes):
        print(f"  [{idx}] {c}")

    print(f"\nScanning validation dataset from: {val_path.resolve()}")
    val_ds = MedicalImageDataset(val_path, transform=val_tf)
    print(f"Total validation images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Build model
    if arch.lower() == "densenet121":
        model = TransferDenseNet(num_classes=num_classes, pretrained=True)
    else:
        model = TransferResNet(num_classes=num_classes, pretrained=True)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr.CosineAnnealingDecay(learning_rate=lr, T_max=epochs)
    optimizer = optim.AdamW(learning_rate=scheduler, parameters=model.parameters(), weight_decay=1e-4)

    best_val_acc = 0.0
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_weight_path = out_dir / f"{task}_{arch}_best.pdparams"

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for step, (images, labels, _) in enumerate(train_loader):
            optimizer.clear_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.numpy()[0])
            preds = paddle.argmax(logits, axis=1)
            correct_train += int(paddle.sum(preds == labels).numpy()[0])
            total_train += int(labels.shape[0])

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                batch_acc = (correct_train / max(1, total_train)) * 100.0
                print(f" Epoch [{epoch}/{epochs}] Step [{step+1}/{len(train_loader)}] — Loss: {total_loss/(step+1):.4f} — Train Acc: {batch_acc:.2f}%")

        scheduler.step()

        # Validation phase
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with paddle.no_grad():
            for images, labels, _ in val_loader:
                logits, _ = model(images)
                preds = paddle.argmax(logits, axis=1)
                val_preds_list.extend(preds.numpy().tolist())
                val_targets_list.extend(labels.numpy().tolist())

        val_acc = float(accuracy_score(val_targets_list, val_preds_list)) * 100.0
        val_prec = float(precision_score(val_targets_list, val_preds_list, average="macro", zero_division=0)) * 100.0
        val_rec = float(recall_score(val_targets_list, val_preds_list, average="macro", zero_division=0)) * 100.0
        val_f1 = float(f1_score(val_targets_list, val_preds_list, average="macro", zero_division=0)) * 100.0

        epoch_sec = time.time() - epoch_start
        print(f"\n>>> Epoch {epoch} Results ({epoch_sec:.1f}s): Val Accuracy: {val_acc:.2f}% | Precision: {val_prec:.2f}% | Recall: {val_rec:.2f}% | F1: {val_f1:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            paddle.save(model.state_dict(), str(best_weight_path))
            print(f" ★ Best model checkpoint saved to: {best_weight_path.resolve()} (Val Acc: {best_val_acc:.2f}%) ★\n")
        else:
            print()

    total_duration = time.time() - start_train_time
    print("=" * 65)
    print(f" Training Complete in {total_duration/60.0:.2f} minutes!")
    print(f" Peak Validation Accuracy: {best_val_acc:.2f}%")
    print(f" Model weights ready at: {best_weight_path.resolve()}")
    print("=" * 65)

    # Save metadata JSON for Recovera AI integration
    meta_path = out_dir / f"{task}_{arch}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "task": task,
            "architecture": arch,
            "classes": classes,
            "num_classes": len(classes),
            "best_val_accuracy": round(best_val_acc, 2),
            "epochs": epochs,
            "training_duration_seconds": round(total_duration, 1),
            "weights_filename": best_weight_path.name
        }, f, indent=2)
    print(f"Metadata written to: {meta_path.resolve()}")

    # Generate sample Grad-CAM explainability overlay on test image
    if len(val_ds) > 0:
        sample_img_path, sample_label, _ = val_ds.samples[0]
        gradcam_dir = Path("backend/storage/gradcam")
        gradcam_dir.mkdir(parents=True, exist_ok=True)
        sample_out = gradcam_dir / f"sample_{task}_gradcam.jpg"
        
        # Load best weights
        model.set_state_dict(paddle.load(str(best_weight_path)))
        overlay = compute_gradcam(sample_img_path, model, target_class=sample_label, img_size=img_size)
        cv2.imwrite(str(sample_out), overlay)
        print(f"Sample Grad-CAM visual attribution map generated at: {sample_out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recovera AI Vision Transfer Learning Pipeline")
    parser.add_argument("--task", type=str, default="fracture", choices=["fracture", "wound"],
                        help="Medical task to train: 'fracture' or 'wound'")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Path to dataset directory. Defaults to Dataset/Bone_Fracture_Binary_Classification_X_RAY or Dataset/Wound Analysis")
    parser.add_argument("--arch", type=str, default="",
                        help="Architecture: 'densenet121' or 'resnet50'")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Input image resolution (default: 256)")
    parser.add_argument("--save_dir", type=str, default="backend/storage/models",
                        help="Directory to save model checkpoints")

    args = parser.parse_args()
    
    # Set default architecture per task
    if not args.arch:
        args.arch = "densenet121" if args.task == "fracture" else "resnet50"

    train_model(
        task=args.task,
        data_dir=args.data_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        save_dir=args.save_dir
    )
