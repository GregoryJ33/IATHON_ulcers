# main.py

import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from dataset import (
    load_classification, split_data,
    load_segmentation, load_ulcer_segmentation, split_segmentation,
    UlcerDataset, SegmentationDataset, get_transforms, CLASS_NAMES
)
from models import Classifier, UNet


# =========================================================
# CONFIG
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE         = 8
EPOCHS_CLF_PHASE1  = 10
EPOCHS_CLF_PHASE2  = 20
EPOCHS_SEG         = 15
EPOCHS_SEG_FINETUNE = 20
LR_HEAD            = 3e-4
LR_BACKBONE        = 3e-5
LR_SEG             = 3e-4
LR_SEG_FINETUNE    = 5e-5
SEG_THRESHOLD      = 0.3
LABEL_SMOOTHING    = 0.1

CLASSIFICATION_DATASET  = "Datasets/UlcereClassification"
SEG_TRAIN_DIR           = "Datasets/FootSegmentation/train"
SEG_VAL_DIR             = "Datasets/FootSegmentation/validation"
ULCER_SEG_DIR           = "Datasets/UlcereSegmentation"


# =========================================================
# UTILS
# =========================================================

train_tf, val_tf = get_transforms()

def denormalize(tensor_img):
    img = tensor_img.permute(1, 2, 0).cpu().numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)


# =========================================================
# PART 1 — ENTRAÎNEMENT SEGMENTATION
# =========================================================

print("\n==============================")
print("SEGMENTATION TRAINING")
print("==============================\n")

train_imgs, train_masks = load_segmentation(SEG_TRAIN_DIR)
val_imgs,   val_masks   = load_segmentation(SEG_VAL_DIR)

seg_train_ds = SegmentationDataset(train_imgs, train_masks, transform=train_tf)
seg_val_ds   = SegmentationDataset(val_imgs,   val_masks,   transform=val_tf)

seg_train_loader = DataLoader(seg_train_ds, batch_size=BATCH_SIZE, shuffle=True)
seg_val_loader   = DataLoader(seg_val_ds,   batch_size=BATCH_SIZE)

seg_model     = UNet().to(DEVICE)
bce_loss      = nn.BCEWithLogitsLoss()
dice_loss     = smp.losses.DiceLoss(mode="binary", from_logits=True)
optimizer_seg = torch.optim.Adam(seg_model.parameters(), lr=LR_SEG)
scheduler_seg = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_seg, mode="min", factor=0.5, patience=3
)

seg_train_losses, seg_val_losses = [], []
best_seg_loss = float("inf")

for epoch in range(EPOCHS_SEG):
    seg_model.train()
    running_loss = 0
    for images, masks in seg_train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer_seg.zero_grad()
        loss = 0.5 * bce_loss(seg_model(images), masks) + 0.5 * dice_loss(seg_model(images), masks)
        loss.backward()
        optimizer_seg.step()
        running_loss += loss.item()
    train_loss = running_loss / len(seg_train_loader)

    seg_model.eval()
    running_val = 0
    running_dice = 0
    running_iou = 0
    with torch.no_grad():
        for images, masks in seg_val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            out = seg_model(images)
            running_val += (0.5 * bce_loss(out, masks) + 0.5 * dice_loss(out, masks)).item()
            running_dice += smp.metrics.f1_score(*smp.metrics.get_stats(out > SEG_THRESHOLD, masks.long(), mode="binary"), reduction="micro")
            running_iou += smp.metrics.iou_score(*smp.metrics.get_stats(out > SEG_THRESHOLD, masks.long(), mode="binary"), reduction="micro")
    val_loss = running_val / len(seg_val_loader)
    val_dice = running_dice / len(seg_val_loader)
    val_iou = running_iou / len(seg_val_loader)

    scheduler_seg.step(val_loss)
    seg_train_losses.append(train_loss)
    seg_val_losses.append(val_loss)

    if val_loss < best_seg_loss:
        best_seg_loss = val_loss
        torch.save(seg_model.state_dict(), "Checkpoints/best_segmentation_model.pth")
        print("Best segmentation model saved.")

    print(f"Epoch {epoch+1}/{EPOCHS_SEG} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Val dice: {val_dice:.4f} | Val iou: {val_iou:.4f} | LR: {optimizer_seg.param_groups[0]['lr']:.6f}")


# =========================================================
# FIGURE 1 — Courbe de loss segmentation
# =========================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(seg_train_losses, label="Train", color="#2196F3", linewidth=2)
ax.plot(seg_val_losses,   label="Val",   color="#FF5722", linewidth=2, linestyle="--")
ax.set_title("Segmentation — Courbe de perte (BCE + Dice)", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Results2/fig1_seg_loss.png", dpi=150)
plt.show()


# =========================================================
# FIGURE 2 — Exemples de segmentation sur les pieds
# =========================================================
seg_model.load_state_dict(torch.load("Checkpoints/best_segmentation_model.pth"))
seg_model.eval()

n_foot = 3
foot_indices = random.sample(range(len(seg_val_ds)), n_foot)

fig, axes = plt.subplots(n_foot, 3, figsize=(9, 3.0 * n_foot))
fig.suptitle("Segmentation — Exemples sur le jeu de validation (pieds)",
             fontsize=12, fontweight="bold", y=1.01)

# Titres des colonnes uniquement sur la première ligne
col_titles = ["Image originale", "Masque prédit (overlay)", "Ground truth"]
for j, title in enumerate(col_titles):
    axes[0, j].set_title(title, fontsize=9, pad=6)

for row, idx in enumerate(foot_indices):
    img_t, mask_t = seg_val_ds[idx]
    with torch.no_grad():
        prob = torch.sigmoid(
            seg_model(img_t.unsqueeze(0).to(DEVICE))
        ).squeeze().cpu().numpy()

    img_np = denormalize(img_t)

    axes[row, 0].imshow(img_np)
    axes[row, 0].axis("off")

    axes[row, 1].imshow(img_np)
    axes[row, 1].imshow(prob, cmap="hot", alpha=0.5, vmin=0, vmax=1)
    axes[row, 1].axis("off")

    axes[row, 2].imshow(mask_t.squeeze().numpy(), cmap="gray")
    axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig("Results2/fig2_seg_examples.png", dpi=150, bbox_inches="tight")
plt.show()


# =========================================================
# PART 1b — FINE-TUNING SEGMENTATION SUR ULCÈRES
# =========================================================

print("\n==============================")
print("SEGMENTATION FINE-TUNING (ulcères)")
print("==============================\n")

ulcer_imgs, ulcer_masks = load_ulcer_segmentation(ULCER_SEG_DIR)
ft_train_imgs, ft_train_masks, ft_val_imgs, ft_val_masks = split_segmentation(
    ulcer_imgs, ulcer_masks, val_ratio=0.15
)

ft_train_ds = SegmentationDataset(ft_train_imgs, ft_train_masks, transform=train_tf)
ft_val_ds   = SegmentationDataset(ft_val_imgs,   ft_val_masks,   transform=val_tf)

ft_batch        = min(4, len(ft_train_ds))
ft_train_loader = DataLoader(ft_train_ds, batch_size=ft_batch, shuffle=True)
ft_val_loader   = DataLoader(ft_val_ds,   batch_size=ft_batch)

# Repart du meilleur checkpoint pieds
seg_model.load_state_dict(torch.load("Checkpoints/best_segmentation_model.pth"))

# lr très faible pour ne pas effacer ce qui a été appris sur les pieds
optimizer_ft = torch.optim.Adam(seg_model.parameters(), lr=LR_SEG_FINETUNE)
scheduler_ft = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_ft, mode="min", factor=0.5, patience=4
)

ft_train_losses, ft_val_losses = [], []
best_ft_loss = float("inf")

for epoch in range(EPOCHS_SEG_FINETUNE):
    seg_model.train()
    running_loss = 0
    for images, masks in ft_train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer_ft.zero_grad()
        out  = seg_model(images)
        loss = 0.5 * bce_loss(out, masks) + 0.5 * dice_loss(out, masks)
        loss.backward()
        optimizer_ft.step()
        running_loss += loss.item()
    train_loss = running_loss / len(ft_train_loader)

    seg_model.eval()
    running_val = 0
    with torch.no_grad():
        for images, masks in ft_val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            out = seg_model(images)
            running_val += (0.5 * bce_loss(out, masks) + 0.5 * dice_loss(out, masks)).item()
    val_loss = running_val / len(ft_val_loader)

    scheduler_ft.step(val_loss)
    ft_train_losses.append(train_loss)
    ft_val_losses.append(val_loss)

    if val_loss < best_ft_loss:
        best_ft_loss = val_loss
        torch.save(seg_model.state_dict(), "Checkpoints/best_segmentation_model.pth")
        print("Best fine-tuned seg model saved.")

    print(f"[FT] Epoch {epoch+1}/{EPOCHS_SEG_FINETUNE} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer_ft.param_groups[0]['lr']:.7f}")


# =========================================================
# FIGURE 1bis — Courbe fine-tuning segmentation ulcères
# =========================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ft_train_losses, label="Train", color="#2196F3", linewidth=2)
ax.plot(ft_val_losses,   label="Val",   color="#FF5722", linewidth=2, linestyle="--")
ax.set_title("Fine-tuning segmentation — ulcères (BCE + Dice)", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Results2/fig1b_seg_finetune_loss.png", dpi=150)
plt.show()


# =========================================================
# FIGURE 2bis — Exemples de segmentation sur ulcères
# =========================================================
seg_model.load_state_dict(torch.load("Checkpoints/best_segmentation_model.pth"))
seg_model.eval()

n_ulcer_ex = min(3, len(ft_val_ds))
ulcer_idxs = random.sample(range(len(ft_val_ds)), n_ulcer_ex)

fig, axes = plt.subplots(n_ulcer_ex, 3, figsize=(9, 3.0 * n_ulcer_ex))
if n_ulcer_ex == 1:
    axes = np.expand_dims(axes, 0)
fig.suptitle("Segmentation fine-tunée — Exemples sur ulcères",
             fontsize=12, fontweight="bold", y=1.01)

axes[0, 0].set_title("Image originale",          fontsize=9, pad=6)
axes[0, 1].set_title("Masque prédit (overlay)",  fontsize=9, pad=6)
axes[0, 2].set_title("Ground truth",             fontsize=9, pad=6)

for row, idx in enumerate(ulcer_idxs):
    img_t, mask_t = ft_val_ds[idx]
    with torch.no_grad():
        prob = torch.sigmoid(
            seg_model(img_t.unsqueeze(0).to(DEVICE))
        ).squeeze().cpu().numpy()
    img_np = denormalize(img_t)

    axes[row, 0].imshow(img_np)
    axes[row, 0].axis("off")
    axes[row, 1].imshow(img_np)
    axes[row, 1].imshow(prob, cmap="hot", alpha=0.5, vmin=0, vmax=1)
    axes[row, 1].axis("off")
    axes[row, 2].imshow(mask_t.squeeze().numpy(), cmap="gray")
    axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig("Results2/fig2b_seg_ulcer_examples.png", dpi=150, bbox_inches="tight")
plt.show()


# =========================================================
# PART 2 — ENTRAÎNEMENT CLASSIFICATION (2 phases)
# =========================================================

print("\n==============================")
print("CLASSIFICATION TRAINING")
print("==============================\n")

samples = load_classification(CLASSIFICATION_DATASET)

train_samples, val_samples, test_samples = split_data(samples)

train_ds = UlcerDataset(train_samples, transform=train_tf)
val_ds   = UlcerDataset(val_samples,   transform=val_tf)
test_ds  = UlcerDataset(test_samples,  transform=val_tf)

sampler      = train_ds.get_weighted_sampler()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=1)

clf_model  = Classifier(num_classes=len(CLASS_NAMES), dropout=0.4).to(DEVICE)
criterion  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

clf_train_losses, clf_val_losses = [], []
best_clf_loss  = float("inf")
phase_boundary = None

# == Phase 1 : warm-up tête ===============================
print(f"\nPhase 1 — Warm-up tête ({EPOCHS_CLF_PHASE1} epochs)")

optimizer_clf = torch.optim.Adam(
    filter(lambda p: p.requires_grad, clf_model.parameters()), lr=LR_HEAD
)
scheduler_clf = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_clf, mode="min", factor=0.5, patience=3
)

for epoch in range(EPOCHS_CLF_PHASE1):
    clf_model.train()
    running_loss = 0
    for images, labels, _ in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer_clf.zero_grad()
        loss = criterion(clf_model(images), labels)
        loss.backward()
        optimizer_clf.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    clf_model.eval()
    running_val = 0
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            running_val += criterion(clf_model(images), labels).item()
    val_loss = running_val / len(val_loader)

    scheduler_clf.step(val_loss)
    clf_train_losses.append(train_loss)
    clf_val_losses.append(val_loss)

    if val_loss < best_clf_loss:
        best_clf_loss = val_loss
        torch.save(clf_model.state_dict(), "Checkpoints/best_classifier_model.pth")
        print("Best classifier saved.")

    print(f"[P1] Epoch {epoch+1}/{EPOCHS_CLF_PHASE1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer_clf.param_groups[0]['lr']:.6f}")

# == Phase 2 : dégel progressif ===========================
phase_boundary = len(clf_train_losses)
print(f"\nPhase 2 — Fine-tuning ({EPOCHS_CLF_PHASE2} epochs)")

clf_model.unfreeze_last_layers(n_layers=2)

optimizer_clf = torch.optim.Adam([
    {"params": clf_model.model.fc.parameters(),     "lr": LR_HEAD},
    {"params": clf_model.model.layer3.parameters(), "lr": LR_BACKBONE},
    {"params": clf_model.model.layer4.parameters(), "lr": LR_BACKBONE},
], weight_decay=1e-4)
scheduler_clf = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_clf, T_max=EPOCHS_CLF_PHASE2, eta_min=1e-7
)

for epoch in range(EPOCHS_CLF_PHASE2):
    clf_model.train()
    running_loss = 0
    for images, labels, _ in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer_clf.zero_grad()
        loss = criterion(clf_model(images), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(clf_model.parameters(), max_norm=1.0)
        optimizer_clf.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    clf_model.eval()
    running_val = 0
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            running_val += criterion(clf_model(images), labels).item()
    val_loss = running_val / len(val_loader)

    scheduler_clf.step()
    clf_train_losses.append(train_loss)
    clf_val_losses.append(val_loss)

    if val_loss < best_clf_loss:
        best_clf_loss = val_loss
        torch.save(clf_model.state_dict(), "Checkpoints/best_classifier_model.pth")
        print("Best classifier saved.")

    print(f"[P2] Epoch {epoch+1}/{EPOCHS_CLF_PHASE2} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")


# =========================================================
# FIGURE 3 — Courbe de loss classification
# =========================================================
epochs_x = list(range(1, len(clf_train_losses) + 1))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs_x, clf_train_losses, label="Train", color="#2196F3", linewidth=2)
ax.plot(epochs_x, clf_val_losses,   label="Val",   color="#FF5722", linewidth=2, linestyle="--")
ax.axvline(x=phase_boundary, color="#9E9E9E", linestyle=":", linewidth=1.5, label="Début Phase 2")
ax.set_title("Classification — Courbe de perte", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Results2/fig3_clf_loss.png", dpi=150)
plt.show()


# =========================================================
# ÉVALUATION SUR LE TEST SET (console)
# =========================================================

clf_model.load_state_dict(torch.load("Checkpoints/best_classifier_model.pth"))
clf_model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels, _ in test_loader:
        preds = torch.argmax(clf_model(images.to(DEVICE)), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n========== CLASSIFICATION REPORT ==========\n")
print(classification_report(all_labels, all_preds,
      labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, zero_division=0))
print("========== CONFUSION MATRIX ==========\n")
print(confusion_matrix(all_labels, all_preds))


# =========================================================
# FIGURE 4 — Pipeline final
# Test set de CLASSIFICATION uniquement (aucune image vue
# pendant l'entraînement de la segmentation → pas de biais)
# 2 colonnes : Original + label vrai | Seg overlay + prédit
# =========================================================
print("\n==============================")
print("FINAL PIPELINE")
print("==============================\n")

seg_model.load_state_dict(torch.load("Checkpoints/best_segmentation_model.pth"))
seg_model.eval()
clf_model.load_state_dict(torch.load("Checkpoints/best_classifier_model.pth"))
clf_model.eval()

n_rows  = 4
indices = random.sample(range(len(test_ds)), min(n_rows, len(test_ds)))

fig, axes = plt.subplots(n_rows, 2, figsize=(7, 3.4 * n_rows))
fig.suptitle("Pipeline Final", fontsize=13, fontweight="bold", y=1.01)

axes[0, 0].set_title("Image originale",               fontsize=9, pad=6)
axes[0, 1].set_title("Segmentation + Classification", fontsize=9, pad=6)

for row, idx in enumerate(indices):
    image, true_label, _ = test_ds[idx]
    inp    = image.unsqueeze(0).to(DEVICE)
    img_np = denormalize(image)

    with torch.no_grad():
        prob = torch.sigmoid(seg_model(inp)).squeeze().cpu().numpy()
        probs      = F.softmax(clf_model(inp), dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item() * 100

    is_correct = pred_class == true_label
    fc_color   = "#2e7d32" if is_correct else "#c62828"

    # -- Col 0 : original + vrai label --------------------
    axes[row, 0].imshow(img_np)
    axes[row, 0].axis("off")
    axes[row, 0].text(
        0.5, 0.03, f"Vrai : {CLASS_NAMES[true_label]}",
        transform=axes[row, 0].transAxes,
        ha="center", va="bottom", fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc="#37474F", alpha=0.82, lw=0)
    )

    # -- Col 1 : seg overlay + classe prédite -------------
    axes[row, 1].imshow(img_np)
    axes[row, 1].imshow(prob, cmap="hot", alpha=0.45, vmin=0, vmax=1)
    axes[row, 1].axis("off")
    axes[row, 1].text(
        0.5, 0.03,
        f"Prédit : {CLASS_NAMES[pred_class]}  ({confidence:.0f}%)",
        transform=axes[row, 1].transAxes,
        ha="center", va="bottom", fontsize=8, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.2", fc=fc_color, alpha=0.85, lw=0)
    )

plt.subplots_adjust(hspace=0.08, wspace=0.05)
plt.savefig("Results2/fig4_pipeline_final.png", dpi=150, bbox_inches="tight")
plt.show()