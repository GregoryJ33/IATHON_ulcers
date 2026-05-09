import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import random
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

# =========================
# CLASSES
# =========================

CLASS_NAMES = [
    "Invalid",
    "Sdti",
    "Stage_I",
    "Stage_II",
    "Stage_III",
    "Stage_IV",
    "Unstageable"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

BINARY_LABELS = {
    "Invalid":     0,
    "Sdti":        1,
    "Stage_I":     1,
    "Stage_II":    1,
    "Stage_III":   1,
    "Stage_IV":    1,
    "Unstageable": 1
}


# =========================
# CLASSIFICATION DATASET
# =========================

class UlcerDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, CLASS_TO_IDX[label], BINARY_LABELS[label]

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Retourne un WeightedRandomSampler pour rééquilibrer les classes.
        Les classes rares sont sur-échantillonnées proportionnellement.
        """
        labels  = [CLASS_TO_IDX[y] for _, y in self.samples]
        counts  = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
        weights = 1.0 / (counts + 1e-6)
        sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float)
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


# =========================
# SEGMENTATION DATASET
# =========================

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None, size=224):
        self.image_paths  = image_paths
        self.mask_paths   = mask_paths
        self.transform    = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img  = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()
        return img, mask


# =========================
# TRANSFORMS
# =========================

def get_transforms(size=224):
    train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train, val


# =========================
# LOAD CLASSIFICATION DATA
# =========================

def load_classification(data_dir):
    samples = []
    for cls in CLASS_NAMES:
        folder = os.path.join(data_dir, cls)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append((os.path.join(folder, f), cls))
    return samples


# =========================
# SPLIT (stratifié sklearn)
# =========================

def split_data(samples):
    """
    Split stratifié 70 / 10 / 20 (train / val / test).
    Stratification sur la classe pour préserver les proportions
    même avec des classes à très peu d'exemples.
    """
    labels = [y for _, y in samples]

    # 1er split : train (80%) | test (20%)
    train, test = train_test_split(
        samples, test_size=0.2, stratify=labels, random_state=42
    )

    # 2ème split : train (87.5% de 80% ≈ 70%) | val (12.5% de 80% ≈ 10%)
    train_labels = [y for _, y in train]
    train, val = train_test_split(
        train, test_size=0.125, stratify=train_labels, random_state=42
    )

    print(f"Split → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


# =========================
# LOAD SEGMENTATION DATA
# (dataset pieds — déjà splitté en train/val par dossier)
# =========================

def load_segmentation(split_dir):
    """Charge un dossier déjà splitté (train ou validation) du dataset pieds."""
    img_dir  = os.path.join(split_dir, "images")
    mask_dir = os.path.join(split_dir, "labels")
    imgs, masks = [], []
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            imgs.append(os.path.join(img_dir, f))
            masks.append(os.path.join(mask_dir, f))
    return imgs, masks


# =========================
# LOAD + SPLIT ULCÈRE SEG
# =========================

def load_ulcer_segmentation(data_dir):
    """
    Charge le dataset UlcereSegmentation (images/ + labels/).
    Retourne toutes les paires (image, masque).
    """
    img_dir  = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "labels")
    imgs, masks = [], []
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path  = os.path.join(img_dir, f)
            mask_path = os.path.join(mask_dir, f)
            if os.path.exists(mask_path):
                imgs.append(img_path)
                masks.append(mask_path)
            else:
                print(f"  [WARN] Masque manquant pour {f}, ignoré.")
    print(f"  UlcèreSegmentation : {len(imgs)} paires image/masque trouvées")
    return imgs, masks


def split_segmentation(imgs, masks, val_ratio=0.15, seed=42):
    """
    Split train/val pour le dataset de segmentation ulcères.
    Avec seulement ~33 images, on garde 85% en train et 15% en val.
    Pas de test set ici, l'évaluation visuelle se fait dans le pipeline final.
    """
    pairs = list(zip(imgs, masks))
    random.seed(seed)
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * val_ratio))
    val_p   = pairs[:n_val]
    train_p = pairs[n_val:]
    train_imgs,  train_masks  = zip(*train_p) if train_p else ([], [])
    val_imgs,    val_masks    = zip(*val_p)   if val_p   else ([], [])
    print(f"  Split ulcère seg → Train: {len(train_imgs)} | Val: {len(val_imgs)}")
    return list(train_imgs), list(train_masks), list(val_imgs), list(val_masks)