import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

# Config des 6 classes cibles (sans la classe Invalid)
CLASS_NAMES = [
    "SDTI",
    "Stage_1",
    "Stage_2",
    "Stage_3",
    "Stage_4",
    "Unstageable"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

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
        return image, CLASS_TO_IDX[label]

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        labels  = [CLASS_TO_IDX[y] for _, y in self.samples]
        counts  = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(float)
        weights = 1.0 / (counts + 1e-6)
        sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float)
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

def get_transforms(size=224):
    train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train, val

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

def split_data(samples):
    labels = [y for _, y in samples]
    train, test = train_test_split(
        samples, test_size=0.2, stratify=labels, random_state=42
    )
    train_labels = [y for _, y in train]
    train, val = train_test_split(
        train, test_size=0.125, stratify=train_labels, random_state=42
    )
    print(f"Split -> Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test