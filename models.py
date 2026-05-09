import torch
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp


# =========================
# CLASSIFIER (ResNet18)
# avec dégel progressif
# =========================

class Classifier(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remplace la tête par une tête avec dropout
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )

        # Phase 1 : geler tout sauf la tête
        self.freeze_backbone()

    def freeze_backbone(self):
        """Gèle tout le backbone ResNet (hors fc)."""
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_last_layers(self, n_layers: int = 2):
        """
        Dégèle les n derniers blocs ResNet pour le fine-tuning.
        ResNet18 a : layer1, layer2, layer3, layer4 + fc
        n_layers=1 → dégèle layer4
        n_layers=2 → dégèle layer3 + layer4
        """
        self.freeze_backbone()
        layers = ["layer4", "layer3", "layer2", "layer1"]
        for layer_name in layers[:n_layers]:
            for param in getattr(self.model, layer_name).parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Classifier] {n_layers} couche(s) dégelée(s) — params entraînables : {n_trainable:,}")

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


# =========================
# SEGMENTATION (U-Net / ResNet34)
# =========================

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x):
        return self.model(x)