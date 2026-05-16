import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# On importe toute la mécanique depuis ton fichier dataset.py
from dataset import load_classification, split_data, get_transforms, UlcerDataset, CLASS_NAMES

# =====================================================================
# 1. ARCHITECTURE DU CLASSIFIER (ResNet18 - 6 Classes)
# =====================================================================
class Classifier(nn.Module):
    def __init__(self, num_classes=6, dropout=0.4):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )
        self.freeze_backbone()

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_last_layers(self, n_layers: int = 2):
        self.freeze_backbone()
        layers_list = ["layer4", "layer3", "layer2", "layer1"]
        for layer_name in layers_list[:n_layers]:
            for param in getattr(self.model, layer_name).parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Classifier] {n_layers} couche(s) dégelée(s) — params entraînables : {n_trainable:,}")

    def forward(self, x):
        return self.model(x)


# =====================================================================
# 2. SCRIPT D'ENTRAÎNEMENT PRINCIPAL WITH ACCURACY
# =====================================================================
if __name__ == "__main__":
    print("\n==============================================")
    print("CLASSIFICATION TRAINING — MODÈLE MULTI-STADES")
    print("==============================================\n")

    PATH_E2 = "dataset_etape2/"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS_PHASE1 = 10
    EPOCHS_PHASE2 = 20
    LR_HEAD = 1e-3
    LR_BACKBONE = 1e-5
    LABEL_SMOOTHING = 0.1

    print(f"Calculs configurés sur l'appareil : {DEVICE}")

    # 1. Chargement et partitionnement des images
    samples = load_classification(PATH_E2)
    print(f"Nombre total d'images trouvées pour les stades : {len(samples)}\n")
    
    train_samples, val_samples, test_samples = split_data(samples)
    train_tf, val_tf = get_transforms()

    train_ds = UlcerDataset(train_samples, transform=train_tf)
    val_ds   = UlcerDataset(val_samples,   transform=val_tf)

    sampler = train_ds.get_weighted_sampler()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # 2. Initialisation du réseau
    clf_model = Classifier(num_classes=len(CLASS_NAMES), dropout=0.4).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Listes pour stocker les métriques
    clf_train_losses, clf_val_losses = [], []
    clf_train_accs, clf_val_accs = [], []
    best_clf_loss = float("inf")
    phase_boundary = None

    # == Phase 1 : warm-up tête ===============================
    print(f"\n ▶ Phase 1 — Warm-up de la tête ({EPOCHS_PHASE1} époques)")

    optimizer_clf = torch.optim.Adam(
        filter(lambda p: p.requires_grad, clf_model.parameters()), lr=LR_HEAD
    )
    scheduler_clf = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_clf, mode="min", factor=0.5, patience=3
    )

    for epoch in range(EPOCHS_PHASE1):
        clf_model.train()
        running_loss = 0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer_clf.zero_grad()
            
            outputs = clf_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_clf.step()
            
            running_loss += loss.item()
            
            # Calcul de l'accuracy train
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        clf_model.eval()
        running_val = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = clf_model(images)
                
                running_val += criterion(outputs, labels).item()
                
                # Calcul de l'accuracy val
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = running_val / len(val_loader)
        val_acc = correct_val / total_val

        scheduler_clf.step(val_loss)
        clf_train_losses.append(train_loss)
        clf_val_losses.append(val_loss)
        clf_train_accs.append(train_acc)
        clf_val_accs.append(val_acc)

        if val_loss < best_clf_loss:
            best_clf_loss = val_loss
            torch.save(clf_model.state_dict(), "best_classifier_model.pth")
            print("  Meilleur modèle sauvegardé.")

        print(f"  [P1] Epoch {epoch+1}/{EPOCHS_PHASE1} | Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}%")

    # == Phase 2 : dégel progressif ===========================
    phase_boundary = len(clf_train_losses)
    print(f"\n ▶ Phase 2 — Fine-tuning par dégel progressif ({EPOCHS_PHASE2} époques)")

    clf_model.unfreeze_last_layers(n_layers=2)

    optimizer_clf = torch.optim.Adam([
        {"params": clf_model.model.fc.parameters(),     "lr": LR_HEAD},
        {"params": clf_model.model.layer3.parameters(), "lr": LR_BACKBONE},
        {"params": clf_model.model.layer4.parameters(), "lr": LR_BACKBONE},
    ], weight_decay=1e-4)
    scheduler_clf = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_clf, T_max=EPOCHS_PHASE2, eta_min=1e-7
    )

    for epoch in range(EPOCHS_PHASE2):
        clf_model.train()
        running_loss = 0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer_clf.zero_grad()
            
            outputs = clf_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf_model.parameters(), max_norm=1.0)
            optimizer_clf.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        clf_model.eval()
        running_val = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = clf_model(images)
                
                running_val += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = running_val / len(val_loader)
        val_acc = correct_val / total_val

        scheduler_clf.step()
        clf_train_losses.append(train_loss)
        clf_val_losses.append(val_loss)
        clf_train_accs.append(train_acc)
        clf_val_accs.append(val_acc)

        if val_loss < best_clf_loss:
            best_clf_loss = val_loss
            torch.save(clf_model.state_dict(), "best_classifier_model.pth")
            print("  Meilleur modèle sauvegardé.")

        print(f"  [P2] Epoch {epoch+1}/{EPOCHS_PHASE2} | Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}%")

    # == Génération des graphiques combinés (Loss et Accuracy) ==
    epochs_x = list(range(1, len(clf_train_losses) + 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Graphe 1 : La Perte (Loss)
    ax1.plot(epochs_x, clf_train_losses, label="Train Loss", color="#2196F3", linewidth=2)
    ax1.plot(epochs_x, clf_val_losses,   label="Val Loss",   color="#FF5722", linewidth=2, linestyle="--")
    ax1.axvline(x=phase_boundary, color="#9E9E9E", linestyle=":", linewidth=1.5, label="Début Phase 2")
    ax1.set_title("Évolution de la Perte (Loss)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Graphe 2 : L'Exactitude (Accuracy)
    ax2.plot(epochs_x, [a * 100 for a in clf_train_accs], label="Train Acc", color="#4CAF50", linewidth=2)
    ax2.plot(epochs_x, [a * 100 for a in clf_val_accs],   label="Val Acc",   color="#E91E63", linewidth=2, linestyle="--")
    ax2.axvline(x=phase_boundary, color="#9E9E9E", linestyle=":", linewidth=1.5, label="Début Phase 2")
    ax2.set_title("Évolution de la Précision (Accuracy)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("monitoring_stades_pytorch.png", dpi=150)
    print("\nEntraînement terminé ! Les graphiques de suivi ont été enregistrés sous : 'monitoring_stades_pytorch.png'")