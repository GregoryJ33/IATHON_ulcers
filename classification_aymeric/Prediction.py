import os
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Importation de TA liste synchronisée depuis ton fichier dataset.py
from dataset import CLASS_NAMES

# =====================================================================
# 1. ARCHITECTURE PYTORCH (Nécessaire pour mapper les poids du .pth)
# =====================================================================
class Classifier(nn.Module):
    def __init__(self, num_classes=6, dropout=0.4):
        super().__init__()
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# =====================================================================
# 2. FONCTION DE PRÉDICTION AVEC RENDU ET SAUVEGARDE UNIQUE
# =====================================================================
def predict_and_visualize(image_path, output_filename, model_keras_1, model_pytorch_2, device):
    """
    Exécute les deux modèles en cascade, extrait le vrai label via le dossier parent,
    génère un graphique à 3 colonnes et le sauvegarde sous 'output_filename'.
    """
    # Sécurité : on vérifie que l'image existe bien avant de bosser
    if not os.path.exists(image_path):
        print(f"❌ Fichier introuvable : '{image_path}' -> Étape ignorée.")
        return

    try:
        image_raw = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Impossible d'ouvrir l'image {image_path} : {e}")
        return

    # --- EXTRACTION ET DROIT AU BUT POUR LE VRAI LABEL ---
    dossier_parent = os.path.basename(os.path.dirname(os.path.abspath(image_path)))
    
    # On compare en majuscules pour éviter de bloquer sur une histoire de "sdti" vs "SDTI"
    classes_upper = [c.upper() for c in CLASS_NAMES]
    
    if dossier_parent.upper() in classes_upper or dossier_parent.lower() == "invalid":
        vrai_label = dossier_parent
    else:
        vrai_label = f"{dossier_parent} (Hors CLASS_NAMES)"

    # Variables par défaut pour le pipeline
    m1_status, m1_cert = "Unknown", 0.0
    m2_status, m2_cert = "Non requis", 0.0
    is_wound = False

   # -----------------------------------------------------------------
    # COLONNE 2 : ANALYSE MODÈLE 1 (KERAS - SÉCURITÉ ET DEBUG TOTAL)
    # -----------------------------------------------------------------
    # 1. On redimensionne l'image brute (comme à l'entraînement)
    img_keras = image_raw.resize((224, 224))
    img_array_keras = tf.keras.utils.img_to_array(img_keras)
    
    # 2. On ajoute la dimension de Batch -> Forme : (1, 224, 224, 3)
    img_array_keras = tf.expand_dims(img_array_keras, 0)

    # 3. Prédiction brute
    preds_keras = model_keras_1.predict(img_array_keras, verbose=0)
    
    # Extraction de la valeur unique de la Sigmoïde
    valeur_brute = float(preds_keras[0][0])
    
    # AFFICHE DANS TON TERMINAL LA VRAIE VALEUR POUR COMPRENDRE
    print(f"-> [DEBUG] Valeur brute renvoyée par Keras pour {os.path.basename(image_path)} : {valeur_brute:.6f}")

    # Logique d'attribution basée sur l'ordre alphabétique strict de Keras
    # 0 = Invalid (proche de 0) | 1 = Wound (proche de 1)
    if valeur_brute >= 0.5:
        m1_status = "PLAIE DÉTECTÉE"
        m1_cert = valeur_brute * 100
        is_wound = True
    else:
        m1_status = "IMAGE INVALID"
        m1_cert = (1.0 - valeur_brute) * 100
        is_wound = False

    # -----------------------------------------------------------------
    # COLONNE 3 : ANALYSE MODÈLE 2 (PYTORCH - STADE MULTI-CLASSE)
    # -----------------------------------------------------------------
    if is_wound:
        preprocess_py = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor_py = preprocess_py(image_raw).unsqueeze(0).to(device)

        model_pytorch_2.eval()
        with torch.no_grad():
            outputs_py = model_pytorch_2(input_tensor_py)
            probabilities_py = torch.softmax(outputs_py, dim=1)[0]
            confidence, class_idx = torch.max(probabilities_py, dim=0)
            
            m2_status = CLASS_NAMES[class_idx.item()]
            m2_cert = confidence.item() * 100

    # -----------------------------------------------------------------
    # GÉNÉRATION DU GRAPHIQUE COMPOSITE (3 COLONNES)
    # -----------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Pipeline d'inférence : {os.path.basename(image_path)}", fontsize=13, fontweight="bold", y=0.98)

    # Colonne 1 : Image brute + Vrai Label
    ax1.imshow(image_raw)
    ax1.set_title("1. Image d'entrée", fontsize=11, fontweight="bold")
    ax1.axis("off")
    ax1.text(0.5, -0.12, f"Vrai Label : {vrai_label}", color="#333333", fontsize=11, 
             fontweight="bold", ha="center", va="top", transform=ax1.transAxes,
             bbox=dict(facecolor='#F5F5F5', alpha=0.8, boxstyle='round,pad=0.5'))

    # Colonne 2 : Verdict Modèle 1
    ax2.set_title("2. Filtre Validité (Modèle 1)", fontsize=11, fontweight="bold")
    ax2.axis("off")
    bg_color_m1 = "#4CAF50" if is_wound else "#F44336"
    ax2.add_patch(plt.Rectangle((0.1, 0.2), 0.8, 0.6, color=bg_color_m1, alpha=0.9, transform=ax2.transAxes))
    ax2.text(0.5, 0.55, m1_status, color="white", fontsize=12, fontweight="bold", ha="center", va="center", transform=ax2.transAxes)
    ax2.text(0.5, 0.40, f"Certitude : {m1_cert:.2f}%", color="white", fontsize=11, ha="center", va="center", transform=ax2.transAxes)

    # Colonne 3 : Verdict Modèle 2
    ax3.set_title("3. Classification Stade (Modèle 2)", fontsize=11, fontweight="bold")
    ax3.axis("off")
    bg_color_m2 = "#2196F3" if is_wound else "#9E9E9E"
    ax3.add_patch(plt.Rectangle((0.1, 0.2), 0.8, 0.6, color=bg_color_m2, alpha=0.9, transform=ax3.transAxes))
    ax3.text(0.5, 0.55, f"Stade : {m2_status}", color="white", fontsize=12, fontweight="bold", ha="center", va="center", transform=ax3.transAxes)
    if is_wound:
        ax3.text(0.5, 0.40, f"Certitude : {m2_cert:.2f}%", color="white", fontsize=11, ha="center", va="center", transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.40, "(Modèle bypassé)", color="#E0E0E0", fontsize=10, style="italic", ha="center", va="center", transform=ax3.transAxes)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    print(f" -> Résultat visuel sauvegardé avec succès sous : '{output_filename}'")
    plt.close() # Ferme la figure en mémoire pour enchaîner proprement la suite


# =====================================================================
# 3. BLOC DE RUN PRINCIPAL (TES 5 TESTS AUTOMATIQUES)
# =====================================================================
if __name__ == "__main__":
    print("\n==============================================")
    print("      INITIALISATION DU PIPELINE HYBRIDE      ")
    print("==============================================\n")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    KERAS_MODEL_PATH = "model_detection_plaie.keras"
    PYTORCH_MODEL_PATH = "best_classifier_model.pth"

    if not os.path.exists(KERAS_MODEL_PATH) or not os.path.exists(PYTORCH_MODEL_PATH):
        print("❌ Fichiers modèles manquants (.keras ou .pth). Arrêt.")
        exit()

    print("Chargement des poids des modèles...")
    model_1_keras = tf.keras.models.load_model(KERAS_MODEL_PATH)
    
    model_2_pytorch = Classifier(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model_2_pytorch.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=DEVICE))
    print("-> Modèles chargés et prêts.")

    print("\n--------------------------------------------------")
    print("EXÉCUTION DES PRÉDICTIONS EN SÉRIE...")
    print("--------------------------------------------------")

    # Vos 5 lignes de tests prêtes à s'exécuter d'un coup
    predict_and_visualize("dataset_etape1/Invalid/Invalid_005.png", "Invalid_test.png", model_1_keras, model_2_pytorch, DEVICE)
    predict_and_visualize("dataset_etape2/SDTI/SDTI_005.png", "SDTI_test.png", model_1_keras, model_2_pytorch, DEVICE)
    predict_and_visualize("dataset_etape2/Unstageable/Unstageable_005.png", "Unstageable_test.png", model_1_keras, model_2_pytorch, DEVICE)
    predict_and_visualize("dataset_etape2/Stage_2/Stage_II_005.png", "Stage2_test.png", model_1_keras, model_2_pytorch, DEVICE)
    predict_and_visualize("dataset_etape2/Stage_4/Stage_IV_005.png", "Stage4_test.png", model_1_keras, model_2_pytorch, DEVICE)

    print("\n==============================================")
    print("✨ Opération terminée ! Les 5 fichiers PNG sont dans votre dossier.")
    print("==============================================")