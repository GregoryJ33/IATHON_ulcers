import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
PATH_E1 = "dataset_etape1/"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# --- CHARGEMENT DU DATASET ---
print("\n[INFO] Chargement du Dataset Étape 1...")
train_ds_e1 = tf.keras.utils.image_dataset_from_directory(
    PATH_E1, validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)

val_ds_e1 = tf.keras.utils.image_dataset_from_directory(
    PATH_E1, validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary'
)

# --- CORRECTION BRUTALE DU DÉSÉQUILIBRE ---
y_train = np.concatenate([y for x, y in train_ds_e1], axis=0).flatten()
nb_invalides = np.sum(y_train == 0)
nb_plaies = np.sum(y_train == 1)
total = len(y_train)

# On équilibre de manière stricte
poids_classes_e1 = {
    0: total / (2.0 * nb_invalides),
    1: total / (2.0 * nb_plaies)
}
print(f"-> Poids configurés : Invalid={poids_classes_e1[0]:.2f}, Wound={poids_classes_e1[1]:.2f}")

train_ds_e1 = train_ds_e1.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds_e1 = val_ds_e1.prefetch(buffer_size=tf.data.AUTOTUNE)

# =====================================================================
# CHANGEMENT D'ARCHITECTURE : PASSAGE SUR RESNET50V2 (Ultra Stable)
# =====================================================================
print("\n[INFO] Initialisation de ResNet50V2 pour casser le blocage...")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
])

# ResNet50V2 gère parfaitement les inputs entre 0 et 1 après rescale
base_model = tf.keras.applications.ResNet50V2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False  # On fige le gros bloc pour commencer propre

inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)  # Normalisation stricte 0-1
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)  # Couche intermédiaire pour donner du relief au modèle
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model_clean = models.Model(inputs, outputs)

# Compilation avec un optimizer mis à jour
model_clean.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =====================================================================
# ENTRAÎNEMENT DE SECOUSSE
# =====================================================================
print("\nEntraînement en cours (Regarde bien si la perte descend)...")
history = model_clean.fit(
    train_ds_e1,
    validation_data=val_ds_e1,
    epochs=20,
    verbose=1,
    class_weight=poids_classes_e1
)

model_clean.save("model_detection_plaie.keras")
print("\n[OK] Nouveau modèle ResNet50V2 enregistré !")