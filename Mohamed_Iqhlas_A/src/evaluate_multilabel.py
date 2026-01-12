import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score, hamming_loss
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from multilabel_utils import single_to_multilabel

# ---------------- CONFIG ----------------
MODEL_PATH = "instrument_classifier_multilabel.keras"
VAL_DIR = "spectrogram_dataset/val"

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
THRESHOLD = 0.5

CLASS_NAMES = sorted(os.listdir(VAL_DIR))
NUM_CLASSES = len(CLASS_NAMES)

print("Classes:", CLASS_NAMES)
print("Number of classes:", NUM_CLASSES)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- DATA LOADER ----------------
datagen = ImageDataGenerator(rescale=1./255)

val_gen = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)

# ---------------- EVALUATION ----------------
y_true = []
y_pred = []

for i in range(len(val_gen)):
    images, labels = next(val_gen)
    preds = model.predict(images)

    preds_bin = (preds >= THRESHOLD).astype(int)

    for j in range(len(labels)):
        y_true.append(
            single_to_multilabel(int(labels[j]), NUM_CLASSES)
        )
        y_pred.append(preds_bin[j])

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ---------------- METRICS ----------------
print("\n📊 Classification Report (Multi-Label):\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
))

macro_f1 = f1_score(y_true, y_pred, average="macro") 
weighted_f1 = f1_score(y_true, y_pred, average="weighted")
hamming = hamming_loss(y_true, y_pred)

print("📌 Macro F1-score:", round(macro_f1, 4))
print("📌 Weighted F1-score:", round(weighted_f1, 4))
print("📌 Hamming Loss:", round(hamming, 4))
