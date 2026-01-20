import sys
import json
import os
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# =========================
# Paths & constants
# =========================
MODEL_PATH = "instrument_classifier.keras"
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = (128, 128)
TEMP_SPEC_PATH = "temp_spec.png"

# =========================
# Load class indices
# =========================
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

# index -> class name
CLASS_NAMES = {v: k for k, v in class_indices.items()}

# =========================
# Load trained model
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# Generate spectrogram IMAGE
# (MUST match training settings)
# =========================
def generate_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    y = librosa.util.normalize(y)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=sr // 2
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(2, 2))
    plt.axis("off")
    plt.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()

# =========================
# Predict TOP-1 instrument
# =========================
def predict_instrument(audio_path):
    # Step 1: Generate spectrogram image
    generate_spectrogram(audio_path, TEMP_SPEC_PATH)

    # Step 2: Load image like training
    img = load_img(TEMP_SPEC_PATH, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Step 3: Predict
    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))

    return {
        "instrument": CLASS_NAMES[idx],
        "confidence": float(preds[idx])
    }

# =========================
# Command-line execution
# =========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py <audio_path>")
        print("Example: python src/inference.py test_audio/Piano/1.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    result = predict_instrument(audio_path)

    print("\n🎵 Prediction Result")
    print(f"Instrument : {result['instrument']}")
    print(f"Confidence : {result['confidence']:.2f}")
