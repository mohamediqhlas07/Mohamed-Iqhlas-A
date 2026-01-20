import streamlit as st
import json
import os
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="CNN-Based Music Instrument Recognition",
    layout="centered"
)

st.title("🎵 CNN-Based Music Instrument Recognition System")
st.write(
    "Upload an audio file (WAV/MP3). "
    "The system converts it into a spectrogram and predicts the dominant musical instrument."
)

# =========================
# Paths & constants
# =========================
MODEL_PATH = "instrument_classifier.keras"
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = (128, 128)
TEMP_SPEC_PATH = "temp_spec.png"

# =========================
# Load model & classes
# =========================
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}
    return model, class_names

model, CLASS_NAMES = load_model_and_classes()

# =========================
# Generate spectrogram IMAGE
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
    # Step 1: create spectrogram image
    generate_spectrogram(audio_path, TEMP_SPEC_PATH)

    # Step 2: load image like training
    img = load_img(TEMP_SPEC_PATH, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Step 3: predict
    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))

    return CLASS_NAMES[idx], float(preds[idx])

# =========================
# Streamlit UI
# =========================
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Save uploaded file temporarily
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.read())

    if st.button("🔍 Predict Instrument"):
        with st.spinner("Analyzing audio..."):
            instrument, confidence = predict_instrument(temp_audio_path)

        st.success("Prediction Completed")

        col1, col2 = st.columns(2)
        col1.metric("Identified Instrument", instrument)
        col2.metric("Confidence", f"{confidence:.2f}")

        st.info(
            f"Based on the spectrogram features extracted from the audio, "
            f"the system predicts **{instrument}** as the dominant instrument."
        )

# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
    "Internship Project | CNN-Based Music Instrument Recognition System | "
    "Developed by Mohamed Iqhlas A"
)
