import json
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from datetime import datetime

MODEL_PATH = "instrument_classifier.keras"
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE = (128, 128)
TEMP_SPEC_PATH = "temp_spec.png"

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.7
LOW_CONFIDENCE_THRESHOLD = 0.5

# Instrument similarity groups (for post-processing suggestions)
SIMILARITY_GROUPS = {
    "Brass Family": ["Trumpet", "Trombone", "Horn"],
    "Guitar Family": ["Acoustic_Guitar", "Electro_Guitar", "Bass_Guitar", "Ukulele", "Mandolin", "Banjo", "Dobro"],
    "Keyboard Family": ["Piano", "Keyboard", "Organ"],
    "Percussion Family": ["Drum_set", "Cymbals", "Hi_Hats", "Floor_Tom", "Tambourine", "Shakers", "cowbell"],
    "Woodwind Family": ["flute", "Clarinet", "Saxophone", "Harmonica"],
    "String Family": ["Violin", "Acoustic_Guitar", "Mandolin"],
    "Reed Family": ["Clarinet", "Saxophone", "Harmonica"]
}

with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)

CLASS_NAMES = {v: k for k, v in class_indices.items()}
model = tf.keras.models.load_model(MODEL_PATH)


def generate_spectrogram(audio_path):
    """Generate mel-spectrogram from audio file."""
    y, sr = librosa.load(audio_path, sr=22050, duration=3, mono=True)
    y = librosa.util.normalize(y)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(3, 3))
    plt.axis("off")
    plt.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
    plt.tight_layout(pad=0)
    plt.savefig(TEMP_SPEC_PATH, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()

    return y, sr, mel_db


def get_top_k_predictions(predictions, k=5):
    """Get top K predictions with confidence scores."""
    top_k_indices = np.argsort(predictions)[::-1][:k]
    top_k_predictions = [
        {
            "instrument": CLASS_NAMES[idx],
            "confidence": float(predictions[idx]),
            "rank": i + 1
        }
        for i, idx in enumerate(top_k_indices)
    ]
    return top_k_predictions


def get_similarity_group(instrument_name):
    """Find which similarity group an instrument belongs to."""
    for group_name, instruments in SIMILARITY_GROUPS.items():
        if instrument_name in instruments:
            return group_name, instruments
    return None, []


def analyze_prediction_confidence(confidence):
    """Analyze prediction confidence level."""
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high", "The model is highly confident in this prediction."
    elif confidence >= LOW_CONFIDENCE_THRESHOLD:
        return "medium", "The model has moderate confidence. Consider checking alternative predictions."
    else:
        return "low", "The model has low confidence. This prediction may be uncertain."


def calculate_audio_features(y, sr):
    """Calculate audio features for analysis."""
    # RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.mean(rms))
    
    # Spectral centroid (brightness)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = float(np.mean(spectral_centroids))
    
    # Zero crossing rate (percussive vs sustained)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))
    
    # Duration
    duration = float(len(y) / sr)
    
    return {
        "energy": round(energy, 4),
        "brightness": round(brightness, 2),
        "zero_crossing_rate": round(zcr_mean, 4),
        "duration": round(duration, 2),
        "sample_rate": sr
    }


def predict(audio_path, top_k=5):
    """
    Enhanced prediction function with Top-K alternatives and analysis.
    
    Args:
        audio_path: Path to audio file
        top_k: Number of top predictions to return (default: 5)
    
    Returns:
        Dictionary with prediction results, alternatives, and analysis
    """
    y, sr, mel_db = generate_spectrogram(audio_path)

    img = load_img(TEMP_SPEC_PATH, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))
    
    # Get primary prediction
    primary_instrument = CLASS_NAMES[idx]
    primary_confidence = float(preds[idx])
    
    # Get Top-K predictions
    top_k_predictions = get_top_k_predictions(preds, k=top_k)
    
    # Analyze confidence
    confidence_level, confidence_message = analyze_prediction_confidence(primary_confidence)
    
    # Get similarity group
    similarity_group, similar_instruments = get_similarity_group(primary_instrument)
    
    # Calculate uncertainty (entropy-based)
    # Normalize predictions to probabilities
    preds_normalized = preds / np.sum(preds)
    entropy = -np.sum(preds_normalized * np.log(preds_normalized + 1e-10))
    max_entropy = np.log(len(preds))
    uncertainty_score = float(entropy / max_entropy)
    
    # Calculate audio features
    audio_features = calculate_audio_features(y, sr)
    
    # Check if alternatives are from same similarity group
    alternatives_in_group = []
    if similarity_group:
        for alt in top_k_predictions[1:]:  # Skip primary
            if alt["instrument"] in similar_instruments:
                alternatives_in_group.append(alt)
    
    return {
        # Primary prediction (backward compatible)
        "instrument": primary_instrument,
        "confidence": primary_confidence,
        
        # Enhanced predictions
        "predictions": {
            "primary": {
                "instrument": primary_instrument,
                "confidence": primary_confidence
            },
            "alternatives": top_k_predictions[1:],  # Exclude primary
            "top_k": top_k_predictions,
            "all_predictions": {
                CLASS_NAMES[i]: float(preds[i]) for i in range(len(preds))
            }
        },
        
        # Analysis
        "analysis": {
            "confidence_level": confidence_level,
            "confidence_message": confidence_message,
            "uncertainty_score": round(uncertainty_score, 4),
            "similarity_group": similarity_group,
            "similar_instruments": similar_instruments,
            "alternatives_in_group": alternatives_in_group
        },
        
        # Audio data (for visualization)
        "wave": y,
        "sr": sr,
        "mel": mel_db,
        
        # Audio features
        "audio_features": audio_features,
        
        # Metadata
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "model_path": MODEL_PATH,
            "num_classes": len(CLASS_NAMES)
        }
    }
