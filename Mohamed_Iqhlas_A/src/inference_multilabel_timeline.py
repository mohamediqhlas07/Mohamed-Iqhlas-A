"""
Multilabel Timeline Inference Script
Performs sliding-window inference on audio files to detect instruments over time.
"""

import os
import json
import librosa
import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
MODEL_PATH = "multilabel_timeline_model.keras"
CLASSES_JSON_PATH = "multilabel_classes.json"
CLASS_INDEX_PATH = "class_indices.json"

# Audio processing (MUST match training)
SR = 22050
WINDOW_SIZE = 3.0  # seconds
HOP_SIZE = 1.0     # seconds
IMG_SIZE = (128, 128)
N_MELS = 128

# Detection threshold
DEFAULT_THRESHOLD = 0.5

# ================= LOAD MODEL AND CLASSES =================
print("Loading model and class mapping...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        f"Please train the model first using train_multilabel_timeline.py"
    )

model = tf.keras.models.load_model(MODEL_PATH)
print(f"✓ Model loaded from {MODEL_PATH}")

# Load class names
with open(CLASSES_JSON_PATH, 'r') as f:
    CLASS_NAMES = json.load(f)

NUM_CLASSES = len(CLASS_NAMES)
print(f"✓ Loaded {NUM_CLASSES} instrument classes")

# ================= FEATURE EXTRACTION =================
def audio_to_mel_spectrogram(y, sr=SR):
    """
    Convert audio to mel-spectrogram image.
    MUST match training preprocessing exactly.
    """
    # Generate mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=N_MELS,
        hop_length=512,
        n_fft=2048
    )
    
    # Convert to dB
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-10)
    
    # Resize to IMG_SIZE
    mel_db = tf.image.resize(
        mel_db[..., np.newaxis], 
        IMG_SIZE,
        method='bilinear'
    )
    
    # Convert to 3-channel (repeat for RGB)
    mel_db = tf.repeat(mel_db, 3, axis=-1)
    
    return mel_db.numpy()

# ================= INFERENCE FUNCTIONS =================
def predict_single_window(audio_clip: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Predict instrument probabilities for a single 3-second audio clip.
    
    Args:
        audio_clip: Audio array (3 seconds)
        sr: Sample rate
    
    Returns:
        Array of shape (NUM_CLASSES,) with probabilities for each instrument
    """
    # Ensure correct length
    expected_length = int(WINDOW_SIZE * sr)
    if len(audio_clip) < expected_length:
        audio_clip = np.pad(audio_clip, (0, expected_length - len(audio_clip)), 'constant')
    elif len(audio_clip) > expected_length:
        audio_clip = audio_clip[:expected_length]
    
    # Extract mel-spectrogram
    mel_spec = audio_to_mel_spectrogram(audio_clip, sr)
    
    # Predict
    mel_spec_batch = np.expand_dims(mel_spec, axis=0)
    predictions = model.predict(mel_spec_batch, verbose=0)[0]
    
    return predictions

def predict_timeline(
    audio_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    return_raw: bool = False
) -> Dict:
    """
    Perform sliding-window inference on an audio file.
    
    Args:
        audio_path: Path to audio file
        threshold: Confidence threshold for detection (default: 0.5)
        return_raw: If True, return raw probabilities for all windows
    
    Returns:
        Dictionary with:
        - 'file': filename
        - 'duration': audio duration in seconds
        - 'detected_instruments': list of detected instruments (above threshold)
        - 'timeline': list of window predictions
        - 'confidence_over_time': dict mapping instrument -> list of confidences
        - 'raw_predictions': (if return_raw) all window predictions
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=SR, mono=True, duration=None)
    duration = len(y) / sr
    
    print(f"Processing {os.path.basename(audio_path)} ({duration:.2f}s)...")
    
    # Sliding window inference
    window_samples = int(WINDOW_SIZE * sr)
    hop_samples = int(HOP_SIZE * sr)
    
    timeline = []
    all_predictions = []
    
    num_windows = max(1, (len(y) - window_samples) // hop_samples + 1)
    
    for i in range(num_windows):
        start_idx = i * hop_samples
        end_idx = start_idx + window_samples
        
        if end_idx > len(y):
            clip = np.pad(y[start_idx:], (0, window_samples - (len(y) - start_idx)), 'constant')
        else:
            clip = y[start_idx:end_idx]
        
        # Predict
        predictions = predict_single_window(clip, sr)
        all_predictions.append(predictions)
        
        # Time for this window (center of window)
        window_center_time = (start_idx + window_samples / 2) / sr
        
        # Get detected instruments (above threshold)
        detected = [
            {
                'instrument': CLASS_NAMES[idx],
                'confidence': float(predictions[idx])
            }
            for idx in range(NUM_CLASSES)
            if predictions[idx] >= threshold
        ]
        
        # Sort by confidence
        detected.sort(key=lambda x: x['confidence'], reverse=True)
        
        timeline.append({
            'time': window_center_time,
            'detected_instruments': detected,
            'all_confidences': {CLASS_NAMES[idx]: float(predictions[idx]) 
                              for idx in range(NUM_CLASSES)}
        })
    
    # Aggregate results across timeline
    # Average confidence for each instrument
    avg_confidences = {}
    for inst_idx in range(NUM_CLASSES):
        inst_name = CLASS_NAMES[inst_idx]
        confidences = [pred[inst_idx] for pred in all_predictions]
        avg_confidences[inst_name] = float(np.mean(confidences))
    
    # Get final detected instruments (average confidence >= threshold)
    detected_instruments = [
        {
            'instrument': inst,
            'average_confidence': avg_confidences[inst],
            'max_confidence': float(max([pred[inst_idx] for pred in all_predictions])),
            'min_confidence': float(min([pred[inst_idx] for pred in all_predictions]))
        }
        for inst_idx, inst in enumerate(CLASS_NAMES)
        if avg_confidences[inst] >= threshold
    ]
    
    # Sort by average confidence
    detected_instruments.sort(key=lambda x: x['average_confidence'], reverse=True)
    
    # Build confidence over time dictionary
    confidence_over_time = {}
    for inst_idx, inst in enumerate(CLASS_NAMES):
        confidence_over_time[inst] = [
            float(pred[inst_idx]) for pred in all_predictions
        ]
    
    result = {
        'file': os.path.basename(audio_path),
        'file_path': audio_path,
        'duration': duration,
        'num_windows': num_windows,
        'detected_instruments': detected_instruments,
        'timeline': timeline,
        'confidence_over_time': confidence_over_time,
        'average_confidences': avg_confidences,
        'threshold': threshold
    }
    
    if return_raw:
        result['raw_predictions'] = all_predictions
    
    return result

def get_top_instruments(
    result: Dict,
    top_k: int = 10,
    use_average: bool = True
) -> List[Dict]:
    """
    Get top K instruments from prediction result.
    
    Args:
        result: Result from predict_timeline()
        top_k: Number of top instruments to return
        use_average: If True, use average confidence; else use max confidence
    
    Returns:
        List of top K instruments with confidence scores
    """
    if use_average:
        key = 'average_confidence'
    else:
        key = 'max_confidence'
    
    all_instruments = [
        {
            'instrument': inst,
            'average_confidence': result['average_confidences'][inst],
            'max_confidence': max(result['confidence_over_time'][inst]),
            'min_confidence': min(result['confidence_over_time'][inst])
        }
        for inst in CLASS_NAMES
    ]
    
    all_instruments.sort(key=lambda x: x[key], reverse=True)
    
    return all_instruments[:top_k]

# ================= MAIN (for testing) =================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference_multilabel_timeline.py <audio_file> [threshold]")
        print("\nExample:")
        print("  python inference_multilabel_timeline.py data_multilabel/audio/song1.mp3 0.5")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Predict
    result = predict_timeline(audio_file, threshold=threshold)
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"File: {result['file']}")
    print(f"Duration: {result['duration']:.2f} seconds")
    print(f"Windows: {result['num_windows']}")
    print(f"Threshold: {threshold}")
    
    print("\nDetected Instruments (average confidence >= threshold):")
    if result['detected_instruments']:
        for inst_info in result['detected_instruments']:
            print(f"  {inst_info['instrument']:20s} | "
                  f"Avg: {inst_info['average_confidence']:.3f} | "
                  f"Max: {inst_info['max_confidence']:.3f} | "
                  f"Min: {inst_info['min_confidence']:.3f}")
    else:
        print("  None (try lowering threshold)")
    
    print("\nTop 10 Instruments (by average confidence):")
    top_10 = get_top_instruments(result, top_k=10)
    for inst_info in top_10:
        print(f"  {inst_info['instrument']:20s} | "
              f"Avg: {inst_info['average_confidence']:.3f} | "
              f"Max: {inst_info['max_confidence']:.3f}")
    
    print("\n✓ Inference completed!")

