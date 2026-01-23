"""
Unique Features for InstruNet
- Real-time microphone analysis
- CNN explainability (Grad-CAM)
- Instrument similarity matrix
- Audio segmentation
"""

import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import io
import soundfile as sf
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

def generate_gradcam(
    model,
    audio_path: str,
    target_instrument_idx: int,
    layer_name: str = 'efficientnetb0',
    sr: int = 22050
) -> np.ndarray:
    """
    Generate Grad-CAM visualization showing what CNN focuses on.
    Unique feature: Model explainability.
    """
    # Load and preprocess audio
    y, sr_actual = librosa.load(audio_path, sr=sr, duration=3, mono=True)
    
    # Generate mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512, n_fft=2048)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-10)
    
    # Resize to 128x128
    mel_db_resized = tf.image.resize(
        mel_db[..., np.newaxis], 
        (128, 128),
        method='bilinear'
    )
    mel_db_resized = tf.repeat(mel_db_resized, 3, axis=-1)
    mel_db_resized = mel_db_resized.numpy()
    
    # Prepare input
    img_array = np.expand_dims(mel_db_resized, axis=0)
    
    # Get model layer
    try:
        target_layer = None
        for layer in model.layers:
            if layer_name.lower() in layer.name.lower():
                target_layer = layer
                break
        
        if target_layer is None:
            # Use last conv layer
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    target_layer = layer
                    break
        
        if target_layer is None:
            return None
        
        # Create Grad-CAM model
        grad_model = tf.keras.models.Model(
            inputs=[model.input],
            outputs=[target_layer.output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, target_instrument_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-10)
        
        # Resize to match mel-spectrogram
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis],
            mel_db.shape,
            method='bilinear'
        ).numpy()[:, :, 0]
        
        return heatmap_resized, mel_db
        
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None, None

def plot_gradcam_explanation(
    audio_path: str,
    model,
    instrument_name: str,
    instrument_idx: int,
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Plot Grad-CAM visualization showing CNN attention.
    """
    heatmap, mel_db = generate_gradcam(model, audio_path, instrument_idx)
    
    if heatmap is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        ax.text(0.5, 0.5, 'Grad-CAM visualization not available', 
               ha='center', va='center', color='white', fontsize=14)
        return fig
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, facecolor='#0E1117', height_ratios=[2, 2, 1])
    
    # Original mel-spectrogram
    ax1.set_facecolor('#0E1117')
    im1 = ax1.imshow(mel_db, origin='lower', aspect='auto', cmap='magma', interpolation='bilinear')
    ax1.set_ylabel('Mel Frequency', color='white', fontsize=11)
    ax1.set_title(f'Original Mel-Spectrogram (CNN Input)', color='white', fontsize=12, fontweight='bold')
    ax1.tick_params(colors='white')
    plt.colorbar(im1, ax=ax1).ax.tick_params(colors='white')
    
    # Grad-CAM heatmap
    ax2.set_facecolor('#0E1117')
    im2 = ax2.imshow(heatmap, origin='lower', aspect='auto', cmap='jet', interpolation='bilinear', alpha=0.8)
    ax2.set_ylabel('Mel Frequency', color='white', fontsize=11)
    ax2.set_title(f'CNN Attention Map (Grad-CAM) - What CNN Focuses on for {instrument_name}', 
                  color='white', fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white')
    plt.colorbar(im2, ax=ax2).ax.tick_params(colors='white')
    
    # Overlay
    ax3.set_facecolor('#0E1117')
    ax3.imshow(mel_db, origin='lower', aspect='auto', cmap='gray', interpolation='bilinear', alpha=0.5)
    ax3.imshow(heatmap, origin='lower', aspect='auto', cmap='jet', interpolation='bilinear', alpha=0.6)
    ax3.set_xlabel('Time (frames)', color='white', fontsize=11)
    ax3.set_ylabel('Mel Frequency', color='white', fontsize=11)
    ax3.set_title('Overlay: Spectrogram + CNN Attention', color='white', fontsize=12, fontweight='bold')
    ax3.tick_params(colors='white')
    
    plt.tight_layout()
    return fig

def compute_instrument_similarity_matrix(
    result: Dict,
    top_k: int = 15
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute similarity matrix between instruments based on CNN confidence patterns.
    Unique feature: Instrument similarity analysis.
    """
    class_names = list(result['average_confidences'].keys())
    
    # Get top K instruments
    top_instruments = sorted(
        [(inst, result['average_confidences'][inst]) for inst in class_names],
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    top_inst_names = [inst for inst, _ in top_instruments]
    
    # Build confidence vectors
    confidence_vectors = []
    for inst in top_inst_names:
        confidences = result['confidence_over_time'][inst]
        confidence_vectors.append(confidences)
    
    confidence_vectors = np.array(confidence_vectors)
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(confidence_vectors)
    
    return similarity_matrix, top_inst_names

def plot_instrument_similarity_matrix(
    result: Dict,
    top_k: int = 15,
    figsize: tuple = (12, 10)
) -> plt.Figure:
    """
    Plot instrument similarity matrix.
    Shows which instruments have similar CNN confidence patterns.
    """
    similarity_matrix, instrument_names = compute_instrument_similarity_matrix(result, top_k)
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    im = ax.imshow(similarity_matrix, cmap='RdYlBu', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(instrument_names)))
    ax.set_yticks(np.arange(len(instrument_names)))
    ax.set_xticklabels(instrument_names, rotation=45, ha='right', color='white', fontsize=9)
    ax.set_yticklabels(instrument_names, color='white', fontsize=9)
    
    # Add text annotations
    for i in range(len(instrument_names)):
        for j in range(len(instrument_names)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="white", fontsize=7)
    
    ax.set_title('Instrument Similarity Matrix (Based on CNN Confidence Patterns)', 
                color='white', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', color='white', fontsize=11)
    cbar.ax.tick_params(colors='white')
    
    plt.tight_layout()
    return fig

def extract_instrument_segments(
    audio_path: str,
    result: Dict,
    instrument_name: str,
    min_confidence: float = 0.5,
    min_duration: float = 1.0,
    sr: int = 22050
) -> List[Dict]:
    """
    Extract audio segments where specific instrument is detected.
    Unique feature: Audio segmentation based on CNN predictions.
    """
    y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
    
    if instrument_name not in result['confidence_over_time']:
        return []
    
    confidences = result['confidence_over_time'][instrument_name]
    times = np.linspace(0, result['duration'], len(confidences))
    
    segments = []
    in_segment = False
    segment_start = 0
    segment_start_idx = 0
    
    window_size = 3.0  # seconds
    hop_size = 1.0     # seconds
    
    for i, (time, conf) in enumerate(zip(times, confidences)):
        if conf >= min_confidence and not in_segment:
            # Start new segment
            in_segment = True
            segment_start = max(0, time - hop_size)  # Include some context
            segment_start_idx = int(segment_start * sr_actual)
        elif conf < min_confidence and in_segment:
            # End segment
            segment_end = time + hop_size
            segment_duration = segment_end - segment_start
            
            if segment_duration >= min_duration:
                segment_end_idx = int(segment_end * sr_actual)
                segment_audio = y[segment_start_idx:segment_end_idx]
                
                segments.append({
                    'start_time': segment_start,
                    'end_time': segment_end,
                    'duration': segment_duration,
                    'average_confidence': np.mean(confidences[
                        max(0, int(segment_start / hop_size)):
                        min(len(confidences), int(segment_end / hop_size))
                    ]),
                    'audio': segment_audio,
                    'sample_rate': sr_actual
                })
            
            in_segment = False
    
    # Handle segment that extends to end
    if in_segment:
        segment_end = result['duration']
        segment_duration = segment_end - segment_start
        if segment_duration >= min_duration:
            segment_end_idx = len(y)
            segment_audio = y[segment_start_idx:segment_end_idx]
            segments.append({
                'start_time': segment_start,
                'end_time': segment_end,
                'duration': segment_duration,
                'average_confidence': np.mean(confidences[int(segment_start / hop_size):]),
                'audio': segment_audio,
                'sample_rate': sr_actual
            })
    
    return segments

def save_segment_audio(
    segment: Dict,
    output_path: str
) -> bool:
    """
    Save extracted audio segment to file.
    """
    try:
        sf.write(output_path, segment['audio'], segment['sample_rate'])
        return True
    except Exception as e:
        print(f"Error saving segment: {e}")
        return False

def analyze_audio_quality(
    audio_path: str,
    sr: int = 22050
) -> Dict:
    """
    Analyze audio quality metrics.
    Unique feature: Audio quality assessment.
    """
    y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
    
    # Signal-to-noise ratio (approximate)
    signal_power = np.mean(y ** 2)
    noise_estimate = np.percentile(np.abs(y), 10) ** 2
    snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
    
    # Dynamic range
    dynamic_range = np.max(y) - np.min(y)
    
    # Zero crossing rate (indicates noise/quality)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
    
    # Spectral centroid (brightness)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_actual)[0])
    
    # RMS energy
    rms = np.mean(librosa.feature.rms(y=y)[0])
    
    # Quality score (0-100)
    quality_score = min(100, max(0, 
        (snr / 20) * 30 +  # SNR contribution
        (dynamic_range / 2) * 20 +  # Dynamic range
        (1 - zcr) * 30 +  # Low ZCR is better
        (rms * 20)  # Energy
    ))
    
    return {
        'snr_db': round(snr, 2),
        'dynamic_range': round(dynamic_range, 4),
        'zero_crossing_rate': round(zcr, 4),
        'spectral_centroid_hz': round(spectral_centroid, 2),
        'rms_energy': round(rms, 4),
        'quality_score': round(quality_score, 1),
        'duration': round(len(y) / sr_actual, 2),
        'sample_rate': sr_actual
    }

def plot_audio_quality_metrics(
    quality_metrics: Dict,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plot audio quality metrics visualization.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    metrics = ['SNR (dB)', 'Dynamic Range', 'ZCR', 'RMS Energy']
    values = [
        quality_metrics['snr_db'] / 20,  # Normalize
        quality_metrics['dynamic_range'] / 2,
        quality_metrics['zero_crossing_rate'],
        quality_metrics['rms_energy']
    ]
    
    colors_list = ['#00C853' if v > 0.5 else '#FFB300' if v > 0.3 else '#FF5252' for v in values]
    
    bars = ax.barh(metrics, values, color=colors_list, alpha=0.7, edgecolor='white')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.02, i, f'{values[i]:.3f}', 
               va='center', color='white', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Normalized Value', color='white', fontsize=12)
    ax.set_title(f'Audio Quality Metrics (Score: {quality_metrics["quality_score"]}/100)', 
                color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.2, color='white', axis='x')
    
    plt.tight_layout()
    return fig

