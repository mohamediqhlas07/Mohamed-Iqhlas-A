"""
Advanced visualization module for InstruNet.
Provides professional, presentation-ready visualizations for audio analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches


def create_enhanced_waveform(y, sr, figsize=(12, 3), max_points: int = 5000):
    """
    Create a 3D-style depth-enhanced waveform visualization.
    
    Args:
        y: Audio time series
        sr: Sample rate
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    # BRUTAL SPEED MODE:
    # This visualization is intentionally downsampled for UI/PDF speed.
    # It does NOT affect predictions (model inference happens elsewhere).
    if y is None or len(y) == 0:
        fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
        ax.set_facecolor('#0E1117')
        ax.text(0.5, 0.5, 'No audio data', ha='center', va='center', color='white', fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
        return fig

    # Downsample aggressively (plotting per-sample is extremely slow)
    if len(y) > max_points:
        step = max(1, len(y) // max_points)
        y_plot = y[::step]
        t_plot = np.linspace(0, len(y) / sr, len(y_plot))
    else:
        y_plot = y
        t_plot = np.linspace(0, len(y) / sr, len(y_plot))

    fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Fast waveform plot (single call)
    ax.plot(t_plot, y_plot, color='#00E5FF', linewidth=1.0, alpha=0.9, label='Waveform')
    ax.fill_between(t_plot, y_plot, alpha=0.18, color='#00E5FF')
    
    # Add zero line
    ax.axhline(y=0, color='white', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Styling
    ax.set_xlabel('Time (seconds)', color='white', fontsize=11)
    ax.set_ylabel('Amplitude', color='white', fontsize=11)
    ax.set_title('Enhanced Waveform Visualization', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    
    # Add peak markers (use scipy instead of librosa)
    from scipy.signal import find_peaks
    amplitude = np.abs(y_plot)
    # distance in samples on the downsampled signal (avoid too many peaks)
    peaks, _ = find_peaks(amplitude, distance=max(1, len(y_plot) // 80))
    if len(peaks) > 0:
        peak_times = t_plot[peaks]
        peak_values = y_plot[peaks]
        ax.scatter(peak_times, peak_values, c='yellow', s=18, alpha=0.65,
                  zorder=5, label='Peaks')
        ax.legend(loc='upper right', facecolor='#0E1117', edgecolor='white', 
                 labelcolor='white')
    
    plt.tight_layout()
    return fig


def create_time_frequency_plot(mel_db, sr, figsize=(12, 6)):
    """
    Create an enhanced time-frequency representation with intensity overlay.
    
    Args:
        mel_db: Mel-spectrogram in dB
        sr: Sample rate
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   facecolor='#0E1117', height_ratios=[3, 1])
    
    # Mel-spectrogram
    ax1.set_facecolor('#0E1117')
    im = ax1.imshow(mel_db, origin='lower', aspect='auto', 
                    cmap='magma', interpolation='bilinear')
    ax1.set_ylabel('Mel Frequency', color='white', fontsize=11)
    ax1.set_title('Mel-Spectrogram with Intensity Overlay', 
                 color='white', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='white')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Magnitude (dB)', color='white', fontsize=10)
    cbar.ax.tick_params(colors='white')
    
    # Intensity over time (RMS energy)
    times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sr)
    # Calculate energy from mel-spectrogram
    energy = np.mean(10**(mel_db/10), axis=0)  # Convert back to linear scale
    energy_db = librosa.power_to_db(energy, ref=np.max)
    
    ax2.set_facecolor('#0E1117')
    ax2.fill_between(times, energy_db, alpha=0.6, color='cyan', label='Energy')
    ax2.plot(times, energy_db, color='white', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Time (seconds)', color='white', fontsize=11)
    ax2.set_ylabel('Energy (dB)', color='white', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='white')
    ax2.legend(loc='upper right', facecolor='#0E1117', edgecolor='white', 
              labelcolor='white')
    
    plt.tight_layout()
    return fig


def create_harmonic_analysis(y, sr, figsize=(12, 6)):
    """
    Create harmonic structure visualization showing fundamental frequencies and harmonics.
    
    Args:
        y: Audio time series
        sr: Sample rate
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   facecolor='#0E1117', height_ratios=[2, 1])
    
    # Harmonic-percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Harmonic spectrogram
    S_harmonic = librosa.stft(y_harmonic)
    S_harmonic_db = librosa.amplitude_to_db(np.abs(S_harmonic), ref=np.max)
    
    ax1.set_facecolor('#0E1117')
    im1 = ax1.imshow(S_harmonic_db, origin='lower', aspect='auto', 
                     cmap='plasma', interpolation='bilinear')
    ax1.set_ylabel('Frequency (Hz)', color='white', fontsize=11)
    ax1.set_title('Harmonic Structure Analysis', 
                 color='white', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='white')
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Magnitude (dB)', color='white', fontsize=10)
    cbar1.ax.tick_params(colors='white')
    
    # Spectral centroid over time (brightness)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=sr)
    
    ax2.set_facecolor('#0E1117')
    ax2.plot(times, spectral_centroids, color='yellow', linewidth=2, 
            label='Spectral Centroid (Brightness)')
    ax2.fill_between(times, spectral_centroids, alpha=0.3, color='yellow')
    ax2.set_xlabel('Time (seconds)', color='white', fontsize=11)
    ax2.set_ylabel('Frequency (Hz)', color='white', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='white')
    ax2.legend(loc='upper right', facecolor='#0E1117', edgecolor='white', 
              labelcolor='white')
    
    plt.tight_layout()
    return fig


def create_instrument_timeline(predictions_data, figsize=(12, 6)):
    """
    Create an instrument activity timeline visualization.
    Shows confidence over time for multiple predictions.
    
    Args:
        predictions_data: List of prediction dictionaries with timestamps
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Extract top predictions
    if isinstance(predictions_data, dict):
        # Single prediction
        top_k = predictions_data.get('predictions', {}).get('top_k', [])
        if not top_k:
            # Fallback to primary prediction
            top_k = [{
                'instrument': predictions_data.get('instrument', 'Unknown'),
                'confidence': predictions_data.get('confidence', 0.0)
            }]
    else:
        # Multiple predictions (for batch)
        top_k = []
        for pred in predictions_data[:3]:  # Top 3 instruments
            if isinstance(pred, dict):
                top_k.append({
                    'instrument': pred.get('instrument', 'Unknown'),
                    'confidence': pred.get('confidence', 0.0)
                })
    
    if not top_k:
        ax.text(0.5, 0.5, 'No prediction data available', 
               ha='center', va='center', color='white', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    # Create timeline bars
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_k)))
    y_positions = np.arange(len(top_k))
    
    for i, pred in enumerate(top_k):
        instrument = pred.get('instrument', 'Unknown')
        confidence = pred.get('confidence', 0.0)
        
        # Create horizontal bar
        ax.barh(i, confidence, color=colors[i], alpha=0.7, 
               edgecolor='white', linewidth=1.5)
        
        # Add confidence label
        ax.text(confidence + 0.02, i, f'{confidence:.2f}', 
               va='center', color='white', fontsize=10, fontweight='bold')
        
        # Add instrument name
        ax.text(-0.05, i, instrument, ha='right', va='center', 
               color='white', fontsize=11, fontweight='bold')
    
    # Styling
    ax.set_xlim(-0.5, 1.1)
    ax.set_ylim(-0.5, len(top_k) - 0.5)
    ax.set_xlabel('Confidence Score', color='white', fontsize=11)
    ax.set_title('Instrument Confidence Timeline', 
                color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.set_yticks([])
    ax.grid(True, alpha=0.2, color='white', axis='x')
    
    # Add confidence threshold lines
    ax.axvline(x=0.7, color='green', linestyle='--', linewidth=1, 
              alpha=0.5, label='High Confidence')
    ax.axvline(x=0.5, color='yellow', linestyle='--', linewidth=1, 
              alpha=0.5, label='Medium Confidence')
    ax.legend(loc='upper right', facecolor='#0E1117', edgecolor='white', 
             labelcolor='white')
    
    plt.tight_layout()
    return fig


def create_confidence_distribution(all_predictions, figsize=(10, 6)):
    """
    Create a confidence distribution visualization for all classes.
    
    Args:
        all_predictions: Dictionary mapping instrument names to confidence scores
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Sort by confidence
    sorted_preds = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
    instruments = [item[0] for item in sorted_preds]
    confidences = [item[1] for item in sorted_preds]
    
    # Color bars based on confidence level
    colors = []
    for conf in confidences:
        if conf >= 0.7:
            colors.append('#00ff00')  # Green (high)
        elif conf >= 0.5:
            colors.append('#ffff00')  # Yellow (medium)
        else:
            colors.append('#ff6666')  # Red (low)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(instruments))
    bars = ax.barh(y_pos, confidences, color=colors, alpha=0.7, 
                  edgecolor='white', linewidth=1)
    
    # Add confidence labels
    for i, (inst, conf) in enumerate(zip(instruments, confidences)):
        ax.text(conf + 0.01, i, f'{conf:.3f}', 
               va='center', color='white', fontsize=9)
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(instruments, color='white', fontsize=9)
    ax.set_xlabel('Confidence Score', color='white', fontsize=11)
    ax.set_title('Confidence Distribution Across All Instruments', 
                color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.2, color='white', axis='x')
    
    # Add threshold lines
    ax.axvline(x=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0.5, color='yellow', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    return fig


def create_comprehensive_analysis_plot(prediction_result, figsize=(16, 10)):
    """
    Create a comprehensive analysis plot combining multiple visualizations.
    
    Args:
        prediction_result: Dictionary from predict() function
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=figsize, facecolor='#0E1117')
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Extract data
    y = prediction_result.get('wave')
    sr = prediction_result.get('sr')
    mel_db = prediction_result.get('mel')
    all_predictions = prediction_result.get('predictions', {}).get('all_predictions', {})
    top_k = prediction_result.get('predictions', {}).get('top_k', [])
    
    # 1. Enhanced Waveform (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0E1117')
    time = np.linspace(0, len(y) / sr, len(y))
    amplitude = np.abs(y)
    normalized_amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-10)
    colors = plt.cm.viridis(normalized_amplitude)
    for i in range(0, len(y) - 1, max(1, len(y)//1000)):  # Sample for performance
        ax1.plot(time[i:i+2], y[i:i+2], color=colors[i], 
                linewidth=0.5 + normalized_amplitude[i] * 2, alpha=0.8)
    ax1.set_xlabel('Time (s)', color='white', fontsize=10)
    ax1.set_ylabel('Amplitude', color='white', fontsize=10)
    ax1.set_title('Enhanced Waveform', color='white', fontsize=12, fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='white')
    
    # 2. Mel-Spectrogram (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0E1117')
    im = ax2.imshow(mel_db, origin='lower', aspect='auto', cmap='magma', interpolation='bilinear')
    ax2.set_ylabel('Mel Frequency', color='white', fontsize=10)
    ax2.set_title('Mel-Spectrogram', color='white', fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white')
    plt.colorbar(im, ax=ax2).ax.tick_params(colors='white')
    
    # 3. Confidence Timeline (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#0E1117')
    if top_k:
        instruments = [p['instrument'] for p in top_k[:5]]
        confidences = [p['confidence'] for p in top_k[:5]]
        colors_bar = plt.cm.Set3(np.linspace(0, 1, len(instruments)))
        y_pos = np.arange(len(instruments))
        ax3.barh(y_pos, confidences, color=colors_bar, alpha=0.7, edgecolor='white')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(instruments, color='white', fontsize=9)
        ax3.set_xlabel('Confidence', color='white', fontsize=10)
        ax3.set_title('Top-K Predictions', color='white', fontsize=12, fontweight='bold')
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.2, color='white', axis='x')
    
    # 4. Confidence Distribution (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#0E1117')
    if all_predictions:
        sorted_preds = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[:10]
        inst_names = [item[0] for item in sorted_preds]
        confs = [item[1] for item in sorted_preds]
        colors_dist = ['#00ff00' if c >= 0.7 else '#ffff00' if c >= 0.5 else '#ff6666' 
                      for c in confs]
        y_pos = np.arange(len(inst_names))
        ax4.barh(y_pos, confs, color=colors_dist, alpha=0.7, edgecolor='white')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(inst_names, color='white', fontsize=8)
        ax4.set_xlabel('Confidence', color='white', fontsize=10)
        ax4.set_title('Top 10 Confidence Scores', color='white', fontsize=12, fontweight='bold')
        ax4.tick_params(colors='white')
        ax4.grid(True, alpha=0.2, color='white', axis='x')
    
    # 5. Energy Timeline (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor('#0E1117')
    rms = librosa.feature.rms(y=y)[0]
    times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    ax5.plot(times_rms, rms, color='cyan', linewidth=2)
    ax5.fill_between(times_rms, rms, alpha=0.3, color='cyan')
    ax5.set_xlabel('Time (s)', color='white', fontsize=10)
    ax5.set_ylabel('RMS Energy', color='white', fontsize=10)
    ax5.set_title('Energy Over Time', color='white', fontsize=12, fontweight='bold')
    ax5.tick_params(colors='white')
    ax5.grid(True, alpha=0.2, color='white')
    
    # 6. Spectral Centroid (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor('#0E1117')
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times_cent = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=sr)
    ax6.plot(times_cent, spectral_centroids, color='yellow', linewidth=2)
    ax6.fill_between(times_cent, spectral_centroids, alpha=0.3, color='yellow')
    ax6.set_xlabel('Time (s)', color='white', fontsize=10)
    ax6.set_ylabel('Frequency (Hz)', color='white', fontsize=10)
    ax6.set_title('Spectral Centroid (Brightness)', color='white', fontsize=12, fontweight='bold')
    ax6.tick_params(colors='white')
    ax6.grid(True, alpha=0.2, color='white')
    
    # Overall title
    fig.suptitle('Comprehensive Audio Analysis Dashboard', 
                color='white', fontsize=16, fontweight='bold', y=0.98)
    
    return fig

