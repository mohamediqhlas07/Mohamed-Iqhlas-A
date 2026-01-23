"""
Fast Optimized Visualizations
Optimized for speed - simplified waveforms, cached processing.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Dict, List
import matplotlib.patches as mpatches

def plot_fast_waveform(
    audio_path: str,
    sr: int = 22050,
    max_samples: int = 5000,  # Heavier downsample for even more speed
    figsize: tuple = (12, 3)
) -> plt.Figure:
    """
    Fast waveform plot - optimized for speed.
    BRUTAL SPEED MODE: limits to 5 seconds + aggressive downsampling.
    This ONLY affects visualization, not CNN predictions.
    """
    # Limit to first 5 seconds for visualization speed (predictions use full audio elsewhere)
    y, sr_actual = librosa.load(audio_path, sr=sr, mono=True, duration=5)
    
    # Downsample for faster plotting
    if len(y) > max_samples:
        step = len(y) // max_samples
        y = y[::step]
        time = np.linspace(0, len(y) / sr_actual * step, len(y))
    else:
        time = np.linspace(0, len(y) / sr_actual, len(y))
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Simple plot - no complex gradients
    ax.plot(time, y, color='cyan', linewidth=1.5, alpha=0.8, label='Waveform')
    ax.fill_between(time, y, alpha=0.3, color='cyan')
    
    # Quick RMS overlay
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr_actual)
    rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-10)
    
    ax2 = ax.twinx()
    ax2.plot(times_rms, rms_normalized, color='yellow', linewidth=2, alpha=0.7, label='Intensity')
    ax2.fill_between(times_rms, rms_normalized, alpha=0.2, color='yellow')
    ax2.set_ylabel('Intensity', color='yellow', fontsize=10)
    ax2.tick_params(colors='yellow')
    ax2.set_ylim(0, 1.1)
    
    ax.set_xlabel('Time (seconds)', color='white', fontsize=11)
    ax.set_ylabel('Amplitude', color='cyan', fontsize=11)
    ax.set_title('Audio Waveform & Intensity', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
             facecolor='#0E1117', edgecolor='white', labelcolor='white', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_fast_mel_spectrogram(
    audio_path: str,
    sr: int = 22050,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Fast mel-spectrogram - optimized.
    """
    y, sr_actual = librosa.load(audio_path, sr=sr, mono=True, duration=10)  # Limit for speed
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, facecolor='#0E1117', height_ratios=[3, 1])
    
    # Mel-spectrogram
    ax1.set_facecolor('#0E1117')
    mel = librosa.feature.melspectrogram(y=y, sr=sr_actual, n_mels=64, hop_length=1024)  # Reduced resolution
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    im = ax1.imshow(mel_db, origin='lower', aspect='auto', cmap='magma', interpolation='bilinear')
    ax1.set_ylabel('Mel Frequency', color='white', fontsize=12)
    ax1.set_title('Mel-Spectrogram (CNN Input Feature)', color='white', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='white')
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Magnitude (dB)', color='white', fontsize=11)
    cbar.ax.tick_params(colors='white')
    
    # Intensity
    ax2.set_facecolor('#0E1117')
    rms = librosa.feature.rms(y=y, hop_length=1024)[0]
    times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr_actual)
    rms_db = librosa.power_to_db(rms, ref=np.max)
    
    ax2.plot(times_rms, rms_db, color='yellow', linewidth=2.5, label='RMS Energy')
    ax2.fill_between(times_rms, rms_db, alpha=0.4, color='yellow')
    ax2.set_xlabel('Time (seconds)', color='white', fontsize=12)
    ax2.set_ylabel('Intensity (dB)', color='white', fontsize=12)
    ax2.set_title('Intensity Over Time', color='white', fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, color='white')
    ax2.legend(loc='upper right', facecolor='#0E1117', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    return fig

