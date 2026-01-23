"""
IMPROVED Visualizations for CNN-Based Music Instrument Recognition
Clear per-instrument waveforms, intensity graphs, mel spectrograms.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Dict, List
import matplotlib.patches as mpatches

def plot_per_instrument_waveform(
    audio_path: str,
    result: Dict,
    detected_instruments: List[str],
    sr: int = 22050,
    figsize: tuple = (16, 10)
) -> plt.Figure:
    """
    Plot clear waveform for each detected instrument with intensity overlay.
    """
    # Load audio
    y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
    
    num_instruments = len(detected_instruments)
    if num_instruments == 0:
        num_instruments = 1
        detected_instruments = ["All Instruments"]
    
    fig, axes = plt.subplots(num_instruments, 1, figsize=figsize, facecolor='#0E1117')
    if num_instruments == 1:
        axes = [axes]
    
    time = np.linspace(0, len(y) / sr_actual, len(y))
    
    for idx, inst in enumerate(detected_instruments):
        ax = axes[idx]
        ax.set_facecolor('#0E1117')
        
        # Plot waveform with color gradient
        amplitude = np.abs(y)
        normalized_amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min() + 1e-10)
        colors = plt.cm.viridis(normalized_amplitude)
        
        # Plot waveform
        ax.plot(time, y, color='cyan', linewidth=1.5, alpha=0.8, label='Waveform')
        
        # Fill area
        ax.fill_between(time, y, alpha=0.3, color='cyan')
        
        # Add intensity overlay
        rms = librosa.feature.rms(y=y)[0]
        times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr_actual)
        rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-10)
        
        ax2 = ax.twinx()
        ax2.plot(times_rms, rms_normalized, color='yellow', linewidth=2, alpha=0.7, label='Intensity')
        ax2.fill_between(times_rms, rms_normalized, alpha=0.2, color='yellow')
        ax2.set_ylabel('Intensity', color='yellow', fontsize=10)
        ax2.tick_params(colors='yellow')
        ax2.set_ylim(0, 1.1)
        
        # Styling
        ax.set_xlabel('Time (seconds)', color='white', fontsize=11)
        ax.set_ylabel('Amplitude', color='cyan', fontsize=11)
        ax.set_title(f'{inst} - Waveform & Intensity Analysis', 
                    color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2, color='white')
        
        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                 facecolor='#0E1117', edgecolor='white', labelcolor='white', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_mel_spectrogram_clear(
    audio_path: str,
    result: Dict,
    sr: int = 22050,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot clear, high-quality mel-spectrogram with intensity overlay.
    """
    y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, facecolor='#0E1117', height_ratios=[3, 1])
    
    # Mel-spectrogram
    ax1.set_facecolor('#0E1117')
    mel = librosa.feature.melspectrogram(y=y, sr=sr_actual, n_mels=128, hop_length=512, n_fft=2048)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    im = ax1.imshow(mel_db, origin='lower', aspect='auto', cmap='magma', interpolation='bilinear')
    ax1.set_ylabel('Mel Frequency', color='white', fontsize=12)
    ax1.set_title('Mel-Spectrogram (CNN Input Feature)', 
                  color='white', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='white')
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Magnitude (dB)', color='white', fontsize=11)
    cbar.ax.tick_params(colors='white')
    
    # Intensity over time
    ax2.set_facecolor('#0E1117')
    rms = librosa.feature.rms(y=y)[0]
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

def plot_instrument_confidence_timeline_clear(
    result: Dict,
    instruments: List[str] = None,
    threshold: float = 0.5,
    figsize: tuple = (16, 8)
) -> plt.Figure:
    """
    Clear confidence timeline for each instrument.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    num_windows = result['num_windows']
    times = np.linspace(0, result['duration'], num_windows)
    
    if instruments is None:
        instruments = [inst['instrument'] for inst in result.get('detected_instruments', [])]
    
    if not instruments:
        ax.text(0.5, 0.5, 'No instruments detected', 
               ha='center', va='center', color='white', fontsize=16,
               transform=ax.transAxes)
        return fig
    
    # Use distinct colors
    colors_list = plt.cm.tab20(np.linspace(0, 1, len(instruments)))
    
    for inst, color in zip(instruments, colors_list):
        if inst in result['confidence_over_time']:
            confidences = result['confidence_over_time'][inst]
            ax.plot(times, confidences, label=inst, linewidth=3, color=color, alpha=0.9)
            ax.fill_between(times, confidences, alpha=0.2, color=color)
    
    # Threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
              alpha=0.8, label=f'Detection Threshold ({threshold})')
    
    # Styling
    ax.set_xlabel('Time (seconds)', color='white', fontsize=13, fontweight='bold')
    ax.set_ylabel('Confidence Score', color='white', fontsize=13, fontweight='bold')
    ax.set_title('CNN Model Confidence Over Time - Per Instrument', 
                color='white', fontsize=16, fontweight='bold')
    ax.tick_params(colors='white', labelsize=11)
    ax.grid(True, alpha=0.3, color='white', linestyle='--')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, result['duration'])
    
    # Legend
    ax.legend(loc='upper right', facecolor='#0E1117', edgecolor='white', 
             labelcolor='white', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    return fig

def plot_detection_heatmap_clear(
    result: Dict,
    threshold: float = 0.5,
    top_k: int = 15,
    figsize: tuple = (16, 10)
) -> plt.Figure:
    """
    Clear detection heatmap showing instrument activity.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    class_names = list(result['average_confidences'].keys())
    top_instruments = sorted(
        [(inst, result['average_confidences'][inst]) 
         for inst in class_names],
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    top_inst_names = [inst for inst, _ in top_instruments]
    
    # Build heatmap data
    num_windows = result['num_windows']
    heatmap_data = []
    
    for inst in top_inst_names:
        confidences = result['confidence_over_time'][inst]
        heatmap_data.append(confidences)
    
    heatmap_data = np.array(heatmap_data)
    
    # Create heatmap with better colormap
    times = np.linspace(0, result['duration'], num_windows)
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', 
                   interpolation='bilinear', vmin=0, vmax=1, origin='lower')
    
    # Set ticks
    ax.set_yticks(np.arange(len(top_inst_names)))
    ax.set_yticklabels(top_inst_names, color='white', fontsize=11, fontweight='bold')
    
    # Time axis
    num_time_ticks = min(15, num_windows)
    time_indices = np.linspace(0, num_windows - 1, num_time_ticks).astype(int)
    time_labels = [f'{times[i]:.1f}s' for i in time_indices]
    ax.set_xticks(time_indices)
    ax.set_xticklabels(time_labels, color='white', fontsize=10)
    
    ax.set_xlabel('Time (seconds)', color='white', fontsize=13, fontweight='bold')
    ax.set_title('CNN Detection Heatmap - Instrument Activity Over Time', 
                color='white', fontsize=16, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('CNN Confidence Score', color='white', fontsize=12)
    cbar.ax.tick_params(colors='white', labelsize=10)
    
    # Add threshold indicator
    threshold_y = threshold * len(top_inst_names)
    ax.axhline(y=threshold_y - 0.5, color='cyan', linestyle='--', 
              linewidth=2, alpha=0.6)
    
    plt.tight_layout()
    return fig

def plot_comprehensive_analysis_clear(
    audio_path: str,
    result: Dict,
    threshold: float = 0.5,
    figsize: tuple = (18, 12)
) -> plt.Figure:
    """
    Comprehensive analysis plot with all visualizations.
    """
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    detected = [inst['instrument'] for inst in result.get('detected_instruments', [])]
    
    fig = plt.figure(figsize=figsize, facecolor='#0E1117')
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # 1. Waveform with intensity
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#0E1117')
    time = np.linspace(0, len(y) / sr, len(y))
    ax1.plot(time, y, color='cyan', linewidth=1.5, alpha=0.8)
    ax1.fill_between(time, y, alpha=0.3, color='cyan')
    rms = librosa.feature.rms(y=y)[0]
    times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(times_rms, rms, color='yellow', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Time (s)', color='white', fontsize=11)
    ax1.set_ylabel('Amplitude', color='cyan', fontsize=11)
    ax1_twin.set_ylabel('Intensity', color='yellow', fontsize=11)
    ax1.set_title('Audio Waveform & Intensity', color='white', fontsize=13, fontweight='bold')
    ax1.tick_params(colors='white')
    ax1_twin.tick_params(colors='yellow')
    ax1.grid(True, alpha=0.2, color='white')
    
    # 2. Mel-spectrogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#0E1117')
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    im2 = ax2.imshow(mel_db, origin='lower', aspect='auto', cmap='magma', interpolation='bilinear')
    ax2.set_ylabel('Mel Frequency', color='white', fontsize=11)
    ax2.set_title('Mel-Spectrogram (CNN Input)', color='white', fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white')
    plt.colorbar(im2, ax=ax2).ax.tick_params(colors='white')
    
    # 3. Confidence timeline
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#0E1117')
    times_conf = np.linspace(0, result['duration'], result['num_windows'])
    colors_list = plt.cm.tab10(np.linspace(0, 1, min(10, len(detected))))
    for idx, inst in enumerate(detected[:10]):
        if inst in result['confidence_over_time']:
            ax3.plot(times_conf, result['confidence_over_time'][inst], 
                    label=inst, linewidth=2, color=colors_list[idx])
    ax3.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Time (s)', color='white', fontsize=11)
    ax3.set_ylabel('Confidence', color='white', fontsize=11)
    ax3.set_title('CNN Confidence Timeline', color='white', fontsize=12, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='white')
    ax3.legend(loc='upper right', facecolor='#0E1117', edgecolor='white', 
              labelcolor='white', fontsize=7)
    
    # 4. Detection heatmap
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor('#0E1117')
    top_10 = sorted(
        [(inst, result['average_confidences'][inst]) for inst in list(result['average_confidences'].keys())],
        key=lambda x: x[1], reverse=True
    )[:10]
    heatmap_data = np.array([result['confidence_over_time'][inst] for inst, _ in top_10])
    im4 = ax4.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='bilinear', vmin=0, vmax=1)
    ax4.set_yticks(np.arange(len(top_10)))
    ax4.set_yticklabels([inst for inst, _ in top_10], color='white', fontsize=10)
    ax4.set_xlabel('Time Windows', color='white', fontsize=11)
    ax4.set_title('CNN Detection Heatmap', color='white', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=ax4).ax.tick_params(colors='white')
    
    # 5. Bar chart
    ax5 = fig.add_subplot(gs[3, :])
    ax5.set_facecolor('#0E1117')
    top_15 = sorted(
        [(inst, result['average_confidences'][inst]) for inst in list(result['average_confidences'].keys())],
        key=lambda x: x[1], reverse=True
    )[:15]
    inst_names = [inst for inst, _ in top_15]
    confs = [conf for _, conf in top_15]
    colors_bar = ['#00C853' if c >= threshold else '#666666' for c in confs]
    y_pos = np.arange(len(inst_names))
    ax5.barh(y_pos, confs, color=colors_bar, alpha=0.7, edgecolor='white', linewidth=1.5)
    ax5.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(inst_names, color='white', fontsize=10)
    ax5.set_xlabel('Average Confidence', color='white', fontsize=11)
    ax5.set_title('CNN Model Predictions - Top 15 Instruments', 
                 color='white', fontsize=12, fontweight='bold')
    ax5.tick_params(colors='white')
    ax5.grid(True, alpha=0.2, color='white', axis='x')
    
    fig.suptitle('CNN-Based Music Instrument Recognition - Complete Analysis', 
                color='white', fontsize=18, fontweight='bold', y=0.98)
    
    return fig

