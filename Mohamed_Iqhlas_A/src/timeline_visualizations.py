"""
Timeline Visualization Functions for Multilabel Instrument Detection
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List

def plot_confidence_timeline(
    result: Dict,
    instruments: List[str] = None,
    threshold: float = 0.5,
    figsize: tuple = (14, 6)
) -> plt.Figure:
    """
    Plot confidence over time for selected instruments.
    
    Args:
        result: Result dictionary from predict_timeline()
        instruments: List of instrument names to plot (None = all detected)
        threshold: Confidence threshold line
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Get time axis
    num_windows = result['num_windows']
    times = np.linspace(0, result['duration'], num_windows)
    
    # Select instruments to plot
    if instruments is None:
        # Plot all detected instruments (above threshold)
        instruments = [inst['instrument'] for inst in result['detected_instruments']]
    
    if not instruments:
        ax.text(0.5, 0.5, 'No instruments detected', 
               ha='center', va='center', color='white', fontsize=14,
               transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    # Plot each instrument
    colors = plt.cm.Set3(np.linspace(0, 1, len(instruments)))
    
    for inst, color in zip(instruments, colors):
        if inst in result['confidence_over_time']:
            confidences = result['confidence_over_time'][inst]
            ax.plot(times, confidences, label=inst, linewidth=2, color=color, alpha=0.8)
            ax.fill_between(times, confidences, alpha=0.2, color=color)
    
    # Threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, 
              alpha=0.7, label=f'Threshold ({threshold})')
    
    # Styling
    ax.set_xlabel('Time (seconds)', color='white', fontsize=12)
    ax.set_ylabel('Confidence', color='white', fontsize=12)
    ax.set_title('Instrument Confidence Over Time', 
                color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.set_ylim(0, 1.05)
    
    # Legend
    ax.legend(loc='upper right', facecolor='#0E1117', edgecolor='white', 
             labelcolor='white', fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_detection_heatmap(
    result: Dict,
    threshold: float = 0.5,
    top_k: int = 15,
    figsize: tuple = (14, 8)
) -> plt.Figure:
    """
    Create a heatmap showing instrument detection over time.
    
    Args:
        result: Result dictionary from predict_timeline()
        threshold: Confidence threshold
        top_k: Number of top instruments to show
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Get class names from result
    class_names = list(result['average_confidences'].keys())
    
    # Get top K instruments by average confidence
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
    
    # Create heatmap
    times = np.linspace(0, result['duration'], num_windows)
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', 
                   interpolation='bilinear', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_yticks(np.arange(len(top_inst_names)))
    ax.set_yticklabels(top_inst_names, color='white', fontsize=9)
    
    # Time axis
    time_ticks = np.linspace(0, num_windows - 1, min(10, num_windows))
    time_labels = [f'{t:.1f}s' for t in np.linspace(0, result['duration'], len(time_ticks))]
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels, color='white', fontsize=9)
    
    ax.set_xlabel('Time (seconds)', color='white', fontsize=12)
    ax.set_title('Instrument Detection Heatmap (Confidence Over Time)', 
                color='white', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Confidence', color='white', fontsize=10)
    cbar.ax.tick_params(colors='white')
    
    # Add threshold indicator
    threshold_line = threshold * len(top_inst_names)
    ax.axhline(y=threshold_line - 0.5, color='cyan', linestyle='--', 
              linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_instrument_bars(
    result: Dict,
    threshold: float = 0.5,
    top_k: int = 15,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot bar chart of average confidence for detected instruments.
    
    Args:
        result: Result dictionary from predict_timeline()
        threshold: Confidence threshold
        top_k: Number of top instruments to show
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0E1117')
    ax.set_facecolor('#0E1117')
    
    # Get class names from result
    class_names = list(result['average_confidences'].keys())
    
    # Get top K instruments
    top_instruments = sorted(
        [(inst, result['average_confidences'][inst],
          max(result['confidence_over_time'][inst]),
          min(result['confidence_over_time'][inst]))
         for inst in class_names],
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    instruments = [inst for inst, _, _, _ in top_instruments]
    avg_confs = [avg for _, avg, _, _ in top_instruments]
    max_confs = [max_conf for _, _, max_conf, _ in top_instruments]
    min_confs = [min_conf for _, _, min_conf, _ in top_instruments]
    
    # Color bars based on average confidence
    colors = []
    for avg_conf in avg_confs:
        if avg_conf >= threshold:
            colors.append('#00C853')  # Green (detected)
        elif avg_conf >= threshold * 0.7:
            colors.append('#FFB300')  # Yellow (near threshold)
        else:
            colors.append('#666666')  # Gray (below threshold)
    
    # Create bars
    y_pos = np.arange(len(instruments))
    bars = ax.barh(y_pos, avg_confs, color=colors, alpha=0.7, 
                  edgecolor='white', linewidth=1)
    
    # Add error bars (min-max range)
    ax.errorbar(avg_confs, y_pos, 
               xerr=[np.array(avg_confs) - np.array(min_confs),
                     np.array(max_confs) - np.array(avg_confs)],
               fmt='none', color='white', alpha=0.5, capsize=3)
    
    # Threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', 
              linewidth=1.5, alpha=0.7, label=f'Threshold ({threshold})')
    
    # Labels
    for i, (inst, conf) in enumerate(zip(instruments, avg_confs)):
        ax.text(conf + 0.02, i, f'{conf:.3f}', 
               va='center', color='white', fontsize=9, fontweight='bold')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(instruments, color='white', fontsize=10)
    ax.set_xlabel('Average Confidence', color='white', fontsize=12)
    ax.set_title('Instrument Detection Summary', 
                color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.2, color='white', axis='x')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#00C853', label='Detected (≥threshold)'),
        mpatches.Patch(facecolor='#FFB300', label='Near threshold'),
        mpatches.Patch(facecolor='#666666', label='Below threshold')
    ]
    ax.legend(handles=legend_elements, loc='lower right', 
             facecolor='#0E1117', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    return fig

def plot_comprehensive_timeline(
    result: Dict,
    threshold: float = 0.5,
    figsize: tuple = (16, 10)
) -> plt.Figure:
    """
    Create comprehensive timeline visualization with multiple subplots.
    
    Args:
        result: Result dictionary from predict_timeline()
        threshold: Confidence threshold
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=figsize, facecolor='#0E1117')
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Get detected instruments
    detected_insts = [inst['instrument'] for inst in result['detected_instruments']]
    
    # 1. Confidence timeline (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0E1117')
    times = np.linspace(0, result['duration'], result['num_windows'])
    
    if detected_insts:
        colors = plt.cm.Set3(np.linspace(0, 1, len(detected_insts)))
        for inst, color in zip(detected_insts, colors):
            if inst in result['confidence_over_time']:
                confs = result['confidence_over_time'][inst]
                ax1.plot(times, confs, label=inst, linewidth=2, color=color)
        ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Time (s)', color='white', fontsize=10)
    ax1.set_ylabel('Confidence', color='white', fontsize=10)
    ax1.set_title('Confidence Timeline', color='white', fontsize=12, fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.2, color='white')
    ax1.set_ylim(0, 1.05)
    if detected_insts:
        ax1.legend(loc='upper right', facecolor='#0E1117', edgecolor='white', 
                  labelcolor='white', fontsize=7)
    
    # 2. Detection heatmap (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0E1117')
    
    class_names = list(result['average_confidences'].keys())
    top_10 = sorted(
        [(inst, result['average_confidences'][inst]) for inst in class_names],
        key=lambda x: x[1], reverse=True
    )[:10]
    
    heatmap_data = np.array([
        result['confidence_over_time'][inst] for inst, _ in top_10
    ])
    
    im = ax2.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', 
                   interpolation='bilinear', vmin=0, vmax=1)
    ax2.set_yticks(np.arange(len(top_10)))
    ax2.set_yticklabels([inst for inst, _ in top_10], color='white', fontsize=8)
    ax2.set_xlabel('Time (s)', color='white', fontsize=10)
    ax2.set_title('Detection Heatmap', color='white', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2).ax.tick_params(colors='white')
    
    # 3. Bar chart (bottom left, spans 2 columns)
    ax3 = fig.add_subplot(gs[1:, :])
    ax3.set_facecolor('#0E1117')
    
    class_names = list(result['average_confidences'].keys())
    top_15 = sorted(
        [(inst, result['average_confidences'][inst]) for inst in class_names],
        key=lambda x: x[1], reverse=True
    )[:15]
    
    inst_names = [inst for inst, _ in top_15]
    confs = [conf for _, conf in top_15]
    colors_bar = ['#00C853' if c >= threshold else '#666666' for c in confs]
    
    y_pos = np.arange(len(inst_names))
    ax3.barh(y_pos, confs, color=colors_bar, alpha=0.7, edgecolor='white')
    ax3.axvline(x=threshold, color='red', linestyle='--', alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(inst_names, color='white', fontsize=9)
    ax3.set_xlabel('Average Confidence', color='white', fontsize=11)
    ax3.set_title('Top 15 Instruments (Average Confidence)', 
                 color='white', fontsize=12, fontweight='bold')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, color='white', axis='x')
    
    fig.suptitle('Comprehensive Timeline Analysis', 
                color='white', fontsize=16, fontweight='bold', y=0.98)
    
    return fig

# Note: CLASS_NAMES are now extracted from result dictionary in each function
# This makes the module more flexible and doesn't require external file loading

