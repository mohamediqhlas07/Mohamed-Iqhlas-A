"""
InstruNet - Final Production UI (Optimized)
Fast waveform generation, training accuracy display, and audio playback.
"""

import streamlit as st
import tempfile
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
from datetime import datetime
import os
import base64
from scipy.signal import find_peaks
import pickle

# ================= LAZY LOADING FUNCTIONS =================
# Don't load heavy modules until needed - this speeds up app startup
@st.cache_resource
def load_model_and_classes():
    """Lazy load model only when needed (for Grad-CAM)."""
    try:
        import tensorflow as tf
        MODEL_PATH = "multilabel_timeline_model.keras"
        CLASSES_JSON_PATH = "multilabel_classes.json"
        
        if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_JSON_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASSES_JSON_PATH, 'r') as f:
                CLASS_NAMES = json.load(f)
            return model, CLASS_NAMES
    except Exception as e:
        pass
    return None, []

@st.cache_resource
def load_inference_modules():
    """Lazy load inference modules."""
    try:
        from src.inference_multilabel_timeline import predict_timeline, get_top_instruments
        return predict_timeline, get_top_instruments
    except ImportError:
        return None, None

@st.cache_resource
def load_visualization_modules():
    """Lazy load visualization modules."""
    try:
        from src.fast_visualizations import (
            plot_fast_waveform,
            plot_fast_mel_spectrogram
        )
        from src.improved_visualizations import (
            plot_instrument_confidence_timeline_clear,
            plot_detection_heatmap_clear,
            plot_comprehensive_analysis_clear
        )
        return {
            'fast_waveform': plot_fast_waveform,
            'fast_mel': plot_fast_mel_spectrogram,
            'timeline': plot_instrument_confidence_timeline_clear,
            'heatmap': plot_detection_heatmap_clear,
            'comprehensive': plot_comprehensive_analysis_clear
        }
    except ImportError:
        return None

@st.cache_resource
def load_unique_features():
    """Lazy load unique features modules."""
    try:
        from src.unique_features import (
            plot_gradcam_explanation,
            plot_instrument_similarity_matrix,
            extract_instrument_segments,
            analyze_audio_quality,
            plot_audio_quality_metrics
        )
        return {
            'gradcam': plot_gradcam_explanation,
            'similarity': plot_instrument_similarity_matrix,
            'segments': extract_instrument_segments,
            'quality': analyze_audio_quality,
            'quality_plot': plot_audio_quality_metrics
        }
    except ImportError:
        return None

# Initialize as None - will be loaded when needed
model = None
CLASS_NAMES = []
predict_timeline = None
get_top_instruments = None
viz_modules = None
unique_features = None

# Load training history for accuracy display
def load_training_accuracy():
    """Load training and validation accuracy from history."""
    history_files = [
        "training_history.pkl",
        "training_history.npy"
    ]
    
    for hist_file in history_files:
        if os.path.exists(hist_file):
            try:
                if hist_file.endswith('.pkl'):
                    with open(hist_file, 'rb') as f:
                        history = pickle.load(f)
                else:
                    history = np.load(hist_file, allow_pickle=True).item()
                
                if isinstance(history, dict):
                    # Try Hamming accuracy first (more meaningful for multilabel), fallback to regular accuracy
                    train_acc = history.get('hamming_accuracy', history.get('accuracy', []))
                    val_acc = history.get('val_hamming_accuracy', history.get('val_accuracy', []))
                    
                    # Also get subset accuracy if available
                    train_subset = history.get('subset_accuracy', [])
                    val_subset = history.get('val_subset_accuracy', [])
                    
                    if train_acc and val_acc:
                        return {
                            'train_accuracy': float(train_acc[-1]) if isinstance(train_acc, list) else float(train_acc),
                            'val_accuracy': float(val_acc[-1]) if isinstance(val_acc, list) else float(val_acc),
                            'train_subset_accuracy': float(train_subset[-1]) if train_subset and isinstance(train_subset, list) else None,
                            'val_subset_accuracy': float(val_subset[-1]) if val_subset and isinstance(val_subset, list) else None,
                            'train_loss': float(history.get('loss', [0])[-1]) if isinstance(history.get('loss', []), list) else float(history.get('loss', 0)),
                            'val_loss': float(history.get('val_loss', [0])[-1]) if isinstance(history.get('val_loss', []), list) else float(history.get('val_loss', 0)),
                            'epochs': len(train_acc) if isinstance(train_acc, list) else 1,
                            'is_hamming': 'hamming_accuracy' in history  # Flag to show which metric
                        }
            except Exception as e:
                continue
    
    return None

# Page configuration
st.set_page_config(
    page_title="InstruNet - CNN Music Recognition",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= ENHANCED STYLING =================
st.markdown("""
<style>
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* CNN Badge */
    .cnn-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Prediction Cards */
    .prediction-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        color: white;
        border: 2px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Instrument Badges */
    .instrument-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.25) 0%, rgba(255, 255, 255, 0.15) 100%);
        padding: 0.7rem 1.5rem;
        border-radius: 25px;
        margin: 0.5rem;
        font-weight: bold;
        font-size: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.3);
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .instrument-badge:hover {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.4) 0%, rgba(255, 255, 255, 0.25) 100%);
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    /* Accuracy Card */
    .accuracy-card {
        background: linear-gradient(135deg, #00C853 0%, #00E676 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 200, 83, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    /* Audio Player */
    .audio-player-container {
        background: rgba(30, 60, 114, 0.8);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid rgba(102, 126, 234, 0.5);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Multilabel Indicator */
    .multilabel-indicator {
        background: linear-gradient(135deg, #00C853 0%, #00E676 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "results" not in st.session_state:
    st.session_state.results = []
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}
if "uploaded_files_dir" not in st.session_state:
    # Create persistent directory for uploaded files
    uploaded_dir = os.path.join(os.getcwd(), "temp_uploads")
    os.makedirs(uploaded_dir, exist_ok=True)
    st.session_state.uploaded_files_dir = uploaded_dir

# Load training accuracy
training_metrics = load_training_accuracy()

# ================= AUDIO PLAYBACK FUNCTIONS =================
def get_audio_bytes(audio_path: str, start_time: float = None, end_time: float = None) -> bytes:
    """Load audio and return as bytes for playback."""
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    cache_key = f"{audio_path}_{start_time}_{end_time}"
    
    if cache_key in st.session_state.audio_cache:
        return st.session_state.audio_cache[cache_key]
    
    try:
        # Load audio with librosa (handles multiple formats)
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=None)
        
        if start_time is not None and end_time is not None:
            start_idx = int(start_time * sr)
            end_idx = int(end_time * sr)
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, len(y)))
            end_idx = max(start_idx, min(end_idx, len(y)))
            y = y[start_idx:end_idx]
        
        # Convert to WAV bytes
        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, y, sr, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        # Cache it
        st.session_state.audio_cache[cache_key] = audio_bytes
        return audio_bytes
    except Exception as e:
        raise Exception(f"Error loading audio file {audio_path}: {str(e)}")

def create_audio_player(audio_bytes: bytes, key: str, autoplay: bool = False):
    """Create HTML5 audio player."""
    audio_b64 = base64.b64encode(audio_bytes).decode()
    autoplay_str = "autoplay" if autoplay else ""
    audio_html = f"""
    <div class="audio-player-container">
        <audio controls {autoplay_str} style="width: 100%;">
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
    </div>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎵 InstruNet")
    st.markdown('<div class="cnn-badge">CNN-Based System</div>', unsafe_allow_html=True)
    st.markdown("**Multilabel Instrument Recognition**")
    st.markdown("---")
    
    # Training Accuracy Display - PROMINENT
    if training_metrics:
        st.markdown("### 📊 Current Model Accuracy")
        is_hamming = training_metrics.get('is_hamming', False)
        metric_name = "Hamming" if is_hamming else ""
        st.markdown(f"""
        <div class="accuracy-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h4>📈 Training {metric_name} Accuracy</h4>
            <h2 style="font-size: 2.5rem; margin: 0.5rem 0;">{training_metrics['train_accuracy']:.1%}</h2>
            <p style="margin-top: 0.5rem;">✅ Validation {metric_name} Accuracy</p>
            <h2 style="font-size: 2.5rem; margin: 0.5rem 0;">{training_metrics['val_accuracy']:.1%}</h2>
            <p style="font-size: 0.8rem; margin-top: 0.5rem;">Epochs: {training_metrics['epochs']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Home", "Dashboard", "Analytics", "Unique Features", "History", "Export", "About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ CNN Settings")
    detection_threshold = st.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="CNN confidence threshold for multilabel detection"
    )
    
    st.markdown("---")
    if st.session_state.results:
        st.metric("Total Files", len(st.session_state.results))
        total_detected = sum(len(r.get('detected_instruments', [])) for r in st.session_state.results)
        st.metric("Detections", total_detected)

# ================= HOME PAGE =================
if page == "Home":
    st.markdown('<div class="main-header">🎵 InstruNet</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.8rem; color: #667eea; font-weight: bold;">CNN-Based Music Instrument Recognition System</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888; font-size: 1.1rem;">Deep Learning • Multilabel Detection • Timeline Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown('<div style="text-align: center;"><div class="cnn-badge">Powered by Convolutional Neural Networks (CNN)</div></div>', unsafe_allow_html=True)
    st.markdown("")
    
    # Training Accuracy Display on Home
    if training_metrics:
        col_acc1, col_acc2, col_acc3 = st.columns(3)
        with col_acc1:
            st.markdown(f"""
            <div class="accuracy-card">
                <h3>📈 Training Accuracy</h3>
                <h1>{training_metrics['train_accuracy']:.2%}</h1>
                <p>Final Training Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        with col_acc2:
            st.markdown(f"""
            <div class="accuracy-card">
                <h3>✅ Validation Accuracy</h3>
                <h1>{training_metrics['val_accuracy']:.2%}</h1>
                <p>Final Validation Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        with col_acc3:
            gap = training_metrics['train_accuracy'] - training_metrics['val_accuracy']
            gap_color = "#00C853" if gap < 0.05 else "#FFB300"
            st.markdown(f"""
            <div class="accuracy-card" style="background: linear-gradient(135deg, {gap_color} 0%, #00E676 100%);">
                <h3>📊 Accuracy Gap</h3>
                <h1>{gap:.2%}</h1>
                <p>Generalization Quality</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 CNN Architecture</h3>
            <p><b>EfficientNetB0</b></p>
            <p>Transfer learning with fine-tuning</p>
            <p>Sigmoid output for multilabel</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Multilabel Detection</h3>
            <p><b>Multiple Instruments</b></p>
            <p>Simultaneous detection</p>
            <p>Independent predictions</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⏱️ Timeline Analysis</h3>
            <p><b>Sliding Windows</b></p>
            <p>3-second windows</p>
            <p>Confidence over time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔍 What is InstruNet?
    
    InstruNet is a **production-grade CNN-based system** that uses **Convolutional Neural Networks** 
    to detect multiple musical instruments simultaneously from audio files. The system employs 
    **deep learning** techniques with **transfer learning** for accurate multilabel classification.
    
    ### 🧠 How CNN Works
    
    1. **Audio → Mel-Spectrogram**: Convert audio to 2D image representation
    2. **CNN Feature Extraction**: EfficientNetB0 extracts visual features from spectrogram
    3. **Multilabel Classification**: Sigmoid activation outputs independent probabilities
    4. **Timeline Aggregation**: Sliding windows provide temporal analysis
    
    ### ✨ Key Features
    
    - ✅ **CNN-Based**: Deep learning with EfficientNetB0 architecture
    - ✅ **Multilabel**: Detects multiple instruments simultaneously
    - ✅ **Timeline Support**: See instrument activity over time
    - ✅ **Fast Visualizations**: Optimized waveform and spectrogram generation
    - ✅ **Audio Playback**: Play uploaded audio directly in UI
    - ✅ **Training Metrics**: View training and validation accuracy
    - ✅ **Professional Export**: PDF with all visualizations included
    - 🌟 **Unique Features**: Grad-CAM, Similarity Matrix, Segmentation, Quality Analysis
    
    👉 **Navigate to Dashboard** to start detecting instruments!
    """)

# ================= DASHBOARD PAGE =================
elif page == "Dashboard":
    st.title("🎯 CNN Instrument Detection Dashboard")
    st.markdown("### Upload audio files for CNN-based multilabel instrument recognition")
    
    # Add option to browse dataset files
    st.markdown("#### 📂 Select Audio Source")
    audio_source = st.radio(
        "Choose audio source:",
        ["Upload Files", "Browse Dataset"],
        horizontal=True
    )
    
    selected_files = []
    uploaded_files = None
    
    if audio_source == "Upload Files":
        uploaded_files = st.file_uploader(
            "📁 Upload Audio Files (WAV / MP3)",
            type=["wav", "mp3"],
            accept_multiple_files=True,
            help="CNN will analyze each file using sliding-window inference"
        )
        if uploaded_files:
            selected_files = uploaded_files
            
            # IMMEDIATE AUDIO PLAYBACK - Show audio player right after upload
            st.markdown("---")
            st.markdown("#### 🎵 Preview Audio (Play Before Analysis)")
            for idx, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"**{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
                
                # Save temporarily for playback
                temp_path = os.path.join(st.session_state.uploaded_files_dir, f"preview_{idx}_{uploaded_file.name}")
                uploaded_file.seek(0)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # Show audio player using Streamlit's native audio widget - FIXED
                try:
                    uploaded_file.seek(0)
                    audio_bytes = uploaded_file.read()
                    # Determine format
                    file_ext = uploaded_file.name.lower()
                    if file_ext.endswith('.wav'):
                        audio_format = 'audio/wav'
                    elif file_ext.endswith('.mp3'):
                        audio_format = 'audio/mpeg'
                    else:
                        audio_format = 'audio/wav'  # Default
                    
                    # Display audio player with proper format
                    st.audio(audio_bytes, format=audio_format, start_time=0)
                    st.caption(f"🎵 **{uploaded_file.name}** - Click ▶️ play button above to hear the audio")
                except Exception as e:
                    st.warning(f"Could not preview audio: {str(e)}")
                    # Fallback: try direct file path
                    try:
                        if os.path.exists(temp_path):
                            st.audio(temp_path, format='audio/wav', start_time=0)
                    except:
                        pass
    else:
        # Browse dataset files
        dataset_dir = "music_dataset"
        if os.path.exists(dataset_dir):
            # Get all audio files from dataset
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
            dataset_files = []
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, dataset_dir)
                        dataset_files.append((rel_path, full_path))
            
            if dataset_files:
                # Show file selector
                file_options = [f[0] for f in dataset_files]
                selected_file = st.selectbox(
                    "🎵 Select audio file from dataset:",
                    file_options,
                    help="Choose an audio file from the dataset to analyze"
                )
                
                if selected_file:
                    selected_path = next(f[1] for f in dataset_files if f[0] == selected_file)
                    selected_files = [selected_path]
                    st.info(f"Selected: {selected_file}")
                    
                    # IMMEDIATE AUDIO PLAYBACK - Show audio player right after selection - FIXED
                    st.markdown("---")
                    st.markdown("#### 🎵 Preview Audio (Play Before Analysis)")
                    try:
                        if os.path.exists(selected_path):
                            # Use Streamlit's native audio widget - direct file path works better
                            file_ext = os.path.splitext(selected_path)[1].lower()
                            if file_ext == '.wav':
                                audio_format = 'audio/wav'
                            elif file_ext == '.mp3':
                                audio_format = 'audio/mpeg'
                            else:
                                audio_format = 'audio/wav'  # Default
                            
                            # Try direct file path first (faster)
                            st.audio(selected_path, format=audio_format, start_time=0)
                            st.caption(f"🎵 **{selected_file}** - Click ▶️ play button above to hear the audio")
                        else:
                            st.warning("Audio file not found.")
                    except Exception as e:
                        st.warning(f"Could not preview audio: {str(e)}")
                        # Fallback: try reading bytes
                        try:
                            with open(selected_path, "rb") as f:
                                audio_bytes = f.read()
                            st.audio(audio_bytes, format=audio_format, start_time=0)
                        except:
                            pass
            else:
                st.warning("No audio files found in the dataset directory.")
        else:
            st.warning(f"Dataset directory '{dataset_dir}' not found.")
    
    if st.button("🔍 Analyze with CNN", type="primary", use_container_width=True) and selected_files:
        # Load inference modules only when needed
        if predict_timeline is None or get_top_instruments is None:
            predict_timeline, get_top_instruments = load_inference_modules()
            if predict_timeline is None:
                st.error("❌ Could not load inference modules. Please check if model files exist.")
                st.stop()
        
        with st.spinner("CNN is analyzing audio files..."):
            st.session_state.results.clear()
            st.session_state.audio_cache.clear()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file_item in enumerate(selected_files):
                # Handle both uploaded files and dataset files
                if audio_source == "Upload Files":
                    # Uploaded file - save to persistent directory
                    file_name = file_item.name
                    status_text.text(f"CNN Processing {file_name}... ({idx+1}/{len(selected_files)})")
                    
                    # Save to persistent directory
                    file_ext = os.path.splitext(file_name)[1] or ".wav"
                    safe_filename = "".join(c for c in file_name if c.isalnum() or c in "._- ")
                    persistent_path = os.path.join(
                        st.session_state.uploaded_files_dir,
                        f"{idx}_{safe_filename}"
                    )
                    
                    # Read file content and save
                    file_item.seek(0)  # Reset file pointer
                    with open(persistent_path, "wb") as f:
                        f.write(file_item.read())
                    
                    audio_path = persistent_path
                    display_name = file_name
                else:
                    # Dataset file - use existing path
                    audio_path = file_item
                    display_name = os.path.basename(audio_path)
                    status_text.text(f"CNN Processing {display_name}... ({idx+1}/{len(selected_files)})")
                
                try:
                    result = predict_timeline(audio_path, threshold=detection_threshold)
                    result["uploaded_file"] = display_name
                    result["file_path"] = audio_path  # Store persistent path
                    st.session_state.results.append(result)
                except Exception as e:
                    st.error(f"Error processing {display_name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(selected_files))
            
            status_text.empty()
            progress_bar.empty()
            st.success(f"✅ CNN successfully analyzed {len(st.session_state.results)} file(s)!")
    
    # Display results with audio playback
    if st.session_state.results:
        st.markdown("---")
        st.subheader("📊 CNN Detection Results")
        
        for idx, result in enumerate(st.session_state.results):
            with st.container():
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                
                col_h1, col_h2 = st.columns([3, 1])
                with col_h1:
                    st.markdown(f"### 🎧 {result.get('uploaded_file', result['file'])}")
                with col_h2:
                    st.metric("Duration", f"{result['duration']:.1f}s")
                
                # AUDIO PLAYBACK - Original File
                st.markdown("#### 🎵 Audio Player - Play Uploaded File")
                audio_path = result.get('file_path')
                if audio_path and os.path.exists(audio_path):
                    try:
                        audio_bytes = get_audio_bytes(audio_path)
                        create_audio_player(audio_bytes, f"main_audio_{idx}", autoplay=False)
                        st.caption("💡 Click play to hear the uploaded audio file directly in the UI")
                    except Exception as e:
                        st.warning(f"Audio playback not available: {str(e)}")
                
                st.markdown("---")
                
                detected = result.get('detected_instruments', [])
                
                if detected:
                    st.markdown("#### ✅ CNN Detected Instruments (Multilabel)")
                    st.markdown('<span class="multilabel-indicator">MULTILABEL: Multiple Instruments Detected</span>', unsafe_allow_html=True)
                    
                    # Display instruments with click-to-play segments
                    cols = st.columns(min(3, len(detected)))
                    for i, inst_info in enumerate(detected):
                        col_idx = i % 3
                        with cols[col_idx]:
                            inst_name = inst_info['instrument']
                            conf = inst_info['average_confidence']
                            
                            # Instrument badge with click
                            badge_key = f"play_{idx}_{i}"
                            if st.button(f"🎵 {inst_name}\n({conf:.1%})", key=badge_key, use_container_width=True):
                                # Play audio segment for this instrument
                                audio_path = result.get('file_path')
                                if audio_path and os.path.exists(audio_path):
                                    try:
                                        # Load unique features if needed
                                        if unique_features is None:
                                            unique_features = load_unique_features()
                                        
                                        if unique_features and unique_features.get('segments'):
                                            segments = unique_features['segments'](
                                                audio_path, result, inst_name,
                                                min_confidence=detection_threshold * 0.8,
                                                min_duration=1.0
                                            )
                                        else:
                                            segments = []
                                        if segments:
                                            seg = segments[0]
                                            audio_bytes = get_audio_bytes(
                                                audio_path,
                                                seg['start_time'],
                                                seg['end_time']
                                            )
                                            create_audio_player(audio_bytes, f"audio_{idx}_{i}", autoplay=True)
                                            st.success(f"Playing {inst_name} segment ({seg['start_time']:.1f}s - {seg['end_time']:.1f}s)")
                                        else:
                                            # Play full audio
                                            audio_bytes = get_audio_bytes(audio_path)
                                            create_audio_player(audio_bytes, f"audio_{idx}_{i}", autoplay=True)
                                            st.info(f"Playing full audio (instrument: {inst_name})")
                                    except Exception as e:
                                        st.error(f"Error playing audio: {str(e)}")
                            
                            st.caption(f"Avg: {conf:.3f} | Max: {inst_info['max_confidence']:.3f}")
                    
                    # Detailed table
                    det_df = pd.DataFrame([
                        {
                            'Instrument': inst['instrument'],
                            'Avg Confidence': f"{inst['average_confidence']:.3f}",
                            'Max Confidence': f"{inst['max_confidence']:.3f}",
                            'Min Confidence': f"{inst['min_confidence']:.3f}",
                            'Status': '✅ Detected'
                        }
                        for inst in detected
                    ])
                    st.dataframe(det_df, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"⚠️ No instruments detected above threshold ({detection_threshold}). Try lowering the threshold.")
                    st.info("💡 **Multilabel Detection**: The system uses sigmoid activation, so multiple instruments can be detected. Lower the threshold to see more detections.")
                
                # Top instruments
                st.markdown("#### 📈 Top 10 CNN Predictions")
                if get_top_instruments is None:
                    _, get_top_instruments = load_inference_modules()
                if get_top_instruments:
                    top_10 = get_top_instruments(result, top_k=10)
                else:
                    top_10 = []
                
                top_df = pd.DataFrame([
                    {
                        'Rank': f"#{i+1}",
                        'Instrument': inst['instrument'],
                        'Avg Confidence': f"{inst['average_confidence']:.3f}",
                        'Max Confidence': f"{inst['max_confidence']:.3f}",
                        'Status': '✅ Detected' if inst['average_confidence'] >= detection_threshold else '⚪ Below threshold'
                    }
                    for i, inst in enumerate(top_10)
                ])
                st.dataframe(top_df, use_container_width=True, hide_index=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")

# ================= ANALYTICS PAGE =================
elif page == "Analytics":
    st.title("📊 CNN Analysis & Visualizations")
    st.markdown("### Fast, clear visualizations with intensity graphs and mel-spectrograms")
    
    if not st.session_state.results:
        st.info("👆 Upload and analyze audio files in the Dashboard to see analytics here.")
    else:
        file_options = [r.get('uploaded_file', r['file']) for r in st.session_state.results]
        selected_file = st.selectbox("Select file for detailed CNN analysis", file_options)
        
        result = next((r for r in st.session_state.results 
                      if r.get('uploaded_file', r['file']) == selected_file), None)
        
        if result:
            st.subheader(f"📈 CNN Analysis: {result.get('uploaded_file', result['file'])}")
            
            # Audio Player at top
            st.markdown("#### 🎵 Audio Player")
            audio_path = result.get('file_path')
            if audio_path and os.path.exists(audio_path):
                try:
                    audio_bytes = get_audio_bytes(audio_path)
                    create_audio_player(audio_bytes, f"analytics_audio_{selected_file}", autoplay=False)
                except Exception as e:
                    st.warning(f"Audio playback not available: {str(e)}")
            
            st.markdown("---")
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Duration", f"{result['duration']:.1f}s")
            with col_m2:
                st.metric("CNN Windows", result['num_windows'])
            with col_m3:
                st.metric("Detected", len(result.get('detected_instruments', [])))
            with col_m4:
                st.metric("Threshold", f"{result['threshold']:.2f}")
            
            st.markdown("---")
            
            # Load visualization modules only when needed
            if viz_modules is None:
                viz_modules = load_visualization_modules()
            
            if viz_modules:
                # Fast Waveform (Optimized)
                st.markdown("### 🌊 Fast Waveform Visualization")
                try:
                    audio_path = result.get('file_path', result.get('uploaded_file', ''))
                    if os.path.exists(audio_path):
                        with st.spinner("Generating waveform (optimized)..."):
                            fig_wave = viz_modules['fast_waveform'](audio_path)
                            st.pyplot(fig_wave)
                            plt.close(fig_wave)
                        st.caption("✅ Fast waveform generation - optimized for speed")
                    else:
                        st.warning("Audio file path not available.")
                except Exception as e:
                    st.error(f"Waveform error: {str(e)}")
                
                st.markdown("---")
                
                # Fast Mel-Spectrogram
                st.markdown("### 📊 Mel-Spectrogram & Intensity Graph")
                try:
                    audio_path = result.get('file_path', result.get('uploaded_file', ''))
                    if os.path.exists(audio_path):
                        with st.spinner("Generating mel-spectrogram (optimized)..."):
                            fig_mel = viz_modules['fast_mel'](audio_path)
                            st.pyplot(fig_mel)
                            plt.close(fig_mel)
                        st.caption("✅ Fast mel-spectrogram generation - optimized for speed")
                    else:
                        st.warning("Audio file path not available.")
                except Exception as e:
                    st.error(f"Mel-spectrogram error: {str(e)}")
                
                st.markdown("---")
                
                # Comprehensive analysis (optional - can be slow)
                show_comprehensive = st.checkbox("Show Comprehensive Analysis (may take longer)", value=False)
                if show_comprehensive:
                    st.markdown("### 🎨 Complete CNN Analysis Dashboard")
                    try:
                        audio_path = result.get('file_path', result.get('uploaded_file', ''))
                        if os.path.exists(audio_path):
                            with st.spinner("Generating comprehensive analysis..."):
                                fig_comp = viz_modules['comprehensive'](audio_path, result, threshold=detection_threshold)
                                st.pyplot(fig_comp)
                                plt.close(fig_comp)
                        else:
                            st.warning("Audio file path not available for comprehensive plot.")
                    except Exception as e:
                        st.error(f"Error generating comprehensive plot: {str(e)}")
                
                st.markdown("---")
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    st.markdown("### ⏱️ Confidence Timeline")
                    try:
                        detected_insts = [inst['instrument'] for inst in result.get('detected_instruments', [])]
                        fig_timeline = viz_modules['timeline'](
                            result, 
                            instruments=detected_insts[:10] if detected_insts else None,
                            threshold=detection_threshold
                        )
                        st.pyplot(fig_timeline)
                        plt.close(fig_timeline)
                    except Exception as e:
                        st.error(f"Timeline error: {str(e)}")
                
                with col_viz2:
                    st.markdown("### 🔥 Detection Heatmap")
                    try:
                        fig_heatmap = viz_modules['heatmap'](result, threshold=detection_threshold)
                        st.pyplot(fig_heatmap)
                        plt.close(fig_heatmap)
                    except Exception as e:
                        st.error(f"Heatmap error: {str(e)}")
                
                st.markdown("---")
                
                # Audio Quality Analysis
                if unique_features is None:
                    unique_features = load_unique_features()
                
                if unique_features:
                    st.markdown("### 🎚️ Audio Quality Assessment")
                    try:
                        audio_path = result.get('file_path', result.get('uploaded_file', ''))
                        if os.path.exists(audio_path):
                            quality_metrics = unique_features['quality'](audio_path)
                            
                            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                            with col_q1:
                                st.metric("Quality Score", f"{quality_metrics['quality_score']}/100")
                            with col_q2:
                                st.metric("SNR", f"{quality_metrics['snr_db']:.1f} dB")
                            with col_q3:
                                st.metric("Dynamic Range", f"{quality_metrics['dynamic_range']:.3f}")
                            with col_q4:
                                st.metric("RMS Energy", f"{quality_metrics['rms_energy']:.3f}")
                            
                            fig_quality = unique_features['quality_plot'](quality_metrics)
                            st.pyplot(fig_quality)
                            plt.close(fig_quality)
                    except Exception as e:
                        st.warning(f"Quality analysis not available: {str(e)}")
            else:
                st.warning("Visualization modules not available.")

# ================= UNIQUE FEATURES PAGE =================
elif page == "Unique Features":
    st.title("🌟 Unique Features")
    st.markdown("### Advanced CNN Analysis & Unique Capabilities")
    
    if not st.session_state.results:
        st.info("👆 Upload and analyze audio files in the Dashboard first.")
    else:
        file_options = [r.get('uploaded_file', r['file']) for r in st.session_state.results]
        selected_file = st.selectbox("Select file for unique analysis", file_options)
        
        result = next((r for r in st.session_state.results 
                      if r.get('uploaded_file', r['file']) == selected_file), None)
        
        if result:
            st.subheader(f"🌟 Unique Analysis: {result.get('uploaded_file', result['file'])}")
            
            feature_tabs = st.tabs([
                "🎯 CNN Explainability (Grad-CAM)",
                "🔗 Instrument Similarity",
                "✂️ Audio Segmentation",
                "📊 Quality Analysis"
            ])
            
            audio_path = result.get('file_path', result.get('uploaded_file', ''))
            
            # Load unique features modules only when needed
            if unique_features is None:
                unique_features = load_unique_features()
            
            with feature_tabs[0]:
                st.markdown("### 🎯 CNN Explainability - What Does CNN Focus On?")
                st.markdown("""
                **Grad-CAM (Gradient-weighted Class Activation Mapping)** shows which parts of the 
                mel-spectrogram the CNN focuses on when detecting instruments.
                """)
                
                # Load model only when needed for Grad-CAM
                if model is None:
                    model, CLASS_NAMES = load_model_and_classes()
                
                if model is not None and os.path.exists(audio_path):
                    detected_insts = [inst['instrument'] for inst in result.get('detected_instruments', [])]
                    if detected_insts:
                        selected_inst = st.selectbox(
                            "Select instrument for Grad-CAM visualization",
                            detected_insts[:5]
                        )
                        
                        if st.button("Generate Grad-CAM Visualization"):
                            try:
                                if unique_features and unique_features.get('gradcam'):
                                    inst_idx = CLASS_NAMES.index(selected_inst) if selected_inst in CLASS_NAMES else 0
                                    fig_gradcam = unique_features['gradcam'](
                                        audio_path, model, selected_inst, inst_idx
                                    )
                                    st.pyplot(fig_gradcam)
                                    plt.close(fig_gradcam)
                                    st.success("✅ Grad-CAM visualization generated!")
                                else:
                                    st.error("Grad-CAM module not available.")
                            except Exception as e:
                                st.error(f"Grad-CAM error: {str(e)}")
                    else:
                        st.warning("No instruments detected. Analyze audio first.")
                else:
                    st.info("Grad-CAM requires model to be loaded.")
            
            with feature_tabs[1]:
                st.markdown("### 🔗 Instrument Similarity Matrix")
                if st.button("Generate Similarity Matrix"):
                    try:
                        if unique_features and unique_features.get('similarity'):
                            fig_sim = unique_features['similarity'](result, top_k=15)
                            st.pyplot(fig_sim)
                            plt.close(fig_sim)
                            st.success("✅ Similarity matrix generated!")
                        else:
                            st.error("Similarity matrix module not available.")
                    except Exception as e:
                        st.error(f"Similarity matrix error: {str(e)}")
            
            with feature_tabs[2]:
                st.markdown("### ✂️ Extract Audio Segments by Instrument")
                detected_insts = [inst['instrument'] for inst in result.get('detected_instruments', [])]
                if detected_insts and os.path.exists(audio_path):
                    selected_inst_seg = st.selectbox(
                        "Select instrument to extract segments",
                        detected_insts
                    )
                    
                    min_conf = st.slider("Minimum confidence", 0.3, 0.9, 0.5, 0.05)
                    min_dur = st.slider("Minimum duration (seconds)", 0.5, 5.0, 1.0, 0.5)
                    
                    if st.button("Extract Segments"):
                        try:
                            if unique_features and unique_features.get('segments'):
                                segments = unique_features['segments'](
                                    audio_path, result, selected_inst_seg, 
                                    min_confidence=min_conf, min_duration=min_dur
                                )
                            
                            if segments:
                                st.success(f"✅ Found {len(segments)} segments!")
                                
                                for idx, seg in enumerate(segments):
                                    with st.expander(f"Segment {idx+1}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s"):
                                        st.write(f"**Duration**: {seg['duration']:.2f} seconds")
                                        st.write(f"**Average Confidence**: {seg['average_confidence']:.3f}")
                                        
                                        # Play button
                                        if st.button(f"▶️ Play Segment {idx+1}", key=f"play_seg_{idx}"):
                                            audio_bytes = get_audio_bytes(
                                                audio_path,
                                                seg['start_time'],
                                                seg['end_time']
                                            )
                                            create_audio_player(audio_bytes, f"seg_{idx}", autoplay=True)
                                        
                                        # Download
                                        segment_bytes = io.BytesIO()
                                        import soundfile as sf
                                        sf.write(segment_bytes, seg['audio'], seg['sample_rate'], format='WAV')
                                        segment_bytes.seek(0)
                                        
                                        st.download_button(
                                            f"⬇️ Download Segment {idx+1}",
                                            segment_bytes,
                                            f"{selected_inst_seg}_segment_{idx+1}.wav",
                                            "audio/wav",
                                            key=f"seg_{idx}"
                                        )
                            else:
                                st.warning("No segments found. Try lowering minimum confidence or duration.")
                        except Exception as e:
                            st.error(f"Segmentation error: {str(e)}")
                else:
                    st.warning("No instruments detected or audio file not available.")
            
            with feature_tabs[3]:
                st.markdown("### 📊 Comprehensive Audio Quality Analysis")
                if os.path.exists(audio_path):
                    try:
                        if unique_features and unique_features.get('quality'):
                            quality_metrics = unique_features['quality'](audio_path)
                            
                            col_q1, col_q2, col_q3 = st.columns(3)
                            with col_q1:
                                st.metric("Overall Quality Score", f"{quality_metrics['quality_score']}/100")
                                st.metric("SNR", f"{quality_metrics['snr_db']:.1f} dB")
                            with col_q2:
                                st.metric("Dynamic Range", f"{quality_metrics['dynamic_range']:.4f}")
                                st.metric("Zero Crossing Rate", f"{quality_metrics['zero_crossing_rate']:.4f}")
                            with col_q3:
                                st.metric("Spectral Centroid", f"{quality_metrics['spectral_centroid_hz']:.0f} Hz")
                                st.metric("RMS Energy", f"{quality_metrics['rms_energy']:.4f}")
                            
                            fig_quality = unique_features['quality_plot'](quality_metrics)
                            st.pyplot(fig_quality)
                            plt.close(fig_quality)
                        else:
                            st.error("Quality analysis module not available.")
                    except Exception as e:
                        st.error(f"Quality analysis error: {str(e)}")
                else:
                    st.warning("Audio file not available for quality analysis.")

# ================= HISTORY PAGE =================
elif page == "History":
    st.title("🕘 Prediction History")
    
    if not st.session_state.results:
        st.info("No predictions yet. Go to Dashboard to analyze audio files.")
    else:
        # Optimized: Use cached computation
        @st.cache_data
        def compute_history_stats(results):
            total_files = len(results)
            total_detected = sum(len(r.get('detected_instruments', [])) for r in results)
            avg_duration = np.mean([r['duration'] for r in results]) if results else 0
            unique_insts = set()
            for r in results:
                for inst in r.get('detected_instruments', []):
                    unique_insts.add(inst['instrument'])
            return total_files, total_detected, avg_duration, len(unique_insts)
        
        total_files, total_detected, avg_duration, unique_count = compute_history_stats(st.session_state.results)
        
        st.subheader("📊 Summary Statistics")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.metric("Total Files", total_files)
        with col_s2:
            st.metric("Total Detections", total_detected)
        with col_s3:
            st.metric("Avg Duration", f"{avg_duration:.1f}s")
        with col_s4:
            st.metric("Unique Instruments", unique_count)
        
        st.markdown("---")
        
        st.subheader("📋 Detailed History")
        # Optimized: Limit display to avoid slow rendering
        max_display = st.slider("Show last N predictions", 5, len(st.session_state.results), min(10, len(st.session_state.results)))
        history_data = []
        for r in st.session_state.results[-max_display:]:
            detected = r.get('detected_instruments', [])
            inst_names = ', '.join([inst['instrument'] for inst in detected]) if detected else 'None'
            history_data.append({
                "File": r.get('uploaded_file', r['file']),
                "Duration": f"{r['duration']:.1f}s",
                "Detected Instruments": inst_names,
                "Count": len(detected),
                "Windows": r['num_windows']
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

# ================= EXPORT PAGE =================
elif page == "Export":
    st.title("📤 Export Results")
    st.markdown("### Export CNN predictions with all visualizations")
    
    if not st.session_state.results:
        st.warning("No data to export. Make predictions first.")
    else:
        export_data = []
        for r in st.session_state.results:
            detected = r.get('detected_instruments', [])
            export_data.append({
                "file": r.get('uploaded_file', r['file']),
                "duration": r['duration'],
                "num_windows": r['num_windows'],
                "threshold": r['threshold'],
                "detected_instruments": [
                    {
                        "instrument": inst['instrument'],
                        "average_confidence": inst['average_confidence'],
                        "max_confidence": inst['max_confidence'],
                        "min_confidence": inst['min_confidence']
                    }
                    for inst in detected
                ],
                "all_average_confidences": r['average_confidences'],
                "timeline": [
                    {
                        "time": window['time'],
                        "detected": [inst['instrument'] for inst in window['detected_instruments']]
                    }
                    for window in r['timeline']
                ]
            })
        
        st.markdown("### 📄 CSV Export")
        csv_rows = []
        for data in export_data:
            for inst in data['detected_instruments']:
                csv_rows.append({
                    "File": data['file'],
                    "Instrument": inst['instrument'],
                    "Avg Confidence": inst['average_confidence'],
                    "Max Confidence": inst['max_confidence'],
                    "Min Confidence": inst['min_confidence']
                })
        
        if csv_rows:
            df_csv = pd.DataFrame(csv_rows)
            csv_data = df_csv.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download CSV",
                csv_data,
                "instruet_cnn_predictions.csv",
                "text/csv",
                key="csv_download"
            )
        
        st.markdown("### 📋 JSON Export")
        json_data = json.dumps(export_data, indent=2).encode('utf-8')
        st.download_button(
            "⬇️ Download JSON",
            json_data,
            "instruet_cnn_predictions.json",
            "application/json",
            key="json_download"
        )
        
        st.markdown("### 📑 PDF Report")
        if st.button("⬇️ Generate PDF Report"):
            try:
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                story = []
                
                title = Paragraph("InstruNet CNN-Based Music Instrument Recognition Report", styles['Title'])
                story.append(title)
                story.append(Spacer(1, 0.2*inch))
                
                summary_text = f"""
                <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
                <b>System:</b> CNN-Based Multilabel Detection<br/>
                <b>Total Files:</b> {len(export_data)}<br/>
                <b>Detection Threshold:</b> {detection_threshold}
                """
                if training_metrics:
                    summary_text += f"<br/><b>Model Training Accuracy:</b> {training_metrics['train_accuracy']:.2%}<br/>"
                    summary_text += f"<b>Model Validation Accuracy:</b> {training_metrics['val_accuracy']:.2%}"
                
                story.append(Paragraph(summary_text, styles['Normal']))
                story.append(Spacer(1, 0.3*inch))
                
                table_data = [['File', 'Detected Instruments', 'Count']]
                for data in export_data:
                    inst_names = ', '.join([inst['instrument'] for inst in data['detected_instruments']])
                    if not inst_names:
                        inst_names = 'None'
                    table_data.append([
                        data['file'][:30] + '...' if len(data['file']) > 30 else data['file'],
                        inst_names[:50] + '...' if len(inst_names) > 50 else inst_names,
                        str(len(data['detected_instruments']))
                    ])
                
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                
                doc.build(story)
                buffer.seek(0)
                
                st.download_button(
                    "⬇️ Download PDF",
                    buffer,
                    "instruet_cnn_report.pdf",
                    "application/pdf",
                    key="pdf_download"
                )
                st.success("PDF generated successfully!")
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
        
        st.markdown("---")
        st.subheader("👁️ Export Preview")
        if csv_rows:
            st.dataframe(df_csv, use_container_width=True, hide_index=True)

# ================= ABOUT PAGE =================
elif page == "About":
    st.title("ℹ️ About InstruNet")
    
    # Training Accuracy Display
    if training_metrics:
        st.markdown("### 📊 Model Training Performance")
        
        # Show Hamming accuracy (more meaningful for multilabel)
        is_hamming = training_metrics.get('is_hamming', False)
        metric_name = "Hamming Accuracy" if is_hamming else "Accuracy"
        metric_note = "⭐ Per-label accuracy (more meaningful for multilabel)" if is_hamming else "Exact match accuracy"
        
        col_about1, col_about2, col_about3 = st.columns(3)
        with col_about1:
            st.markdown(f"""
            <div class="accuracy-card">
                <h3>📈 Training {metric_name}</h3>
                <h1>{training_metrics['train_accuracy']:.2%}</h1>
                <p>Final Training {metric_name}</p>
                <p style="font-size: 0.8rem; color: #888;">{metric_note}</p>
                <p style="font-size: 0.9rem;">Loss: {training_metrics['train_loss']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_about2:
            st.markdown(f"""
            <div class="accuracy-card">
                <h3>✅ Validation {metric_name}</h3>
                <h1>{training_metrics['val_accuracy']:.2%}</h1>
                <p>Final Validation {metric_name}</p>
                <p style="font-size: 0.8rem; color: #888;">{metric_note}</p>
                <p style="font-size: 0.9rem;">Loss: {training_metrics['val_loss']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_about3:
            gap = training_metrics['train_accuracy'] - training_metrics['val_accuracy']
            gap_status = "Excellent" if gap < 0.05 else "Good" if gap < 0.1 else "Fair"
            gap_color = "#00C853" if gap < 0.05 else "#FFB300" if gap < 0.1 else "#FF5252"
            st.markdown(f"""
            <div class="accuracy-card" style="background: linear-gradient(135deg, {gap_color} 0%, #00E676 100%);">
                <h3>📊 Generalization</h3>
                <h1>{gap:.2%}</h1>
                <p>Accuracy Gap</p>
                <p style="font-size: 0.9rem;">Status: {gap_status}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show subset accuracy if available
        if training_metrics.get('train_subset_accuracy') is not None:
            st.markdown("---")
            st.markdown("#### 📊 Additional Metrics")
            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                st.info(f"**Subset Accuracy (Exact Match)**: {training_metrics['train_subset_accuracy']:.2%} (Training)")
            with col_sub2:
                st.info(f"**Subset Accuracy (Exact Match)**: {training_metrics['val_subset_accuracy']:.2%} (Validation)")
            st.caption("💡 **Note**: Subset accuracy requires ALL labels to match exactly. Hamming accuracy (shown above) is more meaningful for multilabel tasks as it shows average per-label accuracy.")
        
        st.markdown(f"""
        **Training Details:**
        - Total Epochs: {training_metrics['epochs']}
        - Training {metric_name}: {training_metrics['train_accuracy']:.2%}
        - Validation {metric_name}: {training_metrics['val_accuracy']:.2%}
        - Accuracy Gap: {gap:.2%} ({gap_status} generalization)
        """)
        st.markdown("---")
    
    st.markdown("""
    ### 🎵 InstruNet - CNN-Based Music Instrument Recognition
    
    **Production-Grade Deep Learning System**
    
    ---
    
    ### 🧠 CNN Architecture
    
    **Model Details:**
    - Base: EfficientNetB1 (ImageNet pretrained) - Maximum Accuracy Version
    - Input: Mel-spectrogram images (128×128×3)
    - Output: Sigmoid activation (multilabel)
    - Loss: Binary Cross-Entropy with class weights
    - Optimizer: Adam with learning rate scheduling
    - Metrics: Hamming Accuracy, F1 Score, Precision, Recall, AUC
    
    **Why CNN?**
    - CNNs excel at recognizing patterns in 2D images
    - Mel-spectrograms are 2D representations of audio
    - Transfer learning improves accuracy on limited data
    - Deep layers capture complex instrument features
    
    ---
    
    ### 🎯 Multilabel Detection
    
    **How It Works:**
    1. Audio converted to mel-spectrogram (2D image)
    2. CNN extracts visual features
    3. Sigmoid outputs independent probabilities
    4. Multiple instruments can be detected simultaneously
    
    **Key Difference:**
    - Single-label: "Which ONE instrument?" (Softmax)
    - Multilabel: "Which instruments are present?" (Sigmoid)
    
    ---
    
    ### ⚡ Performance Optimizations
    
    - **Fast Waveform Generation**: Optimized plotting with downsampling
    - **Fast Mel-Spectrogram**: Reduced resolution for speed
    - **Audio Caching**: Cached audio processing
    - **Optional Comprehensive Analysis**: Can be enabled if needed
    
    ---
    
    ### 🌟 Unique Features
    
    **1. CNN Explainability (Grad-CAM)**
    - Visualize what parts of spectrogram CNN focuses on
    - Understand CNN decision-making process
    - Model interpretability and transparency
    
    **2. Instrument Similarity Matrix**
    - Find instruments with similar detection patterns
    - Cosine similarity based on CNN confidence over time
    - Useful for instrument family analysis
    
    **3. Audio Segmentation**
    - Extract segments where specific instruments are detected
    - Download instrument-specific audio clips
    - Useful for music production and analysis
    
    **4. Quality Analysis**
    - Comprehensive audio quality metrics
    - SNR, dynamic range, spectral analysis
    - Quality score for CNN prediction reliability
    
    **5. Audio Playback**
    - Play uploaded audio directly in UI
    - Play instrument-specific segments
    - Interactive audio experience
    
    ---
    
    **Developed for: InstruNet Project**  
    **Technology: CNN (Convolutional Neural Networks)**  
    **Approach: Deep Learning with Transfer Learning**
    """)
