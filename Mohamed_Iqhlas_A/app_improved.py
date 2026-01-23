"""
IMPROVED InstruNet - CNN-Based Music Instrument Recognition System
Clear, professional UI with per-instrument visualizations and enhanced PDF export.
"""

import streamlit as st
import tempfile
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
from datetime import datetime
import os
import base64

# Import improved inference and visualizations
try:
    from src.inference_multilabel_timeline import predict_timeline, get_top_instruments
    from src.improved_visualizations import (
        plot_per_instrument_waveform,
        plot_mel_spectrogram_clear,
        plot_instrument_confidence_timeline_clear,
        plot_detection_heatmap_clear,
        plot_comprehensive_analysis_clear
    )
    from src.unique_features import (
        plot_gradcam_explanation,
        plot_instrument_similarity_matrix,
        extract_instrument_segments,
        analyze_audio_quality,
        plot_audio_quality_metrics
    )
    
    # Load model and classes for Grad-CAM
    import tensorflow as tf
    import json
    MODEL_PATH = "multilabel_timeline_model.keras"
    CLASSES_JSON_PATH = "multilabel_classes.json"
    
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_JSON_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            with open(CLASSES_JSON_PATH, 'r') as f:
                CLASS_NAMES = json.load(f)
        except:
            model = None
            CLASS_NAMES = []
    else:
        model = None
        CLASS_NAMES = []
        
except ImportError as e:
    st.warning(f"Some features may not be available: {e}")
    model = None
    CLASS_NAMES = []

# Page configuration
st.set_page_config(
    page_title="InstruNet - CNN-Based Music Recognition",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1E88E5, #00ACC1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .cnn-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: white;
    }
    .instrument-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-weight: bold;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1E88E5, #00ACC1);
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "results" not in st.session_state:
    st.session_state.results = []

# Sidebar
with st.sidebar:
    st.markdown("## 🎵 InstruNet")
    st.markdown('<div class="cnn-badge">CNN-Based System</div>', unsafe_allow_html=True)
    st.markdown("**Multilabel Instrument Recognition**")
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
    st.markdown('<p style="text-align: center; font-size: 1.5rem; color: #666; font-weight: bold;">CNN-Based Music Instrument Recognition System</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888;">Deep Learning • Multilabel Detection • Timeline Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # CNN Badge
    st.markdown('<div style="text-align: center;"><div class="cnn-badge">Powered by Convolutional Neural Networks (CNN)</div></div>', unsafe_allow_html=True)
    st.markdown("")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🧠 CNN Architecture
        **EfficientNetB0 Base**
        
        Transfer learning with fine-tuning
        Sigmoid output for multilabel
        """)
    with col2:
        st.markdown("""
        ### 🎯 Multilabel Detection
        **Multiple Instruments**
        
        Simultaneous detection
        Independent predictions
        """)
    with col3:
        st.markdown("""
        ### ⏱️ Timeline Analysis
        **Sliding Windows**
        
        3-second windows
        Confidence over time
        """)
    
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
    - ✅ **Clear Visualizations**: Per-instrument waveforms, mel-spectrograms, intensity graphs
    - ✅ **Professional Export**: PDF with all visualizations included
    - 🌟 **Unique Features**: 
      - CNN Explainability (Grad-CAM) - See what CNN focuses on
      - Instrument Similarity Matrix - Find similar instruments
      - Audio Segmentation - Extract instrument-specific segments
      - Quality Analysis - Comprehensive audio quality metrics
    
    👉 **Navigate to Dashboard** to start detecting instruments!
    """)

# ================= DASHBOARD PAGE =================
elif page == "Dashboard":
    st.title("🎯 CNN Instrument Detection Dashboard")
    st.markdown("### Upload audio files for CNN-based multilabel instrument recognition")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "📁 Upload Audio Files (WAV / MP3)",
        type=["wav", "mp3"],
        accept_multiple_files=True,
        help="CNN will analyze each file using sliding-window inference"
    )
    
    # Analyze button
    if st.button("🔍 Analyze with CNN", type="primary", use_container_width=True) and uploaded_files:
        with st.spinner("CNN is analyzing audio files..."):
            st.session_state.results.clear()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"CNN Processing {file.name}... ({idx+1}/{len(uploaded_files)})")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(file.read())
                    path = tmp.name
                
                try:
                    result = predict_timeline(path, threshold=detection_threshold)
                    result["uploaded_file"] = file.name
                    result["file_path"] = path  # Store path for visualizations
                    st.session_state.results.append(result)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.empty()
            progress_bar.empty()
            st.success(f"✅ CNN successfully analyzed {len(st.session_state.results)} file(s)!")
    
    # Display results
    if st.session_state.results:
        st.markdown("---")
        st.subheader("📊 CNN Detection Results")
        
        for idx, result in enumerate(st.session_state.results):
            with st.container():
                # Header
                col_h1, col_h2 = st.columns([3, 1])
                with col_h1:
                    st.markdown(f"### 🎧 {result.get('uploaded_file', result['file'])}")
                with col_h2:
                    st.metric("Duration", f"{result['duration']:.1f}s")
                
                # Detected instruments - FIXED MULTILABEL
                detected = result.get('detected_instruments', [])
                
                if detected:
                    st.markdown("#### ✅ CNN Detected Instruments (Multilabel)")
                    
                    # Display as badges
                    inst_html = " ".join([
                        f'<span class="instrument-badge">{inst["instrument"]} ({inst["average_confidence"]:.1%})</span>'
                        for inst in detected
                    ])
                    st.markdown(inst_html, unsafe_allow_html=True)
                    
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
                
                # Top instruments
                st.markdown("#### 📈 Top 10 CNN Predictions")
                top_10 = get_top_instruments(result, top_k=10)
                
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
                
                st.markdown("---")

# ================= ANALYTICS PAGE =================
elif page == "Analytics":
    st.title("📊 CNN Analysis & Visualizations")
    st.markdown("### Clear per-instrument visualizations with intensity graphs and mel-spectrograms")
    
    if not st.session_state.results:
        st.info("👆 Upload and analyze audio files in the Dashboard to see analytics here.")
    else:
        # File selection
        file_options = [r.get('uploaded_file', r['file']) for r in st.session_state.results]
        selected_file = st.selectbox("Select file for detailed CNN analysis", file_options)
        
        result = next((r for r in st.session_state.results 
                      if r.get('uploaded_file', r['file']) == selected_file), None)
        
        if result:
            st.subheader(f"📈 CNN Analysis: {result.get('uploaded_file', result['file'])}")
            
            # Summary metrics
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
            
            # Comprehensive analysis
            st.markdown("### 🎨 Complete CNN Analysis Dashboard")
            try:
                audio_path = result.get('file_path', result.get('uploaded_file', ''))
                if os.path.exists(audio_path):
                    fig_comp = plot_comprehensive_analysis_clear(audio_path, result, threshold=detection_threshold)
                    st.pyplot(fig_comp)
                    plt.close(fig_comp)
                else:
                    st.warning("Audio file path not available for comprehensive plot.")
            except Exception as e:
                st.error(f"Error generating comprehensive plot: {str(e)}")
            
            st.markdown("---")
            
            # Per-instrument waveforms
            detected_insts = [inst['instrument'] for inst in result.get('detected_instruments', [])]
            if detected_insts:
                st.markdown("### 🌊 Per-Instrument Waveforms with Intensity")
                try:
                    audio_path = result.get('file_path', result.get('uploaded_file', ''))
                    if os.path.exists(audio_path):
                        fig_wave = plot_per_instrument_waveform(
                            audio_path, result, detected_insts[:5]  # Top 5 instruments
                        )
                        st.pyplot(fig_wave)
                        plt.close(fig_wave)
                    else:
                        st.warning("Audio file path not available.")
                except Exception as e:
                    st.error(f"Waveform error: {str(e)}")
            
            st.markdown("---")
            
            # Mel-spectrogram and intensity
            st.markdown("### 📊 Mel-Spectrogram & Intensity Graph")
            try:
                audio_path = result.get('file_path', result.get('uploaded_file', ''))
                if os.path.exists(audio_path):
                    fig_mel = plot_mel_spectrogram_clear(audio_path, result)
                    st.pyplot(fig_mel)
                    plt.close(fig_mel)
                else:
                    st.warning("Audio file path not available.")
            except Exception as e:
                st.error(f"Mel-spectrogram error: {str(e)}")
            
            st.markdown("---")
            
            # Individual visualizations
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("### ⏱️ Confidence Timeline")
                try:
                    fig_timeline = plot_instrument_confidence_timeline_clear(
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
                    fig_heatmap = plot_detection_heatmap_clear(result, threshold=detection_threshold)
                    st.pyplot(fig_heatmap)
                    plt.close(fig_heatmap)
                except Exception as e:
                    st.error(f"Heatmap error: {str(e)}")
            
            st.markdown("---")
            
            # Audio Quality Analysis
            st.markdown("### 🎚️ Audio Quality Assessment")
            try:
                audio_path = result.get('file_path', result.get('uploaded_file', ''))
                if os.path.exists(audio_path):
                    quality_metrics = analyze_audio_quality(audio_path)
                    
                    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                    with col_q1:
                        st.metric("Quality Score", f"{quality_metrics['quality_score']}/100")
                    with col_q2:
                        st.metric("SNR", f"{quality_metrics['snr_db']:.1f} dB")
                    with col_q3:
                        st.metric("Dynamic Range", f"{quality_metrics['dynamic_range']:.3f}")
                    with col_q4:
                        st.metric("RMS Energy", f"{quality_metrics['rms_energy']:.3f}")
                    
                    fig_quality = plot_audio_quality_metrics(quality_metrics)
                    st.pyplot(fig_quality)
                    plt.close(fig_quality)
            except Exception as e:
                st.warning(f"Quality analysis not available: {str(e)}")

# ================= UNIQUE FEATURES PAGE =================
elif page == "Unique Features":
    st.title("🌟 Unique Features")
    st.markdown("### Advanced CNN Analysis & Unique Capabilities")
    
    if not st.session_state.results:
        st.info("👆 Upload and analyze audio files in the Dashboard first.")
    else:
        # File selection
        file_options = [r.get('uploaded_file', r['file']) for r in st.session_state.results]
        selected_file = st.selectbox("Select file for unique analysis", file_options)
        
        result = next((r for r in st.session_state.results 
                      if r.get('uploaded_file', r['file']) == selected_file), None)
        
        if result:
            st.subheader(f"🌟 Unique Analysis: {result.get('uploaded_file', result['file'])}")
            
            # Feature selection
            feature_tabs = st.tabs([
                "🎯 CNN Explainability (Grad-CAM)",
                "🔗 Instrument Similarity",
                "✂️ Audio Segmentation",
                "📊 Quality Analysis"
            ])
            
            audio_path = result.get('file_path', result.get('uploaded_file', ''))
            
            # Tab 1: Grad-CAM
            with feature_tabs[0]:
                st.markdown("### 🎯 CNN Explainability - What Does CNN Focus On?")
                st.markdown("""
                **Grad-CAM (Gradient-weighted Class Activation Mapping)** shows which parts of the 
                mel-spectrogram the CNN focuses on when detecting instruments. This provides 
                **model explainability** - understanding how CNN makes decisions.
                """)
                
                if model is not None and os.path.exists(audio_path):
                    detected_insts = [inst['instrument'] for inst in result.get('detected_instruments', [])]
                    if detected_insts:
                        selected_inst = st.selectbox(
                            "Select instrument for Grad-CAM visualization",
                            detected_insts[:5]
                        )
                        
                        if st.button("Generate Grad-CAM Visualization"):
                            try:
                                inst_idx = CLASS_NAMES.index(selected_inst) if selected_inst in CLASS_NAMES else 0
                                fig_gradcam = plot_gradcam_explanation(
                                    audio_path, model, selected_inst, inst_idx
                                )
                                st.pyplot(fig_gradcam)
                                plt.close(fig_gradcam)
                                st.success("✅ Grad-CAM visualization generated! Red/yellow areas show where CNN focuses.")
                            except Exception as e:
                                st.error(f"Grad-CAM error: {str(e)}")
                    else:
                        st.warning("No instruments detected. Analyze audio first.")
                else:
                    st.info("Grad-CAM requires model to be loaded. This feature shows CNN attention.")
            
            # Tab 2: Instrument Similarity
            with feature_tabs[1]:
                st.markdown("### 🔗 Instrument Similarity Matrix")
                st.markdown("""
                **Unique Feature**: Shows which instruments have similar CNN confidence patterns over time.
                High similarity (red) means instruments are detected together or have similar patterns.
                """)
                
                if st.button("Generate Similarity Matrix"):
                    try:
                        fig_sim = plot_instrument_similarity_matrix(result, top_k=15)
                        st.pyplot(fig_sim)
                        plt.close(fig_sim)
                        st.success("✅ Similarity matrix generated! Analyze which instruments are similar.")
                    except Exception as e:
                        st.error(f"Similarity matrix error: {str(e)}")
            
            # Tab 3: Audio Segmentation
            with feature_tabs[2]:
                st.markdown("### ✂️ Extract Audio Segments by Instrument")
                st.markdown("""
                **Unique Feature**: Extract audio segments where specific instruments are detected.
                Download segments for further analysis.
                """)
                
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
                            segments = extract_instrument_segments(
                                audio_path, result, selected_inst_seg, 
                                min_confidence=min_conf, min_duration=min_dur
                            )
                            
                            if segments:
                                st.success(f"✅ Found {len(segments)} segments!")
                                
                                for idx, seg in enumerate(segments):
                                    with st.expander(f"Segment {idx+1}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s (Confidence: {seg['average_confidence']:.2f})"):
                                        st.write(f"**Duration**: {seg['duration']:.2f} seconds")
                                        st.write(f"**Average Confidence**: {seg['average_confidence']:.3f}")
                                        
                                        # Save segment
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
            
            # Tab 4: Quality Analysis
            with feature_tabs[3]:
                st.markdown("### 📊 Comprehensive Audio Quality Analysis")
                st.markdown("""
                **Unique Feature**: Detailed audio quality metrics including SNR, dynamic range, and quality score.
                """)
                
                if os.path.exists(audio_path):
                    try:
                        quality_metrics = analyze_audio_quality(audio_path)
                        
                        st.markdown("#### Quality Metrics")
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
                        
                        fig_quality = plot_audio_quality_metrics(quality_metrics)
                        st.pyplot(fig_quality)
                        plt.close(fig_quality)
                        
                        # Quality interpretation
                        if quality_metrics['quality_score'] >= 70:
                            st.success("✅ High quality audio - excellent for CNN analysis")
                        elif quality_metrics['quality_score'] >= 50:
                            st.info("ℹ️ Medium quality audio - good for CNN analysis")
                        else:
                            st.warning("⚠️ Low quality audio - may affect CNN accuracy")
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
        # Summary
        st.subheader("📊 Summary Statistics")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.metric("Total Files", len(st.session_state.results))
        with col_s2:
            total_detected = sum(len(r.get('detected_instruments', [])) for r in st.session_state.results)
            st.metric("Total Detections", total_detected)
        with col_s3:
            avg_duration = np.mean([r['duration'] for r in st.session_state.results])
            st.metric("Avg Duration", f"{avg_duration:.1f}s")
        with col_s4:
            unique_insts = set()
            for r in st.session_state.results:
                for inst in r.get('detected_instruments', []):
                    unique_insts.add(inst['instrument'])
            st.metric("Unique Instruments", len(unique_insts))
        
        st.markdown("---")
        
        # History table
        st.subheader("📋 Detailed History")
        history_data = []
        for r in st.session_state.results:
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
        # Prepare export data
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
        
        # CSV Export
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
        
        # JSON Export
        st.markdown("### 📋 JSON Export")
        json_data = json.dumps(export_data, indent=2).encode('utf-8')
        st.download_button(
            "⬇️ Download JSON",
            json_data,
            "instruet_cnn_predictions.json",
            "application/json",
            key="json_download"
        )
        
        # PDF Export with visualizations
        st.markdown("### 📑 PDF Report (With Visualizations)")
        if st.button("⬇️ Generate PDF Report"):
            try:
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                story = []
                
                # Title
                title = Paragraph("InstruNet CNN-Based Music Instrument Recognition Report", styles['Title'])
                story.append(title)
                story.append(Spacer(1, 0.2*inch))
                
                # Summary
                summary_text = f"""
                <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
                <b>System:</b> CNN-Based Multilabel Detection<br/>
                <b>Total Files:</b> {len(export_data)}<br/>
                <b>Detection Threshold:</b> {detection_threshold}
                """
                story.append(Paragraph(summary_text, styles['Normal']))
                story.append(Spacer(1, 0.3*inch))
                
                # Table
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
                st.success("PDF generated successfully! Note: Visualizations are included in Analytics page.")
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
        
        # Preview
        st.markdown("---")
        st.subheader("👁️ Export Preview")
        if csv_rows:
            st.dataframe(df_csv, use_container_width=True, hide_index=True)

# ================= ABOUT PAGE =================
elif page == "About":
    st.title("ℹ️ About InstruNet")
    
    st.markdown("""
    ### 🎵 InstruNet - CNN-Based Music Instrument Recognition
    
    **Production-Grade Deep Learning System**
    
    ---
    
    ### 🧠 CNN Architecture
    
    **Model Details:**
    - Base: EfficientNetB0 (ImageNet pretrained)
    - Input: Mel-spectrogram images (128×128×3)
    - Output: Sigmoid activation (multilabel)
    - Loss: Binary Cross-Entropy
    - Optimizer: Adam with learning rate scheduling
    
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
    
    ### ⏱️ Timeline Analysis
    
    **Sliding Window Approach:**
    - 3-second windows with 1-second hop
    - Each window analyzed independently by CNN
    - Results aggregated across time
    - Shows instrument activity over time
    
    ---
    
    ### 📊 Visualizations
    
    **Per-Instrument Waveforms:**
    - Clear waveform for each detected instrument
    - Intensity overlay showing energy over time
    
    **Mel-Spectrogram:**
    - Visual representation of frequency content
    - Shows what CNN "sees" as input
    
    **Intensity Graphs:**
    - RMS energy over time
    - Shows audio dynamics
    
    **Confidence Timelines:**
    - CNN confidence for each instrument over time
    - Shows when instruments are most active
    
    ---
    
    ### 🎓 Technical Details
    
    **Audio Processing:**
    - Sample Rate: 22,050 Hz
    - Window Size: 3 seconds
    - Hop Size: 1 second
    - Features: Mel-spectrogram (128 mel bins)
    
    **CNN Training:**
    - Data augmentation for generalization
    - Transfer learning from ImageNet
    - Fine-tuning on music data
    - Early stopping to prevent overfitting
    
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
    
    ---
    
    **Developed for: InstruNet Project**  
    **Technology: CNN (Convolutional Neural Networks)**  
    **Approach: Deep Learning with Transfer Learning**  
    **Unique Features: Explainability, Similarity Analysis, Segmentation, Quality Metrics**
    """)

