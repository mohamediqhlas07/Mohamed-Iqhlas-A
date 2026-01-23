"""
InstruNet - Multilabel Timeline Streamlit Application
Production-ready UI for multilabel instrument detection with timeline support.
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

# Import multilabel inference and visualizations
from src.inference_multilabel_timeline import predict_timeline, get_top_instruments
from src.timeline_visualizations import (
    plot_confidence_timeline,
    plot_detection_heatmap,
    plot_instrument_bars,
    plot_comprehensive_timeline
)

# Page configuration
st.set_page_config(
    page_title="InstruNet - Multilabel Instrument Detection",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
    }
    .stProgress > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if "results" not in st.session_state:
    st.session_state.results = []

# Sidebar
with st.sidebar:
    st.markdown("## 🎵 InstruNet")
    st.markdown("**Multilabel Instrument Detection**")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Home", "Dashboard", "Analytics", "History", "Export", "About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    detection_threshold = st.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Confidence threshold for instrument detection"
    )
    
    st.markdown("---")
    if st.session_state.results:
        st.metric("Total Predictions", len(st.session_state.results))

# ================= HOME PAGE =================
if page == "Home":
    st.markdown('<div class="main-header">🎵 InstruNet</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Multilabel Music Instrument Detection System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🎯 Multilabel Detection
        Detect **multiple instruments** simultaneously
        
        True multilabel classification with timeline support
        """)
    with col2:
        st.markdown("""
        ### ⏱️ Timeline Analysis
        Sliding-window inference
        
        See instrument activity over time
        """)
    with col3:
        st.markdown("""
        ### 📊 Advanced Analytics
        Confidence timelines
        
        Detection heatmaps & visualizations
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔍 What is InstruNet?
    
    InstruNet is a **production-grade CNN-based system** for detecting multiple musical instruments 
    simultaneously in audio files. Unlike single-label classification, InstruNet can identify 
    **all instruments present** in a song using multilabel deep learning.
    
    ### ⚙️ How It Works
    
    1. **Audio Input**: Upload audio files (WAV, MP3)
    2. **Sliding Windows**: Audio is analyzed in 3-second windows with 1-second hop
    3. **Multilabel Prediction**: CNN predicts probabilities for all 28 instruments
    4. **Timeline Aggregation**: Results aggregated across time windows
    5. **Visualization**: Confidence timelines and detection heatmaps
    
    ### ✨ Key Features
    
    - ✅ **True Multilabel**: Detects multiple simultaneous instruments
    - ✅ **Timeline Support**: See instrument activity over time
    - ✅ **28 Instrument Classes**: Comprehensive instrument coverage
    - ✅ **Confidence Thresholding**: Adjustable detection sensitivity
    - ✅ **Professional Visualizations**: Timeline plots, heatmaps, bar charts
    - ✅ **Export Options**: CSV, JSON, PDF reports
    
    ### 🧪 Technical Details
    
    - **Model**: MobileNetV2-based CNN with transfer learning
    - **Output**: Sigmoid activation (multilabel)
    - **Loss**: Binary cross-entropy
    - **Input**: Mel-spectrogram (128×128×3)
    - **Window Size**: 3 seconds
    - **Hop Size**: 1 second
    
    👉 **Navigate to Dashboard** to start detecting instruments!
    """)

# ================= DASHBOARD PAGE =================
elif page == "Dashboard":
    st.title("🎯 Instrument Detection Dashboard")
    st.markdown("Upload audio files to detect multiple instruments simultaneously.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "📁 Upload Audio Files (WAV / MP3)",
        type=["wav", "mp3"],
        accept_multiple_files=True,
        help="Select audio files for multilabel instrument detection"
    )
    
    # Analyze button
    if st.button("🔍 Analyze Instruments", type="primary", use_container_width=True) and uploaded_files:
        with st.spinner("Analyzing audio files..."):
            st.session_state.results.clear()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}... ({idx+1}/{len(uploaded_files)})")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(file.read())
                    path = tmp.name
                
                try:
                    result = predict_timeline(path, threshold=detection_threshold)
                    result["uploaded_file"] = file.name
                    st.session_state.results.append(result)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.empty()
            progress_bar.empty()
            st.success(f"✅ Successfully analyzed {len(st.session_state.results)} file(s)!")
    
    # Display results
    if st.session_state.results:
        st.markdown("---")
        st.subheader("📊 Detection Results")
        
        for idx, result in enumerate(st.session_state.results):
            with st.container():
                # Header
                col_h1, col_h2 = st.columns([3, 1])
                with col_h1:
                    st.markdown(f"### 🎧 {result.get('uploaded_file', result['file'])}")
                with col_h2:
                    st.metric("Duration", f"{result['duration']:.1f}s")
                
                # Detected instruments
                detected = result.get('detected_instruments', [])
                
                if detected:
                    st.markdown("#### ✅ Detected Instruments")
                    
                    # Create columns for instruments
                    num_cols = min(3, len(detected))
                    cols = st.columns(num_cols)
                    
                    for i, inst_info in enumerate(detected):
                        col_idx = i % num_cols
                        with cols[col_idx]:
                            st.metric(
                                inst_info['instrument'],
                                f"{inst_info['average_confidence']:.2%}",
                                delta=f"Max: {inst_info['max_confidence']:.2%}"
                            )
                else:
                    st.warning(f"⚠️ No instruments detected above threshold ({detection_threshold}). Try lowering the threshold.")
                
                # Top instruments (even if below threshold)
                st.markdown("#### 📈 Top 10 Instruments (by Average Confidence)")
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
    st.title("📊 Advanced Analytics")
    
    if not st.session_state.results:
        st.info("👆 Upload and analyze audio files in the Dashboard to see analytics here.")
    else:
        # File selection
        file_options = [r.get('uploaded_file', r['file']) for r in st.session_state.results]
        selected_file = st.selectbox("Select file for detailed analysis", file_options)
        
        result = next((r for r in st.session_state.results 
                      if r.get('uploaded_file', r['file']) == selected_file), None)
        
        if result:
            st.subheader(f"📈 Comprehensive Analysis: {result.get('uploaded_file', result['file'])}")
            
            # Summary metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Duration", f"{result['duration']:.1f}s")
            with col_m2:
                st.metric("Windows", result['num_windows'])
            with col_m3:
                st.metric("Detected", len(result.get('detected_instruments', [])))
            with col_m4:
                st.metric("Threshold", f"{result['threshold']:.2f}")
            
            st.markdown("---")
            
            # Comprehensive timeline plot
            st.markdown("### 🎨 Comprehensive Timeline Analysis")
            try:
                fig_comp = plot_comprehensive_timeline(result, threshold=detection_threshold)
                st.pyplot(fig_comp)
                plt.close(fig_comp)
            except Exception as e:
                st.error(f"Error generating comprehensive plot: {str(e)}")
            
            st.markdown("---")
            
            # Individual visualizations
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("### ⏱️ Confidence Timeline")
                try:
                    detected_insts = [inst['instrument'] for inst in result.get('detected_instruments', [])]
                    fig_timeline = plot_confidence_timeline(
                        result, 
                        instruments=detected_insts[:10] if detected_insts else None,
                        threshold=detection_threshold
                    )
                    st.pyplot(fig_timeline)
                    plt.close(fig_timeline)
                except Exception as e:
                    st.error(f"Timeline error: {str(e)}")
                
                st.markdown("### 📊 Detection Heatmap")
                try:
                    fig_heatmap = plot_detection_heatmap(result, threshold=detection_threshold)
                    st.pyplot(fig_heatmap)
                    plt.close(fig_heatmap)
                except Exception as e:
                    st.error(f"Heatmap error: {str(e)}")
            
            with col_viz2:
                st.markdown("### 📈 Instrument Summary")
                try:
                    fig_bars = plot_instrument_bars(result, threshold=detection_threshold)
                    st.pyplot(fig_bars)
                    plt.close(fig_bars)
                except Exception as e:
                    st.error(f"Bar chart error: {str(e)}")
                
                # Waveform
                st.markdown("### 🌊 Waveform")
                try:
                    y, sr = librosa.load(result['file_path'], sr=22050, mono=True)
                    fig_wave, ax_wave = plt.subplots(figsize=(10, 3), facecolor='#0E1117')
                    ax_wave.set_facecolor('#0E1117')
                    librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='cyan')
                    ax_wave.set_xlabel('Time (s)', color='white')
                    ax_wave.set_ylabel('Amplitude', color='white')
                    ax_wave.set_title('Audio Waveform', color='white', fontweight='bold')
                    ax_wave.tick_params(colors='white')
                    ax_wave.grid(True, alpha=0.2, color='white')
                    st.pyplot(fig_wave)
                    plt.close(fig_wave)
                except Exception as e:
                    st.error(f"Waveform error: {str(e)}")

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
    
    if not st.session_state.results:
        st.warning("No data to export. Make predictions first.")
    else:
        st.markdown("Export your multilabel detection results in various formats.")
        
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
                "instruet_multilabel_predictions.csv",
                "text/csv",
                key="csv_download"
            )
        else:
            st.info("No detected instruments to export.")
        
        # JSON Export
        st.markdown("### 📋 JSON Export")
        json_data = json.dumps(export_data, indent=2).encode('utf-8')
        st.download_button(
            "⬇️ Download JSON",
            json_data,
            "instruet_multilabel_predictions.json",
            "application/json",
            key="json_download"
        )
        
        # PDF Export
        st.markdown("### 📑 PDF Report")
        if st.button("⬇️ Generate PDF Report"):
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = Paragraph("InstruNet Multilabel Detection Report", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 0.2*inch))
            
            # Summary
            summary_text = f"""
            <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
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
                "instruet_multilabel_report.pdf",
                "application/pdf",
                key="pdf_download"
            )
        
        # Preview
        st.markdown("---")
        st.subheader("👁️ Export Preview")
        if csv_rows:
            st.dataframe(df_csv, use_container_width=True, hide_index=True)

# ================= ABOUT PAGE =================
elif page == "About":
    st.title("ℹ️ About InstruNet")
    
    st.markdown("""
    ### 🎵 InstruNet Multilabel Detection System
    
    **Production-Grade CNN-Based Music Instrument Recognition**
    
    ---
    
    ### 📝 Project Information
    
    - **Project Type**: Multilabel Classification System
    - **Domain**: Music Information Retrieval (MIR)
    - **Technology**: Deep Learning (CNN), Audio Processing
    - **Model**: MobileNetV2-based CNN with transfer learning
    - **Classes**: 28 Musical Instruments
    
    ---
    
    ### 🔬 Technical Architecture
    
    **Model Details:**
    - Base: MobileNetV2 (ImageNet pretrained)
    - Input: Mel-spectrogram images (128×128×3)
    - Output: Sigmoid activation (multilabel)
    - Loss: Binary Cross-Entropy
    - Optimizer: Adam
    
    **Audio Processing:**
    - Sample Rate: 22,050 Hz
    - Window Size: 3 seconds
    - Hop Size: 1 second
    - Features: Mel-spectrogram
    
    **Inference:**
    - Sliding-window approach
    - Timeline-based aggregation
    - Confidence thresholding
    
    ---
    
    ### 🎯 Key Features
    
    1. **True Multilabel Classification**
       - Detects multiple simultaneous instruments
       - Independent predictions for each instrument
       - Sigmoid output (not softmax)
    
    2. **Timeline Support**
       - Sliding-window inference
       - Confidence over time
       - Activity heatmaps
    
    3. **28 Instrument Classes**
       - Comprehensive coverage
       - Weak labeling support
       - CSV-based annotation
    
    4. **Professional UI**
       - Clean, modern interface
       - Advanced visualizations
       - Export capabilities
    
    ---
    
    ### 📚 Usage
    
    **Training:**
    ```bash
    python src/train_multilabel_timeline.py
    ```
    
    **Inference:**
    ```bash
    python src/inference_multilabel_timeline.py <audio_file> [threshold]
    ```
    
    **UI:**
    ```bash
    streamlit run app_multilabel.py
    ```
    
    ---
    
    ### 📖 Dataset Format
    
    **CSV Structure (data_multilabel/labels.csv):**
    ```csv
    file,instruments
    song1.mp3,Violin,Piano,Guitar,Drum_set
    song2.mp3,Bass_Guitar,Drum_set
    bgm.mp3,Keyboard,Synth
    ```
    
    **Rules:**
    - Only list instruments confidently present
    - Use exact class names from class_indices.json
    - Comma-separated list
    
    ---
    
    ### 🎓 Academic Notes
    
    **Multilabel vs Single-Label:**
    - Multilabel: Multiple instruments can be present simultaneously
    - Each instrument prediction is independent
    - Uses sigmoid + binary cross-entropy
    
    **Weak Labeling:**
    - Labels at song level, not window level
    - All windows inherit song-level labels
    - Suitable for polyphonic music
    
    ---
    
    **Developed for: InstruNet Project**
    """)

