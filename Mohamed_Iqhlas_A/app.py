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

from src.inference import predict
from src.visualizations import (
    create_enhanced_waveform,
    create_time_frequency_plot,
    create_harmonic_analysis,
    create_instrument_timeline,
    create_confidence_distribution,
    create_comprehensive_analysis_plot
)

# Page configuration
st.set_page_config(
    page_title="InstruNet - Music Instrument Recognition",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .confidence-high {
        color: #00C853;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFB300;
        font-weight: bold;
    }
    .confidence-low {
        color: #FF5252;
        font-weight: bold;
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

# Sidebar navigation
with st.sidebar:
    # External placeholder images can fail on some networks and show a broken "0" icon.
    # Use a local/text header instead (fixes the broken icon).
    st.markdown("## 🎵 InstruNet AI ")
    st.caption("CNN-Based Music Instrument Recognition")
    st.markdown("---")
    page = st.radio(
        "📌 Navigation",
        ["Home", "Dashboard", "Analytics", "History", "Export", "About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 🎯 Quick Stats")
    if st.session_state.results:
        st.metric("Total Predictions", len(st.session_state.results))
        avg_confidence = np.mean([r.get('confidence', 0) for r in st.session_state.results])
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")

# ================= HOME PAGE =================
if page == "Home":
    st.markdown('<div class="main-header">🎵 InstruNet AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">CNN-Based Music Instrument Recognition System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🎯 High Accuracy
        **95.14%** validation accuracy
        
        Trained on **28 instrument classes** with excellent generalization
        """)
    with col2:
        st.markdown("""
        ### 📊 Advanced Analysis
        Top-K confidence ranking
        
        Confidence analysis & uncertainty quantification
        """)
    with col3:
        st.markdown("""
        ### 🎨 Rich Visualizations
        Enhanced waveforms
        
        Time-frequency analysis & instrument timelines
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🔍 What is InstruNet?
    
    InstruNet is an intelligent audio analysis system that automatically identifies **musical instruments** 
    from raw audio using **Convolutional Neural Networks (CNNs)** trained on spectrogram images.
    
    ### ⚙️ System Pipeline
    
    ```
    Audio Input → Spectrogram Conversion → CNN Model → Instrument Prediction → Analysis & Visualization
    ```
    
    ### ✨ Key Features
    
    - 🎯 **Accurate Classification**: 95%+ accuracy on 28 instrument classes
    - 📊 **Top-K Confidence Ranking**: See alternative predictions ranked by confidence (not multilabel)
    - 🎨 **Advanced Visualizations**: 3D-style waveforms, time-frequency plots, harmonic analysis
    - 📈 **Confidence Analysis**: Understand prediction uncertainty
    - 📁 **Batch Processing**: Analyze multiple audio files simultaneously
    - 📤 **Professional Export**: JSON, CSV, and PDF reports
    
    ### 🧪 Technologies
    
    - **Python** - Core language
    - **Librosa** - Audio processing and feature extraction
    - **TensorFlow/Keras** - Deep learning framework
    - **Streamlit** - Interactive web interface
    - **Matplotlib** - Advanced visualizations
    
    ### 🏢 Project Context
    
    This project demonstrates **real-world AI application development** combining:
    - Signal processing and audio analysis
    - Deep learning and CNN architectures
    - User interface design and visualization
    - Model evaluation and performance analysis
    
    👉 **Navigate to Dashboard** to start predicting instruments!
    """)

# ================= DASHBOARD PAGE =================
elif page == "Dashboard":
    st.title("🎯 Instrument Prediction Dashboard")
    st.markdown("Upload audio files to identify musical instruments using our CNN model.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "📁 Upload Audio Files (WAV / MP3)",
        type=["wav", "mp3"],
        accept_multiple_files=True,
        help="Select one or more audio files for instrument recognition"
    )

    # Immediate audio preview (play BEFORE prediction)
    if uploaded_files:
        st.markdown("### 🎵 Audio Preview (Before Prediction)")
        for f in uploaded_files:
            st.markdown(f"**{f.name}**")
            try:
                st.audio(f.getvalue())
            except Exception:
                st.warning("Audio preview not available for this file.")
    
    # Prediction settings
    col_settings1, col_settings2 = st.columns(2)
    with col_settings1:
        top_k = st.slider("Top-K Predictions", min_value=3, max_value=10, value=5, 
                         help="Number of alternative predictions to display")
    with col_settings2:
        show_analysis = st.checkbox("Show Detailed Analysis", value=True,
                                   help="Display confidence analysis and similarity groups")
    
    # Predict button
    if st.button("🔍 Predict Instruments", type="primary", use_container_width=True) and uploaded_files:
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
                    result = predict(path, top_k=top_k)
                    result["file"] = file.name
                    # Store temp path for playback in UI
                    result["audio_path"] = path
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
        st.subheader("📊 Prediction Results")
        
        for idx, result in enumerate(st.session_state.results):
            # Prediction card
            with st.container():
                col_header1, col_header2 = st.columns([3, 1])
                with col_header1:
                    st.markdown(f"### 🎧 {result['file']}")

                    try:
                        audio_path = result.get("audio_path")

                        if audio_path:
                            st.audio(audio_path)
                    except Exception as e:
                        st.warning("Audio playback not available")
                    


                with col_header2:
                    # Confidence badge
                    conf = result.get('confidence', 0)
                    if conf >= 0.7:
                        st.markdown(f'<p class="confidence-high">High Confidence</p>', unsafe_allow_html=True)
                    elif conf >= 0.5:
                        st.markdown(f'<p class="confidence-medium">Medium Confidence</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="confidence-low">Low Confidence</p>', unsafe_allow_html=True)
                
                # Primary prediction
                primary = result.get('predictions', {}).get('primary', {})
                instrument = primary.get('instrument', result.get('instrument', 'Unknown'))
                confidence = primary.get('confidence', result.get('confidence', 0))
                
                col_pred1, col_pred2, col_pred3 = st.columns([2, 2, 2])
                with col_pred1:
                    st.metric("🎵 Predicted Instrument", instrument)
                with col_pred2:
                    st.metric("📈 Confidence", f"{confidence:.2%}")
                    st.progress(confidence)
                with col_pred3:
                    analysis = result.get('analysis', {})
                    uncertainty = analysis.get('uncertainty_score', 0)
                    st.metric("❓ Uncertainty", f"{uncertainty:.2%}")
                
                # Analysis information
                if show_analysis:
                    analysis = result.get('analysis', {})
                    conf_level = analysis.get('confidence_level', 'unknown')
                    conf_msg = analysis.get('confidence_message', '')
                    sim_group = analysis.get('similarity_group', '')
                    
                    if conf_msg:
                        if conf_level == 'high':
                            st.success(f"💡 {conf_msg}")
                        elif conf_level == 'medium':
                            st.warning(f"💡 {conf_msg}")
                        else:
                            st.error(f"💡 {conf_msg}")
                    
                    if sim_group:
                        similar = analysis.get('similar_instruments', [])
                        if similar:
                            st.info(f"🔎 Similarity Group: **{sim_group}** - {', '.join(similar)}")
                
                # Top-K alternatives (NOT multilabel - these are ranked alternatives)
                top_k_preds = result.get('predictions', {}).get('top_k', [])
                if len(top_k_preds) > 1:
                    with st.expander(f"🔍 Top-{len(top_k_preds)} Confidence Ranking (Alternative Predictions)", expanded=False):
                        st.caption("💡 These represent the model's confidence ranking for alternative predictions. This is NOT multilabel classification - these are ranked alternatives, not simultaneously detected instruments.")
                        alt_df = pd.DataFrame([
                            {
                                'Rank': f"#{p.get('rank', i+1)}",
                                'Instrument': p.get('instrument', 'Unknown'),
                                'Confidence': f"{p.get('confidence', 0):.2%}"
                            }
                            for i, p in enumerate(top_k_preds[1:])  # Exclude primary
                        ])
                        st.dataframe(alt_df, use_container_width=True, hide_index=True)
                
                # Audio features
                audio_features = result.get('audio_features', {})
                if audio_features:
                    with st.expander("🎛️ Audio Features", expanded=False):
                        feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
                        with feat_col1:
                            st.metric("Energy", f"{audio_features.get('energy', 0):.3f}")
                        with feat_col2:
                            st.metric("Brightness", f"{audio_features.get('brightness', 0):.0f} Hz")
                        with feat_col3:
                            st.metric("ZCR", f"{audio_features.get('zero_crossing_rate', 0):.4f}")
                        with feat_col4:
                            st.metric("Duration", f"{audio_features.get('duration', 0):.2f}s")
                
                st.markdown("---")

# ================= ANALYTICS PAGE =================
elif page == "Analytics":
    st.title("📊 Advanced Audio Analytics")
    
    if not st.session_state.results:
        st.info("👆 Upload and predict audio files in the Dashboard to see analytics here.")
    else:
        # Select file for detailed analysis
        file_options = [r['file'] for r in st.session_state.results]
        selected_file = st.selectbox("Select file for detailed analysis", file_options)
        
        result = next((r for r in st.session_state.results if r['file'] == selected_file), None)
        
        if result:
            st.subheader(f"📈 Comprehensive Analysis: {result['file']}")
            
            # Comprehensive plot is heavy → keep it OPTIONAL (reduces time massively)
            show_comp = st.checkbox("Show Complete Analysis Dashboard (slower)", value=False)
            if show_comp:
                st.markdown("### 🎨 Complete Analysis Dashboard")
                try:
                    fig_comprehensive = create_comprehensive_analysis_plot(result)
                    st.pyplot(fig_comprehensive)
                    plt.close(fig_comprehensive)
                except Exception as e:
                    st.warning(f"Could not generate comprehensive plot: {str(e)}")
            
            st.markdown("---")
            
            # Individual visualizations
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("### 🌊 Enhanced Waveform (Fast)")
                try:
                    fig_wave = create_enhanced_waveform(result['wave'], result['sr'])
                    st.pyplot(fig_wave)
                    plt.close(fig_wave)
                except Exception as e:
                    st.error(f"Waveform error: {str(e)}")
                
                st.markdown("### 🎵 Harmonic Analysis")
                try:
                    fig_harm = create_harmonic_analysis(result['wave'], result['sr'])
                    st.pyplot(fig_harm)
                    plt.close(fig_harm)
                except Exception as e:
                    st.error(f"Harmonic analysis error: {str(e)}")
            
            with col_viz2:
                st.markdown("### 📊 Time-Frequency Analysis")
                try:
                    fig_tf = create_time_frequency_plot(result['mel'], result['sr'])
                    st.pyplot(fig_tf)
                    plt.close(fig_tf)
                except Exception as e:
                    st.error(f"Time-frequency error: {str(e)}")
                
                st.markdown("### 📈 Confidence Distribution")
                try:
                    all_preds = result.get('predictions', {}).get('all_predictions', {})
                    if all_preds:
                        fig_conf = create_confidence_distribution(all_preds)
                        st.pyplot(fig_conf)
                        plt.close(fig_conf)
                except Exception as e:
                    st.error(f"Confidence distribution error: {str(e)}")
            
            # Instrument timeline
            st.markdown("### ⏱️ Instrument Timeline")
            try:
                fig_timeline = create_instrument_timeline(result)
                st.pyplot(fig_timeline)
                plt.close(fig_timeline)
            except Exception as e:
                st.error(f"Timeline error: {str(e)}")

# ================= HISTORY PAGE =================
elif page == "History":
    st.title("🕘 Prediction History")
    
    if not st.session_state.results:
        st.info("No predictions yet. Go to Dashboard to make predictions.")
    else:
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Total Predictions", len(st.session_state.results))
        with col_stat2:
            avg_conf = np.mean([r.get('confidence', 0) for r in st.session_state.results])
            st.metric("Average Confidence", f"{avg_conf:.2%}")
        with col_stat3:
            high_conf_count = sum(1 for r in st.session_state.results if r.get('confidence', 0) >= 0.7)
            st.metric("High Confidence", high_conf_count)
        with col_stat4:
            unique_instruments = len(set(r.get('instrument', 'Unknown') for r in st.session_state.results))
            st.metric("Unique Instruments", unique_instruments)
        
        st.markdown("---")
        
        # History table
        st.subheader("📋 Detailed History")
        history_data = []
        for r in st.session_state.results:
            primary = r.get('predictions', {}).get('primary', {})
            analysis = r.get('analysis', {})
            history_data.append({
                "File": r['file'],
                "Instrument": primary.get('instrument', r.get('instrument', 'Unknown')),
                "Confidence": f"{primary.get('confidence', r.get('confidence', 0)):.2%}",
                "Confidence Level": analysis.get('confidence_level', 'unknown').title(),
                "Uncertainty": f"{analysis.get('uncertainty_score', 0):.2%}",
                "Similarity Group": analysis.get('similarity_group', 'N/A'),
                "Timestamp": r.get('timestamp', 'N/A')[:19] if r.get('timestamp') else 'N/A'
            })
        
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
        
        # Instrument distribution
        st.markdown("---")
        st.subheader("📊 Instrument Distribution")
        instrument_counts = pd.Series([r.get('instrument', 'Unknown') for r in st.session_state.results]).value_counts()
        st.bar_chart(instrument_counts)

# ================= EXPORT PAGE =================
elif page == "Export":
    st.title("📤 Export Results")
    
    if not st.session_state.results:
        st.warning("No data to export. Make predictions first.")
    else:
        st.markdown("Export your prediction results in various formats.")
        
        # Prepare enhanced data
        export_data = []
        for r in st.session_state.results:
            primary = r.get('predictions', {}).get('primary', {})
            analysis = r.get('analysis', {})
            audio_features = r.get('audio_features', {})
            
            export_data.append({
                "file": r['file'],
                "timestamp": r.get('timestamp', datetime.now().isoformat()),
                "primary_instrument": primary.get('instrument', r.get('instrument', 'Unknown')),
                "primary_confidence": primary.get('confidence', r.get('confidence', 0)),
                "confidence_level": analysis.get('confidence_level', 'unknown'),
                "uncertainty_score": analysis.get('uncertainty_score', 0),
                "similarity_group": analysis.get('similarity_group', ''),
                "energy": audio_features.get('energy', 0),
                "brightness": audio_features.get('brightness', 0),
                "duration": audio_features.get('duration', 0),
                "alternatives": json.dumps([p.get('instrument') for p in r.get('predictions', {}).get('alternatives', [])])
            })
        
        # CSV Export
        st.markdown("### 📄 CSV Export")
        df_csv = pd.DataFrame(export_data)
        csv_data = df_csv.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download CSV",
            csv_data,
            "instruet_predictions.csv",
            "text/csv",
            key="csv_download"
        )
        
        # JSON Export (enhanced)
        st.markdown("### 📋 JSON Export (Enhanced)")
        json_data = json.dumps(export_data, indent=2).encode('utf-8')
        st.download_button(
            "⬇️ Download JSON",
            json_data,
            "instruet_predictions.json",
            "application/json",
            key="json_download"
        )
        
        # PDF Export (enhanced)
        st.markdown("### 📑 PDF Report")
        include_plots = st.checkbox("Include Analytics Plots + History in PDF (slower)", value=True)
        if st.button("⬇️ Generate PDF Report"):
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = Paragraph("InstruNet Prediction Report", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 0.2*inch))
            
            # Summary
            summary_text = f"""
            <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <b>Total Predictions:</b> {len(export_data)}<br/>
            <b>Average Confidence:</b> {np.mean([d['primary_confidence'] for d in export_data]):.2%}
            """
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 0.3*inch))

            # ---------------- Prediction History (MANDATORY) ----------------
            story.append(Paragraph("Prediction History", styles['Heading2']))
            history_table_data = [['File', 'Instrument', 'Confidence', 'Timestamp']]
            for r in st.session_state.results:
                primary = r.get('predictions', {}).get('primary', {})
                inst = primary.get('instrument', r.get('instrument', 'Unknown'))
                conf = primary.get('confidence', r.get('confidence', 0.0))
                ts = r.get('timestamp', '')
                history_table_data.append([
                    str(r.get('file', ''))[:35],
                    str(inst)[:25],
                    f"{conf:.2%}",
                    str(ts)[:19].replace('T', ' ')
                ])

            history_table = Table(history_table_data, colWidths=[2.2*inch, 2.0*inch, 1.2*inch, 1.6*inch])
            history_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            story.append(history_table)
            story.append(Spacer(1, 0.25*inch))
            
            # Table data
            table_data = [['File', 'Instrument', 'Confidence', 'Level']]
            for d in export_data:
                table_data.append([
                    d['file'][:30] + '...' if len(d['file']) > 30 else d['file'],
                    d['primary_instrument'],
                    f"{d['primary_confidence']:.2%}",
                    d['confidence_level'].title()
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
            story.append(Spacer(1, 0.25*inch))

            # ---------------- Analytics Plots (Waveforms & Graphs) ----------------
            if include_plots:
                story.append(Paragraph("Analytics (Waveforms & Graphs)", styles['Heading2']))

                def fig_to_rl_image(fig, width=7.0*inch, height=3.8*inch):
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', dpi=140, bbox_inches='tight')
                    plt.close(fig)
                    img_buf.seek(0)
                    return Image(img_buf, width=width, height=height)

                for r in st.session_state.results:
                    story.append(Spacer(1, 0.15*inch))
                    story.append(Paragraph(f"File: {r.get('file','')}", styles['Heading3']))

                    # Enhanced waveform (now fast, downsampled)
                    fig_wave = create_enhanced_waveform(r.get('wave'), r.get('sr'))
                    story.append(fig_to_rl_image(fig_wave))
                    story.append(Spacer(1, 0.10*inch))

                    # Time-frequency (mel + intensity)
                    fig_tf = create_time_frequency_plot(r.get('mel'), r.get('sr'))
                    story.append(fig_to_rl_image(fig_tf, height=4.6*inch))
                    story.append(Spacer(1, 0.10*inch))

                    # Harmonic analysis
                    fig_h = create_harmonic_analysis(r.get('wave'), r.get('sr'))
                    story.append(fig_to_rl_image(fig_h, height=4.6*inch))
                    story.append(Spacer(1, 0.10*inch))

                    # Instrument timeline
                    fig_tl = create_instrument_timeline(r)
                    story.append(fig_to_rl_image(fig_tl))
                    story.append(Spacer(1, 0.10*inch))

                    # Confidence distribution
                    all_preds = r.get('predictions', {}).get('all_predictions', {})
                    if all_preds:
                        fig_cd = create_confidence_distribution(all_preds)
                        story.append(fig_to_rl_image(fig_cd))
                        story.append(Spacer(1, 0.10*inch))

                    # Comprehensive dashboard (heavy but included because requested)
                    try:
                        fig_comp = create_comprehensive_analysis_plot(r)
                        story.append(fig_to_rl_image(fig_comp, height=4.6*inch))
                    except Exception:
                        pass
                    story.append(Spacer(1, 0.20*inch))
            
            doc.build(story)
            buffer.seek(0)
            
            st.download_button(
                "⬇️ Download PDF",
                buffer,
                "instruet_report.pdf",
                "application/pdf",
                key="pdf_download"
            )
        
        # Preview
        st.markdown("---")
        st.subheader("👁️ Export Preview")
        st.dataframe(df_csv, use_container_width=True, hide_index=True)

# ================= ABOUT PAGE =================
elif page == "About":
    st.title("ℹ️ About InstruNet")
    
    st.markdown("""
    ### 🎵 InstruNet AI
    
    **CNN-Based Music Instrument Recognition System**
    
    ---
    
    ### 📝 Project Information
    
    - **Project Type**: Infosys Internship Project
    - **Domain**: Music Information Retrieval (MIR)
    - **Technology**: Deep Learning (CNN), Audio Processing
    - **Model Accuracy**: 95.14% (Validation Set)
    - **Classes**: 28 Musical Instruments
    
    ---
    
    ### 👨‍💻 Developer
    
    **Mohamed Iqhlas A**
    
    ---
    
    ### 🔬 Technical Details
    
    **Model Architecture:**
    - Base: MobileNetV2 (Transfer Learning)
    - Input: Mel-spectrogram images (128x128)
    - Output: 28-class instrument classification
    - Loss: Categorical Cross-Entropy
    - Optimizer: Adam
    
    **Audio Processing:**
    - Sample Rate: 22,050 Hz
    - Duration: 3 seconds per clip
    - Features: Mel-spectrogram, RMS Energy, Spectral Centroid
    
    **Dataset:**
    - Training Samples: ~27,000
    - Validation Samples: ~15,000
    - Classes: 28 instruments
    
    ---
    
    ### 📊 Performance Metrics
    
    - **Training Accuracy**: 95.17%
    - **Validation Accuracy**: 95.14%
    - **Accuracy Gap**: 0.03% (Excellent generalization)
    - **Best Performing Classes**: Drum_set, Bass_Guitar, Harmonium, Organ, Keyboard, Horn, cowbell, flute (F1 > 0.99)
    
    ---
    
    ### 🎯 Key Features
    
    1. **High Accuracy**: 95%+ classification accuracy
    2. **Top-K Predictions**: See alternative predictions with confidence
    3. **Confidence Analysis**: Understand prediction uncertainty
    4. **Advanced Visualizations**: Enhanced waveforms, time-frequency plots
    5. **Batch Processing**: Analyze multiple files simultaneously
    6. **Professional Export**: JSON, CSV, PDF reports
    
    ---
    
    ### 📚 Limitations & Future Work
    
    **Current Limitations:**
    - Single-instrument classification (not polyphonic)
    - 3-second clip analysis
    - Some confusion between similar instruments (e.g., Trumpet/Trombone)
    - Dataset imbalance for rare classes
    
    **Future Improvements:**
    - Polyphonic music support (true multilabel)
    - Longer temporal context (5-10 seconds)
    - Attention mechanisms for fine-grained discrimination
    - Data augmentation for rare classes
    - Real-time audio streaming
    
    ---
    
    ### 📖 References
    
    - Librosa: Audio and music analysis library
    - TensorFlow/Keras: Deep learning framework
    - MobileNetV2: Efficient CNN architecture
    - Mel-spectrogram: Time-frequency representation
    
    ---
    
    **For questions or feedback, please contact the developer.**
    """)
