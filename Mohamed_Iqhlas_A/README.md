# CNN-Based Music Instrument Recognition System

## Project Overview
This project implements a CNN-based system to recognize musical instruments from audio signals.
Audio inputs are converted into Mel-spectrograms and classified using a deep learning model.

## System Architecture
1. Audio Input (WAV/MP3)
2. Audio Preprocessing (Normalization, Mel-Spectrogram)
3. CNN Model for Instrument Classification
4. Prediction & Confidence Score
5. Streamlit-based User Interface

## Technologies Used
- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Matplotlib
- Streamlit

## Folder Structure
Mohamed_Iqhlas_A/
│── src/
│ ├── train_model.py
│ ├── inference.py
│ ├── generate_spectrograms.py
│ ├── console_evaluate.py
│ └── ...
│── app.py
│── instrument_classifier.keras
│── class_indices.json
│── classification_report.txt
│── final_metrics.txt
│── training_history.npy
│── training_history.pkl
│── README.md

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
### 2. Inference
python src/inference.py test_audio/Piano/1.wav
```bash
### 3.Run Streamlit App
streamlit run app.py
```bash


Output

Predicted Instrument

Confidence Score

Music Style Recommendation

Notes

Model predictions may vary due to similarity between instruments.

The focus of this project is on correct ML pipeline and system design.

Developed as part of Infosys Internship Program.

✅ Save the file.

---

## STEP 2️⃣ Clean cache folders (OPTIONAL but good)

Delete all `__pycache__` folders.

---

## STEP 3️⃣ Git: Stage files properly

Open terminal inside **Demo** folder:

```bash
cd Demo
git checkout mohamed-iqhlas-submission





