import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ---------------- CONFIG ----------------
AUDIO_DATASET_PATH = "music_dataset_grouped"
OUTPUT_PATH = "spectrogram_dataset_grouped"
   # output spectrogram images
SAMPLE_RATE = 22050
DURATION = 3        # seconds
N_MELS = 128

# ----------------------------------------

os.makedirs(OUTPUT_PATH, exist_ok=True)

def generate_mel_spectrogram(audio_path, save_path):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(mel_spec_db, sr=sr, cmap="magma")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")


# ---------------- MAIN ----------------
print("🎵 Starting spectrogram generation...")

for instrument in os.listdir(AUDIO_DATASET_PATH):
    instrument_path = os.path.join(AUDIO_DATASET_PATH, instrument)

    if not os.path.isdir(instrument_path):
        continue

    output_instrument_path = os.path.join(OUTPUT_PATH, instrument)
    os.makedirs(output_instrument_path, exist_ok=True)

    for file in os.listdir(instrument_path):
        if file.endswith(".wav") or file.endswith(".mp3"):
            audio_file = os.path.join(instrument_path, file)
            image_file = file.replace(".wav", ".png").replace(".mp3", ".png")
            save_path = os.path.join(output_instrument_path, image_file)

            generate_mel_spectrogram(audio_file, save_path)

    print(f"✅ Done: {instrument}")
