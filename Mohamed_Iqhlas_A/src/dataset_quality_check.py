import os
import librosa
import numpy as np

DATASET_PATH = "music_dataset"
MIN_DURATION = 1.0  # seconds

print("🔍 Starting Dataset Quality Check...\n")

for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_path):
        continue

    short_clips = 0
    noisy_clips = 0
    total_files = 0

    for file in os.listdir(class_path):
        if not file.endswith(".wav"):
            continue

        total_files += 1
        file_path = os.path.join(class_path, file)

        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            # Very short clips
            if duration < MIN_DURATION:
                short_clips += 1

            # Noise check (low energy variance)
            energy = np.mean(librosa.feature.rms(y=y))
            if energy < 0.01:
                noisy_clips += 1

        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    print(f"Class: {class_name}")
    print(f"  Total samples: {total_files}")
    print(f"  Very short clips (<1s): {short_clips}")
    print(f"  Low-energy / noisy clips: {noisy_clips}\n")

print("✅ Dataset quality analysis completed.")
