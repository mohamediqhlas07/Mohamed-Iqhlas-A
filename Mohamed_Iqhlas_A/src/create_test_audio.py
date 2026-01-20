import librosa
import soundfile as sf
import os
import sys

def split_audio(
    input_audio,
    output_dir="test_audio",
    num_clips=5,
    clip_duration=3  # seconds
):
    os.makedirs(output_dir, exist_ok=True)

    y, sr = librosa.load(input_audio, sr=22050, mono=True)

    samples_per_clip = clip_duration * sr

    for i in range(num_clips):
        start = i * samples_per_clip
        end = start + samples_per_clip

        if end > len(y):
            break

        clip = y[start:end]
        out_path = os.path.join(output_dir, f"clip_{i+1}.wav")
        sf.write(out_path, clip, sr)

        print(f"Saved: {out_path}")

if __name__ == "__main__":
    input_audio = sys.argv[1]
    split_audio(input_audio)
