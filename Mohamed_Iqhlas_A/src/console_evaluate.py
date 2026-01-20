import os
from inference import predict_instrument

TEST_AUDIO_DIR = "test_audio"

total = 0
correct = 0

per_class_total = {}
per_class_correct = {}

print("\n🔍 Running console evaluation...\n")

for true_label in os.listdir(TEST_AUDIO_DIR):
    class_dir = os.path.join(TEST_AUDIO_DIR, true_label)

    if not os.path.isdir(class_dir):
        continue

    per_class_total.setdefault(true_label, 0)
    per_class_correct.setdefault(true_label, 0)

    for file in os.listdir(class_dir):
        if not file.lower().endswith(".wav"):
            continue

        audio_path = os.path.join(class_dir, file)

        result = predict_instrument(audio_path)
        predicted_label = result["instrument"]
        confidence = result["confidence"]

        total += 1
        per_class_total[true_label] += 1

        if predicted_label == true_label:
            correct += 1
            per_class_correct[true_label] += 1
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG"

        print(
            f"{status} | True: {true_label:12s} | "
            f"Predicted: {predicted_label:12s} | "
            f"Conf: {confidence:.2f} | File: {file}"
        )

print("\n================ SUMMARY ================\n")

overall_acc = correct / total if total > 0 else 0
print(f"Overall Accuracy: {overall_acc*100:.2f}% ({correct}/{total})\n")

print("Per-Class Accuracy:")
for cls in per_class_total:
    cls_acc = (
        per_class_correct[cls] / per_class_total[cls]
        if per_class_total[cls] > 0 else 0
    )
    print(
        f"{cls:12s}: {cls_acc*100:.2f}% "
        f"({per_class_correct[cls]}/{per_class_total[cls]})"
    )
