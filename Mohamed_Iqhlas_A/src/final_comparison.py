baseline = {
    "accuracy": 0.95,
    "macro_f1": 0.92,
    "weighted_f1": 0.95
}

final = {
    "accuracy": 0.98,
    "macro_f1": 0.9512,
    "weighted_f1": 0.9749
}

print("\n📊 BASELINE vs FINAL MODEL COMPARISON\n")
print("{:<20} {:<15} {:<15}".format("Metric", "Baseline", "Final"))
print("-" * 50)

for key in baseline:
    print("{:<20} {:<15} {:<15}".format(
        key,
        baseline[key],
        final[key]
    ))