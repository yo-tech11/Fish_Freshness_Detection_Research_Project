from pathlib import Path
import numpy as np
import tensorflow as tf

# Model and dataset paths
MODEL_PATH = Path("models/fish_quality.keras")
TEST_DIR = Path("data/split/test")

# Check model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train.py first."
    )

# Check test dataset
if not TEST_DIR.exists():
    raise FileNotFoundError("Test dataset not found.")

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

# Load test images
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=16,
    shuffle=False,
    label_mode="binary",
    class_names=["fresh", "spoiled"],
)

# Actual labels
y_true = np.concatenate([
    labels.numpy().ravel()
    for _, labels in test_ds
]).astype(int)

# Predictions
print("\nEvaluating model...")

predictions = model.predict(test_ds, verbose=1).ravel()

y_pred = (predictions >= 0.5).astype(int)

# Calculate metrics manually
TP = np.sum((y_true == 1) & (y_pred == 1))
TN = np.sum((y_true == 0) & (y_pred == 0))
FP = np.sum((y_true == 0) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))

def safe_divide(a, b):
    return a / b if b != 0 else 0.0

accuracy = safe_divide(TP + TN, TP + TN + FP + FN)
precision = safe_divide(TP, TP + FP)
recall = safe_divide(TP, TP + FN)

f1_score = safe_divide(
    2 * precision * recall,
    precision + recall
)

# Display results
print("\n========== MODEL EVALUATION ==========")

print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1_score:.4f}")

print("\nConfusion Matrix")
print("----------------")
print("                 Predicted")
print("                 Fresh  Spoiled")
print(f"Actual Fresh      {TN:4d}   {FP:4d}")
print(f"Actual Spoiled    {FN:4d}   {TP:4d}")

print("\nEvaluation completed successfully!")