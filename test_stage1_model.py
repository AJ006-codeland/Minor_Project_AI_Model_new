import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.efficientnet_v2 import preprocess_input

# ==========================================
# CONFIG (must match what stage 1 was trained at!)
# ==========================================

DATASET_PATH = "Original_Data"
TEST_DIR = os.path.join(DATASET_PATH, "test")

IMG_SIZE = 224   # matches your stage 1 checkpoint
BATCH_SIZE = 32

CHECKPOINT = "best_model_stage1.keras"

# ==========================================
# LOAD TEST DATASET
# ==========================================

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,   # keep order so predictions line up with true labels
)

CLASS_NAMES = test_ds.class_names
print("\nClasses:", CLASS_NAMES)

AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = preprocess_input(image)
    return image, label

test_ds_processed = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ==========================================
# LOAD MODEL
# ==========================================

print(f"\nLoading checkpoint: {CHECKPOINT}\n")
model = keras.models.load_model(CHECKPOINT)

# ==========================================
# OVERALL TEST ACCURACY / LOSS
# ==========================================

loss, accuracy = model.evaluate(test_ds_processed)
print(f"\nOverall Test Loss: {loss:.4f}")
print(f"Overall Test Accuracy: {accuracy:.4f}\n")

# ==========================================
# PER-CLASS METRICS (precision / recall / f1 + confusion matrix)
# ==========================================

print("Running predictions for detailed per-class report...\n")

y_true = []
y_pred = []

for images, labels in test_ds_processed:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion matrix (rows = true class, columns = predicted class)
num_classes = len(CLASS_NAMES)
confusion = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    confusion[t, p] += 1

print("Confusion Matrix (rows=true, columns=predicted):")
header = "                " + "".join(f"{name[:12]:>14}" for name in CLASS_NAMES)
print(header)
for i, name in enumerate(CLASS_NAMES):
    row = "".join(f"{confusion[i, j]:>14}" for j in range(num_classes))
    print(f"{name[:15]:>15} {row}")

print()

# Precision / Recall / F1 per class
print(f"{'Class':<15}{'Precision':>12}{'Recall':>12}{'F1-score':>12}{'Support':>10}")
for i, name in enumerate(CLASS_NAMES):
    tp = confusion[i, i]
    fp = confusion[:, i].sum() - tp
    fn = confusion[i, :].sum() - tp
    support = confusion[i, :].sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"{name:<15}{precision:>12.4f}{recall:>12.4f}{f1:>12.4f}{support:>10}")

print(f"\nTotal test samples: {len(y_true)}")