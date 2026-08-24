import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. LOAD YOUR EXACT STAGE 2 MODEL CHECKPOINT
# ------------------------------------------------------------------
MODEL_PATH = r"D:\AJ\sem 6th\Minor_Project\Software_part\Webapp\Backend\best_model_stage2_partial.keras"
TEST_DIR = r"Original_Data\test"
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------------------------------------------
# 2. LOAD YOUR TEST SET
# ------------------------------------------------------------------
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False,
    label_mode='int'
)

class_names = test_dataset.class_names
print("Class order:", class_names, flush=True)

# ------------------------------------------------------------------
# 3. RUN INFERENCE ON THE FULL TEST SET (single batched call, with progress bar)
# ------------------------------------------------------------------
print("Collecting true labels...", flush=True)
y_true = np.concatenate([labels.numpy() for images, labels in test_dataset], axis=0)
print(f"Loaded {len(y_true)} true labels.", flush=True)

print("Running model.predict() on test set (this may take a few minutes on CPU)...", flush=True)
preds = model.predict(test_dataset, verbose=1)

y_pred = np.argmax(preds, axis=1)
y_confidence = np.max(preds, axis=1)
print("Inference complete.", flush=True)

# ------------------------------------------------------------------
# 4. VERIFY ACCURACY MATCHES YOUR REPORTED 99.56%
# ------------------------------------------------------------------
total_images = len(y_true)
correct = np.sum(y_true == y_pred)
misclassified = total_images - correct
accuracy = correct / total_images * 100

print(f"\nTotal test images: {total_images}")
print(f"Correct predictions: {correct}")
print(f"Misclassified: {misclassified}")
print(f"Accuracy: {accuracy:.4f}%")
print("(This should match your reported 99.56% Stage 2 accuracy. "
      "If it shows 100.00%, this is NOT the same checkpoint/test-set "
      "that produced your reported figure.)")

# ------------------------------------------------------------------
# 5. GENERATE AND SAVE THE CONFUSION MATRIX
# ------------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
print(cm)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - EfficientNetV2-S (Stage 2)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("\nSaved confusion_matrix.png", flush=True)
plt.show()

# ------------------------------------------------------------------
# 6. PER-CLASS METRICS (Precision, Recall, F1) FOR YOUR LATEX TABLE
# ------------------------------------------------------------------
print("\nPer-Class Metrics:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# ------------------------------------------------------------------
# 7. CALCULATE SPECIFICITY PER CLASS (not included in sklearn's report)
# ------------------------------------------------------------------
print("\nSpecificity per class:")
for i, class_name in enumerate(class_names):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - tp - fn - fp
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"{class_name}: Specificity = {specificity*100:.2f}%")

# ------------------------------------------------------------------
# 8. FIND ACTUAL MISCLASSIFIED IMAGES (for your misclassification table)
# ------------------------------------------------------------------
misclassified_idx = np.where(y_true != y_pred)[0]
print(f"\n{len(misclassified_idx)} misclassified images found:")

try:
    file_paths = test_dataset.file_paths
    for idx in misclassified_idx:
        print(f"File: {file_paths[idx]}")
        print(f"  Actual: {class_names[y_true[idx]]}, "
              f"Predicted: {class_names[y_pred[idx]]}, "
              f"Confidence: {y_confidence[idx]*100:.2f}%")
except AttributeError:
    print("File paths not available — inspect misclassified_idx manually.")