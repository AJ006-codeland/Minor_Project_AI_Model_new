import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications.efficientnet_v2 import preprocess_input

# ==========================================
# CONFIG (IMG_SIZE must match what stage 1 was trained at!)
# ==========================================

DATASET_PATH = "Original_Data"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "Validate")
TEST_DIR = os.path.join(DATASET_PATH, "test")

IMG_SIZE = 224   # <-- matches your actual stage 1 checkpoint
BATCH_SIZE = 32
EPOCHS_STAGE2 = 10
# How many layers (counting from the END of the backbone) to unfreeze.
# Higher = more compute per step, but more of the backbone adapts.
# 20-30 is a reasonable starting point for a CPU run.
NUM_LAYERS_TO_UNFREEZE = 25

STAGE1_CHECKPOINT = "best_model_stage1.keras"
BEST_MODEL_STAGE2 = "best_model_stage2_partial.keras"
MODEL_NAME = "efficientnetv2_grape_final.keras"

# ==========================================
# GPU CHECK
# ==========================================

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"\nGPU detected: {gpus}\n")
else:
    print("\nNo GPU detected — running on CPU.\n")

# ==========================================
# LOAD DATASET
# ==========================================

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

CLASS_NAMES = train_ds.class_names
print("\nClasses:", CLASS_NAMES)

AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ==========================================
# LOAD STAGE 1 CHECKPOINT
# ==========================================

print(f"\nLoading checkpoint: {STAGE1_CHECKPOINT}\n")
model = keras.models.load_model(STAGE1_CHECKPOINT)
model.summary()

# ==========================================
# FIND THE BASE MODEL LAYER
# ==========================================

base_model = None
for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        base_model = layer
        break

if base_model is None:
    raise RuntimeError(
        "Could not find the EfficientNetV2 base model layer inside the loaded model. "
        "Print [l.name for l in model.layers] to find the correct layer name."
    )

print(f"\nFound base model layer: {base_model.name}")
print(f"Base model has {len(base_model.layers)} internal layers\n")

# ==========================================
# PARTIAL UNFREEZE
# ==========================================
# Freeze everything first, then unfreeze only the last N layers.
# BatchNorm layers are kept frozen regardless, since retraining BN
# stats on a small dataset with few epochs tends to hurt more than help.

base_model.trainable = True

freeze_until = len(base_model.layers) - NUM_LAYERS_TO_UNFREEZE

for i, layer in enumerate(base_model.layers):
    if i < freeze_until:
        layer.trainable = False
    elif isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Unfroze {trainable_count} / {len(base_model.layers)} backbone layers\n")

# ==========================================
# COMPILE FOR PARTIAL FINE-TUNING
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_stage2 = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3,
    ),
    keras.callbacks.ModelCheckpoint(
        BEST_MODEL_STAGE2, save_best_only=True,
    ),
]

# ==========================================
# STAGE 2: PARTIAL FINE-TUNE
# ==========================================

print("\n=== STAGE 2: Partial fine-tuning ===\n")

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks_stage2,
)

# ==========================================
# TEST
# ==========================================

loss, accuracy = model.evaluate(test_ds)
print(f"\nTest Accuracy: {accuracy:.4f}")

# ==========================================
# SAVE FINAL MODEL
# ==========================================

model.save(MODEL_NAME)
print(f"\nModel saved as {MODEL_NAME}")