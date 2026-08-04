import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import EfficientNetV2S
from keras.applications.efficientnet_v2 import preprocess_input

# ==========================================
# CONFIG
# ==========================================

DATASET_PATH = "Original_Data"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "Validate")
TEST_DIR = os.path.join(DATASET_PATH, "test")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 25
EPOCHS_STAGE2 = 15

MODEL_NAME = "efficientnetv2_grape_final.keras"
BEST_MODEL_STAGE1 = "best_model_stage1.keras"
BEST_MODEL_STAGE2 = "best_model_stage2.keras"

# ==========================================
# GPU CHECK (just informational, doesn't stop the script)
# ==========================================

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"\nGPU detected: {gpus}\n")
else:
    print("\nNo GPU detected — running on CPU. This will be SLOW for this config.\n"
          "Consider lowering IMG_SIZE (e.g. 224) and EPOCHS if this is your first test run.\n")

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
NUM_CLASSES = len(CLASS_NAMES)

print("\nClasses:")
print(CLASS_NAMES)

# Sanity check: make sure val/test have the same classes as train
assert val_ds.class_names == CLASS_NAMES, (
    f"Validation classes {val_ds.class_names} don't match train classes {CLASS_NAMES}"
)
assert test_ds.class_names == CLASS_NAMES, (
    f"Test classes {test_ds.class_names} don't match train classes {CLASS_NAMES}"
)

# ==========================================
# PREPROCESSING
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ==========================================
# MODEL
# ==========================================

base_model = EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)

base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.summary()

# ==========================================
# COMPILE (STAGE 1 - frozen base)
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_stage1 = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3,
    ),
    keras.callbacks.ModelCheckpoint(
        BEST_MODEL_STAGE1, save_best_only=True,
    ),
]

# ==========================================
# TRAIN STAGE 1
# ==========================================

print("\n=== STAGE 1: Training with frozen base ===\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks_stage1,
)

# ==========================================
# FINE TUNING (STAGE 2 - unfrozen base)
# ==========================================

print("\n=== STAGE 2: Fine-tuning full model ===\n")

base_model.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Separate callback list + checkpoint file so stage 2 doesn't
# overwrite stage 1's best checkpoint
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

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks_stage2,
)

# ==========================================
# TEST
# ==========================================
# Note: EarlyStopping(restore_best_weights=True) means `model` already
# holds the best weights from stage 2 at this point, not just whatever
# epoch it stopped on.

loss, accuracy = model.evaluate(test_ds)
print(f"\nTest Accuracy: {accuracy:.4f}")

# ==========================================
# SAVE FINAL MODEL
# ==========================================

model.save(MODEL_NAME)
print(f"\nModel saved as {MODEL_NAME}")
print(f"Best stage-1 checkpoint saved as {BEST_MODEL_STAGE1}")
print(f"Best stage-2 checkpoint saved as {BEST_MODEL_STAGE2}")