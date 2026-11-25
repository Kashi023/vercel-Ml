
from pathlib import Path
import os

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "chest_xray"

MODEL_PATH = BASE_DIR / "image_model.keras"
TEMP_MODEL_PATH = BASE_DIR / "image_model_new.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

# 0 = NORMAL, 1 = PNEUMONIA
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

tf.keras.utils.set_random_seed(SEED)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_dataset(split, shuffle=False):
    folder = DATASET_DIR / split

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    for class_name in CLASS_NAMES:
        class_folder = folder / class_name

        if not class_folder.exists():
            raise FileNotFoundError(
                f"Class folder not found: {class_folder}"
            )

        if not any(class_folder.iterdir()):
            raise ValueError(
                f"No files found in: {class_folder}"
            )

    dataset = tf.keras.utils.image_dataset_from_directory(
        folder,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        class_names=CLASS_NAMES,
        shuffle=shuffle,
        seed=SEED,
    )

    return dataset


# --------------------------------------------------
# 3. BUILD MODEL
# --------------------------------------------------

def build_model():
    print("Loading pretrained MobileNetV2...")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze pretrained feature extractor.
    base_model.trainable = False

    inputs = layers.Input(
        shape=(224, 224, 3),
        name="chest_xray_image",
    )

    # Built-in preprocessing:
    # [0, 255] -> [-1, 1]
    # Avoid Lambda(preprocess_input) serialization errors.

    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="preprocessing",
    )(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dropout(
        0.3,
        name="dropout"
    )(x)

    # Sigmoid output:
    # 0 = NORMAL
    # 1 = PNEUMONIA

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="prediction",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="HealthAI_Chest_Xray_Classifier",
    )

    # Metrics for training and validation.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
                threshold=0.5,
            ),
            tf.keras.metrics.Precision(
                name="precision",
                thresholds=0.5,
            ),
            tf.keras.metrics.Recall(
                name="recall",
                thresholds=0.5,
            ),
            tf.keras.metrics.AUC(
                name="auc",
            ),
        ],
    )

    return model


# --------------------------------------------------
# 4. TRAIN, SAVE AND VERIFY
# --------------------------------------------------

def main():
    print("=" * 60)
    print("HealthAI - Chest X-ray Classification")
    print("=" * 60)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {DATASET_DIR}\n"
            "Expected directory structure:\n"
            "chest_xray/train/NORMAL\n"
            "chest_xray/train/PNEUMONIA\n"
            "chest_xray/val/NORMAL\n"
            "chest_xray/val/PNEUMONIA\n"
            "chest_xray/test/NORMAL\n"
            "chest_xray/test/PNEUMONIA"
        )

    print("\nLoading datasets...")

    train_ds = load_dataset("train", shuffle=True)
    val_ds = load_dataset("val", shuffle=False)
    test_ds = load_dataset("test", shuffle=False)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    print("\nBuilding model...")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(TEMP_MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.TerminateOnNaN(),
    ]

    print("\nStarting training...")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save final restored weights to a temporary file.
    print("\nSaving final model...")

    model.save(TEMP_MODEL_PATH)

    # Verify model serialization before replacing old model.
    print("\nVerifying saved model...")

    verified_model = tf.keras.models.load_model(
        TEMP_MODEL_PATH,
        compile=False,
    )

    # Verify output shape and a forward pass.
    for images, _ in test_ds.take(1):
        predictions = verified_model.predict(
            images[:1],
            verbose=0,
        )

        if predictions.shape != (1, 1):
            raise RuntimeError(
                f"Unexpected prediction shape: {predictions.shape}"
            )

        if not np.isfinite(predictions).all():
            raise RuntimeError(
                "Model produced a non-finite prediction."
            )

    print("Model serialization and inference test passed.")

    # Replace old model only after successful verification.
    os.replace(TEMP_MODEL_PATH, MODEL_PATH)

    print(f"\nNew model saved successfully: {MODEL_PATH}")

    # --------------------------------------------------
    # 5. EVALUATE SAVED MODEL
    # --------------------------------------------------

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    model.compile(
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
                threshold=0.5,
            ),
            tf.keras.metrics.Precision(
                name="precision",
                thresholds=0.5,
            ),
            tf.keras.metrics.Recall(
                name="recall",
                thresholds=0.5,
            ),
            tf.keras.metrics.AUC(
                name="auc",
            ),
        ],
    )

    print("\nEvaluating on held-out test dataset...")

    results = model.evaluate(
        test_ds,
        verbose=1,
        return_dict=True,
    )

    print("\nTest results:")

    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    # --------------------------------------------------
    # 6. CLASSIFICATION REPORT
    # --------------------------------------------------

    print("\nGenerating predictions...")

    y_true = np.concatenate([
        labels.numpy().ravel()
        for _, labels in test_ds
    ]).astype(int)

    scores = model.predict(
        test_ds,
        verbose=1,
    ).ravel()

    # Threshold 0.5: classify as pneumonia at >= 0.5.
    y_pred = (scores >= 0.5).astype(int)

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
            digits=4,
        )
    )

    if len(np.unique(y_true)) == 2:
        auc_score = roc_auc_score(y_true, scores)
        print(f"Test ROC-AUC: {auc_score:.4f}")
    else:
        print(
            "ROC-AUC unavailable: "
            "test data contains only one class."
        )

    # --------------------------------------------------
    # 7. COMPLETION
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Training and evaluation completed.")
    print(f"Saved model: {MODEL_PATH}")
    print("Educational prototype only.")
    print("Not clinically validated for medical diagnosis.")
    print("=" * 60)


if __name__ == "__main__":
    main()
