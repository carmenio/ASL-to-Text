"""CNN training workflow for ASL image folders."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_cnn_model(num_classes: int, image_size: tuple[int, int] = (128, 128)):
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(image_size[0], image_size[1], 3)),
            keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(256, (5, 5), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Conv2D(384, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(384, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
    return model


def train_cnn(
    data_dir: str | Path = "data/raw",
    output: str | Path = "models/best_model_CNN.keras",
    epochs: int = 10,
    batch_size: int = 128,
    image_size: tuple[int, int] = (128, 128),
):
    from tensorflow import keras

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_path}")

    train_ds = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    normalizer = keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalizer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalizer(x), y))

    model = build_cnn_model(len(class_names), image_size=image_size)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(output_path, monitor="val_loss", save_best_only=True, mode="min"),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        keras.callbacks.TensorBoard(log_dir="logs"),
    ]
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)
    return history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ASL-to-Text CNN model from image folders.")
    parser.add_argument("--data", default="data/raw", help="Dataset directory containing one folder per class.")
    parser.add_argument("--output", default="models/best_model_CNN.keras", help="Model output path.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_cnn(args.data, args.output, args.epochs, args.batch_size)
    print(f"Best model saved to {args.output}")


if __name__ == "__main__":
    main()
