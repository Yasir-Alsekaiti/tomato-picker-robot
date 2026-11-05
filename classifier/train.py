# CNN for tomato ripeness: ripe / unripe / rotten
# data/train/<class>/*.jpg and data/val/<class>/*.jpg

import argparse

import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE = (224, 224)

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--out", default="tomato_cnn.keras")
args = parser.parse_args()

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{args.data}/train", image_size=IMG_SIZE, batch_size=32)
val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{args.data}/val", image_size=IMG_SIZE, batch_size=32)
classes = train_ds.class_names
print("classes:", classes)

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomBrightness(0.2),
])

model = tf.keras.Sequential([
    layers.Input(shape=IMG_SIZE + (3,)),
    augment,
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(classes), activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=[tf.keras.callbacks.ModelCheckpoint(args.out, save_best_only=True, monitor="val_accuracy")],
)
print("saved", args.out)
