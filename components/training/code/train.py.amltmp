import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str)
parser.add_argument("--test_data", type=str)
parser.add_argument("--epochs", type=int)
parser.add_argument("--output", type=str)
args = parser.parse_args()

def load_images(folder):
    X, y = [], []
    for p in Path(folder).glob("*.jpg"):
        img = tf.keras.utils.load_img(p, target_size=(64, 64))
        X.append(tf.keras.utils.img_to_array(img) / 255.0)
        y.append(int(p.name.split("_")[0]))
    return np.array(X), tf.keras.utils.to_categorical(y, 3)

X_train, y_train = load_images(args.train_data)
X_test, y_test = load_images(args.test_data)

model = tf.keras.Sequential([
    tf.keras.layers.Input((64, 64, 3)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=args.epochs
)

# BELANGRIJK DEEL 1.0
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

model_path = output_dir / "animal-cnn.keras"
model.save(model_path)
