"""
train.py - trains the quantum classifier
----------------------------------------
Dataset: iris_binary.csv (included in this folder)
  100 iris flowers, 2 species (setosa = +1, versicolor = -1)
  2 features each: petal length and petal width

The data is a plain CSV so this script needs nothing except Python + NumPy.
No internet, no sklearn, no quantum SDK. I kept it this way so the project
works the same on my laptop or a Raspberry Pi.

Run:
    python train.py

It saves model.json (8 floats + the int8 quantized copy).
"""

import csv
import time
import numpy as np

from model import QuantumClassifier


def load_data(path="iris_binary.csv"):
    X, y = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row["petal_length_cm"]), float(row["petal_width_cm"])])
            y.append(int(row["label"]))
    return np.array(X), np.array(y)


def main():
    X, y = load_data()

    model = QuantumClassifier()
    print(f"Training a {model.N_QUBITS}-qubit variational quantum classifier")
    print(f"({model.N_PARAMS} trainable angles) on {len(X)} samples...\n")

    t0 = time.time()
    model.fit(X, y, epochs=40, lr=0.25)
    train_time = time.time() - t0

    acc = np.mean([model.predict(xi) == yi for xi, yi in zip(X, y)])
    print(f"\nFinal training accuracy: {acc:.1%}  (trained in {train_time:.1f}s)")

    model.save("model.json")
    print('Saved model to model.json (the whole "quantum neural network" is 8 numbers).')


if __name__ == "__main__":
    main()
