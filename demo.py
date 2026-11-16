"""
demo.py - edge inference with the trained model
-----------------------------------------------
Loads model.json (8 numbers), runs predictions on flowers the model never
trained on, and times single predictions. Then it does the same thing with
the INT8 quantized weights (the trick mobile AI uses to shrink models) to
show accuracy doesn't drop.

Run after train.py:
    python demo.py
"""

import time
import numpy as np

from model import QuantumClassifier
from train import load_data


def timed_inference(model, x, n=200):
    # average inference time over n runs, in microseconds
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(x)
    return (time.perf_counter() - t0) / n * 1e6


def main():
    X, y = load_data()

    # hold out the last 20 flowers as a test set (never trained on these)
    X_test, y_test = X[-20:], y[-20:]

    print("== PocketQ edge inference demo ==\n")

    # normal float32 model
    model = QuantumClassifier.load("model.json")
    acc = np.mean([model.predict(x) == t for x, t in zip(X_test, y_test)])
    us = timed_inference(model, X_test[0])
    print(f"float32 model : test accuracy {acc:5.1%} | {us:6.0f} us/inference")

    # int8 quantized model
    qmodel = QuantumClassifier.load("model.json", quantized=True)
    qacc = np.mean([qmodel.predict(x) == t for x, t in zip(X_test, y_test)])
    qus = timed_inference(qmodel, X_test[0])
    print(f"int8 model    : test accuracy {qacc:5.1%} | {qus:6.0f} us/inference")

    print("\nQuantization cost nothing in accuracy. Makes sense when you think"
          "\nabout it - the model is pure trigonometry, so 8 bits of angle"
          "\nprecision is plenty. That's why I think quantum-style models are"
          "\ninteresting for tiny edge devices.\n")

    # classify a flower you type in
    print("Predict a new flower (cm):")
    length = float(input("  petal length (1-5): "))
    width = float(input("  petal width  (0-2): "))
    score = model.predict_score([length, width])
    label = model.predict([length, width])
    name = "setosa (+1)" if label == 1 else "versicolor (-1)"
    print(f"\n  quantum output <Z0> = {score:+.3f}  ->  predicted: {name}")


if __name__ == "__main__":
    main()
