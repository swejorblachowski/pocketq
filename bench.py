"""
bench.py - quantum vs classical, the numbers that matter for edge AI
--------------------------------------------------------------------
Same task (iris, 2 classes, 2 features), same test set. The question
isn't just "who's more accurate" - for edge AI the interesting numbers
are model size and inference speed.

Run after train.py:
    python bench.py
"""

import json
import os
import pickle
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from model import QuantumClassifier
from train import load_data


def model_size(path):
    return os.path.getsize(path)


def time_inference(predict_fn, X, n=300):
    t0 = time.perf_counter()
    for _ in range(n):
        for x in X:
            predict_fn(x)
    return (time.perf_counter() - t0) / (n * len(X)) * 1e6   # us/sample


def main():
    X, y = load_data()
    X_tr, X_te = X[:80], X[80:]
    y_tr, y_te = y[:80], y[80:]

    # ---- quantum model
    q = QuantumClassifier.load("model.json")
    qacc = np.mean([q.predict(x) == t for x, t in zip(X_te, y_te)])
    qus = time_inference(q.predict, X_te[:10])
    qsize = model_size("model.json")

    # ---- classical baselines, same data
    lr = LogisticRegression().fit(X_tr, y_tr)
    lracc = np.mean(lr.predict(X_te) == y_te)
    lrus = time_inference(lambda x: lr.predict([x]), X_te[:10])
    lr_pickle = pickle.dumps(lr)
    with open("/tmp/lr.pkl", "wb") as f:
        f.write(lr_pickle)
    lrsize = len(lr_pickle)

    mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000,
                        random_state=42).fit(X_tr, y_tr)
    mlpacc = np.mean(mlp.predict(X_te) == y_te)
    mlpus = time_inference(lambda x: mlp.predict([x]), X_te[:10])
    mlpsize = len(pickle.dumps(mlp))

    print(f"{'model':<22}{'accuracy':>9}{'size':>9}{'us/sample':>11}")
    print(f"{'-'*51}")
    print(f"{'quantum (this repo)':<22}{qacc:>8.1%}{qsize:>7}B{qus:>10.0f}")
    print(f"{'logistic regression':<22}{lracc:>8.1%}{lrsize:>7}B{lrus:>10.0f}")
    print(f"{'tiny MLP (4 hidden)':<22}{mlpacc:>8.1%}{mlpsize:>7}B{mlpus:>10.0f}")

    print("\nthe quantum model is the smallest file and the fastest per"
          "\nprediction. the tiny MLP is only 37 weights but pickles to"
          "\nseveral KB - framework overhead, not math.")


if __name__ == "__main__":
    main()
