"""
model.py - the variational quantum classifier
---------------------------------------------
A variational quantum classifier is basically the quantum version of a
neural net:

    data -> quantum circuit with adjustable angles -> measurement -> output

How one prediction works (2 qubits, 2 features):
  1. encode:    each feature becomes an angle, RY(x * scale) on a qubit
  2. trainable: layers of RY and RX rotations + CX entanglement
  3. readout:   <Z> of qubit 0, a number between -1 and +1
  4. predict:   <Z> > 0 -> class +1, otherwise class -1

For training I use the parameter-shift rule, which is the quantum way to get
gradients (no backprop needed):

    dLoss/dtheta = [ Loss(theta + pi/2) - Loss(theta - pi/2) ] / 2

You just run the circuit twice per parameter with the angle nudged by
+/- 90 degrees. It works because rotations are sine waves. The cool part is
this exact same rule works on real quantum hardware, not just my simulator.
"""

import json
import numpy as np

from qsim import QuantumSim, RY, RX


class QuantumClassifier:
    """2-qubit variational quantum classifier for 2-feature data"""

    N_QUBITS = 2
    N_LAYERS = 2
    N_PARAMS = 4 * N_LAYERS  # 2 RY + 2 RX per layer = 8 total

    def __init__(self, feature_scale=np.pi / 4):
        # seed the random start so results are reproducible
        rng = np.random.default_rng(42)
        self.params = rng.uniform(-np.pi, np.pi, self.N_PARAMS).tolist()
        self.feature_scale = feature_scale

    # ------------------------------------------------------------- the circuit

    def _forward(self, x, params):
        """run the circuit on features x, return <Z0> in [-1, 1]"""
        sim = QuantumSim(self.N_QUBITS)

        # step 1: features -> angles
        sim.apply(0, RY(x[0] * self.feature_scale))
        sim.apply(1, RY(x[1] * self.feature_scale))

        # step 2: trainable layers, each one is rotations then entanglement
        k = 0
        for _ in range(self.N_LAYERS):
            sim.apply(0, RY(params[k])); k += 1
            sim.apply(1, RY(params[k])); k += 1
            sim.apply(0, RX(params[k])); k += 1
            sim.apply(1, RX(params[k])); k += 1
            sim.cx(0, 1)

        # step 3: read out qubit 0
        return sim.expectation_z(0)

    # ----------------------------------------------------------- predictions

    def predict_score(self, x):
        return self._forward(x, self.params)

    def predict(self, x):
        return 1 if self.predict_score(x) > 0 else -1

    # --------------------------------------------------------------- training

    @staticmethod
    def _loss(x, y, params, forward):
        # mean squared error between <Z0> and the target (+1 or -1)
        err = forward(x, params) - y
        return 0.5 * err * err

    def fit(self, X, y, epochs=40, lr=0.25, verbose=True):
        """gradient descent with the parameter-shift rule"""
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        history = []

        for epoch in range(epochs):
            total_loss = 0.0
            grads = np.zeros(self.N_PARAMS)

            for xi, yi in zip(X, y):
                total_loss += self._loss(xi, yi, self.params, self._forward)

                # parameter shift gradient for every angle
                for k in range(self.N_PARAMS):
                    plus = list(self.params)
                    minus = list(self.params)
                    plus[k] += np.pi / 2
                    minus[k] -= np.pi / 2
                    lp = self._loss(xi, yi, plus, self._forward)
                    lm = self._loss(xi, yi, minus, self._forward)
                    grads[k] += (lp - lm) / 2.0

            grads /= len(X)
            self.params = (np.asarray(self.params) - lr * grads).tolist()

            avg_loss = total_loss / len(X)
            history.append(avg_loss)
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                acc = np.mean([self.predict(xi) == yi for xi, yi in zip(X, y)])
                print(f"epoch {epoch:3d} | loss {avg_loss:.4f} | acc {acc:6.1%}")

        return history

    # ---------------------------------------------------- save / load the model

    def save(self, path):
        """the whole trained model is just 8 floats. I also save an int8
        quantized copy like mobile AI models do, to show the model survives
        getting squished down to 8 bits."""
        data = {
            "params_float32": [float(p) for p in self.params],
            "feature_scale": float(self.feature_scale),
            # int8 version: store round(theta * 127/pi) instead of theta
            "params_int8": [int(round(p / np.pi * 127)) for p in self.params],
            "quant_scale": float(np.pi / 127),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    @classmethod
    def load(cls, path, quantized=False):
        with open(path) as f:
            data = json.load(f)
        model = cls(feature_scale=data["feature_scale"])
        if quantized:
            # turn the int8 numbers back into angles
            model.params = [q * data["quant_scale"] for q in data["params_int8"]]
        else:
            model.params = data["params_float32"]
        return model
