# PocketQ - a quantum classifier that fits in your pocket

A variational quantum machine learning model built from scratch in pure
Python + NumPy. 8 trainable parameters, 100% test accuracy, about 40
microseconds per prediction, and the whole trained model is a JSON file
smaller than this README. No quantum frameworks. It even runs on a
Raspberry Pi Zero if you want it to.

The trained model also runs live in the browser on the project page:
https://swejorblachowski.github.io/pocketq/

## how a prediction works

A prediction runs a 2 qubit circuit:

```
     feature 0 --RY(x0)--RY(t0)--RX(t2)--*--  measure <Z>
                                         |
     feature 1 --RY(x1)--RY(t1)--RX(t3)--*
```

Each input feature becomes the angle of a qubit rotation, so the data
literally rotates qubits. A CNOT gate ties the two qubits together so the
model can't just split into two independent models. The expectation value
<Z> of qubit 0 is a number between -1 and +1, and that's the prediction.

Training uses the parameter-shift rule, which is the quantum way to get
gradients (no backprop needed):

```
dL/dt = [ L(t + pi/2) - L(t - pi/2) ] / 2
```

You nudge each angle by +/- 90 degrees, see how the loss moves, and step
downhill. It works because rotations are sine waves. The cool part is
this same rule works on real quantum hardware, not just my simulator.

## how to run it

You need Python 3.9+ and NumPy. That's it. The dataset is a CSV in this
folder so you don't need internet after installing numpy.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy

python train.py
```

It trains in about 3 seconds and you should see:

```
Training a 2-qubit variational quantum classifier
(8 trainable angles) on 100 samples...

epoch   0 | loss 0.4847 | acc 58.0%
epoch   5 | loss 0.1940 | acc 97.0%
epoch  10 | loss 0.0614 | acc 99.0%
epoch  15 | loss 0.0363 | acc 100.0%
...

Final training accuracy: 100.0%  (trained in 2.9s)
Saved model to model.json (the whole "quantum neural network" is 8 numbers).
```

Then run the edge demo:

```bash
python demo.py
```

It tests the float32 model and the int8 quantized model side by side
(both 100%), shows inference speed, and then asks you to type in
measurements of your own iris flower and classifies it live:

```
float32 model : test accuracy 100.0% |     42 us/inference
int8 model    : test accuracy 100.0% |     42 us/inference

Predict a new flower (cm):
  petal length (1-5): 4.5
  petal width  (0-2): 1.5
  quantum output <Z0> = -0.859  ->  predicted: versicolor (-1)
```

## the C port (edge AI endgame)

`export_c.py` turns the trained model into a C header, and
`firmware/main.c` is the whole simulator ported to plain C. No Python,
no NumPy, no runtime - just math.h and 8 weights. It prints predictions
for a few test flowers so you can diff against the Python version
(they match to about 1e-6). The int8 weights cost 8 bytes of flash.

```bash
python export_c.py     # writes firmware/model.h from model.json
cd firmware
make && ./pocketq      # float32 weights
./pocketq int8         # same thing with the 8-byte int8 model
```

Output:

```
PocketQ firmware (int8 weights)
flower (len, wid)  class          <Z0>
(1.4, 0.2) cm     setosa      0.870986
(4.5, 1.5) cm     versicolor -0.859037
...
```

There's also `bench.py`, which runs the quantum model against logistic
regression and a tiny MLP on the same data and compares accuracy,
model size, and inference speed. Spoiler: the quantum model is the
smallest file AND the fastest per prediction. The MLP is only 37 weights
but pickles to 29 KB, which is framework overhead, not math.

## reading the code

Read it in this order. Each file is short and commented:

- `qsim.py` - what a quantum computer actually is. It's just a vector of
  2^n complex numbers, gates are matrix math, measurement is the Born rule.
- `model.py` - the classifier. Data as angles, entanglement, the
  parameter-shift rule, and int8 quantization.
- `train.py` - loading the CSV, the training loop, saving an 8 number model.
- `demo.py` - edge inference. Float vs int8, timing, live predictions.
- `export_c.py` - the bridge from Python training to C firmware.
- `firmware/main.c` - the simulator in C, for microcontrollers.

## what I'd do next

- add a 3rd qubit and try 3 classes / 3 features
- try the SPSA optimizer (it's what people use for real hardware)
- get this onto an actual ESP32 / RP2040 with a sensor, not just gcc
- train on phone accelerometer data instead of iris

## things I know are wrong with it

- It's a simulator. A real quantum computer has noise and that changes
  how training behaves.
- 2 qubits can only express pretty simple decision boundaries. Harder
  datasets would need more qubits and more parameters.
- Iris (2 classes, 2 features) is a friendly dataset. I picked it so the
  hard part is the quantum stuff, not the data.

## files

```
qsim.py            the simulator (gates, statevector, measurement)
model.py           the classifier + parameter-shift training
train.py           trains on iris_binary.csv, saves model.json
demo.py            edge inference demo (float vs int8, timing, live)
bench.py           quantum vs logistic regression vs a tiny MLP
export_c.py        exports model.json as a C header
firmware/          the C port (main.c, model.h, Makefile)
iris_binary.csv    the dataset (100 flowers, 2 species, 2 features)
model.json         the trained model (made by train.py)
```
