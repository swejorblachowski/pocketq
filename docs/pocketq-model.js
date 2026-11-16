// the entire trained model. 8 angles (plus an int8 copy). that's it.
// trained by train.py with the parameter-shift rule, see the repo.
const POCKETQ_MODEL = {
  params: [
    2.6723710894080255,
    0.9855813743118543,
    2.2853682671588786,
    1.2400999002927886,
    -2.7844305461293946,
    2.988423371570267,
    2.3154107439089544,
    1.7973950398246918
  ],
  paramsInt8: [108, 40, 92, 50, -113, 121, 94, 73],
  quantScale: 0.024736950028266088,
  featureScale: Math.PI / 4
};

// ---- the quantum simulator, ported from qsim.py to the browser ------------
// 2 qubits, state = [a00, a01, a10, a11] (amplitudes as [re, im] pairs)

function applyGate(state, qubit, g) {
  // g = [[g00,g01],[g10,g11]] with complex numbers as [re,im] pairs
  const bit = 1 << qubit;
  for (let i = 0; i < 4; i++) {
    if ((i & bit) === 0) {
      const j = i | bit;
      const a = state[i], b = state[j];
      state[i] = cadd(cmul(g[0][0], a), cmul(g[0][1], b));
      state[j] = cadd(cmul(g[1][0], a), cmul(g[1][1], b));
    }
  }
}

function cx(state, control, target) {
  const cbit = 1 << control, tbit = 1 << target;
  for (let i = 0; i < 4; i++) {
    if ((i & cbit) && !(i & tbit)) {
      const j = i | tbit;
      const tmp = state[i]; state[i] = state[j]; state[j] = tmp;
    }
  }
}

function RY(t) {
  const c = Math.cos(t / 2), s = Math.sin(t / 2);
  return [
    [[c, 0], [-s, 0]],
    [[s, 0], [c, 0]]
  ];
}
function RX(t) {
  // RX has imaginary off-diagonal entries: [[cos, -i sin], [-i sin, cos]]
  const c = Math.cos(t / 2), s = Math.sin(t / 2);
  return [
    [[c, 0], [0, -s]],
    [[0, -s], [c, 0]]
  ];
}

function cadd(a, b) { return [a[0] + b[0], a[1] + b[1]]; }
function cmul(a, b) { return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]; }

// run the circuit on features [x0, x1], return { z: <Z0>, amps: state }
function runCircuit(x, params) {
  let state = [[1, 0], [0, 0], [0, 0], [0, 0]];
  applyGate(state, 0, RY(x[0] * POCKETQ_MODEL.featureScale));
  applyGate(state, 1, RY(x[1] * POCKETQ_MODEL.featureScale));
  let k = 0;
  for (let layer = 0; layer < 2; layer++) {
    applyGate(state, 0, RY(params[k++]));
    applyGate(state, 1, RY(params[k++]));
    applyGate(state, 0, RX(params[k++]));
    applyGate(state, 1, RX(params[k++]));
    cx(state, 0, 1);
  }
  // <Z0>: qubit 0 is bit 0 of each index
  let z = 0;
  for (let i = 0; i < 4; i++) {
    const p = state[i][0] * state[i][0] + state[i][1] * state[i][1];
    z += ((i & 1) === 0 ? 1 : -1) * p;
  }
  return { z, amps: state };
}

function predict(x, useInt8) {
  const params = useInt8
    ? POCKETQ_MODEL.paramsInt8.map(q => q * POCKETQ_MODEL.quantScale)
    : POCKETQ_MODEL.params;
  return runCircuit(x, params);
}
