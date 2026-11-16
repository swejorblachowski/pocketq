"""
qsim.py - my tiny quantum simulator
-----------------------------------
This is a quantum circuit simulator I wrote from scratch so I could actually
understand what a quantum computer is doing instead of just importing Qiskit.

The main thing I learned: an n-qubit quantum computer's memory is just a list
of 2^n complex numbers (the statevector). n=2 qubits -> 4 numbers. Gates are
matrix multiplies on that vector, and "measuring" means turning amplitudes
into probabilities (probability = amplitude squared, this is the Born rule).

Everything is loops instead of fancy tensor tricks. It's slower but you
can actually read what's happening.
"""

import numpy as np


# --------------------------------------------------------------- basic gates

def RY(theta):
    # rotate a qubit around the Y axis. I use this for encoding data AND
    # for the trainable weights
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def RX(theta):
    # rotate around the X axis (uses imaginary numbers)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def H():
    # Hadamard gate, puts a qubit into 50/50 superposition
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def X():
    # the NOT gate, flips 0 and 1
    return np.array([[0, 1], [1, 0]], dtype=complex)


# ----------------------------------------------------------------- simulator

class QuantumSim:
    """a (very) small quantum computer, simulated"""

    def __init__(self, n_qubits):
        self.n = n_qubits
        # everything starts in |00...0>
        self.state = np.zeros(1 << n_qubits, dtype=complex)
        self.state[0] = 1.0

    def apply(self, qubit, gate):
        """apply a 2x2 gate to one qubit.

        qubit q is bit number q of each basis state index. for every index i
        where that bit is 0, the "partner" index j = i + 2**q is the same
        basis state with the bit set to 1. the gate mixes those two
        amplitudes together.
        """
        bit = 1 << qubit
        for i in range(1 << self.n):
            if i & bit == 0:
                j = i | bit
                a, b = self.state[i], self.state[j]
                self.state[i] = gate[0, 0] * a + gate[0, 1] * b
                self.state[j] = gate[1, 0] * a + gate[1, 1] * b

    def cx(self, control, target):
        """CNOT gate - if control bit is 1, flip the target bit.
        this is what entangles the qubits (I think of it as the qubits
        sharing information)."""
        cbit, tbit = 1 << control, 1 << target
        for i in range(1 << self.n):
            if (i & cbit) and not (i & tbit):
                j = i | tbit
                self.state[i], self.state[j] = self.state[j], self.state[i]

    def probs(self):
        # chance of measuring each basis state (Born rule: |amp|^2)
        return np.abs(self.state) ** 2

    def prob_qubit_is_one(self, qubit):
        p = self.probs()
        return sum(p[i] for i in range(len(p)) if (i >> qubit) & 1)

    def expectation_z(self, qubit):
        """<Z> for one qubit = the average Z value over many measurements.
        +1 means the qubit acts like |0>, -1 means it acts like |1>.
        this is my model's output number."""
        p = self.probs()
        total = 0.0
        for i in range(len(p)):
            z = 1.0 if ((i >> qubit) & 1) == 0 else -1.0
            total += z * p[i]
        return total
