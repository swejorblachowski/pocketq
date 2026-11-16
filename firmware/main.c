/*
 * firmware/main.c - the quantum classifier in C, zero dependencies
 * -----------------------------------------------------------------
 * The same 2-qubit circuit as qsim.py, ported to plain C so the trained
 * model can run on a microcontroller (or anywhere gcc exists). No
 * Python, no NumPy, no runtime - just math.h and 8 weights.
 *
 * Build:  make && ./pocketq
 * (or: gcc -O2 -o pocketq main.c -lm)
 *
 * It prints predictions for a few test flowers and the <Z> score, so you
 * can diff the output against the Python version. They match.
 */

#include <stdio.h>
#include <math.h>
#include "model.h"

typedef struct { float re, im; } cpx;

static void c_add(cpx *a, const cpx *b) { a->re += b->re; a->im += b->im; }
static cpx c_mul(const cpx *a, const cpx *b) {
    cpx r = { a->re * b->re - a->im * b->im,
              a->re * b->im + a->im * b->re };
    return r;
}

/* apply a 2x2 complex gate to qubit `q` of the 4-amplitude state */
static void apply_gate(cpx state[4], int q, cpx g[2][2]) {
    int bit = 1 << q, i, j;
    for (i = 0; i < 4; i++) {
        if (!(i & bit)) {
            cpx a = state[i], b = state[j = i | bit], t1, t2;
            t1 = c_mul(&g[0][0], &a); t2 = c_mul(&g[0][1], &b); c_add(&t1, &t2);
            state[i] = t1;
            t1 = c_mul(&g[1][0], &a); t2 = c_mul(&g[1][1], &b); c_add(&t1, &t2);
            state[j] = t1;
        }
    }
}

static void cx(cpx state[4], int control, int target) {
    int cbit = 1 << control, tbit = 1 << target, i, j;
    for (i = 0; i < 4; i++) {
        if ((i & cbit) && !(i & tbit)) {
            cpx tmp = state[i];
            state[i] = state[j = i | tbit];
            state[j] = tmp;
        }
    }
}

static void RY(cpx g[2][2], float t) {
    float c = cosf(t / 2), s = sinf(t / 2);
    g[0][0] = (cpx){ c, 0 }; g[0][1] = (cpx){ -s, 0 };
    g[1][0] = (cpx){ s, 0 }; g[1][1] = (cpx){  c, 0 };
}

static void RX(cpx g[2][2], float t) {
    float c = cosf(t / 2), s = sinf(t / 2);
    /* RX = [[cos, -i sin], [-i sin, cos]] */
    g[0][0] = (cpx){ c, 0 };   g[0][1] = (cpx){ 0, -s };
    g[1][0] = (cpx){ 0, -s };  g[1][1] = (cpx){ c, 0 };
}

/* run the circuit on features x0, x1. returns <Z0> */
static float run_circuit(float x0, float x1, const float params[8]) {
    cpx state[4] = { {1,0}, {0,0}, {0,0}, {0,0} };
    cpx g[2][2];
    int layer, k = 0;

    /* encode: features become angles */
    RY(g, x0 * PQ_FEATURE_SCALE); apply_gate(state, 0, g);
    RY(g, x1 * PQ_FEATURE_SCALE); apply_gate(state, 1, g);

    /* trainable layers */
    for (layer = 0; layer < 2; layer++) {
        RY(g, params[k++]); apply_gate(state, 0, g);
        RY(g, params[k++]); apply_gate(state, 1, g);
        RX(g, params[k++]); apply_gate(state, 0, g);
        RX(g, params[k++]); apply_gate(state, 1, g);
        cx(state, 0, 1);
    }

    /* <Z0>: qubit 0 is bit 0 of each index */
    float z = 0;
    for (int i = 0; i < 4; i++) {
        float p = state[i].re * state[i].re + state[i].im * state[i].im;
        z += ((i & 1) == 0 ? 1.0f : -1.0f) * p;
    }
    return z;
}

static const char *classify(float z) { return z > 0 ? "setosa" : "versicolor"; }

int main(int argc, char **argv) {
    /* test flowers: [petal length, petal width] in cm */
    float tests[][2] = {
        {1.4f, 0.2f}, {4.5f, 1.5f}, {5.1f, 1.8f},
        {1.0f, 0.1f}, {6.0f, 2.4f}, {3.0f, 1.0f}
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    int use_int8 = argc > 1;   /* any argument = use the int8 weights */

    printf("PocketQ firmware (%s weights)\n", use_int8 ? "int8" : "float32");
    printf("%-22s %-8s %10s\n", "flower (len, wid)", "class", "<Z0>");

    for (int i = 0; i < n; i++) {
        float z;
        if (use_int8) {
            float p8[8];
            for (int j = 0; j < 8; j++)
                p8[j] = pq_params_int8[j] * PQ_QUANT_SCALE;
            z = run_circuit(tests[i][0], tests[i][1], p8);
        } else {
            z = run_circuit(tests[i][0], tests[i][1], pq_params);
        }
        printf("(%.1f, %.1f) cm       %-9s %10.6f\n",
               tests[i][0], tests[i][1], classify(z), z);
    }
    return 0;
}
