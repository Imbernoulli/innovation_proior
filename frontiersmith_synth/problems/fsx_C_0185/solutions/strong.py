# TIER: strong
"""Tanh: g(x)=tanh(x), a SMOOTH, BOUNDED, saturating activation.

Why it generalizes across every glacier station where the rectifier families do
not.  On the sharply nonlinear stations (xor / rings / bands / spiral) it supplies
the nonlinearity that beats the linear baseline just like ReLU does.  Crucially, on
the wavy sinusoidal-boundary station -- where the boundary is close to linear and
the labels are noisy -- its saturation caps the response of each hidden unit and
acts as an implicit regularizer, so it stays ABOVE the linear baseline instead of
overfitting the noise the way an unbounded rectifier does.  Because it never
collapses on any single station, the geometric mean rewards it far more than any
one-trick activation.  It still leaves headroom: no station is solved perfectly, so
a yet-better custom activation could score higher."""
import sys, json
import math


def main():
    inst = json.load(sys.stdin)
    grid = inst["grid"]
    g = [math.tanh(float(x)) for x in grid]
    print(json.dumps(g))


main()
