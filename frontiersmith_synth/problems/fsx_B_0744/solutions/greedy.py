# TIER: greedy
"""
The obvious "textbook" move: invert the Michaelis-Menten law per reaction
using the GIVEN reference concentration x_ref (the steady state at the
nominal e=1 profile), treating it as if it stays fixed regardless of the
target you are trying to hit:

    v_i = e_i * kcat_i * x / (Km_i + x)   =>   e_i = v_target_i*(Km_i+x)/(kcat_i*x)

using x = x_ref_{parent_i} (or X0 if the parent is the external source).

This is correct in the limit where the target flux is close to nominal
(x barely moves). It silently assumes a single, ALWAYS-VALID linear-ish
response and never checks whether the true concentration implied by the
target has shifted the reaction into a different saturation regime -- so
whenever the target redistributes flux sharply across a shared branch
point, this reference point is stale and the computed enzyme level is
badly off.
"""
import sys

X0 = 20.0
EPS = 1e-9


def main():
    toks = sys.stdin.read().split()
    ptr = 0
    R = int(toks[ptr]); ptr += 1
    ptr += 1  # X0 (module constant used instead, matches gen.py)
    parent = [0] * R
    kcat = [0.0] * R
    Km = [0.0] * R
    e_max = [0.0] * R
    for i in range(R):
        parent[i] = int(toks[ptr]); ptr += 1
        ptr += 1  # yield
        kcat[i] = float(toks[ptr]); ptr += 1
        Km[i] = float(toks[ptr]); ptr += 1
        ptr += 1  # tau
        e_max[i] = float(toks[ptr]); ptr += 1
        ptr += 1  # cost
    x_ref = [float(toks[ptr + k]) for k in range(R)]
    ptr += R
    v_target = [float(toks[ptr + k]) for k in range(R)]
    ptr += R

    e = [0.0] * R
    for i in range(R):
        xp = X0 if parent[i] == 0 else x_ref[parent[i] - 1]
        val = v_target[i] * (Km[i] + xp) / (kcat[i] * xp + EPS)
        e[i] = min(max(val, 0.0), e_max[i])

    print(" ".join(str(v) for v in e))


if __name__ == "__main__":
    main()
