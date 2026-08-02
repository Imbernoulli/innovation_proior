#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- deterministic scorer for reaction-network pruning.

Feasibility: the submitted reaction subset must parse exactly (k distinct valid indices,
no extra/garbage tokens, all finite) and, integrated with the SAME fixed-step RK4 scheme
as the full network, must keep the target species within `epsilon` of the full-network
trajectory for EVERY held-out condition. Any violation -> Ratio: 0.0.

Objective (minimize): F = number of kept reactions. Baseline B = m (keep everything,
always feasible). Ratio = min(1, 0.1*B/F).
"""
import sys
import math


def fail(msg):
    print(f"INFEASIBLE: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_floats(tokens, i, k):
    vals = []
    for _ in range(k):
        if i >= len(tokens):
            return None, i
        try:
            v = float(tokens[i])
        except ValueError:
            return None, i
        if not math.isfinite(v):
            return None, i
        vals.append(v)
        i += 1
    return vals, i


def simulate_target(n, reacts, c0, T_horizon, N_steps, target):
    dt = T_horizon / N_steps
    c = list(c0)

    def deriv(state):
        d = [0.0] * n
        for (r, p, rate) in reacts:
            f = rate * state[r]
            d[r] -= f
            d[p] += f
        return d

    traj = [c[target]]
    for _ in range(N_steps):
        k1 = deriv(c)
        c2 = [c[i] + 0.5 * dt * k1[i] for i in range(n)]
        k2 = deriv(c2)
        c3 = [c[i] + 0.5 * dt * k2[i] for i in range(n)]
        k3 = deriv(c3)
        c4 = [c[i] + dt * k3[i] for i in range(n)]
        k4 = deriv(c4)
        c = [c[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n)]
        traj.append(c[target])
        for v in c:
            if not math.isfinite(v):
                return None
    return traj


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        in_tokens = f.read().split()
    ip = 0
    n = int(in_tokens[ip]); ip += 1
    m = int(in_tokens[ip]); ip += 1
    target = int(in_tokens[ip]); ip += 1
    P = int(in_tokens[ip]); ip += 1
    T_horizon = float(in_tokens[ip]); ip += 1
    N_steps = int(in_tokens[ip]); ip += 1
    eps = float(in_tokens[ip]); ip += 1

    reactions = []
    for _ in range(m):
        r = int(in_tokens[ip]); ip += 1
        p = int(in_tokens[ip]); ip += 1
        rate = float(in_tokens[ip]); ip += 1
        reactions.append((r, p, rate))

    conditions = []
    for _ in range(P):
        c = [float(in_tokens[ip + j]) for j in range(n)]
        ip += n
        conditions.append(c)

    if not (0 <= target < n):
        fail("bad instance: target out of range")

    # ---- parse participant output strictly ----
    try:
        out_text = open(outf).read()
    except Exception:
        fail("cannot read output")
    out_tokens = out_text.split()
    if len(out_tokens) == 0:
        fail("empty output")
    try:
        k_val = float(out_tokens[0])
    except ValueError:
        fail("first token is not a number")
    if not math.isfinite(k_val):
        fail("k is not finite")
    if k_val != int(k_val):
        fail("k is not an integer")
    k = int(k_val)
    if k < 0 or k > m:
        fail(f"k={k} out of range [0,{m}]")
    if len(out_tokens) != 1 + k:
        fail(f"expected exactly {1 + k} tokens, got {len(out_tokens)}")

    idx_vals, _ = read_floats(out_tokens, 1, k)
    if idx_vals is None:
        fail("could not parse k indices")
    kept = []
    for v in idx_vals:
        if v != int(v):
            fail("reaction index is not an integer")
        iv = int(v)
        if not (0 <= iv < m):
            fail(f"reaction index {iv} out of range [0,{m - 1}]")
        kept.append(iv)
    if len(set(kept)) != len(kept):
        fail("duplicate reaction index")

    if k == 0:
        fail("empty reaction set cannot reach the target")

    kept_reacts = [reactions[i] for i in kept]

    # ---- feasibility: full vs reduced network must agree within eps on EVERY condition ----
    for p_idx, c0 in enumerate(conditions):
        full_traj = simulate_target(n, reactions, c0, T_horizon, N_steps, target)
        red_traj = simulate_target(n, kept_reacts, c0, T_horizon, N_steps, target)
        if full_traj is None or red_traj is None:
            fail(f"non-finite trajectory (condition {p_idx})")
        dev = max(abs(a - b) for a, b in zip(full_traj, red_traj))
        if dev > eps:
            fail(f"condition {p_idx}: deviation {dev:.6f} > epsilon {eps:.6f}")

    B = float(m)
    F = float(k)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"OK: kept {k} of {m} reactions, baseline {m}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
