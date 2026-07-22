#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance of "Enzyme Staging for Target Flux" to stdout.

A small branching metabolic tree: reaction i (1..R) consumes metabolite parent_i
(0 = external, unlimited-supply source; else an already-produced internal
metabolite) and produces yield_i units of its OWN metabolite i, via Michaelis-
Menten kinetics v_i = e_i * kcat_i * x_{parent_i} / (Km_i + x_{parent_i}).
Internal metabolite i's steady-state concentration is set by a dilution/uptake
balance: x_i = tau_i * max(yield_i*v_i - sum of children's consumption, 0) --
a genuine per-node fixed point, since the children's consumption term is
itself a function of x_i.

The instance also gives a REFERENCE steady state x_ref (computed at a nominal
e=1 profile for every reaction) alongside the TARGET flux vector v_target,
which is deliberately redistributed away from the nominal profile -- most
sharply at shared branch points -- so that a solver assuming the reference
concentration stays valid gets badly misled on several cases.

Output (stdout):
  R X0
  R lines: parent_i yield_i kcat_i Km_i tau_i e_max_i cost_i   (i=1..R)
  1 line : x_ref_1 ... x_ref_R
  1 line : v_target_1 ... v_target_R
"""
import sys
import random

X0 = 20.0

TRAP_IDS = {3, 5, 7, 9, 10}


def scale_for(t):
    table = {1: 7, 2: 8, 3: 8, 4: 9, 5: 9, 6: 10, 7: 11, 8: 12, 9: 12, 10: 13}
    return table[t]


def simulate(R, parent, yield_, kcat, Km, tau, e):
    """Exact topological per-node bisection solve for the steady state.
    The tree structure has no real cycles: metabolite i's balance equation
    depends only on an ALREADY-finalized ancestor concentration plus a
    self-term (its own children's consumption, monotone in x_i), so a
    single pass in increasing index order, with a per-node bisection,
    finds the unique fixed point exactly (up to bisection precision)."""
    children = [[] for _ in range(R + 1)]
    for j in range(R):
        children[parent[j]].append(j)
    x = [0.0] * R
    for i in range(R):
        xp_val = X0 if parent[i] == 0 else x[parent[i] - 1]
        v_i = e[i] * kcat[i] * xp_val / (Km[i] + xp_val)
        C = yield_[i] * v_i
        if C <= 0.0:
            x[i] = 0.0
            continue
        kids = children[i + 1]

        def Sfun(xi, kids=kids):
            s = 0.0
            for k in kids:
                s += e[k] * kcat[k] * xi / (Km[k] + xi)
            return s

        lo, hi = 0.0, tau[i] * C
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            val = mid - tau[i] * max(C - Sfun(mid), 0.0)
            if val < 0:
                lo = mid
            else:
                hi = mid
        x[i] = 0.5 * (lo + hi)
    xx = [X0] + x
    v = [0.0] * R
    for j in range(R):
        xp = xx[parent[j]]
        v[j] = e[j] * kcat[j] * xp / (Km[j] + xp)
    return x, v


def build(t):
    R = scale_for(t)
    rnd = random.Random(20260713 + 131 * t)

    parent = [0] * R
    for j in range(1, R):
        parent[j] = 0 if rnd.random() < 0.35 else rnd.randint(1, j)

    yield_ = [rnd.choice([1, 1, 1, 2]) for _ in range(R)]
    kcat = [round(rnd.uniform(0.8, 2.5), 4) for _ in range(R)]
    Km = [round(rnd.uniform(0.5, 3.0), 4) for _ in range(R)]
    tau = [round(rnd.uniform(0.5, 2.5), 4) for _ in range(R)]
    e_max = [round(rnd.uniform(3.0, 6.0), 4) for _ in range(R)]
    cost = [round(rnd.uniform(0.5, 2.0), 4) for _ in range(R)]

    e_ref = [1.0] * R
    x_ref, v_ref = simulate(R, parent, yield_, kcat, Km, tau, e_ref)

    is_trap = t in TRAP_IDS
    FRAC = 0.85  # reserve slack so every metabolite's implied budget stays > 0
    v_target = [0.0] * R
    remaining_budget = [None] * (R + 1)
    for j in range(R):
        if is_trap:
            scale = rnd.uniform(2.0, 3.5) if rnd.random() < 0.5 else rnd.uniform(0.15, 0.35)
        else:
            scale = rnd.uniform(0.35, 2.2)
        raw = max(0.0, v_ref[j] * scale)
        m = parent[j]
        if m == 0:
            vt = raw
        else:
            if remaining_budget[m] is None:
                remaining_budget[m] = FRAC * yield_[m - 1] * v_target[m - 1]
            vt = min(raw, max(0.0, remaining_budget[m]))
            remaining_budget[m] -= vt
        v_target[j] = round(vt, 6)

    return R, parent, yield_, kcat, Km, tau, e_max, cost, x_ref, v_target


def main():
    t = int(sys.argv[1])
    R, parent, yield_, kcat, Km, tau, e_max, cost, x_ref, v_target = build(t)
    out = []
    out.append(f"{R} {X0:.6f}")
    for i in range(R):
        out.append(
            f"{parent[i]} {yield_[i]} {kcat[i]:.6f} {Km[i]:.6f} "
            f"{tau[i]:.6f} {e_max[i]:.6f} {cost[i]:.6f}"
        )
    out.append(" ".join(f"{v:.6f}" for v in x_ref))
    out.append(" ".join(f"{v:.6f}" for v in v_target))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
