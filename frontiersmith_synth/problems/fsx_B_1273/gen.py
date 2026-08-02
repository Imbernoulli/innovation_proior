#!/usr/bin/env python3
"""gen.py <testId> -- fsx_B_1273 (pension-glidepath-design).

Prints ONE instance to stdout. testId 1..10 is a difficulty ladder: small,
calm instances first, growing into longer horizons with deliberately
engineered "trap" return blocks (an early rally into overfunding followed
by a crash, and a persistently-underfunded run rescued only by a late
rally) from testId 4 onward. All randomness is seeded purely from testId,
so the same testId always reproduces byte-identical output.
"""
import random
import sys

# Fixed policy-grid geometry shared by every instance (also restated in the
# input so a solution never needs to hardcode it).
BOUNDARIES = [0.80, 1.00, 1.20, 1.50]   # 5 funded-ratio buckets
FLEX = [1.6, 1.25, 1.0, 0.7, 0.3]        # contribution-flexibility multiplier per bucket

TESTS = [
    dict(T=6,  funded0=1.00, archs=["steady", "calm_bull", "calm_bear"]),
    dict(T=8,  funded0=0.95, archs=["calm_bull", "calm_bear", "choppy"]),
    dict(T=10, funded0=0.85, archs=["steady", "calm_bull", "crash_early", "choppy"]),
    dict(T=12, funded0=1.05, archs=["overfunded_then_crash", "calm_bull", "steady", "choppy"]),
    dict(T=14, funded0=0.75, archs=["underfunded_persist_late_rally", "calm_bear", "steady", "rate_shock_up"]),
    dict(T=16, funded0=0.90, archs=["overfunded_then_crash", "underfunded_persist_late_rally",
                                     "calm_bull", "crash_late", "steady"]),
    dict(T=18, funded0=1.10, archs=["crash_late", "rate_shock_down", "calm_bear", "choppy",
                                     "overfunded_then_crash"]),
    dict(T=20, funded0=0.80, archs=["overfunded_then_crash", "underfunded_persist_late_rally",
                                     "crash_early", "crash_late", "calm_bull", "rate_shock_up"]),
    dict(T=25, funded0=0.70, archs=["underfunded_persist_late_rally", "overfunded_then_crash",
                                     "choppy", "rate_shock_down", "calm_bear", "steady"]),
    dict(T=30, funded0=0.95, archs=["overfunded_then_crash", "underfunded_persist_late_rally",
                                     "crash_early", "crash_late", "rate_shock_up", "rate_shock_down",
                                     "calm_bull"]),
]


def archetype_returns(name, T, rng):
    """One deterministic economic archetype -> T rows of (r_risky, r_safe, dr, g)."""
    out = []
    for t in range(1, T + 1):
        safe = 0.014 + rng.uniform(-0.003, 0.003)
        g = 0.045 + rng.uniform(-0.004, 0.004)
        dr = rng.uniform(-0.0015, 0.0015)
        base = 0.045
        if name == "steady":
            base = 0.045
        elif name == "calm_bull":
            base = 0.075
        elif name == "calm_bear":
            base = 0.005
        elif name == "choppy":
            base = 0.10 if t % 2 == 1 else -0.08
        elif name == "crash_early":
            base = {2: -0.32, 3: -0.20, 4: -0.13}.get(t, 0.065)
        elif name == "crash_late":
            tail = {T - 2: -0.30, T - 1: -0.18, T: -0.11}
            base = tail.get(t, 0.06)
        elif name == "overfunded_then_crash":
            rally_end = max(1, T // 3)
            crash_end = rally_end + 3
            if t <= rally_end:
                base = 0.14
            elif t <= crash_end:
                idx = t - rally_end
                base = {1: -0.32, 2: -0.24, 3: -0.11}.get(idx, -0.11)
            else:
                base = 0.025
        elif name == "underfunded_persist_late_rally":
            split = (2 * T) // 3
            base = -0.03 if t <= split else 0.13
        elif name == "rate_shock_up":
            base = 0.05
            if t in (2, 3):
                dr = 0.010 + rng.uniform(-0.001, 0.001)
        elif name == "rate_shock_down":
            base = 0.05
            if t in (2, 3):
                dr = -0.010 + rng.uniform(-0.001, 0.001)
        r_risky = base + rng.uniform(-0.035, 0.035)
        out.append((r_risky, safe, dr, g))
    return out


def build_instance(test_id):
    spec = TESTS[test_id - 1]
    T = spec["T"]
    funded0 = spec["funded0"]
    A0 = 100.0
    L0 = A0 / funded0
    c_base = 0.006 * L0
    blocks = []
    for i, name in enumerate(spec["archs"]):
        rng = random.Random(1000 * test_id + i)
        blocks.append(archetype_returns(name, T, rng))
    return T, A0, L0, c_base, blocks


def main():
    test_id = int(sys.argv[1])
    T, A0, L0, c_base, blocks = build_instance(test_id)
    M = len(blocks)
    out = []
    out.append(f"{T} {M}")
    out.append(f"{A0:.6f} {L0:.6f} {c_base:.6f}")
    out.append(" ".join(f"{x:.6f}" for x in BOUNDARIES))
    out.append(" ".join(f"{x:.6f}" for x in FLEX))
    for block in blocks:
        for (r_risky, r_safe, dr, g) in block:
            out.append(f"{r_risky:.6f} {r_safe:.6f} {dr:.6f} {g:.6f}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
