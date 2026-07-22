#!/usr/bin/env python3
"""
gen.py <testId>  ->  ONE instance of the "drum-corps bridge" comb-dodging-stiffness
problem: a discretized bridge (S stiffness segments / S-1 free masses, fixed-fixed
ends) and a marching-drum forcing comb (frequency + weight lines).

Physical model (fixed-fixed uniform mass-spring chain, S identical springs of
stiffness BASE_K, S-1 unit point masses) has a CLOSED-FORM eigenfrequency spectrum:
    omega_m = 2*sqrt(BASE_K)*sin(m*pi / (2*(N+1))),   N = S-1,  m = 1..N
The drum corps' forcing comb is planted EXACTLY on this uniform-baseline spectrum
(heavier weight on the middle third of modes) -- see AGENT_BRIEF: this is the trap.
Any solver that just "shifts" stiffness uniformly reproduces the same spectrum and
collides head-on with every comb line.

Difficulty ladder (testId 1..10 -> S, more segments = more modes = a denser comb):
    S = 6, 6, 8, 8, 10, 10, 12, 14, 16, 20
Fully deterministic: the instance is a pure function of testId (no RNG at all --
BASE_K is a fixed constant, so nothing needs seeding).
"""
import sys, math

LADDER = [6, 6, 8, 8, 10, 10, 12, 14, 16, 20]
BASE_K = 12  # baseline per-segment stiffness unit (uniform reference beam)


def uniform_omegas(S, k):
    """Closed-form eigenfrequencies of the fixed-fixed UNIFORM chain: S identical
    springs (stiffness k), S-1 unit masses.  omega_m = 2*sqrt(k)*sin(m*pi/(2*(N+1)))."""
    N = S - 1
    return [2.0 * math.sqrt(k) * math.sin(m * math.pi / (2.0 * (N + 1))) for m in range(1, N + 1)]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    S = LADDER[(t - 1) % len(LADDER)]
    BUDGET = S * BASE_K
    N = S - 1

    om = uniform_omegas(S, BASE_K)
    om_sorted = sorted(om)
    gaps = [om_sorted[i + 1] - om_sorted[i] for i in range(len(om_sorted) - 1)]
    spacing = (sum(gaps) / len(gaps)) if gaps else om_sorted[0]
    eps = 0.5 * spacing  # scoring resolution: half the average uniform-mode gap

    # target band = middle third of the mode index range (heavier comb weight);
    # the rest of the spectrum still carries light weight (so a fix must not
    # merely trade the target band's resonance for worse resonance elsewhere).
    lo, hi = N // 3, (2 * N) // 3
    if hi <= lo:
        hi = lo + 1
    idx_target = set(range(lo, hi))

    lines = []
    for i in range(N):
        w = 8 if i in idx_target else 1
        lines.append((om[i], w))
    lines.sort(key=lambda p: p[0])  # deterministic order, no dict/set iteration leakage

    print(S, BUDGET)
    print("%.10f" % eps)
    print(len(lines))
    for f, w in lines:
        print("%.10f %d" % (f, w))


if __name__ == "__main__":
    main()
