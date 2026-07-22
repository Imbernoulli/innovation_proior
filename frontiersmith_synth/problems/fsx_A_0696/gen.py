#!/usr/bin/env python3
"""gen.py <testId> -> one instance on stdout.

Theme: barges transiting the bottleneck chamber of a staircase of canal locks.
Family: lock-chamber-green-wave (chamber-batching + direction-alternation +
shared-water-budget).

Barges arrive in ROUNDS aligned to the cycle duration t: round r arrives at tick
r*t and contributes up to C up-bound and C down-bound barges simultaneously (C =
the chamber's per-cycle capacity), so a batching-aware scheduler can clear a whole
round's one-direction backlog in a single cycle. Because both directions become
due-urgent at the same instant every round, a scheduler that always chases the
single most urgent waiting barge is forced to alternate direction almost every
cycle. Switching direction costs far more water (wa) than staying (ws), and the
reservoir regenerates only while the chamber sits IDLE (not while it cycles), so
that chase strategy drains the reservoir after a handful of cycles and strands
the rest of the fleet -- the family's planted trap. Tests 7, 9, 10 use the
tightest due slack (the trap is most severe there); tests 1-3 use generous slack
(gentle); tests 4-6, 8 are in between. All randomness is seeded from testId only.

Instance format (all integers):
    n C L t H W0 rho ws wa
    a_1 d_1 len_1 due_1 wt_1
    ...
    a_n d_n len_n due_n wt_n
"""
import sys

N_LADDER = [8, 10, 14, 18, 24, 30, 38, 48, 60, 75]
TRAP_TESTS = {7, 9, 10}

T_CYCLE = 10  # fixed cycle duration (ticks)
WS = 5        # water cost of a same-direction cycle
WA = 25       # water cost of a direction-switching cycle (5x)
RHO = 1       # reservoir trickle refill per IDLE tick only


class LCG:
    """Tiny deterministic linear-congruential RNG, stable across Python versions."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def nxt(self):
        self.s = (6364136223846793005 * self.s + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.s

    def randint(self, lo, hi):
        span = hi - lo + 1
        return lo + (self.nxt() >> 33) % span


def category(t):
    if t in TRAP_TESTS:
        return "trap"
    if t <= 3:
        return "gentle"
    return "moderate"


def build_instance(t, n):
    C = max(3, n // 12 + 3)
    L = C * 4
    round_size = 2 * C
    num_rounds = (n + round_size - 1) // round_size
    K_needed = 2 * num_rounds
    H = T_CYCLE * (K_needed + 6)
    target_S = max(1, (num_rounds + 1) // 2)
    W0 = K_needed * WS + target_S * (WA - WS) + 15

    cat = category(t)
    if cat == "trap":
        due_slack = 2
    elif cat == "gentle":
        due_slack = 5
    else:
        due_slack = 3

    rng = LCG(900001 + 131 * t + n)

    barges = []
    remaining = n
    for r in range(num_rounds):
        a = r * T_CYCLE
        cnt = min(round_size, remaining)
        remaining -= cnt
        n_dir0 = (cnt + 1) // 2
        n_dir1 = cnt - n_dir0
        if cat == "gentle":
            # long single-direction runs across rounds instead of a 50/50 split
            d_seq = [r % 2] * cnt
        elif cat == "trap":
            d_seq = [i % 2 for i in range(cnt)]
        else:
            d_seq = [0] * n_dir0 + [1] * n_dir1
            rng_shift = rng.randint(0, 1)
            if rng_shift:
                d_seq = d_seq[::-1]
        for d in d_seq:
            ln = rng.randint(1, 4)
            wt = rng.randint(1, 9)
            # per-barge jitter so due-order is NOT just a monotonic re-labelling
            # of arrival-order (otherwise "sort by due" and "sort by arrival"
            # coincide and the due-aware/blind dispatchers behave identically)
            jitter = rng.randint(-(T_CYCLE // 2), T_CYCLE // 2)
            due = a + max(T_CYCLE, due_slack * T_CYCLE + jitter)
            barges.append((a, d, ln, due, wt))

    return n, C, L, T_CYCLE, H, W0, RHO, WS, WA, barges[:n]


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    if t > len(N_LADDER):
        t = len(N_LADDER)
    n = N_LADDER[t - 1]

    n, C, L, tc, H, W0, rho, ws, wa, barges = build_instance(t, n)

    lines = ["%d %d %d %d %d %d %d %d %d" % (n, C, L, tc, H, W0, rho, ws, wa)]
    for (a, d, ln, due, wt) in barges:
        lines.append("%d %d %d %d %d" % (a, d, ln, due, wt))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
