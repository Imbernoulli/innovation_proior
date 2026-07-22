#!/usr/bin/env python3
"""gen.py <testId> -- prints one Quasi-Stationary Casino Floor instance to stdout.
Deterministic: all randomness is seeded ONLY from testId. testId 1..10 is a
size/difficulty ladder (small -> large). Every instance plants the SAME trap
structure (a long low-cashout corridor loop vs. tempting high-cashout feature
stations) so that a per-node greedy split lands far from the cycle-aware optimum.
"""
import random, sys

# (num corridor pits, num feature stations) ladder, small -> large.
SIZES = [
    (6, 2), (8, 3), (10, 3), (14, 4), (20, 6),
    (30, 10), (45, 15), (60, 20), (90, 30), (130, 40),
]

# Global per-arc floors (minimum probability mass any USED door must carry).
F_RING, F_SHORT, F_HRET, F_HRING = 0.03, 0.01, 0.02, 0.02
LO_LO, LO_HI = 0.02, 0.05     # corridor pits: small mandatory cashier floor
HI_LO, HI_HI = 0.80, 0.90     # feature stations: large mandatory cashier floor


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    tid = int(sys.argv[1])
    idx = max(1, min(len(SIZES), tid)) - 1
    Nc, K = SIZES[idx]

    rng = random.Random(1_000_003 * tid + 7)

    delta_lo = [round(rng.uniform(LO_LO, LO_HI), 6) for _ in range(Nc)]
    delta_hi = [round(rng.uniform(HI_LO, HI_HI), 6) for _ in range(K)]
    shortcut_target = [rng.randrange(K) for _ in range(Nc)]
    hubret_target = [rng.randrange(Nc) for _ in range(K)]

    out = []
    out.append(f"{Nc} {K}")
    out.append(f"{F_RING:.6f} {F_SHORT:.6f} {F_HRET:.6f} {F_HRING:.6f}")
    out.append(" ".join(f"{x:.6f}" for x in delta_lo))
    out.append(" ".join(f"{x:.6f}" for x in delta_hi))
    out.append(" ".join(str(x) for x in shortcut_target))
    out.append(" ".join(str(x) for x in hubret_target))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
