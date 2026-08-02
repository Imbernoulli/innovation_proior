# TIER: greedy
"""Textbook first attempt: "crank the heat and feed steadily." Hold the reactor
at the HOTTEST available level for the whole run (fastest possible growth rate,
lowest possible nucleation threshold) and feed the precursor budget evenly
across every step, picking a plausible middle-of-the-road surfactant.

This legitimately beats the slow/mid-heat trivial baseline (faster growth,
easier to get particles moving). But it never separates nucleation from
growth in TIME: because the threshold at the hottest level is low, each new
top-up keeps pushing the monomer pool back over threshold, so nucleation
re-fires every few steps for the entire run instead of happening once. The
result is particles of many different ages/final sizes -- a broad
distribution that only partially overlaps a tight target window, and whose
non-uniform ages make it vulnerable to Ostwald-ripening redistribution
whenever the pool dips low, which further separates the early (big) cohorts
from the late (small) ones instead of leaving a narrow, single-age batch."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]; pos += 1
        return v

    T = int(nxt()); L = int(nxt()); S = int(nxt())
    nxt(); nxt()
    nxt(); nxt()
    for _ in range(L):
        nxt(); nxt(); nxt()
    for _ in range(S):
        nxt(); nxt()
    C0 = float(nxt())
    nxt()
    nxt(); nxt()

    temp = [L - 1] * T
    inject = [C0 / T] * T
    surf = S // 2

    out = [" ".join(map(str, temp)), " ".join(f"{x:.6f}" for x in inject), str(surf)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
