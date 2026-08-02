# TIER: greedy
"""The obvious first attempt: cool at a CONSTANT rate from T_start to T_min over
the N batch steps (the textbook "smooth ramp, no shocks" move -- much gentler
than dumping straight to T_min), and pick the seed option with the most total
surface area (count * radius^2), reasoning "more crystal area growing = safer,
consumes supersaturation faster". Both pieces sound reasonable in isolation.

What this ignores: the solubility curve (given in the input) is steep near T_min
and nearly flat near T_start -- most of the crystallizable mass only becomes
available in the LAST slice of the temperature range. A constant cooling rate
spends most of its steps in the flat region doing very little, then rushes
through the steep region late, generating a supersaturation pulse with little
batch time left for the newly formed crystals to grow out -- and on the
high-yield cases that pulse is required to be big, so it frequently detonates
enough nucleation (or misses the yield target outright) to land far below a
schedule that gets into the steep region early instead."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    N = int(nxt())
    T_start = float(nxt())
    T_min = float(nxt())
    for _ in range(6):  # skip kb b kg g r0 kv_rho
        nxt()
    M = int(nxt())
    for _ in range(M):
        nxt(); nxt()
    nxt()  # required_yield
    K = int(nxt())
    lib = []
    for _ in range(K):
        cnt = float(nxt())
        rad = float(nxt())
        lib.append((cnt, rad))

    best_i, best_area = 0, -1.0
    for i, (cnt, rad) in enumerate(lib):
        area = cnt * rad * rad
        if area > best_area:
            best_area = area
            best_i = i
    seed_idx = best_i + 1

    sched = [T_start - (T_start - T_min) * ((t + 1) / N) for t in range(N)]

    out = [str(seed_idx)] + [repr(x) for x in sched]
    print(" ".join(out))


if __name__ == "__main__":
    main()
