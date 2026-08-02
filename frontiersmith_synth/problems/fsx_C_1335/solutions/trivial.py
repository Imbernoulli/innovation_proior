# TIER: trivial
"""Reproduces the checker's own baseline exactly: slow CONTINUOUS nucleation --
hold at a constant mid-level heat (index 2) and feed the whole precursor budget
evenly across every step, using the weakest surfactant. Always feasible, never
reasons about timing at all -- this IS the checker's reference construction, so
it scores ~0.1 by definition."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]; pos += 1
        return v

    T = int(nxt()); L = int(nxt()); S = int(nxt())
    nxt(); nxt()          # r0 v0 (unused)
    nxt(); nxt()          # theta_ripen ripening_rate (unused)
    for _ in range(L):
        nxt(); nxt(); nxt()  # thr cap gcoef (unused)
    for _ in range(S):
        nxt(); nxt()         # bind p (unused)
    C0 = float(nxt())
    nxt()                    # max_inject (unused)
    nxt(); nxt()              # target disp_limit (unused)

    temp = [2] * T
    inject = [C0 / T] * T
    surf = 0

    out = [" ".join(map(str, temp)), " ".join(f"{x:.6f}" for x in inject), str(surf)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
