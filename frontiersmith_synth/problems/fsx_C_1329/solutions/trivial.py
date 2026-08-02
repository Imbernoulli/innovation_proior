# TIER: trivial
"""Always use template 0 (the safe reference) run exactly at its own
kinetic sweet spot. Template 0's window is generated to always fully
contain the framework window, so this is always feasible; it reproduces
the checker's internal baseline construction."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = data[p]
        p += 1
        return v

    K = int(nxt())
    c = int(nxt())
    D_target = float(nxt())
    q_target = float(nxt())
    for _ in range(4):
        nxt()
    w1, w2, w3 = float(nxt()), float(nxt()), float(nxt())

    # template 0
    s = float(nxt()); q = float(nxt()); f = float(nxt())
    Tlo = float(nxt()); Thi = float(nxt())
    pHlo = float(nxt()); pHhi = float(nxt())
    Topt = float(nxt()); pHopt = float(nxt())
    R = float(nxt()); r = float(nxt())

    print("0 %.6f %.6f" % (Topt, pHopt))


if __name__ == "__main__":
    main()
