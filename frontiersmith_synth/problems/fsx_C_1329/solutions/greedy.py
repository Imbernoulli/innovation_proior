# TIER: greedy
"""The obvious textbook approach: rank templates purely by geometric
structure-directing fit (SDI) and pick the single best-fitting one, then
run it at ITS OWN preferred (Topt, pHopt) point. This treats "pick the
template" and "check it can actually be synthesized" as a sequential
two-step process -- it never looks at whether that template's preferred
conditions are reachable inside the framework's crystallization window,
nor at its removal cost, until after the choice is locked in."""
import sys

Q_NORM = 2.0
F_NORM = 1.0


def f_ideal(c):
    return 0.15 * c + 0.1


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

    fid = f_ideal(c)
    best_idx = -1
    best_sdi = -1.0
    best_pt = None

    for i in range(K):
        s = float(nxt()); q = float(nxt()); f = float(nxt())
        Tlo = float(nxt()); Thi = float(nxt())
        pHlo = float(nxt()); pHhi = float(nxt())
        Topt = float(nxt()); pHopt = float(nxt())
        R = float(nxt()); r = float(nxt())

        size_m = max(0.0, 1.0 - abs(s - D_target) / D_target)
        charge_m = max(0.0, 1.0 - abs(q - q_target) / Q_NORM)
        shape_m = max(0.0, 1.0 - abs(f - fid) / F_NORM)
        val = w1 * size_m + w2 * charge_m + w3 * shape_m

        if val > best_sdi:
            best_sdi = val
            best_idx = i
            best_pt = (Topt, pHopt)

    print("%d %.6f %.6f" % (best_idx, best_pt[0], best_pt[1]))


if __name__ == "__main__":
    main()
