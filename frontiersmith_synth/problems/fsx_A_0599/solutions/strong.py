# TIER: strong
# INSIGHT: the corner sweep gives every part of the SAME catalog value one shared
# signed perturbation. So if a divider's top and bottom are both built from the SAME
# value, both scale by the identical factor and the ratio b/(a+b) is EXACTLY invariant
# at every corner -- the correlated error cancels structurally. A nominal-fit search
# never sees this. We therefore realize each target as a matched integer ratio
# b/(a+b) of value[0], picking the best (a,b) with a+b <= DCAP within the part budget.
# Approximation residual is the only error left (corner spread = 0), which the moderate
# DCAP keeps above zero -> leaves headroom above this reference.
import sys

DCAP = 15   # reference resolution cap (RL can push higher for a better score)

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    K = int(next(it)); M = int(next(it)); P = int(next(it))
    C = int(next(it)); TPM = int(next(it))
    targets = [float(next(it)) for _ in range(K)]
    catalog = [int(next(it)) for _ in range(M)]

    # per-tap best matched ratio b/den, den in 2..DCAP, 1<=b<=den-1
    picks = []
    for k in range(K):
        r = targets[k]
        best = None
        for den in range(2, DCAP + 1):
            b = round(r * den)
            b = min(max(b, 1), den - 1)
            e = abs(b / den - r)
            if best is None or e < best[0] - 1e-12 or (abs(e - best[0]) <= 1e-12 and den < best[1]):
                best = (e, den, b)
        _, den, b = best
        picks.append((den, b))

    # respect the total part budget (den sum). Trim the largest denominators if needed.
    while sum(den for den, _ in picks) > P:
        # shrink the currently largest-den tap toward a coarser matched ratio
        kmax = max(range(K), key=lambda k: picks[k][0])
        den, b = picks[kmax]
        r = targets[kmax]
        nd = den - 1
        if nd < 2:
            break
        nb = min(max(round(r * nd), 1), nd - 1)
        picks[kmax] = (nd, nb)

    lines = []
    for k in range(K):
        den, b = picks[k]
        a = den - b
        top = [0] * M; bot = [0] * M
        top[0] = a; bot[0] = b
        lines.append(" ".join(map(str, top + bot)))
    sys.stdout.write("\n".join(lines) + "\n")

main()
