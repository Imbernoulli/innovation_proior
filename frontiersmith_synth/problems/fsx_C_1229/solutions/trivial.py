# TIER: trivial
import sys, math

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    V = int(next(it)); T = int(next(it))
    variants = []
    for _ in range(V):
        fleet = int(next(it))
        p = float(next(it))
        reward = float(next(it))
        penalty = float(next(it))
        D = int(next(it))
        W = float(next(it))
        variants.append((fleet, p, reward, penalty, D, W))

    out = []
    for (fleet, p, reward, penalty, D, W) in variants:
        # One cautious single-shot cohort at t=0, sized exactly at the risk-budget cap.
        c = math.floor(W / p) if p > 0 else 0
        c = min(c, fleet)
        if c > 0:
            out.append("1")
            out.append("0 %d" % c)
        else:
            out.append("0")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
