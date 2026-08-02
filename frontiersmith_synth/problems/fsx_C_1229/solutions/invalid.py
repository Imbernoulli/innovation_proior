# TIER: invalid
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    V = int(next(it)); T = int(next(it))
    variants = []
    for _ in range(V):
        fleet = int(next(it))
        p = float(next(it)); reward = float(next(it)); penalty = float(next(it))
        D = int(next(it)); W = float(next(it))
        variants.append((fleet, p, reward, penalty, D, W))

    out = []
    for (fleet, p, reward, penalty, D, W) in variants:
        # Deliberately infeasible: schedule the WHOLE fleet in a single stage 10x
        # over the fleet's own size (impossible cohort sum) at an out-of-range time.
        out.append("1")
        out.append("%d %d" % (T + 5, fleet * 10 + 1000))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
