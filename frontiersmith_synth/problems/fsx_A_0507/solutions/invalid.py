# TIER: invalid
# Emits infeasible upgrades: cells on the outer border (insulating walls, not open interior
# cells). The checker must reject these -> Ratio 0.0.
import sys

def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    H = int(next(it)); W = int(next(it)); KHI = int(next(it)); K = int(next(it))
    # claim K upgrades, all on the top border row (which are walls / not open) -> infeasible
    out = [str(K)]
    for j in range(K):
        out.append("0 %d" % (j % W))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
