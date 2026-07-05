# TIER: invalid
# Emits overlapping zones that also cross the central cordon -> must score 0.
import sys

def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    k = min(N, 3)
    out = [str(k)]
    for _ in range(k):
        out.append("0.0 0.0 0.95")   # all stacked at the origin, inside the cordon
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
