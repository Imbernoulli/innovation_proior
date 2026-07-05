# TIER: invalid
# Emits a single all-ones term that does NOT reconstruct the tensor -> checker must
# reject it (Ratio 0.0). Demonstrates the feasibility gate.
import sys

def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    P = int(next(it)); m = int(next(it)); n = int(next(it))
    # consume tensor (not needed)
    for _ in range(P * m * n):
        next(it)
    out = ["1"]
    out.append(" ".join(["1"] * m))
    out.append(" ".join(["1"] * n))
    out.append(" ".join(["1"] * P))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
