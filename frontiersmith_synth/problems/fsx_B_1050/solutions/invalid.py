# TIER: invalid
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))

    T = N - K
    # Structurally the right shape (correct merge count) but the very first
    # merge references cluster id 0, which is never active (labels start at
    # 1) -- must be rejected by the feasibility check, not merely by a
    # count mismatch.
    out = [str(T)]
    out.append("0 1")
    for t in range(2, T + 1):
        out.append(f"{t} {t}")  # also self-merges, doubly infeasible
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
