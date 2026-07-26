# TIER: trivial
"""Feed the single cheapest tank every day of the whole horizon. This is exactly the
checker's own baseline construction: 'keep one culture going, all the time'."""
import sys


def main():
    tokens = sys.stdin.read().split()
    idx = 0

    def nxt(k):
        nonlocal idx
        vals = tokens[idx: idx + k]
        idx += k
        return [int(x) for x in vals]

    T, H, M, BUDGET = nxt(4)
    costs = []
    for _ in range(T):
        row = nxt(6)
        costs.append(row[5])
    # orders unused by this tier

    cheapest = min(range(T), key=lambda i: (costs[i], i))
    lines = [str(H)]
    for d in range(H):
        lines.append(f"{cheapest} {d}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
