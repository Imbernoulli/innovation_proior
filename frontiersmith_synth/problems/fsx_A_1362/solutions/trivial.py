# TIER: trivial
"""Equal split: every agent gets exactly 1/n of every item. Always feasible and
always envy-free (every agent's bundle is literally identical), reproducing the
checker's own baseline B exactly."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    for _ in range(n):
        for _ in range(m):
            next(it)  # valuations unused by this construction
    share = 1.0 / n
    out_lines = []
    for _j in range(m):
        out_lines.append(" ".join(f"{share:.9f}" for _ in range(n)))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
