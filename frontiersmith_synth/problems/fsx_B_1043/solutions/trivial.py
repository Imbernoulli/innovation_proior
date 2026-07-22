# TIER: trivial
"""Weld struts in input order, every side +1 -- exactly the checker's own
reference baseline construction. No attempt at cancellation at all."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    for _ in range(m):
        next(it); next(it); next(it)  # skip u w eff
    out = [str(m)]
    for i in range(m):
        out.append(f"{i} 1")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
