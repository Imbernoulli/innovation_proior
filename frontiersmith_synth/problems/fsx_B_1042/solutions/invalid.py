# TIER: invalid
"""Deliberately infeasible: claims every pond grows the entire horizon at full
declared feed cap simultaneously (violates the shared feed cap constraint)."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it))
    T = int(next(it))
    C = float(next(it))
    for _ in range(P):
        next(it); next(it); next(it); next(it); next(it)

    out = [str(P)]
    for _p in range(P):
        out.append(f"0 {T}")
        out.append(" ".join(f"{C:.6f}" for _ in range(T)))  # each pond takes the FULL cap alone
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
