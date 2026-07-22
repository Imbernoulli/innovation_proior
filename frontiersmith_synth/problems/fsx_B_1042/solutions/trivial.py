# TIER: trivial
"""Reproduces the checker's own baseline construction: feed the WHOLE line to
pond 0 for the entire horizon and ignore every other pond in the fleet (each
of them harvests immediately at t=0, with zero growth)."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it))
    T = int(next(it))
    C = float(next(it))
    for _ in range(P):
        next(it); next(it); next(it); next(it); next(it)  # a, b0, e0, decay, tau -- unused

    out = [str(P)]
    out.append(f"0 {T}")
    out.append(" ".join(f"{C:.6f}" for _ in range(T)))
    for _p in range(P - 1):
        out.append("0 0")
        out.append("")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
