# TIER: trivial
# Flat overnight charge: hold one constant rate for the whole window (one long
# session, no attempt to exploit cheap prices or the quadratic loss shape).
# This reproduces the checker's own internal baseline construction.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    dt = float(next(it))
    Rmax = float(next(it))
    Fee = float(next(it))
    D = float(next(it))
    r_min = float(next(it))
    prices = [float(next(it)) for _ in range(T)]
    alpha = [float(next(it)) for _ in range(T)]
    E_target = float(next(it))

    r_avg = max(E_target / (T * dt), r_min)
    print(" ".join(f"{r_avg:.6f}" for _ in range(T)))


if __name__ == "__main__":
    main()
