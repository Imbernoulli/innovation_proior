# TIER: invalid
# Emits a plan that forces a head-on meet: every train enters every block at
# tick 0, so opposing trains occupy the same single-track block simultaneously.
# The checker must reject this -> Ratio 0.0.
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    S = int(next(it)); H = int(next(it)); TMAX = int(next(it))
    cap = [int(next(it)) for _ in range(S)]
    N = int(next(it))
    for _ in range(N):
        int(next(it)); int(next(it)); int(next(it)); int(next(it)); int(next(it))
    line = " ".join(["0"] * (S - 1))
    out = [line for _ in range(N)]
    sys.stdout.write("\n".join(out) + "\n")


main()
