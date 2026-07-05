# TIER: invalid
# Emits an infeasible family: it includes the all-0, all-1 and all-2 schedules,
# which are all-distinct at every intersection -> a grid-lock resonance (a line).
# The checker must reject this and score 0.
import sys, itertools


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    out = ["0" * n, "1" * n, "2" * n]
    # pad with some {0,1} vectors so it is not trivially tiny
    for bits in itertools.product("01", repeat=n):
        out.append("".join(bits))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
