# TIER: invalid
# Emits an infeasible sequence: an input vertex is not an intermediate, so the
# set-equality feasibility check fails -> checker must score 0.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    V = int(next(it)); E = int(next(it)); M = int(next(it)); N = int(next(it))
    intermediates = list(range(M, V - N))
    if intermediates:
        intermediates[-1] = 0  # replace one intermediate with input id 0 -> invalid set
    sys.stdout.write(" ".join(map(str, intermediates)) + "\n")


if __name__ == "__main__":
    main()
