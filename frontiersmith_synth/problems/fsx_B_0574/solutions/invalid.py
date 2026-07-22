# TIER: invalid
# Emits an infeasible artifact (engine count 0 and non-finite propellant) so the
# grader's feasibility gate must reject it -> 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    S = int(float(toks[0]))
    out = ["0 nan" for _ in range(S)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
