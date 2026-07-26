# TIER: invalid
# Emits an all-zero schedule: never meets the steam demand (S_t > 0 on every test
# instance), so the checker's feasibility gate must reject it -> Ratio 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    T = int(toks[0])
    out = ["0.0 0.0 0.0 0.0 0.0" for _ in range(T)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
