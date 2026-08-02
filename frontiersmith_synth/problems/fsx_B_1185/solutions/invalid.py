# TIER: invalid
# Emits an out-of-range artifact: a negative location and an over-cap
# severity -- the checker must reject it with Ratio 0.0 regardless of the
# instance (both bounds are violated: x_hat < 0 and s_hat > S_MAX_OUT=0.5).
import sys


def main():
    sys.stdin.read()
    sys.stdout.write("-999.0 999.0\n")


if __name__ == "__main__":
    main()
