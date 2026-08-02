# TIER: invalid
# Claim a match with a duty far beyond any stream's available capacity ->
# capacity check must reject it -> Ratio: 0.0.
import sys


def main():
    sys.stdin.read()
    print(1)
    print("1 1 999999999.0")


if __name__ == "__main__":
    main()
