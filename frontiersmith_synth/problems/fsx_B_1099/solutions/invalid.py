# TIER: invalid
# A negative thigmotropism weight -- physically nonsensical and explicitly
# barred by the feasibility gate (all weights must be >= 0), so the checker
# rejects it with Ratio: 0.0 regardless of the (otherwise well-formed) rest
# of the output.
import sys


def main():
    sys.stdin.read()
    print("1.0 1.0 -1.0")
    print("0")


if __name__ == "__main__":
    main()
