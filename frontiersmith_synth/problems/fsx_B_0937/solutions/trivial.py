# TIER: trivial
import sys

# Reads (but does not fit) the train sample and emits a blind, uncalibrated
# guess for the area-scaling coefficient: "some textbook-ish 1/(4*pi)-ish
# number", applied with NO perimeter term and NO data fitting at all. This
# reproduces the checker's own internal baseline construction.


def main():
    sys.stdin.read()  # consume the train sample (unused)
    print("0.092 * a * b * lam")


if __name__ == "__main__":
    main()
