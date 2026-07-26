# TIER: invalid
# Emits a stack whose non-final layer thicknesses already reach/exceed the
# fixed sensor depth D (leaving no room for the mandatory final layer), which
# the checker's feasibility gate must reject outright -> Ratio: 0.0.
import sys


def main():
    sys.stdin.read()  # ignore input
    print(3)
    print("6.0 1.40")
    print("6.0 1.20")
    print("2.0 1.10")


if __name__ == "__main__":
    main()
