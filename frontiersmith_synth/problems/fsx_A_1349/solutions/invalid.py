# TIER: invalid
import sys


def main():
    sys.stdin.read()  # instance is irrelevant to this deliberately broken machine
    # A single state that self-loops on every symbol, including blank: it
    # never reaches ACCEPT or REJECT, so it must time out under the step
    # bound on every pair -> checker must score this Ratio: 0.0.
    sys.stdout.write("1\n0 0 0\n")


if __name__ == "__main__":
    main()
