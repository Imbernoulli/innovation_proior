# TIER: invalid
# Emits a syntactically-shaped but wrong circuit (outputs a single input wire),
# which disagrees with the required outputs -> must score 0.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    # one trivial gate, output it -- will not match the planted function
    sys.stdout.write("1\n")
    sys.stdout.write("ZERO\n")
    sys.stdout.write("OUT %d\n" % n)  # wire n = the ZERO gate -> constant 0 (wrong)


if __name__ == "__main__":
    main()
