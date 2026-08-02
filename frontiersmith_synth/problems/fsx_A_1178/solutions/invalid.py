# TIER: invalid
# Ignores the probes entirely and emits a fixed-count garbage vector of
# negative numbers -- fails both the token-count-vs-N-1 check in easy cases
# and the non-negativity / exact-probe-match feasibility checks universally.
import sys


def main():
    sys.stdin.read()
    print("\n".join(["-999.0"] * 5))


if __name__ == "__main__":
    main()
