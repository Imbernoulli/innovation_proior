# TIER: trivial
# Do-nothing baseline: ignore the ledger's growth rate and structure entirely,
# predict T(n) ~= n (one op per portion, no batching overhead at all). This
# reproduces the checker's calibrated floor.
import sys


def main():
    sys.stdin.read()  # ignore the ledger
    print("BASE 1 0 0")
    print("REC n")


if __name__ == "__main__":
    main()
