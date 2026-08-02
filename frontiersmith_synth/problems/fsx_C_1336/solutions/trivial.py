# TIER: trivial
# Always pick coformer 0 at ratio 1: the checker's own "reference" baseline
# construction. Reproduces B exactly -> Ratio ~= 0.1 on every case.
import sys


def main():
    data = sys.stdin.read()
    if not data.strip():
        return
    print("0 1")


if __name__ == "__main__":
    main()
