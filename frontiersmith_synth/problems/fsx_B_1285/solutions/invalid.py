# TIER: invalid
# Emits a garbage artifact: a wrong node count and out-of-alphabet decision
# characters. Must score Ratio: 0.0.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    M = 2 ** (T + 1) - 1
    print(M + 3)                 # wrong count
    print("X" * M + "??")        # invalid tokens, wrong length too

if __name__ == "__main__":
    main()
