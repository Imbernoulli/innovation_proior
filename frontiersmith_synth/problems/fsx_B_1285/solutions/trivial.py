# TIER: trivial
# Settle at the very first opportunity, unconditionally. This reproduces the
# checker's own internal baseline B exactly (Ratio ~= 0.1).
import sys

def depth(idx):
    return (idx + 1).bit_length() - 1

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    for _ in range(T):
        next(it)
    M = 2 ** (T + 1) - 1
    # decision for every node; only node 0's decision ever matters here
    policy = ["S"] * M
    print(M)
    print(" ".join(policy))

if __name__ == "__main__":
    main()
