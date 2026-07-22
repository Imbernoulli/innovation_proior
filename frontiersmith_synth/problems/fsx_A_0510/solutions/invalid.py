# TIER: invalid
# Emits out-of-range residues (>= n) -> the checker's feasibility gate rejects -> 0.
import sys

def main():
    tk = sys.stdin.buffer.read().split()
    it = iter(tk)
    t = int(next(it)); ms = [int(next(it)) for _ in range(t)]; k = int(next(it))
    n = 1
    for m in ms:
        n *= m
    print(k)
    for _ in range(k):
        print(n + 7)

if __name__ == "__main__":
    main()
