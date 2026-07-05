# TIER: trivial
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); b = int(next(it))
    flooded = set(next(it) for _ in range(b))
    diag = ["0" * n] + ["".join("1" if j == i else "0" for j in range(n)) for i in range(n)]
    chosen = [c for c in diag if c not in flooded]
    print(len(chosen))
    if chosen:
        print("\n".join(chosen))

main()
