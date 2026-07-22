# TIER: trivial
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); c = int(next(it))
    # do nothing: no injection at all
    print(" ".join("0" for _ in range(n)))

main()
