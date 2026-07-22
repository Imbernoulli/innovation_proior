# TIER: trivial
# Remove the first k edges by index -- exactly the checker's baseline construction.
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); m = int(next(it)); k = int(next(it))
    print(" ".join(str(i) for i in range(1, min(k, m) + 1)))

main()
