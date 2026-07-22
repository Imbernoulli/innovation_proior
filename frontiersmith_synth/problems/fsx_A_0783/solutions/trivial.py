# TIER: trivial
"""Do nothing: zero doses everywhere. Matches the checker's own baseline construction
exactly, so this reproduces Ratio ~= 0.1 by design."""
import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    print(" ".join("0" for _ in range(N)))

if __name__ == "__main__":
    main()
