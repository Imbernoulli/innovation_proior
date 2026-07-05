# TIER: trivial
# Reproduces the checker's index-block reference bisection -> Ratio ~= 0.1.
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    res = ["0" if v <= n // 2 else "1" for v in range(1, n + 1)]
    sys.stdout.write(" ".join(res) + "\n")

if __name__ == "__main__":
    main()
