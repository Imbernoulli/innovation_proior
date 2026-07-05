# TIER: invalid
# Emits 0s -- outside the {-1,+1} alphabet -- so the feasibility gate rejects it
# and it scores Ratio: 0.0.
import sys

def main():
    n = int(sys.stdin.read().split()[0])
    rows = [" ".join(["0"] * n) for _ in range(n)]
    sys.stdout.write("\n".join(rows) + "\n")

if __name__ == "__main__":
    main()
