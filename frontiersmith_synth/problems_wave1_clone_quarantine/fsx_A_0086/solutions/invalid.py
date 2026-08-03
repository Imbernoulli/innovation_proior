# TIER: invalid
# Emits entries outside {-1,+1} (uses 0s), so the feasibility gate rejects it -> Ratio 0.0.
import sys

def main():
    n = int(sys.stdin.read().split()[0])
    rows = ["0 " * n for _ in range(n)]
    sys.stdout.write("\n".join(r.strip() for r in rows) + "\n")

if __name__ == "__main__":
    main()
