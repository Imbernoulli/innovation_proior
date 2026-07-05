# TIER: invalid
# Emits entries outside {-1,+1} (all zeros), so the feasibility gate rejects it -> Ratio 0.0.
import sys

def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    rows = [" ".join(["0"] * n) for _ in range(n)]
    sys.stdout.write("\n".join(rows) + "\n")

if __name__ == "__main__":
    main()
