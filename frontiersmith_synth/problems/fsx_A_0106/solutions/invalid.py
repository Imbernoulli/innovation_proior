# TIER: invalid
# Emit an out-of-alphabet artifact (all zeros): every token is 0, which is not in
# {-1, +1}, so the feasibility gate rejects it and the score is Ratio: 0.0.
import sys

def main():
    n = int(sys.stdin.read().split()[0])
    rows = ["0 " * n for _ in range(n)]
    sys.stdout.write("\n".join(r.strip() for r in rows) + "\n")

if __name__ == "__main__":
    main()
