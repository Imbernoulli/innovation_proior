# TIER: invalid
# Emits out-of-range entries (2) -> infeasible -> must score 0.
import sys

def main():
    N = int(sys.stdin.read().split()[0])
    out = []
    for i in range(N):
        out.append(" ".join("2" for _ in range(N)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
