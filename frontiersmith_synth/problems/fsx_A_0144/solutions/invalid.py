# TIER: invalid
# Infeasible: places every station at the same point far outside the reservoir.
# Fails the containment / coincidence checks -> score 0.
import sys

def main():
    n = int(sys.stdin.read().split()[0])
    out = ["3.0 3.0" for _ in range(n)]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
