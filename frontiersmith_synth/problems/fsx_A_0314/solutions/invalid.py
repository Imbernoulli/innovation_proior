# TIER: invalid
# Emits turbines far outside the field -> feasibility gate must reject it (score 0).
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    out = ["3.0 3.0" for _ in range(n)]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
