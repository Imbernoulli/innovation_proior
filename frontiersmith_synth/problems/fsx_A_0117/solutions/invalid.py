# TIER: invalid
# Emits an infeasible layout: too few slots AND an out-of-range value.
# Must score Ratio 0.0 (checker rejects on count/range violation).
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0]); M = int(data[1])
    # Declare n slots but emit garbage: one out-of-range slot, wrong count.
    out = [str(n), str(M + 999), "-5"]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
