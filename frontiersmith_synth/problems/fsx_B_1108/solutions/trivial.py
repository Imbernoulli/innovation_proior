# TIER: trivial
# Baseline construction: place the k sinks at junctions 0..k-1, exactly
# reproducing the checker's internal baseline (the shell block of feeble,
# strongly-radiating sources that never needed help) -> scores ~0.1.
import sys


def main():
    it = sys.stdin.read().split()
    k = int(it[2])
    out = [str(k)] + [str(i) for i in range(k)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
