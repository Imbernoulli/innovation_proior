# TIER: invalid
# Emits an infeasible schedule: a negative production value and wrong count.
# Must be rejected by the checker -> Ratio 0.0.
import sys

def main():
    n, V = map(int, sys.stdin.read().split())
    sys.stdout.write("-5 %d 0\n" % (V + 999))

if __name__ == "__main__":
    main()
