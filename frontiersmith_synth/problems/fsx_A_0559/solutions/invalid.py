# TIER: invalid
# Emits an out-of-range schedule (marginal rate > RATE_MAX and a non-zero first
# threshold) -> the checker must reject it and score 0.
import sys

def main():
    sys.stdin.read()
    print(2)
    print("5.000000 1.500000")   # first threshold != 0 and rate > 0.95
    print("100.000000 2.000000")

main()
