# TIER: trivial
# Reproduces the checker's internal baseline: a single mild flat tax.
import sys

def main():
    data = sys.stdin.read().split()
    # (we do not even need to read the population; just emit the baseline)
    print(1)
    print("%.6f %.6f" % (0.0, 0.06))

main()
