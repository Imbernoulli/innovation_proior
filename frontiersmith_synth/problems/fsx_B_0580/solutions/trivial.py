# TIER: trivial
# Uniform equal subsidy: split the budget evenly across every household.
import sys

def main():
    dat = sys.stdin.buffer.read().split()
    n = int(dat[0]); B = int(dat[2])
    u = B // n
    sys.stdout.write(("%d " % u) * n + "\n")

main()
