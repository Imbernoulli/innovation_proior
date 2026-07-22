# TIER: invalid
# Overspends the budget wildly -> feasibility gate must score 0.
import sys

def main():
    dat = sys.stdin.buffer.read().split()
    n = int(dat[0]); B = int(dat[2])
    sys.stdout.write((("%d " % (B + 1)) * n) + "\n")

main()
