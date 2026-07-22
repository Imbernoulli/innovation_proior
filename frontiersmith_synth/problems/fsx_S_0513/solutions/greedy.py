# TIER: greedy
# The obvious approach: "why would removing roads ever help?" -- keep the WHOLE
# network and let equilibrium sort it out. This walks straight into every Braess
# trap (the tempting shortcuts stay in and inflate the selfish-routing cost).
import sys


def main():
    raw = sys.stdin.read().split()
    it = iter(raw)
    N = int(next(it)); M = int(next(it))
    print(" ".join(str(i) for i in range(1, M + 1)))


main()
