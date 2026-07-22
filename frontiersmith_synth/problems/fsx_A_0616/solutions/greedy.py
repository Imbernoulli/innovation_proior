# TIER: greedy
# The recipe: spread the budget evenly over all T releases (uniform schedule).
# Splitting into many pulses already beats a single dump, but a flat schedule
# under-invests early, so it never programs the terrain into an efficient trap.
import sys

def main():
    tok = sys.stdin.read().split()
    T = int(tok[2])
    Mtot = float(tok[3])
    sched = [Mtot / T] * T
    print(" ".join("%.6f" % x for x in sched))

main()
