# TIER: invalid
# Emits an over-budget schedule (twice the sediment budget) -> infeasible -> 0.
import sys

def main():
    tok = sys.stdin.read().split()
    T = int(tok[2])
    Mtot = float(tok[3])
    sched = [2.0 * Mtot / T] * T   # total = 2*Mtot, violates the mass budget
    print(" ".join("%.6f" % x for x in sched))

main()
