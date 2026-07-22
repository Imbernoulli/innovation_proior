# TIER: trivial
# Dump the entire sediment budget in the FINAL release (the checker's baseline).
# It never shapes the router, so the flood channelises straight to the sea edge.
import sys

def main():
    tok = sys.stdin.read().split()
    T = int(tok[2])
    Mtot = float(tok[3])
    sched = [0.0] * (T - 1) + [Mtot]
    print(" ".join("%.6f" % x for x in sched))

main()
