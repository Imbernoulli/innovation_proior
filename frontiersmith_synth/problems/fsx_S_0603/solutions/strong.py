# TIER: strong
# Insight: drive the SAME remodelling rule with a FLUCTUATING mix of loads --
# the aggregate load *and* every single-sink and half-group load, all given a
# turn -- instead of the single steady aggregate load alone. Letting each sink
# reinforce its own path some of the time preserves the redundant loops that a
# steady aggregate-only load prunes away, without giving up on the aggregate's
# own efficiency signal.
import sys
def main():
    t=sys.stdin.read().split();it=iter(t)
    R=int(next(it));C=int(next(it));S=int(next(it));K=int(next(it))
    [next(it) for _ in range(K)]
    next(it);next(it);M=int(next(it))
    w=[0.0]*M
    w[0]=1.0                    # scenario 0 = aggregate load, still gets a turn
    # scenarios 1..K are the single-sink (fluctuating) loads; K+1,K+2 half-groups.
    for m in range(1,K+1):
        w[m]=1.0
    for m in range(K+1,M):
        w[m]=0.5
    print("0.35")
    print(" ".join("%.4f"%x for x in w))
main()
