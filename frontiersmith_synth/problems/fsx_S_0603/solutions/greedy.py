# TIER: greedy
# Obvious recipe: reinforce the network under THE demand it must carry -- the
# single aggregate load (all sinks drawn together) -- and tune the exponent for
# transport efficiency.  Ignores the fluctuating single-sink regime, so it prunes
# a fragile trunk on spread layouts.
import sys
def main():
    t=sys.stdin.read().split();it=iter(t)
    R=int(next(it));C=int(next(it));S=int(next(it));K=int(next(it))
    [next(it) for _ in range(K)]
    next(it);next(it);M=int(next(it))
    w=["0.0"]*M
    w[0]="1.0"           # scenario 0 = aggregate load
    print("0.35")
    print(" ".join(w))
main()
