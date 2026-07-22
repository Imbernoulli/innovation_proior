# TIER: invalid
# Infeasible artifact: gamma far outside the allowed range and a nan weight.
import sys
def main():
    t=sys.stdin.read().split();it=iter(t)
    R=int(next(it));C=int(next(it));S=int(next(it));K=int(next(it))
    [next(it) for _ in range(K)]
    next(it);next(it);M=int(next(it))
    print("999.0")
    print(" ".join(["nan"]+["1.0"]*(M-1)))
main()
