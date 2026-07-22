# TIER: trivial
# Do-nothing: gamma=0 => the reinforcement exponent is 0, so every edge keeps
# unit conductance -> the uniform mesh (the checker's internal baseline).
import sys
def main():
    t=sys.stdin.read().split();it=iter(t)
    R=int(next(it));C=int(next(it));S=int(next(it));K=int(next(it))
    [next(it) for _ in range(K)]
    next(it);next(it);M=int(next(it))
    print("0.0")
    print(" ".join(["1.0"]*M))
main()
