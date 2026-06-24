import sys
from functools import lru_cache

# Problem: reduce N to 1. Operations:
#  - subtract 1 (v -> v-1), cost 1   (only if v>=2 of course, result >=1)
#  - if v % p == 0 for an allowed prime p, v -> v/p, cost 1
# allowed primes P. Minimize steps to reach 1.

def solve_dp(N, P):
    # dp over 1..N
    INF = float('inf')
    dp = [INF]*(N+1)
    dp[1]=0
    for v in range(2,N+1):
        best = dp[v-1]+1
        for p in P:
            if v % p == 0:
                best = min(best, dp[v//p]+1)
        dp[v]=best
    return dp[N]

def solve_greedy(N, P):
    Ps = sorted(P, reverse=True)
    v=N; steps=0
    while v>1:
        divided=False
        for p in Ps:
            if v % p ==0:
                v//=p; steps+=1; divided=True; break
        if not divided:
            v-=1; steps+=1
    return steps

import random
trap_found=0
for it in range(20000):
    N = random.randint(2,200)
    k = random.randint(1,3)
    P = random.sample([2,3,5,7,11,13], k)
    a = solve_dp(N,P)
    b = solve_greedy(N,P)
    if a!=b:
        trap_found+=1
        if trap_found<=10:
            print("TRAP N=%d P=%s dp=%d greedy=%d"%(N,P,a,b))
print("traps:",trap_found)
