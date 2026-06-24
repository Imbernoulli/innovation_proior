import random

def powers_upto(N,k):
    res=[]
    b=1
    while b**k <= N:
        res.append(b**k); b+=1
    return res

def solve_dp(N,k):
    INF=float('inf')
    P=powers_upto(N,k)
    dp=[INF]*(N+1); dp[0]=0
    for v in range(1,N+1):
        best=INF
        for p in P:
            if p>v: break
            if dp[v-p]+1<best: best=dp[v-p]+1
        dp[v]=best
    return dp[N]

def solve_greedy(N,k):
    P=powers_upto(N,k)
    v=N; steps=0
    while v>0:
        # take largest power <= v
        for p in reversed(P):
            if p<=v:
                v-=p; steps+=1; break
    return steps

# count traps for k=2 and k=3
for k in [2,3]:
    traps=0; ex=[]
    for N in range(1,500):
        a=solve_dp(N,k); b=solve_greedy(N,k)
        if a!=b:
            traps+=1
            if len(ex)<6: ex.append((N,a,b))
    print("k=%d traps in 1..499: %d examples %s"%(k,traps,ex))
