import time
def solve_dp(N):
    INF=10**9
    cubes=[]; b=1
    while b**3<=N: cubes.append(b**3); b+=1
    dp=[INF]*(N+1); dp[0]=0
    for v in range(1,N+1):
        best=INF
        for c in cubes:
            if c>v: break
            x=dp[v-c]+1
            if x<best: best=x
        dp[v]=best
    return dp[N]
t=time.time()
print(solve_dp(10**6))  # warm but in python just to see #cubes
print("cubes upto 1e6:", int(round((10**6)**(1/3))))
print("time(py for 1e6):", time.time()-t)
