import sys

def solve():
    data = sys.stdin.read().split()
    idx = 0
    q = int(data[idx]); idx += 1
    MOD = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        cx = int(data[idx]); cy = int(data[idx+1]); ex = int(data[idx+2]); ey = int(data[idx+3])
        idx += 4
        # Count monotone lattice paths (0,0)->(ex,ey) that pass through (cx,cy),
        # using an exhaustive grid DP (count[x][y] = number of right/up paths to (x,y)).
        # First count paths (0,0)->(cx,cy), then paths (cx,cy)->(ex,ey), multiply.
        def grid_paths(ax, ay, bx, by):
            # number of monotone paths from (ax,ay) to (bx,by)
            W = bx - ax
            H = by - ay
            # dp over a (W+1) x (H+1) grid
            dp = [[0]*(H+1) for _ in range(W+1)]
            for x in range(W+1):
                for y in range(H+1):
                    if x == 0 and y == 0:
                        dp[x][y] = 1
                    else:
                        v = 0
                        if x > 0:
                            v += dp[x-1][y]
                        if y > 0:
                            v += dp[x][y-1]
                        dp[x][y] = v
            return dp[W][H]
        leg1 = grid_paths(0, 0, cx, cy)
        leg2 = grid_paths(cx, cy, ex, ey)
        ans = (leg1 % MOD) * (leg2 % MOD) % MOD
        out.append(str(ans))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

solve()
