I would call this method the Aliens trick, the standard name inherited from the IOI 2016 problem "Aliens" that made the technique famous. It is a Lagrangian-relaxation, or parametric-search, device for exact-count dynamic programs whose optimal-value sequence is discretely convex.

The underlying task is to cover a collection of intervals with exactly K larger intervals while minimizing the sum of squared spans. After preprocessing—sorting by left endpoint, breaking ties by decreasing right endpoint, and discarding every interval that is contained in an earlier kept interval—the remaining left endpoints and right endpoints are both strictly increasing. This ordering means that any optimal solution can be viewed as choosing a sequence of contiguous blocks of the reduced intervals. If the last chosen block starts at t and ends at i-1, its incremental contribution is the square of the span r[i-1]-l[t]+1, minus any overlap that was already paid for by the previous chosen block, namely max(0, r[t-1]-l[t]+1)^2, with the convention that the overlap is zero when t=0.

The exact-count recurrence is therefore f[i][j] = min over t<i of f[t][j-1] + cost(t,i), with f[0][0]=0 and the natural infinities for impossible states. Solving this directly costs O(n^2 K) time, which is prohibitive when K can be large. The first simplification is that for a fixed layer j the minimization over t is one-dimensional, and cost(t,i) can be expanded into a line in the variable x=r[i-1]. That turns each fixed-count layer into an O(n log n) lower-envelope problem, but it does not remove the iteration over the count coordinate.

The decisive observation is that the sequence F(k)=f[n][k] of optimal values for exactly k intervals is discretely convex. In symbols, F(k-1)-F(k) >= F(k)-F(k+1); equivalently, the marginal saving from adding one more interval is non-increasing as k grows. The first few intervals can exploit obvious clusters, but once those clusters are exhausted each additional interval saves no more than the previous one. This shape is what makes a penalty on the number of intervals behave monotonically.

Rather than forcing exactly K intervals, I add a price C to every chosen interval and solve the unconstrained problem G(C)=min_k (F(k)+C k). Inside the prefix DP, this only changes the transition by adding C to each split; the line expansion from the fixed-layer observation remains valid, and I simply track the number of intervals used by each candidate solution. When two candidates give the same penalized value, I keep the one with the larger count, because the binary search will rely on the rightmost minimizer.

Because of convexity, increasing C makes extra intervals less attractive, so the optimal count is monotone non-increasing in C. At C=0 intervals are free, so the optimum uses as many useful intervals as possible; at C equal to the square of the maximum span, a single interval is always sufficient. I binary-search the largest integer C whose optimal count is still at least K. If S_k=F(k-1)-F(k) denotes the saving from the k-th interval, this search lands on a value C with S_K >= C and S_K < C+1; since the costs are integers, S_K=C. Thus K belongs to the contact set of the line y=C k + G(C) with the convex chain (k,F(k)).

The answer is recovered as F(K)=G(C)-C K. It is essential to subtract C times the requested K, not C times the count that happened to be returned by the tie-breaker. If the probe at C returns exactly K the formula is immediate. If the returned count jumps from above K to below K at the next integer price, convexity implies that the skipped counts all lie on the same linear segment of the chain, so F(k)+C k is constant across that segment and the same tangent evaluation works for every k in it, including K.

This transforms the original O(n^2 K) computation into O(log M) probes, each an O(n log n) lower-envelope sweep, where M bounds the largest possible span. For verification and illustration, the Python script below uses a direct O(n^2) probe instead of a Li Chao tree, compares the Aliens-trick answer against a brute-force exact-count DP on small random instances, and prints the convex sequence F(k) so the shrinking marginal savings are visible.

```python
import random

def reduce_intervals(row, col):
    seg = []
    for a, b in zip(row, col):
        l, r = min(a, b), max(a, b)
        seg.append((l, r))
    seg.sort(key=lambda x: (x[0], -x[1]))
    keep = []
    far = -10**9
    for l, r in seg:
        if r > far:
            keep.append((l, r))
            far = r
    return keep

def block_cost(l, r, t, i):
    span = r[i - 1] - l[t] + 1
    if t == 0:
        ov = 0
    else:
        ov = max(0, r[t - 1] - l[t] + 1)
    return span * span - ov * ov

def brute_force(row, col, K):
    seg = reduce_intervals(row, col)
    n = len(seg)
    if n == 0 or K <= 0:
        return 0
    K = min(K, n)
    l = [s[0] for s in seg]
    r = [s[1] for s in seg]
    INF = 10**30
    dp = [[INF] * (K + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        for k in range(1, K + 1):
            for t in range(i):
                if dp[t][k - 1] < INF:
                    c = block_cost(l, r, t, i)
                    dp[i][k] = min(dp[i][k], dp[t][k - 1] + c)
    return dp[n][K]

def aliens_trick(row, col, K):
    seg = reduce_intervals(row, col)
    n = len(seg)
    if n == 0 or K <= 0:
        return 0
    K = min(K, n)
    l = [s[0] for s in seg]
    r = [s[1] for s in seg]
    INF = 10**30

    def probe(C):
        dp = [INF] * (n + 1)
        cnt = [0] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            best_val = INF
            best_cnt = 0
            for t in range(i):
                if dp[t] < INF:
                    val = dp[t] + block_cost(l, r, t, i) + C
                    if val < best_val or (val == best_val and cnt[t] + 1 > best_cnt):
                        best_val = val
                        best_cnt = cnt[t] + 1
            dp[i] = best_val
            cnt[i] = best_cnt
        return dp[n], cnt[n]

    max_span = r[-1] - l[0] + 1
    lo, hi = 0, max_span * max_span
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, c = probe(mid)
        if c >= K:
            lo = mid
        else:
            hi = mid - 1
    val, _ = probe(lo)
    return val - lo * K

random.seed(0)
for _ in range(500):
    n = random.randint(1, 8)
    K = random.randint(1, n)
    row = [random.randint(1, 10) for _ in range(n)]
    col = [random.randint(1, 10) for _ in range(n)]
    a = brute_force(row, col, K)
    b = aliens_trick(row, col, K)
    if a != b:
        print("Mismatch!", row, col, K, a, b)
        break
else:
    print("All random tests passed.")

print("\nConvexity check (marginal savings should be non-increasing):")
row = [1, 2, 3, 4, 5]
col = [3, 5, 7, 9, 11]
prev = None
for k in range(1, len(row) + 1):
    Fk = brute_force(row, col, k)
    print(f"F({k}) = {Fk}")
```
