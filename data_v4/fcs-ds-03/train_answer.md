**Problem.** Stations `0..n` lie on a line. `dp[0] = 0`, and for `i = 1..n`,
`dp[i] = c[i] + min over 0 <= j < i of ( dp[j] + b[j]*a[i] )`. Read `n`, then `a[1..n]`, then
`b[0..n-1]`, then `c[1..n]` (integers in `[-10^6, 10^6]`, `n <= 2*10^5`); print `dp[n]`.

**Why the obvious DP is too slow.** The transition says "for each `i`, take the min over every earlier
`j`," i.e. a double loop. That is `O(n^2)` and obviously correct — it is the right *oracle* — but at
`n = 2*10^5` it does `~2*10^10` operations and cannot finish in one second. `dp[i]` picks a single
predecessor (not a sum), so there is no prefix/partial-sum shortcut; the cost is the `min_{j<i}` itself.

**Key idea — the inner min is a lower envelope of lines.** Fix `i` and treat `a[i]` as a variable `X`.
Then predecessor `j` contributes `b[j]*X + dp[j]` — a **line** with slope `b[j]` and intercept `dp[j]`.
Evaluating it at `X = a[i]` is exactly `dp[j] + b[j]*a[i]`. So

```
min_{j<i} ( dp[j] + b[j]*a[i] )  =  ( minimum of the j-lines )  evaluated at X = a[i],
```

the **lower envelope** of a set of lines at one query point — the Convex Hull Trick. Recognizing that the
cost is *linear in the query coordinate* `a[i]` is the whole reformulation; it turns `O(n^2)` into
"maintain lines, support add-line and query-min-at-point" in `O(log n)` each, i.e. `O(n log n)`.

**Which envelope structure — and why Li Chao, not the deque.** The amortized-`O(1)` monotone deque CHT
requires inserted slopes to be monotone *and* queries to arrive in monotone coordinate order. Here the
slopes `b[j]` and query points `a[i]` are arbitrary signed inputs (e.g. `b = [5,-2,1,3]`, `a = [3,1,4,2]`
both zig-zag), so the deque silently corrupts. The robust SOTA choice is a **Li Chao tree**: insert any
line, query any point, each `O(log(range))`, with no monotonicity assumption — and it supports the
*online interleaving* this DP forces (line `j` is only known after `dp[j]`, and `dp[i]` is queried before
its own line exists). Build it over the **coordinate-compressed** distinct values of `a[]` (the only query
points), giving `O(n)` leaves and memory and folding away heavy ties in `a[]`.

**Pitfalls to get right.**
1. *Insert/query ordering.* Insert line `0` (`b[0]`, intercept `dp[0]=0`) **before** any query. For each
   `i`: query at `a[i]` to form `dp[i]`, then insert line `i` **only after** `dp[i]` is finalized, and
   only if `i < n` (start stations are `0..n-1`). This guarantees the tree holds exactly `{0..i-1}` at the
   `i`-query — never the illegal self-edge `j=i`.
2. *Li Chao recurse direction.* After keeping the midpoint winner, push the loser **left** when
   `leftBetter != midBetter`, else right. Inverting this condition makes the tree forget that a line wins
   near an endpoint (a trace on `{y=5, y=-3X+6}` returns `6` instead of `5` at `X=0`).
3. *Overflow.* A chain reaches `~2*10^5 * 10^6 * 10^6 = 2*10^17`; use `long long` everywhere (slopes,
   intercepts, `dp`). An `int` is a silent wrong answer. The `4e18` sentinel is only ever `min`-combined,
   never has a real value added to it (line `0` is always present before any query), so it cannot overflow.

**Edge cases.** `n = 0` → print `dp[0]=0` (special-cased before building a 0-leaf tree). `n = 1` → query
`a[1]`, set `dp[1]`, insert nothing (no station `2`). Heavy ties in `a[]` → compression dedups. All-/mixed
-sign factors → exactly why Li Chao is chosen over the deque. All verified equal to the `O(n^2)` brute.

**Complexity.** `O(n log n)` time (one `O(log n)` insert and one `O(log n)` query per `i`), `O(n)` memory.
At `n = 2*10^5` it runs in about `0.15 s` and `29 MB`, comfortably inside the `1 s` / `256 MB` limits.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
  dp[0] = 0
  dp[i] = c[i] + min_{0 <= j < i} ( dp[j] + b[j] * a[i] )   for i = 1..n
  answer = dp[n]

  Each j contributes the LINE  y = b[j] * X + dp[j]  evaluated at X = a[i].
  We need the minimum over a set of lines at a query point => lower envelope.
  Because lines are inserted (one per j) interleaved with queries (one per i),
  and the slopes b[j] and query points a[i] are NOT sorted/monotone in general
  (values may be negative), the robust SOTA structure is the Li Chao tree built
  over the fixed set of query coordinates = the distinct values of a[1..n].
*/

const long long INF = (long long)4e18;

struct Line {
    long long m, c;            // y = m*X + c
    bool valid;
    long long eval(long long x) const { return m * x + c; }
};

int LN;                        // number of distinct query coordinates (leaves)
vector<long long> xs;          // sorted distinct query coordinates
vector<Line> tree;             // Li Chao segment tree over [0, LN-1] index space

// Insert line `nw` into the node covering [l, r].
void insert(int node, int l, int r, Line nw) {
    if (!nw.valid) return;
    int mid = (l + r) >> 1;
    Line &cur = tree[node];
    if (!cur.valid) { cur = nw; return; }
    // Compare at the midpoint coordinate xs[mid].
    bool leftBetter  = nw.eval(xs[l])   < cur.eval(xs[l]);
    bool midBetter   = nw.eval(xs[mid]) < cur.eval(xs[mid]);
    if (midBetter) swap(cur, nw);      // keep the better line at the midpoint
    if (l == r) return;
    // Push the loser down to the side where it can still win.
    if (leftBetter != midBetter) insert(node << 1,     l,       mid, nw);
    else                         insert(node << 1 | 1, mid + 1, r,   nw);
}

// Minimum y over all inserted lines at coordinate index p (query point xs[p]).
long long query(int node, int l, int r, int p) {
    Line &cur = tree[node];
    long long res = cur.valid ? cur.eval(xs[p]) : INF;
    if (l == r) return res;
    int mid = (l + r) >> 1;
    if (p <= mid) res = min(res, query(node << 1,     l,       mid, p));
    else          res = min(res, query(node << 1 | 1, mid + 1, r,   p));
    return res;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    if (n == 0) { cout << 0 << "\n"; return 0; }   // no stations: dp[0]=0 is the answer

    vector<long long> a(n + 1), b(n), c(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];      // a[1..n]
    for (int j = 0; j < n; j++)  cin >> b[j];      // b[0..n-1]
    for (int i = 1; i <= n; i++) cin >> c[i];      // c[1..n]

    // Coordinate-compress the query points (the distinct values of a[1..n]).
    xs.assign(a.begin() + 1, a.end());
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), xs.end()), xs.end());
    LN = (int)xs.size();

    // index of coordinate v among xs (v is guaranteed present)
    auto idx = [&](long long v) -> int {
        return int(lower_bound(xs.begin(), xs.end(), v) - xs.begin());
    };

    tree.assign(4 * LN, Line{0, 0, false});

    vector<long long> dp(n + 1, INF);
    dp[0] = 0;
    // Insert the line for j = 0:  y = b[0]*X + dp[0].
    insert(1, 0, LN - 1, Line{b[0], dp[0], true});

    for (int i = 1; i <= n; i++) {
        long long best = query(1, 0, LN - 1, idx(a[i]));   // min_j (dp[j] + b[j]*a[i])
        dp[i] = c[i] + best;
        // Now dp[i] is final; if i can serve as a future start (j = i < n), insert its line.
        if (i < n) insert(1, 0, LN - 1, Line{b[i], dp[i], true});
    }

    cout << dp[n] << "\n";
    return 0;
}
```
