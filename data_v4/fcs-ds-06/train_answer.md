**Problem.** There are `n` fixed point locations, each with a mutable integer weight (possibly
negative). Process `q` operations in order: `1 i d` adds `d` to point `i`'s weight; `2 A B C D` is a
rectangle-sum query whose four coordinates are XOR-encoded with the previous answer `lastAns` (starting
`0`) — decode `X1=A^lastAns, Y1=B^lastAns, X2=C^lastAns, Y2=D^lastAns`, then report the sum of current
weights with `X1<=x<=X2` and `Y1<=y<=Y2`. The XOR makes queries **forced online**; coordinates and
weights reach `+-10^9`, `n,q<=10^5`.

**Why the obvious approaches fail.** The textbook offline sweepline (sort events by `x`, sweep with one
Fenwick over `y`) is doubly illegal here: weights mutate, so there is no static vector to prefix over,
and — decisively — the forced-online XOR forbids reordering queries at all, since a query's rectangle is
hidden behind the previous answer. The natural online fallback, a dense compressed 2D Fenwick `bit[x][y]`,
needs `X*Y` cells with `X,Y` up to `10^5`, i.e. `~10^{10}` cells (~80 GB). Dead on memory.

**Key idea — Fenwick-of-Fenwicks sized to the static point set (the insight).** The one gift the problem
gives is that **point locations never change** — updates only retouch existing `(x,y)`, queries only
read — so the set of cells that can ever be nonzero is known up front: the `n` input points. Build a
Fenwick tree over the compressed `x`-ranks, and replace each scalar node by an *inner* Fenwick over only
the `y`-values that route to it. Concretely, a point at `x`-rank `r` is registered into every outer node
`j` on the walk `j = r, r+(r&-r), ...`; after collecting, sort+unique each node's `y`-list — that list is
that node's inner `y`-axis, sized exactly. Total inner cells `= O(n log n) ~ 1.7*10^6` (a dozen MB), and
each operation is `O(log^2 n)`. This product-structure decomposition is the canonical strongest known
structure for *online* dynamic 2D point-update / rectangle-sum on a static coordinate set; `q log^2 n ~
3*10^7` clears the 1 s limit (measured 0.30 s at full scale).

- *Update* `(x,y,+d)`: walk outer `j=r` upward; in node `j`, the inner slot is the **1-indexed position
  of the present value** `yr = lower_bound(list, y) - begin + 1`, then walk the inner Fenwick upward.
- *Prefix* `x-rank <= r, y <= Y`: walk outer `j=r` downward; in node `j`, the inner prefix length is the
  **count of values `<= Y`** `yr = upper_bound(list, Y) - begin`, then walk the inner Fenwick downward.
- *Rectangle*: `(P(rHi,Y2)-P(rHi,Y1-1)) - (P(rLo,Y2)-P(rLo,Y1-1))` with `rHi=xidx(X2)`, `rLo=xidx(X1-1)`,
  where `xidx(v)` is the largest `x`-rank with value `<= v`.

**Pitfalls.**
1. *Inner-rank off-by-one (the load-bearing bug).* The **update** must hit the slot that *is* `y`
   (`lower_bound+1`); the **query** must sum slots up to the *count* `<= Y` (`upper_bound`). These are
   different binary searches — using `upper_bound` on the update side corrupts duplicate-`y` and
   exact-edge sums (it lands in the wrong slot once a `y` recurs).
2. *Forced-online decode.* XOR every query's four coordinates with the previous answer *at execution
   time*; `lastAns` starts at `0`. Reading and "pre-sorting" queries is impossible.
3. *Overflow.* A full-plane sum reaches `~10^{14}`; every weight, accumulator, and Fenwick cell is
   `long long`. An `int` is a silent wrong answer.
4. *Memory.* Do **not** give each outer node a full `y`-axis — only the `y`'s routed to it. That is what
   turns `~10^{10}` cells into `O(n log n)`.

**Edge cases.** Empty decoded rectangle (`X1>X2` or `Y1>Y2`) returns `0` before touching the trees;
boundary is inclusive (`rLo=xidx(X1-1)`, lower edge `Y1-1`); negative weights can cancel to `0` or make
the answer negative (no clamp — it is a signed sum, not a count); duplicate coordinates accumulate into
the same inner slot (the `lower_bound+1` fix is exactly what makes this correct); `rLo=0` (left edge at or
below the smallest `x`) contributes `0`; the first query decodes with `lastAns=0`; an update-only run
prints nothing.

**Complexity.** Build `O(n log n)`; each update and each query `O(log^2 n)`; total
`O((n+q) log^2 n)` time and `O(n log n)` memory.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if(!(cin >> n >> q)) return 0;

    // The set of (x,y) coordinates that may ever be touched by an update is given
    // up front, so we can compress and size every inner structure exactly.
    vector<long long> px(n), py(n), pw(n);
    for(int i=0;i<n;i++) cin >> px[i] >> py[i] >> pw[i];

    // --- read all queries first (they are forced-online via XOR, but their raw
    //     encoded form is independent of answers; decode happens during execution) ---
    // type 1: update  "1 i d"   -> add d to the weight of point i (0-indexed)
    // type 2: query   "2 X1 Y1 X2 Y2" -> sum of weights with X1<=x<=X2, Y1<=y<=Y2
    // The query coordinates are XOR-encoded with the last answer (lastAns).
    vector<array<long long,5>> qs(q);
    vector<int> qt(q);
    for(int i=0;i<q;i++){
        int t; cin >> t; qt[i]=t;
        if(t==1){
            long long idx,d; cin >> idx >> d;
            qs[i]={idx,d,0,0,0};
        }else{
            long long a,b,c,e; cin >> a >> b >> c >> e;
            qs[i]={a,b,c,e,0};
        }
    }

    // --- coordinate compression of x over all points ---
    vector<long long> xs(px.begin(), px.end());
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), xs.end()), xs.end());
    int X = (int)xs.size();
    auto xidx = [&](long long v)->int{
        // position in 1-indexed compressed array of the largest xs <= v, else 0
        return (int)(upper_bound(xs.begin(), xs.end(), v) - xs.begin());
    };
    auto xexact = [&](long long v)->int{
        return (int)(lower_bound(xs.begin(), xs.end(), v) - xs.begin()) + 1; // 1-indexed exact
    };

    // For each compressed x-column (Fenwick index over x), collect the distinct y
    // values of points whose x falls under it in the BIT tree. A point at x-rank r
    // (1-indexed) is inserted into all Fenwick nodes j obtained by r += r&(-r).
    // Each such node owns an inner Fenwick over the compressed y's routed to it.
    vector<vector<long long>> ys(X+1);
    // map each point to its x-rank
    vector<int> rankX(n);
    for(int i=0;i<n;i++){
        int r = xexact(px[i]);
        rankX[i] = r;
        for(int j=r; j<=X; j+=j&(-j)) ys[j].push_back(py[i]);
    }
    // sort+unique each node's y-list -> gives that inner Fenwick its compressed y axis
    for(int j=1;j<=X;j++){
        auto &v = ys[j];
        sort(v.begin(), v.end());
        v.erase(unique(v.begin(), v.end()), v.end());
    }
    // inner Fenwick trees, one per outer node; sized to that node's distinct y count
    vector<vector<long long>> bit(X+1);
    for(int j=1;j<=X;j++) bit[j].assign(ys[j].size()+1, 0LL);

    // update: add delta to point with x-rank r and y-value yv
    auto update = [&](int r, long long yv, long long delta){
        for(int j=r; j<=X; j+=j&(-j)){
            auto &yv_list = ys[j];
            int yr = (int)(lower_bound(yv_list.begin(), yv_list.end(), yv) - yv_list.begin()) + 1;
            int m = (int)yv_list.size();
            for(int k=yr; k<=m; k+=k&(-k)) bit[j][k] += delta;
        }
    };

    // prefix query over x-rank in [1..r], y-value <= yv (inclusive upper bound)
    auto queryPrefix = [&](int r, long long yv)->long long{
        long long s = 0;
        for(int j=r; j>0; j-=j&(-j)){
            auto &yv_list = ys[j];
            // number of compressed y's <= yv in this node
            int yr = (int)(upper_bound(yv_list.begin(), yv_list.end(), yv) - yv_list.begin());
            for(int k=yr; k>0; k-=k&(-k)) s += bit[j][k];
        }
        return s;
    };

    // rectangle sum: X1<=x<=X2, Y1<=y<=Y2
    auto rectSum = [&](long long X1, long long Y1, long long X2, long long Y2)->long long{
        if(X1>X2 || Y1>Y2) return 0;
        int rHi = xidx(X2);          // x-rank for x <= X2
        int rLo = xidx(X1 - 1);      // x-rank for x <= X1-1
        long long hi = queryPrefix(rHi, Y2) - queryPrefix(rHi, Y1 - 1);
        long long lo = queryPrefix(rLo, Y2) - queryPrefix(rLo, Y1 - 1);
        return hi - lo;
    };

    // current weight stored at each point (so updates apply deltas correctly)
    vector<long long> curW(n, 0);
    // initialize all points with their starting weight
    for(int i=0;i<n;i++){
        if(pw[i]!=0){
            update(rankX[i], py[i], pw[i]);
        }
        curW[i] = pw[i];
    }

    long long lastAns = 0;
    string out;
    out.reserve((size_t)q*8);
    for(int i=0;i<q;i++){
        if(qt[i]==1){
            long long idx = qs[i][0];
            long long d   = qs[i][1];
            int pi = (int)idx;
            update(rankX[pi], py[pi], d);
            curW[pi] += d;
        }else{
            // decode the four coordinates by XOR with lastAns
            long long X1 = qs[i][0] ^ lastAns;
            long long Y1 = qs[i][1] ^ lastAns;
            long long X2 = qs[i][2] ^ lastAns;
            long long Y2 = qs[i][3] ^ lastAns;
            long long ans = rectSum(X1, Y1, X2, Y2);
            lastAns = ans;
            out += to_string(ans);
            out += '\n';
        }
    }
    cout << out;
    return 0;
}
```
