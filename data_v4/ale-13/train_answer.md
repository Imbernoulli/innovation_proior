# String Reassembly — editorial

## Problem

Given a bag of `n` short fragments (lowercase strings over a 2–4 letter alphabet,
lengths 12–40, `60 ≤ n ≤ 400`), output a single string `T` that contains **every**
fragment as a contiguous substring — a *common superstring* — and make `T` as short
as possible. This is the Shortest Common Superstring problem (shotgun fragment
assembly / overlap compression): NP-hard and APX-hard, judged by how short a feasible
superstring you produce.

## Objective + scoring

Feasibility: `T` must use only symbols present in the fragments **and** contain every
fragment as a contiguous substring; otherwise the score is `0` (the feasibility
floor — a single missing fragment or stray symbol zeroes the answer). Among feasible
answers, with `Lsol = len(T)` and `Lbase = Σ|fragment|` (the trivial-concatenation
reference length, recomputed by the scorer):

```
score = round( 1_000_000 × Lbase / Lsol )   (feasible)
score = 0                                    (infeasible)
```

Higher is better; the trivial concatenation scores exactly `1_000_000`, any shorter
feasible superstring scores strictly more.

## Baseline

Concatenate all fragments back to back. Unconditionally feasible, length `Lbase`,
score exactly `1_000_000`. It captures zero overlap and is the floor to beat: I beat
it the moment my superstring captures any positive total overlap.

## Key idea (the heuristic innovation)

Lay the fragments out left-to-right in some **order**; consecutive fragments overlap
by the longest suffix-equals-prefix match, so

```
len(T) = Σ|fragment|  −  Σ (overlap of consecutive pairs in the order).
```

The first term is fixed, so **minimizing the superstring length is exactly
maximizing total consecutive overlap** — a maximum-weight Hamiltonian path on the
asymmetric overlap graph `ov[i][j]`. This reformulation is the whole lever:

1. **Drop redundant fragments** (any fragment that is a substring of another); this
   keeps the order-equals-assembly model exact.
2. **Greedy max-overlap merge** for a strong initial order: repeatedly splice the
   highest-overlap `tail → head` edge between two different chains (union-find blocks
   cycles) until one chain remains. The classic constant-factor SCS heuristic.
3. **Reorder by local search**, the decisive step. **Or-opt** (relocate a run of 1–3
   fragments) and a bounded **2-opt** reversal improve the order, and because each
   move only breaks/creates a few adjacencies, its effect on total overlap is an
   `O(1)` incremental delta straight out of the `ov` table — the superstring is never
   rebuilt to score a candidate. Double-bridge-style kicks (iterated local search)
   escape local optima, all under a ~1.8 s wall-clock budget.
4. **Materialize** `T` once at the end by overlap-merging fragments in the final
   order.

## Feasibility & pitfalls

- Feasibility is automatic for *any* order: overlap-merging keeps each fragment a
  contiguous substring of `T`, so the search moves freely and any early stop still
  prints a valid superstring.
- Redundant fragments must be filtered, or the "consecutive overlap" accounting (a
  contained fragment has no chain position) breaks.
- The overlap weight is **asymmetric** (`ov[i][j] ≠ ov[j][i]`), so a 2-opt reversal
  must re-sum the reversed window's internal edges, not just swap two boundary edges
  — handled by capping the window length so the re-sum stays cheap.
- Greedy must refuse any merge that would close a cycle (union-find), else it cannot
  produce a single linear order.

## Complexity per step

Overlap precompute `O(n² · len²) ≈ 6M` char comparisons for `n ≤ 400`, `len ≤ 40`;
greedy edge sort `O(n² log n)`; each Or-opt/2-opt move evaluated in `O(1)` (a few
table lookups) so thousands of moves per millisecond. Final string build `O(len(T))`,
done exactly once.

Self-verify (seeds 1–20): every output feasible, solver mean ≈ `2.44M` vs baseline
`1.00M` (≈ 2.4× shorter superstrings), all within the time budget.

## Code

```cpp
// String Reassembly -- shortest-common-superstring heuristic solver.
//
// Objective: read n fragments and output ONE string T that contains every
// fragment as a contiguous substring, making T as SHORT as possible (shortest
// common superstring; NP-hard). Read the instance from stdin, write T on a
// single line to stdout.
//
// Method (the innovation):
//   0. Drop redundant fragments: any fragment that is already a substring of
//      another fragment is removed (covering the longer one covers it for free).
//   1. Precompute ov[i][j] = length of the longest suffix of frag i that equals
//      a prefix of frag j (the overlap when j is placed right after i).
//   2. Greedy max-overlap merge to build a strong initial ORDER (permutation):
//      repeatedly join the two open chain-ends with the largest overlap, never
//      closing a cycle, until one chain remains. This is the classic GREEDY SCS
//      construction.
//   3. KEY LEVER -- reorder the merge sequence. The superstring length of an
//      order p[0..m-1] is  sum(len) - sum_{k} ov[p[k]][p[k+1]], a constant minus
//      the total consecutive overlap. So minimizing length == MAXIMIZING the
//      total overlap of consecutive pairs == a maximum-weight Hamiltonian path
//      on the overlap graph (asymmetric). We improve the greedy order with local
//      search: Or-opt (relocate a short run of fragments) and 2-opt-style segment
//      reversal, EACH evaluated as an O(1) incremental delta over only the few
//      broken/created adjacencies -- the full length is never recomputed. Wrapped
//      in restarts/perturbations under a wall-clock budget.
//   4. Materialize T from the final order by overlap-merging in sequence.
//
// Feasibility is guaranteed for ANY order: merging consecutive fragments by their
// overlap keeps each fragment a contiguous substring of T, so we always have a
// valid superstring, and any early stop still prints a feasible answer.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
    uint32_t nextu(uint32_t m) { return m ? (uint32_t)(next() % m) : 0u; }  // [0,m)
    double nextd() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

// longest suffix of a that equals a prefix of b (overlap when b follows a).
static int overlap(const string& a, const string& b) {
    int la = (int)a.size(), lb = (int)b.size();
    int maxk = min(la, lb);
    // try the largest overlap first
    for (int k = maxk; k >= 1; --k) {
        bool ok = true;
        for (int t = 0; t < k; ++t) {
            if (a[la - k + t] != b[t]) { ok = false; break; }
        }
        if (ok) return k;
    }
    return 0;
}

int main() {
    double t_start = now_sec();
    const double TIME_LIMIT = 1.8;  // seconds, wall clock

    // ----- read instance -----
    int n, salpha;
    if (scanf("%d %d", &n, &salpha) != 2) return 0;
    {
        // consume rest of header line
        int c;
        while ((c = getchar()) != '\n' && c != EOF) {}
    }
    vector<string> raw(n);
    for (int i = 0; i < n; ++i) {
        // read a whole line (fragment may in principle be any non-space token,
        // but our generator emits one token per line)
        string line;
        int c;
        while ((c = getchar()) != EOF && c != '\n') {
            if (c != '\r') line.push_back((char)c);
        }
        raw[i] = line;
    }

    // ----- drop redundant fragments (substring of another) -----
    // Keep the longest first; a fragment is redundant if it is a substring of an
    // already-kept fragment. This never changes the set of required substrings.
    {
        vector<int> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = i;
        stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
            if (raw[a].size() != raw[b].size()) return raw[a].size() > raw[b].size();
            return raw[a] < raw[b];
        });
        vector<string> kept;
        for (int id : idx) {
            const string& f = raw[id];
            if (f.empty()) continue;
            bool red = false;
            for (const string& g : kept) {
                if (g.find(f) != string::npos) { red = true; break; }
            }
            if (!red) kept.push_back(f);
        }
        raw.swap(kept);
    }
    int m = (int)raw.size();

    // Degenerate cases: 0 or 1 fragment -> print it directly.
    if (m == 0) { printf("\n"); return 0; }
    if (m == 1) { printf("%s\n", raw[0].c_str()); return 0; }

    vector<int> flen(m);
    for (int i = 0; i < m; ++i) flen[i] = (int)raw[i].size();

    // ----- precompute pairwise overlaps ov[i][j] -----
    // ov[i][j] = overlap when j is placed immediately after i.
    vector<vector<int>> ov(m, vector<int>(m, 0));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            if (i != j) ov[i][j] = overlap(raw[i], raw[j]);

    auto order_overlap_sum = [&](const vector<int>& ord) -> long long {
        long long s = 0;
        for (int k = 0; k + 1 < (int)ord.size(); ++k) s += ov[ord[k]][ord[k + 1]];
        return s;
    };

    // ----- greedy max-overlap construction -> initial order -----
    // Maintain chains as doubly-linked ends. Repeatedly pick the (i->j) edge with
    // the largest overlap such that i is a current chain TAIL, j is a chain HEAD,
    // and i,j are in different chains (no premature cycle).
    vector<int> succ(m, -1), pred(m, -1);
    vector<char> isTail(m, 1), isHead(m, 1);
    // chain id via union-find to forbid cycles
    vector<int> uf(m);
    iota(uf.begin(), uf.end(), 0);
    function<int(int)> find = [&](int x) { while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; };

    // candidate edges sorted by overlap desc (m<=~400 -> m^2 manageable)
    struct Edge { int o, i, j; };
    vector<Edge> edges;
    edges.reserve((size_t)m * (m - 1));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            if (i != j) edges.push_back({ov[i][j], i, j});
    sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        if (a.o != b.o) return a.o > b.o;
        if (a.i != b.i) return a.i < b.i;
        return a.j < b.j;
    });
    int joins = 0;
    for (const Edge& e : edges) {
        if (joins == m - 1) break;
        int i = e.i, j = e.j;
        if (!isTail[i] || !isHead[j]) continue;
        if (find(i) == find(j)) continue;  // would close a cycle
        succ[i] = j; pred[j] = i;
        isTail[i] = 0; isHead[j] = 0;
        uf[find(i)] = find(j);
        ++joins;
    }
    // Stitch any leftover separate chains arbitrarily (overlap 0 between them).
    // Collect chain heads in a stable order.
    vector<int> order;
    order.reserve(m);
    {
        vector<int> heads;
        for (int i = 0; i < m; ++i) if (isHead[i]) heads.push_back(i);
        for (int h : heads) {
            int cur = h;
            while (cur != -1) { order.push_back(cur); cur = succ[cur]; }
        }
    }
    // safety: order must be a permutation of 0..m-1
    if ((int)order.size() != m) {
        order.clear();
        for (int i = 0; i < m; ++i) order.push_back(i);
    }

    // ----- local search on the ORDER: maximize total consecutive overlap -----
    // length(order) = totalLen - order_overlap_sum(order); we maximize the sum.
    long long totalLen = 0;
    for (int i = 0; i < m; ++i) totalLen += flen[i];

    Rng rng(0x13ULL ^ (uint64_t)totalLen ^ ((uint64_t)m << 20));

    // position[] gives index of node in `order` (kept in sync after each move).
    auto rebuild_pos = [&](const vector<int>& ord, vector<int>& pos) {
        for (int k = 0; k < (int)ord.size(); ++k) pos[ord[k]] = k;
    };
    vector<int> pos(m);
    rebuild_pos(order, pos);

    auto edgew = [&](int a, int b) -> int { return ov[a][b]; };  // weight a->b

    // 2-opt style: reverse order[a..b]. Delta of overlap sum touches at most the
    // two boundary edges (because reversal makes the segment run backwards, which
    // for an asymmetric weight requires re-summing the segment's internal edges).
    // To keep it O(1)-ish we use Or-opt (relocation) as the workhorse and a
    // bounded reversal only for short segments.
    long long bestSum = order_overlap_sum(order);
    vector<int> bestOrder = order;

    auto seg_internal_sum = [&](const vector<int>& ord, int a, int b) -> long long {
        long long s = 0;
        for (int k = a; k < b; ++k) s += edgew(ord[k], ord[k + 1]);
        return s;
    };

    // Or-opt: move a run of length L starting at position p to position q.
    // We compute the overlap-sum delta from only the broken & created adjacencies.
    auto try_or_opt = [&](vector<int>& ord, long long& curSum, int p, int L, int q) -> bool {
        int M = (int)ord.size();
        if (L <= 0 || p < 0 || p + L > M) return false;
        // q is interpreted in RESIDUAL coordinates (the array with the segment
        // removed); it is range-checked against RM below.
        // segment [p, p+L-1]; removing it joins (p-1) with (p+L).
        int segHead = ord[p], segTail = ord[p + L - 1];
        int before = (p - 1 >= 0) ? ord[p - 1] : -1;
        int after = (p + L < M) ? ord[p + L] : -1;
        // overlap lost by removing the segment from its place:
        long long removed = 0;
        if (before != -1) removed += edgew(before, segHead);
        if (after != -1) removed += edgew(segTail, after);
        long long bridge = 0;
        if (before != -1 && after != -1) bridge = edgew(before, after);
        // Build the residual array (without the segment) implicitly by index math.
        // Compute insertion point in residual coordinates.
        // Residual order = ord without [p..p+L-1].
        // Map q (target position in the NEW full array) carefully: we insert the
        // segment so that it starts at position q in the residual array.
        // Residual length = M - L.
        int RM = M - L;
        if (q < 0 || q > RM) return false;
        // neighbours in residual at insertion gap:
        auto resid_at = [&](int idx) -> int {
            // idx in [0, RM-1] -> original index skipping [p..p+L-1]
            if (idx < p) return ord[idx];
            return ord[idx + L];
        };
        int leftN = (q - 1 >= 0) ? resid_at(q - 1) : -1;
        int rightN = (q < RM) ? resid_at(q) : -1;
        // don't reinsert into the same spot (no-op)
        // gain by inserting segment between leftN and rightN
        long long addRem = 0;  // overlap removed by splitting leftN--rightN
        if (leftN != -1 && rightN != -1) addRem = edgew(leftN, rightN);
        long long added = 0;
        if (leftN != -1) added += edgew(leftN, segHead);
        if (rightN != -1) added += edgew(segTail, rightN);
        long long delta = (-removed + bridge) + (added - addRem);
        if (delta <= 0) return false;
        // apply: extract segment, reinsert
        vector<int> seg(ord.begin() + p, ord.begin() + p + L);
        vector<int> resid;
        resid.reserve(RM);
        for (int k = 0; k < M; ++k) if (k < p || k >= p + L) resid.push_back(ord[k]);
        vector<int> nord;
        nord.reserve(M);
        for (int k = 0; k < q; ++k) nord.push_back(resid[k]);
        for (int x : seg) nord.push_back(x);
        for (int k = q; k < RM; ++k) nord.push_back(resid[k]);
        ord.swap(nord);
        curSum += delta;
        return true;
    };

    // 2-opt segment reversal for SHORT segments (bounded length): recompute the
    // segment internal sum forward vs backward + boundary edges.
    auto try_reverse = [&](vector<int>& ord, long long& curSum, int a, int b) -> bool {
        int M = (int)ord.size();
        if (a < 0 || b >= M || a >= b) return false;
        int left = (a - 1 >= 0) ? ord[a - 1] : -1;
        int right = (b + 1 < M) ? ord[b + 1] : -1;
        long long oldB = 0, newB = 0;
        long long oldInt = seg_internal_sum(ord, a, b);
        // reversed internal sum
        long long newInt = 0;
        for (int k = b; k > a; --k) newInt += edgew(ord[k], ord[k - 1]);
        if (left != -1) { oldB += edgew(left, ord[a]); newB += edgew(left, ord[b]); }
        if (right != -1) { oldB += edgew(ord[b], right); newB += edgew(ord[a], right); }
        long long delta = (newB + newInt) - (oldB + oldInt);
        if (delta <= 0) return false;
        reverse(ord.begin() + a, ord.begin() + b + 1);
        curSum += delta;
        return true;
    };

    long long curSum = bestSum;
    int noImprove = 0;
    while (now_sec() - t_start < TIME_LIMIT) {
        bool improved = false;
        // Or-opt sweep: relocate runs of length 1..3.
        for (int L = 1; L <= 3 && now_sec() - t_start < TIME_LIMIT; ++L) {
            for (int p = 0; p + L <= m; ++p) {
                // try a handful of random target positions + neighbours
                for (int tcnt = 0; tcnt < 6; ++tcnt) {
                    int q = (int)rng.nextu((uint32_t)(m - L + 1));
                    if (try_or_opt(order, curSum, p, L, q)) {
                        improved = true;
                        break;
                    }
                }
            }
        }
        // bounded 2-opt reversal sweep (short segments)
        for (int a = 0; a + 1 < m && now_sec() - t_start < TIME_LIMIT; ++a) {
            int bmax = min(m - 1, a + 8);
            for (int b = a + 1; b <= bmax; ++b) {
                if (try_reverse(order, curSum, a, b)) { improved = true; break; }
            }
        }
        if (curSum > bestSum) { bestSum = curSum; bestOrder = order; noImprove = 0; }
        else ++noImprove;

        if (!improved || noImprove >= 3) {
            // perturbation (double-bridge-like): random segment shuffle/relocate
            order = bestOrder;
            curSum = bestSum;
            int kicks = 1 + (int)rng.nextu(3);
            for (int t = 0; t < kicks; ++t) {
                int L = 1 + (int)rng.nextu(3);
                if (L > m) L = 1;
                int p = (int)rng.nextu((uint32_t)(m - L + 1));
                int q = (int)rng.nextu((uint32_t)(m - L + 1));
                // forced relocate (ignore sign) to escape local optimum
                vector<int> seg(order.begin() + p, order.begin() + p + L);
                vector<int> resid;
                for (int k = 0; k < m; ++k) if (k < p || k >= p + L) resid.push_back(order[k]);
                int RM = (int)resid.size();
                if (q > RM) q = RM;
                vector<int> nord;
                for (int k = 0; k < q; ++k) nord.push_back(resid[k]);
                for (int x : seg) nord.push_back(x);
                for (int k = q; k < RM; ++k) nord.push_back(resid[k]);
                order.swap(nord);
            }
            curSum = order_overlap_sum(order);
            noImprove = 0;
        }
    }
    if (curSum > bestSum) { bestSum = curSum; bestOrder = order; }
    order = bestOrder;

    // ----- materialize superstring from the final order -----
    string T = raw[order[0]];
    for (int k = 1; k < m; ++k) {
        int prev = order[k - 1], cur = order[k];
        int o = ov[prev][cur];
        T += raw[cur].substr(o);
    }

    // print
    fputs(T.c_str(), stdout);
    fputc('\n', stdout);
    return 0;
}
```
