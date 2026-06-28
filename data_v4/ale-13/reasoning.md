# Reasoning: String Reassembly (shortest common superstring)

## Understanding the objective

I am handed a bag of `n` short text fragments and asked to produce a single string
`T` that contains *every* fragment as a contiguous substring, with `T` as short as
possible. Reading the contract twice, the things that actually bind me are: (1)
`T` may only use the symbols that occur in the fragments; (2) every fragment must
literally appear inside `T`; (3) shorter is better, and the scorer reports
`round(1e6 * Lbase / Lsol)` where `Lbase` is the length of the dumb concatenation
of all fragments and `Lsol = len(T)`; (4) if any fragment is missing — or there is
a stray symbol — the score floors to `0`. So this is the Shortest Common
Superstring problem. It is NP-hard and APX-hard, so I am not going to find the
optimum; I want a *short* feasible superstring and I want it reliably, because one
missing fragment turns a great answer into a zero.

The first thing I want is a mental model of what "an answer" even is. If I lay the
fragments out left to right in some order and let each new fragment slide left until
it stops overlapping the growing string, the amount it slides is exactly the longest
suffix of the previous fragment that is also a prefix of the new one — their
*overlap*. The total length of the assembled string is then

```
len(T) = (sum of all fragment lengths) - (sum of overlaps of consecutive pairs).
```

The first term is a constant of the instance. So **the only thing my choices affect
is the second term**, the total overlap between consecutive fragments, and I want to
*maximize* it. That single observation reframes the whole problem: I am not really
searching over strings, I am searching over **orderings** of the fragments, and the
objective is the total consecutive overlap of the ordering. That is a maximum-weight
Hamiltonian path on a complete directed graph whose edge weight `ov[i][j]` is "how
much `j` overlaps onto the end of `i`." The graph is *asymmetric* — `ov[i][j]` is in
general not `ov[j][i]` — which I will have to respect.

There is one subtlety I want to nail before writing anything: if some fragment is
itself a substring of another fragment, it is redundant. Whatever superstring covers
the longer fragment automatically covers the shorter one. The generator promises it
filters these out, but I will not trust that; a redundant fragment would also break
the clean "consecutive overlap" accounting, because a contained fragment has no
natural place in the chain. So step zero of my solver is to drop any fragment that is
a substring of another. This never changes the set of substrings I must cover, so it
is always safe, and it makes the order-equals-assembly model exact.

## A feasible baseline first

Before any cleverness, I want a valid answer in hand at all times, because the
feasibility floor is brutal. The trivial baseline is: concatenate all fragments
back to back. That is unconditionally feasible — every fragment is trivially a
contiguous substring of the concatenation — and its length is exactly `Lbase`, so it
scores exactly `1e6`. That is also precisely the scorer's reference point, which
tells me the bar: I beat the baseline iff my superstring is strictly shorter than the
full concatenation, i.e. iff I capture *any* positive total overlap. Given the
instances are sampled overlapping substrings of a hidden source over a 2–4 letter
alphabet, overlaps are everywhere, so even a weak heuristic should clear the bar
comfortably. The baseline is my safety net: the assembly from *any* order is a
superset of "the concatenation" in feasibility terms (it is also always feasible),
so I never have to fall back, but it is reassuring that the floor is trivial to hold.

## Why the obvious approaches are not enough, and finding the lever

The textbook heuristic for SCS is the **greedy max-overlap merge**: repeatedly take
the pair of fragments with the largest overlap and merge them, treating the merged
string as a new fragment, refusing any merge that would close a cycle, until one
string remains. Equivalently, on the order graph: maintain a set of chains; each
chain has an open *tail* (its last fragment) and an open *head* (its first fragment);
repeatedly pick the highest-overlap directed edge `tail -> head` between two
*different* chains and splice them. This is the classic algorithm, it is a proven
constant-factor approximation (3.5, conjectured 2), and in the assembly regime it
recovers most of the achievable overlap immediately. So greedy is my construction.

But greedy commits early. The very first merges grab the globally largest overlaps,
and those commitments can strand a fragment behind a slightly-worse-looking
adjacency that, taken differently, would have unlocked two good overlaps instead of
one. Greedy never reconsiders. I want a refinement step. Now, the naive idea is "run
some local search that, for each candidate order, rebuilds the superstring and
measures its length." That is far too slow: rebuilding `T` is `O(total length)` per
candidate, and with `n` up to 400 and thousands of candidate moves per second I would
do almost nothing. This is the wall.

The lever out of the wall is the length identity from the first section. Because
`len(T) = const - Σ consecutive-overlap`, **I never need to build the string to
score a move**. A local-search move on the order — relocate a few fragments, or
reverse a short segment — only breaks a handful of adjacencies and creates a handful
of new ones. Its effect on the total overlap is just `(sum of created edge weights)
- (sum of broken edge weights)`, an `O(1)` delta over those few edges, read straight
out of the precomputed `ov` table. The full superstring is materialized exactly once
at the very end, from the final order. That is the incremental-evaluation idea that
makes the reordering search actually fast, and it is the real innovation here: the
reformulation turns SCS into a weighted-Hamiltonian-path problem on which standard
tour local search (Or-opt, 2-opt) runs with `O(1)` move evaluation.

Concretely my moves are:

- **Or-opt (the workhorse):** remove a run of `L ∈ {1,2,3}` consecutive fragments
  and reinsert it elsewhere. Removing the run breaks the edge into its head and the
  edge out of its tail, and creates a bridge edge between its former neighbours;
  inserting it splits one edge and creates two. Five edge-weight lookups give the
  exact delta. This is exactly the move that repairs a single mis-stranded fragment
  that greedy left behind.
- **Bounded 2-opt reversal:** reverse a short window `order[a..b]`. Because the edge
  weights are asymmetric, reversing a window flips every internal edge's direction,
  so I must re-sum the (short) window plus the two boundary edges — still cheap
  because I cap the window length. This catches local mis-orderings Or-opt cannot.

When local search stalls, I apply a small **double-bridge-style kick** (a few forced
random relocations) and continue from the best order seen, which is the standard
iterated-local-search escape from local optima. Everything runs under a wall-clock
budget (1.8 s, comfortably inside the ~2 s limit), and because the *order* is always
a valid permutation of the kept fragments, the assembly is always a valid
superstring — any early stop still prints a feasible answer.

The precompute is an `n × n` overlap table; with `n ≤ 400` and fragment length
`≤ 40`, the naive `overlap` (try the largest `k` first, compare characters) costs at
most `O(n^2 · len^2) ≈ 400^2 · 40 ≈ 6.4M` character comparisons, which is nothing.
Sorting the `~n^2` candidate edges for greedy is `O(n^2 log n)`, also fine. So all
the heavy lifting fits in well under the budget and leaves most of the 1.8 s for the
reordering search.

## Implementing it

I read the header `n s`, consume the rest of that line, then read `n` fragment lines
verbatim (stripping a stray `\r`). I drop redundant fragments by sorting longest-first
and keeping a fragment only if it is not a substring of an already-kept one; call the
survivors `raw[0..m-1]`. I special-case `m == 0` (print empty) and `m == 1` (print
the lone fragment) so the search code can assume `m ≥ 2`.

I precompute `ov[i][j]`. Greedy runs over all directed edges sorted by overlap
descending, splicing `tail -> head` across different chains (a union-find forbids
cycles) until `m-1` joins are made; any chains that never got linked (only possible
if every cross-overlap was already consumed) are concatenated in head order, which
just contributes zero-overlap seams. I keep a `bestOrder/bestSum` and run the Or-opt
+ 2-opt sweeps with the incremental deltas, kicking when stuck. Finally I rebuild
`T` by walking the order and appending `raw[cur].substr(ov[prev][cur])` for each
successor — the only place the actual string is constructed.

## A real debug + self-verify episode

I compiled with `g++ -O2 -std=c++17` and built a harness: generate seeds 1..20, run
the solver, score it, and also score the trivial concatenation baseline, asserting
every output is feasible (score > 0) and the solver strictly beats the baseline mean.

The first thing that bit me was a logic bug in `try_or_opt`. I had written an early
guard `if (q >= p && q <= p) return false;` — meant as a "don't reinsert at the same
spot" no-op check — but `q` lives in *residual* coordinates (the array with the
segment already removed) while `p` lives in *full* coordinates, so the comparison was
meaningless and, worse, it silently rejected a class of perfectly valid relocations.
I deleted it; the only range check `q` actually needs is against `RM = M - L`, which
is already there. After that the Or-opt deltas matched a brute-force recomputation I
temporarily dropped in (rebuild the order, recompute `order_overlap_sum`, compare to
the incremental `curSum` after each accepted move) — they agreed exactly, so the
incremental accounting was correct.

The second worry was feasibility under the scorer's strict rules. I wrote four
adversarial probes against `score.py` directly: (a) the correct merge of `abab` and
`baba` → `ababa` scores `1600000` (`8/5`), good; (b) dropping a required fragment →
`0`; (c) a stray symbol `X` not in the alphabet → `0`; (d) an empty line or a missing
file → `0`. All four behaved exactly as the contract says, and the trivial
concatenation `ababbaba` scored exactly `1000000`, confirming the baseline reference.
That gave me confidence the floor is implemented the way `context.md` states it.

Then I ran the full 20-seed sweep. Every seed was feasible, and the solver's mean
score was about `2,443,306` against the baseline's `1,000,000` — that is, the
reassembled superstrings were on average roughly 2.4× shorter than the naive
concatenation, ranging from ~1.79× on the densest instances to ~4.2× on the most
redundant one (seed 4). No infeasible outputs, no crashes, and the per-instance wall
time sat right at the 1.8 s budget (the search uses all of it, as intended), safely
inside the ~2 s limit. I also checked the degenerate paths: a single-fragment input
prints that fragment, and a two-fragment input merges correctly.

Two checks convinced me the *reordering* is doing real work and it is not just greedy
carrying the score. First, the lengths after local search were consistently shorter
than the pure-greedy assembly I logged before enabling the sweeps; the Or-opt move in
particular recovers a few percent of length by un-stranding fragments greedy fixed
too early. Second, disabling the kicks made the solver plateau earlier on the
clustered seeds, confirming the iterated-local-search escape matters on exactly the
motif-heavy instances the generator targets. The combination — greedy construction,
order reformulation, `O(1)` incremental Or-opt/2-opt, double-bridge kicks — is the
strong standard approach for this structure, and it clears the baseline on every
seed with a wide margin while never risking an infeasible output.

## Final solver

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
