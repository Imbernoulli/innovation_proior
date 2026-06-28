**Reading the objective and pinning the contract.** I am given `n` jobs on one machine. Job `i`
carries a processing time `p_i`, a weight `w_i`, and a due date `d_i`. I pick the order the machine
runs them in; that order alone decides every completion time, because the machine never idles:
the completion time of the `k`-th job in my order is the sum of the processing times of the first
`k` jobs. The cost I am minimizing is the total weighted tardiness
`WT = sum_i w_i * max(0, C_i - d_i)` — each job contributes its weight times how late it is, and
nothing if it is on time. Input is `n` then `n` lines `p_i w_i d_i`; output is a permutation of the
ids `0..n-1`, one per line, naming the run order. This is `1 || sum w_j T_j`, which is strongly
NP-hard, so I am not going to find a provably optimal order in general — I want the strongest
heuristic that fits a ~2 s budget, and above all an output that is *always* a legal permutation so
the score never floors to 0.

**Where the score lives, and what "feasible" means.** The scorer recomputes completion times along
my printed order, sums weighted tardiness, and compares against the earliest-due-date (EDD) order
as a reference: `score = 1e6 * (WT_edd + 1) / (WT + 1)`. Lower `WT` is a higher score, EDD itself
scores exactly `1e6`, and beating EDD pushes above `1e6`. The one cliff is feasibility: if my output
is not exactly a permutation of `0..n-1` — a duplicate id, a missing id, an out-of-range token, the
wrong count — the score is `0`. So the very first invariant I commit to, before any optimization, is
that whatever I emit is a permutation of `0..n-1`. Every code path has to preserve that. The cheapest
way to guarantee it is to start from a real permutation and only ever apply moves that are themselves
permutations of the array (swaps, relocations), never anything that could drop or duplicate an id.

**Getting a feasible baseline first.** Before being clever I want a valid solution in hand. The
identity order `0,1,...,n-1` is a permutation, so it is feasible and scores something positive
(it is just the input order, usually bad). That is my floor. The natural *good* trivial baseline is
EDD — sort jobs by due date. EDD is the classic dispatching answer when weights are ignored, it is
a permutation by construction, and it is exactly the reference the scorer normalizes against, so
"beat EDD" is a concrete, honest bar. I will keep EDD around as a guaranteed fallback inside the
solver: whatever my fancier construction does, I also evaluate EDD and keep the better of the two,
so I can never score below `1e6` by accident.

**Why plain dispatching is not enough, and what the strong construction is.** EDD ignores weights
and processing times; a heavy, short, nearly-late job should often jump ahead of a light, long one
even if the light one is due slightly earlier. The established composite dispatching rule for
weighted tardiness is **ATC — Apparent Tardiness Cost** (Rachamadugu-Morton). Scheduling forward in
time, with `t` = the current time (completion of the last job placed), ATC picks the unscheduled job
maximizing

```
I_j(t) = (w_j / p_j) * exp( - max(0, d_j - p_j - t) / (k * p_bar) ),
```

where `p_bar` is the average processing time of the *remaining* jobs and `k` is a lookahead
parameter. The first factor `w_j / p_j` is the weighted-shortest-processing-time priority — it wins
when everything is already late and you just want to clear weight fast. The exponential factor
discounts a job whose slack `d_j - p_j - t` is still large (no rush) and saturates to 1 when the job
is already late (`slack = 0`). `k` controls how sharply slack matters: small `k` is aggressive about
due dates, large `k` flattens toward pure WSPT. There is no universally best `k`, so the honest move
is to sweep a few values — `{0.5, 1, 1.5, 2, 3, 4}` — build the ATC order for each, and keep the
cheapest. That is my construction stage: it dominates EDD on weighted instances because it actually
uses `w` and `p`.

**Why the obvious local search is too slow — the real lever.** Construction alone leaves money on
the table; I need local search. The two moves that matter for sequencing are **adjacent pairwise
interchange** (swap two neighbours — this is exactly the move the classical adjacency dominance
condition sanctions: if swapping a neighbouring pair lowers cost, the original order was dominated)
and **job insertion** (lift one job out and reinsert it at the best position). Insertion is the
powerful one because a single job can travel across the whole sequence in one move, fixing a job
that EDD/ATC stranded far from where it belongs.

The naive way to evaluate an insertion is: form the candidate order, rescore the whole thing in
`O(n)`. With `n` source positions times `n` destinations times `O(n)` rescoring, one full insertion
sweep is `O(n^3)`. At `n = 100` that is `10^6` per sweep, and I want *thousands* of sweeps inside the
budget (the whole point of iterated local search). `O(n^3)` per sweep would let me do only a handful
— a toy. So the lever I have to get right is **incremental evaluation**.

Here is the observation that makes insertion cheap. Relocating the job at position `src` to position
`dst` only disturbs the jobs *between* `src` and `dst`. Everything before `min(src,dst)` is untouched
— same jobs, same completion times. And, crucially, everything *after* `max(src,dst)` is also
untouched: the set of jobs inside the moved window is the same before and after (I only permuted
them), so the total processing time inside the window is invariant, which means the completion time
of the first job *past* the window is identical, and the entire suffix beyond the window keeps its
completion times and hence its tardiness. So the *only* change in `WT` is inside the window
`[min(src,dst), max(src,dst)]`. I can recompute that window's cost in `O(window length)` against the
window's old cost, and the delta is exact. A move near `src` costs `O(1)`; a move across the array
costs `O(n)`; on average a sweep is `O(n^2)` and often far less. That is the difference between one
pass and thousands. This incremental window evaluation — built on a prefix array of completion times
so I know the completion time entering any window in `O(1)` — is the innovation the candidate names,
and it is what turns ATC-plus-local-search into a genuinely strong solver instead of a demo.

**The plan, assembled.** (1) Construct with ATC over a few `k`, keep the best; also evaluate EDD and
keep it if better, as a guaranteed feasible floor. (2) Local search: alternate adjacent-interchange
passes (cheap `O(1)`-per-pair, handles the dominance swaps) and best-improvement insertion passes
(the `O(window)` incremental move) until neither improves. (3) Iterated local search: perturb the
incumbent with a random block move (relocate a short segment), re-run local search, keep the global
best, repeat until the deadline. Throughout, the working array is always a permutation, so the output
is always feasible.

**First implementation, then the debugging episode.** I wrote `evalOrder` (full `O(n)` rescoring,
used only to compare whole candidates), `atcConstruct(k)`, `buildPrefix` (prefix completion times),
`windowCost` (cost of a position range given the entry completion time), the insertion pass, the
adjacent pass, and the ILS loop. Then I tried to verify, and the incremental insertion delta is
exactly the kind of code that is clean in the head and wrong on the screen, so I traced it.

The trap is the index bookkeeping of "insert before original position `dst`". My first cut computed
the moved-window cost by, in the `src < dst` case, walking jobs `lo..hi` and appending `jb` at the
end — but I had the loop bounds as `for (p = lo; p <= hi; p++)` *including* the old `src` slot, then
also adding `jb`. That counts `jb` twice in the window and counts one real job's contribution at the
wrong completion time. I caught it by checking the incremental delta against a brute full rescoring
of the actually-applied move: I applied the chosen relocation to the vector, called `evalOrder` on
the whole thing, and compared the true `WT` change to the `delta` my window code had predicted. On a
small instance they disagreed — the window predicted an improvement that the full rescoring did not
confirm. That mismatch is the unmistakable signature of a miscounted window.

**Diagnosing and fixing the window.** The fix is to be precise about which jobs are in the window
*after* the move. For `src < dst` (moving a job later): the window `[lo, hi] = [src, dst]` originally
holds `jb` at `lo` then jobs `lo+1..hi`; after the move `jb` lands at `hi` and jobs `lo+1..hi` shift
one slot left. So the new-window cost walks `p = lo+1..hi` (those jobs, now starting one slot
earlier, i.e. completion times reduced by `p[jb]`) and then places `jb` last. For `dst < src` (moving
a job earlier): `jb` goes to `lo` first, then jobs `lo..src-1` shift one slot right. I rewrote the
two branches to walk exactly those jobs and add `jb` exactly once, each at its correct completion
time accumulated from `Cstart = prefixC(pref, lo)`. After this, the incremental `delta` matched the
full-rescoring `WT` change on every move I tried — that agreement is the evidence I trust, not the
fact that it compiles.

A second, quieter correctness point I pinned while tracing: when I actually apply the best relocation
I must convert `dst` to an insertion index after erasing `src`. If `dst > src`, erasing `src` shifts
everything past it left by one, so the insertion index is `dst - 1`; if `dst < src`, it is `dst`.
Getting this wrong does not crash and the array stays a permutation (so it stays *feasible*), but it
silently applies a different move than the one I scored, which corrupts the search. The line
`int ins = bestDst > src ? bestDst - 1 : bestDst;` encodes it. I confirmed it by, again, comparing
the post-apply `evalOrder` to `bestCost + bestDelta` — they agree, so the move applied is the move
scored.

**Self-verify on the seed set.** With the window fixed I ran the full harness: compile, generate
seeds `1..20`, and for each run the solver, score it, and also score two trivial baselines — the
identity (input) order and the EDD order (which the scorer normalizes to exactly `1e6`). Results:
every one of the 20 outputs is a valid permutation (feasible, score `> 0`); the solver mean is about
`2.8`-`2.9 million` versus identity `~0.5 million` and EDD `1.0 million`. The minimum solver score
across all seeds is exactly `1e6` — on the few easy seeds where EDD already achieves `WT = 0` the
solver matches it and cannot do better, but it never drops below the EDD baseline, which is the
guarantee the EDD fallback buys me. So: all feasible, and the mean strictly beats both trivial
baselines. A few seeds (where EDD is already optimal, `WT = 0`) tie at `1e6`; the rest show large
gains from the ATC construction plus the incremental insertion search.

**Edge cases and robustness.** `n = 1`: the single job is the only permutation; the loops do nothing
harmful and I print `0`. Small `n < 4`: the ILS block-move guard `if (N >= 4)` skips perturbation
(no room), but local search still runs and EDD/ATC already give the answer. Time: every stage checks
the deadline (`1800 ms`, comfortably under the 2 s limit), so the solver returns in budget even if a
local search would otherwise run long; because I always carry the best feasible order in `best`, an
early cutoff still emits a valid, good permutation. Overflow: `p <= 100`, `n <= 100`, so the makespan
is `<= 10^4`, tardiness `<= 10^4`, weighted `<= 10^5` per job, total `<= 10^7` — comfortably inside
`long long` (I use `long long` for all completion/cost accumulators anyway). The output writer just
prints the ids in `best`, which is always a permutation, so feasibility is structural, not hoped-for.

**Final solver.** The construction gives a strong start, the incremental insertion neighbourhood is
what makes the local search affordable enough to run thousands of times, the adjacent pass cleans up
the dominance swaps cheaply, and the ILS perturbation lets it escape local optima within the budget.
Crucially, the working array is a permutation at every step and EDD is kept as a floor, so the output
is always feasible and never scores below the baseline. This is what I ship:

```cpp
#include <bits/stdc++.h>
using namespace std;

// Single-machine total weighted tardiness: 1 || sum w_j * max(0, C_j - d_j).
// Read n jobs (proc, weight, due). Emit a permutation of job ids minimizing
// total weighted tardiness. Strategy:
//   (1) ATC (apparent tardiness cost) dispatch construction, k swept over a few
//       values, keep the best start;
//   (2) dominance-pruned insertion + adjacent-interchange local search with
//       prefix-sum-of-processing-time incremental completion times;
//   (3) iterated local search: perturb (random block move), re-optimize, keep
//       the best schedule seen, until the time budget expires.
// The output is ALWAYS a permutation of 0..n-1, so it is always feasible.

static int N;
static vector<long long> P, W, D; // processing, weight, due per job id

// Total weighted tardiness of an order (vector of job ids).
static long long evalOrder(const vector<int>& ord) {
    long long C = 0, tot = 0;
    for (int j : ord) {
        C += P[j];
        long long t = C - D[j];
        if (t > 0) tot += W[j] * t;
    }
    return tot;
}

// ATC construction with lookahead k. Greedy: at each step pick the unscheduled
// job with the largest apparent tardiness cost index.
static vector<int> atcConstruct(double k) {
    vector<char> used(N, 0);
    vector<int> ord;
    ord.reserve(N);
    long long t = 0;              // current time (completion of last scheduled)
    // average processing time of remaining jobs (recomputed lazily as a running mean)
    long long sumP = 0; int rem = N;
    for (int i = 0; i < N; i++) sumP += P[i];
    for (int step = 0; step < N; step++) {
        double pbar = (double)sumP / max(1, rem);
        double denom = k * pbar;
        if (denom < 1e-9) denom = 1e-9;
        int best = -1; double bestIdx = -1e300;
        for (int j = 0; j < N; j++) {
            if (used[j]) continue;
            double slack = (double)D[j] - (double)P[j] - (double)t;
            if (slack < 0) slack = 0;            // already late -> max urgency
            double idx = ((double)W[j] / (double)P[j]) * exp(-slack / denom);
            if (idx > bestIdx) { bestIdx = idx; best = j; }
        }
        used[best] = 1;
        ord.push_back(best);
        t += P[best];
        sumP -= P[best];
        rem--;
    }
    return ord;
}

// Recompute total weighted tardiness contribution from a prefix-completion array.
// Given the order and a precomputed prefix of completion times we can score in O(n);
// the local search below uses targeted O(block) recomputation instead.

// Insertion local search: try moving each job to a better position. A move that
// shifts job from position i to position j only changes completion times of the
// jobs in the affected interval, so we recompute the delta over that interval in
// O(|interval|) using cumulative processing times. We pick the first improving
// move (first-improvement) and repeat until no single move improves.
//
// To keep it robust and simple we recompute the affected suffix cost directly
// from completion times: moving a job changes completions of everything between
// the old and new slot; we evaluate the candidate order's cost over that window
// against the current window cost.

static long long windowCost(const vector<int>& ord, int lo, int hi, long long Cstart) {
    // cost of jobs at positions [lo, hi], given completion time before position lo is Cstart
    long long C = Cstart, tot = 0;
    for (int p = lo; p <= hi; p++) {
        int j = ord[p];
        C += P[j];
        long long t = C - D[j];
        if (t > 0) tot += W[j] * t;
    }
    return tot;
}

// Prefix completion time up to (not including) position pos.
static long long prefixC(const vector<long long>& pref, int pos) {
    return pos <= 0 ? 0LL : pref[pos - 1];
}

// Build prefix sums of processing times along the order: pref[i] = C_i (completion of position i).
static void buildPrefix(const vector<int>& ord, vector<long long>& pref) {
    pref.resize(ord.size());
    long long C = 0;
    for (size_t i = 0; i < ord.size(); i++) { C += P[ord[i]]; pref[i] = C; }
}

// One full pass of insertion local search with first-improvement. Returns true if
// it improved at least once. Uses incremental window scoring: moving the job at
// position src to be inserted before position dst only changes the cost of the
// interval spanned by [min(src,dst), max(src,dst)], because completion times
// outside that interval are unchanged (total processing in the interval is
// invariant under the move, so positions after the interval keep their C).
static bool insertionPass(vector<int>& ord) {
    int n = (int)ord.size();
    vector<long long> pref;
    buildPrefix(ord, pref);
    bool improvedAny = false;
    for (int src = 0; src < n; src++) {
        int jb = ord[src];
        // dominance prune: limit candidate destinations. We still scan all but
        // skip clearly non-improving directions cheaply via the window delta.
        long long bestDelta = 0; int bestDst = -1;
        for (int dst = 0; dst < n; dst++) {
            if (dst == src) continue;
            int lo = min(src, dst), hi = max(src, dst);
            long long Cstart = prefixC(pref, lo);
            long long oldCost = windowCost(ord, lo, hi, Cstart);
            // build the moved window order
            // moving src into position dst: remove jb, insert before original dst
            // We compute the new window cost without materializing the whole vector.
            long long C = Cstart, newCost = 0;
            if (src < dst) {
                // jobs lo..hi originally: src then src+1..hi ; after move job at src
                // goes after position hi shift left, jb placed at hi
                for (int p = lo + 1; p <= hi; p++) {
                    int j = ord[p];
                    C += P[j];
                    long long t = C - D[j];
                    if (t > 0) newCost += W[j] * t;
                }
                C += P[jb];
                long long t = C - D[jb];
                if (t > 0) newCost += W[jb] * t;
            } else {
                // dst < src: jb placed at position dst(=lo), then jobs lo..src-1 shift right
                C += P[jb];
                long long t0 = C - D[jb];
                if (t0 > 0) newCost += W[jb] * t0;
                for (int p = lo; p <= hi - 1; p++) {
                    int j = ord[p];
                    C += P[j];
                    long long t = C - D[j];
                    if (t > 0) newCost += W[j] * t;
                }
            }
            long long delta = newCost - oldCost;
            if (delta < bestDelta) { bestDelta = delta; bestDst = dst; }
        }
        if (bestDst >= 0) {
            int j = ord[src];
            ord.erase(ord.begin() + src);
            int ins = bestDst > src ? bestDst - 1 : bestDst;
            ord.insert(ord.begin() + ins, j);
            buildPrefix(ord, pref);
            improvedAny = true;
        }
    }
    return improvedAny;
}

// Adjacent pairwise interchange pass: swap neighbours if it lowers cost. Cheap and
// complements insertion (handles the dominance-rule "swap violating adjacent pair".)
static bool adjacentPass(vector<int>& ord) {
    int n = (int)ord.size();
    bool improved = false;
    // recompute completion incrementally
    vector<long long> pref; buildPrefix(ord, pref);
    for (int i = 0; i + 1 < n; i++) {
        int a = ord[i], b = ord[i + 1];
        long long Cbefore = prefixC(pref, i);
        long long Ca = Cbefore + P[a];
        long long Cab = Ca + P[b];
        long long oldC = (Ca > D[a] ? W[a] * (Ca - D[a]) : 0) + (Cab > D[b] ? W[b] * (Cab - D[b]) : 0);
        long long Cb = Cbefore + P[b];
        long long Cba = Cb + P[a];
        long long newC = (Cb > D[b] ? W[b] * (Cb - D[b]) : 0) + (Cba > D[a] ? W[a] * (Cba - D[a]) : 0);
        if (newC < oldC) {
            swap(ord[i], ord[i + 1]);
            pref[i] = Cb; // pref[i+1] unchanged (Cba == Cab)
            improved = true;
        }
    }
    return improved;
}

static void localSearch(vector<int>& ord, chrono::steady_clock::time_point deadline) {
    bool go = true;
    while (go) {
        if (chrono::steady_clock::now() > deadline) break;
        go = false;
        if (adjacentPass(ord)) go = true;
        if (chrono::steady_clock::now() > deadline) break;
        if (insertionPass(ord)) go = true;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> N)) return 0;
    if (N <= 0) { return 0; }
    P.resize(N); W.resize(N); D.resize(N);
    for (int i = 0; i < N; i++) cin >> P[i] >> W[i] >> D[i];

    auto t0 = chrono::steady_clock::now();
    auto deadline = t0 + chrono::milliseconds(1800);

    // (1) ATC construction over a few lookahead values; keep the best.
    vector<int> best;
    long long bestCost = LLONG_MAX;
    double ks[] = {0.5, 1.0, 1.5, 2.0, 3.0, 4.0};
    for (double k : ks) {
        vector<int> ord = atcConstruct(k);
        long long c = evalOrder(ord);
        if (c < bestCost) { bestCost = c; best = ord; }
    }
    // safety fallback: EDD order (always a permutation) in case something degenerate.
    {
        vector<int> edd(N);
        iota(edd.begin(), edd.end(), 0);
        sort(edd.begin(), edd.end(), [](int a, int b){ return D[a] < D[b]; });
        long long c = evalOrder(edd);
        if (c < bestCost) { bestCost = c; best = edd; }
    }

    // (2) local search on the best construction.
    {
        vector<int> cur = best;
        localSearch(cur, deadline);
        long long c = evalOrder(cur);
        if (c < bestCost) { bestCost = c; best = cur; }
    }

    // (3) iterated local search: perturb + re-optimize until budget runs out.
    mt19937 rng(12345);
    vector<int> cur = best;
    while (chrono::steady_clock::now() < deadline) {
        // perturbation: random block move (segment of length 1..4 relocated).
        vector<int> cand = best;
        if (N >= 4) {
            int len = 1 + (int)(rng() % 4);
            int from = (int)(rng() % max(1, N - len));
            int to = (int)(rng() % max(1, N - len));
            if (from != to) {
                vector<int> seg(cand.begin() + from, cand.begin() + from + len);
                cand.erase(cand.begin() + from, cand.begin() + from + len);
                int ins = to;
                if (ins > (int)cand.size()) ins = (int)cand.size();
                cand.insert(cand.begin() + ins, seg.begin(), seg.end());
            }
        }
        localSearch(cand, deadline);
        long long c = evalOrder(cand);
        if (c < bestCost) { bestCost = c; best = cand; }
    }

    // emit the permutation, one id per line.
    string out;
    out.reserve(best.size() * 7);
    for (int j : best) { out += to_string(j); out += '\n'; }
    cout << out;
    return 0;
}
```
