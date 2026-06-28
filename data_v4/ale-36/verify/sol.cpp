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
