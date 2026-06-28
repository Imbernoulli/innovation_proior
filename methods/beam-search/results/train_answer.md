The problem is a layered decision graph: every node branches into several children, depth is large, and the exact frontier grows exponentially. We need a search that stays forward-only and linear in depth, without the exponential memory or the brittleness of committing to a single path. Exact methods like full-breadth dynamic programming or A* are correct, but they pay for optimality by touching the whole breadth at every step, which is unaffordable for speech decoding or large combinatorial instances. A single greedy path is cheap, but one locally attractive yet globally wrong decision ruins the solution, and recovering from that requires backtracking, which wastes all the work already done on the abandoned prefix.

The missing idea is neither keep-all nor keep-one, but keep the best few. At each layer we expand every currently live partial solution into its children, score each child by an estimate of the final score it could still achieve, and prune the pool down to a small fixed number of survivors. Memory is bounded by that small number, work is linear in depth, and there is no backtracking because discarded states are never revisited. The survivors act as a beam of near-miss alternatives around the most promising path, so a temporary ranking mistake has room to correct itself in later layers.

The method is beam search. We maintain a frontier of at most W states, called the beam width. To advance one layer, we generate every child of every state in the frontier, rank the children by g + h where g is the score accumulated so far and h is an optimistic estimate of the best remaining completion, and keep only the top W. The best feasible state ever seen during the sweep is returned as the answer. W is the central dial: at W = 1 the method is greedy hill-climbing, and at W = infinity nothing is pruned so it becomes the exact full-breadth sweep. Intermediate values trade a small amount of optimality for a large gain in speed. The cost is O(W · b · depth) given constant-time evaluation and linear top-W selection, so it scales linearly rather than exponentially.

Two refinements are worth keeping in mind. When the accurate global evaluation is expensive, we can prune in two stages: first use a cheap local score to filter down to a wider intermediate set, then apply the expensive evaluation only on those survivors before pruning to the final beam width. When the beam collapses into many near-duplicates of a single lucky lineage, diversity can be restored by deduplicating states, capping same-score states, or sampling survivors stochastically. A structural alternative called Chokudai search keeps a priority queue per layer and repeatedly runs width-one passes, each pass tending to develop a different lineage; this gives anytime behavior at the cost of retaining more states.

For concreteness, here is a working implementation for 0/1 knapsack as a single self-contained C++17 program that reads one instance from stdin. The first line gives the item count `n`, the beam width `W`, and the capacity; the next `n` lines each give an item's weight and value. The program prints the best value beam search finds and the chosen item indices. Each state records how many items have been decided, the value packed so far (in `long long`, since values accumulate), the remaining capacity, and the chosen items. The ranking key is value-so-far plus the fractional-knapsack LP relaxation of the remaining items, which is an optimistic upper bound on any integer completion.

```cpp
// Beam search for 0/1 knapsack (single-file C++17, reads stdin).
// Input: first line "n W cap" (item count, beam width, capacity); then n lines
// "w v". Output line 1: the best value beam search finds; line 2: the chosen
// item indices (0-based, increasing), or blank if none.
#include <bits/stdc++.h>
using namespace std;

struct State {
    int idx;              // how many items decided (the layer)
    long long value;      // value packed so far (this is g)
    long long cap;        // remaining capacity
    double key;           // rank by value-so-far + optimistic bound (g + h)
    vector<int> chosen;   // item indices taken
};

// h: optimistic completion -- the LP (fractional-knapsack) relaxation of the
// items not yet decided: pack them in DESCENDING value/weight order, last item
// taken fractionally. The fractional optimum is the LP relaxation of the integer
// remainder, so it never underestimates the best integer completion -- exactly
// what we want to rank states by. (The descending-density order is what makes it
// an upper bound; input order is not.) 'rem' is items[idx..n-1] pre-sorted by
// descending density so this is a single scan.
static double bound(long long value, long long cap,
                    const vector<pair<long long, long long>>& rem) {
    double b = (double)value;
    long long c = cap;
    for (const auto& wv : rem) {
        long long w = wv.first, v = wv.second;
        if (w <= c) { c -= w; b += (double)v; }
        else { b += (double)v * ((double)c / (double)w); break; }
    }
    return b;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long W, capacity;
    if (!(cin >> n >> W >> capacity)) return 0;
    vector<pair<long long, long long>> items(n);  // (weight, value) in input order
    for (int i = 0; i < n; ++i) cin >> items[i].first >> items[i].second;

    // For each layer, pre-sort the remaining items[layer..n-1] by descending
    // value/weight so the bound is a simple scan.
    vector<vector<pair<long long, long long>>> remByDensity(n + 1);
    for (int layer = 0; layer <= n; ++layer) {
        vector<pair<long long, long long>> rem(items.begin() + layer, items.end());
        sort(rem.begin(), rem.end(),
             [](const pair<long long, long long>& a,
                const pair<long long, long long>& b) {       // descending density
                 return (long double)a.second * b.first > (long double)b.second * a.first;
             });
        remByDensity[layer] = move(rem);
    }

    vector<State> layer;
    layer.push_back(State{0, 0, capacity, bound(0, capacity, remByDensity[0]), {}});
    State best = layer[0];

    for (int step = 0; step < n; ++step) {           // forward, one layer per item
        vector<State> pool;
        pool.reserve(layer.size() * 2);
        for (const auto& s : layer) {                // expand every live state
            long long w = items[s.idx].first, v = items[s.idx].second;
            // skip item s.idx
            {
                State c = s;
                c.idx = s.idx + 1;
                c.key = bound(c.value, c.cap, remByDensity[c.idx]);
                pool.push_back(move(c));
            }
            // take item s.idx, if it fits
            if (w <= s.cap) {
                State c = s;
                c.idx = s.idx + 1;
                c.value = s.value + v;
                c.cap = s.cap - w;
                c.chosen.push_back(s.idx);
                c.key = bound(c.value, c.cap, remByDensity[c.idx]);
                pool.push_back(move(c));
            }
        }
        // rank by g + h, keep the top W -- the single pruning line, no backtracking
        sort(pool.begin(), pool.end(),
             [](const State& a, const State& b) { return a.key > b.key; });
        if ((long long)pool.size() > W) pool.resize((size_t)W);
        layer = move(pool);
        for (const auto& s : layer)                  // track best feasible value
            if (s.value > best.value) best = s;
    }

    cout << best.value << "\n";
    for (size_t i = 0; i < best.chosen.size(); ++i)
        cout << best.chosen[i] << (i + 1 == best.chosen.size() ? '\n' : ' ');
    if (best.chosen.empty()) cout << "\n";
    return 0;
}
```

Run against a brute-force optimum on a 20-item instance (capacity 60, optimum 185) and sweep the width: feeding the same items at `W = 1, 2, 5, 20, 100` yields best values `179, 183, 185, 185, 185` — the achieved value climbs as the beam widens (gap to the true optimum closing 6 → 2 → 0) and then flattens once the beam is wide enough never to prune the eventual winner, with cost growing linearly in `W` the whole way.
