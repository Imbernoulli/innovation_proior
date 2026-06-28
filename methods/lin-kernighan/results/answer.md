# The Lin–Kernighan Algorithm for the Traveling-Salesman Problem

## Problem

Symmetric TSP: given an `n × n` symmetric distance matrix `c(i,j)`, find a tour (Hamiltonian cycle
visiting every city once) of minimum total length. Local-search heuristics improve a tour by
`k`-exchanges — deleting `k` links and reconnecting with `k` others — but fixing `k` in advance forces
a poor tradeoff: small `k` (2-opt, 3-opt) stalls at mediocre local optima, large `k` is `O(n^k)` and
unaffordable, and the right `k` cannot be known before running. Lin–Kernighan removes the fixed-`k`
drawback with a *variable-depth* sequential exchange.

## Key idea

Build the exchange one link at a time and let the depth `k` be decided dynamically. Maintain a city
chain `t_1, t_2, t_3, ...` where `x_i = (t_{2i-1}, t_{2i})` is a tour link to break and
`y_i = (t_{2i}, t_{2i+1})` is a new link to add; consecutive links share endpoints, so the sequence
`x_1, y_1, x_2, y_2, ...` is an adjoining chain. Three rules make it work:

- **Sequential exchange + feasibility (close-up).** Given the previously added `y_{i-1}` reaching
  `t_{2i-1}`, the broken link `x_i` is forced to be the one tour link whose removal leaves the chain
  *closeable*: adding `(t_{2i}, t_1)` yields a single tour. So at every depth there is a valid tour
  one link away, and the search may stop at any `i`.
- **Positive-gain criterion.** With `g_i = c(x_i) - c(y_i)` and running gain `G_i = g_1 + ... + g_i`,
  extend the chain only while `G_i > 0`. This loses no improving move: if a sequence of gains has a
  positive total sum, some cyclic permutation has all partial sums positive, so trying every start
  `t_1` recovers any improving sequential exchange. The proof: with prefix sums `S_j = g_1+...+g_j`
  (`S_0 = 0`, `S_n > 0`), let `k` be the largest index minimizing `S_{k-1}` over `S_0,...,S_{n-1}`;
  reading from `g_k`, a partial sum ending at `j ≥ k` is `S_j - S_{k-1} > 0`, and one wrapping to
  `j < k` is `(S_n - S_{k-1}) + S_j ≥ S_n > 0`.
- **Best close-up wins.** Track `G* = max_i (G_{i-1} + g_i*)` where `g_i* = c(x_i) - c(t_{2i}, t_1)`
  is the gain of closing up at depth `i`, with `k` the achieving depth. Stop when no admissible
  `x_i, y_i` remain or when `G_i ≤ G*`; if `G* > 0`, perform the depth-`k` exchange (new length
  `f(T) - G*`) and restart.

Supporting choices: prefer nearest neighbors for `y_i` (small `|y_i|` ⇒ large `g_i`, tight
branching); keep `X` and `Y` disjoint (no link broken then re-added, or vice versa, within an
iteration); allow the single infeasible alternate `x_2` (only at `i = 2`) to recover some
non-sequential moves; and backtrack only at levels 1 and 2 with ~5 candidates each, since the
improving move is almost always the first candidate (mean choice 1.2 and 1.8). The resulting local
optima are at least 3-opt, reached in time growing about as `n^{2.2}`.

## Algorithm

```
1.  Generate a random tour T.
2.  G* = 0; choose a start city t1; let x1 be a tour link at t1; i = 1.
3.  From the other end t2 of x1, choose y1 = (t2, t3) ∉ T with g1 = |x1| - |y1| > 0.
        If none exists, backtrack (Step 6).
4.  i = i + 1.
    (a) x_i = (t_{2i-1}, t_{2i}) is the unique tour link such that joining t_{2i} to t1 can close up.
    (b) Close-up check: with yi* = (t_{2i}, t1), gi* = |x_i| - |yi*|. If G_{i-1} + gi* > G*,
        set G* = G_{i-1} + gi*, k = i.
    (c) Choose y_i = (t_{2i}, t_{2i+1}) ∉ T, nearest first, with:
          - G_i = G_{i-1} + g_i > 0          (positive-gain criterion)
          - x_i, y_i disjoint from earlier links
          - an x_{i+1} can still be broken   (so close-up stays possible next step)
        If such y_i exists, go to 4; else go to 5.
5.  Stop the chain when no admissible x_i, y_i remain or G_i ≤ G*. If G* > 0, take the depth-k
    exchange: f(T') = f(T) - G*, T = T', go to 2.
6.  If G* = 0, limited backtracking: alternate y2 (increasing length, g1+g2 > 0); then alternate x2
    (infeasible close-up allowed only here); then alternate y1; then alternate x1; then new t1.
7.  When all n choices of t1 are exhausted without profit, T is a local optimum; optionally restart.
```

## Code

A faithful single-file C++17 implementation of the Lin–Kernighan step: the `t1..t_{2i}` city chain,
the broken/added edge sets `X`/`Y`, the positive-gain and feasibility (close-up) checks, and
backtracking limited to levels 1 and 2. It reads `n` and an `n × n` symmetric distance matrix from
stdin, runs the variable-depth search from the identity tour, and prints the resulting tour (city
indices in visit order) and its length to stdout.

```cpp
// Lin-Kernighan variable-depth local search for the symmetric TSP.
// Reads: n, then an n x n symmetric distance matrix (row-major) from stdin.
// Writes: the tour (n city indices, 0-based, in visit order) and its length to stdout.
#include <bits/stdc++.h>
using namespace std;

static int N;
static vector<vector<double>> D;                 // symmetric cost matrix
static vector<vector<int>> NB;                   // neighbour lists, nearest first

static inline double dist(int i, int j) { return D[i][j]; }

// undirected edge as an ordered pair (a < b)
static inline long long key(int i, int j) {
    if (i > j) swap(i, j);
    return (long long)i * N + j;
}

struct Tour {
    vector<int> order;                           // city visit order
    vector<int> pos;                             // pos[city] = index in order
    unordered_set<long long> edges;              // current tour edges (membership)
    int size = 0;

    explicit Tour(const vector<int>& o) : order(o), size((int)o.size()) {
        pos.assign(size, 0);
        edges.reserve(size * 2);
        for (int i = 0; i < size; i++) pos[order[i]] = i;
        for (int i = 0; i < size; i++) edges.insert(key(order[(i + size - 1) % size], order[i]));
    }

    // the two tour-neighbours (predecessor, successor) of a city
    pair<int,int> around(int node) const {
        int idx = pos[node];
        int pred = order[(idx + size - 1) % size];
        int succ = order[(idx + 1) % size];
        return {pred, succ};
    }

    bool contains(long long e) const { return edges.count(e) != 0; }

    // New edge set = (tour - broken) + joined; rebuild and check it is ONE Hamiltonian cycle.
    // The feasibility / close-up oracle. On success, fills out with the rebuilt order.
    bool generate(const unordered_set<long long>& broken,
                  const unordered_set<long long>& joined,
                  vector<int>& out) const {
        // assemble the resulting edge list as adjacency (degree <= 2 expected)
        vector<array<int,2>> adj(size, {-1, -1});
        vector<int> deg(size, 0);
        auto addEdge = [&](int a, int b) {
            if (deg[a] < 2) adj[a][deg[a]] = b;
            deg[a]++;
            if (deg[b] < 2) adj[b][deg[b]] = a;
            deg[b]++;
        };
        int edgeCount = 0;
        for (long long e : edges) {
            if (broken.count(e)) continue;
            int a = (int)(e / N), b = (int)(e % N);
            addEdge(a, b);
            edgeCount++;
        }
        for (long long e : joined) {
            int a = (int)(e / N), b = (int)(e % N);
            addEdge(a, b);
            edgeCount++;
        }
        if (edgeCount != size) return false;
        for (int i = 0; i < size; i++) if (deg[i] != 2) return false;

        // walk from 0; a single cycle visits all `size` nodes before returning
        out.clear();
        out.reserve(size);
        int prev = -1, cur = 0;
        for (int step = 0; step < size; step++) {
            out.push_back(cur);
            int a = adj[cur][0], b = adj[cur][1];
            int nxt = (step == 0) ? min(a, b)               // canonical orientation at the start
                                  : ((a != prev) ? a : b);
            prev = cur;
            cur = nxt;
            if (cur == 0 && step + 1 < size) return false;  // premature return => subtours
        }
        return cur == 0 && (int)out.size() == size;
    }
};

struct LinKernighan {
    vector<int> path;            // current best tour order
    double cost;                 // its length
    unordered_set<string> seen;  // tours already reached (cycle guard)

    explicit LinKernighan(const vector<int>& p) : path(p) { cost = pathCost(p); }

    static double pathCost(const vector<int>& p) {
        double c = dist(p.back(), p.front());
        for (size_t i = 1; i < p.size(); i++) c += dist(p[i - 1], p[i]);
        return c;
    }

    static string tourKey(const vector<int>& p) {
        string s;
        s.reserve(p.size() * 4);
        for (int x : p) { s += to_string(x); s += ','; }
        return s;
    }

    // Candidate y_i = (t2i, node): keep only positive running gain G_i, not broken, not a tour
    // edge; order by how good the next break looks. `gain` already holds G_{i-1} + |x_i|.
    // Returns (node, G_i) pairs, best first.
    vector<pair<int,double>> closest(int t2i, const Tour& tour, double gain,
                                     const unordered_set<long long>& broken,
                                     const unordered_set<long long>& joined) const {
        // candidates kept in first-insertion order (neighbour-list order), then stably
        // sorted by `diff` descending -- mirrors Python's dict + stable sorted().
        vector<int> nodes;                               // distinct candidate nodes, in order
        unordered_map<int, pair<double,double>> cand;    // node -> {diff, Gi}
        for (int node : NB[t2i]) {
            long long yi = key(t2i, node);
            double Gi = gain - dist(t2i, node);          // running gain if we ADD y_i
            if (Gi <= 0 || broken.count(yi) || tour.contains(yi))
                continue;                                // POSITIVE-GAIN CRITERION + disjointness
            auto pr = tour.around(node);
            for (int s : {pr.first, pr.second}) {        // the x_{i+1} we could break next
                long long xi = key(node, s);
                if (!broken.count(xi) && !joined.count(xi)) {
                    double diff = dist(node, s) - dist(t2i, node);
                    auto it = cand.find(node);
                    if (it == cand.end()) { cand[node] = {diff, Gi}; nodes.push_back(node); }
                    else if (diff > it->second.first) it->second = {diff, Gi};
                }
            }
        }
        stable_sort(nodes.begin(), nodes.end(),          // descending by diff, ties keep order
                    [&](int a, int b){ return cand[a].first > cand[b].first; });
        vector<pair<int,double>> out;                    // (node, Gi)
        for (int node : nodes) out.emplace_back(node, cand[node].second);
        return out;
    }

    // Choose y_i from the close-up-ordered candidates: 5 at level 2, top-ranked only deeper.
    bool chooseY(const Tour& tour, int t1, int t2i, double gain,
                 const unordered_set<long long>& broken,
                 const unordered_set<long long>& joined) {
        auto ordered = closest(t2i, tour, gain, broken, joined);
        int top = (broken.size() == 2) ? 5 : 1;
        for (auto& nc : ordered) {
            int node = nc.first; double Gi = nc.second;
            auto added = joined; added.insert(key(t2i, node));   // y_i = (t2i, node)
            if (chooseX(tour, t1, node, Gi, broken, added))
                return true;
            if (--top == 0) return false;
        }
        return false;
    }

    // Choose x_i to break from `last`; try to close up, else extend via chooseY.
    bool chooseX(const Tour& tour, int t1, int last, double gain,
                 const unordered_set<long long>& broken,
                 const unordered_set<long long>& joined) {
        vector<int> aroundNodes;
        auto pr = tour.around(last);
        if (broken.size() == 4) {                        // deep: commit to the longer x_i
            aroundNodes.push_back(dist(pr.first, last) > dist(pr.second, last) ? pr.first : pr.second);
        } else {
            aroundNodes.push_back(pr.first);             // both tour links are candidate x_i
            aroundNodes.push_back(pr.second);
        }
        for (int t2i : aroundNodes) {
            long long xi = key(last, t2i);
            double Gi = gain + dist(last, t2i);          // add |x_i| to the running gain
            if (joined.count(xi) || broken.count(xi)) continue;  // keep X and Y disjoint
            auto added = joined; added.insert(key(t2i, t1));     // close-up edge (t2i, t1)
            auto removed = broken; removed.insert(xi);
            double relink = Gi - dist(t2i, t1);          // improvement G* if we close up here
            vector<int> newTour;
            bool isTour = tour.generate(removed, added, newTour);
            if (!isTour && added.size() > 2) continue;   // infeasible close-up allowed only at i = 2
            if (isTour && seen.count(tourKey(newTour))) return false; // already seen -> avoid cycling
            if (isTour && relink > 1e-12) {              // strictly better tour: take it
                path = newTour;
                cost -= relink;
                return true;
            }
            bool choice = chooseY(tour, t1, t2i, Gi, removed, joined); // else extend the chain
            if (broken.size() == 2) {                    // full backtracking at level 2
                if (choice) return true;
            } else {
                return choice;                           // single shot for i > 2
            }
        }
        return false;
    }

    // Level 1: every start city, both choices of x1, up to five y1 candidates.
    bool improve() {
        Tour tour(path);
        for (int t1 : path) {                            // try every start city
            auto ar = tour.around(t1);
            for (int t2 : {ar.first, ar.second}) {       // both choices of x1 (alternate x1)
                unordered_set<long long> broken; broken.insert(key(t1, t2));
                double gain = dist(t1, t2);              // |x1|
                auto close = closest(t2, tour, gain, broken, {}); // y1 with g1 > 0
                int tries = 5;                           // limited level-1 backtracking (~5)
                for (auto& nc : close) {
                    int t3 = nc.first; double Gi = nc.second;
                    if (t3 == ar.first || t3 == ar.second) continue; // t3 not a tour-neighbour of t1
                    unordered_set<long long> joined; joined.insert(key(t2, t3));
                    if (chooseX(tour, t1, t3, Gi, broken, joined))
                        return true;                     // improvement -> restart
                    if (--tries == 0) break;
                }
            }
        }
        return false;
    }

    void optimise() {
        // candidate neighbour lists, ordered by distance ascending (nearest first)
        NB.assign(N, {});
        for (int i = 0; i < N; i++) {
            vector<pair<double,int>> nbrs;
            for (int j = 0; j < N; j++)
                if (j != i && dist(i, j) > 0) nbrs.emplace_back(dist(i, j), j);
            sort(nbrs.begin(), nbrs.end());
            for (auto& pr : nbrs) NB[i].push_back(pr.second);
        }
        seen.clear();
        bool better = true;
        while (better) {                                 // restart at every improving move
            better = improve();
            seen.insert(tourKey(path));
        }
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> N) || N <= 0) return 0;
    D.assign(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) cin >> D[i][j];

    vector<int> start(N);
    iota(start.begin(), start.end(), 0);                 // identity tour 0,1,...,n-1
    LinKernighan lk(start);
    lk.optimise();

    for (int i = 0; i < N; i++) cout << lk.path[i] << (i + 1 < N ? ' ' : '\n');
    cout << fixed << setprecision(4) << lk.cost << '\n';
    return 0;
}
```
