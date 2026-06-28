We are given the symmetric traveling-salesman problem: an $n \times n$ symmetric matrix of distances $c(i,j)$, and we want the shortest tour — a single Hamiltonian cycle through all $n$ cities. With $(n-1)!/2$ distinct tours, enumeration is hopeless, and the exact methods of the day (Held–Karp, then branch and bound when an instance falls outside their tractable class) choke past a few dozen cities — the largest reported exact solution is 64 cities. So the realistic goal is not a certificate of optimality but reliably good tours, fast, with running time that grows gently with $n$. That puts us squarely in iterative improvement: start from a random tour $T$, find a transformation to a shorter tour $T'$, jump there, repeat until nothing improves (a local optimum), then restart from a fresh random tour and keep the best. The entire quality of this scheme lives in one place — the transformation in the middle. A weak step leaves many bad local optima; a strong step makes the local optima few and good, so a random start often lands on the global one.

The standard step is the $k$-opt interchange: delete $k$ tour links, reconnect the resulting paths with $k$ new links (reversing subpaths as needed) so the result is again a tour, and keep it if it is shorter. Croes's 2-opt fixes $k = 2$ — pull two links and reverse the one subsegment that reconnects them; cheap and always feasible, but a shallow neighborhood riddled with mediocre local optima. Lin's 3-opt fixes $k = 3$ — markedly better tours, but the neighborhood scan costs on the order of $n^3$, and one can keep climbing to $k = 4, 5$ for still-better quality at still-worse cost. The decisive frustration is that $k$ must be chosen *in advance*. The work grows like $O(n^k)$, there is no useful bound on how many improving exchanges a tour admits, and there is no way to know the right $k$ for a given instance before running: too small and we stall at a poor optimum, too large and every move is ruinous. The right depth surely varies from instance to instance and even from move to move. Fixing $k$ up front is exactly backwards — the data should tell us how deep to go.

I propose the Lin–Kernighan algorithm: a *variable-depth sequential exchange* in which $k$ is not an input but an outcome. Rather than commit to a $k$ and examine every $k$-subset, I build the set $X = \{x_1, x_2, \dots\}$ of links to remove and the set $Y = \{y_1, y_2, \dots\}$ of links to add one element at a time, choosing the most profitable pair available at each step, and let a stopping rule decide when to quit. The bookkeeping that makes this tractable is additivity of gains. Define the gain of swapping $x_i$ out for $y_i$ in as $g_i = |x_i| - |y_i|$ — length removed minus length added. If a whole group of swaps simply adds up, then for any reachable $T'$ we have $f(T) - f(T') = \sum_i g_i$, and $T'$ is shorter exactly when that sum is positive. This lets me reason about *partial* progress, not just finished exchanges.

To keep the construction one step away from a valid tour at every moment, I tie the broken and added links to a city chain $t_1, t_2, t_3, \dots$. Pick a start city $t_1$; let $x_1 = (t_1, t_2)$ be one of its two tour links — the first break. From $t_2$ add a new link $y_1 = (t_2, t_3)$ to some other city $t_3$. In general
$$x_i = (t_{2i-1}, t_{2i}) \in T, \qquad y_i = (t_{2i}, t_{2i+1}) \notin T,$$
so consecutive links share endpoints and the sequence $x_1, y_1, x_2, y_2, \dots$ zig-zags through the cities as an adjoining chain. This sequential structure is what lets me close up to a tour at any moment. After adding $y_{i-1}$ to reach $t_{2i-1}$ I am holding a Hamiltonian path; of the two tour links at $t_{2i-1}$, exactly one choice of $x_i$ leaves the chain *closeable* — adding $(t_{2i}, t_1)$ yields a single closed tour rather than two disjoint loops. So $x_i$ is forced, uniquely determined by the demand that I can always snap shut. At every depth $i$ there is a legal tour one link away; I never assemble a pile of swaps only to find they cannot be made into a tour.

This immediately gives a stop test. Before committing the open-ended $y_i$, tentatively close up with $y_i^\ast = (t_{2i}, t_1)$; the gain of closing here is $g_i^\ast = |x_i| - |y_i^\ast|$, and because feasibility held this really is a tour, improving on $T$ by $G_{i-1} + g_i^\ast$, where $G_{i-1} = g_1 + \dots + g_{i-1}$ is the running gain of the open chain. So I keep a running best
$$G^\ast = \max_i \, \bigl(G_{i-1} + g_i^\ast\bigr),$$
recording the depth $k$ that achieves it; $G^\ast$ starts at 0 and only rises, and at the end, if $G^\ast > 0$, I perform the depth-$k$ exchange, yielding $f(T') = f(T) - G^\ast$.

What keeps this from exploding is the choice of $y_i$ and the rule for when to stop extending. To make $g_i = |x_i| - |y_i|$ large I want $|y_i|$ small, so I try the nearest neighbors of $t_{2i}$ first — short new links are where the gain is — which bounds the branching to a few candidates. The stopping rule is the heart of the method. The naive answer, "stop as soon as some $g_i$ goes negative," is too greedy: a step that loses a little can set up a later step that wins a lot, the classic escape past a local minimum. The opposite extreme, "keep going while the total could still come out positive," is almost no constraint and lets the search wander forever. The right rule is the *positive-gain criterion*: extend the chain only while the running sum $G_i = g_1 + \dots + g_i$ stays strictly positive, i.e. insist every partial sum is positive, and abandon the chain the instant it hits zero.

It looks far too restrictive to demand positivity at every prefix — surely that throws away good moves whose gain dips negative in the middle. It does not, because of a fact about gains read cyclically: if a sequence $g_1, \dots, g_n$ has positive total sum, then some cyclic permutation of it has all partial sums positive. The proof is the load-bearing step. Write prefix sums $S_j = g_1 + \dots + g_j$ with $S_0 = 0$ and $S_n > 0$, and let $k$ be the *largest* index for which $S_{k-1}$ is the minimum among $S_0, S_1, \dots, S_{n-1}$. Read the sequence starting at $g_k$. A partial sum ending at index $j$ with $k \le j \le n$ equals $g_k + \dots + g_j = S_j - S_{k-1}$; since $S_{k-1}$ is the minimum and $k$ is the *largest* index attaining it, $S_j > S_{k-1}$ for every $j \ge k$, so this is strictly positive. A partial sum that wraps around, ending at $j$ with $1 \le j < k$, equals $(S_n - S_{k-1}) + S_j$, and since $S_j \ge S_{k-1}$ this is $\ge S_n > 0$. Every partial sum is positive. The consequence is exactly what I need: any improving sequential exchange — any chain with positive total gain — can be re-read, starting from the right city, so that every running partial sum stays positive. So if I enforce $G_i > 0$ at every step and am willing to try every starting city $t_1$, I miss no improving move; I only require starting the chain in the right place. In return, the criterion lets me prune away an enormous mass of fruitless deep searches, which is the only reason variable-depth search is affordable at all.

A few supporting choices make it honest and effective. The sets $X$ and $Y$ must stay disjoint within an iteration — a broken link is not re-added and an added link is not re-broken — which stops the chain from undoing its own work, simplifies the gain accounting, keeps running time down, and gives a clean finite stop. Each $y_i$ must also leave room for some $x_{i+1}$ to be broken next, so feasibility survives one more step. Purely sequential chains capture most improving moves but not all — the smallest non-sequential exchange touches four links and admits no adjoining closing order — so at level $i = 2$ only I also allow the *alternate* $x_2$, the infeasible-at-that-moment choice, to recover some of that lost power cheaply; I never permit such a feasibility violation deeper. Finally, backtracking is limited to levels 1 and 2 with about five candidates each. Full backtracking everywhere would find the optimum but is exhaustive search in disguise; empirically the improving move, when it exists, is almost always the very first candidate tried (mean choice number about 1.2 at level 1 and 1.8 at level 2), so capping to five contenders loses essentially no optima while nearly halving the running time. The resulting local optima are at least 3-opt — the first levels reproduce a 2- or 3-exchange — reached in time growing on average about as $n^{2.2}$, with roughly $n/4$ to $n/3$ improvements per local optimum: early moves are deep, large-$k$ chains, and as the tour sharpens they shrink to small $k$ around 2 to 7.

Concretely, I land it as a single self-contained C++17 program that reads $n$ and an $n \times n$ symmetric distance matrix from stdin, runs the search from the identity tour, and prints the resulting tour and its length to stdout. A tour is an ordered city list plus its edge set; `generate(broken, joined)` forms $(\text{tour edges} - \text{broken}) + \text{joined}$, walks the successor map, and reports whether the result is one Hamiltonian cycle — the feasibility / close-up oracle. `closest` is where the nearest-neighbor preference and the positive-gain gate live: from the chain end $t_{2i}$ with running gain $G_{i-1} + |x_i|$ already accumulated, it scores each candidate $y_i = (t_{2i}, \text{node})$ by $G_i = \text{gain} - |y_i|$ and keeps only those with $G_i > 0$ that are neither broken nor a tour edge. `improve` runs level 1 over every start city and both choices of $x_1$, capped at five $y_1$ candidates; `chooseX` picks the forced (or, deep in the chain, the longer) $x_i$, tests the close-up, takes the move when `relink` $> 0$, and otherwise recurses; `chooseY` extends with five candidates at level 2 and only the top-ranked one deeper.

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
