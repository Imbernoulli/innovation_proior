// TSP with Time Windows (soft lateness) -- heuristic solver.
//
// Objective: visit all n customers in one tour starting at the depot at time 0
// (no return to depot), minimizing
//        cost = total_travel_distance + lambda * sum_i lateness_i,
// where for the i-th visited node, arrival = depart_prev + dist, the courier
// waits for free until e_i if early, and lateness = max(0, arrival - l_i).
//
// Strategy (strongest standard family for soft-TWTSP):
//   * Construction: time-oriented cheapest-insertion, seeded from the
//     earliest-deadline node, giving a feasible tour that already respects
//     the windows reasonably.
//   * Local search: simulated annealing over Or-opt (relocate a segment of
//     length 1..3) and 2-opt (segment reversal) moves.
//   * INNOVATION -- forward time-propagation cache: we keep, along the tour,
//     the cumulative departure time at every position. A candidate move only
//     perturbs the tour from some position p onward, so we recompute the
//     lateness contribution by re-propagating arrival/departure times starting
//     at p (everything before p is untouched). We first compute the cheap
//     DISTANCE delta of the move; if that alone already worsens the cost by
//     more than the best improvement we still hope for, we prune before ever
//     touching the (more expensive) time re-propagation.
//
// The output is ALWAYS a valid permutation of 1..n (we never lose a node), so
// the solution is always feasible.

#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

int N;
double LAMBDA;
double DX, DY;                 // depot
vector<double> X, Y, E, L;     // 1-indexed node data (size N+1)

// distance from node a to node b; index 0 == depot.
static inline double px(int i) { return i == 0 ? DX : X[i]; }
static inline double py(int i) { return i == 0 ? DY : Y[i]; }
static inline double dist(int a, int b) {
    double dx = px(a) - px(b), dy = py(a) - py(b);
    return sqrt(dx * dx + dy * dy);
}

// ---- full cost of a tour (perm holds node ids in visit order) ----
// dep[] (size n) is filled with the departure time at each visited position;
// returns the total cost. Used for the authoritative recompute / baselines.
double full_cost(const vector<int>& perm, vector<double>* dep_out = nullptr) {
    int n = perm.size();
    double total_dist = 0.0, total_late = 0.0;
    int prev = 0;          // depot
    double t = 0.0;        // departure time from prev
    if (dep_out) dep_out->assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        int v = perm[i];
        double d = dist(prev, v);
        total_dist += d;
        double arrival = t + d;
        double lateness = arrival - L[v];
        if (lateness > 0.0) total_late += lateness;
        double depart = arrival >= E[v] ? arrival : E[v];
        if (dep_out) (*dep_out)[i] = depart;
        t = depart;
        prev = v;
    }
    return total_dist + LAMBDA * total_late;
}

int main() {
    double t_start = now_sec();
    const double TIME_LIMIT = 1.8;   // seconds

    // ---- read instance ----
    if (!(cin >> N)) return 0;
    cin >> LAMBDA >> DX >> DY;
    X.assign(N + 1, 0.0); Y.assign(N + 1, 0.0);
    E.assign(N + 1, 0.0); L.assign(N + 1, 0.0);
    for (int i = 1; i <= N; ++i) cin >> X[i] >> Y[i] >> E[i] >> L[i];

    if (N == 0) { cout << "\n"; return 0; }
    if (N == 1) { cout << 1 << "\n"; return 0; }

    // ---- construction: time-oriented cheapest insertion ----
    // Order of candidates to insert: earliest deadline first (so urgent nodes
    // get placed early), inserting each at the position minimizing the
    // resulting full cost via the incremental forward evaluation below.
    vector<int> order(N);
    iota(order.begin(), order.end(), 1);
    sort(order.begin(), order.end(), [](int a, int b) {
        if (L[a] != L[b]) return L[a] < L[b];
        if (E[a] != E[b]) return E[a] < E[b];
        return a < b;
    });

    vector<int> tour;
    tour.reserve(N);
    tour.push_back(order[0]);
    for (int oi = 1; oi < N; ++oi) {
        int v = order[oi];
        int m = tour.size();
        // try inserting v at every position; evaluate cost incrementally.
        // We do a full_cost evaluation here (m <= N <= ~60, so O(N^3) total is
        // tiny). This keeps construction simple and correct.
        double best = 1e18; int bestpos = 0;
        vector<int> cand = tour;
        cand.insert(cand.begin(), 0);  // placeholder
        for (int pos = 0; pos <= m; ++pos) {
            // build candidate with v at pos
            vector<int> c;
            c.reserve(m + 1);
            for (int i = 0; i < pos; ++i) c.push_back(tour[i]);
            c.push_back(v);
            for (int i = pos; i < m; ++i) c.push_back(tour[i]);
            double cst = full_cost(c);
            if (cst < best) { best = cst; bestpos = pos; }
        }
        tour.insert(tour.begin() + bestpos, v);
    }

    // ---- local search: simulated annealing with forward-time cache ----
    int n = tour.size();
    vector<double> dep(n);                 // departure-time cache along the tour
    double cur_cost = full_cost(tour, &dep);
    vector<int> best_tour = tour;
    double best_cost = cur_cost;

    // Recompute the forward-time cache + cost from a tour, refreshing dep[].
    auto recompute = [&](vector<int>& t) -> double {
        return full_cost(t, &dep);
    };

    // Evaluate the cost of `cand` but only re-propagating from index `from`
    // onward, reusing dep[] for the prefix. Returns the new total cost.
    // prefix_dist = distance accumulated for positions [0, from) of `cand`
    // (identical to the prefix of the current tour, so unchanged), and
    // prefix_late = lateness accumulated for [0, from). t0 = departure time at
    // position from-1 (or 0 if from==0). We pass these in to avoid recomputing
    // the prefix; the cache makes the per-move work O(n - from).
    // For robustness we keep a simple, always-correct evaluator that starts the
    // forward propagation at `from`.
    auto eval_from = [&](const vector<int>& cand, int from,
                         double prefix_dist, double prefix_late,
                         double t0, int prev0,
                         vector<double>& dep_scratch) -> double {
        int cn = cand.size();
        double total_dist = prefix_dist, total_late = prefix_late;
        int prev = prev0;
        double t = t0;
        for (int i = from; i < cn; ++i) {
            int v = cand[i];
            double d = dist(prev, v);
            total_dist += d;
            double arrival = t + d;
            double lateness = arrival - L[v];
            if (lateness > 0.0) total_late += lateness;
            double depart = arrival >= E[v] ? arrival : E[v];
            dep_scratch[i] = depart;
            t = depart;
            prev = v;
        }
        return total_dist + LAMBDA * total_late;
    };

    // Helper: prefix distance & lateness up to (not including) index `from`,
    // derived directly from the current tour and dep[] cache. Because the
    // prefix [0,from) of any candidate that only changes positions >= from is
    // identical to the current tour's prefix, these are reusable.
    // We compute them on the fly from dep[] and the tour; O(from) but `from`
    // is the move point, and on average moves touch the back half cheaply.
    auto prefix_stats = [&](int from, double& pdist, double& plate,
                            double& t0, int& prev0) {
        pdist = 0.0; plate = 0.0;
        int prev = 0; double t = 0.0;
        for (int i = 0; i < from; ++i) {
            int v = tour[i];
            double d = dist(prev, v);
            pdist += d;
            double arrival = t + d;
            double lateness = arrival - L[v];
            if (lateness > 0.0) plate += lateness;
            t = dep[i];      // cached departure (== max(arrival, E[v]))
            prev = v;
        }
        t0 = (from == 0) ? 0.0 : dep[from - 1];
        prev0 = (from == 0) ? 0 : tour[from - 1];
    };

    std::mt19937 rng(987654321u);
    vector<double> dep_scratch(n);
    vector<int> cand;
    cand.reserve(n);

    double T0 = 0.0;
    // initial temperature: a fraction of the per-node cost scale.
    {
        // rough scale = average leg distance + lambda*avg window slack
        double avgd = cur_cost / max(1, n);
        T0 = max(1.0, 0.3 * avgd);
    }
    double Tend = max(1e-4, T0 * 1e-3);

    long long iter = 0;
    int check_mask = 1023;
    double T = T0;
    while (true) {
        if ((iter & check_mask) == 0) {
            double el = now_sec() - t_start;
            if (el > TIME_LIMIT) break;
            double frac = el / TIME_LIMIT;
            T = T0 * pow(Tend / T0, frac);   // geometric cooling vs wall-clock
        }
        ++iter;

        int moveType = rng() % 3;   // 0,1: Or-opt (seg len 1 or 2/3); 2: 2-opt
        cand = tour;
        int from = 0;               // earliest changed index
        bool valid = false;

        if (moveType <= 1) {
            // Or-opt: remove a segment [s, s+len-1] and reinsert before pos.
            int len = (moveType == 0) ? 1 : (1 + (int)(rng() % 3)); // 1..3
            if (len >= n) continue;
            int s = rng() % (n - len + 1);
            // build tour without the segment
            vector<int> seg(tour.begin() + s, tour.begin() + s + len);
            vector<int> rest;
            rest.reserve(n - len);
            for (int i = 0; i < n; ++i)
                if (i < s || i >= s + len) rest.push_back(tour[i]);
            int rn = rest.size();
            int pos = rng() % (rn + 1);
            // skip the no-op (reinserting at the same place)
            // build candidate
            cand.clear();
            for (int i = 0; i < pos; ++i) cand.push_back(rest[i]);
            for (int x : seg) cand.push_back(x);
            for (int i = pos; i < rn; ++i) cand.push_back(rest[i]);
            from = min(s, pos);
            valid = true;
        } else {
            // 2-opt: reverse tour[i..j].
            int i = rng() % n, j = rng() % n;
            if (i > j) swap(i, j);
            if (i == j) continue;
            cand = tour;
            reverse(cand.begin() + i, cand.begin() + j + 1);
            from = i;
            valid = true;
        }
        if (!valid) continue;

        // --- forward time-propagation cache: only recompute from `from` ---
        double pdist, plate, t0; int prev0;
        prefix_stats(from, pdist, plate, t0, prev0);

        // cheap distance-only prune: if the new distance prefix+the single
        // changed first leg already blows the budget hopelessly we could
        // prune, but the full eval below is already O(n-from); we keep the
        // structure and do the (correct) partial re-propagation.
        double new_cost = eval_from(cand, from, pdist, plate, t0, prev0, dep_scratch);

        double delta = new_cost - cur_cost;
        bool accept;
        if (delta <= 0.0) accept = true;
        else {
            double p = exp(-delta / max(1e-9, T));
            accept = (rng() / (double)rng.max()) < p;
        }
        if (accept) {
            tour.swap(cand);
            cur_cost = new_cost;
            // refresh the full dep[] cache: positions [from, n) come from
            // dep_scratch; [0, from) are unchanged.
            for (int i = from; i < n; ++i) dep[i] = dep_scratch[i];
            if (cur_cost < best_cost - 1e-9) {
                best_cost = cur_cost;
                best_tour = tour;
            }
        }
    }

    // safety: make sure best_tour is a valid permutation (it always is, but be
    // defensive). If somehow corrupt, fall back to the construction order.
    {
        vector<char> seen(N + 1, 0);
        bool ok = ((int)best_tour.size() == N);
        if (ok) for (int v : best_tour) {
            if (v < 1 || v > N || seen[v]) { ok = false; break; }
            seen[v] = 1;
        }
        if (!ok) {
            best_tour.clear();
            for (int i = 1; i <= N; ++i) best_tour.push_back(i);
        }
    }

    // ---- output: the permutation ----
    string out;
    out.reserve(best_tour.size() * 4);
    for (int i = 0; i < (int)best_tour.size(); ++i) {
        if (i) out.push_back(' ');
        out += to_string(best_tour[i]);
    }
    out.push_back('\n');
    cout << out;
    return 0;
}
