// Simulated Epidemic Containment (control loop) -- heuristic solver.
//
// Objective: over T days, each day choose at most b of the n regions to LOCK
// DOWN so as to MINIMIZE the total new infections of a deterministic
// SIR-on-a-graph epidemic. We read the instance from stdin:
//     n m T b
//     beta gamma kappa
//     u_e v_e w_e               (m undirected weighted edges)
//     I0_0 .. I0_{n-1}          (initial infected fraction per region)
// and write to stdout a SCHEDULE of T lines; line t = "c_t  id ... id"
// (c_t <= b distinct region ids locked on day t).
//
// Dynamics (must match the scorer exactly). Each day t, in order:
//   factor_r = kappa if r locked today else 1.
//   lambda_r = factor_r * (beta*I_r + sum_{(j,w) ~ r} w*beta*I_j*factor_j).
//   newinf_r = S_r * (1 - exp(-lambda_r)); newrec_r = gamma*I_r.
//   update S,I,R simultaneously (all use pre-update I); accumulate sum newinf_r.
//
// Method (the innovation): a ROLLING-HORIZON controller with a k-step LOOK-AHEAD
// marginal-infection score. A myopic "lock the most-infected-today" rule is weak
// because today's hottest region may already be burning out (little S left) while
// a still-susceptible region next to the front would, if left open, ignite a much
// larger wave a few days from now. So instead of ranking regions by current I, we
// score a candidate lockdown by how many infections it AVERTS over the next
// horizon H: we roll the cached current state forward H days under a cheap default
// future policy, once WITHOUT the candidate locked today and once WITH it, and the
// drop in projected cumulative new infections is the candidate's value. Because the
// b picks for one day interact (locking two neighbours overlaps), we build the day's
// set by GREEDY MARGINAL GAIN: pick the single best region, fix it, re-score the
// rest given it is already locked, pick the next best, and so on up to b. The
// lookahead always reuses the SAME cached (S,I,R) snapshot, so each rollout is a
// few cheap SIR steps. Any schedule we hold is feasible by construction (<= b
// distinct ids per day, every id in range), so we never emit an invalid output.
#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
               steady_clock::now().time_since_epoch())
        .count();
}

int n, m, T, b;
double beta_, gamma_, kappa_;
// CSR adjacency
vector<int> adj_head, adj_to;
vector<double> adj_w;

// one SIR step given a "locked" mask (factor = kappa for locked, else 1).
// advances S,I,R in place using simultaneous update; returns sum of new infections.
static inline double step(vector<double> &S, vector<double> &I, vector<double> &R,
                          const vector<char> &locked, vector<double> &factor,
                          vector<double> &newinf, vector<double> &newrec) {
    for (int r = 0; r < n; ++r) factor[r] = locked[r] ? kappa_ : 1.0;
    double total = 0.0;
    for (int r = 0; r < n; ++r) {
        double cross = 0.0;
        for (int e = adj_head[r]; e < adj_head[r + 1]; ++e) {
            int j = adj_to[e];
            cross += adj_w[e] * beta_ * I[j] * factor[j];
        }
        double lam = factor[r] * (beta_ * I[r] + cross);
        double ni = S[r] * (1.0 - exp(-lam));
        if (ni < 0.0) ni = 0.0;
        if (ni > S[r]) ni = S[r];
        newinf[r] = ni;
        newrec[r] = gamma_ * I[r];
    }
    for (int r = 0; r < n; ++r) {
        S[r] -= newinf[r];
        I[r] += newinf[r] - newrec[r];
        R[r] += newrec[r];
        total += newinf[r];
    }
    return total;
}

// Roll the cached state (S0,I0,R0) forward `horizon` days. On day 0 of the rollout
// the regions in `today_lock` are locked; on subsequent rollout days we apply a
// cheap DEFAULT future policy: lock the b regions with the largest current I (this
// stands in for "we will keep containing" so the lookahead is not pessimistically
// open). Returns the projected cumulative new infections over the horizon.
static double rollout(const vector<double> &S0, const vector<double> &I0,
                      const vector<double> &R0, const vector<char> &today_lock,
                      int horizon,
                      // scratch buffers (reused to avoid allocation):
                      vector<double> &S, vector<double> &I, vector<double> &R,
                      vector<char> &locked, vector<double> &factor,
                      vector<double> &newinf, vector<double> &newrec,
                      vector<int> &order) {
    S = S0; I = I0; R = R0;
    double total = 0.0;
    for (int d = 0; d < horizon; ++d) {
        if (d == 0) {
            for (int r = 0; r < n; ++r) locked[r] = today_lock[r];
        } else {
            // default future policy: lock the b most-infected regions.
            for (int r = 0; r < n; ++r) locked[r] = 0;
            order.resize(n);
            for (int r = 0; r < n; ++r) order[r] = r;
            partial_sort(order.begin(), order.begin() + min(b, n), order.end(),
                         [&](int a, int c) {
                             if (I[a] != I[c]) return I[a] > I[c];
                             return a < c;
                         });
            for (int t = 0; t < min(b, n); ++t) locked[order[t]] = 1;
        }
        total += step(S, I, R, locked, factor, newinf, newrec);
    }
    return total;
}

int main() {
    double t_start = now_sec();
    const double TIME_LIMIT = 1.8;  // seconds; leave margin under a 2s budget

    if (scanf("%d %d %d %d", &n, &m, &T, &b) != 4) return 0;
    if (scanf("%lf %lf %lf", &beta_, &gamma_, &kappa_) != 3) return 0;

    vector<vector<pair<int, double>>> g(n);
    for (int e = 0; e < m; ++e) {
        int u, v; double w;
        scanf("%d %d %lf", &u, &v, &w);
        g[u].push_back({v, w});
        g[v].push_back({u, w});
    }
    vector<double> I0(n);
    for (int r = 0; r < n; ++r) scanf("%lf", &I0[r]);

    // build CSR adjacency for cache-friendly inner loop
    adj_head.assign(n + 1, 0);
    for (int r = 0; r < n; ++r) adj_head[r + 1] = adj_head[r] + (int)g[r].size();
    adj_to.assign(adj_head[n], 0);
    adj_w.assign(adj_head[n], 0.0);
    for (int r = 0; r < n; ++r) {
        int p = adj_head[r];
        for (auto &pr : g[r]) { adj_to[p] = pr.first; adj_w[p] = pr.second; ++p; }
    }

    // true (canonical) state we actually advance day by day
    vector<double> S(n), I(n), R(n, 0.0);
    for (int r = 0; r < n; ++r) { S[r] = 1.0 - I0[r]; I[r] = I0[r]; }

    // scratch buffers for rollouts / steps
    vector<double> sS(n), sI(n), sR(n), factor(n), newinf(n), newrec(n);
    vector<char> locked(n), today(n);
    vector<int> order;

    // the produced schedule: for each day, the (<= b) locked region ids.
    vector<vector<int>> schedule(T);

    for (int t = 0; t < T; ++t) {
        // adaptive horizon: longer when we have time, but bounded so each day is cheap.
        int remaining_days = T - t;
        int H = min(remaining_days, 6);
        if (H < 1) H = 1;

        // GREEDY MARGINAL-GAIN selection of up to b regions to lock today.
        // Start from "nothing locked today" and repeatedly add the region whose
        // marginal averted-infections (vs. the current chosen set) is largest and
        // positive. The look-ahead always rolls the SAME cached (S,I,R) snapshot.
        for (int r = 0; r < n; ++r) today[r] = 0;
        vector<int> chosen;
        chosen.reserve(b);

        // base projected infections with the current `today` set (initially empty)
        double base_proj =
            rollout(S, I, R, today, H, sS, sI, sR, locked, factor, newinf, newrec, order);

        for (int pick = 0; pick < b; ++pick) {
            int best_r = -1;
            double best_gain = 1e-12;  // require a strictly positive averted amount
            double best_proj = base_proj;
            // time guard: if we are running low, stop refining and keep what we have
            if (now_sec() - t_start > TIME_LIMIT) break;
            for (int r = 0; r < n; ++r) {
                if (today[r]) continue;  // already locked today
                today[r] = 1;
                double proj = rollout(S, I, R, today, H, sS, sI, sR, locked, factor,
                                      newinf, newrec, order);
                today[r] = 0;
                double gain = base_proj - proj;  // infections averted by adding r
                if (gain > best_gain) {
                    best_gain = gain;
                    best_r = r;
                    best_proj = proj;
                }
            }
            if (best_r < 0) break;  // no region helps anymore; stop early
            today[best_r] = 1;
            chosen.push_back(best_r);
            base_proj = best_proj;
        }

        // record today's locks (feasible by construction: <= b distinct ids)
        schedule[t] = chosen;

        // ADVANCE the true state by one real day under today's chosen locks.
        for (int r = 0; r < n; ++r) locked[r] = today[r];
        step(S, I, R, locked, factor, newinf, newrec);
    }

    // emit the schedule: T lines, each "c id id ...".
    string out;
    out.reserve(T * 8);
    char buf[32];
    for (int t = 0; t < T; ++t) {
        int c = (int)schedule[t].size();
        snprintf(buf, sizeof(buf), "%d", c);
        out += buf;
        for (int id : schedule[t]) {
            out += ' ';
            snprintf(buf, sizeof(buf), "%d", id);
            out += buf;
        }
        out += '\n';
    }
    fputs(out.c_str(), stdout);
    return 0;
}
