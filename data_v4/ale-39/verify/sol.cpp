// Parameter Placement for a Simulated Controller -- strong heuristic solver.
//
// We tune S segments of K = 3 gains each (a PD-on-error + velocity-damping triple)
// that drive a deterministic scalar plant to track a reference trajectory. Each gain
// is an integer CODE in [0, Q] mapping to value LO_k + (HI_k-LO_k)*code/Q. The
// objective is to MINIMIZE the squared tracking cost
//     COST = sum_t (ref[t] - x_after(t))^2
// and the reported score grows as COST drops below the open-loop (zero-gain) cost.
//
// THE LEVER -- PREFIX-STATE CACHE / LOCALIZED RE-SIMULATION.
//   The plant is deterministic and each segment owns DISJOINT gains, so changing a
//   single segment s's gain perturbs the trajectory ONLY from the start of segment s
//   onward. We treat the forward simulation as a black-box oracle but CACHE the plant
//   state (x, v, e_prev) and the accumulated cost at every segment boundary. A
//   single-code perturbation in segment s then re-runs the sim from boundary s only --
//   O((S-s)*seg_len) work instead of a full O(T) re-evaluation. That cheap incremental
//   delta is what makes a dense coordinate-descent / simulated-annealing search over
//   the S*K codes affordable.
//
// METHOD.
//   (1) Greedy coordinate-descent construction. Sweep segments left to right; for each
//       gain try a coarse grid of codes and keep the best, re-simulating only the tail
//       from that segment using the cached prefix state. Gives a strong warm start.
//   (2) Simulated annealing over single-code moves with the same localized re-sim and
//       O(1)-amortized acceptance, escaping the coordinate-descent local minimum.
//   (3) Final coordinate-descent polish to a local optimum.
// Every code stays in [0, Q] at all times, so every intermediate -- and the final
// output -- is FEASIBLE by construction. The simulation matches the scorer exactly
// (same DRAG, same fixed-point reads, same divergence guard).
#include <bits/stdc++.h>
using namespace std;

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static inline uint64_t xr() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static inline double urand() { return (xr() >> 11) * (1.0 / 9007199254740992.0); }

static const double SCALE = 1000.0;
static const double DRAG = 0.02;
static const double DIVERGE = 1e15;

int S, K, Q, T, seg_len;
vector<double> LO, HI;         // per-gain box bounds (K entries)
vector<double> rf, dst;        // reference + disturbance, length T

// current codes: code[s*K + k]
vector<int> code;
// gain VALUE for a code on gain k
inline double codeVal(int k, int c) { return LO[k] + (HI[k] - LO[k]) * (double)c / (double)Q; }

// cached plant state at each segment boundary b in [0, S]:
// state just BEFORE simulating segment b (i.e. at global step b*seg_len).
struct St { double x, v, ep, cost; bool ok; };
vector<St> boundary;           // size S+1; boundary[0] is the fixed initial state

// Re-simulate from segment boundary `b0` to the end, using boundary[b0] as the start
// state. Fills boundary[b0+1 .. S]. Returns total COST (boundary[S].cost) or +inf if
// the simulation diverges. Uses the CURRENT `code` for the gains.
double resimulate(int b0) {
    St cur = boundary[b0];
    if (!cur.ok) return numeric_limits<double>::infinity();
    double x = cur.x, v = cur.v, ep = cur.ep, cost = cur.cost;
    for (int s = b0; s < S; s++) {
        double g0 = codeVal(0, code[s * K + 0]);
        double g1 = codeVal(1, code[s * K + 1]);
        double g2 = codeVal(2, code[s * K + 2]);
        int t0 = s * seg_len, t1 = t0 + seg_len;
        for (int t = t0; t < t1; t++) {
            double e = rf[t] - x;
            double f = g0 * e + g1 * (e - ep) + g2 * v;
            v = v + f - DRAG * v + dst[t];
            x = x + v;
            double err = rf[t] - x;
            cost += err * err;
            ep = e;
            if (!isfinite(x) || !isfinite(v) || !isfinite(cost) ||
                fabs(x) > DIVERGE || fabs(v) > DIVERGE || cost > DIVERGE) {
                // mark all downstream boundaries diverged
                for (int b = s + 1; b <= S; b++) boundary[b] = {0, 0, 0, 0, false};
                return numeric_limits<double>::infinity();
            }
        }
        boundary[s + 1] = {x, v, ep, cost, true};
    }
    return boundary[S].cost;
}

int main() {
    auto t_start = chrono::steady_clock::now();
    const double TIME_LIMIT = 1.7; // seconds, comfortably under the 2s budget
    auto elapsed = [&]() {
        return chrono::duration<double>(chrono::steady_clock::now() - t_start).count();
    };

    // ---- read instance ----
    if (scanf("%d %d %d", &S, &K, &Q) != 3) return 0;
    LO.assign(K, 0); HI.assign(K, 0);
    for (int k = 0; k < K; k++) {
        long long lo, hi;
        if (scanf("%lld %lld", &lo, &hi) != 2) return 0;
        LO[k] = lo / SCALE; HI[k] = hi / SCALE;
    }
    if (scanf("%d", &T) != 1) return 0;
    rf.assign(T, 0); dst.assign(T, 0);
    for (int t = 0; t < T; t++) { long long r; if (scanf("%lld", &r) != 1) return 0; rf[t] = r / SCALE; }
    for (int t = 0; t < T; t++) { long long d; if (scanf("%lld", &d) != 1) return 0; dst[t] = d / SCALE; }
    seg_len = T / S;

    // ---- fixed initial boundary state ----
    boundary.assign(S + 1, St{0, 0, 0, 0, false});
    boundary[0] = {rf[0], 0.0, 0.0, 0.0, true};    // x0 = ref[0], v0 = 0, e_prev0 = 0

    // ---- start from all-zero codes (feasible) ----
    code.assign(S * K, 0);
    double curCost = resimulate(0);

    // ---- (1) greedy coordinate-descent construction, left to right ----
    // For each segment/gain, scan a coarse code grid and keep the best, re-simulating
    // only the tail from that segment. The cached prefix state makes each trial cheap.
    {
        const int GRID = 21;
        bool stop = false;
        for (int s = 0; s < S && !stop; s++) {
            for (int k = 0; k < K; k++) {
                int idx = s * K + k;
                int bestC = code[idx];
                double bestCost = curCost;
                for (int gi = 0; gi <= GRID; gi++) {
                    int c = (int)llround((double)Q * gi / GRID);
                    if (c < 0) c = 0; if (c > Q) c = Q;
                    code[idx] = c;
                    double cst = resimulate(s);   // only seg s..S-1 re-run
                    if (cst < bestCost) { bestCost = cst; bestC = c; }
                }
                code[idx] = bestC;
                curCost = resimulate(s);          // commit the chosen code, refresh cache
                if (elapsed() > TIME_LIMIT * 0.4) { stop = true; break; } // warm start only
            }
        }
    }

    // best-so-far snapshot
    vector<int> best = code;
    double bestCost = curCost;

    // ---- (2) simulated annealing over single-code moves ----
    // A move perturbs one code in some segment s by a (decaying) step; we re-simulate
    // only segments s..S-1 from the cached boundary[s]. Accept by Metropolis on the
    // cost delta. Keep the best feasible coloring seen.
    {
        // characteristic cost scale for the temperature schedule
        double costScale = max(1.0, bestCost / max(1, T));
        double T0 = costScale * 50.0, T1 = costScale * 0.05;
        long long iter = 0;
        // we must keep boundary[] consistent with `code`; resimulate(0) once to sync
        curCost = resimulate(0);
        while (true) {
            if ((iter & 1023) == 0) {
                if (elapsed() > TIME_LIMIT * 0.9) break;
            }
            iter++;
            double frac = elapsed() / TIME_LIMIT;
            if (frac > 1) frac = 1;
            double Temp = T0 * pow(T1 / T0, frac);

            int idx = (int)(xr() % (uint64_t)(S * K));
            int s = idx / K;
            int oldC = code[idx];
            // step size shrinks over time; at least 1
            int span = max(1, (int)llround((double)Q * (0.5 - 0.45 * frac)));
            int step = (int)(xr() % (uint64_t)(2 * span + 1)) - span;
            int newC = oldC + step;
            if (newC < 0) newC = 0; if (newC > Q) newC = Q;
            if (newC == oldC) continue;

            code[idx] = newC;
            double cand = resimulate(s);          // localized re-sim from segment s
            double d = cand - curCost;
            if (d <= 0 || urand() < exp(-d / Temp)) {
                curCost = cand;                   // accept; boundary[] now matches code
                if (curCost < bestCost) { bestCost = curCost; best = code; }
            } else {
                code[idx] = oldC;                 // reject; restore code...
                resimulate(s);                    // ...and the cached boundary state
            }
        }
    }

    // ---- (3) final coordinate-descent polish on the best codes ----
    code = best;
    curCost = resimulate(0);
    {
        bool improved = true;
        int rounds = 0;
        bool stop = false;
        const int offs[] = {1, -1, 5, -5, 20, -20, 60, -60};
        while (improved && rounds < 50 && !stop) {
            improved = false; rounds++;
            for (int s = 0; s < S && !stop; s++) {
                for (int k = 0; k < K; k++) {
                    int idx = s * K + k;
                    int oldC = code[idx];
                    int bestC = oldC; double bc = curCost;
                    for (int o : offs) {
                        int c = oldC + o;
                        if (c < 0 || c > Q) continue;
                        code[idx] = c;
                        double cst = resimulate(s);
                        if (cst < bc) { bc = cst; bestC = c; }
                    }
                    code[idx] = bestC;
                    curCost = resimulate(s);
                    if (bestC != oldC) improved = true;
                    if (elapsed() > TIME_LIMIT) { stop = true; break; }
                }
            }
        }
        if (curCost < bestCost) { bestCost = curCost; best = code; }
    }

    // ---- output: S lines of K codes (always in [0,Q] -> feasible) ----
    code = best;
    string out;
    out.reserve(8 * S * K);
    for (int s = 0; s < S; s++) {
        for (int k = 0; k < K; k++) {
            out += to_string(code[s * K + k]);
            out.push_back(k + 1 < K ? ' ' : '\n');
        }
    }
    fputs(out.c_str(), stdout);
    return 0;
}
