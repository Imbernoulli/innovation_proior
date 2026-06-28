// Heat-Diffusion Tile Coloring -- strong heuristic solver.
//
// Energy to MINIMIZE over a binary coloring x[r][c] in {0,1} (pins fixed):
//     E = W * (# 4-adjacent pairs with differing coatings)
//       + sum_cells  h[i] * [ x[i] != t[i] ]
//
// This is an Ising/Potts pairwise energy with a unary field -- a submodular binary
// MRF. The method follows the problem-specific RELAXATION lever:
//
//   (1) CONTINUOUS RELAXATION + HEAT DIFFUSION. Relax x_i to u_i in [0,1] and replace
//       the binary energy by its quadratic surrogate
//           Q = W * sum_{i~j} (u_i-u_j)^2 + sum_i h_i (u_i - t_i)^2 .
//       Coordinate-minimizing Q (Gauss-Seidel) gives the closed-form update
//           u_i <- ( W * sum_{j~i} u_j + h_i * t_i ) / ( W * deg_i + h_i ),
//       clamped to pins. This is exactly a heat-diffusion sweep with a source term;
//       a few sweeps drive u to the smooth steady state. THRESHOLD at 0.5 to round
//       it into a binary coloring -- a strong warm start that already respects both
//       smoothness and the field.
//
//   (2) BOUNDARY LOCAL SEARCH with O(1) energy delta (ICM + annealing). Flipping one
//       free cell changes E by an amount computed from only that cell's field term and
//       its <=4 neighbours -- an O(1) delta, no global recompute. We sweep the
//       interface cells (those with a differing neighbour, where every gain lives),
//       greedily accepting improving flips (Iterated Conditional Modes), and run a
//       short simulated-annealing pass that also accepts small uphill flips to escape
//       the local minima ICM gets stuck in. Pinned cells are never flipped, so every
//       intermediate -- and the final output -- is FEASIBLE by construction.
//
// Output: N rows of N space-separated bits. Always feasible within the time budget.
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

int N, W;
vector<int> H, T, P;          // field, target, pin (-1 free, else fixed bit)
vector<char> x;               // current coloring
inline int ID(int r, int c) { return r * N + c; }

// Energy delta of flipping free cell i (from x[i] to 1-x[i]). O(1).
inline long long flip_delta(int r, int c) {
    int i = ID(r, c);
    int a = x[i], b = 1 - a;
    long long d = 0;
    // field term
    d += (long long)H[i] * ((b != T[i]) - (a != T[i]));
    // interface term over 4 neighbours
    if (r > 0)      { int j = ID(r - 1, c); d += (long long)W * ((b != x[j]) - (a != x[j])); }
    if (r + 1 < N)  { int j = ID(r + 1, c); d += (long long)W * ((b != x[j]) - (a != x[j])); }
    if (c > 0)      { int j = ID(r, c - 1); d += (long long)W * ((b != x[j]) - (a != x[j])); }
    if (c + 1 < N)  { int j = ID(r, c + 1); d += (long long)W * ((b != x[j]) - (a != x[j])); }
    return d;
}

long long total_energy() {
    long long E = 0;
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) {
            int i = ID(r, c);
            if (c + 1 < N && x[ID(r, c + 1)] != x[i]) E += W;
            if (r + 1 < N && x[ID(r + 1, c)] != x[i]) E += W;
            if (x[i] != T[i]) E += H[i];
        }
    return E;
}

int main() {
    auto t_start = chrono::steady_clock::now();
    const double TIME_LIMIT = 1.8; // seconds, comfortably under the 2s budget

    if (scanf("%d %d", &N, &W) != 2) return 0;
    int M = N * N;
    H.assign(M, 0); T.assign(M, 0); P.assign(M, -1);
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++) {
            int i = ID(r, c);
            if (scanf("%d %d %d", &H[i], &T[i], &P[i]) != 3) return 0;
        }

    // ---- (1) Continuous relaxation: Gauss-Seidel heat-diffusion sweeps. ----
    vector<double> u(M, 0.0);
    for (int i = 0; i < M; i++) {
        if (P[i] != -1) u[i] = P[i];            // pin clamps the field
        else u[i] = T[i] ? 0.75 : 0.25;          // warm init toward the target
    }
    // degree of each cell (number of in-grid neighbours)
    auto deg = [&](int r, int c) {
        return (r > 0) + (r + 1 < N) + (c > 0) + (c + 1 < N);
    };
    int sweeps = 60;
    for (int s = 0; s < sweeps; s++) {
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
                int i = ID(r, c);
                if (P[i] != -1) { u[i] = P[i]; continue; }
                double num = 0.0, den = (double)W * deg(r, c) + (double)H[i];
                if (r > 0)     num += W * u[ID(r - 1, c)];
                if (r + 1 < N) num += W * u[ID(r + 1, c)];
                if (c > 0)     num += W * u[ID(r, c - 1)];
                if (c + 1 < N) num += W * u[ID(r, c + 1)];
                num += (double)H[i] * T[i];
                u[i] = (den > 0.0) ? num / den : (T[i] ? 1.0 : 0.0);
            }
    }
    // ---- threshold / round into a binary coloring (pins respected) ----
    x.assign(M, 0);
    for (int i = 0; i < M; i++) {
        if (P[i] != -1) x[i] = (char)P[i];
        else x[i] = (u[i] >= 0.5) ? 1 : 0;
    }

    // ---- (2a) ICM: sweep free cells, greedily flip while it strictly improves. ----
    bool improved = true;
    int icm_rounds = 0;
    while (improved && icm_rounds < 200) {
        improved = false;
        icm_rounds++;
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
                int i = ID(r, c);
                if (P[i] != -1) continue;
                if (flip_delta(r, c) < 0) { x[i] ^= 1; improved = true; }
            }
        if ((icm_rounds & 7) == 0) {
            double el = chrono::duration<double>(chrono::steady_clock::now() - t_start).count();
            if (el > TIME_LIMIT * 0.5) break;
        }
    }

    // keep the best coloring seen
    vector<char> best = x;
    long long bestE = total_energy();
    long long curE = bestE;

    // free-cell list for random sampling during annealing
    vector<int> freeCells;
    for (int i = 0; i < M; i++) if (P[i] == -1) freeCells.push_back(i);

    // ---- (2b) Simulated annealing on single free-cell flips (O(1) delta). ----
    if (!freeCells.empty()) {
        double T0 = max(1.0, (double)W * 1.5), T1 = 0.05;
        long long iter = 0;
        while (true) {
            if ((iter & 2047) == 0) {
                double el = chrono::duration<double>(chrono::steady_clock::now() - t_start).count();
                if (el > TIME_LIMIT) break;
            }
            iter++;
            double frac = chrono::duration<double>(chrono::steady_clock::now() - t_start).count() / TIME_LIMIT;
            if (frac > 1) frac = 1;
            double Temp = T0 * pow(T1 / T0, frac);

            int i = freeCells[xr() % freeCells.size()];
            int r = i / N, c = i % N;
            long long d = flip_delta(r, c);
            if (d <= 0 || urand() < exp(-(double)d / Temp)) {
                x[i] ^= 1;
                curE += d;
                if (curE < bestE) { bestE = curE; best = x; }
            }
        }
    }

    // ---- final greedy clean-up pass on the best coloring (ICM to a local min) ----
    x = best;
    improved = true;
    while (improved) {
        improved = false;
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) {
                int i = ID(r, c);
                if (P[i] != -1) continue;
                if (flip_delta(r, c) < 0) { x[i] ^= 1; improved = true; }
            }
    }

    // ---- output: N rows of N bits (pins are already honoured) ----
    string out;
    out.reserve(2 * M);
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            out.push_back('0' + x[ID(r, c)]);
            out.push_back(c + 1 < N ? ' ' : '\n');
        }
    }
    fputs(out.c_str(), stdout);
    return 0;
}
