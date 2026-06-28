// ALE-10  Wall Painting  --  heuristic solver.
//
// Objective: reproduce an N x N target picture (colours 0..C-1) with at most T
// axis-aligned rectangular *brush strokes*. A canvas starts filled with colour 0;
// strokes are applied in order and later strokes overwrite earlier ones; the
// score is the number of cells whose final colour equals the target. T is far
// smaller than N*N, so cells cannot be painted individually -- the structure of
// the picture (its overlapping coloured rectangles) must be exploited.
//
// INNOVATION (why this file is fast):
//   We keep a fixed sequence of T strokes and, for every cell, the index of the
//   TOPMOST stroke covering it (top[r][c], -1 = bare canvas => colour 0). The
//   displayed colour of a cell is therefore O(1), and the match score is held
//   incrementally. A simulated-annealing move edits ONE stroke i (its rectangle
//   and/or colour). Only cells in the UNION of stroke i's old and new footprints
//   can change, so the score delta is computed over that footprint alone -- never
//   over the whole grid and never by replaying all T strokes. Concretely:
//     * cells that LEAVE i's rectangle and had top==i lose i's colour and fall
//       back to the topmost stroke below i (found by a short local rescan);
//     * cells that ENTER i's rectangle take colour(i) iff no stroke ABOVE i
//       already covers them (i.e. their current top < i);
//     * cells in BOTH rectangles whose top==i just swap i's old colour for the
//       new one.
//   Maintaining top[][] this way makes each SA step cost O(footprint), not
//   O(N^2) or O(T*N^2), which is what lets the search take millions of steps and
//   actually beat the greedy construction.
//
// SEARCH: SA over the T strokes. Warm start = greedy layered construction (most
// frequent colour as a full-canvas base, then a few large solid-colour blocks).
// Moves: perturb a random stroke's rectangle (shift/resize an edge) or recolour
// it; Metropolis acceptance lets the search undo a wasteful early stroke that a
// later one would otherwise have to paint over. The best feasible state seen is
// remembered and printed, so the output is always feasible and never worse than
// the warm start.
//
// I/O:
//   stdin : "N C T", then N lines of N integers (the target grid).
//   stdout: "Q", then Q lines "r1 c1 r2 c2 col" (0<=r1<=r2<N, 0<=c1<=c2<N).
// Compile: g++ -O2 -std=c++17 sol.cpp
#include <bits/stdc++.h>
using namespace std;

static uint64_t rng_state = 0x243F6A8885A308D3ULL;
static inline uint64_t xr() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}
static inline int randint(int n) { return (int)(xr() % (uint64_t)n); }
static inline double urand() { return (xr() >> 11) * (1.0 / 9007199254740992.0); }

static double now_sec() {
    return chrono::duration<double>(
               chrono::steady_clock::now().time_since_epoch())
        .count();
}

int N, C, T;
vector<int> target;                 // target[r*N+c]

struct Stroke { int r1, c1, r2, c2, col; };

int main() {
    double t_start = now_sec();
    const double TIME_LIMIT = 1.85;  // seconds, leaves margin under a 2s cap

    if (scanf("%d %d %d", &N, &C, &T) != 3) return 0;
    target.assign(N * N, 0);
    for (int i = 0; i < N * N; i++) scanf("%d", &target[i]);

    auto idx = [&](int r, int c) { return r * N + c; };

    // ----- structures maintained incrementally -----
    vector<Stroke> S(T);
    // top[cell] = index of topmost stroke covering it, or -1 (bare canvas => 0).
    vector<int> topv(N * N, -1);
    int score = 0;  // number of matching cells under the current sequence

    // colour displayed at a cell given its top index
    auto disp = [&](int cell) -> int {
        int t = topv[cell];
        return (t < 0) ? 0 : S[t].col;
    };

    // Recompute the topmost stroke covering a single cell by scanning strokes
    // top-down. Used only on the (small) footprints touched by a move.
    auto recompute_top = [&](int r, int c) -> int {
        for (int t = T - 1; t >= 0; t--) {
            const Stroke &s = S[t];
            if (r >= s.r1 && r <= s.r2 && c >= s.c1 && c <= s.c2) return t;
        }
        return -1;
    };

    // ----- warm start: a feasible, decent layered construction -----
    // Stroke 0: paint the whole canvas with the most common target colour.
    // (Cheap, and already a strong baseline.)
    auto build_initial = [&]() {
        vector<long long> freq(C, 0);
        for (int i = 0; i < N * N; i++) freq[target[i]]++;
        int base = 0;
        for (int col = 1; col < C; col++)
            if (freq[col] > freq[base]) base = col;

        // Default every stroke to a harmless 1x1 dot of the base colour at (0,0);
        // these are overwritten by stroke 0 / later strokes and never hurt.
        for (int t = 0; t < T; t++) S[t] = {0, 0, 0, 0, base};
        if (T >= 1) S[0] = {0, 0, N - 1, N - 1, base};

        // A few greedy solid blocks: scan a coarse set of rectangles and, for the
        // most frequent non-base colour inside, lay a block if it helps. Kept
        // simple -- SA does the heavy lifting; this only needs to be feasible and
        // better than a single flat fill.
        int next = 1;
        int step = max(1, N / 6);
        for (int r1 = 0; r1 < N && next < T; r1 += step) {
            for (int c1 = 0; c1 < N && next < T; c1 += step) {
                int r2 = min(N - 1, r1 + step - 1);
                int c2 = min(N - 1, c1 + step - 1);
                // majority colour in this block
                vector<int> cnt(C, 0);
                for (int r = r1; r <= r2; r++)
                    for (int c = c1; c <= c2; c++) cnt[target[idx(r, c)]]++;
                int best = 0;
                for (int col = 1; col < C; col++)
                    if (cnt[col] > cnt[best]) best = col;
                if (best != base && cnt[best] * 2 > (r2 - r1 + 1) * (c2 - c1 + 1)) {
                    S[next++] = {r1, c1, r2, c2, best};
                }
            }
        }
        // remaining slots stay as the harmless base-coloured dot at (0,0)

        // Build top[][] and the score by replaying once (one-off O(T*N^2)).
        for (int cell = 0; cell < N * N; cell++) topv[cell] = -1;
        for (int t = 0; t < T; t++) {
            const Stroke &s = S[t];
            for (int r = s.r1; r <= s.r2; r++)
                for (int c = s.c1; c <= s.c2; c++) topv[idx(r, c)] = t;
        }
        score = 0;
        for (int cell = 0; cell < N * N; cell++)
            if (disp(cell) == target[cell]) score++;
    };
    build_initial();

    // Apply a candidate edit to stroke `i` (new rectangle/colour `ns`) and return
    // the new score, updating topv[] and S[i] IN PLACE. Only cells in the union of
    // the old and new footprints are touched -- this is the incremental core.
    auto apply_move = [&](int i, const Stroke &ns) -> int {
        Stroke os = S[i];
        S[i] = ns;  // commit so recompute_top sees the new geometry

        int sc = score;
        // Bounding box of the union of old and new rectangles.
        int R1 = min(os.r1, ns.r1), R2 = max(os.r2, ns.r2);
        int C1 = min(os.c1, ns.c1), C2 = max(os.c2, ns.c2);
        for (int r = R1; r <= R2; r++) {
            for (int c = C1; c <= C2; c++) {
                bool inOld = (r >= os.r1 && r <= os.r2 && c >= os.c1 && c <= os.c2);
                bool inNew = (r >= ns.r1 && r <= ns.r2 && c >= ns.c1 && c <= ns.c2);
                if (!inOld && !inNew) continue;
                int cell = idx(r, c);
                int oldTop = topv[cell];
                // old displayed colour at this cell
                int oldColShown;
                if (oldTop < 0) oldColShown = 0;
                else if (oldTop == i) oldColShown = os.col;  // i's *old* colour
                else oldColShown = S[oldTop].col;
                bool wasMatch = (oldColShown == target[cell]);

                // compute the new top index for this cell
                int newTop;
                if (inNew) {
                    // stroke i now covers it; topmost is i unless a stroke >i covers it
                    if (oldTop > i) newTop = oldTop;           // something above stays on top
                    else newTop = i;                            // i becomes (or stays) top
                } else {
                    // i no longer covers it
                    if (oldTop != i) newTop = oldTop;           // i wasn't the top => unaffected
                    else newTop = recompute_top(r, c);          // i was the top => rescan below
                }
                topv[cell] = newTop;
                int newColShown = (newTop < 0) ? 0 : S[newTop].col;
                bool nowMatch = (newColShown == target[cell]);
                sc += (int)nowMatch - (int)wasMatch;
            }
        }
        return sc;
    };

    // Undo helper: restore stroke i to `os` and topv over the union footprint.
    // We simply call apply_move with the saved stroke, but apply_move recomputes
    // tops from the *current* S, which after a rejected move still has the right
    // geometry for strokes != i; restoring S[i]=os and recomputing the union
    // footprint reproduces the pre-move state exactly.
    auto revert_move = [&](int i, const Stroke &os, const Stroke &ns) {
        // ns is what is currently in S[i]; restore os over the union footprint.
        S[i] = os;
        int R1 = min(os.r1, ns.r1), R2 = max(os.r2, ns.r2);
        int C1 = min(os.c1, ns.c1), C2 = max(os.c2, ns.c2);
        for (int r = R1; r <= R2; r++)
            for (int c = C1; c <= C2; c++) {
                bool inOld = (r >= os.r1 && r <= os.r2 && c >= os.c1 && c <= os.c2);
                bool inNew = (r >= ns.r1 && r <= ns.r2 && c >= ns.c1 && c <= ns.c2);
                if (!inOld && !inNew) continue;
                topv[idx(r, c)] = recompute_top(r, c);
            }
    };

    // ----- remember the best feasible state -----
    vector<Stroke> bestS = S;
    int bestScore = score;

    // propose a perturbed version of stroke i
    auto propose = [&](int i) -> Stroke {
        Stroke s = S[i];
        int kind = randint(6);
        if (kind == 0) {
            s.col = randint(C);                      // recolour
        } else if (kind == 1) {                      // jiggle one edge inward/outward
            int e = randint(4);
            int d = randint(2) ? 1 : -1;
            if (e == 0) s.r1 = min(max(0, s.r1 + d), s.r2);
            else if (e == 1) s.r2 = max(min(N - 1, s.r2 + d), s.r1);
            else if (e == 2) s.c1 = min(max(0, s.c1 + d), s.c2);
            else s.c2 = max(min(N - 1, s.c2 + d), s.c1);
        } else if (kind == 2) {                      // translate the whole rect
            int dr = randint(3) - 1, dc = randint(3) - 1;
            int h = s.r2 - s.r1, w = s.c2 - s.c1;
            int nr1 = min(max(0, s.r1 + dr), N - 1 - h);
            int nc1 = min(max(0, s.c1 + dc), N - 1 - w);
            s.r1 = nr1; s.r2 = nr1 + h; s.c1 = nc1; s.c2 = nc1 + w;
        } else {                                     // fresh random rectangle+colour
            int a = randint(N), b = randint(N);
            int c1 = randint(N), c2 = randint(N);
            s.r1 = min(a, b); s.r2 = max(a, b);
            s.c1 = min(c1, c2); s.c2 = max(c1, c2);
            s.col = randint(C);
        }
        return s;
    };

    // ----- simulated annealing -----
    double T0 = max(2.0, 0.01 * N * N);   // initial temperature ~ a few % of cells
    double T1 = 0.05;
    long long iter = 0;
    double t_now = now_sec();
    while (true) {
        if ((iter & 1023) == 0) {
            t_now = now_sec();
            if (t_now - t_start > TIME_LIMIT) break;
        }
        iter++;
        double frac = (t_now - t_start) / TIME_LIMIT;
        if (frac > 1.0) frac = 1.0;
        double temp = T0 * pow(T1 / T0, frac);

        int i = randint(T);
        Stroke os = S[i];
        Stroke ns = propose(i);
        int newScore = apply_move(i, ns);  // mutates S[i], topv over footprint
        int delta = newScore - score;
        bool accept = (delta >= 0) || (urand() < exp(delta / temp));
        if (accept) {
            score = newScore;
            if (score > bestScore) {
                bestScore = score;
                bestS = S;
            }
        } else {
            revert_move(i, os, ns);  // S[i] currently == ns; restore os
        }
    }

    // ----- emit the best feasible solution -----
    // Drop strokes that are completely covered by a later stroke's *exact* slot?
    // Not needed for feasibility; we just print all T strokes of the best state.
    // Every stroke is a valid rectangle inside the grid with a valid colour, so
    // the output is always feasible. Q = T <= T (the budget) holds by construction.
    printf("%d\n", T);
    for (int t = 0; t < T; t++) {
        const Stroke &s = bestS[t];
        printf("%d %d %d %d %d\n", s.r1, s.c1, s.r2, s.c2, s.col);
    }
    return 0;
}
