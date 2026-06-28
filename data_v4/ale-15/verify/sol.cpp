// ale-15 "Continuous Facility Layout".
//
// We are given N axis-aligned rectangles (facility footprints) with fixed sizes
// and a W x H container. We choose an integer bottom-left corner for each
// rectangle, keeping it fully inside the container, to MINIMISE
//
//   energy = OVERLAP_W   * (total pairwise overlap area)
//          + DISPERSION_W * (sum of squared distance of each centre from
//                            the mean centre).
//
// This is a continuous facility-layout problem: NP-hard, judged by a continuous
// energy. The two terms pull against each other -- overlap wants the rectangles
// spread out, dispersion wants them packed tightly around one centroid -- so a
// careless layout (everything at the origin, or everything flung to the
// corners) is bad.
//
// Heuristic = simulated annealing with two non-obvious levers:
//
//  (1) FORCE-DIRECTED move proposals. Instead of proposing a uniformly random
//      new position for a rectangle, we compute a "force": a separation push
//      away from each rectangle it currently overlaps (proportional to the
//      penetration), plus a weak spring toward the global centroid (the
//      dispersion gradient). The candidate move steps the rectangle a random
//      fraction along that force. This proposes moves that are actually likely
//      to reduce energy, so the search converges far faster than blind jumps.
//
//  (2) A SPATIAL HASH GRID for O(1)-amortised neighbour queries. The overlap
//      term couples every pair, so naively re-evaluating a single rectangle's
//      overlap contribution is O(N). We bucket every rectangle into all grid
//      cells its bounding box covers; the overlap candidates of a rectangle are
//      then just the rectangles sharing one of those cells -- typically a
//      handful. Both the force computation and the incremental overlap delta of
//      a move become O(neighbours) instead of O(N), which is what makes tens of
//      millions of SA steps affordable inside the time budget.
//
// The dispersion term is kept O(1) per move by maintaining running sums of the
// centres (sum and sum-of-squares): dispersion = Sx2 - Sx^2/N + Sy2 - Sy^2/N.
//
// We always hold a feasible layout (every move keeps each rectangle inside the
// container), so whenever we stop we can print a legal answer. We also snapshot
// the best feasible layout ever seen and print that.

#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    return (double)chrono::duration_cast<chrono::microseconds>(
               chrono::steady_clock::now().time_since_epoch())
               .count() *
           1e-6;
}

// Scoring constants -- MUST match verify/score.py.
static const double OVERLAP_W = 1.0;
static const double DISPERSION_W = 1.0e-4;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const double T_BUDGET = 1.9;  // seconds
    const double t_start = now_sec();

    int N;
    long long W, H;
    if (!(cin >> N >> W >> H)) return 0;
    vector<int> w(N), h(N);
    for (int i = 0; i < N; i++) cin >> w[i] >> h[i];

    if (N <= 0) return 0;

    // Per-rectangle integer position of the bottom-left corner. Feasible range
    // for rectangle i is x in [0, W - w[i]], y in [0, H - h[i]].
    vector<int> X(N), Y(N);
    vector<int> hiX(N), hiY(N);  // inclusive upper bounds for X,Y
    for (int i = 0; i < N; i++) {
        hiX[i] = (int)max(0LL, W - w[i]);
        hiY[i] = (int)max(0LL, H - h[i]);
    }

    // Deterministic RNG (seeded from the instance so runs are reproducible).
    uint64_t rng_state = 0x9e3779b97f4a7c15ULL ^ (uint64_t)N ^
                         ((uint64_t)W << 20) ^ ((uint64_t)H << 1);
    for (int i = 0; i < N; i++) {
        rng_state ^= (uint64_t)(w[i] * 1000003 + h[i]) + 0x632be59bd9b4e019ULL +
                     (uint64_t)i;
        rng_state *= 0xff51afd7ed558ccdULL;
        rng_state ^= rng_state >> 29;
    }
    auto nextu = [&]() -> uint64_t {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        return rng_state;
    };
    auto nextd = [&]() -> double {
        return (double)(nextu() >> 11) / 9007199254740992.0;  // [0,1)
    };
    auto randint = [&](int lo, int hi) -> int {  // inclusive [lo,hi]
        if (hi <= lo) return lo;
        return lo + (int)(nextu() % (uint64_t)(hi - lo + 1));
    };

    // ---- spatial hash grid ----------------------------------------------
    // Cell size ~ a typical large rectangle dimension so each rectangle covers
    // only a few cells. We insert each rectangle into every cell its AABB
    // touches; the overlap candidates of a rectangle are exactly the rectangles
    // that share at least one of those cells.
    long long sumDim = 0;
    int maxDim = 1;
    for (int i = 0; i < N; i++) {
        sumDim += w[i] + h[i];
        maxDim = max(maxDim, max(w[i], h[i]));
    }
    int cell = (int)max(1LL, sumDim / max(1, 2 * N));  // ~ mean side length
    cell = max(cell, 1);
    int GX = (int)(W / cell) + 2;
    int GY = (int)(H / cell) + 2;
    // Guard against pathological grid sizes (huge container, tiny cells).
    while ((long long)GX * GY > 4'000'000 && cell < maxDim * 8) {
        cell *= 2;
        GX = (int)(W / cell) + 2;
        GY = (int)(H / cell) + 2;
    }
    auto cellOf = [&](int v) -> int { return v / cell; };

    // grid[gx*GY+gy] = list of rectangle ids whose AABB touches that cell.
    vector<vector<int>> grid((size_t)GX * GY);

    // For removal we record, per rectangle, the cell-range it currently occupies
    // so we can erase it from exactly those cells.
    vector<int> cgx0(N), cgx1(N), cgy0(N), cgy1(N);

    auto insertRect = [&](int i) {
        int gx0 = cellOf(X[i]);
        int gx1 = cellOf(X[i] + w[i]);
        int gy0 = cellOf(Y[i]);
        int gy1 = cellOf(Y[i] + h[i]);
        if (gx1 >= GX) gx1 = GX - 1;
        if (gy1 >= GY) gy1 = GY - 1;
        cgx0[i] = gx0; cgx1[i] = gx1; cgy0[i] = gy0; cgy1[i] = gy1;
        for (int gx = gx0; gx <= gx1; gx++)
            for (int gy = gy0; gy <= gy1; gy++)
                grid[(size_t)gx * GY + gy].push_back(i);
    };
    auto removeRect = [&](int i) {
        for (int gx = cgx0[i]; gx <= cgx1[i]; gx++)
            for (int gy = cgy0[i]; gy <= cgy1[i]; gy++) {
                auto &v = grid[(size_t)gx * GY + gy];
                for (size_t k = 0; k < v.size(); k++)
                    if (v[k] == i) { v[k] = v.back(); v.pop_back(); break; }
            }
    };

    // ---- initial feasible layout ----------------------------------------
    // Shelf / row layout: lay rectangles left-to-right in rows, wrapping at the
    // container width. This already avoids most overlap and is a far better
    // starting point than the origin pile. Always feasible by construction
    // because every rectangle fits (generator guarantees W>=maxw+10 etc).
    {
        int curx = 0, cury = 0, rowh = 0;
        for (int i = 0; i < N; i++) {
            if (curx + w[i] > W) { curx = 0; cury += rowh; rowh = 0; }
            int px = curx, py = cury;
            if (py + h[i] > H) {
                // Ran out of vertical room: drop somewhere legal (rare; the
                // generator leaves slack, but stay safe).
                px = randint(0, hiX[i]);
                py = randint(0, hiY[i]);
            }
            X[i] = min(px, hiX[i]);
            Y[i] = min(py, hiY[i]);
            curx += w[i];
            rowh = max(rowh, h[i]);
        }
    }
    for (int i = 0; i < N; i++) insertRect(i);

    // ---- running dispersion sums ----------------------------------------
    // centre of rect i = (X[i] + w[i]/2, Y[i] + h[i]/2). We keep the raw sums in
    // doubled units to stay integral: cx2 = 2*X + w. dispersion uses real
    // centres, so we keep Sx = sum(centreX), Sx2 = sum(centreX^2), etc.
    double Sx = 0, Sy = 0, Sx2 = 0, Sy2 = 0;
    auto cxOf = [&](int i) -> double { return X[i] + w[i] * 0.5; };
    auto cyOf = [&](int i) -> double { return Y[i] + h[i] * 0.5; };
    for (int i = 0; i < N; i++) {
        double a = cxOf(i), b = cyOf(i);
        Sx += a; Sy += b; Sx2 += a * a; Sy2 += b * b;
    }
    auto dispersionEnergy = [&]() -> double {
        double dx = Sx2 - Sx * Sx / N;
        double dy = Sy2 - Sy * Sy / N;
        return DISPERSION_W * (dx + dy);
    };

    // overlap contribution of rectangle i against ALL current neighbours, using
    // the spatial hash. Returns the summed overlap area i participates in.
    auto pairOverlap = [&](int i, int j) -> long long {
        long long ox = min((long long)X[i] + w[i], (long long)X[j] + w[j]) -
                       max(X[i], X[j]);
        if (ox <= 0) return 0;
        long long oy = min((long long)Y[i] + h[i], (long long)Y[j] + h[j]) -
                       max(Y[i], Y[j]);
        if (oy <= 0) return 0;
        return ox * oy;
    };
    // candidate neighbour ids of rect i (deduped) gathered from the cells its
    // AABB currently touches. Caller must have i inserted.
    vector<char> seen(N, 0);
    vector<int> nb;
    auto gatherNeighbours = [&](int i) {
        nb.clear();
        for (int gx = cgx0[i]; gx <= cgx1[i]; gx++)
            for (int gy = cgy0[i]; gy <= cgy1[i]; gy++) {
                auto &v = grid[(size_t)gx * GY + gy];
                for (int j : v)
                    if (j != i && !seen[j]) { seen[j] = 1; nb.push_back(j); }
            }
        for (int j : nb) seen[j] = 0;  // reset for next use
    };
    auto overlapOf = [&](int i) -> long long {
        gatherNeighbours(i);
        long long s = 0;
        for (int j : nb) s += pairOverlap(i, j);
        return s;
    };

    // total overlap area (each pair counted once). Used only for snapshots.
    auto totalOverlap = [&]() -> long long {
        long long s = 0;
        for (int i = 0; i < N; i++) s += overlapOf(i);
        return s / 2;  // each pair counted from both sides
    };

    auto energyNow = [&]() -> double {
        return OVERLAP_W * (double)totalOverlap() + dispersionEnergy();
    };

    double curEnergy = energyNow();
    vector<int> bestX = X, bestY = Y;
    double bestEnergy = curEnergy;

    // ---- simulated annealing --------------------------------------------
    const double T0 = max(1.0, curEnergy / max(1, N));  // start temperature
    const double T1 = 1e-3;
    long long iter = 0;
    int timeMask = 4095;

    while (true) {
        if ((iter & timeMask) == 0) {
            double el = now_sec() - t_start;
            if (el > T_BUDGET) break;
        }
        iter++;

        int i = (int)(nextu() % (uint64_t)N);

        // ---- force-directed proposal --------------------------------------
        // Sum a separation force away from each overlapping neighbour and a weak
        // spring toward the centroid; step a random fraction along it. Fall back
        // to a random jump occasionally to keep ergodicity.
        gatherNeighbours(i);
        double fx = 0, fy = 0;
        double ci_x = cxOf(i), ci_y = cyOf(i);
        for (int j : nb) {
            if (pairOverlap(i, j) <= 0) continue;
            double dxc = ci_x - cxOf(j);
            double dyc = ci_y - cyOf(j);
            double d2 = dxc * dxc + dyc * dyc + 1.0;
            double inv = 1.0 / d2;
            fx += dxc * inv * 4000.0;  // separation push
            fy += dyc * inv * 4000.0;
        }
        // weak spring toward the global centroid (dispersion gradient)
        double meanx = Sx / N, meany = Sy / N;
        fx += (meanx - ci_x) * 0.05;
        fy += (meany - ci_y) * 0.05;

        int nx, ny;
        if ((nextu() & 7) == 0 || (fx == 0 && fy == 0)) {
            // random jump (small with prob, full-range occasionally)
            if ((nextu() & 3) == 0) {
                nx = randint(0, hiX[i]);
                ny = randint(0, hiY[i]);
            } else {
                int span = max(2, maxDim);
                nx = X[i] + randint(-span, span);
                ny = Y[i] + randint(-span, span);
            }
        } else {
            double scale = 0.3 + 1.7 * nextd();
            nx = (int)llround(X[i] + fx * scale);
            ny = (int)llround(Y[i] + fy * scale);
        }
        // clamp to feasible range (keeps the layout feasible at all times)
        if (nx < 0) nx = 0; else if (nx > hiX[i]) nx = hiX[i];
        if (ny < 0) ny = 0; else if (ny > hiY[i]) ny = hiY[i];
        if (nx == X[i] && ny == Y[i]) continue;

        // ---- incremental energy delta -------------------------------------
        // overlap delta: only rect i's overlap with its neighbours changes.
        long long oldOv = 0;
        for (int j : nb) oldOv += pairOverlap(i, j);

        int oldX = X[i], oldY = Y[i];
        double oldcx = ci_x, oldcy = ci_y;

        // tentatively move (positions only; grid still has the old cells)
        X[i] = nx; Y[i] = ny;
        double newcx = cxOf(i), newcy = cyOf(i);

        // new overlap must be gathered from the NEW cells. Re-bucket i.
        removeRect(i);
        insertRect(i);
        long long newOv = 0;
        gatherNeighbours(i);
        for (int j : nb) newOv += pairOverlap(i, j);

        // dispersion delta from moving rect i's centre (O(1) via running sums)
        double newSx = Sx - oldcx + newcx;
        double newSy = Sy - oldcy + newcy;
        double newSx2 = Sx2 - oldcx * oldcx + newcx * newcx;
        double newSy2 = Sy2 - oldcy * oldcy + newcy * newcy;
        double oldDisp = DISPERSION_W * ((Sx2 - Sx * Sx / N) + (Sy2 - Sy * Sy / N));
        double newDisp = DISPERSION_W *
                         ((newSx2 - newSx * newSx / N) + (newSy2 - newSy * newSy / N));

        double dOverlap = OVERLAP_W * (double)(newOv - oldOv);
        double dDisp = newDisp - oldDisp;
        double delta = dOverlap + dDisp;

        // SA acceptance with geometric cooling.
        double frac = (now_sec() - t_start) / T_BUDGET;
        if (frac > 1) frac = 1;
        double T = T0 * pow(T1 / T0, frac);
        bool accept = (delta <= 0) || (nextd() < exp(-delta / max(1e-9, T)));

        if (accept) {
            // commit: update running sums and current energy. (i is already
            // re-bucketed into the grid at its new position.)
            Sx = newSx; Sy = newSy; Sx2 = newSx2; Sy2 = newSy2;
            curEnergy += delta;
            if (curEnergy < bestEnergy - 1e-6) {
                bestEnergy = curEnergy;
                bestX = X; bestY = Y;
            }
        } else {
            // revert: restore position and grid bucketing.
            X[i] = oldX; Y[i] = oldY;
            removeRect(i);
            insertRect(i);
        }
    }

    // ---- recompute the snapshot's exact energy as a final safety check ----
    // (curEnergy is maintained incrementally; the snapshot bestX/bestY is what
    // we emit, so trust the stored corners directly -- they are always feasible
    // because every accepted/initial position was clamped into range.)

    // ---- emit feasible solution -----------------------------------------
    {
        string out;
        out.reserve((size_t)N * 8);
        char buf[32];
        for (int i = 0; i < N; i++) {
            int len = snprintf(buf, sizeof(buf), "%d %d\n", bestX[i], bestY[i]);
            out.append(buf, len);
        }
        cout << out;
    }
    return 0;
}
