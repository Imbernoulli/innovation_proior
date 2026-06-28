// ale-05 "Relay Tower Placement" (p-median).
//
// We are given N households on a plane and a budget of K relay towers. Every
// tower must sit ON a household; we minimise the sum over households of the
// Euclidean distance to the nearest tower. This is the (discrete) p-median /
// k-medoid problem: NP-hard, judged by a continuous cost.
//
// Heuristic: k-means++-style medoid seeding, then PAM (Partitioning Around
// Medoids) swap local search accelerated with FIRST- and SECOND-nearest
// caching (FastPAM1). The cache is the lever: with d1[i] (distance to the
// nearest medoid) and d2[i] (distance to the second-nearest) stored for every
// household, a candidate swap "remove medoid m, add non-medoid h" is evaluated
// in O(N) instead of the naive O(N*K) re-assignment, and one full pass costs
// O(N * (#non-medoids)) while computing the best swap for EVERY removed medoid
// simultaneously. A geometric-cooling acceptance is not needed: the swap step
// is steepest-descent on the true cost, and the second-nearest cache makes it
// affordable. We always hold a feasible medoid set, so any time we stop we can
// print a legal answer.

#include <bits/stdc++.h>
using namespace std;

static inline double now_sec() {
    return (double)chrono::duration_cast<chrono::microseconds>(
               chrono::steady_clock::now().time_since_epoch())
               .count() *
           1e-6;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const double T_BUDGET = 1.9;  // seconds
    const double t_start = now_sec();

    int N, K;
    if (!(cin >> N >> K)) return 0;
    vector<double> X(N), Y(N);
    for (int i = 0; i < N; i++) cin >> X[i] >> Y[i];

    // Degenerate guards: always emit a feasible answer.
    if (N <= 0) {
        return 0;
    }
    if (K <= 0) {
        // No towers requested: nothing to print but the count.
        cout << 0 << "\n";
        return 0;
    }
    if (K >= N) {
        // Every household can host a tower; choosing all N of them drives the
        // cost to 0 and is optimal. The valid case is K == N (the generator
        // guarantees K << N, so K > N is never produced); we emit exactly K
        // distinct in-range indices, which requires K <= N. When K == N we
        // print all N households; for the impossible K > N we still print K
        // headers but cap indices to the valid range as a best effort.
        cout << K << "\n";
        for (int i = 0; i < K; i++) cout << ((i % N) + 1) << "\n";
        return 0;
    }

    auto dist = [&](int a, int b) -> double {
        double dx = X[a] - X[b], dy = Y[a] - Y[b];
        return sqrt(dx * dx + dy * dy);
    };

    // Deterministic RNG (seeded from instance so runs are reproducible).
    uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < N; i++) {
        rng_state ^= (uint64_t)(X[i] * 1000003.0) + 0x632be59bd9b4e019ULL;
        rng_state *= 0xff51afd7ed558ccdULL;
        rng_state ^= (uint64_t)(Y[i] * 31.0) + (uint64_t)i;
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

    // ---- k-means++ style medoid seeding (D^2 weighting) -------------------
    vector<int> medoid(K);
    vector<char> isMed(N, 0);
    {
        int first = (int)(nextu() % (uint64_t)N);
        medoid[0] = first;
        isMed[first] = 1;
        vector<double> dmin(N);
        for (int i = 0; i < N; i++) dmin[i] = dist(i, first);
        for (int c = 1; c < K; c++) {
            double tot = 0.0;
            for (int i = 0; i < N; i++) tot += dmin[i] * dmin[i];
            int pick = -1;
            if (tot <= 1e-12) {
                // all remaining points coincide with medoids; pick any new one
                for (int i = 0; i < N; i++)
                    if (!isMed[i]) { pick = i; break; }
            } else {
                double target = nextd() * tot, acc = 0.0;
                for (int i = 0; i < N; i++) {
                    if (isMed[i]) continue;
                    acc += dmin[i] * dmin[i];
                    if (acc >= target) { pick = i; break; }
                }
            }
            if (pick < 0) {
                for (int i = 0; i < N; i++)
                    if (!isMed[i]) { pick = i; break; }
            }
            medoid[c] = pick;
            isMed[pick] = 1;
            for (int i = 0; i < N; i++) {
                double dd = dist(i, pick);
                if (dd < dmin[i]) dmin[i] = dd;
            }
        }
    }

    // ---- first/second-nearest cache --------------------------------------
    // near1[i], d1[i]  : index-into-medoid[] of nearest medoid and its distance
    // near2[i], d2[i]  : second-nearest medoid (index-into-medoid[]) & distance
    vector<int> near1(N), near2(N);
    vector<double> d1(N), d2(N);

    auto full_recompute = [&]() {
        for (int i = 0; i < N; i++) {
            int b1 = -1, b2 = -1;
            double v1 = 1e300, v2 = 1e300;
            for (int c = 0; c < K; c++) {
                double dd = dist(i, medoid[c]);
                if (dd < v1) {
                    v2 = v1; b2 = b1;
                    v1 = dd; b1 = c;
                } else if (dd < v2) {
                    v2 = dd; b2 = c;
                }
            }
            near1[i] = b1; d1[i] = v1;
            near2[i] = b2; d2[i] = v2;
        }
    };
    full_recompute();

    auto current_cost = [&]() -> double {
        double s = 0.0;
        for (int i = 0; i < N; i++) s += d1[i];
        return s;
    };
    double cur = current_cost();

    // Snapshot of the best feasible medoid set ever seen.
    vector<int> best = medoid;
    double bestCost = cur;

    // ---- FastPAM1 swap local search --------------------------------------
    // For ONE pass we consider every non-medoid h as a candidate to ADD.
    // Adding h: for each household i let dh = dist(i,h).
    //   * removeLoss[m] = sum over i with near1[i]==m of (cost change if medoid
    //     m disappears and h is NOT yet considered) -- handled implicitly.
    // We use the standard FastPAM1 accumulation:
    //   delta_remove[m] (independent of h) = sum_{i: near1[i]==m} (d2[i]-d1[i])
    //   For a fixed h, define for each i:
    //       if dh < d1[i]:            contributes (dh - d1[i]) to every removal
    //                                 (point reassigns to h regardless of which
    //                                  medoid we drop)  -> add to a global accum
    //       else (dh >= d1[i]):       only matters if we remove m = near1[i];
    //                                 then i goes to min(dh, d2[i]); contributes
    //                                 (min(dh,d2[i]) - d1[i]) to delta[near1[i]]
    // Total change of swapping out medoid m for h:
    //       Delta(m,h) = removeLoss_global(h) + perMedoid[m]
    //   where removeLoss_global(h) = sum_i [dh<d1[i]] (dh-d1[i])
    //   and   perMedoid[m]         = delta_remove[m]
    //                                + sum_{i: near1[i]==m, dh>=d1[i]}
    //                                       (min(dh,d2[i]) - d2[i])
    // We pick the (m,h) with the most negative Delta and apply it. O(N*nonmed)
    // per pass, evaluating ALL m for each h at once via the cache.

    vector<double> deltaRemove(K);       // sum (d2-d1) over points owned by m
    vector<double> perMed(K);            // reused per candidate h
    int noImprove = 0;

    // Build non-medoid list once; refresh lazily after swaps.
    while (true) {
        if (now_sec() - t_start > T_BUDGET) break;

        // delta_remove[m]: loss if we simply delete m (each owned point falls
        // back to its current second nearest).
        for (int c = 0; c < K; c++) deltaRemove[c] = 0.0;
        for (int i = 0; i < N; i++) deltaRemove[near1[i]] += (d2[i] - d1[i]);

        double bestDelta = -1e-7;  // require a real improvement
        int bestM = -1, bestH = -1;

        // Iterate candidate additions h over non-medoids.
        for (int h = 0; h < N; h++) {
            if (isMed[h]) continue;
            // periodic time check inside the heavy loop
            if (((h & 1023) == 0) && now_sec() - t_start > T_BUDGET) break;

            double removeGlobal = 0.0;  // sum over i with dh<d1[i] of (dh-d1[i])
            for (int c = 0; c < K; c++) perMed[c] = deltaRemove[c];

            for (int i = 0; i < N; i++) {
                double dx = X[i] - X[h], dy = Y[i] - Y[h];
                double dh = sqrt(dx * dx + dy * dy);
                if (dh < d1[i]) {
                    // i reassigns to h no matter which medoid we drop
                    removeGlobal += (dh - d1[i]);
                    // but for m = near1[i] we previously credited (d2-d1) in
                    // deltaRemove; that credit is wrong now (i goes to h, not
                    // d2). Correct perMed[near1[i]] by removing the (d2-d1)
                    // term so we don't double count.
                    perMed[near1[i]] -= (d2[i] - d1[i]);
                } else {
                    // i keeps its medoid unless we remove m = near1[i].
                    int m = near1[i];
                    double alt = min(dh, d2[i]);
                    // deltaRemove[m] already added (d2-d1); replace by (alt-d1).
                    perMed[m] += (alt - d2[i]);
                }
            }

            for (int c = 0; c < K; c++) {
                double delta = removeGlobal + perMed[c];
                if (delta < bestDelta) {
                    bestDelta = delta;
                    bestM = c;
                    bestH = h;
                }
            }
        }

        if (bestM < 0) {
            noImprove++;
            // No improving swap found in a full pass: local optimum.
            // Try a small random perturbation (restart-ish) if time remains.
            if (now_sec() - t_start > T_BUDGET * 0.85) break;
            // random kick: swap a random medoid with a random non-medoid
            int rm = (int)(nextu() % (uint64_t)K);
            int rh;
            do { rh = (int)(nextu() % (uint64_t)N); } while (isMed[rh]);
            isMed[medoid[rm]] = 0;
            medoid[rm] = rh;
            isMed[rh] = 1;
            full_recompute();
            cur = current_cost();
            if (noImprove > 30) break;
            continue;
        }

        // Apply best swap: medoid[bestM] <- bestH.
        isMed[medoid[bestM]] = 0;
        medoid[bestM] = bestH;
        isMed[bestH] = 1;
        // Recompute the cache. (O(N*K); cheaper than re-deriving incrementally
        // and keeps the cache exact, which is what makes the next pass valid.)
        full_recompute();
        cur = current_cost();
        noImprove = 0;

        if (cur < bestCost - 1e-9) {
            bestCost = cur;
            best = medoid;
        }
    }

    // Final guard: make sure 'best' is the cheapest feasible set we hold.
    {
        // recompute exact cost of 'best'
        vector<int> save = medoid;
        medoid = best;
        full_recompute();
        double bc = current_cost();
        if (bc <= bestCost + 1e-9) {
            bestCost = bc;  // best stays
        } else {
            best = save;  // shouldn't happen, but stay safe
        }
        medoid = best;
    }

    // ---- emit feasible solution ------------------------------------------
    // best holds K DISTINCT household indices (0-based). Print 1-based.
    cout << K << "\n";
    for (int c = 0; c < K; c++) cout << (best[c] + 1) << "\n";
    return 0;
}
