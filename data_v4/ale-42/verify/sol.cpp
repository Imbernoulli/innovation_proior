// Interactive Adaptive Probing (find hidden hotspots) -- ale-42.
//
// A hidden H x W reward field r(x,y) >= 0 sits in stdin (rows of integers); a
// cell is HOT iff r >= thr. We are a probing agent with a budget of Q probes. A
// probe at (x,y) OBSERVES the (2*rad+1)^2 Chebyshev window around it and returns
// a NOISY reading of each observed cell. We must end with a REPORT: a set of
// cells we declare hot. Scoring (see score.py): sum of TRUE r over reported
// cells minus `penalty` per reported cold cell; a reported cell must be observed
// (within rad of a declared probe); >Q probes / unobserved report / malformed
// -> score 0. Normalized vs a uniform-grid probing baseline.
//
// CONTRACT DISCIPLINE. We treat the field as hidden: we only ever consult a
// cell's reward through a probe, and we never use more than Q probes. Each probe
// returns a noisy reading (true value + deterministic measurement noise of std
// sigma*SCALE); we never see the true field directly, and we decide where to
// probe and what to report from those noisy readings alone.
//
// INNOVATION -- a two-phase adaptive probe schedule over TWO Gaussian-smoothed
// BELIEF grids. The field is smooth, so a probe near a hotspot reads elevated
// values even at a distance; we exploit that to *locate* bumps from sparse
// readings, then zoom in. Each noisy reading is folded (precision-weighted) into
// (a) a WIDE-kernel exploration belief `mu`, whose only job is to localize bumps
// -- a reading is evidence about the whole smooth neighbourhood, and (b) a
// NARROW-kernel report belief `rmu`, which keeps a hot core's magnitude intact
// for the final hot/cold call. Phase 1 lays a coarse uniform lattice (about 2/3
// of the budget) so no region is a blind spot, seeding the wide belief. Phase 2
// greedily spends the remaining probes by placing each where its window collects
// the most elevated `mu` belief (chasing the belief PEAK, discounting cells
// already read), which discovers hot mass the sweep only hinted at and gives the
// hottest cells extra overlapping reads to denoise them. Finally we report a cell
// when its EXPECTED credit under the narrow belief is positive (decision-
// theoretic, penalty-aware).

#include <bits/stdc++.h>
using namespace std;

static const int SCALE = 1000;

int H, W, Q, rad, thr, penalty;
double sigma;
vector<vector<int>> grid;   // TRUE field, only consulted through probe()

// deterministic measurement noise: a function of (x,y) only, so re-probing the
// same cell is consistent (a real noisy sensor reading we cannot un-see).
static inline uint64_t mix(uint64_t a) {
    a ^= a >> 33; a *= 0xff51afd7ed558ccdULL;
    a ^= a >> 33; a *= 0xc4ceb9fe1a85ec53ULL;
    a ^= a >> 33; return a;
}
double noisy_read(int x, int y) {
    uint64_t h = mix((uint64_t)(x) * 1000003u + (uint64_t)(y) + 0x9E3779B97F4A7C15ULL);
    double u1 = ((h & 0xFFFFFFFFu) + 1.0) / 4294967297.0;
    double u2 = (((h >> 32) & 0xFFFFFFFFu) + 1.0) / 4294967297.0;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * acos(-1.0) * u2);
    double val = grid[x][y] + z * sigma * SCALE;
    return val;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> H >> W >> Q >> rad >> sigma >> thr >> penalty)) return 0;
    grid.assign(H, vector<int>(W, 0));
    for (int x = 0; x < H; x++)
        for (int y = 0; y < W; y++)
            cin >> grid[x][y];

    const int N = H * W;
    auto id = [&](int x, int y) { return x * W + y; };

    // ---- belief grids: posterior mean mu and accumulated precision `info`.
    // We fold each noisy reading into nearby cells with a Gaussian kernel: the
    // field is smooth, so a reading at (x,y) is evidence about its neighbourhood.
    // A WIDE kernel lets a single probe hint at a bump several cells away, which
    // is what makes sparse exploration able to localize hotspots at all.
    vector<double> mu(N, 0.0);
    vector<double> info(N, 0.0);
    vector<char> probed(N, 0);
    vector<char> observed(N, 0);

    // Number of DIRECT reads of each cell (it fell inside a probe window). Used
    // to discount re-reads when choosing zoom probes: a cell already read several
    // times yields less new information than a fresh one.
    vector<int> dcnt(N, 0);

    // Two smoothing scales serve two jobs. A WIDE exploration kernel (`mu`)
    // lets a probe several cells from a bump still register elevated belief there
    // -- that is what makes sparse probes *localize* hotspots. But a wide kernel
    // also pulls a hot core's estimate down toward its cold surroundings, so we
    // do NOT trust `mu`'s magnitude for the report; instead we denoise reportable
    // cells with extra direct reads (phase 2) and a NARROW report kernel (`rmu`)
    // that barely smooths.
    int srad = 4;                            // wide evidence radius (Chebyshev)
    double ks = 2.2;                         // wide kernel std (cells)
    int K = 2 * srad + 1;
    vector<vector<double>> ker(K, vector<double>(K, 0.0));
    for (int dx = -srad; dx <= srad; dx++)
        for (int dy = -srad; dy <= srad; dy++)
            ker[dx + srad][dy + srad] =
                exp(-(double)(dx * dx + dy * dy) / (2.0 * ks * ks));

    double prior = 0.10 * SCALE;             // weak prior; uncertainty drives explore
    double prior_prec = 1.0 / (0.40 * SCALE * 0.40 * SCALE);
    for (int i = 0; i < N; i++) { mu[i] = prior; info[i] = prior_prec; }

    // NARROW report-belief grid: same Bayesian fold but with a tight kernel, so
    // a hot core keeps (most of) its magnitude while still denoising via the few
    // overlapping reads it gets. Used ONLY for the final report decision.
    int rsrad = 1;
    double rks = 0.8;
    int RK = 2 * rsrad + 1;
    vector<vector<double>> rker(RK, vector<double>(RK, 0.0));
    for (int dx = -rsrad; dx <= rsrad; dx++)
        for (int dy = -rsrad; dy <= rsrad; dy++)
            rker[dx + rsrad][dy + rsrad] =
                exp(-(double)(dx * dx + dy * dy) / (2.0 * rks * rks));
    vector<double> rmu(N, prior);
    vector<double> rinfo(N, prior_prec);

    double noisevar = sigma * sigma * SCALE * SCALE + 1.0;

    // fold a noisy reading at (x,y) into the belief of its kernel neighbourhood.
    auto fold = [&](int x, int y, double reading) {
        // wide exploration belief
        for (int dx = -srad; dx <= srad; dx++) {
            int xx = x + dx; if (xx < 0 || xx >= H) continue;
            for (int dy = -srad; dy <= srad; dy++) {
                int yy = y + dy; if (yy < 0 || yy >= W) continue;
                double w = ker[dx + srad][dy + srad];
                if (w < 1e-6) continue;
                double prec = w / noisevar;
                int j = id(xx, yy);
                double ni = info[j] + prec;
                mu[j] = (mu[j] * info[j] + reading * prec) / ni;
                info[j] = ni;
            }
        }
        // narrow report belief
        for (int dx = -rsrad; dx <= rsrad; dx++) {
            int xx = x + dx; if (xx < 0 || xx >= H) continue;
            for (int dy = -rsrad; dy <= rsrad; dy++) {
                int yy = y + dy; if (yy < 0 || yy >= W) continue;
                double w = rker[dx + rsrad][dy + rsrad];
                if (w < 1e-6) continue;
                double prec = w / noisevar;
                int j = id(xx, yy);
                double ni = rinfo[j] + prec;
                rmu[j] = (rmu[j] * rinfo[j] + reading * prec) / ni;
                rinfo[j] = ni;
            }
        }
    };

    vector<pair<int,int>> probes;
    probes.reserve(Q);

    auto do_probe = [&](int x, int y) {
        probes.push_back({x, y});
        probed[id(x, y)] = 1;
        for (int dx = -rad; dx <= rad; dx++) {
            int xx = x + dx; if (xx < 0 || xx >= H) continue;
            for (int dy = -rad; dy <= rad; dy++) {
                int yy = y + dy; if (yy < 0 || yy >= W) continue;
                int j = id(xx, yy);
                observed[j] = 1;
                double rd = noisy_read(xx, yy);
                dcnt[j] += 1;                        // one more direct read
                fold(xx, yy, rd);                    // update both belief grids
            }
        }
    };

    // ---- Phase 1: uniform coverage sweep (about two thirds of the budget) -----
    // We FIRST guarantee broad coverage with no blind region: a regular lattice
    // of probes, aspect-matched to the grid so the windows tile the plane evenly.
    // The wide-kernel fold turns these scattered noisy readings into a smoothed
    // belief that localizes the bumps for the adaptive phase. We commit ~2/3 of
    // the budget here; placing too few sweep probes risks missing a hotspot
    // entirely (the dominant failure mode), so we err toward coverage. Any
    // leftover sweep budget fills the largest uncovered gaps. We keep ~1/3 of the
    // budget for adaptive zoom.
    int sweepN = max(1, (2 * Q + 2) / 3);
    if (sweepN > Q) sweepN = Q;
    int rows = max(1, (int)llround(sqrt((double)sweepN * H / max(1, W))));
    rows = min(rows, sweepN);
    int cols = max(1, sweepN / rows);
    {
        int placed = 0;
        for (int i = 0; i < rows && placed < sweepN; i++) {
            int px = (int)((i + 0.5) * H / rows);
            px = min(H - 1, max(0, px));
            for (int j = 0; j < cols && placed < sweepN; j++) {
                int py = (int)((j + 0.5) * W / cols);
                py = min(W - 1, max(0, py));
                if (probed[id(px, py)]) continue;
                do_probe(px, py);
                placed++;
            }
        }
        // leftover sweep budget -> fill the largest uncovered gaps (cells with no
        // direct reading and the highest belief uncertainty).
        while (placed < sweepN) {
            int best = -1; double bv = -1e18;
            for (int i = 0; i < N; i++) {
                if (probed[i] || dcnt[i] > 0) continue;
                double unc = 1.0 / sqrt(info[i]);
                if (unc > bv) { bv = unc; best = i; }
            }
            if (best < 0) break;
            do_probe(best / W, best % W);
            placed++;
        }
    }

    // ---- Phase 2: greedy adaptive zoom onto the belief peaks ------------------
    // Each remaining probe is placed where its (2 rad+1)^2 window is expected to
    // sit on the MOST promising belief. We score a candidate probe center by the
    // window sum of (mu - localBackground) above 0 -- i.e. how elevated the wide
    // belief is there relative to the field's floor -- weighted DOWN for cells we
    // have already read directly (a re-read still denoises, but a fresh read buys
    // more). Critically we chase the belief PEAK, not "mu over thr": the wide
    // kernel pulls a hot core's mu below thr, so an over-thr test would refuse to
    // zoom in on real but under-estimated hotspots (the dominant miss). Zooming
    // gives the hottest cells several overlapping direct reads, which the NARROW
    // report belief then turns into a confident hot call. Never re-probe a cell.
    double floorv = prior;     // belief background level
    while ((int)probes.size() < Q) {
        int best = -1; double bestv = -1e18;
        for (int cx = 0; cx < H; cx++) {
            for (int cy = 0; cy < W; cy++) {
                if (probed[id(cx, cy)]) continue;
                double val = 0.0;
                for (int dx = -rad; dx <= rad; dx++) {
                    int xx = cx + dx; if (xx < 0 || xx >= H) continue;
                    for (int dy = -rad; dy <= rad; dy++) {
                        int yy = cy + dy; if (yy < 0 || yy >= W) continue;
                        int j = id(xx, yy);
                        double elev = mu[j] - floorv;
                        if (elev <= 0) continue;
                        double w = 1.0 / (1.0 + 0.7 * dcnt[j]);
                        val += elev * w;
                    }
                }
                if (val > bestv) { bestv = val; best = id(cx, cy); }
            }
        }
        if (best < 0) break;
        do_probe(best / W, best % W);
    }

    // ---- report --------------------------------------------------------------
    // Decision-theoretic rule. For an observed cell, model its true reward as
    // r ~ Normal(rmu, sd) with sd = 1/sqrt(rinfo) (the narrow-belief estimate and
    // its uncertainty). The scorer credits a reported cell with r if it is hot
    // (r >= thr) and r - penalty if cold, so the EXPECTED credit of reporting is
    //     E[r] - penalty * P(r < thr) = rmu - penalty * Phi((thr - rmu) / sd).
    // We report a cell exactly when that expectation is positive (plus a small
    // safety bias `eps` scaled by penalty, to stay on the profitable side of the
    // boundary given that our noise model is only approximate). This collects far
    // more genuine hot mass than a hard "estimate clears thr" test while keeping
    // false positives in check via the penalty term.
    auto Phi = [](double z) {                        // standard normal CDF
        return 0.5 * erfc(-z * M_SQRT1_2);
    };
    // Safety bias: require the expected credit to clear a margin proportional to
    // the penalty. A false positive in a region with NO real hot mass can sink a
    // whole instance to 0, so we stay firmly on the profitable side -- better to
    // skip a borderline cell than to report a phantom. We also gate on a minimum
    // amount of evidence (rinfo well above the prior precision): a cell seen only
    // through the smoothing tail of one far probe is too uncertain to claim.
    double eps = 0.20 * (double)penalty;
    double min_info = 1.6 * prior_prec;
    vector<pair<int,int>> report;
    for (int x = 0; x < H; x++) {
        for (int y = 0; y < W; y++) {
            int i = id(x, y);
            if (!observed[i]) continue;            // can only report observed cells
            if (rinfo[i] < min_info) continue;     // too little evidence to claim
            double sd = 1.0 / sqrt(rinfo[i]);
            if (sd < 1.0) sd = 1.0;
            double pcold = Phi(((double)thr - rmu[i]) / sd);
            double exp_credit = rmu[i] - (double)penalty * pcold;
            if (exp_credit > eps) report.push_back({x, y});
        }
    }

    // ---- emit: P, P probe cells, M, M report cells ---------------------------
    string out;
    out.reserve(16 * (probes.size() + report.size()) + 32);
    out += to_string((int)probes.size()); out += '\n';
    for (auto &p : probes) {
        out += to_string(p.first); out += ' ';
        out += to_string(p.second); out += '\n';
    }
    out += to_string((int)report.size()); out += '\n';
    for (auto &p : report) {
        out += to_string(p.first); out += ' ';
        out += to_string(p.second); out += '\n';
    }
    cout << out;
    return 0;
}
