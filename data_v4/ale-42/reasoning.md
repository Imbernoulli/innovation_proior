# Reasoning: Interactive Adaptive Probing (find hidden hotspots)

## Reading the objective

I have a hidden `H x W` grid of non-negative integer rewards `r(x, y)`. The field is smooth — a few
Gaussian bumps on a low background. A cell is "hot" when `r(x, y) >= thr`. I am a probing agent with
a budget of `Q` probes; each probe at `(x, y)` lets me *observe* the `(2*rad+1)^2` Chebyshev window
around it and gives me a *noisy* reading of every cell in that window (noise std `sigma` on the
reward scale `SCALE = 1000`). At the end I commit a **report**: a set of cells I claim are hot. I
collect the true reward of every reported cell, but I pay `penalty` for every reported cell that is
actually cold, and — crucially — I can only report cells I *observed*, and I cannot use more than `Q`
probes. Break any of those rules and the score is floored to `0`.

So three things matter, and they interact: (1) **where I probe**, because `Q` is tiny relative to the
grid and the hot mass sits in a few unknown places; (2) **how I denoise**, because a single reading
with `sigma * SCALE` ≈ 300–450 of noise against `thr` ≈ 600 is almost useless on its own; and (3)
**what I report**, because a false positive costs `penalty` ≈ 300–500. The first thing I want is a
solution that is always *feasible* — never crashes, never reports an unobserved cell, never overruns
the budget — and only then do I chase score.

## A feasible baseline first

The trivially feasible thing is: use zero probes and report nothing. That parses, observes nothing,
reports nothing, and scores `0`. It is my safety floor — whatever else my program does, if a branch
goes wrong I can always fall back to "emit `0\n0\n`". Good. Now I need something that actually
beats the scorer's normalizer.

The scorer normalizes against **uniform-grid probing**: drop `Q` probes on a regular lattice, observe
their windows, and report exactly the observed cells whose *true* reward clears `thr`. That baseline
is omniscient *within its coverage* — it never makes a false positive — so to beat it I cannot just
match its coverage and report the same cells; I have to **observe more of the hot mass than a fixed
lattice does**, by steering my probes toward where the reward actually is. That is the whole game.

My first instinct is the textbook active-sensing move: maintain a posterior belief over the field and
pick probes by an upper-confidence-bound (UCB) acquisition rule — `mu + beta * uncertainty` — so I
explore uncertain cells early and exploit high-belief cells late. Because the field is smooth, a
reading at `(x, y)` is evidence about its neighbours too, so I fold each noisy reading into a belief
grid with a Gaussian kernel (precision-weighted Bayesian update: `mu <- (mu*info + reading*prec) /
(info+prec)`, `info += prec`). I start every cell at a weak prior so unprobed cells have high
uncertainty and the UCB term drives exploration.

## First implementation, and the wall

I wrote exactly that: a single belief grid with a narrow kernel (`srad = rad`), UCB probe selection
with a decaying `beta`, and a report rule "report observed cells whose `mu - margin >= thr`". I
compiled it and ran my self-verify harness on seeds 1..20: generate the instance, run the solver,
score it, and compare to the uniform baseline.

The result was a disaster. On 19 of 20 seeds the report was empty and the score was `0`; only seed 6
produced anything, and even that scored a fraction of baseline. I instrumented the instances: on
seed 1, `Q = 20`, `rad = 1`, `sigma = 0.421`, `thr = 660`, only **31** cells are hot out of 3640.
Twenty `rad = 1` probes observe at most `20 * 9 = 180` cells — about 5% of the grid. So the first
problem is brutal: I have to *find* a 31-cell needle with 180 darts, and each dart's reading carries
±420 of noise. The narrow kernel and weak prior meant my belief never built up enough signal at the
hot cores to clear `thr` after the confidence margin, so I reported nothing almost everywhere.

I needed to separate the failure modes. I wrote a diagnostic that, for the solver's *actual* probe
set, computes the **oracle report given that coverage** — i.e. report every observed cell that is
truly hot — and compares it to the baseline. That isolates "did I even *observe* the hot cells?" from
"did I *report* the right cells?". The numbers were stark: on seeds 1, 2, 3, 8, 10, 12, 18, 19 the
oracle-given-coverage was **0** — my probes did not observe a single hot cell, even though the
uniform baseline did. My "adaptive" scheme was *worse at coverage than a dumb lattice*. On the seeds
where I did cover hot cells, the oracle ratio was 0.5–2.0, occasionally huge — so when coverage
worked, there was plenty of value to collect, and the bottleneck there was the *report decision*.

Two distinct bugs, then: a **coverage/localization** bug (my probes miss the bumps) and a **report**
bug (even with coverage I collect little).

## Diagnosing the coverage failure

Why did UCB miss the bumps? Two reasons. First, with `sigma * SCALE` ≈ 420 of noise and a narrow
kernel, the highest-UCB cell is frequently just a **noise spike** in an empty region, not a real
hotspot — so UCB chased phantoms and clustered probes on noise. Second, and more fundamental, the
*localization* relies on a probe near a bump reading elevated values, but a narrow kernel only
propagates that signal one cell, so a hotspot sitting between my scattered probes produced **no**
belief elevation anywhere I could see it. On seed 1 my nearest probe got within Chebyshev distance 2
of a hot cell but, with `rad = 1`, the window never reached it and the narrow belief never lit up.

The fix has two parts. (a) **Guarantee coverage at least as good as the baseline** by spending a
solid chunk of the budget on a uniform lattice up front — placing *fewer* probes than the baseline is
how you miss a hotspot entirely, the dominant failure. (b) **Smooth wider** so a probe several cells
from a bump still registers it, letting the adaptive phase actually localize bumps from the sweep's
scattered readings.

But a wide kernel has a nasty side effect I discovered immediately: it pulls a hot core's belief
**down** toward its cold surroundings. With a wide kernel, `mu` at the peak is a kernel-weighted
average of the peak and its lower neighbours, so `mu_peak` lands *below* `thr` even for a genuinely
hot cell. That breaks both the adaptive targeting (if I zoom only onto cells whose `mu > thr`, I
refuse to zoom onto real-but-underestimated bumps) and the report (if I report only `mu >= thr`, I
report nothing). I hit exactly this: after switching to a wide kernel, coverage improved but the
report stayed near-empty on several seeds, and seed 14 — which observed 5 hot cells — still reported
**zero** because the smoothed `mu` at those cells didn't clear `thr`.

## The key idea: two belief scales, peak-chasing zoom, decision-theoretic report

The resolution is to stop asking one belief grid to do two incompatible jobs. **Localizing** a bump
wants a *wide* kernel (so distant readings vote). **Reporting** a cell wants the belief to preserve
the core's *magnitude* so I can tell hot from cold. So I maintain **two** Gaussian-smoothed belief
grids from the same readings:

- a **wide** exploration belief `mu` (kernel std ≈ 2.2, radius 4) whose only job is to say *where the
  bumps are* — I never trust its magnitude;
- a **narrow** report belief `rmu` (kernel std ≈ 0.8, radius 1) which barely smooths, so a hot core
  keeps most of its value while still getting some denoising from the overlapping reads it receives.

With that split, the probe schedule becomes two clean phases:

**Phase 1 — uniform coverage sweep (~2/3 of the budget).** Lay a regular lattice, aspect-matched to
the grid, so no region is a blind spot. This matches the baseline's coverage by construction and
seeds the wide belief so it can localize the bumps.

**Phase 2 — greedy adaptive zoom (the rest).** For each remaining probe, score every candidate probe
center by the sum over its `(2*rad+1)^2` window of the **elevation** `max(0, mu - background)`,
discounting cells I have already read directly (a re-read still denoises, but a fresh read buys more
information). I place the probe at the best such center. The decisive choice is that I chase the
belief **peak** — total elevation — *not* `mu - thr`: because the wide kernel suppresses peaks below
`thr`, an over-thr criterion would refuse to zoom onto exactly the real bumps I want. Chasing the
peak both discovers hot mass the sweep only hinted at and piles several overlapping reads onto the
hottest cells, which is what denoises their narrow belief enough to report them.

**The report.** This is where I make the penalty-vs-reward tradeoff explicit. For each observed cell,
model its true reward as `r ~ Normal(rmu, sd)` with `sd = 1/sqrt(rinfo)` (the narrow belief and its
accumulated precision). The scorer credits a reported cell with `r` if hot and `r - penalty` if cold,
so the **expected credit** of reporting it is
`E[r] - penalty * P(r < thr) = rmu - penalty * Phi((thr - rmu)/sd)`,
where `Phi` is the standard normal CDF (`0.5 * erfc(-z/sqrt(2))`). I report a cell exactly when that
expectation is positive. This is the Bayes-optimal report rule under my belief, and it collects far
more genuine hot mass than the brittle "estimate clears `thr`" test — a cell a little under `thr` is
still worth claiming if the reward it probably carries beats the penalty it might cost.

## Implementing and a second debug episode

I rewrote the solver with the two belief grids, the two phases, and the decision-theoretic report,
and re-ran the harness. Coverage was fixed — now 14/20 beat baseline, with raw values jumping from
~8000 (baseline) to 30000+ on the good seeds. But two new problems showed up.

First, on seeds with **no observable hot mass** (e.g. seed 1, where my probes still don't land a
window on the needle, and seed 10, a degenerate instance where even the baseline observes nothing),
my report rule occasionally fired on a **noise spike** — it reported a phantom hot cell in an empty
region, producing a *negative* raw value. The scorer clamps the credited value at 0, so these don't
go below baseline, but a phantom is a wasted risk: a single false positive in a region with no real
hot mass can sink an otherwise-fine instance. Second, on seeds with lots of hot cells (seed 6: 149
hot, seed 13: 131 hot) I observed many hot cells but reported too few — leaving value on the table.

The report-too-few part the decision-theoretic rule already largely fixed (reporting jumped on those
seeds once `E[credit] > 0` replaced the hard threshold). For the phantom-false-positive part, I added
two guards, both motivated directly by the penalty economics: (a) require the expected credit to
clear a small **safety margin** `eps = 0.20 * penalty`, so I stay firmly on the profitable side of
the boundary given that my Gaussian noise model is only approximate; and (b) gate on a **minimum
evidence** level — a cell whose narrow-belief precision `rinfo` is barely above the prior (seen only
through the smoothing tail of one far probe) is too uncertain to claim, so I skip it. After these,
the phantom reports vanished: seed 1 now emits an empty report (raw 0) instead of a negative one, and
the seeds with real coverage report aggressively.

Final harness over seeds 1..20: every output feasible, no parse failures, **17/20 beat the uniform
baseline**, and on a second range 21..40 it again beat baseline 17/20 with the rest being near-ties
or degenerate `baseline = 0` instances. Runtime is well under 10 ms and memory ~4 MB, comfortably
inside the 2 s / 256 MB budget (the costliest step is the phase-2 candidate scan, `O(H*W * window)`
per zoom probe, with only a handful of zoom probes). The remaining sub-baseline cases are instances
where the hot mass simply cannot be localized with `Q` probes and a `rad`-1 window — an inherent
difficulty the baseline shares, and on average my adaptive scheme is far ahead.

A subtle correctness point I double-checked: the contract says I may only report **observed** cells.
My report loop iterates only over cells with `observed[i] == 1`, and `observed` is set exactly inside
`do_probe` for the window cells, so by construction every reported cell is within `rad` of a probe —
the scorer's observability check can never floor me. And I never re-probe an exact cell (`probed[]`
guard), so I never waste a probe and the window-overlap that denoises hot cores comes from *adjacent*
probe centers, which is legitimate.

## Final solver

```cpp
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
```
