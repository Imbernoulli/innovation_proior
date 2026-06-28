#include <bits/stdc++.h>
using namespace std;

/*
  ale-41  Online Bin Assignment (sequential, partial information).

  Protocol (interactive, enforced by the scorer):
    read "K N"
    read K capacities
    then, repeatedly: read one item "s v", decide, print one bin id in 1..K
    (or 0 to DROP), and FLUSH, before the next item is revealed.

  Strategy -- adaptive value-density threshold (secretary / online-knapsack
  threshold family) with a per-bin fill-scaled reservation price:

    * Each item has density d = v / s. The lever is: accept only items whose
      density clears an adaptive cutoff estimated from the empirical density
      distribution seen so far, and place an accepted item into the bin whose
      *fill-scaled* threshold it clears with the tightest fit.
    * The cutoff is set so that we keep only the densest alpha-fraction of the
      remaining size-mass, where alpha = remaining-capacity / projected-future-
      demand. It is therefore the (1 - alpha) empirical density quantile, which
      tracks the non-stationary value distribution AND the instance's contention
      automatically (no magic constant baked to one regime). When capacity is
      not binding (alpha >= 1) we accept everything that fits -- dropping would
      be pure loss.
    * A bin that is nearly full demands a higher density (reservation price
      r(fill) rises with the fraction of capacity already used), so early
      capacity is spent freely and scarce late capacity is reserved for the
      best items -- this is what an adaptive threshold under partial info buys.
    * A short warmup accepts anything that fits to seed the empirical density
      distribution; an end-game relaxation fades the bar to 0 in the tail so we
      never leave capacity idle that a plain first-fit would have used.

  Feasibility is structural: we ONLY ever return a bin whose remaining capacity
  is >= the item size, and 0 (drop) otherwise. The scorer can never see an
  overflow, so the score is never floored to 0.
*/

int main() {
    ios::sync_with_stdio(false);
    // NOTE: do NOT untie cin/cout -- we must flush after every decision.

    int K;
    long long N;
    if (!(cin >> K >> N)) return 0;
    vector<long long> cap(K), rem(K);
    for (int i = 0; i < K; i++) { cin >> cap[i]; rem[i] = cap[i]; }

    // Empirical density samples. For each quantile query we copy and run
    // nth_element; exact over all densities seen so far, cheap at these N.
    vector<double> dens;            // all densities seen so far
    dens.reserve((size_t)N);

    // Observation/warmup length: a tiny prefix where the density distribution
    // is too thin to trust a quantile, so we accept anything that fits to seed
    // the empirical distribution (and not waste these early arrivals).
    long long warmup = max<long long>(20, N / 50);  // ~2% of the stream

    // Running total of item sizes seen, to project total demand and thus the
    // affordable acceptance rate.
    long long seenSize = 0;

    // Quantile over all densities seen so far (nearest-rank on a scratch copy).
    // O(M) per call via nth_element on a copy. M is the count seen; with N a
    // few thousand this is comfortably within the time budget.
    auto quantile = [&](double q) -> double {
        if (dens.empty()) return 0.0;
        if (q <= 0.0) return *min_element(dens.begin(), dens.end());
        if (q >= 1.0) return *max_element(dens.begin(), dens.end());
        vector<double> tmp = dens;
        size_t k = (size_t)floor(q * (tmp.size() - 1) + 0.5);
        if (k >= tmp.size()) k = tmp.size() - 1;
        nth_element(tmp.begin(), tmp.begin() + k, tmp.end());
        return tmp[k];
    };

    for (long long it = 0; it < N; it++) {
        long long s, v;
        if (!(cin >> s >> v)) break;            // stream ended early
        double d = (s > 0) ? (double)v / (double)s : (double)v;
        dens.push_back(d);
        seenSize += s;

        int choice = 0;  // default: drop

        // Remaining free capacity across all bins.
        long long R = 0;
        for (int b = 0; b < K; b++) R += rem[b];
        double frac = (double)it / (double)N;         // progress in [0,1)
        double remFrac = 1.0 - frac;                  // fraction of stream still to come

        if (it < warmup || R <= 0) {
            // Warmup: not enough samples for a stable quantile. Accept anything
            // that fits (tightest fit) to both seed the distribution and bank
            // early value. If R<=0 nothing fits anyway.
            int best = -1; double bestSlack = 0;
            for (int b = 0; b < K; b++) {
                if (rem[b] < s) continue;
                double slack = (double)(rem[b] - s);
                if (best < 0 || slack < bestSlack) { best = b; bestSlack = slack; }
            }
            if (best >= 0) choice = best + 1;
        } else {
            // -- Affordable acceptance rate. Project the remaining demand
            //    (size-mass still to arrive) from the rate seen so far, and
            //    compare to remaining free capacity R. alpha = the fraction of
            //    the *future* size-mass we can afford to keep. If we can afford
            //    everything (alpha>=1) capacity is not binding -> accept all
            //    that fits. Otherwise we keep only the densest alpha-fraction,
            //    so the acceptance THRESHOLD is the (1-alpha) density quantile.
            double sizeRate = (double)seenSize / (double)(it + 1);  // avg size/item
            double projRemDemand = sizeRate * remFrac * (double)N;  // future size-mass
            double alpha = (projRemDemand > 1e-9)
                ? (double)R / projRemDemand : 2.0;     // capacity vs future demand

            if (alpha >= 1.0) {
                // Capacity not binding from here on: dropping is pure loss.
                // Accept anything that fits (tightest fit).
                int best = -1; double bestSlack = 0;
                for (int b = 0; b < K; b++) {
                    if (rem[b] < s) continue;
                    double slack = (double)(rem[b] - s);
                    if (best < 0 || slack < bestSlack) { best = b; bestSlack = slack; }
                }
                if (best >= 0) choice = best + 1;
            } else {
                // Selective: keep only the densest alpha-fraction of size-mass.
                // Base threshold = (1-alpha) empirical density quantile. This
                // self-tunes to the non-stationary value distribution and to
                // how contended this particular instance is. Clamp the quantile
                // to a sane band so a few outliers cannot push it pathological.
                double qlevel = 1.0 - alpha;
                qlevel = min(0.97, max(0.05, qlevel));

                // End-game relaxation: as the stream nears its end there is
                // little future to reserve for, so we lower the bar toward 0 in
                // the last stretch. This greedily flushes any capacity that the
                // threshold would otherwise leave idle -- guaranteeing we do not
                // end up under-filled relative to a plain first-fit. The bar
                // fades to 0 over the final 15% of the stream.
                double tail = 0.15;
                if (remFrac < tail) qlevel *= (remFrac / tail);

                double baseCut = quantile(qlevel);

                // Per-bin reservation: a fuller bin demands a higher density,
                // so the last slots in a bin are spent on the very best items.
                // Pick, among fitting bins whose (fill-scaled) bar this item
                // clears, the one it clears by the most; tie -> tightest fit.
                int best = -1;
                double bestClear = -1e18, bestSlackKey = 0.0;
                for (int b = 0; b < K; b++) {
                    if (rem[b] < s) continue;            // hard feasibility
                    double fill = 1.0 - (double)rem[b] / (double)cap[b]; // used frac
                    double r = 1.0 + 0.6 * fill * fill;  // reservation multiplier
                    double thr = baseCut * r;
                    double clear = d - thr;
                    if (clear < 0) continue;             // misses this bin's bar
                    double slack = (double)(rem[b] - s);
                    if (clear > bestClear + 1e-12 ||
                        (fabs(clear - bestClear) <= 1e-12 && slack < bestSlackKey)) {
                        bestClear = clear; bestSlackKey = slack; best = b;
                    }
                }
                if (best >= 0) choice = best + 1;
                // else: drop, reserving scarce capacity for denser future items.
            }
        }

        if (choice >= 1) rem[choice - 1] -= s;  // commit (already feasible)
        cout << choice << "\n";
        cout.flush();                            // MUST flush: interactive
    }
    return 0;
}
