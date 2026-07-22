// TIER: strong
// Same run-order construction as the greedy tier (EDF latest-free-slot for stat
// deadlines, ascending fill otherwise) -- the insight is entirely in how the two
// reset budgets are placed GIVEN that order, and it is genuinely joint, not just
// "greedy plus a parameter tweak":
//   1. TIP SELECTION is redone as the TRUE marginal saving of fixing position j,
//      given a fixed recalibration placement: saving(j) = GAP_COEF*gap(j) +
//      DRIFT_COEF*BOOST*e(j)^2, where e(j) = min(cnt(j), CAP) is the usage count
//      AFTER the recalibration pass (e(j)=1 right after a recalibration, larger
//      elsewhere). Picking the top-K1 positions by this TRUE saving is provably
//      at least as good as greedy's "top-K1 by raw gap alone" for any FIXED
//      recalibration placement, because it directly maximizes the quantity the
//      objective actually charges instead of a proxy for it.
//   2. RECALIBRATION PLACEMENT is where the joint reasoning lives. Near-equal
//      spacing is optimal for the base (non-boosted) quadratic drift term alone,
//      but letting a boundary drift toward a large local gap can additionally
//      kill the (1+BOOST) amplification exactly where an uncorrected carryover
//      break already sits -- a real trade-off between "even segments" and
//      "align with the break". Rather than guess which side of that trade-off
//      wins on a given instance, this solution builds BOTH candidate placements
//      (even spacing, and a bounded-window snap to the locally largest gap),
//      pairs each with its own true-marginal-saving tip selection, evaluates the
//      REAL objective for both, and keeps the winner. Since candidate A is
//      exactly greedy's recalibration placement plus a provably-better tip rule,
//      this construction can never do worse than greedy on the resource-
//      placement side -- the joint search on top can only add further savings by
//      co-locating resets where carryover already broke the sequence.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static const ll CAP = 60;
static const double WINFRAC = 0.2; // recal search window, as a fraction of the ideal segment length

int n, K1, K2;
ll GAP_COEF, DRIFT_COEF, BOOST;
vector<ll> conc, gap;
vector<int> order;

// Candidate A: greedy-style near-equal spacing (identical construction to
// solutions/greedy.cpp's recalibration placement).
vector<int> recalEven() {
    vector<int> recal(n + 1, 0);
    if (K2 > 0) {
        int step = max(1, n / K2);
        int used = 0;
        set<int> placed;
        for (int k = 1; k <= K2; k++) {
            int posn = min(n, k * step);
            if (placed.count(posn)) continue;
            placed.insert(posn);
            recal[posn] = 1;
            used++;
            if (used >= K2) break;
        }
    }
    return recal;
}

// Candidate B: near-equal spacing, each boundary refined within a bounded
// window toward the locally largest gap.
vector<int> recalSnap() {
    vector<int> recal(n + 1, 0);
    if (K2 > 0) {
        int seg = max(1, n / K2);
        int win = max(1, (int)(seg * WINFRAC));
        for (int k = 1; k <= K2; k++) {
            int ideal = min(n, k * seg);
            int lo = max(1, ideal - win);
            int hi = min(n, ideal + win);
            int best = ideal;
            for (int j = lo; j <= hi; j++) {
                if (!recal[j] && gap[j] > gap[best]) best = j;
            }
            while (recal[best] && best < n) best++;
            recal[best] = 1;
        }
    }
    return recal;
}

// Given a fixed recalibration placement, pick the top-K1 tip positions by the
// TRUE marginal saving (see header comment).
vector<int> bestTip(const vector<int>& recal) {
    vector<ll> cnt(n + 1, 0);
    for (int j = 1; j <= n; j++) cnt[j] = recal[j] ? 1 : cnt[j - 1] + 1;
    vector<ll> saving(n + 1, 0);
    for (int j = 1; j <= n; j++) {
        if (gap[j] <= 0) continue;
        ll e = min(cnt[j], CAP);
        saving[j] = GAP_COEF * gap[j] + DRIFT_COEF * BOOST * e * e;
    }
    vector<int> bySaving(n + 1);
    for (int j = 1; j <= n; j++) bySaving[j] = j;
    sort(bySaving.begin() + 1, bySaving.end(), [&](int a, int b) {
        if (saving[a] != saving[b]) return saving[a] > saving[b];
        return a < b;
    });
    vector<int> tip(n + 1, 0);
    for (int k = 0; k < K1 && k < n && saving[bySaving[k + 1]] > 0; k++) tip[bySaving[k + 1]] = 1;
    return tip;
}

// Exact objective, mirroring chk.cc.
ll evalF(const vector<int>& tip, const vector<int>& recal) {
    ll F = 0, prevConc = 0, cnt = 0;
    for (int j = 1; j <= n; j++) {
        ll c = conc[order[j]];
        ll g = (j == 1) ? 0 : max((ll)0, prevConc - c);
        ll carry = tip[j] ? 0 : GAP_COEF * g;
        cnt = recal[j] ? 1 : cnt + 1;
        ll e = min(cnt, CAP);
        bool boosted = (j > 1) && (g > 0) && !tip[j];
        ll drift = DRIFT_COEF * e * e * (boosted ? (1 + BOOST) : 1);
        F += carry + drift;
        prevConc = c;
    }
    return F;
}

int main() {
    scanf("%d %d %d %lld %lld %lld", &n, &K1, &K2, &GAP_COEF, &DRIFT_COEF, &BOOST);
    conc.assign(n + 1, 0);
    vector<int> isStat(n + 1, 0), deadline(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        scanf("%lld %d %d", &conc[i], &isStat[i], &deadline[i]);
    }

    // ---- build run order (identical to greedy) -------------------------------
    vector<int> statIdx;
    for (int i = 1; i <= n; i++) if (isStat[i]) statIdx.push_back(i);
    sort(statIdx.begin(), statIdx.end(), [&](int a, int b) { return deadline[a] < deadline[b]; });

    vector<int> parent(n + 2);
    for (int x = 0; x <= n + 1; x++) parent[x] = x;
    function<int(int)> find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };

    order.assign(n + 1, 0);
    for (int s : statIdx) {
        int slot = find(min(deadline[s], n));
        order[slot] = s;
        parent[slot] = slot - 1;
    }
    vector<int> nonStat;
    for (int i = 1; i <= n; i++) if (!isStat[i]) nonStat.push_back(i);
    sort(nonStat.begin(), nonStat.end(), [&](int a, int b) { return conc[a] < conc[b]; });
    int ptr = 0;
    for (int j = 1; j <= n; j++) {
        if (order[j] == 0) order[j] = nonStat[ptr++];
    }

    // ---- gaps ---------------------------------------------------------------
    gap.assign(n + 1, 0);
    for (int j = 2; j <= n; j++) gap[j] = max((ll)0, conc[order[j - 1]] - conc[order[j]]);

    // ---- two candidates, keep the winner -------------------------------------
    vector<int> recalA = recalEven();
    vector<int> tipA = bestTip(recalA);
    ll FA = evalF(tipA, recalA);

    vector<int> recalB = recalSnap();
    vector<int> tipB = bestTip(recalB);
    ll FB = evalF(tipB, recalB);

    const vector<int>& tip = (FB < FA) ? tipB : tipA;
    const vector<int>& recal = (FB < FA) ? recalB : recalA;

    for (int j = 1; j <= n; j++) printf("%d%c", order[j], j == n ? '\n' : ' ');
    for (int j = 1; j <= n; j++) printf("%d%c", tip[j], j == n ? '\n' : ' ');
    for (int j = 1; j <= n; j++) printf("%d%c", recal[j], j == n ? '\n' : ' ');
    return 0;
}
