// TIER: strong
// The insight: evaluate every add/remove/swap move against the TRUE full
// objective -- which correctly re-prices the crowd penalty of every
// already-selected neighbour, not just the moving node's own charge. That
// externality-aware delta lets local search find each cluster's true
// crowd-respecting core instead of overshooting it. Multiple randomized
// restarts (construction order + random subsets) with add/remove/swap hill
// climbing keep the best full team found. The search is stopped by a fixed,
// input-size-derived OPERATION budget (never by a wall-clock read), so the
// same input always drives the same sequence of moves and produces the
// exact same output regardless of machine speed.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M;
vector<ll> v, cap, pen;
vector<vector<pair<int,int>>> adj; // neighbor, s

vector<char> selected;
vector<ll> degS;
ll curF;

ll opCount = 0;      // deterministic work counter (NOT wall-clock)
ll opBudget = 0;      // set from N,M in main()

static inline ll penaltyOf(ll deg, ll c, ll p) {
    ll ex = deg - c;
    return ex > 0 ? p * ex : 0;
}

// True delta F if we add currently-unselected node k (accounts for the
// crowd-penalty change it retroactively imposes on already-selected
// neighbours).
ll deltaAdd(int k) {
    opCount += 1 + (ll)adj[k].size();
    ll d = v[k];
    ll synGain = 0, penChange = 0;
    for (auto& pr : adj[k]) {
        int j = pr.first, s = pr.second;
        if (selected[j]) {
            synGain += s;
            penChange += penaltyOf(degS[j] + 1, cap[j], pen[j]) - penaltyOf(degS[j], cap[j], pen[j]);
        }
    }
    penChange += penaltyOf(degS[k], cap[k], pen[k]); // k's own new penalty
    return d + synGain - penChange;
}

// True delta F if we remove currently-selected node k.
ll deltaRemove(int k) {
    opCount += 1 + (ll)adj[k].size();
    ll d = -v[k];
    ll synLoss = 0, penSave = 0;
    for (auto& pr : adj[k]) {
        int j = pr.first, s = pr.second;
        if (selected[j]) {
            synLoss += s;
            penSave += penaltyOf(degS[j], cap[j], pen[j]) - penaltyOf(degS[j] - 1, cap[j], pen[j]);
        }
    }
    penSave += penaltyOf(degS[k], cap[k], pen[k]); // k's own penalty removed
    return -synLoss + penSave + d;
}

void applyAdd(int k) {
    ll d = deltaAdd(k);
    selected[k] = 1;
    // degS[j] = # of j's neighbours that are selected, for EVERY j (selected
    // or not) -- must be updated unconditionally, or an unselected node's
    // degS goes stale and its own future penalty is mis-scored.
    for (auto& pr : adj[k]) degS[pr.first]++;
    curF += d;
}
void applyRemove(int k) {
    ll d = deltaRemove(k);
    selected[k] = 0;
    for (auto& pr : adj[k]) degS[pr.first]--;
    curF += d;
}

void resetState() {
    selected.assign(N, 0);
    degS.assign(N, 0);
    curF = 0;
}

// Local search to a hill-climbing optimum: repeatedly apply the single best
// improving add/remove move until none improves (or the op budget is spent).
void localSearch() {
    bool improved = true;
    while (improved && opCount < opBudget) {
        improved = false;
        int bestNode = -1; ll bestDelta = 0; bool bestIsAdd = false;
        for (int k = 0; k < N; k++) {
            if (!selected[k]) {
                ll d = deltaAdd(k);
                if (d > bestDelta) { bestDelta = d; bestNode = k; bestIsAdd = true; }
            } else {
                ll d = deltaRemove(k);
                if (d > bestDelta) { bestDelta = d; bestNode = k; bestIsAdd = false; }
            }
        }
        if (bestNode != -1) {
            if (bestIsAdd) applyAdd(bestNode); else applyRemove(bestNode);
            improved = true;
        }
    }
}

int main() {
    scanf("%d %d", &N, &M);
    v.resize(N); cap.resize(N); pen.resize(N);
    for (int i = 0; i < N; i++) scanf("%lld", &v[i]);
    for (int i = 0; i < N; i++) scanf("%lld", &cap[i]);
    for (int i = 0; i < N; i++) scanf("%lld", &pen[i]);
    adj.assign(N, {});
    for (int e = 0; e < M; e++) {
        int a, b, s;
        scanf("%d %d %d", &a, &b, &s);
        adj[a].push_back({b, s});
        adj[b].push_back({a, s});
    }

    mt19937 rng(12345);
    // Fixed, input-size-derived operation budget: bounds work the same way
    // on every machine (a slower CPU just takes longer wall time -- it never
    // does LESS work or picks a different sequence of moves).
    opBudget = 200000000LL / (ll)(N + 4 * (M / max(1, N) + 1) + 8);
    if (opBudget < 400000) opBudget = 400000;
    if (opBudget > 60000000) opBudget = 60000000;

    vector<char> bestSel;
    ll bestF = -1;

    auto recordIfBetter = [&]() {
        if (curF > bestF) { bestF = curF; bestSel = selected; }
    };

    // Restart 0: empty set is always a safe baseline.
    resetState();
    recordIfBetter();

    int restart = 0;
    while (opCount < opBudget) {
        restart++;

        resetState();
        int mode = restart % 3;
        if (mode == 0) {
            // Correct-marginal greedy construction (order by true current delta).
            for (int iter = 0; iter < N; iter++) {
                int best = -1; ll bd = 0;
                for (int k = 0; k < N; k++) if (!selected[k]) {
                    ll d = deltaAdd(k);
                    if (d > bd) { bd = d; best = k; }
                }
                if (best == -1) break;
                applyAdd(best);
            }
        } else if (mode == 1) {
            // Randomized order construction: shuffle candidates, add if positive.
            vector<int> order(N);
            iota(order.begin(), order.end(), 0);
            shuffle(order.begin(), order.end(), rng);
            for (int k : order) {
                if (!selected[k] && deltaAdd(k) > 0) applyAdd(k);
            }
        } else {
            // Random-density seed then let local search clean it up.
            double p = uniform_real_distribution<double>(0.05, 0.6)(rng);
            for (int k = 0; k < N; k++)
                if (!selected[k] && (double)rng() / rng.max() < p) applyAdd(k);
        }

        localSearch();

        // A handful of random swap attempts to escape local optima, then
        // re-run local search if any swap stuck.
        for (int t = 0; t < 150; t++) {
            if (opCount >= opBudget) break;
            vector<int> selList, unselList;
            for (int i = 0; i < N; i++) (selected[i] ? selList : unselList).push_back(i);
            if (selList.empty() || unselList.empty()) break;
            int k1 = selList[rng() % selList.size()];
            int k2 = unselList[rng() % unselList.size()];
            ll d1 = deltaRemove(k1);
            applyRemove(k1);
            ll d2 = deltaAdd(k2);
            applyAdd(k2);
            if (d1 + d2 <= 0) {
                // revert
                ll u1 = deltaRemove(k2); applyRemove(k2);
                ll u2 = deltaAdd(k1); applyAdd(k1);
                (void)u1; (void)u2;
            }
        }
        localSearch();
        recordIfBetter();
    }

    vector<int> pick;
    for (int i = 0; i < N; i++) if (i < (int)bestSel.size() && bestSel[i]) pick.push_back(i);
    printf("%d\n", (int)pick.size());
    for (size_t i = 0; i < pick.size(); i++) printf("%d%c", pick[i], (i + 1 == pick.size()) ? '\n' : ' ');
    if (pick.empty()) printf("\n");
    return 0;
}
