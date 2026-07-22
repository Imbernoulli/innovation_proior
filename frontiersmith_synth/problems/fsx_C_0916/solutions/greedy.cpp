// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// The "try a few standard 1-D clustering recipes and keep whichever wins"
// engineering habit: quantile bucketing, equal-length bucketing, Lloyd's
// algorithm (from a uniform seed and from a weighted farthest-point seed,
// updating by weighted MEDIAN or weighted MEAN), and greedy incremental
// facility placement -- ALL of them under the SYMMETRIC (two-sided)
// distance a textbook k-median/k-means solver assumes. Every one of these
// is the textbook-correct facility choice for ordinary two-sided k-median,
// and every one of them is wrong here: this problem's cost is ONE-SIDED (a
// stake only helps positions at or beyond it), so a symmetric center wastes
// reach on points below it that gain nothing, while the low edge of what it
// "should" serve falls back on a distant earlier stake. Keeping the best of
// several standard recipes is a realistic engineering habit, not an insight
// about one-sidedness -- none of these candidates ever reasons about which
// side of a stake a query falls on.

static inline ll ceilDiv(ll a, ll b) { return (a + b - 1) / b; }

static ll evalCost(const vector<ll> &pos, const vector<ll> &cnt, ll W, vector<ll> stakes) {
    stakes.push_back(0);
    sort(stakes.begin(), stakes.end());
    stakes.erase(unique(stakes.begin(), stakes.end()), stakes.end());
    ll F = 0;
    for (size_t i = 0; i < pos.size(); i++) {
        auto it = upper_bound(stakes.begin(), stakes.end(), pos[i]);
        --it;
        F += cnt[i] * ceilDiv(pos[i] - *it, W);
    }
    return F;
}

static vector<ll> lloydRun(vector<ll> center, const vector<ll> &pos, const vector<ll> &cnt,
                            int K, int Q, bool useMean) {
    vector<int> assign(Q, 0);
    for (int iter = 0; iter < 25; iter++) {
        bool changed = false;
        for (int i = 0; i < Q; i++) {
            int best = 0;
            ll bestD = llabs(pos[i] - center[0]);
            for (int k = 1; k < K; k++) {
                ll d = llabs(pos[i] - center[k]);
                if (d < bestD) { bestD = d; best = k; }
            }
            if (assign[i] != best) { assign[i] = best; changed = true; }
        }
        vector<vector<int>> members((size_t)K);
        for (int i = 0; i < Q; i++) members[assign[i]].push_back(i);
        for (int k = 0; k < K; k++) {
            if (members[k].empty()) continue;
            if (useMean) {
                ll tot = 0; long double sum = 0;
                for (int idx : members[k]) { tot += cnt[idx]; sum += (long double)cnt[idx] * pos[idx]; }
                if (tot > 0) center[k] = (ll)llround((double)(sum / tot));
            } else {
                ll tot = 0;
                for (int idx : members[k]) tot += cnt[idx];
                ll half = (tot + 1) / 2, run = 0;
                ll med = pos[members[k].front()];
                for (int idx : members[k]) {
                    run += cnt[idx];
                    if (run >= half) { med = pos[idx]; break; }
                }
                center[k] = med;
            }
        }
        if (!changed && iter > 0) break;
    }
    return center;
}

static vector<ll> quantileMedian(const vector<ll> &pos, const vector<ll> &cnt, int K, int Q) {
    ll S = 0;
    for (int i = 0; i < Q; i++) S += cnt[i];
    vector<ll> stakes;
    int i = 0;
    for (int b = 1; b <= K && i < Q; b++) {
        ll target = (S * (ll)b) / K;
        int start = i; ll acc = 0;
        while (i < Q && acc < target) { acc += cnt[i]; i++; }
        if (i == start) { if (i < Q) { acc += cnt[i]; i++; } else break; }
        ll half = (acc + 1) / 2, run = 0, med = pos[start];
        for (int j = start; j < i; j++) { run += cnt[j]; if (run >= half) { med = pos[j]; break; } }
        stakes.push_back(med);
    }
    while ((int)stakes.size() < K) stakes.push_back(stakes.empty() ? 0 : stakes.back());
    return stakes;
}

static vector<ll> equalLenMedian(ll L, const vector<ll> &pos, const vector<ll> &cnt, int K, int Q) {
    vector<ll> stakes;
    int i = 0;
    for (ll b = 0; b < K; b++) {
        ll lo = L * b / K, hi = L * (b + 1) / K;
        int start = i;
        while (i < Q && pos[i] < hi) i++;
        int end = i;
        if (end > start) {
            ll tot = 0;
            for (int j = start; j < end; j++) tot += cnt[j];
            ll half = (tot + 1) / 2, run = 0, med = pos[start];
            for (int j = start; j < end; j++) { run += cnt[j]; if (run >= half) { med = pos[j]; break; } }
            stakes.push_back(med);
        } else stakes.push_back((lo + hi) / 2);
    }
    return stakes;
}

static vector<ll> greedyIncremental(const vector<ll> &pos, const vector<ll> &cnt, int K, int Q) {
    vector<ll> curBest(Q);
    for (int i = 0; i < Q; i++) curBest[i] = llabs(pos[i] - 0);
    vector<ll> chosen;
    vector<char> used(Q, 0);
    for (int step = 0; step < K; step++) {
        int bestC = -1; long double bestReduction = -1;
        for (int c = 0; c < Q; c++) {
            if (used[c]) continue;
            long double reduction = 0;
            for (int i = 0; i < Q; i++) {
                ll d = llabs(pos[i] - pos[c]);
                if (d < curBest[i]) reduction += (long double)(curBest[i] - d) * cnt[i];
            }
            if (reduction > bestReduction) { bestReduction = reduction; bestC = c; }
        }
        if (bestC < 0) break;
        used[bestC] = 1;
        chosen.push_back(pos[bestC]);
        for (int i = 0; i < Q; i++) { ll d = llabs(pos[i] - pos[bestC]); if (d < curBest[i]) curBest[i] = d; }
    }
    while ((int)chosen.size() < K) chosen.push_back(0);
    return chosen;
}

int main() {
    ll L, K, W;
    int Q;
    if (!(cin >> L >> K >> Q >> W)) return 0;
    vector<ll> pos(Q), cnt(Q);
    for (int i = 0; i < Q; i++) cin >> pos[i] >> cnt[i];

    if (K <= 0 || Q == 0) { for (ll i = 0; i < K; i++) cout << 0 << " \n"[i + 1 == K]; return 0; }
    int Ki = (int)K;

    vector<ll> uniform(Ki);
    for (ll i = 1; i <= K; i++) uniform[i - 1] = L * i / (K + 1);

    // weighted farthest-point (k-means++ style) seeding
    vector<ll> farseed;
    int firstI = 0; ll bw = -1;
    for (int i = 0; i < Q; i++) if (cnt[i] > bw) { bw = cnt[i]; firstI = i; }
    farseed.push_back(pos[firstI]);
    vector<ll> dist(Q, LLONG_MAX);
    for (int i = 0; i < Q; i++) dist[i] = llabs(pos[i] - farseed[0]);
    while ((int)farseed.size() < Ki) {
        int bi = 0; long double bs = -1;
        for (int i = 0; i < Q; i++) {
            long double sc = (long double)dist[i] * (long double)cnt[i];
            if (sc > bs) { bs = sc; bi = i; }
        }
        farseed.push_back(pos[bi]);
        for (int i = 0; i < Q; i++) dist[i] = min(dist[i], llabs(pos[i] - pos[bi]));
    }
    sort(farseed.begin(), farseed.end());

    vector<ll> qm = quantileMedian(pos, cnt, Ki, Q);
    vector<ll> el = equalLenMedian(L, pos, cnt, Ki, Q);
    vector<ll> gi = greedyIncremental(pos, cnt, Ki, Q);

    vector<vector<ll>> cands = {
        uniform,
        lloydRun(uniform, pos, cnt, Ki, Q, false), lloydRun(farseed, pos, cnt, Ki, Q, false),
        lloydRun(uniform, pos, cnt, Ki, Q, true),  lloydRun(farseed, pos, cnt, Ki, Q, true),
        lloydRun(gi, pos, cnt, Ki, Q, false),      lloydRun(qm, pos, cnt, Ki, Q, false),
        gi, qm, el
    };

    ll bestCost = LLONG_MAX;
    vector<ll> best = uniform;
    for (auto &cand : cands) {
        ll c = evalCost(pos, cnt, W, cand);
        if (c < bestCost) { bestCost = c; best = cand; }
    }

    for (size_t k = 0; k < best.size(); k++)
        cout << best[k] << (k + 1 < best.size() ? ' ' : '\n');
    return 0;
}
