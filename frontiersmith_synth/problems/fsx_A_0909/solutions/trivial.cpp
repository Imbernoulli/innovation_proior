// TIER: trivial
// Naive baseline: visit each demand type once, in the GIVEN INPUT ORDER, using a
// pattern that repeats that type's width as many times as usefully fits -- but with
// NO batching insight (produce exactly 1 unit per run, revisiting qty_i times) and
// NO reachability planning (bridge straight to the next type in input order).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static pair<vector<ll>, ll> fullPattern(ll W, int K, ll width) {
    ll c = min((ll)K, (W - 1) / width);
    while (c >= 0) {
        ll remSlots = K + 1 - c;
        if (remSlots <= 0) { c--; continue; }
        ll fillerCount = remSlots - 1;
        ll remainder = W - c * width - fillerCount * 1;
        if (remainder >= 1) {
            vector<ll> segs;
            for (ll k = 0; k < c; k++) segs.push_back(width);
            for (ll k = 0; k < fillerCount; k++) segs.push_back(1);
            segs.push_back(remainder);
            vector<ll> pos; ll run = 0;
            for (size_t k = 0; k + 1 < segs.size(); k++) { run += segs[k]; pos.push_back(run); }
            return {pos, c};
        }
        c--;
    }
    vector<ll> pos; ll run = 0;
    for (int k = 0; k < K; k++) { run += max((ll)1, (W - 1) / (K + 1)); pos.push_back(min(run, W - 1)); }
    return {pos, 0};
}

static vector<vector<ll>> bridgeTo(vector<ll> cur, const vector<ll> &target, ll m, int K) {
    vector<vector<ll>> path;
    vector<ll> p = cur;
    int guard = 0;
    while (true) {
        ll maxd = 0;
        for (int j = 0; j < K; j++) maxd = max(maxd, llabs(target[j] - p[j]));
        if (maxd == 0) break;
        vector<ll> rp(K);
        if (maxd <= m) {
            rp = target;
        } else {
            double f = (double)m / (double)maxd;
            for (int j = 0; j < K; j++) {
                double v = p[j] + (target[j] - p[j]) * f;
                ll iv = (ll)llround(v);
                if (iv > p[j] + m) iv = p[j] + m;
                if (iv < p[j] - m) iv = p[j] - m;
                rp[j] = iv;
            }
            for (int j = 1; j < K; j++) if (rp[j] <= rp[j - 1]) rp[j] = rp[j - 1] + 1;
            for (int j = 0; j < K; j++) {
                if (rp[j] > p[j] + m) rp[j] = p[j] + m;
                if (rp[j] < p[j] - m) rp[j] = p[j] - m;
            }
            for (int j = 1; j < K; j++) if (rp[j] <= rp[j - 1]) rp[j] = rp[j - 1] + 1;
        }
        path.push_back(rp);
        p = rp;
        if (++guard > 200000) break;
    }
    return path;
}

int main() {
    ll W, m; int K;
    cin >> W >> K >> m;
    vector<ll> S(K);
    for (int j = 0; j < K; j++) cin >> S[j];
    int n; cin >> n;
    vector<ll> width(n), qty(n);
    for (int i = 0; i < n; i++) cin >> width[i] >> qty[i];
    ll maxSetups, setupCost, penalty;
    cin >> maxSetups >> setupCost >> penalty;

    vector<vector<ll>> outPos; vector<ll> outR;
    vector<ll> cur = S;
    ll setups = 0;
    for (int i = 0; i < n && setups < maxSetups; i++) {
        auto pr = fullPattern(W, K, width[i]);
        vector<ll> pos = pr.first;
        auto path = bridgeTo(cur, pos, m, K);
        for (size_t s = 0; s < path.size() && setups < maxSetups; s++) {
            outPos.push_back(path[s]); outR.push_back(1); setups++;
            cur = path[s];
        }
        for (ll u = 0; u < qty[i] && setups < maxSetups; u++) {
            outPos.push_back(pos); outR.push_back(1); setups++;
        }
    }
    if (outPos.empty()) { outPos.push_back(S); outR.push_back(1); }

    cout << outPos.size() << "\n";
    for (size_t t = 0; t < outPos.size(); t++) {
        for (int j = 0; j < K; j++) cout << outPos[t][j] << " ";
        cout << outR[t] << "\n";
    }
    return 0;
}
