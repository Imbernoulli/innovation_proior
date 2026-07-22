// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Textbook "sqrt(n)-spaced skip pointer" recipe: allocate the global rope
// budget proportionally to each lane's LENGTH, and space ropes uniformly by
// INDEX within each lane. Never looks at which lanes are actually queried,
// never looks at values, never looks at how many convoys share a lane. This
// is the obvious first approach and it is the trap: it wastes budget on
// decoy lanes no convoy touches, and even on touched lanes its uniform
// index-breakpoints rarely align with where the dispatcher actually stalls
// (that depends on the VALUES of the lanes it is paired against).
int main() {
    int M, Q, B;
    if (!(cin >> M >> Q >> B)) return 0;
    vector<ll> n(M + 1);
    ll total = 0;
    for (int c = 1; c <= M; c++) {
        cin >> n[c];
        total += n[c];
        for (ll j = 0; j < n[c]; j++) { ll x; cin >> x; }
    }
    // ignore the Q query blocks entirely -- greedy never looks at them

    vector<ll> alloc(M + 1, 0);
    ll used = 0;
    vector<pair<ll,int>> order;
    for (int c = 1; c <= M; c++) {
        ll cap = max((ll)0, n[c] - 1);
        ll a = (total > 0) ? (ll)((double)n[c] / (double)total * B) : 0;
        if (a > cap) a = cap;
        alloc[c] = a;
        used += a;
        order.push_back({n[c], c});
    }
    sort(order.rbegin(), order.rend());
    // hand out any leftover budget to the largest lanes first, still capped
    size_t idx = 0;
    int guard = 0;
    while (used < B && !order.empty() && guard < 4 * M + 10) {
        int c = order[idx % order.size()].second;
        ll cap = max((ll)0, n[c] - 1);
        if (alloc[c] < cap) { alloc[c]++; used++; }
        idx++; guard++;
    }

    for (int c = 1; c <= M; c++) {
        ll k = alloc[c];
        ll len = n[c];
        vector<pair<ll,ll>> ropes;
        if (k > 0 && len >= 2) {
            ll step = max((ll)1, (len - 1) / (k + 1));
            set<ll> used_pos;
            for (ll i = 1; i <= k; i++) {
                ll pos = (ll)i * (len - 1) / (k + 1);
                if (pos > len - 2) pos = len - 2;
                if (pos < 0) pos = 0;
                if (used_pos.count(pos)) continue;
                ll target = pos + step;
                if (target > len - 1) target = len - 1;
                if (target <= pos) target = min(len - 1, pos + 1);
                if (target <= pos) continue; // lane too short, skip
                used_pos.insert(pos);
                ropes.push_back({pos, target});
            }
        }
        cout << ropes.size() << "\n";
        for (auto &pr : ropes) cout << pr.first << " " << pr.second << "\n";
    }
    return 0;
}
