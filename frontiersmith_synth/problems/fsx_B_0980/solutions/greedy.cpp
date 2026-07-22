// TIER: greedy
// The "obvious first idea": rank segments by raw NATURAL (no-extraction)
// pre-mix temperature only -- ignore eta and hardware capacity entirely --
// take the K hottest-looking segments, then at each one (processed in
// position order) extract as much as its capacity/Tsink limit allows.
// This is a trap: it never accounts for the fact that draining an upstream
// segment hard dumps cold water into the mix and suppresses every later
// chosen segment's ACTUAL temperature far below what the natural profile
// promised, and it can waste a slot on a very hot but low-quality (low eta)
// decoy segment.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int T, K, Tsink;
    scanf("%d %d %d", &T, &K, &Tsink);
    vector<ll> q(T + 1), theta(T + 1), eta(T + 1), cap(T + 1);
    for (int i = 1; i <= T; i++)
        scanf("%lld %lld %lld %lld", &q[i], &theta[i], &eta[i], &cap[i]);

    vector<double> pre(T + 1);
    {
        double F = 0.0, Tcur = 0.0;
        for (int i = 1; i <= T; i++) {
            double Fnew = F + (double)q[i];
            double Tpre = (F == 0.0) ? (double)theta[i] : (F * Tcur + (double)q[i] * theta[i]) / Fnew;
            F = Fnew;
            pre[i] = Tpre;
            Tcur = Tpre;
        }
    }

    vector<int> order(T);
    iota(order.begin(), order.end(), 1);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (pre[a] != pre[b]) return pre[a] > pre[b];
        return a < b;
    });
    vector<char> chosen(T + 1, 0);
    for (int k = 0; k < K && k < T; k++) chosen[order[k]] = 1;

    vector<pair<int,ll>> out;
    double F = 0.0, Tcur = 0.0;
    for (int i = 1; i <= T; i++) {
        double Fnew = F + (double)q[i];
        double Tpre = (F == 0.0) ? (double)theta[i] : (F * Tcur + (double)q[i] * theta[i]) / Fnew;
        F = Fnew;
        if (chosen[i]) {
            double feasMax = F * (Tpre - Tsink);
            double xmax = min((double)cap[i], feasMax);
            if (xmax < 0) xmax = 0;
            ll x = (ll)floor(xmax);
            out.push_back({i, x});
            double Tpost = Tpre - (double)x / F;
            if (Tpost < Tsink) Tpost = Tsink;
            Tcur = Tpost;
        } else {
            Tcur = Tpre;
        }
    }

    printf("%d\n", (int)out.size());
    for (auto& pr : out) printf("%d %lld\n", pr.first, pr.second);
    return 0;
}
