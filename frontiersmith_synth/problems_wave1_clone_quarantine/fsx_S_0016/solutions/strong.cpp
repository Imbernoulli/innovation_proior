// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static inline ll cheb(ll ax, ll ay, ll bx, ll by) {
    return max(llabs(ax - bx), llabs(ay - by));
}

int P, Q;
ll dpx, dpy;
vector<ll> px, py, dx, dy, q, w, f;

// event: (type, idx); type 0 = pickup, 1 = delivery
vector<pair<int,int>> ev;
vector<char> served;

static inline void ptOf(const pair<int,int>& e, ll& X, ll& Y) {
    int i = e.second;
    if (e.first == 0) { X = px[i]; Y = py[i]; }
    else              { X = dx[i]; Y = dy[i]; }
}

// Try to insert request r's pickup+delivery as an adjacent pair at the best gap.
// Accept only if it strictly reduces the objective.
bool tryInsert(int r) {
    int L = (int)ev.size();
    vector<ll> loadBefore(L + 1, 0), onboardF(L + 1, 0);
    ll cl = 0, cf = 0;
    for (int k = 0; k < L; k++) {
        int i = ev[k].second;
        if (ev[k].first == 0) { cl += q[i]; cf += f[i]; }
        else                  { cl -= q[i]; cf -= f[i]; }
        loadBefore[k + 1] = cl;
        onboardF[k + 1]  = cf;
    }

    ll bestDelta = LLONG_MAX;
    int bestG = -1;
    for (int g = 0; g <= L; g++) {
        if (loadBefore[g] + q[r] > Q) continue;  // no room in cage at this gap
        ll bx, by, ax, ay;
        if (g == 0) { bx = dpx; by = dpy; } else ptOf(ev[g - 1], bx, by);
        if (g == L) { ax = dpx; ay = dpy; } else ptOf(ev[g],     ax, ay);

        ll added = cheb(bx, by, px[r], py[r])
                 + cheb(px[r], py[r], dx[r], dy[r])
                 + cheb(dx[r], dy[r], ax, ay)
                 - cheb(bx, by, ax, ay);
        // Inserting two events at gap g pushes every request on board there 2 stops later.
        ll carryImpact = 2 * onboardF[g];
        ll delta = added + carryImpact - w[r];   // hops_r = 0 (adjacent), so no f[r] term
        if (delta < bestDelta) { bestDelta = delta; bestG = g; }
    }

    if (bestG >= 0 && bestDelta < 0) {
        ev.insert(ev.begin() + bestG, {1, r});
        ev.insert(ev.begin() + bestG, {0, r});
        served[r] = 1;
        return true;
    }
    return false;
}

int main() {
    if (scanf("%d %d", &P, &Q) != 2) return 0;
    scanf("%lld %lld", &dpx, &dpy);
    px.resize(P + 1); py.resize(P + 1); dx.resize(P + 1); dy.resize(P + 1);
    q.resize(P + 1); w.resize(P + 1); f.resize(P + 1);
    for (int i = 1; i <= P; i++)
        scanf("%lld %lld %lld %lld %lld %lld %lld",
              &px[i], &py[i], &dx[i], &dy[i], &q[i], &w[i], &f[i]);
    served.assign(P + 1, 0);

    // Priority by standalone attractiveness (penalty minus solo round trip).
    vector<int> ord;
    for (int i = 1; i <= P; i++) ord.push_back(i);
    sort(ord.begin(), ord.end(), [&](int a, int b) {
        ll sa = cheb(dpx, dpy, px[a], py[a]) + cheb(px[a], py[a], dx[a], dy[a]) + cheb(dx[a], dy[a], dpx, dpy);
        ll sb = cheb(dpx, dpy, px[b], py[b]) + cheb(px[b], py[b], dx[b], dy[b]) + cheb(dx[b], dy[b], dpx, dpy);
        return (w[a] - sa) > (w[b] - sb);
    });

    for (int r : ord) tryInsert(r);

    // Remove-and-reinsert local search: re-evaluate every request against the current tour.
    for (int pass = 0; pass < 3; pass++) {
        for (int r = 1; r <= P; r++) {
            if (served[r]) {
                vector<pair<int,int>> ne;
                ne.reserve(ev.size());
                for (auto& e : ev) if (e.second != r) ne.push_back(e);
                ev.swap(ne);
                served[r] = 0;
            }
            tryInsert(r);   // may or may not (re)add r
        }
    }

    printf("%d\n", (int)ev.size());
    for (auto& e : ev) printf("%d %d\n", e.first, e.second);
    return 0;
}
