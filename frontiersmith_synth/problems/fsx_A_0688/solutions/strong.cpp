// TIER: strong
// The insight: opening cost only gates WHICH sites you may use -- once open, more
// capacity never hurts -- so sites should be ranked by capacity-served PER UNIT
// BUDGET (a value/cost density, using the nearest-facility catchment load as the
// locality signal), not by raw cost. Assignment then processes the largest
// arrival-rate demand nodes first and routes each to whichever open site
// minimizes travel cost PLUS the MARGINAL increase in that site's queueing
// term lambda/(mu-lambda) -- since that marginal term explodes near saturation,
// this naturally spreads load onto sites with headroom well before any site is
// driven to the brink. A bounded local-relocation pass then mops up.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int n, m; ll Bud, QW;
    scanf("%d %d %lld %lld", &n, &m, &Bud, &QW);
    vector<ll> dx(n+1), dy(n+1), dlam(n+1);
    for (int i = 1; i <= n; i++) scanf("%lld %lld %lld", &dx[i], &dy[i], &dlam[i]);
    vector<ll> fx(m+1), fy(m+1), fmu(m+1), fcost(m+1);
    for (int j = 1; j <= m; j++) scanf("%lld %lld %lld %lld", &fx[j], &fy[j], &fmu[j], &fcost[j]);

    auto dist = [&](ll ax, ll ay, ll bx, ll by)->double{
        double dxv = (double)(ax - bx), dyv = (double)(ay - by);
        return sqrt(dxv*dxv + dyv*dyv);
    };

    ll TotalLambda = 0;
    for (int i = 1; i <= n; i++) TotalLambda += dlam[i];

    // locality signal: nearest-facility catchment load (pure geometry, all m sites)
    vector<ll> catchLoad(m+1, 0);
    for (int i = 1; i <= n; i++) {
        int best = 1; double bd = dist(dx[i], dy[i], fx[1], fy[1]);
        for (int j = 2; j <= m; j++) {
            double d = dist(dx[i], dy[i], fx[j], fy[j]);
            if (d < bd) { bd = d; best = j; }
        }
        catchLoad[best] += dlam[i];
    }

    vector<int> order(m);
    for (int j = 0; j < m; j++) order[j] = j + 1;
    sort(order.begin(), order.end(), [&](int a, int b){
        double va = (double)catchLoad[a] + 0.25 * (double)fmu[a];
        double vb = (double)catchLoad[b] + 0.25 * (double)fmu[b];
        double ra = va / (double)fcost[a], rb = vb / (double)fcost[b];
        if (ra != rb) return ra > rb;
        return a < b;
    });

    vector<char> isOpen(m + 1, 0);
    ll spent = 0, capOpen = 0;
    for (int j : order) {
        if (spent + fcost[j] <= Bud) { spent += fcost[j]; isOpen[j] = 1; capOpen += fmu[j]; }
    }
    // safety net: keep adding highest-capacity affordable sites until total open
    // capacity can plausibly cover all demand
    if (capOpen < TotalLambda) {
        vector<int> byMu(m);
        for (int j = 0; j < m; j++) byMu[j] = j + 1;
        sort(byMu.begin(), byMu.end(), [&](int a, int b){ return fmu[a] > fmu[b]; });
        for (int j : byMu) {
            if (capOpen >= TotalLambda) break;
            if (!isOpen[j] && spent + fcost[j] <= Bud) { spent += fcost[j]; isOpen[j] = 1; capOpen += fmu[j]; }
        }
    }
    vector<int> open;
    for (int j = 1; j <= m; j++) if (isOpen[j]) open.push_back(j);
    if (open.empty()) { open.push_back(order[0]); isOpen[order[0]] = 1; }

    vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i + 1;
    sort(idx.begin(), idx.end(), [&](int a, int b){ return dlam[a] > dlam[b]; });

    auto qterm = [&](ll L, ll mu)->double{
        if (L <= 0) return 0.0;
        if (L >= mu) return 1e18;
        return (double)L / (double)(mu - L);
    };

    vector<ll> load(m + 1, 0);
    vector<int> assign(n + 1, -1);
    for (int i : idx) {
        int best = -1; double bestCost = 1e30;
        for (int f : open) {
            ll newLoad = load[f] + dlam[i];
            if (newLoad >= fmu[f]) continue;
            double travel = (double)dlam[i] * dist(dx[i], dy[i], fx[f], fy[f]);
            double dq = (double)QW * (qterm(newLoad, fmu[f]) - qterm(load[f], fmu[f]));
            double c = travel + dq;
            if (c < bestCost) { bestCost = c; best = f; }
        }
        if (best == -1) {
            ll bestSlack = -1;
            for (int f : open) {
                ll slack = fmu[f] - load[f];
                if (slack > bestSlack) { bestSlack = slack; best = f; }
            }
        }
        load[best] += dlam[i];
        assign[i] = best;
    }

    // bounded local-relocation pass
    for (int sweep = 0; sweep < 3; sweep++) {
        bool improved = false;
        for (int i = 1; i <= n; i++) {
            int cur = assign[i];
            ll curLoadWithout = load[cur] - dlam[i];
            double curContrib = (double)dlam[i] * dist(dx[i], dy[i], fx[cur], fy[cur])
                               + (double)QW * (qterm(load[cur], fmu[cur]) - qterm(curLoadWithout, fmu[cur]));
            int best = cur; double bestDelta = 0.0;
            for (int f : open) {
                if (f == cur) continue;
                ll newLoad = load[f] + dlam[i];
                if (newLoad >= fmu[f]) continue;
                double addContrib = (double)dlam[i] * dist(dx[i], dy[i], fx[f], fy[f])
                                   + (double)QW * (qterm(newLoad, fmu[f]) - qterm(load[f], fmu[f]));
                double delta = addContrib - curContrib;
                if (delta < bestDelta - 1e-9) { bestDelta = delta; best = f; }
            }
            if (best != cur) {
                load[cur] -= dlam[i];
                load[best] += dlam[i];
                assign[i] = best;
                improved = true;
            }
        }
        if (!improved) break;
    }

    printf("%d\n", (int)open.size());
    for (int f : open) printf("%d ", f);
    printf("\n");
    for (int i = 1; i <= n; i++) printf("%d ", assign[i]);
    printf("\n");
    return 0;
}
