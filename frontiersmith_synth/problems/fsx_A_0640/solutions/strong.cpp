// TIER: strong
// The insight: feasibility is governed by how much of the wire is ALREADY
// shaped at the moment a station is bent, not by the station's position in
// the part. Bending strictly in geometric order always swings the entire
// unshaped remainder on early stations (worst case). Instead use a
// BALANCE-POINT RECURSION: bend the station that splits the remaining
// contiguous index range most evenly FIRST (it braces both halves at once
// for everything that comes later), then recurse on the two halves. This
// keeps every subsequent reach bounded by roughly half of what remains
// instead of the whole tail, so far fewer stations ever collide -- including
// the heavy near-base stations that plain path order loses. Precompensation
// uses the same budgeted springback fix as greedy.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int m; ll L; int c, TOL; ll K;
vector<ll> theta, delta, w;
vector<pair<int, ll>> outLines;

void rec(int lo, int hi) {
    if (lo > hi) return;
    int mid = (lo + hi) / 2;
    ll comp = -delta[mid];
    if (comp > c) comp = c;
    if (comp < -c) comp = -c;
    outLines.push_back({mid, theta[mid] + comp});
    rec(lo, mid - 1);
    rec(mid + 1, hi);
}

int main() {
    scanf("%d %lld %d %d %lld", &m, &L, &c, &TOL, &K);
    vector<ll> x(m + 1);
    theta.assign(m + 1, 0); delta.assign(m + 1, 0); w.assign(m + 1, 0);
    for (int i = 1; i <= m; i++)
        scanf("%lld %lld %lld %lld", &x[i], &theta[i], &delta[i], &w[i]);
    rec(1, m);
    for (auto& p : outLines) printf("%d %lld\n", p.first, p.second);
    return 0;
}
