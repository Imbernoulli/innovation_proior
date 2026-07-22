// TIER: strong
// The insight: edge cost is a function of the schedule itself, so topology
// and visiting rhythm must be chosen TOGETHER, online, rather than fixing a
// static-weight tree first and serializing it blindly. At every time step we
// pick the connecting measurement that minimizes the CURRENT incremental
// cost -- inherited error + each side's drift accumulated since it was last
// touched + the geometric floor. This naturally: (a) front-loads high-drift
// unconnected elements (their d_b*t term only grows with time), and (b)
// reuses a just-touched hub again immediately whenever that is cheapest,
// instead of leaving it stale while unrelated branches get processed -- no
// separate "reorder" logic is needed, it falls out of re-optimizing the
// schedule at every step. Once every element is connected, any leftover time
// budget is spent on redundant cross-check re-touches, always picking the
// pair that reduces total error the most.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N; ll T;
    scanf("%d %lld", &N, &T);
    vector<ll> X(N), Y(N), D(N);
    for (int i = 0; i < N; i++) scanf("%lld %lld %lld", &X[i], &Y[i], &D[i]);

    auto w = [&](int a, int b) { return llabs(X[a] - X[b]) + llabs(Y[a] - Y[b]); };

    vector<char> connected(N, 0);
    vector<ll> err(N, 0), last(N, 0);
    connected[0] = 1;
    int nConnected = 1;

    string out;
    out.reserve((size_t)T * 8);

    for (ll t = 1; t <= T; t++) {
        if (nConnected < N) {
            // find the connecting pair (a connected, b unconnected) with the
            // minimum incremental cost given the CURRENT clock state.
            ll bestCost = LLONG_MAX;
            int bestA = -1, bestB = -1;
            for (int a = 0; a < N; a++) {
                if (!connected[a]) continue;
                ll deltaA = D[a] * (t - last[a]);
                for (int b = 0; b < N; b++) {
                    if (connected[b]) continue;
                    ll cost = err[a] + deltaA + D[b] * t + w(a, b);
                    if (cost < bestCost) { bestCost = cost; bestA = a; bestB = b; }
                }
            }
            int a = bestA, b = bestB;
            ll deltaA = D[a] * (t - last[a]);
            ll deltaB = D[b] * t;
            err[b] = err[a] + deltaA + deltaB + w(a, b);
            last[a] = t; last[b] = t;
            connected[b] = 1;
            nConnected++;
            out += to_string(a); out += ' '; out += to_string(b); out += '\n';
        } else {
            // everyone connected: spend the step on the most valuable
            // redundant cross-check re-touch (or a harmless repeat if none
            // helps).
            ll bestGain = -1;
            int bu = 0, bv = 1;
            for (int u = 0; u < N; u++) {
                ll deltaU = D[u] * (t - last[u]);
                for (int v = u + 1; v < N; v++) {
                    ll deltaV = D[v] * (t - last[v]);
                    ll ww = w(u, v);
                    ll candU = err[v] + deltaV + deltaU + ww;
                    ll candV = err[u] + deltaU + deltaV + ww;
                    ll gain = 0;
                    if (candU < err[u]) gain += err[u] - candU;
                    if (candV < err[v]) gain += err[v] - candV;
                    if (gain > bestGain) { bestGain = gain; bu = u; bv = v; }
                }
            }
            int u = bu, v = bv;
            ll deltaU = D[u] * (t - last[u]);
            ll deltaV = D[v] * (t - last[v]);
            ll ww = w(u, v);
            ll candU = err[v] + deltaV + deltaU + ww;
            ll candV = err[u] + deltaU + deltaV + ww;
            err[u] = min(err[u], candU);
            err[v] = min(err[v], candV);
            last[u] = t; last[v] = t;
            out += to_string(u); out += ' '; out += to_string(v); out += '\n';
        }
    }
    fputs(out.c_str(), stdout);
    return 0;
}
