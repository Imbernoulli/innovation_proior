// TIER: strong
// Coordinate descent starting from the all-on baseline. Holding all other
// generators fixed, replace one generator's whole schedule with the cheapest of
// a few candidate patterns (off entirely, on-only-where-forced expanded to warm-up,
// full baseload, or unchanged), accepting only strict improvements. Sweeping the
// fleet to a fixed point strips idle no-load waste and trims start-up churn --
// the baseload-plus-peakers structure the myopic greedy never finds.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int T, G;
vector<ll> P, b, rate, K, U;
vector<ll> D, W, R;
vector<int> order; // by fuel rate for dispatch

ll cost(const vector<vector<char>>& x, bool* feas) {
    *feas = true; ll total = 0;
    vector<char> prev(G, 0);
    for (int t = 0; t < T; t++) {
        ll capON = 0;
        for (int g = 0; g < G; g++) if (x[t][g]) {
            if (!prev[g]) total += K[g];
            total += b[g]; capON += P[g];
        }
        if (capON < R[t]) *feas = false;
        ll rem = R[t];
        for (int idx = 0; idx < G && rem > 0; idx++) {
            int g = order[idx];
            if (!x[t][g]) continue;
            ll use = min(P[g], rem); total += use * rate[g]; rem -= use;
        }
        for (int g = 0; g < G; g++) prev[g] = x[t][g];
    }
    return total;
}

int main() {
    if (scanf("%d %d", &T, &G) != 2) return 0;
    P.resize(G); b.resize(G); rate.resize(G); K.resize(G); U.resize(G);
    for (int g = 0; g < G; g++) scanf("%lld %lld %lld %lld %lld",&P[g],&b[g],&rate[g],&K[g],&U[g]);
    D.resize(T); W.resize(T); R.resize(T);
    for (int t = 0; t < T; t++) { scanf("%lld %lld",&D[t],&W[t]); R[t] = max(0LL, D[t]-W[t]); }

    order.resize(G);
    for (int g = 0; g < G; g++) order[g] = g;
    sort(order.begin(), order.end(), [&](int a, int c){
        if (rate[a] != rate[c]) return rate[a] < rate[c];
        return a < c;
    });

    vector<vector<char>> x(T, vector<char>(G, 1)); // start all-on
    bool f; ll cur = cost(x, &f);

    auto minupOK = [&](const vector<char>& col, int g)->bool{
        int t = 0;
        while (t < T) {
            if (col[t]) { int a=t; while(t<T&&col[t])t++; int len=t-a; if(t!=T && len<U[g]) return false; }
            else t++;
        }
        return true;
    };

    bool improved = true; int passes = 0;
    while (improved && passes < 20) {
        improved = false; passes++;
        for (int g = 0; g < G; g++) {
            // capacity from all other generators at each step
            vector<ll> othersCap(T, 0);
            for (int t = 0; t < T; t++) {
                ll c = 0;
                for (int h = 0; h < G; h++) if (h != g && x[t][h]) c += P[h];
                othersCap[t] = c;
            }
            // candidate patterns for generator g
            vector<vector<char>> cands;
            // (0) current
            { vector<char> col(T); for (int t=0;t<T;t++) col[t]=x[t][g]; cands.push_back(col); }
            // (1) off entirely
            cands.push_back(vector<char>(T, 0));
            // (2) full baseload
            cands.push_back(vector<char>(T, 1));
            // (3) on only where forced by capacity, then expanded to warm-up
            {
                vector<char> col(T, 0);
                for (int t=0;t<T;t++) if (othersCap[t] < R[t]) col[t]=1;
                // enforce min-up by extending each run rightward
                int t=0;
                while (t<T) {
                    if (col[t] && (t==0 || !col[t-1])) {
                        int end = min(T-1, t + (int)U[g] - 1);
                        for (int k=t;k<=end;k++) col[k]=1;
                        t = end + 1;
                    } else t++;
                }
                cands.push_back(col);
            }
            // pick cheapest feasible candidate
            ll bestC = cur; int bestI = -1;
            for (int ci = 0; ci < (int)cands.size(); ci++) {
                const auto& col = cands[ci];
                // feasibility: capacity + min-up
                bool ok = true;
                for (int t=0;t<T && ok;t++) if (othersCap[t] + (col[t]?P[g]:0) < R[t]) ok=false;
                if (ok) ok = minupOK(col, g);
                if (!ok) continue;
                for (int t=0;t<T;t++) x[t][g]=col[t];
                bool ff; ll c = cost(x, &ff);
                if (ff && c < bestC) { bestC = c; bestI = ci; }
            }
            // commit the best (bestI==-1 keeps current, which is cands[0])
            const auto& chosen = (bestI < 0) ? cands[0] : cands[bestI];
            for (int t=0;t<T;t++) x[t][g]=chosen[t];
            if (bestI >= 0 && bestC < cur) { cur = bestC; improved = true; }
            else { bool ff; cur = cost(x, &ff); }
        }
    }

    for (int t = 0; t < T; t++) {
        for (int g = 0; g < G; g++) { if (g) fputc(' ', stdout); fputc(x[t][g] ? '1':'0', stdout); }
        fputc('\n', stdout);
    }
    return 0;
}
