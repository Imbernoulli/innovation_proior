// TIER: trivial
// Baseline: get every phase constraint RIGHT (via the same chain-propagation
// every valid solution needs) but ignore width-sharing entirely -- every
// piece gets its own private column (x=0 for all), serialized one after
// another. This wastes almost all of the fabric's width, which is exactly
// the checker's calibration point (B assumes perfect width tiling).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N; ll W, r;
    if (scanf("%d %lld %lld", &N, &W, &r) != 3) return 0;
    vector<ll> w(N), h(N), req(N);
    for (int i = 0; i < N; i++) scanf("%lld %lld %lld", &w[i], &h[i], &req[i]);
    int M; scanf("%d", &M);
    vector<int> topTo(N, -1);
    vector<char> hasBottomUsed(N, 0);
    for (int k = 0; k < M; k++) {
        int i, ei, j, ej; scanf("%d %d %d %d", &i, &ei, &j, &ej);
        if (ei == 1 && ej == 0) { topTo[i] = j; hasBottomUsed[j] = 1; }
        else if (ei == 0 && ej == 1) { topTo[j] = i; hasBottomUsed[i] = 1; }
    }

    auto target = [&](int i, int f) -> ll {
        if (f == 0) return ((-req[i]) % r + r) % r;
        return ((req[i] - h[i]) % r + r) % r;
    };

    vector<ll> required(N, -1);
    vector<int> chosenF(N, 0);
    vector<char> visited(N, 0);
    vector<int> chainStart;
    for (int i = 0; i < N; i++) {
        if (hasBottomUsed[i]) continue;
        chainStart.push_back(i);
        visited[i] = 1; chosenF[i] = 0; required[i] = target(i, 0);
        int cur = i;
        while (topTo[cur] != -1) {
            int nxt = topTo[cur];
            ll outPhase = (required[cur] + h[cur]) % r;
            ll t0 = target(nxt, 0), t1 = target(nxt, 1);
            int fch = (t0 == outPhase) ? 0 : ((t1 == outPhase) ? 1 : 0);
            chosenF[nxt] = fch; required[nxt] = outPhase; visited[nxt] = 1;
            cur = nxt;
        }
    }

    vector<ll> X(N, 0), Y(N, 0);
    ll cursor = 0;
    for (int s : chainStart) {
        int cur = s;
        while (true) {
            ll rem = ((cursor % r) + r) % r;
            ll delta = ((required[cur] - rem) % r + r) % r;
            ll ypos = cursor + delta;
            X[cur] = 0; Y[cur] = ypos;
            cursor = ypos + h[cur];
            if (topTo[cur] == -1) break;
            cur = topTo[cur];
        }
    }

    string buf;
    for (int i = 0; i < N; i++) {
        buf += to_string(X[i]); buf += ' ';
        buf += to_string(Y[i]); buf += ' ';
        buf += to_string(chosenF[i]); buf += '\n';
        if (buf.size() > (1u << 20)) { fputs(buf.c_str(), stdout); buf.clear(); }
    }
    fputs(buf.c_str(), stdout);
    return 0;
}
