// TIER: strong
// Insight: QUOTIENT BY THE PHASE LATTICE FIRST. Walk every seam chain once,
// from its free-bottom end, to derive the unique required residue (y mod r)
// and rotation for EVERY piece before any packing decision is made. Because
// each chain's internal congruence is now fully resolved into one number per
// piece, the chains "collapse into independent components that then nest
// freely": pack with a normal width-aware shelf heuristic (NFDH: sort by
// height, fill rows left to right), and only round each piece's own y UP (by
// at most r-1, within its own row -- rows are at different x per piece so
// this never causes a new overlap) to hit its already-known target residue.
// No chain is ever revisited after packing, so no cascade, no wasted column.
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

    // ---- quotient by the phase lattice: solve every chain up front ----
    vector<ll> required(N, -1);
    vector<int> chosenF(N, 0);
    vector<char> visited(N, 0);
    for (int i = 0; i < N; i++) {
        if (hasBottomUsed[i]) continue;
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

    // ---- now nest freely: NFDH shelf packing, height-descending order ----
    vector<int> ord(N);
    for (int i = 0; i < N; i++) ord[i] = i;
    sort(ord.begin(), ord.end(), [&](int a, int b) { return h[a] > h[b]; });

    vector<ll> X(N, 0), Y(N, 0);
    ll rowY = 0, curX = 0, rowNextH = 0;
    for (int idx = 0; idx < N; idx++) {
        int p = ord[idx];
        if (curX + w[p] > W) { rowY += rowNextH; curX = 0; rowNextH = 0; }
        ll rem = ((rowY % r) + r) % r;
        ll delta = ((required[p] - rem) % r + r) % r;
        ll ypos = rowY + delta;
        X[p] = curX; Y[p] = ypos;
        curX += w[p];
        rowNextH = max(rowNextH, ypos + h[p] - rowY);
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
