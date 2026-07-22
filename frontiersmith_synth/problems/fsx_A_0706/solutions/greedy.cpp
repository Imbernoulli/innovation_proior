// TIER: greedy
// The obvious approach: nest first, worry about stripes after. Assume every
// piece keeps its as-cut orientation (f=0 -- rotation "obviously" shouldn't
// matter for packing) and pack width-aware with a standard shelf heuristic,
// rounding each piece's y up to its own OWN stripe requirement under f=0.
// Only THEN check each seam chain: does the f=0-everywhere guess actually
// satisfy every seam in it? A chain of length k matches by luck with
// probability ~2^-(k-1), because the seam relation pins two DIFFERENT
// pieces' residues together through a fixed offset chain, and once a piece's
// own phase constraint is honored under f=0 its y mod r is a NON-NEGOTIABLE
// number -- translation can never repair a seam mismatch caused by the wrong
// rotation guess (shifting y by any amount preserves y mod r). So whenever
// the guess is wrong anywhere in a chain, greedy is forced to throw the
// WHOLE chain's nice shelf placement away and re-lay it out from scratch,
// single-column, at the very end of the fabric -- it never revisits its
// rotation assumption early enough to avoid this, and it never reconsiders
// f for the pieces it originally guessed correctly either (no global view).
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

    // correct (only used as a fallback once the f=0 guess is caught failing)
    vector<ll> correctReq(N, -1);
    vector<int> correctF(N, 0);
    // per-chain bookkeeping
    vector<vector<int>> chainMembers;
    vector<char> chainOkZero;
    vector<char> visited(N, 0);

    for (int i = 0; i < N; i++) {
        if (hasBottomUsed[i]) continue;
        vector<int> members;
        // correct solve (needed for feasibility no matter what)
        visited[i] = 1; correctF[i] = 0; correctReq[i] = target(i, 0);
        members.push_back(i);
        int cur = i;
        while (topTo[cur] != -1) {
            int nxt = topTo[cur];
            ll outPhase = (correctReq[cur] + h[cur]) % r;
            ll t0 = target(nxt, 0), t1 = target(nxt, 1);
            int fch = (t0 == outPhase) ? 0 : ((t1 == outPhase) ? 1 : 0);
            correctF[nxt] = fch; correctReq[nxt] = outPhase; visited[nxt] = 1;
            members.push_back(nxt);
            cur = nxt;
        }
        // does the naive "f=0 everywhere" guess also satisfy this chain?
        bool okZero = true;
        ll cz = target(i, 0);
        cur = i;
        while (topTo[cur] != -1) {
            int nxt = topTo[cur];
            ll outPhase = (cz + h[cur]) % r;
            ll t0 = target(nxt, 0);
            if (t0 != outPhase) { okZero = false; break; }
            cz = t0; cur = nxt;
        }
        chainMembers.push_back(members);
        chainOkZero.push_back(okZero);
    }

    vector<ll> X(N, 0), Y(N, 0);
    vector<int> Fo(N, 0);
    vector<char> placed(N, 0);

    // ---- efficient shelf pack for every chain whose f=0 guess happens to work ----
    vector<int> caseAPieces;
    for (size_t c = 0; c < chainMembers.size(); c++) {
        if (chainOkZero[c]) {
            for (int p : chainMembers[c]) { caseAPieces.push_back(p); Fo[p] = 0; }
        }
    }
    sort(caseAPieces.begin(), caseAPieces.end(), [&](int a, int b) { return h[a] > h[b]; });
    ll rowY = 0, curX = 0, rowNextH = 0;
    for (int p : caseAPieces) {
        if (curX + w[p] > W) { rowY += rowNextH; curX = 0; rowNextH = 0; }
        ll req0 = target(p, 0);
        ll rem = ((rowY % r) + r) % r;
        ll delta = ((req0 - rem) % r + r) % r;
        ll ypos = rowY + delta;
        X[p] = curX; Y[p] = ypos;
        curX += w[p];
        rowNextH = max(rowNextH, ypos + h[p] - rowY);
        placed[p] = 1;
    }
    ll shelfEnd = rowY + rowNextH;

    // ---- chains where the guess failed: re-lay out single-column at the tail ----
    ll cursor = shelfEnd;
    for (size_t c = 0; c < chainMembers.size(); c++) {
        if (chainOkZero[c]) continue;
        for (int p : chainMembers[c]) {
            ll rem = ((cursor % r) + r) % r;
            ll delta = ((correctReq[p] - rem) % r + r) % r;
            ll ypos = cursor + delta;
            X[p] = 0; Y[p] = ypos; Fo[p] = correctF[p];
            cursor = ypos + h[p];
            placed[p] = 1;
        }
    }

    string buf;
    for (int i = 0; i < N; i++) {
        buf += to_string(X[i]); buf += ' ';
        buf += to_string(Y[i]); buf += ' ';
        buf += to_string(Fo[i]); buf += '\n';
        if (buf.size() > (1u << 20)) { fputs(buf.c_str(), stdout); buf.clear(); }
    }
    fputs(buf.c_str(), stdout);
    return 0;
}
