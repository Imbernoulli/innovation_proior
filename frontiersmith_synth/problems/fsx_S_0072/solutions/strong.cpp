// TIER: strong
// Per-ship dynamic program that trades power price against gantry re-mobilisation
// (G per contiguous run) subject to the shared spot capacity. Ships are handled
// tight-window-first so scarce spot slots go where slack is smallest; each ship's
// DP picks, for every hour, pause / spot (if a shared slot remains) / shore, then
// the used spot slots are committed for later ships.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll INF = (ll)1e18;

int main() {
    int T, J, G;
    if (scanf("%d %d %d", &T, &J, &G) != 3) return 0;
    vector<ll> sp(T), od(T);
    vector<int> C(T);
    for (int t = 0; t < T; t++) {
        long a,b,c; scanf("%ld %ld %ld", &a,&b,&c);
        sp[t]=a; od[t]=b; C[t]=(int)c;
    }
    vector<int> A(J), B(J), W(J), order(J);
    for (int j = 0; j < J; j++) { scanf("%d %d %d", &A[j], &B[j], &W[j]); order[j]=j; }

    // tight-window-first: smallest slack (window - demand), tie by shorter window
    sort(order.begin(), order.end(), [&](int x, int y){
        int sx=(B[x]-A[x])-W[x], sy=(B[y]-A[y])-W[y];
        if (sx!=sy) return sx<sy;
        return (B[x]-A[x])<(B[y]-A[y]);
    });

    vector<int> rem = C;                 // remaining shared spot capacity
    vector<vector<int>> outH(J), outM(J); // per-ship chosen hours / modes

    for (int oi = 0; oi < J; oi++) {
        int j = order[oi];
        int a = A[j], b = B[j], w = W[j];
        int L = b - a;
        // per-position (p=hour a+p) base cost of working + mode choice
        vector<ll> base(L);   // min(spot if slot, shore)
        vector<int> mode(L);  // 0 spot, 1 shore
        for (int p = 0; p < L; p++) {
            int t = a + p;
            ll cs = (rem[t] > 0) ? sp[t] : INF;
            ll cd = od[t];
            if (cs <= cd) { base[p] = cs; mode[p] = 0; }
            else          { base[p] = cd; mode[p] = 1; }
        }
        // dp[k][r]: min cost, k hours worked among processed positions, r=last worked?
        // iterate positions; reconstruct with parent choices.
        vector<vector<array<ll,2>>> dp(L + 1,
            vector<array<ll,2>>(w + 1, {INF, INF}));
        // choice[i][k][r] = 0 skip, 1 work (came into state after position i-1)
        vector<vector<array<char,2>>> ch(L + 1,
            vector<array<char,2>>(w + 1, {(char)-1,(char)-1}));
        dp[0][0][0] = 0;
        for (int i = 0; i < L; i++) {
            for (int k = 0; k <= w; k++) {
                for (int r = 0; r < 2; r++) {
                    ll cur = dp[i][k][r];
                    if (cur >= INF) continue;
                    // skip position i
                    if (cur < dp[i+1][k][0]) { dp[i+1][k][0] = cur; ch[i+1][k][0] = 0 + (r<<1); }
                    // work position i
                    if (k + 1 <= w && base[i] < INF) {
                        ll add = base[i] + (r ? 0 : (ll)G);
                        ll nv = cur + add;
                        if (nv < dp[i+1][k+1][1]) { dp[i+1][k+1][1] = nv; ch[i+1][k+1][1] = 1 + (r<<1); }
                    }
                }
            }
        }
        int endr = -1; ll best = INF;
        for (int r = 0; r < 2; r++) if (dp[L][w][r] < best) { best = dp[L][w][r]; endr = r; }
        // reconstruct
        vector<pair<int,int>> picks; // (hour, mode)
        int k = w, r = endr;
        for (int i = L; i >= 1; i--) {
            char c = ch[i][k][r];
            int act = c & 1;         // 0 skip, 1 work
            int pr  = (c >> 1) & 1;  // previous r
            if (act == 1) {
                int p = i - 1;
                picks.push_back({a + p, mode[p]});
                k -= 1;
            }
            r = pr;
        }
        // commit spot usage
        for (auto& pm : picks) if (pm.second == 0) rem[pm.first]--;
        for (auto& pm : picks) { outH[j].push_back(pm.first); outM[j].push_back(pm.second); }
    }

    for (int j = 0; j < J; j++) {
        for (size_t i = 0; i < outH[j].size(); i++)
            printf("%d %d ", outH[j][i], outM[j][i]);
        printf("\n");
    }
    return 0;
}
