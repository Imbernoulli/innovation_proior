// TIER: trivial
// Reproduces the checker's own baseline B: pick the single common line that,
// used alone, saves the most cut length, and activate only that one.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N, M;
    scanf("%d %d", &N, &M);
    vector<ll> H(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &H[i]);
    vector<vector<int>> mem(M + 1);
    vector<ll> L(M + 1);
    vector<ll> baseline(N + 1, 0);
    for (int g = 1; g <= M; g++){
        int k; ll Lg; scanf("%d %lld", &k, &Lg);
        L[g] = Lg; mem[g].resize(k);
        for (int j = 0; j < k; j++){ scanf("%d", &mem[g][j]); baseline[mem[g][j]] += Lg / 2; }
    }
    vector<ll> Rextra(N + 1);
    for (int i = 1; i <= N; i++) Rextra[i] = H[i] - baseline[i];

    int bestG = -1; ll bestVal = -1; vector<int> bestParts;
    for (int g = 1; g <= M; g++){
        vector<int> ok;
        for (int p : mem[g]) if (Rextra[p] >= L[g] / 2) ok.push_back(p);
        if ((int)ok.size() >= 2){
            ll val = (ll)(ok.size() - 1) * L[g];
            if (val > bestVal){ bestVal = val; bestG = g; bestParts = ok; }
        }
    }
    if (bestG < 0){ printf("0\n"); return 0; }
    printf("1\n%d %d", bestG, (int)bestParts.size());
    for (int p : bestParts) printf(" %d", p);
    printf("\n");
    return 0;
}
