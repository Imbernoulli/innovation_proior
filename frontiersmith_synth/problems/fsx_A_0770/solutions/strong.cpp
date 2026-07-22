// TIER: strong
// Insight: distance to the listener is not observable in the objective at all --
// only the phi2 coupling table is. Reformulate as marginal-gain (submodular)
// selection under the checker's OWN Lorentzian objective: repeatedly install the
// affordable panel that most reduces the true F right now (re-evaluating after
// every pick, since a channel's marginal benefit shrinks as it gets damped).
// This automatically finds a resonant channel's antinode -- wherever it is in
// the room, including far from every listener -- because that is exactly where
// phi2 is largest for the channel dominating F.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static const ll SCALE2 = 1000000LL;
static const ll EK_CAP = 1000000000LL;

int NM;
vector<ll> Delta, Gamma0, S, L;

ll computeF(const vector<ll>& gE){
    ll F = 0;
    for (int k = 0; k < NM; k++){
        ll Gk = Gamma0[k] + gE[k];
        ll denom = Delta[k]*Delta[k] + Gk*Gk;
        if (denom < 1) denom = 1;
        ll Ek = (S[k]*S[k]) * SCALE2 / denom;
        if (Ek > EK_CAP) Ek = EK_CAP;
        F += Ek * L[k];
    }
    return F;
}

int main(){
    int W,H,sx,sy; scanf("%d %d",&W,&H); scanf("%d %d",&sx,&sy);
    int NL; scanf("%d",&NL);
    for(int i=0;i<NL;i++){ int a,b; scanf("%d %d",&a,&b); }
    scanf("%d",&NM);
    Delta.resize(NM); Gamma0.resize(NM); S.resize(NM); L.resize(NM);
    for(int k=0;k<NM;k++) scanf("%lld %lld %lld %lld",&Delta[k],&Gamma0[k],&S[k],&L[k]);
    ll Budget; scanf("%lld",&Budget);
    int M; scanf("%d",&M);
    vector<ll> cost(M+1), alpha(M+1);
    vector<vector<ll>> phi2(M+1, vector<ll>(NM));
    for(int i=1;i<=M;i++){
        int id,x,y; scanf("%d %d %d %lld %lld",&id,&x,&y,&cost[i],&alpha[i]);
        for(int k=0;k<NM;k++) scanf("%lld",&phi2[i][k]);
    }

    vector<ll> gammaExtra(NM, 0);
    vector<char> used(M+1, 0);
    ll spent = 0;
    ll curF = computeF(gammaExtra);
    vector<int> chosen;

    // precompute per-candidate contribution vectors once
    while (true){
        int bestId = -1; double bestDensity = 0; ll bestNewF = curF;
        for (int i = 1; i <= M; i++){
            if (used[i] || spent + cost[i] > Budget) continue;
            vector<ll> tmp(NM);
            for (int k = 0; k < NM; k++) tmp[k] = gammaExtra[k] + alpha[i]*phi2[i][k]/1000;
            ll nf = computeF(tmp);
            ll reduction = curF - nf;
            if (reduction <= 0) continue;
            double density = (double)reduction / (double)cost[i];
            if (density > bestDensity){ bestDensity = density; bestId = i; bestNewF = nf; }
        }
        if (bestId == -1) break;
        used[bestId] = 1; spent += cost[bestId];
        for (int k = 0; k < NM; k++) gammaExtra[k] += alpha[bestId]*phi2[bestId][k]/1000;
        curF = bestNewF;
        chosen.push_back(bestId);
    }

    printf("%d\n", (int)chosen.size());
    for (int id : chosen) printf("%d ", id);
    printf("\n");
    return 0;
}
