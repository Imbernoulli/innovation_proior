// TIER: greedy
// Load-balancing greedy: cut the position-sorted jobs into K contiguous zones so each crane
// gets (almost) equal TOTAL hoisting work, cranes 1..K.  This is the obvious "balance the
// load" recipe -- it ignores the safety gaps (so on blob instances it slices a congested
// cluster and pays serialization) and ignores the standby powers.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int K, M; long long S, alpha, gamma;
    if (scanf("%d %d %lld %lld %lld", &K, &M, &S, &alpha, &gamma) != 5) return 0;
    vector<long long> P(K);
    for (int c = 0; c < K; c++) scanf("%lld", &P[c]);
    vector<long long> pos(M), work(M);
    for (int i = 0; i < M; i++) scanf("%lld %lld", &pos[i], &work[i]);

    vector<int> ord(M);
    for (int i = 0; i < M; i++) ord[i] = i;
    sort(ord.begin(), ord.end(), [&](int a, int b){ return pos[a] < pos[b]; });

    long long total = 0;
    for (int i = 0; i < M; i++) total += work[i];

    // assign zones by cumulative-work quantiles
    vector<int> zoneOfInput(M);
    long long acc = 0; int c = 0;
    for (int r = 0; r < M; r++) {
        int i = ord[r];
        // advance crane while we have passed its work target and jobs remain for the rest
        while (c < K - 1 && acc >= (long long)(c + 1) * total / K && (M - r) > (K - 1 - c))
            c++;
        zoneOfInput[i] = c;
        acc += work[i];
    }

    // per-crane duration
    vector<long long> W(K,0), mn(K,LLONG_MAX), mx(K,LLONG_MIN), D(K,0);
    vector<char> ne(K,0);
    for (int i=0;i<M;i++){ int z=zoneOfInput[i]; ne[z]=1; W[z]+=work[i]; mn[z]=min(mn[z],pos[i]); mx[z]=max(mx[z],pos[i]); }
    for (int cc=0;cc<K;cc++) if (ne[cc]) D[cc]=W[cc]+alpha*(mx[cc]-mn[cc]);

    // minimal staggered offsets
    vector<int> order; for (int cc=0;cc<K;cc++) if(ne[cc]) order.push_back(cc);
    vector<long long> s(K,0);
    long long runClock=0, prevMax=LLONG_MIN; bool first=true;
    for (int cc : order) {
        long long gap = first ? (S+1) : (mn[cc]-prevMax);
        if (first || gap>=S) { s[cc]=0; runClock=D[cc]; }
        else { s[cc]=runClock; runClock+=D[cc]; }
        prevMax=mx[cc]; first=false;
    }

    for (int i=0;i<M;i++) printf("%d%c", zoneOfInput[i]+1, i+1<M?' ':'\n');
    for (int cc=0;cc<K;cc++) printf("%lld%c", s[cc], cc+1<K?' ':'\n');
    return 0;
}
