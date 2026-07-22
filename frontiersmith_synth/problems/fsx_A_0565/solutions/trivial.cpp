// TIER: trivial
// Even-count partition: split the position-sorted jobs into K contiguous zones of (almost)
// equal job count, cranes 1..K, with the minimal staggered offsets.  This is EXACTLY the
// checker's baseline construction, so it scores the calibration point ratio ~= 0.1.
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

    // zone (crane) per sorted rank, equal-count
    vector<int> zoneOfInput(M);
    vector<long long> W(K,0), mn(K,LLONG_MAX), mx(K,LLONG_MIN), D(K,0);
    vector<char> ne(K,0);
    for (int r = 0; r < M; r++) {
        int c = (int)((long long)r * K / M); if (c >= K) c = K-1;
        int i = ord[r];
        zoneOfInput[i] = c;
        ne[c]=1; W[c]+=work[i]; mn[c]=min(mn[c],pos[i]); mx[c]=max(mx[c],pos[i]);
    }
    for (int c=0;c<K;c++) if (ne[c]) D[c]=W[c]+alpha*(mx[c]-mn[c]);

    // minimal staggered offsets
    vector<int> order; for (int c=0;c<K;c++) if(ne[c]) order.push_back(c);
    vector<long long> s(K,0);
    long long runClock=0, prevMax=LLONG_MIN; bool first=true;
    for (int c : order) {
        long long gap = first ? (S+1) : (mn[c]-prevMax);
        if (first || gap>=S) { s[c]=0; runClock=D[c]; }
        else { s[c]=runClock; runClock+=D[c]; }
        prevMax=mx[c]; first=false;
    }

    // output
    for (int i=0;i<M;i++) printf("%d%c", zoneOfInput[i]+1, i+1<M?' ':'\n');
    for (int c=0;c<K;c++) printf("%lld%c", s[c], c+1<K?' ':'\n');
    return 0;
}
