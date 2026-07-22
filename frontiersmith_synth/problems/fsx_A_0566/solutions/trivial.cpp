// TIER: trivial
// Balanced BINARY search tree over 1..N (one tile per page). This reproduces the
// checker's own baseline exactly -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;

int N, B, S;
vector<int> K, L, R;   // per node (index = id-1)
int cnt = 0;

int build(int lo, int hi){
    if (lo > hi) return 0;
    int mid = (lo + hi) / 2;
    int id = ++cnt;
    K.push_back(mid); L.push_back(0); R.push_back(0);
    int l = build(lo, mid - 1);
    int r = build(mid + 1, hi);
    L[id - 1] = l; R[id - 1] = r;
    return id;
}

int main(){
    scanf("%d %d %d", &N, &B, &S);
    // consume the rest of the input (weights + scans) -- unused
    long long tmp;
    for (int i = 0; i < N; i++) scanf("%lld", &tmp);
    for (int i = 0; i < S; i++){ long long a,b,c; scanf("%lld %lld %lld", &a,&b,&c); }

    K.reserve(N); L.reserve(N); R.reserve(N);
    int root = build(1, N);

    printf("%d %d\n", cnt, root);
    for (int id = 1; id <= cnt; id++)
        printf("1 %d %d %d\n", K[id - 1], L[id - 1], R[id - 1]);
    return 0;
}
