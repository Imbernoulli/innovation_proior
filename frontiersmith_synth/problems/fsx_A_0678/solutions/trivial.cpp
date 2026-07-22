// TIER: trivial
// Do-nothing uniform phasing: phi_i = 0 for every element. Reproduces the
// checker's internal baseline B exactly -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N, M, D1000;
    scanf("%d %d %d", &N, &M, &D1000);
    for (int i = 0; i < N; i++){ int x; scanf("%d", &x); }
    for (int m = 0; m < M; m++){ int x; scanf("%d", &x); }
    for (int m = 0; m < M; m++){ int x; scanf("%d", &x); }
    for (int m = 0; m < M; m++){ int x; scanf("%d", &x); }
    int K; scanf("%d", &K);
    for (int i = 0; i < K; i++){ int x; scanf("%d", &x); }
    int thresh; scanf("%d", &thresh);

    for (int i = 0; i < N; i++) printf("%d%c", 0, i + 1 == N ? '\n' : ' ');
    return 0;
}
