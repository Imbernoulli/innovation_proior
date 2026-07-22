// TIER: invalid
// Deliberately infeasible: prints ignition index 0, which is out of the valid
// range [1,N] on every case -> must score 0.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    for (int k = 0; k < M; k++) printf("0%c", k + 1 < M ? ' ' : '\n');
    return 0;
}
