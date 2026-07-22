// TIER: trivial
// Do-nothing: post no tolls. This is exactly the baseline the checker measures,
// so it scores ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N, M, D, R; long long T;
    if(!(cin >> N >> M >> D >> T >> R)) return 0;
    for(int e = 0; e < M; e++){ if(e) putchar(' '); printf("0"); }
    printf("\n");
    return 0;
}
