// TIER: trivial
// Do-nothing baseline: build no new footbridges. The augmented graph equals the original, so its
// algebraic connectivity equals the checker's internal baseline -> ratio ~ 0.1 by construction.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int n,m; long long B;
    if(scanf("%d %d %lld",&n,&m,&B)!=3) return 0;
    printf("0\n");
    return 0;
}
