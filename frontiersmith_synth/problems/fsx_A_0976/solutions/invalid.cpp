// TIER: invalid
// Deliberately infeasible: prints S copies of the out-of-range candidate index C
// (valid indices are 0..C-1). The checker's bounded readInt(0,C-1,...) rejects it.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N,M,C,S,K;
    scanf("%d %d %d %d %d", &N,&M,&C,&S,&K);
    for (int i=0;i<S;i++) printf("%d ", C);
    printf("\n");
    return 0;
}
