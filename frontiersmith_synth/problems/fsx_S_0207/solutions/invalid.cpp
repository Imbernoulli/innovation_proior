// TIER: invalid
// Deliberately infeasible: places two type-0 patches at the same offset -> overlap
// (or exceeds supply if cnt[0]==1). Must score 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int H,W,T; scanf("%d %d %d",&H,&W,&T);
    for(int r=0;r<H;r++)for(int c=0;c<W;c++){int x;scanf("%d",&x);}
    for(int i=0;i<T;i++){int cn,s;scanf("%d %d",&cn,&s);for(int k=0;k<s;k++){int a,b;scanf("%d %d",&a,&b);}}
    printf("2\n0 0 0 0\n0 0 0 0\n");
    return 0;
}
