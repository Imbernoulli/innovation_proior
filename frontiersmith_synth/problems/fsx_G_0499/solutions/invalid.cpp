// TIER: invalid
// Deliberately infeasible: select two fragments with NO bond -> the assembled
// "molecule" is disconnected -> checker's connectivity check fails -> score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    ll M,C,Wmax,Lam,Rb;
    if(!(cin>>M>>C>>Wmax>>Lam>>Rb)) return 0;
    // read+ignore the rest is unnecessary; we just print two disconnected picks.
    if(M<2){ printf("1\n1\n0\n"); return 0; }
    printf("2\n1 2\n0\n");   // 2 fragments, 0 bonds -> not connected
    return 0;
}
