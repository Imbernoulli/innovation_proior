// TIER: invalid
// Deliberately infeasible: prints porosity level Pmax+1 everywhere, which is out of
// the checker's accepted range [0,Pmax] and must score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main(){
    ll W,H,Wb,Pmax,Ymin,Ymax,K;
    cin>>W>>H>>Wb>>Pmax>>Ymin>>Ymax>>K;
    for (ll x=0;x<Wb;x++){
        for (ll y=0;y<H;y++) cout<<(Pmax+1)<<(y+1==H?'\n':' ');
    }
    return 0;
}
