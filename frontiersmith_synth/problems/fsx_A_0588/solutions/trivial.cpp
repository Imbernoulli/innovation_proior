// TIER: trivial
// Do-nothing: no exchangers. Every stream is served entirely by external utility,
// reproducing the checker's baseline B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int NH, NC; long long dTmin;
    if(!(cin>>NH>>NC>>dTmin)) return 0;
    long long a,b,cp;
    for(int i=0;i<NH;i++) cin>>a>>b>>cp;
    for(int j=0;j<NC;j++) cin>>a>>b>>cp;
    cout<<0<<"\n";
    return 0;
}
