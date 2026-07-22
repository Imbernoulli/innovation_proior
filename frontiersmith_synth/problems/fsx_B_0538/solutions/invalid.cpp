// TIER: invalid
// Deliberately infeasible: a rule value of 2 (must be 0/1) -> checker rejects -> 0.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N,K,T,r0,c0,b;
    if(!(cin>>N>>K>>T>>r0>>c0>>b)) return 0;
    string s; for(int r=0;r<N;r++) cin>>s;
    cout << "2 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1\n";
    cout << "1\n" << r0 << " " << c0 << "\n";
    return 0;
}
