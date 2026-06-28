// Trivial baseline: choose the first K household indices as towers.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N,K;
    if(!(cin>>N>>K)) return 0;
    // read & discard coords
    for(int i=0;i<N;i++){long long x,y;cin>>x>>y;}
    if(K>N)K=N;
    cout<<K<<"\n";
    for(int i=0;i<K;i++) cout<<(i+1)<<"\n";
    return 0;
}
