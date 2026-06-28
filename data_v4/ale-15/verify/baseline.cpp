// Trivial baseline for ale-15: place every rectangle's bottom-left corner at
// (0,0). This is always feasible (the generator guarantees W >= maxw+10 and
// H >= maxh+10, so every rectangle fits) but piles all facilities on top of one
// another, so the overlap energy is huge. It is the floor the real solver must
// beat.
#include <bits/stdc++.h>
using namespace std;
int main(){
    int N; long long W, H;
    if(!(cin>>N>>W>>H)) return 0;
    for(int i=0;i<N;i++){long long w,h;cin>>w>>h;}
    string out;
    for(int i=0;i<N;i++) out += "0 0\n";
    cout<<out;
    return 0;
}
