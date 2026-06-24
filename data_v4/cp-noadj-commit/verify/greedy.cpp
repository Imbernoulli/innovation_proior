#include <bits/stdc++.h>
using namespace std;
int main(){
    int n; if(!(cin>>n)) return 0;
    vector<long long> a(n);
    for(auto&x:a) cin>>x;
    // greedy: repeatedly take the largest remaining positive element, block its neighbors
    vector<char> used(n,0), blocked(n,0);
    long long total=0;
    while(true){
        int best=-1;
        for(int i=0;i<n;i++) if(!used[i]&&!blocked[i]&&a[i]>0&&(best==-1||a[i]>a[best])) best=i;
        if(best==-1) break;
        total+=a[best]; used[best]=1;
        if(best>0) blocked[best-1]=1;
        if(best+1<n) blocked[best+1]=1;
    }
    cout<<total<<"\n";
    return 0;
}
