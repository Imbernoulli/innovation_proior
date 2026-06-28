#include <bits/stdc++.h>
using namespace std;
int main(){int q;cin>>q;while(q--){long long n,k;cin>>n>>k;long long r=0;for(long long m=2;m<=n;++m)r=(r+k)%m;cout<<r+1<<"\n";}}
