#include <bits/stdc++.h>
using namespace std;
typedef long long ll; typedef __int128 lll;
int main(){
    int n; ll L; if(scanf("%d %lld",&n,&L)!=2) return 0;
    vector<ll> a(n); for(auto&x:a) scanf("%lld",&x);
    vector<ll> S(n+1,0); for(int i=0;i<n;i++) S[i+1]=S[i]+a[i];
    bool have=false; ll bn=0,bd=1;
    for(int i=0;i<=n;i++) for(int j=i+(int)L;j<=n;j++){
        ll num=S[j]-S[i], den=j-i;
        if(!have){bn=num;bd=den;have=true;}
        else { lll l=(lll)num*(lll)bd, r=(lll)bn*(lll)den; if(l>r){bn=num;bd=den;} }
    }
    ll g=std::__gcd(bn<0?-bn:bn, bd); if(g==0)g=1; bn/=g; bd/=g;
    printf("%lld/%lld\n",bn,bd); return 0;
}
