#include <bits/stdc++.h>
using namespace std;
int main(){
    int n; long long K,C;
    if(!(cin>>n>>K>>C)) return 0;
    vector<long long> w(n),v(n);
    for(int i=0;i<n;i++) cin>>w[i]>>v[i];
    if(K<0||K>n){cout<<"INFEASIBLE\n";return 0;}
    int Kc=(int)K,Cc=(int)C;
    // BUG: whole table initialized to 0 (so "k parcels, weight c" is treated as
    // already-achievable with profit 0 even when it is not).
    vector<vector<long long>> dp(Kc+1, vector<long long>(Cc+1, 0));
    for(int i=0;i<n;i++){
        long long wi=w[i],vi=v[i];
        if(wi>(long long)Cc) continue;
        int wint=(int)wi;
        for(int k=Kc-1;k>=0;k--)
            for(int c=Cc-wint;c>=0;c--)
                dp[k+1][c+wint]=max(dp[k+1][c+wint], dp[k][c]+vi);
    }
    long long ans=LLONG_MIN;
    for(int c=0;c<=Cc;c++) ans=max(ans,dp[Kc][c]);
    cout<<ans<<"\n";
    return 0;
}
