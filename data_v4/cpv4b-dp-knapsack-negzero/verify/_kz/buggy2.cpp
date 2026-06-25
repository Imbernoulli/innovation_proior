#include <bits/stdc++.h>
using namespace std;
int main(){
    int n; long long K,C;
    if(!(cin>>n>>K>>C)) return 0;
    vector<long long> w(n),v(n);
    for(int i=0;i<n;i++) cin>>w[i]>>v[i];
    if(K<0||K>n){cout<<"INFEASIBLE\n";return 0;}
    const long long NEG=LLONG_MIN/4;
    int Kc=(int)K,Cc=(int)C;
    vector<vector<long long>> dp(Kc+1, vector<long long>(Cc+1, NEG));
    dp[0][0]=0;
    for(int i=0;i<n;i++){
        long long wi=w[i],vi=v[i];
        if(wi>(long long)Cc) continue;
        int wint=(int)wi;
        // BUG: counts iterated UPWARD -> a single parcel can fill several of the K slots.
        for(int k=0;k<=Kc-1;k++)
            for(int c=Cc-wint;c>=0;c--){
                if(dp[k][c]==NEG) continue;
                dp[k+1][c+wint]=max(dp[k+1][c+wint], dp[k][c]+vi);
            }
    }
    long long ans=NEG;
    for(int c=0;c<=Cc;c++) if(dp[Kc][c]!=NEG) ans=max(ans,dp[Kc][c]);
    if(ans==NEG) cout<<"INFEASIBLE\n"; else cout<<ans<<"\n";
    return 0;
}
