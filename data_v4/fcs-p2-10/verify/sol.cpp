#include <bits/stdc++.h>
using namespace std;
int main(){
    int n;
    if(!(cin>>n)) return 0;
    vector<long long> p(n+1, 0);
    for(int i=1;i<=n;i++) cin>>p[i];
    // dp[L] = max revenue obtainable from a rod of length L.
    // dp[0]=0; dp[L]=max over first-piece length k in 1..L of p[k]+dp[L-k].
    vector<long long> dp(n+1, 0);
    for(int L=1;L<=n;L++){
        long long best = LLONG_MIN;
        for(int k=1;k<=L;k++){
            long long cand = p[k] + dp[L-k];
            if(cand>best) best=cand;
        }
        dp[L]=best;
    }
    cout << dp[n] << "\n";
    return 0;
}
