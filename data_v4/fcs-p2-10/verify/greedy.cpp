#include <bits/stdc++.h>
using namespace std;
int main(){
    int n;
    if(!(cin>>n)) return 0;
    vector<long long> p(n+1, 0);
    for(int i=1;i<=n;i++) cin>>p[i];
    // WRONG greedy: repeatedly cut the piece length with the best price-per-length
    // ratio that still fits in the remaining rod.
    long long remaining = n, total = 0;
    while(remaining > 0){
        int bestk = -1; double bestRatio = -1e18;
        for(int k=1;k<=remaining;k++){
            double r = (double)p[k] / k;
            if(r > bestRatio){ bestRatio = r; bestk = k; }
        }
        if(bestk == -1) break;
        total += p[bestk];
        remaining -= bestk;
    }
    cout << total << "\n";
    return 0;
}
