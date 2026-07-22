// TIER: trivial
// Freeze up to K target pixels inside the box with the identity rule.
// Reproduces exactly the checker baseline B -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N, K, T, r0, c0, b;
    if(!(cin >> N >> K >> T >> r0 >> c0 >> b)) return 0;
    vector<string> g(N);
    for(int r=0;r<N;r++) cin >> g[r];

    // identity rule: B*=0, S*=1
    vector<int> rule(18, 0);
    for(int j=9;j<18;j++) rule[j]=1;

    vector<pair<int,int>> seed;
    for(int r=r0;r<r0+b && (int)seed.size()<K;r++)
        for(int c=c0;c<c0+b && (int)seed.size()<K;c++)
            if(g[r][c]=='1') seed.push_back({r,c});

    for(int j=0;j<18;j++) cout << rule[j] << (j<17?' ':'\n');
    cout << seed.size() << "\n";
    for(auto&p:seed) cout << p.first << " " << p.second << "\n";
    return 0;
}
