// TIER: greedy
// Standalone-benefit greedy: serve a request iff its skip-penalty exceeds the cost
// of a dedicated depot->hub->clinic->depot detour; output the chosen requests
// serialized (P_i D_i) in decreasing order of benefit. No reordering / batching.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static inline ll md(ll ax, ll ay, ll bx, ll by){ return llabs(ax-bx)+llabs(ay-by); }

int main(){
    int P, Q; ll x0, y0;
    if(scanf("%d %d", &P, &Q)!=2) return 0;
    scanf("%lld %lld", &x0, &y0);
    vector<ll> px(P+1),py(P+1),dx(P+1),dy(P+1),q(P+1),w(P+1);
    for(int i=1;i<=P;i++)
        scanf("%lld %lld %lld %lld %lld %lld",&px[i],&py[i],&dx[i],&dy[i],&q[i],&w[i]);

    vector<pair<ll,int>> cand; // (benefit, i)
    for(int i=1;i<=P;i++){
        ll solo = md(x0,y0,px[i],py[i]) + md(px[i],py[i],dx[i],dy[i]) + md(dx[i],dy[i],x0,y0);
        ll benefit = w[i] - solo;
        if(benefit > 0) cand.push_back({benefit, i});
    }
    sort(cand.begin(), cand.end(), [](const pair<ll,int>&a, const pair<ll,int>&b){
        return a.first > b.first;
    });

    printf("%d\n", (int)cand.size()*2);
    for(auto &c : cand){
        int i = c.second;
        printf("0 %d\n", i);
        printf("1 %d\n", i);
    }
    return 0;
}
