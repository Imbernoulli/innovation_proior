// TIER: greedy
// Standalone-benefit greedy: perform a task alone iff its omission penalty
// exceeds its solo dock round-trip travel (Chebyshev) plus instrumentation.
// Each performed task is mounted then immediately read (rack load = q_i <= Q).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static inline ll cheb(ll ax, ll ay, ll bx, ll by){ return max(llabs(ax-bx), llabs(ay-by)); }

int main() {
    int P; ll Q;
    scanf("%d %lld", &P, &Q);
    ll x0,y0; scanf("%lld %lld",&x0,&y0);
    vector<ll> px(P),py(P),dx(P),dy(P),q(P),c(P),w(P);
    for (int i=0;i<P;i++)
        scanf("%lld %lld %lld %lld %lld %lld %lld",&px[i],&py[i],&dx[i],&dy[i],&q[i],&c[i],&w[i]);

    vector<int> served;
    for (int i=0;i<P;i++){
        ll solo = cheb(x0,y0,px[i],py[i]) + cheb(px[i],py[i],dx[i],dy[i])
                + cheb(dx[i],dy[i],x0,y0) + c[i];
        if (w[i] > solo) served.push_back(i);
    }
    // benefit order: largest (w - solo) first (does not change F for serialized plan,
    // but gives a stable deterministic order)
    sort(served.begin(), served.end(), [&](int a,int b){
        ll sa = w[a]-(cheb(x0,y0,px[a],py[a])+cheb(px[a],py[a],dx[a],dy[a])+cheb(dx[a],dy[a],x0,y0)+c[a]);
        ll sb = w[b]-(cheb(x0,y0,px[b],py[b])+cheb(px[b],py[b],dx[b],dy[b])+cheb(dx[b],dy[b],x0,y0)+c[b]);
        return sa > sb;
    });

    printf("%d\n", (int)served.size()*2);
    for (int idx: served){
        printf("0 %d\n", idx+1);
        printf("1 %d\n", idx+1);
    }
    return 0;
}
