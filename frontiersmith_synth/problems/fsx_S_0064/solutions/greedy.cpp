// TIER: greedy
// Solo-benefit greedy: serve every request whose penalty exceeds its solo
// load-weighted round-trip haulage. Serve them one at a time (capture then
// immediately release), chained in capture-coordinate order. No batching.
#include <bits/stdc++.h>
using namespace std;

static long long manh(long long ax,long long ay,long long bx,long long by){
    return llabs(ax-bx)+llabs(ay-by);
}

int main(){
    int P; long long Q;
    if(!(cin>>P>>Q)) return 0;
    long long x0,y0; cin>>x0>>y0;
    vector<long long> px(P),py(P),dx(P),dy(P),q(P),w(P);
    for(int i=0;i<P;i++) cin>>px[i]>>py[i]>>dx[i]>>dy[i]>>q[i]>>w[i];

    vector<int> served;
    for(int i=0;i<P;i++){
        long long solo = manh(x0,y0,px[i],py[i])*1
                       + manh(px[i],py[i],dx[i],dy[i])*(1+q[i])
                       + manh(dx[i],dy[i],x0,y0)*1;
        if(w[i] > solo) served.push_back(i);
    }
    // chain in capture-coordinate order (cheap deterministic ordering)
    sort(served.begin(), served.end(), [&](int a,int b){
        if(px[a]!=px[b]) return px[a]<px[b];
        return py[a]<py[b];
    });

    printf("%d\n", (int)served.size()*2);
    for(int idx : served){
        printf("0 %d\n", idx+1);
        printf("1 %d\n", idx+1);
    }
    return 0;
}
