// TIER: greedy
// Sequential-nearest-request heuristic: repeatedly pick the unserved job whose
// pickup cache is nearest to the current position, then fully serve it (pickup
// then immediate delivery) before choosing the next. Shortens travel versus the
// index-order serialized plan but never batches under capacity.
#include <bits/stdc++.h>
using namespace std;

static inline long long edist(long long x1,long long y1,long long x2,long long y2){
    double dx=(double)(x1-x2), dy=(double)(y1-y2);
    return (long long)llround(sqrt(dx*dx+dy*dy));
}

int main(){
    int N,Q; scanf("%d %d",&N,&Q);
    long long x0,y0; scanf("%lld %lld",&x0,&y0);
    vector<long long> px(N+1),py(N+1),dx(N+1),dy(N+1); vector<int> q(N+1);
    for(int i=1;i<=N;i++)
        scanf("%lld %lld %lld %lld %d",&px[i],&py[i],&dx[i],&dy[i],&q[i]);

    vector<char> done(N+1,0);
    vector<pair<int,int>> out;
    long long cx=x0, cy=y0;
    for(int step=0; step<N; step++){
        int best=-1; long long bd=LLONG_MAX;
        for(int i=1;i<=N;i++) if(!done[i]){
            long long d=edist(cx,cy,px[i],py[i]);
            if(d<bd){bd=d;best=i;}
        }
        done[best]=1;
        out.push_back({0,best});
        out.push_back({1,best});
        cx=dx[best]; cy=dy[best];
    }

    printf("%d\n",(int)out.size());
    for(auto&e:out) printf("%d %d\n",e.first,e.second);
    return 0;
}
