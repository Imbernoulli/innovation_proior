// TIER: greedy
// Serve every task whose penalty covers its standalone round-trip cost; chain the
// selected tasks by nearest-neighbour on pickups, emitting each as P then D consecutively.
#include <bits/stdc++.h>
using namespace std;
static inline long long man(long long ax,long long ay,long long bx,long long by){
    return llabs(ax-bx)+llabs(ay-by);
}
int main(){
    long long Xd,Yd; int m;
    if(!(cin>>Xd>>Yd>>m)) return 0;
    vector<long long> px(m),py(m),dx(m),dy(m),w(m);
    for(int j=0;j<m;j++) cin>>px[j]>>py[j]>>dx[j]>>dy[j]>>w[j];

    vector<int> sel;
    for(int j=0;j<m;j++){
        long long base = man(Xd,Yd,px[j],py[j])+man(px[j],py[j],dx[j],dy[j])+man(dx[j],dy[j],Xd,Yd);
        if(w[j] >= base) sel.push_back(j);
    }

    // nearest-neighbour ordering on pickups starting from depot
    vector<char> used(sel.size(),0);
    vector<int> ord;
    long long cx=Xd, cy=Yd;
    for(size_t step=0; step<sel.size(); step++){
        int best=-1; long long bd=LLONG_MAX;
        for(size_t i=0;i<sel.size();i++){
            if(used[i]) continue;
            long long d=man(cx,cy,px[sel[i]],py[sel[i]]);
            if(d<bd){bd=d;best=(int)i;}
        }
        used[best]=1; ord.push_back(sel[best]);
        cx=dx[sel[best]]; cy=dy[sel[best]]; // move to its delivery (task done consecutively)
    }

    printf("%d\n", (int)ord.size()*2);
    for(int j: ord){
        printf("%d 0\n", j+1);
        printf("%d 1\n", j+1);
    }
    return 0;
}
