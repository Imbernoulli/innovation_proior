// TIER: strong
// Cheapest-insertion with precedence: keep a single depot-to-depot route; repeatedly try to
// insert each unserved task's pickup and delivery (pickup earlier) at the cheapest positions,
// accepting whenever the insertion cost is below its penalty. Multiple passes let shared travel
// pull in tasks a standalone estimate would reject.
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

    // route stored as list of (task,type). depot is implicit at both ends.
    vector<pair<int,int>> route; // (task, type 0/1)
    vector<char> served(m,0);

    auto nodeX=[&](int idx)->long long{
        if(idx<0||idx>=(int)route.size()) return Xd;
        return route[idx].second==0?px[route[idx].first]:dx[route[idx].first];
    };
    auto nodeY=[&](int idx)->long long{
        if(idx<0||idx>=(int)route.size()) return Yd;
        return route[idx].second==0?py[route[idx].first]:dy[route[idx].first];
    };
    // coordinate of gap-neighbour: gap g sits between element g-1 and g;
    // A(g) = coord of element index g-1 (depot if g==0), B(g)=coord of element index g (depot if g==L)
    auto coordAt=[&](int elem, long long&X, long long&Y){
        if(elem<0){X=Xd;Y=Yd;} else if(elem>=(int)route.size()){X=Xd;Y=Yd;}
        else {X=nodeX(elem);Y=nodeY(elem);}
    };

    // sort tasks by penalty desc for insertion priority
    vector<int> ord(m); iota(ord.begin(),ord.end(),0);
    sort(ord.begin(),ord.end(),[&](int a,int b){return w[a]>w[b];});

    for(int pass=0; pass<3; pass++){
        bool changed=false;
        for(int j: ord){
            if(served[j]) continue;
            int L=(int)route.size();
            // best insertion of P at gap g1 (0..L) and D at gap g2 (g1..L)
            long long bestDelta=LLONG_MAX; int bg1=-1,bg2=-1; bool bestSame=false;
            for(int g1=0; g1<=L; g1++){
                long long ax,ay,bx,by;
                coordAt(g1-1,ax,ay); coordAt(g1,bx,by);
                long long baseEdge=man(ax,ay,bx,by);
                long long insP = man(ax,ay,px[j],py[j])+man(px[j],py[j],bx,by)-baseEdge;
                // same-gap: P then D in this gap
                {
                    long long same = man(ax,ay,px[j],py[j])+man(px[j],py[j],dx[j],dy[j])
                                     +man(dx[j],dy[j],bx,by)-baseEdge;
                    if(same<bestDelta){bestDelta=same;bg1=g1;bg2=g1;bestSame=true;}
                }
                // different gaps: P at g1, D at g2>g1 (positions in the ORIGINAL route)
                for(int g2=g1+1; g2<=L; g2++){
                    long long cx,cy,dxx,dyy;
                    coordAt(g2-1,cx,cy); coordAt(g2,dxx,dyy);
                    long long baseEdge2=man(cx,cy,dxx,dyy);
                    long long insD = man(cx,cy,dx[j],dy[j])+man(dx[j],dy[j],dxx,dyy)-baseEdge2;
                    long long delta = insP + insD;
                    if(delta<bestDelta){bestDelta=delta;bg1=g1;bg2=g2;bestSame=false;}
                }
            }
            if(bg1<0) continue;
            if(bestDelta < w[j]){ // beneficial: travel added < penalty saved
                if(bestSame){
                    route.insert(route.begin()+bg1, make_pair(j,1));
                    route.insert(route.begin()+bg1, make_pair(j,0));
                } else {
                    // insert D first at g2 (higher index) then P at g1 so indices stay valid
                    route.insert(route.begin()+bg2, make_pair(j,1));
                    route.insert(route.begin()+bg1, make_pair(j,0));
                }
                served[j]=1; changed=true;
            }
        }
        if(!changed) break;
    }

    printf("%d\n", (int)route.size());
    for(auto&pr: route) printf("%d %d\n", pr.first+1, pr.second);
    return 0;
}
