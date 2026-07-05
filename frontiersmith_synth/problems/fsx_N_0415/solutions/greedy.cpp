// TIER: greedy
// Greedy cover (hubs first) then a ONE-PASS farthest-point fill over a grid lattice +
// corners of the whole basin: spare pods spread into the EMPTY regions the demand-only
// baseline cannot reach. Maximizes dispersion; ignores the resonance bonus.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static inline ll L1(ll ax,ll ay,ll bx,ll by){ return llabs(ax-bx)+llabs(ay-by); }

int main(){
    int W,H,m,R,D,a,b,K,r0;
    if(scanf("%d %d %d %d %d %d %d %d %d",&W,&H,&m,&R,&D,&a,&b,&K,&r0)!=9) return 0;
    vector<int> dx(D),dy(D);
    for(int i=0;i<D;i++) scanf("%d %d",&dx[i],&dy[i]);

    vector<ll> dmind(D,LLONG_MAX);
    vector<char> used(D,0);
    vector<int> ox,oy; ox.reserve(m); oy.reserve(m);
    int placed=0;
    unordered_set<ll> occ; occ.reserve(m*2+16);
    auto put=[&](int x,int y){
        ox.push_back(x); oy.push_back(y); placed++; occ.insert((ll)x*(ll)H+y);
        for(int j=0;j<D;j++){ ll dd=L1(x,y,dx[j],dy[j]); if(dd<dmind[j])dmind[j]=dd; }
    };
    for(int i=0;i<D && placed<m;i++) if(dmind[i]>R){ used[i]=1; put(dx[i],dy[i]); }

    // candidate lattice over the basin (dense enough to supply every spare)
    int NG=max(48,(int)ceil(sqrt(2.0*m))+3); if(NG>150)NG=150;
    vector<int> cx,cy;
    for(int gi=0;gi<=NG;gi++) for(int gj=0;gj<=NG;gj++)
        cx.push_back((int)((ll)gi*(W-1)/NG)), cy.push_back((int)((ll)gj*(H-1)/NG));
    int NC=(int)cx.size();
    vector<ll> cmind(NC,LLONG_MAX);
    for(int c=0;c<NC;c++){
        ll key=(ll)cx[c]*(ll)H+cy[c];
        if(occ.count(key)){ cmind[c]=-1; continue; }
        ll best=LLONG_MAX;
        for(int i=0;i<placed;i++){ ll dd=L1(cx[c],cy[c],ox[i],oy[i]); if(dd<best)best=dd; }
        cmind[c]=best;
    }
    while(placed<m){
        int best=-1; ll bv=-1;
        for(int c=0;c<NC;c++) if(cmind[c]>bv){bv=cmind[c];best=c;}
        if(best<0||bv<0) break;
        int x=cx[best],y=cy[best];
        cmind[best]=-1; occ.insert((ll)x*(ll)H+y); put(x,y);
        for(int c=0;c<NC;c++){ if(cmind[c]<0)continue; ll dd=L1(cx[c],cy[c],x,y); if(dd<cmind[c])cmind[c]=dd; }
    }
    for(int j=0;j<D && placed<m;j++){ ll key=(ll)dx[j]*(ll)H+dy[j]; if(occ.count(key))continue; used[j]=1; put(dx[j],dy[j]); }

    for(int i=0;i<placed;i++) printf("%d %d\n",ox[i],oy[i]);
    return 0;
}
