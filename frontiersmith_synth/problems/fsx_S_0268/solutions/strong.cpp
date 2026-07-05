// TIER: strong
// Profitable-set selection + load-aware nearest-neighbour event interleaving,
// then a drop/add local search that reroutes and keeps changes only when the
// realized objective improves. Beats greedy by batching pickups (amortizing legs
// under the load-weighted fuel model) and by reoptimizing the served set.
#include <bits/stdc++.h>
using namespace std;
static long long manh(long long x1,long long y1,long long x2,long long y2){
    return llabs(x1-x2)+llabs(y1-y2);
}
int P; long long Q,K,X0,Y0;
vector<long long> ax,ay,bx,by,mm,cc,ww;

// route a given selected set with load-aware nearest-neighbour; return events + fuel
long long routeFuel(const vector<char>& sel, vector<pair<int,int>>* out){
    vector<char> picked(P,0), deliv(P,0);
    long long load=0, fuel=0, cx=X0, cy=Y0;
    int remaining=0;
    for(int i=0;i<P;i++) if(sel[i]) remaining++;
    remaining*=2; // pickup + delivery per selected
    if(out) out->clear();
    while(remaining>0){
        long long best=LLONG_MAX; int bt=-1,bi=-1;
        for(int i=0;i<P;i++){
            if(!sel[i]) continue;
            if(!picked[i] && load+mm[i]<=Q){
                long long inc = manh(cx,cy,ax[i],ay[i])*(K+load);
                if(inc<best || (inc==best && (bt!=0 || i< bi))){ best=inc; bt=0; bi=i; }
            }
            if(picked[i] && !deliv[i]){
                long long inc = manh(cx,cy,bx[i],by[i])*(K+load);
                if(inc<best || (inc==best && (bt==1 && i<bi) )){ best=inc; bt=1; bi=i; }
            }
        }
        if(bt<0) break; // should not happen (m<=Q guarantees progress)
        if(bt==0){ picked[bi]=1; fuel+=manh(cx,cy,ax[bi],ay[bi])*(K+load); load+=mm[bi]; cx=ax[bi]; cy=ay[bi]; }
        else { deliv[bi]=1; fuel+=manh(cx,cy,bx[bi],by[bi])*(K+load); load-=mm[bi]; cx=bx[bi]; cy=by[bi]; }
        if(out) out->push_back({bt,bi});
        remaining--;
    }
    fuel += manh(cx,cy,X0,Y0)*(K+load); // load should be 0 here
    return fuel;
}

long long objOf(const vector<char>& sel){
    long long fuel = routeFuel(sel, nullptr);
    long long F = fuel;
    for(int i=0;i<P;i++){ if(sel[i]) F+=cc[i]; else F+=ww[i]; }
    return F;
}

int main(){
    if(scanf("%d %lld %lld",&P,&Q,&K)!=3) return 0;
    scanf("%lld %lld",&X0,&Y0);
    ax.assign(P,0);ay.assign(P,0);bx.assign(P,0);by.assign(P,0);mm.assign(P,0);cc.assign(P,0);ww.assign(P,0);
    for(int i=0;i<P;i++)
        scanf("%lld %lld %lld %lld %lld %lld %lld",&ax[i],&ay[i],&bx[i],&by[i],&mm[i],&cc[i],&ww[i]);

    // initial selection: solo-benefit
    vector<char> sel(P,0);
    for(int i=0;i<P;i++){
        long long solo = manh(X0,Y0,ax[i],ay[i])*K
                       + manh(ax[i],ay[i],bx[i],by[i])*(K+mm[i])
                       + manh(bx[i],by[i],X0,Y0)*K + cc[i];
        if(solo < ww[i]) sel[i]=1;
    }

    long long cur = objOf(sel);
    // local search: try single flips (drop or add) keeping best improving move
    for(int sweep=0; sweep<8; sweep++){
        bool improved=false;
        long long bestGain=0; int bestIdx=-1;
        for(int i=0;i<P;i++){
            sel[i]^=1;
            long long v = objOf(sel);
            sel[i]^=1;
            long long gain = cur - v;
            if(gain>bestGain){ bestGain=gain; bestIdx=i; }
        }
        if(bestIdx>=0){
            sel[bestIdx]^=1;
            cur -= bestGain;
            improved=true;
        }
        if(!improved) break;
    }

    vector<pair<int,int>> ev;
    routeFuel(sel, &ev);
    printf("%d\n",(int)ev.size());
    for(auto&e:ev) printf("%d %d\n", e.first, e.second+1);
    return 0;
}
