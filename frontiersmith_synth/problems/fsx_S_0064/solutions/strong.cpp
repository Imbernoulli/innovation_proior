// TIER: strong
// Cheapest-insertion selection + load-aware adjacent-swap local search.
// 1. Sort requests by profitability (penalty minus estimated solo haulage).
// 2. Insert each profitable request as a [capture,release] block at the gap of
//    the shared tour with least incremental load-weighted haulage; only serve it
//    when that incremental cost is below its skip penalty. Blocks keep each
//    request's carry distance minimal (release right after capture), which is
//    strong under the load-weighted metric.
// 3. Adjacent-swap local search (O(1) delta, precedence + capacity safe) to
//    interleave neighbouring requests where that shortens the tour.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll manh(ll ax,ll ay,ll bx,ll by){ return llabs(ax-bx)+llabs(ay-by); }

int P; ll Q, DX0, DY0;
vector<ll> px,py,dx,dy,q,w;

struct Ev { int type; int req; }; // type 0=capture,1=release

// coordinates of an event
static inline ll ex(const Ev&e){ return e.type==0?px[e.req]:dx[e.req]; }
static inline ll ey(const Ev&e){ return e.type==0?py[e.req]:dy[e.req]; }
static inline ll delta(const Ev&e){ return e.type==0? q[e.req] : -q[e.req]; }

int main(){
    if(!(cin>>P>>Q)) return 0;
    cin>>DX0>>DY0;
    px.resize(P);py.resize(P);dx.resize(P);dy.resize(P);q.resize(P);w.resize(P);
    for(int i=0;i<P;i++) cin>>px[i]>>py[i]>>dx[i]>>dy[i]>>q[i]>>w[i];

    // profitability ordering
    vector<int> order(P);
    for(int i=0;i<P;i++) order[i]=i;
    vector<ll> solo(P);
    for(int i=0;i<P;i++){
        solo[i]= manh(DX0,DY0,px[i],py[i])*1
               + manh(px[i],py[i],dx[i],dy[i])*(1+q[i])
               + manh(dx[i],dy[i],DX0,DY0)*1;
    }
    sort(order.begin(),order.end(),[&](int a,int b){
        return (w[a]-solo[a]) > (w[b]-solo[b]);
    });

    vector<Ev> route; // current tour (excluding depot endpoints)

    // build point + preload arrays for current route
    auto buildAux=[&](vector<ll>&PX,vector<ll>&PY,vector<ll>&L){
        int M=route.size();
        PX.assign(M+2,0);PY.assign(M+2,0);L.assign(M+1,0);
        PX[0]=DX0;PY[0]=DY0;PX[M+1]=DX0;PY[M+1]=DY0;
        ll load=0;
        for(int k=0;k<M;k++){
            L[k]=load;                 // load carried on leg into event k
            PX[k+1]=ex(route[k]);PY[k+1]=ey(route[k]);
            load+=delta(route[k]);
        }
        L[M]=load; // should be 0
    };

    for(int oi=0; oi<P; oi++){
        int r=order[oi];
        int M=route.size();
        vector<ll> PX,PY,L; buildAux(PX,PY,L);
        // try inserting block [capture r, release r] into each gap g (0..M)
        // gap g sits between point index g and g+1; carried load there = L[g].
        // capture at load L[g], release at load L[g]+q[r] (must be <= Q).
        ll bestDelta = LLONG_MAX; int bestG=-1;
        ll cx=px[r],cy=py[r],rx=dx[r],ry=dy[r], qr=q[r];
        for(int g=0; g<=M; g++){
            ll lb=L[g];
            if(lb+qr>Q) continue; // capacity
            ll ax=PX[g],ay=PY[g], bx=PX[g+1],by=PY[g+1];
            ll oldLeg = manh(ax,ay,bx,by)*(1+lb);
            ll newLeg = manh(ax,ay,cx,cy)*(1+lb)
                      + manh(cx,cy,rx,ry)*(1+lb+qr)
                      + manh(rx,ry,bx,by)*(1+lb);
            ll d = newLeg-oldLeg;
            if(d<bestDelta){ bestDelta=d; bestG=g; }
        }
        if(bestG>=0 && bestDelta < w[r]){
            route.insert(route.begin()+bestG, Ev{1,r}); // release
            route.insert(route.begin()+bestG, Ev{0,r}); // capture (before release)
        }
    }

    // ---- adjacent-swap local search ----
    {
        int M=route.size();
        vector<ll> PX,PY,L; buildAux(PX,PY,L);
        bool improved=true; int passes=0;
        while(improved && passes<60){
            improved=false; passes++;
            for(int e=0;e+1<M;e++){
                Ev A=route[e], B=route[e+1];
                if(A.type==0 && B.type==1 && A.req==B.req) continue; // precedence
                ll Le=L[e];
                ll dA=delta(A), dB=delta(B);
                if(Le+dB>Q) continue;            // load after B
                if(Le+dB+dA>Q) continue;         // load after A (net)
                ll prevx=PX[e], prevy=PY[e];
                ll nextx=PX[e+2], nexty=PY[e+2];
                ll ax=ex(A),ay=ey(A), bx=ex(B),by=ey(B);
                ll oldC = manh(prevx,prevy,ax,ay)*(1+Le)
                        + manh(ax,ay,bx,by)*(1+Le+dA)
                        + manh(bx,by,nextx,nexty)*(1+Le+dA+dB);
                ll newC = manh(prevx,prevy,bx,by)*(1+Le)
                        + manh(bx,by,ax,ay)*(1+Le+dB)
                        + manh(ax,ay,nextx,nexty)*(1+Le+dA+dB);
                if(newC<oldC){
                    swap(route[e],route[e+1]);
                    // update aux locally
                    swap(PX[e+1],PX[e+2]); swap(PY[e+1],PY[e+2]);
                    L[e+1]=Le+dB;
                    improved=true;
                }
            }
        }
    }

    // ---- output ----
    printf("%d\n",(int)route.size());
    for(auto&e:route) printf("%d %d\n", e.type, e.req+1);
    return 0;
}
