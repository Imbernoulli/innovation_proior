// TIER: strong
// Nearest-feasible-event construction with capacity batching: at each step move
// to the nearest of {a pickup whose demand still fits the rack, a delivery of an
// on-board cartridge}. This interleaves pickups and deliveries so nearby caches
// and stations are chained together. Also computes the sequential-nearest-request
// tour and outputs whichever realized tour is shorter, so it never loses to the
// no-batching greedy.
#include <bits/stdc++.h>
using namespace std;

static inline long long edist(long long x1,long long y1,long long x2,long long y2){
    double dx=(double)(x1-x2), dy=(double)(y1-y2);
    return (long long)llround(sqrt(dx*dx+dy*dy));
}

struct Inst {
    int N,Q; long long x0,y0;
    vector<long long> px,py,dx,dy; vector<int> q;
};

static long long tourLen(const Inst&I, const vector<pair<int,int>>&ev){
    long long F=0, cx=I.x0, cy=I.y0;
    for(auto&e:ev){
        long long nx = (e.first==0)? I.px[e.second] : I.dx[e.second];
        long long ny = (e.first==0)? I.py[e.second] : I.dy[e.second];
        F+=edist(cx,cy,nx,ny); cx=nx; cy=ny;
    }
    F+=edist(cx,cy,I.x0,I.y0);
    return F;
}

// nearest-feasible-event with batching
static vector<pair<int,int>> buildBatched(const Inst&I){
    int N=I.N;
    vector<char> picked(N+1,0), delivered(N+1,0);
    vector<pair<int,int>> out; out.reserve(2*N);
    long long cx=I.x0, cy=I.y0, load=0;
    int served=0;
    while(served < N){
        int bt=-1,bi=-1; long long bd=LLONG_MAX;
        for(int i=1;i<=N;i++){
            if(!picked[i] && load+I.q[i]<=I.Q){
                long long d=edist(cx,cy,I.px[i],I.py[i]);
                if(d<bd){bd=d;bt=0;bi=i;}
            }
            if(picked[i] && !delivered[i]){
                long long d=edist(cx,cy,I.dx[i],I.dy[i]);
                if(d<bd){bd=d;bt=1;bi=i;}
            }
        }
        if(bi==-1){
            // rack full and nothing deliverable should be impossible; fall back
            // by delivering any on-board job
            for(int i=1;i<=N;i++) if(picked[i]&&!delivered[i]){bt=1;bi=i;break;}
        }
        if(bt==0){
            picked[bi]=1; load+=I.q[bi];
            out.push_back({0,bi}); cx=I.px[bi]; cy=I.py[bi];
        } else {
            delivered[bi]=1; load-=I.q[bi]; served++;
            out.push_back({1,bi}); cx=I.dx[bi]; cy=I.dy[bi];
        }
    }
    return out;
}

// sequential-nearest-request (no batching)
static vector<pair<int,int>> buildSeq(const Inst&I){
    int N=I.N;
    vector<char> done(N+1,0);
    vector<pair<int,int>> out; out.reserve(2*N);
    long long cx=I.x0, cy=I.y0;
    for(int step=0;step<N;step++){
        int best=-1; long long bd=LLONG_MAX;
        for(int i=1;i<=N;i++) if(!done[i]){
            long long d=edist(cx,cy,I.px[i],I.py[i]);
            if(d<bd){bd=d;best=i;}
        }
        done[best]=1;
        out.push_back({0,best}); out.push_back({1,best});
        cx=I.dx[best]; cy=I.dy[best];
    }
    return out;
}

int main(){
    Inst I;
    scanf("%d %d",&I.N,&I.Q);
    scanf("%lld %lld",&I.x0,&I.y0);
    int N=I.N;
    I.px.assign(N+1,0);I.py.assign(N+1,0);I.dx.assign(N+1,0);I.dy.assign(N+1,0);I.q.assign(N+1,0);
    for(int i=1;i<=N;i++)
        scanf("%lld %lld %lld %lld %d",&I.px[i],&I.py[i],&I.dx[i],&I.dy[i],&I.q[i]);

    auto a=buildBatched(I);
    auto b=buildSeq(I);
    auto&best = (tourLen(I,a) <= tourLen(I,b)) ? a : b;

    printf("%d\n",(int)best.size());
    for(auto&e:best) printf("%d %d\n",e.first,e.second);
    return 0;
}
