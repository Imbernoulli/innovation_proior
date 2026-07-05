// TIER: strong
// Selection + nearest-neighbor stack construction with a local improvement pass.
// 1) Served set: start from profitable-by-solo requests.
// 2) Route the served set with a nearest-neighbor STACK heuristic: at each step
//    take the cheapest legal move -- pick up any not-yet-picked served request
//    while the rack has room, or deliver the current top tote. This batches
//    nearby shelves under the LIFO discipline.
// 3) Also build the simple depth-1 chain (like greedy) and keep whichever tour
//    has smaller objective F; then try dropping individually-unprofitable served
//    requests. Output is always LIFO-feasible.
#include <bits/stdc++.h>
using namespace std;

static int P, H;
static long long X0, Y0;
static vector<long long> px, py, dx, dy, w;

static inline long long manh(long long ax,long long ay,long long bx,long long by){
    return llabs(ax-bx)+llabs(ay-by);
}

// evaluate objective F of an event list (assumed feasible); returns travel+penalty
static long long evalF(const vector<pair<int,int>>& ev) {
    long long D = 0, cx = X0, cy = Y0;
    vector<char> served(P+1,0), pick(P+1,0);
    for (auto& e : ev) {
        int t = e.first, i = e.second;
        long long nx, ny;
        if (t==0) { nx=px[i]; ny=py[i]; pick[i]=1; }
        else      { nx=dx[i]; ny=dy[i]; served[i]=1; }
        D += manh(cx,cy,nx,ny); cx=nx; cy=ny;
    }
    if (!ev.empty()) D += manh(cx,cy,X0,Y0);
    long long pen = 0;
    for (int i=1;i<=P;i++) if(!served[i]) pen += w[i];
    return D + pen;
}

// nearest-neighbor stack tour over a served set
static vector<pair<int,int>> nnStack(const vector<int>& S) {
    vector<pair<int,int>> ev;
    vector<char> picked(P+1,0);
    vector<int> stk;
    long long cx=X0, cy=Y0;
    int remaining = (int)S.size();
    // set for quick iteration
    while (remaining > 0 || !stk.empty()) {
        long long best = LLONG_MAX; int bestType=-1, bestId=-1;
        if ((int)stk.size() < H) {
            for (int r : S) if (!picked[r]) {
                long long d = manh(cx,cy,px[r],py[r]);
                if (d < best) { best=d; bestType=0; bestId=r; }
            }
        }
        if (!stk.empty()) {
            int top = stk.back();
            long long d = manh(cx,cy,dx[top],dy[top]);
            if (d < best) { best=d; bestType=1; bestId=top; }
        }
        if (bestType==0) {
            picked[bestId]=1; stk.push_back(bestId); remaining--;
            cx=px[bestId]; cy=py[bestId];
            ev.push_back({0,bestId});
        } else {
            stk.pop_back();
            cx=dx[bestId]; cy=dy[bestId];
            ev.push_back({1,bestId});
        }
    }
    return ev;
}

// depth-1 chain, shelves visited nearest-neighbor
static vector<pair<int,int>> chain(const vector<int>& S) {
    vector<pair<int,int>> ev;
    vector<char> used(P+1,0);
    long long cx=X0, cy=Y0;
    for (size_t k=0;k<S.size();k++){
        int best=-1; long long bd=LLONG_MAX;
        for (int r : S) if(!used[r]){
            long long d=manh(cx,cy,px[r],py[r]);
            if(d<bd){bd=d;best=r;}
        }
        used[best]=1; ev.push_back({0,best}); ev.push_back({1,best});
        cx=dx[best]; cy=dy[best];
    }
    return ev;
}

int main(){
    if(!(cin>>P>>H)) return 0;
    cin>>X0>>Y0;
    px.assign(P+1,0);py.assign(P+1,0);dx.assign(P+1,0);dy.assign(P+1,0);w.assign(P+1,0);
    for(int i=1;i<=P;i++) cin>>px[i]>>py[i]>>dx[i]>>dy[i]>>w[i];

    vector<int> S;
    for(int i=1;i<=P;i++){
        long long solo = manh(X0,Y0,px[i],py[i])+manh(px[i],py[i],dx[i],dy[i])
                       + manh(dx[i],dy[i],X0,Y0);
        if (w[i] > solo) S.push_back(i);
    }

    // candidate tours; keep best
    vector<pair<int,int>> bestEv;
    long long bestF = LLONG_MAX;
    auto consider = [&](const vector<pair<int,int>>& ev){
        long long f = evalF(ev);
        if (f < bestF){ bestF=f; bestEv=ev; }
    };
    consider(vector<pair<int,int>>{});      // serve nobody
    if(!S.empty()){
        consider(chain(S));
        consider(nnStack(S));
    }

    // local drop pass: remove one served request at a time if it lowers F
    // (re-route the reduced set with nnStack). Bounded iterations for speed.
    vector<int> cur = S;
    bool improved = true; int iters=0;
    while(improved && iters<200 && (int)cur.size()>0){
        improved=false; iters++;
        // baseline F of current best route on cur
        vector<pair<int,int>> curEv = nnStack(cur);
        long long curF = evalF(curEv);
        consider(curEv);
        int dropIdx=-1; long long bestDropF=curF;
        for(size_t j=0;j<cur.size();j++){
            vector<int> t; t.reserve(cur.size()-1);
            for(size_t k=0;k<cur.size();k++) if(k!=j) t.push_back(cur[k]);
            long long f = evalF(nnStack(t));
            if(f<bestDropF){bestDropF=f;dropIdx=(int)j;}
        }
        if(dropIdx>=0){
            cur.erase(cur.begin()+dropIdx);
            improved=true;
        }
    }
    if(!cur.empty()) consider(nnStack(cur));

    printf("%d\n",(int)bestEv.size());
    for(auto&e:bestEv) printf("%d %d\n", e.first, e.second+0); // ids already 1-based
    return 0;
}
