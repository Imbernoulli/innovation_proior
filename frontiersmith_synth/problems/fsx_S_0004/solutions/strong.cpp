// TIER: strong
// Cheapest-insertion selection (respecting precedence + capacity) followed by a
// feasibility-checked 2-opt / or-opt local search on the event sequence. Requests
// are admitted only when their best insertion cost is below their skip-penalty, so
// batching interleaved pickups/deliveries lets more requests become profitable than
// the standalone greedy admits, and the local search tightens the tour.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static inline ll md(ll ax, ll ay, ll bx, ll by){ return llabs(ax-bx)+llabs(ay-by); }

int P, Q; ll DX0, DY0;
vector<ll> px,py,dx,dy,q,w;

// event = (type, req): type 0 pickup, 1 delivery
static inline ll evx(const pair<int,int>&e){ return e.first==0?px[e.second]:dx[e.second]; }
static inline ll evy(const pair<int,int>&e){ return e.first==0?py[e.second]:dy[e.second]; }

// feasibility + tour distance for an event sequence
bool feasDist(const vector<pair<int,int>>& ev, ll& outDist){
    static vector<int> st; st.assign(P+1,0);
    ll load=0, dist=0, cx=DX0, cy=DY0;
    for(const auto& e : ev){
        int t=e.first, i=e.second;
        if(t==0){
            if(st[i]!=0) return false;
            st[i]=1; load+=q[i];
            if(load>Q) return false;
            dist += md(cx,cy,px[i],py[i]); cx=px[i]; cy=py[i];
        } else {
            if(st[i]!=1) return false;
            st[i]=2; load-=q[i];
            dist += md(cx,cy,dx[i],dy[i]); cx=dx[i]; cy=dy[i];
        }
    }
    if(!ev.empty()) dist += md(cx,cy,DX0,DY0);
    outDist=dist;
    return true;
}

int main(){
    if(scanf("%d %d",&P,&Q)!=2) return 0;
    scanf("%lld %lld",&DX0,&DY0);
    px.assign(P+1,0);py.assign(P+1,0);dx.assign(P+1,0);dy.assign(P+1,0);
    q.assign(P+1,0);w.assign(P+1,0);
    for(int i=1;i<=P;i++)
        scanf("%lld %lld %lld %lld %lld %lld",&px[i],&py[i],&dx[i],&dy[i],&q[i],&w[i]);

    // order requests by standalone benefit (heuristic admission order)
    vector<int> order;
    for(int i=1;i<=P;i++) order.push_back(i);
    sort(order.begin(), order.end(), [&](int a,int b){
        ll sa = md(DX0,DY0,px[a],py[a])+md(px[a],py[a],dx[a],dy[a])+md(dx[a],dy[a],DX0,DY0);
        ll sb = md(DX0,DY0,px[b],py[b])+md(px[b],py[b],dx[b],dy[b])+md(dx[b],dy[b],DX0,DY0);
        return (w[a]-sa) > (w[b]-sb);
    });

    vector<pair<int,int>> seq;   // current event sequence
    ll curDist = 0;

    // cheapest-insertion admission
    for(int i : order){
        int m = (int)seq.size();
        ll best = LLONG_MAX;
        int bestA=-1, bestB=-1;
        vector<pair<int,int>> cand;
        cand.reserve(m+2);
        for(int a=0; a<=m; a++){
            for(int b=a; b<=m; b++){
                // insert pickup at gap a, delivery at gap b (delivery after pickup)
                cand.clear();
                for(int k=0;k<a;k++) cand.push_back(seq[k]);
                cand.push_back({0,i});
                for(int k=a;k<b;k++) cand.push_back(seq[k]);
                cand.push_back({1,i});
                for(int k=b;k<m;k++) cand.push_back(seq[k]);
                ll d;
                if(feasDist(cand,d)){
                    ll added = d - curDist;
                    if(added < best){ best=added; bestA=a; bestB=b; }
                }
            }
        }
        if(bestA>=0 && best < w[i]){
            // commit best insertion
            vector<pair<int,int>> ns;
            ns.reserve(m+2);
            for(int k=0;k<bestA;k++) ns.push_back(seq[k]);
            ns.push_back({0,i});
            for(int k=bestA;k<bestB;k++) ns.push_back(seq[k]);
            ns.push_back({1,i});
            for(int k=bestB;k<m;k++) ns.push_back(seq[k]);
            seq.swap(ns);
            feasDist(seq, curDist);
        }
    }

    // 2-opt (segment reversal) + or-opt (single move), feasibility-checked
    bool improved = true;
    int guard = 0;
    while(improved && guard < 40){
        improved = false; guard++;
        int m = (int)seq.size();
        // 2-opt
        for(int a=0;a<m;a++){
            for(int b=a+1;b<m;b++){
                reverse(seq.begin()+a, seq.begin()+b+1);
                ll d;
                if(feasDist(seq,d) && d < curDist){
                    curDist = d; improved = true;
                } else {
                    reverse(seq.begin()+a, seq.begin()+b+1);
                }
            }
        }
        // or-opt: move one event to another position
        m = (int)seq.size();
        for(int a=0;a<m && !improved;a++){
            auto moved = seq[a];
            vector<pair<int,int>> reduced;
            reduced.reserve(m-1);
            for(int k=0;k<m;k++) if(k!=a) reduced.push_back(seq[k]);
            for(int b=0;b<=(int)reduced.size();b++){
                vector<pair<int,int>> ns;
                ns.reserve(m);
                for(int k=0;k<b;k++) ns.push_back(reduced[k]);
                ns.push_back(moved);
                for(int k=b;k<(int)reduced.size();k++) ns.push_back(reduced[k]);
                ll d;
                if(feasDist(ns,d) && d < curDist){
                    seq.swap(ns); curDist=d; improved=true; break;
                }
            }
        }
    }

    printf("%d\n", (int)seq.size());
    for(auto &e : seq) printf("%d %d\n", e.first, e.second);
    return 0;
}
