// Drone Courier under Time Windows -- heuristic solver (ale-03)
//
// Maximize the number of served requests by a single drone that starts and
// ends at a depot, where each request has a spatial location, a time window
// [r,d] (service must START within it; early arrival waits), a service time s,
// and the whole tour must return to the depot by horizon T.
//
// METHOD: greedy/regret INSERTION construction, then LARGE NEIGHBORHOOD SEARCH
// (ruin-and-recreate / LNS): repeatedly remove a batch of requests (worst-cost
// removal + a Shaw-style related removal) and greedily re-insert all currently
// unserved requests. Accept the new route iff it serves more (tie: less time).
//
// The lever that makes LNS affordable is an O(1) feasibility test per candidate
// insertion position, built on forward earliest-start times and a backward
// "max admissible push" (forward time slack). We never replay the whole route
// to test a single insertion.
//
#include <bits/stdc++.h>
using namespace std;

static const int L = 1000;

struct Req { long long x, y, r, d, s; };

static int N;
static long long T;
static vector<Req> req;   // index 0 = depot, 1..N = requests

// travel time = ceil(euclidean - 1e-9), identical to the scorer
static inline long long travel(long long ax, long long ay, long long bx, long long by){
    double dx = (double)(ax - bx), dy = (double)(ay - by);
    double dist = sqrt(dx*dx + dy*dy);
    long long c = (long long)ceil(dist - 1e-9);
    if (c < 0) c = 0;
    return c;
}
static inline long long dt(int a, int b){
    return travel(req[a].x, req[a].y, req[b].x, req[b].y);
}

// ---- route state ----------------------------------------------------------
// route: sequence of request ids (depot is implicit at both ends).
// For a route node at position p (0-based over the K served requests):
//   start[p]  = time service STARTS at route[p]   (>= r, <= d)
//   We also keep, going backward, the maximum value start[p] could take while
//   keeping everything after p (incl. return to depot) feasible: maxStart[p].
// An insertion of u between route[p-1] and route[p] is feasible iff:
//   (i)  the new node itself can be served: arrU <= d[u]
//   (ii) the push it imposes on route[p..] does not exceed the slack maxStart[p].
struct Route {
    vector<int> seq;            // served request ids in order
    vector<long long> start;    // start[i] for seq[i]
    vector<long long> maxStart; // latest feasible start[i]
    long long endTime = 0;      // arrival back at depot

    void recompute(){
        int K = seq.size();
        start.assign(K, 0);
        maxStart.assign(K, 0);
        // forward earliest starts
        long long t = 0; int prev = 0; // depot
        for(int i=0;i<K;i++){
            int u = seq[i];
            long long arr = t + dt(prev, u);
            long long st = max(arr, req[u].r);
            start[i] = st;
            t = st + req[u].s;
            prev = u;
        }
        endTime = t + dt(prev, 0); // back to depot
        // backward latest feasible starts
        // after last node we must reach depot by T.
        // For node i: maxStart[i] = min( d[seq[i]] ,
        //                                maxStart[i+1] - s[seq[i]] - dt(seq[i],seq[i+1]) )
        // last node: bounded by T via return to depot.
        for(int i=K-1;i>=0;i--){
            int u = seq[i];
            long long lim = req[u].d;
            if(i+1<K){
                int w = seq[i+1];
                lim = min(lim, maxStart[i+1] - req[u].s - dt(u, w));
            } else {
                // must return to depot by T
                lim = min(lim, T - req[u].s - dt(u, 0));
            }
            maxStart[i] = lim;
        }
    }

    bool empty_() const { return seq.empty(); }
};

// Try to find the best feasible insertion position for request u into r.
// Returns {feasible, position, addedTime}. addedTime is the increase in total
// time (used as a cheapness tie-break for greedy insertion).
struct InsRes { bool ok=false; int pos=-1; long long add=LLONG_MAX; };

static InsRes bestInsertion(const Route& R, int u){
    InsRes res;
    int K = R.seq.size();
    // walk every gap p in [0..K]; node before gap p is seq[p-1] (or depot),
    // node after gap p is seq[p] (or depot).
    long long tBefore;   // time we have finished service of node before gap
    int prevId;
    for(int p=0;p<=K;p++){
        if(p==0){ prevId=0; tBefore=0; }
        else { prevId = R.seq[p-1]; tBefore = R.start[p-1] + req[R.seq[p-1]].s; }
        long long arrU = tBefore + dt(prevId, u);
        long long startU = max(arrU, req[u].r);
        if(startU > req[u].d) {
            // arriving here already too late; later gaps depart even later, so
            // once we pass the window we can break for monotonic forward time...
            // but waiting can reset; forward time is non-decreasing, so startU
            // is non-decreasing in p once past release. Safe to continue though.
            continue;
        }
        long long finishU = startU + req[u].s;
        int nextId = (p<K)? R.seq[p] : 0;
        long long arrNext = finishU + dt(u, nextId);
        long long add; // increase in total tour time
        if(p<K){
            // pushed start of seq[p]:
            long long newStartNext = max(arrNext, req[R.seq[p]].r);
            // feasibility: newStartNext must not exceed maxStart[p]
            if(newStartNext > R.maxStart[p]) continue;
            // also node u itself fine (checked). compute added time vs old.
            long long oldStartNext = R.start[p];
            // The push amount on the suffix equals (newStartNext - oldStartNext)
            // but the tour-time increase is the new path minus old direct edge.
            long long oldEdge = (p==0? dt(0, R.seq[0]) : dt(R.seq[p-1], R.seq[p]));
            // Approx added time: detour cost. Use sum of new edges minus old edge
            // plus any extra wait. Cheaper-insertion heuristic only; feasibility
            // already guaranteed above.
            long long newEdges = dt(prevId,u) + dt(u,nextId);
            add = newEdges - oldEdge;
            (void)oldStartNext;
        } else {
            // inserting at the end: must still return to depot by T
            long long backArr = arrNext; // arrNext already = finishU + dt(u,0)
            if(backArr > T) continue;
            long long oldEdge = (K==0? 0 : dt(R.seq[K-1], 0));
            long long newEdges = dt(prevId,u) + dt(u,0);
            add = newEdges - oldEdge;
        }
        if(add < res.add){ res.ok=true; res.pos=p; res.add=add; }
    }
    return res;
}

static void doInsert(Route& R, int u, int pos){
    R.seq.insert(R.seq.begin()+pos, u);
    R.recompute();
}

// Greedy insertion of all requests in `pool` that are not yet served.
// Uses a regret-2 style choice: pick the request whose (secondBest - best)
// regret is largest among feasible ones, inserting at its best position.
static void greedyInsertAll(Route& R, vector<char>& served){
    while(true){
        int bestU=-1, bestPos=-1; long long bestKey=LLONG_MIN;
        long long bestAddForU=0;
        // For regret we need top-2 add costs per candidate. Recompute per loop
        // (route changed). N is small enough; LNS removes only a few per round
        // but the very first build inserts up to N -- still fine for these sizes.
        for(int u=1;u<=N;u++){
            if(served[u]) continue;
            // find best and second-best feasible insertion cost
            InsRes b1; long long secondAdd=LLONG_MAX; int b1pos=-1; long long b1add=LLONG_MAX;
            int K=R.seq.size();
            long long tBefore; int prevId;
            for(int p=0;p<=K;p++){
                if(p==0){ prevId=0; tBefore=0; }
                else { prevId=R.seq[p-1]; tBefore=R.start[p-1]+req[R.seq[p-1]].s; }
                long long arrU=tBefore+dt(prevId,u);
                long long startU=max(arrU,req[u].r);
                if(startU>req[u].d) continue;
                long long finishU=startU+req[u].s;
                int nextId=(p<K)?R.seq[p]:0;
                long long arrNext=finishU+dt(u,nextId);
                long long add;
                if(p<K){
                    long long newStartNext=max(arrNext,req[R.seq[p]].r);
                    if(newStartNext>R.maxStart[p]) continue;
                    long long oldEdge=(p==0?dt(0,R.seq[0]):dt(R.seq[p-1],R.seq[p]));
                    add=dt(prevId,u)+dt(u,nextId)-oldEdge;
                } else {
                    if(arrNext>T) continue;
                    long long oldEdge=(K==0?0:dt(R.seq[K-1],0));
                    add=dt(prevId,u)+dt(u,0)-oldEdge;
                }
                if(add<b1add){ secondAdd=b1add; b1add=add; b1pos=p; }
                else if(add<secondAdd){ secondAdd=add; }
            }
            if(b1pos<0) continue; // not insertable now
            long long regret;
            if(secondAdd==LLONG_MAX) regret = (long long)4e18; // only one slot: very urgent
            else regret = secondAdd - b1add;
            // tie-break toward cheaper insertion
            long long key = regret*1000000LL - b1add;
            if(key>bestKey){ bestKey=key; bestU=u; bestPos=b1pos; bestAddForU=b1add; }
            (void)bestAddForU; (void)b1;
        }
        if(bestU<0) break;
        doInsert(R, bestU, bestPos);
        served[bestU]=1;
    }
}

// xorshift RNG (deterministic)
static uint64_t rngState=88172645463325252ULL;
static inline uint64_t rnd(){ rngState^=rngState<<13; rngState^=rngState>>7; rngState^=rngState<<17; return rngState; }
static inline int rndi(int n){ return (int)(rnd()% (uint64_t)n); }

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if(!(cin>>N>>T)){ cout<<0<<"\n"; return 0; }
    req.assign(N+1, Req{});
    req[0] = Req{L/2, L/2, 0, 0, 0};
    for(int i=1;i<=N;i++){
        cin>>req[i].x>>req[i].y>>req[i].r>>req[i].d>>req[i].s;
    }

    auto t_start = chrono::steady_clock::now();
    auto elapsed_ms = [&](){
        return chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now()-t_start).count();
    };
    const long long TIME_LIMIT_MS = 1800;

    // ---- initial construction: regret insertion ----
    Route best;
    vector<char> served(N+1, 0);
    greedyInsertAll(best, served);
    best.recompute();

    // record best by (#served, -endTime)
    auto curServed = [&](const Route& R){ return (int)R.seq.size(); };
    int bestCount = curServed(best);

    // ---- LNS: ruin & recreate ----
    Route cur = best;
    vector<char> curServedFlag = served;

    while(elapsed_ms() < TIME_LIMIT_MS){
        Route trial = cur;
        vector<char> flag = curServedFlag;

        int K = trial.seq.size();
        // batch size to remove
        int kRemove = 1 + rndi(max(1, min(K, 8)));
        if(kRemove > K) kRemove = K;

        // --- RUIN ---
        int mode = rndi(3);
        vector<int> removeIdx; // positions in trial.seq to remove
        if(K==0){
            // nothing to remove; just (re)insert everyone
        } else if(mode==0){
            // worst-removal: remove the nodes whose "detour cost" is largest.
            // detour(i) = dt(prev,u)+dt(u,next)-dt(prev,next)
            vector<pair<long long,int>> cost;
            for(int i=0;i<K;i++){
                int u=trial.seq[i];
                int pv = (i==0)?0:trial.seq[i-1];
                int nx = (i+1<K)?trial.seq[i+1]:0;
                long long c = dt(pv,u)+dt(u,nx)-dt(pv,nx);
                // add a little randomness so we don't always pick identical sets
                cost.push_back({c*1000 + (long long)(rnd()%900), i});
            }
            sort(cost.rbegin(), cost.rend());
            for(int i=0;i<kRemove;i++) removeIdx.push_back(cost[i].second);
        } else if(mode==1){
            // Shaw / related removal: pick a seed node, remove it and its
            // spatially+temporally nearest companions.
            int seed = rndi(K);
            int su = trial.seq[seed];
            vector<pair<long long,int>> rel;
            for(int i=0;i<K;i++){
                int u=trial.seq[i];
                long long sp = dt(su,u);
                long long tw = llabs(req[su].r - req[u].r);
                rel.push_back({sp*4 + tw, i});
            }
            sort(rel.begin(), rel.end());
            for(int i=0;i<kRemove && i<K;i++) removeIdx.push_back(rel[i].second);
        } else {
            // random removal
            vector<int> all(K); iota(all.begin(), all.end(), 0);
            for(int i=0;i<kRemove;i++){
                int j = i + rndi(K-i);
                swap(all[i], all[j]);
                removeIdx.push_back(all[i]);
            }
        }
        // apply removals (descending position so indices stay valid)
        sort(removeIdx.rbegin(), removeIdx.rend());
        removeIdx.erase(unique(removeIdx.begin(), removeIdx.end()), removeIdx.end());
        for(int idx : removeIdx){
            int u = trial.seq[idx];
            flag[u]=0;
            trial.seq.erase(trial.seq.begin()+idx);
        }
        trial.recompute();

        // --- RECREATE ---
        greedyInsertAll(trial, flag);
        trial.recompute();

        int tc = curServed(trial);
        int cc = curServed(cur);
        // accept if more served, or equal served with strictly less endTime
        bool accept = false;
        if(tc > cc) accept = true;
        else if(tc == cc && trial.endTime < cur.endTime) accept = true;
        // occasional random restart-ish acceptance to diversify (only if not worse by much)
        else if(tc == cc && (rnd()%100) < 20) accept = true;

        if(accept){
            cur = trial;
            curServedFlag = flag;
            if(tc > bestCount || (tc==bestCount && trial.endTime<best.endTime)){
                best = trial;
                bestCount = tc;
            }
        }
    }

    // ---- output (always feasible: best is a replayed valid route) ----
    // Defensive: best is guaranteed feasible by construction; print it.
    cout << best.seq.size() << "\n";
    for(size_t i=0;i<best.seq.size();i++){
        cout << best.seq[i] << (i+1<best.seq.size()? ' ' : '\n');
    }
    if(best.seq.empty()) cout << "\n";
    return 0;
}
