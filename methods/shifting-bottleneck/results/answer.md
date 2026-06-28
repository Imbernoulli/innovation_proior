# The Shifting Bottleneck Procedure

## Problem

Job shop makespan minimization. `n` jobs, `m` machines; each job is a fixed route of operations,
operation `i` runs on a prescribed machine for an uninterruptible duration `d_i`; a machine processes
one operation at a time. Choose, for every machine, the order of its operations to minimize the
makespan (completion time of the last job).

On the **disjunctive graph** `G = (N, A, E)` — operations as nodes, conjunctive arcs `A` for job
precedence (arc `i -> j` of weight `d_i`), disjunctive cliques `E_k` for the operations sharing
machine `k` — scheduling means orienting every clique acyclically. Once oriented into a DAG `D_S`, the
**makespan equals the longest path from source to sink**. Orienting all cliques jointly to minimize
the longest path is NP-hard.

## Key idea

Decompose by sequencing machines **one at a time**, always next sequencing the machine that is the
worst **bottleneck**, and re-optimizing the already-sequenced machines after each insertion.

For a partially built schedule with `M_0` the set of already-sequenced machines, an unsequenced
machine `k` defines a **single-machine subproblem**: keep the job arcs `A` and the fixed selections
`S_p` (`p ∈ M_0`), delete the other unsequenced cliques, and read each operation's **head** and
**tail** from longest paths in the current graph:

    r_i = L(0, i)            (head: earliest start)
    q_i = L(i, n) - d_i      (tail: downstream work after i)

With `H = L(0,n)`, this is `1|r_i, q_i|C_max`: minimize
`B(k,M_0) = max_i(C_i + q_i)`. It is equivalent to minimizing maximum lateness `1|r_i|L_max` with due
date `f_i = H - q_i`, since `C_i - f_i = C_i + q_i - H`. The code can return
`ell(k,M_0) = B(k,M_0) - H`; at a fixed iteration `H` is common to every candidate machine, so ranking
by `ell` is the same as ranking by `B`. The actual increase in the partial graph is
`max(0, ell(k,M_0))`, and a large `ell` measures bottleneck quality as a degree, not a yes/no
criticality flag.

## Algorithm

```
M_0 = ∅
while M_0 ≠ M:
    for each unsequenced machine k:
        compute heads/tails from longest paths (job arcs + S_p, p∈M_0; ignore other unsequenced cliques)
        ell(k, M_0) = optimal L_max of k's single-machine 1|r|L_max subproblem
    bottleneck m = argmax_k ell(k, M_0)
    fix m's optimal selection into the graph;  M_0 ← M_0 ∪ {m}
    reoptimize: for ~3 cycles, pull each already-sequenced machine out, recompute its heads/tails,
                re-solve it, drop it back (worst-first); plus a non-critical-machine perturbation
final local reoptimization: keep cycling until a full sweep yields no improvement
```

Notes:
- The single-machine subproblem is solved **exactly** (Carlier 1982 branch-and-bound: Schrage
  dispatch for an upper bound, critical-block lower bound `h(J) = min_J r + Σ_J d + min_J q`, branch on
  the critical job before/after the block; trees rarely exceed `2n` nodes). Exactness matters because
  the machine *ranking* depends on `ell(k, M_0)`. A clean modern equivalent is a CP-SAT model.
- **Re-optimization** is essential: inserting a machine changes every other machine's heads and tails,
  so earlier greedy choices become improvable.
- Longest paths drive the inner loop; an acyclic complete order is the transitive closure of its
  Hamiltonian path, so only the consecutive arcs matter, giving an **O(n)** longest-path labeling over
  per-job and per-machine adjacency lists.
- Cycle from a fixed selection: re-solve that subproblem with the offending precedence enforced.
- `max_k B(k, ∅)` is the first-level lower bound on the optimal makespan; equivalently,
  `H_0 + max_k ell(k, ∅)` when `H_0` is the initial job-precedence longest path.

## Code

Single-file C++17 program. It reads a job-shop instance from stdin in the standard OR-Library
format — first line `n m` (jobs, machines), then `n` job rows, each a route of `m` pairs
`machine duration` in processing order — and prints the makespan found by the shifting-bottleneck
procedure (a valid upper bound on the optimum). The single-machine `1|r_j|L_max` subproblem is
solved exactly by Carlier's branch-and-bound; longest paths over the per-job / per-machine adjacency
lists give heads and tails. All time arithmetic uses `long long`.

```cpp
// Shifting Bottleneck Procedure for job-shop makespan minimization.
//
// I/O contract: reads a job-shop instance from stdin in the standard OR-Library
// format -- first line "n m" (n jobs, m machines), then n lines, each job a route
// of m pairs "machine duration" in processing order. Prints the makespan found by
// the shifting-bottleneck procedure on stdout (a valid upper bound on the optimum).
//
// Operations are numbered op = job*m + position. Makespan is the longest path in
// the oriented disjunctive graph; the bottleneck (largest single-machine optimal
// L_max) is sequenced first, and already-sequenced machines are re-optimized in
// cycles. The single-machine 1|r_j|L_max subproblem is solved exactly by Carlier's
// branch-and-bound. All path/time arithmetic uses long long to avoid overflow.

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n_jobs, n_mach, n_ops;            // jobs, machines, operations
vector<ll> dur;                       // dur[op]
vector<int> mach;                     // machine of op
// disjunctive graph: each op has a job-successor/predecessor (col 0, fixed) and a
// machine-successor/predecessor (col 1, oriented as machines are sequenced).
vector<array<int,2>> succ, pred;      // -1 = none

inline void add_arc(int s, int e){ succ[s][1]=e; pred[e][1]=s; }
inline void remove_arc(int s,int e){ succ[s][1]=-1; pred[e][1]=-1; }

// Find one cycle in the current oriented graph; returns a parent map node->next
// along the cycle (following successors). Used to repair an infeasible selection.
unordered_map<int,int> succ_cycle(){
    vector<int> state(n_ops,0);           // 0=unseen,1=active,2=done
    vector<int> parent(n_ops,-1);
    unordered_map<int,int> cyc;
    // iterative DFS to avoid deep recursion
    for(int s=0;s<n_ops && cyc.empty(); ++s){
        if(state[s]) continue;
        vector<pair<int,int>> st; st.push_back({s,0});
        while(!st.empty()){
            auto& [v,ci] = st.back();
            if(ci==0) state[v]=1;
            if(ci<2){
                int w=succ[v][ci]; ci++;
                if(w<0) continue;
                if(state[w]==0){ parent[w]=v; st.push_back({w,0}); }
                else if(state[w]==1){      // back edge v->w : cycle
                    int cur=v; cyc[v]=w;
                    while(cur!=w){ int p=parent[cur]; cyc[p]=cur; cur=p; }
                    break;
                }
            } else { state[v]=2; st.pop_back(); }
        }
        if(!cyc.empty()) break;
    }
    return cyc;
}

// Longest-path labeling over the (at most 2-in / 2-out) sparse graph. forward=true
// gives completion-style labels (dist[op] = longest path ending at op, including
// dur[op]); forward=false gives labels from op to the sink (including dur[op]).
// Returns true on success (DAG); on a cycle returns false and fills cyc.
bool longest_path(bool forward, vector<ll>& dist, unordered_map<int,int>* cyc=nullptr){
    const auto& nbr = forward ? succ : pred;   // direction we relax along
    const auto& rev = forward ? pred : succ;   // gives in-degree
    dist.assign(n_ops, 0);
    vector<int> indeg(n_ops,0);
    for(int v=0; v<n_ops; ++v)
        for(int c=0;c<2;++c) if(rev[v][c]>=0) indeg[v]++;
    vector<int> stk;
    for(int v=0; v<n_ops; ++v) if(indeg[v]==0){ dist[v]=dur[v]; stk.push_back(v); }
    int seen=0;
    while(!stk.empty()){
        int v=stk.back(); stk.pop_back(); seen++;
        for(int c=0;c<2;++c){
            int w=nbr[v][c]; if(w<0) continue;
            if(dist[v]+dur[w] > dist[w]) dist[w]=dist[v]+dur[w];
            if(--indeg[w]==0) stk.push_back(w);
        }
    }
    if(seen<n_ops){                 // cycle present
        if(cyc) *cyc = succ_cycle();
        return false;
    }
    return true;
}

// Current makespan (longest source-to-sink path) and the critical path of ops.
// Returns -1 on a cycle.
ll makespan(vector<int>* critical=nullptr){
    // forward labeling with predecessor tracking for the critical path
    vector<ll> dist(n_ops,0);
    vector<int> indeg(n_ops,0), prv(n_ops,-1);
    for(int v=0; v<n_ops; ++v)
        for(int c=0;c<2;++c) if(pred[v][c]>=0) indeg[v]++;
    vector<int> stk;
    for(int v=0; v<n_ops; ++v) if(indeg[v]==0){ dist[v]=dur[v]; stk.push_back(v); }
    int seen=0;
    while(!stk.empty()){
        int v=stk.back(); stk.pop_back(); seen++;
        for(int c=0;c<2;++c){
            int w=succ[v][c]; if(w<0) continue;
            if(dist[v]+dur[w] > dist[w]){ dist[w]=dist[v]+dur[w]; prv[w]=v; }
            if(--indeg[w]==0) stk.push_back(w);
        }
    }
    if(seen<n_ops) return -1;
    int sink=0; for(int v=1;v<n_ops;++v) if(dist[v]>dist[sink]) sink=v;
    if(critical){ critical->clear(); int v=sink; while(v>=0){ critical->push_back(v); v=prv[v]; } reverse(critical->begin(),critical->end()); }
    return dist[sink];
}

// ---------------------------------------------------------------------------
// Exact single-machine 1|r_j|L_max via Carlier (1982) branch and bound.
// Each task has processing time p, head (release) r, tail q. Objective: minimize
// max over jobs of (completion + q). We return that optimum (C_max with tails) and
// a job order achieving it. Optional extra precedences (a before b) are enforced
// by inflating heads/tails, used to repair cycles.
// ---------------------------------------------------------------------------
struct CTask { ll p, r, q; int id; };

// Schrage heuristic: among released jobs, dispatch the one with largest tail.
// Returns the C_max-with-tails value and fills seq with the dispatch order (ids).
ll schrage(const vector<CTask>& T, vector<int>& seq){
    int N=T.size();
    vector<char> done(N,0);
    seq.clear();
    ll t = LLONG_MAX;
    for(auto& x:T) t=min(t,x.r);
    ll best=0;
    for(int cnt=0;cnt<N;++cnt){
        int pick=-1;
        for(int i=0;i<N;++i) if(!done[i] && T[i].r<=t){
            if(pick<0 || T[i].q>T[pick].q || (T[i].q==T[pick].q && T[i].p>T[pick].p)) pick=i;
        }
        if(pick<0){ // none released; jump clock to next release
            ll nr=LLONG_MAX; for(int i=0;i<N;++i) if(!done[i]) nr=min(nr,T[i].r);
            t=nr; --cnt; continue;
        }
        ll c=t+T[pick].p;
        best=max(best,c+T[pick].q);
        done[pick]=1; seq.push_back(T[pick].id);
        // advance clock
        ll nr=LLONG_MAX; for(int i=0;i<N;++i) if(!done[i]) nr=min(nr,T[i].r);
        t = max(c, nr==LLONG_MAX? c : nr);
    }
    return best;
}

ll carlier_best;           // incumbent C_max-with-tails
vector<int> carlier_seq;   // incumbent order

void carlier_rec(vector<CTask> T){
    int N=T.size();
    if(N==0) return;
    vector<int> seq;
    ll ub = schrage(T, seq);
    if(ub < carlier_best){ carlier_best=ub; carlier_seq=seq; }
    // rebuild the Schrage schedule to find the critical path / critical job
    // recompute completion times in dispatch order
    unordered_map<int,int> idx; for(int i=0;i<N;++i) idx[T[i].id]=i;
    vector<ll> comp(N,0);
    ll t=LLONG_MAX; for(auto&x:T) t=min(t,x.r);
    ll cmax=0; int critJobPos=-1; ll critVal=-1;
    // track, for each scheduled job, its completion
    {
        ll clk=t;
        for(int s=0;s<N;++s){
            int id=seq[s], i=idx[id];
            clk = max(clk, T[i].r) + T[i].p;
            comp[i]=clk;
            if(clk + T[i].q > critVal){ critVal=clk+T[i].q; critJobPos=s; }
        }
        cmax=critVal;
    }
    // Critical path: the block of consecutive jobs (in dispatch order) ending at
    // the job realizing cmax, back to where the machine last had idle time before
    // a contiguous busy run. Identify block [a..b] (positions) that runs with no
    // idle and ends at critJobPos.
    int b=critJobPos;
    int a=b;
    {
        // start of the contiguous busy block ending at b
        ll clk=t;
        vector<ll> startp(N), endp(N);
        clk=t;
        for(int s=0;s<N;++s){ int i=idx[seq[s]]; ll st=max(clk,T[i].r); startp[s]=st; endp[s]=st+T[i].p; clk=endp[s]; }
        a=b;
        while(a>0){ int i=idx[seq[a]]; if(startp[a]==max(endp[a-1], T[i].r) && startp[a]==endp[a-1]) a--; else break; }
        // ensure block start respects release: find smallest a' with no idle before each
        while(a>0 && startp[a]==endp[a-1]) a--;
    }
    // The critical sequence J' = jobs at positions [a..b]. Find critical job c:
    // largest position k in [a..b-1] with q_{seq[k]} < q_{seq[b]} (tail of last).
    int bId=seq[b];
    ll qlast=T[idx[bId]].q;
    int cpos=-1;
    for(int s=a; s<b; ++s){ if(T[idx[seq[s]]].q < qlast) cpos=s; }
    if(cpos<0) return;   // no critical job: Schrage is optimal for this block
    int cId=seq[cpos];
    // block J = positions (cpos, b]
    vector<int> J;
    for(int s=cpos+1;s<=b;++s) J.push_back(idx[seq[s]]);
    // lower bound h(J) = min r + sum p + min q over J
    ll minr=LLONG_MAX,sump=0,minq=LLONG_MAX;
    for(int i:J){ minr=min(minr,T[i].r); sump+=T[i].p; minq=min(minq,T[i].q); }
    ll hJ = minr+sump+minq;
    if(hJ >= carlier_best) return;     // prune

    int ci = idx[cId];
    // Branch 1: critical job AFTER J -> new head r'_c = max(r_c, min_{J} r + sum_J p)
    {
        vector<CTask> T1=T;
        ll newr = max(T[ci].r, minr+sump);
        T1[ci].r = newr;
        // lower bound for this child
        ll childLB = max(hJ, /*overall lb*/ 0LL);
        if(childLB < carlier_best) carlier_rec(T1);
    }
    // Branch 2: critical job BEFORE J -> new tail q'_c = max(q_c, sum_J p + min_J q)
    {
        vector<CTask> T2=T;
        ll newq = max(T[ci].q, sump+minq);
        T2[ci].q = newq;
        ll childLB = max(hJ, 0LL);
        if(childLB < carlier_best) carlier_rec(T2);
    }
}

// Solve 1|r_j|L_max exactly. ops: operation ids on this machine. head[op],tail[op]
// give release/tail (tail q here is the downstream work after the op). extraPrec:
// pairs (a,b) meaning op a must precede op b (cycle repair). Returns the optimal
// "L_max" = (C_max_with_tails - H) and the order; we keep H implicit by working in
// raw C_max-with-tails and the caller subtracts H. Here we return ell directly.
struct LmaxResult { ll ell; vector<int> order; };

LmaxResult solve_lmax(const vector<int>& ops, const vector<ll>& head,
                      const vector<ll>& tail, ll H,
                      const vector<pair<int,int>>& extraPrec){
    vector<CTask> T; T.reserve(ops.size());
    unordered_map<int,int> pos;
    for(int op:ops){ pos[op]=T.size(); T.push_back({dur[op], head[op], tail[op], op}); }
    // enforce extra precedences a->b by inflating heads/tails (a before b):
    // bump b's head past a, and a's tail past b. Repeat to a fixed point.
    for(int it=0; it<(int)extraPrec.size()+1; ++it){
        bool ch=false;
        for(auto& pr:extraPrec){
            int a=pos.count(pr.first)?pos[pr.first]:-1;
            int b=pos.count(pr.second)?pos[pr.second]:-1;
            if(a<0||b<0) continue;
            ll nr = T[a].r + T[a].p;            // b cannot start before a finishes
            if(T[b].r < nr){ T[b].r=nr; ch=true; }
            ll nq = T[b].q + T[b].p;            // a's downstream includes b
            if(T[a].q < nq){ T[a].q=nq; ch=true; }
        }
        if(!ch) break;
    }
    carlier_best=LLONG_MAX; carlier_seq.clear();
    // seed incumbent with Schrage
    { vector<int> s; ll v=schrage(T,s); carlier_best=v; carlier_seq=s; }
    carlier_rec(T);
    LmaxResult R; R.order=carlier_seq; R.ell=carlier_best - H;
    return R;
}

// ---------------------------------------------------------------------------
// Read heads and tails (q_i = L(i,n)-d_i) off two longest-path passes.
// ---------------------------------------------------------------------------
void heads_tails(vector<ll>& head, vector<ll>& tail, ll& H){
    vector<ll> fwd, bwd;
    longest_path(true, fwd);     // dist ending at op, includes dur
    longest_path(false, bwd);    // dist from op to sink, includes dur
    H=0; for(int v=0;v<n_ops;++v) H=max(H,fwd[v]);
    head.assign(n_ops,0); tail.assign(n_ops,0);
    for(int op=0;op<n_ops;++op){
        head[op]=fwd[op]-dur[op];     // earliest start
        tail[op]=bwd[op]-dur[op];     // downstream work after op
    }
}

// Find the offending precedence pair from a cycle, to enforce on re-solve.
pair<int,int> offending_pair(const unordered_map<int,int>& cyc, const vector<int>& order){
    unordered_set<int> oset(order.begin(), order.end());
    for(int s: order){
        if(cyc.count(s)){
            int nb=cyc.at(s);
            while(!oset.count(nb) && cyc.count(nb)) nb=cyc.at(nb);
            // walk forward while the *next* node is still in order
            while(cyc.count(nb) && oset.count(cyc.at(nb))) nb=cyc.at(nb);
            if(s!=nb) return {s,nb};
        }
    }
    // fallback: any consecutive pair in the order
    return {order.front(), order.back()};
}

// Insert one machine: solve its 1|r|L_max, lay the order's consecutive arcs into
// the graph; if that creates a cycle, enforce the offending precedence and retry.
// Returns the chosen order and its ell (= L_max).
struct InsertResult { vector<int> order; ll ell; ll ms; };

InsertResult insert_machine(const vector<int>& ops){
    vector<pair<int,int>> precs;
    while(true){
        vector<ll> head,tail; ll H; heads_tails(head,tail,H);
        LmaxResult lr = solve_lmax(ops, head, tail, H, precs);
        for(size_t i=0;i+1<lr.order.size();++i) add_arc(lr.order[i], lr.order[i+1]);
        vector<int> crit;
        ll ms = makespan(&crit);
        if(ms>=0) return {lr.order, lr.ell, ms};
        // cycle: detect it on the graph WITH the trial arcs, then undo and enforce
        // the offending precedence before re-solving.
        unordered_map<int,int> cyc = succ_cycle();
        for(size_t i=0;i+1<lr.order.size();++i) remove_arc(lr.order[i], lr.order[i+1]);
        precs.push_back(offending_pair(cyc, lr.order));
    }
}

int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    if(!(cin>>n_jobs>>n_mach)) return 0;
    n_ops = n_jobs*n_mach;
    dur.assign(n_ops,0); mach.assign(n_ops,0);
    succ.assign(n_ops,{-1,-1}); pred.assign(n_ops,{-1,-1});
    for(int j=0;j<n_jobs;++j)
        for(int k=0;k<n_mach;++k){
            int op=j*n_mach+k; ll m,d; cin>>m>>d; mach[op]=(int)m; dur[op]=d;
            // job route arc (fixed, col 0)
            if(k<n_mach-1){ succ[op][0]=op+1; pred[op+1][0]=op; }
        }
    // operations per machine
    vector<vector<int>> machine_ops(n_mach);
    for(int op=0;op<n_ops;++op) machine_ops[mach[op]].push_back(op);

    vector<vector<int>> sol(n_mach);    // committed order per machine
    vector<int> to_schedule, scheduled;
    for(int m=0;m<n_mach;++m) to_schedule.push_back(m);

    auto reopt_machine = [&](int m){
        // pull m out, re-solve, drop back in
        for(size_t i=0;i+1<sol[m].size();++i) remove_arc(sol[m][i], sol[m][i+1]);
        InsertResult r = insert_machine(machine_ops[m]);
        sol[m]=r.order;
        return r.ell;
    };

    for(int it=0; it<n_mach; ++it){
        ll bestEll=LLONG_MIN; int bm=-1; vector<int> bperm;
        for(int k: to_schedule){
            InsertResult r = insert_machine(machine_ops[k]);
            if(r.ell>bestEll){ bestEll=r.ell; bm=k; bperm=r.order; }
            for(size_t i=0;i+1<r.order.size();++i) remove_arc(r.order[i], r.order[i+1]);  // undo trial
        }
        sol[bm]=bperm;
        for(size_t i=0;i+1<bperm.size();++i) add_arc(bperm[i], bperm[i+1]);
        // re-optimize previously sequenced machines (a few worst-first cycles)
        if(it>0 && it<n_mach-1){
            for(int cyc=0; cyc<3; ++cyc){
                vector<pair<ll,int>> scores;
                for(int m: scheduled) scores.push_back({reopt_machine(m), m});
                sort(scores.rbegin(), scores.rend());
                scheduled.clear();
                for(auto& pr:scores) scheduled.push_back(pr.second);
            }
        }
        to_schedule.erase(find(to_schedule.begin(),to_schedule.end(),bm));
        scheduled.push_back(bm);
    }

    // final local re-optimization: cycle until a full sweep yields no improvement
    ll best_ms = makespan();
    for(int iter=0; iter<200; ++iter){
        bool improved=false;
        for(int m: scheduled){
            reopt_machine(m);
            ll ms = makespan();
            if(ms<best_ms){ best_ms=ms; improved=true; }
        }
        if(!improved) break;
    }
    cout << best_ms << "\n";
    return 0;
}
```

The makespan is itself the exact evaluation of a fully oriented schedule: it is the longest
source-to-sink path in the oriented disjunctive graph (`end == start + duration`, job precedence,
no-overlap per machine all encoded by the conjunctive arcs and the committed machine orders), so the
printed value is the schedule's true completion time, a valid upper bound on the optimum.
