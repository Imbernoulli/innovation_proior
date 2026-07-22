// TIER: strong
// The insight: flux is conserved, so no placement can destroy wind energy -- it can
// only be relocated (jetted to the two nearest lanes, or diffused more broadly one
// column downwind). Since the objective is a SUM OF SQUARES (convex), concentrating
// removed momentum into a jet is punished harder than spreading it thinly, and a
// solid wall that doesn't span the whole lateral field just forces its blocked flux
// out past its own tips, onto whatever crops sit there. So this solution actually
// SIMULATES the exact mechanics (mirroring the checker) and ranks every candidate
// (column, lane, porosity level) by simulated marginal score-per-cost, then greedily
// fills the budget and runs a couple of local-improvement passes against the
// current grid. This routinely prefers several cheap, leaky (low-porosity)
// plantings over one expensive solid one.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll W,H,Wb,Pmax,Ymin,Ymax,K;
static vector<ll> inflow;
static ll driftNum, driftDen;
static vector<ll> cost, drag, jetfrac;
static int M;
static vector<vector<int>> cropsAtX;

static inline ll reflect(ll k, ll Hn){
    if (Hn <= 1) return 0;
    ll period = 2 * (Hn - 1);
    ll r = k % period; if (r < 0) r += period;
    if (r >= Hn) r = period - r;
    return r;
}
static void distribute(ll amount, const vector<ll>& targets, const vector<ll>& weights, vector<ll>& dst){
    ll wsum = 0; for (ll w : weights) wsum += w;
    if (wsum <= 0 || amount == 0) return;
    vector<ll> share(targets.size());
    ll used = 0;
    for (size_t i = 0; i < targets.size(); i++){ share[i] = (amount * weights[i]) / wsum; used += share[i]; }
    ll rem = amount - used;
    vector<int> order(targets.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
    sort(order.begin(), order.end(), [&](int a, int b){
        if (weights[a] != weights[b]) return weights[a] > weights[b];
        return a < b;
    });
    int oi = 0;
    while (rem > 0 && !order.empty()){ share[order[oi % order.size()]] += 1; rem--; oi++; }
    for (size_t i = 0; i < targets.size(); i++) dst[targets[i]] += share[i];
}

static ll simulate(const vector<vector<int>>& grid){
    vector<ll> s(H), pool(H, 0);
    for (ll y = 0; y < H; y++) s[y] = inflow[y];
    ll shiftAcc = 0, F = 0;
    for (ll x = 0; x < W; x++){
        for (ll y = 0; y < H; y++){ s[y] += pool[y]; pool[y] = 0; }
        shiftAcc += driftNum;
        if (shiftAcc >= driftDen){
            shiftAcc -= driftDen;
            vector<ll> ns(H);
            for (ll y = 0; y < H; y++) ns[y] = s[(y - 1 + H) % H];
            s.swap(ns);
        }
        if (x < Wb){
            vector<ll> s0 = s;
            vector<ll> delta(H, 0), poolNext(H, 0);
            for (ll y = Ymin; y <= Ymax; y++){
                int p = grid[x][y];
                if (p <= 0) continue;
                ll removed = (s0[y] * drag[p]) / 1000;
                delta[y] -= removed;
                ll jetAmt = (removed * jetfrac[p]) / 1000;
                ll diffAmt = removed - jetAmt;
                vector<ll> jt = {reflect(y-1,H), reflect(y+1,H)};
                vector<ll> jw = {1,1};
                distribute(jetAmt, jt, jw, delta);
                vector<ll> dt = {reflect(y-2,H), reflect(y-1,H), reflect(y,H), reflect(y+1,H), reflect(y+2,H)};
                vector<ll> dw = {1,2,4,2,1};
                distribute(diffAmt, dt, dw, poolNext);
            }
            for (ll y = 0; y < H; y++) s[y] = s0[y] + delta[y];
            pool = poolNext;
        }
        if (x >= Wb){
            for (int y : cropsAtX[x]) F += s[y] * s[y];
        }
    }
    return F;
}

int main(){
    cin>>W>>H>>Wb>>Pmax>>Ymin>>Ymax>>K;
    inflow.assign(H, 0);
    for (ll y=0;y<H;y++) cin>>inflow[y];
    cin>>driftNum>>driftDen;
    cost.assign(Pmax+1,0); drag.assign(Pmax+1,0); jetfrac.assign(Pmax+1,0);
    for (ll p=1;p<=Pmax;p++) cin>>cost[p];
    for (ll p=1;p<=Pmax;p++) cin>>drag[p];
    for (ll p=1;p<=Pmax;p++) cin>>jetfrac[p];
    cin>>M;
    cropsAtX.assign(W, {});
    for (int i=0;i<M;i++){ ll x,y; cin>>x>>y; cropsAtX[x].push_back((int)y); }

    vector<vector<int>> grid(Wb, vector<int>(H, 0));
    ll F0 = simulate(grid);

    struct Cand{ ll x,y; int p; double eff; };
    vector<Cand> cands;
    for (ll x=0;x<Wb;x++){
        for (ll y=Ymin;y<=Ymax;y++){
            int bestP = 0; double bestEff = -1e18;
            for (int p=1;p<=(int)Pmax;p++){
                grid[x][y] = p;
                ll F = simulate(grid);
                grid[x][y] = 0;
                ll df = F0 - F; // positive = improvement (minimizing F)
                double eff = (double)df / (double)cost[p];
                if (df > 0 && eff > bestEff){ bestEff = eff; bestP = p; }
            }
            if (bestP > 0) cands.push_back({x,y,bestP,bestEff});
        }
    }
    sort(cands.begin(), cands.end(), [](const Cand&a, const Cand&b){ return a.eff > b.eff; });

    ll budget = K;
    for (auto &c : cands){
        if (grid[c.x][c.y] != 0) continue;
        if (cost[c.p] <= budget){
            grid[c.x][c.y] = c.p;
            budget -= cost[c.p];
        }
    }

    ll curF = simulate(grid);
    for (int pass = 0; pass < 2; pass++){
        for (ll x=0;x<Wb;x++){
            for (ll y=Ymin;y<=Ymax;y++){
                int cur = grid[x][y];
                int bestP = cur; ll bestF = curF; ll bestCostDelta = 0;
                for (int p=0;p<=(int)Pmax;p++){
                    if (p == cur) continue;
                    ll costDelta = (p>0?cost[p]:0) - (cur>0?cost[cur]:0);
                    if (budget - costDelta < 0) continue;
                    grid[x][y] = p;
                    ll F = simulate(grid);
                    grid[x][y] = cur;
                    if (F < bestF){ bestF = F; bestP = p; bestCostDelta = costDelta; }
                }
                if (bestP != cur){
                    budget -= bestCostDelta;
                    grid[x][y] = bestP;
                    curF = bestF;
                }
            }
        }
    }

    // Safety net: on tiny/idiosyncratic instances the marginal-then-refine heuristic
    // can occasionally under-perform the naive "solid wall at the front" recipe, so
    // also build that recipe and keep whichever grid the simulator actually scores
    // better -- a real strong solver never ships a solution it hasn't checked beats
    // the obvious one.
    {
        vector<vector<int>> altGrid(Wb, vector<int>(H, 0));
        vector<ll> lanes;
        for (ll y=Ymin;y<=Ymax;y++) lanes.push_back(y);
        sort(lanes.begin(), lanes.end(), [&](ll a, ll b){ return inflow[a] > inflow[b]; });
        ll altBudget = K;
        for (ll y : lanes){
            if (altBudget >= cost[Pmax]){ altGrid[0][y] = (int)Pmax; altBudget -= cost[Pmax]; }
        }
        ll altF = simulate(altGrid);
        if (altF < curF) grid = altGrid;
    }

    for (ll x=0;x<Wb;x++){
        for (ll y=0;y<H;y++) cout<<grid[x][y]<<(y+1==H?'\n':' ');
    }
    return 0;
}
