#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Brewery batches sharing one thermal storage tank".
//
// Input:  N H C ; then N lines  type vol e l p1 p2
//   type=0 donor:    p1=temp, p2 unused
//   type=1 consumer: p1=req,  p2=price
// Output: N integers s_1..s_N with e_i<=s_i<=l_i (bounded read enforces this).
//
// Simulation (shared by baseline + participant): sort task indices by
// (s_i, i), replay in that order on an initially empty tank (V=0,T=0):
//   donor:    a = min(vol, C-V);  if a>0: T = (V*T + a*temp)/(V+a); V += a.
//   consumer: e = min(vol, V); sh = vol - e;
//             cost += price * (e*max(0,req-T) + sh*req);  V -= e.
// F = total cost (MIN objective).
//
// Baseline B = same simulation on the earliest-start schedule (s_i = e_i for
// every task) -- exactly what solutions/trivial.cpp reproduces.
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int N; ll H, C;
vector<int> type_;
vector<ll> vol_, e_, l_, p1_, p2_;

double simulate(const vector<ll> &s){
    vector<int> idx(N);
    for (int i = 0; i < N; i++) idx[i] = i;
    sort(idx.begin(), idx.end(), [&](int a, int b){
        if (s[a] != s[b]) return s[a] < s[b];
        return a < b;
    });
    double V = 0.0, T = 0.0, cost = 0.0;
    for (int id : idx){
        if (type_[id] == 0){
            double a = min((double)vol_[id], (double)C - V);
            if (a > 0){
                T = (V * T + a * (double)p1_[id]) / (V + a);
                V += a;
            }
        } else {
            double eff = min((double)vol_[id], V);
            double sh = (double)vol_[id] - eff;
            double deficit = max(0.0, (double)p1_[id] - T);
            cost += (double)p2_[id] * (eff * deficit + sh * (double)p1_[id]);
            V -= eff;
        }
    }
    return cost;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    H = inf.readLong();
    C = inf.readLong();
    type_.resize(N); vol_.resize(N); e_.resize(N); l_.resize(N); p1_.resize(N); p2_.resize(N);
    for (int i = 0; i < N; i++){
        type_[i] = inf.readInt();
        vol_[i]  = inf.readLong();
        e_[i]    = inf.readLong();
        l_[i]    = inf.readLong();
        p1_[i]   = inf.readLong();
        p2_[i]   = inf.readLong();
    }

    // ---- internal baseline B: earliest-start schedule ----
    vector<ll> sBase(N);
    for (int i = 0; i < N; i++) sBase[i] = e_[i];
    double Bd = simulate(sBase);
    ll B = (ll)llround(Bd);
    if (B <= 0) B = 1;

    // ---- read participant schedule (bounded reads enforce feasibility) ----
    vector<ll> s(N);
    for (int i = 0; i < N; i++){
        s[i] = ouf.readLong(e_[i], l_[i], "s_i");
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the N start times");

    double Fd = simulate(s);
    if (!isfinite(Fd)) quitf(_wa, "non-finite objective");
    ll F = (ll)llround(Fd);
    if (F < 0) F = 0;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
