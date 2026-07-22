// TIER: strong
// The insight: the boiler's oscillation is a deterministic, fully known
// trajectory (the recurrence is entirely given by the input), so a solver can
// SIMULATE it ahead of time instead of forcing anything. Two phases:
//  (1) WAIT-AND-INTERCEPT pass: walk ticks 0..Tmax-1 on the natural (no-flush)
//      trajectory; at each tick, among orders currently pending whose window
//      the trajectory is ALREADY inside, pull the highest-weight one (never
//      "serve on arrival" -- an order just sits until its own moment in the
//      sweep arrives, which costs nothing but a little wait budget).
//  (2) FLUSH-RESCUE pass: for every order the sweep never reaches in time
//      (this is exactly the trap the greedy tier falls into), try inserting
//      ONE flush a few ticks after its arrival, RE-SIMULATE the full trajectory
//      with that flush added, and check both that the order now lands in its
//      window AND that no already-committed pull is broken downstream by the
//      changed trajectory. Only commit if both hold and the scarce flush
//      budget allows it -- flushes are spent on rescues nothing else can reach,
//      not spent as a blunt substitute for patience.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static int N, Tmax, H0, FMAX;
static ll TLO, THI, THOT, PH, QH, TCOLD, PC, QC, T0, FLUSH_DROP;
static vector<ll> A, D, LO, HI, W;

static inline ll stepT(ll T, int H) {
    ll diff = H ? (THOT - T) : (TCOLD - T);
    ll p = H ? PH : PC, q = H ? QH : QC;
    T += (diff * p) / q;
    return T;
}
static inline int nextH(ll T, int H) {
    if (T <= TLO) return 1;
    if (T >= THI) return 0;
    return H;
}
static inline bool inWindow(ll T, int i) { return T >= LO[i] && T <= HI[i]; }

// simulate the full trajectory given a boolean flush mask of length Tmax
static vector<ll> simulate(const vector<char> &flushAt) {
    vector<ll> T(Tmax + 1);
    T[0] = T0;
    int H = H0;
    for (int t = 0; t < Tmax; t++) {
        ll cur = T[t];
        if (flushAt[t]) { cur -= FLUSH_DROP; if (cur < 0) cur = 0; }
        ll nt = stepT(cur, H);
        T[t + 1] = nt;
        H = nextH(nt, H);
    }
    return T;
}

int main() {
    if (scanf("%d %d %lld %lld %lld %lld %lld %lld %lld %lld %lld %d %lld %d",
               &N, &Tmax, &TLO, &THI, &THOT, &PH, &QH, &TCOLD, &PC, &QC, &T0, &H0, &FLUSH_DROP, &FMAX) != 14) return 0;
    A.resize(N); D.resize(N); LO.resize(N); HI.resize(N); W.resize(N);
    for (int i = 0; i < N; i++) scanf("%lld %lld %lld %lld %lld", &A[i], &D[i], &LO[i], &HI[i], &W[i]);

    vector<char> flushAt(Tmax, 0);
    vector<ll> traj = simulate(flushAt);   // natural trajectory (no flush yet)

    vector<char> usedTick(Tmax, 0);
    vector<char> served(N, 0);
    vector<pair<ll,int>> pulls;   // (tick, orderIdx)
    int flushUsed = 0;

    // ---- phase 1: wait-and-intercept on the natural trajectory ----
    for (int t = 0; t < Tmax; t++) {
        if (usedTick[t]) continue;
        int best = -1;
        for (int i = 0; i < N; i++) {
            if (served[i] || A[i] > t || D[i] < t) continue;
            if (!inWindow(traj[t], i)) continue;
            if (best == -1 || W[i] > W[best] || (W[i] == W[best] && D[i] < D[best])) best = i;
        }
        if (best != -1) {
            pulls.push_back({t, best});
            served[best] = 1;
            usedTick[t] = 1;
        }
    }

    // ---- phase 2: flush-rescue for unserved orders, highest weight first ----
    vector<int> unserved;
    for (int i = 0; i < N; i++) if (!served[i]) unserved.push_back(i);
    sort(unserved.begin(), unserved.end(), [](int x, int y) { return W[x] > W[y]; });

    for (int i : unserved) {
        if (served[i] || flushUsed >= FMAX) continue;
        ll loTf = A[i], hiTf = min(D[i] - 1, A[i] + 8);
        bool done = false;
        for (ll tf = loTf; tf <= hiTf && !done; tf++) {
            if (tf < 0 || tf >= Tmax || usedTick[tf]) continue;
            vector<char> trial = flushAt;
            trial[tf] = 1;
            vector<ll> trialTraj = simulate(trial);
            // find earliest free tick in (tf, D[i]] where order i lands in window
            ll pt = -1;
            for (ll t = tf + 1; t <= D[i] && t < Tmax; t++) {
                if (usedTick[t]) continue;
                if (inWindow(trialTraj[t], i)) { pt = t; break; }
            }
            if (pt == -1) continue;
            // verify no already-committed pull downstream of tf is broken
            bool ok = true;
            for (auto &pr : pulls) {
                if (pr.first > tf && !inWindow(trialTraj[pr.first], pr.second)) { ok = false; break; }
            }
            if (!ok) continue;
            // commit
            flushAt = trial;
            traj = trialTraj;
            usedTick[tf] = 1;
            usedTick[pt] = 1;
            pulls.push_back({pt, i});
            served[i] = 1;
            flushUsed++;
            done = true;
        }
    }

    // ---- assemble output: pulls + flush ticks, sorted by tick ----
    vector<pair<ll,int>> actions = pulls;
    for (int t = 0; t < Tmax; t++) if (flushAt[t]) actions.push_back({t, -1});
    sort(actions.begin(), actions.end());

    printf("%d\n", (int) actions.size());
    for (auto &pr : actions) printf("%lld %d\n", pr.first, pr.second);
    return 0;
}
