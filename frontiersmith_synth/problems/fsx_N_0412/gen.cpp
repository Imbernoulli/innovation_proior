#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Clause { ll w; int r; vector<int> lits; };

int n;
vector<int> A; // hidden near-optimal assignment A[1..n]
vector<Clause> C;

// sample k distinct variables in 1..n
vector<int> sampleVars(int k) {
    if (k > n) k = n;
    set<int> s;
    while ((int)s.size() < k) s.insert(rnd.next(1, n));
    return vector<int>(s.begin(), s.end());
}

// planted route: signals random, resonance target set so it PAYS under hidden A*
void addPlanted(ll wt, int k) {
    auto vs = sampleVars(k);
    vector<int> lits; int c = 0;
    for (int v : vs) {
        int sign = rnd.next(0, 1) ? 1 : -1;
        lits.push_back(sign > 0 ? v : -v);
        bool sat = (sign > 0) ? (A[v] == 1) : (A[v] == 0);
        if (sat) c++;
    }
    C.push_back({wt, c % 3, lits});
}

// trap route: naive "meet every signal" gives c=k, but target = (k-1)%3, so a
// greedy that maximizes met signals lands off-phase and gets NOTHING.
void addTrap(ll wt, int k) {
    auto vs = sampleVars(k);
    vector<int> lits;
    for (int v : vs) {
        int sign = rnd.next(0, 1) ? 1 : -1;
        lits.push_back(sign > 0 ? v : -v);
    }
    int kk = (int)lits.size();
    C.push_back({wt, (kk - 1 + 3) % 3, lits});
}

// baseline route: target chosen so it PAYS under all-WEST -> guarantees B>0
void addBaseline(ll wt, int k) {
    auto vs = sampleVars(k);
    vector<int> lits; int c = 0;
    for (int v : vs) {
        int sign = rnd.next(0, 1) ? 1 : -1;
        lits.push_back(sign > 0 ? v : -v);
        // under all-WEST (x=0): positive literal not met, negative literal met
        if (sign < 0) c++;
    }
    C.push_back({wt, c % 3, lits});
}

// noise route: random resonance target
void addNoise(ll wt, int k) {
    auto vs = sampleVars(k);
    vector<int> lits;
    for (int v : vs) {
        int sign = rnd.next(0, 1) ? 1 : -1;
        lits.push_back(sign > 0 ? v : -v);
    }
    C.push_back({wt, rnd.next(0, 2), lits});
}

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure/size ladder ----
    if (testId == 1) {
        n = 6;                 // tiny, example scale
    } else {
        // grow to the envelope by testId 10: n up to ~1490
        n = 8 + (testId - 1) * 165;
        if (n > 1500) n = 1500;
    }

    int m;
    if (testId == 1) m = 8;
    else m = 8 * n;            // dense route set
    if (m > 12000) m = 12000;

    int kmax = min(n, 3 + testId / 2);   // clause size grows: 3..8, capped by n
    if (kmax < 2) kmax = min(n, 2);
    int kmin = min(n, 2);

    auto randK = [&]() {
        int lo = kmin, hi = kmax;
        if (hi < lo) hi = lo;
        return rnd.next(lo, hi);
    };

    // hidden near-optimal assignment A*
    A.assign(n + 1, 0);
    for (int v = 1; v <= n; v++) A[v] = rnd.next(0, 1);

    C.clear();
    C.reserve(m);

    // route-type budget (fractions of m)
    int nBaseline = max(2, m / 6);           // guarantee positive baseline, modest weight
    int nPlanted  = (int)(m * 0.42);         // the achievable optimum lives here
    int nTrap     = (int)(m * 0.16);         // punishes "meet everything" greedy
    int nNeedle   = (testId >= 3) ? 2 + testId / 3 : 1; // few high-value planted routes
    int placed    = nBaseline + nPlanted + nTrap + nNeedle;
    int nNoise    = max(0, m - placed);

    // baseline routes: small weight so B stays a modest reference
    for (int i = 0; i < nBaseline; i++) addBaseline(rnd.next(1, 3), randK());
    // planted routes consistent with A*: the reward gradient
    for (int i = 0; i < nPlanted; i++) addPlanted(rnd.next(3, 9), randK());
    // trap routes
    for (int i = 0; i < nTrap; i++) addTrap(rnd.next(3, 9), randK());
    // needle routes: rare, high-value, only pay when the hidden structure is found
    for (int i = 0; i < nNeedle; i++) addPlanted(rnd.next(60, 100), randK());
    // noise routes
    for (int i = 0; i < nNoise; i++) addNoise(rnd.next(1, 6), randK());

    // shuffle so route order carries no positional information
    shuffle(C.begin(), C.end());

    int mm = (int)C.size();
    printf("%d %d\n", n, mm);
    for (auto& c : C) {
        printf("%lld %d %d", c.w, c.r, (int)c.lits.size());
        for (int l : c.lits) printf(" %d", l);
        printf("\n");
    }
    return 0;
}
