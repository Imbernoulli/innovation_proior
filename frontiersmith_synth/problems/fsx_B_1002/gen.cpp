#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "The Surfing Barista".  family: single-boiler-surfing-queue
//
// Builds a boiler (bang-bang hysteresis thermal law) and N drink orders.
// Category mix across the 10 tests:
//   1-4  : random orders on top of the natural oscillation (general difficulty ramp)
//   5-6  : PLANTED phase-swept clusters -- several orders per cluster whose windows
//          are pinned to CONSECUTIVE ticks of one heating/cooling leg, but all
//          arrive together several ticks before the first target tick. "Serve
//          ASAP on arrival" (FIFO/EDF ignoring temperature) serves the whole
//          cluster back-to-back starting at arrival -- a constant phase offset
//          from every order's true target tick -- while a scheduler that WAITS
//          for the sweep intercepts every one of them for near-full quality.
//   7-8  : TRAP flush-dependent orders -- window pinned to the value the boiler
//          would reach ONE FLUSH below its value at arrival, with a deadline of
//          arrival+1: reachable ONLY by flushing at arrival then pulling next
//          tick. A no-flush strategy always misses (temperature far outside
//          the narrow window), no matter which tick within [a,d] it tries.
//   9    : NEEDLE -- one very high weight order sharing a tight target tick with
//          several low-weight fillers whose deadlines are strictly EARLIER, so a
//          deadline-only scheduler serves the fillers and pushes the needle past
//          its narrow window.
//   10   : mixed -- planted + trap + needle + random, largest test, fills the
//          declared constraint envelope.
// -----------------------------------------------------------------------------

struct Order { ll a, d, lo, hi, w; };

static inline ll stepT(ll T, int H, ll THOT, ll PH, ll QH, ll TCOLD, ll PC, ll QC) {
    ll diff = H ? (THOT - T) : (TCOLD - T);
    ll p = H ? PH : PC, q = H ? QH : QC;
    T += (diff * p) / q;
    return T;
}
static inline int nextH(ll T, int H, ll TLO, ll THI) {
    if (T <= TLO) return 1;
    if (T >= THI) return 0;
    return H;
}

struct Params { ll TLO, THI, THOT, PH, QH, TCOLD, PC, QC, T0; int H0; };

static Params makeParams() {
    Params p;
    p.TLO = 9000 + rnd.next(-150, 150);
    ll band = 620 + rnd.next(0, 150);           // 620..770
    p.THI = p.TLO + band;
    p.THOT = p.THI + 5500 + rnd.next(-400, 400);
    p.TCOLD = p.TLO - 6000 + rnd.next(-400, 400);
    if (p.TCOLD < 250) p.TCOLD = 250;
    p.PH = 1; p.QH = 30 + rnd.next(-6, 6);
    p.PC = 1; p.QC = 30 + rnd.next(-6, 6);
    p.T0 = p.TLO + rnd.next(0, (int) band);
    p.H0 = rnd.next(0, 1);
    return p;
}

static vector<ll> naturalTrajectory(const Params &p, int Tmax) {
    vector<ll> nat(Tmax + 1);
    nat[0] = p.T0;
    int H = p.H0;
    for (int t = 0; t < Tmax; t++) {
        ll nt = stepT(nat[t], H, p.THOT, p.PH, p.QH, p.TCOLD, p.PC, p.QC);
        nat[t + 1] = nt;
        H = nextH(nt, H, p.TLO, p.THI);
    }
    return nat;
}

// Orders arrive in small BATCHES (a realistic queue: several customers show up
// within a couple of ticks of each other), not spread uniformly thin. This is
// what makes "give up the moment your exact arrival tick is taken" (trivial)
// meaningfully weaker than "search forward for the next free tick" (greedy).
static void addRandomOrders(vector<Order> &orders, const vector<ll> &nat, int Tmax, int count) {
    int i = 0;
    while (i < count) {
        int batch = min(count - i, 2 + rnd.next(0, 3));   // 2..5
        int slack = 15 + rnd.next(0, 35);
        int aBase = rnd.next(0, max(1, Tmax - slack - 4));
        for (int b = 0; b < batch; b++) {
            int a = min(Tmax - 2, aBase + rnd.next(0, 2));
            int d = min(Tmax - 1, a + slack);
            int tt = rnd.next(a, d);
            ll center = nat[tt];
            ll hw = 60 + rnd.next(0, 200);
            Order o;
            o.a = a; o.d = d;
            o.lo = max(0LL, center - hw); o.hi = center + hw;
            o.w = 1 + rnd.next(0, 4);
            orders.push_back(o);
            i++;
        }
    }
}

// PLANTED: one cluster of `len` orders pinned to consecutive natural ticks
// s, s+1, ..., s+len-1, all arriving `bufA` ticks BEFORE the first target tick.
static void addPlantedCluster(vector<Order> &orders, const vector<ll> &nat, int Tmax) {
    int len = 4 + rnd.next(0, 3);                       // 4..6
    int s = rnd.next(10, max(11, Tmax - len - 12));
    int bufA = 5 + rnd.next(0, 5);                       // 5..10 ticks of enforced waiting
    int bufD = 2 + rnd.next(0, 3);                       // slack after each own target tick
    int a0 = max(0, s - bufA);
    for (int j = 0; j < len; j++) {
        int tt = s + j;
        if (tt >= Tmax) break;
        ll center = nat[tt];
        ll hw = 25 + rnd.next(0, 25);                    // narrow: forces real timing
        Order o;
        o.a = a0; o.d = min(Tmax - 1, tt + bufD);
        o.lo = max(0LL, center - hw); o.hi = center + hw;
        o.w = 2 + rnd.next(0, 3);
        orders.push_back(o);
    }
}

// TRAP: window reachable only by flushing exactly at arrival, pulling next tick.
static void addTrapOrder(vector<Order> &orders, const Params &p, const vector<ll> &nat, int Tmax, ll FLUSH_DROP) {
    int tf = rnd.next(8, Tmax - 4);
    // simulate one tick starting from a flush at tf, from the natural state at tf.
    // reconstruct heater state at tf by replaying nextH along nat[] (nat has no flush,
    // but H only depends on the trajectory itself, which up to tf is identical since no
    // flush happened before tf in THIS local replay).
    int H = p.H0;
    for (int t = 0; t < tf; t++) H = nextH(nat[t + 1], H, p.TLO, p.THI);
    ll cur = max(0LL, nat[tf] - FLUSH_DROP);
    ll after = stepT(cur, H, p.THOT, p.PH, p.QH, p.TCOLD, p.PC, p.QC);
    ll hw = 20 + rnd.next(0, 15);
    Order o;
    o.a = tf; o.d = min(Tmax - 1, tf + 1);
    o.lo = max(0LL, after - hw); o.hi = after + hw;
    o.w = 2 + rnd.next(0, 3);
    orders.push_back(o);
}

// NEEDLE: one high-weight order plus low-weight fillers with strictly earlier
// deadlines contending for the same tick.
static void addNeedleGroup(vector<Order> &orders, const vector<ll> &nat, int Tmax, ll needleW) {
    int Tstar = rnd.next(20, Tmax - 10);
    int bufA = 4 + rnd.next(0, 4);
    int a0 = max(0, Tstar - bufA);
    ll centerStar = nat[Tstar];
    ll hwStar = 25 + rnd.next(0, 15);
    Order needle;
    needle.a = a0; needle.d = min(Tmax - 1, Tstar + 2);
    needle.lo = max(0LL, centerStar - hwStar); needle.hi = centerStar + hwStar;
    needle.w = needleW;
    orders.push_back(needle);
    int nf = 3 + rnd.next(0, 2);
    for (int i = 0; i < nf; i++) {
        ll center = nat[max(0, Tstar - 1)];
        ll hw = 200 + rnd.next(0, 150);              // wide: easy for a naive scheduler to "solve"
        Order f;
        f.a = a0; f.d = min(Tmax - 1, Tstar + 1);      // deadline strictly before the needle's
        f.lo = max(0LL, center - hw); f.hi = center + hw;
        f.w = 1;
        orders.push_back(f);
    }
}

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    static const int Ns[10]    = {6, 20, 40, 65, 80, 100, 70, 110, 140, 220};
    static const int Tmaxs[10] = {150, 300, 500, 650, 750, 900, 500, 700, 1000, 1800};
    int N = Ns[t - 1];
    int Tmax = Tmaxs[t - 1];

    Params p = makeParams();
    vector<ll> nat = naturalTrajectory(p, Tmax);
    ll band = p.THI - p.TLO;
    ll FLUSH_DROP = max(150LL, (ll) (band * 0.55) + rnd.next(-30, 30));

    vector<Order> orders;
    int trapCount = 0;

    if (t <= 4) {
        addRandomOrders(orders, nat, Tmax, N);
    } else if (t == 5 || t == 6) {
        int used = 0;
        while ((int) orders.size() < N - 6 && used < 40) { addPlantedCluster(orders, nat, Tmax); used++; }
        addRandomOrders(orders, nat, Tmax, max(0, N - (int) orders.size()));
    } else if (t == 7 || t == 8) {
        int wantTraps = (t == 7) ? N / 3 : N / 3 + N / 6;
        for (int i = 0; i < wantTraps && (int) orders.size() < N; i++) { addTrapOrder(orders, p, nat, Tmax, FLUSH_DROP); trapCount++; }
        addRandomOrders(orders, nat, Tmax, max(0, N - (int) orders.size()));
    } else if (t == 9) {
        int groups = 3;
        for (int i = 0; i < groups; i++) addNeedleGroup(orders, nat, Tmax, 45 + rnd.next(0, 15));
        addRandomOrders(orders, nat, Tmax, max(0, N - (int) orders.size()));
    } else { // t == 10: mixed, largest
        int used = 0;
        while ((int) orders.size() < N / 3 && used < 60) { addPlantedCluster(orders, nat, Tmax); used++; }
        int wantTraps = N / 6;
        for (int i = 0; i < wantTraps && (int) orders.size() < 2 * N / 3; i++) { addTrapOrder(orders, p, nat, Tmax, FLUSH_DROP); trapCount++; }
        int groups = 3;
        for (int i = 0; i < groups; i++) addNeedleGroup(orders, nat, Tmax, 45 + rnd.next(0, 20));
        addRandomOrders(orders, nat, Tmax, max(0, N - (int) orders.size()));
    }

    // clip / trim to exactly N orders
    if ((int) orders.size() > N) orders.resize(N);
    while ((int) orders.size() < N) {
        vector<Order> extra;
        addRandomOrders(extra, nat, Tmax, N - (int) orders.size());
        for (auto &o : extra) orders.push_back(o);
    }
    N = (int) orders.size();

    // stable sort by arrival tick (ties keep construction order) -- keeps the input in a
    // natural "arrival order" a solver can rely on for I/O convenience.
    stable_sort(orders.begin(), orders.end(), [](const Order &x, const Order &y) { return x.a < y.a; });

    int FMAX = max(3, N / 9) + trapCount / 2 + 2;

    printf("%d %d %lld %lld %lld %lld %lld %lld %lld %lld %lld %d %lld %d\n",
           N, Tmax, p.TLO, p.THI, p.THOT, p.PH, p.QH, p.TCOLD, p.PC, p.QC, p.T0, p.H0, FLUSH_DROP, FMAX);
    for (auto &o : orders) printf("%lld %lld %lld %lld %lld\n", o.a, o.d, o.lo, o.hi, o.w);
    return 0;
}
