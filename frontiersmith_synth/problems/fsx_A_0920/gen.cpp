#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Tunnel Relay generator  (family: time-expanded-contention-routing)
//
// The network is built from independent CELLS glued onto a shared node/edge
// pool (cells never share a node or edge, so contention only happens within
// a cell). Every packet's path is 3 hops: a private entry edge (unique to
// that packet), a SHARED middle segment, and a private exit edge.
//
//  * GENERIC-SCARCITY cell: P packets all want the SAME single shared edge
//    (their only route). Its narrow slot window admits only (P - dropCount)
//    of them; who gets in should be decided by VALUE, not arrival order.
//
//  * TRAP cell: two populations share one hub-to-hub BOTTLENECK edge (short,
//    duration 2) that is also the durationwise-shortest route for EVERY
//    packet in the cell:
//      - CAPTIVE packets: deadline too tight for anything but the bottleneck
//        (a hidden DETOUR of duration 8 exists in the graph, but is too slow
//        for them).
//      - FLEXIBLE packets: deadline is generous enough that the slower
//        detour also finishes them in time -- taking the bottleneck is only
//        "locally" better for them, never necessary.
//    FLEXIBLE packets outnumber the bottleneck's narrow admission window
//    (flexCount = C + 2 >= window), and are assigned STRICTLY HIGHER value
//    than captives, and are emitted BEFORE captives in the packet list.
//    Any router that (a) always takes the durationwise-shortest path and
//    (b) processes packets by arrival order or by value will therefore let
//    the flexible packets exhaust the bottleneck's slots first -- stranding
//    every captive packet, even though the flexibles would have arrived
//    fine via the idle detour. The insight: reserve the bottleneck's scarce
//    slots for packets with no detour, and steer the others off it.
//
// gen.cpp plants >=3 test ids (4,6,7,8,9,10) with multiple TRAP cells so the
// obvious shortest-path-first heuristic lands far below a detour-aware
// reservation strategy. A NEEDLE packet (one captive with far higher value
// than its neighbours) appears in the two largest tests.
// -----------------------------------------------------------------------------

int N = 0;
vector<array<ll,3>> E;   // u, v, dur
vector<array<ll,5>> P;   // s, t, r, d, v
ll maxDeadlineSeen = 0;

int newNode(){ return N++; }
int addEdge(ll u, ll v, ll dur){ E.push_back({u, v, dur}); return (int)E.size() - 1; }
void addPacket(ll s, ll t, ll r, ll d, ll v){
    P.push_back({s, t, r, d, v});
    maxDeadlineSeen = max(maxDeadlineSeen, d);
}

// A handful of totally private, uncontended single-edge packets: always
// deliverable by anyone, keeps scale/diversity honest without affecting the
// contention story.
void addEasyAnchor(int cnt){
    for (int i = 0; i < cnt; i++){
        int a = newNode(), b = newNode();
        addEdge(a, b, 2);
        ll val = rnd.next(10, 60);
        addPacket(a, b, 0, 60, val);
    }
}

// P_ packets, each with a private entry/exit pair, all funnelling through
// ONE shared edge whose slot window only fits (P_ - dropCount) of them.
// Optimal handling here is a pure "keep the top-value (window) packets"
// choice -- rewards value-aware admission even with no detour anywhere.
// Packets are emitted in INCREASING value order on purpose: arrival order
// carries no information about importance in general, and this is the
// worst case for an arrival-order scheduler (it always keeps the low-value
// early arrivals and drops the high-value latecomers), making the reward
// for scoring by value -- not by order -- robust and deterministic rather
// than a lucky artifact of the random value draw.
void addGenericScarcity(int P_, int dropCount, ll valLo, ll valHi){
    int Lg = newNode(), Rg = newNode();
    addEdge(Lg, Rg, 2);
    int window = max(1, P_ - dropCount);
    ll slack = window - 1;
    ll deadline = 1 + 2 + 1 + slack;  // depart-window width == `window`
    vector<ll> vals(P_);
    for (int i = 0; i < P_; i++) vals[i] = rnd.next(valLo, valHi);
    sort(vals.begin(), vals.end());
    for (int i = 0; i < P_; i++){
        int g = newNode(), t = newNode();
        addEdge(g, Lg, 1);
        addEdge(Rg, t, 1);
        addPacket(g, t, 0, deadline, vals[i]);
    }
}

// The trap: C captive + (C+2) flexible packets sharing one bottleneck, with
// an unused (to naive routing) detour available for the flexible ones.
void addTrapCell(int C, ll needleValue = -1){
    int Lhub = newNode(), Rhub = newNode(), Dmid = newNode();
    addEdge(Lhub, Rhub, 2);   // bottleneck: durationwise-shortest, scarce
    addEdge(Lhub, Dmid, 4);   // detour half 1
    addEdge(Dmid, Rhub, 4);   // detour half 2 (total detour duration 8)

    int flexCount = C + 2;
    int window = C + 2;
    ll slackCap = window - 1;
    ll capDeadline = 1 + 2 + 1 + slackCap;              // = C + 5
    ll slackFlex = flexCount + 3;
    ll flexDeadline = 1 + 8 + 1 + slackFlex;            // = C + 15, comfortably admits the detour

    // Flexible packets are emitted FIRST (both by index and, since their
    // value range strictly (if narrowly) dominates captives', by value
    // too) so both the arrival-order baseline and the value-first greedy
    // fall into the same trap: they let flexible packets exhaust the
    // bottleneck's window before any captive gets a turn. The value gap is
    // kept narrow on purpose -- captives are worth almost as much as
    // flexibles, so losing them all costs the naive approaches dearly.
    for (int i = 0; i < flexCount; i++){
        int s = newNode(), t = newNode();
        addEdge(s, Lhub, 1);
        addEdge(Rhub, t, 1);
        ll val = rnd.next(100, 101);
        addPacket(s, t, 0, flexDeadline, val);
    }
    for (int i = 0; i < C; i++){
        int s = newNode(), t = newNode();
        addEdge(s, Lhub, 1);
        addEdge(Rhub, t, 1);
        ll val = (i == 0 && needleValue > 0) ? needleValue : rnd.next(98, 99);
        addPacket(s, t, 0, capDeadline, val);
    }
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    switch (testId){
        case 1:
            addGenericScarcity(4, 1, 10, 60);
            break;
        case 2:
            addGenericScarcity(6, 2, 10, 80);
            addEasyAnchor(3);
            break;
        case 3:
            addTrapCell(13);
            addGenericScarcity(14, 9, 10, 500);
            break;
        case 4:
            addTrapCell(22);
            addTrapCell(17);
            addGenericScarcity(16, 10, 10, 500);
            break;
        case 5:
            addTrapCell(18);
            addEasyAnchor(4);
            addGenericScarcity(16, 10, 10, 500);
            break;
        case 6:
            addTrapCell(24);
            addTrapCell(26);
            addGenericScarcity(20, 13, 10, 600);
            break;
        case 7:
            addTrapCell(28);
            addTrapCell(24);
            addTrapCell(19, 900);   // needle: one huge-value captive
            addGenericScarcity(20, 13, 10, 600);
            break;
        case 8:
            addTrapCell(30);
            addTrapCell(26);
            addTrapCell(24);
            addTrapCell(20);
            addGenericScarcity(20, 13, 10, 600);
            break;
        case 9:
            addTrapCell(32);
            addTrapCell(28);
            addTrapCell(25);
            addEasyAnchor(6);
            addGenericScarcity(22, 14, 10, 600);
            break;
        default: // 10 and beyond: largest / adversarial
            addTrapCell(36);
            addTrapCell(32);
            addTrapCell(28);
            addTrapCell(23, 1400);  // needle
            addGenericScarcity(22, 14, 10, 600);
            break;
    }

    ll T = maxDeadlineSeen + 5;
    int M = (int)E.size();
    int K = (int)P.size();

    printf("%d %d %d %lld\n", N, M, K, T);
    for (auto& e : E) printf("%lld %lld %lld\n", e[0], e[1], e[2]);
    for (auto& p : P) printf("%lld %lld %lld %lld %lld\n", p[0], p[1], p[2], p[3], p[4]);
    return 0;
}
