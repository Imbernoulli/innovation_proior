#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ============================================================================
// Family: parallax-scope-pairing (theme: telescope fleet catching transients
// that need two eyes).  Mechanisms composed: co-observation-matching +
// decaying-target-value + slew-cost-sequencing.
//
// Telescope roster per test (built incrementally, T grows with testId):
//   several ANCHOR pairs   -- one site-0 + one site-1 telescope co-located
//                             exactly on a dedicated zero-travel, zero-decay
//                             target (a=0). Count scales with testId so the
//                             checker's baseline B (sum of all such freebie
//                             payouts) scales with instance size the same
//                             way the strong solution's harvest does --
//                             this keeps the score ceiling open at large
//                             scale instead of saturating.
//   one SCOUT pair          -- one site-0 + one site-1 telescope positioned
//                             (via distance = speed * R, same R both sides)
//                             so their exact-arrival ticks to one 'obvious'
//                             rendezvous match EXACTLY, regardless of W.
//   nA site-0 regular       -- home azimuth ~0 deg (+-30 jitter), each with
//     + nB site-1 regular      its OWN zero-travel, zero-decay personal lure
//                             (worth more, undecayed, than any rendezvous a
//                             per-scope-only chaser would detour for).
//
// Targets: one freebie per anchor pair, one obvious rendezvous, HID hidden
// rendezvous (azimuth chosen -- by searching (u,v) regular-cluster pairs --
// so a SPECIFIC, non-obvious cross-site pair's natural, undirected arrival
// ticks land within W; a per-scope-only chaser has no way to know which
// partner's geometry syncs where), one personal lure per regular telescope,
// and small filler noise.
//
// A telescope that only ever "chases whichever value IT can reach" (blind
// to whether a same-tick, different-site partner will show up) grabs its
// own zero-decay personal lure -- which can never pair, being same-site --
// and has no mechanism to notice that scope X's own geometry happens to
// sync with scope Y at a specific hidden rendezvous. A strategy that
// reformulates the input as a time-expanded PAIR-assignment graph (evaluate
// every (target, site-0 scope, site-1 scope) triple by the payout it would
// achieve from each scope's CURRENT tail state, commit the best, repeat)
// finds these -- and its slew sequencing falls out of the commitment order.
// ============================================================================

static int H, W;
static ll Pn, Qd;

struct Tel { int site, pos, speed; };
struct Tgt { int a, pos, v, o; };

static inline int angdist(int a, int b) {
    int d = abs(a - b) % 360;
    return min(d, 360 - d);
}
static inline int travelTicks(int p, int q, int speed) {
    int d = angdist(p, q);
    if (d == 0) return 0;
    return (d + speed - 1) / speed;
}
static inline int norm360(int x) { x %= 360; if (x < 0) x += 360; return x; }

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    double f = (t - 1) / 9.0;

    int nA = max(1, (int)llround(1 + f * 8));    // 2..9 regular per site
    int nB = nA;
    H = (int)llround(80 + f * 260);              // 80..340
    W = (t % 2 == 0) ? 3 : (3 + (t % 5));         // varies 3..7ish, deterministic in t
    if (W < 2) W = 2;
    if (W > 10) W = 10;
    Qd = 1000;
    Pn = 970 - (ll)llround(f * 140);              // 970..830  (mild -> stronger decay)
    if (Pn < 700) Pn = 700;
    if (Pn >= Qd) Pn = Qd - 1;

    int numAnchors = max(1, (int)llround(1 + f * 6)); // 1..7 freebie pairs

    vector<Tel> tel;
    vector<Tgt> tgt;
    set<int> usedPos;

    auto freshPos = [&]() {
        int p;
        int tries = 0;
        do { p = rnd.next(0, 359); tries++; } while (usedPos.count(p) && tries < 200);
        usedPos.insert(p);
        return p;
    };

    // ---- anchor pairs: dedicated zero-travel freebies (scales with size) ----
    // Every target position placed below is checked against `usedPos` (which
    // accumulates BOTH telescope start positions and target positions) so no
    // two distinct targets -- or a target and an unrelated telescope's park
    // spot -- ever land on the exact same azimuth. Without this, a later
    // random target could coincidentally re-land on an anchor's freebie
    // azimuth and silently inflate the baseline B beyond what any single
    // (telescope-pair, target) commitment can actually collect.
    for (int k = 0; k < numAnchors; k++) {
        int FP = freshPos();
        int freebieV = 900 + rnd.next(0, 700);
        tel.push_back({0, FP, 5 + rnd.next(0, 20)});
        tel.push_back({1, FP, 5 + rnd.next(0, 20)});
        tgt.push_back({0, FP, freebieV, 2});
    }

    // ---- obvious rendezvous: exact-synced scouts ----
    int OP = freshPos();
    int obvV = 2400 + rnd.next(0, 900);
    {
        int spd0 = 6 + rnd.next(0, 16);
        int R0 = 2 + rnd.next(0, 4);
        int dist0 = spd0 * R0;
        if (dist0 > 175) { R0 = 175 / spd0; if (R0 < 1) R0 = 1; dist0 = spd0 * R0; }
        int pos0 = norm360(OP - dist0);
        usedPos.insert(pos0);
        tel.push_back({0, pos0, spd0});

        int spd1 = 6 + rnd.next(0, 16);
        int dist1 = spd1 * R0; // SAME R0 -> exact tick match regardless of W
        if (dist1 > 175) { spd1 = max(6, 175 / R0); dist1 = spd1 * R0; }
        int pos1 = norm360(OP + dist1);
        usedPos.insert(pos1);
        tel.push_back({1, pos1, spd1});
    }
    tgt.push_back({0, OP, obvV, 3});

    // ---- regular clusters + personal lures ----
    vector<int> idxA, idxB;
    for (int k = 0; k < nA; k++) {
        int pos, tries = 0;
        do { pos = norm360(0 + rnd.next(-30, 30)); tries++; }
        while (usedPos.count(pos) && tries < 200);
        usedPos.insert(pos);
        int spd = 4 + rnd.next(0, 22);
        idxA.push_back((int)tel.size());
        tel.push_back({0, pos, spd});
        int lureV = 550 + rnd.next(0, 1100);
        tgt.push_back({0, pos, lureV, 2 + rnd.next(0, 2)});
    }
    for (int k = 0; k < nB; k++) {
        int pos, tries = 0;
        do { pos = norm360(180 + rnd.next(-30, 30)); tries++; }
        while (usedPos.count(pos) && tries < 200);
        usedPos.insert(pos);
        int spd = 4 + rnd.next(0, 22);
        idxB.push_back((int)tel.size());
        tel.push_back({1, pos, spd});
        int lureV = 550 + rnd.next(0, 1100);
        tgt.push_back({0, pos, lureV, 2 + rnd.next(0, 2)});
    }

    // ---- hidden rendezvous: plant by searching over (u,v) pairs + azimuth ----
    int hidWanted = (int)llround(2 + f * 14); // 2..16
    int attempts = 0, planted = 0;
    while (planted < hidWanted && attempts < hidWanted * 40 + 200) {
        attempts++;
        int P = rnd.next(0, 359);
        if (usedPos.count(P)) continue;
        int u = idxA[rnd.next(0, (int)idxA.size() - 1)];
        int v = idxB[rnd.next(0, (int)idxB.size() - 1)];
        int Ru = travelTicks(tel[u].pos, P, tel[u].speed);
        int Rv = travelTicks(tel[v].pos, P, tel[v].speed);
        if (Ru == 0 || Rv == 0) continue;      // degenerate / collides with a lure spot
        if (abs(Ru - Rv) > W) continue;
        if (max(Ru, Rv) >= H - 6) continue;    // must leave room to actually observe
        int hv = 1300 + rnd.next(0, 1500);
        tgt.push_back({0, P, hv, 2 + rnd.next(0, 2)});
        usedPos.insert(P);
        planted++;
    }

    // ---- small filler noise: low value, scattered, various appear times ----
    int fillerN = 3 + rnd.next(0, 3 + t);
    for (int i = 0; i < fillerN; i++) {
        int P, tries = 0;
        do { P = rnd.next(0, 359); tries++; } while (usedPos.count(P) && tries < 50);
        if (usedPos.count(P)) continue; // give up this slot rather than collide
        usedPos.insert(P);
        int a = rnd.next(0, max(1, H / 3));
        int v = 40 + rnd.next(0, 250);
        int o = 2 + rnd.next(0, 3);
        if (a + o > H - 1) a = max(0, H - 1 - o);
        tgt.push_back({a, P, v, o});
    }

    int T = (int)tel.size();
    int M = (int)tgt.size();

    // ---- emit ----
    string out;
    out.reserve((size_t)(T + M) * 20 + 64);
    char line[64];
    int hl = sprintf(line, "%d %d %d %d %lld %lld\n", T, M, H, W, Pn, Qd);
    out.append(line, hl);
    for (int i = 0; i < T; i++) {
        hl = sprintf(line, "%d %d %d\n", tel[i].site, tel[i].pos, tel[i].speed);
        out.append(line, hl);
    }
    for (int j = 0; j < M; j++) {
        hl = sprintf(line, "%d %d %d %d\n", tgt[j].a, tgt[j].pos, tgt[j].v, tgt[j].o);
        out.append(line, hl);
    }
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
