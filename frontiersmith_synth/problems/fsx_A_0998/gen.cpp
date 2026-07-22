#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Ridgeline Valley: Planting the New Rootstock"   family: zealot-majority-tipping
//
// n households, m trust ties (weighted, undirected). Household i farms p_i
// acres, starts on rootstock s_i in {0=OLD,1=NEW}, and costs c_i to recruit
// as a permanent "seed grower" (zealot, fixed to NEW forever). Every season
// for R seasons every non-zealot household SIMULTANEOUSLY re-weighs its
// trusted neighbors' PREVIOUS-season choice (weighted majority; exact ties
// keep last season's choice) and updates. Solver picks a zealot set; the
// checker simulates R seasons and requires >= tau% of total ACREAGE on NEW
// by the end; objective is to minimize total recruitment cost c_i summed
// over the zealot set.
//
// PLANTED STRUCTURE / TRAP (innovation hook): a household's vote is only as
// "buyable" as the fraction of ITS OWN trust-weight a single tie commands.
// A HUB household has huge weighted degree (many ties) but each tie is a
// small share of each neighbor's OWN degree (neighbors are themselves
// decently connected background households) -- so seeding the hub, even
// though the hub itself is cheap, buys almost no cascade: the hub's many
// neighbors are barely moved by the one hub tie and stay OLD, and flipping
// the hub itself contributes negligible acreage (hub's own acreage is
// small). A DENSE POCKET (a small near-clique of high-acreage households
// whose ties are almost entirely to EACH OTHER, hence individually
// low-degree) is the opposite: each tie inside the pocket is a LARGE share
// of a pocket member's degree, so seeding roughly half the pocket already
// gives every remaining member a same-season majority -> the whole pocket
// self-sustains NEW after 1-2 seasons, for cheap, and it carries real
// acreage. A degree-first ("recruit the best-connected household") recipe
// is drawn straight to the hub and wastes its budget; a leverage-aware
// strategy nucleates pockets instead. Trap tests (5..8) make pocket acreage
// alone comfortably clear the tau target while a hub-first spend does not.
// -----------------------------------------------------------------------------

struct Node { int p, c, s; };

static void addEdgeIfNew(int u, int v, int w, vector<array<int,3>> &edges,
                          set<pair<int,int>> &seen){
    if (u == v) return;
    int a = min(u, v), b = max(u, v);
    if (seen.count({a, b})) return;
    seen.insert({a, b});
    edges.push_back({u, v, w});
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    // ---- overall scale ladder ----
    int n;
    bool trapMode;      // dominant hub + high-acreage pockets that alone clear tau
    int numPockets, pocketLo, pocketHi;
    int tau, R;
    double pocketFracTarget;   // desired share of total acreage sitting in pockets
    bool hasHub;

    if (testId == 1){
        n = 9; trapMode = false; numPockets = 1; pocketLo = 4; pocketHi = 4;
        tau = 60; R = 5; pocketFracTarget = 0.45; hasHub = false;
    } else if (testId <= 4){
        // medium random ramps: no dominant trick, general capability check
        n = 30 + (int)llround(f * 170);              // ~30..200ish across 2..4
        trapMode = false;
        numPockets = 2 + (testId - 2);
        pocketLo = 5; pocketHi = 9;
        tau = 55 + 5 * (testId - 2);
        R = 6 + (testId - 2) * 2;
        pocketFracTarget = 0.40 + 0.05 * (testId - 2);
        hasHub = (testId >= 3);
    } else if (testId <= 7){
        // TRAP zone: dominant cheap hub, several high-acreage pockets that
        // alone comfortably clear tau; degree-first recruiting is misled.
        n = 250 + (testId - 5) * 200;                  // 250, 450, 650
        trapMode = true;
        numPockets = 4 + (testId - 5);
        pocketLo = 7; pocketHi = 12;
        tau = 82 + 3 * (testId - 5);
        R = 8 + 2 * (testId - 5);
        pocketFracTarget = 0.90;
        hasHub = true;
    } else {
        // large scale: fill the constraint envelope, keep trap pressure
        n = 1200 + (testId - 8) * 900;                 // 1200, 2100, 3000..up to ~3900
        if (testId == 10) n = 3900;
        trapMode = true;
        numPockets = 8 + (testId - 8) * 3;
        pocketLo = 8; pocketHi = 14;
        tau = 80 + 2 * (testId - 8);
        R = 14 + 3 * (testId - 8);
        pocketFracTarget = 0.85;
        hasHub = true;
    }
    R = min(R, 40);

    // ---- assign nodes: pockets first, then hub (if any), then background ----
    vector<vector<int>> pockets(numPockets);
    vector<int> pocketOf(n + 1, -1);   // -1 = not a pocket member
    int nextId = 1;
    for (int k = 0; k < numPockets; k++){
        int sz = pocketLo + rnd.next(0, max(0, pocketHi - pocketLo));
        for (int t = 0; t < sz && nextId <= n; t++){
            pockets[k].push_back(nextId);
            pocketOf[nextId] = k;
            nextId++;
        }
    }
    int hubId = -1;
    if (hasHub && nextId <= n){ hubId = nextId; nextId++; }
    vector<int> background;
    for (; nextId <= n; nextId++) background.push_back(nextId);

    // ---- populations (acreage) ----
    vector<int> p(n + 1, 1), c(n + 1, 1), s(n + 1, 0);
    ll pocketPopSum = 0;
    for (int k = 0; k < numPockets; k++)
        for (int v : pockets[k]){
            p[v] = (trapMode ? 55 : 30) + rnd.next(0, 35);   // high acreage
            c[v] = 4 + rnd.next(0, 8);                        // moderately cheap
            pocketPopSum += p[v];
        }
    if (hubId != -1){
        p[hubId] = 1 + rnd.next(0, 4);      // tiny acreage: converting it wins little
        c[hubId] = 1 + rnd.next(0, 3);      // deliberately CHEAP: tempts a degree-greedy
    }
    // background acreage sized so pocketPopSum / totalPop ~= pocketFracTarget
    int nBg = (int)background.size();
    if (nBg > 0){
        double wanted = max(1.0, pocketPopSum / max(0.05, pocketFracTarget) - pocketPopSum
                             - (hubId != -1 ? p[hubId] : 0));
        int perNode = max(1, (int)llround(wanted / nBg));
        for (int v : background){
            int jitter = rnd.next(-min(perNode - 1, 4), 4);
            p[v] = max(1, min(200, perNode + jitter));
            c[v] = 5 + rnd.next(0, 12);
        }
    }

    // ---- initial opinions: mostly OLD, a light sprinkle of early NEW adopters ----
    for (int v = 1; v <= n; v++) s[v] = (rnd.next(0, 99) < 6) ? 1 : 0;

    // ---- edges ----
    vector<array<int,3>> edges;
    set<pair<int,int>> seen;

    // pockets: near-clique (random graph, edge prob ~0.65), high weight, few internal
    for (int k = 0; k < numPockets; k++){
        auto &grp = pockets[k];
        int sz = (int)grp.size();
        for (int i = 0; i < sz; i++)
            for (int j = i + 1; j < sz; j++)
                if (rnd.next(0, 99) < 65)
                    addEdgeIfNew(grp[i], grp[j], 3 + rnd.next(0, 4), edges, seen);
        // make sure pocket is internally connected even if random draw was sparse
        for (int i = 1; i < sz; i++)
            addEdgeIfNew(grp[i-1], grp[i], 3 + rnd.next(0, 4), edges, seen);
    }

    // background: ring (min degree 2, connected) + random chords, moderate weight
    int nb = (int)background.size();
    if (nb >= 3){
        for (int i = 0; i < nb; i++)
            addEdgeIfNew(background[i], background[(i + 1) % nb], 1 + rnd.next(0, 2), edges, seen);
        int extraChords = nb * 2;
        for (int t = 0; t < extraChords; t++){
            int a = background[rnd.next(0, nb - 1)];
            int b = background[rnd.next(0, nb - 1)];
            addEdgeIfNew(a, b, 1 + rnd.next(0, 2), edges, seen);
        }
    } else if (nb == 2){
        addEdgeIfNew(background[0], background[1], 1 + rnd.next(0, 2), edges, seen);
    }

    // hub: many LOW-weight ties into a wide, random slice of the background
    // (its neighbors keep their own moderate degree from the ring/chords
    // above, so a single hub tie is a small share of a neighbor's degree).
    if (hubId != -1 && nb > 0){
        int hubDeg = trapMode ? max(6, nb * 2 / 3) : max(3, nb / 3);
        hubDeg = min(hubDeg, nb);
        vector<int> pool = background;
        for (int i = (int)pool.size() - 1; i > 0; i--) swap(pool[i], pool[rnd.next(0, i)]);
        for (int t = 0; t < hubDeg; t++)
            addEdgeIfNew(hubId, pool[t], 1, edges, seen);
    }

    // a few narrow bridges from each pocket into the background, so a
    // flipped pocket can (but need not) nucleate a little further out
    for (int k = 0; k < numPockets; k++){
        if (nb == 0) continue;
        int bridges = 1 + rnd.next(0, 1);
        for (int t = 0; t < bridges; t++){
            int a = pockets[k][rnd.next(0, (int)pockets[k].size() - 1)];
            int b = background[rnd.next(0, nb - 1)];
            addEdgeIfNew(a, b, 1 + rnd.next(0, 1), edges, seen);
        }
    }

    // light extra background-background noise for larger tests
    if (testId >= 8 && nb > 4){
        int extra = nb / 2;
        for (int t = 0; t < extra; t++){
            int a = background[rnd.next(0, nb - 1)];
            int b = background[rnd.next(0, nb - 1)];
            addEdgeIfNew(a, b, 1 + rnd.next(0, 2), edges, seen);
        }
    }

    int m = (int)edges.size();
    printf("%d %d %d %d\n", n, m, R, tau);
    for (int v = 1; v <= n; v++) printf("%d %d %d\n", p[v], c[v], s[v]);
    for (auto &e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
