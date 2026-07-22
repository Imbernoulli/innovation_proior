#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "The Oasis Cooperative's Fair Ditch Web"   family: seepage-fair-irrigation-web
//
// Node 0 = spring. Junction nodes 1..J are the routing backbone (a random
// spanning tree over spring+junctions plus extra redundant "loop" edges).
// Field nodes J+1..J+F each attach via EXACTLY ONE inlet edge (degree 1 in the
// whole candidate graph) so a field can never be a mid-route pass-through node.
//
// PLANTED TRAPS (>=3 of 10 tests):
//   - CHAIN trap (t=3,4,9,10): one field hangs off a long sequential (degree-2)
//     junction chain with NO shortcut -- multiplicative retention compounds over
//     many hops, and a naive equal-split-at-forks allocation starves it further.
//   - RETENTION-VS-LENGTH trap (t=5,6,9,10): a field is reachable via a SHORT
//     but low-retention ("bad soil") edge, and ALSO via a LONGER chain of
//     high-retention edges that nets a higher delivered fraction despite more
//     ditch length -- the length-shortest route is a trap for a solver that
//     equates "shortest network" with "best route".
//   - CAPACITY trap (t=7,8, and mildly present everywhere): CAP is small
//     relative to D, so no single spring-side branch can carry all of D --
//     multiple parallel branches ("redundant loops") out of the spring are
//     required to actually use most of the discharge.
// -----------------------------------------------------------------------------

struct Edge { int u, v; ll len; int ret; };
vector<Edge> E;
set<pair<int,int>> seenPairs;
int nextNode = 1; // 0 reserved for spring

int newNode(){ return nextNode++; }

bool addEdgeSafe(int u, int v, ll len, int ret){
    if (u == v) return false;
    int a = min(u, v), b = max(u, v);
    if (seenPairs.count({a, b})) return false;
    seenPairs.insert({a, b});
    E.push_back({u, v, len, ret});
    return true;
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    double f = (t - 1) / 9.0;

    // ---- parameters by testId ----
    int baseJ = 5 + (int)llround(f * 45.0);              // 5..50 backbone junctions
    int extra = 3 + (int)llround(f * 15.0);               // extra redundant backbone edges
    ll D = 2500 + 450LL * t;
    bool capTight = (t == 7 || t == 8);
    ll CAP = capTight ? (ll)llround(D / 5.2) : (ll)llround(D / 2.3);
    if (CAP < 20) CAP = 20;

    int Farr[11] = {0, 2, 3, 4, 5, 4, 5, 5, 6, 8, 10};
    int F = Farr[t];

    bool chainTrap = (t == 3 || t == 4 || t == 9 || t == 10);
    bool retTrap   = (t == 5 || t == 6 || t == 9 || t == 10);
    int chainLen = chainTrap ? (t <= 4 ? 10 + 3 * (t - 3) : 14 + 2 * (t - 9)) : 0; // t3:10 t4:13 t9:14 t10:16
    int altLen   = retTrap   ? (t <= 6 ? 4 + 2 * (t - 5) : 6 + 2 * (t - 9)) : 0;   // t5:4 t6:6 t9:6 t10:8

    // ---- 1) backbone: random spanning tree over spring + baseJ junctions ----
    vector<int> pool; pool.push_back(0);
    for (int i = 0; i < baseJ; i++){
        int id = newNode();
        int par = pool[rnd.next(0, (int)pool.size() - 1)];
        ll len = 2 + rnd.next(0, 4);              // 2..6
        int ret = 880 + rnd.next(0, 110);          // 880..990
        addEdgeSafe(par, id, len, ret);
        pool.push_back(id);
    }
    // ---- 2) extra redundant backbone edges (loops) among pool only ----
    for (int i = 0; i < extra; i++){
        if ((int)pool.size() < 2) break;
        int a = pool[rnd.next(0, (int)pool.size() - 1)];
        int b = pool[rnd.next(0, (int)pool.size() - 1)];
        ll len = 2 + rnd.next(0, 6);
        int ret = 860 + rnd.next(0, 120);
        addEdgeSafe(a, b, len, ret);
    }
    // ensure spring has enough distinct candidate edges for genuine multi-branch use
    int minSpringDeg = capTight ? 6 : 4;
    {
        int deg0 = 0;
        for (auto &e : E) if (e.u == 0 || e.v == 0) deg0++;
        int tries = 0;
        while (deg0 < minSpringDeg && tries < 200){
            tries++;
            int b = pool[rnd.next(1, (int)pool.size() - 1)];
            ll len = 2 + rnd.next(0, 4);
            int ret = 880 + rnd.next(0, 100);
            if (addEdgeSafe(0, b, len, ret)) deg0++;
        }
    }

    ll extraLenReserve = 0;
    int fieldJunction[11]; // which junction each field (0-indexed) attaches to
    ll fieldInletLen[11]; int fieldInletRet[11];
    int nextFieldSlot = 0;
    int chainFieldSlot = -1, retFieldSlot = -1;
    int forceShareNode = -1; // a junction another field must also route through,
                              // so the chain-trap field never gets a private branch

    // ---- 3) chain trap: sequential degree-2 junction chain, forced route ----
    // The chain must NOT start at the spring itself (pool[0]): that would hand the
    // chain field its own private, undiluted spring branch under a length-shortest
    // routing, defeating the trap. pool[1] is guaranteed spring-adjacent (it is
    // the very first backbone node, whose only possible parent was pool[0]), so
    // starting there -- and forcing a normal field to also route through it below
    // -- guarantees the branch is shared and the equal-split trap actually bites.
    if (chainTrap){
        chainFieldSlot = nextFieldSlot++;
        int start = pool[1];
        forceShareNode = start;
        int prev = start;
        for (int i = 0; i < chainLen; i++){
            int cur = newNode();
            ll len = 2 + rnd.next(0, 2);           // 2..4
            int ret = 900 + rnd.next(0, 70);        // 900..970
            addEdgeSafe(prev, cur, len, ret);
            extraLenReserve += len;
            prev = cur;
        }
        fieldJunction[chainFieldSlot] = prev;
        fieldInletLen[chainFieldSlot] = 1 + rnd.next(0, 1);
        fieldInletRet[chainFieldSlot] = 900 + rnd.next(0, 70);
        extraLenReserve += fieldInletLen[chainFieldSlot];
    }

    // ---- 4) retention-vs-length trap: short-bad edge + long-clean alt path ----
    if (retTrap){
        retFieldSlot = nextFieldSlot++;
        int hubA = pool[rnd.next(0, min(4, (int)pool.size() - 1))];
        int hubB = newNode();
        // short but lossy direct edge
        addEdgeSafe(hubA, hubB, 2, 500 + rnd.next(0, 150)); // len 2, ret 500..650
        // longer chain of clean edges from hubA to hubB (parallel alt route)
        int prev = hubA;
        ll altTotalLen = 0;
        for (int i = 0; i < altLen; i++){
            int cur = (i == altLen - 1) ? hubB : newNode();
            ll len = 3 + rnd.next(0, 2);            // 3..5
            int ret = 950 + rnd.next(0, 35);         // 950..985
            if (cur != hubB || i == altLen - 1){
                if (addEdgeSafe(prev, cur, len, ret)) altTotalLen += len;
            }
            prev = cur;
        }
        extraLenReserve += altTotalLen;
        fieldJunction[retFieldSlot] = hubB;
        fieldInletLen[retFieldSlot] = 1 + rnd.next(0, 1);
        fieldInletRet[retFieldSlot] = 900 + rnd.next(0, 70);
        extraLenReserve += fieldInletLen[retFieldSlot];
    }

    // ---- 5) remaining "normal" (easy/control) fields: short inlet near spring ----
    bool sharedForced = false;
    while (nextFieldSlot < F){
        int slot = nextFieldSlot++;
        int j;
        if (forceShareNode != -1 && !sharedForced){
            // guarantee at least one other field's shortest route shares the
            // trap field's spring branch, so a length/hop-based grouping must
            // split that branch instead of handing it over privately.
            j = forceShareNode;
            sharedForced = true;
        } else {
            int nearCount = min(8, (int)pool.size());
            j = pool[rnd.next(0, nearCount - 1)];
        }
        fieldJunction[slot] = j;
        fieldInletLen[slot] = 1 + rnd.next(0, 2);       // 1..3
        fieldInletRet[slot] = 900 + rnd.next(0, 85);     // 900..985
    }

    // ---- 6) materialize field nodes + inlet edges ----
    vector<int> fieldIds(F);
    for (int i = 0; i < F; i++){
        int fid = newNode();
        fieldIds[i] = fid;
        bool ok = addEdgeSafe(fieldJunction[i], fid, fieldInletLen[i], fieldInletRet[i]);
        if (!ok){
            // extremely unlikely (duplicate inlet target already used exactly);
            // nudge the length to keep the pair distinct in the edge-set.
            addEdgeSafe(fieldJunction[i], fid, fieldInletLen[i] + 1, fieldInletRet[i]);
        }
    }

    int N = nextNode;
    int M = (int)E.size();

    // ---- 7) size L: BFS hop-shortest union (what 'trivial' needs) + explicit
    //         reserve for the trap structures' extra length, generous factor. ----
    vector<vector<pair<int,int>>> adj(N); // (neighbor, edgeIdx)
    for (int i = 0; i < M; i++){
        adj[E[i].u].push_back({E[i].v, i});
        adj[E[i].v].push_back({E[i].u, i});
    }
    vector<int> parentEdge(N, -1), dist(N, -1);
    queue<int> q; q.push(0); dist[0] = 0;
    while (!q.empty()){
        int u = q.front(); q.pop();
        for (auto &pr : adj[u]){
            int v = pr.first, eid = pr.second;
            if (dist[v] == -1){ dist[v] = dist[u] + 1; parentEdge[v] = eid; q.push(v); }
        }
    }
    vector<char> used(M, 0);
    ll unionLen = 0;
    for (int i = 0; i < F; i++){
        int v = fieldIds[i];
        while (v != 0 && parentEdge[v] != -1){
            int eid = parentEdge[v];
            if (!used[eid]){ used[eid] = 1; unionLen += E[eid].len; }
            v = (E[eid].u == v) ? E[eid].v : E[eid].u;
        }
    }
    ll L = (ll)ceil(2.8 * (double)unionLen) + extraLenReserve + 40;

    // ---- print ----
    printf("%d %d %d %lld %lld %lld\n", N, M, F, D, CAP, L);
    for (int i = 0; i < F; i++) printf("%d%c", fieldIds[i], i + 1 == F ? '\n' : ' ');
    for (auto &e : E) printf("%d %d %lld %d\n", e.u, e.v, e.len, e.ret);
    return 0;
}
