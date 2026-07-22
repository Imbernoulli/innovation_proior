#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Retensioning the Ground-Braced Mast Array"   family: tensegrity-prestress-shaping
//
// A 1-D tensegrity: node 0 is the ground anchor (position 0, fixed). Every other
// node is added incrementally:
//   - STRUT node: rigidly welded (fixed signed offset d) to an earlier GROUNDED
//     node -> joins the grounded (rigid) backbone. Its position is FOREVER
//     exactly target[node] (struts are not tunable).
//   - FREE node: a single-DOF mast tip. It is NOT welded to anything; instead it
//     is later given two elastic CABLES to two DIFFERENT grounded backbone nodes
//     (one with smaller target position "L", one with larger target position "R"
//     -- both are guaranteed to exist because node 0 is always grounded at the
//     low end and node n-1 is FORCED to be a strut/grounded node at the high end).
//     The solver must choose this free node's two cable rest lengths.
//
// A few extra DECOY cables (free node -> a third grounded node) are sprinkled in:
// a naive solver tensions them too (for nothing); the intended solution safely
// slacks them.
//
// PLANTED TRAP (from testId 3 on): the two structural cables of many free nodes
// use deliberately MISMATCHED stiffness (k_L vs k_R differing by >=10), and from
// testId 5/7/9 on some gaps are made deliberately tiny (1..3). A solver that
// tensions "gap minus a fixed margin" on every cable (ignoring stiffness) creates
// an unbalanced force at the target shape and drifts off it; on tiny gaps the
// margin can even clamp to zero, causing large distortions.
//
// Output (of gen): one test case per call to stdout, format:
//   n s c
//   target_0 ... target_{n-1}
//   s lines: u v d        (strut: x_v - x_u = d exactly)
//   c lines: u v k        (cable: stiffness k; solver outputs its rest length)
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n = 6 + 12 * (testId - 1);           // 6 .. 114
    double strutProb = 0.35;
    bool asymTrap  = (testId >= 3);          // stiffness-mismatch trap active
    bool tinyTrap  = (testId == 5 || testId == 7 || testId == 9 || testId == 10);
    bool wantDecoy = (testId >= 2);

    vector<ll> target(n);
    target[0] = 0;
    for (int i = 1; i < n; i++) {
        ll gap;
        if (tinyTrap && rnd.next(0, 99) < 35) gap = rnd.next(1, 3);
        else gap = rnd.next(8, 35);
        target[i] = target[i - 1] + gap;
    }

    vector<int> parent(n);
    for (int i = 0; i < n; i++) parent[i] = i;
    function<int(int)> find = [&](int x) { while (parent[x] != x) x = parent[x] = parent[parent[x]]; return x; };
    vector<char> grounded(n, 0);
    grounded[0] = 1;
    vector<int> groundedList = {0};

    vector<array<ll,3>> struts;   // u v d
    vector<char> isFree(n, 0);

    for (int v = 1; v < n; v++) {
        bool forceStrut = (v == n - 1);
        bool makeStrut = forceStrut || (rnd.next(0, 99) < (int)(strutProb * 100));
        if (makeStrut) {
            int u = groundedList[rnd.next(0, (int)groundedList.size() - 1)];
            ll d = target[v] - target[u];
            struts.push_back({(ll)u, (ll)v, d});
            parent[v] = find(u);
            grounded[v] = 1;
            groundedList.push_back(v);
        } else {
            isFree[v] = 1;
        }
    }

    // Pass 2: give every free node its nearest grounded L/R neighbor.
    vector<array<ll,3>> cables;  // u v k  (u,v node ids; exactly one endpoint grounded... here: u=free,v=grounded by convention but we just store the pair)
    for (int v = 1; v < n; v++) {
        if (!isFree[v]) continue;
        int uL = -1, uR = -1;
        for (int g : groundedList) {
            if (target[g] < target[v] && (uL == -1 || target[g] > target[uL])) uL = g;
            if (target[g] > target[v] && (uR == -1 || target[g] < target[uR])) uR = g;
        }
        // node 0 guarantees uL exists (target 0 < target[v] for v>=1);
        // node n-1 (forced grounded) guarantees uR exists.
        ll kL, kR;
        if (asymTrap && rnd.next(0, 99) < 65) {
            if (rnd.next(0, 1) == 0) { kL = rnd.next(1, 5);  kR = rnd.next(15, 20); }
            else                     { kL = rnd.next(15, 20); kR = rnd.next(1, 5); }
        } else {
            kL = rnd.next(4, 18);
            kR = rnd.next(4, 18);
        }
        cables.push_back({(ll)v, (ll)uL, kL});
        cables.push_back({(ll)v, (ll)uR, kR});

        if (wantDecoy && groundedList.size() >= 3 && rnd.next(0, 99) < 18) {
            int ux;
            for (int tries = 0; tries < 20; tries++) {
                int cand = groundedList[rnd.next(0, (int)groundedList.size() - 1)];
                if (cand != uL && cand != uR) { ux = cand; break; }
                ux = -1;
            }
            if (ux != -1) {
                ll kx = rnd.next(1, 20);
                cables.push_back({(ll)v, (ll)ux, kx});
            }
        }
    }

    // Shuffle cable order (structure must still be recoverable from k / target, not order)
    for (int i = (int)cables.size() - 1; i > 0; i--) swap(cables[i], cables[rnd.next(0, i)]);

    int s = (int)struts.size();
    int c = (int)cables.size();
    printf("%d %d %d\n", n, s, c);
    for (int i = 0; i < n; i++) printf("%lld%c", target[i], i + 1 == n ? '\n' : ' ');
    for (auto &e : struts) printf("%lld %lld %lld\n", e[0], e[1], e[2]);
    for (auto &e : cables) printf("%lld %lld %lld\n", e[0], e[1], e[2]);
    return 0;
}
