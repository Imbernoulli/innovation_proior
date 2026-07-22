#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Firebreak Grid generator  (family: cascade-capacity-hardening)
//
// The grid is built from independent CELLS. Each cell has:
//   * a BACKBONE chain  Be -> B1 -> ... (Be = entrance has base load, downstream
//     lines idle at base 0), every backbone line high value VB. Reroute along the
//     chain uses num=1000 (no shedding): if Be trips, the WHOLE backbone cascades.
//   * a POCKET line: high base load (bait), tiny value VP, NO reroute edges
//     (everything it carries is shed to GROUND). A perfect sacrificial firebreak.
//   * several TRIGGER lines: modest value VT; on trip each sends num_tb/1000 of its
//     load to the cell's backbone entrance Be and num_tp/1000 to the pocket.
//
// HARD cells have large trigger loads -> the surge onto Be overwhelms any capacity
// that merely tracks base load. EASY cells have small trigger loads -> they survive
// even the uniform grid, so the uniform baseline stays positive (and not tiny).
//
// TRAP for the obvious "capacity proportional to base load" heuristic: the pocket's
// huge base load lures a big proportional slice, which is wasted (the pocket still
// trips under the cascade and grounds anyway), starving the backbone entrance so it
// trips in the worst scenario -> the whole backbone is lost.
//
// INSIGHT (strong): sacrifice pockets (cap 0 -> they trip to ground harmlessly) and
// pour the freed budget into surge margin on the backbone entrances, so every
// cascade in the sweep dies in a low-value pocket instead of crossing the backbone.
//
// Budget B = total_base_load * 1.10  (covers every base load + a thin 10% margin).
// Proportional gets only a 10% surge cushion -> fails hard cells. The firebreak
// design frees the pockets' base load for surge -> protects all backbones.
// -----------------------------------------------------------------------------

static const ll DEN = 1000;
static const int VB = 1000;   // backbone value (dominates demand)
static const int VP = 30;     // pocket value (cheap to sacrifice)
static const int VT = 15;     // trigger value

struct Redge { int u, j, num; };

vector<ll> W;                 // base load per line
vector<ll> Vv;               // value per line
vector<Redge> edges;
vector<int> hardTrig;        // (index, load) collected for scenario selection
vector<ll>  hardTrigLoad;
vector<int> easyTrig;

int addLine(ll w, ll v){ W.push_back(w); Vv.push_back(v); return (int)W.size()-1; }

// Build one cell. hard => big triggers (dangerous surge). Returns nothing.
void buildCell(bool hard, int bc, int tr, int wB, int wP, int num_tb, int num_tp,
               int wtLo, int wtHi){
    // backbone chain
    vector<int> bb(bc);
    bb[0] = addLine(wB, VB);                 // entrance carries base load
    for (int i = 1; i < bc; i++) bb[i] = addLine(0, VB);   // idle reserves
    for (int i = 0; i + 1 < bc; i++) edges.push_back({bb[i], bb[i+1], (int)DEN}); // full forward
    // pocket (ground sink: no out edges)
    int pk = addLine(wP, VP);
    // triggers
    for (int k = 0; k < tr; k++){
        int wt = rnd.next(wtLo, wtHi);
        int t = addLine(wt, VT);
        edges.push_back({t, bb[0], num_tb});   // surge onto backbone entrance
        edges.push_back({t, pk,   num_tp});    // and onto the pocket
        if (hard){ hardTrig.push_back(t); hardTrigLoad.push_back(wt); }
        else       easyTrig.push_back(t);
    }
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- size ladder ----
    // The score is a MIN over scenarios; with independent cells the greedy-vs-strong
    // gap is essentially "one lost backbone", so the CELL COUNT must stay modest and
    // roughly constant (otherwise one backbone becomes negligible vs total demand and
    // both heuristics saturate). We grow size mainly through backbone length and
    // trigger count, keeping the demand-to-baseline ratio in a stable band.
    int Nh, Ne, needle, bcHard, bcEasy, trHard;
    if (testId <= 1){
        Nh = 3; Ne = 1; needle = 0; bcHard = 5; bcEasy = 6; trHard = 3;
    } else {
        Nh = 5 + (testId % 4);          // 5..8 hard cells
        Ne = 1 + (testId % 2);          // 1..2 easy anchor cells (baseline)
        bcHard = 5 + (testId % 5);      // 5..9 backbone length
        bcEasy = 7 + (testId % 4);      // 7..10 (easy backbone a bit longer)
        trHard = 4 + (testId % 3);      // 4..6 triggers per hard cell
        needle = (testId >= 3) ? 1 : 0; // an oversized backbone hidden among the rest
    }

    // build easy anchor cell(s): small triggers -> survive even the uniform grid,
    // keeping the uniform baseline positive and scaled with the grid.
    for (int c = 0; c < Ne; c++)
        buildCell(false, bcEasy, 3, 140, 250, 700, 200, 50, 130);

    // build hard cells: big triggers, big pocket bait
    for (int c = 0; c < Nh; c++){
        int wP = rnd.next(600, 800);
        buildCell(true, bcHard, trHard, 300, wP, 700, 200, 600, 950);
    }

    // needle cell: an extra-long, extra-valuable backbone hidden among the rest.
    if (needle)
        buildCell(true, bcHard + 6, 4, 300, 750, 700, 200, 650, 950);

    int L = (int)W.size();
    int M = (int)edges.size();

    // ---- budget: cover every base load + 10% margin ----
    ll totalBase = 0; for (int i = 0; i < L; i++) totalBase += W[i];
    ll B = totalBase + (totalBase / 10) + 1;

    // ---- scenarios: the most dangerous hard triggers (adversarial), capped at 50 ----
    // sort hard triggers by load desc, keep the biggest.
    vector<int> order(hardTrig.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
    sort(order.begin(), order.end(), [&](int a, int b){
        return hardTrigLoad[a] > hardTrigLoad[b];
    });
    vector<int> scen;
    for (int idx : order){ if ((int)scen.size() >= 50) break; scen.push_back(hardTrig[idx]); }
    // if very few hard triggers, top up with easy ones
    for (size_t i = 0; i < easyTrig.size() && (int)scen.size() < 50; i++)
        scen.push_back(easyTrig[i]);
    if (scen.empty()) scen.push_back(0);
    int S = (int)scen.size();

    // ---- emit ----
    printf("%d %d %d %lld\n", L, M, S, B);
    for (int i = 0; i < L; i++) printf("%lld %lld\n", W[i], Vv[i]);
    for (auto& e : edges) printf("%d %d %d\n", e.u, e.j, e.num);
    for (int i = 0; i < S; i++) printf("%d%c", scen[i], i + 1 < S ? ' ' : '\n');
    return 0;
}
