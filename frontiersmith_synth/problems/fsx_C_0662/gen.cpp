#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Plates that split one cut two ways" (shared-kerf-pairing).
//
// Model: N plate parts, each with a heat-affected-zone budget H_i. M "common
// lines": a common line g has member parts S_g (parts whose edge lies on that
// physical line) and a nominal cut length L_g (even). If a part cuts its edge
// SOLO it costs L_g/2 heat; if it joins the shared cut used by >=2 members of
// S_g it costs the FULL L_g heat (but the sheet only pays for the cut once, so
// the shared cut, if it recruits t members, saves (t-1)*L_g of total cut
// length). Every part must eventually cut every line it lies on -- solo unless
// it opts in to sharing -- so the *baseline* heat use is
//   baseline_i = sum over lines g with i in S_g of L_g/2,
// and choosing to SHARE line g costs an EXTRA L_g/2 on top of baseline. We
// therefore hand each part its "extra" budget R_i directly; the true H_i
// printed is baseline_i + R_i (so R_i is always feasible: doing nothing is
// always legal). This lets us plant tight vs. generous R_i independent of how
// many lines a part happens to sit on.
// -----------------------------------------------------------------------------

vector<ll> R;                       // R[0] unused, R[i] = extra ("shareable") budget of part i
vector<pair<vector<int>, ll>> G;    // groups: (members, L)

int addPart(ll r){ R.push_back(r); return (int)R.size() - 1; }
void addGroup(vector<int> members, ll L){ G.push_back({members, L}); }

// A "trap" cluster: 2 hub parts, each linked to cPerHub private leaf parts by a
// cheap pair-line (L = Lsmall), plus ONE expensive line joining the two hubs
// directly (L = Lgiant = cPerHub * Lsmall). Each hub's extra budget is tuned to
// exactly cover ALL of its own cheap pair-lines (total win = 2*cPerHub*Lsmall)
// OR the single expensive line alone (win = Lgiant = cPerHub*Lsmall) -- never
// both. Splitting into many cheap shared contacts beats grabbing the single
// most valuable line by a clean factor of ~2.
void addTrapCluster(int cPerHub, ll Lsmall){
    ll Rhub = (ll)cPerHub * (Lsmall / 2) + 5;
    int h1 = addPart(Rhub);
    int h2 = addPart(Rhub);
    ll Lgiant = 2 * (Rhub - 5); // = cPerHub * Lsmall, even
    addGroup({h1, h2}, Lgiant);
    for (int t = 0; t < cPerHub; t++){ int leaf = addPart(Lsmall / 2); addGroup({h1, leaf}, Lsmall); }
    for (int t = 0; t < cPerHub; t++){ int leaf = addPart(Lsmall / 2); addGroup({h2, leaf}, Lsmall); }
}

// A block of N0 freshly-created "filler" parts with random slack in [rlo,rhi].
vector<int> addFillerParts(int N0, ll rlo, ll rhi){
    vector<int> ids;
    for (int i = 0; i < N0; i++) ids.push_back(addPart(rnd.next(rlo, rhi)));
    return ids;
}

// A random group over an existing pool of part ids.
void addRandomGroup(const vector<int>& pool, int kmin, int kmax, ll Lmin, ll Lmax){
    int k = min((int)pool.size(), rnd.next(kmin, kmax));
    if (k < 2) return;
    vector<int> idx = pool;
    shuffle(idx.begin(), idx.end());
    vector<int> members(idx.begin(), idx.begin() + k);
    ll L = rnd.next((int)(Lmin / 2), (int)(Lmax / 2)) * 2;
    if (L < 2) L = 2;
    addGroup(members, L);
}

void printInstance(){
    int N = (int)R.size() - 1;
    int M = (int)G.size();
    vector<ll> baseline(N + 1, 0);
    for (auto &g : G) for (int m : g.first) baseline[m] += g.second / 2;
    printf("%d %d\n", N, M);
    for (int i = 1; i <= N; i++) printf("%lld%c", baseline[i] + R[i], i == N ? '\n' : ' ');
    for (auto &g : G){
        printf("%d %lld\n", (int)g.first.size(), g.second);
        for (size_t j = 0; j < g.first.size(); j++)
            printf("%d%c", g.first[j], j + 1 == g.first.size() ? '\n' : ' ');
    }
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    R.push_back(0); // index 0 unused

    if (testId == 1){
        // Tiny hand-built sanity case, mirrors the worked example in statement.txt.
        addPart(32); addPart(35); addPart(22); addPart(22); addPart(11); addPart(16);
        addGroup({1, 2}, 20);
        addGroup({1, 3, 4}, 16);
        addGroup({2, 5}, 10);
        addGroup({5, 6}, 8);
    } else if (testId == 2){
        // Small random instance.
        auto pool = addFillerParts(18, 30, 120);
        for (int i = 0; i < 14; i++) addRandomGroup(pool, 2, 4, 4, 60);
    } else if (testId == 3){
        // TRAP: 2 small clusters.
        addTrapCluster(6, 20);
        addTrapCluster(7, 22);
    } else if (testId == 4){
        // TRAP: 3 clusters, medium.
        addTrapCluster(8, 20);
        addTrapCluster(10, 24);
        addTrapCluster(9, 18);
    } else if (testId == 5){
        // PLANTED: several generous (non-trap) clusters where sharing everything is
        // fully affordable, mixed with decoy lines almost nobody can afford (k<2
        // eligible), to test that a solver isn't distracted by size alone.
        for (int c = 0; c < 3; c++){
            int cPerHub = 5 + c * 2;
            ll Lsmall = 16 + 4 * c;
            ll Rhub = (ll)cPerHub * Lsmall + 50; // generous: affords BOTH giant and all smalls
            int h1 = addPart(Rhub), h2 = addPart(Rhub);
            addGroup({h1, h2}, cPerHub * Lsmall);
            for (int t = 0; t < cPerHub; t++){ int leaf = addPart(Lsmall); addGroup({h1, leaf}, Lsmall); }
            for (int t = 0; t < cPerHub; t++){ int leaf = addPart(Lsmall); addGroup({h2, leaf}, Lsmall); }
        }
        auto pool = addFillerParts(20, 4, 10); // stingy filler pool
        for (int i = 0; i < 15; i++) addRandomGroup(pool, 3, 6, 400, 600); // decoys: nobody can afford
    } else if (testId == 6){
        // NEEDLE: one big, fully-affordable high-value cluster hidden among many
        // near-worthless tiny groups on a separate stingy pool.
        auto rich = addFillerParts(9, 2000, 2200);
        vector<int> needleMembers(rich.begin(), rich.end());
        addGroup(needleMembers, 900);
        auto poor = addFillerParts(120, 1, 2);
        for (int i = 0; i < 90; i++) addRandomGroup(poor, 2, 2, 2, 2);
    } else if (testId == 7){
        // TRAP: 4 clusters, larger.
        addTrapCluster(12, 20);
        addTrapCluster(14, 26);
        addTrapCluster(13, 22);
        addTrapCluster(15, 18);
    } else if (testId == 8){
        // Dense random, large.
        auto pool = addFillerParts(180, 12, 35);
        for (int i = 0; i < 40; i++) addRandomGroup(pool, 2, 6, 4, 80);
    } else if (testId == 9){
        // Sparse random, large.
        auto pool = addFillerParts(220, 25, 70);
        for (int i = 0; i < 26; i++) addRandomGroup(pool, 2, 3, 4, 120);
    } else {
        // testId 10: ADVERSARIAL LARGE, fills the size envelope: several trap
        // clusters of increasing size + dense random background + one needle.
        addTrapCluster(16, 20);
        addTrapCluster(18, 24);
        addTrapCluster(20, 22);
        addTrapCluster(17, 26);
        addTrapCluster(19, 18);
        auto rich = addFillerParts(8, 3000, 3200);
        addGroup(rich, 1400);
        auto pool = addFillerParts(80, 40, 160);
        for (int i = 0; i < 70; i++) addRandomGroup(pool, 2, 5, 4, 80);
    }

    printInstance();
    return 0;
}
