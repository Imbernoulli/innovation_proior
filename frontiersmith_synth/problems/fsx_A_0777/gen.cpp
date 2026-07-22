#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Weaving a Fast Gossip Network Across a Long Ridge of Villages"
// family: wirelength-expander-loom
//
// n = 2^k villages sit on a ridge (a line), at x-coordinates built by a
// recursive "fold" process: at the top, the whole ridge is split into a left
// half [0..n/2) and a right half [n/2..n), separated by a fold-gap of size
// GAP0 (this is the biggest fold on the ridge -- crossing it is expensive).
// Each half is then recursively split again, with fold-gaps that shrink by a
// factor of 2 every recursion level. Because a level-d split creates 2^d
// gaps of size GAP0/2^d, EVERY level of the hierarchy contributes exactly the
// same total length GAP0 to the ridge -- the classic self-similar / summable
// wirelength-budget structure the family is built around.
//
// The reference "hierarchical bridge" construction (function addCrossEdges)
// links, at every level of that same recursion, the two extreme villages of
// each half-pair with two edges: (leftmost of left half <-> rightmost of
// right half) and (rightmost of left half <-> leftmost of right half). This
// is a fixed, index-only topology (independent of the concrete coordinates)
// with max degree k and O(n) edges, connected by induction, whose exact
// wirelength Wc and max degree Dc are computed HERE so the degree cap d and
// wirelength budget W can be set (with a controlled slack) to exactly fit it
// -- by construction the strong solution is always feasible, with no
// hand-derived asymptotics required.
//
// TRAP (>=3 of 10 tests, sharpening with n): degree cap set to Dc+1 (almost
// no slack). A naive "connect nearby villages first" band/mesh strategy then
// gets its whole degree budget consumed by purely LOCAL edges (any two
// villages within the same fine cluster are cheap and plentiful) and never
// reaches a fold-crossing edge, so its network stays stuck at O(n/d) hop
// diameter while the hierarchical bridge constructions get O(log n) -- a
// regime change the local strategy cannot see, and it worsens as n grows.
// -----------------------------------------------------------------------------

static const ll GAP0 = 4096;
static const ll UNIT = 1;

int K;
int N;
vector<ll> X;
ll cursor;

void buildPos(int lo, int hi, int depth) {
    if (hi - lo == 1) { X[lo] = cursor; cursor += UNIT; return; }
    int mid = (lo + hi) / 2;
    buildPos(lo, mid, depth + 1);
    cursor += (GAP0 >> depth);
    buildPos(mid, hi, depth + 1);
}

void addCrossEdges(int lo, int hi, vector<pair<int,int>>& edges) {
    if (hi - lo <= 1) return;
    int mid = (lo + hi) / 2;
    if (hi - lo == 2) {
        edges.push_back({lo, mid});
    } else {
        edges.push_back({lo, hi - 1});
        edges.push_back({mid - 1, mid});
    }
    addCrossEdges(lo, mid, edges);
    addCrossEdges(mid, hi, edges);
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // k (=> n=2^k), degree slack, wirelength-budget slack multiplier (x1000),
    // and whether to carve a flat "needle" window into the ridge.
    static const int    KS[11]     = {0,3,4,4,5,5,6,6,7,7,8};
    static const int    SLACKD[11] = {0,6,6,1,6,1,6,1,6,1,1};
    static const int    SLACKW[11] = {0,1300,1300,1300,1300,1050,1300,1300,1050,1300,1100};
    static const int    NEEDLE[11] = {0,0,0,0,0,0,0,0,1,1,1};

    K = KS[testId];
    N = 1 << K;
    X.assign(N, 0);
    cursor = 0;
    buildPos(0, N, 0);

    if (NEEDLE[testId]) {
        // Flatten a random contiguous window of villages to a uniform 1-unit
        // spacing (destroying the fine hierarchy locally) while preserving
        // the total width of that window, seeded deterministically by testId.
        rnd.setSeed(900000 + testId);
        int winLen = max(4, N / 8);
        int start = rnd.next(0, N - winLen - 1);
        ll lo = X[start], hi = X[start + winLen];
        ll span = hi - lo;
        for (int t = 0; t <= winLen; t++) {
            X[start + t] = lo + (span * t) / winLen;
        }
        // re-establish strict monotonicity (guard against any rounding collapse)
        for (int i = 1; i < N; i++) if (X[i] <= X[i - 1]) X[i] = X[i - 1] + 1;
    }

    // Reference hierarchical-bridge construction: compute its wirelength Wc
    // and max degree Dc on THIS instance's coordinates.
    vector<pair<int,int>> edges;
    addCrossEdges(0, N, edges);
    vector<int> deg(N, 0);
    ll Wc = 0;
    for (auto &e : edges) {
        deg[e.first]++; deg[e.second]++;
        Wc += llabs(X[e.second] - X[e.first]);
    }
    int Dc = 0;
    for (int i = 0; i < N; i++) Dc = max(Dc, deg[i]);

    int d = Dc + SLACKD[testId];
    ll W = (Wc * (ll)SLACKW[testId] + 999) / 1000;

    printf("%d %d %d %lld\n", N, K, d, W);
    for (int i = 0; i < N; i++) printf("%lld%c", X[i], i + 1 == N ? '\n' : ' ');
    return 0;
}
