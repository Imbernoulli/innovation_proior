// TIER: strong
// Insight: expansion PER UNIT WIRELENGTH is maximized by a distance-doubling
// hierarchy, not by spending budget on the cheapest edges available (that's
// the greedy trap) and not by a random/uniform expander (that blows the
// budget). Recursively bisect the village INDEX range (not the geometry --
// the topology below needs no coordinates at all, only n): at every level,
// link the two extreme villages of each half-pair with two "fold-crossing"
// bridges (leftmost<->rightmost-of-other-half and rightmost<->leftmost-of-
// other-half). This is connected by induction (every split is bridged), has
// max degree O(k)=O(log n) (one bridge edge per recursion level touches each
// village at most once per level), and because the ridge's fold-gaps are
// themselves geometrically summable (every level of the recursion contains
// exactly the same total fold length), the total wirelength stays low even
// though a handful of the bridges are genuinely long-range. The result is a
// small-world / hypercube-like graph: O(log n) hop-diameter for a wirelength
// budget that a naive local mesh (greedy) or a geometry-blind random
// d-regular graph could never afford for the same expansion.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N;
vector<pair<int,int>> edges;

void addCrossEdges(int lo, int hi) {
    if (hi - lo <= 1) return;
    int mid = (lo + hi) / 2;
    if (hi - lo == 2) {
        edges.push_back({lo, mid});
    } else {
        edges.push_back({lo, hi - 1});
        edges.push_back({mid - 1, mid});
    }
    addCrossEdges(lo, mid);
    addCrossEdges(mid, hi);
}

int main() {
    int n, k, d; ll W;
    if (!(cin >> n >> k >> d >> W)) return 0;
    vector<ll> x(n);
    for (auto &v : x) cin >> v;
    N = n;

    addCrossEdges(0, N);

    // Defensive re-check against the actual caps (should always hold by
    // construction, since the generator sized d/W to fit exactly this
    // topology on these coordinates -- but never emit an infeasible edge).
    vector<int> deg(n, 0);
    ll total = 0;
    vector<pair<int,int>> keep;
    for (auto &e : edges) {
        int a = e.first, b = e.second;
        ll cost = llabs(x[b] - x[a]);
        if (deg[a] + 1 > d || deg[b] + 1 > d) continue;
        if (total + cost > W) continue;
        deg[a]++; deg[b]++;
        total += cost;
        keep.push_back(e);
    }

    cout << keep.size() << "\n";
    for (auto &e : keep) cout << (e.first + 1) << ' ' << (e.second + 1) << "\n";
    return 0;
}
