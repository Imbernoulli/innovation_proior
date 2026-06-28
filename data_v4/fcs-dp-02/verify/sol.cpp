#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

static int n, p;
static vector<ll> x;       // sorted well positions, 0-indexed: x[0..n-1]
static vector<ll> pre;     // pre[t] = x[0] + ... + x[t-1]

const ll INF = (ll)4e18;

// cost(j, i): minimum total walking distance if a SINGLE pump serves the
// contiguous block of wells [j, i] (inclusive, 0-indexed), the pump placed
// optimally. For positions on a line the sum of absolute deviations is
// minimized at the (lower) median, so we anchor at x[m], m = (j+i)/2.
static inline ll cost(int j, int i) {
    if (j > i) return 0;
    int m = (j + i) / 2;                                  // median index
    ll med = x[m];
    // wells [j, m-1] sit at or left of the median; [m+1, i] sit at or right.
    ll left  = med * (m - j) - (pre[m] - pre[j]);         // sum(med - x[t]), t in [j, m-1]
    ll right = (pre[i + 1] - pre[m + 1]) - med * (i - m); // sum(x[t] - med), t in [m+1, i]
    return left + right;
}

static vector<ll> prevLayer; // dp for k-1 pumps
static vector<ll> curLayer;  // dp for k   pumps

// Divide-and-conquer DP optimization for one layer.
// Fills curLayer[i] for i in [lo, hi]. curLayer[i] = min cost to cover the
// first i wells with this layer's number of pumps, where the last pump covers
// the block [j, i-1] and the first i-1 of these pumps cover the first j wells:
//     curLayer[i] = min_{j in [optlo, min(opthi, i-1)]} prevLayer[j] + cost(j, i-1).
// Because cost satisfies the quadrangle inequality, the optimal j = opt(i) is
// monotonic non-decreasing in i, so we recurse with shrinking [optlo, opthi].
static void compute(int lo, int hi, int optlo, int opthi) {
    if (lo > hi) return;
    int mid = (lo + hi) / 2;
    ll best = INF;
    int bestj = -1;
    int jhi = min(opthi, mid - 1);
    for (int j = optlo; j <= jhi; ++j) {
        if (prevLayer[j] >= INF) continue;
        ll cand = prevLayer[j] + cost(j, mid - 1);  // last group = wells [j, mid-1]
        if (cand < best) { best = cand; bestj = j; }
    }
    curLayer[mid] = best;
    if (bestj == -1) bestj = optlo;                 // no feasible split; keep window valid
    compute(lo, mid - 1, optlo, bestj);
    compute(mid + 1, hi, bestj, opthi);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n >> p)) return 0;
    x.resize(n);
    for (int t = 0; t < n; ++t) cin >> x[t];
    sort(x.begin(), x.end());   // positions are given sorted; sort defensively

    pre.assign(n + 1, 0);
    for (int t = 0; t < n; ++t) pre[t + 1] = pre[t] + x[t];

    // dp_k[i] = min total distance to cover the first i wells with exactly k pumps.
    // Base layer k = 1: a single pump covers [0, i-1], so dp_1[i] = cost(0, i-1).
    prevLayer.assign(n + 1, INF);
    prevLayer[0] = 0;            // 0 wells with 0 pumps
    curLayer.assign(n + 1, INF);

    int kcap = min(p, n);        // more than n pumps can never beat n pumps (cost 0)
    for (int k = 1; k <= kcap; ++k) {
        fill(curLayer.begin(), curLayer.end(), INF);
        // Need i >= k wells to fill k non-empty groups; j (wells before last group)
        // ranges in [k-1, i-1], so globally j in [k-1, n-1].
        compute(k, n, k - 1, n - 1);
        swap(prevLayer, curLayer);
    }

    cout << prevLayer[n] << "\n";
    return 0;
}
