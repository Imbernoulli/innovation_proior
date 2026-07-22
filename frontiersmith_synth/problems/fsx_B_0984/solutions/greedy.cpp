// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// The "obvious" monotone-intuition heuristic:
//   1. Model curvature as roughly LINEAR in the number of cuts, using the
//      TANGENT slope of the true cubic law at r=k (full material). This is
//      exactly the "more cuts = more bend, proportionally" instinct.
//   2. Estimate how many cuts each column needs to hit its target under that
//      linear model.
//   3. Serve the columns with the largest estimated need FIRST (spend the
//      shared slit budget on the neediest-looking columns), clamping at the
//      strength floor and whatever budget remains.
// Because the true law is convex (accelerates near the floor), the tangent
// line UNDER-predicts curvature away from r=k, so this over-estimates the
// cuts really needed -> burns the shared budget on the first few columns it
// processes and floors them needlessly, starving the rest.
int main() {
    int m, k, S;
    ll M, C;
    cin >> m >> k >> S >> M >> C;
    vector<ll> traw(m);
    vector<double> t(m);
    for (int i = 0; i < m; i++) { cin >> traw[i]; t[i] = traw[i] / 1000.0; }

    double curvK = (double)M / ((double)k * k * k);
    double slope0 = 3.0 * (double)M / ((double)k * k * k * k);   // tangent slope at r=k
    int maxU = k - S;

    vector<ll> uPred(m);
    vector<int> order(m);
    for (int i = 0; i < m; i++) {
        double gap = t[i] - curvK;
        ll up = 0;
        if (gap > 0 && slope0 > 1e-12) up = (ll)llround(gap / slope0);
        if (up < 0) up = 0;
        if (up > maxU) up = maxU;
        uPred[i] = up;
        order[i] = i;
    }
    // Explicit total order (predicted need desc, index asc tie-break) so the
    // result is identical regardless of std::sort implementation/stability.
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (uPred[a] != uPred[b]) return uPred[a] > uPred[b];
        return a < b;
    });

    ll remaining = C;
    vector<int> r(m, k);
    for (int idx : order) {
        ll use = min(uPred[idx], remaining);
        if (use > maxU) use = maxU;
        r[idx] = k - (int)use;
        remaining -= use;
    }
    for (int i = 0; i < m; i++) cout << r[i] << (i + 1 == m ? '\n' : ' ');
    return 0;
}
