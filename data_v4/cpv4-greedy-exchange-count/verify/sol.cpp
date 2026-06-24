#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<pair<long long,long long>> iv(n); // (l, r)
    for (int i = 0; i < n; i++) cin >> iv[i].first >> iv[i].second;

    // Greedy stabbing: sort intervals by right endpoint. Sweep; whenever the
    // current interval is not yet pierced by the last placed point, place a new
    // point at its right endpoint. This yields the minimum number of points
    // such that every interval contains at least one point.
    sort(iv.begin(), iv.end(), [](const pair<long long,long long>& a,
                                   const pair<long long,long long>& b){
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });

    vector<long long> pts;            // placed inspection points, increasing
    long long last = LLONG_MIN;       // position of the most recent point
    for (int i = 0; i < n; i++) {
        long long l = iv[i].first, r = iv[i].second;
        if (last < l) {               // current interval not yet pierced
            last = r;                 // place a point at its right endpoint
            pts.push_back(last);
        }
    }
    long long numPoints = (long long)pts.size();

    // Second quantity: how many intervals contain >= 2 of the placed points?
    // The placed points are sorted increasingly. For interval [l, r], the count
    // of placed points inside is (upper_bound(r) - lower_bound(l)); it has two
    // or more iff there exists a placed point <= r-... i.e. the index gap >= 2.
    long long multi = 0;
    for (int i = 0; i < n; i++) {
        long long l = iv[i].first, r = iv[i].second;
        // number of placed points p with l <= p <= r
        long long hi = upper_bound(pts.begin(), pts.end(), r) - pts.begin();
        long long lo = lower_bound(pts.begin(), pts.end(), l) - pts.begin();
        if (hi - lo >= 2) multi++;
    }

    cout << numPoints << " " << multi << "\n";
    return 0;
}
