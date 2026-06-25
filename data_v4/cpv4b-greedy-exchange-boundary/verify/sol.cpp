#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 -> no segments -> 0 checkpoints
    vector<pair<long long,long long>> seg(n); // (right endpoint, left endpoint)
    for (int i = 0; i < n; i++) {
        long long l, r;
        cin >> l >> r;
        seg[i] = {r, l};                   // sort key is the right endpoint
    }

    sort(seg.begin(), seg.end());          // by right endpoint ascending

    long long checkpoints = 0;
    long long last = LLONG_MIN;            // coordinate of the most recent checkpoint
    for (int i = 0; i < n; i++) {
        long long r = seg[i].first, l = seg[i].second;
        // The segment is already inspected iff l <= last <= r. Since we process in
        // right-endpoint order, last <= r always holds once a checkpoint exists, so
        // the only test that matters is whether l <= last. If l > last the segment
        // is NOT covered and we must open a new checkpoint at r (the latest spot that
        // still inspects this segment, maximizing future coverage).
        if (last < l) {                    // strict: l == last means already covered
            checkpoints++;
            last = r;
        }
    }

    cout << checkpoints << "\n";
    return 0;
}
