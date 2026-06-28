#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // Meet in the middle. Split items into two halves of size <= 20 each.
    int la = n / 2;          // size of first half
    int lb = n - la;         // size of second half

    // Enumerate all subsets of the second half into (weight, value) pairs.
    int sb = 1 << lb;
    vector<pair<long long,long long>> B;     // (weight, value)
    B.reserve(sb);
    for (int mask = 0; mask < sb; mask++) {
        long long sw = 0, sv = 0;
        for (int j = 0; j < lb; j++) {
            if (mask & (1 << j)) {
                sw += w[la + j];
                sv += v[la + j];
            }
        }
        B.push_back({sw, sv});
    }
    // Sort by weight; build a prefix maximum of value so that for any weight
    // budget the best achievable value among weights <= budget is queryable.
    sort(B.begin(), B.end());
    vector<long long> bw(sb), bestv(sb);
    long long run = LLONG_MIN;
    for (int i = 0; i < sb; i++) {
        bw[i] = B[i].first;
        run = max(run, B[i].second);
        bestv[i] = run;                      // best value using weight <= bw[i]
    }

    // Enumerate all subsets of the first half; for each, binary-search the
    // remaining capacity in the second half and add its best value.
    int sa = 1 << la;
    long long ans = 0;
    for (int mask = 0; mask < sa; mask++) {
        long long sw = 0, sv = 0;
        for (int j = 0; j < la; j++) {
            if (mask & (1 << j)) {
                sw += w[j];
                sv += v[j];
            }
        }
        if (sw > C) continue;                // first half alone already too heavy
        long long rem = C - sw;
        // largest index i with bw[i] <= rem
        int lo = 0, hi = sb - 1, pos = -1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (bw[mid] <= rem) { pos = mid; lo = mid + 1; }
            else hi = mid - 1;
        }
        if (pos >= 0) {
            ans = max(ans, sv + bestv[pos]);
        }
        // (pos == -1 is impossible since the empty subset has weight 0 <= rem)
    }

    cout << ans << "\n";
    return 0;
}
