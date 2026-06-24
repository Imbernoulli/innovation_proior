#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }

    // Coordinate-compress the values so a Fenwick tree can be indexed by rank.
    vector<long long> srt(a);
    sort(srt.begin(), srt.end());
    srt.erase(unique(srt.begin(), srt.end()), srt.end());
    int m = (int)srt.size();
    auto rankOf = [&](long long v) -> int {
        // 1-based rank in the sorted distinct array.
        return (int)(lower_bound(srt.begin(), srt.end(), v) - srt.begin()) + 1;
    };

    // bit[r] = max f-value over compressed ranks in the Fenwick prefix ending at r.
    // We query the prefix max over ranks STRICTLY LESS THAN rank(a[i]) (values < a[i]),
    // then point-update rank(a[i]) with f[i].
    const long long NEG = LLONG_MIN / 4;
    vector<long long> bit(m + 1, NEG);

    auto queryPrefixMax = [&](int r) -> long long { // max over ranks [1..r]
        long long best = NEG;
        for (; r > 0; r -= r & (-r))
            best = max(best, bit[r]);
        return best;
    };
    auto updatePoint = [&](int r, long long val) {  // bit[r] = max(bit[r], val)
        for (; r <= m; r += r & (-r))
            bit[r] = max(bit[r], val);
    };

    long long answer = NEG;
    for (int i = 0; i < n; i++) {
        int r = rankOf(a[i]);
        // best f among earlier elements with strictly smaller value:
        long long bestPrev = queryPrefixMax(r - 1);
        long long f = a[i] + max(bestPrev, 0LL); // a[i] alone if no smaller predecessor
        updatePoint(r, f);
        answer = max(answer, f);
    }

    cout << answer << "\n";
    return 0;
}
