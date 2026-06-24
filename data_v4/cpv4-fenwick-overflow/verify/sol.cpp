#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<int> a(n);
    for (auto &x : a) cin >> x;

    // Coordinate-compress the values so the Fenwick index space is O(n).
    vector<int> srt(a);
    sort(srt.begin(), srt.end());
    srt.erase(unique(srt.begin(), srt.end()), srt.end());
    int m = (int)srt.size();
    auto rankOf = [&](int v) {
        return int(lower_bound(srt.begin(), srt.end(), v) - srt.begin()) + 1; // 1..m
    };

    // Two Fenwick trees indexed by compressed value:
    //   cnt[r] = how many earlier elements have value-rank r,
    //   sum[r] = sum of those earlier element VALUES (this is the part that overflows int).
    vector<long long> bitCnt(m + 1, 0), bitSum(m + 1, 0);
    auto add = [&](vector<long long> &bit, int i, long long delta) {
        for (; i <= m; i += i & (-i)) bit[i] += delta;
    };
    auto query = [&](vector<long long> &bit, int i) {
        long long s = 0;
        for (; i > 0; i -= i & (-i)) s += bit[i];
        return s;
    };

    long long answer = 0;
    // Sweep left to right. For element j (value v, rank r), every earlier element with a
    // strictly greater value forms an inversion (i<j, a[i] > a[j]); each contributes a[i]*a[j].
    // Summed over those earlier elements that is v * (sum of their values).
    for (int j = 0; j < n; j++) {
        long long v = a[j];
        int r = rankOf(a[j]);
        // earlier elements with rank in (r, m]  =>  value strictly greater than a[j].
        long long greaterValueSum = query(bitSum, m) - query(bitSum, r);
        answer += v * greaterValueSum;     // v fits int but the product / accumulator do not
        // insert this element
        add(bitCnt, r, 1);
        add(bitSum, r, v);
    }

    cout << answer << "\n";
    return 0;
}
