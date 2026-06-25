#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Count non-empty subarrays [l, r] with a[l] + ... + a[r] == S.
    // Prefix sums: P[0] = 0, P[k] = a[0] + ... + a[k-1].
    // [l, r] sums to S  <=>  P[r+1] - P[l] = S  <=>  P[l] = P[r+1] - S.
    // Sweep r+1 = 1..n. BEFORE inserting P[r+1], the map holds exactly
    // {P[0], ..., P[r]} = all valid left endpoints l in [0, r], so a length-0
    // subarray (l = r+1) can never be matched. Order matters: query then insert.
    unordered_map<long long, long long> cnt;
    cnt.reserve(n * 2 + 16);
    cnt.max_load_factor(0.7);

    long long pref = 0;        // P[0] = 0, the empty prefix (a valid left endpoint l = 0)
    cnt[pref] = 1;             // seed P[0] exactly once
    long long answer = 0;
    for (int i = 0; i < n; i++) {
        pref += a[i];          // now pref = P[i+1]
        auto it = cnt.find(pref - S);   // need P[l] = P[i+1] - S
        if (it != cnt.end()) answer += it->second;
        cnt[pref] += 1;        // insert P[i+1] AFTER querying
    }

    cout << answer << "\n";
    return 0;
}
