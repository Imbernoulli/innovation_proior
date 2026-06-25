#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;

    // count[p] = how many prefix sums P[0..j-1] equal value p have been seen so far.
    // We accumulate the answer as we sweep j from 0..n-1.
    unordered_map<long long, long long> seen;
    seen.reserve(n * 2);
    seen.max_load_factor(0.7f);

    long long prefix = 0;        // P[j] after consuming a[0..j-1]; prefix sum can exceed 32-bit
    long long answer = 0;        // number of subarrays; can reach ~n^2/2, exceeds 32-bit
    seen[0] = 1;                 // empty prefix P[0] = 0 seen once

    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        prefix += x;             // prefix = P[i+1]
        // subarrays ending at index i with sum S correspond to a prior prefix == prefix - S
        auto it = seen.find(prefix - S);
        if (it != seen.end()) answer += it->second;
        seen[prefix] += 1;
    }

    cout << answer << "\n";
    return 0;
}
