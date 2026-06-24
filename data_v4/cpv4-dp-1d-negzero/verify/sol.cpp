#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long c;
    if (!(cin >> n >> c)) return 0;          // empty input -> profit 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // best = max profit of a NON-EMPTY contiguous block ending at i, AFTER paying c.
    //   cur = best subarray sum ending at i (Kadane), base -inf so a single day is forced in.
    // Running-on profit of a block is (sum of block) - c; we compare against doing nothing (0).
    const long long NEG = LLONG_MIN / 4;
    long long cur = NEG;          // best subarray sum ending at current index (non-empty)
    long long bestSum = NEG;      // best subarray sum over all non-empty blocks
    for (int i = 0; i < n; i++) {
        cur = max(a[i], cur + a[i]);   // extend or restart; never "empty" -> base NEG
        bestSum = max(bestSum, cur);
    }

    long long answer = 0;                          // do nothing
    if (bestSum > NEG) answer = max(answer, bestSum - c);   // run the best block, pay c once

    cout << answer << "\n";
    return 0;
}
