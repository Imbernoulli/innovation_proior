#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long D;
    if (!(cin >> n >> D)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Count contiguous blocks [l, r] (0-indexed, inclusive) of length >= 2 with
    // max(block) - min(block) <= D.
    //
    // Two pointers with two monotonic deques (max-deque and min-deque holding
    // indices). For each right endpoint r we advance l to the smallest index so
    // that window [l, r] satisfies max - min <= D. Every l' in [l, r] yields a
    // valid window [l', r]; among those, the ones of length >= 2 are exactly the
    // l' in [l, r-1], i.e. (r - l) windows (clamped at 0 when l == r).
    deque<int> mx, mn; // indices, mx decreasing values, mn increasing values
    int l = 0;
    long long answer = 0;
    for (int r = 0; r < n; r++) {
        while (!mx.empty() && a[mx.back()] <= a[r]) mx.pop_back();
        mx.push_back(r);
        while (!mn.empty() && a[mn.back()] >= a[r]) mn.pop_back();
        mn.push_back(r);

        // Shrink from the left until window is valid.
        while (a[mx.front()] - a[mn.front()] > D) {
            l++;
            if (mx.front() < l) mx.pop_front();
            if (mn.front() < l) mn.pop_front();
        }

        // Windows [l', r] with l <= l' <= r are all valid. Length >= 2 needs
        // l' <= r - 1, so the count is (r - l). When l == r there are none.
        answer += (long long)(r - l);
    }

    cout << answer << "\n";
    return 0;
}
