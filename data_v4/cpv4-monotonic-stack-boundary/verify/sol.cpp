#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    const long long MOD = 1000000007LL;

    // For each i, count subarrays whose minimum is a[i].
    // left[i]  = number of consecutive elements strictly greater than a[i]
    //            immediately to the left (so a[i] is the min over that reach).
    // right[i] = number of consecutive elements >= a[i] immediately to the right.
    // The strict/non-strict split (strict left, non-strict right) makes each
    // subarray credited to exactly one index when minima tie.
    vector<long long> left(n), right(n);

    // left: previous strictly-smaller-OR-EQUAL element acts as the wall.
    // Pop while stack top value > a[i]  (those are strictly greater -> in reach).
    {
        vector<int> st; // indices, increasing-ish by value (non-decreasing)
        for (int i = 0; i < n; i++) {
            while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
            int prev = st.empty() ? -1 : st.back();
            left[i] = i - prev;           // count of positions in (prev, i]
            st.push_back(i);
        }
    }
    // right: next strictly-smaller element is the wall.
    // Pop while stack top value >= a[i] (those are >= -> still in reach).
    {
        vector<int> st;
        for (int i = n - 1; i >= 0; i--) {
            while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
            int nxt = st.empty() ? n : st.back();
            right[i] = nxt - i;           // count of positions in [i, nxt)
            st.push_back(i);
        }
    }

    long long ans = 0;
    for (int i = 0; i < n; i++) {
        long long cnt = (left[i] % MOD) * (right[i] % MOD) % MOD;
        long long val = ((a[i] % MOD) + MOD) % MOD;
        ans = (ans + cnt * val) % MOD;
    }
    cout << ans << "\n";
    return 0;
}
