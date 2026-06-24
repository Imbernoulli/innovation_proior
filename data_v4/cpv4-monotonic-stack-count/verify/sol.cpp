#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // c[i] = number of subarrays whose minimum is a[i], with ties broken by the
    // LEFTMOST minimum index. a[i] owns [l, r] (l <= i <= r) iff:
    //   - no element < a[i] in [l, r]            (else that element's value is the min)
    //   - no element <= a[i] strictly left of i  (an equal element to the left is a
    //                                              more-left minimum => it owns instead)
    //   - elements equal to a[i] to the RIGHT are fine (i stays leftmost min).
    // So:  left[i]  = nearest j < i with a[j] <= a[i]   -> l in (left[i], i]
    //      right[i] = nearest j > i with a[j] <  a[i]   -> r in [i, right[i])
    //      c[i] = (i - left[i]) * (right[i] - i).
    // Mixing <= / < across the two sides is what prevents double counting equal mins.

    vector<int> left_b(n), right_b(n);
    vector<int> st;
    st.reserve(n);

    // left[i]: previous index with a[j] <= a[i]; pop while a[top] > a[i].
    for (int i = 0; i < n; i++) {
        while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
        left_b[i] = st.empty() ? -1 : st.back();
        st.push_back(i);
    }
    st.clear();
    // right[i]: next index with a[j] < a[i]; pop while a[top] >= a[i].
    for (int i = n - 1; i >= 0; i--) {
        while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
        right_b[i] = st.empty() ? n : st.back();
        st.push_back(i);
    }

    long long ans = 0;
    for (int i = 0; i < n; i++) {
        long long cnt = (long long)(i - left_b[i]) * (long long)(right_b[i] - i); // exact, fits in 64-bit
        cnt %= MOD;
        ans = (ans + (long long)(i % MOD) * cnt) % MOD;
    }

    cout << ans << "\n";
    return 0;
}
