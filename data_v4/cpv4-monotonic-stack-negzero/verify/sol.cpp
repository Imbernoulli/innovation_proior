#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) { cout << 0 << "\n"; return 0; } // empty input -> empty subarray
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // For each i, find the widest window [L, R] in which a[i] is a minimum.
    // left[i]  = index of nearest position to the left with a value STRICTLY less than a[i] (-1 if none)
    // right[i] = index of nearest position to the right with a value <= a[i] (n if none)
    // This (strict-left, non-strict-right) tie-break makes each window counted once.
    vector<int> left(n), right(n);
    vector<int> st; // monotonic stack of indices

    st.clear();
    for (int i = 0; i < n; i++) {
        while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
        left[i] = st.empty() ? -1 : st.back();
        st.push_back(i);
    }
    st.clear();
    for (int i = n - 1; i >= 0; i--) {
        while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
        right[i] = st.empty() ? n : st.back();
        st.push_back(i);
    }

    // Empty subarray is allowed and scores 0; that is the base value.
    long long best = 0;
    for (int i = 0; i < n; i++) {
        long long width = (long long)(right[i] - left[i] - 1);
        long long score = a[i] * width;          // min over the window is exactly a[i]
        if (score > best) best = score;
    }

    cout << best << "\n";
    return 0;
}
