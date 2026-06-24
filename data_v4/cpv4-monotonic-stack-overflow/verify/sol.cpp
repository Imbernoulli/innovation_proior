#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // For each i, count subarrays in which a[i] is the (chosen) minimum.
    // left[i]  = number of consecutive positions ending at i (incl. i) for which
    //            a[i] is a STRICT minimum vs the left  (a[j] > a[i] for j in that run).
    // right[i] = number of consecutive positions starting at i (incl. i) for which
    //            a[i] is a NON-strict minimum vs the right (a[j] >= a[i]).
    // Strict on one side, non-strict on the other breaks ties so each subarray's
    // minimum is attributed to exactly one index.
    vector<long long> left(n), right(n);
    vector<int> st; // indices, values strictly increasing from bottom to top

    // left: previous index with a value STRICTLY less than a[i].
    for (int i = 0; i < n; i++) {
        while (!st.empty() && a[st.back()] >= a[i]) st.pop_back();
        int prev = st.empty() ? -1 : st.back();
        left[i] = i - prev;                // run length to the left
        st.push_back(i);
    }
    st.clear();
    // right: next index with a value strictly less-OR-equal... we use next strictly
    // less to keep ties on the left side; here pop while top value strictly greater.
    for (int i = n - 1; i >= 0; i--) {
        while (!st.empty() && a[st.back()] > a[i]) st.pop_back();
        int nxt = st.empty() ? n : st.back();
        right[i] = nxt - i;                // run length to the right
        st.push_back(i);
    }

    long long answer = 0;
    for (int i = 0; i < n; i++) {
        // a[i] is the minimum of left[i] * right[i] subarrays.
        answer += a[i] * (left[i] * right[i]);
    }

    cout << answer << "\n";
    return 0;
}
