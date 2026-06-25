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

    // prefix[k] = a[0] + ... + a[k-1], prefix[0] = 0, length n+1.
    // A window [l, r) (0 <= l < r <= n) has sum prefix[r] - prefix[l];
    // we want the shortest window with prefix[r] - prefix[l] >= S, i.e. minimize r - l.
    vector<long long> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) prefix[i + 1] = prefix[i] + a[i];

    // Monotonic deque of candidate left endpoints (indices into prefix), with
    // strictly increasing prefix values. For each r we pop from the front while
    // prefix[r] - prefix[front] >= S (that left can never be beaten by a larger r,
    // since r only grows), and we keep the deque increasing from the back so a
    // smaller-or-equal prefix at a later index dominates earlier larger ones.
    deque<int> dq;
    int best = INT_MAX;
    for (int r = 0; r <= n; r++) {
        while (!dq.empty() && prefix[r] - prefix[dq.front()] >= S) {
            best = min(best, r - dq.front());
            dq.pop_front();
        }
        while (!dq.empty() && prefix[dq.back()] >= prefix[r]) {
            dq.pop_back();
        }
        dq.push_back(r);
    }

    cout << (best == INT_MAX ? -1 : best) << "\n";
    return 0;
}
