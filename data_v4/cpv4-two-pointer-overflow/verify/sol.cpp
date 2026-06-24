#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long B;
    if (!(cin >> n >> B)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Two-pointer / sliding window. All a[i] >= 1, so the window sum is
    // monotone in window width: as we extend the right end, the running sum
    // only grows, and shrinking the left end only shrinks it. For each right
    // we advance left just far enough that the window sum is <= B, then the
    // current window is the widest (hence, since values are positive, the
    // largest-sum) window ending at `right` that stays within budget.
    long long cur = 0;     // sum of a[left..right], must fit 64-bit
    long long best = 0;    // best window sum found so far (empty window = 0)
    int left = 0;
    for (int right = 0; right < n; right++) {
        cur += a[right];
        while (cur > B) {          // shrink from the left until within budget
            cur -= a[left];
            left++;
        }
        // now cur = sum(a[left..right]) <= B and is the max such ending here
        if (cur > best) best = cur;
    }

    cout << best << "\n";
    return 0;
}
