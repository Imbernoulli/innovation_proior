#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // Two pointers: for each right end, shrink from the left while the
    // window sum exceeds S. Every valid window ending at `right` with left
    // boundary in [left, right] is admissible, contributing (right-left+1).
    long long count = 0;   // up to ~n*(n+1)/2 = 2*10^10 -> must be 64-bit
    long long sum = 0;     // up to n*max(w) = 2*10^14 -> must be 64-bit
    int left = 0;
    for (int right = 0; right < n; right++) {
        sum += w[right];
        while (left <= right && sum > S) {
            sum -= w[left];
            left++;
        }
        // windows [left..right], [left+1..right], ..., [right..right]
        count += (long long)(right - left + 1);
    }

    cout << count << "\n";
    return 0;
}
