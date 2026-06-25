#include <bits/stdc++.h>
using namespace std;

// Count unordered pairs (i<j) with a[i]+a[j] <= K, on a SORTED array a.
// Classic two-pointer: for each right, the leftmost l with a[l]+a[right] <= K.
static long long countLE(const vector<long long>& a, long long K) {
    int n = (int)a.size();
    long long cnt = 0;
    int l = 0, r = n - 1;
    while (l < r) {
        if (a[l] + a[r] <= K) {
            // a[l..r-1] all pair with a[r] to satisfy <= K (array sorted)
            cnt += (long long)(r - l);
            l++;
        } else {
            r--;
        }
    }
    return cnt;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    long long lo, hi;
    cin >> lo >> hi;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    sort(a.begin(), a.end());

    // An empty band (lo > hi) admits no pair. Guard it BEFORE subtracting:
    // countLE(hi) - countLE(lo-1) is the count in [lo, hi] only when lo <= hi,
    // i.e. when {s <= hi} is a superset of {s <= lo-1}. If lo > hi the
    // subtraction can go negative, so the answer must be pinned to 0 here.
    long long ans;
    if (lo > hi) {
        ans = 0;
    } else {
        // pairs with sum in [lo, hi] = countLE(hi) - countLE(lo - 1)
        ans = countLE(a, hi) - countLE(a, lo - 1);
    }

    cout << ans << "\n";
    return 0;
}
