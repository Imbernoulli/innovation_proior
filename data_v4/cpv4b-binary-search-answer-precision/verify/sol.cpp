#include <bits/stdc++.h>
using namespace std;

typedef unsigned long long u64;
typedef __int128 i128;

int n, m;
long long K;
vector<u64> a, b;

// count how many products a[i]*b[j] are <= x  (all values >= 1, so products are >= 1)
// x is a 128-bit threshold; products can reach 1.6e19 which overflows u64, so compare in i128.
i128 countLE(i128 x) {
    i128 cnt = 0;
    for (size_t i = 0; i < a.size(); i++) {
        // largest b[j] with a[i]*b[j] <= x  <=>  b[j] <= x / a[i]
        // Do NOT divide x by a[i] in floating point and do NOT count via b[j] <= x/a[i] with
        // truncation surprises; instead count b[j] with (i128)a[i]*b[j] <= x by walking sorted b.
        i128 ai = (i128)a[i];
        // binary search the number of b[j] (b sorted ascending) with ai*b[j] <= x
        int lo = 0, hi = (int)b.size(); // first index with ai*b[hi] > x
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ai * (i128)b[mid] <= x) lo = mid + 1;
            else hi = mid;
        }
        cnt += lo;
    }
    return cnt;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> m >> K)) return 0;
    a.resize(n);
    b.resize(m);
    for (auto &x : a) cin >> x;
    for (auto &x : b) cin >> x;
    sort(b.begin(), b.end());

    // Binary search the answer x in [1, MAXPROD]: smallest x with countLE(x) >= K.
    i128 lo = 1;
    i128 hi = (i128)4000000000ULL * (i128)4000000000ULL; // 1.6e19, exceeds u64 max
    while (lo < hi) {
        i128 mid = lo + (hi - lo) / 2;
        if (countLE(mid) >= (i128)K) hi = mid;
        else lo = mid + 1;
    }

    // print lo as a base-10 number (it can exceed u64, so render the i128 by hand)
    i128 v = lo;
    if (v == 0) { cout << 0 << "\n"; return 0; }
    string s;
    while (v > 0) { s += char('0' + (int)(v % 10)); v /= 10; }
    reverse(s.begin(), s.end());
    cout << s << "\n";
    return 0;
}
