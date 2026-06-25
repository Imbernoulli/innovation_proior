#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;            // empty input -> n = 0 -> answer 0

    // p = running parity of popcount over the prefix; p == 0 for the empty prefix.
    // A window (l, r] has XOR with even popcount  <=>  p[l] == p[r]
    // (popcount parity is linear over XOR). Count prefixes in each parity class
    // (including the empty prefix) and pair equal-parity prefixes.
    long long cnt0 = 1, cnt1 = 0;          // empty prefix has parity 0
    int p = 0;
    for (int i = 0; i < n; i++) {
        unsigned int x;
        cin >> x;
        p ^= (__builtin_popcount(x) & 1);  // fold this element's popcount parity in
        if (p == 0) cnt0++; else cnt1++;
    }

    long long ans = cnt0 * (cnt0 - 1) / 2 + cnt1 * (cnt1 - 1) / 2;
    cout << ans << "\n";
    return 0;
}
