#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, K;
    if (!(cin >> n >> L >> K)) return 0;   // empty input -> load nothing
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // We must load between L and K crates (inclusive), 0 <= L <= K <= n. Each
    // crate's net delta c[i] may be positive, zero, or negative. Maximize the sum
    // of the loaded crates' deltas.
    //
    // Exchange argument: for a FIXED count t, the best choice is the t largest
    // deltas (swap any omitted larger delta for a chosen smaller one to improve).
    // So sort descending and look only at prefix sums. The best count t (in
    // [L, K]) is: keep adding the next-largest delta while it is positive, but we
    // are FORCED to reach at least L crates even if that drags in negatives.
    sort(c.begin(), c.end(), greater<long long>());

    // prefix[t] = sum of the t largest deltas. Forced floor at t = L.
    long long forced = 0;          // sum of the first L crates (mandatory quota)
    for (long long i = 0; i < L; i++) forced += c[i];

    long long best = forced;       // baseline: exactly the L mandatory crates
    long long run = forced;
    for (long long t = L; t < K; t++) {
        run += c[t];               // adding the (t+1)-th largest delta
        best = max(best, run);
    }

    cout << best << "\n";
    return 0;
}
