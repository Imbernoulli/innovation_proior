#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long P;
    if (!(cin >> n >> P)) return 0;
    vector<long long> s(n);
    for (auto &x : s) cin >> x;

    // pages printed by all presses by time T = sum_i floor(T / s[i]).
    // This is nondecreasing in T, so binary-search the minimum T with pages(T) >= P.
    // Upper bound: a single press with the smallest period needs P*minPeriod seconds
    // to alone reach P pages, which is a safe (over-)estimate for the whole set.
    long long minPeriod = *min_element(s.begin(), s.end());

    auto pages = [&](long long T) -> long long {
        long long total = 0;
        for (long long period : s) {
            total += T / period;
            if (total >= P) return total; // early exit also caps the running sum
        }
        return total;
    };

    long long lo = 0;                 // pages(0) = 0 < P (P >= 1)
    long long hi = P * minPeriod;     // pages(hi) >= P guaranteed
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (pages(mid) >= P) hi = mid;
        else lo = mid + 1;
    }

    cout << lo << "\n";
    return 0;
}
