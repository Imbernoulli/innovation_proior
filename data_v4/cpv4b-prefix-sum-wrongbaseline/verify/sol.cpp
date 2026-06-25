#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long m;
    if (!(cin >> n >> m)) return 0;

    // count[r] = number of prefix sums seen so far whose normalized remainder mod m is r.
    // prefix sum 0 (the empty prefix, before index 0) is present from the start.
    vector<long long> count(m, 0);
    count[0] = 1;

    long long prefix = 0;     // running prefix sum (64-bit: can reach ~2*10^14)
    long long answer = 0;     // number of windows (64-bit: can reach ~2*10^10)

    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        prefix += x;
        // Normalize the remainder into [0, m): C++ % can be negative for negative prefix.
        long long r = prefix % m;
        if (r < 0) r += m;
        answer += count[r];   // every earlier prefix with the same remainder closes a divisible window
        count[r]++;
    }

    cout << answer << "\n";
    return 0;
}
