#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long m, t;
    if (!(cin >> n >> m >> t)) return 0;     // n machines, modulus m, target residue t

    // cnt[r] = how many frequencies are congruent to r modulo m.
    vector<long long> cnt(m, 0);
    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        long long r = ((x % m) + m) % m;     // safe modulo for any sign (here x >= 0)
        cnt[r]++;
    }

    // We want pairs (i<j) with (a_i + a_j) % m == t.
    // Pair residues (r, s) with (r + s) % m == t. For r != s the count is cnt[r]*cnt[s]
    // (each unordered residue pair counted once); for the self-paired residue r == s it
    // is cnt[r]*(cnt[r]-1)/2 (choose 2 within the bucket).
    long long answer = 0;
    for (long long r = 0; r < m; r++) {
        long long s = ((t - r) % m + m) % m; // residue that completes r to t (mod m)
        if (r < s) {
            answer += cnt[r] * cnt[s];       // cnt[r]*cnt[s] can exceed 32-bit: long long
        } else if (r == s) {
            answer += cnt[r] * (cnt[r] - 1) / 2;
        }
        // r > s already handled when the loop variable was s
    }

    cout << answer << "\n";
    return 0;
}
