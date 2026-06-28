#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // empty input -> no pairs

    const int B = 20;                      // values are < 2^20
    const int SZ = 1 << B;                 // 1048576 masks
    const int FULL = SZ - 1;               // 0b11..1 (20 ones)

    // f[m] will become "how many array values are submasks of m".
    // Start as the multiplicity histogram of the input values, then
    // run the sum-over-subsets (SOS) transform in place.
    vector<long long> f(SZ, 0);
    vector<int> a(n);
    long long zeros = 0;                   // count of values equal to 0
    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;
        a[i] = x;
        f[x] += 1;
        if (x == 0) zeros++;
    }

    // SOS DP: after bit b, f[m] = #values whose mask is a submask of m
    // when restricted to differing only in bits <= b. After all B bits,
    // f[m] = total #array values that are submasks of m.
    for (int b = 0; b < B; b++) {
        for (int m = 0; m < SZ; m++) {
            if (m & (1 << b)) {
                f[m] += f[m ^ (1 << b)];
            }
        }
    }

    // For index i, the values disjoint from a[i] are exactly the submasks
    // of comp_i = FULL ^ a[i]. f[comp_i] counts them over ALL indices,
    // including i itself iff a[i] == 0 (0 is a submask of everything).
    long long ordered = 0;                  // ordered pairs (i, j), i may equal j
    for (int i = 0; i < n; i++) {
        ordered += f[FULL ^ a[i]];
    }

    // ordered counts: each unordered pair {i,j}, i!=j, disjoint, twice;
    // plus a self-pair (i,i) once for every value 0 (since 0 AND 0 == 0).
    // Remove the self-pairs, then halve to get unordered i<j pairs.
    long long answer = (ordered - zeros) / 2;

    cout << answer << "\n";
    return 0;
}
