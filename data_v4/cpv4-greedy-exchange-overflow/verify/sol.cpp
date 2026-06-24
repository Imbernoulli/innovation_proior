#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // no jobs -> total cost 0
    vector<long long> p(n), w(n);
    for (int i = 0; i < n; i++) cin >> p[i];
    for (int i = 0; i < n; i++) cin >> w[i];

    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    // Smith's rule: order by p/w ascending. Compare p_i/w_i < p_j/w_j as a
    // cross-multiplication p_i * w_j < p_j * w_i with NO division. Both products
    // can reach 1e5 * 1e5 = 1e10, which overflows 32-bit; p and w are long long.
    sort(idx.begin(), idx.end(), [&](int i, int j) {
        long long lhs = p[i] * w[j];
        long long rhs = p[j] * w[i];
        if (lhs != rhs) return lhs < rhs;
        return i < j;                       // deterministic tie-break
    });

    long long clock = 0;                    // running completion time (up to 2e9)
    long long answer = 0;                   // sum of w_i * C_i (up to ~2e18)
    for (int k = 0; k < n; k++) {
        int i = idx[k];
        clock += p[i];                      // this job finishes at 'clock'
        answer += w[i] * clock;             // weighted completion time
    }

    cout << answer << "\n";
    return 0;
}
