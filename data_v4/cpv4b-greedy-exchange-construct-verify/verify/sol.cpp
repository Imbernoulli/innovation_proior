#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;                 // no input -> nothing to print

    // Build the lexicographically smallest Sidon set (distinct pairwise sums) of size n.
    // Greedy: x[0] = 0; each next value is the smallest integer > x[last] that keeps EVERY
    // pairwise sum distinct. The exchange argument shows this greedy prefix is lex-smallest.
    //
    // We track which pairwise sums are already used with a presence array `seen`, indexed by
    // the sum value. A candidate v is admissible iff none of the new sums it creates --
    // v + x[k] for every existing x[k], and 2v -- is already present. The new sums are also
    // mutually distinct automatically (v + x[k] are distinct for distinct x[k], and 2v differs
    // from each v + x[k] since x[k] != v), so checking each against `seen` is sufficient.

    vector<long long> x;
    x.reserve(n);
    vector<char> seen;                          // seen[s] = 1 if sum s already used
    long long cand = 0;

    while ((int)x.size() < n) {
        long long top = 2 * cand;               // largest new sum this candidate would create
        if ((long long)seen.size() <= top) seen.resize(top + 1, 0);

        bool ok = true;
        for (long long y : x) {                 // check v + x[k] for existing x[k]
            if (seen[cand + y]) { ok = false; break; }
        }
        if (ok && seen[top]) ok = false;        // check the self-sum 2v

        if (ok) {
            for (long long y : x) seen[cand + y] = 1;
            seen[top] = 1;
            x.push_back(cand);
        }
        cand++;
    }

    // Output the set, space-separated, on one line.
    for (int i = 0; i < n; i++) {
        cout << x[i];
        cout << (i + 1 < n ? ' ' : '\n');
    }
    if (n == 0) cout << "\n";
    return 0;
}
