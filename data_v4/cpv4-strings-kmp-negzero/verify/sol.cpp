#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n)) return 0;          // no input at all
    vector<long long> t(n);
    for (auto &x : t) cin >> x;
    cin >> m;
    vector<long long> p(m);
    for (auto &x : p) cin >> x;

    // A resonance at position i means there is a constant c with t[i+j] = p[j] + c
    // for all 0 <= j < m. Subtracting consecutive entries cancels c, so this is
    // equivalent to: the (m-1)-length difference sequence of p equals the
    // (m-1)-length difference sequence of the window t[i..i+m-1].
    //
    // Special cases on the pattern length BEFORE building any difference array:
    //   m == 0 : an empty pattern. No resonance is defined for an empty pattern
    //            (there is no window to align), so the answer is 0 with no positions.
    //   m == 1 : the difference sequence is empty, so EVERY start position
    //            i in [0, n-1] resonates (a single value always matches up to shift).
    //            If n == 0 there are still zero positions.
    // These are exactly the base cases a naive "run KMP on the diff arrays" forgets.

    if (m == 0) {
        cout << 0 << "\n" << "\n";   // count line, then (empty) positions line
        return 0;
    }
    if (m == 1) {
        // Every position 0..n-1 is a resonance (window of length 1).
        cout << n << "\n";
        for (int i = 0; i < n; i++) cout << i << (i + 1 < n ? ' ' : '\n');
        if (n == 0) cout << "\n";    // empty positions line when there are none
        return 0;
    }
    if (m > n) {
        // Pattern longer than text: no window fits.
        cout << 0 << "\n" << "\n";
        return 0;
    }

    // Build difference sequences (lengths m-1 and n-1). Values can be negative,
    // zero, or positive; comparisons are plain integer equality so sign is irrelevant
    // to correctness as long as we use a signed 64-bit type.
    int pm = m - 1;
    vector<long long> pd(pm), td(n - 1);
    for (int j = 0; j < pm; j++) pd[j] = p[j + 1] - p[j];
    for (int j = 0; j + 1 < n; j++) td[j] = t[j + 1] - t[j];

    // KMP failure function over the pattern-difference sequence pd (length pm >= 1).
    vector<int> fail(pm, 0);
    for (int i = 1; i < pm; i++) {
        int k = fail[i - 1];
        while (k > 0 && pd[i] != pd[k]) k = fail[k - 1];
        if (pd[i] == pd[k]) k++;
        fail[i] = k;
    }

    // Scan the text-difference sequence. A full match of pd ending at td index e
    // corresponds to a window of m consecutive original values starting at
    // (e - pm + 1) in the difference array, i.e. text start index s = e - pm + 1.
    vector<int> hits;
    int k = 0;
    int tn = n - 1; // length of td
    for (int i = 0; i < tn; i++) {
        while (k > 0 && td[i] != pd[k]) k = fail[k - 1];
        if (td[i] == pd[k]) k++;
        if (k == pm) {
            int s = i - pm + 1;     // start index in difference array == text start
            hits.push_back(s);
            k = fail[k - 1];
        }
    }

    cout << (int)hits.size() << "\n";
    for (size_t i = 0; i < hits.size(); i++)
        cout << hits[i] << (i + 1 < hits.size() ? ' ' : '\n');
    if (hits.empty()) cout << "\n";
    return 0;
}
