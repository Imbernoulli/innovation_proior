#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, k, A, B, M;
    if (!(cin >> n >> k >> A >> B >> M)) return 0;   // empty input

    // Count strings of length n over k colors such that every MAXIMAL
    // monochromatic run has length in [A, B], modulo M.
    //
    // The empty string (n == 0) has no runs, so it is vacuously valid: 1 string.
    if (n == 0) { cout << (1 % M) << "\n"; return 0; }

    // f[i] = number of valid colorings of cells 1..i in which a maximal run
    //        ENDS exactly at position i. Keying on "a run ends here" is what
    //        prevents double counting: every valid string is counted once, at
    //        the position where its last run terminates.
    //
    // A run ending at i has length len in [A, B], so it occupies cells
    // (i-len+1 .. i). Let p = i - len be the position just before the run.
    //   - p == 0: this is the FIRST run; it may be any of the k colors.
    //             Valid only when len == i lies in [A, B].
    //   - p >= 1: a run ends at p, and the new run's color must DIFFER from the
    //             previous run's color -> (k-1) choices. Contributes
    //             f[p] * (k-1).
    //
    // Answer = f[n].
    long long kmod = k % M;
    long long km1  = (k - 1) % M;             // k >= 1, so k-1 >= 0

    vector<long long> f(n + 1, 0);
    // pref[i] = (f[0] + f[1] + ... + f[i]) mod M, with the convention f[0] = 0
    // for the prefix only; the FIRST-run case is handled separately so we never
    // confuse the empty prefix with a real "run ends at 0".
    vector<long long> pref(n + 1, 0);

    for (long long i = 1; i <= n; i++) {
        // Window of valid previous-end positions p so that len = i - p is in
        // [A, B] and p >= 1 (non-first run):  A <= i - p <= B  =>
        //   p in [i - B, i - A], intersected with [1, i-1].
        long long plo = max(1LL, i - B);
        long long phi = i - A;                // len >= A  =>  p <= i - A
        long long ways = 0;
        if (phi >= plo) {
            // sum of f[plo..phi] via prefix sums
            long long s = pref[phi] - (plo >= 1 ? pref[plo - 1] : 0);
            s %= M; if (s < 0) s += M;
            ways = s % M * (km1 % M) % M;     // (k-1) color choices for new run
        }
        // First-run case: the whole prefix 1..i is one run, length i in [A,B].
        if (i >= A && i <= B) {
            ways = (ways + kmod) % M;
        }
        f[i] = ways % M;
        pref[i] = (pref[i - 1] + f[i]) % M;
    }

    cout << (f[n] % M) << "\n";
    return 0;
}
