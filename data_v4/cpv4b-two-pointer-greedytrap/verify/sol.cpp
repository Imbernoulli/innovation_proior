#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long B;
    if (!(cin >> n >> B)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // Left crane removes a prefix of i containers; right crane removes a suffix
    // of j containers. They share one fuel budget B, so pref[i] + suf[j] <= B,
    // and they must not overlap, so i + j <= n. Maximize i + j.
    //
    // pref[i] = sum of first i containers; suf[j] = sum of last j containers.
    vector<long long> pref(n + 1, 0);
    for (int i = 0; i < n; i++) pref[i + 1] = pref[i] + w[i];
    vector<long long> suf(n + 1, 0);
    for (int j = 0; j < n; j++) suf[j + 1] = suf[j] + w[n - 1 - j];

    // Sweep i (prefix count) upward over every affordable prefix. For each i the
    // suffix may use at most n - i containers and at most B - pref[i] fuel. As i
    // increases, pref[i] grows so the suffix budget shrinks; the overlap bound
    // n - i also shrinks. Hence the largest affordable suffix count j is
    // non-increasing in i, so a single pointer sliding inward gives O(n) total.
    long long best = 0;
    int j = n; // largest suffix count we will ever consider; shrinks as i grows
    for (int i = 0; i <= n; i++) {
        if (pref[i] > B) break;            // no longer affordable; larger i is worse
        if (j > n - i) j = n - i;          // respect the no-overlap bound
        while (j > 0 && (suf[j] > B - pref[i])) j--; // shrink until suffix fits
        best = max(best, (long long)i + j);
    }

    cout << best << "\n";
    return 0;
}
