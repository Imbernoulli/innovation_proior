#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> t(n);
    for (auto &x : t) cin >> x;

    sort(t.begin(), t.end());

    // A snapshot at integer time s captures every pulse with s <= t < s + L
    // (the right end s+L is EXCLUDED -- half-open window [s, s+L)).
    // Greedy: take the earliest still-uncaptured pulse p, place s = t[p].
    // That window covers exactly the pulses with value in [t[p], t[p]+L).
    long long snapshots = 0;
    int i = 0;
    while (i < n) {
        snapshots++;
        long long cover_end = t[i] + L;        // exclusive right boundary
        while (i < n && t[i] < cover_end) i++;  // strict: t[i] == cover_end is NOT covered
    }

    cout << snapshots << "\n";
    return 0;
}
