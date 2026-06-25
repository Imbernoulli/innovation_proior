#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // no performances -> profit 0
    vector<long long> p(n);
    vector<int> d(n);
    for (int i = 0; i < n; i++) {
        cin >> p[i] >> d[i];
        // Only the first n slots can ever be used (at most n acts get scheduled),
        // so a deadline beyond n is equivalent to a deadline of n. This caps slot
        // numbers at n and keeps the DSU array O(n) even when d[i] is up to 1e9.
        if (d[i] > n) d[i] = n;
    }

    // Order indices by profit descending; ties broken arbitrarily (does not affect the total).
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int x, int y) { return p[x] > p[y]; });

    // Disjoint-set-union "find latest free slot <= d": parent[s] points to the
    // largest still-free slot index that is <= s. Slot 0 is the sentinel "no slot".
    vector<int> parent(n + 1);
    iota(parent.begin(), parent.end(), 0); // parent[s] = s initially (all free)

    function<int(int)> find = [&](int s) {
        while (parent[s] != s) { parent[s] = parent[parent[s]]; s = parent[s]; }
        return s;
    };

    long long total = 0;
    for (int idx : order) {
        int s = find(d[idx]);   // latest free slot <= deadline, or 0 if none
        if (s > 0) {
            total += p[idx];    // take the job into slot s
            parent[s] = s - 1;  // slot s now points to the next-lower free slot
        }
    }

    cout << total << "\n";
    return 0;
}
