#include <bits/stdc++.h>
using namespace std;

// Disjoint-set "next free slot" structure.
// nxt[s] points to the earliest slot >= s that is still free (or T+1 if none).
static int nxt[200005];

int find_free(int s) {
    // Path-compressed walk to the representative free slot.
    while (nxt[s] != s) {
        nxt[s] = nxt[nxt[s]];
        s = nxt[s];
    }
    return s;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long T;
    if (!(cin >> n >> T)) return 0;

    vector<int> l(n), r(n), order(n);
    for (int i = 0; i < n; i++) {
        cin >> l[i] >> r[i];
        order[i] = i;
    }

    // Earliest-deadline-first: process requests by increasing right endpoint,
    // breaking ties by increasing left endpoint.
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (r[a] != r[b]) return r[a] < r[b];
        return l[a] < l[b];
    });

    // Initialize the "next free slot" pointers for slots 1..T, plus a sentinel T+1.
    // (We only ever touch slots that appear as some l[i], so T may be huge; we
    //  index the DSU only up to the largest endpoint actually needed, but here
    //  endpoints are <= T and T fits in the array bound by the constraints.)
    int Ti = (int)T;
    for (int s = 1; s <= Ti + 1; s++) nxt[s] = s;

    vector<int> assign(n, -1);
    bool feasible = true;

    for (int idx : order) {
        int s = find_free(l[idx]);     // earliest free slot >= l[idx]
        if (s > r[idx]) {              // no free slot within [l, r]
            feasible = false;
            break;
        }
        assign[idx] = s;               // serve request idx in slot s
        nxt[s] = s + 1;                // slot s is now taken; point past it
    }

    if (!feasible) {
        cout << "NO\n";
    } else {
        cout << "YES\n";
        for (int i = 0; i < n; i++) {
            cout << assign[i] << (i + 1 < n ? ' ' : '\n');
        }
        if (n == 0) cout << '\n';
    }
    return 0;
}
