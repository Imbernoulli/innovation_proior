#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;          // n = 0 / empty input handled below

    vector<long long> bal(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> bal[i];

    // DSU with union by size; root carries the guild's total balance and member count.
    vector<int> par(n + 1), sz(n + 1, 1);
    for (int i = 1; i <= n; i++) par[i] = i;

    function<int(int)> find = [&](int x) {
        while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; }
        return x;
    };

    // Multiset of balances of guilds that currently have >= 2 members.
    // A merged balance can be SMALLER than its parts (negative members), so a
    // running max over merge-time balances is wrong; the multiset stays exact.
    multiset<long long> multi;

    string out;
    out.reserve(1 << 16);

    for (int e = 0; e < q; e++) {
        int type;
        cin >> type;
        if (type == 1) {
            int u, v;
            cin >> u >> v;
            int ru = find(u), rv = find(v);
            if (ru == rv) continue;          // already same guild: no change
            // A root with sz >= 2 is currently represented in the multiset.
            if (sz[ru] >= 2) multi.erase(multi.find(bal[ru]));
            if (sz[rv] >= 2) multi.erase(multi.find(bal[rv]));
            if (sz[ru] < sz[rv]) swap(ru, rv);
            par[rv] = ru;
            sz[ru] += sz[rv];
            bal[ru] += bal[rv];              // merged balance = sum (may be negative)
            multi.insert(bal[ru]);           // ru now has >= 2 members
        } else {
            // type == 2: max balance over multi-member guilds, but only if positive.
            long long ans = 0;
            if (!multi.empty()) {
                long long mx = *multi.rbegin();
                if (mx > 0) ans = mx;
            }
            out += to_string(ans);
            out += '\n';
        }
    }

    cout << out;
    return 0;
}
