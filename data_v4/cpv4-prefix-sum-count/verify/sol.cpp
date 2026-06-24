#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // prefix[r] = (a[0]+...+a[r-1]) mod m, with prefix[0] = 0 (empty prefix).
    // A window [l, r-1] (0-based, 0 <= l <= r-1) has sum divisible by m iff
    // prefix[l] == prefix[r] (as residues). Count unordered pairs i < j with
    // prefix[i] == prefix[j]: for each residue with count c, that is c*(c-1)/2.
    // Negative values: normalize residue into [0, m-1].
    vector<long long> cnt(m, 0);
    long long pref = 0;
    cnt[0] = 1; // the empty prefix prefix[0] = 0
    long long answer = 0;
    for (int i = 0; i < n; i++) {
        pref = ((pref + a[i]) % m + m) % m; // normalize into [0, m-1]
        // every earlier prefix equal to this residue closes a balanced window ending at i
        answer += cnt[pref];
        cnt[pref]++;
    }

    cout << answer << "\n";
    return 0;
}
