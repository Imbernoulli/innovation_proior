// TIER: trivial
// Reproduces the judge's greedy nested matching exactly -> energy == baseline B -> ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;

static bool comp(char a, char b) {
    auto is = [&](char x, char y) { return (a == x && b == y) || (a == y && b == x); };
    return is('C', 'G') || is('A', 'U') || is('G', 'U');
}

int main() {
    int L;
    if (!(cin >> L)) return 0;
    string S; cin >> S;
    int P; cin >> P;
    for (int k = 0; k < P; k++) { int p, v; cin >> p >> v; }

    vector<int> partner(L, -1), st;
    for (int j = 0; j < L; j++) {
        int found = -1;
        for (int idx = (int)st.size() - 1; idx >= 0; idx--) {
            int i = st[idx];
            if (j - i - 1 >= 3 && comp(S[i], S[j])) { found = idx; break; }
        }
        if (found >= 0) {
            int i = st[found];
            partner[i] = j; partner[j] = i;
            st.resize(found);
        } else {
            st.push_back(j);
        }
    }

    vector<pair<int,int>> out;
    for (int i = 0; i < L; i++) if (partner[i] > i) out.push_back({i + 1, partner[i] + 1});

    cout << out.size() << "\n";
    for (auto& pr : out) cout << pr.first << " " << pr.second << "\n";
    return 0;
}
