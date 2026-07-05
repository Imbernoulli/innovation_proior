// TIER: greedy
// Weight-greedy: enumerate all valid base pairs, add highest-weight-first if it keeps the
// structure nested and reuses no position. Packs strong C-G pairs but is blind to stacking
// and to pockets, so it beats the trivial baseline but leaves cooperative energy on the table.
#include <bits/stdc++.h>
using namespace std;

static int pairW(char a, char b) {
    auto is = [&](char x, char y) { return (a == x && b == y) || (a == y && b == x); };
    if (is('C', 'G')) return 3;
    if (is('A', 'U')) return 2;
    if (is('G', 'U')) return 1;
    return -1;
}

int main() {
    int L;
    if (!(cin >> L)) return 0;
    string S; cin >> S;
    int P; cin >> P;
    for (int k = 0; k < P; k++) { int p, v; cin >> p >> v; }

    struct Cand { int w, i, j; };
    vector<Cand> cs;
    for (int i = 0; i < L; i++)
        for (int j = i + 4; j < L; j++) {
            int w = pairW(S[i], S[j]);
            if (w > 0) cs.push_back({w, i, j});
        }
    // highest weight first; tie-break shorter span (more likely to stack later)
    sort(cs.begin(), cs.end(), [](const Cand& a, const Cand& b) {
        if (a.w != b.w) return a.w > b.w;
        return (a.j - a.i) < (b.j - b.i);
    });

    vector<int> partner(L, -1);
    vector<pair<int,int>> sel;   // selected intervals (i,j)
    auto compatible = [&](int i, int j) {
        for (auto& s : sel) {
            int c = s.first, d = s.second;
            bool disjoint = (j < c) || (i > d);
            bool nested = (i < c && d < j) || (c < i && j < d);
            if (!(disjoint || nested)) return false;
        }
        return true;
    };
    for (auto& c : cs) {
        if (partner[c.i] != -1 || partner[c.j] != -1) continue;
        if (!compatible(c.i, c.j)) continue;
        partner[c.i] = c.j; partner[c.j] = c.i;
        sel.push_back({c.i, c.j});
    }

    vector<pair<int,int>> out;
    for (int i = 0; i < L; i++) if (partner[i] > i) out.push_back({i + 1, partner[i] + 1});
    cout << out.size() << "\n";
    for (auto& pr : out) cout << pr.first << " " << pr.second << "\n";
    return 0;
}
