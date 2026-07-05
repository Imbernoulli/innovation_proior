// TIER: strong
// Helix-and-pocket heuristic: enumerate maximal complementary stems (helices), score each by
// base weight + cooperative stacking + the value of the pockets it would enclose, greedily
// select a nested compatible set of stems, then fill remaining positions with high-weight
// single pairs. Captures cooperative energy AND deliberately encloses high-value pockets.
#include <bits/stdc++.h>
using namespace std;

static const int STK = 3;
static int pairW(char a, char b) {
    auto is = [&](char x, char y) { return (a == x && b == y) || (a == y && b == x); };
    if (is('C', 'G')) return 3;
    if (is('A', 'U')) return 2;
    if (is('G', 'U')) return 1;
    return -1;
}
static bool comp(char a, char b) { return pairW(a, b) > 0; }

int main() {
    int L;
    if (!(cin >> L)) return 0;
    string S; cin >> S;
    int P; cin >> P;
    vector<long long> pval(L, 0);      // pocket value at position (0 if none)
    for (int k = 0; k < P; k++) { int p, v; cin >> p >> v; pval[p - 1] += v; }
    // prefix sums of pocket value for O(1) range queries
    vector<long long> pre(L + 1, 0);
    for (int i = 0; i < L; i++) pre[i + 1] = pre[i] + pval[i];
    auto rangeVal = [&](int a, int b) -> long long {   // sum over [a,b] inclusive, clamped
        if (a < 0) a = 0; if (b > L - 1) b = L - 1;
        if (a > b) return 0;
        return pre[b + 1] - pre[a];
    };

    // enumerate maximal complementary stems: an outer pair (i,j) with (i-1,j+1) NOT a pair.
    struct Stem { long long score; int i, j, len; };
    vector<Stem> stems;
    for (int i = 0; i < L; i++)
        for (int j = i + 4; j < L; j++) {
            if (!comp(S[i], S[j])) continue;
            if (i - 1 >= 0 && j + 1 < L && comp(S[i - 1], S[j + 1])) continue; // not maximal-outer
            // extend inward while complementary and min-loop still satisfied
            int len = 0; long long base = 0;
            while (comp(S[i + len], S[j - len]) && (j - len) - (i + len) - 1 >= 3) {
                base += pairW(S[i + len], S[j - len]);
                len++;
            }
            if (len == 0) continue;
            long long sc = base + (long long)STK * (len - 1);
            // pocket value enclosed in the loop region (i+len .. j-len), if this stem is a helix
            if (len >= 2) sc += rangeVal(i + len, j - len);
            stems.push_back({sc, i, j, len});
        }
    sort(stems.begin(), stems.end(), [](const Stem& a, const Stem& b) {
        if (a.score != b.score) return a.score > b.score;
        return (a.j - a.i) < (b.j - b.i);
    });

    vector<int> partner(L, -1);
    vector<pair<int,int>> sel;   // selected outer intervals
    auto compatible = [&](int i, int j) {
        for (auto& s : sel) {
            int c = s.first, d = s.second;
            bool disjoint = (j < c) || (i > d);
            bool nested = (i < c && d < j) || (c < i && j < d);
            if (!(disjoint || nested)) return false;
        }
        return true;
    };
    auto armsFree = [&](int i, int j, int len) {
        for (int k = 0; k < len; k++)
            if (partner[i + k] != -1 || partner[j - k] != -1) return false;
        return true;
    };

    for (auto& s : stems) {
        if (!armsFree(s.i, s.j, s.len)) continue;
        if (!compatible(s.i, s.j)) continue;
        for (int k = 0; k < s.len; k++) {
            partner[s.i + k] = s.j - k;
            partner[s.j - k] = s.i + k;
        }
        sel.push_back({s.i, s.j});
    }

    // fill leftovers with high-weight single pairs (weight-greedy over free positions)
    struct Cand { int w, i, j; };
    vector<Cand> cs;
    for (int i = 0; i < L; i++) {
        if (partner[i] != -1) continue;
        for (int j = i + 4; j < L; j++) {
            if (partner[j] != -1) continue;
            int w = pairW(S[i], S[j]);
            if (w > 0) cs.push_back({w, i, j});
        }
    }
    sort(cs.begin(), cs.end(), [](const Cand& a, const Cand& b) {
        if (a.w != b.w) return a.w > b.w;
        return (a.j - a.i) < (b.j - b.i);
    });
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
