// Checker/scorer for "Cooperative Ribozyme Folding with Aptamer Pockets".
// Reads the RNA + pockets from inf, the participant's base pairs from ouf, validates
// feasibility strictly (min-loop, complementarity, matching, non-crossing), computes the
// cooperative energy F, and scores it against the greedy-nested-matching baseline B.
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static const int STK = 3;

// base-pair weight; -1 if the two bases are not an allowed pair
static int pairW(char a, char b) {
    auto is = [&](char x, char y) { return (a == x && b == y) || (a == y && b == x); };
    if (is('C', 'G')) return 3;
    if (is('A', 'U')) return 2;
    if (is('G', 'U')) return 1;
    return -1;
}
static bool comp(char a, char b) { return pairW(a, b) > 0; }

// Cooperative energy of a structure given partner[] (partner[i] = j or -1).
static long long energy(int L, const string& seq, const vector<int>& partner,
                        const vector<pair<int,int>>& pockets) {
    long long F = 0;
    // base pairing
    for (int i = 0; i < L; i++) {
        int j = partner[i];
        if (j > i) F += pairW(seq[i], seq[j]);
    }
    // cooperative stacking: pair (i,j) whose inner neighbor (i+1,j-1) is also a pair
    for (int i = 0; i < L; i++) {
        int j = partner[i];
        if (j > i && i + 1 < L && j - 1 >= 0 && partner[i + 1] == j - 1) F += STK;
    }
    // helical flag: a pair with a stacked neighbor (inner or outer)
    vector<char> helical(L, 0);
    for (int i = 0; i < L; i++) {
        int j = partner[i];
        if (j < 0) continue;
        int a = min(i, j), b = max(i, j);
        bool st = false;
        if (a + 1 < b - 1 && partner[a + 1] == b - 1) st = true;
        if (a - 1 >= 0 && b + 1 < L && partner[a - 1] == b + 1) st = true;
        if (st) helical[i] = 1;
    }
    // pockets: unpaired AND enclosed by some helical pair
    for (auto& pk : pockets) {
        int p = pk.first, v = pk.second;
        if (p < 0 || p >= L) continue;
        if (partner[p] != -1) continue;
        bool ok = false;
        for (int i = 0; i < L && !ok; i++) {
            int j = partner[i];
            if (j > i && i < p && p < j && helical[i]) ok = true;
        }
        if (ok) F += v;
    }
    return F;
}

// reference greedy nested matching (the baseline B is this structure's energy).
// Scan j left to right, keeping a stack of unpaired positions. For each j, find the
// nearest position i still on the stack (topmost first) that satisfies min-loop and
// complementarity; pair them and discard every stack entry above i (they stay unpaired).
// Otherwise push j. This is non-crossing by construction.
static vector<int> greedyStruct(int L, const string& seq) {
    vector<int> partner(L, -1);
    vector<int> st;
    for (int j = 0; j < L; j++) {
        int found = -1;
        for (int idx = (int)st.size() - 1; idx >= 0; idx--) {
            int i = st[idx];
            if (j - i - 1 >= 3 && comp(seq[i], seq[j])) { found = idx; break; }
        }
        if (found >= 0) {
            int i = st[found];
            partner[i] = j; partner[j] = i;
            st.resize(found);
        } else {
            st.push_back(j);
        }
    }
    return partner;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int L = inf.readInt();
    string seq = inf.readWord();
    int P = inf.readInt();
    vector<pair<int,int>> pockets;
    for (int k = 0; k < P; k++) {
        int p = inf.readInt() - 1;
        int v = inf.readInt();
        pockets.push_back({p, v});
    }

    // --- read + strictly validate participant output ---
    int K = ouf.readInt(0, L, "K");
    vector<int> partner(L, -1);
    for (int k = 0; k < K; k++) {
        int i = ouf.readInt(1, L, "i") - 1;
        int j = ouf.readInt(1, L, "j") - 1;
        if (i >= j) quitf(_wa, "pair %d: need i < j", k + 1);
        if (j - i - 1 < 3) quitf(_wa, "pair %d: loop length %d < 3", k + 1, j - i - 1);
        if (!comp(seq[i], seq[j]))
            quitf(_wa, "pair %d: bases %c-%c not complementary", k + 1, seq[i], seq[j]);
        if (partner[i] != -1 || partner[j] != -1)
            quitf(_wa, "pair %d: a position is reused", k + 1);
        partner[i] = j; partner[j] = i;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // non-crossing (nested) validation via a bracket scan
    {
        vector<int> stk;
        for (int pos = 0; pos < L; pos++) {
            if (partner[pos] == -1) continue;
            if (partner[pos] > pos) stk.push_back(pos);
            else {
                if (stk.empty() || stk.back() != partner[pos])
                    quitf(_wa, "pairs cross (structure is not nested)");
                stk.pop_back();
            }
        }
        if (!stk.empty()) quitf(_fail, "internal: unbalanced partner array");
    }

    long long F = energy(L, seq, partner, pockets);
    vector<int> bp = greedyStruct(L, seq);
    long long B = energy(L, seq, bp, pockets);
    if (B < 1) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    if (sc < 0) sc = 0;
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
}
