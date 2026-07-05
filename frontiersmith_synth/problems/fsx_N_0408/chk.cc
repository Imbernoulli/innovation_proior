#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, L;
vector<vector<int>> pat;   // 1-indexed firing patterns

// difference-count buffer for the peak-coupling computation
vector<int> diffCnt;       // size 2L+1, indexed by (a-b)+L
vector<int> touched;

// c(A,B) = max_t | { (a,b): a-b = t } |   (discrete cross-correlation peak)
ll coupling(const vector<int>& A, const vector<int>& B) {
    touched.clear();
    ll best = 0;
    for (int a : A) for (int b : B) {
        int idx = a - b + L;                 // in [0, 2L-1]
        int v = ++diffCnt[idx];
        if (v == 1) touched.push_back(idx);
        if (v > best) best = v;
    }
    for (int idx : touched) diffCnt[idx] = 0; // clear only touched entries
    return best;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    L = inf.readInt();
    diffCnt.assign(2 * L + 2, 0);

    pat.assign(n + 1, {});
    for (int i = 1; i <= n; i++) {
        int s = inf.readInt();
        pat[i].resize(s);
        int prev = -1;
        for (int j = 0; j < s; j++) {
            int p = inf.readInt();
            pat[i][j] = p;
            prev = p;
        }
    }

    // ---- internal baseline: identity schedule cost B ----
    ll B = 0;
    for (int i = 1; i < n; i++) B += coupling(pat[i], pat[i + 1]);
    if (B <= 0) B = 1;   // guard (generator keeps identity costly)

    // ---- read & validate participant's permutation ----
    vector<int> perm(n);
    vector<char> seen(n + 1, 0);
    for (int i = 0; i < n; i++) {
        int x = ouf.readInt(1, n, "order");
        if (seen[x]) quitf(_wa, "observation %d appears more than once", x);
        seen[x] = 1;
        perm[i] = x;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- objective: schedule cost F ----
    ll F = 0;
    for (int i = 0; i + 1 < n; i++)
        F += coupling(pat[perm[i]], pat[perm[i + 1]]);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
