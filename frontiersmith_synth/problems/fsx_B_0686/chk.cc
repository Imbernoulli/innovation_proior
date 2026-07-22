#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- Wildcard Pattern Superstring.
// Feasibility: every input pattern p_i must occur in the participant's S as a
// wildcard-respecting substring ('?' in p_i matches any character of S; S itself must
// contain only concrete letters). Objective (min): F = |S|. Internal baseline B = sum of
// all pattern lengths (the "lay every pattern end to end, no merging" trivial construction).
// Score: sc = min(1000, 100*B/max(1,F)); ratio = sc/1000, capped at 1.0.

static inline bool isConcrete(char c) { return c >= 'A' && c <= 'F'; }

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    vector<string> pat(n);
    long long sumL = 0;
    for (int i = 0; i < n; i++) {
        pat[i] = inf.readWord();
        sumL += (long long)pat[i].size();
    }

    long long B = max(1LL, sumL);
    long long MAXOUT = 5 * sumL + 100;

    long long Ldecl = ouf.readLong(1, MAXOUT, "L");
    string S = ouf.readWord();
    long long M = (long long)S.size();

    if (M != Ldecl)
        quitf(_wa, "declared length L=%lld does not match actual |S|=%lld", Ldecl, M);
    if (M < 1) quitf(_wa, "empty output string");
    if (M > MAXOUT) quitf(_wa, "output string too long: |S|=%lld > cap=%lld", M, MAXOUT);
    for (char c : S) {
        if (!isConcrete(c))
            quitf(_wa, "output contains a character outside {A..F} (no wildcards allowed in S): '%c'", c);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after S");

    for (int i = 0; i < n; i++) {
        const string &p = pat[i];
        long long L = (long long)p.size();
        bool found = false;
        if (L <= M) {
            for (long long t = 0; t + L <= M && !found; t++) {
                bool ok = true;
                for (long long j = 0; j < L; j++) {
                    char pc = p[j];
                    if (pc != '?' && pc != S[t + j]) { ok = false; break; }
                }
                if (ok) found = true;
            }
        }
        if (!found)
            quitf(_wa, "pattern %d (%s) does not occur in S as a wildcard-respecting substring", i, p.c_str());
    }

    long long F = M;
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
