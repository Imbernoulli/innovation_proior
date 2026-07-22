#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Wildcard Pattern Superstring.
//
// testId is a difficulty/structure ladder. Each test is built from zero or more PLANTED
// "clusters": a cluster of `count` patterns is cut as overlapping windows (window i starts
// at i*step) of a single hidden random core string of length `patLen + (count-1)*step`,
// then each pattern is independently wildcard-masked with probability `maskP` (a position
// becomes '?' instead of its true letter). Consecutive windows in the SAME cluster share a
// true underlying overlap of `patLen - step` characters, but at high maskP a plain
// literal-character overlap check (treating '?' as an ordinary character) sees almost none
// of it, while the wildcard-compatible overlap check sees the full true overlap. Tests are
// also padded with independent "noise" patterns (unrelated random windows, low mask) so the
// planted clusters must be found among distractors rather than handed to the solver in a
// convenient order (the whole pattern list is shuffled before printing).
//
// Tests 6, 7 are heavy TRAP cases (high maskP, tight step): literal-overlap greedy collapses
// almost to the trivial baseline. Test 8 is a NEEDLE case: a small compressible cluster is
// hidden among a majority of unrelated noise patterns. Tests 9-10 combine multiple clusters
// at scale and fill the declared size envelope (n up to 200, |p_i| up to 30).
static const string ALPHA = "ABCDEF";

static string randomCore(int len) {
    string s(len, 'A');
    for (int i = 0; i < len; i++) s[i] = ALPHA[rnd.next((int)ALPHA.size())];
    return s;
}

static string maskPattern(const string &src, double maskP) {
    string p = src;
    bool anyConcrete = false;
    for (char &c : p) {
        if (rnd.next(0.0, 1.0) < maskP) c = '?';
        else anyConcrete = true;
    }
    if (!anyConcrete) p[0] = src[0]; // guard: keep at least one concrete anchor
    return p;
}

// A cluster of `count` overlapping windows of a fresh hidden core, each masked independently.
static vector<string> makeCluster(int count, int patLen, int step, double maskP) {
    int coreLen = patLen + (count - 1) * step;
    string T = randomCore(coreLen);
    vector<string> out;
    out.reserve(count);
    for (int k = 0; k < count; k++) {
        int start = k * step;
        string window = T.substr(start, patLen);
        out.push_back(maskPattern(window, maskP));
    }
    return out;
}

static vector<string> makeNoise(int count, int Lmin, int Lmax, double maskP) {
    vector<string> out;
    out.reserve(count);
    for (int i = 0; i < count; i++) {
        int L = rnd.next(Lmin, Lmax);
        string s = randomCore(L);
        out.push_back(maskPattern(s, maskP));
    }
    return out;
}

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    vector<string> patterns;

    switch (idx) {
        case 1: {
            auto c = makeCluster(4, 8, 3, 0.0);
            auto nz = makeNoise(1, 5, 7, 0.10);
            patterns.insert(patterns.end(), c.begin(), c.end());
            patterns.insert(patterns.end(), nz.begin(), nz.end());
            break;
        }
        case 2: {
            auto c = makeCluster(8, 9, 3, 0.0);
            auto nz = makeNoise(2, 6, 9, 0.10);
            patterns.insert(patterns.end(), c.begin(), c.end());
            patterns.insert(patterns.end(), nz.begin(), nz.end());
            break;
        }
        case 3: {
            auto c = makeCluster(14, 10, 3, 0.0);
            auto nz = makeNoise(2, 7, 10, 0.15);
            patterns.insert(patterns.end(), c.begin(), c.end());
            patterns.insert(patterns.end(), nz.begin(), nz.end());
            break;
        }
        case 4: {
            auto c1 = makeCluster(18, 12, 3, 0.0);
            auto c2 = makeCluster(10, 10, 3, 0.0);
            auto nz = makeNoise(3, 7, 11, 0.15);
            patterns.insert(patterns.end(), c1.begin(), c1.end());
            patterns.insert(patterns.end(), c2.begin(), c2.end());
            patterns.insert(patterns.end(), nz.begin(), nz.end());
            break;
        }
        case 5: {
            auto c = makeCluster(30, 14, 3, 0.15);
            auto nz = makeNoise(10, 8, 13, 0.15);
            patterns.insert(patterns.end(), c.begin(), c.end());
            patterns.insert(patterns.end(), nz.begin(), nz.end());
            break;
        }
        case 6: { // TRAP: heavy masking, tight step, single big cluster
            auto c = makeCluster(50, 16, 2, 0.55);
            auto nz = makeNoise(30, 8, 14, 0.15);
            patterns.insert(patterns.end(), c.begin(), c.end());
            patterns.insert(patterns.end(), nz.begin(), nz.end());
            break;
        }
        case 7: { // TRAP: two heavily-masked clusters
            auto c1 = makeCluster(40, 14, 2, 0.60);
            auto c2 = makeCluster(30, 14, 2, 0.55);
            auto nz = makeNoise(30, 8, 14, 0.15);
            patterns.insert(patterns.end(), c1.begin(), c1.end());
            patterns.insert(patterns.end(), c2.begin(), c2.end());
            patterns.insert(patterns.end(), nz.begin(), nz.end());
            break;
        }
        case 8: { // NEEDLE: small compressible cluster hidden among mostly-noise patterns
            auto c = makeCluster(25, 18, 2, 0.60);
            auto nz = makeNoise(95, 8, 14, 0.10);
            patterns.insert(patterns.end(), c.begin(), c.end());
            patterns.insert(patterns.end(), nz.begin(), nz.end());
            break;
        }
        case 9: { // large scale, multi-cluster
            auto c1 = makeCluster(50, 18, 2, 0.30);
            auto c2 = makeCluster(50, 16, 2, 0.30);
            auto nz = makeNoise(40, 8, 14, 0.15);
            patterns.insert(patterns.end(), c1.begin(), c1.end());
            patterns.insert(patterns.end(), c2.begin(), c2.end());
            patterns.insert(patterns.end(), nz.begin(), nz.end());
            break;
        }
        case 10: { // largest / most adversarial, fills the size envelope
            auto c1 = makeCluster(70, 20, 2, 0.60);
            auto c2 = makeCluster(60, 18, 2, 0.55);
            auto nz = makeNoise(70, 10, 16, 0.15);
            patterns.insert(patterns.end(), c1.begin(), c1.end());
            patterns.insert(patterns.end(), c2.begin(), c2.end());
            patterns.insert(patterns.end(), nz.begin(), nz.end());
            break;
        }
    }

    // Shuffle so the input order gives no hint about cluster membership or chain order.
    for (int i = (int)patterns.size() - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(patterns[i], patterns[j]);
    }

    int n = (int)patterns.size();
    printf("%d\n", n);
    for (int i = 0; i < n; i++) printf("%s\n", patterns[i].c_str());

    return 0;
}
