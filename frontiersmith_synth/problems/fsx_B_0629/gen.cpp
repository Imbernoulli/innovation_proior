#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Master Keys for the Wing Complex"   family: masterkey-template-cover
//
// K building wings (scenarios), each with a set of demanded door-code strings over
// alphabet {0,1,2,3}, length L. The solver designs a small set of MASTER-KEY
// TEMPLATES; a demand is served if it lies within the template's Hamming radius.
// A per-template radius costs (r+1)^2 units from a shared budget (radius-covering-
// design). Score = MIN over wings of the served fraction (min-scenario-coverage).
//
// PLANTED STRUCTURE (the trap): each wing's demands are corrupted copies of a few
// hidden base "motifs". TWO "showcase" wings are built from a single dominant motif
// each (low motif-entropy, huge population) -- they are cheap and enormous, so any
// raw-frequency / pooled-coverage approach finds them immediately. The remaining
// "annex" wings are built from FIVE motifs SHARED verbatim across every annex wing
// (70% of an annex wing's demands) plus one motif UNIQUE to that annex wing (30%).
// A frequency-pooled greedy discovers the two showcase motifs and the five shared
// motifs (their pooled aggregate is large), but the eight small unique residuals
// look individually unattractive and a FIXED, budget-hungry radius exhausts the
// budget before all of them are reached -- several annex wings are permanently
// stuck at ~70% while the showcase wings sit at 100%. The intended insight is to
// notice the objective is a MINIMUM (not a sum): apportion the radius budget by
// each wing's remaining shortfall, and recognize that one template near a SHARED
// motif lifts every annex wing's floor at once for the price of a single template --
// freeing budget to close the small unique residuals that a sum-maximizer ignores.
// -----------------------------------------------------------------------------

static const int L = 24;
static const int Q = 4;              // alphabet '0'..'3'
static const int RMAX = 6;

static string randMotif() {
    string s(L, '0');
    for (int i = 0; i < L; i++) s[i] = char('0' + rnd.next(0, Q - 1));
    return s;
}

// flip exactly c distinct positions of `base` to a DIFFERENT symbol.
static string corrupt(const string &base, int c) {
    string s = base;
    vector<int> pos(L);
    for (int i = 0; i < L; i++) pos[i] = i;
    for (int i = 0; i < c && i < L; i++) {
        int j = rnd.next(i, L - 1);
        swap(pos[i], pos[j]);
        int p = pos[i];
        int old = s[p] - '0';
        int nw = rnd.next(0, Q - 2);
        if (nw >= old) nw++;
        s[p] = char('0' + nw);
    }
    return s;
}

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);            // 1..10

    static const int Ks[10] = {3, 4, 5, 6, 7, 8, 8, 9, 10, 10};
    int K = Ks[t - 1];
    // always leave at least 2 annex ("hard") wings so the shared-motif structure
    // (the whole point of the family) is present even on the smallest tests.
    int EASY_COUNT = min(2, max(1, K - 2));
    int HARD_COUNT = K - EASY_COUNT;

    int EASY_N = 400 + 700 * (t - 1);      // 400 .. 6700
    int HARD_N = 60 + 70 * (t - 1);        // 60  .. 690

    int M_max = min(4 * K, 40);
    ll Budget = 16LL * K;

    // ---- the shared "annex motif" pool: 5 base strings common to EVERY hard wing ----
    vector<string> shared(5);
    for (auto &s : shared) s = randMotif();

    printf("%d %d %d %d %lld\n", K, M_max, L, RMAX, Budget);

    for (int k = 0; k < K; k++) {
        if (k < EASY_COUNT) {
            // showcase wing: single dominant motif, small corruption spread {0,1,2}
            string motif = randMotif();
            int n = EASY_N;
            printf("%d\n", n);
            for (int i = 0; i < n; i++) {
                int c = rnd.next(0, 2);
                printf("%s\n", corrupt(motif, c).c_str());
            }
        } else {
            // annex wing: 70% mass on the 5 SHARED motifs, 30% on one unique motif,
            // corruption spread {0,1,2,3,4} -- wide enough that TWO raw samples of
            // the same motif can sit up to distance 8 apart, so a fixed-radius
            // template centered on a raw sample only ever reaches a slice of its
            // own motif; a denoised (majority-vote) center reaches the whole motif
            // for the same convex per-radius budget.
            string uniq = randMotif();
            int n = HARD_N;
            printf("%d\n", n);
            for (int i = 0; i < n; i++) {
                int c = rnd.next(0, 4);
                const string *base;
                if (rnd.next(0, 99) < 70) base = &shared[rnd.next(0, 4)];
                else base = &uniq;
                printf("%s\n", corrupt(*base, c).c_str());
            }
        }
    }
    return 0;
}
