#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- "Bidders Who Agreed Beforehand"
//
// Prints T independent bundled markets. Market 0 of every test is a PLANTED easy case:
// a guaranteed size-2 ring whose members are deliberately given HIGH true rates (so the
// grader's own "highest total bid" baseline construction always catches at least one true
// ring member somewhere -> baseline B is provably positive on every test). Every other
// market is a realistic TRAP case: cartel members' rates are drawn from the SAME range as
// honest bidders (never outliers), and roughly a third of the honest bidders are expensive
// "decoys" (legitimately high true cost) so that raw high-bid flagging is actively misled.

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int T = 6 + 2 * (testId - 1);           // 6 .. 24
    int nmax = min(16, 6 + testId);         // grows with testId, capped at 16
    int nmin = 4;
    int mmax = min(14, 6 + testId);
    int mmin = 6;

    printf("%d\n", T);

    for (int t = 0; t < T; t++) {
        bool forceEasy = (t == 0);

        int n = rnd.next(nmin, nmax);
        int m = rnd.next(mmin, mmax);

        vector<int> s(m);
        for (int a = 0; a < m; a++) s[a] = rnd.next(15, 300);

        vector<int> k(n);
        for (int i = 0; i < n; i++) k[i] = rnd.next(3, 45);

        bool hasCartel = false;
        int groupSize = 0;
        vector<int> members;
        vector<int> rotOrder;
        int markup = 0;

        if (forceEasy) {
            hasCartel = true;
            groupSize = 2;
        } else {
            double p = min(0.35 + 0.05 * min(testId, 6), 0.65);
            hasCartel = (rnd.next(0.0, 1.0) < p);
        }

        if (hasCartel) {
            vector<int> idx(n);
            iota(idx.begin(), idx.end(), 0);
            for (int i = n - 1; i > 0; i--) { int j = rnd.next(0, i); swap(idx[i], idx[j]); }
            if (!forceEasy) groupSize = (n >= 5 && rnd.next(0, 1) == 1) ? 3 : 2;
            members.assign(idx.begin(), idx.begin() + groupSize);

            if (forceEasy) {
                for (int mem : members) k[mem] = rnd.next(55, 70);  // dominates all honest k here
            } else {
                for (int mem : members) k[mem] = rnd.next(3, 45);   // same range as honest -> no outlier
            }

            markup = rnd.next(1, 5);

            rotOrder = members;
            for (int i = (int)rotOrder.size() - 1; i > 0; i--) {
                int j = rnd.next(0, i);
                swap(rotOrder[i], rotOrder[j]);
            }

            // avoid a degenerate k collision that would make a member's implied-rate set
            // collapse to a single value (own rate == some cover-round implied rate)
            for (int guard = 0; guard < 60; guard++) {
                bool collision = false;
                for (int a : members)
                    for (int b : members)
                        if (a != b && k[a] == k[b] + markup) collision = true;
                if (!collision) break;
                int victim = members[rnd.next(0, (int)members.size() - 1)];
                k[victim] = forceEasy ? rnd.next(55, 70) : rnd.next(3, 45);
            }
        }

        // decoys: legitimately expensive honest bidders (trap ammunition); never in market 0
        if (!forceEasy) {
            for (int i = 0; i < n; i++) {
                bool isMember = hasCartel && find(members.begin(), members.end(), i) != members.end();
                if (isMember) continue;
                if (rnd.next(0.0, 1.0) < 0.30) k[i] = rnd.next(70, 140);
            }
        }

        printf("%d %d\n", n, m);
        for (int a = 0; a < m; a++) printf("%d%c", s[a], a + 1 < m ? ' ' : '\n');

        for (int i = 0; i < n; i++) {
            bool isMember = hasCartel && find(members.begin(), members.end(), i) != members.end();
            for (int a = 0; a < m; a++) {
                int rate;
                if (isMember) {
                    int winner = rotOrder[a % (int)rotOrder.size()];
                    rate = (winner == i) ? k[i] : (k[winner] + markup);
                } else {
                    rate = k[i];
                }
                int noise = rnd.next(0, s[a] - 1);
                int bid = rate * s[a] + noise;
                printf("%d%c", bid, a + 1 < m ? ' ' : '\n');
            }
        }
    }
    return 0;
}
