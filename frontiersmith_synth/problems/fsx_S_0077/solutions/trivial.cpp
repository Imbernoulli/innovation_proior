// TIER: trivial
// Fully-serial baseline: every pass on its fastest dam, run one after another on a
// single global timeline. Makespan == B, so ratio == 0.1 exactly.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    long long clock = 0;
    for (int j = 0; j < n; j++) {
        int o; scanf("%d", &o);
        string line;
        for (int k = 0; k < o; k++) {
            int c; scanf("%d", &c);
            int bestDam = -1, bestDur = INT_MAX;
            for (int e = 0; e < c; e++) {
                int dam, dur; scanf("%d %d", &dam, &dur);
                if (dur < bestDur) { bestDur = dur; bestDam = dam; }
            }
            line += to_string(bestDam) + " " + to_string(clock);
            if (k + 1 < o) line += " ";
            clock += bestDur;
        }
        printf("%s\n", line.c_str());
    }
    return 0;
}
