// TIER: trivial
// Serial default: assign every phase its FIRST eligible crew and run all phases one
// after another on a single global timeline. Makespan == B, so ratio == 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    long long cur = 0;
    // buffer output per stage
    for (int j = 0; j < n; j++) {
        int o;
        scanf("%d", &o);
        vector<pair<int, long long>> out;  // (crew, start)
        for (int k = 0; k < o; k++) {
            int e;
            scanf("%d", &e);
            int firstCrew = -1, firstDur = 0;
            for (int idx = 0; idx < e; idx++) {
                int c, d;
                scanf("%d %d", &c, &d);
                if (idx == 0) { firstCrew = c; firstDur = d; }
            }
            out.push_back({firstCrew, cur});
            cur += firstDur;
        }
        for (int k = 0; k < o; k++)
            printf("%d %lld%c", out[k].first, out[k].second, k + 1 == o ? '\n' : ' ');
    }
    return 0;
}
