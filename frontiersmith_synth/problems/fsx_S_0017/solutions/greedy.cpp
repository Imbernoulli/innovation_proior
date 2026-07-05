// TIER: greedy
// Job-order list scheduling: process modules one at a time in input order.
// Each operation is appended to the earliest free slot at the END of its
// workstation's current timeline (no gap insertion). Packs machines in
// parallel, so it beats the sequential baseline, but the rigid job order
// leaves it well short of a good interleaving.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int J, M;
    if (scanf("%d %d", &J, &M) != 2) return 0;
    vector<long long> machineEnd(M + 1, 0);
    vector<long long> starts;
    for (int j = 0; j < J; j++) {
        int k; scanf("%d", &k);
        long long jobReady = 0;
        for (int o = 0; o < k; o++) {
            int m, d; scanf("%d %d", &m, &d);
            long long s = max(jobReady, machineEnd[m]);
            starts.push_back(s);
            long long e = s + d;
            machineEnd[m] = e;
            jobReady = e;
        }
    }
    for (size_t i = 0; i < starts.size(); i++)
        printf("%lld\n", starts[i]);
    return 0;
}
