// TIER: trivial
// Fully sequential schedule: run every operation back-to-back on one global
// timeline. Makespan = sum of all durations = the checker's baseline B -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int J, M;
    if (scanf("%d %d", &J, &M) != 2) return 0;
    long long cum = 0;
    vector<long long> starts;
    for (int j = 0; j < J; j++) {
        int k; scanf("%d", &k);
        for (int o = 0; o < k; o++) {
            int m, d; scanf("%d %d", &m, &d);
            starts.push_back(cum);
            cum += d;
        }
    }
    for (size_t i = 0; i < starts.size(); i++)
        printf("%lld\n", starts[i]);
    return 0;
}
