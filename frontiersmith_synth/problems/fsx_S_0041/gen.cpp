#include "testlib.h"
#include <vector>
#include <set>
#include <algorithm>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // Difficulty ladder: grow jobs, machines, chain length, flexibility, and
    // introduce a skewed "slow bottleneck asset" at higher testIds.
    int J = min(14, 2 + testId);                 // testId1 -> 3 jobs, testId10 -> 12
    int M = min(8, 2 + (testId - 1) / 2);        // testId1 -> 2, testId10 -> 6
    if (M < 2) M = 2;

    int kmin = 2;
    int kmax = min(10, 2 + testId / 2);          // chain length ceiling grows
    if (kmax < kmin) kmax = kmin;

    int maxElig = min(M, 2 + testId / 3);        // flexibility grows, capped by M
    if (maxElig < 1) maxElig = 1;

    bool skew = (testId >= 5);
    int slow = 1 + rnd.next(M);                  // a designated bottleneck asset

    // Respect the total-task cap of 200.
    int cap = 200;

    // First build per-job chains.
    vector<vector<vector<pair<int,int>>>> jobs; // jobs[j][task] = list of (asset,dur)
    int total = 0;
    for (int j = 0; j < J; j++) {
        int k = rnd.next(kmin, kmax);
        if (total + k > cap) k = cap - total;
        if (k <= 0) break;
        total += k;
        vector<vector<pair<int,int>>> chain;
        for (int t = 0; t < k; t++) {
            int e = rnd.next(1, maxElig);
            // pick e distinct assets
            set<int> chosen;
            while ((int)chosen.size() < e) chosen.insert(1 + rnd.next(M));
            vector<int> assets(chosen.begin(), chosen.end());
            shuffle(assets.begin(), assets.end());
            vector<pair<int,int>> opts;
            for (int a : assets) {
                int d;
                if (skew && a == slow) d = rnd.next(55, 99);       // bottleneck is slow
                else d = rnd.next(1, skew ? 30 : 99);              // alternatives faster
                opts.push_back({a, d});
            }
            chain.push_back(opts);
        }
        jobs.push_back(chain);
    }

    J = (int)jobs.size();

    printf("%d %d\n", J, M);
    for (int j = 0; j < J; j++) {
        printf("%d\n", (int)jobs[j].size());
        for (auto& op : jobs[j]) {
            printf("%d", (int)op.size());
            for (auto& pr : op) printf(" %d %d", pr.first, pr.second);
            printf("\n");
        }
    }
    return 0;
}
