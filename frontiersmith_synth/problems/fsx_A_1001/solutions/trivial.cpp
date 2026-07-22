// TIER: trivial
// Do-nothing baseline: dedicate ONE probe to EACH fault (set its required
// positions, leave everything else 0).  No attempt to notice that two
// dedicated probes might already be compatible.  P = N exactly -> Ratio 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int K, N;
    scanf("%d %d", &K, &N);
    vector<vector<pair<int,int>>> req(N);
    for (int i = 0; i < N; i++) {
        int c; scanf("%d", &c);
        req[i].resize(c);
        for (int k = 0; k < c; k++) scanf("%d %d", &req[i][k].first, &req[i][k].second);
    }
    printf("%d\n", N);
    string probe(K, '0');
    for (int i = 0; i < N; i++) {
        fill(probe.begin(), probe.end(), '0');
        for (auto &pr : req[i]) probe[pr.first] = (pr.second ? '1' : '0');
        probe.push_back('\n');
        fwrite(probe.data(), 1, probe.size(), stdout);
        probe.pop_back();
    }
    return 0;
}
