// TIER: trivial
// Order-of-arrival: accept each bid, scanned in input order, iff every item it
// needs still has spare supply. No sorting, no lookahead. This reproduces the
// checker's own internal baseline B exactly.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int M, N;
    scanf("%d %d", &M, &N);
    vector<ll> cap(M + 1);
    for (int j = 1; j <= M; j++) scanf("%lld", &cap[j]);

    vector<int> k(N + 1);
    vector<ll> p(N + 1);
    vector<vector<int>> items(N + 1);
    for (int i = 1; i <= N; i++) {
        scanf("%d %lld", &k[i], &p[i]);
        items[i].resize(k[i]);
        for (int t = 0; t < k[i]; t++) scanf("%d", &items[i][t]);
    }

    vector<ll> rem = cap;
    vector<int> accepted;
    for (int i = 1; i <= N; i++) {
        bool ok = true;
        for (int it : items[i]) if (rem[it] < 1) { ok = false; break; }
        if (ok) {
            for (int it : items[i]) rem[it]--;
            accepted.push_back(i);
        }
    }

    printf("%d\n", (int)accepted.size());
    for (size_t i = 0; i < accepted.size(); i++)
        printf("%d%c", accepted[i], i + 1 == accepted.size() ? '\n' : ' ');
    if (accepted.empty()) printf("\n");
    return 0;
}
