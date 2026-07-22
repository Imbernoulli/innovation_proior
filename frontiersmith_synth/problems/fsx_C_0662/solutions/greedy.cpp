// TIER: greedy
// The obvious "max-weight matching" recipe: sort common lines by cut length
// DESCENDING (biggest single pairing/cluster first) and greedily recruit every
// currently-affordable member. This locks the few heat-tolerant hub parts into
// whichever giant line comes first, exhausting their budgets before the many
// cheaper lines that also needed those hubs ever get a chance to reach quorum.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N, M;
    scanf("%d %d", &N, &M);
    vector<ll> H(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &H[i]);
    vector<vector<int>> mem(M + 1);
    vector<ll> L(M + 1);
    vector<ll> baseline(N + 1, 0);
    for (int g = 1; g <= M; g++){
        int k; ll Lg; scanf("%d %lld", &k, &Lg);
        L[g] = Lg; mem[g].resize(k);
        for (int j = 0; j < k; j++){ scanf("%d", &mem[g][j]); baseline[mem[g][j]] += Lg / 2; }
    }
    vector<ll> remain(N + 1);
    for (int i = 1; i <= N; i++) remain[i] = H[i] - baseline[i];

    vector<int> ord(M); for (int g = 1; g <= M; g++) ord[g - 1] = g;
    sort(ord.begin(), ord.end(), [&](int a, int b){
        if (L[a] != L[b]) return L[a] > L[b];
        return a < b;
    });

    vector<pair<int, vector<int>>> plan;
    for (int g : ord){
        vector<int> ok;
        for (int p : mem[g]) if (remain[p] >= L[g] / 2) ok.push_back(p);
        if ((int)ok.size() >= 2){
            for (int p : ok) remain[p] -= L[g] / 2;
            plan.push_back({g, ok});
        }
    }
    printf("%d\n", (int)plan.size());
    for (auto &pr : plan){
        printf("%d %d", pr.first, (int)pr.second.size());
        for (int p : pr.second) printf(" %d", p);
        printf("\n");
    }
    return 0;
}
