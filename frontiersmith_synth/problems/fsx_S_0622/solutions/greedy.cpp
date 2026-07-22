// TIER: greedy
// The obvious recipe: immunize people in decreasing order of raw STRUCTURAL
// degree (ignore edge thresholds/scenarios), skipping anyone who would blow the
// budget. Community hubs/members have far more degree than the handful of
// degree-2 bridge people, so this recipe burns the whole budget deep inside one
// redundant community -- which barely dents the network-wide giant lump, since
// many alternate internal paths survive -- and never reaches the bridges that
// actually hold separate communities together.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int N, M, ELL, K, S;
    if (scanf("%d %d %d %d %d", &N, &M, &ELL, &K, &S) != 5) return 0;
    (void)ELL; (void)S;
    vector<int> deg(N + 1, 0);
    for (int j = 0; j < M; j++){
        int u, v, t;
        scanf("%d %d %d", &u, &v, &t);
        deg[u]++; deg[v]++;
    }
    vector<int> cost(N + 1);
    for (int i = 1; i <= N; i++) scanf("%d", &cost[i]);
    // p_s scenario strengths are ignored by this naive approach.
    vector<int> ord(N);
    for (int i = 0; i < N; i++) ord[i] = i + 1;
    stable_sort(ord.begin(), ord.end(), [&](int a, int b){ return deg[a] > deg[b]; });

    vector<int> chosen;
    long long spent = 0;
    for (int id : ord){
        if (spent + cost[id] > K) continue;
        spent += cost[id];
        chosen.push_back(id);
    }
    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++)
        printf("%d%c", chosen[i], (i + 1 < chosen.size()) ? ' ' : '\n');
    if (chosen.empty()) printf("\n");
    return 0;
}
