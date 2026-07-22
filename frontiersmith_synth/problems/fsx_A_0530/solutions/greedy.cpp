// TIER: greedy
// The obvious approach: DEFEND THE FRONTIER. Each step, among the stands about to burn
// (unburnt, unprotected neighbours of the fire) protect the B with the highest value.
// Because the front is wider than B and every valuable grove sits behind a gateway of
// C > B stands, this reacts too late: it only rescues at most B grove stands at the very
// instant a grove is breached and seals no distant gateway -> far below the insight.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N, M, B, S;
    if (scanf("%d %d %d %d", &N, &M, &B, &S) != 4) return 0;
    vector<ll> w(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &w[i]);
    vector<int> src(S);
    for (int i = 0; i < S; i++) scanf("%d", &src[i]);
    vector<vector<int>> adj(N + 1);
    for (int e = 0; e < M; e++){
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v); adj[v].push_back(u);
    }

    vector<char> burnt(N + 1, 0), prt(N + 1, 0);
    vector<int> frontier;
    for (int s : src) if (!burnt[s]){ burnt[s] = 1; frontier.push_back(s); }

    vector<vector<int>> schedule;   // schedule[t-1] = protected ids at step t
    while (!frontier.empty()){
        // candidate threatened valued stands = unburnt/unprotected neighbours with w>0
        vector<int> cand;
        for (int u : frontier)
            for (int v : adj[u])
                if (!burnt[v] && !prt[v] && w[v] > 0) cand.push_back(v);
        sort(cand.begin(), cand.end());
        cand.erase(unique(cand.begin(), cand.end()), cand.end());
        sort(cand.begin(), cand.end(), [&](int a, int b){
            if (w[a] != w[b]) return w[a] > w[b];
            return a < b;
        });
        vector<int> pick;
        for (int i = 0; i < (int)cand.size() && (int)pick.size() < B; i++){
            pick.push_back(cand[i]); prt[cand[i]] = 1;
        }
        schedule.push_back(pick);
        // spread
        vector<int> nxt;
        for (int u : frontier)
            for (int v : adj[u])
                if (!burnt[v] && !prt[v]){ burnt[v] = 1; nxt.push_back(v); }
        frontier.swap(nxt);
    }

    // trim trailing empty steps
    while (!schedule.empty() && schedule.back().empty()) schedule.pop_back();
    printf("%d\n", (int)schedule.size());
    for (auto &st : schedule){
        printf("%d", (int)st.size());
        for (int id : st) printf(" %d", id);
        printf("\n");
    }
    return 0;
}
