// TIER: greedy
// The obvious strong-coder approach: single continuous marginal-gain pass. Round
// by round, ignite whichever remaining candidate adds the most total detonated
// value RIGHT NOW (given what is already ignited), never revisiting the choice.
// This is a legitimate, generally-good heuristic (it never wastes a pick inside an
// already-saturated cluster, since a redundant pick shows zero marginal gain) --
// but it is myopic across components: a shielded target that needs TWO distinct
// feeder chains lit together never looks attractive round-by-round, because
// lighting just one feeder, by itself, unlocks nothing. Each feeder's marginal
// gain (its own small chain value alone) loses to a decoy's larger standalone
// value or another cluster's immediate payoff, so the greedy pass never commits
// budget to both feeders of the same target and the shielded value is left dark.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M;
vector<ll> X, Y, V, R, T;
vector<int> C;
vector<vector<int>> outAdj;

ll simulate(const vector<int>& ignite){
    vector<char> det(N, 0);
    vector<int> cnt(N, 0);
    vector<int> q; q.reserve(N);
    size_t head = 0;
    for (int i : ignite) if (!det[i]){ det[i] = 1; q.push_back(i); }
    while (head < q.size()){
        int u = q[head++];
        for (int j : outAdj[u]){
            if (det[j]) continue;
            if (++cnt[j] >= T[j]){ det[j] = 1; q.push_back(j); }
        }
    }
    ll tot = 0;
    for (int i = 0; i < N; i++) if (det[i]) tot += V[i];
    return tot;
}

int main(){
    if (scanf("%d %d", &N, &M) != 2) return 0;
    X.assign(N, 0); Y.assign(N, 0); V.assign(N, 0); R.assign(N, 0); T.assign(N, 0);
    C.assign(N, 0);
    for (int i = 0; i < N; i++)
        scanf("%lld %lld %lld %lld %lld %d", &X[i], &Y[i], &V[i], &R[i], &T[i], &C[i]);

    outAdj.assign(N, {});
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j){
                ll dx = X[i] - X[j], dy = Y[i] - Y[j];
                if (dx * dx + dy * dy <= R[i] * R[i]) outAdj[i].push_back(j);
            }

    vector<int> chosen;
    vector<char> used(N, 0);
    for (int step = 0; step < M; step++){
        ll bestVal = -1; int bestI = -1;
        for (int i = 0; i < N; i++){
            if (C[i] != 1 || used[i]) continue;
            vector<int> trial = chosen; trial.push_back(i);
            ll val = simulate(trial);
            if (val > bestVal || (val == bestVal && (bestI == -1 || i < bestI))){
                bestVal = val; bestI = i;
            }
        }
        if (bestI == -1) break;
        used[bestI] = 1;
        chosen.push_back(bestI);
    }
    for (size_t i = 0; i < chosen.size(); i++)
        printf("%d%c", chosen[i] + 1, i + 1 < chosen.size() ? ' ' : '\n');
    return 0;
}
