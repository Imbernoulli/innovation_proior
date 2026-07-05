// TIER: greedy
// Append-greedy: consider jobs in order of decreasing penalty; tentatively tack
// [pickup, delivery] onto the END of the current route. Serve the job iff the
// marginal driving increase is strictly cheaper than the forfeited penalty.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, depot;
vector<int> X, Y, D, P, Q, W;

static inline ll darr(int a, int b) {
    return (ll)abs(X[a] - X[b]) + abs(Y[a] - Y[b]) + D[b];
}

int main() {
    if (scanf("%d %d %d", &N, &M, &depot) != 3) return 0;
    X.assign(N + 1, 0); Y.assign(N + 1, 0); D.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) scanf("%d %d %d", &X[i], &Y[i], &D[i]);
    P.assign(M + 1, 0); Q.assign(M + 1, 0); W.assign(M + 1, 0);
    for (int j = 1; j <= M; j++) scanf("%d %d %d", &P[j], &Q[j], &W[j]);

    vector<int> order(M);
    for (int j = 0; j < M; j++) order[j] = j + 1;
    sort(order.begin(), order.end(), [&](int a, int b){ return W[a] > W[b]; });

    // route as a list of visited intersection ids (excluding the depot endpoints)
    vector<int> loc;             // locations in order
    vector<pair<int,int>> ev;    // (type, job) matching loc
    int lastLoc = depot;         // last visited intersection before returning

    for (int j : order) {
        // current cost of returning home from lastLoc
        ll oldTail = (loc.empty() ? 0 : darr(lastLoc, depot));
        // appended legs: lastLoc -> p_j -> q_j -> depot
        ll add = darr(lastLoc, P[j]) + darr(P[j], Q[j]) + darr(Q[j], depot);
        ll marginal = add - oldTail;
        if (marginal < W[j]) {
            ev.push_back({1, j});
            ev.push_back({2, j});
            loc.push_back(P[j]);
            loc.push_back(Q[j]);
            lastLoc = Q[j];
        }
    }

    printf("%d\n", (int)ev.size());
    for (auto& e : ev) printf("%d %d\n", e.first, e.second);
    return 0;
}
