// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
// The obvious "recipe" approach: rank each network's nodes by degree and pair
// same-rank nodes together (hub<->hub, leaf<->leaf). This LOOKS balanced/robust
// ("keep important things paired with important things") but it is exactly the
// trap: it routes power-hub failures straight onto the fragile comm-net relay
// hubs, fragmenting the comm layer's own giant component and triggering a
// cascading collapse whenever the attack targets power hubs.
int main() {
    int N, Mp, Mc;
    if (!(cin >> N >> Mp >> Mc)) return 0;
    vector<int> degP(N, 0), degC(N, 0);
    for (int e = 0; e < Mp; e++) {
        int a, b; cin >> a >> b; a--; b--;
        degP[a]++; degP[b]++;
    }
    for (int e = 0; e < Mc; e++) {
        int a, b; cin >> a >> b; a--; b--;
        degC[a]++; degC[b]++;
    }
    // scenarios are irrelevant to this heuristic; consume nothing further.

    vector<int> pOrder(N), cOrder(N);
    for (int i = 0; i < N; i++) pOrder[i] = i, cOrder[i] = i;
    sort(pOrder.begin(), pOrder.end(), [&](int a, int b) {
        if (degP[a] != degP[b]) return degP[a] > degP[b];
        return a < b;
    });
    sort(cOrder.begin(), cOrder.end(), [&](int a, int b) {
        if (degC[a] != degC[b]) return degC[a] > degC[b];   // descending: hub<->hub
        return a < b;
    });

    vector<int> matchPtoC(N);
    for (int r = 0; r < N; r++) matchPtoC[pOrder[r]] = cOrder[r];

    for (int i = 0; i < N; i++) printf("%d\n", matchPtoC[i] + 1);
    return 0;
}
