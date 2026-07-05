// TIER: greedy
// Cheap-first interdiction: cut the lowest-cost lines one by one, keeping s-t connected and
// within budget. Maximizes the NUMBER of cuts but ignores where current actually flows, so it
// raises R modestly and often wastes budget on lines that carry little current.
#include <bits/stdc++.h>
using namespace std;

int n, m, s, t;
long long B;
vector<int> eu, ev, er, ec;

bool connected(const vector<char>& alive) {
    vector<vector<int>> g(n + 1);
    for (int e = 0; e < m; e++) if (alive[e]) { g[eu[e]].push_back(ev[e]); g[ev[e]].push_back(eu[e]); }
    vector<char> seen(n + 1, 0); vector<int> st = {s}; seen[s] = 1;
    while (!st.empty()) { int u = st.back(); st.pop_back(); for (int w : g[u]) if (!seen[w]) { seen[w] = 1; st.push_back(w); } }
    return seen[t];
}

int main() {
    scanf("%d %d %d %d %lld", &n, &m, &s, &t, &B);
    eu.resize(m); ev.resize(m); er.resize(m); ec.resize(m);
    for (int e = 0; e < m; e++) scanf("%d %d %d %d", &eu[e], &ev[e], &er[e], &ec[e]);

    vector<int> order(m);
    for (int i = 0; i < m; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (ec[a] != ec[b]) return ec[a] < ec[b];      // cheapest first
        return er[a] < er[b];
    });

    vector<char> alive(m, 1);
    long long spent = 0;
    vector<int> cut;
    for (int e : order) {
        if (spent + ec[e] > B) continue;
        alive[e] = 0;
        if (connected(alive)) { spent += ec[e]; cut.push_back(e + 1); }
        else alive[e] = 1;                              // would island t -> revert
    }

    printf("%d\n", (int)cut.size());
    for (size_t i = 0; i < cut.size(); i++) printf("%d%c", cut[i], i + 1 == cut.size() ? '\n' : ' ');
    if (cut.empty()) printf("\n");
    return 0;
}
