#include <bits/stdc++.h>
using namespace std;

// 2-SAT via implication graph + Tarjan SCC (iterative), O(V + C).
// Variable t in [0..n-1] has two literals: true -> node 2*t, false -> node 2*t+1.
// A clause (lit_a OR lit_b) adds the two implications (~a -> b) and (~b -> a).

int n;                     // number of boolean variables
vector<vector<int>> adj;   // implication graph on 2n nodes

// literal encoding: node id = 2*var + (val?0:1)
static inline int litNode(int var, int val) { return 2 * var + (val ? 0 : 1); }
static inline int negNode(int node)         { return node ^ 1; }

// Tarjan iterative
vector<int> comp, low, num;
vector<char> onstk;
vector<int> stk;
int idx_counter, comp_counter;

void tarjan_all(int N) {
    comp.assign(N, -1);
    low.assign(N, 0);
    num.assign(N, -1);
    onstk.assign(N, 0);
    stk.clear();
    idx_counter = 0;
    comp_counter = 0;

    // iterative DFS frame: node + position in its adjacency list
    vector<pair<int,int>> frame;
    frame.reserve(N);
    for (int s = 0; s < N; s++) {
        if (num[s] != -1) continue;
        frame.push_back({s, 0});
        num[s] = low[s] = idx_counter++;
        stk.push_back(s);
        onstk[s] = 1;
        while (!frame.empty()) {
            int u = frame.back().first;
            int &pi = frame.back().second;
            if (pi < (int)adj[u].size()) {
                int v = adj[u][pi++];
                if (num[v] == -1) {
                    num[v] = low[v] = idx_counter++;
                    stk.push_back(v);
                    onstk[v] = 1;
                    frame.push_back({v, 0});
                } else if (onstk[v]) {
                    low[u] = min(low[u], num[v]);
                }
            } else {
                if (low[u] == num[u]) {
                    while (true) {
                        int w = stk.back();
                        stk.pop_back();
                        onstk[w] = 0;
                        comp[w] = comp_counter;
                        if (w == u) break;
                    }
                    comp_counter++;
                }
                frame.pop_back();
                if (!frame.empty()) {
                    int p = frame.back().first;
                    low[p] = min(low[p], low[u]);
                }
            }
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int m;                  // number of clauses
    if (!(cin >> n >> m)) return 0;

    adj.assign(2 * n, {});

    for (int c = 0; c < m; c++) {
        int i, a, j, b;
        cin >> i >> a >> j >> b;   // clause (var i == a) OR (var j == b)
        int la = litNode(i, a);    // literal "var i is a"
        int lb = litNode(j, b);    // literal "var j is b"
        // (~la -> lb) and (~lb -> la)
        adj[negNode(la)].push_back(lb);
        adj[negNode(lb)].push_back(la);
    }

    tarjan_all(2 * n);

    // satisfiable iff for every variable, the two literals are in different SCCs
    for (int t = 0; t < n; t++) {
        if (comp[2 * t] == comp[2 * t + 1]) {
            cout << "NO\n";
            return 0;
        }
    }

    // Tarjan numbers SCCs in reverse topological order:
    // comp index increases as we move "earlier" in topo order.
    // A literal is TRUE if its component comes later in topological order
    // than its negation's, i.e. smaller Tarjan comp index.
    cout << "YES\n";
    for (int t = 0; t < n; t++) {
        // node 2*t is literal "var t = true"; node 2*t+1 is "var t = false"
        // choose value true iff comp[true-literal] < comp[false-literal]
        int val = (comp[2 * t] < comp[2 * t + 1]) ? 1 : 0;
        cout << val;
        cout << (t + 1 < n ? ' ' : '\n');
    }
    if (n == 0) cout << "\n";
    return 0;
}
