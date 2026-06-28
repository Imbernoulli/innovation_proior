#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int k, m;
    if (!(cin >> k >> m)) return 0;
    vector<string> kmers(m);
    for (int i = 0; i < m; i++) cin >> kmers[i];

    // Special case: empty list of k-mers -> empty reconstruction.
    if (m == 0) {
        cout << "\n";
        return 0;
    }

    // Build the de Bruijn graph: each k-mer s is a directed edge from its
    // (k-1)-character prefix to its (k-1)-character suffix. Nodes are the
    // distinct (k-1)-mers; a valid reconstruction that uses every k-mer once
    // is exactly an Eulerian path over these edges.
    unordered_map<string,int> id;
    id.reserve(2 * m * 2 + 16);
    auto getId = [&](const string &s) -> int {
        auto it = id.find(s);
        if (it != id.end()) return it->second;
        int v = (int)id.size();
        id.emplace(s, v);
        return v;
    };

    int n = 0;                 // number of distinct nodes (filled as we go)
    vector<vector<int>> adj;   // adjacency: for node u, list of destination nodes
    vector<int> indeg, outdeg;

    auto ensure = [&](int v) {
        while ((int)adj.size() <= v) { adj.push_back({}); indeg.push_back(0); outdeg.push_back(0); }
    };

    vector<int> uEdge(m), vEdge(m);
    for (int i = 0; i < m; i++) {
        const string &s = kmers[i];
        string pre = s.substr(0, k - 1);
        string suf = s.substr(1, k - 1);
        int u = getId(pre);
        int v = getId(suf);
        ensure(u); ensure(v);
        uEdge[i] = u; vEdge[i] = v;
    }
    n = (int)id.size();
    ensure(n - 1);

    for (int i = 0; i < m; i++) {
        adj[uEdge[i]].push_back(vEdge[i]);
        outdeg[uEdge[i]]++;
        indeg[vEdge[i]]++;
    }

    // Eulerian-path existence test (directed multigraph).
    // Degree condition: every node balanced, OR exactly one node with
    // outdeg-indeg=+1 (start) and exactly one with indeg-outdeg=+1 (end).
    int start = 0;            // default start: any node with an edge
    int plusOne = -1, minusOne = -1;
    bool degreeOk = true;
    for (int v = 0; v < n; v++) {
        int d = outdeg[v] - indeg[v];
        if (d == 1) {
            if (plusOne != -1) { degreeOk = false; }
            plusOne = v;
        } else if (d == -1) {
            if (minusOne != -1) { degreeOk = false; }
            minusOne = v;
        } else if (d != 0) {
            degreeOk = false;
        }
    }
    if ((plusOne == -1) != (minusOne == -1)) degreeOk = false; // must come in a pair
    if (plusOne != -1) start = plusOne;

    if (!degreeOk) { cout << "IMPOSSIBLE\n"; return 0; }

    // Connectivity: every node that touches an edge must lie in a single
    // connected component (consider the underlying undirected graph, since
    // the degree condition already guarantees strong reachability there).
    // We walk forward edges from `start`; a node with an edge that is not
    // reachable means the Euler trail cannot cover all edges.
    vector<char> seen(n, 0);
    // BFS over the underlying undirected graph among edge-bearing nodes.
    vector<vector<int>> und(n);
    for (int v = 0; v < n; v++)
        for (int w : adj[v]) { und[v].push_back(w); und[w].push_back(v); }
    {
        queue<int> q; q.push(start); seen[start] = 1;
        while (!q.empty()) {
            int v = q.front(); q.pop();
            for (int w : und[v]) if (!seen[w]) { seen[w] = 1; q.push(w); }
        }
    }
    for (int v = 0; v < n; v++) {
        if ((indeg[v] + outdeg[v]) > 0 && !seen[v]) { cout << "IMPOSSIBLE\n"; return 0; }
    }

    // Hierholzer's algorithm, iterative, O(E). `ptr[v]` is the next unused
    // outgoing edge of v. We push the Euler trail of nodes onto `circuit`
    // in reverse order, then reverse it.
    vector<int> ptr(n, 0);
    vector<int> stk;          // node stack
    vector<int> circuit;      // resulting node sequence (reversed)
    circuit.reserve(m + 1);
    stk.reserve(m + 1);
    stk.push_back(start);
    while (!stk.empty()) {
        int v = stk.back();
        if (ptr[v] < (int)adj[v].size()) {
            int w = adj[v][ptr[v]++];
            stk.push_back(w);
        } else {
            circuit.push_back(v);
            stk.pop_back();
        }
    }
    reverse(circuit.begin(), circuit.end());

    // A full Euler trail visits exactly m+1 nodes (m edges). If fewer, some
    // edges were unreachable -> impossible (defensive; connectivity should
    // already have caught it).
    if ((int)circuit.size() != m + 1) { cout << "IMPOSSIBLE\n"; return 0; }

    // Reconstruct the string: print the (k-1)-mer of the first node, then one
    // new character per subsequent node (the last char of its (k-1)-mer).
    // We recover each node's label from the stored id map.
    vector<string> label(n);
    for (auto &kv : id) label[kv.second] = kv.first;

    string out;
    out.reserve((k - 1) + m);
    out += label[circuit[0]];
    for (int i = 1; i < (int)circuit.size(); i++) {
        const string &lab = label[circuit[i]];
        out += lab.empty() ? string() : string(1, lab.back());
    }

    cout << out << "\n";
    return 0;
}
