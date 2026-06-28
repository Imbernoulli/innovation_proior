#include <bits/stdc++.h>
using namespace std;

/*
  Maximum common independent set of two matroids on the SAME ground set of m
  edges:
    M1 = graphic matroid  : a subset is independent iff it is a forest
    M2 = partition matroid: at most cap[color] edges of each color

  We want the largest subset of edges that is simultaneously a forest and
  respects every per-color capacity. This is exactly maximum-cardinality
  matroid intersection, solved by the augmenting-path / exchange-graph
  algorithm: repeatedly find a shortest augmenting path from a free source
  set to a free sink set in the directed exchange graph, then flip the
  membership of every element on that path, growing the common independent
  set by one. With no augmenting path, the current set is optimal.
*/

int n, m, K;
vector<int> eu, ev, ec;     // endpoints and color of each edge
vector<int> cap_;           // capacity per color (1-indexed)

// ---- graphic-matroid (forest) helpers over a given edge subset S ----
// inForest1(S): is S a forest? (no cycle)
// Used as a building block; the real per-edge exchange tests below are
// computed directly via cycle/path analysis on S.

// Build adjacency of the forest induced by the current set S (edges with in[e]).
// For M1 exchange edges we need, for x in S and y not in S:
//   y -> x  (an M1-arc) iff x lies on the unique cycle that y would close
//             when added to S, i.e. x is on the tree path between y's endpoints.
// And free M1-sink: y not in S is M1-free iff S+y is independent
//             (y connects two different components / is a self loop? no).
//
// For M2 (partition): each color is its own "component". For x in S, y not in S:
//   x -> y  (an M2-arc) iff removing x frees capacity that lets y in, i.e.
//             x and y share a color and that color is at capacity.
//   y not in S is M2-free (source) iff its color is below capacity.

int main() {
    if (!(scanf("%d %d %d", &n, &m, &K) == 3)) return 0;
    eu.resize(m); ev.resize(m); ec.resize(m);
    cap_.assign(K + 1, 0);
    for (int i = 1; i <= K; i++) scanf("%d", &cap_[i]);
    for (int i = 0; i < m; i++) {
        scanf("%d %d %d", &eu[i], &ev[i], &ec[i]);
    }

    vector<char> inS(m, 0);          // membership in current common indep set
    vector<int> colorCount(K + 1, 0);

    // DSU over vertices to test forest membership quickly when needed.
    // But for exchange arcs we need tree paths, so we rebuild adjacency of S.

    auto buildForestAdj = [&](vector<vector<pair<int,int>>> &adj) {
        // adj[v] = list of (neighbor, edgeIndex) for edges currently in S
        adj.assign(n + 1, {});
        for (int e = 0; e < m; e++) if (inS[e]) {
            adj[eu[e]].push_back({ev[e], e});
            adj[ev[e]].push_back({eu[e], e});
        }
    };

    // For a candidate edge y not in S, find the set of edges of S on the tree
    // path between y's endpoints (these are the x with arc y->x). If endpoints
    // are in different components, the path is empty and y is M1-free.
    // We answer this via a BFS/DFS from one endpoint.
    auto treePathEdges = [&](vector<vector<pair<int,int>>> &adj, int s, int t,
                             vector<int> &outEdges) -> bool {
        // returns true if s and t connected; fills outEdges with edge indices on path
        outEdges.clear();
        if (s == t) return true; // self-loop edge: path empty, but it forms a loop
        vector<int> par(n + 1, -1), parE(n + 1, -1);
        vector<char> vis(n + 1, 0);
        queue<int> q; q.push(s); vis[s] = 1;
        bool found = false;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (u == t) { found = true; break; }
            for (auto &pr : adj[u]) {
                int w = pr.first, e = pr.second;
                if (!vis[w]) { vis[w] = 1; par[w] = u; parE[w] = e; q.push(w); }
            }
        }
        if (!found) return false;
        int cur = t;
        while (cur != s) {
            outEdges.push_back(parE[cur]);
            cur = par[cur];
        }
        return true;
    };

    // One augmentation step: build exchange graph, BFS shortest path from any
    // M2-free source (over not-in-S elements whose color has spare capacity)
    // to any M1-free sink (not-in-S elements that keep the forest a forest),
    // flip the path. Returns true if augmented.
    auto augment = [&]() -> bool {
        vector<vector<pair<int,int>>> adj;
        buildForestAdj(adj);

        // Precompute, for each y NOT in S, whether y is M1-free and the path edges.
        // Also classify sources (M2-free) for elements NOT in S.
        // Exchange graph nodes = all edges [0..m). We do a multi-source BFS.

        // arcs:
        //   for y not in S:
        //     M1: y is sink if S+y independent (forest). Else for each x in S on
        //         the cycle that y closes: arc y -> x.
        //     M2: y is source if colorCount[color]<cap. Else for each x in S with
        //         same color: arc x -> y.
        // BFS from all sources following arcs, looking for a sink. Shortest path
        // in edge count -> a valid shortest augmenting path (no shortcut needed
        // because BFS already yields shortest; the classic correctness uses the
        // shortest augmenting path).

        const int INF = INT_MAX;
        vector<int> dist(m, INF), prevNode(m, -1);
        // Precompute color->list of in-S edges (for M2 arcs x->y) lazily.
        // We'll just iterate when needed.

        // Determine sources and sinks, and an adjacency we can traverse.
        // To avoid O(m^2) blowups in path reconstruction we compute arcs on the
        // fly during BFS expansion.

        // For M1 arcs we need tree paths; cache path edges per y when first needed.
        // For M2 arcs x->y: y not in S, x in S, same color, color full.

        // group not-in-S edges by color for M2 forward arcs
        vector<vector<int>> notInByColor(K + 1);
        vector<vector<int>> inByColor(K + 1);
        for (int e = 0; e < m; e++) {
            if (inS[e]) inByColor[ec[e]].push_back(e);
            else notInByColor[ec[e]].push_back(e);
        }

        // sinks: y not in S, M1-free (S+y independent forest)
        // sources: y not in S, M2-free (colorCount[color] < cap)
        auto isM1Free = [&](int y) -> bool {
            // S + y is a forest iff endpoints in different components and not a
            // self-loop closing a cycle. A self-loop (eu==ev) always forms a loop.
            if (eu[y] == ev[y]) return false;
            vector<int> tmp;
            bool connected = treePathEdges(adj, eu[y], ev[y], tmp);
            return !connected; // not connected => adding y keeps forest
        };
        auto isM2Free = [&](int y) -> bool {
            return colorCount[ec[y]] < cap_[ec[y]];
        };

        queue<int> bfs;
        for (int e = 0; e < m; e++) if (!inS[e]) {
            if (isM2Free(e)) { dist[e] = 0; prevNode[e] = -1; bfs.push(e); }
        }

        int sinkFound = -1;
        // If a source is itself a sink, that's a length-0 augmenting "path".
        for (int e = 0; e < m; e++) if (!inS[e] && dist[e] == 0) {
            if (isM1Free(e)) { sinkFound = e; break; }
        }

        if (sinkFound == -1) {
            while (!bfs.empty() && sinkFound == -1) {
                int u = bfs.front(); bfs.pop();
                if (!inS[u]) {
                    // u is a not-in-S element. Outgoing arcs:
                    //  M1 arcs u -> x (x in S on the cycle u closes)
                    vector<int> pathE;
                    bool connected = treePathEdges(adj, eu[u], ev[u], pathE);
                    if (eu[u] == ev[u]) {
                        // self-loop: closes a cycle with no S edge => no M1 arcs,
                        // and it's never M1-free; it's a dead element. skip.
                    } else if (connected) {
                        for (int x : pathE) {
                            if (dist[x] == INF) {
                                dist[x] = dist[u] + 1;
                                prevNode[x] = u;
                                bfs.push(x);
                            }
                        }
                    }
                } else {
                    // u is an in-S element x. Outgoing arcs:
                    //  M2 arcs x -> y (y not in S, same color, color full)
                    int col = ec[u];
                    if (colorCount[col] >= cap_[col]) {
                        for (int y : notInByColor[col]) {
                            if (dist[y] == INF) {
                                dist[y] = dist[u] + 1;
                                prevNode[y] = u;
                                bfs.push(y);
                                if (isM1Free(y)) { sinkFound = y; }
                            }
                        }
                    }
                }
                if (sinkFound != -1) break;
            }
        }

        if (sinkFound == -1) return false;

        // flip path: walk prevNode from sinkFound back to a source.
        int cur = sinkFound;
        while (cur != -1) {
            inS[cur] = inS[cur] ? 0 : 1;
            if (inS[cur]) colorCount[ec[cur]]++;
            else colorCount[ec[cur]]--;
            cur = prevNode[cur];
        }
        return true;
    };

    int size = 0;
    while (augment()) size++;

    printf("%d\n", size);
    return 0;
}
