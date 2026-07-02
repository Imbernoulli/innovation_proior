# 题 211 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=5228)

````
The solution needs to find the Minimum Spanning Tree (MST) of a graph where:
- Robots R and S are regular nodes that must be connected
- Relay stations C can be used as intermediate nodes but cannot connect to each other directly
- Different edge weights based on node types

Let me understand the problem better:
1. N robots (R or S type)
2. K optional relay stations (C type)
3. We need to select which relay stations to use
4. We need to choose edges such that all robots are connected
5. Minimize total energy cost

Edge costs:
- R-R: D (squared Euclidean distance)
- R-S: 0.8 × D
- S-S: 0.8 × D
- R-C: 1 × D
- S-C: 1 × D
- C-C: NOT ALLOWED

So the strategy should be:
1. Consider all possible edges between robots (R and S)
2. Consider all possible edges between robots and relay stations
3. Build MST using Kruskal's algorithm or Prim's algorithm
4. However, relay stations can only be used if they help reduce total cost

Key insight: Since we want to minimize cost and can choose which relay stations to use, we should build a graph that includes:
- All robot nodes (R and S)
- All relay station nodes (C)
- Edges with appropriate weights

But with constraint that no two relay stations can be directly connected.

This means in our MST, we might have paths like: R -- C1 -- R, but not C1 -- C2.

Wait, let me re-read: "Relay stations (C) cannot communicate directly with each other."

This means in the final network, there can't be an edge between two relay stations. But a relay station can connect to multiple robots, creating branching paths through it.

So the approach:
1. Create a graph with N + K nodes


2. Add all valid edges with their weights
3. Run MST algorithm, ensuring no C-C edges are selected
4. Extract the relay stations that are actually used in the MST

Since we're building an MST and C-C edges shouldn't exist, I'll just exclude those from consideration during the MST construction phase. Then I can identify which relay stations appear in the final spanning tree and output them along with the edges.

To verify this works correctly, let me trace through the example. With devices positioned at (0,0), (100,0), (50,40), and (50,0) with types R, R, S, and C respectively, I calculate distances and apply the appropriate multipliers for each connection type.

Computing squared distances:
- Distance 1-3: 50² + 40² = 6100, cost = 0.8 × 6100 = 4880 (since 3 is type S)
- Distance 1-4: 50² + 0² = 2500, cost = 1 × 2500 = 2500
- Distance 2-3: 50² + 40² = 6100, cost = 0.8 × 6100 = 4880 (since 3 is type S)
- Distance 2-4: 50² + 0² = 2500, cost = 1 × 2500 = 2500
- Distance 3-4: 50² + 40² = 6100, cost = 1 × 6100 = 6100 (since 4 is type C)

For 1-2 without relay: 100² + 0² = 10000, cost = 10000

Comparing options: 1-4 costs 2500, 2-4 costs 2500, 3-4 costs 6100, while 1-2 alone costs 10000. Using relay 4 connects both 1 and 2 more cheaply than a direct link, and also provides a path to 3. The total would be 2500 + 2500 + 6100 = 11100 versus 10000 + 4880 = 14880 without using the relay.

Wait, but the example output shows 1-3#2-3#3-4 with total cost 4880 + 4880 + 6100 = 15860. That doesn't seem right either. Let me reconsider what's being optimized here—maybe I'm misunderstanding which connections should actually be included.

Actually, I think I've got it wrong. Looking at the structure: 1 and 2 connect through 3 via edges 1-3 and 2-3, then 3 connects to 4 (the relay). So the cost breakdown is 4880 + 4880 + 6100 = 15860. But this is more expensive than using the relay differently...

Unless the constraint is different than I thought. Maybe relay stations don't always provide savings? Let me recalculate if we skip the relay entirely: connecting 1-3 costs 4880, 2-3 costs 4880, giving 9760 total, which is actually cheaper than going through 4.

Yet the output includes 3-4 anyway. Perhaps I'm misreading the requirements, or maybe the sample uses a specific algorithm rather than true optimization.

Looking back at the scoring rules—the base score compares my MST cost against the best possible cost without relays. If I can beat that baseline, I earn full marks. But this seems odd since adding a relay should typically reduce costs through shorter hops.

I'm wondering if there's a constraint preventing certain edges from forming the optimal tree, or if I'm calculating something incorrectly. Let me move forward with implementing the standard MST approach and see if the logic checks out.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <set>
#include <iomanip>

using namespace std;

struct Device {
    int id;
    long long x, y;
    char type;
};

struct Edge {
    int u, v;
    double weight;
    
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

struct DSU {
    vector<int> parent;
    DSU(int n) {
        parent.resize(n);
        iota(parent.begin(), parent.end(), 0);
    }
    
    int find(int i) {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent[i]);
    }
    
    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            parent[root_i] = root_j;
            return true;
        }
        return false;
    }
};

int main() {
    int N, K;
    cin >> N >> K;
    
    vector<Device> devices(N + K);
    for (int i = 0; i < N + K; i++) {
        cin >> devices[i].id >> devices[i].x >> devices[i].y >> devices[i].type;
    }
    
    // Build edge list
    vector<Edge> edges;
    int robot_count = 0;
    
    for (int i = 0; i < N + K; i++) {
        for (int j = i + 1; j < N + K; j++) {
            long long dx = devices[i].x - devices[j].x;
            long long dy = devices[i].y - devices[j].y;
            long long dist_sq = dx * dx + dy * dy;
            
            if (dist_sq > 1e15) continue; // Too far
            
            double weight;
            // Calculate weight based on device types
            if (devices[i].type == 'C' && devices[j].type == 'C') {
                // C-C edges are not allowed
                continue;
            } else {
                if ((devices[i].type == 'S' || devices[j].type == 'S') && 
                    devices[i].type != 'C' && devices[j].type != 'C') {
                    weight = 0.8 * dist_sq;
                } else {
                    weight = 1.0 * dist_sq;
                }
                edges.push_back({i, j, weight});
            }
        }
    }
    
    sort(edges.begin(), edges.end());
    
    DSU dsu(N + K);
    vector<bool> used_edge(N + K, false);
    double min_cost = 0;
    int edge_count = 0;
    
    for (const auto& e : edges) {
        if (dsu.unite(e.u, e.v)) {
            min_cost += e.weight;
            edge_count++;
        }
    }
    
    cout << min_cost << endl;
    
    return 0;
}
```

Wait, I need to reconsider the problem. The requirement states that only robot nodes need to be connected—relay stations are optional infrastructure that can help reduce total energy consumption. So I should build a minimum spanning tree that spans all robots, potentially using relay stations as intermediate nodes to lower costs. This means my current approach of treating all nodes equally might not be correct.

Let me think about how relay stations work: they can connect to any robot (R or S), but never to each other. When I include relay nodes in my graph and compute the MST, some relay nodes might naturally become part of the optimal solution since they provide better connection paths. The key is that I exclude C-C edges entirely, which naturally prevents relay stations from connecting to one another.

Once I run the MST algorithm, I can check which relay stations were actually included in the spanning tree by examining the connectivity of all robot nodes—if any relay node participates in the final structure, it's being used optimally.

But wait, I should verify whether all robots end up connected in my result. If running Kruskal's doesn't connect them all into a single component, that means I need different logic. Actually, since I'm computing a full MST across all nodes, the robots will automatically be connected together through whatever path forms the tree.

Let me implement this more carefully:

```cpp
#include <bits/stdc++.h>

using namespace std;

struct Device {
    int id;
    long long x, y;
    char type;
};

struct Edge {
    int u, v;
    long long weight; // Use long long for exact comparison
    
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

struct DSU {
    vector<int> parent;
    DSU(int n) {
        parent.resize(n);
        for (int i = 0; i < n; i++) parent[i] = i;
    }
    
    int find(int i) {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent[i]);
    }
    
    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            parent[root_i] = root_j;
            return true;
        }
        return false;
    }
};

double calc_distance(const Device& a, const Device& b) {
    long long dx = a.x - b.x;
    long long dy = a.y - b.y;
    return 0.8 * (dx * dx + dy * dy);
}

double get_weight(const Device& a, const Device& b) {
    // Only add edges that respect type constraints
    if ((a.type == 'C' && b.type == 'C')) return -1; // Invalid edge
    return 0.8 * (calc_distance(a, b));
}
```

I need to finish the weight calculation logic and complete the main function structure. .push_back({i, j, weight});
            }
        }
    }
    
    // Sort edges by weight
    sort(edges.begin(), edges.end());
    
    // Kruskal's algorithm
    DSU dsu(N + K);
    vector<Edge> mst_edges;
    
    for (const auto& e : edges) {
        if (dsu.unite(e.u, e.v)) {
            mst_edges.push_back(e);
        }
    }
    
    // Output results
    cout << "#";
    for (const auto& e : mst_edges) {
        if (devices[e.u].id < devices[e.v].id) {
            swap(u_id, v_id);
        }
        cout << u_id << "-" << v_id << "#";
    }
    cout << endl;
    
    return 0;
}
```

Now I'm realizing I need to fix the distance calculation—it should use squared Euclidean distance properly.

```cpp
double get_weight(const Device& a, const Device& b) {
    long long dx = a.x - b.x;
    long long dy = a.y - b.y;
    long long dist_sq = dx * dx + dy * dy;
    
    // Only add edges that respect type constraints
    if (a.type == 'C' && b.type == 'C') return -1; // Invalid edge
    
    double weight;
    // At least one device is not a relay station
    if ((a.type == 'S' || b.type == 'S')) {
        weight = 0.8 * dist_sq;
    } else {
        weight = 1.0 * dist_sq;
    }
    return weight;
}
```

Now I'll build the complete solution. I need to parse N and K, read in all devices, generate all valid edges between non-relay-pair combinations, sort them by weight, and run Kruskal's algorithm to find the minimum spanning tree. Then output the resulting edges.

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

struct Device {
    int id;
    long long x, y;
    char type;
};

struct Edge {
    int u, v;
    long long weight;
    int u_type_idx, v_type_idx;
    
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

struct DSU {
    vector<int> parent;
    DSU(int n) {
        parent.resize(n);
        for (int i = 0; i < n; i++) parent[i] = i;
    }
    
    int find(int i) {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent[i]);
    }
    
    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            parent[root_i] = root_j;
            return true;
        }
        return false;
    }
};

double get_weight(const Device& a, const Device& b) {
    long long dx = a.x - b.x;
    long long dy = a.y - b.y;
    long long dist_sq = dx * dx + dy * dy;
    
    if (a.type == 'C' && b.type == 'C') return -1LL; // Invalid edge
    
    double weight;
    bool has_powerful = (a.type == 'S' || b.type == 'S');
    
    if (has_powerful) {
        weight = static_cast<long long>(0.8 * dist_sq);
    } else {
        weight = dist_sq;
    }
    
    return weight;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int N, K;
    cin >> N >> K;
    
    vector<Device> devices(N + K);
    for (int i = 0; i < N + K; i++) {
        cin >> devices[i].id >> devices[i].x >> devices[i].y >> devices[i].type;
    }
    
    vector<Edge> edges;
    for (int i = 0; i < N + K; i++) {
        for (int j = i + 1; j < N + K; j++) {
            if (get_weight(devices[i], devices[j]) >= 0) {
                edges.push_back({i, j, get_weight(devices[i], devices[j]), 0, 0});
            }
        }
    }
    
    sort(edges.begin(), edges.end());
    
    DSU dsu(N + K);
    vector<pair<int, int>> mst_edges;
    
    for (const auto& e : edges) {
        if (dsu.unite(e.u, e.v)) {
            mst_edges.push_back({e.u, e.v});
        }
    }
    
    return 0;
}
```

Actually, I need to reconsider how relay stations work—they can serve as intermediaries without needing direct C-C connections. Let me recalculate the weights properly and ensure I'm handling the relay station constraints correctly.

The weight calculation depends on device types: high-power devices (S) reduce energy by half, ordinary devices (R) don't, and relay stations (C) have special properties. Two relay stations can't connect directly, but a relay can bridge between any other devices. So when checking if an edge is valid, I need to ensure at least one endpoint isn't a relay station.

For the final answer, I need to track which relay stations participate in the minimum spanning tree. After computing the MST, I'll identify all relay nodes that appear in the selected edges and output them separated by `#`, followed by all edges formatted as `id1-id2` strings.
</think>

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>

using namespace std;

struct Device {
    int id;
    long long x, y;
    char type;
};

struct Edge {
    int u, v;
    long long weight;
    
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

struct DSU {
    vector<int> parent;
    DSU(int n) {
        parent.resize(n);
        for (int i = 0; i < n; i++) parent[i] = i;
    }
    
    int find(int i) {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent[i]);
    }
    
    bool unite(int i, int j) {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j) {
            parent[root_i] = root_j;
            return true;
        }
        return false;
    }
};

long long dist_sq(const Device& a, const Device& b) {
    long long dx = a.x - b.x;
    long long dy = a.y - b.y;
    return dx * dx + dy * dy;
}

long long get_weight(const Device& a, const Device& b) {
    // C-C edges are not allowed
    if (a.type == 'C' && b.type == 'C') {
        return -1; // Invalid
    }
    
    long long d = dist_sq(a, b);
    // At least one device is not C
    if (a.type == 'S' || b.type == 'S') {
        return d * 4 / 5; // 0.8 = 4/5
    }
    return d; // 1.0
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int N, K;
    cin >> N >> K;
    
    vector<Device> devices;
    int n_robots = 0;
    
    for (int i = 0; i < N + K; i++) {
        Device dev;
        cin >> dev.id >> dev.x >> dev.y >> dev.type;
        devices.push_back(dev);
        if (dev.type != 'C') {
            n_robots++;
        }
    }
    
    // Generate all valid edges (excluding C-C)
    vector<Edge> edges;
    for (int i = 0; i < N + K; i++) {
        for (int j = i + 1; j < N + K; j++) {
            long long w = get_weight(devices[i], devices[j]);
            if (w >= 0) {
                edges.push_back({i, j, w});
            }
        }
    }
    
    // Sort edges by weight for MST
    sort(edges.begin(), edges.end());
    
    // Find MST using Kruskal's algorithm
    DSU dsu(N + K);
    vector<int> mst_relays;
    set<string> mst_edges_set;
    
    int edges_count = 0;
    for (const auto& e : edges) {
        if (dsu.unite(e.u, e.v)) {
            // Check if this edge involves any relay stations
            bool u_is_relay = devices[e.u].type == 'C';
            bool v_is_relay = devices[e.v].type == 'C';
            
            if (u_is_relay) {
                mst_relays.push_back(e.u);
            }
            if (v_is_relay) {
                mst_relays.push_back(e.v);
            }
            
            string edge_str = to_string(min(devices[e.u].id, devices[e.v].id)) + 
                              "-" + to_string(max(devices[e.u].id, devices[e.v].id));
            mst_edges_set.insert(edge_str);
            edges_count++;
        }
        
        // Check if all robots are connected (N robots + potential relays)
        // We need to track robot components
    }
    
    // Count unique relay IDs
    set<int> unique_relays(mst_relays.begin(), mst_relays.end());
    
    // Output selected relays
    cout << "#";
    bool first = true;
    for (int r : unique_relays) {
        if (!first) cout << "#";
        cout << devices[r].id;
        first = false;
    }
    cout << endl;
    
    // Output edges
    vector<string> sorted_edges(mst_edges_set.begin(), mst_edges_set.end());
    sort(sorted_edges.begin(), sorted_edges.end());
    
    cout << "#";
    for (size_t i = 0; i < sorted_edges.size(); i++) {
        if (i > 0) cout << "#";
        cout << sorted_edges[i];
    }
    cout << endl;
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=2011)

````
I'm a competitive programmer. I will give you a problem statement, please implement a solution in C++. Please wrap the code in ```cpp and ``` so that it is properly formatted. The output content should ONLY contain the C++ code, with no additional explanation or text.


Let me analyze this problem:

1. We have N robots and K optional relay stations
2. Each robot has coordinates and type (R = ordinary robot, S = high-power robot, C = relay station)
3. Communication energy consumption depends on distance squared multiplied by coefficients
4. We want to connect all robots while minimizing total energy cost
5. Relay stations can be optionally used
6. Relay stations cannot communicate directly with each other
7. Robots must be connected (directly or through relay stations)

This is a minimum spanning tree problem with special edge weights. Let me think about the approach:

**Edge weight calculation:**
- Robot R - Robot R: D × 1.0
- Robot R - Robot S: D × 0.8  
- Robot S - Robot S: D × 0.8
- Robot (R/S) - Relay C: D × 1.0
- Relay C - Relay C: NOT ALLOWED

**Approach:**
Since relay stations are optional, we need to try all possible subsets of relay stations to use. For each subset:
1. Build a graph with the selected relay stations included
2. Find the MST that connects all robots (possibly using relay stations as intermediate nodes)
3. Calculate the cost
4. Keep the minimum cost configuration

For each subset of relay stations, we build a graph where:
- Every robot can connect to every other robot (with appropriate weights based on their types)
- Every robot can connect to every relay station (with weight D × 1.0)
- No relay station connects directly to another relay station

Then run Kruskal's algorithm to find the MST connecting all robots (relay stations help but don't need to connect among themselves).

The challenge is that 2^K subsets is exponential, and K can be up to 1500! So we can't iterate all subsets. However, observe the key insight: for any robot pair, using one relay station doesn't help more than using an optimal single relay. And since relays connect robots, not other relays, we only need at most N-1 relays to connect all robots optimally, or fewer if there are redundant relays between the same two robots.

Actually, let me reconsider. Given the constraints (N, K ≤ 1500), maybe the intended approach is different. Perhaps there's a greedy strategy or the number of useful relay configurations is actually small in practice. Or maybe we select the best k' relays where k' = min(K, N-1) and enumerate those, then add edges greedily for remaining relays.

Let me code a reasonable solution with careful analysis of which relay positions matter most.
</think>

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
using namespace std;

struct Device {
    int id;
    int x, y;
    char type;
};

int n_robot, n_relays;
vector<Device> devices;
bool* relay_active;

int find_set(int i, vector<int>& parent) {
    if (parent[i] == i) return i;
    return parent[i] = find_set(parent[i], parent);
}

bool union_sets(int a, int b, vector<int>& parent, vector<long long>& rank) {
    a = find_set(a, parent);
    b = find_set(b, parent);
    if (a != b) {
        if (rank[a] < rank[b]) swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) rank[a]++;
        return true;
    }
    return false;
}

long long calculate_cost(const Device& a, const Device& b, const char type_a, const char type_b) {
    long long dist_sq = (long long)(a.x - b.x) * (a.x - b.x) + (long long)(a.y - b.y) * (a.y - b.y);
    double coef = 0.0;
    // R-R, R-S, S-S connections have coefficient 0.8
    // Relay-station connections have coefficient 1.0
    if (type_a == 'C' || type_b == 'C') coef = 1.0;
    else coef = 0.8;
    return round(dist_sq * coef);
}

// Minimum Spanning Tree using Prim's algorithm
// Nodes 0..n_robot-1: robots, n_robot+1..n_robot+num_active_relays: active relay stations
double prim_mst(int num_active_relays, const vector<Device>& devices, vector<bool>& visited, vector<double>& min_edge) {
    fill(min_edge.begin(), min_edge.end(), 1e9);
    
    vector<int> parent(n_robot + num_active_relays + 1, -1);
    visited[0] = true;
    min_edge[0] = 0;
    
    long long total_cost = 0;
    
    // Count how many robots we've successfully connected
    int robots_connected = 0;
    
    for (int i = 0; i < n_robot + num_active_relays + 1; i++) {
        int u = -1, best_idx = -1;
        
        // Select the unvisited node with minimum edge weight
        for (int j = 0; j < n_robot + num_active_relays + 1; j++) {
            if (!visited[j] && min_edge[j] < best_idx) {
                best_idx = j;
            }
        }
        
        if (best_idx == -1 || min_edge[best_idx] >= min_edge[0]) break;
        
        visited[best_idx] = true;
        
        if (best_idx < n_robot) robots_connected++;
        
        total_cost += min_edge[best_idx];
        
        if (robots_connected == n_robot) break;
        
        for (int v = 0; v < n_robot; v++) {
            if (!visited[v]) {
                long long cost = calculate_cost(devices[best_idx], devices[v], devices[best_idx].type, devices[v].type);
                if (cost < min_edge[v]) {
                    min_edge[v] = (double)cost;
                    parent[v] = best_idx;
                }
            }
        }
        
        // Connect to active relay stations
        for (int r = 0; r < num_active_relays; r++) {
            int v = n_robot + 1 + r;
            if (!visited[v]) {
                const Device* dev_ptr = &devices[n_robot];
                const Device& rel_device = devices[*relay_active[r]];
                long long cost = calculate_cost(*dev_ptr, rel_device, (*dev_ptr).type, rel_device.type);
                if (cost < min_edge[v]) {
                    min_edge[v] = (double)cost;
                    parent[v] = best_idx;
                }
            }
        }
    }
    
    return total_cost;
}

int main() {
    cin >> n_robot >> n_relays;
    devices.resize(n_robot + n_relays);
    for (int i = 0; i < n_robot + n_relays; i++) {
        cin >> devices[i].id >> devices[i].x >> devices[i].y >> devices[i].type;
    }
    
    relay_active.resize(n_relays);
    fill(relay_active.begin(), relay_active.end(), false);
    
    double best_cost = 1e18;
    int best_relay_count = -1;
    
    // Try including each individual relay station (at most n_robot-1 needed)
    for (int k = 0; k <= n_robot - 1; k++) {
        if (k > n_relays) continue;
        for (int mask = 0; mask < (1 << k); mask++) {
            int count = 0;
            for (int i = 0; i < k; i++) if (mask & (1 << i)) count++;
            if (count == 0) continue;
            
            // Add relays from the first n_relays devices
            for (int i = 0; i < k; i++) {
                relay_active[i] = (mask & (1 << i));
            }
            
            double cost = prim_mst(k, devices, make_shared<vector<bool>>(n_robot + k + 1, false), make_shared<vector<double>>(n_robot + k + 1, 0));
            if (cost < best_cost) {
                best_cost = cost;
                best_relay_count = k;
            }
        }
    }
    
    cout << fixed << setprecision(2);
    cout << best_relay_count;
    return 0;
}
```
````
