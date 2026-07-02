# 题 253 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 0.0,completion_tokens=1911)

````
The problem is an interactive problem where I need to determine which roads are repaired. We have n intersections and m roads. Some subset of roads are repaired such that the city is connected using only repaired roads.

Let me understand the interaction:
1. Block a road (x): Mark road x as blocked if not already blocked
2. Unblock a road (x): Mark road x as unblocked if it was blocked
3. Delivery query (? k y1 y2 ... yk): Randomly picks Y from {y1, ..., yk}, checks if there's a path from s to Y using repaired AND unblocked roads, returns 1 or 0

Key constraints:
- Total cost limit: 50000 for full score
- Cost of ?: 0.5 + log2(k+1)
- Cost of block/unblock: 2 each
- Maximum requests: 100*m per test case

Since we know the graph is connected using repaired roads, if I can find a spanning tree of repaired roads, I can identify all repaired roads.

The challenge is that we don't know which roads are repaired initially, but we know they form a connected subgraph.

Strategy:
1. Try to use queries to check reachability
2. Since k=1 has cost 0.5+log2(2) = 1.5, this is reasonable
3. With k nodes in the query, cost is 0.5 + log2(k+1)

Wait, I need to think more carefully. The key insight is:
- If I know two nodes are in the same component (using repaired roads), I can query between them
- But I don't know which roads are repaired

One approach: For each road i, try to block it and see if connectivity breaks. But checking connectivity with queries costs money.

Actually, let me think about this differently. 

I'm exploring ways to efficiently determine repaired roads while minimizing query costs. The key challenge is finding a method that allows precise identification without excessive resource expenditure.

Blocking roads seems promising because it could help me systematically test road connections. By strategically blocking and querying, I might be able to build up information about which roads remain intact while keeping total query costs low.

My current strategy focuses on using targeted delivery queries to map out the underlying road network, potentially revealing the repaired infrastructure through clever probing of the graph structure.

The critical constraint is managing query costs effectively, especially when trying to identify specific road states across potentially complex interconnected networks.

I need to design an approach that minimizes redundant queries while ensuring complete network understanding within the given budget constraints.

The key strategy involves selectively probing road connectivity by blocking potential routes and testing path existence between specific node pairs. This allows systematic identification of which roads remain functional after selective obstruction.

Critical steps involve tracking which roads connect components, determining repair status through strategic query placement, and maintaining cost efficiency throughout the investigation process. I realize I'm dealing with a fundamental challenge in identifying which roads remain connected. The core issue involves systematically probing road connections while minimizing query complexity. My current approach needs refinement to handle different graph structures efficiently.

The problem requires developing a strategy that works across varying road configurations without making assumptions about initial connectivity. I'll need to create a more adaptive method that can dynamically explore the road network's structure through careful, targeted queries.

A binary search-like technique might help reduce query overhead, potentially allowing more efficient exploration of potential road repairs. The goal is to determine repair status with minimal interaction rounds while respecting strict cost constraints.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Solution {
    int n, m;
    vector<pair<int,int>> roads;
    vector<bool> repaired(m+1, false);
    
    void reset() {
        repaired.assign(m+1, false);
    }
    
    bool isConnected(int s, int t) {
        // Binary search to find if there's a path from s to t
        queue<int> q;
        vector<bool> vis(n+1, false);
        
        q.push(s);
        vis[s] = true;
        
        while(!q.empty()){
            int u = q.front();
            q.pop();
            
            if(u == t) return true;
            
            for(auto& [_, v] : adj[u]){
                if(vis[v]) continue;
                vis[v] = true;
                q.push(v);
            }
        }
        
        return false;
    }
    
    void solve() {
        cin >> n >> m;
        roads.clear();
        roads.resize(m+1);
        adj.assign(n+1, vector<vector<int>>(m+1));
        
        for(int i = 1; i <= m; ++i){
            cin >> roads[i].first >> roads[i].second;
            adj[roads[i].first].push_back({i, roads[i].second});
            adj[roads[i].second].push_back({i, roads[i].first});
        }
        
        vector<bool> ans(m+1, true);
        
        cout << "? 2 " << 1 << " " << n << endl;
        cout.flush();
        int response;
        cin >> response;
        
        if(response == 1) {
            // All roads might be needed, but we still need to identify
            for(int i = 1; i <= m; ++i) {
                // Check if road i is necessary
                vector<bool> removed(i == 1 ? 2 : -1);
                
                // Try removing road i and check connectivity
                cout << "- " << i << endl;
                cout.flush();
                
                cout << "? 2 " << 1 << " " << n << endl;
                cout.flush();
                cin >> response;
                
                if(response == 0 || (response == 0 && i == 1)) {
                    ans[i] = true;
                } else {
                    ans[i] = true;
                }
                
                cout << "+ " << i << endl;
                cout.flush();
            }
        }
        
        // Final answer
        cout << "!";
        for(int i = 1; i <= m; ++i) {
            cout << (ans[i] ? " 1" : " 0");
        }
        cout << endl;
        cout.flush();
    }
};

vector<vector<int>> adj;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int t;
    cin >> t;
    
    while(t--) {
        cin >> n >> m;
        
        roads.resize(m+1);
        for(int i = 1; i <= m; ++i) {
            cin >> roads[i].first >> roads[i].second;
        }
        
        // Initialize adjacency for BFS
        adj.assign(n+1, vector<vector<int>>(m+1));
        for(int i = 1; i <= m; ++i) {
            adj[roads[i].first].push_back({i, roads[i].second});
            adj[roads[i].second].push_back({i, roads[i].first});
        }
        
        vector<bool> candidate(m+1, true);
        
        // First query: check basic connectivity
        cout << "? 2 1 " << n << endl;
        cout.flush();
        int r1;
        cin >> r1;
        if(r1 != 1) {
            cout << "!";
            for(int i = 1; i <= m; ++i) cout << " 0";
            cout << endl;
            cout.flush();
            return;
        }
        
        // Strategy: For each edge, temporarily remove it and query
        for(int i = 1; i <= m; ++i) {
            // Remove road i
            cout << "- " << i << endl;
            cout.flush();
            
            cout << "? 2 1 " << n << endl;
            cout.flush();
            cin >> r1;
            
            if(r1 == 0) {
                // Connectivity broken - road i must be repaired
                candidate[i] = true;
            } else {
                candidate[i] = false;
            }
            
            // Restore road i
            cout << "+ " << i << endl;
            cout.flush();
        }
        
        // Output result
        cout << "!";
        for(int i = 1; i <= m; ++i) {
            cout << " " << (candidate[i] ? "1" : "0");
        }
        cout << endl;
        cout.flush();
        
        int verdict;
        cin >> verdict;
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=1405)

````
I see a potential issue with the query logic. The query returns 1 if there's a path from s to Y through REPAIRED AND unblocked roads, but our block/unblock operations only affect unblocking, not blocking repaired roads. Let me fix this:

The problem is that when we try to "block" road x using `- x`, this only blocks it for future queries. But in the original approach, we need to be able to verify whether each road is actually repaired or just an alternative connection.

Let me rethink this more carefully...

Actually, I realize there's a fundamental issue with my approach - I'm trying to use block/unblock to test individual roads, but the block command only prevents using that road in paths, while what I really need is to distinguish between:
- A road that is NOT repaired (never contributes to any path)
- A road that IS repaired but redundant (contributed as an alternative)

The key insight is: **a road i is NOT repaired if and only if no query ever returns 1 due to road i being used.**

So I can track which roads have contributed to a positive answer at least once. If a road never helps connect anything, it must not be repaired.

Let me implement this correctly:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t;
    cin >> t;
    while (t--) {
        int n, m;
        cin >> n >> m;
        vector<pair<int, int>> roads(m + 1);
        for (int i = 1; i <= m; i++) {
            cin >> roads[i].first >> roads[i].second;
        }
        
        bool repaired[2005]; // tracks if road i has contributed to ANY connection
        fill(repaired, repaired + m + 1, false);
        
        set<int> blocked_roads;     // currently blocked roads (can only unblock, not permanently keep blocked)
        
        auto add_road = [&](int u, int v) -> int {
            int best_v = 0, max_dist = 0;
            
            for (auto& [u_r, v_r] : roads) {
                if (u_r == u && v_r != v) continue;
                if (v_r == v && u_r != u) continue;
                
                const char* q_res = "?";
                cout << "?" << " " << n;
                cout << " ";
                for (int i = 1; i <= n; i++) cout << i << " ";
                cout << endl;
                int res; cin >> res;
                if (res < 0 || res > 1) { exit(0); }
                
                if (res == 0) return 0;
                
                int d = __builtin_popcount(u ^ v);
                if (d >= max_dist) {
                    max_dist = d;
                    best_v = v_r;
                }
            }
            return best_v;
        };
        
        for (int i = 1; i <= m; i++) {
            int added_v = add_road(roads[i].first, roads[i].second);
            if (added_v != roads[i].second) {
                repaired[i] = true;
            } else {
                repaired[i] = !repaired[i];
            }
        }
        
        cout << "!";
        for (int i = 1; i <= m; i++) {
            cout << (repaired[i] ? 1 : 0) << " ";
        }
        cout << endl;
    }
    return 0;
}
```

I'll now verify this solution works correctly by testing it against different input scenarios and checking its performance characteristics.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t;
    cin >> t;
    while (t--) {
        int n, m;
        cin >> n >> m;
        vector<pair<int,int>> roads(m + 1);
        for (int i = 1; i <= m; i++) cin >> roads[i].first >> roads[i].second;
        
        bool repaired[2005];
        fill(repaired, repaired + m + 1, false);
        set<int> blocked_roads;
        
        auto add_road = [&](int u, int v) -> int {
            int best_v = 0, max_dist = 0;
            for (auto& r : roads) {
                if ((r.first == u && r.second == v) || 
                    (r.first == v && r.second == u)) continue;
                if (blocked_roads.count(r.first) && blocked_roads.count(r.second)) continue;
                if (!blocked_roads.empty()) {
                    const string cmd("?");
                    cout << " " << cmd << " " << blocked_roads.size() << "\n";
                    cout << " \n";
                }
                const string cmd("?");
                cout << " " << cmd << " " << n << "\n";
                for (int i = 1; i <= n; i++) cout << i << " ";
                cout << endl;
                int res; cin >> res;
                if (res < 0 || res > 1) exit(0);
                if (res == 0) return 0;
                int d = __builtin_popcount(u ^ v);
                if (d >= max_dist) max_dist = d, best_v = v;
            }
            return best_v;
        };
        
        for (int i = 1; i <= m; i++) {
            int added_v = add_road(roads[i].first, roads[i].second);
            if (added_v != roads[i].second) repaired[i] = true;
            else repaired[i] = !repaired[i];
        }
        
        cout << "!";
        for (int i = 1; i <= m; i++) cout << (repaired[i] ? 1 : 0) << " ";
        cout << endl;
    }
    return 0;
}
```
````
