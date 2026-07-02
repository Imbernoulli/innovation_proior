# 题 177 — base均分 64.9 vs 直接SFT均分 0.0(掉点 64.9)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=4,得分 95.8,completion_tokens=2200)

````
The user wants me to solve a graph 3-coloring problem where the goal is to minimize conflicting edges. A conflicting edge is one where both endpoints have the same color. I need to output a valid coloring for all vertices using colors 1, 2, or 3.

Let me think about this problem:
1. This is essentially trying to get as close to a proper 3-coloring as possible
2. A proper 3-coloring would have b = 0 (no conflicting edges) and score = 1
3. However, some graphs might not be 3-colorable in general
4. But we're asked to find ANY coloring that minimizes conflicts - this is always possible with any colors since we can always assign some color combination
5. The greedy approach should work well here

Key observations:
- Since m ≤ 200,000 and n ≤ 60,000, we need an efficient algorithm
- A simple greedy approach where we try to avoid assigning the same color as neighbors should work
- For each vertex, we should pick a color different from as many neighbors as possible

Strategy:
1. Process vertices in some order (possibly by degree)
2. For each vertex, choose a color that doesn't conflict with already-colored neighbors
3. If multiple such colors exist, pick one arbitrarily
4. If no such color exists (all three colors are used by neighbors), pick the least restrictive one

Actually, a simpler approach that works well:
1. Assign colors greedily
2. For each vertex, count how many times each color appears among its colored neighbors
3. Pick the color with minimum count among neighbors

Wait, but we need to consider ALL edges. Let me think again...

Actually, a very effective strategy for this problem:
1. Use a randomized approach or priority-based assignment
2. Sort vertices by degree (highest first) to handle high-degree nodes early
3. For each vertex, try to pick a color that minimizes conflicts


4. This greedy heuristic typically produces good results on most graphs

I'll implement a solution that prioritizes high-degree vertices and assigns colors to minimize immediate conflicts.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

const int MAXN = 60005;
vector<int> adj[MAXN];
int color[MAXN];

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n, m;
    cin >> n >> m;
    
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    vector<int> order(n + 1);
    for (int i = 1; i <= n; i++) order[i] = i;
    
    // Sort by degree (descending) - higher degree vertices first
    sort(order.begin(), order.end(), [&](int a, int b) {
        return adj[a].size() > adj[b].size();
    });
    
    // Greedy coloring
    for (int idx : order) {
        if (color[idx] != 0) continue; // Already colored
        
        int cnt[4] = {0, 0, 0, 0};
        for (int neighbor : adj[idx]) {
            if (color[neighbor] != 0) {
                cnt[color[neighbor]]++;
            }
        }
        
        // Pick color with minimum usage among neighbors
        int bestColor = 1;
        for (int c = 1; c <= 3; c++) {
            if (cnt[c] < cnt[bestColor]) {
                bestColor = c;
            }
        }
        
        color[idx] = bestColor;
    }
    
    for (int i = 1; i <= n; i++) {
        cout << color[i] << (i == n ? "" : " ");
    }
    cout << "\n";
    
    return 0;
}
```

This greedy approach processes highest-degree vertices first, tracking neighbor color counts to select the optimal color for each uncolored node. , so we'd want to break ties deterministically.

Actually, the real challenge here is ensuring the coloring works well even for complex graphs. With only three colors available, conflicts might still occur in dense regions, so the greedy strategy of picking the least-used color locally helps minimize total conflicts across the entire graph.

A simpler greedy implementation that sorts vertices by degree then assigns each one the color appearing least frequently among already-colored neighbors should handle most cases efficiently.

For better handling of dense subgraphs where all three colors appear in neighbors, randomization could help escape local optima. I'll use a deterministic approach first—sorting vertices by degree descending and greedily selecting the least-used color among neighbors—which provides solid baseline performance before considering more complex strategies like simulated annealing or constraint propagation.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
using namespace std;

const int MAXN = 60005;
vector<int> adj[MAXN];
int color[MAXN];

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n, m;
    if (!(cin >> n >> m)) return 0;
    
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    // Greedy coloring with vertex ordering by degree
    for (int i = 1; i <= n; i++) {
        color[i] = 0; // 0 means uncolored
    }
    
    // Create index array and sort by degree
    vector<int> p(n);
    iota(p.begin(), p.end(), 1);
    sort(p.begin(), p.end(), [&](int a, int b) {
        return adj[a].size() > adj[b].size();
    });
    
    // Color each vertex greedily
    for (int i = 0; i < n; i++) {
        int idx = p[i];
        if (color[idx] != 0) continue;
        
        int cnt[4] = {0, 0, 0, 0};
        for (int neighbor : adj[idx]) {
            if (color[neighbor] != 0) {
                cnt[color[neighbor]]++;
            }
        }
        
        // Try colors 1, 2, 3 and pick the one that causes minimum conflicts
        // First try colors that haven't been used by neighbors
        for (int c = 1; c <= 3; c++) {
            if (cnt[c] == 0) {
                color[idx] = c;
                break;
            }
        }
        
        // If all colors are used, pick the one with minimum count
        if (color[idx] == 0) {
            int minCount = cnt[1];
            int bestC = 1;
            for (int c = 2; c <= 3; c++) {
                if (cnt[c] < minCount) {
                    minCount = cnt[c];
                    bestC = c;
                }
            }
            color[idx] = bestC;
        }
    }
    
    // Output the coloring
    for (int i = 1; i <= n; i++) {
        cout << color[i] << (i == n ? "" : " ");
    }
    cout << "\n";
    
    return 0;
}
```

Now I need to verify the output format requirements - each color should be printed separated by spaces, with a newline at the end.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

const int MAXN = 60005;
vector<int> adj[MAXN];
int color[MAXN];

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n, m;
    if (!(cin >> n >> m)) return 0;
    
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    for (int i = 1; i <= n; i++) {
        color[i] = 0;
    }
    
    vector<int> p(n);
    iota(p.begin(), p.end(), 1);
    sort(p.begin(), p.end(), [&](int a, int b) {
        return adj[a].size() > adj[b].size();
    });
    
    for (int i = 0; i < n; i++) {
        int idx = p[i];
        if (color[idx] != 0) continue;
        
        int cnt[4] = {0, 0, 0, 0};
        for (int neighbor : adj[idx]) {
            if (color[neighbor] != 0) {
                cnt[color[neighbor]]++;
            }
        }
        
        int bestC = 1;
        for (int c = 1; c <= 3; c++) {
            if (cnt[c] < cnt[bestC]) {
                bestC = c;
            }
        }
        
        color[idx] = bestC;
    }
    
    for (int i = 1; i <= n; i++) {
        cout << color[i] << (i == n ? "" : " ");
    }
    cout << "\n";
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=3,得分 0.0,completion_tokens=1891)

````
I need to minimize the number of conflicting edges in a graph 3-coloring problem. An edge is conflicting if its two endpoints have the same color.

Let me think about the structure:
- For an independent set (no edges between them), any coloring has score = 1
- For a complete graph, we can use 2 colors to get score = 0 (since all vertices get different colors except one pair)
- The score depends on which connected component each vertex belongs to

Key insight:
- In a connected component with n₁ vertices and m₁ edges, if n₁ ≤ m₁ + 1 (a tree-like structure), I can achieve at most 2/3 score within that component
- If n₁ > m₁ + 1 (contains cycles), I can achieve even better by using both types of strategies

Strategy:
- Assign vertices to groups based on their component and whether they're in tree or cyclic parts
- Group A: vertices from tree components - these contribute 2/3 per vertex when isolated
- Group B: vertices from cyclic components - these contribute 4/9 each but share an edge bonus
- Use a knapsack-style greedy to maximize total score
- Within each group, optimize local score through balanced coloring

Implementation approach:
1. Build adjacency list and compute degrees
2. Find independent sets (nodes with degree 1 or leaves after removing leaves)
3. Classify remaining nodes as part of tree or cycle structures  
4. Use dynamic programming to select best combination of groups for global optimum
5. Reconstruct the optimal assignment and count conflicts

This balances between avoiding conflicts locally while considering the overall edge count across the entire graph.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

const int INF = 0x3f3f3f3f;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    vector<bool> used(n + 1, false);

    int t_cnt = 0, c_cnt = 0, e_cnt = 0;
    vector<int> t_part(n + 1, 0), c_part(n + 1, 0);

    auto process_component = [&](const vector<int>& ids, bool tree_like) -> void {
        vector<pair<int, int>> leaf_edge;
        vector<bool> comp_visited(ids.size());
        
        queue<int> q;
        for (int x : ids) q.push(x), comp_visited[ids.size()]--;
        
        while (!q.empty()) {
            int u = q.front(); q.pop();
            used[u] = true;
            
            if (g[u].size() <= 1 || 
                (tree_like && all_of(g[u].begin(), g[u].end(), [](int v){ return !used[v]; }))) {
                leaf_edge.push_back({u, -1});
            } else {
                comp_visited[ids.size()]--;
            }
        }

        if (tree_like && leaf_edge.size() >= 2) {
            sort(leaf_edge.begin(), leaf_edge.end());
            for (int i = 0; i < leaf_edge.size(); i += 2) {
                int u = leaf_edge[i].first;
                int v = leaf_edge[i+1].first;
                if (!used[v]) {
                    used[v] = true;
                    t_cnt++;
                    leaf_edge.push_back({v, -1});
                    leaf_edge.sort();
                }
            }
        }

        if (used.count(ids.back()) == 0 && leaf_edge.size() >= 2) {
            sort(leaf_edge.begin(), leaf_edge.end());
            for (int i = 0; i < leaf_edge.size(); i += 2) {
                int u = leaf_edge[i].first;
                int v = leaf_edge[i+1].first;
                if (!used[v]) {
                    used[v] = true;
                    c_cnt++;
                    leaf_edge.push_back({v, -1});
                    leaf_edge.sort();
                }
            }
        }

        int start_node = -1;
        for (int x : ids) if (!used[x]) { start_node = x; break; }

        if (start_node != -1) {
            vector<int> stack_list;
            queue<int> st_q;
            st_q.push(start_node);
            used[start_node] = true;
            st_list:;
            while(!st_q.empty()){
                int u = st_q.front(); st_q.pop();
                stack_list.push_back(u);
                for(int v : g[u]){
                    if(!used[v] && (stack_list.empty() || abs(stack_list.back()-v) > 1)) {
                        used[v] = true;
                        st_q.push(v);
                        goto st_list;
                    }
                }
                // Fallthrough for next node
                continue;
            }
            for(int u : stack_list) {
                t_part[u] = tree_like ? c_part[u] = t_cnt++ : c_part[u] = c_cnt++;
            }
            e_cnt = max(e_cnt, (int)floor(t_cnt * 2.0 / 3.0));
        }
    };

    for (int i = 1; i <= n; ++i) {
        if (!used[i]) {
            vector<int> ids;
            function<void(int)> dfs_ids = [&](int u) { ids.push_back(u); for(int v : g[u]) if(!used[v]) dfs_ids(v); };
            dfs_ids(i);
            process_component(ids, ids.size() <= g[i].front() ? (true) : (false));
        }
    }

    vector<double> A(1, 0.6667);
    vector<double> B(1, 0.4445);

    vector<double> F1(A.size()), F2(B.size());
    iota(F1.begin(), F1.end(), 0);
    iota(F2.begin(), F2.end(), 0);
    
    for(int j=1;j<=min((long long)c_cnt,(long long)B.size()-1);j++) {
        double sum = accumulate(B.begin()+c_cnt-j,B.end(),0.0);
        double cost = floor(sum+e_cnt*0.3333)+0.5;
        F2[j]=max(cost,F1[c_cnt]-sum);
    }
    F1.assign(c_cnt+1, 0.0);
    F1[0]=0.0;
    if (A.size() > 1) {
        for(int j=1;j<=min((long long)t_cnt,(long long)A.size()-1);j++){
            double sum=A[t_cnt-j]+0.6667*e_cnt;
            F1[j]=max((double)F1[j],F1[j-1]-sum);
        }
    }
    
    vector<int> pA(c_cnt+1,-1);
    vector<int> pB(c_cnt+1,-1);
    
    int best_idx = 0, max_val = -1;
    for(int i=0;i<=t_cnt;i++){
        for(int j=0;j<=min((long long)c_cnt,B.size()-1);j++){
            int cur=i+j,cost=j+i;
            double val=F1[i]+F2[j];
            if(val>max_val){
                max_val=val;best_idx=cur;
            }
        }
    }
    
    priority_queue<pair<double,int>,vector<pair<double,int>>,greater<pair<double,int>>> pq;
    pq.push({0.0,0});
    while(pq.size()>0){
        pair<double,int> pr=pq.top();pq.pop();
        if(pr.second>=best_idx||pr.first<=max_val)break;
    }
    
    for (int k = 1; k <= n; k++) {
        cout << ((t_part[k] > c_part[k] ? 1 : (t_part[k] == c_part[k] ? 2 : 3))) << " ";
    }
    cout << "\n";

    return 0;
}
```
````
