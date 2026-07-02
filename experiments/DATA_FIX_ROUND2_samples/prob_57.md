# 题 57 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 0.0,completion_tokens=4166)

````
The problem is an interactive problem where we need to determine the values of all nodes in a tree. The tree has $n$ nodes, each with value 1 or -1. We can query the sum of path sums from the root to various nodes, and we can toggle node values.

Key observations:
1. For any node $u$, let $f(u)$ be the sum of values on the path from root to $u$.
2. If we pick two nodes $u$ and $v$, then $f(u) + f(v) = f(\text{LCA}(u,v)) \times 2 + \sum_{\text{path}(root,u)} + \sum_{\text{path}(root,v)}$ 
   Actually more precisely: $f(u) = \text{val}(root) + \dots + \text{val}(u)$
   $f(v) = \text{val}(root) + \dots + \text{val}(v)$
   So if LCA is $l$, then $f(u) + f(v) = 2 \cdot f(l) + \sum \text{values on paths from } l \text{ to } u \text{ and } l \text{ to } v$ (excluding $l$ once)

3. A useful property: if we query single nodes one by one, we get $f(u)$ for each node $u$. But we want to minimize queries.

4. Important observation: For any edge $(u, v)$, if we know the structure of the tree, then toggling a node changes the sum along certain paths.

5. Let's think about what we can learn:
   - If we query ? 1 k u1 u2 ... uk, we get $\sum f(ui)$
   - If we query ? 2 u, we flip node u's value

6. Key insight: If we fix the structure of the tree, we can use BFS/DFS to establish parent-child relationships. Then for each node, we can relate $f(child) = f(parent) + val(child)$.

7. But wait - we don't know which node is the root! That's important.

8. However, the relative relationships are fixed regardless of which node we consider root. For any two nodes $u, v$, their path relationship is fixed.

9. Strategy:
   - First, we need to figure out the tree structure (which nodes are connected)
   - Then, we need to determine actual node values
   
Let me think about a different approach:
- Querying single nodes gives us $f(u)$
- If we have $f(u)$ and $f(v)$ where $v$ is a child of $u$, then $f(v) - f(u) = \text{value}(v)$
- So if we can determine parent-child relationships and get enough $f()$ values, we can compute individual values

But the tree is unrooted initially. However, we can query to learn about it.

Actually, I realize that the tree structure is given to us in the input (n-1 edges). So we DO know the adjacency structure, just not who is the root.

Let me reconsider: Given the tree structure, if we pick ANY node as root and run DFS/BFS, we get parent-child relationships. The root choice is arbitrary but consistent within our analysis.

Wait, there's a constraint: we can only make at most $n+1000$ queries total. Since $n \leq 1000$, this means around $2n$ queries maximum per test case roughly speaking (though the scoring says n queries gets 100 points, up to n+1000).

Let me think about what minimal information we need:
1. We need to know the value of every node
2. Each type-1 query with k=1 gives us f(u) for some u
3. Each type-2 query toggles a node value (useful if needed during interaction)

If we know f(u) for ALL nodes AND we know the tree structure, can we recover all node values?

Yes! Here's how:
- Pick any node as root (say node 1)
- Compute f(root) = value(root)  
- For each child v of parent p: f(v) = f(p) + value(v), so value(v) = f(v) - f(p)
- This propagates through the entire tree

So strategy:
1. Build adjacency list from input
2. Pick node 1 as our "virtual root" for traversal purposes
3. Do BFS/DFS to build parent pointers
4. Query ? 1 1 1 2 ... n (all nodes) to get f(1), f(2), ..., f(n)
5. Use the parent-child relationships to extract individual node values
6. Output answers

Wait, querying all n nodes would be n queries which is acceptable for full score. But let's check the constraints again...

n <= 1000, sum of n <= 1000 across test cases, max queries n+1000 allowed.
So n queries exactly is fine for maximum score.

However, there's a catch: after some type-2 queries happen (value flips), the original f(u) values become invalid. But looking at the problem statement more carefully, it seems type-2 queries will be issued BY THE INTERACTOR after we print our answer? No wait, reading the problem:

"Cow the Nerd wins if he guesses the value of every node correctly (the values of the final tree, after performing the queries)"

So we may issue type-2 queries ourselves to help gather information, and we need to track those changes.

Actually, looking at the example:
- They issue ? 1 queries first
- Then ? 2 queries to toggle
- Then ? 1 queries again
- Finally ! answer

So the interactor might be issuing responses to our type-2 queries. We need to keep track of state changes.

Given the complexity here, let me focus on a cleaner solution:

1. Read tree structure and build adjacency list
2. Perform BFS starting from node 1 to establish parent-child relationships (even though we don't know true root, this gives us a consistent tree traversal)
3. Query all nodes individually using type-1 queries to get f values
4. Compute node values from these f values using the parent-child relationships
5. Store current node values
6. When we issue a type-2 query, update our knowledge base accordingly

But wait, the problem doesn't say the interactor makes type-2 queries. Looking again:

"To ask this query, print: ? 2 u and read the response (no output, just toggle)."

So type-2 is OUR query to flip a node. The response is implicitly the flipped state acknowledged.

Actually rereading: when we print "? 2 u", we're telling the interactor to flip node u. The interactor confirms this somehow (but the problem says "read the response (no output, just toggle)") which is confusing. Let me look at the example interaction format.

From example:
```
? 1 3 1 2 4
? 1 2 3 1
? 2 4
? 1 3 1 2 4
? 1 2 3 1
! -1 -1 -1 -1
```

So they're alternating between type-1 queries and a type-2 query. After type-2 ? 2 4, there's no explicit response shown in the example output (just next query continues). But typically in interactive problems, you'd read a response.

Looking at "read the response (no output, just toggle)" - this probably means we send the query and the interactor processes it (flips the value), and we continue. There's no additional integer to read after type-2 query based on this phrasing.

So strategy refined:
1. Build adjacency list
2. Root at node 1 for traversal (arbitrary)
3. Run BFS to get parent array and level/order for propagation
4. Query all nodes one-by-one or in batches via type-1
5. Compute initial node values
6. Handle any type-2 updates if we decide to make them
7. Print final answer

Simplest optimal approach: Just query each node individually n times, calculate values, submit. No need for complex batching or type-2 queries unless necessary.

Actually wait - if I query f(u) for all u, then:
- Set node i's value = f(i) - f(parent[i]) except for root where value = f(root)

This works perfectly!

Let me code this solution.

Wait, one more consideration - what about Type-2 queries being REQUIRED by the problem? Re-reading:

"Cow the Nerd wins if he guesses the value of every node correctly"

And there's a scoring system where:
- At most n queries = 100 points
- More than n+1000 queries = 0 points
- Between n and n+1000 = linear interpolation

So making exactly n queries (type-1, one per node) should give 100 points IF that's sufficient to solve. It should be since with f(values) and tree structure we can always compute individual values.

Let me implement this solution:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t;
    cin >> t;
    while(t--) {
        int n;
        cin >> n;
        
        vector<vector<int>> adj(n + 1);
        for(int i = 0; i < n - 1; i++) {
            int u, v;
            cin >> u >> v;
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        
        // Build parent-child relationship with node 1 as root
        vector<int> parent(n + 1, 0);
        vector<bool> visited(n + 1, false);
        queue<int> q;
        
        q.push(1);
        visited[1] = true;
        parent[1] = 0;
        
        while(!q.empty()) {
            int u = q.front();
            q.pop();
            
            for(int v : adj[u]) {
                if(!visited[v]) {
                    visited[v] = true;
                    parent[v] = u;
                    q.push(v);
                }
            }
        }
        
        // Get f values for all nodes
        vector<int> f(n + 1);
        vector<int> values(n + 1);
        
        // Build order of processing from BFS
        vector<int> bfs_order;
        q.push(1);
        visited.assign(n + 1, false);
        while(!q.empty()){
            int u = q.front();
            q.pop();
            bfs_order.push_back(u);
            visited[u] = true;
            for(int v : adj[u]){
                if(!visited[v]){
                    visited[v] = true;
                    parent[v] = u;
                    q.push(v);
                }
            }
        }
        
        // Clear visited for re-use if needed
        fill(visited.begin(), visited.end(), false);
        
        // Query each node to get f(node)
        for(int i = 1; i <= n; i++) {
            cout << "? 1 1 " << i << "\n";
            flush(cout);
            cin >> f[i];
        }
        
        // Compute actual node values
        for(auto &node : bfs_order) {
            if(parent[node] == 0) {
                values[node] = f[node]; // Root node
            } else {
                values[node] = f[node] - f[parent[node]];
            }
        }
        
        // Output result
        for(int i = 1; i <= n; i++) {
            cout << values[i] << (i == n ? "" : " ");
        }
        cout << "\n";
    }
    return 0;
}
```

Wait, I'm double-traversing the graph unnecessarily. Also, I need to handle the flushing properly. Let me simplify.

Actually, there's one more concern I need to address: The problem states the values change when we query ? 2 u. If I never use ? 2 queries, the values won't change, so my solution works. But does the interactor ever call back with changes? 

Looking at the interaction protocol: When I print "? 2 u", I tell the interactor to flip node u. The interactor acknowledges this internally. If I later query that node, I'll get the new f value which accounts for the flip.

For my simple n-query approach without type-2, I'm essentially getting the current state and solving it immediately. Since I query all nodes and compute values right away, the answer should be correct for the current tree state.

Let me optimize the code a bit:

```cpp
#include <iostream>
#include <vector>
#include <queue>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t;
    cin >> t;
    while(t--) {
        int n;
        cin >> n;
        
        vector<vector<int>> adj(n + 1);
        for(int i = 0; i < n - 1; i++) {
            int u, v;
            cin >> u >> v;
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        
        // BFS to establish parent-child relationships (rooting at node 1)
        vector<int> parent(n + 1, 0);
        vector<int> bfs_order;
        vector<bool> visited(n + 1, false);
        
        queue<int> q;
        q.push(1);
        visited[1] = true;
        
        while(!q.empty()) {
            int u = q.front();
            q.pop();
            bfs_order.push_back(u);
            
            for(int v : adj[u]) {
                if(!visited[v]) {
                    visited[v] = true;
                    parent[v] = u;
                    q.push(v);
                }
            }
        }
        
        vector<int> f(n + 1);
        vector<int> values(n + 1);
        
        // Query each node individually
        for(int i = 1; i <= n; i++) {
            cout << "? 1 1 " << i << "\n";
            cout.flush();
            cin >> f[i];
        }
        
        // Calculate actual node values
        for(int node : bfs_order) {
            if(parent[node] == 0) {
                values[node] = f[node];
            } else {
                values[node] = f[node] - f[parent[node]];
            }
        }
        
        // Output results
        for(int i = 1; i <= n; i++) {
            cout << values[i] << (i == n ? "" : " ");
        }
        cout << "\n";
        cout.flush();
    }
    return 0;
}
```

I've streamlined the implementation, focusing on the core algorithm while maintaining readability and efficiency.

Potential concerns include handling the tree traversal carefully and ensuring the calculation logic correctly extracts individual node values from path sums.

I'll verify the parent relationship computation and ensure proper node indexing throughout the solution.

When computing node values, I'm tracking path sum differences carefully. Each node's value derives from its path sum minus its parent's path sum, capturing the precise contribution.

The output formatting ensures clean space separation between values, with a newline termination. Traversal order from BFS guarantees logical parent-before-child processing, maintaining computational integrity.

Memory management looks efficient, allocating vectors only for necessary size plus one, avoiding unnecessary overhead. Time complexity scales linearly with tree size, meeting performance requirements.

One critical detail remains: confirming the `parent` initialization prevents unintended default value interpretation for root node calculations.

The BFS traversal ensures `parent[root]` stays at zero, preventing incorrect computations for the root's value assignment. This handles the special case elegantly within the main value calculation loop.

I'll trust the BFS approach to systematically process nodes, guaranteeing each child encounters its parent before independent value calculation.
</think>

```cpp
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;

        vector<vector<int>> adj(n + 1);
        for (int i = 0; i < n - 1; i++) {
            int u, v;
            cin >> u >> v;
            adj[u].push_back(v);
            adj[v].push_back(u);
        }

        // BFS to establish parent-child relationships (treating node 1 as root)
        vector<int> parent(n + 1, 0);
        vector<int> bfs_order;
        vector<bool> visited(n + 1, false);

        queue<int> q;
        q.push(1);
        visited[1] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            bfs_order.push_back(u);

            for (int v : adj[u]) {
                if (!visited[v]) {
                    visited[v] = true;
                    parent[v] = u;
                    q.push(v);
                }
            }
        }

        vector<int> f(n + 1);
        vector<int> values(n + 1);

        // Query each node individually to get f(u) = sum of path from virtual root to u
        for (int i = 1; i <= n; i++) {
            cout << "? 1 1 " << i << "\n";
            cout.flush();
            cin >> f[i];
        }

        // Compute actual node values
        // f(u) = f(parent[u]) + value(u), so value(u) = f(u) - f(parent[u])
        // For root: value(root) = f(root)
        for (int node : bfs_order) {
            if (parent[node] == 0) {
                values[node] = f[node];
            } else {
                values[node] = f[node] - f[parent[node]];
            }
        }

        // Output results
        for (int i = 1; i <= n; i++) {
            cout << values[i] << (i == n ? "" : " ");
        }
        cout << "\n";
        cout.flush();
    }

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=858)

````
I'll call the queries type-1 and type-2 for convenience. In a type-1 query, we get back $\sum_{j=1}^{k} f(u_j)$ where each $f(u_j)$ is the sum of values on the path from root to $u_j$.
This means $f(u) = \text{value of node } u + \sum_{\text{children } v \text{ of } u} f(v)$.

**Key insight:** When we ask "? 1" for a set of nodes, if the tree had no edges (just isolated nodes), then $f(u)$ would equal the value of node $u$, which is $\pm 1$, so the sum would be very small in absolute value. However, with edges present, $f(u)$ can accumulate many values along paths, making the total sum potentially much larger. We can use this as a proxy for how many paths are actually connected.

Consider the case where every edge has exactly one incident endpoint among our queried nodes, and all other endpoints have value -1. For each such edge, it contributes approximately +1 to its own endpoint's $f$-value. Summing over all these nodes gives us roughly the number of edges with both endpoints involved. If we query all $n$ nodes, we're essentially counting edges that contribute to the sums—this could reveal structural information about the tree when we look at subsets of nodes.

The trick is to make this count non-zero in a controlled way. By ensuring only certain nodes appear as endpoints of queried edges (the "odd-degree" endpoints relative to our selection), we force their $f$-values to be around ±1, while most others cancel out (being surrounded by -1s or not being edge endpoints). This lets us isolate edge counts from individual node values in type-1 responses, letting us piece together both structure and values.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int n;
vector<int> parent(1005);
vector<int> deg(1005);
vector<pair<int,int>> adj[1005];

void dfs(int u, int p=0, int val=0) {
    parent[u]=p;
    deg[u]=adj[u].size();
    cout<<"? 2 "<<u<<endl;
    for(auto [v,_]:adj[u])if(v!=p){
        int t=0;
        cout<<"? 1 2 ";
        for(int i=u,v1=v,u2=parent[v];i!=u&&i!=-1;i=u2){
            if(i==u)break;
            cout<<i<<" ";
            u2=v1;
            v1=u2;
        }
        cout<<u<<" ";
        cout<<"?"<<endl;
        cin>>t;
        if((t+val)%4==2){
            vector<int>t1={};
            int i=0,j=0;
            cout<<"? 1 1 ";for(;i<u;i++)cout<<u<<" ";cout<<endl;cin>>t;
            for(t+=1-(deg[i]%2==3?-1:1);!j;){
                for(int x=i;x>=0&&x<=u&&!j;x-=1,i--)if(deg[x]%2==1)t+=-1*(deg[x]%2==3?-1:1);if(!j){j=1;j=t;}
            }
            //print answer
        }
    }
}

main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);
    int tc;cin>>tc;while(tc--){
        cin>>n;
        for(int i=1;i<n;i++){int u,v;cin>>u>>v;adj[u].pb({v,0});adj[v].pb({u,0});}
        dfs(0);
        cout<<"!";
    }
}
```
````
