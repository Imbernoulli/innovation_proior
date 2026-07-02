# 题 120 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=4,得分 0.0,completion_tokens=6028)

````
The problem is to reconstruct an undirected simple graph with 100 vertices using queries. Each query gives us three distinct vertices a, b, c, and we get back the number of edges among them (0, 1, 2, or 3).

Let me think about this systematically:

1. If we query {a, b, c} and get 0 edges, it means no edges exist among these three vertices.
2. If we get 1 edge, exactly one pair has an edge.
3. If we get 2 edges, two pairs have edges.
4. If we get 3 edges, all three pairs have edges (triangle).

The key insight is that we need to determine all edges in the graph. Since there are n=100 vertices, there can be up to n*(n-1)/2 = 4950 edges.

For efficiency (Q < 3400), we need a clever approach. Let me think...

One approach: 
- For each vertex i from 1 to n-1, we can try to determine its connections to vertices j > i.
- We can use queries strategically to build up information.

Better approach:
- For each pair (i, j) where i < j, we want to know if edge (i, j) exists.
- We can use vertex k as a pivot to help determine edge (i, j).

Actually, let me think more carefully. If we query {i, j, k}:
- 3 edges means (i,j), (j,k), (k,i) all exist
- 2 edges means exactly 2 of these exist
- 1 edge means exactly 1 of these exists
- 0 edges means none exist

Here's a strategy:
1. First, fix one vertex (say vertex 1) and try to determine which vertices are connected to it.


2. For other vertices, we can use vertex 1 as a reference point in our queries.

A smart strategy would leverage vertex 1 as a reference. For each pair (i, j) where both are not 1, I can query {1, i, j}. The response tells me something about whether edges (1,i), (1,j), and (i,j) exist.

Specifically:
- Response 3 means all three edges exist
- Response 2 means 2 edges exist (either (1,i) and (1,j), or (1,i) and (i,j), or (1,j) and (i,j))
- Response 1 means 1 edge exists
- Response 0 means no edges exist

This approach uses approximately n(n-1) queries - around 10000 total - but we only have budget for ~3400 queries. So I need something much more efficient.

Actually, I should first figure out vertex 1's neighbors by querying triplets that include 1, then use that information to deduce the rest of the graph structure.

Once I know which vertices connect to vertex 1, I can process the remaining vertices more intelligently. For vertices without any edges to 1, I might still extract some adjacency information through careful query selection. The real opportunity lies in exploiting the fact that each query reveals relationships between three vertices simultaneously.

Let me reframe this as a systematic approach: query all pairs involving vertex 1 to map out its neighborhood, then leverage that mapping to understand connections among other vertices by strategically choosing third points in subsequent queries. .neighbors[j].count(i));
            }
        }
    }
    
    for (int i = 2; i <= n; i++) {
        // Query with vertex 1 already determined
        for (int j = 2; j <= n && j != i; j++) {
            if (!has_edge(1, i) && !has_edge(1, j)) {
                int result = query(1, i, j);
                // If result is 3, then both (1,i) and (1,j) should be 1, contradiction
                // Wait, if neither connects to 1, result must be 0 or 1 or 2
                
                // result 2 means either (i,j) exists plus one of (1,i),(1,j) exists
                // But we said neither connects to 1, so result must be 0 or 1
                
                if (result == 2) {
                    // This shouldn't happen if neither connects to 1
                    // Unless my logic is wrong
                    has_edge(i, j) = true;
                } else if (result == 1) {
                    has_edge(i, j) = true;
                } else if (result == 0) {
                    // No edge between i and j
                }
            }
        }
    }
    
    output();
}
```

There's a flaw in my reasoning. When neither i nor j connects to vertex 1, the query on {1, i, j} can only return 0 or 1—never 2 or 3, since at least one edge would require a connection to vertex 1 that doesn't exist. Yet when I check if both neighbors[1] contain i and j, I'm getting unexpected results that suggest edges are being recorded incorrectly.

I realize now that querying {1, i, j} directly only works efficiently when I already know most of the connections to vertex 1. To make this feasible within the query limit, I need a smarter strategy: first identify vertex 1's complete neighborhood using roughly n queries, then handle remaining vertex pairs more selectively based on what I've learned so far.

My revised approach: For each unqueried vertex pair (i, j), construct a query sequence that determines their connection by leveraging known relationships to vertex 1. Specifically, if I query {1, i, j} and get 0, I can immediately infer certain structural constraints about the triangle formed by these three vertices.

However, the fundamental challenge persists—determining all edges efficiently requires minimizing redundant queries while extracting maximum information from each response.

A better strategy emerges: for every vertex i, identify its degree by examining how many other vertices it connects to, then reconstruct the full adjacency matrix systematically using minimal queries.

Since we're allowed up to 3400 queries for 100 vertices, and there are potentially 4950 edges, we need an approach that extracts multiple edge states per query rather than querying pairs individually.

The key insight is using fixed pivot vertices strategically—if we pre-determine vertex 1's complete adjacency profile, we can then efficiently determine edges among remaining vertices by checking triangles involving that pivot point.

When querying {1, i, j}, the response encodes exactly 1 bit per potential edge among the three vertices. If we know edges (1,i) and (1,j), then response 3 forces (i,j)=1, response 2 indicates (i,j)=0, response 1 implies inconsistency, and response 0 contradicts the setup.

But if we haven't fully determined (1,i) or (1,j), the same query yields less definitive information—we'd need additional queries to resolve ambiguity, making this approach less efficient overall.

I should instead focus on determining edges among non-pivot vertices through direct pair queries, then reconstruct vertex 1's connections separately. This separates the problem into manageable subproblems rather than trying to solve everything in parallel.

Wait, I need to reconsider. With Q limited to 3400 and n at 100, I can actually afford roughly O(n²/3) queries if structured right. My current estimate of ~5000 queries exceeds this budget significantly.

The breakthrough comes from recognizing that knowing vertex 1's full neighborhood unlocks substantial inference power. Once I've identified all n-1 edges connecting to vertex 1, I can use that information to determine remaining edges among other vertices with just one query per missing edge.

So the real cost is: one query per edge incident to vertex 1 to establish those initial connections, then one query per edge in the induced subgraph of vertices 2-n to determine their mutual relationships. This totals roughly n + n*(n-3)/2 queries, which for n=100 gives approximately 4850—still too many.

I need to find a way to learn edges between other vertices without querying each pair individually. Perhaps by batching information or using the structure of vertex 1's neighborhood more cleverly.

The core observation: when I query {1, i, j}, the response reveals the state of edge (i,j) provided I already know (1,i) and (1,j). So once vertex 1's connections are mapped, each remaining edge costs just one query.

For efficiency, I should minimize upfront work determining vertex 1's neighborhood. I could sample strategically rather than exhaustively query, though this risks missing critical information. Given the 3400 query budget, maybe I can afford to skip some (1,k) checks if the resulting queries on {1, i, j} still yield useful deductions.

Let me try a concrete algorithm: initialize adjacency tracking for vertices 1-100, first scan all triplets containing vertex 1 to populate its neighborhood list, then systematically verify remaining edges using the query results. d_edge[i] ? 1 : 0, adj[i][1];
        
        int ans = query(a, b, c);
        
        // Determine edges based on response
        if (ans == 3) {
            add_edge(a, b);
            add_edge(b, c);
            add_edge(a, c);
        } else if (ans == 2) {
            if (adj[a][b] || adj[b][c] || adj[c][a]) {
                add_edge(a, b);
            } else {
                add_edge(b, c);
            }
        } else if (ans == 1) {
            if (adj[a][b]) {
                add_edge(a, c);
            } else if (adj[b][c]) {
                add_edge(c, a);
            } else if (adj[c][a]) {
                add_edge(a, b);
            }
        }
    }
    
    cout << "!" << endl;
    print_adjacency_matrix();
}
```

With this approach, I'd run roughly 15,000 queries, which exceeds the constraint. I need to optimize by processing vertices more systematically rather than exhaustively.

First, I'll determine vertex 1's complete neighborhood by checking each potential neighbor. For each vertex i from 2 to 100, I query {1, i, n+1} where n+1 serves as a fresh test vertex to avoid collisions. Based on the response—whether it's 0, 1, 2, or 3—I can deduce whether an edge exists between 1 and i. Once I know all of vertex 1's connections, I can efficiently resolve the remaining edges.

Now for the second phase, I examine every pair (i, j) where both exceed 1 and haven't been tested yet. Using the known edges from vertex 1 and the test vertex n+1, I can fill in the missing information about edges within the larger set.

If all edges to the test vertex exist and all edges to vertex 1 don't, I can directly add edges between i and j based on the query result. Otherwise, I need to combine information from previous queries to determine the edge state.

However, this approach creates a problem: querying with vertex n+1 becomes expensive as n grows, and the query count will quickly exceed the limit.

Instead, I should recognize that once vertex 1's neighborhood is determined, I only need to discover the actual edges between vertices 2-100. For any missing edge (i, j), I can query {1, i, j}. If I already know both (1, i) and (1, j) exist, then the query result directly tells me whether (i, j) exists—specifically, if the answer is 3, all three edges exist; if it's 2, only two edges exist, so (i, j) must be absent.

But if I don't know all the necessary edges, this approach breaks down. A better strategy focuses entirely on building vertex 1's connections first, then using those to deduce the rest.

To establish whether (1, i) exists, I query {1, i, j} for some j where I'm uncertain about (1, j). The key is finding a j such that the response reveals information about (1, i) without circular dependencies.

Alternatively, I can use a simpler method: fix i and pick various j values, then check responses across different combinations. If responses vary, I can isolate (1, i)'s state.

The cleanest path forward: dedicate initial queries to mapping vertex 1's full adjacency profile, then use that complete picture to efficiently determine all remaining edges in the graph.

For each vertex i from 2 to n, I'll query {1, i, n+1} repeatedly. By observing responses like 0, 1, 2, or 3, I can infer whether (1, i) exists—the pattern across queries reveals the truth. Then, armed with knowledge of all edges incident to vertex 1, I can compute the final answer by checking remaining edges.

However, I realize the sample shows s_{i,j} should be '0' if there's no edge. Looking at the sample output again, the adjacency matrix format needs clarification—whether rows represent outgoing edges or something else.

The sample shows four lines after the query results. Line 1 has '0001', meaning vertex 1 connects to vertex 4. Lines 2-3 also show '0001' for vertices 2 and 3, each connecting only to vertex 4. Line 4 shows '1110', indicating vertex 4 connects to vertices 1, 2, and 3.

So vertex 4 forms a triangle with vertices 1, 2, and 3—that explains why {1, 2, 4} returns 2, {1, 3, 4} returns 2, but {1, 2, 3} returns 0.

This confirms the graph is just a K₄ triangle with vertex 4 at the center. Now I need to implement the solution. First, I'll initialize the adjacency matrix to zero, then systematically determine which edges exist in the graph.

For each vertex i, I'll query {1, i, n+1} to gather information. When the response is 3, edges exist between all three vertices including (i, n+1), confirming both (1, i) and (i, n+1). When it's 2, I check whether (1, n+1) was already established to narrow down which specific edges exist. A response of 1 or 0 helps eliminate possibilities, and I can track this by counting occurrences of specific patterns across all queries.

After processing all vertices, I'll have enough information to reconstruct the graph. If no edges were found during the initial phase, I'll need additional queries to determine connectivity, but otherwise the accumulated data should reveal the structure. , n + 1):
            r = query(1, i, n + 1)
            responses.append(r)
    
    # Count each type of response
    count_0 = responses.count(0)
    count_1 = responses.count(1)
    count_2 = responses.count(2)
    count_3 = responses.count(3)
    
    # Vertex 1 connects to vertices with response >= 1
    for i in range(1, n):
        if responses[i] >= 1:
            adj[1][i] = 1
            adj[i][1] = 1
    
    # Now determine edges among 2..n
    for i in range(2, n + 1):
        for j in range(i + 1, n + 1):
            # Check edge (i, j) using query with vertex 1
            r = query(1, i, j)
            
            if adj[1][i] and adj[1][j]:
                # Both connected to 1, so check if they're connected to each other
                if r == 3:
                    adj[i][j] = 1
                    adj[j][i] = 1
                elif r == 1:
                    pass  # Edge doesn't exist
            
    # Output the adjacency matrix
    for i in range(1, n + 1):
        row = "".join(str(adj[i][j]) for j in range(1, n + 1))
        print(row)

I need to reconsider the strategy here. The approach assumes I can determine edges among non-pivot vertices by querying with vertex 1, but I should verify this logic holds for the actual graph structure.

For each unconnected pair (i, j) where neither connects to vertex 1, querying {1, i, j} will give me a response. If that response is 0, the edge doesn't exist. If it's 1, the edge does exist. Responses of 2 or 3 contradict the assumption that neither i nor j connects to 1, so those cases won't occur. This means I can reliably determine the edge state based on the query result.

However, this approach still requires too many queries overall. I need a more efficient strategy that leverages the initial information I gathered about vertex 1's connections.

The key realization is that once I determine whether (1, i) exists for all i, I can use that as a foundation. If I already know the edges incident to vertex 1, then for any query {1, i, j}, I can directly infer whether (i, j) exists by comparing the response against what I already know about (1, i) and (1, j). This eliminates redundant queries and dramatically reduces the total count needed.

However, there's a practical issue: initially I have incomplete information about which vertices connect to vertex 1. I can't simply query every {1, i, j} because I'll waste queries on pairs where I haven't yet determined the relationship to vertex 1. Instead, I should focus on establishing the connections to vertex 1 first, then systematically fill in the remaining edges among other vertices using those known relationships as anchors.

Given the query budget constraint, I need to balance the upfront cost of mapping vertex 1's neighborhood against the benefit of having reliable information for subsequent queries. Perhaps a hybrid strategy works best—determining a subset of vertex 1's connections precisely enough to unlock enough information about the rest of the graph, rather than trying to determine everything at once.

Actually, let me reconsider. If I assume vertex 1 isn't connected to most vertices initially, then queries involving {1, i, j} where neither connects to 1 will give 0, and I can only confirm the edge (i, j) exists if I get 1. But this still requires too many queries.

The real solution is simpler: query {1, i, j} whenever I'm uncertain about both (1, i) and (1, j). The response immediately tells me whether (i, j) exists without needing prior knowledge of the individual edges to vertex 1.

However, this gets messy quickly. Let me step back and think about the core principle: each query on {1, i, j} provides one bit of information about the triangle formed by those three vertices. If I have good partial information about some edges, I can infer others. If I have little information, I'm constrained.

A viable strategy: start by determining all edges incident to vertex 1 using queries of the form {1, i, n} where n is a carefully chosen vertex. Once I know which vertices connect to 1, I can fill in the rest of the graph by querying remaining missing edges.

But wait—I'm overcomplicating this. What if I take a probabilistic angle? Randomly select query triples and observe the statistics. Over time, the frequency distribution of responses might reveal something about the graph's density and structure. Though this feels impractical for exact reconstruction.

Let me settle on a concrete approach: prioritize establishing vertex 1's connections completely first, then tackle the remaining edges. The trade-off is accepting some inefficiency in later steps, but the certainty of accurate reconstruction matters more than perfect query optimization.

For each vertex i, I'll perform two queries to determine whether (1, i) exists—one with vertex 1 and another with a different anchor point. Based on the responses, I can deduce the edge status. Once I have all of vertex 1's neighbors identified, I can systematically check the remaining edge pairs using triangular queries that leverage the known information.

However, I'm realizing this approach still requires excessive queries. I need to reconsider the strategy entirely.

What if I abandon trying to fully map vertex 1's neighborhood upfront? Instead, I could attempt to infer most edges between vertices 2 through n by making strategic choices about which query to construct next, potentially saving resources on vertices 1's connections.

Or perhaps I should embrace randomness—using randomly selected triples across the entire graph until I've confidently determined sufficient edges to reconstruct the solution.

Actually, I think I can leverage the constraint more effectively. The goal is Q < 3400, which gives me roughly O(n²) queries. Since n=100, I can afford about one query per pair on average, minus some overhead.

The key insight: by fixing vertex 1 as a common element in queries, I gain significant information from each query about the relationships between the other two vertices.

Let me implement a cleaner version: query all triples containing vertex 1 to establish baseline information, then use that knowledge to efficiently determine the remaining edges.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int N = 100;
int adj[105][105];

void flush() {
    cout.flush();
}

int query(int a, int b, int c) {
    cout << "?" << " " << a << " " << b << " " << c << "\n";
    flush();
    int res;
    cin >> res;
    flush();
    return res;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    memset(adj, 0, sizeof(adj));
    
    // Phase 1: Determine all edges from vertex 1
    for (int i = 2; i <= N; i++) {
        // Pick a vertex j that we suspect may connect to 1
        // Use i+1 cyclically to avoid issues
        int j = (i % (N - 1)) + 2;
        
        // Try to determine if (1, i) exists
        // If response is 3, both (1,i) and (1,j) must exist (since they share 1)
        // Actually, let's be systematic: query {1, i, j} where we track results
        
        // Better: Use binary search-style approach for each edge (1,i)
        // For simplicity: assume we query multiple times
        
        for (int k = 0; k < 2; k++) {
            int t1 = ((i * 7) + 13) % N + 1;
            int t2 = ((i * 11) + 17) % N + 1;
            
            if (t1 == i) t1 = (i + 3) % N + 1;
            if (t2 == i) t2 = (i + 5) % N + 1;
            
            int r = query(t1, t2, i);
            
            if (r == 3) {
                adj[t1][i] = adj[i][t1] = 1;
                adj[t2][i] = adj[i][t2] = 1;
            } else if (r == 2) {
                if (adj[t1][t2]) {
                    // Already know (t1,t2), so one of the others exists
                    adj[i][t1] = 1;
                }
            }
        }
        
        // Finalize (1,i) determination
        for (int k = 0; k < N; k++) {
            if (k == i) continue;
            int j = (k == 1) ? 2 : (k > 1 ? k : 1);
            if (!adj[1][k] && !adj[1][j] && k != i && j != i) {
                int r = query(1, i, k);
                if (r == 1) adj[1][i] = adj[i][1] = 1;
            }
        }
    }
    
    // More efficient Phase 1: Use random-like systematic approach
    memset(adj, 0, sizeof(adj));
    
    // Directly determine (1,i) for all i using smart queries
    for (int i = 2; i <= N; i++) {
        bool connected = false;
        for (int k = 0; k < 5; k++) {
            int j = (2 + i + k * 7) % N + 1;
            if (j == 1) j = 2;
            int r = query(1, i, j);
            
            // If we get 3 or 2, likely connected
            if (r == 3) {
                connected = true;
                break;
            } else if (r == 2 && adj[1][j]) {
                // Could be (1,i) exists, or (i,j) exists when (1,j) exists
                // Need to track more carefully
            }
        }
        
        if (connected) adj[1][i] = adj[i][1] = 1;
    }
    
    // Phase 2: Determine all remaining edges using known (1,i) info
    for (int i = 2; i <= N; i++) {
        for (int j = i + 1; j <= N; j++) {
            // Query {1, i, j}
            int r = query(1, i, j);
            
            int e1 = adj[1][i];
            int e2 = adj[1][j];
            
            if (e1 && e2) {
                // Both connect to 1, r tells us about (i,j)
                if (r == 3) adj[i][j] = adj[j][i] = 1;
            } else {
                // At least one doesn't connect to 1
                // r=2 means exactly one of {(1,i),(1,j),(i,j)} exists
                // r=1 means exactly one exists
                // r=0 means none exist
                
                if (r == 2 && !adj[i][j]) adj[i][j] = adj[j][i] = 1;
                else if (r == 1) adj[i][j] = adj[j][i] = 1;
            }
        }
    }
    
    // Output the graph
    cout << "! \n";
    flush();
    
    for (int i = 1; i <= N; i++) {
        string s = "";
        for (int j = 1; j <= N; j++) {
            s += (adj[i][j] ? '1' : '0');
        }
        cout << s << "\n";
        flush();
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=1108)

````
I need to reconstruct an undirected graph with 100 vertices using queries that return the number of edges among any three chosen vertices.

Key observations:
1. A query returns how many edges exist in the induced subgraph on three vertices (0-3 edges)
2. I can determine edge existence by combining queries - a key insight is XORing query results
3. When checking {a,b,c} and {a,b,d}, the common elements are a and b, so if these queries return different values, then one of a-b, c-d must be an edge.

Strategy:
- Start with vertex 1 as reference
- For each other vertex i, check all pairs {1, j} where j < i, testing {1, j, i}: if result=2, edge(1,i) exists; if result=1, edge(1,j) exists but not edge(1,i)
- Build adjacency matrix row by row from vertex 1, then for subsequent vertices use the same approach against known neighbors

Output format: 100 lines of 100-character binary strings where s[i][j]=1 iff there's an edge between i and j (i and j from 1-100). Note that the sample shows row i corresponds to vertex i, not the string index.
</think>

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

int query(int a, int b, int c) {
    cout << "? " << a << " " << b << " " << c << endl;
    int ans; cin >> ans;
    return ans;
}

int main() {
    vector<vector<int>> adj(105, vector<int>(105, 0));

    // Step 1: Reconstruct edges from vertex 1
    for (int i = 2; i <= 100; i++) {
        int e1 = query(1, i, 2), e2 = query(1, i, 3);
        if (e1 == 2 || e2 == 2) adj[1][i] = adj[i][1] = 1;
        else {
            // If no edge incident to 1 in {1,i,2} or {1,i,3}, then either
            // edge(1,i)=edge(2,3) OR edge(1,i)=edge(2,i) etc.
            if ((e1 == 1 && e2 == 1) || (e1 == 0 && e2 == 0)) {
                // Query edge(2,3):
                // edge(1,2)=1 implies one is fixed -> query edge(1,3)
                if (query(1, 2, 3) == 1)
                    adj[1][i] = adj[i][1] = 1;
            }
        }
    }

    // Step 2: For each new vertex u, determine its connections to earlier ones
    for (int u = 2; u <= 100; u++) {
        for (int v = 1; v < u; v++) {
            bool fixed = false;
            for (int w = 1; w < v; w++) {
                // Check triangle {v,w,u}: value 2 => edge(v,w) already set -> edge(v,u) not yet determined
                if (adj[v][w]) {
                    if (query(v, w, u) == 2) fixed = true;
                    break;
                }
            }
            // Also check first two neighbors of v to confirm
            if (!fixed && adj[v][2] && query(v, 2, u) == 2) fixed = true;
            if (!fixed && adj[v][3] && query(v, 3, u) == 2) fixed = true;

            if (fixed) {
                int t1 = query(1, v, u), t2 = query(1, 2, u);
                // Since edge(v,2)=1 already known, compare triangles {1,v,u} vs {1,2,u}
                if ((t1 == 2) != (t2 == 2))
                    adj[v][u] = adj[u][v] = 1;
            }
        }
    }

    // Output the reconstructed graph: 100 rows, each a 100-char binary string
    // row i encodes edges FROM vertex i TO vertices j (where j ranges 1..100)
    for (int i = 1; i <= 100; i++) {
        string row = "";
        for (int j = 1; j <= 100; j++) {
            row += (adj[i][j] ? '1' : '0');
        }
        cout << row << endl;
    }
    cout << "!" << endl;

    return 0;
}
```
````
