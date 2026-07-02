# 题 187 — base均分 72.9 vs 直接SFT均分 1.5(掉点 71.4)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 75.7,completion_tokens=14957)

````
The problem asks us to find a clique cover of an undirected graph $G = (V, E)$. A clique cover is a partitioning of the vertex set $V$ into subsets such that each subset forms a clique. We want to minimize the number of cliques used (i.e., minimize the maximum ID assigned).

This problem is equivalent to the minimum coloring of the complement graph $\bar{G}$. If we have a valid coloring of $\bar{G}$ with $k$ colors, then for each color class in $\bar{G}$, no two vertices are connected in $\bar{G}$, meaning they *are* connected in $G$, forming a clique. So finding a clique cover in $G$ is exactly finding a vertex coloring in $\bar{G}$.

Since $N \le 500$, this suggests an algorithm with complexity around $O(N^3)$ or maybe slightly higher is acceptable within 2 seconds. The greedy strategy for coloring graphs often produces good results but not necessarily optimal ones. However, since we just need to maximize the score ($K^*/K$), minimizing $K$ is key.

Let's look at constraints: $N \le 500$.
A standard greedy coloring algorithm processes vertices one by one and assigns them the smallest available color that doesn't conflict with their already colored neighbors. In the context of the complement graph $\bar{G}$, this means assigning a new clique ID if the vertex shares an edge with some already processed vertices in $G$? No, wait.
Let's re-read carefully.
Clique Cover in $G$: Partition $V$ into sets $S_1, S_2, \dots, S_k$ where each $S_j$ induces a complete subgraph (clique) in $G$.
Vertex Coloring in $\bar{G}$: Partition $V$ into independent sets $I_1, I_2, \dots, I_k$ in $\bar{G}$, which corresponds to cliques in $G$.

Greedy Algorithm for Graph Coloring (First-Fit):
Sort vertices in some order. For each vertex $u$, assign it the smallest positive integer $c$ such that no neighbor $v$ of $u$ in the target graph (here $\bar{G}$) has color $c$.
Wait, if we are doing Clique Cover on $G$, we want vertices with same ID to be connected in $G$. This means if $u, v$ have same ID, $(u,v) \in E(G)$.
So if we process vertices, say vertex $u$, and we want to put it in a group with existing vertices $v_1, \dots, v_{current\_size}$, then $u$ must be connected to all $v_i$ in $E(G)$.
Basically, we can try to form maximal cliques or use a heuristic to pack vertices into fewest groups.

Since it's equivalent to coloring $\bar{G}$, let's construct $\bar{G}$. An edge exists in $\bar{G}$ iff no edge exists in $G$. Then we want to color $\bar{G}$ minimally.
Greedy coloring on $\bar{G}$:
For each vertex $v$ (in some order), assign the smallest color $c \ge 1$ such that no neighbor $u$ of $v$ in $\bar{G}$ has color $c$.
Since constructing $\bar{G}$ explicitly takes $O(N^2)$, and checking neighbors takes time, total time could be around $O(N \cdot (\text{neighbors}))$. Since sum of degrees in $\bar{G}$ is at most $N(N-1)/2$, naive check is fine.
Wait, checking "smallest color" requires iterating over used colors. With max colors up to $N$, this is roughly $O(N^3)$ overall. Given $N=500$, $N^3 = 1.25 \times 10^8$, which fits well within 2.0s.

However, the order of vertices matters for the quality of the coloring (greedy). Standard ordering like sorted degree helps but here we might want to sort based on something else.
Actually, the "Maximum Cardinality Search" or similar heuristics work well.
Or simply sorting vertices by degree in $\bar{G}$ descending usually works better than random or ascending.
Wait, high degree in $\bar{G}$ means low degree in $G$.
Vertices with high degree in $G$ (low degree in $\bar{G}$) are hard to exclude from cliques.
Actually, let's think about the Clique Cover heuristic directly.
We want to group vertices that are fully connected.
One simple heuristic: Greedy clique packing.
Pick a vertex $u$. Start a new clique group $C = \{u\}$.
Iterate through other unassigned vertices $v$. If $v$ is connected to all currently selected members of $C$ (including $u$), add $v$ to $C$.
Once $C$ cannot grow anymore (or we just stop arbitrarily), close the group and start next.
Is this efficient? Checking connectivity for each candidate against current group.
If current group size is $s$, check takes $O(s)$.
Total candidates $N$. Total operations roughly $\sum_{groups} N \cdot s_i$. Worst case $\sum N \approx N^2$ if $s_i$ small, or $\sum N^2$ ?
Actually, simpler: iterate vertices. Maintain a list of open cliques. For each vertex $v$, try to fit into first compatible open clique. If none, create new.
Checking compatibility: for a specific open clique $C$, is $v$ adjacent to all $u \in C$?
To speed up this, maintain bitsets or boolean adjacency matrix.
With $N=500$, `std::bitset<500>` allows fast AND operation.
Checking if $v$ connects to all nodes in clique $C$ is equivalent to checking `adj[v] & mask[C] == mask[C]`.
Bitset operations take $O(N/64)$. Total time $O(N^2 \cdot N/64) \approx O(N^3 / 64)$. Very fast.
Strategy:
Iterate $i$ from 1 to $N$ (vertex index).
Maintain a list of active open cliques (represented by their masks of covered vertices so far).
When considering vertex $v$, iterate through active cliques. Find the first one where $v$ is connected to ALL members in that clique.
To do this efficiently, we don't strictly need to iterate all members if we store a bitmask for the clique.
If we find a compatible clique, assign $v$ to it and update its mask.
If $v$ is not compatible with any existing clique, create a new clique containing only $v$.
Wait, does this guarantee validity? Yes, because we only add if connected to everyone. Does it minimize cliques?
It's a greedy approach. By trying to add to existing cliques before starting a new one, it tends to pack vertices well.
The order of processing vertices matters. Sorting vertices by degree in $G$ (descending) is a common heuristic for coloring/clique packing. Vertices with high degree in $G$ should likely be part of large cliques together.
Specifically, if $v$ has high degree in $G$, it has fewer non-neighbors. It is easier to satisfy "connected to all in clique". Actually, the condition is that we must be connected to *everyone* in the clique. If the clique is growing, it restricts candidates more.
Heuristic for ordering:
In the context of Clique Cover of $G$:
We want to pick vertices that can belong to many potential cliques?
Usually, sorting by degree in $G$ ascending makes sense for coloring $\bar{G}$. Wait.
Coloring $\bar{G}$: we want to assign different colors to neighbors in $\bar{G}$. Neighbors in $\bar{G}$ are non-neighbors in $G$.
So if degree in $\bar{G}$ is high (degree in $G$ is low), the constraints are tight.
Standard greedy coloring sorts vertices by degree in the graph being colored. Here graph is $\bar{G}$. So sort by degree in $\bar{G}$ descending (which is degree in $G$ ascending).
Let's stick to the direct Clique Cover construction using bitsets.

Algorithm Refined:
1. Read $N, M$.
2. Build Adjacency Matrix (or Bitsets). `adj[i][j]` is true if edge exists.
3. Compute degrees of vertices in $G$ (number of edges incident).
4. Sort vertices based on degree. Heuristic: Process vertices with highest degree first? Or lowest?
   Let's consider the goal: minimize number of cliques.
   This is equivalent to coloring $\bar{G}$.
   In $\bar{G}$, high degree nodes are those with low degree in $G$.
   Greedy coloring generally prefers sorting by decreasing degree. So process vertices with lower degree in $G$ first (higher degree in $\bar{G}$)?
   Actually, DSatur algorithm is often cited as better than simple degree heuristics for coloring, but that's complex to implement.
   Simple degree heuristic is usually sufficient for competitive programming unless the test cases are malicious.
   Let's try sorting by degree in $G$ (ascending).
   
   Wait, if I sort by degree in $G$ descending, high degree nodes in $G$ come first. High degree node in $G$ means it *wants* to join big cliques easily? Not necessarily. It is connected to many nodes, but to join a clique, it must be connected to *all* currently picked nodes in that clique.
   If I pick a high degree node, it is likely connected to almost everyone, so it can potentially merge into a clique started by others.
   If I pick a low degree node (sparse in $G$), it might fail to connect to nodes in an existing clique immediately.
   Usually for Vertex Coloring (which is our equivalent problem), sorting by decreasing degree in the graph being colored is best.
   Our graph to color is $\bar{G}$.
   Degree in $\bar{G}(u) = (N-1) - \text{Degree}_G(u)$.
   So decreasing degree in $\bar{G}$ corresponds to increasing degree in $G$.
   So, sorting vertices by degree in $G$ **ascending** seems like the safer bet for coloring $\bar{G}$.
   
   Let's check example:
   5 vertices. Edges: (1,2), (2,3), (3,4), (4,5), (5,1), (2,5).
   Degrees in G:
   1: 2 (2, 5)
   2: 3 (1, 3, 5)
   3: 2 (2, 4)
   4: 2 (3, 5) -> Wait (3,4), (4,5). Yes 2.
   5: 3 (4, 1, 2)
   
   Sorted ascending: 1(2), 3(2), 4(2), 2(3), 5(3).
   Or 3, 4, 1...
   Let's try logic again.
   Actually, there is a specific algorithm often used for Maximal Clique Enumeration which is Bron-Kerbosch, but that finds cliques, not covers efficiently.
   But since we just need a valid solution, not exact minimal, greedy with a good heuristic is expected.
   
   Another strong heuristic for Clique Cover is based on Maximum Independent Set on $\bar{G}$? No, that's NP-hard.
   
   Let's go with the Bitset approach.
   We maintain a list of cliques defined by their vertex indices or masks.
   Actually, keeping track of the "open" status for every pair might be heavy.
   With $N=500$, $O(N^2)$ bitset ops is perfect.
   
   Data structure:
   `vector<int> adj[N+1]` : for quick lookup or bitset `adj[N+1]`. `std::bitset<501> adj[501]`.
   Initialize `id[N+1]` array to 0.
   
   Processing:
   Order vertices $p_1, p_2, \dots, p_N$.
   For each $u$ in $p$:
     Try to find a clique index $k$ such that $u$ can be added to clique $k$.
     To verify: for all $v$ currently in clique $k$, $is_connected(u, v)$ must be true.
     Condition: `(adj[u] & mask[k]) == mask[k]`. (Here mask[k] stores bits set for members of clique k).
     If found, update `mask[k] |= (1<<u)`, `id[u] = k`. Break.
     Else, create new clique $k$, set `mask[k] = (1<<u)`, `id[u] = k`.
     
   Optimization: Which clique to try first?
   To minimize cliques created, we should try to extend existing cliques.
   If there are multiple existing cliques that accept $u$, picking the one with largest mask size (most restrictive?) or smallest?
   Smallest? Larger cliques are harder to extend (need to be connected to more nodes).
   But actually, we just need to reduce the count. Any valid assignment counts as 1.
   However, filling a small clique completely might leave larger room elsewhere?
   Maybe try to fit $u$ into the clique with the most existing members?
   Intuitively, fitting into larger cliques is harder, so prioritize them? Or maybe fitting into the clique where $u$ is most constrained?
   Given the constraints and problem type, simply iterating $k$ from 1 to CurrentMaxID and taking the first valid one is simplest and usually fast enough for random/sparse graphs. For dense graphs, the loop will rarely run full length.
   Wait, if we process by degree, maybe we should refine.
   Sorting vertices by degree (ascending) in $G$ (so descending in $\bar{G}$) is standard for coloring $\bar{G}$.
   
   Is there any case where $N=500$ causes TLE?
   Operations per vertex: Loop over existing cliques (up to $N$). Inside loop, Bitset check (cost $N/64$).
   Total ops: $\sum_{k=1}^N (\text{num\_cliques}) \times (N/64)$.
   Num cliques can be up to $N$. So worst case $N \cdot N \cdot (N/64) = 1.25 \times 10^8 / 64 \approx 2 \times 10^6$. Very fast.
   Even without bitsets, raw iteration:
   Worst case $N^3$ loop checks. $500^3 = 1.25 \times 10^8$. Feasible in 2.0s (usually ~1e8 ops/sec, often ~5e8 simple ops).
   Using bitsets is safer.
   
   Implementation details:
   Input reading: fast I/O.
   Adjacency matrix bool or bitset. Since $N \le 500$, bitset is easy.
   Mask vector: `vector<std::bitset<500>> masks;`
   Reserve size $N+1$.
   
   One detail: The output format requires printing ID for each vertex 1..N.
   So after computing IDs, print `id[1]`...`id[N]`.
   
   Let's double check the constraints.
   Memory limit 512MB. We use negligible memory.
   Time 2.0s.
   
   What about Multiple edges?
   Problem says: "Multiple edges may appear but should be treated as a single constraint."
   Reading input, just set `adj[u][v] = adj[v][u] = 1`.
   
   Corner cases:
   $N=2$.
   Disconnected components.
   Empty graph $M=0$ (though constraints say $M \ge 1$). If $M=0$, $K=N$. My algo: each vertex fails to match previous (since no edges) -> creates new cliques. Correct.
   Complete graph $M=N(N-1)/2$. My algo: first vertex starts clique. Subsequent vertices check clique 1. They connect to everyone in clique 1 (all previous vertices). They get added. All get same ID. $K=1$. Optimal. Correct.
   
   Sorting Strategy Check:
   Does sorting order affect $K$ significantly?
   For coloring $\bar{G}$, sorting by degree in $\bar{G}$ (descending) is best for First-Fit greedy.
   Degree in $\bar{G}$ is $(N-1) - \text{deg}_G$.
   So sorting by $\text{deg}_G$ ascending is correct.
   
   Let's write down the logic clearly.
   1. Parse Input. Store adjacencies in `bitset`. Also store degrees.
   2. Create permutation vector `P` of size $N$. Fill with $1..N$.
   3. Sort `P` based on degrees of original vertices. Ascending order of degree in $G$.
      Comparator: if degrees differ, smaller degree comes first. If equal, arbitrary (e.g., smaller index).
   4. `vector<int> ans(N + 1)` storing result IDs.
   5. `vector<bitset<500>> cliques_masks`. Keep track of number of cliques `cnt = 0`.
   6. Iterate `u` in `P`:
      Found = false.
      Iterate `k` from 0 to `cnt - 1`:
         // Check if u compatible with clique k
         // i.e. u must be adjacent to all nodes already in clique k
         // `cliques_masks[k]` has 1s at positions of members of clique k.
         // `adj[u]` has 1s at neighbors of u.
         // Intersection must be `cliques_masks[k]`.
         if `((cliques_masks[k] ^ adj[u]) & cliques_masks[k]).none()`?
         Wait, logical XOR isn't quite right for subset check.
         Subset check: `(cliques_masks[k] & adj[u]) == cliques_masks[k]`.
         If true:
             ans[u_original_idx] = k + 1
             cliques_masks[k].set(u_original_idx)
             Found = true
             Break
      if !Found:
          cnt++
          ans[u_original_idx] = cnt
          cliques_masks.push_back(bitset)
          cliques_masks[cnt-1].set(u_original_idx)
          
   Wait, indices mapping. `ans` needs to correspond to vertex 1..N.
   `u` in iteration is index in `P`. `P[iter]` gives vertex index.
   
   Optimization for checking compatibility:
   With `bitset<500>`, intersection check is fast.
   However, creating a new `bitset` inside loop might be costly? No, copying small struct is fast. But we modify it.
   `cliques_masks` stores state.
   
   Complexity:
   Outer loop N times.
   Inner loop K times (up to N).
   Bitset op cost N/64.
   Total $N^3/64$. With $N=500$, approx $2 \times 10^6$ ops. Very safe.
   
   Wait, is sorting by degree in $G$ ascending really the way to go?
   Let's trace Example.
   Edges: (1,2), (2,3), (3,4), (4,5), (5,1), (2,5).
   Deg G:
   1: 2 (2,5)
   2: 3 (1,3,5)
   3: 2 (2,4)
   4: 2 (3,5)
   5: 3 (2,4,1)
   
   Wait, my manual trace of deg earlier was slightly off?
   1: 2-1, 5-1. Yes 2.
   2: 1-2, 3-2, 5-2. Yes 3.
   3: 2-3, 4-3. Yes 2.
   4: 3-4, 5-4. Yes 2.
   5: 4-5, 1-5, 2-5. Yes 3.
   
   Degrees:
   1: 2
   2: 3
   3: 2
   4: 2
   5: 3
   
   Sorted (Ascending): {1, 3, 4}, {2, 5}. Tie breaking by index.
   Order: 1, 3, 4, 2, 5.
   
   Processing:
   1. Process 1. Clique 1: {1}. Ans[1]=1.
   2. Process 3. Clique 1 ({1}). Edge (1,3)? No. Need new. Clique 2: {3}. Ans[3]=2.
   3. Process 4.
      Check Clique 1 ({1}). Edge (1,4)? No.
      Check Clique 2 ({3}). Edge (3,4)? Yes. OK.
      Update Clique 2: {3, 4}. Ans[4]=2.
   4. Process 2.
      Check Clique 1 ({1}). Edge (1,2)? Yes. OK.
      Add 2 to Clique 1: {1, 2}. Ans[2]=1.
   5. Process 5.
      Check Clique 1 ({1, 2}). Edges needed: (1,5) and (2,5). Both exist?
      (1,5) yes. (2,5) yes.
      Add 5 to Clique 1: {1, 2, 5}. Ans[5]=1.
   
   Final Ans:
   1: 1
   2: 1
   3: 2
   4: 2
   5: 1
   Output lines:
   1
   1
   2
   2
   1
   This matches example output perfectly!
   
   The heuristic seems robust.
   Just ensure `adj` includes self-loops? No, problem says $u \neq v$.
   Self connection check? The clique definition is pairwise distinct vertices.
   When adding $u$ to clique $C$, $u$ must be connected to ALL $v \in C$.
   $C$ initially contains just $u$, so loop is empty?
   Logic holds. `cliques_masks[k]` tracks *previous* members.
   
   One subtle optimization:
   In inner loop, iterate over cliques. Which order?
   If I have cliques of sizes 5 and 2, and $u$ can join both.
   Joining size 5 keeps K constant. Joining size 2 keeps K constant.
   Is it beneficial to join the largest clique?
   Often yes, to keep cliques "full" or utilize capacity? Not necessarily for count.
   However, if we greedily fill small cliques, they stay small.
   If we fill large cliques, they might become too "exclusive" later?
   Actually, usually in coloring, we prefer to color highly constrained nodes (those with many conflicts) early.
   In our greedy logic, processing order is determined by initial sort.
   Once processed, we just fit them into the *first* available bucket.
   Iterating buckets $k=1..cnt$. Small $k$ filled earlier.
   Should we prioritize buckets that are most "compatible"?
   Checking compatibility is fast. The order of bucket scan determines behavior.
   Scanning by size? Largest compatible clique first?
   Or just scan linearly?
   Linear scan is simplest. Given $N=500$, optimization is probably unnecessary unless specific adversarial cases exist.
   However, scanning from largest clique to smallest *might* help avoid fragmenting cliques unnecessarily?
   Wait, if we scan $k=1..cnt$, we prefer older cliques (likely larger or at least established).
   If we process vertices with ascending degree in G (high connectivity in G, low in $\bar{G}$), these are "easy" to put into cliques.
   
   Actually, I will stick to the example logic which worked. It scans $k$ from 0 upwards.
   One thing: `bitset` indexing. `std::bitset` uses 0-based indices. Vertex indices are 1-based.
   Map vertex $v$ to bit `v-1`.
   
   Constraints:
   $N \le 500$.
   `std::bitset<501>` is fine.
   
   Include `<iostream>`, `<vector>`, `<numeric>`, `<algorithm>`, `<bitset>`.
   
   Code structure:
   ```cpp
   #include <bits/stdc++.h>
   using namespace std;
   
   int main() {
       ios_base::sync_with_stdio(false);
       cin.tie(NULL);
       
       int N, M;
       if (!(cin >> N >> M)) return 0;
       
       vector<vector<bool>> adj(N + 1, vector<bool>(N + 1, false));
       vector<int> deg(N + 1, 0);
       
       for(int i=0; i<M; ++i) {
           int u, v;
           cin >> u >> v;
           adj[u][v] = adj[v][u] = true;
           deg[u]++;
           deg[v]++;
       }
       
       vector<int> p(N);
       iota(p.begin(), p.end(), 1);
       sort(p.begin(), p.end(), [&](int a, int b){
           if(deg[a] != deg[b]) return deg[a] < deg[b];
           return a < b; 
       });
       
       vector<int> result_id(N + 1, 0);
       // Use vector of bitsets for masks. 
       // Resize dynamically or push_back.
       // Max cliques <= N.
       // Storing full N-bitset per clique.
       // N*N words. 500*500/64 bytes approx 4KB. Negligible.
       vector<bitset<501>> clique_masks;
       reserve not needed for dynamic vector but good practice.
       
       for(int u : p) {
           int k = 0; // index of clique (0-based)
           // Try to fit into existing cliques
           while(k < clique_masks.size()) {
               // check compatibility
               // clique_masks[k] represents set of vertices already in this clique
               // u must be connected to ALL of them
               // intersection of adj[u] and clique_masks[k] must equal clique_masks[k]
               
               // Using temporary bitset?
               // bitwise_and
               if ((adj[u] & clique_masks[k]).count() == clique_masks[k].count()) {
                   // Valid
                   result_id[u] = k + 1; // 1-based ID
                   clique_masks[k].set(u); // Update mask
                   break; // Move to next vertex
               }
               k++;
           }
           
           if (!result_id[u]) { // Didn't fit anywhere (should handle initialization)
                // Wait, if loop finishes without setting result_id, means new clique
                result_id[u] = k + 1;
                clique_masks.resize(clique_masks.size() + 1);
                clique_masks.back().reset();
                clique_masks.back().set(u);
                // Note: result_id initialized to 0 above.
                // Actually logic needs to be clearer.
           }
       }
       
       // Correction on loop logic:
       // If we initialize `result_id` to 0, and inside loop we set it, then check if 0?
       // But `k` loop variable goes out of scope? No, local var.
       // Better structure:
       /*
       int num_cliques = 0;
       for(int u : p) {
           int assigned = 0;
           bool found = false;
           for(int k=0; k<num_cliques; ++k) {
               if ((adj[u] & clique_masks[k]).test() == ... ) Wait `operator&` returns bitset.
               // comparison of bitset equality with another bitset
               // `bitset` doesn't support efficient compare? Yes it has `operator==`.
               // `(adj[u] & clique_masks[k]) == clique_masks[k]`
               if ((adj[u] & clique_masks[k]) == clique_masks[k]) {
                   result_id[u] = k + 1;
                   clique_masks[k].set(u);
                   found = true;
                   break;
               }
           }
           if (!found) {
               assigned = num_cliques + 1;
               result_id[u] = assigned;
               clique_masks.push_back(bitset<501>());
               clique_masks.back().set(u);
               num_cliques++;
           }
       }
       */
       
       // Print result_id[1..N]
       
   ```
   Wait, `(adj[u] & clique_masks[k]) == clique_masks[k]` checks if intersection equals clique mask.
   Yes, if `A & B == B`, then B is a subset of A. Meaning every bit in `B` is set in `A`.
   This correctly verifies $u$ is connected to all vertices in clique $k$.
   
   One subtle issue: `adj` is `vector<vector<bool>>`. Bitwise operations require `std::bitset`.
   I should convert `adj` to `bitset<501>[]` or `vector<bitset<501>>`.
   Given $N \le 500$, `bitset<501>` is fixed size template.
   So `bitset<501> adj_row[N+1]`.
   
   This makes the code cleaner.
   
   Wait, `vector<bitset<501>>` initialization of masks.
   Default constructor initializes all bits to 0.
   `push_back` works.
   
   One potential issue:
   The bitset size should be strictly covering N. 501 covers 0..500.
   
   Double check: `std::bitset` operator `&` returns a temporary `bitset`. Comparison `==` works.
   
   Time complexity again:
   `vector<bitset<501>>` copy/move overhead.
   Inside loop:
   `if ((adj[u] & clique_masks[k]) == clique_masks[k])`
   Creates temp bitset, performs comparison.
   Cost is small constant (bit manipulation).
   Number of comparisons per vertex $\le$ number of cliques.
   Max number of cliques $\le N$.
   Total $O(N^2 \cdot \frac{N}{64})$.
   
   Space: $N \cdot \frac{N}{64}$ bytes $\approx 500 \cdot 8 = 4000$ bytes. Minimal.
   
   Wait, the prompt says "Output exactly N lines".
   And "The i-th line must contain one integer: id[i]".
   Where `id` corresponds to vertices $1 \dots N$.
   
   Let's refine the implementation plan.
   
```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <bitset>

using namespace std;

const int MAXN = 505;

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    // Use bitset for adjacency to leverage bitwise operations efficiently
    vector<bitset<MAXN>> adj(N + 1);
    vector<int> deg(N + 1, 0);

    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        // Ensure unique edges handling implicitly done by setting true
        adj[u][v] = 1;
        adj[v][u] = 1;
        deg[u]++;
        deg[v]++;
    }

    // Prepare vertices for processing order
    // Heuristic: Sort vertices by degree in G (ascending)
    // Equivalent to sorting by degree in complement graph (descending)
    vector<int> p(N);
    iota(p.begin(), p.end(), 1); // 1 to N
    sort(p.begin(), p.end(), [&](int a, int b) {
        if (deg[a] != deg[b]) return deg[a] < deg[b];
        return a < b;
    });

    // id[i] stores the clique ID for vertex i
    // Initially 0.
    vector<int> id(N + 1, 0);
    
    // Stores a bitmask representing the vertices currently in each clique
    vector<bitset<MAXN>> cliques_masks;

    for (int u : p) {
        int current_clique_index = -1;
        
        // Try to fit u into existing cliques
        // Iterate backwards or forwards? 
        // Let's iterate forwards 0..size-1.
        // Optimization: Check if u compatible with existing clique
        // Condition: (adj[u] & mask) == mask
        
        for (int k = 0; k < (int)cliques_masks.size(); ++k) {
            if ((adj[u] & cliques_masks[k]) == cliques_masks[k]) {
                current_clique_index = k;
                break; 
            }
        }

        if (current_clique_index != -1) {
            id[u] = current_clique_index + 1;
            cliques_masks[current_clique_index].set(u);
        } else {
            // Start new clique
            int new_k = cliques_masks.size();
            cliques_masks.push_back(bitset<MAXN>());
            id[u] = new_k + 1;
            cliques_masks.back().set(u);
        }
    }

    // Output results
    for (int i = 1; i <= N; ++i) {
        cout << id[i] << "\n";
    }

    return 0;
}
```
   
   Wait, is `ios_base::sync_with_stdio(false);` necessary? Probably good practice.
   
   Re-check "Multiple edges may appear".
   Input `u v` can repeat. `adj[u][v] = 1` handles duplicates naturally (idempotent).
   
   Constraints check:
   $N=500$.
   `bitset<505>` is fine.
   `deg` vector size `N+1`.
   `id` vector size `N+1`.
   `p` size `N`.
   Everything looks correct.
   
   Wait, what if graph has isolated vertices?
   `deg`=0. They come early in sort order.
   Processed: New clique created. Correct.
   
   One minor optimization:
   If `cliques_masks` grows large, iterating through all of them takes time.
   Can we optimize finding a compatible clique?
   Maybe maintain a list of cliques sorted by size? Or something?
   But given the constraints and nature of the problem (heuristic), iterating existing cliques is $O(K)$ where $K \le N$.
   If average $K$ is small, it's fast.
   Even if $K \approx N/2$, total complexity is bounded by $\sum_{u} K_u \cdot N/64$.
   Worst case: Star graph in $\bar{G}$ (Empty star in $G$?).
   If $\bar{G}$ is a matching (many small cliques), $K \approx N/2$.
   Checking takes $O(N)$. Total $O(N^2)$.
   Wait, bitset op is $O(N/64)$.
   Checking $K$ cliques: $O(K \cdot N/64)$.
   Summing over $N$ vertices: $\sum K_u \cdot N/64 \approx N^2 \cdot N/64$.
   Wait, if $K$ reaches $N$, loop runs $N$ times.
   Total ops $\approx N^2 \cdot N/64$.
   Wait, the number of cliques $K$ is at most $N$.
   Average number of cliques might be much less.
   Even in worst case where we have $N$ cliques, we process each vertex and check against all $k$ cliques?
   No, if we have $N$ cliques (isolated vertices in G), then for vertex $u$, we try to match existing cliques.
   For vertex 1: 0 cliques.
   Vertex 2: matches nothing (no edges to 1? wait).
   In $G$ disconnected: vertex 1 gets clique 1. Vertex 2 (not connected to 1) checks clique 1. Not compatible. Starts clique 2.
   So we scan all previous cliques.
   Scan takes $O(\text{prev\_cliques} \cdot N/64)$.
   If prev cliques $\approx i$, cost is $i \cdot N/64$.
   Total cost $\sum_{i=1}^N i \cdot N/64 = \frac{N^3}{2 \cdot 64}$.
   $500^3 \approx 1.25 \times 10^8$. Divided by 64 $\approx 2 \times 10^6$.
   This is extremely fast. 0.01s approx.
   So no further optimization needed.
   
   One catch: `vector<bitset<MAXN>> cliques_masks;` inside loop or global?
   Local is fine.
   `MAXN = 505`.
   
   Let's check headers.
   `#include <iostream>`
   `#include <vector>`
   `#include <numeric>` for `iota`
   `#include <algorithm>` for `sort`
   `#include <bitset>`
   
   Final sanity check on logic.
   Example trace again with code logic.
   1->id 1, mask {1}
   3->id 2, mask {3} (check 1: no edge)
   4->id 2, mask {3,4} (check 1: no edge. check 2: edge (3,4) ok)
   2->id 1, mask {1,2} (check 1: edge (1,2) ok)
   5->id 1, mask {1,2,5} (check 1: (1,5)yes, (2,5)yes ok)
   
   Result:
   1:1, 2:1, 3:2, 4:2, 5:1
   Matches example.
   
   Another case: Cycle C3: 1-2, 2-3, 3-1.
   Deg all 2. Order 1, 2, 3.
   1: id 1, mask {1}
   2: check 1: (1,2) yes. id 1, mask {1,2}
   3: check 1: need (1,3) and (2,3). Both yes. id 1, mask {1,2,3}
   Result: 1 1 1. K=1. Correct (C3 is a clique).
   
   Case: Path P4: 1-2-3-4.
   Edges: (1,2), (2,3), (3,4).
   Deg: 1:1, 2:2, 3:2, 4:1.
   Sort Asc: 1, 4, 2, 3 (indices tie break). Or 4, 1, 2, 3.
   Let's assume order 1, 4, 2, 3.
   1: id 1, mask {1}
   4: check 1: no (1,4). id 2, mask {4}
   2: check 1: (1,2) yes. id 1, mask {1,2}
   3: check 1: need (1,3) NO. check 2: need (4,3) NO.
      Wait, (4,3) yes. (1,2) is in clique 1.
      Check 1 (mask {1,2}): need (1,3) & (2,3). (2,3) yes, (1,3) no. Fail.
      Check 2 (mask {4}): need (4,3). Yes. Match.
      id 2. mask {4, 3}.
   Ids: 1:1, 4:2, 2:1, 3:2.
   Cliques: {1,2}, {4,3}.
   Valid? {1,2} edge. {3,4} edge.
   Number of cliques K=2. Optimal for P4 is 2 ({1,2}, {3,4} or {1},{2,3,4}? No {2,3,4} is not clique. {2,3} is edge).
   Min clique cover P4: 2 cliques.
   
   Looks solid.

   Final check on "positive integer clique ID".
   My code outputs `k+1` where `k` is 0-indexed. So min value is 1. Correct.
   
   One corner case: `N` up to 500. `MAXN` 505 is sufficient.
   
   Memory Limit 512MB. Vector overhead small.
   Time Limit 2.0s. $O(N^3/64)$ is trivial.
   
   Wait, is there any case where we prefer `deg` descending?
   Some sources suggest DSatur is superior, but simple degree heuristic is widely accepted for CP unless "hard" inputs specifically targeting greedy are known. Given the problem asks for *a* solution maximizing score, and $N$ is small, any valid solution with reasonable $K$ is expected.
   However, note that if the graph is a complement of a disjoint union of cliques, then $G$ is a disjoint union of independent sets.
   Then any clique in $G$ has size 1. $K=N$.
   If $G$ is complete, $K=1$.
   My code produces $K=N$ for independent sets (sorted by degree doesn't matter, each needs new clique).
   For complete graph, it produces $K=1$.
   
   The solution seems robust.

```cpp
/*
 * Solution for Clique Cover Challenge
 * Approach:
 * 1. Construct the graph G.
 * 2. The problem asks for a clique cover in G, equivalent to coloring the complement graph G_bar.
 * 3. We use a greedy strategy for coloring: iterate through vertices and assign the smallest valid color (clique ID).
 *    In G terms: assign the smallest clique ID such that the vertex is connected to all existing vertices with that ID.
 * 4. To improve the greedy solution, we sort vertices based on a heuristic. 
 *    Ordering by degree in G (ascending) corresponds to ordering by degree in G_bar (descending), 
 *    which is a standard heuristic for graph coloring problems.
 * 5. For efficiency with N <= 500, we use bitsets to represent the membership of vertices in each clique.
 *    Checking compatibility becomes a bitwise AND and equality check.
 */

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <bitset>

using namespace std;

// Define maximum N plus a small buffer
const int MAXN = 505;

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    // Adjacency matrix represented as bitsets for efficient bit manipulation
    vector<bitset<MAXN>> adj(N + 1);
    // Degree array for heuristics
    vector<int> deg(N + 1, 0);

    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        // Mark adjacency
        adj[u][v] = 1;
        adj[v][u] = 1;
        // Count degrees
        deg[u]++;
        deg[v]++;
    }

    // Prepare vertices for processing
    vector<int> p(N);
    iota(p.begin(), p.end(), 1); // Indices 1 to N
    
    // Sort vertices based on degree in G (ascending).
    // This heuristic helps vertices with fewer connections (and thus more constrained in G_bar)
    // to be processed early, potentially leading to fewer cliques overall.
    sort(p.begin(), p.end(), [&](int a, int b) {
        if (deg[a] != deg[b]) return deg[a] < deg[b];
        return a < b; // Tie-breaker
    });

    // id[u] stores the clique ID assigned to vertex u
    vector<int> id(N + 1, 0);
    
    // cliques_masks[k] holds the bitset of vertices belonging to clique k+1 (0-indexed internally)
    // A vertex u can be added to clique k if it is connected to ALL vertices currently in clique k.
    vector<bitset<MAXN>> cliques_masks;

    for (int u : p) {
        int assigned_clique_idx = -1;
        
        // Try to fit u into an existing clique
        for (int k = 0; k < (int)cliques_masks.size(); ++k) {
            // Check if u is compatible with clique k
            // Compatibility condition: u must share an edge with every vertex currently in clique k.
            // This means (adj[u] & clique_mask[k]) must contain all bits set in clique_mask[k].
            // In bitwise logic: (adj[u] & mask) == mask
            if ((adj[u] & cliques_masks[k]) == cliques_masks[k]) {
                assigned_clique_idx = k;
                break; 
            }
        }

        if (assigned_clique_idx != -1) {
            // Assign existing clique
            id[u] = assigned_clique_idx + 1;
            cliques_masks[assigned_clique_idx].set(u);
        } else {
            // Create new clique
            int new_idx = cliques_masks.size();
            cliques_masks.push_back(bitset<MAXN>());
            id[u] = new_idx + 1;
            cliques_masks.back().set(u);
        }
    }

    // Output results
    for (int i = 1; i <= N; ++i) {
        cout << id[i] << "\n";
    }

    return 0;
}
```
   
   Wait, one very small detail. The bitset comparison `operator==` compares the entire bitset (all 505 bits).
   Since we only care about bits 0..N, the rest should be 0.
   `cliques_masks` starts with all zeros. `set(u)` sets one bit.
   `adj[u]` is computed based on input.
   Bits > N are always 0 in both `adj[u]` and `cliques_masks[k]`.
   So `==` check is safe regardless of N value, provided MAXN is fixed.
   It will return true/false correctly based on the relevant bits.
   
   Is there any case where `adj[u]` has garbage? No, constructed from input.
   So this is safe.
   
   Double check constraints again. `2 <= N <= 500`.
   The logic holds for small N.
   
   One edge case: `M` edges.
   If duplicate edges given, handled by `adj[u][v] = 1`.
   
   The solution seems complete and optimized enough.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <bitset>

using namespace std;

const int MAXN = 505;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    // Adjacency matrix using bitset for efficient operations
    vector<bitset<MAXN>> adj(N + 1);
    vector<int> deg(N + 1, 0);

    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        // Input guarantees u != v
        adj[u][v] = adj[v][u] = 1;
        deg[u]++;
        deg[v]++;
    }

    // List of vertices to process
    vector<int> p(N);
    iota(p.begin(), p.end(), 1); 

    // Sort vertices by degree in G (ascending).
    // This order corresponds to sorting by degree in complement graph (descending),
    // which is a good heuristic for minimizing the chromatic number (max clique cover size).
    sort(p.begin(), p.end(), [&](int a, int b) {
        if (deg[a] != deg[b]) return deg[a] < deg[b];
        return a < b;
    });

    // id[u] will store the clique identifier for vertex u
    vector<int> id(N + 1, 0);
    
    // Each element in cliques_masks is a bitset representing the set of vertices in a specific clique
    vector<bitset<MAXN>> cliques_masks;

    for (int u : p) {
        int assigned_clique_idx = -1;
        
        // Iterate over existing cliques to see if u can be added to any of them
        // Condition: u must be connected to ALL members of the clique
        for (int k = 0; k < (int)cliques_masks.size(); ++k) {
            // Bitwise AND checks common neighbors, compare with mask to check subset property
            if ((adj[u] & cliques_masks[k]) == cliques_masks[k]) {
                assigned_clique_idx = k;
                break; 
            }
        }

        if (assigned_clique_idx != -1) {
            id[u] = assigned_clique_idx + 1;
            cliques_masks[assigned_clique_idx].set(u);
        } else {
            // No compatible clique found, start a new one
            int new_idx = (int)cliques_masks.size();
            cliques_masks.push_back(bitset<MAXN>());
            id[u] = new_idx + 1;
            cliques_masks.back().set(u);
        }
    }

    // Output the clique IDs for vertices 1 to N
    for (int i = 1; i <= N; ++i) {
        cout << id[i] << "\n";
    }

    return 0;
}
```
One last check. The problem statement says:
"The execution time and memory limit are also stated in the statement so be aware of the complexity of the program."
My complexity analysis: Time $O(N^3 / W)$ where $W=64$, Space $O(N^2 / W)$.
$N=500$, $N^3 = 1.25 \times 10^8$, divided by 64 is $2 \times 10^6$ basic operations.
Very well within limits.

Memory: `vector<bitset<505>>` takes $500 \times 505$ bits $\approx 32$ KB. Very low.

The constraints: "2 <= N". Handled.
"1 <= M". Handled.

Final confirmation on output: "Output exactly N lines. The i-th line must contain one integer: id[i]".
My code loops $i=1..N$ and prints `id[i]`. Correct.

I'm confident in this solution.

Wait, I should check if including `<iomanip>` or anything is needed? No.
Using `vector<bitset>` inside `main` is fine.

Double check include guards? No need for CP.
Just standard header.

One small thing: `deg` calculation.
If multiple edges exist between $u$ and $v$, `deg[u]++` increments twice.
Does this affect the heuristic?
Sorting by degree works even with incorrect degrees due to multi-edges, as long as the relative order is somewhat consistent.
Since "multiple edges may appear", `deg` would technically count multiplicity.
If I treat it as simple graph, I should ignore duplicates in `deg` or just rely on `adj` being true.
Strictly speaking, if input has parallel edges, `deg` increases.
But `adj[u][v]` stays 1.
Is degree defined on multigraph or simple graph?
Problem says "undirected graph G=(V,E)". "Multiple edges... treated as a single constraint".
Usually implies simple graph properties for the task, but the input format allows multigraph input description.
Usually for coloring heuristics, simple graph degree is preferred.
I should probably read edges, build `adj`, and compute degree based on `adj`?
Or just read edges and increment. If duplicate edges, `deg` increases.
If I use `deg` from raw input count, I might penalize a vertex with parallel edges.
Better to use `deg` based on unique neighbors.
To fix this:
After reading all edges, calculate `deg` from `adj`.
Or during read, only increment if `adj[u][v]` was false?
Wait, if input is:
1 2
1 2
Then it's effectively 1 edge.
If I do `deg[1]++` twice, it thinks degree is 2.
Correct degree is 1.
Since `N` is small, recomputing degree is cheap.
I'll update `deg` logic to be accurate.
Initialize `deg` to 0.
Read $u, v$. Set `adj`.
Then iterate $i=1..N$ and `deg[i] = adj[i].count()`.
This ensures simple graph degree is used for heuristic, which is semantically more correct for the problem.
Although for heuristic, slight inaccuracies might not break the solution given the relaxed constraints ($N=500$).
But correctness-wise, counting unique neighbors is better.

Updated loop for input reading + degree computation:
```cpp
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u][v] = adj[v][u] = 1;
    }
    for(int i=1; i<=N; ++i) deg[i] = adj[i].count();
```
This is cleaner and safer.
Complexity: $O(M)$ to read, $O(N^2/64)$ to count degrees. Still negligible.

Okay, final version ready.

One tiny thing: The problem says "score = K*/K * 100". It mentions K* is known to judge.
This implies minimizing K is important.
Is there any risk with sorting order?
Some heuristics for Clique Cover (equivalent to Chromatic Number of Complement) involve DSatur or sorting by Degree in Complement.
My sort order: Degree in G ascending <=> Degree in $\bar{G}$ descending. This is the standard "DSatur" base or "Degree" heuristic for coloring.
Given time limits, I cannot implement DSatur which involves calculating saturation levels dynamically.
Static degree sorting is the most appropriate choice here.

One more thought: Does `std::bitset::count()` work efficiently? Yes, it's usually optimized (popcount instruction).
Even if not, it iterates over words, very fast.
So recomputing degree is cheap.

Wait, I need to make sure I didn't miss any headers.
`vector`, `iostream`, `numeric`, `algorithm`, `bitset`. All good.

One syntax thing: `vector<bitset<MAXN>> cliques_masks` default initialization.
Inside `else` block, `cliques_masks.push_back(bitset<MAXN>())`.
`bitset` default ctor creates all zero bits.
`set(u)` sets bit `u`.
Since vertices are 1-based, I must map `u` to `u-1` when accessing bit?
OR: Use 1-based bitsetting and ignore bit 0.
`bitset` index 0 to MAXN-1.
Vertices are 1 to N.
I access `cliques_masks[k].set(u)`. This uses index `u`.
Valid since `MAXN = 505` > 500.
Same for `adj[u][v] = 1`. Index `u` valid.
So 1-based indexing for bits works fine.
`adj` row `u` has bit `v` set.
Comparison `(adj[u] & mask) == mask`.
Mask has bits corresponding to vertices in clique.
`adj[u]` has bits corresponding to neighbors of `u`.
Intersection must equal mask.
Yes, this works perfectly with 1-based indexing for bits.

Ready to generate response.

One detail: `cout << "\n"` vs `endl`. `\n` is faster. Good.

Final Check:
Input:
5 6
1 2
...
My code reads M lines.
Correct.

Wait, I should confirm the bitset indexing again.
`bitset<N>::set(pos)`. `pos` must be in `[0, N-1]`.
If I use `set(u)` where `u` is `1..N`.
Size is `MAXN = 505`.
Indices `0..504` available.
If `N=500`, `u=500` is valid.
So `set(u)` is correct.

Wait, `adj` is declared `vector<bitset<MAXN>> adj(N + 1);`.
`adj[u][v] = 1`. `u, v` in `1..N`.
Access `adj[500][500]`.
Vector size 501. Index 500 is valid (last element).
Index 0 is unused.
Everything aligns.

Wait, `deg` calculation using `adj[i].count()`.
`adj[i]` is the row for vertex `i`.
It contains bits `j` where `adj[i][j] == 1`.
`count()` returns number of set bits.
Correct.

The code looks solid.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <bitset>

using namespace std;

const int MAXN = 505;

int main() {
    // Optimize I/O operations for performance
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    // Use bitset for adjacency to leverage bitwise operations efficiently.
    // Size is MAXN to accommodate 1-based vertex indices up to 500.
    vector<bitset<MAXN>> adj(N + 1);
    
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u][v] = adj[v][u] = 1;
    }

    // Calculate degree for each vertex based on the adjacency bitset.
    // This represents the number of edges incident to each vertex in G.
    vector<int> deg(N + 1, 0);
    for (int i = 1; i <= N; ++i) {
        deg[i] = adj[i].count();
    }

    // Create a list of vertices 1 to N
    vector<int> p(N);
    iota(p.begin(), p.end(), 1);

    // Sort vertices by degree in G (ascending).
    // This heuristic maps to sorting by degree in the complement graph (descending).
    // This order typically yields a better coloring of the complement graph (clique cover of G).
    sort(p.begin(), p.end(), [&](int a, int b) {
        if (deg[a] != deg[b]) return deg[a] < deg[b];
        return a < b;
    });

    // id[u] stores the clique ID assigned to vertex u
    vector<int> id(N + 1, 0);
    
    // Stores a bitmask for each active clique.
    // cliques_masks[k] has the j-th bit set if vertex j is in the k-th clique.
    vector<bitset<MAXN>> cliques_masks;

    for (int u : p) {
        int assigned_clique_idx = -1;
        
        // Try to fit u into an existing clique
        for (int k = 0; k < (int)cliques_masks.size(); ++k) {
            // Check if u is connected to ALL members of clique k.
            // This requires (adj[u] & mask) == mask.
            if ((adj[u] & cliques_masks[k]) == cliques_masks[k]) {
                assigned_clique_idx = k;
                break; 
            }
        }

        if (assigned_clique_idx != -1) {
            id[u] = assigned_clique_idx + 1;
            cliques_masks[assigned_clique_idx].set(u);
        } else {
            // If no existing clique accepts u, create a new one.
            int new_idx = (int)cliques_masks.size();
            cliques_masks.push_back(bitset<MAXN>());
            id[u] = new_idx + 1;
            cliques_masks.back().set(u);
        }
    }

    // Output results for vertices 1 to N
    for (int i = 1; i <= N; ++i) {
        cout << id[i] << "\n";
    }

    return 0;
}
```
This is the final code. Checked all aspects.
</think>

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <bitset>

using namespace std;

const int MAXN = 505;

int main() {
    // Optimize I/O operations for performance
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    // Use bitset for adjacency to leverage bitwise operations efficiently.
    // Size is MAXN to accommodate 1-based vertex indices up to 500.
    vector<bitset<MAXN>> adj(N + 1);
    
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u][v] = adj[v][u] = 1;
    }

    // Calculate degree for each vertex based on the adjacency bitset.
    // This represents the number of edges incident to each vertex in G.
    // We use this for the sorting heuristic.
    vector<int> deg(N + 1, 0);
    for (int i = 1; i <= N; ++i) {
        deg[i] = adj[i].count();
    }

    // Create a list of vertices 1 to N
    vector<int> p(N);
    iota(p.begin(), p.end(), 1);

    // Sort vertices by degree in G (ascending).
    // This heuristic maps to sorting by degree in the complement graph (descending).
    // In the context of the complement graph coloring problem (which is equivalent to clique cover),
    // processing vertices with higher degree in the target graph (complement of G) first is optimal for First-Fit greedy.
    // Higher degree in complement means lower degree in G.
    sort(p.begin(), p.end(), [&](int a, int b) {
        if (deg[a] != deg[b]) return deg[a] < deg[b];
        return a < b;
    });

    // id[u] stores the clique ID assigned to vertex u
    vector<int> id(N + 1, 0);
    
    // Stores a bitmask for each active clique.
    // cliques_masks[k] has the j-th bit set if vertex j is in the k-th clique.
    // We use bitsets to quickly verify if a vertex is connected to all members of a clique.
    vector<bitset<MAXN>> cliques_masks;

    for (int u : p) {
        int assigned_clique_idx = -1;
        
        // Try to fit u into an existing clique
        // Iterate through existing cliques to find the first one compatible with u.
        for (int k = 0; k < (int)cliques_masks.size(); ++k) {
            // Check if u is connected to ALL members of clique k.
            // Compatibility condition:
            // u must be a neighbor of every vertex v where cliques_masks[k][v] is set.
            // In bitwise terms, (adj[u] & mask) must equal mask.
            if ((adj[u] & cliques_masks[k]) == cliques_masks[k]) {
                assigned_clique_idx = k;
                break; 
            }
        }

        if (assigned_clique_idx != -1) {
            id[u] = assigned_clique_idx + 1;
            cliques_masks[assigned_clique_idx].set(u);
        } else {
            // If no existing clique accepts u, create a new one.
            int new_idx = (int)cliques_masks.size();
            cliques_masks.push_back(bitset<MAXN>());
            id[u] = new_idx + 1;
            cliques_masks.back().set(u);
        }
    }

    // Output results for vertices 1 to N
    for (int i = 1; i <= N; ++i) {
        cout << id[i] << "\n";
    }

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=733)

````
The Clique Cover problem is NP-hard in general. Given the constraints and the nature of competitive programming, I need to implement a solution that:
1. Is valid (every group shares an ID forms a clique)
2. Uses as few cliques as possible

The most practical approach for this constraint range (N ≤ 500) is a greedy algorithm with backtracking:
- Greedy coloring on the complement graph (which corresponds to clique partitioning on G)
- Backtrack if a valid cover isn't found at the estimated minimum size

I'll use:
1. Adjacency matrix for O(1) edge lookup
2. Greedy coloring with maximum degree priority (color highest-degree uncolored vertices first)
3. Check each color class forms a clique before accepting it
4. Backtrack if the result exceeds our estimate of K*
5. Output all colors from 1 up to max used
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, m;
int adj[505][505];
vector<int> color(505);
int bestK = INT_MAX;

bool same_clique(int u, int v) {
    return adj[u][v] || adj[v][u];
}

void dfs(int u) {
    if (color[u]) return;
    for (int i = n - 1; i >= 0; --i) {
        if (!color[i] + 1 && dfs(i)) return;
    }
    for (int i = 0; i <= u; ++i) {
        if (color[i] == n) continue;
        bool can = true;
        for (int j = i + 1; j <= u; ++j) {
            if (color[j] != color[i]) {
                if (same_clique(i, j) != true) {
                    can = false; break;
                }
            }
        }
        if (can) color[i] = color[u];
    }
}

int main() {
    scanf("%d%d", &n, &m);
    memset(adj, 0, sizeof(adj));
    vector<pair<int,int>> edges;
    for (int i = 0; i < m; ++i) {
        int u, v;
        scanf("%d%d", &u, &v);
        adj[u][v] = adj[v][u] = 1;
    }

    int c = 0;
    memset(color, 0, sizeof(color));
    for (int i = n - 1; i >= 0; --i) {
        bool same = true;
        for (int j = 0; j < i; ++j) {
            if (color[j] != color[i] && !adj[i][j] && !adj[j][i]) same = false;
        }
        if (!same) color[i] = ++c;
    }

    int K = c + (c > 0 ? 0 : 1);
    printf("%d\n", (K == 0 ? 1 : K));
    for (int i = 1; i <= K; ++i) printf("1\n");
    
    fflush(stdout);
    return 0;
}
```
````
