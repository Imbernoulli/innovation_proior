# 题 183 — base均分 80.4 vs 直接SFT均分 18.1(掉点 62.2)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=2,得分 100.0,completion_tokens=17623)

````
The problem asks us to find an independent set of vertices in a given undirected graph. We want to maximize the size of this independent set, as the scoring function rewards solutions closer to the optimal Maximum Independent Set (MIS). Specifically, if $K$ is our found size and $K^*$ is the optimal, Score = $(N - K^*) / (N - K) * 100$. Since we don't know $K^*$, we need to do our best to approximate it. If we output an invalid solution (edges between selected vertices), the score is 0.

Finding the Maximum Independent Set (MIS) on general graphs is NP-hard. Given $N \le 10,000$, an exact algorithm won't work within the time limit for all cases. However, heuristic algorithms or approximation algorithms are acceptable here since it's described as a "heuristic optimization problem" and we just need to maximize $K$ relative to $K^*$. Wait, usually in competitive programming problems like this with hidden $K^*$, there might be specific structure or constraints that allow better approximations, or it might be a "challenge" where standard heuristics perform well enough on average test cases.

Let's analyze the constraints. $N=10,000$, $M=500,000$. Time limit 2.0s.
Since MIS is NP-hard, we can't solve it exactly for dense or complex graphs. However, we want to get close to $K^*$.
Note that the complement of an Independent Set is a Vertex Cover. Finding a minimum vertex cover is equivalent to finding a maximum independent set ($|VC| + |MIS| = N$). Approximating Minimum Vertex Cover is possible with a factor of 2 using a simple greedy strategy: pick both endpoints of any edge not covered by current selection? No, the standard greedy strategy for Vertex Cover picks vertices with highest degree? Or repeatedly removes edges and takes one vertex?
A 2-approximation for Vertex Cover works like this: Initialize VC = empty. While edges remain: pick an arbitrary edge $(u, v)$, add both $u, v$ to VC, remove incident edges. This gives $|VC| \le 2 \times K_{opt}^{VC}$.
Thus, selecting vertices NOT in this vertex cover yields an Independent Set with size $\ge N/2 \times K_{opt}^{MIS}$?
Wait, let's rephrase.
If we find a vertex cover $V_C$ such that $|V_C| \le 2 \times \text{min\_vc}$, then the independent set $S = V \setminus V_C$ has size $|S| = N - |V_C| \ge N - 2 \times \text{min\_vc} = 2 \times \text{max\_is} - \dots$ No, that logic doesn't directly translate to a good bound on $K$ because $N$ is fixed.
Actually, since $K^* + VC^* = N$, minimizing VC maximizes IS.
If we find a VC of size $k$, then IS size is $N-k$.
A simple greedy strategy for Vertex Cover: pick vertex with max degree, add to VC, repeat. This gives a constant factor approximation but can be slow if degrees change.
Another greedy strategy for VC: Pick any edge $(u, v)$, pick both $u$ and $v$, remove incident edges. Repeat until no edges left. This guarantees $|VC| \le 2 \cdot |VC_{opt}|$.
Why? Because $VC_{opt}$ must cover all edges processed. For each edge $(u,v)$ picked, at least one endpoint is in $VC_{opt}$. Since we pick both, $|VC| \le 2 \cdot |VC_{opt}|$.
So, using this strategy, we get $K \ge N - 2(N-K^*) = 3K^* - N$ ? No.
$|VC| \le 2 |VC^*| = 2(N - K^*)$.
So $K = N - |VC| \ge N - 2(N - K^*) = 2K^* - N$.
This isn't great. If $K^*$ is small, $K$ could be negative (impossible). It means $K$ could be very small compared to $K^*$. For example, if $G$ is a matching, $K^* = N/2$, $|VC^*| = N/2$. Our algo picks one vertex per edge from matching? No, picking edge $(u,v)$ adds BOTH to VC. So VC size becomes $N/2$, IS size becomes $N/2$. This is perfect.
What if graph is star? Center connected to leaves.
Edges $(c, l_i)$. $K^* = N-1$ (all leaves). $VC^* = 1$ (center).
Greedy strategy: Pick edge $(c, l_1)$. Add $c, l_1$ to VC. Remove all edges incident to $c$ and $l_1$. Remaining edges are none (if only one neighbor). VC = $\{c, l_1\}$. Size 2. IS = $\{l_2, \dots, l_N\}$. Size $N-2$.
Optimal IS size $N-1$. We got $N-2$. This is quite close.
However, consider a complete bipartite graph $K_{n,n}$. $N=2n$. Edges $(u_i, v_j)$ for all $i,j$.
$VC^* = n$. $IS^* = n$.
Greedy Strategy: Pick edge $(u_1, v_1)$. Add $u_1, v_1$ to VC. Remove row 1 and col 1.
Remaining graph is $K_{n-1, n-1}$.
Repeat $n$ times. VC size $2n = N$. IS size 0.
This is terrible! The greedy edge-picking strategy fails catastrophically on bipartite graphs where an edge covers everything else.

Wait, does it really fail?
In $K_{n,n}$, if I pick $(u_1, v_1)$, I must pick both to cover the edge $(u_1, v_1)$. Yes.
Then all other edges incident to $u_1$ or $v_1$ are covered. The remaining edges form $K_{n-1, n-1}$.
Eventually VC = all vertices. IS = empty.
But actually, if I use the "degree heuristic" for Vertex Cover:
Pick vertex with max degree. In $K_{n,n}$, all degrees are $n$. Pick any, say $u_1$.
Add $u_1$ to VC. Remove incident edges. All $v_j$ lose an edge.
New degrees of $v_j$ are $n-1$. Max degree is still $n-1$.
Pick another vertex...
If we keep picking vertices until no edges, we might pick all $u$'s. Then VC size $n$, IS size $n$.
This matches optimal.
So degree-based greedy for Vertex Cover works better than arbitrary edge picking.

Let's check the constraints again. $N=10,000$. $O(N^2)$ is $10^8$, might be okay for 2 seconds, but tight. $O(M \log N)$ or similar is safer.
Degree based greedy for Vertex Cover:
1. Maintain degrees of all vertices.
2. While edges exist:
   a. Find vertex $v$ with max degree.
   b. Add $v$ to VC.
   c. Remove all edges incident to $v$ (decrement neighbors' degrees).
   d. Note: removing $v$ effectively deletes it from consideration for VC. Vertices remaining in the graph update their degrees.

Complexity:
To find max degree, we can scan array: $O(N)$.
Removing $v$: iterate over neighbors. Total iterations over all steps is $2M$.
Scanning array $N$ times?
Wait, number of vertices in VC is at most $N$.
If we scan array every step, complexity is $O(N^2)$. With $N=10000$, $10^8$ ops. C++ handles $\sim 10^8$ in 1s easily. 2.0s is generous.
So Degree Based Greedy for Vertex Cover is feasible.

Algorithm details:
- Adjacency list `adj[N+1]`.
- Array `deg[N+1]` initialized by counting.
- Boolean array `in_vc[N+1]` initialized false.
- Iterate while there is an active vertex (check `deg[v] > 0`):
    - Select vertex $u$ with maximum `deg[u]`.
    - Mark `in_vc[u] = true`.
    - For each neighbor $v$ of $u$:
        - Decrement `deg[v]`.
        - If `deg[v]` becomes 0, it is effectively removed (no longer needs covering).
        - Optimization: Actually, we just need to ensure that any edge not yet covered gets covered. The condition `deg[v] > 0` implies $v$ is part of some uncovered edge.
        - Wait, if `deg[v]` decreases but stays > 0, $v$ is still involved in edges.
    - To speed up finding max degree, use a priority queue (max heap). But pushing neighbors when degree drops adds overhead?
    - Using a PQ: Initial build $O(N + M \log N)$? No, initial insertions $O(N \log N)$. Updates: When degree changes, we can just push new degree?
    - Standard approach: Store `{deg[u], u}` in PQ.
    - Issue: If we update degree of $u$, the old value in PQ is stale. We need to handle lazy deletion.
    - Complexity with PQ: Each edge removal corresponds to updating a vertex. A vertex is removed once. So total updates $O(M)$. Lazy deletion ensures we don't process outdated entries too much. Total time roughly $O(M \log N)$.
    - With $N=10000, M=500000$, $M \log N \approx 5 \cdot 10^5 \cdot 14 \approx 7 \cdot 10^6$, which is very fast.

Alternative: Simple loop to find max degree.
In each step, we pick one vertex, remove edges.
How many vertices do we select for VC? At most $N$.
Loop runs $N$ times (or fewer).
Inside loop, scan $N$ vertices to find max degree -> $O(N^2)$.
Update neighbors -> sum of neighbors over all picked vertices = $2M$ worst case (each edge processed twice at most).
Total time $O(N^2 + M)$.
Given $N=10,000$, $N^2 = 10^8$. 2.0s time limit allows $\approx 10^8$ ops typically, maybe slightly more. This is risky.
Better to use the Priority Queue approach for finding max degree.
With Lazy PQ:
1. Build graph. Compute degrees.
2. Push all vertices into PQ: `{degree[u], u}`.
3. Loop:
   While PQ is not empty and top element has `degree < PQ.top().degree`: pop stale.
   Wait, simply:
   Get top `(d, u)`.
   Check if `current_degree[u] == d`.
   If yes, this is the valid max degree vertex.
   Select `u` for VC.
   For each neighbor `v`:
     `degree[v]--`
     Push `{degree[v], v}` to PQ. (Only if `degree[v] > 0`?)
   Repeat.
   Stop when no edges left. How do we know edges left? If sum of degrees == 0.
   But we need to track edges carefully.
   Sum of degrees starts at $2M$.
   Whenever we remove $u$ from consideration (add to VC), we conceptually delete its incident edges.
   For neighbor $v$, if `degree[v]` goes to 0, it means $v$ has no uncovered edges anymore. We stop processing $v$ (it won't contribute to degree anymore).
   Actually, the stopping condition is simpler:
   The loop continues as long as there are edges.
   Edges can be identified by checking if any vertex has `degree > 0`.
   Since we reduce degrees, eventually all become 0.
   Wait, is `degree[v]` the count of *uncovered* incident edges?
   Initially yes.
   When we pick $u$ for VC:
   For all neighbors $v$ of $u$:
     Edge $(u,v)$ is now covered.
     $v$ loses one potential connection that requires covering.
     So decrement `degree[v]`.
     Does $u$ contribute to edge count after this? No, $u$ is in VC, so edges touching $u$ are "done".
     But what about $v$? If `degree[v]` becomes 0, it means $v$ is incident to NO uncovered edges. It effectively becomes isolated in the subgraph of uncovered edges. It will never be needed in VC again.
   So we can just continue until all degrees are 0?
   Wait, if $u$ is added to VC, does it mean $v$ cannot be added?
   No, in the context of Vertex Cover, we select $u$ and all edges incident to $u$ are satisfied.
   However, vertices still connected to other vertices need coverage.
   If we select $u$, we don't care about edges touching $u$ anymore.
   So, for VC construction:
   Repeat:
     Find $u$ with max degree among vertices with `degree[u] > 0`.
     Add $u$ to VC.
     For each neighbor $v$ of $u$:
       Edge $(u, v)$ is covered.
       Decrement `degree[v]` (since $u$ was a source of requirement for $v$).
   Wait, strictly speaking, Vertex Cover selects vertices to touch every edge.
   If we pick $u$, every edge incident to $u$ is touched.
   We don't need to visit $u$ again.
   For neighbor $v$, edge $(u,v)$ is touched. Any other edge incident to $v$ remains untouched.
   So reducing `degree[v]` counts how many *remaining uncovered edges* are incident to $v$.
   Correct.
   The algorithm terminates when max degree is 0 (all edges covered).
   
   Is this correct?
   Let's trace Star graph: Center 1 connected to 2..N.
   Deg(1)=N-1, Deg(others)=1.
   Step 1: Pick 1. Add to VC.
   Neighbors 2..N: deg becomes 0.
   PQ becomes empty or all 0.
   End. VC = {1}. Size 1. IS = {2..N}. Size N-1. Optimal.
   Trace $K_{n,n}$:
   All degrees $n$.
   Pick $u_1$. Add to VC.
   Neighbors $v_1..v_n$ degrees become $n-1$.
   Next max degree is $n-1$ (many vertices).
   Say pick $v_1$. Add to VC.
   Neighbors of $v_1$: $u_1..u_n$. $u_1$ is already in VC?
   Wait, vertices in VC shouldn't be processed for degree updates?
   If $v_1$ is picked, edges $(v_1, u_i)$ are covered.
   $u_i$ decrements degree.
   Note: We should ignore nodes already in VC?
   Once a node is in VC, its edges are covered. Its degree in the *uncovered* subgraph is 0.
   Wait, my degree update logic decrements neighbors' degree. It doesn't touch itself.
   But we shouldn't consider a vertex already in VC as a candidate again.
   So mark `picked[u] = true`.
   If `picked[u]`, skip.
   Also, if `degree[v]` becomes 0, it has no uncovered edges, so it won't be picked.
   
   Implementation Details:
   - Use `std::vector<int> adj[N+1]`
   - `int deg[N+1]`
   - `bool used[N+1]`
   - `priority_queue<pair<int,int>> pq;` stores `{deg[i], i}`
   - Loop:
     While PQ not empty:
       Pop `p`.
       Check `used[p.second]`. If true, continue (stale/picked).
       Check `deg[p.second] == p.first`. If not, continue (outdated degree).
       Else:
         Pick `u = p.second`.
         `used[u] = true`.
         For `v` in `adj[u]`:
           `deg[v]--`.
           If `deg[v] > 0` (and `!used[v]`):
             Push `{deg[v], v}` to PQ.
             // Wait, if we just push, PQ might grow large.
             // Is it necessary to push every time?
             // Only if degree changes significantly?
             // Actually, standard PQ doesn't support decrease-key efficiently. Pushing is fine.
             // But duplicates?
             // We check `deg[p.second] == p.first` before processing. Duplicates are ignored.
             // Memory usage: each edge causes 1 push?
             // No, degree decreases multiple times. Worst case M pushes?
             // With $M=500,000$, pushing $O(M)$ items is fine.
     
     One detail: initially all vertices have degree. Insert all non-zero degree vertices.
     Actually, insert all $1..N$.
     
     Is the condition `deg[v] > 0` sufficient?
     What if `deg[v]` was updated to something lower, but we don't push immediately?
     The standard way:
     `pq.push({d, v})` whenever `d` changes.
     If `d` changes multiple times, multiple entries in PQ.
     Condition `deg[curr] == stored_d` filters out stale entries.
     
     Wait, is this algorithm optimal?
     This is the "Degree Greedy" for Vertex Cover.
     As discussed, it provides a 2-approximation for Min Vertex Cover.
     Thus IS size $\ge N - 2(V_{opt}) = N - 2(N - I_{opt}) = 2 I_{opt} - N$.
     If $I_{opt} \ge N/2$, this guarantee is poor ($K \ge 2(N/2) - N = 0$).
     Wait, $VC_{opt} + IS_{opt} = N$.
     $|VC_{greedy}| \le 2 |VC_{opt}| = 2 (N - K^*)$.
     $K = N - |VC_{greedy}| \ge N - 2(N - K^*) = 2 K^* - N$.
     Example: $K^* = N/2$. $K \ge 0$.
     But empirically it works reasonably well.
     Is there a better heuristic?
     Maybe "Max Weighted Independent Set" where weights are all 1?
     Equivalent to Max Weighted Vertex Cover where weights are 1?
     Wait, Independent Set on $G$ is Clique on Complement $\bar{G}$.
     Finding max clique is also hard.
     
     Consider Bron-Kerbosch algorithm variants? They are exponential.
     For general graphs with $N=10000$, exact is impossible.
     
     There is a randomized algorithm or iterative improvement?
     Simulated Annealing? Tabu Search?
     These might get stuck in local optima.
     
     However, the problem statement says: "This is a heuristic optimization problem."
     And "You are NOT required to find the theoretically optimal solution."
     So a standard 2-approximation (Greedy VC) is likely intended, OR perhaps a better heuristic exists.
     
     Is there a case where Greedy VC is bad?
     Bipartite graph $K_{n,n}$.
     Degrees start at $n$.
     Pick any $u_1$. VC += $u_1$. Remove edges.
     Remaining graph $K_{n-1, n-1}$ minus $u_1$'s edges?
     Wait, $u_1$ is in VC. Its edges are covered.
     Neighbors $v_1..v_n$ had their degree decreased by 1.
     New degree $n-1$.
     Pick $v_1$. VC += $v_1$.
     Edges incident to $v_1$ covered.
     Neighbors of $v_1$ include $u_1$ (already in VC) and $u_2..u_n$.
     $u_1$'s degree becomes irrelevant (ignored).
     $u_2..u_n$ degree becomes $n-2$.
     Next step, we can pick $u_2$.
     Effectively we alternate picking one from $U$ and one from $V$.
     After $n$ steps, we pick all $U$ and all $V$?
     Wait, in $K_{n,n}$, $VC^* = n$ (one side). $IS^* = n$ (other side).
     Greedy VC picks:
     1. $u_1$ (deg $n$). Removes edges incident to $u_1$.
        $v_i$ deg becomes $n-1$.
        Used $U$: {1}, $V$: {}.
     2. $v_1$ (deg $n-1$). Removes edges incident to $v_1$.
        $u_j$ (for $j>1$) deg becomes $n-2$.
        Used $U$: {1}, $V$: {1}.
     ...
     Eventually we pick all $n$ vertices from $U$ AND $n$ from $V$?
     Wait, if we pick $u_1$, we cover edges $(u_1, v_1)...(u_1, v_n)$.
     Do we still need to pick $v_1$?
     $v_1$ is incident to edges $(v_1, u_1)$ [covered], $(v_1, u_2)$ [not covered].
     So yes, we need to cover $(v_1, u_2)$.
     So we pick $v_1$.
     This seems to lead to $VC$ containing all vertices if we are unlucky?
     Actually, notice that for $K_{n,n}$, any vertex cover MUST contain either ALL $U$ or ALL $V$ (or mixtures summing to $n$? No, min cover is size $n$).
     Wait, if we pick all $U$, size is $n$. Valid VC.
     Can Greedy VC find a VC of size $n$ for $K_{n,n}$?
     Step 1: Pick $u_1$. Covers $(u_1, \forall v)$.
     Remaining edges: all edges between $U \setminus \{u_1\}$ and $V$.
     This is $K_{n-1, n}$.
     Wait, the definition of "edge covered" is satisfied.
     If $u_1$ is in VC, edge $(u_1, v_1)$ is covered.
     Does this help $v_1$? Yes, but $v_1$ is still incident to $(u_2, v_1)$.
     So $v_1$ still has degree 1? No, degree is $n-1$ because $(u_1, v_1)$ is gone.
     Edges incident to $v_1$ remaining: $(u_2, v_1), \dots, (u_n, v_1)$. Count $n-1$.
     So the structure of remaining edges is indeed $K_{n-1, n}$.
     Greedy continues on $K_{n-1, n}$.
     Vertices in $U'$ have degree $n$. Vertices in $V$ have degree $n-1$.
     Max degree is $n$ (at $u_2$). Pick $u_2$.
     ...
     Repeat until we pick all $U$.
     At this point, all edges between $U$ and $V$ are covered (since every $v$ connects to some $u \in U$, which is picked).
     So VC = $U$. Size $n$.
     Wait, earlier I thought we pick $v_1$ too. Why?
     In $K_{n-1, n}$ (after picking $u_1$), degrees of $U'$ are $n$, degrees of $V$ are $n-1$.
     We pick vertex with MAX degree. That is any $u \in U'$.
     So we keep picking from $U$ until $U$ is exhausted.
     Then VC = $U$. Size $n$.
     This is optimal!
     So for $K_{n,n}$, Greedy VC with tie-breaking preference for high degree works perfectly.
     
     Is there a graph where it fails badly?
     Usually greedy strategies for VC are within factor 2.
     Given the constraints and "heuristic" nature, this seems sufficient.
     Also note, for sparse graphs (like paths), it performs very well.
     
     So the plan:
     Implement Greedy Vertex Cover using Priority Queue.
     Output Independent Set = $V \setminus VC$.
     
     One more thing:
     Is there a constraint about "multiple edges"?
     "Multiple edges between the same pair of vertices may appear but imply the same constraint."
     Adjacency list can store duplicates, or use `std::set` or sort and unique.
     Duplicates affect degree count?
     If multiple edges $(u, v)$, picking $u$ covers all of them.
     If we store duplicates in `adj`, iterating `adj[u]` will see $v$ multiple times.
     We must decrement `deg[v]` for EACH occurrence.
     But wait, if there are parallel edges, the constraint "no edge" applies.
     Does the algorithm hold?
     Yes, degree represents number of edges requiring covering.
     Parallel edges require covering too.
     So if we have 2 edges between $u, v$, and we pick $u$, both are covered.
     So $v$'s degree should decrease by 2.
     So duplicates should be counted in degree.
     Just store all edges in adjacency list.
     Complexity increases?
     $M \le 500,000$. Even with duplicates, $M$ lines input.
     So total edge occurrences $\le M$.
     Sum of sizes of adj lists $\le 2M$.
     So complexity remains proportional to $M$. OK.

     Corner case: Disconnected components? Handled naturally.
     Self loops? "u != v". OK.
     
     Data structures:
     `adj` vector array.
     `deg` array.
     `used` boolean array.
     `pq` priority queue.
     
     Input parsing:
     Read N, M.
     Loop M times: read u, v.
     `adj[u].push_back(v); adj[v].push_back(u);`
     `deg[u]++; deg[v]++;`
     
     Initialization:
     For i = 1 to N: `pq.push({deg[i], i});`
     
     Main Loop:
     While (!pq.empty()) {
       pair<int, int> top = pq.top(); pq.pop();
       int d = top.first; int u = top.second;
       
       if (d < deg[u]) continue; // Stale
       if (used[u]) continue; // Already picked
       
       // Select u
       used[u] = true;
       // Now process neighbors
       for (int v : adj[u]) {
         deg[v]--;
         // Check if v needs reconsideration
         if (deg[v] > 0 && !used[v]) {
             // We must update its priority.
             // We can just push again.
             // But wait, if we push to PQ, we might have multiple entries for v.
             // The check `if (d < deg[u])` at the top of loop handles this lazily.
             // We just need to ensure we push the NEW degree.
             // Is it guaranteed that the entry with the smaller degree comes out later?
             // No, PQ orders by key. Smaller degree comes out LATER.
             // Wait, we want max degree.
             // We push `{new_deg, v}`.
             // If `new_deg < d`, it goes deeper in PQ.
             // Later if `old_deg` entry pops (larger value), it checks `deg[v] == old_deg`. Fails (because `deg` changed).
             // If `new_deg` pops, it is valid.
             // Correct.
             // BUT: if `new_deg` is still large, it might come out before we need it?
             // That's fine, it processes correctly.
             pq.push({deg[v], v});
         }
       }
     }
     
     Wait, one catch:
     If we update `deg[v]`, and push `{deg[v], v}`,
     Later we might encounter the old `{old_deg, v}` entry in PQ.
     `old_deg > deg[v]`.
     When it pops, `top.first (old_deg) > deg[v]`.
     Condition `d < deg[u]` (using `d=top.first`).
     Wait, inside loop: `d` is the degree associated with the popped item.
     Current actual degree is `deg[v]`.
     We want to execute ONLY IF `d == deg[v]`.
     Since PQ orders by max first:
     If `d > deg[v]`, the item is outdated and represents an HIGHER degree than current.
     Since we only pick nodes with MAX degree, processing a stale HIGH degree item is dangerous?
     Wait. If `d > deg[v]`, it means previously $v$ was considered to have degree $d$.
     We haven't picked $v$ yet.
     Now its degree dropped.
     If we process it now (thinking it has degree $d$), we might pick it.
     But we can only pick it if it is the max degree.
     Actually, if `d > deg[v]`, this is definitely not the current degree.
     But could it be that `v` is still one of the candidates?
     Yes, but it's no longer true that its degree is `d`.
     We should treat it as "invalid priority".
     Since we want the VERITABLE max degree, ignoring stale entries is correct IF we don't process it.
     We should process it only if `d == deg[v]`.
     Wait, what if `d > deg[v]`?
     The real degree is smaller.
     If there is another node `w` with degree $D_{w} \ge d$, then $w$ would be picked first.
     If `w` was picked, then we return to loop.
     Eventually `v` enters as max.
     If we have multiple entries for `v`: `{10, v}` and `{9, v}`.
     `{10, v}` pops first.
     `deg[v]` is actually 9.
     So `10 != 9`. Ignore.
     `{9, v}` pops. `9 == 9`. Process.
     This logic holds.
     
     Wait, what if `d` is smaller than current `deg`? Impossible since we only decrease.
     Wait, degree decreases. So `d >= deg[v]` always.
     So the condition is `d == deg[v]`.
     Wait, is it possible `deg[v]` became 0?
     We only push if `deg[v] > 0`.
     Wait, if `deg[v]` becomes 0, we shouldn't push.
     But we need to know if $v$ is still available.
     If `deg[v] == 0`, $v$ has no uncovered edges. It doesn't need to be in VC.
     So `if (deg[v] > 0)` is correct to avoid unnecessary work.
     Wait, if `deg[v]` becomes 0, should we add to VC? No.
     But wait, could $v$ be part of VC even if degree is 0?
     If $deg[v]=0$, it has no uncovered edges. Including it in VC costs 1 unit and saves 0. Never beneficial.
     Unless it's mandatory? No, if edges are covered, no need to pick it.
     
     Wait, there's a risk:
     If we simply ignore stale entries, how do we ensure we eventually pick nodes if they are needed?
     Nodes are only needed if they have edges.
     Edges are removed when endpoints are picked.
     When all edges are removed, no one is needed.
     If a node has edges, its degree > 0.
     Eventually, if it is max degree, it will be picked.
     
     One optimization:
     When `deg[v]` decreases, we push to PQ.
     However, `deg` decreases by 1 multiple times?
     Yes, if $u$ is picked, $v$ loses one edge.
     If $u$ is picked again? No, $u$ is marked `used`.
     So `deg[v]` decreases at most `initial_deg[v]` times.
     Total pushes $\le \sum deg = 2M$.
     So PQ operations $O(M \log N)$ or $O(M \log M)$. Fine.
     
     Wait, memory limit 512MB.
     $N=10000$, vectors take little space.
     
     One edge case:
     If `deg[v]` becomes 0, we stop pushing.
     But we must be careful: `used[u]` prevents picking it.
     Is it possible `deg[v]` becomes 0 but $v$ is never pushed?
     If we start pushing when `deg` changes.
     Initially all pushed.
     If `deg` becomes 0, we don't push.
     But it was already pushed with positive degree.
     Wait, if `deg` becomes 0, it will stay 0.
     If there are old entries with `d > 0`, they will pop.
     Check `d == deg[v]` (which is 0).
     Fail. Ignore.
     Loop continues.
     PQ eventually empties or contains only zeros (which we ignore).
     Wait, if PQ contains `{0, v}`, it might sit at bottom?
     No, `priority_queue` is max heap.
     Zeros might be at top? No, only if all remaining degrees are 0.
     If max degree is 0, loop terminates.
     Wait, if PQ has `{0, v}` and all others are `0`, we process.
     Check `d==deg[v]` (0==0). True.
     Then we process `v`.
     `used[v]` check?
     If `used[v]` is false, we add `v` to VC.
     But `deg[v]=0` means no edges. Adding to VC is useless.
     Should we prevent processing `v` if `deg[v]==0`?
     Yes, explicitly check `deg[v] > 0`?
     Actually, if we rely on `d == deg[v]`, then `d` must be 0.
     If `d > 0` initially pushed.
     If `deg[v]` hits 0, we don't push new.
     Old entries `{d>0, v}` pop.
     Check `d == 0`. False. Ignored.
     So eventually all `{d>0}` disappear.
     When all degrees are 0, PQ might still contain stale entries.
     We process them, ignore.
     Empty PQ.
     Result: VC contains vertices that were picked when degree > 0.
     Vertices with degree 0 never picked (correct).
     
     Wait, we need to check `deg[v] > 0` before processing `v`?
     Suppose `v` has degree 1. We pick `u` adjacent. `deg[v]` becomes 0.
     `v` is NOT pushed.
     Eventually `v` is processed via stale entry `{1, v}`.
     `1 != 0`. Ignored.
     So `v` is not picked.
     Correct.
     
     Wait, initialization.
     For `v` with `deg[v] > 0`, push to PQ.
     Loop condition: while `!pq.empty()`.
     Inside: `pair top = pq.top(); pq.pop();`
     `int d = top.first; int u = top.second;`
     `if (d != deg[u]) continue;`
     `if (used[u]) continue;`
     `// Pick u.`
     
     Is it possible `deg[u]` is 0 but `d` is 0?
     If we pushed `{0, u}`?
     Only if `deg[u]` became 0?
     My code logic above said: `if (deg[v] > 0) pq.push`.
     So we never push 0.
     Except initial push? `deg[u]` could be 0.
     If `deg[u] == 0`, push `{0, u}`.
     Pops. `d=0`. `deg[u]=0`. Match.
     `used[u]` false.
     Pick `u`.
     `adj[u]` empty.
     `used[u] = true`.
     Loop continues.
     But we pick vertices with no edges.
     This increases VC size unnecessarily.
     Wait, we don't want to pick vertices with degree 0.
     They are irrelevant.
     So filter: `if (deg[u] == 0) continue;`?
     Wait, if `d=0` and `deg[u]=0`, `d!=deg[u]` check passes.
     Then `used[u]` check passes.
     Then we pick `u`.
     This is BAD.
     Fix:
     1. Don't push vertices with initial degree 0.
     2. Or, when popping, if `deg[u] == 0`, ignore it (since we won't need it for VC unless graph has self-loops, but input says u!=v).
        Actually, if a component is just isolated vertices, any subset is independent set.
        We want to MAXIMIZE independent set -> MINIMIZE VC.
        Isolates should NOT be in VC.
        So if `deg[u] == 0`, skip it.
     
     Wait, if I don't push 0-degrees, how do I handle the case where `deg` decreases to 0?
     My previous logic: if `deg` drops to 0, DO NOT PUSH.
     Then the last state `deg[u]` is tracked.
     Wait, if `deg` is 0, we assume it's fine.
     But we must ensure we don't pick it.
     The `if (d != deg[u])` check works if we don't push 0.
     If `deg` becomes 0, it won't have a 0-entry in PQ.
     So we never pick it.
     Exception: if it was 0 initially.
     If `deg[u]` is 0, don't push.
     Then it is never popped.
     Perfect.
     
     Final logic refinement:
     Loop:
       Pop `d, u`.
       `if (d != deg[u]) continue;`
       `if (deg[u] == 0) continue;` // Safety check
       `if (used[u]) continue;`
       
       `used[u] = true;`
       `VC.add(u);`
       For `v` in `adj[u]`:
         `deg[v]--`;
         `if (deg[v] > 0) pq.push({deg[v], v});` // Only push if still relevant
     
     Wait, if `deg[v]` was 1, becomes 0.
     We do not push.
     Correct.
     
     What if `d != deg[u]` logic is tricky?
     `d` comes from PQ. `deg[u]` is current.
     If we update `deg[v]` without pushing, we rely on old entry failing `d != deg[u]`.
     Correct.
     
     Wait, we need to populate `deg` first.
     
     One subtle issue: `pq.top()` access on empty queue. Check not empty.
     
     Code structure looks solid.
     Includes: `iostream`, `vector`, `queue`, `algorithm`.
     IO optimization `cin.tie(NULL); ios_base::sync_with_stdio(false);`.
     
     Let's write the solution.
     
     Wait, one more check.
     Problem asks for $N$ lines of 0/1.
     If $x_u = 1$ (in IS), means $u \notin VC$.
     So print 1 if `!used[u]`, 0 otherwise.
     
     Wait, I/O format:
     "Output exactly N lines. The i-th line..."
     
     Double check example.
     4 3
     1 2
     2 3
     3 4
     Graph: 1-2-3-4.
     Deg: 1:1, 2:2, 3:2, 4:1.
     PQ: {(2, 2), (2, 3), (1, 1), (1, 4)} (sorted desc).
     1. Pop (2, 2). Pick 2.
        adj[2]: 1, 3.
        deg[1] -> 0. (Don't push).
        deg[3] -> 1. Push (1, 3).
        used[2] = true.
     2. Pop (1, 3). (Assuming PQ order).
        Wait, if (1, 3) pops.
        Check deg[3]=1. Match. Not used.
        Pick 3.
        adj[3]: 2, 4.
        deg[2] -> 0 (was 2, now effectively 1 but decremented?).
        Wait, deg update logic:
        When picking 2, we decremented 3's degree from 2 to 1.
        When picking 3, we decrement neighbors.
        Neighbor 2: used=true?
        If we iterate neighbors regardless:
        deg[2] --. Becomes 0 (if it was 1?).
        Wait, initially deg[2]=2.
        Pick 2.
        decr 1 -> 0.
        decr 3 -> 1.
        Push 3.
        Next pop (1, 3).
        Pick 3.
        decr 2 -> 0 (from 2?). No, 3 sees 2.
        2 is in adj[3].
        deg[2] decremented. It was 2? No.
        Wait, `deg` array tracks uncovered edges.
        When 2 picked, edges (2,1) and (2,3) removed.
        So 2 has 0 uncovered edges. `deg[2]` conceptually 0.
        But my code doesn't zero `deg[2]` explicitly on pick.
        It sets `used[2] = true`.
        Does `deg` matter for `used` node?
        No, we skip `used[u]`.
        But neighbor 2 of 3:
        `deg[2]` decremented. From what?
        If I didn't modify `deg[2]` when picking 2.
        Then `deg[2]` is still 2.
        When picking 3:
        `deg[2]--` -> 1.
        Push `(1, 2)`.
        Next pop `(1, 2)`.
        `used[2]` is true. Skip.
        This works fine?
        Wait, `deg[2]` is technically garbage if `used[2]` is true.
        But `if (used[u]) continue` handles it.
        
        However, logic consistency:
        If I pick 3, edge (3, 4) is removed. `deg[4]` decrements.
        `deg[4]` was 1. Becomes 0.
        Do not push.
        
        Next pop?
        PQ might have (1, 4) if it was there.
        If 4 was never pushed (wait, it was).
        Pop (1, 4).
        Check `deg[4]` is 0.
        Logic:
        `if (d != deg[u]) continue;` (1 != 0) -> Continue.
        So we skip 4.
        Correct.
        Wait, why did 4 have 1? It's isolated in remaining graph.
        It's covered by 3's edges?
        Edge (3, 4) covered by picking 3.
        So 4 needs no covering.
        Correct.
        So 4 is NOT in VC.
        So IS includes 4.
        Also 1 was not in VC. IS includes 1.
        IS = {1, 4}. Size 2.
        Vertices 2, 3 in VC.
        Output:
        1: 0? No, 1 is IS. Output 1.
        Wait, my manual trace result IS={1, 4}.
        My code picked VC={2, 3}.
        So IS = V \ VC = {1, 4}.
        Matches example output (1, 0, 0, 1).
        Wait, 2 and 3 are 0.
        Yes.
        Matches.
        
     Wait, example output says:
     1
     0
     0
     1
     Implies S = {1, 4}.
     My trace produced S = {1, 4}.
     Looks consistent.
     
     One more check on time complexity.
     Each edge reduces sum of degrees.
     Max pushes per vertex change is bounded?
     Actually, sum of degrees is $2M$.
     Every `deg[v]--` corresponds to an edge being covered.
     Each edge covered once.
     So total decrements across all vertices $\le M$ (actually $M$ edges, each covered once by one endpoint).
     Wait, an edge $(u, v)$ is covered when $u$ is picked or $v$ is picked.
     If $u$ picked, $v$'s degree drops.
     If $v$ picked, $u$'s degree drops.
     Wait, we decrement neighbors.
     So edge $(u,v)$ contributes to decrement of $v$ when $u$ picked.
     And $u$ when $v$ picked.
     So each edge triggers at most 1 decrement? No.
     If we pick $u$, we decrement all neighbors $v$.
     Edges $(u, v_1), (u, v_2) \dots$
     Each edge $(u, v_i)$ is handled.
     Then later, if $v_i$ picked, we check its neighbors.
     But $u$ is already picked (`used[u]=true`).
     Does it matter if we process $v_i$'s neighbors?
     $u$ is skipped due to `used`.
     But we still decrement `deg` of other neighbors.
     So complexity is proportional to $O(\sum \text{neighbors of picked vertices})$.
     In worst case, we pick all vertices? $O(N + M)$.
     Total operations $\le O(M)$.
     Heap operations $O(M \log N)$.
     Safe.
     
     Wait, what if I pick $u$, then later pick $v$?
     The logic holds.
     
     Is it possible `priority_queue` is slow due to many elements?
     Max elements $M+N$.
     $5 \cdot 10^5$.
     $\log(5 \cdot 10^5) \approx 19$.
     $10^7$ ops. Very safe.
     
     Include headers: `<bits/stdc++.h>` is fine for CP context, or `<iostream>`, `<vector>`, `<queue>`, `<algorithm>`.
     
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

const int MAXN = 10005;
vector<int> adj[MAXN];
int deg[MAXN];
bool used[MAXN];

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
        deg[u]++;
        deg[v]++;
    }

    priority_queue<pair<int, int>> pq;
    for (int i = 1; i <= N; ++i) {
        if (deg[i] > 0) {
            pq.push({deg[i], i});
        }
    }

    fill(used, used + N + 1, false);

    while (!pq.empty()) {
        pair<int, int> top = pq.top();
        pq.pop();

        int d = top.first;
        int u = top.second;

        if (used[u] || d < deg[u]) { 
            // Wait, d <= deg[u] is always true because we only decrement.
            // So d > deg[u] is impossible?
            // Actually, since we push updated degrees, d should match deg[u].
            // If d > deg[u], it's a stale entry from a previous larger degree.
            // Wait, can d be greater than current deg?
            // We only push if we decrement? No.
            // We push INITIAL degree.
            // If we update, we push new LOWER degree.
            // So `deg[u]` only decreases.
            // So current `deg[u]` <= old `deg[u]`.
            // If `d` (old) > `deg[u]` (new), it is stale.
            // Wait, I wrote `d < deg[u]` in thought earlier?
            // If `deg` decreases, `d` (old) > `deg[u]` (new).
            // So `d > deg[u]` implies stale.
            // If `d == deg[u]`, it's valid.
            // If `d < deg[u]`, it implies `deg` increased? Impossible.
            // So we check: `if (d != deg[u]) continue;`
            // Wait, strict inequality?
            // If we push `{deg[v], v}` after decrementing.
            // The `deg[v]` is smaller than previous.
            // So if we have `{old, v}` in PQ, `old > new`.
            // So `d > deg[u]` happens.
            // Correct check: `if (d != deg[u]) continue;`
            // Because `deg[u]` can never increase.
            // If `d < deg[u]`, impossible.
            // So `d != deg[u]` covers `d > deg[u]`.
            continue;
        }
        
        // Wait, what about the condition used in my trace: `d == deg[u]`
        // If `d != deg[u]`, continue.
        // Is `deg[u]` guaranteed to be non-negative? Yes.
        
        // But I also have `d < deg[u]` check in the code snippet above.
        // Let's stick to `if (d != deg[u]) continue;`
        // But wait, what if `d < deg[u]`? Impossible.
        
        // Re-evaluating the logic `d < deg[u]`.
        // If I initialize PQ with `deg`.
        // Then update `deg` and push.
        // So `deg` always monotonically decreases.
        // So `d` (popped) >= `current deg[u]`.
        // Wait. If I pop `d=5`, current `deg[u]` is 5. OK.
        // If I pop `d=5`, current `deg[u]` is 4 (because updated).
        // `d > deg[u]`. So it's stale.
        // If I pop `d=5`, current `deg[u]` is 6. Impossible.
        // So condition `if (d != deg[u]) continue;` is robust.
        
        // WAIT.
        // What if `d < deg[u]` is possible? No.
        
        // Okay, code:
        if (used[u] || d != deg[u]) continue;
        
        // One crucial thing:
        // If `d` is from an entry that was pushed but `deg[u]` was higher at push time?
        // Wait.
        // If I push `{10, u}`.
        // Then `deg[u]` drops to 8. Push `{8, u}`.
        // Then I pop `{10, u}`.
        // `d=10`. `deg[u]=8`. `d != deg[u]`. Continue.
        // Then I pop `{8, u}`.
        // `d=8`. `deg[u]=8`. `d == deg[u]`. Process.
        // Correct.
        
        // Is it possible `d < deg[u]`? No.
        // Is it possible `d > deg[u]`? Yes, stale.
        
        // What if `d < deg[u]`?
        // Only if `deg` increased. Not happening.
        
        // Wait, logic error in my comment earlier?
        // If I wrote `d < deg[u]` in the block, it was for stale entry logic?
        // Usually stale check is `d < deg[u]` in some contexts (like Dijkstra where dist decreases).
        // Here `deg` is upper bound decreasing. So `d > deg[u]` is stale.
        // But `d < deg[u]` would mean we found a path/edge that increased cost? No.
        // So `if (d != deg[u]) continue;` is correct.
        
        // However, if `deg[u]` is currently 5. And `d` is 10.
        // `d != deg[u]`.
        // Correct.
        // What if `deg[u]` is 10. `d` is 10.
        // Correct.
        
        // Wait, what if `d > deg[u]`?
        // This means `d` was an older, larger value.
        // But wait, if `deg[u]` is currently 10. `d` could be 10.
        // If `deg[u]` became 5. `d` is 10.
        // `d != deg[u]`.
        // So `continue` skips.
        // This seems correct.
        
        // Wait, what if `d < deg[u]`?
        // Can `deg[u]` be greater than `d`?
        // Only if `d` was a pushed value from somewhere else?
        // No, `d` belongs to `u`.
        // If `deg[u]` increased? Impossible.
        // So `d` is always $\ge$ `current deg[u]`.
        // Wait. `d` is the degree at the time of push.
        // `deg[u]` is current.
        // Since degree decreases, `deg[u]` $\le d$.
        // So `d > deg[u]` means stale.
        // `d == deg[u]` means fresh.
        // So `if (d != deg[u]) continue;` is logically equivalent to `if (d > deg[u]) continue;`.
        // Is `d < deg[u]` possible? No.
        
        // Wait, I'll use `if (d != deg[u])` for safety/simplicity.
        
        // Wait, there is a case where I pushed `{8, u}` but `deg[u]` is 8.
        // But there was ALSO `{10, u}` in PQ.
        // If I process `{10, u}`, it is skipped.
        // If I process `{8, u}`, it is accepted.
        
        // Wait, one logic hole:
        // If `deg[u]` was 10.
        // Updated to 8.
        // Pushed 8.
        // PQ has 10, 8.
        // Pop 10. `10 != 8`. Skip.
        // Pop 8. `8 == 8`. Process.
        // Correct.
        
        // What if `deg[u]` becomes 0?
        // Update to 0. Push 0?
        // My logic says `if (deg[v] > 0) push`.
        // So we don't push 0.
        // So PQ has 10, 8.
        // Both fail check?
        // 10 != 0 (skipped).
        // 8 != 0 (skipped).
        // Neither processed.
        // Correct.
        
        // Wait, check `deg[u] == 0` condition again.
        // If `deg[u] == 0`, we don't want to pick it.
        // If `d != 0`, it will be skipped.
        // If `d` was 0?
        // If we don't push 0, we never get `d=0` from PQ.
        // So `d != 0` implies stale.
        // Correct.
        // EXCEPT: what if `deg[u]` is 0 initially?
        // We don't push.
        // So never popped.
        // Correct.
        
        // So final check:
        if (used[u] || deg[u] == 0 || d != deg[u]) continue;
        // Note: `deg[u] == 0` check is redundant if `d != deg[u]` holds for non-zero pushes,
        // BUT if `d` somehow matches 0 (e.g. we push 0).
        // To be safe, explicit check `deg[u] == 0` avoids picking 0-degree nodes if we ever pushed them.
        // Since I decided NOT to push 0s, `d` will never be 0.
        // So `d != deg[u]` check will trigger for all stale entries that reached 0.
        // Wait. If `d` corresponds to 0-deg, it means `d=0`.
        // But we assumed we don't push 0.
        // If `d` came from a push, `d` was $>0$.
        // But `deg[u]` became 0.
        // Then `d != deg[u]` is $d \neq 0$. Since $d>0$, this is true.
        // So skip.
        // So explicit `deg[u] == 0` is not needed for correctness but good for clarity.
        
        // Wait, I will add `if (deg[u] == 0) continue;` to be absolutely sure.
        
        used[u] = true;
        
        // Since we are building Independent Set,
        // Actually, I am building Vertex Cover.
        // IS is everything NOT in VC.
        
        // Process neighbors
        for (int v : adj[u]) {
            deg[v]--;
            if (deg[v] > 0 && !used[v]) {
                pq.push({deg[v], v});
            }
        }
    }
    
    for (int i = 1; i <= N; ++i) {
        // if not in VC (used is false), it's in IS -> print 1
        // else print 0
        cout << (used[i] ? 0 : 1) << "\n";
    }

    return 0;
}
```

Wait, I should double check the condition `if (d != deg[u])`.
Is it possible `d < deg[u]`?
As established, `deg` decreases. So `current deg` $\le$ `past deg` (`d`).
So `d >= current deg`.
So `d < current deg` is impossible.
So `d != deg[u]` implies `d > deg[u]`.
Correct.

Wait, looking at the code snippet in thought:
`if (used[u] || d < deg[u]) continue;`
Wait, why did I think `d < deg[u]` earlier?
Usually in Dijkstra `dist` decreases, and we check `if (curr.d > d[u]) continue`.
Here we want to process if `d[u] == d`.
Since `d` decreases, if we find `d < d[u]`, it implies `d` is older but larger? No.
If `d` is from PQ, it's an old value.
Old values are $\ge$ Current value.
So `d >= d[u]` is always true.
If `d > d[u]`, it's stale.
If `d == d[u]`, it's fresh.
So `if (d != deg[u])` is the check.

One logic issue:
Can `d` be less than `deg[u]`?
No.
So `d < deg[u]` check is always false?
Wait, if I write `if (d < deg[u])` as a stale check, it means nothing will be skipped?
Because `d` is never less.
It MUST be `d > deg[u]`.
So the check `d != deg[u]` is correct.
Wait, `d != deg[u]` covers `d > deg[u]`.
And `d == deg[u]` covers the valid case.
Since `d` cannot be smaller, `d !=` works.

Wait, if `d` is stale, `d > deg[u]`.
Example: `deg` was 10. Updated to 5. Push 5.
PQ: `{10, u}`, `{5, u}`.
Pop `{10, u}`. `deg[u]` is 5. `10 != 5`. Skip.
Pop `{5, u}`. `deg[u]` is 5. `5 == 5`. Process.
Logic holds.

What if `deg[u]` is 5.
Push 5.
Later `deg` becomes 4. Push 4.
Pop 5. Skip.
Pop 4. Process.
Works.

Wait, I missed one thing:
The loop `while(!pq.empty())` will terminate when PQ is empty.
If `deg` becomes 0, we stop pushing.
Stale entries with `d > 0` remain.
They are checked against `deg[u]` (which is 0).
`d != 0`. Skipped.
Eventually PQ becomes empty.
Wait, if all `d` are skipped, we need to ensure we don't process invalid ones.
My code checks `d != deg[u]`.
So it skips all stale entries where `deg` dropped.
It processes only fresh entries where `d == deg[u]`.
And `deg[u] > 0` check.
This ensures we only pick vertices with `current degree > 0` that have a matching entry in PQ (which means it was pushed recently or init).
But wait.
If `deg` was 10.
Updated to 9. Push 9.
Updated to 8. Push 8.
PQ has 10, 9, 8.
Pop 10. `deg` is 8. `10 != 8`. Skip.
Pop 9. `deg` is 8. `9 != 8`. Skip.
Pop 8. `deg` is 8. Process.
This logic assumes `deg[u]` hasn't dropped *again* between pushing 9 and popping 9.
That's guaranteed because `deg` only changes when `u` is neighbor of a picked vertex.
If `u`'s degree changes, it is done by picking some `w`.
We pick `w`. Neighbors `u`'s degree drops.
If we pick `w` again? No `used[w]`.
So `deg[u]` drops.
So `deg` is monotonic.
The sequence of entries pushed is a sequence of decreasing degrees for `u`.
When we pop `d`, if `d` is the latest one, it equals `deg[u]`.
Since it is a max heap, we might pop older (larger) entries first.
Those are skipped.
Eventually we hit the largest entry that equals `deg[u]`.
Wait, is it the largest equal?
If `d` matches `deg[u]`, it is valid.
Are there multiple entries with same `d`?
Yes. `deg` drops from 10 to 5.
We push 5.
Do we push 5 again?
Only if it changes further.
Wait, `deg[u]` drops from 5 to 4.
Push 4.
So distinct values.
Wait, could `deg` drop multiple times?
Yes.
Initial: 10. Push 10.
Drop 5. Push 5.
Drop 4. Push 4.
PQ: 10, 5, 4.
Pop 10. Skip (8).
Pop 5. Skip (4).
Pop 4. Match.
Seems correct.

However, consider case where `deg` stays same?
Does `deg` ever stay same?
Only if we don't update it.
If we don't push, it's not in PQ.
Wait, if `deg` updates multiple times?
Example: Drop 2. Drop 1. Drop 0.
Push 2, 1. (0 not pushed).
PQ: ..., 2, 1.
Pop 2. Skip (1).
Pop 1. Match?
Wait. If `deg` is 1. `d=1`. Match.
Process.
But wait. `deg` drops from 1 to 0.
Push nothing.
Current `deg` is 0.
Entry 1 is stale. `1 != 0`. Skip.
Correct.

Wait, is it possible `deg` increases? No.
So `d != deg[u]` check is robust.

Final check on time limit 2.0s.
Input reading: $10^5$ ints. Fast I/O needed.
Adjacency list: vector.
PQ operations: max $M$ pushes.
Total time complexity: $O(M \log N)$.
With $N=10000, M=500000$, $M \log N \approx 7 \cdot 10^6$ ops.
Very fast.
Memory: $O(N+M)$. Fits 512MB easily.

Code looks good.

```cpp
/*
Problem: Maximum Independent Set Challenge
Solution: Heuristic Algorithm (Approximate Vertex Cover)

Strategy:
We seek a Maximum Independent Set (MIS), which corresponds to a Minimum Vertex Cover (MinVC) in terms of $|V| - |MIS| = |VC|$.
Since finding the exact MinVC is NP-hard, we use a greedy approximation.
A standard greedy approach for MinVC is to iteratively pick the vertex with the highest degree, add it to the Vertex Cover, and remove all incident edges.
This provides a 2-approximation ratio.
For the given constraints (N=10,000, M=500,000), this approach fits well within the time limits.

Data Structures:
- Adjacency list `adj` to store the graph.
- `deg` array to track the current degree (number of uncovered edges) of each vertex.
- `used` array to track vertices already included in the Vertex Cover.
- `priority_queue` to efficiently retrieve the vertex with the maximum degree.
- Lazy deletion is used for the priority queue to handle dynamic degree updates.
*/

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

// Define maximum number of vertices as per constraints
const int MAXN = 10005;

// Global variables for graph and state
vector<int> adj[MAXN];
int deg[MAXN];
bool used[MAXN];

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
        deg[u]++;
        deg[v]++;
    }

    // Priority queue to store {degree, vertex_index} ordered by degree descending
    priority_queue<pair<int, int>> pq;
    for (int i = 1; i <= N; ++i) {
        // Only push vertices with at least one incident edge
        if (deg[i] > 0) {
            pq.push({deg[i], i});
        }
    }

    // Fill used array with false initially
    fill(used, used + N + 1, false);

    // Greedy Vertex Cover Construction
    while (!pq.empty()) {
        pair<int, int> top = pq.top();
        pq.pop();

        int d = top.first;
        int u = top.second;

        // Check validity of the popped entry:
        // 1. If 'u' is already selected for VC, skip.
        // 2. If 'd' (stored degree) differs from current 'deg[u]', it's a stale entry.
        //    Since degrees only decrease, 'd > deg[u]' means it's outdated.
        //    'd < deg[u]' is impossible in this logic.
        // 3. If 'deg[u]' is 0, it has no uncovered edges and shouldn't be picked.
        if (used[u] || d != deg[u] || deg[u] == 0) {
            continue;
        }

        // Select vertex 'u' for Vertex Cover
        used[u] = true;

        // Remove all edges incident to 'u'. This involves decrementing the degree of neighbors.
        for (int v : adj[u]) {
            deg[v]--;
            // If 'v' still has uncovered edges, update its position in the heap.
            // We push the new degree. Duplicate entries are handled by the check 'd != deg[v]'.
            if (deg[v] > 0 && !used[v]) {
                pq.push({deg[v], v});
            }
        }
    }

    // Output the Independent Set
    // x_i = 1 if vertex i is NOT in Vertex Cover (Independent Set), else 0
    for (int i = 1; i <= N; ++i) {
        cout << (used[i] ? 0 : 1) << "\n";
    }

    return 0;
}
```
One small fix:
Inside loop `if (used[v])` check.
If `v` is already in VC (`used[v]`), its edges are already covered.
Decrementing `deg[v]` is meaningless but harmless.
Checking `!used[v]` before pushing to PQ saves some operations, though negligible.
Wait, if `v` is `used`, it shouldn't be considered for VC again anyway.
So skipping push for `used[v]` is optimization.
Also `deg[v]--` is safe.
The logic `deg[u] == 0` handles termination.
If `deg[v]` drops to 0, we don't push.
This relies on `deg[v]` tracking uncovered edges.
If `deg[v]` reaches 0, it means all edges incident to `v` are covered.
Wait.
Suppose `v` has neighbors `a, b`. `deg[v]` initially 2.
If `a` is picked, `deg[v]` becomes 1. Push 1.
If `b` is picked next. `deg[v]` becomes 0. Do not push.
Correct.
What if `a` and `b` are both picked?
When `a` picked, `deg[v]` becomes 1.
When `b` picked, `deg[v]` becomes 0.
Correct.
Wait, if `b` is picked, `v`'s degree decremented.
Wait, does `b` seeing `v` imply edge `(v,b)` is covered?
Yes.
So logic is sound.

Wait, one detail:
`if (deg[v] > 0 && !used[v])`
If `used[v]` is true, we don't push.
If `deg[v]` is 0, we don't push.
So `v` with `used[v]` might still be in PQ?
Yes, stale entries.
They will be popped and skipped because `used[v]` is checked.
Correct.

Wait, `fill(used, used + N + 1, false);`
Indices are 1 to N. Correct.

Looks ready.
</think>

```cpp
/*
Problem: Maximum Independent Set Challenge
Solution: Heuristic Algorithm (Approximate Vertex Cover)

Strategy:
We seek a Maximum Independent Set (MIS), which corresponds to a Minimum Vertex Cover (MinVC) in terms of |V| - |MIS| = |VC|.
Since finding the exact MinVC is NP-hard, we use a greedy approximation.
A standard greedy approach for MinVC is to iteratively pick the vertex with the highest degree, add it to the Vertex Cover, and remove all incident edges.
This provides a 2-approximation ratio.
For the given constraints (N=10,000, M=500,000), this approach fits well within the time limits.

Data Structures:
- Adjacency list `adj` to store the graph.
- `deg` array to track the current degree (number of uncovered edges) of each vertex.
- `used` array to track vertices already included in the Vertex Cover.
- `priority_queue` to efficiently retrieve the vertex with the maximum degree.
- Lazy deletion is used for the priority queue to handle dynamic degree updates.
*/

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

// Define maximum number of vertices as per constraints
const int MAXN = 10005;

// Global variables for graph and state
vector<int> adj[MAXN];
int deg[MAXN];
bool used[MAXN];

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
        deg[u]++;
        deg[v]++;
    }

    // Priority queue to store {degree, vertex_index} ordered by degree descending
    priority_queue<pair<int, int>> pq;
    for (int i = 1; i <= N; ++i) {
        // Only push vertices with at least one incident edge
        if (deg[i] > 0) {
            pq.push({deg[i], i});
        }
    }

    // Fill used array with false initially
    fill(used, used + N + 1, false);

    // Greedy Vertex Cover Construction
    while (!pq.empty()) {
        pair<int, int> top = pq.top();
        pq.pop();

        int d = top.first;
        int u = top.second;

        // Check validity of the popped entry:
        // 1. If 'u' is already selected for VC, skip.
        // 2. If 'd' (stored degree) differs from current 'deg[u]', it's a stale entry.
        //    Since degrees only decrease, 'd > deg[u]' means it's outdated.
        //    'd < deg[u]' is impossible in this logic.
        // 3. If 'deg[u]' is 0, it has no uncovered edges and shouldn't be picked.
        if (used[u] || d != deg[u] || deg[u] == 0) {
            continue;
        }

        // Select vertex 'u' for Vertex Cover
        used[u] = true;

        // Remove all edges incident to 'u'. This involves decrementing the degree of neighbors.
        for (int v : adj[u]) {
            deg[v]--;
            // If 'v' still has uncovered edges, update its position in the heap.
            // We push the new degree. Duplicate entries are handled by the check 'd != deg[v]'.
            if (deg[v] > 0 && !used[v]) {
                pq.push({deg[v], v});
            }
        }
    }

    // Output the Independent Set
    // x_i = 1 if vertex i is NOT in Vertex Cover (Independent Set), else 0
    for (int i = 1; i <= N; ++i) {
        cout << (used[i] ? 0 : 1) << "\n";
    }

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=1581)

````
Let me think about this problem. The challenge is to find a maximum independent set in a graph, which is NP-hard. However, the scoring formula gives us an interesting incentive structure:

Score = (N - K*) / (N - K) * 100

If we can find the optimal solution (K = K*), then Score = 100%.
If our solution is suboptimal, the score decreases as K increases relative to K*.

The key insight is that the **degree of freedom** for vertex i is N - d_i, where d_i is its degree (number of edges connected to it). A higher degree vertex has fewer ways to be part of an independent set (because including it restricts more neighbors), so it's often better to exclude high-degree vertices.

One effective strategy is greedy with randomization:
1. Sort vertices by decreasing degree
2. For each vertex in order, try adding it to the solution if possible
3. Use multiple restarts (random shuffling) to avoid local optima

Let me implement this approach:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

int main() {
    int N, M;
    cin >> N >> M;
    
    vector<vector<int>> adj(N + 1);
    vector<int> degree(N + 1);
    vector<pair<int, int>> edges(M);
    vector<int> p(N); // Permutation for ordering
    
    for(int i = 0; i < N; i++) p[i] = i + 1;
    
    for(int i = 0; i < M; i++){
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
        degree[u]++;
        degree[v]++;
        edges[i] = {u, v};
    }
    
    bool best_valid[50005] = {0};
    int max_count = 0;
    
    double best_score = -1.0;
    double score = -1.0;
    
    int num_trials = 10;
    for (int trial = 0; trial < num_trials; trial++) {
        random_shuffle(p.begin(), p.end());
        
        int count = 0;
        bool valid = true;
        
        for(int i : p){
            bool can_take = true;
            for(int neighbor : adj[i]){
                if(best_valid[neighbor]) {
                    can_take = false;
                    break;
                }
            }
            
            if(can_take){
                count++;
                best_valid[i] = true;
            } else {
                best_valid[i] = false;
            }
        }
        
        for(int e_idx = 0; e_idx < M; e_idx++){
            if(best_valid[edges[e_idx].first] && best_valid[edges[e_idx].second]){
                valid = false;
                break;
            }
        }
        
        if(valid){
            score = 100.0 * max(1, N - count) / max(1, N - max_count);
            if(score > best_score){
                best_score = score;
                max_count = count;
            }
        }
    }
    
    for(int i = 1; i <= N; i++) cout << best_valid[i] << endl;
}
```

This uses greedy selection with degree-based heuristics across multiple randomized trials to maximize independent set size while maintaining validity through conflict checking. I notice there are a few potential issues with my current approach. Let me trace through some cases to verify correctness and see where improvements might be needed.

The greedy approach selects vertices greedily based on whether they conflict with already-selected ones. With random reordering, I should get reasonable solutions quickly. Let me check the scoring logic though - when I select K vertices instead of the optimal K*, the penalty should scale appropriately. For instance, selecting only one vertex from four isolated nodes gives score 75% since I'm missing three out of the available selections. When K reaches N (the full graph), the denominator approaches zero and I should cap it at N-1 to avoid division issues. My implementation handles these edge cases with max() calls before division.

Actually, I realize there's still room for improvement. Let me reconsider the scoring carefully - if N=4, K*=2, K=3, then I'd want score = 1/1 * 100 = 100%, but currently I'm getting 2/(N-K) = 2. That suggests the scoring formula might be using (N-K*)/(N-K) directly rather than multiplying by 100, which would give different results than expected.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int N, M;
    cin >> N >> M;
    
    vector<vector<int>> adj(N + 1);
    vector<int> degree(N + 1);
    vector<pair<int, int>> edges(M);
    
    for(int i = 0; i < M; i++){
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
        degree[u]++;
        degree[v]++;
        edges[i] = {u, v};
    }
    
    double best_score = -1.0;
    int best_K = -1;
    vector<bool> selected(N + 1);
    
    for(int restart = 0; restart < 20; restart++){
        vector<int> p;
        for(int i = 1; i <= N; i++) p.push_back(i);
        
        sort(p.begin(), p.end(), [&](int a, int b){
            return degree[a] < degree[b];
        });
        
        fill(selected.begin(), selected.end(), false);
        
        for(int v : p){
            bool ok = true;
            for(int neighbor : adj[v]) {
                if(selected[neighbor]) {
                    ok = false;
                    break;
                }
            }
            if(ok) selected[v] = true;
        }
        
        bool valid = true;
        int K = 0;
        for(int i = 1; i <= N; i++) if(selected[i]) K++;
        
        for(auto& e : edges) {
            if(selected[e.first] && selected[e.second]) {
                valid = false;
                break;
            }
        }
        
        if(valid) {
            double sc = ((double)(N - best_K)) / max(1, (double)(N - K)) * 100.0;
            if(sc > best_score) {
                best_score = sc;
                best_K = K;
            }
        }
    }
    
    for(int i = 1; i <= N; i++) cout << selected[i] << "\n";
}
```
````
