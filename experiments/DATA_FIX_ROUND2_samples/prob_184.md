# 题 184 — base均分 56.3 vs 直接SFT均分 0.8(掉点 55.5)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=4,得分 85.4,completion_tokens=12940)

````
The problem asks me to find an independent set in a given undirected graph. An independent set is a set of vertices where no two vertices are adjacent. I need to output N lines, each containing either 0 or 1 to indicate whether vertex i is selected.

Key constraints:
- N ≤ 1000 (number of vertices)
- M ≤ 500,000 (number of edges)
- Time Limit: 2.0s
- Memory Limit: 512MB

The problem explicitly states this is a **heuristic optimization problem**. It says "You are NOT required to find the theoretically optimal solution." This is crucial because finding the Maximum Independent Set is NP-hard. However, for competitive programming with these constraints, there might be a specific heuristic or algorithm expected that performs well on average cases.

Since K* (the optimal size) is not known and we want to maximize K/K*, we need a good approximation algorithm.

Given N ≤ 1000, exponential algorithms like $O(2^N)$ will time out. Standard exact algorithms for Max Independent Set often run in $O(1.1996^n)$ using color-coding or similar techniques, which is too slow for N=1000.

However, looking at the problem description again: "This is a heuristic optimization problem... Instead, you should try to maximize the size of the set S". This suggests I don't need the perfect answer, just a large one. A greedy approach is usually simple but might not perform well. Simulated annealing, local search, or branch-and-bound heuristics are common approaches for this type of problem in competitions where K* is hidden.

Let's consider some approaches:
1.  **Greedy**: Pick a vertex with minimum degree, add it to S, remove it and its neighbors. Repeat. Or pick max degree and exclude. Or random permutation.
    -   Greedy by picking min-degree node: If we pick a node with low degree, we leave more freedom for other nodes. This is related to the kernelization of MWIS on general graphs (approximation ratio doesn't hold well for max, but often gets decent results). Actually, standard greedy (pick arbitrary node, pick if possible) is bad. The better greedy is picking nodes based on degrees.
    -   Algorithm: Sort vertices by degree ascending. Iterate through them. If current vertex $v$ has no neighbor currently in consideration (i.e., already added to set or removed), mark as part of IS. But wait, once we mark $v$, we must mark all its neighbors as forbidden? No, independent set means no two selected. So if I pick $v$, I cannot pick any neighbor.
    -   Revised Greedy Strategy:
        1.  Build adjacency list.
        2.  Compute degrees.
        3.  While vertices remain:
            Pick a vertex $u$.
            Check if any of $u$'s neighbors have been picked. If so, discard $u$.
            If not, pick $u$, increment count, mark neighbors as unavailable.
    -   Which order? Randomized greedy often works better than sorted greedy to avoid getting stuck in early choices that block many later options (though blocking neighbors immediately is correct for the constraint).
    -   Better strategy for general graphs (often called Maximal Independent Set):
        Start with empty set. Repeatedly pick an unpicked vertex $u$ such that none of its neighbors are in the set? No, that's finding an independent set.
        Actually, a classic randomized local search or simply picking high/low degree nodes iteratively.
        
    Let's look at the constraints again. $N \le 1000$. This is quite small for heuristic purposes. Even a slightly complex heuristic might pass.
    
    Common Heuristic for MIS (Maximum Independent Set):
    -   **Randomized Hill Climbing / Local Search**:
        1. Initialize $S$ randomly. Ensure validity.
        2. Try to flip a variable from 1 to 0 to resolve conflicts.
        3. Try to flip a variable from 0 to 1 if it increases score and maintains validity.
        4. Repeat until convergence.
    -   **Simulated Annealing**:
        Similar to local search but allows occasional invalid moves to escape local optima.
    -   **DSatur (Degree of Saturation)**: Used for coloring, can be adapted.
    -   **Simple Greedy with Shuffle**:
        Shuffle vertices. For each vertex in order, if it has no neighbors in the chosen set, choose it.
        Then maybe repeat multiple times with different shuffles and keep the best result.
        
    Given the scoring system depends on K* (unknown), we really just want K to be as close to K* as possible. Since K* is unknown, we can't check optimality. We just need valid sets with large cardinality.
    
    A very effective simple heuristic for Max Independent Set is **randomized greedy**:
    Run the process $T$ times (e.g., $N$ or $10N$ iterations):
    1. Copy adjacency lists to allow modification efficiently or rebuild.
    2. Shuffle the list of vertices.
    3. Maintain a boolean array `selected` initialized to false.
    4. Iterate through shuffled vertices $u$:
       If $u$ is not selected AND no neighbor of $u$ is selected:
          Select $u$ (`selected[u] = true`).
          Mark all neighbors of $u$ as unselectable? Wait, if we select $u$, we can never select any neighbor of $u$ in future steps. So we effectively remove $u$ and its neighbors from the pool of candidates.
          Optimization: Maintain a set of available vertices. When selecting $u$, remove $u$ and all neighbors $v$ from available set.
    5. Count selected vertices.
    6. Keep track of the maximum size found across runs.
    
    This guarantees a valid Independent Set (Maximal Independent Set). Is it Maximum? Not necessarily. However, running it multiple times with random permutations helps explore the space.
    
    With $N=1000$, doing $O(N \cdot (N+M))$ per run is too slow if we do too many runs. $M$ can be up to 500,000. Removing neighbors takes time proportional to degree. Sum of degrees over all operations is $M$. One full run is $O(M + N)$ if done carefully?
    Let's analyze complexity:
    - Iterate vertices: $N$.
    - For selected vertex, iterate neighbors: Total edges $2M$.
    - Checking if neighbor selected: $O(1)$ lookup.
    So one full greedy run is roughly $O(M)$.
    Running it $K$ times: $O(K \cdot M)$.
    If $K=10$, $10 \times 500,000 = 5,000,000$ ops, which is very fast (well within 2 seconds).
    However, constructing the adjacency list takes $O(M)$. Rebuilding/shuffling for every run.
    
    Can we improve?
    Using a static structure where we just maintain state?
    State: `status[v]` = 0 (available), 1 (in set), 2 (forbidden).
    Shuffle vertex indices.
    Loop $i$ from 0 to $N-1$:
      $u$ = $p[i]$
      if `status[u]` == 0:
         // Check neighbors. Wait, checking neighbors one by one to see if they are status 1 or 2.
         // To do this efficiently without scanning all neighbors every time, we need to know if any neighbor is ALREADY selected.
         // In the loop, we build the set incrementally.
         
    Actually, the logic is:
    1. Shuffle vertices.
    2. For $u$ in shuffled:
       if $u$ not yet marked 2:
         Check if any neighbor $v$ of $u$ is marked 1.
         If yes -> mark $u$ as 2 (forbidden).
         If no -> mark $u$ as 1 (selected).
    Problem: Checking neighbors requires iterating over the adjacency list.
    In worst case $N=1000, M=500,000$ (dense-ish). Average degree $\sim 500$.
    Iterating all vertices and checking neighbors: $\sum \deg(u) = 2M$.
    So one run is $O(M)$.
    With $M=500,000$, 100 runs would be $5 \times 10^7$ operations, which fits easily in 2.0s.
    
    However, checking "any neighbor is marked 1" requires iterating the whole adjacency list.
    Can we optimize?
    Maybe pre-marking?
    If I use a `visited` array for the neighbors?
    
    Optimized Greedy Logic:
    1. Reset `is_independent[1..N]` = false, `is_available[1..N]` = true.
    2. Shuffle vertex order `order`.
    3. For `u` in `order`:
       if `!is_available[u]` continue.
       bool conflict = false;
       for `v` in adj[u]:
           if `is_independent[v]` { conflict = true; break; }
       
       if (!conflict) {
           `is_independent[u] = true`;
           `is_available[u] = false`;
           for `v` in adj[u]:
               `is_available[v] = false`; 
       }
       
    Wait, the condition for conflict is: "is there any neighbor that is ALREADY IN THE SET".
    Because we process sequentially, neighbors processed earlier are checked against. Neighbors processed later don't care about us yet (we make our choice now). If a neighbor appears later in the order, when we get to it, we will check if *it* sees us selected.
    So:
    `is_independent` tracks the set we built so far.
    `is_available` tracks vertices that haven't been skipped due to being a neighbor of someone else in the past. Actually, simpler: `is_selected` tracks members of S.
    
    Correct Logic for Greedy (Maximal IS):
    1. Init `S = {}`, `blocked = []`.
    2. Vertices $V$.
    3. Shuffle $V$.
    4. For $u$ in $V$:
       if $u$ is blocked, skip.
       check neighbors of $u$: if any $v \in S$, then $u$ becomes blocked.
       else add $u$ to $S$, block all neighbors of $u$ (add to blocked set).
    
    Complexity of step 4 check: Iterate neighbors.
    If $u$ is added, we iterate neighbors again to block them.
    Each edge $(u,v)$ contributes to processing at most twice (once for $u$ checking $v$, once for $v$ checking $u$?). No.
    If $u$ is picked, we scan neighbors. If neighbor is picked, we stop.
    If neighbor not picked, we eventually pick it or block it.
    Essentially each edge is visited constant times. Total complexity $O(M)$ per run.
    
    Given N=1000, M=500,000, $O(M)$ is fine.
    To boost performance, we should run this greedy strategy multiple times with different random seedings/orderings and take the best $S$.
    How many iterations?
    Since N is small (1000), running 50-100 iterations should provide a robust result.
    100 * 500,000 = 50,000,000 operations. In C++, this is typically < 0.2s.
    
    Can we improve further?
    Maybe use "Local Search" to refine the greedy solution.
    Take a maximal independent set (from greedy). Try to swap vertices to increase size.
    Since maximizing Independent Set is hard, greedy is the baseline. Local search adds value.
    But with 2 seconds, pure randomized greedy repeated 100 times is probably sufficient for a "Heuristic" challenge. Most judges accept good approximations.
    Given the prompt "You are NOT required to find the theoretically optimal solution", efficient heuristics are preferred over complex ones that risk TLE.
    
    Wait, there's another common heuristic:
    **Coloring Heuristic**. The chromatic number $\chi(G)$ satisfies $\alpha(G) \ge N / \chi(G)$. If we color the graph greedily (DSATUR), we get a valid coloring with $k$ colors. The set of vertices assigned each color forms an independent set. We can pick the largest color class.
    DSATUR is generally good.
    However, building a coloring involves ordering nodes by saturation degree.
    This is also roughly $O(N^2)$ or $O(N \log N)$ depending on implementation.
    Given $N=1000$, $O(N^2)$ is $10^6$, very cheap. Even with adjacency traversal ($M$ edges involved in updates), it's comparable.
    Actually, greedy coloring takes $O(N + M)$.
    The size of the largest color class gives a valid Independent Set.
    However, the greedy coloring produces a valid coloring, meaning vertices of same color are non-adjacent.
    Does the largest color class correspond to a larger independent set than the greedy MIS construction?
    Usually, Greedy Coloring is used to approximate Independent Set by partitioning V.
    Specifically: Size of IS $\ge$ Size of Largest Color Class.
    Greedy coloring (using min-degrees or saturation) is better than just random shuffle.
    BUT, the problem is specifically finding MAX independent set. The largest color class might be small.
    Example: Star graph. Center connected to leaves.
    Color center (color 1). Leaves can be color 2.
    IS: {leaves} size N-1.
    Greedy coloring might put all leaves in color 2 if order is Center first?
    Wait, Center (1), Leaves (2). Color classes: {Center}, {Leaves}.
    Max of these is {Leaves}, size N-1. Correct.
    What if we pick leaves first? Leaf 1 (1), Center (2), Leaf 2 (1)...
    Color 1: All leaves. Color 2: Center. Max IS: All leaves.
    Generally, $\alpha(G) + \tau(G) = N$ (vertex cover)? No.
    The relation $\alpha(G) \ge N/k$ where k is chromatic number.
    So if we minimize k, we get a lower bound.
    
    Let's stick to the "Repeated Randomized Greedy" approach but refined.
    Why? Because building the Independent Set directly by selection usually yields better results than taking a subset of a coloring partition (which partitions V into many disjoint sets).
    Wait, actually, any maximal independent set can be constructed by this greedy.
    Repeating it is key.
    
    Implementation Details:
    - Adjacency list `vector<int> adj[N+1]`.
    - Function `solve()` returns vector<int> of size N.
    - Inside `solve`:
      - Create copy of input graph? Or just work on a working boolean array and reset?
      - Working arrays: `used[N+1]` (bool, if in IS), `blocked[N+1]` (bool, if neighbor of IS).
      - Wait, if `used[v]` is true, `blocked` isn't needed separately, just check `used`.
      - Actually `blocked` helps us skip vertices quickly. `status[u]`: 0=free, 1=in_IS, 2=blocked_by_neighbor.
      - Initialization: fill all with 0.
      - Loop through vertices.
    
    Optimization:
    Use `std::mt19937` for randomness.
    Loop X times (e.g. 50).
    Shuffle vertices.
    Process.
    Record max size.
    
    Wait, is $M$ really up to 500,000? Yes. $N=1000$.
    This implies density $\approx M/N \approx 500$.
    Dense graphs make greedy harder? Or easier?
    In dense graphs, few nodes can be picked. Max IS size is small.
    In sparse graphs, Max IS size is large.
    Greedy works reasonably well.
    
    Is there a risk of Time Limit Exceeded?
    100 runs * 500,000 edges = 50 million ops.
    C++ handles ~10^8 ops/sec conservatively. 50 million is safe.
    However, memory access patterns matter. `adj` vector is accessed frequently.
    Using a single vector for edges and indexing might be faster than `vector<vector<int>>` but for N=1000 overhead is negligible.
    
    Let's refine the loop count.
    Is 100 enough?
    Maybe 10 is risky. 50-100 is safer.
    Can we make it adaptive?
    We can also implement a simple local improvement phase.
    Once we have a candidate S, try to move nodes from outside S into S if they don't conflict, or remove conflicting nodes from S to add others.
    Given the strict instruction "Output exactly N lines", efficiency in outputting matters (just print).
    
    One detail: The scoring relies on K*. Since we can't calculate K*, we just need K to be large.
    Sometimes, removing low-degree vertices first is good, sometimes high-degree.
    Actually, the classic heuristic for MWIS is "Pick node with highest degree" -> Add to VC? No.
    For IS:
    1. **Greedy Min-Degree**: Pick min degree node. If no neighbors in current set (which is empty), pick it. Remove neighbors.
       This is equivalent to: Sort vertices by degree. Iterate. If $u$ has no neighbors selected yet, select $u$, mark neighbors as invalid.
       Wait, "neighbors of $u$ selected yet". If I process in sorted order, I only care if any neighbor *already processed* is selected? No, if I process $u$ now, any neighbor $v$ that comes *later* won't affect the validity of $u$ regarding the *current* set.
       But $v$ needs to know about $u$.
       So, the algorithm should be:
       List vertices.
       While List not empty:
          Pick $u$ from List.
          If $u$ is still a candidate (no neighbors in final S):
             Add $u$ to S.
             Remove all neighbors of $u$ from List.
          Else:
             Discard $u$.
          
          To implement efficiently:
          Maintain `candidates`.
          Initially `candidates` = all vertices.
          Priority Queue (by degree) to pick next vertex.
          Update degrees dynamically? Dynamic degree maintenance is costly ($O(M)$ or $O(\sqrt{E})$).
          Static degree sort + greedy is easier.
          
          Let's stick to the Randomized Shuffle approach. It explores the space better than static sorting which gets trapped in local optima determined by initial order.
    
    Refined Plan:
    1. Read N, M.
    2. Read edges. Store in adjacency list. Handle duplicate edges (multiset not needed for correctness, but store unique).
       Input might have duplicates. `vector<int> adj[N+1]` is fine. Duplicates just mean we check the same neighbor multiple times. To be clean, we can `sort` and `unique` adjacency lists after reading.
    3. Define a function `get_mis_random(seed)` that:
       - Shuffles `1..N`.
       - Initializes `in_s` array to false.
       - Initializes `visited_for_scan` (to avoid re-scanning neighbors unnecessarily, though checking `in_s` is sufficient).
         Wait, simpler:
         `status[v]` = 0 (available), 1 (taken/is_blocked).
         Iterate `u` in shuffled:
           If `status[u] == 0`:
             Check neighbors `v`. If any `in_s[v]` is true, then `status[u] = 1` (skip).
             Else:
               `in_s[u] = true`
               For `v` in neighbors `u`:
                 `in_s[v] = true`? NO.
                 Wait, `in_s` marks vertices IN the set.
                 If I pick $u$, I cannot pick any neighbor. So any neighbor $v$ becomes INVALID for the set.
                 Does it matter if I mark them `true` in `in_s`?
                 If I mark them `true`, when I reach $v$ later, `status[v]` check?
                 Let's use distinct statuses.
                 `state[v]`: 0 = Free, 1 = Selected, 2 = Forbidden.
                 
                 Iterate `u`:
                   If `state[u] != 0`: continue.
                   
                   bool ok = true;
                   for `v` in adj[u]:
                      if `state[v] == 1` { ok = false; break; }
                   
                   if ok:
                      `state[u] = 1`
                   else:
                      `state[u] = 2`
                      // Note: if u is forbidden, we don't need to propagate because u itself is not selected, so it doesn't forbid anything new.
                      // ONLY selected nodes forbid their neighbors.
                      // Wait, if I process `u` and it's forbidden, it stays forbidden.
                      // If I process `u` and select it, I MUST forbid all neighbors?
                      // Actually, `state[v] == 1` check covers that IF `v` was already processed.
                      // If `v` hasn't been processed, it is `state[v]=0`.
                      // Later when we visit `v`, we will check `state[u]`. Since `state[u]=1`, we reject `v`.
                      // So we DO NOT need to explicitly mark neighbors as forbidden immediately to prevent them from being selected LATER in THIS iteration.
                      // Just checking `if (state[v] == 1)` is enough!
           
                       // Wait, this logic is:
                       // We iterate a fixed permutation P.
                       // For u in P:
                       //    if u is compatible with {x | x in P before u}:
                       //         add u to IS
                       
                       // Is this correct?
                       // Suppose edges (1,2). P = [2, 1].
                       // Visit 2: Compatible. Add to IS.
                       // Visit 1: Neighbor 2 is in IS. Reject.
                       // Result IS={2}. Size 1. Correct (size 1). Max is 1.
                       
                       // P = [1, 2].
                       // Visit 1: Compatible. Add to IS.
                       // Visit 2: Neighbor 1 is in IS. Reject.
                       // Result IS={1}. Size 1.
                       
                       // This generates a MAXIMAL Independent Set, but not necessarily MAXIMUM.
                       // By shuffling P, we get different maximal sets.
                       // This is the standard way to solve MWIS via heuristics.
                       
    4. We also need to count size.
    5. Keep global max.
    
    Number of rounds:
    With N=1000, maybe 20-50 rounds.
    If M=500,000, checking neighbors for every node might take time.
    Sum of checks over one round = $\sum_{u \in S} \deg(u) + \sum_{u \notin S} \deg(u) \text{ (checked once)} = \sum_{all v} \deg(v) = 2M$.
    So cost is O(M).
    Total cost = Rounds * M.
    Rounds = 50 -> 25,000,000 operations.
    Safe.
    
    Let's write the code.
    Include necessary headers.
    Fast I/O.
    Main loop.
    Output formatting.
    
    Corner cases:
    - N=2, M=1.
    - Disconnected components.
    - Multiple edges between same pair.
    - Self loops? Problem says $u \neq v$.
    
    Handling Multiple Edges:
    Adjacency list will contain duplicates.
    When checking neighbors, we might see same neighbor twice. No harm, just redundant checks.
    Can optimize with `sort` and `unique` on `adj[u]` after reading input. Cost negligible.
    
    Code Structure:
    ```cpp
    #include <iostream>
    #include <vector>
    #include <algorithm>
    #include <random>
    #include <chrono>

    using namespace std;

    int main() {
        ios_base::sync_with_stdio(false);
        cin.tie(NULL);

        int N, M;
        if (!(cin >> N >> M)) return 0;

        vector<vector<int>> adj(N + 1);
        for (int i = 0; i < M; ++i) {
            int u, v;
            cin >> u >> v;
            adj[u].push_back(v);
            adj[v].push_back(u);
        }

        // Optional: Clean adjacency list to reduce checks
        for(int i=1; i<=N; ++i) {
            sort(adj[i].begin(), adj[i].end());
            adj[i].erase(unique(adj[i].begin(), adj[i].end()), adj[i].end());
        }

        int best_count = 0;
        vector<int> best_is(N + 1, 0);

        // State tracking
        // We can reuse vectors inside loop
        
        mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
        
        // Number of tries
        int tries = 100; 

        for (int t = 0; t < tries; ++t) {
            vector<int> p;
            p.reserve(N);
            for(int i=1; i<=N; ++i) p.push_back(i);
            
            shuffle(p.begin(), p.end(), rng);
            
            // current_set flags
            vector<int> status(N + 1, 0); // 0: free, 1: selected, 2: rejected
            
            int current_count = 0;
            
            for (int u : p) {
                if (status[u] != 0) continue;
                
                bool can_select = true;
                for (int v : adj[u]) {
                    if (status[v] == 1) {
                        can_select = false;
                        break;
                    }
                }
                
                if (can_select) {
                    status[u] = 1;
                    current_count++;
                } else {
                    status[u] = 2; // explicitly reject, though continue skips it
                }
            }
            
            if (current_count > best_count) {
                best_count = current_count;
                best_is.assign(1, 0); // resize/clear
                // Reconstruct best_is from status?
                // Just copying status to best_is is better
                best_is.clear();
                best_is.resize(N + 1, 0);
                for(int i=1; i<=N; ++i) {
                    if (status[i] == 1) best_is[i] = 1;
                }
            }
        }
        
        for (int i = 1; i <= N; ++i) {
            cout << best_is[i] << "\n";
        }

        return 0;
    }
    ```
    Wait, I need to check one optimization.
    Inside the inner loop (checking neighbors), if `adj[u]` is large, we iterate all.
    If `status[u]` is checked against neighbors.
    If we simply rely on `status` being 0, 1, or 2.
    Wait, does `status[v] == 1` check suffice?
    Yes. If $v$ was selected in a previous step, it remains 1. If $v$ was rejected, it is 2. If free, 0.
    If $v$ is a neighbor of $u$ and $v$ was selected previously, $u$ cannot be selected.
    If $v$ was rejected previously, $u$ CAN be selected (since rejection of $v$ means $v$ didn't select anyone else, or rather $v$ wasn't in the set, so no conflict with $v$).
    Wait, rejection of $v$ means $v$ was skipped. Why? Because $v$ had a neighbor selected BEFORE $v$.
    This does not imply anything about compatibility with $u$. $v$'s existence is irrelevant if it's not in the set.
    So checking `status[v] == 1` is the correct and necessary condition for compatibility.
    
    Is `tries = 100` enough?
    If we want to be safer against TLE, reduce tries to 50 or 40.
    If we want higher quality, increase tries or add local search.
    However, 2.0s is generous for 50 million ops in C++.
    I will stick with 100 to maximize score likelihood.
    
    Another thought:
    Should I use `status` or `vector<int>` `is_independent`?
    Using integers 0, 1, 2 is fine.
    Actually, we can just use `bool in_is[N+1]`.
    And `bool blocked[N+1]`?
    No, `status` logic above handles both. `status[u]==1` means in IS. `status[u]==2` means excluded (forbidden).
    Wait, if I set `status[u]=2`, it effectively excludes it from future consideration.
    In the loop: `if (status[u] != 0) continue;` handles skipping.
    
    Wait, there's a logical gap.
    My logic:
    `if (status[u] == 0)` -> process.
    Check neighbors.
    If OK -> `status[u] = 1`.
    If NO -> `status[u] = 2`.
    
    Is it possible that `status[u]=2` prevents a neighbor $w$ from being considered later?
    No, because $w$ will be processed independently. If $u$ and $w$ are neighbors, and $u$ is rejected ($status[u]=2$), then when we process $w$, we check `status[u]`. It will be 2. Since `status[u] != 1`, we don't reject $w$ based on $u$.
    This is correct. $w$ can join IS if $u$ is not in IS.
    
    One catch:
    If $u$ is rejected because $z$ was in IS, then $u$ is definitely not in IS.
    But $u$ blocks nothing?
    Wait, if $u$ is not in IS, it doesn't block its neighbors.
    So my logic holds.
    
    One potential issue: `status` array needs to be reset for each try.
    Allocation inside loop `vector<int> status(N + 1, 0);` is fast enough?
    Doing this 100 times with size 1001 allocs is trivial.
    But declaring outside and `fill` might be slightly faster. `vector<int> status(N+1);` outside, `iota` or `fill`.
    Actually, since I'm writing a script, local vector allocation is optimized well by modern compilers. But let's keep it outside to be sure.
    
    Also, `adj` cleaning with `unique` is good practice.
    
    Double check constraints:
    N=1000, M=500,000.
    My complexity: Tries * (N + M).
    100 * 600,000 ≈ 6e7 ops.
    In C++, 6e7 simple integer ops is < 0.5s.
    Memory is negligible.
    
    Any chance of Stack Overflow? Recursion? None used.
    Heap usage? 500,000 ints for adj lists = 2MB. Fits in 512MB easily.
    
    Final check on problem statement:
    "Input Format... 1 ≤ M ≤ 500,000".
    If M is small, M=N*(N-1)/2 could be 500k for N=1000? Yes.
    Complete graph K1000 has ~500k edges.
    In complete graph, Max IS size = 1.
    Greedy with shuffle will pick one, then block all. Result 1. Correct.
    
    Graph with no edges (M=0).
    Greedy picks all N. Correct.
    
    Graph with isolated vertices. Picks all isolated. Correct.
    
    What about a star graph?
    Center 1, Leaves 2..N.
    Shuffle:
    Case 1: 1 comes first. Pick 1. Block all leaves. Count = 1.
    Case 2: 1 comes last. First N-1 leaves come up. They check center. Center is 0 (not picked yet). So leaf picked. Mark center as blocked? No, my logic:
    Wait, my logic: `status[v] == 1`.
    If I pick leaf 2. `status[2] = 1`.
    Neighbor is center 1. `status[1]` is 0.
    Do I update `status[1]`?
    Wait!
    If I pick leaf 2, then center 1 is FORBIDDEN for the rest of the process?
    In my code:
    ```cpp
    if (can_select) {
        status[u] = 1;
        current_count++;
    } else {
        status[u] = 2;
    }
    ```
    I am NOT updating `status` for neighbors of `u`.
    This means if I pick leaf 2, `status[1]` remains 0.
    When I process leaf 3 later: check neighbors (center 1). `status[1]` is 0. Pick leaf 3.
    ...
    Process leaf N. Pick leaf N.
    Finally Process center 1.
    Check neighbors (leaf 2). `status[leaf 2]` is 1.
    Conflict! Cannot pick center.
    Result: Leaves selected. Count = N-1.
    Max IS = N-1.
    Correct.
    
    Wait, what if the graph is disconnected?
    Greedy finds max IS for component? No, just local decisions.
    With shuffling, we mix components.
    Usually okay.
    
    The critical observation in my manual trace above:
    If I do NOT update neighbors' status, it works fine because the check `status[v] == 1` happens when visiting `v`.
    However, there is one nuance.
    Does `status[u]` need to be set to something else to indicate "blocked" for subsequent processing?
    The loop structure:
    `for (int u : p) { if (status[u] != 0) continue; ... }`
    Here, if I pick `u` (`status[u]=1`), next time I encounter `u`?
    Wait, `p` contains unique vertices. Each vertex visited exactly once in the loop order `p`.
    So `if (status[u] != 0)` is basically "Have we already decided on `u`?".
    Wait, why?
    Ah, if I modify `status` of neighbors inside the loop?
    In the previous code snippet:
    ```cpp
    if (can_select) {
        status[u] = 1;
        // NO update to neighbors
    }
    ```
    Since each `u` in `p` is visited once, `status[u]` is 0 at start of body.
    We change it to 1 or 2.
    Since `u` is not in `p` again, it doesn't matter what it is anymore.
    It matters for its neighbors $v$.
    When we visit $v$ later in `p` (since `p` contains ALL vertices), we check `status[u]`.
    If `status[u]` is 1, $v$ sees conflict.
    If `status[u]` is 2 (rejected), $v$ sees no conflict with $u$.
    So NOT updating neighbors explicitly is CORRECT AND NECESSARY EFFICIENCY for this specific loop structure.
    
    Wait, if `status[u]` is 2 (rejected), it means $u$ has a neighbor in IS.
    Does $u$ impose restrictions on other neighbors? No.
    Only selected nodes restrict neighbors.
    So yes, logic holds.
    
    Wait, `if (status[u] != 0) continue;`?
    Is it possible for `status[u]` to be modified by someone else?
    In my proposed logic:
    `for (int u : p)`
       ... `status[u] = 1` or `2`.
    Neighbors $v$ are not visited until later.
    Their `status[v]` is checked. It reads `status[u]`? No, it checks `adj[v]` for `u`.
    It reads `status[u]`.
    But `status[u]` was modified when we were at step $u$ in loop `p`.
    Wait, if $u$ is visited AFTER $v$ in `p`.
    At step $v$, `u` is not processed yet. `status[u]` is 0.
    So $v$ checks `status[u]`. It is 0. So no conflict.
    $v$ selects itself. `status[v] = 1`.
    Later at step $u$, `status[u]` is 0.
    Check neighbors. $v$ is neighbor. `status[v]` is 1.
    Conflict. $u$ rejected.
    Correct.
    
    So we don't need to update `status` of neighbors. Just setting `status[u]` is enough.
    And since `p` iterates once per vertex, `if (status[u] != 0)` is technically redundant if we assume `status` starts at 0 and we only process `u` once?
    Ah, `status` is initialized to 0.
    If I modify `status[u]` inside the loop, does it affect anything?
    Wait, `status[u]` is only read when visiting `u`.
    Is it possible `u` is visited in the loop multiple times? No, `p` is a permutation of 1..N.
    So the check `if (status[u] != 0)` is dead code in this specific implementation unless I modify `status` of `u` somehow from another thread? No.
    Unless I implement dynamic updates to `status` of neighbors?
    As established, I don't modify `status` of neighbors.
    So `if (status[u] != 0)` check is useless?
    Wait, if I don't check, I just process `u`.
    If I did mark neighbors, then I would need it.
    Without marking neighbors, `status[u]` is always 0 at entry.
    So I can just remove `if (status[u] != 0) continue;` or keep it for safety.
    Wait, wait.
    In the loop `for (int u : p)`, I iterate every vertex.
    `status[u]` is 0 initially.
    I execute logic.
    So yes, check is redundant.
    
    However, let's re-verify the "Shuffle" requirement.
    Randomization is crucial. `shuffle` does that.
    
    One edge case:
    Input graph might contain multiple edges.
    `adj` stores duplicates.
    `for (int v : adj[u])` will iterate duplicates.
    `if (status[v] == 1)` will be true if true.
    It just adds redundant checks. `unique` fixes this.
    
    Okay, looks solid.
    
    One tiny optimization:
    `best_count` update.
    Since we only need to output the solution, we need to save `best_is`.
    `vector<int> best_is(N+1, 0);` inside loop or copy `status`?
    Copy `status` is cheaper. `memcpy` or loop.
    
    Wait, I noticed I defined `status` inside the loop. That's fine.
    Allocating vector of size 1000 100 times is 100KB * 100 = 10MB bandwidth. Negligible.
    
    Is it worth trying to improve greedy with local search?
    Local search involves iterating through the IS, finding a vertex $u \in IS$ that can be replaced by $v \notin IS$ to increase size or resolve conflict?
    Usually "Flip": try to add a vertex from $V \setminus S$ if valid. If valid, add it. If adding causes conflicts, remove conflicting vertices.
    This is basically "Greedy" on top of current solution.
    Given time limit, extra code complexity might introduce bugs.
    Repeating the base heuristic (randomized greedy) 100 times is standard competitive programming tactic for NP-hard problems with small N.
    With 2 seconds, we have slack.
    I'll increase iterations to 200 just to be safe.
    200 * 0.6s? No. 200 * 5e7 ops?
    Wait, 5e7 ops per run. 200 runs -> 1e10 ops. Too slow.
    Wait, I miscalculated ops.
    Checking neighbors:
    $\sum_{u} \text{degree}(u) = 2M$.
    If graph is dense, M ~ 500k.
    Ops per run ~ 10^5 to 10^6 (depending on cache).
    Wait, 500k edges. Loop over edges is 500k iterations.
    Inside loop: check `status[v] == 1`. Array access. Very fast.
    Total ops ~ 500,000 * cycle_cost. Cycle cost maybe 1-5 cycles?
    So 1 million cycles? No.
    500,000 edges. Access memory.
    Modern CPU ~ 3GHz.
    500,000 instructions fit in milliseconds.
    Let's say 1 microsecond per edge (conservative).
    0.5ms per run.
    100 runs -> 50ms.
    1000 runs -> 500ms.
    So 1000 runs is feasible.
    Let's go with 200 or 500.
    I'll stick to 500 runs. Safety margin.
    
    Wait, is memory 512MB enough? Yes.
    
    Final check on headers.
    `<vector>`, `<iostream>`, `<algorithm>`, `<random>`, `<chrono>`.
    All standard.
    
    One tricky thing: "Output exactly N lines".
    Code outputs `cout << best_is[i] << "\n";`
    `best_is` is 1-based index. Loop 1 to N.
    Matches format.
    
    One detail: The problem says "Multiple edges between the same pair of vertices may appear but imply the same constraint."
    Handled by `unique` or ignored logic. `unique` makes code cleaner/faster.
    
    Complexity summary:
    $O(T \cdot (N + M))$
    $T = 500$
    $N = 1000$
    $M = 500,000$
    Ops $\approx 2.5 \cdot 10^8$.
    This is slightly high for 2.0s if constant factor is large.
    Wait, $2.5 \cdot 10^8$ might be pushing it for 2s (usually limit is $10^8$ ops per sec).
    But these ops are very simple.
    Vector iteration is sequential. Branch prediction works.
    However, `adj` is `vector<vector<int>>`. Inner vector iterator. Cache misses possible.
    To optimize, maybe flatten adjacency?
    Head-to-Tail linked lists or CSR format?
    Given 2.0s and $M$ up to 500k, $N=1000$ (dense), cache locality is poor anyway.
    Actually, for $N=1000$, the graph is likely dense.
    In dense graph, adjacency vectors are short (avg deg 500).
    Iterating 500 items is cache-friendly (L1 cache).
    Access pattern: Random access `status[v]`. `status` array size 1000 fits in L1.
    So `adj[u]` iteration loads vertices `v`. Random `status[v]`. `status` fits in cache.
    Should be very fast.
    Let's reduce Tries to 200 to be absolutely safe.
    
    Wait, I should double check the logic of "Status" vs "Blocked".
    My logic:
    `status[v] == 1` -> conflict.
    If I have a clique.
    Order 1..N.
    1 checks nothing. Pick 1. `status[1]=1`.
    2 checks 1. Conflict. Skip.
    3 checks 1. Conflict. Skip.
    ...
    Count = 1.
    
    If I have a path 1-2-3-4.
    Order 2, 1, 3, 4.
    2 picks. `status[2]=1`.
    1 checks 2. Conflict. Skip.
    3 checks 2. Conflict. Skip.
    4 checks 3 (status 0). Pick 4. `status[4]=1`.
    Count 2. ({2, 4}). Valid.
    Order 1, 3.
    1 picks.
    3 checks 1 (neighbor?). No edge (1,3).
    3 picks.
    2 checks 1. Conflict.
    4 checks 3. Conflict.
    Count 2. ({1, 3}). Valid.
    
    Looks correct.
    
    One optimization: `status` vector reuse.
    Move `vector<int> status(N + 1, 0);` outside loop.
    Inside loop: `fill(status.begin(), status.end(), 0);`
    Faster than reallocating.
    
    Wait, `std::shuffle` might produce bad sequences?
    `std::mt19937` is good. Seed with hardware clock or `random_device`.
    
    I think the solution is ready.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>

using namespace std;

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    // Adjacency list
    vector<vector<int>> adj(N + 1);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Optional optimization: remove duplicate edges
    for (int i = 1; i <= N; ++i) {
        sort(adj[i].begin(), adj[i].end());
        adj[i].erase(unique(adj[i].begin(), adj[i].end()), adj[i].end());
    }

    // State array reused to avoid reallocation
    vector<int> status(N + 1); 
    
    int best_count = 0;
    vector<int> best_solution(N + 1, 0);
    
    // Seeding random generator
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

    // Number of greedy iterations. 
    // With N=1000, M=500000, T=500 gives ~2.5e8 ops, feasible within 2s.
    const int NUM_TRIES = 500;

    for (int t = 0; t < NUM_TRIES; ++t) {
        // Reset status for new trial
        // Filling with 0 is fast enough for size 1001
        fill(status.begin(), status.end(), 0);

        // Generate random permutation of vertices
        vector<int> p;
        p.reserve(N);
        for (int i = 1; i <= N; ++i) p.push_back(i);
        shuffle(p.begin(), p.end(), rng);

        int current_count = 0;
        
        // Greedy selection based on permutation
        for (int u : p) {
            // status[u] should be 0 here.
            // Check if any neighbor is ALREADY SELECTED (status == 1)
            bool conflict = false;
            for (int v : adj[u]) {
                if (status[v] == 1) {
                    conflict = true;
                    break;
                }
            }

            if (!conflict) {
                // Select vertex u
                status[u] = 1;
                current_count++;
            } else {
                // Reject vertex u
                // Status remains whatever it is or set to 2 to indicate rejected/processed
                // Setting to 2 ensures if we revisit u (we don't in this loop logic) 
                // or for clarity. Actually just leaving as 0 is fine for 'processed'?
                // No, strictly speaking if we check status[v] later, 0 means free.
                // Since we only iterate p ONCE, u won't be revisited.
                // But we must ensure status[u] does not act as a blocker for others 
                // unless u is selected. If u is not selected, it blocks nothing.
                // So we don't need to set status[u] = 2 for correctness logic.
                // But keeping 0 is wrong? No, 0 means "Not yet selected/rejected in IS context"?
                // My logic: status[v] == 1 triggers conflict. status[v] == 0 does not.
                // If u is not selected (status[u] == 0), it acts like a non-existent vertex for future checks.
                // This is CORRECT. Non-selected vertices do not impose constraints.
            }
        }

        // Update best solution if current is larger
        if (current_count > best_count) {
            best_count = current_count;
            // Copy status to best_solution
            for (int i = 1; i <= N; ++i) {
                best_solution[i] = (status[i] == 1) ? 1 : 0;
            }
        }
    }

    // Output results
    for (int i = 1; i <= N; ++i) {
        cout << best_solution[i] << "\n";
    }

    return 0;
}
```
Wait, one subtle point in the comment logic:
"If u is not selected... status[u] == 0... acts like a non-existent vertex".
Yes, if `status[u]` is 0, `if (status[v] == 1)` is false. So no conflict.
So non-selected vertices are transparent to future selections.
This is correct behavior for an Independent Set.
If I marked `status[u] = 2` for rejected, it would still satisfy `!= 1`. So 2 is also fine. 0 is fine.
So logic stands.

One final check on `vector<int> p` allocation inside loop.
Allocating 1000 ints 500 times. 500 * 4KB = 2MB total allocation churn. Trivial.
Using `vector` constructor `vector<int> p(N); iota(p.begin(), p.end(), 1);` is slightly cleaner but push_back loop is fine.
Actually `iota` is in numeric. Added include.

Is there any case where `adj` is empty?
Yes M=0. Loop over `adj[u]` does nothing. `conflict` remains false. All picked. Correct.

Double check the `fill` function speed. `memset` is often faster but for `int` 0, `fill` is optimized.
Given time limit, `std::fill` is perfectly fine.

Final verification of problem statement:
"Score = K / K* * 100".
We maximize K.
Output: "Output exactly N lines."
Format: "i-th line ... integer x_i".
My code does this.

Constraints: N=1000.
Time limit 2.0s.
Memory 512MB.
Code uses ~several MBs.
Logic is sound for heuristic maximization.

One minor thing: The problem says "2 <= N <= 1000".
My code handles N>=1.
M >= 1.
Everything consistent.

I will formulate the response now.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <numeric>

using namespace std;

int main() {
    // Optimize I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    // Adjacency list
    // Using 1-based indexing for vertices
    vector<vector<int>> adj(N + 1);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Optimization: Remove duplicate edges to reduce checks
    for (int i = 1; i <= N; ++i) {
        sort(adj[i].begin(), adj[i].end());
        adj[i].erase(unique(adj[i].begin(), adj[i].end()), adj[i].end());
    }

    // State array to track vertex status during greedy selection
    // 0: free (available to be selected)
    // 1: selected (in Independent Set)
    // 2: processed/junk (unused logic simplifies to 0 or 1)
    vector<int> status(N + 1); 

    int best_count = 0;
    vector<int> best_solution(N + 1, 0);
    
    // Seeding random generator with hardware entropy
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

    // Number of greedy trials. 
    // With N=1000, M=500000, O(Tries * M) should be around 10^8 ops for 500 tries, fitting 2.0s.
    const int NUM_TRIES = 500;

    for (int t = 0; t < NUM_TRIES; ++t) {
        // Reset status array for new trial
        fill(status.begin(), status.end(), 0);

        // Generate random permutation of vertices
        vector<int> p;
        p.reserve(N);
        for (int i = 1; i <= N; ++i) p.push_back(i);
        shuffle(p.begin(), p.end(), rng);

        int current_count = 0;
        
        // Greedy selection based on current random permutation
        for (int u : p) {
            // Check if any neighbor is ALREADY SELECTED (status == 1)
            // Unselected neighbors (status 0) do not cause conflict.
            bool conflict = false;
            for (int v : adj[u]) {
                if (status[v] == 1) {
                    conflict = true;
                    break;
                }
            }

            if (!conflict) {
                // Select vertex u
                status[u] = 1;
                current_count++;
            } else {
                // Vertex u cannot be selected due to neighbors.
                // It effectively blocks nothing for future nodes since it's not in the set.
                // status[u] remains 0 (or could be set to 2, but 0 suffices logic-wise).
            }
        }

        // Update best solution if current independent set is larger
        if (current_count > best_count) {
            best_count = current_count;
            // Copy status to best_solution
            for (int i = 1; i <= N; ++i) {
                best_solution[i] = (status[i] == 1) ? 1 : 0;
            }
        }
    }

    // Output results
    for (int i = 1; i <= N; ++i) {
        cout << best_solution[i] << "\n";
    }

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=776)

````
The Maximum Independent Set problem is NP-hard, but for N up to 1000 with only 2 seconds allowed, a complete exact solution (like max-flow min-cut) would be too slow on general graphs. The key insight for this heuristic challenge is that independent sets are complementary to vertex covers: selecting an independent set S means excluding its complement V\S, where V\S forms a valid vertex cover. Since minimum vertex covers can be efficiently found using maximum matching in bipartite graphs via maximum flow, we can use this structure to approximate MIS by iteratively removing well-connected vertices from a candidate set and re-evaluating the remaining subgraph's independence number.
</think>

#include <bits/stdc++.h>
using namespace std;

// Fast I/O
ios_base::sync_with_stdio(false);
cin.tie(nullptr);

int main() {
    int N, M;
    cin >> N >> M;
    
    // Build adjacency list (vertices 1..N)
    vector<vector<int>> adj(N + 1);
    for (int i = 0; i < M; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    // Greedy construction of a large independent set
    // Iteratively remove the most "independent" vertex (lowest degree) from candidates
    // This biases toward low-degree nodes which are easier to include while maintaining independence
    vector<int> deg(N + 1), cand;
    iota(deg.begin(), deg.end(), 0);
    cand.push_back(0); // dummy placeholder
    
    for (int i = 1; i <= N; i++)
        if (!adj[i].empty()) cand.push_back(i);
    shuffle(cand.begin() + 1, cand.end());
    
    vector<char> inS(N + 1, 0);
    
    while ((int)cand.size() > 1) {
        char best = 0;
        size_t best_idx = 0;
        
        for (size_t i = 1; i < cand.size(); i++) {
            const int &u = cand[i];
            int neighbors_in_cand = 0;
            bool safe = true;
            
            for (const int &v : adj[u]) {
                if (cand.count((long long)v)) {
                    ++neighbors_in_cand;
                    if (inS[v]) {
                        safe = false;
                        break;
                    }
                }
            }
            
            // Prefer vertices with few active constraints and no conflicts
            if (safe && (best == 0 || neighbors_in_cand < deg[best])) {
                best = (char)1;
                best_idx = i;
            }
        }
        
        if (best) {
            const int &u = cand[best_idx];
            inS[u] = 1;
            for (const int &v : adj[u])
                if (cand.count((long long)v))
                    cand.erase(--cand.end(), find(cand.begin(), cand.end(), (long long)v));
        }
        
        cand.erase(cand.begin() + best_idx);
    }
    
    cout << (inS[1] ? 1 : 0) << '\n';
    for (int i = 2; i <= N; i++) cout << (inS[i] ? 1 : 0) << '\n';
    
    return 0;
}
```
````
