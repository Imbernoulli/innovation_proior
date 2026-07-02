# 题 180 — base均分 38.4 vs 直接SFT均分 5.7(掉点 32.7)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 84.0,completion_tokens=24019)

````
The user wants a C++ solution for the "Graph Isomorphism" problem.
This problem asks to find a permutation of vertices of graph $G_2$ that maximizes the number of edges mapped correctly to edges in graph $G_1$.
Basically, we want to maximize $\sum_{(u,v) \in E_2} I((p(u), p(v)) \in E_1)$, where $I$ is the indicator function.

Constraints:
$n \le 2000$.
$m$ can be up to $n(n-1)/2$, which is roughly $2 \times 10^6$.
Time limit is usually around 1-2 seconds for competitive programming problems, though not explicitly stated here, standard limits apply.
Memory limit is also not explicitly stated but usually fits within standard RAM.

Since finding the *exact* maximum score corresponds to the General Graph Isomorphism Problem (which is hard), solving it exactly is NP-hard in general cases, though practical solvers exist. However, since this is a "soft scoring rule" and the score is simply the ratio of matched edges, an approximate solution or a randomized local search heuristic might work better than exact exponential time algorithms given $N=2000$. Exact isomorphism algorithms like NAUTY are very good, but implementing them from scratch is too much code.

Let's consider the nature of the scoring. We want to match as many edges as possible.
If the graphs are regular, it's easier. If they are random, any mapping might yield $m/2$ on average.
With $N=2000$, an $O(N^3)$ or $O(N^4)$ algorithm might be too slow. $O(N^2)$ is preferred.

However, this specific problem statement mentions "evaluated with a soft scoring rule". This implies we don't necessarily need a perfect match, just a high one. A common technique for graph matching/heuristics is:
1. Calculate degrees of vertices in both graphs.
2. Only map vertex $i$ from $G_2$ to vertex $j$ from $G_1$ if their degrees are equal.
   Wait, maximizing degree-preserving matches is a strong heuristic. If degrees differ significantly, mapping them won't contribute to the edge count easily because neighbors won't match.
   Actually, strict degree preservation is required for perfect isomorphism. For partial isomorphism (maximizing matches), preserving degree is still a good heuristic because edges connect vertices of similar degrees.

Let's formalize the objective:
Maximize $\sum_{e=(u,v) \in E_2} \text{adj}_{G_1}(p(u), p(v))$.
This looks like the Maximum Common Subgraph problem, specifically induced or non-induced? Here it's non-induced matching of edges.
Wait, if $(u,v)$ exists in $G_2$ and maps to $(p(u), p(v))$, we gain a point. If $p(u)=p(v)$ is impossible (permutation).
Actually, since it's undirected edges, $(u,v)$ is same as $(v,u)$. The condition is $(p(u), p(v)) \in E_1$.

For small $N$, backtracking works. For $N=2000$, we need something smarter.
Given the constraints and the "soft scoring", maybe a greedy approach with some swapping (simulated annealing or simple hill climbing) could work.
Or, simpler:
Group vertices by degree in both graphs.
We only care about mapping a vertex of degree $d$ in $G_2$ to a vertex of degree $d$ in $G_1$ if we hope to preserve edges efficiently.
If we map a degree-$k$ vertex to a degree-$l$ vertex ($k \neq l$), at most $\min(k, l)$ incident edges can possibly match. Since we want to maximize matches globally, focusing on same-degree nodes is a strong pruning strategy.

Algorithm Idea 1: Bipartite Matching based on neighborhoods?
If we fix the mapping of one vertex, say $u \to v$, then all its neighbors must be mapped to neighbors of $v$.
Since $N$ is large, we can't explore full possibilities.
However, we can use a simplified version of the Weisfeiler-Lehman algorithm or just basic degree refinement.
Degree refinement sorts vertices into buckets based on degree.
Then, maybe try to match vertices within the same bucket.
But just mapping same-degree vertices greedily might lead to suboptimal results (local optima).

Let's look at the "score". It's just `matched_edges / m`.
If $G_1$ and $G_2$ are identical, score is 1.
If they are completely different, score 0.
With $N=2000$, maybe a randomized hill-climbing approach is best.
Start with an initial permutation (e.g., identity after degree sorting).
Calculate current score.
Try to swap two elements in the permutation.
If score improves, keep the swap.
Repeat until no improvement (or for fixed time).

Complexity of one check:
Iterate through edges of $G_2$ (size $M$). Check existence in $G_1$ using adjacency matrix or hash set.
Adjacency matrix for $G_1$ takes $O(N^2)$ space and $O(1)$ query. With $N=2000$, matrix is $4 \times 10^6$ bits/bytes, perfectly fine.
Calculating score takes $O(M)$.
Number of swaps? Potentially $O(N^2)$. Total $O(M \cdot N^2)$ is too much ($2 \cdot 10^6 \cdot 4 \cdot 10^6 \approx 10^{13}$).
We need faster updates.

Optimized Hill Climbing:
Current permutation $P$.
Score depends on edges of $G_2$.
Edge $e = (u, v)$ in $G_2$ contributes 1 if $(P[u], P[v])$ in $E_1$.
When we swap $P[i]$ and $P[j]$:
Only edges involving $i$ or $j$ change status.
Edges incident to $i$ in $G_2$: $\delta_{G_2}(i)$. Let neighbors be $N_2(i)$.
For each $w \in N_2(i)$, edge $(i, w)$ previously checked $(P[i], P[w])$. Now checks $(P[j], P[w])$.
Similarly for $j$.
So we only need to re-evaluate edges connected to $i$ and $j$ in $G_2$.
Total edges affected is roughly $deg_G2(i) + deg_G2(j)$.
Max degree is $N$. So update takes $O(deg(i) + deg(j))$.
In worst case $O(N)$.
If we do $K$ iterations, complexity $O(K \cdot N)$.
How many iterations? Maybe $1000 \dots 10000$?
We also need a good initial permutation.
Initial Permutation: Sort vertices of $G_1$ and $G_2$ by degree. Then try to align them.
Within same degree groups, use other properties (like sorted neighbor degrees) to order them.
This is basically degree refinement / WL pre-coloring.

Let's refine the initialization.
1. Compute degrees of all vertices in $G_1$ and $G_2$.
2. Create list of pairs `(vertex, degree)` for both.
3. Sort these lists primarily by degree.
4. Within the same degree, sort by a signature. A simple signature could be the sum of degrees of neighbors (or just list of neighbor degrees).
5. Map $u \in G_2$ to $v \in G_1$ such that their signatures match index-wise in the sorted list.
This gives a good starting guess. If they are isomorphic, this should give identity mapping.

Hill Climbing Steps:
1. Calculate current score with initial mapping.
2. Loop max_iter times (e.g. 2000):
    Best_score = current_score.
    Best_swap = none.
    For $i$ from 1 to $N$:
        For $j$ from $i+1$ to $N$:
            Swap $P[i], P[j]$ temporarily.
            Update score delta:
                Edges touching $i$: $\{ (i, k) \in E_2 \}$.
                Edges touching $j$: $\{ (j, k) \in E_2 \}$.
                Note: Edge $(i, j)$ if exists counted twice, needs care.
                For each neighbor $k$ of $i$ or $j$ in $G_2$:
                    old contribution: $(old\_mapping[k])$ vs $(old\_mapping[neighbor])$.
                    new contribution: ...
                Actually, simpler logic:
                Current contribution comes from $E_2$.
                Contribution is sum over $e \in E_2$ of $A_{G_1}(P[e.u], P[e.v])$.
                If we swap $u$ and $v$ in indices of $P$ (meaning $P[u]$ becomes $P[v]$'s old value)?
                Wait, permutation $P$ usually means $P[i]$ is image of vertex $i$ from $G_2$.
                Let's stick to definition: Output $p_1, ..., p_n$ where $p_i = j$ means vertex $i$ of $G_2$ goes to $j$ of $G_1$.
                So array `perm` of size $N+1$, where `perm[i]` stores target vertex in $G_1$ for vertex `i` in $G_2$.
                Score calculation:
                Iterate all edges $(u, v)$ in $G_2$. Add `adj1[perm[u]][perm[v]]`.
                
                Swapping `perm[i]` and `perm[j]`:
                Only edges $(i, k)$ and $(j, k)$ and $(i, j)$ in $G_2$ are affected.
                We can iterate all neighbors $k$ of $i$ and $j$ in $G_2$.
                Check if edge existed in $G_1$ before and after.
                Keep swap if score increases.
            
            Restore permutation.
            
    If no improvement found in outer loop passes, break or restart.

Optimization for $N=2000$:
Swapping loops are $O(N^2)$. Each check is $O(deg(i) + deg(j))$. Worst case $O(N)$. Total per iteration $O(N^3)$. Too slow.
We can improve the swap selection.
Instead of trying all pairs, pick random pairs, or pairs that have high probability of conflict resolution.
Or use "Simulated Annealing": occasionally accept worse swaps to escape local minima.
Or implement the "degree signature" more robustly during the greedy phase.
Given $N=2000$, $O(N^2)$ total operations is fine. $O(N^2)$ iterations of checking score? No.
We can afford to recalculate full score every few steps, but updating incrementally is key.

Incremental Update Logic:
Precompute `edges_in_G2[u]` as a list/vector of neighbors.
Maintain `current_score`.
When swapping `perm[i]` and `perm[j]`:
   Identify neighbors of `i` in $G_2$: $N_i$.
   Identify neighbors of `j` in $G_2$: $N_j$.
   For each $k \in N_i \cup N_j$ (and potentially $i, j$ themselves if connected):
       // Edge $(i, k)$
       Old contribution was `adj1[perm[i]][perm[k]]`.
       New contribution would be `adj1[perm[j]][perm[k]]`.
       If `adj1[perm[j]][perm[k]] == 1 && adj1[perm[i]][perm[k]] == 0`, delta += 1.
       Else if `adj1[perm[j]][perm[k]] == 0 && adj1[perm[i]][perm[k]] == 1`, delta -= 1.
       
       // Edge $(j, k)$ - careful not to double count if $k=j$. But $k \in N_i$, so distinct.
       // Also need to handle edge $(i, j)$ specifically if it exists.
       
   Wait, this is still potentially $O(N)$ per swap check.
   To make it fast, we need a better heuristic for picking swaps or limiting candidates.
   Usually, we just swap random elements. If we just run randomized local search with limited iterations, say 500-1000 passes, checking all $N(N-1)/2$ pairs is $2000^2 \approx 4 \times 10^6$, multiplied by $O(N)$ cost -> $8 \times 10^9$ ops. Too slow.
   Cost is proportional to degree. Average degree is $m/n$. If $m$ is small, it's fast. If $m \approx n^2/2$, cost is $O(n)$, so $O(n^3)$.
   
   We need to prune the search space.
   Most swaps will not increase the score.
   If we maintain the permutation such that degrees match (strict filtering), we reduce candidates.
   Better yet: construct a bipartite matching between vertices of $G_1$ and $G_2$ based on neighborhoods intersection size?
   
   Let's rethink.
   Is there an exact algorithm that works well enough?
   Given this is likely a CP problem with soft scores, often the test cases aren't pathological.
   Sometimes a simple "sort by degree, then map" works surprisingly well.
   But to beat random initialization, we need refinement.
   
   Let's try to optimize the "random pair swap" to $O(1)$?
   We can store the score contribution of each vertex.
   $S_u = \sum_{v \in Adj(G_2, u)} \mathbb{1}( (p(u), p(v)) \in E_1 )$.
   Total score is $\frac{1}{2} \sum S_u$? No, edges are counted once.
   When we change $p(i)$, only terms involving edges incident to $i$ change.
   Let $Neighbors_2(i)$ be the list of neighbors of $i$ in $G_2$.
   Current Score contribution from edges incident to $i$: $\sum_{v \in Neighbors_2(i)} A_1(p(i), p(v))$.
   Note: Summing over all vertices counts each edge twice.
   Let $Val(i)$ be the contribution of $i$ being at position $p(i)$ assuming other positions are fixed.
   $Val(i) = \sum_{v \in Neighbors_2(i)} A_1(p(i), p(v))$.
   But this value changes when other $p(k)$ change.
   
   Alternative: Randomized Local Search (Simulated Annealing variant).
   Start with Degree-based init.
   Run for fixed time. In each step, randomly pick a pair $(u, v)$ from $G_2$ vertices.
   Check swap impact.
   Impact computation: $O(deg(u) + deg(v))$.
   Average degree might be low. If dense, this is slow.
   However, usually in dense graphs, almost any permutation has many matches if graphs are isomorphic. If not, score is low anyway.
   Wait, if graphs are dense ($m \approx n^2$), computing score takes $O(n^2)$. We can do $O(1)$ updates if we track row sums?
   
   Actually, let's consider the constraints again.
   $N \le 2000$.
   Time limit usually 1s or 2s.
   We have 1 second for $10^8$ operations roughly.
   $O(N^2)$ is acceptable. $O(N^2 \log N)$ or $O(N^2 \sqrt N)$ borderline.
   $O(N^3)$ is definitely TLE.
   
   Can we solve it via Maximum Bipartite Matching?
   Construct a bipartite graph where left side is $V(G_2)$, right side is $V(G_1)$.
   Connect $u \in V(G_2)$ to $v \in V(G_1)$ with weight?
   Max flow doesn't directly solve Max Clique/Subgraph issues for graphs.
   But for isomorphism, we need to preserve adjacency.
   If we assume the permutation preserves degrees (very strong assumption), we restrict candidates.
   
   What if we just use the "Sort by Degree" heuristic as the primary strategy, and maybe a lightweight optimization?
   Actually, there's a technique called "Vertex Coloring Refinement".
   Use canonical labeling or colorings.
   We can assign colors to vertices iteratively.
   Round 0: Color by degree.
   Round 1: Color by (degree, sorted neighbor degrees).
   Round 2: ...
   Eventually, the coloring distinguishes vertices sufficiently.
   Then we can map vertices with same color tuple.
   This gives a deterministic and fast initialization that handles "structural" differences.
   If $G_1$ and $G_2$ are close, this will group them tightly.
   
   Let's implement the "Color Refinement" to initialize the permutation.
   Step 1: Compute degrees.
   Step 2: List of (color, id) for both graphs.
   Initial color: degree.
   Iteratively refine colors:
     For each vertex, collect multiset of neighbor colors.
     Form new color: (old_color, sorted_neighbor_colors).
     Update colors.
   Stop when colors stabilize (no change) or max iterations (say $\log N$ or 10).
   This usually produces a partition.
   Vertices with same final color sequence are candidates for each other.
   We create blocks. Within each block, we try to establish a bijection.
   Since $N=2000$, blocks should be small if graphs are distinct. If isomorphic, blocks are singletons or trivial.
   We can build the permutation greedily within blocks.
   
   After this initialization, we perform a local search (swaps between compatible vertices) to polish the result.
   "Compatible" meaning we stay within the same structural class if possible? Or just any swap?
   Just doing random swaps among those with same initial color (from degree) is a safe bet to not violate structure too much.
   Actually, allowing swaps across different refined color classes might help merge things if refinement got stuck.
   But refining helps us start near the optimum.
   
   Let's design the solution:
   1. Read input. Store adjacency lists and adj matrices.
      Adj Matrix is needed for $O(1)$ edge lookup. With $N=2000$, `bitset<2000> adj[2000]` or `bool adj[2001][2001]` is fast enough. Memory ~ 4MB for bool array.
   2. Initialize permutation based on Vertex Coloring (Degree + Neighbor Degrees).
      Let's just do degree + sum of neighbor degrees to save time on sorting multisets.
      Or simpler:
      Assign rank to each vertex based on (degree, sum of neighbors degrees, etc.).
      Map $u$ (in $G_2$) to $v$ (in $G_1$) such that their ranks are closest.
      Specifically:
      Vector $A$ of structs $\{id, score\}$ for $G_1$.
      Vector $B$ of structs $\{id, score\}$ for $G_2$.
      Sort both.
      For $k=0..n-1$, if $A[k].score == B[k].score$, potential match.
      Since multiple vertices can have same score, we need finer grouping.
      
      Refined Coloring Implementation:
      `vector<int> adj[N+1]`
      `int deg[N+1]`
      Initial color: `deg[u]`.
      Colors stored in `vector<pair<int, int>> nodes_by_degree`.
      Wait, multiset of neighbor colors is slow ($O(N \log N)$ per iter).
      Using `vector<int>` sorted neighbor colors: construction takes $O(D \log D)$ or $O(D)$. Summing over all vertices: $\sum D = 2M$. Sorting takes $\sum D \log D$.
      For $M \approx N^2$, this is $O(N^2 \log N)$. Acceptable.
      
      Algorithm:
      1. `vector<int> c1(N+1, 0)`, `c2(N+1, 0)`. Init with degree.
      2. Loop max 20 times:
         Next colors `nc1`, `nc2`.
         For each $u$:
           Collect `c1[v]` for $v \in Adj_1[u]$.
           Sort them (conceptually, store in vector).
           Form signature `pair<int, vector<int>>` (or compressed string/hash).
           Find rank among all unique signatures.
           Assign `nc1[u]`.
         Repeat.
      
      Wait, comparing vectors is heavy.
      We can assign IDs to unique signatures.
      Since we want to avoid $O(N \log N)$ comparisons inside the loop, maybe simpler signature is enough.
      Signature = `c[u]`? No, that's previous.
      Signature = `pair(c[u], sum_deg_neighbors)`.
      Or even: Hash of (c[u], {c[v]...}).
      
      Actually, let's look at the problem source or typical solutions for "Graph Isomorphism" with soft scoring.
      Usually, a randomized local search with a good start is sufficient.
      Start: Identity mapping (if degrees allow) or sorted by degree.
      Heuristic: Greedy match neighbors.
      
      Let's combine:
      1. Build `adj` matrix for fast lookup.
      2. Group vertices by degree in both graphs.
      3. Map vertices $u \in G_2$ to $v \in G_1$ within same degree group greedily.
         How? Sort degree groups by secondary characteristic (e.g. sum of neighbor degrees).
         This creates a candidate permutation.
      4. Apply Simulated Annealing or Hill Climbing to improve.
         Number of iterations: maybe 5000.
         In each iteration, propose a swap.
         Probability of improving: try random pairs?
         If we use SA, we can swap any pair.
         Cost of evaluation: $O(deg(u) + deg(v))$.
         With $N=2000$, $O(N)$ check.
         Doing 5000 iterations: $5000 \times N^2$ pairs is too much.
         We should pick pairs strategically or limit the number of proposed swaps.
         
         Wait, we can just iterate edges of $G_2$ to compute score? No, $M$ can be large.
         But we established incremental update is $O(deg(u) + deg(v))$.
         So we just need to limit the number of proposals.
         Try 1000 random pairs? Or 10000?
         Or perform a "Greedy Swap": Find the pair $(i, j)$ whose swap yields the highest increase (or smallest decrease below threshold).
         Finding best pair is $O(N^2 \cdot (\text{avg deg}))$.
         If we just scan random pairs, expected $k$ tries to find best.
         
         Actually, let's step back.
         Does the soft scoring imply we don't need global optimum? Yes.
         Just getting reasonably close is fine.
         Often, sorting vertices by degree (and maybe another invariant) gives a decent isomorphism for many graphs.
         Maybe adding some randomness is sufficient.
         
         Let's write a solution that:
         1. Computes degrees and sorts vertices by degree + sum of neighbor degrees.
         2. Uses this sort to define an initial permutation.
         3. Runs a hill climbing / random swap optimization to try to fix mismatches.
         
         Complexity of Optimization:
         We can maintain the current score.
         In each step, we select a random pair $(u, v)$ to swap in $G_2$ vertices mapping.
         Let $curr\_p$ be the array.
         $swap(curr\_p[u], curr\_p[v])$.
         Re-calculate score difference.
         If diff > 0, accept (with probability or always for greedy).
         Do this for 2000 steps.
         In the worst case (dense graph), $deg(u) \approx N$. Delta calc is $O(N)$.
         Total time $2000 \times 2000 \times N \approx 8 \cdot 10^9$.
         Too slow.
         
         Wait, we can speed up delta calculation.
         Precompute contributions.
         But actually, we don't need to calculate delta from scratch.
         We only need to know how many edges incident to $u$ or $v$ now match.
         Wait, edges incident to $u$ in $G_2$ involve neighbors $w$.
         The check is: was $(curr\_p[u], curr\_p[w])$ in $G_1$?
         Is $(new\_p[u], curr\_p[w])$ in $G_1$?
         We can just store `vector<int> neighbors[u]` for $G_2$.
         For each neighbor $w$ of $u$:
             old_match = adj[curr_p[u]][curr_p[w]];
             new_match = adj[new_p[u]][curr_p[w]]; // wait, $new\_p[u]$ is $curr\_p[v]$
         This is $O(deg(u))$.
         
         If graph is sparse ($M$ small), $deg$ is small, this is fast.
         If graph is dense, $M$ is large, but then almost any mapping is likely to have high score?
         Wait, if $G_1$ is a complete graph, any permutation matches all edges.
         If $G_1$ is empty, score 0.
         Hard cases are usually semi-regular graphs.
         In dense graphs, calculating score is fast because `adj` access is fast, but iterating edges is slow.
         BUT, in dense graphs, $deg$ is large, so $O(deg)$ update is slow.
         Is there a bottleneck here?
         If $G_1$ is dense, we check `adj[x][y]`. That's $O(1)$.
         We iterate $w \in neighbors(w)$ in $G_2$. There are $O(N)$ such neighbors.
         So yes, update is $O(N)$.
         With 2000 swaps, we do $2000 \times N^2$ ops $\approx 8 \cdot 10^9$.
         We need fewer swaps or cheaper updates.
         
         Can we check only $k$ random swaps? Say 500 swaps?
         With random swaps, we expect to cover the space somewhat.
         500 iterations should be enough for many test cases.
         We can iterate 1000 times. $1000 \times N^2 \approx 4 \cdot 10^9$. Still risky for 1 sec.
         Usually $N=2000$ allows $O(N^2)$ or maybe $O(N^2 \log N)$ total time.
         $N^2 = 4 \cdot 10^6$.
         So we can afford $O(N^2)$ work per swap if constant factor is tiny? No.
         We can afford $O(N^2)$ total work.
         So we can only perform very few updates if they take $O(N)$ time.
         Or we need $O(1)$ updates.
         
         To get $O(1)$ updates, we need to know the score quickly.
         This seems hard because changing one mapping affects edges.
         However, notice that in the output format, we just need *one* line.
         And this is "soft scoring".
         Perhaps we don't need to optimize much.
         Maybe the "Degree + Neighbor Degree Sort" is sufficient for a very high score.
         Let's rely on that.
         
         Algorithm Revised:
         1. Parse $G_1$ and $G_2$.
         2. Build Adjacency Matrix for $G_1$. $O(N^2)$ space/time.
         3. Compute Degree for all vertices in $G_1$ and $G_2$.
         4. Construct a custom comparator.
            Key for vertex $u$ in $G_k$: `(degree[u], sum_of_neighbors_degrees[u], ...)`
            Actually, just `(degree[u], sum_of_neighbors_degrees[u])` might not be unique enough.
            Let's use `(degree[u], sorted_neighbor_degrees_string)`.
            String of degrees of neighbors?
            Length $\approx N$. Building string takes $O(N)$.
            Comparison takes $O(N)$.
            Sorting takes $O(N^2)$.
            Doing this for all vertices is $O(N^2)$ or $O(N^2 \log N)$ depending on implementation.
            With $N=2000$, this is feasible.
         5. Sort vertices of $G_1$ based on this key $\to L_1$.
         6. Sort vertices of $G_2$ based on this key $\to L_2$.
         7. Define permutation: $L_2[i]$ maps to $L_1[i]$.
         8. This gives initial $P$.
         9. Calculate initial score.
         10. Perform a limited number of random swaps (e.g. 500-1000) to improve.
             Since we calculated the score initially, we can check if swap improves score.
             Check takes $O(deg(u) + deg(v))$.
             Limit iterations to keep runtime under limit.
             
         Is this sufficient?
         Soft scoring means we just want to maximize matches.
         Usually isomorphism heuristics work quite well.
         Let's write the code.
         
         Wait, to implement "sorted neighbor degrees", we need `deg` array.
         `vector<int> n_deg[u]`.
         Fill it for all $u$.
         Then sort `n_deg[u]` to create a consistent representation.
         
         Corner cases:
         $N=2$.
         Disconnected components? Handled naturally by degree and neighbor properties.
         Multiple valid permutations? Any is fine.
         
         One detail: "Output exactly one line: n integers".
         
         Optimization:
         Instead of building full strings, use `long long` hash or just sort.
         Given $N=2000$, $O(N^2)$ is roughly $4 \times 10^6$ operations, negligible compared to overhead.
         Even $O(N^2)$ inside sort (using std::sort) is fine? No, std::sort is $N \log N$ comparisons.
         Comparison of two vertices involves comparing arrays of length $N$.
         So one comparison is $O(N)$. Total sorting $O(N^2 \log N)$.
         $2000 \times 2000 \times 11 \approx 4.4 \times 10^7$. Very fast.
         
         Wait, we can do better.
         Bitmask of neighbors?
         Represent neighborhood as a bitset.
         Compare bitsets.
         Construction: $O(N^2)$. Comparison $O(N/64)$.
         Total sorting $O(N^2 \log N / 64)$.
         This is even faster.
         
         Let's use this:
         1. For each $G_1$ vertex $u$, store bitmask of neighbors. (Actually `std::bitset<2000>`).
         2. Same for $G_2$.
         3. Key for vertex sorting:
            Primary: degree.
            Secondary: bitmask of neighbors. (Or better, bitmask of neighbors' degrees? No, raw bitmask is stronger).
            Wait, raw bitmask represents adjacency.
            If we map vertices with same neighbor pattern, they are structurally similar locally.
            Is it unique enough?
            In many random graphs, no.
            But for structured graphs, yes.
            Combined with degree, it's strong.
            Wait, two vertices can have same neighbors but in different positions relative to other vertices.
            However, since we map $u \to v$ if they share same neighbor bitmask, we might map neighbors to neighbors?
            If $u \in G_2$ and $v \in G_1$ have same neighbor mask (relative to their respective vertex set), then their neighborhoods are structurally identical?
            Not necessarily, the vertices labeled differently might have different structures.
            But `bitset` is a canonical representation of the neighborhood *subset*.
            We want to match $u$ to $v$ if $N(u)$ in $G_2$ maps to $N(v)$ in $G_1$.
            But we don't know the mapping yet.
            So sorting by (degree, bitmask) is essentially grouping vertices by their local topological structure.
            Vertices with same local structure are candidates.
            This is extremely fast ($O(N^2/64)$ sort).
            Then we do local search.
         
         Wait, if we use `bitset`, we need to store `bitset<2000> adj[n]`.
         Comparing two bitsets takes $2000/64 \approx 32$ ops.
         This is efficient.
         However, `std::bitset` is fixed size. $N \le 2000$.
         
         Plan:
         1. `adj` matrix as `bool` or `vector<bool>`.
            Wait, `bitset` is convenient but requires compile time size. $N \le 2000$. `bitset<2005>`.
         2. Compute degree for each node.
         3. Construct `key` for $G_1$: `{degree, bitset}`.
            Actually, `bitset` comparison is sufficient?
            Let's add degree first.
            `struct Info { int d; std::vector<char> b; }`.
            Wait, bitset is better.
            We can't use `std::vector<int>` for neighbors directly as the key because sorting might not respect "structural equivalence".
            But `std::sort` will put vertices with same bitsets together.
            Is it better than degree?
            In $G_2$, vertex $u$ is connected to set $A \subset V$.
            In $G_1$, vertex $v$ connected to set $B \subset V$.
            We want to map $A$ to $B$.
            If we just sort vertices by bitset, vertices with same connections to others are grouped.
            This is basically the WL 1-step refinement.
         
         Let's implement "Sort by (Degree, NeighborBitset)".
         But wait, the bitset refers to vertices $1..N$.
         The labels matter.
         Actually, `bit` $k$ is set if $k$ is a neighbor.
         So `bitset` represents the adjacency pattern.
         Two vertices have same bitset if they have same connection pattern (to specific indices).
         This assumes indices have meaning? No, just that we compare patterns.
         If $u$ and $x$ have same bitsets in $G_2$, and $v$ and $y$ have same bitsets in $G_1$,
         does it mean $u$ should map to $v$?
         Not necessarily, but it's a good heuristic.
         Because if $u$ connects to $\{1, 2\}$ and $x$ to $\{1, 2\}$,
         and $v$ connects to $\{a, b\}$ and $y$ to $\{a, b\}$,
         we should map $u \to v$ or $u \to y$ consistently.
         But which?
         The bitset values (1, 2) correspond to actual vertex indices in the input.
         Since indices in $G_1$ and $G_2$ are independent, we can't compare bitsets directly between graphs!
         Ah, that's a catch. The bitsets are indexed by original labels.
         We cannot compare bitsets from $G_1$ and $G_2$ because label 1 in $G_1$ means nothing in $G_2$.
         So the bitset is useless for cross-graph matching unless normalized.
         Normalizing means replacing vertex ID with a local ID or sorting neighbors by degree.
         Sorting neighbors by degree IS what we were discussing earlier.
         
         So back to sorting neighbors by degree.
         `vector<int> nbrs(u)`. Sort `nbrs` by `deg[v]`.
         Key = `{degree[u], sorted(nbrs_degrees)}`.
         This is fully comparable between $G_1$ and $G_2$.
         Construction cost:
         1. Count degrees: $O(N+M)$.
         2. Get neighbor list: $O(N+M)$.
         3. For each vertex, sort neighbors: $\sum O(d_i \log d_i) \approx O(M \log N)$.
         4. Extract signature (degrees of sorted neighbors). Vector of ints.
         5. Sort vertices by this signature.
         Since signature length varies, comparison takes length of signature $\approx d_i \le N$.
         Total sort time: $O(N^2)$?
         No, sorting vertices takes $N \log N$ comparisons.
         Each comparison takes $O(N)$ (length of neighbor degree list).
         Total: $O(N^2 \log N)$.
         This is acceptable.
         
         So, the Initialization:
         1. Compute `deg[u]` for $u \in 1..N$.
         2. For each $u$, build `vector<int> neighbor_degs`.
         3. Sort `neighbor_degs` for each $u$.
         4. Define tuple signature: `(degree, neighbor_deg_vector)`.
         5. Store vertices of $G_1$ as list $L_1$, $G_2$ as list $L_2$.
         6. Sort $L_1$ and $L_2$ by signature.
         7. Initial Permutation $P$: $P[L_2[i]] = L_1[i]$.
         
         Then Optimization:
         Simulated Annealing or Random Local Search.
         Given the constraints, simple greedy local search (pick random pair, if improves keep) works well for many graphs.
         To prevent getting stuck, sometimes we try to flip pairs with low penalty.
         Or just use SA (Metropolis).
         
         Since time limit is not specified but typical is 1s.
         Python is slower, but C++ is fast.
         Let's use Randomized Hill Climbing.
         Max swaps: 20000?
         Swap check cost: $O(deg(u) + deg(v))$.
         In dense graph, this is slow.
         Can we optimize swap check?
         We can maintain `current_score`.
         For a swap $(u, v)$ in $G_2$ vertices:
         Update:
         For each $w \in Neighbors_G2(u)$:
             $match_{old} = adj1[P[u]][P[w]]$
             $match_{new} = adj1[P[v]][P[w]]$ (assuming $P[u]$ becomes $P[v]$)
             diff += $match_{new} - match_{old}$
         Similar for $v$'s neighbors.
         Plus check edge $(u, v)$ in $G_2$.
         This is correct.
         We just need $O(N^2)$ memory for `adj1` (bool) and lists.
         
         Let's finalize the code structure.
         
         Include headers: `<iostream>, <vector>, <algorithm>, <cmath>, <cstdlib>, <ctime>`.
         Fast IO.
         
         Code structure:
         ```cpp
         #include <bits/stdc++.h>
         using namespace std;
         
         const int MAXN = 2005;
         bool adj1[MAXN][MAXN];
         int n, m;
         vector<int> adj2[MAXN];
         
         int main() {
             ios_base::sync_with_stdio(false); cin.tie(NULL);
             if (!(cin >> n >> m)) return 0;
             
             // Read G1
             for(int i=0; i<m; ++i) {
                 int u, v; cin >> u >> v;
                 adj1[u][v] = adj1[v][u] = true;
             }
             
             // Read G2
             for(int i=0; i<m; ++i) {
                 int u, v; cin >> u >> v;
                 adj2[u].push_back(v);
             }
             
             // Prepare data structures for sorting
             struct Node {
                 int u; // original index in G1/G2 (for storage)
                 int deg;
                 vector<int> n_deg; // sorted neighbor degrees
             };
             vector<Node> nodes1, nodes2;
             nodes1.reserve(n);
             nodes2.reserve(n);
             
             // We need to store mapping P later. 
             // Let's use a vector 'target' where target[i] is the target in G1 for G2 node i.
             vector<int> target(n + 1);
             
             // Helper to generate signature
             auto get_sig = [&](int u, const vector<int>& neighbors, const vector<int>& degrees) {
                 vector<int> sig;
                 sig.push_back(degrees[u]);
                 for(int v : neighbors) {
                     sig.push_back(degrees[v]);
                 }
                 // sort sig? No, push_back keeps order? No, we need canonical representation.
                 // Wait, we need to sort neighbors' DEGREES.
                 // But to ensure stability/canonicalness, we just push them and sort the vector inside the struct.
                 // Actually, inside the lambda, we construct a temp vector.
                 // It's better to construct a struct with a vector<int> field and sort it.
                 return {}; // placeholder
             };
             
             // Degrees in G1
             vector<int> deg1(n + 1, 0);
             vector<int> deg2(n + 1, 0);
             vector<vector<int>> neighbors1(n + 1), neighbors2(n + 1);
             
             // Read edges properly to populate adj1 (for fast lookup) and also build degs
             // Already read adj1 bools. Need to reconstruct lists for degree computation?
             // Actually, we can just iterate.
             // But reading edges twice isn't allowed in stdin usually (unless buffered).
             // Let's rebuild.
             // Wait, I read m lines for G1, storing them in adj1 directly.
             // I didn't store lists. I can iterate adj1 to build lists.
             
             for(int i=1; i<=n; ++i) {
                 for(int j=i+1; j<=n; ++j) {
                     if(adj1[i][j]) {
                         neighbors1[i].push_back(j);
                         neighbors1[j].push_back(i);
                         deg1[i]++;
                         deg1[j]++;
                     }
                 }
             }
             
             // deg2 is already populated if I filled adj2.
             // Let's refill neighbors2 while reading? No, read loop was separate.
             // Let's re-read carefully.
             // Input:
             // Line 1: n m
             // Next m: u v (G1)
             // Next m: u v (G2)
             // So I should store G1 edges in adj1 bools.
             // Store G2 edges in adj2 vector.
             // While storing G2, compute deg2.
             
             // Redo reading part in thought process:
             /*
             // G1
             for(m) cin u v; adj1[u][v]=true;
             // G2
             for(m) cin u v; adj2[u].push(v); deg2[u]++; deg2[v]++;
             */
             
             // Construct Nodes1 and Nodes2
             for(int i=1; i<=n; ++i) {
                 // For G1
                 vector<int> s = deg1[i]; // temporary holder?
                 // We need a vector of neighbor degrees, sorted.
                 vector<int> s_deg;
                 s_deg.reserve(neighbors1[i].size());
                 for(int v : neighbors1[i]) s_deg.push_back(deg1[v]);
                 sort(s_deg.begin(), s_deg.end());
                 
                 nodes1.push_back({i, deg1[i], s_deg}); // Copying s_deg is costly?
                 // Copying takes O(deg). Sum of deg is M.
                 // Creating nodes1 takes O(N+M log M).
                 // Total N nodes. Each copy O(N). Copy total O(N^2).
                 // Fine.
             }
             // Do same for G2.
             
             // Wait, copying s_deg every time.
             // vector<int> s_deg moves? Yes.
             // Total time O(N^2 + M log N).
             
             // Sort nodes1 and nodes2.
             // Sort criteria: deg, then s_deg.
             sort(nodes1.begin(), nodes1.end(), [](const Node& a, const Node& b){
                 if(a.deg != b.deg) return a.deg < b.deg;
                 if(a.n_deg.size() != b.n_deg.size()) return a.n_deg.size() < b.n_deg.size();
                 return a.n_deg < b.n_deg; // Lexicographical comparison
             });
             
             sort(nodes2.begin(), nodes2.end(), [](const Node& a, const Node& b){
                 if(a.deg != b.deg) return a.deg < b.deg;
                 if(a.n_deg.size() != b.n_deg.size()) return a.n_deg.size() < b.n_deg.size();
                 return a.n_deg < b.n_deg;
             });
             
             // Init Permutation
             for(int i=0; i<n; ++i) {
                 target[nodes2[i].u] = nodes1[i].u;
             }
             
             // Compute initial score
             int current_score = 0;
             for(auto e : edges2) {
                 if(adj1[target[e.u]][target[e.v]]) current_score++;
             }
             
             // Hill Climbing
             // Pick 2000 random swaps?
             // Or iterative improvement.
             // To optimize swap check, we need to efficiently access adj1.
             // adj1 is [MAXN][MAXN].
             // target[u] is the mapped vertex in G1.
             
             // Check if edge exists: adj1[target[u]][target[v]].
             
             // Let's use a simpler randomized improvement.
             // We want to maximize score.
             // Try random swaps for K iterations.
             
             int max_iterations = 10000; // Enough to shake things up?
             // Maybe 5000 is safer.
             
             // But we need to check score change quickly.
             // Change only depends on neighbors of u and v in G2.
             // Let's precompute neighbors2 list.
             
             // Random shuffle for pairs?
             // Actually, if we just iterate 10000 times, and pick random pair (u, v) from 1..N.
             // Probability of finding good swap decreases as score increases.
             // This is basically Simulated Annealing without temperature cooling (just greedy with restart or random acceptance).
             // Greedy is fine.
             
             // Let's try to swap vertices that have mismatched edge counts?
             // No, random is easier.
             
             // Optimizations for dense graphs:
             // If graph is dense, deg is large.
             // Iterating neighbors is slow.
             // However, if score is already high (near 1), few swaps needed.
             // If score is low, maybe we need many swaps.
             
             // Let's just do O(N^2) random swaps with early exit.
             // If we don't improve for X iterations, stop.
             
             vector<int> g2_adj_list[MAXN]; // From G2 input
             
             // ... fill g2_adj_list ...
             
             // Inside loop:
             // u = rand(), v = rand()
             // swap(target[u], target[v])
             // delta = 0
             // neighbors of u in G2: w
             //   old_term = adj1[target_old[u]][target[w]]
             //   new_term = adj1[target_new[u]][target[w]]
             //   delta += new - old
             // repeat for v.
             // if delta > 0: accept, else swap back (maybe probabilistic?)
             
             // Note: Edge (u, v) handled twice in loop above (once for u as neighbor of v, once for v as neighbor of u).
             // Be careful with double counting.
             // Condition `w != v` and `w != u`? No, neighbors include each other if connected.
             // Handle (u, v) edge separately or ensure loop covers it correctly.
             // Simpler:
             // Iterate neighbors of u in G2 (including v if connected).
             // Iterate neighbors of v in G2 (including u if connected).
             // For each neighbor w:
             //   term_old = (w==u || w==v) ? (adj1[target_old[u]][target_v]? No)
             //   Wait, simpler:
             //   The contribution of u in G2 is sum_{w in Adj(u)} adj1[target[u]][target[w]].
             //   The contribution of v is sum_{w in Adj(v)} adj1[target[v]][target[w]].
             //   We need total contribution of edges. Sum of contributions counts edges incident to u and v.
             //   Edge (u, v) is counted in u's contrib and v's contrib.
             //   So summing u_contrib + v_contrib counts (u, v) twice.
             //   Original Sum: (u, v) counted twice.
             //   New Sum: (u, v) counted twice.
             //   So we can just compute delta of u_contrib + v_contrib.
             //   Wait, if u->v edge exists in G2.
             //   u_contrib includes adj1[target[u]][target[v]].
             //   v_contrib includes adj1[target[v]][target[u]].
             //   Since adj is symmetric, they are equal.
             //   Both contribute 2 to the sum (if matched).
             //   If swapped: u gets t_new[u], v gets t_new[v].
             //   u_contrib term uses t_new[u]. v_contrib uses t_new[v].
             //   Correct.
             
             // Implementation details:
             // To avoid O(deg) scan every time, we can't do better easily.
             // But we limit iterations.
             // 5000 iterations * avg_degree. Avg degree approx 2M/N.
             // Total Ops ~ 5000 * M.
             // M <= N^2/2 = 2*10^6.
             // 5000 * 2*10^6 = 10^10. TOO SLOW.
             
             // We MUST limit iterations or speed up.
             // Speed up? No.
             // Limit iterations to few hundreds?
             // If we limit to 500 iterations:
             // 500 * 2*10^6 = 10^9. Still tight.
             // Maybe 100 iterations.
             // 100 * 2*10^6 = 2*10^8. Feasible.
             // But 100 iterations might be insufficient for complex graphs.
             
             // Is there a way to check faster?
             // If we store `cnt[u]` = number of neighbors w such that `adj1[target[u]][target[w]]` is 1.
             // Wait, this depends on `target[w]`.
             // If we swap `target[u]` and `target[v]`.
             // `target[w]` remains same for all w except if w=u or w=v.
             // So for all w (neighbors of u, excluding u and v):
             //   `adj1[target[u]][target[w]]` changes to `adj1[target[v]][target[w]]`.
             // This is exactly the check we are doing.
             // So it's intrinsic.
             
             // Wait, the constraint `M` is up to `N*(N-1)/2`.
             // In practice, random dense graphs are easy.
             // Sparse graphs have small M.
             // The worst case is `M` large but structure complex?
             // Or `M` moderate.
             // If `M` is large, maybe we can optimize.
             // If `M` is close to `N^2`, almost everything is 1.
             // `adj1` is mostly 1.
             // `adj1[x][y]` check is fast.
             // Iterating neighbors is $O(N)$.
             
             // Can we assume `M` is smaller?
             // No.
             // However, we can prioritize random swaps over full scan?
             // Actually, we can just sample a subset of neighbors? No, accuracy matters.
             
             // Let's reconsider the "Sort" strategy.
             // Maybe the initial sorted mapping is ALMOST optimal.
             // If we perform very few swaps, say 500, is it enough?
             // Competitive programming solutions often use limited local search.
             // Let's try to set iterations dynamically or based on progress.
             // If score == M, break.
             
             // Another heuristic:
             // If we swap, check only neighbors in `g2` that are NOT already matched?
             // No, matching status is binary.
             
             // Let's stick to 100-200 iterations. It should pass given "soft scoring".
             // Also, we can use a small amount of randomness in swap choice to avoid local optima.
             
             // Wait, 10^8 ops is usually 0.5s.
             // If we do 200 iterations, and max M = 2e6.
             // Worst case 4e8. Might be TLE.
             // But average degree is smaller than max.
             // Unless M is truly quadratic (complete graph minus few edges).
             // In complete graph, degree is N. Checking neighbors is N ops.
             // But if M is complete, then `target` mapping doesn't really matter?
             // If G1 and G2 are complete graphs, score is 1 regardless of permutation.
             // We just output anything.
             // My code will output identity-like.
             // Does my code slow down?
             // Yes, scanning neighbors takes time.
             // But we can detect if graph is dense?
             // If `m > 0.9 * n*n`, treat as special?
             // Actually, if `m` is huge, `adj1` is dense.
             // Then `adj1[x][y]` is likely 1.
             // We only need to check if `target` changed edge presence.
             // In dense graph, most edges are present.
             // So score change is rarely significant (mostly noise or small variations).
             // However, we want to avoid worst case TLE.
             
             // Let's implement `m` dependent check.
             // If `m` is very large (e.g. > 10000), the cost of `adj1` access dominates? No.
             // The loop runs `num_neighbors` times.
             // If `m` is large, `num_neighbors` is large.
             // BUT, if `m` is large, we probably just need to verify.
             // Maybe we can cap the number of iterations.
             
             // Better approach for optimization:
             // Don't check ALL neighbors.
             // Only check neighbors `w` where `adj1[old_target[u]][target[w]] != adj1[new_target[u]][target[w]]`.
             // We iterate all neighbors. Check both conditions.
             // If `new` != `old`, update delta.
             // This is exactly what we do.
             
             // Let's trust the $N=2000$ constraint and $O(N^2)$ complexity for sort/init.
             // Optimization is the risk.
             // I will limit iterations to 1000.
             // 1000 * M is dangerous.
             // But note: in most cases, $M$ is not maximal.
             // If $M$ is maximal, sorting gives arbitrary result which is valid.
             // Wait, if $M$ is maximal, ANY permutation is valid.
             // But my code spends time to check validity.
             // We can skip check if score >= total_matched_max_possible?
             // We don't know max possible.
             
             // But actually, if graph is complete, loop runs 1000 times.
             // For each swap, we iterate $N$ neighbors.
             // Total $1000 \times N^2 \approx 4 \cdot 10^9$. Definitely TLE.
             // We need to handle dense graphs faster.
             // OR just rely on the fact that if dense, score is usually 1 immediately?
             // If $G_1, G_2$ are both complete graphs, score starts at $M$. No swaps needed.
             // If $G_1$ complete, $G_2$ sparse.
             // Matched edges in $G_2$?
             // Vertices of $G_2$ map to $G_1$. All edges in $G_1$ exist.
             // So `adj1` always returns 1 for any distinct pair.
             // So we just need to count edges in $G_2$.
             // My code does this.
             // The issue is calculating the delta for swaps.
             // If $G_1$ is complete, `adj1` is always 1.
             // So delta is always 0 (unless self-loops, but problem says $u \neq v$).
             // So we can check: if `adj1` is uniform (all 1), stop.
             // But checking uniformity is also $O(N^2)$.
             // If we just assume `adj1` is dense, we don't need to check neighbors?
             // Wait, if `adj1` is dense, then `adj1[x][y]` is 1.
             // Then `delta` is 0.
             // So we can detect "almost complete" graph.
             // If `m/n > 0.9*N`?
             // Threshold: if `m` is very close to `n*(n-1)/2`.
             
             // Let's add a condition:
             // If `n * (n - 1) / 2 - m < threshold`, then `adj1` is dense.
             // If `adj1` is dense, score is just $|E_2|$?
             // No. `adj1[p(u)][p(v)]` must be 1.
             // If `G1` is complete, then for ANY $u, v$, `p(u), p(v)` are distinct (since permutation), so edge exists in $G1$.
             // So score is $|E_2|$. Which is $M$.
             // So if `G1` is complete, score is always 1.
             // Similarly if `G2` is complete.
             // So if either is complete, we can output any permutation (identity) and finish.
             // What if neither is complete but edges overlap heavily?
             // Probably okay.
             
             // Also, to speed up the loop:
             // If `m` is large, `deg` is large.
             // If we pick 1000 random swaps, that might be enough for "soft scoring".
             // We shouldn't optimize out correctness, but correctness of "finding a high score" is the goal.
             // We won't TLE if we check fewer swaps for dense graphs or just assume they are easy.
             // Let's stick to a reasonable number of swaps, say 500.
             // 500 * N^2 is ~ 4 * 10^9. Still too high.
             // Wait, is it 500 * N^2 or 500 * M?
             // Loop runs 500 times. Inside, iterate neighbors.
             // Sum of neighbors of u and v is $deg(u) + deg(v)$.
             // Expected sum is $2 \cdot (2M/N) = 4M/N$.
             // So one swap check takes $O(M/N)$.
             // Total time $K \cdot M/N$.
             // With $K=500$, $M=N^2/2$.
             // $500 \cdot (N^2/2)/N = 250 \cdot N \approx 500,000$.
             // This is FAST.
             // The worst case is $M=N^2/2$.
             // In that case $deg(u) \approx N$.
             // Then loop takes $O(N)$.
             // Total time $K \cdot N$.
             // $500 \cdot 2000 = 1,000,000$.
             // Wait, my previous calculation was $K \cdot N^2$.
             // Why did I think $N^2$?
             // Because I assumed we iterate $N$ neighbors for each swap check.
             // Yes, if dense, $deg(u) \approx N$.
             // If we do 500 swaps, total cost is $500 \times N$.
             // Wait, why did I think $K \cdot N^2$ before?
             // Ah, I was thinking of iterating all $N^2$ pairs to find the BEST swap.
             // If we pick RANDOM pairs, we iterate pairs $K$ times.
             // Each pair check costs $deg(u)+deg(v)$.
             // So it is $O(K \cdot N)$ in dense case.
             // So even 10000 swaps is $2 \cdot 10^7$ ops.
             // That is totally fine!
             // The mistake was thinking we check ALL pairs or that cost is $O(N^2)$ per swap.
             // It is $O(deg)$.
             // So random swaps are safe.
             
             // Final plan:
             // 1. Init by degree+signature sort.
             // 2. Hill climb with 5000 random swaps.
             // 3. Return permutation.
             
             // One minor correction:
             // When checking swap, we modify `target` array.
             // If we swap, we accept/reject.
             // We can also implement a "best-so-far" tracking.
             
             // One detail:
             // "Soft scoring" implies we just want high score.
             // Hill climbing with restarts or just plain greedy.
             
             // Let's write it.
             
             // Headers
             // Fast I/O
             
             // Data structures
             // Global adj for G1 (since N fixed).
             // Vectors for G2.
             
             // Main logic.
             
             // Double check: N <= 2000.
             // stack size might be issue for recursion? No recursion used.
             // Heap allocation vectors are fine.
             
             // Memory:
             // adj1 [2005][2005] bool -> 4MB.
             // Other vectors negligible.
             
             // Constraints:
             // 2 <= n <= 2000.
             // m >= 1.
             
             // Example trace:
             // G1: 1-2
             // G2: 1-2
             // degs match. Order preserved.
             
             // Output formatting:
             // p1 ... pn.
             
             // Corner cases:
             // Disconnected graphs: degree handles components.
             // Multiple components with same degrees?
             // Sorting handles this by assigning indices.
             // If multiple nodes have same signature, order doesn't matter much (symmetric).
             // But local search might improve.
             
             // Code looks solid.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

using namespace std;

const int MAXN = 2005;
bool adj1[MAXN][MAXN];
int n, m;
vector<int> adj2[MAXN];

struct NodeInfo {
    int u;
    int deg;
    vector<int> neighbor_degrees;
};

// Comparator for sorting
bool compareNodes(const NodeInfo& a, const NodeInfo& b) {
    if (a.deg != b.deg) return a.deg < b.deg;
    if (a.neighbor_degrees.size() != b.neighbor_degrees.size())
        return a.neighbor_degrees.size() < b.neighbor_degrees.size();
    return a.neighbor_degrees < b.neighbor_degrees;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n >> m)) return 0;

    // Reset global array if reused, though here strictly one run
    for(int i=0; i<=n; ++i) {
        for(int j=0; j<=n; ++j) adj1[i][j] = false;
    }

    // Read G1
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj1[u][v] = true;
        adj1[v][u] = true;
    }

    // Read G2
    // We also need to compute degrees for G2
    vector<int> deg2(n + 1, 0);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj2[u].push_back(v);
        adj2[v].push_back(u);
        deg2[u]++;
        deg2[v]++;
    }

    // Build NodeInfo for G1
    // First, compute degrees for G1 and gather neighbors
    vector<int> deg1(n + 1, 0);
    vector<vector<int>> neighbors1(n + 1);
    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            if (adj1[i][j]) {
                deg1[i]++;
                neighbors1[i].push_back(j);
            }
        }
    }

    vector<NodeInfo> nodes1, nodes2;
    nodes1.reserve(n);
    nodes2.reserve(n);

    // Helper lambda to fill NodeInfo
    auto fill_node_info = [&](int idx, const vector<int>& adj_list, const vector<int>& degrees_vec) -> NodeInfo {
        NodeInfo info;
        info.u = idx;
        info.deg = degrees_vec[idx];
        info.neighbor_degrees.reserve(adj_list.size());
        for (int neighbor : adj_list) {
            info.neighbor_degrees.push_back(degrees_vec[neighbor]);
        }
        sort(info.neighbor_degrees.begin(), info.neighbor_degrees.end());
        return info;
    };

    for (int i = 1; i <= n; ++i) {
        nodes1.push_back(fill_node_info(i, neighbors1[i], deg1));
    }
    
    // Reconstruct adj2 degrees into a vector for filling
    // Note: we need neighbor degrees based on GLOBAL structure?
    // Yes, we sort neighbor degrees.
    // We have adj2, so we can compute deg2 array.
    // But we need deg2 vector ready.
    // Wait, I computed deg2 partially in loop. Let's ensure it's correct.
    // Actually, in reading G2 loop I updated deg2[u] and deg2[v].
    // So deg2 is ready.
    
    for (int i = 1; i <= n; ++i) {
        // Need neighbors for sorting? No, I have adj2[u]
        nodes2.push_back(fill_node_info(i, adj2[i], deg2));
    }

    sort(nodes1.begin(), nodes1.end(), compareNodes);
    sort(nodes2.begin(), nodes2.end(), compareNodes);

    // Initial Mapping
    vector<int> p(n + 1); // p[u] = image of vertex u in G2
    for (int i = 0; i < n; ++i) {
        p[nodes2[i].u] = nodes1[i].u;
    }

    // Function to calculate score
    // Returns score (count of matched edges)
    auto get_score = [&]() {
        int count = 0;
        // Iterate over edges of G2
        // Optimization: we can use adj2 list.
        // But we must avoid double counting edges (undirected).
        // Iterate 1 to n, check neighbors > i?
        // Better: iterate edges list.
        // We didn't store edge list of G2 explicitly, just adjacency lists.
        // We can iterate adj2.
        // To avoid duplicates, iterate u from 1 to n, v in adj2[u]. If v > u, count.
        // Or just sum and divide by 2.
        // Dividing by 2 is safer.
        
        // Let's collect edges to avoid double counting logic mess
        // Actually, simple: sum over all u, all v in adj[u].
        // If adj1[p[u]][p[v]] then score++.
        // Finally score /= 2.
        
        int total_matches = 0;
        for(int u = 1; u <= n; ++u) {
            for(int v : adj2[u]) {
                if(adj1[p[u]][p[v]]) {
                    total_matches++;
                }
            }
        }
        return total_matches / 2;
    };
    
    // Wait, `get_score` is O(N^2) or O(M). O(M) is fast enough?
    // If M is 2e6, doing this 100 times is 2e8. Fine.
    // But for local search, we can calculate delta.

    int current_score = get_score();

    // Optimization variables
    // Since N is up to 2000, we can perform randomized local search.
    // Max iterations 5000.
    // Random number generator
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> dist(1, n);

    // To optimize delta calculation:
    // Precompute neighbors for G2 (already in adj2)
    
    // Hill Climbing
    // We can try to improve.
    // We don't need Simulated Annealing (temperature) to strictly avoid local optima,
    // just random restarts or greedy swaps.
    // Let's do a fixed number of random swaps.
    // If we find a better one, keep it.
    // Or better: keep improving.
    
    // To prevent TLE in dense graphs, limit iterations?
    // Or rely on random seed.
    
    int max_swaps = 5000; 
    // Actually, 5000 swaps * avg_deg check.
    // If avg_deg is high, it's O(N). 5000*2000 = 10^7 ops.
    // If we do 20000 swaps, 20000*2000 = 4*10^7.
    // This is very safe.
    // So we can do many swaps.
    
    // Wait, checking score incrementally:
    // If we swap p[u] and p[v].
    // Score change = sum_{w in Neighbors(u)} (adj1[new][old_w] - adj1[old][old_w])
    //                 + sum_{w in Neighbors(v)} (adj1[new'][old'_w] - ...)
    // Where new/old refers to p[u]/p[v].
    // We need to be careful about double counting edge (u, v).
    // The sum formula:
    // Edges incident to u in G2 contribute 1/2 if adj1 holds.
    // Wait, `total_matches` iterates all directed edges and divides by 2.
    // So each edge (u,v) contributes 2 to total_matches (once at u, once at v).
    // So delta should be calculated based on directed edges.
    // delta_u = 0.
    // for w in adj2[u]:
    //    val_old = adj1[p[u]][p[w]]
    //    val_new = adj1[p[v]][p[w]]
    //    delta_u += val_new - val_old
    
    // delta_v = 0.
    // for w in adj2[v]:
    //    // We must exclude w == u because u is in adj2[v] and handled in delta_u loop?
    //    // Yes, adj2 contains neighbors. If u is neighbor of v, it appears in adj2[v] AND adj2[u].
    //    // So we handle edge (u, v) twice.
    //    // In delta_u, we processed (u, v) using p[u]->p[v].
    //    // In delta_v, we process (v, u) using p[v]->p[u].
    //    // Both are valid.
    //    val_old = adj1[p[v]][p[w]]
    //    val_new = adj1[p[u]][p[w]]
    //    delta_v += val_new - val_old
    
    // Total Delta = delta_u + delta_v.
    // Note: for edge (u, v), u sees neighbor v.
    // val_old = adj1[p[u]][p[v]]
    // val_new = adj1[p[v]][p[v]] ?? NO.
    // If we swap p[u] and p[v], let's call them A=p[u], B=p[v].
    // New A' = B, B' = A.
    // Edge (u, v) in G2.
    // At u: p[w] where w=v is B.
    //   val_old = adj1[A][B].
    //   val_new = adj1[B'][B] = adj1[A][B]. (Same!)
    // At v: p[w] where w=u is A.
    //   val_old = adj1[B][A].
    //   val_new = adj1[A][A]. (Wait, self-loop?)
    //   Wait, if w=u, p[w] is unchanged? No, we swapped p[u] and p[v].
    //   So w=u implies index u. p[u] became B.
    //   So val_new uses B.
    //   Wait, `w` is a neighbor. `w` is a vertex index in G2.
    //   Its image p[w] stays the SAME unless w is u or w is v.
    //   So for w=v (neighbor of u), p[w] is the image of v.
    //   Wait, p[w] is the variable holding the image.
    //   If w == v, then p[w] is involved in the swap?
    //   Yes, p[w] is p[v].
    //   So we need to track carefully.
    
    // Correct Logic for Swap(p[u], p[v]):
    // Save old_images = {p[u], p[v]}
    // Swap p[u], p[v].
    // Iterate all w in adj2[u]:
    //    if w == v: continue; // Edge (u, v) handled?
    //    // Wait, edge (u, v) is undirected.
    //    // If we iterate all neighbors of u and all neighbors of v.
    //    // The edge (u, v) appears in adj2[u] (at v) and adj2[v] (at u).
    //    // We just need to check contribution of (u, w) and (v, w).
    //    // Contribution of edge {x, y} is adj1[p[x]][p[y]].
    //    // It is counted twice in the total_matches logic.
    //    // So delta calculation for directed edges is sufficient.
    
    //   Let's trace edge (u, v).
    //   In adj2[u], we see v. Corresponding image is p[v].
    //   Before swap: term is adj1[old_p[u]][old_p[v]].
    //   After swap: term is adj1[old_p[v]][old_p[v]]? NO.
    //   p[v] is the image. If we swap images p[u] <-> p[v].
    //   So new_image_for_u is old_p[v].
    //   new_image_for_v is old_p[u].
    //   Neighbor v's image is old_p[v].
    //   So term becomes adj1[old_p[v]][old_p[v]]. Self loop?
    //   Wait. p[v] is a vertex in G1.
    //   So for edge {u, v}:
    //   Term is adj1[image(u), image(v)].
    //   Initially image(u)=A, image(v)=B. Term adj1[A][B].
    //   After swap image(u)=B, image(v)=A. Term adj1[B][A].
    //   Since G1 is undirected, adj1[A][B] == adj1[B][A].
    //   So swapping images of endpoints of an edge DOES NOT CHANGE the match status of that edge!
    //   (Provided u, v are distinct).
    //   Thus, for edge (u, v) in G2, the delta is 0.
    //   We only care about edges where ONE endpoint is u or v, and OTHER is neither.
    //   So we just need to exclude w == v when processing u, and w == u when processing v.
    //   But wait, if we process u's neighbors:
    //   If w is a neighbor of u.
    //   If w == v: edge is (u, v). As established, delta is 0.
    //   If w != v: edge is (u, w).
    //   Image of w does NOT change.
    //   Image of u changes A -> B.
    //   So we check adj1[B][image(w)] vs adj1[A][image(w)].
    
    // So optimized logic:
    // Swap(p[u], p[v]).
    // For w in adj2[u]:
    //    if w == v continue;
    //    if (!adj1[old_p[v]][p[w]] && adj1[old_p[u]][p[w]]) old_score_loss++?
    //    Actually we need delta.
    //    delta += adj1[new_p[u]][p[w]] - adj1[old_p[u]][p[w]];
    // For w in adj2[v]:
    //    if w == u continue;
    //    delta += adj1[new_p[v]][p[w]] - adj1[old_p[v]][p[w]];
    //
    // If delta > 0, keep. Else revert.
    
    // Wait, what about edge (u, v) itself?
    // If edge (u, v) exists in G2.
    // Does its match status change?
    // Old: adj1[A][B].
    // New: adj1[B][A].
    // Since G1 is undirected, they are equal.
    // So delta contribution from (u, v) is 0.
    // Correct.
    
    // Wait, what if w=v?
    // In adj2[u], we encounter v.
    // p[w] = p[v].
    // We excluded w=v.
    // Is the contribution of edge (u, v) accounted for elsewhere?
    // It is NOT accounted for in w loop because of exclusion.
    // And as derived, its contribution change is 0.
    // So it is safe to ignore.
    
    // Wait, we must ensure we don't double count or miss anything.
    // Edges incident to u (excluding v) contribute to delta.
    // Edges incident to v (excluding u) contribute to delta.
    // Edges between u and v: contribution 0.
    // Is it possible that (u, v) was not matched before, but becomes matched?
    // No, because adj1[A][B] == adj1[B][A].
    // So status of edge (u, v) never changes.
    // Great.
    
    // Final check:
    // What if we have a triangle u-v-w in G2.
    // Swap u and v.
    // Edges (u, v), (u, w), (v, w).
    // (u, v): ignored.
    // (u, w): w != v. Checked in u loop.
    // (v, w): w != u. Checked in v loop.
    // Correct.
    
    // Complexity: O(deg(u) + deg(v)).
    // Perfect.
    
    int best_score = current_score;
    // Use a loop with random seeds?
    // If score doesn't improve for long, break?
    // Let's just run fixed iterations.
    
    // Also, for the initial sort, if we want to be robust against isomorphisms,
    // we might need to sort by something more unique?
    // Neighbor degrees is good.
    // Maybe neighbor neighbor degrees?
    // Given time constraints, 1-level refinement (neighbor deg) is standard.
    
    // Implementation details.
    // We need to store original `p` array to revert.
    vector<int> original_p = p;
    
    // Wait, to avoid reallocating inside loop, declare delta vars outside.
    
    // Loop
    for(int i=0; i<max_swaps; ++i) {
        int u = dist(rng); // 1-based
        int v = dist(rng);
        if(u == v) continue;
        
        int du = adj2[u].size();
        int dv = adj2[v].size();
        
        int delta = 0;
        // We need old values to check condition, but since we check before swap
        // and know what they become, we can just simulate.
        // Wait, we need to know p[u] (A) and p[v] (B).
        int Au = p[u];
        int Av = p[v];
        
        // Check u's neighbors
        for(int w : adj2[u]) {
            if(w == v) continue;
            // Image of w is p[w].
            // Old term: adj1[Au][p[w]]
            // New term: adj1[Av][p[w]]
            if(!adj1[Au][p[w]] && adj1[Av][p[w]]) delta++;
            else if(adj1[Au][p[w]] && !adj1[Av][p[w]]) delta--;
        }
        
        // Check v's neighbors
        for(int w : adj2[v]) {
            if(w == u) continue;
            // Old term: adj1[Av][p[w]]
            // New term: adj1[Au][p[w]]
            if(!adj1[Av][p[w]] && adj1[Au][p[w]]) delta++;
            else if(adj1[Av][p[w]] && !adj1[Au][p[w]]) delta--;
        }
        
        // Edge (u, v) in G2?
        // If adj2[u] contains v, it was skipped.
        // Contribution is 0.
        // BUT, does (u, v) exist in G2?
        // We check if v is in adj2[u] AND u is in adj2[v].
        // If so, skipping is correct.
        // So no extra check needed.
        
        if(delta > 0) {
            // Apply swap
            p[u] = Av;
            p[v] = Au;
            current_score += delta;
            best_score = current_score;
        } else {
            // Revert
            p[u] = Au;
            p[v] = Av;
        }
    }
    
    // Output
    for(int i=1; i<=n; ++i) {
        cout << p[i] << (i==n?"":" ");
    }
    cout << "\n";

    return 0;
}
```
One small logic bug check:
In `adj2[u]` loop:
We iterate `w`. If `w` is `v`, we continue.
But `w` is vertex index in `G2`.
Is it guaranteed that `w` is in `adj2[u]`? Yes.
If `w == v`, we skip.
Correct.
What about `w` not in `adj2[v]`?
We process `u`'s neighbors and `v`'s neighbors independently.
Seems correct.

Double check the time complexity again.
Sorting: $O(N^2 \log N)$.
Loop: 5000 iterations.
Inside loop: $O(deg(u) + deg(v))$.
Worst case $O(N)$.
Total $O(N^2 \log N + K \cdot N)$.
With $N=2000, K=5000$, this is roughly $8 \cdot 10^6 + 10^7$.
Very fast.
Even if $M$ is huge (dense), logic holds.

One thing: `dist` generates random numbers in range `[1, n]`.
Need `uniform_int_distribution<int> dist(1, n);`.
Also `mt19937` header.
And `chrono`.

Wait, the prompt says "Your response should ONLY contain the C++ code".
I should wrap it in ` ```cpp ... ``` `.

Another check: `nodes2[i].u` accesses. `nodes2` sorted.
The mapping `p` is initialized.
Is it possible that two nodes in `nodes2` have same signature but belong to different connected components of $G_2$?
Yes.
Does the order matter?
If they are disconnected, they are effectively identical structures.
Mapping them arbitrarily is valid.
The sorting is stable? No, `std::sort` is not stable.
But stability doesn't hurt.
Just consistency.

Wait, `vector<int> adj2[MAXN]`.
In the loop `for(int i=0; i<m; ++i) { cin >> u >> v; adj2[u].push_back(v); adj2[v].push_back(u); ...}`
This fills `adj2` with neighbors.
This is correct.

One edge case: `m` is large, memory usage?
Vectors of vectors for `adj2` store $2M$ integers.
$2 \times 2 \cdot 10^6 \times 4$ bytes $\approx 16$ MB. Fine.
`adj1` is `bool` array $4$ MB.
Total memory well within limits.

Final check on imports.
`<iostream>`, `<vector>`, `<algorithm>`, `<numeric>`, `<random>`, `<chrono>`.
Looks good.

One detail: `best_score` is not used, only `current_score`.
But keeping `best_score` doesn't hurt.
Wait, if we use simple greedy (only accept if delta > 0), we might get stuck in local optima.
Should I implement simulated annealing or multi-start?
"Soft scoring" + "competitive programmer" usually implies that a good heuristic is expected.
With 5000 iterations, greedy might settle.
Adding a chance to accept bad moves helps.
Or restarting.
Given I can't spend too much time on tuning constants, let's add a small probability to accept bad moves.
Say, 0.2 probability.
`int r = dist(rng); if(r % 5 == 0 && delta < 0) { apply swap; }`
Actually, `delta` is integer.
Let's add this simple twist. It prevents trapping easily.
Probability $1/K$ or something.
If `delta < 0` but small? Or just random.
With `mt19937`, `rand()` is uniform.
Let's say: `if (delta > 0 || abs(delta) * 50 < current_score)`? No.
Just `if (delta > 0)` is greedy.
Let's stick to greedy but randomize pairs more aggressively?
No, random pairs is enough.
Maybe increasing max_swaps to 20000?
$20000 \times 2000 = 4 \times 10^7$.
Totally fine.
Increasing iterations reduces risk of local optima.
Let's set `max_swaps = 20000`.

Wait, `adj2` is filled with `vector<int>`.
Is it ordered? No.
Order doesn't matter for correctness.
Sorting `adj2` might help cache locality slightly or deterministic behavior, but unnecessary.

One subtle thing:
`adj1` is `bool adj1[MAXN][MAXN]`.
Accessing `adj1[u][v]` is fast.
However, if `u` and `v` are large, array indexing is fine.
Indices 1..2000. Array size 2005.
Safe.

Wait, `nodes1` and `nodes2` sort.
If two nodes have same `neighbor_degrees` vector, `compareNodes` uses `<` operator on vector.
It compares element by element.
Vector comparison is lexicographical.
Correct.

Wait, `adj1` is global.
The `fill_node_info` captures `deg1` and `neighbors1` by reference.
Wait, `neighbors1` is vector of vector.
Capturing `neighbors1` inside lambda?
Lambda capture `[&]` or explicit.
Explicit `[&]` is safer/cleaner.
Wait, `neighbors1` is defined outside loop.
Yes.

Code structure looks good.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>

using namespace std;

const int MAXN = 2005;
bool adj1[MAXN][MAXN];
int n, m;
vector<int> adj2[MAXN];

struct NodeInfo {
    int u;
    int deg;
    vector<int> neighbor_degrees;
};

bool compareNodes(const NodeInfo& a, const NodeInfo& b) {
    if (a.deg != b.deg) return a.deg < b.deg;
    if (a.neighbor_degrees.size() != b.neighbor_degrees.size())
        return a.neighbor_degrees.size() < b.neighbor_degrees.size();
    return a.neighbor_degrees < b.neighbor_degrees;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n >> m)) return 0;

    for(int i=0; i<=n; ++i) {
        for(int j=0; j<=n; ++j) adj1[i][j] = false;
    }

    // Read G1
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj1[u][v] = true;
        adj1[v][u] = true;
    }

    // Read G2
    vector<int> deg2(n + 1, 0);
    // We'll read and populate adj2 simultaneously
    // Note: m can be 0? Problem says 1 <= m.
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj2[u].push_back(v);
        adj2[v].push_back(u);
        deg2[u]++;
        deg2[v]++;
    }
    
    // Recompute deg2 carefully because loop above increments
    // Actually the loop logic is correct.

    // Read G1 edges again to populate neighbors1 and deg1?
    // Or iterate adj1 matrix.
    vector<int> deg1(n + 1, 0);
    vector<vector<int>> neighbors1(n + 1);
    for (int i = 1; i <= n; ++i) {
        for (int j = i+1; j <= n; ++j) {
            if (adj1[i][j]) {
                neighbors1[i].push_back(j);
                neighbors1[j].push_back(i);
                deg1[i]++;
                deg1[j]++;
            }
        }
    }

    // Prepare NodeInfo
    vector<NodeInfo> nodes1, nodes2;
    nodes1.reserve(n);
    nodes2.reserve(n);

    // Lambda to build NodeInfo
    auto build_info = [&](int idx, const vector<int>& adj_list, const vector<int>& degs) -> NodeInfo {
        NodeInfo info;
        info.u = idx;
        info.deg = degs[idx];
        info.neighbor_degrees.reserve(adj_list.size());
        for (int neighbor : adj_list) {
            info.neighbor_degrees.push_back(degs[neighbor]);
        }
        // Sort degrees to make signature canonical
        sort(info.neighbor_degrees.begin(), info.neighbor_degrees.end());
        return info;
    };

    for (int i = 1; i <= n; ++i) {
        nodes1.push_back(build_info(i, neighbors1[i], deg1));
    }
    
    for (int i = 1; i <= n; ++i) {
        nodes2.push_back(build_info(i, adj2[i], deg2));
    }

    sort(nodes1.begin(), nodes1.end(), compareNodes);
    sort(nodes2.begin(), nodes2.end(), compareNodes);

    // Initial Permutation
    vector<int> p(n + 1);
    for (int i = 0; i < n; ++i) {
        p[nodes2[i].u] = nodes1[i].u;
    }

    // Hill Climbing / Local Search
    mt19937 rng(time(nullptr)); // Seed with time
    uniform_int_distribution<int> dist(1, n);

    int max_swaps = 5000; 
    
    // Calculate initial score to track improvements
    // Though we only track delta during swaps.
    // Initial score is not strictly needed for the logic, just to validate.
    
    for(int i=0; i<max_swaps; ++i) {
        int u = dist(rng);
        int v = dist(rng);
        if(u == v) continue;
        
        int Au = p[u];
        int Av = p[v];
        int delta = 0;
        
        // Check neighbors of u in G2
        for(int w : adj2[u]) {
            if(w == v) continue;
            // Check contribution of edge (u, w)
            // Old: adj1[Au][p[w]]
            // New: adj1[Av][p[w]]
            if(adj1[Au][p[w]] && !adj1[Av][p[w]]) delta--;
            else if(!adj1[Au][p[w]] && adj1[Av][p[w]]) delta++;
        }
        
        // Check neighbors of v in G2
        for(int w : adj2[v]) {
            if(w == u) continue;
            // Check contribution of edge (v, w)
            // Old: adj1[Av][p[w]]
            // New: adj1[Au][p[w]]
            if(adj1[Av][p[w]] && !adj1[Au][p[w]]) delta--;
            else if(!adj1[Av][p[w]] && adj1[Au][p[w]]) delta++;
        }
        
        // If delta > 0, accept
        if(delta > 0) {
            p[u] = Av;
            p[v] = Au;
        } else {
            // Revert
            p[u] = Au;
            p[v] = Av;
        }
    }

    // Output
    for(int i=1; i<=n; ++i) {
        cout << p[i] << (i==n?"":" ");
    }
    cout << "\n";

    return 0;
}
```
Wait, `time(nullptr)` might be invalid inside function scope?
No, it is fine.
Using `chrono` is better or just `srand`?
Standard random.
Also, check if `nodes2` sort order is consistent.
Yes.
One final check: `p` indices.
$p$ maps $G_2$ vertex $i$ to $G_1$ vertex $p[i]$.
Output requires $p_1, p_2, \dots, p_n$.
My loop prints `p[i]` for $i=1..n$.
Correct.

The code seems complete and correct within constraints.

One tiny optimization:
`time(nullptr)` returns current time. Good.
`srand` isn't used if using `mt19937`.
The provided solution is good.

Wait, I missed including `<cmath>` for `abs`?
Wait, I don't use `abs` anymore in the final snippet (removed bad move logic).
Removed `best_score` and `delta` handling.
Wait, I removed `abs` logic.
The `delta` logic:
```cpp
            if(adj1[Au][p[w]] && !adj1[Av][p[w]]) delta--;
            else if(!adj1[Au][p[w]] && adj1[Av][p[w]]) delta++;
```
This logic relies on `bool` logic.
If both true or both false, delta doesn't change.
If opposite, delta +/- 1.
Correct.

Is `abs` needed anywhere? No.
Wait, `adj2` loop:
`for(int w : adj2[u])`
This iterates neighbors.
Correct.

One issue:
`vector<NodeInfo> nodes1` copies `vector<int> neighbor_degrees`.
Copy constructor allocates memory.
Sum of sizes is $M$.
Copying takes $O(M)$.
With $M$ up to $2 \cdot 10^6$, it's fine.
Memory is not an issue.

Final check on `main` return type and arguments. `int main()`.
Everything looks good.

One small edge case:
If `n` is small, say 2.
Loops run correctly.
`p` vector size `n+1`.
Output `p[1] p[2]`.
Correct.

Double check input reading:
`adj2[u].push_back(v)` adds duplicate edges if input has duplicates?
Problem says: "Note that there are no duplicate edges."
So input is clean.
However, in reading loop, `cin >> u >> v; adj2[u].push_back(v); adj2[v].push_back(u);`
Adds 2 entries to total list of edges (u->v and v->u) which is correct for undirected traversal.
Correct.

The code looks solid.

```cpp
/*
Solution for Graph Isomorphism problem.
Uses degree sorting and neighbor degree signatures for initialization,
followed by randomized hill climbing to maximize matched edges.
*/
```
I will strip comments to keep it clean.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

const int MAXN = 2005;
bool adj1[MAXN][MAXN];
int n, m;
vector<int> adj2[MAXN];

struct NodeInfo {
    int u;
    int deg;
    vector<int> neighbor_degrees;
};

// Comparator for sorting vertices based on degree and neighbor degrees signature
bool compareNodes(const NodeInfo& a, const NodeInfo& b) {
    if (a.deg != b.deg) return a.deg < b.deg;
    if (a.neighbor_degrees.size() != b.neighbor_degrees.size())
        return a.neighbor_degrees.size() < b.neighbor_degrees.size();
    return a.neighbor_degrees < b.neighbor_degrees;
}

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n >> m)) return 0;

    // Initialize adjacency matrix for G1
    for(int i=0; i<=n; ++i) {
        for(int j=0; j<=n; ++j) adj1[i][j] = false;
    }

    // Read G1 edges
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj1[u][v] = true;
        adj1[v][u] = true;
    }

    // Read G2 edges and prepare degrees
    vector<int> deg2(n + 1, 0);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj2[u].push_back(v);
        adj2[v].push_back(u);
        deg2[u]++;
        deg2[v]++;
    }

    // Prepare data structures for G1
    // We need to iterate adj1 to get neighbor lists and degrees since adj1 was bools
    vector<int> deg1(n + 1, 0);
    vector<vector<int>> neighbors1(n + 1);
    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j) {
            if (adj1[i][j]) {
                neighbors1[i].push_back(j);
                neighbors1[j].push_back(i);
                deg1[i]++;
                deg1[j]++;
            }
        }
    }

    // Helper lambda to create NodeInfo
    auto build_info = [&](int idx, const vector<int>& adj_list, const vector<int>& degs) -> NodeInfo {
        NodeInfo info;
        info.u = idx;
        info.deg = degs[idx];
        info.neighbor_degrees.reserve(adj_list.size());
        for (int neighbor : adj_list) {
            info.neighbor_degrees.push_back(degs[neighbor]);
        }
        sort(info.neighbor_degrees.begin(), info.neighbor_degrees.end());
        return info;
    };

    vector<NodeInfo> nodes1, nodes2;
    nodes1.reserve(n);
    nodes2.reserve(n);

    for (int i = 1; i <= n; ++i) {
        nodes1.push_back(build_info(i, neighbors1[i], deg1));
    }
    
    for (int i = 1; i <= n; ++i) {
        nodes2.push_back(build_info(i, adj2[i], deg2));
    }

    // Sort vertices by their signature to find a good initial mapping
    sort(nodes1.begin(), nodes1.end(), compareNodes);
    sort(nodes2.begin(), nodes2.end(), compareNodes);

    // Initialize permutation: map G2 vertex u to G1 vertex p[u]
    vector<int> p(n + 1);
    for (int i = 0; i < n; ++i) {
        p[nodes2[i].u] = nodes1[i].u;
    }

    // Randomized Hill Climbing / Local Search to improve the mapping
    mt19937 rng(time(nullptr));
    uniform_int_distribution<int> dist(1, n);
    int max_swaps = 20000; 
    
    for(int i=0; i<max_swaps; ++i) {
        int u = dist(rng);
        int v = dist(rng);
        if(u == v) continue;
        
        int Au = p[u];
        int Av = p[v];
        int delta = 0;
        
        // Check neighbors of u in G2
        for(int w : adj2[u]) {
            if(w == v) continue; // Skip edge (u,v) as it's undirected and contribution change is 0
            // Contribution of edge (u, w)
            // Old: adj1[Au][p[w]]
            // New: adj1[Av][p[w]]
            if(adj1[Au][p[w]] && !adj1[Av][p[w]]) delta--;
            else if(!adj1[Au][p[w]] && adj1[Av][p[w]]) delta++;
        }
        
        // Check neighbors of v in G2
        for(int w : adj2[v]) {
            if(w == u) continue; // Skip edge (v,u)
            // Contribution of edge (v, w)
            // Old: adj1[Av][p[w]]
            // New: adj1[Au][p[w]]
            if(adj1[Av][p[w]] && !adj1[Au][p[w]]) delta--;
            else if(!adj1[Av][p[w]] && adj1[Au][p[w]]) delta++;
        }
        
        if(delta > 0) {
            p[u] = Av;
            p[v] = Au;
        } else {
            // Revert if no improvement
            p[u] = Au;
            p[v] = Av;
        }
    }

    // Output the permutation
    for(int i=1; i<=n; ++i) {
        cout << p[i] << (i==n?"":" ");
    }
    cout << "\n";

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=1099)

````
I need to find a permutation of vertices of the second graph G₂ that maximizes the number of matched edges with the first graph G₁.

Key observations:
1. This is finding the best isomorphism mapping between two graphs - it's computationally hard (Graph Isomorphism problem)
2. For n ≤ 2000, we can't try all n! permutations - that would be too slow
3. I should focus on preserving the **degree structure** of vertices
4. Vertices in corresponding positions (u ↔ p(u)) must have matching degrees

Strategy:
- Group vertices in each graph by their degree (vertices with same degree go together)
- Use backtracking search trying different mappings while respecting degree constraints
- Track which edges are successfully mapped
- Stop when we've found good enough matches or time limit reached

Implementation details:
- Precompute adjacency lists for both graphs
- Calculate degree for each vertex
- Organize vertices by degree into groups
- Backtrack through possible mappings within each degree group
- Count how many edges match under the constructed mapping
- Keep track of best solution found so far

This runs in approximately O(n²) due to the limited ways vertices of the same degree can map to each other and edge validation taking O(m).
</think>

```cpp
#include <vector>
#include <algorithm>
using namespace std;
int adj[2005][2005];
int deg[2005], g1_deg[2005], g1_adj[2005][2005], g2_adj[2005][2005];
int n, m, best_score;
int P[2005], used_g2[2005];

void backtrack(int g2, vector<int>& group_g1, vector<int>& group_g2, int idx, long long score, bool is_first = true) {
    if (score <= best_score || used_g2[g2]) return;
    P[g2] = group_g1[idx]; used_g2[P[g2]] = 1;
    best_score = max(best_score, score);
    for (size_t i = 0; i < group_g1.size(); ++i) {
        if (!used_g2[group_g1[i]]) {
            long long s = 0;
            if (is_first && deg[group_g1[i]] != deg[group_g2[i]]) continue;
            int g1v = group_g1[i], g2v = group_g2[idx];
            bool valid = true;
            for (int u : g2_adj[g2v]) {
                if (adj[P[u]][P[g1v]] != adj[g1v][g1v]) valid = false; break;
                if (u < g2v) { s += adj[g1v][g1v]; break; }
            }
            if (valid && !is_first) s = 0; // count new edges only
            backtrack(g2, group_g1, group_g2, idx + 1, score + s, !is_first);
            used_g2[P[g2]] = 0;
        }
    }
}

int main() {
    cin >> n >> m;
    for (int i = 1; i <= m; ++i) {
        cin >> u >> v; adj[u][v] = adj[v][u] = 1;
        g1_adj[u][v] = g1_adj[v][u] = 1;
        ++deg[u]; ++deg[v];
    }
    for (int i = 1; i <= m; ++i) {
        cin >> u >> v; g2_adj[u][v] = g2_adj[v][u] = 1;
        ++g1_deg[u]; ++g1_deg[v];
    }
    vector<int> groups_g1[2005], groups_g2[2005];
    for (int i = 1; i <= n; ++i) groups_g1[deg[i]].push_back(i);
    for (int i = 1; i <= n; ++i) groups_g2[g1_deg[i]].push_back(i);
    long long total_edges = 0;
    for (int u = 1; u <= n; ++u)
        for (int v = u + 1; v <= n; ++v)
            if (adj[u][v]) total_edges++;
    fill(P, P + n + 1, 0); fill(used_g2, used_g2 + n + 1, 0);
    best_score = 0;
    if (groups_g1[0].size() && groups_g2[0].size())
        backtrack(groups_g1[0].empty() ? 0 : groups_g1[0][0], groups_g2[0], groups_g1[0]);
    cout << "0\n";
    return 0;
}
```
````
