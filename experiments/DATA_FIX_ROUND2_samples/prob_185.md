# 题 185 — base均分 16.7 vs 直接SFT均分 0.0(掉点 16.7)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 83.3,completion_tokens=30808)

````
The problem asks us to find a large clique in a given undirected graph. A clique is a subset of vertices where every two distinct vertices are adjacent. This is the classic Maximum Clique problem, which is NP-hard. However, since $N$ is up to 1000 and we only need *a* large clique (heuristic optimization) rather than the absolute maximum, we can employ heuristic algorithms like Bron-Kerbosch with pivoting, Branch and Bound, or greedy heuristics combined with local search. Given the constraints ($N=1000$), an exponential time complexity algorithm is likely too slow if the optimal clique is small but the graph is dense, but often competitive programming problems with "find any large clique" have specific structures or allow reasonable timeouts for good heuristics.

Wait, the problem statement says:
"The value K* is known to the judge but not to the contestant."
"You are NOT required to find the theoretically optimal solution."
"Invalid solutions ... receive score = 0."
"Higher score is better."

This implies it's a contest problem where we get points proportional to how close our clique size is to the true maximum. If we output an invalid set (not a clique), we get 0. So correctness is paramount.

Constraints: $N \le 1000$. A general Maximum Clique algorithm like exact backtracking might be too slow in the worst case (e.g., random graphs near the Ramsey threshold). However, typical competitive programming instances for this type of problem might not hit the absolute worst cases of general graphs. But wait, usually finding max clique on N=1000 requires heuristics because exact algorithms are extremely slow on average hard instances.

Let's consider the complexity. $O(N^3)$ or $O(M \cdot N)$ might be acceptable.
Bron-Kerbosch with pivoting runs efficiently on many graphs but can degrade on certain ones. With $N=1000$, pure backtracking is risky.

However, there is a specialized technique for this kind of challenge. If we assume the test cases aren't specifically designed to kill backtracking, a strong backtracking solver with optimizations (pivoting, ordering vertices by degree) is standard. Another approach is using a randomized greedy strategy or simulated annealing, but those risk invalidity if not checked carefully, though checking validity at each step is crucial.

Given the scoring mechanism (Score = 0 if invalid), we MUST ensure the output is a valid clique. Heuristic approaches that stop early or make mistakes might produce invalid cliques. Therefore, we should prioritize finding a *valid* large clique. The most robust method for ensuring validity while trying to maximize size is a Maximal Clique finder.

Actually, simply running a fast heuristic search is best. A common efficient approach for "large clique" in limited time is:
1. Order vertices by degree.
2. Use a recursive backtracking algorithm (Bron-Kerbosch-like) but prune aggressively.
3. Or use a simple greedy construction and try to extend it.

Since $N=1000$, let's look at time limit: 2.0s. That gives roughly $10^8$ operations. An $O(2^{N/2})$ is impossible. We need something closer to polynomial or very efficient combinatorial search.
However, note that if the graph is dense (many edges), a clique is likely to exist. If the graph is sparse, the clique is small.
Maximum Clique is NP-Hard. For $N=1000$, typically no exact solution exists unless the graph has special properties or the clique is very small/large.
But the problem says "heuristic optimization". It suggests we don't need the global optimum.
A very strong heuristic is to sort vertices by degree descending and try to greedily add them if they connect to all currently selected nodes, then backtrack or restart if stuck? No, greedy doesn't guarantee optimality but guarantees validity.

Let's refine the greedy strategy.
Sort vertices by degree. Iterate through sorted vertices. Maintain a candidate set of vertices compatible with current clique.
Or better: Start with the vertex having the highest degree. Then iterate over its neighbors, filtering out those not connected to previous selections.
This finds a maximal clique rooted at that vertex. We should try multiple starting points or restart strategies.

Actually, a more robust "local search" or "simulated annealing" style approach could work. We start with an empty set. Pick random vertices or high degree ones. Add if compatible. If we reach a deadlock (cannot add any more), check if removing one vertex allows adding another, thus increasing size.

However, implementing a complex local search is error-prone during a coding session.
Is there a simpler logic?
The constraint $N \le 1000$ suggests maybe $O(N^2)$ or slightly higher.
Let's consider the adjacency matrix/bitset approach. Using `std::bitset` or a custom bitset representation speeds up intersection checks from $O(N)$ to $O(N/64)$.
Algorithm:
1. Build adjacency bitsets.
2. Sort vertices by degree (descending).
3. Try to construct a clique.
   Greedy construction: Start with vertex $u$. Current clique $C=\{u\}$. Remaining candidates $R = Neighbors(u)$.
   While $R$ is not empty:
     Pick $v \in R$ with max degree within $R$. Add to $C$. Update $R = R \cap Neighbors(v)$. Remove $v$ from consideration? No, just update $R$.
   This finds *a* clique quickly.
   To improve, run this process multiple times with different starting vertices or random permutations of neighbors selection.

To maximize $K$, we can do the following:
Iterate over all vertices as potential starting points. For each starting vertex, run a "greedy" expansion to find a large clique. Track the maximum size found.
With bitsets, checking compatibility is fast.
Complexity: $N$ starts $\times$ (Greedy steps). In each step, we intersect bitsets. Intersection takes $O(N/W)$. Number of steps is bounded by clique size. Total time roughly $O(N \cdot K_{max} \cdot N/W)$. Since $K_{max} \le N$, worst case $O(N^3/W)$. With $N=1000$, $N^3 \approx 10^9$. With $W=64$, it's $1.5 \cdot 10^7$ ops, which fits easily in 2s.

So the plan:
1. Read input. Store graph as adjacency bitsets (`std::vector<std::bitset<1001>>`).
2. Compute degrees for each vertex.
3. Loop through each vertex $i$ from 1 to $N$:
    Treat $i$ as the first element of the clique.
    Let `candidates` be the neighbor bitset of $i$.
    Let `current_clique` contain $i$.
    While `candidates` has bits set:
       Pick a vertex $j$ from `candidates`. Strategy: pick $j$ with maximum number of neighbors inside `candidates` (or just pick highest degree globally). Actually, a simple heuristic is to pick the vertex in `candidates` that has the highest degree among `candidates`.
       Add $j$ to `current_clique`.
       Update `candidates` = `candidates` & `adj[j]`.
       (Need to handle index mapping carefully).
    Record size of `current_clique`.
    Repeat.
   
Wait, simple greedy (always pick highest degree node available) isn't necessarily finding the max clique. But trying all starting vertices helps significantly.
Is there a risk of Time Limit Exceeded (TLE)?
Bitset intersection is fast. $1000 \times 1000 / 64 \approx 15625$ ops per inner loop iteration?
Actually, iterating vertices to find the "best" candidate in `candidates` takes $O(K \cdot N)$. Doing this $N$ times outside, and potentially iterating candidates...
Optimization: Instead of scanning candidates to find max degree, we can maintain a list of vertices in candidates?
Actually, since we want to code it correctly within limits, let's simplify the heuristic.
Randomized Greedy + Local Search.
Start with a clique. Try to swap members to increase size.
However, "score = 0 if invalid". So we must strictly ensure clique property.
The most reliable way to generate a valid clique that isn't trivially small is the Bron-Kerbosch algorithm with some pruning.
Given the "Heuristic Optimization" context, we should probably implement a decent implementation of Bron-Kerbosch with pivoting, which is much faster on random inputs than naive recursion, and handles dense subgraphs well.
For $N=1000$, pure BK is okay if tuned. But BK can visit many states.
Wait, if I use BK, I will find the MAXIMUM clique (K*). Wait, BK with pivoting finds ALL maximal cliques (usually) or is restricted to a branch and bound search for Max Clique.
Standard BK enumerates maximal cliques. If we stop after finding one, it's a maximal clique, but not necessarily maximum.
If the graph has many maximal cliques, finding the largest one among them is tricky without exploring them all.
However, usually "Max Clique Challenge" in contests (like ICPC regionals or online judges) accepts a reasonably sized clique found via heuristics.
A known trick is: **Vertex Ordering**.
Order vertices $v_1, v_2, ..., v_N$ such that their degrees are decreasing.
Then apply BK/branching with this ordering.
Pruning: If the max possible clique size we can achieve from current state plus current clique size $\le$ best found so far, backtrack.
This effectively implements Branch-and-Bound for Maximum Clique.
With bitset optimization, the intersection checks are fast.
Pseudo-code for Max Clique (Branch & Bound):
Global `best_clique_size = 0`
`visited` array.
Function `solve(candidates)` where `candidates` is the set of vertices compatible with current partial clique.
Sort `candidates`? No, order matters.
We maintain a set of candidate vertices `P`, a set of excluded vertices `X` (to avoid duplicates in maximal clique enumeration), and `R` (remaining vertices to add to clique, but here we just track the current upper bound).
Actually, a simplified recursive function:
`solve(current_clique_size, P)`:
  Upper bound calculation: `UB = |P|`? No. `UB = |P| + current_clique_size`.
  Prune if `UB <= best_so_far`.
  Pick a pivot $p \in P \cup X$ (from the graph's perspective, to reduce branching).
  Iterate over $v \in P \setminus Neighbors(p)$:
    Recurse with `new_P = P \cap Neighbors(v)`.
    `new_X = X \cap Neighbors(v)`.
    Add $v$ to `current_clique`.
    Recurse.
    Remove $v$ from `current_clique` and `P`.
    
With $N=1000$, this might still be slow if the graph is hard.
Let's reconsider the bitset greedy approach with bit manipulation. It's very fast in C++.
Strategy:
Use `std::bitset<1001> adj[1001]`.
Compute `degree[v]` for all $v$.
Loop `start_node` from 1 to `N`.
  `clique` contains `{start_node}`.
  `candidates` = `adj[start_node]`.
  While `candidates.any()`:
    Find $v \in candidates$ that maximizes `count_adj[candidates]` or `degree`.
    A simple heuristic: pick the $v$ in `candidates` that has the highest `degree` in the ORIGINAL graph. Or better, highest intersection with current candidates? Calculating intersection size takes time. High original degree is a safe tie-breaker or proxy.
    Actually, the standard heuristic for this specific "Heuristic Max Clique" task (often found in Codeforces/AtCoder training) is:
    1. Sort all vertices by degree descending.
    2. Maintain a list of potential vertices $S$. Initially $S$ = all vertices.
    3. Iterate through sorted vertices.
    4. If $u \in S$ and $u$ is compatible with all currently chosen clique members, add $u$ to clique. Else remove $u$ from $S$.
    This is a filter-based greedy.
    Algorithm:
      Clique $C = \emptyset$.
      Candidate Set $S = V$.
      While $S$ is not empty:
        Pick $v \in S$ with max degree (or first in sorted list).
        If $v$ is connected to all $u \in C$:
          Add $v$ to $C$.
          $S = S \cap Neighbors(v)$.
        Else:
          Remove $v$ from $S$. (Or keep $S$ same but don't add).
      
      Wait, this logic removes vertices. Removing $v$ means we give up on using $v$ in this clique attempt.
      This finds a maximal clique determined by the order.
      To optimize: Try all permutations of the sorted vertices? Too many.
      
      Correct heuristic often used:
      Sort vertices by degree descending.
      Construct a maximal clique by processing vertices in this order.
      Maintain `potential_nodes`. Initialize to all vertices? No, we pick the first one, restrict next candidates to its neighbors.
      To beat the initial greedy, restart with different pivots?
      
      Since we need to submit C++ code and time limit is 2s, let's write a Branch and Bound algorithm optimized with bitsets. This is the standard solution for $N=100$-ish problems. For $N=1000$, it needs to be very tight.
      However, the problem says "You are NOT required to find the theoretically optimal solution".
      Maybe we can afford to run the greedy search many times with random restarts?
      
      Let's try a combination:
      1. Precompute adjacency bitsets.
      2. Identify "heavy" vertices (high degree). These are good candidates for cliques.
      3. Run a "local search": Start with a clique of size > 1. Try to swap vertices to increase size.
         This is safer than raw backtracking.
         
      Wait, if I start with a clique, say size $k$. Can I verify if I can improve it?
      Iterative Improvement:
      Start with $C = \{v_{max\_degree}\}$. Expand to maximal clique using greedy (pick node in candidates with max degree).
      Now we have a maximal clique. Is it maximum? Probably not.
      Can we grow it?
      We can check neighbors of vertices in $C$. If a neighbor $x \notin C$ connects to all $C \setminus \{y\}$ for some $y \in C$, swapping $y$ with $x$ might help.
      This is the "vertex exchange" neighborhood search.
      
      Better: Since we want a *large* clique, and we have a 2s time limit, a simple greedy with pivoting is the most robust.
      Let's implement a Bitset-based Max Clique heuristic.
      
      Algorithm details:
      - Input $N, M$.
      - Adjacency Matrix `adj[N+1][N+1]` or `bitset`. Since $N=1000$, `bitset<1001>` is perfect.
      - Sort vertices by degree.
      - Global `best_clique_size = 0`, `best_clique_mask`.
      - Recursive function `BK(P, R, X)`? No, just standard Branch and Bound.
      
      Let's look at the constraints again. $N \le 1000$. Time 2s.
      A strict Max Clique solver might TLE on worst case random graphs (around $N/2$ density).
      However, since we just need "High Score", maybe we can find a clique of size $K \approx K^*/2$ or $K^*/1.5$ easily.
      A "Fast Maximal Clique" finder is sufficient if we run it multiple times with different seedings/orders.
      
      Proposed Solution Logic:
      1. Read input, build adjacency bitsets.
      2. Calculate degrees.
      3. Sort vertices by degree descending.
      4. Perform a randomized search:
         Shuffle vertices.
         Greedily build a clique based on the shuffled order.
         Repeat this ~50-100 times.
         Keep the largest valid clique found.
         
      How does the Greedy build work?
      We select an ordered sequence of vertices $v_1, v_2, \dots, v_N$.
      We want to find indices $i_1 < i_2 < \dots < i_k$ such that $\{v_{i_j}, v_{i_l}\} \in E$ for all $j, l$.
      This is equivalent to: start with $v_{i_1}$. Next pick $v_{i_2}$ from remaining vertices that connect to $v_{i_1}$. And so on.
      BUT, we don't pick *any* available neighbor. We pick based on some priority.
      Standard algorithm:
      Let `active_set` initially be all vertices.
      Pick $u \in active\_set$ with highest degree (tie break with original index).
      Start clique $C = \{u\}$.
      Filter `active_set` to only include neighbors of $u$.
      Repeat until `active_set` is empty or we want to stop? No, we can continue picking from remaining valid nodes.
      But to avoid getting stuck, once we filter `active_set`, we should pick the one with highest degree in the *original* graph (or induced subgraph).
      Usually, sorting vertices by degree and then picking the first one works as a pivot.
      
      Wait, the simplest greedy that works well:
      Sort vertices by degree descending.
      Let `candidates` = all vertices.
      Pick $v_1 = candidates[0]$ (highest degree).
      $C = \{v_1\}$.
      $candidates$ = $neighbors(v_1) \cap candidates$.
      From remaining, pick $v_2$ with max degree.
      Update $candidates = neighbors(v_2) \cap candidates$.
      Repeat until no candidates left.
      This yields one maximal clique rooted at $v_1$.
      Since we fixed the order based on degree, doing this for *each* vertex as start might be needed?
      Or just do it once?
      Usually, picking the start vertex matters. Trying multiple start vertices (especially the highest degree ones) is useful.
      
      Wait, "Try multiple start vertices" is basically iterating over the top $K$ vertices as roots.
      Also, random shuffling of the vertex list before applying the "filter" logic helps escape bad local optima of the greedy path.
      
      Let's combine:
      Loop 50 times:
        Shuffle vertices $P$.
        Greedily construct clique $C$ from $P$:
          $C = \{ P[0] \}$.
          $Candidates = Neighbors(P[0])$.
          For $i = 1$ to $N-1$:
            Let $u = P[i]$.
            If $u \in Candidates$:
              $Candidates = Candidates \cap Neighbors(u)$.
              Add $u$ to $C$.
            // Check if Candidates is empty? If so break.
        Update Best.
        
      Wait, this is "finding a clique compatible with prefix". No, that's not right.
      Correct Greedy with Shuffled List:
      Iterate through the shuffled list. Maintain the set of vertices that are compatible with the current clique $C$. Initially $Candidates = \{1..N\}$.
      If $u \in Candidates$, consider adding to $C$.
      Should we add every compatible vertex? That forms a clique?
      Yes! Because if we process in some order and pick ANY vertex compatible with CURRENT $C$, since $C$ only changes by addition, if $u$ connects to all current, it connects to all future additions (because future additions must also connect to all current).
      Wait, if we add $u$, then any subsequent $v$ must connect to $u$ AND all previous nodes.
      So, algorithm:
        Shuffle permutation $P$.
        $C = \emptyset$.
        $Candidates = AllVertices$.
        For $u$ in $P$:
          If $u \in Candidates$:
             Is $u$ connected to ALL vertices in $C$?
             Actually, maintaining $Candidates$ ensures $u$ is connected to all PREVIOUSLY ADDED vertices in $C$.
             But wait, $Candidates$ definition: initially $V$. When we add $u$ to $C$, new $Candidates = Candidates \cap Neighbors(u)$.
             Any $v$ picked later from new Candidates automatically connects to $u$.
             So yes, the condition simplifies to: $u \in Candidates$.
             If so, add $u$ to $C$.
        
        Is this correct?
        Suppose $P = [u, v, w]$. Edges: $(u,v), (v,w)$. Not $(u,w)$.
        Start $Candidates = \{u, v, w\}$.
        $u$: Connects to nothing yet. Add to $C=\{u\}$. Update $Candidates = \{u, v\}$ (neighbors of u). Assume $v \in N(u)$, $w \notin N(u)$.
        Next $v$: $v \in Candidates$. Add to $C=\{u, v\}$. Update $Candidates = \{v\} \cap \{w\}$ (neighbors of $v$).
        Next $w$: $w \notin Candidates$. Skip.
        Result: $\{u, v\}$. Valid.
        If we had $u, w$ connected but $u, v$ not?
        Sorted order matters.
        
        Complexity: $O(T \cdot N \cdot (N/64))$. With $T=100$, $N=1000$, operations $\approx 100 \cdot 1000 \cdot 16 \approx 1.6 \cdot 10^6$. Very fast.
        This seems too easy?
        Why is $N=1000$ given? Maybe to ensure we use bitsets.
        But this simple greedy usually finds a small clique.
        We want to maximize it.
        Does the order of processing matter? Yes.
        Vertices with higher degree should likely come earlier in the permutation to capture more connections.
        Sorting by degree descending BEFORE random shuffle makes sense.
        
      Refined Heuristic:
      1. Compute degrees.
      2. Sort vertices by degree descending. Let this array be `sorted_v`.
      3. Shuffle `sorted_v` partially or fully to explore different orderings.
         Actually, fully randomizing the top-k vertices is enough?
         Let's just perform the "Sorted Greedy" algorithm with a few restarts using random perturbations.
         Restart strategy:
         For $k = 1$ to NumRestarts:
           Create a copy of vertices, sort by degree.
           Randomly permute the copy.
           Run greedy clique finding on permuted list.
           Keep track of max clique found.
           
         Wait, if I use the "Shuffle then Greedy", I should sort by degree *then* shuffle?
         Or shuffle degrees?
         Actually, the most effective heuristic for large cliques is often "Greedy" with vertices sorted by degree.
         However, simply sorting by degree and taking the first one might trap you.
         Let's try:
         Outer loop (randomized):
           Initialize `candidates` to all vertices.
           Sort `candidates` by degree descending.
           $C = \emptyset$.
           For each $v$ in `candidates`:
             If $v$ is connected to ALL $u \in C$:
               Add $v$ to $C$.
               Update `candidates`: remove $v$ and any non-neighbors of $v$. (Actually, just intersect $Candidates$ with $Neighbors(v)$).
           This finds a clique.
           
           Wait, is "connected to all $u \in C$" guaranteed by intersecting `candidates` with `Neighbors(v)`?
           Yes, if we maintain the invariant that `candidates` contains only neighbors of all currently selected vertices in `C`.
           Initial `candidates` = All vertices. `C` = Empty.
           Pick $v$ (e.g., first in sorted list). $C = \{v\}$. `candidates` = `candidates` $\cap$ `Neighbors(v)`.
           Pick next $w$ from `candidates` (since we iterate sequentially, $w$ is naturally a neighbor of $v$).
           $C = \{v, w\}$. `candidates` = `candidates` $\cap$ `Neighbors(w)`.
           Repeat.
           Since we iterate $v$ in sorted order (by degree), we prioritize high-degree vertices.
           This creates a "Maximal Clique" rooted at the current starting point.
           But wait, we pick ONLY from `candidates`. Once a vertex is removed from `candidates`, it's gone.
           Does this cover all possibilities? No.
           Example: Triangle $1-2, 2-3, 3-1$. Degrees equal.
           Order 1, 2, 3.
           Start 1. `candidates` becomes {1, 2, 3}.
           Next 2. Compatible. `candidates` becomes {2}.
           Next 3. Incompatible (removed).
           Clique {1, 2}. Size 2. Missed {1, 3}? No, {1, 3} is edge. But algorithm picks 2 first.
           If we pick 3 before 2?
           Order 1, 3, 2.
           Start 1. `candidates` = {1, 2, 3}.
           Next 3. Compatible. `candidates` = {3}. (Intersection of {1,2,3} and Neighbors(3)={1,2}).
           Next 2. Compatible? No, 2 is in {3}? Wait.
           My manual trace is flawed.
           Let's re-verify the set logic.
           `candidates` represents vertices that can be added to $C$ WITHOUT violating edge conditions WITH EXISTING C members.
           Wait, if I process vertex $u$, and I decide to include it in $C$, then *subsequent* candidates must also be neighbors of $u$.
           What about vertices I skipped? They were in `candidates` but we chose not to add them (because we iterate the whole list and pick *every* compatible one?).
           NO. The standard Greedy algorithm for Max Clique *with an ordering*:
           Iterate $v$ in Order:
             If $v$ is compatible with $C$ (which is maintained by keeping `current_valid_candidates`):
               Add $v$ to $C$.
               Filter `current_valid_candidates`.
             
           Wait, if we skip a vertex $v$, we might regret it later.
           Actually, the "Maximal Clique Finder using Vertex Ordering" typically works as follows:
           You want to find *all* maximal cliques, but we just need *one*.
           To find a maximal clique rooted at a specific vertex $r$, we do:
           $C = \{r\}$. $R = Neighbors(r)$.
           While $R$ not empty:
             Pick $u \in R$ (e.g. max degree).
             Add $u$ to $C$.
             $R = R \cap Neighbors(u)$.
             Remove $u$ from $R$ (and others not compatible?).
             (Here we are essentially finding the branch of the search tree).
             
           To optimize for *largest*, we need to try different starting vertices $r$, and different orders for selecting from $R$.
           
           The most practical solution:
           Implement the "Bron-Kerbosch with pivoting" using bitsets.
           It is guaranteed to run reasonably fast for typical CP graphs unless constructed maliciously. With $N=1000$, it might be on the edge.
           However, we are allowed to output *any* valid clique, maximizing score.
           A simplified BK is:
           Function `Search(C, P, X)`:
             If $P$ is empty and $X$ is empty: return (Found Maximal Clique C)
             Pivot $u$: vertex in $P \cup X$ maximizing $|P \cap N(u)|$.
             Iterate $v \in P \setminus N(u)$:
                Recurse `Search(C U {v}, P \cap N(v), X \cap N(v))`
                Remove $v$ from $P$, add to $X$.
             Return
            
           This finds ALL maximal cliques. We track max size.
           Optimization: We only need ONE large clique. If $|C| + |P| \le BestSize$, prune.
           With bitsets, calculating $|P|$ is just `.count()`. Checking intersection is fast.
           Pivot choice reduces branching factor.
           This is the "Best Bet" for competitive programming "Max Clique" with $N=1000$.
           Given the 2.0s limit, we must ensure efficient bitset usage.
           
           Implementation Details:
           - Use `std::bitset<1001> adj[1001]`.
           - Function `count_set_bits()`.
           - Pass by reference or global variables? Pass by value (for bitsets) copies are cheap (bitset copy is block copy). $1000$ bits is 16 integers. Copy is negligible.
           - `main` calls `BK({bit(0)}, adj_all, adj_none)`.
           - `BK` tracks global `max_clique`.
           - `best_clique` mask to print later.
           
           Corner Case: Disconnected components? The algorithm handles it (returns all).
           Multiple edges? Handled (set logic).
           Self loops? Constraint says $u \neq v$.
           
           One catch: Finding the exact Maximum Clique is still hard for BK on specific "tough" graphs.
           However, for a heuristic problem where we just need a big clique, and "Score = 0" for invalid, BK finds VALID maximal cliques. Among maximal cliques, we want the largest.
           BK with pivoting is efficient enough for finding the Maximum Clique on many inputs.
           If there are many maximal cliques of similar size, it will finish quickly.
           If there's a unique giant clique, it finds it quickly.
           The worst case is when the optimal clique is small but there are exponentially many maximal cliques of size slightly smaller? No, usually hard cases are around $N \log N$ or similar.
           
           Let's write the BK solution. It's the most rigorous valid approach within reason.
           Wait, with $N=1000$, copying bitsets recursively might add overhead.
           We should pass by value for `P` and `X` if possible, but that might cause stack overflow or memory bloat?
           Depth of recursion is at most $N$. Stack depth 1000 is fine.
           Copying `bitset<1001>` is $\approx 128$ bytes. $1000 \times 128$ is small.
           However, creating too many objects might slow down.
           Optimization: Don't create `P` and `X` bitsets in arguments if we can avoid it.
           Just pass pointers or references?
           Wait, modifying `P` (intersection) in place is harder because recursion branches.
           Usually people pass by value. It works.
           
           Also, we should initialize `P` to `AllVertices` and `X` to `Empty`.
           But `P` and `X` logic:
           Initial call: `BK(P, X)`. `P` is all vertices. `X` is empty.
           Inside `BK`:
             Pivot $u \in P \cup X$. Maximize $|P \cap N(u)|$.
             Note: $u$ must be in $P \cup X$.
             If we use pivoting, we iterate $v \in P \setminus N(u)$.
             Wait, `N(u)` refers to the neighbors in the GRAPH.
             Pivot rule: choose $u \in P \cup X$ such that $|P \cap N(u)|$ is maximized.
             Then iterate $v \in P \setminus N(u)$.
             Why? Because every clique in $P$ must either be in $N(u)$ or it must contain $u$ (actually, the pivot logic ensures we don't miss any maximal clique by covering subsets).
             Specifically, if a maximal clique contains a vertex $v$ from the iteration set, it won't be missed if we ensure we explore necessary paths.
             The standard pivot rule: Pick $p \in P \cup X$ such that $|P \cap N(p)|$ is maximized.
             Then we only need to iterate over $v \in P \setminus N(p)$.
             Reason: Any clique containing vertices from $N(p) \cap P$ can be handled by other iterations.
             Actually, the standard BK is:
             ```cpp
             void BronKerbosch(bitset<N> P, bitset<N> X, vector<int>& res_clique) {
                 if (P.empty() && X.empty()) {
                     update_max_clique(res_clique);
                     return;
                 }
                 int pivot = -1;
                 int max_neighbors = -1;
                 bitset<N> candidates = P; // We need to pick pivot from P U X
                
                 // Efficiently find pivot
                 // Iterate vertices?
                 // We can optimize this loop.
                 
                 for (int i = 0; i < N; ++i) {
                     if ((P[i] || X[i])) { // is i in P or X
                         int n_cnt = (P & adj[i]).count(); // count neighbors in P
                         if (n_cnt > max_neighbors) {
                             max_neighbors = n_cnt;
                             pivot = i;
                         }
                     }
                 }
                 
                 if (pivot == -1) return; // Should not happen if P not empty
                 
                 bitset<N> P_copy = P;
                 // We iterate v in P such that v is NOT neighbor of pivot.
                 // But we need to update P and X.
                 // Since P changes, we iterate through the current P.
                 // To make it stable, we collect vertices to iterate.
                 
                 vector<int> vertices_in_P;
                 for(int i=0; i<N; ++i) if(P[i]) vertices_in_P.push_back(i);
                 
                 for(int v : vertices_in_P) {
                     // Check if v != neighbor of pivot?
                     // Condition: iterate v in P s.t. v not in N(pivot).
                     if (P_test(v, pivot) && !is_neighbor(v, pivot)) { 
                        // Wait, P[i] check is sufficient.
                     }
                     
                     // Optimization: Only visit v if not neighbor of pivot
                     if (v != pivot && !(adj[pivot][v])) { // Need correct check
                        // Recurse
                    }
                 }
                 // This O(N) scan for pivot inside recursion is expensive.
                 // We can precompute pivot index outside? No, depends on P.
                 // But we can compute intersection size quickly.
             }
             ```
           Wait, iterating $N$ inside every recursion level is $O(N^2)$ total?
           Depth is $N$. At each level, we iterate $N$ to find pivot. Total $O(N^2)$.
           Plus intersections $O(N/64)$.
           With branching, we might repeat.
           Actually, we don't need to scan $N$ every time. We can iterate over `P` directly.
           `for (int v = 0; v < N; ++v) if (P[v]) ...`
           This is efficient enough.
           
           One critical detail: The problem requires outputting the clique.
           We need to store the best clique found so far.
           `std::vector<int> best_clique`.
           Global `int max_k`.
           Whenever we find a full clique, update `max_k` and `best_clique`.
           
           Wait, checking `P` and `X` emptiness is $O(N)$ or $O(1)$ with bitsets.
           The recursion structure is key.
           
           Another optimization:
           Use `std::vector<int>` to represent `P` and `X` instead of bitsets?
           Bitsets are great for intersection, but iterating elements in `P` to modify/recursion needs to extract indices.
           Extracting indices from `bitset` takes time.
           Maybe passing `vector<int>` for `P` and bitset for `adj` lookup is faster?
           Or keep bitset for `P` and `X` for fast intersection, and iterate using `test()`?
           Iterating 1000 bits in recursion might be slow if depth is deep and many recursive calls.
           
           Actually, a simpler heuristic:
           Just use the "Greedy with Bitset" multiple times.
           Why? Because it's easier to implement and less prone to infinite recursion or timeout if the graph is a line (where BK explodes).
           Line graph: $1-2-3-4...$. Max clique is 2.
           BK explores every pair.
           For $N=1000$ line graph: $O(N^2)$. Fast.
           For $N=1000$ random graph: BK is fast.
           The hard case is when there are many maximal cliques.
           E.g. disjoint triangles, etc.
           BK handles this well.
           
           Let's stick to BK with bitsets and pivoting.
           To make it robust:
           1. Use `vector<int>` for `P` elements to avoid scanning bitset if we can avoid.
           Wait, `P` as a `bitset` allows fast intersection. `P.intersect(Q)` returns new bitset.
           Extracting active elements: `P._Find_first()`? `std::bitset` doesn't expose internal storage easily for iteration. `P.test(i)` is fast.
           Iterating 1000 bits: $1000$ iterations. Inside recursion, we iterate to find pivot.
           Total cost: $\sum (\text{number of recursive calls}) \times N$.
           If number of calls is large, this TLEs.
           
           Alternative: Maintain `P` as a dynamic list.
           `void Solve(vector<int>& P, vector<int>& X)`
           - Pivot selection: $O(|P| + |X|)$.
           - Intersection: Vector intersection is $O(\min(|P|, |Q|))$.
           - This might be slower if sets are dense (size $O(N)$). Bitset is $O(N/64)$.
           So Bitset is better.
           But extracting list of vertices from bitset takes $O(N)$.
           So we can't extract and iterate quickly?
           Actually, we iterate $v$ from $0$ to $N-1$. Check `if (P[v] && v != pivot)`. This is $O(N)$.
           It seems unavoidable.
           
           Wait, we can optimize pivot finding.
           We only care about maximizing neighbors in `P`.
           If we know the order of vertices, we can check degrees?
           Not exactly.
           
           Let's reconsider the "Greedy" approach.
           The problem asks for HEURISTIC OPTIMIZATION.
           It explicitly mentions "Heuristic optimization problem".
           This strongly implies we don't need the exact Max Clique, just a good one.
           So we should prioritize speed and simplicity to get a LARGE clique.
           The Greedy strategy:
           Sort vertices by degree.
           Run the "Filter" loop:
           Iterate through sorted vertices. If vertex connects to ALL already picked, pick it.
           This is $O(N^2)$ (naively) or $O(N \cdot K_{found} \cdot N)$?
           Maintaining `candidates` bitset makes it $O(N \cdot N/64)$.
           This is very fast.
           But does it find a huge clique?
           On dense graphs, yes.
           On specific graphs, maybe not.
           Combining with multiple restarts (shuffles) increases coverage.
           
           Let's write a code that:
           1. Implements the Bitset-based Greedy Maximal Clique Finder.
           2. Runs it for the original sorted order.
           3. Also runs it with the reversed order.
           4. Also runs it with random shuffles.
           5. Keeps the largest result.
           This is highly likely to yield a high score without risking TLE due to worst-case BK.
           Also, it satisfies "Valid Solution" requirement perfectly.
           
           Code Structure:
           - Adjacency Matrix `bitset<1001> adj[1001]`.
           - `vector<int> degree`.
           - Function `get_clique(bitset<1001> candidates)`:
               This is tricky. The `get_clique` logic described earlier was slightly loose.
               Let's formalize:
               `vector<int> clique = {starting_node};`
               `bitset<1001> current_intersection = adj[starting_node];`
               `vector<int> candidates_list;` (indices where `current_intersection` is 1).
               While loop:
                  Pick best `v` from `candidates_list` (heuristic: highest degree).
                  Add `v` to `clique`.
                  Intersect `current_intersection` with `adj[v]`.
                  Update `candidates_list` with new bits.
               Return `clique`.
               
           Wait, this finds a clique *rooted* at `starting_node`.
           We should try all `i` in $1..N$ as starting_node?
           $N$ runs. Each run takes $O(K_{avg} \cdot N/64)$.
           Total time $O(N^2/64)$. Extremely fast.
           Wait, finding "best `v`" from list.
           We can maintain `candidates_list` indices.
           To avoid scanning, we can pre-sort vertices by degree.
           Iterate `v` in the pre-sorted order.
           If `v` is in `current_intersection`, we pick it and recurse? No.
           
           Let's use the "Order-Based Max Clique" algorithm:
           We process vertices in a fixed order $u_1, u_2, \dots, u_N$.
           We build a clique $C$ initialized empty.
           Iterate $i$ from 1 to $N$:
             If $u_i$ is compatible with $C$ (check edges against all $u \in C$):
               Add $u_i$ to $C$.
               Remove $u_i$ from consideration? No, keep it in $C$, it blocks others.
               Mark incompatible vertices?
             
           Wait, if we skip $u_i$, it remains compatible for future?
           Yes.
           But if we add $u_i$, we enforce that future choices must connect to $u_i$.
           
           This simple greedy produces a clique.
           To optimize, we should process vertices by degree (descending).
           Why? Because high degree nodes are more central, likely part of a larger clique.
           By putting them early, we are forced to accept them (or reject them?), wait.
           If we put $v$ early and it's compatible, we *add* it. This restricts the space.
           If we put $v$ late, we might have built a clique $C$ without it. Then we encounter $v$. If $v$ is compatible with $C$, we add it.
           This seems like a "Filter" algorithm.
           Let's trace:
           Order: $v_1, \dots, v_N$.
           $C = \emptyset$.
           $CompatibleSet = \{v_1, \dots, v_N\}$.
           For $v$ in Order:
             If $v \in CompatibleSet$:
               Is $v$ connected to all in $C$?
               Note: If $v \in CompatibleSet$, it implies it was connected to all previously ADDED vertices?
               Wait. $CompatibleSet$ needs to be updated.
               Correct logic:
               $CandidateSet = \{1, \dots, N\}$.
               $C = \emptyset$.
               For $u$ in Permutation:
                  If $u \in CandidateSet$:
                     Add $u$ to $C$.
                     $CandidateSet = CandidateSet \cap Neighbors(u)$.
               
               This logic finds the largest clique consistent with the permutation IF we greedily take EVERY compatible vertex.
               Does taking EVERY compatible vertex form a clique?
               Suppose $u, v$ are processed. $u$ comes first.
               $u$ enters $C$. $CandidateSet$ updates to exclude non-neighbors of $u$.
               $v$ comes second. $v$ is in updated $CandidateSet$, so $v$ is neighbor of $u$. $v$ added to $C$.
               $CandidateSet$ updates further.
               Any $w$ comes third. If $w$ in set, $w$ is neighbor of $u$ and $v$.
               Yes, this forms a clique.
               
               So the algorithm is:
               1. Define permutation $P$.
               2. $CandidateSet = \{1..N\}$.
               3. $C = \emptyset$.
               4. For $u$ in $P$:
                   If $u \in CandidateSet$:
                      $C.add(u)$.
                      $CandidateSet = CandidateSet \cap Neighbors(u)$.
               5. Return $|C|$.
               
               Is this optimal? No, it depends on $P$.
               However, if we run this for multiple permutations (especially degree sorted ones), we find a large clique.
               Specifically, sorting $P$ by degree descending is the best single heuristic.
               Running it $T$ times with random shuffles of degree-sorted list (top $N/10$ vertices?) covers bases.
               Wait, random shuffling the whole list destroys the degree benefit.
               Better: Sort by degree. Then try permutations of the top $K$ elements?
               Or just one sorted pass.
               Wait, sorting by degree puts highest degree first.
               If graph is star (center degree $N-1$, leaves degree 1).
               Sorted: Center, Leaves.
               Run:
               1. Process Center. Add to $C$. $Candidates \cap Neighbors(Center) = Leaves$.
               2. Process Leaf 1. Add to $C$. $Candidates \cap Neighbor(Leaf1) = \{Center, Leaf 1\} \cap \{...\}$.
                  Wait. $Neighbor(Leaf1)$ contains Center? Yes. Contains Leaf1? No.
                  Intersection removes Leaf 1 (not self loop).
                  So $C=\{Center, Leaf 1\}$. $Candidates$ becomes empty (since Leaves not connected to Leaf 2).
               3. Stop. $C=\{Center, Leaf 1\}$. Size 2.
               Optimal is $\{Center, Leaf 1\}$ (or any leaf).
               
               Graph: Triangle 1-2, 2-3, 3-1. Degree 2, 2, 2.
               Sorted: 1, 2, 3.
               1. 1. $C=\{1\}$. Candidates=$\{2, 3\}$.
               2. 2. $C=\{1, 2\}$. Candidates=$\{2, 3\} \cap \{1, 3\} = \{3\}$.
               3. 3. $C=\{1, 2, 3\}$. Candidates=$\{3\} \cap \{1, 2\} = \emptyset$.
               Size 3.
               
               What if 1-2, 2-3, 3-4? Line.
               Sorted: 2(2), 3(2), 1(1), 4(1).
               1. 2. $C=\{2\}$. Cand=$\{1, 3\}$.
               2. 3. $C=\{2, 3\}$. Cand=$\{1, 3\} \cap \{2, 4\} = \emptyset$. (Assuming 1 not neighbor 3).
               Wait, 1 is neighbor 2. 4 is neighbor 3.
               Intersection $\{1, 3\} \cap \{2, 4\}$. 3 is in $\{2, 4\}$? No. 1 is in $\{2, 4\}$? No.
               Empty.
               Size 2. Correct max is 2.
               
               It seems this greedy algorithm with "Degree Sort" is quite good.
               We should run this on the sorted list.
               But to be sure we don't miss the max, we can reverse the list (Low degree to High degree)?
               Maybe.
               Let's implement this greedy algorithm with a few variations:
               1. Original Sort (Desc).
               2. Reverse Sort (Asc).
               3. Random Shuffle.
               
               Wait, for the "Reverse Sort" (Asc), we pick low degree nodes first.
               Star graph: Center (High), Leaves (Low).
               Asc: Leaf 1, Leaf 2...
               1. Leaf 1. $C=\{L1\}$. Cand=$\{Center\}$.
               2. Leaf 2. In Cand? No (assuming Leaf 2 not neighbor L1).
               ...
               Result $C=\{L1\}$. Size 1. Terrible.
               
               So Desc sort is critical.
               However, what if the triangle case?
               Nodes 1, 2, 3.
               Desc: 1, 2, 3 -> OK.
               
               Are there cases where Desc sort fails?
               Yes, "Star of Triangles".
               Central node $v$. Surrounding nodes $u_1 \dots u_k$ connected to each other in triangles?
               If we have a large clique of high degree nodes, they should appear first.
               If we have a clique of size $K$, these nodes have high degree (at least $K-1$).
               Sorting by degree groups them.
               The risk is picking a "false peak" - a node with high degree but low connectivity among other high degree nodes.
               Example: $v$ connected to $u_1 \dots u_{100}$ (Star). $u_i$ connected to $w_i$ (Leaves).
               $deg(v)=100, deg(u_i)=2$.
               Sorted: $v, u_1 \dots$.
               Pick $v$. $C=\{v\}$. Candidates = $\{u_1 \dots u_{100}\}$.
               Next $u_1$. $C=\{v, u_1\}$. Candidates = $\{u_1 \dots u_{100}\} \cap \{v, w_1\}$.
               Intersection removes $u_2 \dots u_{100}$ (not neighbors of $u_1$).
               Leaves $\{v\}$ (if present in intersection? No, candidates filtered from prev step).
               Wait, $v$ is in neighbors($u_1$). So $v$ stays? No, $v$ was removed as "used"?
               Wait, my logic "Process in order" adds to C if in CandidateSet.
               $v$ processed first. Added.
               $u_1$ processed. In CandidateSet? Yes. Added.
               $u_2$ processed. In CandidateSet?
               $CandidateSet$ was $\{u_1 \dots u_{100}\}$.
               Intersection with $N(u_1) = \{v, w_1\}$.
               So $CandidateSet$ becomes $\{v, w_1\} \cap \{u_1 \dots\} = \{v\}$ (if $v$ in $u$'s set? No).
               Wait, the set of nodes remaining to be considered must be subset of $CandidateSet$.
               If we update $CandidateSet$ to $N(u) \cap CurrentSet$, we lose nodes that were not neighbors.
               If $v$ is already in $C$, do we need to keep it in $CandidateSet$?
               No, because $v$ is already selected. We just need to see who connects to $v$ and future nodes.
               The $CandidateSet$ should strictly contain nodes NOT yet in $C$ but compatible with $C$.
               When $v$ is processed, $C=\{v\}$. $CandidateSet = V \setminus \{v\}$. (Actually initialized as $V$).
               After $v$, $CandidateSet = Candidates \cap N(v)$.
               $u_1$ in $Candidates$. In $N(v)$? Yes.
               So $u_1$ selected. $C=\{v, u_1\}$.
               New $Candidates = Candidates \cap N(u_1)$.
               $Candidates$ originally had $u_2 \dots$.
               $N(u_1) = \{v, w_1\}$.
               Intersection with $\{u_2 \dots u_{100}, w_1\}$.
               None of $u_i$ (i>1) are in $N(u_1)$ (except $v$).
               So $u_2 \dots$ are dropped.
               $w_1$ is kept.
               Next in order: $u_2$. Not in $Candidates$. Skip.
               $w_1$. In $Candidates$? Yes.
               $w_1$ selected? $C=\{v, u_1, w_1\}$.
               But $\{v, u_1, w_1\}$ is NOT a clique (no edge $u_1-w_1$).
               Ah! $w_1$ is neighbor of $u_1$, not $v$.
               Wait, edge $v-w_1$? No.
               My example: Star center $v$, leaves $u_i$. $u_i$ connected to $w_i$.
               Edges: $(v, u_i)$ for all $i$. $(u_i, w_i)$ for all $i$.
               Degree($v$)=100. Degree($u_i$)=2. Degree($w_i$)=1.
               Sorted: $v, u_1, w_1, u_2, \dots$. (Tie break arbitrary).
               1. $v$. $C=\{v\}$. Cand=$\{u_i\}$.
               2. $u_1$. In Cand? Yes. Connected to $v$? Yes.
                  $C=\{v, u_1\}$. Cand=$\{u_i\} \cap N(u_1) = \{w_1\}$ (Assuming $u_i$ for $i>1$ not neighbors of $u_1$).
               3. $w_1$. In Cand? Yes. Connected to $C$?
                  $N(w_1) = \{u_1\}$.
                  Does $w_1$ connect to $v$? No.
                  But we didn't check $C$! We rely on $Candidates$ maintenance.
                  Is $w_1$ in $Candidates$?
                  Step 2 updated $Candidates$ to $N(u_1) \cap \{u_i\}$.
                  $w_1$ is in $N(u_1)$. So $w_1$ is in $Candidates$.
                  So the algorithm thinks $w_1$ is compatible with $C$.
                  BUT $C=\{v, u_1\}$. $w_1$ is NOT compatible with $v$.
                  The flaw: $Candidates$ tracks compatibility with *selected nodes* implicitly?
                  In the algorithm:
                  `Candidates` tracks nodes compatible with ALL currently SELECTED nodes?
                  Let's trace:
                  Init $Candidates = V$.
                  $C = \emptyset$.
                  Process $u$:
                    If $u \in Candidates$:
                       $C.add(u)$.
                       $Candidates = Candidates \cap N(u)$.
                  
                  Trace Star example again:
                  $Candidates = \{v, u_1 \dots u_{100}, w_1 \dots w_{100}\}$.
                  1. $v$. In Cand? Yes.
                     $C=\{v\}$.
                     $Candidates = V \cap N(v) = \{u_1 \dots u_{100}\}$.
                  2. $u_1$. In Cand? Yes.
                     $C=\{v, u_1\}$.
                     $Candidates = \{u_1 \dots\} \cap N(u_1) = \{u_1 \dots\} \cap \{v, w_1\}$.
                     Note: Intersection is with $N(u_1)$. $u_1$ not in $N(u_1)$. $v$ IS in $N(u_1)$. $w_1$ IS in $N(u_1)$.
                     So $Candidates = \{v, w_1\}$.
                     Wait. $u_2 \dots u_{100}$ are removed.
                  3. $u_2$. In Cand? No (Removed).
                  4. $w_1$. In Cand? Yes.
                     $C=\{v, u_1, w_1\}$.
                     $Candidates = \{v, w_1\} \cap N(w_1) = \{v, w_1\} \cap \{u_1\}$.
                     Empty.
                  Final $C=\{v, u_1, w_1\}$.
                  Is this a clique?
                  Edges needed: $(v, u_1)$ YES.
                  $(v, w_1)$ NO.
                  $(u_1, w_1)$ YES.
                  Fail.
                  
                  Why did it fail?
                  Because $Candidates$ was updated by $N(u_1)$. It included $v$.
                  But $v$ was ALREADY processed.
                  The check "Is $u \in Candidates$?" is insufficient if we don't ensure $u$ connects to PREVIOUSLY PROCESSED AND SELECTED nodes.
                  Wait. If $u \in Candidates$, does it mean $u$ connects to ALL selected nodes?
                  Initially yes (connects to nothing).
                  After step 1 ($v$ selected), $Candidates$ becomes $N(v)$. So yes, any $u \in Candidates$ connects to $v$.
                  After step 2 ($u_1$ selected), $Candidates$ becomes $N(v) \cap N(u_1)$.
                  So any $u \in Candidates$ connects to $v$ AND $u_1$.
                  So why did $w_1$ end up in $Candidates$?
                  $N(v) \cap N(u_1) = \{u_1 \dots\} \cap \{v, w_1\}$.
                  Is $w_1 \in N(v)$? NO.
                  So $w_1$ should NOT be in $Candidates$!
                  In my trace, I said $Candidates = \{v, w_1\}$.
                  Why?
                  Step 1: $Candidates = \{u_1 \dots\}$. ($v$ removed from consideration or just ignored).
                  Actually, we should treat $Candidates$ as "nodes to consider adding".
                  So $Candidates$ should not contain $v$ anymore?
                  If $Candidates = \{u_1 \dots u_{100}\}$.
                  Step 2: $Candidates = Candidates \cap N(u_1)$.
                  $N(u_1) = \{v, w_1\}$.
                  Intersection: $\{u_1 \dots\} \cap \{v, w_1\}$.
                  Since $v$ is not in $\{u_1 \dots\}$, $v$ disappears.
                  $w_1$ appears?
                  Is $w_1 \in \{u_1 \dots u_{100}\}$? NO.
                  So $w_1$ disappears too!
                  Wait, $w_1$ was never in the initial $Candidates$ (restricted to neighbors of $v$).
                  So $Candidates$ becomes empty.
                  Wait, $u_1$ was removed?
                  Initial $Candidates = \{u_1 \dots\}$.
                  Select $u_1$.
                  Update $Candidates = \{u_1 \dots\} \cap N(u_1)$.
                  If $u_1 \notin N(u_1)$ (no self loop), then $u_1$ is removed.
                  Wait, $u_1$ was selected. It shouldn't be in Candidates.
                  So logic holds: Candidates shrinks.
                  In the example, $Candidates$ became empty.
                  So algorithm stops.
                  $C = \{v, u_1\}$.
                  Valid clique of size 2.
                  Wait, is there a clique of size 2? Yes.
                  Is there a larger one?
                  $v$ is connected to all $u$. $u_i$ are not connected to $u_j$.
                  Max clique is $\{v, u_i\}$. Size 2.
                  Wait, $u_i$ connected to $w_i$.
                  $\{u_i, w_i\}$ is size 2.
                  Wait, I thought I found a clique of size 3 earlier?
                  $\{v, u_1, w_1\}$. Edges $(v, u_1)$ yes. $(u_1, w_1)$ yes. $(v, w_1)$ NO.
                  So size 3 is invalid.
                  So my manual trace that led to failure was actually producing a valid clique (size 2) and my confusion came from thinking it produced invalid.
                  The algorithm produced $\{v, u_1\}$. Which is valid.
                  
                  What if the graph is Triangle + isolated edge?
                  1-2, 2-3, 3-1.
                  4-5.
                  Degs: 1,2,3 have 2. 4,5 have 1.
                  Order: 1, 2, 3, 4, 5.
                  1. 1. $C=\{1\}$. Cand=$\{2, 3\}$.
                  2. 2. $C=\{1, 2\}$. Cand=$\{3\}$.
                  3. 3. $C=\{1, 2, 3\}$. Cand=$\emptyset$.
                  4. 4. Not in Cand.
                  5. 5. Not in Cand.
                  Result $\{1, 2, 3\}$. Correct.
                  
                  Seems correct.
                  The algorithm:
                  Iterate $u$ in $P$.
                  If $u \in Candidates$:
                    Add $u$ to $C$.
                    $Candidates = Candidates \cap N(u)$.
                  This logic assumes $Candidates$ initially contains all nodes.
                  But if we update $Candidates$ by intersection, we must exclude nodes that are already selected or irrelevant.
                  Actually, the set $Candidates$ should be maintained as "Nodes available to extend $C$".
                  When we add $u$, we intersect with $N(u)$.
                  This works because $Candidates$ represents $\bigcap_{v \in C} N(v)$.
                  Any node $z$ picked from $Candidates$ satisfies $z \in \bigcap N(C)$.
                  So yes, this greedy construction works and is valid.
                  The "failure" in my trace was me assuming $Candidates$ contained everything.
                  With proper intersection, it works.
                  
                  So, the core algorithm is:
                  `sort_vertices(descending_degree)`
                  `run_greedy(vertices_sorted)`
                  
                  To improve:
                  Since picking the wrong "first" node can lead to a smaller clique (as seen in Star vs Triangle mix?), maybe we should try starting with a random subset of nodes?
                  No, sorting is deterministic.
                  We can simply run this on `shuffle(sorted_vertices)`.
                  Wait, if we shuffle, we lose the degree benefit.
                  Actually, for Max Clique, the standard heuristic is just degree sort.
                  But sometimes picking the vertex with highest degree (which is $v$ in Star) is bad?
                  In Star graph, $v$ has max degree. We pick it. Then we pick one leaf. Size 2.
                  If we picked a leaf first? $u_1$. $C=\{u_1\}$. Cand=$\{v, w_1\}$.
                  Next $v$. In Cand? Yes. $C=\{u_1, v\}$. Cand=$\{w_1\} \cap \{...\}$ -> $\emptyset$. Size 2.
                  Both give 2.
                  What about the "Triangle attached to Star"?
                  $v$ connected to $u_1, u_2$. $u_1$ connected to $1, 2$. $1-2$ connected.
                  Graph: $v-u_1$, $u_1-1$, $1-2$, $2-u_1$, $v-u_2$.
                  Edges: $(v, u_1), (v, u_2), (u_1, 1), (u_1, 2), (1, 2)$.
                  Clue: $\{1, 2, u_1\}$ is triangle.
                  $\{u_1, v, ?\}$ no.
                  Max clique is 3.
                  Degrees: $u_1$: 3 ($v, 1, 2$). $v$: 2. $1$: 2. $2$: 2. $u_2$: 1.
                  Sort: $u_1, v, 1, 2, u_2$.
                  1. $u_1$. $C=\{u_1\}$. Cand=$\{v, 1, 2\}$.
                  2. $v$. In Cand? Yes. $C=\{u_1, v\}$. Cand=$\{v, 1, 2\} \cap N(v) = \{v, 1, 2\} \cap \{u_1, u_2\} = \{u_1, u_2\}$.
                     Wait. Intersection of $\{v, 1, 2\}$ and $\{u_1, u_2\}$.
                     Does $v$ have $1$ or $2$? No.
                     So $\{1, 2\}$ removed.
                     $u_1$ removed (processed). $u_2$ not neighbor of $u_1$?
                     Wait, $u_1$ connected to $v$. $u_2$ connected to $v$. $u_1$ not connected to $u_2$.
                     So $u_2$ removed.
                     $Candidates$ becomes empty.
                  3. 1. Not in Cand.
                  ...
                  Result $\{u_1, v\}$. Size 2.
                  BUT max clique is $\{u_1, 1, 2\}$.
                  So degree sort FAILED.
                  It picked $v$ early (degree 2 vs 3? No $u_1$ degree 3).
                  Wait, sort: $u_1(3), v(2), 1(2), 2(2), u_2(1)$.
                  Why did it fail?
                  1. Pick $u_1$. $C=\{u_1\}$. Cand=$\{v, 1, 2\}$.
                  2. Pick $v$. (Next in sort).
                  Is $v$ compatible with $u_1$? Yes.
                  So add $v$.
                  Now $C=\{u_1, v\}$.
                  Remaining candidates: intersection of Cand with $N(v)$.
                  $Cand = \{v, 1, 2\}$. (Note: $v$ was in it).
                  $N(v) = \{u_1, u_2\}$.
                  Intersection $\{v, 1, 2\} \cap \{u_1, u_2\} = \emptyset$ (ignoring $u_1$).
                  Wait. $1$ is not neighbor of $v$. $2$ is not neighbor of $v$.
                  So they are dropped.
                  Result $\{u_1, v\}$.
                  
                  The clique $\{u_1, 1, 2\}$ requires $u_1$ to be present.
                  $v$ should have been avoided? Or processed last?
                  If we processed $1$ then $2$ then $u_1$.
                  Sort was: $u_1, v, 1, 2...$
                  Wait, if we process $1, 2$ first?
                  Say we shuffle such that $1$ is first.
                  1. 1. $C=\{1\}$. Cand=$\{u_1, 2\}$.
                  2. 2. $C=\{1, 2\}$. Cand=$\{u_1\}$.
                  3. $u_1$. $C=\{1, 2, u_1\}$. Cand=$\emptyset$.
                  Found size 3!
                  
                  So, randomizing the order is EXTREMELY important.
                  Degree sort alone failed on this case.
                  Why did it fail? Because $v$ "blocked" $1$ and $2$.
                  Even though $1$ and $2$ have lower degree than $u_1$ (same as $v$).
                  Basically $v$ shares fewer connections with the set $\{1, 2\}$ than $u_1$ does.
                  Actually $u_1$ connects to BOTH 1 and 2. $v$ connects to NONE.
                  Wait, $v$ connected to $u_1$ and $u_2$. Not 1, 2.
                  The triangle is $\{1, 2, u_1\}$.
                  We picked $u_1$ (good).
                  But then we encountered $v$. $v$ is neighbor of $u_1$.
                  Since $v$ appeared early, we accepted it, and discarded 1, 2.
                  But 1, 2 were compatible with $u_1$!
                  Why were they discarded?
                  Because we intersected Candidates with $N(v)$.
                  $N(v)$ did NOT contain 1 or 2.
                  So 1, 2 were valid wrt $u_1$, but rejected by $v$.
                  Since we greedily accepted $v$, we cut off the other clique.
                  
                  Conclusion: The simple Greedy with Fixed Order is not enough.
                  We MUST run it with different orders.
                  However, running $2^{1000}$ is impossible.
                  But we can try Randomized Permutations.
                  How many?
                  If we run 50-100 iterations of random permutations.
                  With $N=1000$, we hope to hit the "correct" order for the target clique.
                  The "correct" order places all clique members relative to non-clique blockers intelligently.
                  Specifically, we want clique members to appear "before" blockers.
                  Blockers are nodes that connect to clique members but don't connect to other clique members (blocking the extension).
                  Or nodes that are "better" than clique members (higher degree) but prevent the clique.
                  This is a tough heuristic problem.
                  However, a very common improvement is to run the algorithm on `permuted(degree_sorted_vertices)`.
                  Actually, just random shuffling the `Candidates` selection or something?
                  
                  Let's go back to BK.
                  BK finds maximal cliques. If we find a clique of size 3, we keep it.
                  If we continue, we might find size 3 or 4.
                  BK will definitely find $\{1, 2, u_1\}$ because it explores branches.
                  Branching on $u_1$ (added to $C$). Then $P$ restricted to neighbors.
                  Neighbors of $u_1$: $\{v, 1, 2\}$.
                  Then we branch on $v, 1, 2$.
                  Option 1: Pick $v$. $C=\{u_1, v\}$. $P$ restricted to $\{u_1, u_2\}$.
                  Option 2: Pick $1$. $C=\{u_1, 1\}$. $P$ restricted to $\{2\}$ (neighbor of 1 in P).
                     Next pick 2. $C=\{u_1, 1, 2\}$. Max clique found.
                  So BK finds both.
                  Is BK fast enough?
                  In this case, yes.
                  For $N=1000$?
                  BK with pivoting is very fast for most instances.
                  The issue is only when there are *exponentially many* maximal cliques.
                  In that case, finding the maximum is hard.
                  However, since it's a "Challenge", likely the graph isn't pathological.
                  
                  I will provide the BK solution with pivoting and bitset optimization. It is the most robust for "Correctness".
                  It guarantees valid clique. It explores enough ground to likely beat Greedy.
                  
                  Implementation Plan:
                  - Class `Solver`.
                  - `adj` bitsets.
                  - `vector<int> best_clique`.
                  - `int max_k`.
                  - Recursive function `solve(P, X)`.
                  - Optimization: `P` is bitset, `X` is bitset.
                  - Pivot selection: Scan `P | X`.
                  - Iteration: Scan `P` (using `_Find_first` loop or linear scan).
                  - Memoization? Not really feasible.
                  - Pruning: `if (P.count() + C.size() <= max_k) return;`
                  
                  Wait, `P.count()` takes $O(N/64)$. Good.
                  
                  One optimization: Order vertices such that `adj` bitsets are populated efficiently?
                  Reading input is $O(M)$.
                  
                  Memory limit 512MB is plenty for bitsets.
                  Time 2.0s.
                  
                  The recursive BK can be deep. Stack size might be an issue? Default stack is usually 8MB.
                  Depth 1000 frame is small. Safe.
                  
                  Double check: `X` set.
                  We need to track vertices in `X` to avoid reporting same clique multiple times.
                  When `P` is empty, we have a maximal clique.
                  If `X` is also empty, it's a NEW maximal clique (standard BK).
                  Wait, standard BK iterates over `P \ N(pivot)`.
                  If `P` empty and `X` empty, return.
                  But we need to track the clique.
                  Pass `current_clique` by value or maintain global `C`?
                  Maintain `global_max_C`.
                  When maximal found (P==0 && X==0), compare with `max_k`.
                  Wait, `P==0 && X==0` condition is for maximal cliques found.
                  But in BK, we only enter the base case when we cannot extend.
                  If we have visited the node in `X` (excluded previously), we shouldn't report it again.
                  Wait, the standard BK algorithm:
                  If `P` is empty:
                    If `X` is empty (or specific condition depending on variant), update max.
                    Actually, standard BK enumerates maximal cliques.
                    We just want the largest.
                    
                    Wait, if `P` is not empty, we recurse.
                    If `P` is empty, it means we found a maximal clique.
                    However, if `X` is not empty, we might have already covered it?
                    Actually, in the standard "exact" version, we just want to know the largest.
                    If `P` is empty, the current set $C$ is a maximal clique.
                    Is it always a *new* one?
                    We must ensure we don't double count? No, we just want max size.
                    Even if we find same size multiple times, it's fine.
                    Wait, if `P` is empty, $C$ is maximal.
                    We update max size.
                    Do we need to check `X`?
                    The standard algorithm includes `X` to avoid redundant work.
                    Here we can ignore `X` for correctness (just want size), but for performance (pruning), we use it?
                    Actually, if `X` is empty, we found a maximal clique that wasn't covered by pivoting?
                    Actually, the pivot logic ensures we cover everything.
                    The condition `P.isEmpty()` is sufficient to say "Current C is maximal".
                    Wait. Is it?
                    If `P` is empty, no vertices remain to extend $C$. So yes, maximal.
                    So we just record $|C|$.
                    But wait, standard BK uses `X` to prevent visiting same maximal clique multiple times.
                    Without tracking `X` properly (or passing it), we might re-examine the same maximal clique?
                    With pivoting, we iterate $v \in P \setminus N(p)$.
                    If we don't use `X`, do we revisit?
                    Actually, if we don't pass `X` and use a flag array `visited`, we might visit again?
                    No, `P` and `X` partition vertices.
                    We need `X` in the state.
                    But maybe we can just use a `vector<bool> visited` globally?
                    No, standard BK is preferred.
                    Let's implement with `X`.
                    State: `(P, X)`.
                    Base case: `P` is empty. Update max.
                    Recursive step: Pivot $u \in P \cup X$.
                    Iterate $v \in P \setminus N(u)$.
                    
                    Wait, `P \setminus N(u)` logic:
                    $P \cap \neg N(u)$.
                    So we iterate over $v$ such that $v \in P$ AND $v \notin N(u)$.
                    Correct.
                    
                    One detail: `X` is passed to recursive calls.
                    We need to ensure we don't re-enter the same state.
                    However, with `P` reducing (subset), states are unique.
                    So we are safe.
                    
                    One more thing:
                    If $N=1000$, passing bitsets by value is fine.
                    Bitset copy is fast.
                    
                    What if $K^*$ is small?
                    Then BK is fast.
                    What if $K^*$ is large?
                    Then BK might be slow.
                    But BK is good for large cliques.
                    
                    Wait, the greedy heuristic ran out of luck on "Triangle + Star". BK worked.
                    So BK is a safer bet.
                    
                    Final check on Constraints.
                    Time: 2.0s.
                    With $N=1000$, BK might TLE on specific graphs (e.g. many maximal cliques).
                    Is there a way to make it faster?
                    We can just run BK.
                    If TLE is a concern, we can add a cutoff (limit recursion depth).
                    But then we might miss the optimal.
                    However, we want a heuristic solution.
                    I'll stick with BK but optimize heavily.
                    1. Use fast IO.
                    2. Use `__builtin_popcountll` or `bitset::count`.
                    3. Pivot optimization: If $P \setminus N(u)$ is empty, we can stop? No, just skip loop.
                    4. `best_so_far` pruning.
                   
                    Wait, I can implement a "Hybrid":
                    Run BK to find a large clique.
                    Also run the Greedy heuristic.
                    Take the max of both.
                    This balances robustness.
                    Greedy is super fast. BK is powerful.
                    Given time limit, running Greedy (with 50 random permutations) takes negligible time (~0.01s).
                    Running BK (on same graph) takes unknown time.
                    Maybe just BK is enough.
                    I will implement BK.
                    But wait, `P \setminus N(u)` involves iterating.
                    With $N=1000$, iterating $v \in P$ takes $O(N)$.
                    Total complexity is proportional to number of nodes in recursion tree.
                    If recursion tree is large, we die.
                    
                    Actually, the prompt says "Heuristic optimization problem. You are NOT required to find the theoretically optimal solution."
                    This implies BK (which tries to find optimal) might be overkill or risky.
                    A robust heuristic is simply:
                    Sort by degree.
                    Greedy Expansion with Pivoting (Pick neighbor with max degree).
                    Wait, that's exactly the Greedy I analyzed.
                    To make it stronger:
                    1. Sort by degree.
                    2. Maintain `Candidates` set.
                    3. Loop $v \in Candidates$. Pick $v$ with highest degree (from original graph).
                    4. Add to clique. Update Candidates.
                    
                    To fix the "Triangle+Star" failure:
                    The failure happened because we picked $u_1$ (part of clique), then $v$ (blocker), killing $1, 2$.
                    Notice $u_1$ was picked before $1, 2$.
                    But $v$ was picked between them.
                    If we can reorder vertices such that blockers are pushed back.
                    How to detect blockers? Hard.
                    But random shuffling is the easiest fix.
                    
                    Revised Plan:
                    Combine Greedy with Random Shuffling.
                    Run Greedy 50 times with random shuffles of vertices.
                    This is very unlikely to TLE.
                    It's unlikely to find the MAX clique, but likely to find a VERY good clique.
                    It guarantees validity.
                    
                    Wait, is it possible to TLE with 50 x $O(N^2)$?
                    $50 \times 1000^2 \approx 5 \cdot 10^7$.
                    Operations inside Greedy are bitwise (fast).
                    Intersection: 1000 bits / 64 $\approx 16$ ints.
                    Cost per vertex added: $O(16)$.
                    Cost per vertex considered: Check bit? $O(1)$ or $O(16)$.
                    Actually, checking if $u \in Candidates$: $O(1)$.
                    Updating Candidates: $O(N/64)$.
                    Total per Greedy run: $O(N \cdot (N/K + N/W))$. Roughly $O(N^2)$.
                    Wait, number of iterations is $N$.
                    In each iteration, we update Candidates.
                    Intersection takes $N/64$.
                    So $O(N^2/64)$.
                    $50 \times 1000^2 / 64 \approx 50 \times 15000 \approx 750,000$ ops.
                    This is TRIVIAL.
                    Wait, I calculated $10^7$ before.
                    $N^2/64$ is roughly $1.5 \cdot 10^4$.
                    So 50 runs is $7.5 \cdot 10^5$.
                    Wait, is it?
                    Let's re-evaluate.
                    $N=1000$.
                    `bitset` op is $O(N/64)$.
                    We do this for every vertex in permutation.
                    So $N$ intersections.
                    Total ops: $N \times (N/64) \approx 1000 \times 16 = 16000$.
                    50 runs -> $800,000$.
                    This is instant.
                    Why did I think $N^2$? Because of naive intersection.
                    With bitset, it's very fast.
                    
                    So, 50-100 runs of Greedy with Random Shuffle is perfectly viable and very strong.
                    It explores diverse subspaces.
                    
                    Wait, I need to store the result.
                    Output format: $N$ lines.
                    So we need the final clique mask.
                    We will update a `mask` vector/array for each run.
                    
                    Wait, I should confirm the complexity of updating `Candidates`.
                    `Candidates` is a `bitset`.
                    `Candidates &= adj[u]`.
                    This takes $O(N/64)$.
                    We also need to iterate the permutation. $O(N)$.
                    Wait, finding if $u \in Candidates$?
                    `Candidates.test(u)` is $O(1)$.
                    So loop body is $O(1)$ (excluding intersection).
                    Intersection happens when we ADD $u$.
                    But we don't know if we add $u$ until we check.
                    Actually, we always add?
                    Wait. "If $u$ is compatible with current $C$".
                    My previous logic: `Candidates` maintains compatibility.
                    So if `Candidates[u]` is 1, then $u$ is compatible with ALL currently selected vertices in $C$.
                    Wait. If $u \in Candidates$, is it compatible with *all* previous selected?
                    Yes, because `Candidates` was updated at each selection step by intersecting with neighbors.
                    So `Candidates` tracks intersection of neighbors of all selected vertices.
                    Thus, any bit set in `Candidates` corresponds to a node connected to ALL in $C$.
                    Wait. The logic:
                    1. Init `Candidates = All`.
                    2. Loop $u$ in Perm:
                       If `Candidates[u]`:
                          Add $u$ to $C$.
                          `Candidates &= adj[u]`.
                    
                    Does `Candidates[u]` imply $u$ connects to ALL in $C$?
                    Start: $C=\emptyset$. `Candidates` = All.
                    Pick $u_1$. `Candidates`[u1]=1. Add $u_1$. `Candidates` $\leftarrow$ `Candidates` & `adj[u1]`.
                    Pick $u_2$. `Candidates`[u2]=1.
                    This implies $u_2 \in C$'s neighbors?
                    Yes, because `Candidates` was intersected with `adj[u1]`.
                    So $u_2$ is neighbor of $u_1$.
                    Add $u_2$. `Candidates` $\leftarrow$ `Candidates` & `adj[u2]`.
                    Any bit now set in `Candidates` is neighbor of $u_1$ AND $u_2$.
                    Correct.
                    So we just add every compatible node.
                    This constructs a clique.
                    And it is valid.
                    
                    So the algorithm is:
                    For 50 iterations:
                      Generate random permutation of vertices.
                      `cand` = All bits set.
                      `res` = Empty.
                      For each `u` in Permutation:
                        if `cand[u]`:
                           `res`.push(`u`).
                           `cand` &= `adj[u]`.
                      Update global best `res`.
                    
                    This is extremely efficient.
                    Why would this not find the max clique?
                    Because the permutation order forces the selection.
                    With random permutations, we are likely to find a large clique.
                    Especially if we shuffle the degree-sorted array.
                    Wait, random shuffle destroys degree correlation.
                    Better: Shuffle only the top $N/10$ vertices?
                    Or just shuffle.
                    Actually, for $N=1000$, full shuffle is fast enough.
                    
                    Let's code this. It's clean, fast, and likely to win high score (close to 100) on random/typical inputs.
                    On adversarial inputs, it might be lower, but BK would be needed.
                    However, BK is complex to implement bug-free.
                    The Greedy is bug-free logic.
                    And the constraints $N=1000$ is borderline for exact BK but perfect for Greedy.
                    Wait, $N=1000$ Greedy is super fast.
                    
                    I will add one optimization: Sort by degree initially. Then apply "Local Randomization" (swap elements in permutation).
                    Or just random permutation.
                    I'll stick to Random Permutation to be safe against structural traps.
                    Wait, if the graph is disconnected, random is good.
                    If graph is random Erdos-Renyi, random is good.
                    
                    Wait, the example input.
                    5 vertices.
                    Shuffle might pick 1, 4, 2, 3, 5.
                    1. 1. `cand` = {2,3,4,5} (neighbors). `cand` &= {2,3,4} -> {2,3}.
                    2. 4. Not in `cand`.
                    3. 2. In `cand`. `cand` &= {1,3,5} (neighbors of 2) -> {3}.
                    4. 3. In `cand`. `cand` &= {1,2} -> {}.
                    5. 5. Not in `cand`.
                    Result {1, 2, 3}. Size 3. Matches Example.
                    
                    If order 1, 4, 2, 3, 5 gave {1, 2, 3}.
                    If order 5, 2, 1, 3, 4.
                    1. 5. `cand`={2,3} (neighbors of 5). `cand` &= {2,3} -> {2,3}.
                    2. 2. In `cand`. `cand` &= {1,3,5} -> {3}.
                    3. 1. Not in `cand`.
                    4. 3. In `cand`. `cand` &= {1,2} -> {}.
                    Result {5, 2, 3}. Size 3.
                    Both valid cliques of size 3.
                    
                    Looks solid.
                    
                    One edge case: $N$ up to 1000.
                    `std::bitset<1001>` is static size.
                    Input indices are 1-based. Map to 0-based.
                    
                    Wait, `cand &= adj[u]`.
                    `adj` needs to be 0-based.
                    
                    Wait, memory limit 512MB.
                    We need to store `adj` array.
                    1000 bitsets. $1000 \times 1000$ bits $\approx 1$ MB. Negligible.
                    
                    Code structure:
                    - Includes.
                    - Main function.
                    - Fast IO.
                    - Read N, M.
                    - Read edges, fill adj.
                    - Variable `max_n` = 0. `best_permutation_indices`.
                    - Loop 100 times:
                       - Random shuffle `1..N`.
                       - Run Greedy.
                       - Update max.
                    - Print `best_permutation_indices`? No, print the clique.
                    
                    Wait, we need to output the vertices in the final clique.
                    Store `best_clique` as a bitmask (vector<bool> or bitset).
                    
                    Final check:
                    Does this satisfy the requirements?
                    - Undirected graph.
                    - Output 1/0.
                    - Valid clique.
                    - Maximize size.
                    - Time limit 2s.
                    
                    Complexity: $100 \times N \times (N/64)$.
                    $100 \times 1000 \times 16 \approx 1.6 \times 10^6$ ops.
                    Plenty of time.
                    
                    What if the score is lower than greedy?
                    The problem says "Heuristic Optimization".
                    If we find a suboptimal clique, score is still positive.
                    Is it possible to optimize further?
                    We can try to locally improve?
                    Swap elements in clique?
                    If we have a clique $C$, and we find a node $u \notin C$.
                    If we remove $x \in C$ and add $u$, is it still a clique?
                    If $u$ connects to all $C \setminus \{x\}$, yes.
                    If we find such $x$, replace.
                    If $|C|$ doesn't change, we might improve connectivity (more candidates).
                    But since we need to maximize size, if $|C| < |C'|$ and valid, update.
                    We can add this local search step.
                    Given the low cost, it's worth adding.
                    Local Search Loop:
                    `improved = true`
                    `while improved`:
                       `improved = false`
                       `best_replace = -1`
                       For `u` in `NonClique`:
                          Count neighbors in `C`.
                          If count == `|C|`:
                             // Adding u keeps C valid if we don't remove anyone.
                             // So just `Add u`. (This is actually just extending).
                             // Wait, my greedy stopped when no more compatible nodes were found?
                             // Yes, because `cand` became empty.
                             // But if we shuffle, `cand` becomes empty because we exhausted order.
                             // But maybe there are nodes compatible that came earlier but were skipped?
                             // NO. `cand` tracks compatibility with CURRENT C.
                             // If we skipped a node, it means it was NOT compatible with previous C.
                             // So we cannot add it later.
                             // So the clique is MAXIMAL for that specific permutation order.
                             // Wait, does "skipped" mean "never compatible"?
                             // In my logic: "If `cand[u]`".
                             // `cand` is updated. If `u` was in `cand` (compatible) but we didn't pick it,
                             // that implies `u` came AFTER all compatible nodes?
                             // No. We iterate *in order*.
                             // If we encounter `u` and it's compatible, we pick it.
                             // So we greedily pick everything compatible.
                             // This implies for the chosen permutation, we pick the "Longest Chain of Compatibility"?
                             // No, it picks *all* vertices compatible with *all* previous picks.
                             // So it forms a clique.
                             // Since we iterate all nodes, any node not picked was incompatible with *some* prefix of the clique?
                             // Yes.
                             // So we can't simply "Add".
                             // Local swap is valid only if we find a replacement.
                    This seems like a valid optimization.
                    However, finding a swap might take time.
                    Let's skip to ensure we stay well within limits and keep code simple.
                    100 restarts is enough to beat most test cases.
                    
                    Wait, one issue:
                    If I use `std::random_shuffle` (deprecated) or `std::shuffle`.
                    Use `<algorithm>`.
                    
```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <bitset>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    vector<bitset<1001>> adj(N + 1);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].set(v);
        adj[v].set(u);
    }

    // Greedy construction with multiple random permutations
    vector<int> best_clique;
    int max_k = 0;

    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    
    // We can optimize by sorting vertices by degree initially, 
    // but shuffling destroys that.
    // Let's shuffle the vertices.
    
    vector<int> p(N);
    iota(p.begin(), p.end(), 1); // 1 to N
    
    // Run multiple times
    // To ensure coverage, we do several passes.
    // A mix of sorted (deg) and random is good, but random is safer for exploration.
    // Let's just do purely random.
    // Or perhaps "Shuffle Top K".
    // Actually, sorting by degree first might help identify dense cores.
    // Let's just rely on random permutations.
    
    // To save time, precalculate degrees
    vector<int> degrees(N + 1, 0);
    for (int i = 1; i <= N; ++i) {
        degrees[i] = adj[i].count();
    }
    
    // We will shuffle the vector 'p' in each iteration.
    for (int iter = 0; iter < 200; ++iter) {
        // Randomize p
        shuffle(p.begin(), p.end(), rng);
        
        vector<bitset<1001>> candidates(N + 1);
        candidates.assign(N + 1, bitset<1001>(true)); // All bits set
        
        // But assigning bitset<1001>(true) sets all bits.
        // However, bitset<N> constructor default is all zero. 
        // We want all bits 1 except maybe unused index 0? 
        // Indices are 1..N. So we need 0-th bit? Doesn't matter.
        // Actually, initializing 1000 bits is fast?
        // `bitset<1001> temp; temp.set();` sets all.
        
        // Optimized candidate initialization:
        // Just use a separate check or initialize once.
        // Re-initializing inside loop might be slow if done poorly.
        // Better: `candidates` variable.
        
        bitset<1001> cand;
        cand.set(); 
        
        vector<int> current_clique;
        
        // To optimize, we can check degrees?
        // Just iterate p.
        
        // Pre-checking if vertex is in candidates: `cand[u]`.
        // But `cand` update is bitset op.
        
        // Resetting cands for next iteration
        cand.reset();
        cand.set();
        
        // We can maintain a list of active candidates?
        // No, bitset is fastest.
        
        bool found = false; // optimization? No.
        
        // Optimization: If current clique size already >= best_clique.size(), maybe skip?
        // No, we might find better in this run.
        
        // Let's implement the loop
        // To avoid reallocation, declare vectors outside?
        // current_clique is temporary.
        
        // Actually, 200 iterations with N=1000 is ~ 2*10^5 ops * bitset overhead.
        // Bitset assign/copy takes time.
        // But wait, we only update `cand`. `cand` is local.
        // Creating it 200 times is fine.
        
        // However, we want to keep the best.
        
        // Wait, `cand` reset/set is O(N/64).
        
        // Logic:
        // for u in p:
        //   if cand[u]:
        //      current_clique.push(u)
        //      cand &= adj[u]
        
        // After loop, check size.
        
        // But wait, the `best_clique` stores indices.
        // Let's track max size.
        
        // To make it faster:
        // 1. Sort p by degree initially. Then shuffle?
        // 2. Use `std::vector` reserve.
        
        // Let's optimize the shuffle.
        // Shuffle is O(N).
        
        // Let's write clean code.
        // The time limit is generous.
    }
    
    // Wait, the loop above is pseudocode.
    // I need to implement the actual run.
}
```
Correction:
The logic `candidates[N+1]` is unnecessary if we use a single `bitset<1001>` named `cand`.
We need to iterate `p`.
Inside loop: `if (cand[u]) { ... cand &= adj[u]; }`
`cand` must be initialized to all 1s (1..N bits).
`cand.set()` does that.

Wait, `cand.set()` sets all bits including 0.
It doesn't matter, we check `cand[u]` where u is 1..N.
Indices are 1-based.

One small optimization:
If `cand.none()` breaks early.
Also, we can maintain `current_clique` vector.

Double check the constraints: N=1000.
Vector allocation is fast.
Code looks good.

Wait, `std::bitset<1001>` initialization inside loop?
Yes, `bitset<1001>()` constructor initializes to 0.
We need to set all bits.
`cand |= ~(bitset<1001>())` ? No.
Just `cand.set()` is better.

Wait, `cand` should track nodes *available to be added*.
Initially all nodes are candidates.
When we pick `u`, we filter `cand` to neighbors of `u`.
Correct.

The loop runs 200 times.
Total operations roughly $200 \times 1000 \times 16 \approx 3.2 \times 10^6$ bitwise ops.
Plus shuffle $200 \times 1000$.
Total very low.

Final plan:
1. Include headers.
2. `main` reads input.
3. Build adj.
4. 200 iterations:
    Shuffle `p` (1..N).
    `bitset<1001> cand`. `cand.set()`.
    `vector<int> current_clique`.
    Loop `u` in `p`:
      If `cand[u]`:
         `current_clique.push_back(u)`
         `cand &= adj[u]`
      If `current_clique.size() > max_k`:
         Update `max_k`.
         Copy `current_clique` to `final_ans`.
5. Print `final_ans`.
   (Output 1 for `i` in `final_ans`, 0 else).

Wait, `current_clique` copy can be costly if frequent.
Better: track indices of `best_clique` as `vector<int> best_clique_indices`.
Only update if `current_clique.size() > best_clique_indices.size()`.
Then copy `current_clique` to `best`.

One edge case: if N=0? Constraints say N>=2.
Multiple edges? Handled by adjacency matrix (idempotent).
Self loops? Constraints u!=v.

Looks ready.

Wait, I should check if I need to import `<chrono>` or `<ctime>` for random seed.
Using `mt19937` is good.

One slight optimization:
The shuffle can be biased towards degree sorted order.
Sort `p` by degree.
Then random swaps.
I'll stick to full shuffle for simplicity and robustness.
The probability of hitting the optimal structure in a random permutation is high enough for heuristic problems.

Wait, `std::iota` fills 1..N.
`shuffle` modifies it.

Let's double check the score criteria.
"Invalid solutions receive score = 0."
My solution guarantees validity.
So score > 0.
"Higher score is better."
Maximizing $K$ maximizes score (assuming $K^*$ is fixed).
We approximate $K^*$.
Usually $K^*$ is around $\sqrt{N}$ to $N$ depending on graph.
Greedy heuristic finds a clique of size roughly logarithmic or constant fraction of max clique.
With random restarts, we get close to max clique size on average cases.

Wait, there is a possibility that $K^*$ is very small, e.g., 2.
Then any valid clique of size $\ge 2$ is good.
Greedy finds cliques. If graph is disconnected edges, size 1 or 2.
Wait, if graph has no triangles, max clique is 2.
Greedy will find size 2. Score 100.
If graph has many cliques of size 5, greedy might find 5.
Seems sufficient.

One check: The problem says "multiple edges between the same pair ... imply same constraint".
Handled.

Output format:
N lines.
x_i in {0, 1}.

Example:
Output:
1
1
1
0
0
My code:
Reads `best_clique` indices.
For i from 1 to N: if i in best -> cout << 1\n else 0\n.

Wait, `best_clique` is `vector<int>`.
Checking membership is $O(1)$ with `unordered_set` or boolean array.
Boolean array `bool in_best_clique[1001]`.

```cpp
// Final Check
// Constraints: N <= 1000.
// Time 2.0s.
// My algo complexity: 200 * N * (N/64). ~3M ops.
// Space: Adj + Bitsets. Minimal.
```
Ready to write.

One minor thing:
`bitset<1001>` default constructor sets to 0.
`cand.set()` sets all bits to 1.
This works.

Wait, `adj` vector size `N+1`.
Loop `i` from 1 to N.
Input indices 1 to N.
Code aligns.

One logic check: `cand` tracks vertices compatible with *already selected* vertices.
When we pick `u`, we check `if (cand[u])`.
Since `cand` was updated at previous steps, `cand[u]` being 1 means `u` is neighbor of *all* previous selected vertices.
Therefore, if we add `u` to `current_clique`, the new clique is valid (since `u` connects to all previous, and previous connect to each other by induction).
Wait, we need to ensure `u` connects to all previous.
Yes, `cand` intersection property ensures this.
Base case: `current_clique` empty. `cand` = All.
Any `u` is compatible with empty set.
Step: `current_clique` has $\{v_1, \dots, v_k\}$.
`cand` = $N(v_1) \cap \dots \cap N(v_k)$.
If `cand[u]` is 1, `u` connects to all $v_i$.
So $\{v_1, \dots, v_k, u\}$ is a clique.
Logic is sound.

Wait, what if `cand` is large?
It's fine.
What if we skip `u`?
If `cand[u]` is 0, `u` is not connected to at least one previous selected vertex.
So we cannot add `u`.
We MUST skip.
So this greedy strategy constructs a valid clique.

Correctness confirmed.

Wait, could `cand[u]` be true but `u` was already processed and skipped?
No. `cand` tracks availability.
Once we pick `u`, we update `cand` to neighbors of `u`.
Does `cand` remove `u`?
`adj[u]` has bit `u` as 0 (no self loops).
So `cand` will clear bit `u`.
So `u` is removed from `cand` in next steps.
Wait, this is important!
If `cand` doesn't remove `u`, and we have `u` appearing again later in permutation?
But permutation has unique vertices.
So `u` is never seen again.
And `cand[u]` becoming 0 doesn't affect anything else except removing `u` from candidates (if we ever saw `u` again).
Wait, `u` is not seen again.
So it's fine.
The crucial part is that `cand[u]` must be 0 to indicate incompatibility.
Wait.
Suppose `cand` = $\{v_1\}$. `u` is picked. `adj[u]` contains `v_1`.
New `cand` = $\{v_1\} \cap \{v_1, \dots\}$.
Is `v_1` in `cand`?
Wait, `cand` should contain nodes *to be considered* in future steps.
Should `cand` contain nodes already in `current_clique`?
No.
My code `cand &= adj[u]`.
If `adj[u]` does not contain `u` (no self loop), then `u` will be 0 in `cand`.
So `u` is removed.
Future vertices must be neighbors of `u`.
So this works.
Wait, what about `v_1`?
`v_1` was already picked.
Should `cand` contain `v_1`?
It might, if `v_1` connects to `u`.
But `v_1` is already processed.
Will it be processed again?
No, permutation has unique indices.
So `cand` content for processed nodes doesn't matter.
Only unprocessed nodes matter.
So removing `u` is fine.
Wait, if `cand` still contains `v_1`, does it hurt?
Only if `v_1` appears later in `p`. It doesn't.
So logic holds.

One detail: `adj[u]` bitset does not have `u` set.
Correct.
So `cand` will drop `u` immediately.

Final Check:
Is it possible that `cand` becomes empty but we haven't finished permutation?
Yes, `if (cand.none()) break;` optimization.
Also helpful.

I'll add `break` optimization.

One more thing: `adj` is `bitset<1001>`.
We need to read carefully.
Input: N M.
Then M lines u v.
Constraints 2<=N<=1000.
Memory 512MB.

```cpp
/*
Problem: Maximum Clique Challenge
Solution: Greedy Maximal Clique Construction with Random Restarts
Approach:
1. Parse the graph into an adjacency matrix represented by std::bitset for fast intersection.
2. Perform a greedy search:
   - Maintain a set of 'candidate' vertices compatible with the current clique.
   - Iterate through a permutation of vertices.
   - If a vertex is compatible (in candidates), add it to the clique and filter candidates.
   - Repeat this for multiple random permutations to increase chance of finding a large clique.
3. Track the largest valid clique found across all permutations.
4. Output the result.
*/
```
I will incorporate comments briefly.

Time Complexity: $O(T \cdot N^2 / W)$ where $W$ is word size (64). With $T=200$, $N=1000$, approx $3 \cdot 10^6$ ops. Well within 2s.

Space Complexity: $O(N^2 / W)$ for adjacency. Approx 1-2 MB.

One subtle detail:
Can `best_clique` be initialized to empty?
Yes. If no clique found (impossible for $N \ge 2$ and at least 1 edge), it outputs all 0.
Wait, max clique size is at least 1 (single vertex is clique).
The algorithm finds cliques of size $\ge 1$ because we pick first compatible node (always true for first node).
Wait, if graph has no edges?
Adjacency bitsets empty.
Permutation: pick 1st node. `cand` becomes empty.
Result size 1. Correct (any single vertex is a clique).
Wait, if graph is 2 vertices, no edge.
Perm: u, v.
Pick u. `cand` cleared.
Pick v. `cand[v]` false.
Result size 1.
Wait, max clique for empty graph is 1. Correct.
Wait, if graph is 2 vertices with edge.
Pick u. `cand`={v}.
Pick v. `cand`={}.
Result size 2. Correct.

Everything seems correct.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <bitset>

using namespace std;

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    // Use bitsets for adjacency matrix. Size 1001 to accommodate 1-based indexing.
    vector<bitset<1001>> adj(N + 1);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].set(v);
        adj[v].set(u);
    }

    // We will use a vector to store the best clique found so far
    vector<int> best_clique;
    int max_k = 0;

    // Random number generator
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    
    // Vector for permutation (vertices 1 to N)
    vector<int> p(N);
    iota(p.begin(), p.end(), 1); 

    // Pre-calculate degrees to optionally sort? 
    // Simple random shuffle is generally robust enough and avoids degree sorting bias.
    // However, sorting vertices by degree initially is a common heuristic.
    // Given the constraints and problem type, purely random permutations with many restarts 
    // is a solid approach for finding a large clique.

    // Run multiple random restarts. 200 iterations should be sufficient.
    for (int iter = 0; iter < 200; ++iter) {
        // Shuffle vertices randomly
        shuffle(p.begin(), p.end(), rng);
        
        // 'cand' bitset tracks vertices compatible with the current clique.
        // Initially, all vertices are candidates.
        bitset<1001> cand;
        cand.set(); // Sets all bits to 1
        
        // To avoid clearing the whole bitset repeatedly (though .set() is fast enough),
        // we can rely on the fact that we only need to set relevant bits.
        // But resetting to 1 is necessary.
        // Optimizing: Use a vector of ints for active candidates if sparsity is high?
        // Bitset intersection is very fast (16 integers), vector intersection is slower.
        // Stick with bitset.
        
        // Optimization: If cand becomes empty, we stop processing.
        // But we need to continue the loop to check remaining vertices? 
        // No, if cand is empty, no vertex can be added.
        // However, the 'cand' variable persists.
        // If we break, we just proceed to next iteration of outer loop.
        
        vector<int> current_clique;
        current_clique.reserve(N); // Reserve memory
        
        for (int u : p) {
            if (cand.test(u)) {
                // u is compatible with all currently selected vertices in current_clique
                current_clique.push_back(u);
                
                // Update candidates: must be neighbors of u
                // Note: We must ensure u itself is removed from candidates 
                // (unless self-loops exist, which constraints say no).
                // adj[u] has 0 at position u, so cand &= adj[u] clears bit u.
                cand &= adj[u];
            }
            
            // Small optimization: check if candidates are empty
            if (!cand.any()) {
                // If no candidates left, no more vertices can be added.
                // However, we must ensure that 'current_clique' is consistent.
                // The 'current_clique' is valid so far.
                // But wait, if cand becomes empty, does it mean we MUST stop?
                // Yes, because any future vertex will require compatibility with current clique.
                // Since no vertex satisfies this, we can't add anything.
                // And we've checked all vertices in 'p' anyway.
                // But for efficiency, we can break early?
                // If we break, we skip checking remaining p vertices.
                // They won't be added anyway.
                // Is it possible a vertex appears later in p that connects to current clique?
                // If cand is empty, NO vertex connects to ALL in current_clique.
                // So yes, we can break.
                // Wait, 'cand' tracks compatibility with ALL in current_clique.
                // If 'cand' is empty, NO vertex connects to all.
                // So no extensions possible.
                // Correct.
                break;
            }
        }
        
        // Check if this clique is better
        if ((int)current_clique.size() > max_k) {
            max_k = (int)current_clique.size();
            best_clique = current_clique;
        }
    }

    // Output results
    // We need to output N lines.
    // Using a boolean array for quick lookup
    vector<bool> is_in_best(N + 1, false);
    for (int u : best_clique) {
        is_in_best[u] = true;
    }

    for (int i = 1; i <= N; ++i) {
        if (is_in_best[i]) {
            cout << "1\n";
        } else {
            cout << "0\n";
        }
    }

    return 0;
}
```
Wait, I missed `<chrono>` header in snippet. Added.
Also `is_in_best` initialization.
Looks good.

One final check on constraints:
`2 ≤ N ≤ 1000`.
`1 ≤ M ≤ 500,000`.
If M is large, input reading dominates. `cin` is fast with sync disabled.
Code handles it.

The logic `cand &= adj[u]` is correct.
If `u` is in `current_clique`, `adj[u]` does not have bit `u` set.
So `cand[u]` becomes 0.
In next iteration, if `u` appears again? (Impossible, permutation has unique elements).
So `u` is effectively removed from consideration.
Wait, what if `u` was not added but remained in `cand`?
If `u` was not added (because `cand[u]` was false), it means `u` was incompatible.
It stays in `cand`? No, we didn't intersect.
Wait, if `u` is not compatible, it is not added.
It remains in `cand`?
The logic `if (cand.test(u)) { ... cand &= adj[u]; }`
If `cand[u]` is 0, we do nothing. `cand` remains as is.
This is correct. We only intersect if we add `u` to restrict candidates.
If `u` is skipped, `cand` does NOT intersect with `u`.
This means we are effectively considering `u` as "not compatible with prefix".
Wait.
If `cand[u]` is 0, `u` is incompatible.
Can `u` become compatible later?
No. `cand` accumulates restrictions.
If `u` was 0, it means there is some `v \in current_clique` such that `u` is not connected to `v`.
As we add more nodes to `current_clique`, `cand` shrinks further.
So `cand[u]` will remain 0.
Thus `u` will never be added.
This is logically correct for a greedy construction based on permutation.

Wait, if I have a clique $\{1, 2\}$.
$p = [1, 2, 3]$.
1. Process 1. In cand. Add. `cand` = neighbors of 1.
2. Process 2. In cand? Yes. Add. `cand` = neighbors of 2 & 1.
3. Process 3. In cand? Maybe.
If $p = [3, 1, 2]$.
Edges $(1,2), (2,3), (1,3)$.
1. Process 3. In cand. Add. `cand` = neighbors of 3 ({1, 2}).
2. Process 1. In cand. Add. `cand` = neighbors of 1 & 3 ({2}).
3. Process 2. In cand. Add. `cand` = neighbors of 2 & 1 & 3 ({}).
Result {3, 1, 2}.
Correct.

What if $p=[3, 1, 2]$ but 3 not connected to 1?
Edges $(1,2)$.
1. Process 3. In cand. Add. `cand` = neighbors of 3.
2. Process 1. In cand?
   If 3 not connected to 1, then 1 not in `adj[3]`.
   So `cand[1]` is 0.
   Skip 1.
3. Process 2.
   If 3 connected to 2. `cand` intersects `adj[3]` (contains 2).
   Is 2 in `cand`? Yes.
   Add 2.
   `cand` &= `adj[2]`.
   Result {3, 2}.
   Is {3, 2} valid? Edge $(3,2)$ exists.
   Is {3, 1} valid? No.
   Wait, skipping 1 meant we didn't form clique {1, 2}.
   We formed {3, 2}.
   Is {3, 2} a clique? Yes.
   Is {3, 1} a clique? No.
   Is {1, 2} a clique? Yes.
   Did we miss {1, 2}?
   Since 3 blocked 1, we took path 3 -> 2.
   Since 3 was first, it committed us to 3's neighborhood.
   So {1, 2} was unreachable because 1 was incompatible with 3.
   This is inherent limitation of greedy with fixed order.
   But with random restarts, we might pick order [1, 2, 3] and find {1, 2}.
   Or [2, 1, 3].
   So we are fine.

Final check.
The `best_clique` is `vector<int>`.
Printing requires `is_in_best` boolean array.
Code does this.
Correct.

One tiny thing:
`best_clique` is empty initially.
If N=1, loop runs. `current_clique` size 1.
Update `max_k` = 1. `best_clique` = {1}.
Output 1.
Correct.

Wait, if `cand` was empty initially (N>0, M=0).
`cand` initialized to 1s.
First `u` (say 1). `cand[1]` is 1.
Add 1. `cand` &= `adj[1]` (empty).
`cand` becomes all 0.
Break.
Result {1}.
Correct.

Okay, code is solid.

```cpp
// Imports
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <bitset>
#include <chrono>

using namespace std;

int main() {
    // ...
}
```
Include `<chrono>`?
`chrono` is C++11. Should be supported.
Also `steady_clock`.
Yes.

Final verification of memory/time.
Memory: Bitsets + vectors. $O(N^2)$ bits. $1000^2 = 10^6$ bits $\approx 125$ KB.
Vectors: minimal.
Time: 200 * N * N/64 $\approx 3 \cdot 10^6$ ops.
Time limit 2s is $2 \cdot 10^8$ ops.
Safe.

One paranoid check:
`cand.set()` sets ALL bits.
We access indices 1 to N.
Index 0 is unused.
It doesn't matter.

Okay.

```cpp
// Final Code Construction
```
Wait, I should define `max_k` as integer.
Comparison `current_clique.size()` (unsigned long usually, vector member type) vs `max_k`.
Cast to int is safer.
I did `(int)current_clique.size()`.
Good.

Wait, `best_clique` assignment `best_clique = current_clique`.
Vector copy. Length $\le N$.
Copying vector inside loop is expensive if done every time?
Wait, `best_clique` only updated if strictly better.
Updates happen rarely.
Average length is small.
Even if copied every time, cost is small.
$O(N)$ copy.
$200 \times 1000 = 2 \cdot 10^5$.
Negligible.

Okay.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <bitset>
#include <chrono>

using namespace std;

int main() {
    // Optimize standard I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    // Use bitsets for adjacency matrix. 
    // Size 1001 to accommodate 1-based indexing comfortably.
    vector<bitset<1001>> adj(N + 1);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].set(v);
        adj[v].set(u);
    }

    // Vector to store the best clique found so far
    vector<int> best_clique;
    int max_k = 0;

    // Random number generator seeded with current time
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    
    // Helper vector for permutation (vertices 1 to N)
    vector<int> p(N);
    iota(p.begin(), p.end(), 1); 

    // Run multiple random restarts to increase likelihood of finding a large clique.
    // 200 iterations provides a good balance between execution time and search quality.
    const int ITERATIONS = 200;
    
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        // Shuffle vertices randomly in each iteration
        shuffle(p.begin(), p.end(), rng);
        
        // 'cand' bitset tracks vertices compatible with the current clique.
        // Initialized to all 1s (all bits set).
        bitset<1001> cand;
        cand.set(); 
        
        vector<int> current_clique;
        current_clique.reserve(N);
        
        // Build the clique greedily according to the permutation order
        for (int u : p) {
            // Check if vertex u is still a candidate (i.e., compatible with all currently selected vertices)
            if (cand.test(u)) {
                // Add u to the clique
                current_clique.push_back(u);
                
                // Update candidates: new candidates must be neighbors of u.
                // This effectively maintains the intersection of neighborhoods.
                // Note: Since adj[u] has 0 at bit u (no self-loops), u is removed from cand.
                cand &= adj[u];
            }
            
            // Optimization: If no candidates remain, we cannot add any more vertices.
            // Break early to save time.
            if (!cand.any()) {
                break;
            }
        }
        
        // Update the best clique found if current one is larger
        if ((int)current_clique.size() > max_k) {
            max_k = (int)current_clique.size();
            best_clique = current_clique;
        }
    }

    // Prepare output: mark vertices in the best clique
    vector<bool> is_in_best(N + 1, false);
    for (int u : best_clique) {
        is_in_best[u] = true;
    }

    // Output exactly N lines
    for (int i = 1; i <= N; ++i) {
        if (is_in_best[i]) {
            cout << "1\n";
        } else {
            cout << "0\n";
        }
    }

    return 0;
}
```
This looks complete.

Wait, `max_k` initialized to 0.
If N=1000, M=0.
Loop runs.
`cand` set.
`u=1`. `cand[1]=1`. Push 1. `cand` &= 0 -> 0. Break.
`current_clique` size 1.
`max_k` becomes 1.
Output 1s?
Wait, `best_clique` = {1}.
Loop 1..N. `is_in_best` true for 1.
Output:
1
0
...
0
This is a valid clique of size 1.
Wait, if M=0, is {1} a clique? Yes.
Wait, problem says "Undirected graph G=(V,E)".
Subset S is clique if for every pair u,v in S, edge exists.
Condition for singleton {u}: No pairs u,v distinct. Condition vacuously true.
So {1} is valid.
Max clique size is 1.
So this is correct.

Double check: "for **every** pair of distinct vertices u, v ∈ S, there **IS** an edge".
Yes.

One small detail: `const int ITERATIONS = 200;`
Can I increase it?
$200 \times 16 \mu s = 3.2 ms$.
Can go up to 500.
But 200 is safe.

I will stick to 200.

One thought: `best_clique` assignment inside loop copies vector.
If `max_k` updated often (early iterations produce good results), copy is frequent.
But size grows? No, we overwrite.
Actually, if we find a clique of size 1000, `best_clique` is large.
Copy takes time.
But size 1000 vector copy is fast (memory move or byte copy).
Still fine.

The solution seems robust.

Final Code Block.
</think>

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <bitset>
#include <chrono>

using namespace std;

int main() {
    // Optimize standard I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N, M;
    if (!(cin >> N >> M)) return 0;

    // Use bitsets for adjacency matrix. 
    // Size 1001 to accommodate 1-based indexing comfortably.
    // N <= 1000, so bitset size 1001 is sufficient.
    vector<bitset<1001>> adj(N + 1);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].set(v);
        adj[v].set(u);
    }

    // Vector to store the best clique found so far
    vector<int> best_clique;
    int max_k = 0;

    // Random number generator seeded with current time for reproducibility/randomness balance
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    
    // Helper vector for permutation (vertices 1 to N)
    vector<int> p(N);
    iota(p.begin(), p.end(), 1); 

    // Run multiple random restarts to increase likelihood of finding a large clique.
    // 200 iterations provides a good balance between execution time (well within 2.0s) and search quality.
    const int ITERATIONS = 200;
    
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        // Shuffle vertices randomly in each iteration
        // This explores different orderings, which is crucial for heuristic Max Clique
        shuffle(p.begin(), p.end(), rng);
        
        // 'cand' bitset tracks vertices compatible with the current clique.
        // Initialized to all 1s (all bits set).
        // Indices 1..N correspond to vertices. Index 0 is unused.
        bitset<1001> cand;
        cand.set(); 
        
        vector<int> current_clique;
        current_clique.reserve(N);
        
        // Build the clique greedily according to the permutation order
        // If vertex u is compatible (in candidates), add it to clique and restrict candidates.
        for (int u : p) {
            // Check if vertex u is still a candidate (i.e., connected to all currently selected vertices)
            if (cand.test(u)) {
                // Add u to the clique
                current_clique.push_back(u);
                
                // Update candidates: new candidates must be neighbors of u.
                // This effectively maintains the intersection of neighborhoods.
                // Since the problem states no self-loops (u != v), adj[u] has 0 at bit u.
                // Consequently, u is removed from cand, preventing re-selection or invalidation.
                cand &= adj[u];
            }
            
            // Optimization: If no candidates remain, we cannot add any more vertices compatible with current clique.
            // Break early to save time.
            if (!cand.any()) {
                break;
            }
        }
        
        // Update the best clique found if current one is strictly larger
        if ((int)current_clique.size() > max_k) {
            max_k = (int)current_clique.size();
            best_clique = current_clique;
        }
    }

    // Prepare output: mark vertices in the best clique
    // Using a boolean array for O(1) lookup
    vector<bool> is_in_best(N + 1, false);
    for (int u : best_clique) {
        is_in_best[u] = true;
    }

    // Output exactly N lines as required
    for (int i = 1; i <= N; ++i) {
        if (is_in_best[i]) {
            cout << "1\n";
        } else {
            cout << "0\n";
        }
    }

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=1002)

````
I'll implement a heuristic approach to find a large clique in an undirected graph.

Since finding the maximum clique is NP-hard, I'll use a randomized greedy strategy with vertex ordering by degree.

The key insights:
1. Start with high-degree vertices (more edges = more likely to be part of a large clique)
2. Iteratively build the clique from candidates that are connected to all current clique members
3. Add randomness to avoid bad local optima
4. Use early termination with a minimum threshold

Algorithm:
- Randomly shuffle vertices and compute degrees
- Process vertices by descending degree order
- Build the clique iteratively, adding vertices connected to ALL current clique members
- Try random perturbations (removing some vertices temporarily, then rebuilding)
- Early exit if the remaining candidates are too few
</think>

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

int main() {
    int N, M;
    cin >> N >> M;
    
    vector<vector<int>> adj(N + 1);
    for (int i = 0; i < M; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    static vector<int> p(N);
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    iota(p.begin(), p.end(), 0);
    
    int best_k = 0;
    vector<int> best_clique;
    
    for (int k = 0; k < 60; k++) {
        shuffle(p.begin(), p.end(), rng);
        
        int max_remind_threshold = max(40000 - 500 * k, 5000);
        vector<int> visited(p.begin(), p.end(), 0);
        
        int rem_size = 0;
        int start_pos = -1;
        
        for (int x : p) {
            if (adj[x].size() >= max_remind_threshold && !visited[x]) {
                rem_size++;
                start_pos = x;
            }
        }
        
        vector<bool> x(N + 1, false);
        vector<int> candidate;
        if (start_pos != -1) {
            visited[start_pos] = 1;
            x[start_pos] = true;
            candidate.push_back(start_pos);
            rem_size--;
        }
        
        bool changed = true;
        while (changed && rem_size > 0) {
            changed = false;
            int best_x = -1;
            int max_common = 0;
            int common_count = candidate.size();
            
            for (int x : p) {
                if (x <= best_x) continue;
                if (!candidate.empty() && x == candidate[0]) continue;
                if (x == candidate.back()) continue;
                
                if (visited[x] || !x_in(adj[x], candidate)) continue;
                
                if (common_count == 1 && x >= candidate[back(candidate)] && candidate.size() >= 3) {
                    break;
                }
                
                if (common_count == 2 && x >= back(candidate[1]) && x <= front(candidate)) {
                    break;
                }
                
                int new_common = count(common_adj(x, candidate));
                if (new_common > max_common) {
                    max_common = new_common;
                    best_x = x;
                }
            }
            
            if (best_x != -1) {
                visited[best_x] = 1;
                x[best_x] = true;
                candidate.push_back(best_x);
                rem_size--;
                changed = true;
            }
            
            if (rem_size <= max_remind_threshold / 100 && candidate.empty()) {
                rem_size += 50;
            }
            
            if ((double)candidate.size() / (double)(candidate.size() + rem_size + cand_cnt) <= 0.07) {
                break;
            }
        }
        
        if (candidate.size() > best_k) {
            best_k = candidate.size();
            best_clique = candidate;
        }
    }
    
    cout << best_clique.size() << endl;
    for (auto &v : best_clique) {
        cout << v << " ";
    }
}
```
````
