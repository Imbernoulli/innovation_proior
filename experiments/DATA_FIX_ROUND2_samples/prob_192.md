# 题 192 — base均分 70.6 vs 直接SFT均分 13.5(掉点 57.1)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=2,得分 82.1,completion_tokens=12418)

````
The user wants a C++ solution for the Max-Cut problem.

**Problem Analysis:**
Max-Cut is a classic NP-hard optimization problem. The goal is to partition the vertices of a graph into two disjoint sets $S$ and $T$ such that the number of edges connecting a vertex in $S$ to a vertex in $T$ is maximized.

**Constraints:**
- $n \le 1000$
- $m \le 20000$
- Time limit isn't explicitly stated, but standard limits are around 1-2 seconds.
- Memory limit isn't explicit, usually 256MB+.

**Complexity:**
Since Max-Cut is NP-hard, finding an exact solution for general graphs requires exponential time or heuristic/approximation algorithms. However, $N=1000$ is quite large for exponential time complexity ($O(2^n)$).
Wait, let me double check the problem constraints and context. Usually, competitive programming problems have either small $N$ (like $N \le 30$) for exponential solutions, or specific structure (bipartite checking) for polynomial solutions, or they ask for approximation.
Is this problem actually asking for *any* valid cut, or the *maximum* cut? "maximize the number of edges crossing the partition". This definitely means Maximum Cut.

However, with $N=1000$, solving Exact Max-Cut is computationally infeasible.
Let's look at the scoring: "score = c / m". This implies we want to maximize the ratio.
Is there a trick?
Usually, Max-Cut is approximated by 0.878-factor using Semidefinite Programming, or randomized local search / greedy heuristics (like Kernighan-Lin) can be used.
But wait, maybe the graph has special properties or the test cases are weak?
Or maybe I am overestimating the difficulty?
Actually, for $N=1000$, an exact solution is impossible. A randomized local search or simulated annealing often works well on random instances. But competitive programming platforms might have weak test cases or expect an approximation algorithm if it's hard.
Wait, looking at similar problems from contests (e.g., CodeChef, HackerRank), sometimes "Max-Cut" problems appear with smaller constraints or require specific logic.
However, there is a specific case where Max-Cut is easy: Bipartite graphs. If the graph is bipartite, the max cut is all edges ($c=m$).
If it's not bipartite, it's harder.

Let's reconsider the problem source. This looks like a variation of the Max-Cut problem often used to test approximation algorithms or heuristics when $N$ is small enough for some exponential approaches (which 1000 is not) or large enough for heuristics.
Given the constraints $N \le 1000$, the intended solution is likely a heuristic or an approximation, OR the problem statement implies something simpler that I'm missing.
Wait, could it be solvable using Minimum Cut (Min-Cut)?
By min-cut/max-flow duality? No, Min-Cut minimizes edges between two partitions. We want to MAXIMIZE.
Max-Cut $\approx N/2$ edges on average.
Actually, there is a known randomized algorithm for Max-Cut:
Assign vertices randomly. Expected cut size is $m/2$.
Flip assignment if it improves the cut.
Repeat until no improvement.
This gives a good approximation (actually optimal for many practical instances).
Also, local search:
1. Initialize cut (e.g., random or first vertex to set 0, propagate bipartite matching).
2. While there exists a vertex moving its side that increases cut size, move it.
3. Return current cut.
This converges quickly. With $N=1000$, this is $O(N \cdot M)$ or $O(N^2)$ per iteration, which fits easily within time limits if iterations are few.

There is also a branch-and-bound approach or dynamic programming, but those are too slow.
Is there any chance this problem is actually asking for something else? "partition the vertices into two sets to maximize..."
Okay, since it's NP-hard, and $N=1000$, it must be expecting an approximation or a specific property.
However, often in these "solve this" requests, if it's from a contest where $N=1000$, they might allow $O(N \log N)$ or $O(N)$ approximations, or the test cases are easy.
But the most standard way to handle Max-Cut on larger graphs in CP contexts (when exact is needed but $N$ allows it via specific solvers) is... wait.
If this is from a platform like Hackerrank or similar, usually there's a "Greedy" or "Local Search" solution expected.
A common greedy strategy:
1. Place first vertex in S.
2. For subsequent vertices, place in T if connected to more vertices in S than in T (this is the "greedy" placement).
3. Or simply try both assignments? No, $2^{1000}$.

Let's assume the question expects a high-quality heuristic/local search because it's NP-Hard.
Algorithm:
1. Assign vertex 1 to Set 0.
2. Propagate: if edge $(u, v)$, assign $v$ to opposite of $u$. Wait, this finds a 2-coloring if bipartite. If cycle is odd, it conflicts.
3. Start with random assignment or BFS-based assignment. Then perform Local Search:
   Iterate through all vertices. Check if flipping vertex $i$ increases the number of cut edges. If yes, flip.
   Repeat until convergence.
   Since the objective function decreases as cuts decrease? No, we maximize cuts. Flipping a node changes degree $d_i$ cut status.
   Specifically, if we flip $v$:
     New Cut = Old Cut + (edges to same set) - (edges to diff set)
     Benefit = neighbors_in_same_set - neighbors_in_diff_set
     If Benefit > 0, flip.
   To ensure getting closer to optimum, repeat multiple passes or use hill climbing.
   
   Is this enough? For $N=1000$, simple local search usually gets very close to maximum cut.
   
   Another perspective:
   Is it possible to map this to something easier?
   There is no direct mapping to Min-Cut for Maximization.
   However, given the prompt format ("You are a competitive programmer"), providing an approximate solution is risky unless specified. But exact is impossible.
   Maybe the constraints are loose enough that $O(M \cdot N)$ is acceptable?
   Let's refine the Local Search.
   
   Algorithm:
   1. Initialize `assign` array (e.g., all 0s or random).
   2. Calculate initial cut size.
   3. Loop:
      improved = false
      For i from 1 to n:
         Calculate gain if flipped.
         Gain = (count neighbors currently 0) - (count neighbors currently 1).
         If Gain > 0:
            Flip i.
            Update cut size.
            improved = true
      If not improved, break.
      
   Complexity: In each pass $O(M)$ (or $O(N+N_{edges})$). How many passes? Usually it stabilizes fast.
   Worst case might be cycles oscillating, but with $N=1000$, usually it's fine.
   
   Wait, there's a risk of getting stuck in local optima. A better approach is Simulated Annealing or restarting from different seeds.
   Given "Competitive Programmer" persona, typically one would implement the most robust heuristic available without external libraries.
   
   Actually, there is a specific observation for Max-Cut:
   If we just run the local search (also called the "Kernighan-Lin" type or simple FLIP heuristics), it performs quite well.
   Also, the score calculation suggests we care about $c/m$.
   
   Let's write the solution implementing Local Search.
   Data structures: Adjacency list.
   Vector `s` storing partition (0 or 1).
   Array `degree[n]` counting neighbors in opposite set (cut edges incident to `i`).
   Wait, to check gain efficiently:
   When `s[i]` flips:
     Gain = `neighbors_same_side` - `neighbors_opposite_side`.
     Note: `neighbors_opposite_side` contributes to cut edges.
     If `s[i]` changes from 0 to 1:
       Edges to 0 become cut edges.
       Edges to 1 stop being cut edges.
       Change in total cut edges = (edges to 0) - (edges to 1).
     So we need to count how many neighbors are in same group vs other group.
   
   Optimization:
   Store adjacency lists.
   Maintain `current_cut` count.
   Precompute degrees.
   
   Implementation details:
   Input parsing: fast IO.
   Graph representation: `vector<int> adj[N+1]`.
   `s[N+1]` initialized to 0.
   Iterate $k$ times (e.g., 100 passes).
   Inside pass: iterate all nodes $i$. Calculate potential gain. If positive, flip.
   
   Is 100 passes enough? For $N=1000$, likely yes for typical test cases.
   To avoid getting stuck, shuffle the order of vertices or restart.
   Multiple restarts with different random seeds improve result quality.
   Given the time limit is usually 1-2s, 10-20 restarts of a full sweep takes minimal time if implemented efficiently.
   
   One crucial detail:
   The output must be `n integers s1 ... sn`.
   Indices are 1-based.
   
   Let's draft the code.
   
```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Heuristic initialization
    // Start with a simple bipartite coloring attempt or random
    vector<int> s(n + 1, 0);
    
    // Try random initialization to escape bad local optima?
    // Actually, deterministic first pass then local search is standard.
    // Let's start by assigning based on connectivity or just 0.
    // Better: Use a simple heuristic like greedy.
    // However, pure 0 initialization is risky.
    // Let's stick to Local Search starting with all 0s. 
    // It's often sufficient for weak test cases, but strong ones need restarts.
    // With N=1000, running 20 passes is safe.
    
    // Current cut value
    long long cut_size = 0;

    // Helper to count neighbors of x in set 0 vs 1
    auto update_info = [&](int x) {
        // Not needed if we recalculate fully or maintain state
        // Maintaining state: 
        // cnt[x][side] ?
    };
    
    // Efficient calculation of neighbors in same/opposite set
    // When evaluating flip of i:
    // gain = neighbors[opposite] - neighbors[same]
    // Because current contribution is neighbors[opposite].
    // After flip, contribution becomes neighbors[same].
    // Change = new - old = neighbors[same] - neighbors[opposite] (if flip 0->1)
    // Wait, let's trace carefully.
    // Let current color of neighbor j be s[j].
    // Edge (i,j) is a cut edge if s[i] != s[j].
    // If we flip s[i], edge (i,j) becomes non-cut if it was cut, and cut if it was not.
    // So number of cut edges incident to i changes from deg_cross(i) to deg_same(i).
    // Total cut change = sum over j of (+1 if now cut, -1 if no longer cut).
    // New Cut = Old Cut - deg_cross(i) + deg_same(i).
    // Improvement = deg_same(i) - deg_cross(i).
    // We need deg_same(i) and deg_cross(i).
    // deg_cross(i) = number of neighbors j where s[i] != s[j].
    // deg_same(i) = number of neighbors j where s[i] == s[j].
    
    // Let's maintain counts on the fly or recompute. Recomputing for every node every pass is O(M).
    // Pass is O(N*M). Total passes P. Total P*N*M.
    // With N=1000, M=20000, N*M = 2*10^7. 20 passes -> 4*10^8 ops. Might be tight for 1 sec.
    // Optimized: 
    // Calculate global best.
    // Just loop.
    
    // To speed up:
    // We only need to check improvement.
    // Optimization: compute 'deg_cross' incrementally?
    // No, simpler is O(M) per pass.
    // Can we optimize the check?
    // For a fixed pass, calculating deg_same and deg_cross for all nodes:
    // Iterate i from 1 to n:
    //    count_0 = 0, count_1 = 0
    //    for neighbor in adj[i]:
    //       if s[neighbor] == 0:
    //          if s[i] == 0 count_0++ else count_1++
    //       else
    //          if s[i] == 0 count_1++ else count_0++
    // This is O(deg(i)). Summing over all i is O(2M) = O(M).
    // So one full sweep is O(M).
    // Number of sweeps?
    // Local search converges fast. Usually less than 10-20 for dense/random, maybe more for structured.
    // With 2*10^7 ops per sweep, 20 sweeps is 4*10^8, borderline.
    // We can reduce sweeps if we don't touch everything.
    // Only pick nodes that can improve? Or just random selection?
    // Standard Hill Climbing: pick best improving node?
    // If we greedily find ANY node that improves, flip it and continue.
    // This is faster (fewer flips needed to stabilize? No).
    // Greedily finding best flip: still O(M) per step, but fewer steps.
    
    // Let's try a hybrid:
    // 1. Run several sweeps of full pass with updates only when beneficial.
    // 2. Limit total operations or time? (Hard to control exactly without chrono, but 10^8 is usually OK in C++ for 1s if simple ops).
    // Operations here are additions and array accesses. Very cache friendly? Adjacency list traversal might jump.
    
    // To be safe, let's use a randomized restart with fewer iterations.
    // Actually, simply iterating through the graph and fixing local optimality is a standard technique.
    // Let's implement a multi-start local search.
    
    best_score = -1;
    best_s = s;
    
    // Random seed for reproducibility or variance?
    // We want consistent output usually, but maximizing score.
    // Deterministic approach: 0 initialization.
    
    for(int iter=0; iter<10; ++iter) {
        // Reset s
        // ... calculate scores ...
        
        while(true) {
            bool improved = false;
            // Find node with max positive gain
            int best_node = -1;
            int best_gain = -1;
            
            // This is O(N*M) worst case per outer loop if we scan all.
            // Instead, scan nodes once, pick one that improves, flip, and repeat inner loop.
            // Or just one pass (sweep), flip all that improve.
            // Let's do full sweep, accumulating flips.
            
            // Actually, standard Local Search is:
            // while exists i such that flip(i) improves:
            //    pick i
            //    flip(i)
            
            // Doing "pick best i" or "pick first i"? "Pick first i" is faster to check stability but slower convergence?
            // Actually, if we pick any i that improves, it might create ripple effect.
            // To guarantee convergence, we must ensure we don't cycle.
            // Flipping changes state. We can track visited states? No, too many.
            // But objective (cut size) strictly increases. Finite states. So termination guaranteed.
            // Problem: state space is $2^N$.
            // With N=1000, we can't visit all. But objective is integer, max $M$.
            // It stops when no i improves.
            
            // Implementation of single sweep local optimization:
            // For i = 1 to n:
            //    calculate gain.
            //    if gain > 0:
            //        update cut size locally.
            //        s[i] = 1 - s[i];
            //        mark improved
            
            // If after one full scan no gains found, we stopped.
            // This is O(M) per iteration.
            // How many scans?
            // Empirically small number for random graphs.
            // For specific crafted cases, could be larger.
            // Let's add random restarts to be safe against local optima.
            
        }
    }
}
```

Refining the strategy to be safer:
1. **Multiple Starts**: Run the local search procedure $K$ times (e.g., 50). Keep the best configuration.
   Each start starts with a random partition.
   Then applies greedy local search (flip any node that improves, repeatedly).
   This ensures escaping shallow local optima.
   
2. **Efficient Update**:
   When flipping $i$:
   Recalculate neighbors' contribution? No, update current total cut size in $O(\text{deg}(i))$.
   To determine gain:
   Count neighbors with $s[j] == s[i]$ and neighbors with $s[j] \neq s[i]$.
   Since `adj` stores neighbors, iterating them takes $O(\text{deg}(i))$.
   In the worst case, inside the `while(improved)` loop, we do work proportional to $M$ (sum of degrees).
   The cost per flip is $O(\text{deg}(i))$.
   The total cost depends on how many flips happen.
   Usually, flips diminish.
   
   Wait, recalculating gain for *all* nodes in every sweep is safer than picking one, but slightly slower.
   Let's do the sweep method:
   `changed = true`
   `while changed`:
     `changed = false`
     `for v in 1..n`:
       `gain = count(same) - count(diff)`
       `if gain > 0`:
          `update_total_cuts(v, -gain)` // Careful with signs
          `s[v] ^= 1`
          `changed = true`
          
   Is updating `total_cuts` inside the loop safe? Yes, `total_cuts` tracks global state.
   However, if we iterate `v` sequentially, the state of neighbors might change before we process the next `v`. Does this matter?
   Yes, it matters for the sequence. That's what makes it a local search.
   This is basically "Greedy with respect to immediate gain, processing nodes in index order".
   Since we want to maximize, any improvement is good.
   Restarting with random ordering/shuffling might help.
   
   Wait, strict time constraints ($N=1000$, 1s) mean $O(N^2)$ or $O(EM)$ is preferred.
   If we do $K$ restarts, and each restart runs for say 1000 flips, total ops $K \times (\text{avg\_flip\_cost} \times \text{deg\_avg})$.
   Avg degree $\approx 2M/N$. Cost per flip $O(deg)$.
   Ops $\approx K \times F \times 2M/N \times (\text{something?})$.
   Basically $O(K \cdot M)$.
   With $K=20$, $M=20000$, Ops $\approx 400,000$. Very fast.
   Wait, why did I think it was $O(NM)$?
   In a pass, scanning all edges to compute gains is $O(M)$.
   If we just flip, we update.
   So one pass is $O(M)$.
   If number of passes is small (say < 100), total is $O(P \cdot M)$.
   With $P \approx 100$, $P \cdot M \approx 2 \cdot 10^6$. Extremely fast.
   Wait, does it converge in 100 passes?
   Usually, yes.
   The danger is oscillating between two configurations without improvement.
   But since `improved` is defined strictly (>0), if we get to a stable state, we stop.
   However, in the sweep loop, we modify `s` dynamically.
   Later nodes see updated `s` values.
   This prevents cycling back to previous states easily because we move forward.
   
   Okay, so the complexity per run is dominated by the number of passes times M.
   We can afford ~200 passes easily.
   
   Wait, let's verify the "Gain" logic.
   If $s[i]=0$:
     Neighbors in 0: same set.
     Neighbors in 1: diff set (cut).
     Moving to 1:
       Edges to 0 become cut (Gain +size)
       Edges to 1 become same (Lose -size)
     Net change = count_neighbors(0) - count_neighbors(1).
     If positive, flip.
   Correct.
   
   So, Plan:
   1. Read input.
   2. Function `run_local_search()` returns `(cut_value, s_vector)`.
   3. Call `run_local_search` multiple times with random init.
   4. Output best result.
   
   Since $N=1000$, creating vectors and random seeds is cheap.
   
   Double check: The problem asks for "a solution", not necessarily *the* optimal solution, but usually we aim for the best. Given it's NP-hard, we provide the best heuristic we can.
   
   Wait, is there a greedy algorithm that guarantees 0.9 or something?
   Goemans-Williamson is SDP. Too complex to implement.
   Randomized rounding + local search is standard.
   
   Wait, I should consider if the graph is disconnected.
   If disconnected, solve for each component independently?
   No, the two sets are global. Vertices from different components can go to same set or diff sets.
   Actually, for a connected component, the partition (S,T) can be swapped (T,S) without changing the number of cut edges.
   So we can fix vertex 1's position arbitrarily (it's always 0 or 1 relative to others).
   But since we just want to find *a* partition maximizing cuts, we can treat components independently and then combine?
   Actually, for a component, we want to maximize its internal cut edges.
   But we also have freedom to swap the whole component's labels.
   Does swapping affect cross-component edges?
   Edges only exist within components (by definition of connectivity).
   So components are independent.
   We can solve max cut for each component separately?
   Wait, if I swap component A's sets (0->1, 1->0), cut edges within A stay same.
   What about interaction with B?
   There are NO edges between A and B.
   So interactions are zero.
   So yes, we can solve for each connected component independently and concatenate results.
   Wait, if I solve Component 1 and Component 2 independently, I get two pairs of (sets).
   e.g. C1: (S1, T1), C2: (S2, T2).
   Any combination of (S1/S2...) doesn't introduce cross edges.
   So solving independently is valid and saves time?
   Wait, if I find the optimal split for C1 and C2, I get a valid split for G.
   So yes! This reduces $N$ to sum of $N_{comp}$.
   It doesn't change complexity asymptotically, but good for small components.
   
   However, my Local Search handles this naturally (nodes far apart don't interact much).
   Explicitly handling components isn't necessary for correctness, but maybe helpful for performance if many small components.
   Given $N=1000$, overhead of BFS for components might outweigh benefits. I'll skip explicit component detection to keep code short and focus on the core algorithm.
   
   **Code Structure:**
   - Includes.
   - Fast IO.
   - `vector<int> adj[1005]`
   - `main` logic.
   
   One optimization for the sweep:
   Instead of checking all neighbors for every node, precalculate degrees.
   Actually, just iterating the adjacency list is fine.
   
   Random Initialization Strategy:
   Shuffle `1..n` and assign randomly to 0 or 1.
   Or just assign 0/1 based on parity of index, or random.
   Random is better.
   
   Max Iterations?
   Limit passes to say 2000. If no change for a block of passes, stop?
   Or just fix `MAX_PASS = 100`.
   Usually 10 passes of full graph scan is plenty for local search.
   Let's try 50 passes.
   And 10 random restarts.
   
   Wait, "Time limit" is implicit.
   C++ loops $10^8$ take ~0.2-0.5s.
   $10 \times 100 \times 20000 = 2 \times 10^7$ ops. Totally safe.
   
   Wait, the "score" calculation logic provided in problem description is for *scoring*, not what we need to output.
   We just output the partition.
   Wait, "Output exactly one line: n integers s1 s2 ... sn".
   So we don't output the score, just the partition string.
   
   One corner case: $N=1, M=0$.
   Output 0 or 1.
   
   Corner Case: Disconnected graph. Handled naturally.
   
   Corner Case: Multiple edges? Problem says "no multiple edges or self-loops".
   
   Final check on constraints:
   $N \le 1000$.
   $M \le 20000$.
   Solution complexity: roughly $O(Restarts \times Passes \times (N + M))$.
   $10 \times 100 \times 20000 \approx 2 \times 10^7$.
   Very efficient.
   
   Wait, is there a possibility of TLE with this many passes?
   Some graphs are nasty for Local Search (many local optima).
   Using multiple restarts is key.
   Maybe implement a check: stop if no change in X consecutive passes?
   Yes, "stabilized".
   
   ```cpp
   #include <iostream>
   #include <vector>
   #include <random>
   #include <numeric>
   #include <algorithm>

   using namespace std;

   int main() {
       ios_base::sync_with_stdio(false);
       cin.tie(NULL);

       int n, m;
       if (!(cin >> n >> m)) return 0;

       vector<vector<int>> adj(n + 1);
       for (int i = 0; i < m; ++i) {
           int u, v;
           cin >> u >> v;
           adj[u].push_back(v);
           adj[v].push_back(u);
       }

       // Variables to store best solution found
       vector<int> best_s(n + 1);
       int max_cuts = -1;

       // RNG setup
       mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
       
       // Helper to initialize a random vector
       auto init_random = [&](vector<int>& s) {
           fill(s.begin(), s.end(), 0);
           // Assign each node randomly to 0 or 1
           uniform_int_distribution<> dis(0, 1);
           for(int i=1; i<=n; ++i) {
               s[i] = dis(rng);
           }
       };

       // Try a few random restarts
       const int NUM_RESTARTS = 10; 
       const int MAX_PASSES = 100; 

       for (int r = 0; r < NUM_RESTARTS; ++r) {
           vector<int> s(n + 1);
           init_random(s);

           // Current cut size calculation
           int current_cuts = 0;
           
           // Pre-calculate degrees? No, just iterate.
           
           // Optimization: Compute initial cuts
           for(int i=1; i<=n; ++i) {
               for(int neighbor : adj[i]) {
                   if(i < neighbor && s[i] != s[neighbor]) {
                       current_cuts++;
                   }
               }
           }

           bool improved = true;
           int pass_count = 0;

           while (improved && pass_count < MAX_PASSES) {
               improved = false;
               pass_count++;
               
               // Iterate and flip if beneficial
               // Note: Iterating 1..n modifies s. 
               // This acts like a greedy pass.
               for(int i=1; i<=n; ++i) {
                   int same = 0;
                   int diff = 0;
                   
                   // Count neighbors in same/diff sets
                   for(int neighbor : adj[i]) {
                       if(s[neighbor] == s[i]) same++;
                       else diff++;
                   }
                   
                   // If diff is current contribution.
                   // If we flip: contribution becomes same.
                   // Change = same - diff.
                   if(same > diff) {
                       // Flip i
                       s[i] = 1 - s[i];
                       // Update current_cuts?
                       // Actually we just need to know if we made progress.
                       // But to optimize, we can track cuts.
                       // Wait, recalculating global cuts every time is O(M) or O(M/N)?
                       // Tracking change is O(deg(i)).
                       
                       // Let's just track change to avoid recomputing global cuts constantly.
                       // But wait, if we flip multiple nodes in one pass, their changes interact?
                       // No, we just need the state to be correct.
                       // Global cuts update:
                       // current_cuts += same - diff;
                       improved = true;
                   }
               }
           }
           
           // Check if better than global best
           int final_cuts = 0;
           for(int i=1; i<=n; ++i) {
               for(int neighbor : adj[i]) {
                   if(i < neighbor && s[i] != s[neighbor]) {
                       final_cuts++;
                   }
               }
           }
           
           // Compare best
           if(final_cuts > max_cuts) {
               max_cuts = final_cuts;
               best_s = s;
           }
       }

       // Output result
       for(int i=1; i<=n; ++i) {
           cout << best_s[i] << (i == n ? "" : " ");
       }
       cout << "\n";

       return 0;
   }
   ```
   
   Refinement on the Local Search Loop:
   The nested loop structure inside `while(improved)` is:
   `for i=1..n`, check condition, update `s[i]` immediately.
   This is correct for "Hill Climbing" (specifically Steepest Ascent? No, First Descent/Ascent depending on choice).
   Here we check all `i`. Since we want to maximize, `same > diff` is correct.
   If multiple nodes satisfy this, we pick the first one encountered.
   Updating `s[i]` immediately affects the checks for subsequent `j` (since `adj[i]` contains `j`).
   This is a form of Sequential Hill Climbing. It's fast and generally effective.
   However, tracking `current_cuts` update:
   When `s[i]` flips:
     `change = same - diff`
     `current_cuts += change`
   
   Wait, if we flip `s[i]`, `diff` becomes `same` and `same` becomes `diff`?
   Yes.
   If `same > diff`:
     New cut incident to `i` will be `old_same` (which was non-cut) minus `old_diff` (which was cut).
     Wait.
     Before flip:
       Edges to Same Set: `same` count. These are NOT cut edges.
       Edges to Diff Set: `diff` count. These ARE cut edges.
     Contribution to total cut count from `i` = `diff`.
     After flip:
       Nodes previously in Same are now Diff. (Contribution `same`).
       Nodes previously in Diff are now Same. (Contribution `diff` lost).
       New Contribution = `same`.
     Change = `New - Old` = `same - diff`.
     Yes, logic holds.
     
   Wait, is `current_cuts` maintained correctly?
   Yes.
   
   One issue:
   If we update `s` during the loop, and later in the same pass another node flips, the global cut calculation needs to be consistent?
   Yes, `current_cuts` maintains the invariant of the current state `s`.
   However, comparing against `max_cuts` requires recalculating `final_cuts` accurately at the end of each restart.
   My code recalculates `final_cuts` fully at the end, which is safe and O(M).
   
   Potential Bug: `init_random` fills `s` with 0 or 1. `fill` puts 0. Loop puts 0/1. Correct.
   `best_s` needs to be reset or overwritten. `best_s = s` copies. Correct.
   
   Constraints check:
   $N=1000$, $M=20000$.
   Memory: vectors `adj` take $4 \times 20000 \times 2 \approx 160KB$. Trivial.
   Time: 10 restarts $\times$ 100 passes $\times$ ($N + M$) ops.
   $10 \times 100 \times 30000 \approx 3 \times 10^7$ ops.
   Should run in ~0.1s.
   
   Is there any library header missing?
   `<chrono>` is used for seeding `mt19937`. It's standard C++11.
   `<random>` is needed.
   
   Final verification:
   The prompt asks for C++ code wrapped in ````cpp.
   No explanation.
   
   Wait, `s` is 1-based?
   `s` size `n+1`. Indices `1` to `n`.
   Loops `1` to `n`.
   Correct.
   
   One subtle thing:
   The problem statement says "Score is clamped to [0, 1]" but we just output the partition.
   The scoring is for evaluation, irrelevant for our generation.
   
   The solution looks solid for the constraints and problem type.
   Max-Cut is NP-hard, so exact solution is out. Heuristic is required.
   Local Search is the standard heuristic.
   
   Wait, I should consider if `MAX_PASSES` should be higher.
   Sometimes it takes more passes to reach good local optimum.
   With random restarts, even a few passes per restart give decent coverage.
   Let's increase passes to 500 just to be sure, won't hurt significantly for $N=1000$.
   Actually 500 passes might be overkill and slow down. 100 is standard for this scale.
   Let's stick to 100.
   
   Also, include `<vector>`, `<iostream>`, `<random>`, `<numeric>`, `<algorithm>`.
   
   Double check variable names and scope.
   `adj` vector resized correctly.
   `best_s` initialized to size `n+1`.
   
   Looks good.

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

// Function to calculate the number of cut edges given the partition
int count_cuts(const vector<int>& s, const vector<vector<int>>& adj, int n) {
    int cuts = 0;
    for (int i = 1; i <= n; ++i) {
        for (int neighbor : adj[i]) {
            if (i < neighbor && s[i] != s[neighbor]) {
                cuts++;
            }
        }
    }
    return cuts;
}

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Variables to store best solution found
    vector<int> best_s(n + 1);
    int max_cuts = -1;

    // Setup Random Number Generator
    unsigned seed = chrono::steady_clock::now().time_since_epoch().count();
    mt19937 rng(seed);
    uniform_int_distribution<> dis(0, 1);

    // Hyperparameters for the local search heuristic
    const int NUM_RESTARTS = 20; 
    const int MAX_PASSES = 100; 

    for (int r = 0; r < NUM_RESTARTS; ++r) {
        vector<int> s(n + 1);
        
        // Initialize partition randomly
        for(int i = 1; i <= n; ++i) {
            s[i] = dis(rng);
        }

        // Evaluate initial partition
        // We will track current cuts incrementally to save time?
        // Given the complexity, full recount at end is fine, incremental in loop is optional.
        // Let's do incremental update during passes to be faster.
        // But to simplify logic, let's just rely on full recount at end.
        // Incremental update is easy though.
        
        int current_cuts = 0;
        for(int i = 1; i <= n; ++i) {
             for(int neighbor : adj[i]) {
                 if(i < neighbor && s[i] != s[neighbor]) {
                     current_cuts++;
                 }
             }
        }

        bool improved = true;
        int pass_count = 0;

        while (improved && pass_count < MAX_PASSES) {
            improved = false;
            pass_count++;
            
            // Iterate through all nodes and flip if it increases the cut
            for(int i = 1; i <= n; ++i) {
                int same_neighbors = 0;
                int diff_neighbors = 0;

                for(int neighbor : adj[i]) {
                    if(s[neighbor] == s[i]) same_neighbors++;
                    else diff_neighbors++;
                }

                // If we flip s[i], edges to 'same' neighbors become cut edges.
                // Edges to 'diff' neighbors stop being cut edges.
                // Net change = same_neighbors - diff_neighbors.
                // We want change > 0 => same > diff.
                
                if (same_neighbors > diff_neighbors) {
                    s[i] = 1 - s[i];
                    current_cuts += (same_neighbors - diff_neighbors);
                    improved = true;
                }
            }
        }

        // Update global best if this partition is better
        // Recompute to ensure accuracy (though delta accumulation is usually fine)
        int cuts = 0;
        for(int i = 1; i <= n; ++i) {
            for(int neighbor : adj[i]) {
                if(i < neighbor && s[i] != s[neighbor]) {
                    cuts++;
                }
            }
        }
        
        if(cuts > max_cuts) {
            max_cuts = cuts;
            best_s = s;
        }
    }

    // Output result
    for(int i = 1; i <= n; ++i) {
        cout << best_s[i] << (i == n ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
Wait, one logic error in the loop.
When `s[i]` is updated, `current_cuts` is updated.
Later in the same pass, `j` is considered.
Its calculation `same_neighbors` and `diff_neighbors` uses the *updated* `s` array (because `s` was modified).
Is this correct?
Yes, this is the essence of the sequential improvement strategy (steepest ascent or rather just greedy improvement).
Does it lead to infinite loops?
Since `current_cuts` strictly increases with every flip, and max cut is bounded by M, it terminates.
Is it correct to update `current_cuts`?
Yes, `current_cuts` reflects the true cut size of the *current* `s`.
However, note that if we flip `i`, it might change `diff_neighbors` for `j`.
My code recalculates `same_neighbors` and `diff_neighbors` for `j` by iterating `adj[j]` fresh. So it sees the updated `s[i]`. This is correct.
This ensures consistency.

One tiny optimization:
Inside the loop, `s[i]` is changed.
Does it make sense to process `j` where `j < i`? No, usually `j > i` or arbitrary order.
Since `i` goes 1 to n, dependencies are handled by the fact that `s` is fully scanned.
It effectively is a wave of improvements.
Seems robust enough.

Final check on headers: `<vector>`, `<iostream>`, `<random>`, `<chrono>`. All present.

Double check: "The graph may be disconnected."
Logic handles disconnected graphs correctly because adjacency lists just won't contain connections.
Double check: "1 <= u, v <= n, u != v".
Handled.
Double check: "no multiple edges or self-loops".
Handled (logic doesn't depend on uniqueness, but input respects it).

Seems ready.

One last thought: Is it better to check `same > diff` or `same >= diff`?
Strict inequality `>` ensures strictly increasing cut size, preventing cycles.
If `same == diff`, flipping yields 0 gain. Sticking avoids wasted moves (though doesn't hurt much). Strict is safer.

One potential issue:
In `current_cuts += (same_neighbors - diff_neighbors);`
If `same > diff`, gain is positive.
Correct.

Wait, `s` is 1-based, size `n+1`.
Loop `1` to `n`.
Input indices 1-based.
Matches.

```cpp
// Final Code Assembly
```
I will remove comments to make it cleaner or keep them for clarity?
"Response should ONLY contain the C++ code". Comments are fine.
Just wrap in blocks.

One minor point: `num_starts`. 20 is good.
`max_passes`. 100 is good.
This is a solid heuristic implementation.

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> best_s(n + 1);
    int max_cuts = -1;

    unsigned seed = chrono::steady_clock::now().time_since_epoch().count();
    mt19937 rng(seed);
    uniform_int_distribution<> dis(0, 1);

    const int NUM_RESTARTS = 20;
    const int MAX_PASSES = 100;

    for (int r = 0; r < NUM_RESTARTS; ++r) {
        vector<int> s(n + 1);
        for(int i = 1; i <= n; ++i) {
            s[i] = dis(rng);
        }

        // Initial cut count
        int current_cuts = 0;
        for(int i = 1; i <= n; ++i) {
            for(int neighbor : adj[i]) {
                if(i < neighbor && s[i] != s[neighbor]) {
                    current_cuts++;
                }
            }
        }

        bool improved = true;
        int pass_count = 0;

        while (improved && pass_count < MAX_PASSES) {
            improved = false;
            pass_count++;
            
            for(int i = 1; i <= n; ++i) {
                int same_neighbors = 0;
                int diff_neighbors = 0;

                for(int neighbor : adj[i]) {
                    if(s[neighbor] == s[i]) same_neighbors++;
                    else diff_neighbors++;
                }

                if (same_neighbors > diff_neighbors) {
                    s[i] = 1 - s[i];
                    current_cuts += (same_neighbors - diff_neighbors);
                    improved = true;
                }
            }
        }

        // Verify cuts
        int cuts = 0;
        for(int i = 1; i <= n; ++i) {
            for(int neighbor : adj[i]) {
                if(i < neighbor && s[i] != s[neighbor]) {
                    cuts++;
                }
            }
        }
        
        if(cuts > max_cuts) {
            max_cuts = cuts;
            best_s = s;
        }
    }

    for(int i = 1; i <= n; ++i) {
        cout << best_s[i] << (i == n ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
Looks perfect.

One check: "s_i = 0 means vertex i is in set S, 1 means T".
We print 0 or 1.
Correct.

Wait, I should check if there's any overflow for `current_cuts`?
$m \le 20000$, fits in `int`.
`n \le 1000`.
Fits `int`.

Ready.

```cpp
/*
    Max-Cut Heuristic Solution
    Uses Local Search (Hill Climbing) with Random Restarts.
    Time Complexity: O(Restarts * Passes * (N + M))
    Space Complexity: O(N + M)
*/
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> best_s(n + 1);
    int max_cuts = -1;

    unsigned seed = chrono::steady_clock::now().time_since_epoch().count();
    mt19937 rng(seed);
    uniform_int_distribution<> dis(0, 1);

    // Parameters for Local Search
    const int NUM_RESTARTS = 20;
    const int MAX_PASSES = 100;

    for (int r = 0; r < NUM_RESTARTS; ++r) {
        vector<int> s(n + 1);
        // Initialize with random partition
        for(int i = 1; i <= n; ++i) {
            s[i] = dis(rng);
        }

        int current_cuts = 0;
        for(int i = 1; i <= n; ++i) {
            for(int neighbor : adj[i]) {
                if(i < neighbor && s[i] != s[neighbor]) {
                    current_cuts++;
                }
            }
        }

        bool improved = true;
        int pass_count = 0;

        while (improved && pass_count < MAX_PASSES) {
            improved = false;
            pass_count++;
            
            for(int i = 1; i <= n; ++i) {
                int same_neighbors = 0;
                int diff_neighbors = 0;

                for(int neighbor : adj[i]) {
                    if(s[neighbor] == s[i]) same_neighbors++;
                    else diff_neighbors++;
                }

                // If flipping node i increases the cut edges
                if (same_neighbors > diff_neighbors) {
                    s[i] = 1 - s[i];
                    current_cuts += (same_neighbors - diff_neighbors);
                    improved = true;
                }
            }
        }

        // Recalculate cuts to be safe/consistent
        int cuts = 0;
        for(int i = 1; i <= n; ++i) {
            for(int neighbor : adj[i]) {
                if(i < neighbor && s[i] != s[neighbor]) {
                    cuts++;
                }
            }
        }
        
        if(cuts > max_cuts) {
            max_cuts = cuts;
            best_s = s;
        }
    }

    for(int i = 1; i <= n; ++i) {
        cout << best_s[i] << (i == n ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
</think>

```cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

int main() {
    // Optimize standard I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    // Adjacency list to store the graph
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Variable to store the best partition found
    vector<int> best_s(n + 1);
    int max_cuts = -1;

    // Setup Random Number Generator for initialization and restarts
    unsigned seed = chrono::steady_clock::now().time_since_epoch().count();
    mt19937 rng(seed);
    uniform_int_distribution<> dis(0, 1);

    // Local Search Configuration
    // Multiple random restarts help escape local optima
    const int NUM_RESTARTS = 20; 
    // Number of passes allowed per restart
    const int MAX_PASSES = 100; 

    for (int r = 0; r < NUM_RESTARTS; ++r) {
        vector<int> s(n + 1);
        // Initialize partition randomly
        for(int i = 1; i <= n; ++i) {
            s[i] = dis(rng);
        }

        // Calculate initial number of cut edges
        int current_cuts = 0;
        for(int i = 1; i <= n; ++i) {
            for(int neighbor : adj[i]) {
                if(i < neighbor && s[i] != s[neighbor]) {
                    current_cuts++;
                }
            }
        }

        bool improved = true;
        int pass_count = 0;

        // Perform Local Search (Hill Climbing)
        while (improved && pass_count < MAX_PASSES) {
            improved = false;
            pass_count++;
            
            // Iterate through vertices and try to flip them if it improves the cut
            for(int i = 1; i <= n; ++i) {
                int same_neighbors = 0;
                int diff_neighbors = 0;

                for(int neighbor : adj[i]) {
                    if(s[neighbor] == s[i]) same_neighbors++;
                    else diff_neighbors++;
                }

                // If we flip s[i]:
                // Edges to 'same_neighbors' (currently not cut) become cut.
                // Edges to 'diff_neighbors' (currently cut) stop being cut.
                // Net gain = same_neighbors - diff_neighbors.
                // We want gain > 0, so same_neighbors > diff_neighbors.
                if (same_neighbors > diff_neighbors) {
                    s[i] = 1 - s[i];
                    current_cuts += (same_neighbors - diff_neighbors);
                    improved = true;
                }
            }
        }

        // Final verification of cut size for this configuration
        int cuts = 0;
        for(int i = 1; i <= n; ++i) {
            for(int neighbor : adj[i]) {
                if(i < neighbor && s[i] != s[neighbor]) {
                    cuts++;
                }
            }
        }
        
        // Update global best if this configuration is superior
        if(cuts > max_cuts) {
            max_cuts = cuts;
            best_s = s;
        }
    }

    // Output the partition corresponding to the best solution found
    for(int i = 1; i <= n; ++i) {
        cout << best_s[i] << (i == n ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=987)

````
I need to solve the Max-Cut problem.

## Key Observations:

1. **Max-Cut is NP-hard in general graphs** - so I can't use a simple polynomial-time algorithm
2. Since n ≤ 1000, even though it's NP-hard, this constraint suggests either:
   - A randomized approach (which works well in practice)
   - An efficient approximation that runs quickly

## Best Approach: Randomized Greedy + Local Search

### Method 1: Simple Random Greedy (Good baseline)
- Assign each vertex to set 0 or 1 randomly
- Time: O(n²) per trial
- With many trials, gets close to optimal on average

### Method 2: Local Search (Better) - What I'll implement:
After an initial assignment, try to flip vertices to improve the cut.

**Greedy Flipping Strategy:**
1. For each unvisited vertex, find its neighbors in different sets vs same set
2. If more neighbors are in the opposite set than the current set, flip it to gain edges
3. Repeat until no improvement possible

Then run multiple trials and pick the best result.

## Implementation Details:

```cpp
int adj[MAXN][MAXN]; // adjacency matrix for O(1) edge lookup
int diff[MAXN], same[MAXN]; // count of neighbors in different/same set


```

For each node, I track how many neighbors would cross the partition if flipped. The greedy flip rule says: only flip a vertex when it has more neighbors across the cut than within it. I apply these flips repeatedly to refine the initial random assignment. Then I repeat the whole process with fresh random assignments several times, keeping whichever gives the largest cut. Finally, I output which vertices ended up in each partition.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

const int MAXN = 1005;
int adj[MAXN][MAXN];
int diff[MAXN], same[MAXN];
int assignment[MAXN];
int n, m;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n >> m)) return 0;

    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        --u; --v;
        adj[u][v] = adj[v][u] = 1;
    }

    int bestCut = 0;
    vector<int> bestAssignment(n);

    int trials = 30;
    for (int t = 0; t < trials; t++) {
        memset(adj, 0, sizeof(adj));
        memset(diff, 0, sizeof(diff));
        memset(same, 0, sizeof(same));
        memset(assignment, 0, sizeof(assignment));

        // Random initialization
        for (int i = 0; i < n; i++) {
            assignment[i] = rand() % 2;
        }

        // Compute initial diff and same counts
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (adj[i][j]) {
                    if (assignment[i] != assignment[j]) diff[i]++;
                    else same[i]++;
                }
            }
        }

        // Greedy flipping
        bool improved = true;
        while (improved) {
            improved = false;
            for (int i = 0; i < n; i++) {
                if (diff[i] > same[i]) {
                    assignment[i] ^= 1;
                    improved = true;
                }
            }
        }

        int currentCut = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (adj[i][j] && assignment[i] != assignment[j]) {
                    currentCut++;
                }
            }
        }

        if (currentCut > bestCut) {
            bestCut = currentCut;
            bestAssignment = assignment;
        }
    }

    for (int i = 0; i < n; i++) {
        cout << bestAssignment[i] << (i == n - 1 ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
````
