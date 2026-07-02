# 题 174 — base均分 72.6 vs 直接SFT均分 1.9(掉点 70.8)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 79.6,completion_tokens=18725)

````
The user wants a C++ solution for a graph 3-coloring problem where the goal is to minimize conflicting edges (edges with both endpoints having the same color).
The constraints are $n \le 1000$ and $m \le n(n-1)/2$.

This problem asks to assign colors from $\{1, 2, 3\}$ to vertices such that the number of monochromatic edges is minimized.
Since we want to minimize conflicting edges, this is equivalent to maximizing non-conflicting edges.
Actually, minimizing monochromatic edges isn't necessarily related to finding a valid 3-coloring (which has 0 conflicts), but rather finding an assignment that gets "close" to being valid if that's impossible. Since we can always just pick one color for everyone (all conflicts = m) or try to spread them out.
Wait, standard 3-coloring finds a partition into 3 independent sets. If the graph is not 3-colorable, there will be some conflicts. However, usually, we might just be allowed any coloring. The problem statement says "minimize the number of conflicting edges". It does not say we must find a valid 3-coloring if one exists. It implies we should output *any* assignment, but the one that minimizes conflicts gives the best score. Wait, looking at the scoring: Score = $1 - b/m$. To maximize score, we minimize $b$. This means we need the minimum number of monochromatic edges possible given 3 colors.

However, finding the absolute minimum number of monochromatic edges in a general graph with k colors is NP-hard (Minimizing Monochromatic Edges). With $N=1000$, exact optimization is impossible. This suggests there might be a specific structure or a heuristic/ randomized approach expected, OR the constraints allow something else.
Actually, wait. Is this a known problem? Minimizing monochromatic edges with $k$ colors.
If $k=3$, $N=1000$.
Maybe there is a specific property I'm missing, or it expects a local search / greedy approach / flow-based approximation?
Usually, for competitive programming problems with $N=1000$ and a coloring objective like this, either:
1. It's solvable via matching/max-flow (unlikely for general graphs without special structure).
2. It requires a randomized greedy approach (like simulated annealing or random restarts).
3. There is a mathematical bound or simple construction.

Let's re-read carefully. "Your task is to assign each vertex one of three colors... minimize the number of conflicting edges."
This is effectively the Multiway Cut problem variant or Min-Uncut problem, but specifically 3-coloring.
Actually, if we treat each color as an independent set, we want to partition vertices into $V_1, V_2, V_3$ to minimize edges within $V_i$.
This looks like a variation of the Quadratic Assignment Problem or similar clustering.
But wait, $N=1000$ suggests $O(N^2)$ or $O(N^2 \log N)$ is acceptable. Maybe even $O(M \cdot N)$.
Given the time limit usually associated with such problems (typically 1-2 seconds), and the nature of the problem (NP-hard generally), a pure exhaustive search is out.
However, sometimes these problems have weak constraints on the score, or specific test cases. Or perhaps the intended solution is much simpler.
Is it possible the question implies we just need *a* valid coloring if one exists, and otherwise anything goes? No, it says minimize conflicting edges.
Let's check the complexity. Finding the optimal 3-partition to minimize internal edges is hard.
However, maybe we can use local search.
Initial assignment: Randomly assign colors.
Improvement strategy: Move a vertex from one color class to another if it reduces conflicts.
Since there are only 3 colors, we can check all moves.
For a vertex $u$, let current color be $c$. We can move it to $c' \in \{1, 2, 3\} \setminus \{c\}$.
We calculate the change in conflicts.
Conflicts involving $u$:
Edges $(u, v)$ where $color(v) == color(u)$.
If $u$ moves from $C_{old}$ to $C_{new}$:
Decrease conflicts by count of neighbors in $C_{old}$.
Increase conflicts by count of neighbors in $C_{new}$.
Net change = Neighbors in $C_{new}$ - Neighbors in $C_{old}$.
If change < 0, update color.

This is a local search (Hill Climbing) approach.
It might get stuck in local optima. To handle this, we can run multiple random restarts or use Simulated Annealing.
Given the constraints and the nature of the request ("Generate solution code"), a randomized greedy/local search is often the standard practical solution for such CSPs when $N$ is small enough for $O(M)$ checks but large for exponential DP.
With $N=1000$, doing $O(M)$ per step is fine if total steps aren't huge.
Let's consider the maximum conflicts. At worst $M$.
Local search iterations: if we swap vertices, max conflict reduction is bounded.
Simulated Annealing parameters: temperature decay.
Or simply random shuffling + hill climbing until convergence, repeated several times.

Algorithm Plan:
1. Read input. Build adjacency list. Store edge counts per color class.
2. Function `calc_conflicts()` computes current total conflicting edges.
   Actually, maintaining the count is better. When moving a vertex, update the conflict count incrementally.
3. Initialize colors randomly.
4. Loop for a fixed number of iterations or until stable:
   Iterate over all vertices $u$.
   Try changing $color[u]$ to other 2 colors.
   Pick the one that strictly decreases conflicts.
   If no move decreases conflicts, this state is a local optimum. Break.
5. Repeat step 3-4 multiple times (random restarts) to avoid bad local optima.
6. Output the configuration with minimum conflicts found across restarts.

Complexity:
Each iteration scans all vertices and neighbors -> $O(N+M)$.
Number of restarts: say 10-20.
Convergence speed: hopefully fast.
Total complexity roughly $K \times S \times (N+M)$ where $S$ is iterations.
With $N=1000, M \approx 500,000$, $O(M)$ is okay for reasonable number of steps.
Since we just need to output the result, getting close to global optimum is likely sufficient. Usually competitive programming problems of this sort accept solutions that don't guarantee optimality if they are heuristics, OR the test cases are such that greedy works well. But given "minimize", usually exact is preferred if possible. However, 3-coloring min-mono-edge is APX-hard. So heuristic is necessary.

Wait, is there a simpler constraint or interpretation?
Maybe bipartite matching?
If the graph is bipartite, 2-coloring yields 0 conflicts.
If it's not, 3-coloring allows breaking cycles.
Actually, if the chromatic number $\chi(G) \le 3$, then answer is 0.
Determining $\chi(G) \le 3$ is easy? Yes, check if triangle-free? No, triangle implies 3 colors needed (but still valid 3-coloring exists).
Non-bipartite means odd cycles exist.
If $G$ contains triangles, 3-coloring might still be perfect.
Wait, $\chi(G) \le 3$ is checkable efficiently using greedy with backtracking (since $N=1000$). If $\chi(G) \le 3$, then min conflicts = 0.
If $\chi(G) > 3$, we can't achieve 0. Then we need to minimize.
Checking if $\chi(G) \le 3$: We can run a BFS/DFS. Since $\chi \le 3$, the width of the graph is small.
Actually, checking 3-colorability is NP-complete.
So we cannot assume 0 conflicts is achievable.
Thus, the local search heuristic is the way to go.

Refined Algorithm:
1. Parse Graph.
2. Implement `solve` function using randomized local search.
   - `colors` array initialized to random values {1, 2, 3}.
   - Compute initial conflicts.
   - `best_colors`, `best_conflicts`.
   - Loop for e.g., 5000 generations? Or just while improving.
   - In each generation, iterate all nodes. For node $u$, find neighbor counts in current colors.
     - Let $deg[u][c]$ be number of neighbors of $u$ having color $c$.
     - If we switch from $c_{curr}$ to $c_{new}$:
       - change = $deg[u][c_{new}] - deg[u][c_{curr}]$
       - if change < 0, apply change and update total conflict count.
   - After trying to improve all nodes once (local search pass), repeat multiple times with different initial random assignments.

Optimization:
Count neighbors per color efficiently.
`vector<int> adj[N]`
To compute degree in specific color quickly:
Iterate neighbors: `if (adj[v][k] == color[v]) ...`
This takes $O(\text{degree})$. Summing over all nodes is $O(2M)$.
Updating global count takes $O(1)$ if we know how many neighbors were affected.
Specifically:
Current conflicts $B$.
Move $u$ from $c$ to $c'$.
$u$ had neighbors in $c$ (conflict contribution). These become non-conflicts.
$u$ has neighbors in $c'$ (non-conflict before). These become conflicts.
Delta = (neighbors of $u$ with color $c'$) - (neighbors of $u$ with color $c$).
We need to maintain count of neighbors per color for each vertex.
`cnt[u][c]`: number of neighbors of $u$ with color $c$.
When moving $u$:
Remove old color $c$:
`B -= cnt[u][c]`
`cnt[u][c']++`
Wait, this changes `cnt` for $u$ and also affects `cnt` of its neighbors!
Updating `cnt` for all neighbors of $u$ is slow ($O(\text{deg}(u))$).
Total updates per full scan: $\sum \text{deg}(u) = 2M$.
So calculating `cnt` naively every step is $O(M)$.
If we do many passes, $O(M \cdot \text{passes})$.
Is $M$ up to $5 \cdot 10^5$? Yes.
If we do 100 passes, $5 \cdot 10^7$ ops. Fine.
So we can just maintain `adj` and recalculate deltas or use arrays for degrees.
Wait, if we update `cnt` for $u$ immediately upon change, subsequent checks for neighbors of $u$ need updated info.
So:
Maintain `cnt[u][color]` for all $u, color$. Initially $O(M)$.
When switching $u$ from $c_1$ to $c_2$:
  Update `cnt[u][c_1]--`
  Update `cnt[u][c_2]++`
  For each neighbor $v$ of $u$:
    Update `cnt[v][c_1]` (decrease? No, $u$ changed from $c_1$ to $c_2$)
    So `cnt[v][c_1]--`
    `cnt[v][c_2]++`
    And also `total_conflicts` adjustment logic.
  Total update cost is $O(\text{deg}(u))$.
  Iterating all nodes once: $O(2M)$.
  Multiple rounds: $O(2M \times \text{rounds})$.
  Rounds needed? Maybe 100. Total operations $\approx 100 \times 10^6 = 10^8$. Might be tight for 1 second in Python, but C++ should handle $10^8$ easily in 0.5s.
  Let's optimize: Just iterate neighbors once.
  Actually, we don't need `cnt[v]` to store neighbor counts for all colors permanently.
  Wait, to decide whether moving $u$ improves the score, we need `cnt[u][current_color]` and `cnt[u][other_color]`.
  So we DO need `cnt[u][c]` maintained globally.
  
  Structure:
  `vector<int> adj[1005];`
  `int c[1005];` // current color
  `int deg[1005][4];` // number of neighbors with specific color
  
  Initialization:
  For each u:
    For each v in adj[u]:
      deg[u][c[v]]++;
  
  Update function `move(u, new_c)`:
    old_c = c[u];
    if (new_c == old_c) return;
    
    // Conflict change calculation
    // u's conflict contribution = deg[u][old_c]
    // new contribution = deg[u][new_c]
    // delta = deg[u][new_c] - deg[u][old_c]
    
    // Apply move locally to 'deg' array for correctness
    deg[u][old_c]--;
    deg[u][new_c]++;
    
    // Now update neighbors' deg arrays because u's color changed
    // This changes their perspective of u.
    // For each v in adj[u]:
    //   deg[v][old_c]--;
    //   deg[v][new_c]++;
    //   // Also track conflicts change for those neighbors? 
    //   // The TOTAL conflict count needs to be tracked carefully.
    //   // Total conflicts B = sum(deg[u][old_c] for all u where c[u] == old_c?)
    //   // No. An edge (u, v) contributes 1 to B if c[u] == c[v].
    //   // Total B = (1/2) * sum_{u} (number of neighbors v s.t. c[v] == c[u])
    //           = (1/2) * sum_{u} deg[u][c[u]].
    
    // So we just need to maintain B correctly.
    // Before move:
    // B_new_local = sum(neighbors of u with color new_c)
    // B_old_local = sum(neighbors of u with color old_c)
    // Delta for u = B_new_local - B_old_local
    
    // Wait, the neighbors' deg arrays reflect neighbors' view of U.
    // When U changes, the conflict status of edges (U, V) might flip or stay same.
    // Specifically, for a specific neighbor V:
    // Old status: was it conflict? (c[V] == old_c)
    // New status: is it conflict? (c[V] == new_c)
    // Change for edge (U, V):
    // if c[V] == old_c: old contrib 1, new contrib 0 -> -1
    // if c[V] == new_c: old contrib 0, new contrib 1 -> +1
    // if c[V] != old_c and != new_c: 0 change.
    // This depends on V's color, which we know.
    
    // Optimization:
    // Total Conflicts B = sum_{u} (deg[u][c[u]]) / 2
    // Because each edge counted twice.
    // When moving u from old_c to new_c:
    // Term (u, c[u]) becomes:
    // remove deg[u][old_c], add deg[u][new_c].
    // Change in sum = deg[u][new_c] - deg[u][old_c].
    // Wait, is that sufficient?
    // Yes! Because sum_{u} deg[u][c[u]] sums exactly twice every monochromatic edge.
    // If we change c[u] from old_c to new_c:
    // The term deg[u][c[u]] changes.
    // Before: contributed deg[u][old_c].
    // After: contributes deg[u][new_c].
    // Change = deg[u][new_c] - deg[u][old_c].
    // We also need to update the `deg` array so future queries use new colors.
    // BUT `deg[u][x]` stores neighbors of u with color x.
    // Does this depend on u's color? No.
    // `deg[u][c]` is purely about colors of u's neighbors.
    // So when we change c[u], `deg[u][...]` doesn't change!
    // ONLY `deg[v][...]` for neighbors v changes.
    // Wait, `deg[u][c]` means neighbors of u having color c.
    // If I change color of u, my own degree array `deg[u]` (neighbor counts) does NOT change.
    // Only the `deg[v]` entries where v is neighbor of u change.
    // Correct.
    
    // So to maintain data structures efficiently:
    // We MUST update `deg[v]` for all neighbors. Cost O(deg(u)).
    // To maintain Total Conflicts B:
    // Current B = sum_{u} deg[u][c[u]] / 2.
    // Change = (new contribution of u) - (old contribution of u) ?
    // Yes. Contribution of u to the sum is deg[u][c[u]].
    // Note that this double counts edges.
    // Example: Edge (u, v) with c[u]=1, c[v]=1.
    // deg[u][1] includes v. deg[v][1] includes u.
    // Sum += deg[u][c[u]] + deg[v][c[v]] = 1 + 1 = 2.
    // Divide by 2 -> 1 conflict.
    // If we move u from 1 to 2. c[u]=2.
    // deg[u][2] (neighbors with color 2).
    // deg[u][1] (neighbors with color 1).
    // Change in u's term: deg[u][2] - deg[u][1].
    // Does this affect the sum?
    // Yes. But we also need to update `deg` for neighbors because `deg[v][old]` etc.
    // Wait, `deg[u][1]` stays the same physically?
    // No! If u changes color to 2, u is now a neighbor of v with color 2.
    // So for any v, `deg[v][2]` increases (if v was not 2? No if u becomes 2).
    // v's perspective of u changes.
    // So `deg[v]` MUST be updated.
    // So updating data structures takes O(deg(u)).
    // Updating Total Conflict Score can be done incrementally if we know B.
    // But since we are updating `deg` arrays anyway, we can just recompute B fully after stabilization or track it.
    // Given N=1000, recomputing B is O(N) or O(M)?
    // Recomputing B: Iterate all u, sum deg[u][c[u]].
    // That is O(N). Very fast.
    // So inside the loop:
    //   Find best move for u.
    //   Commit move.
    //   Update neighbors `deg`.
    //   Recalculate B (optional, or keep track).
    // Since we need to iterate all nodes to see if we are stuck, doing O(N) B-calc is negligible.
    // Wait, recomputing B is O(N) but we have to access `deg` which depends on `c`.
    // Accessing `deg[u][c[u]]` is O(1).
    // Summing over N nodes is O(N).
    // So:
    // Pass over nodes.
    // Check best move for u.
    // If better, execute, update `deg` for neighbors.
    // After a pass, check if B improved? No, we just need the final B.
    // Wait, if we update `deg` for neighbors mid-pass, subsequent nodes' checks will be consistent with the state.
    // So yes, just loop through all u, try to move.
    // Run this until no moves improve B.
    // Repeat K times.

One detail: $M=0$ case handled naturally (score 1). Code should handle this.
Output format: space separated integers.

Implementation details:
Use `std::vector<int> adj[MAXN]`
`int c[MAXN]`
`int deg[MAXN][4]` (indices 1,2,3)
`memset` deg.
Init `deg`: loop u, loop v in adj[u], deg[u][c[v]]++.
Loop restarts:
  Init `c` random.
  Rebuild `deg`.
  Local Search:
    Loop until no improvement in whole pass.
    Or just loop fixed times (e.g. 1000 passes). Fixed is safer against infinite loops if not careful, though improvements decrease monotonically here. Since finite states, it terminates. But to ensure speed, cap at MAX_ITERATIONS.
  Keep track of Global Minimum Conflicts and corresponding colors.
  Output Global Minimum.

Corner cases:
$n=1$: Output 1.
$m=0$: Any output 1 1 ... 1.
Disconncted components: Handled automatically.
Max $N=1000, M \approx 500,000$.
Memory: `adj` takes ~4MB. `deg` takes 4MB. Safe.
Time: Heuristic.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <ctime>

using namespace std;

const int MAXN = 1005;
const int MAXM = 500005; // Max edges approx 5*10^5

vector<int> adj[MAXN];
int c[MAXN]; // Color for each vertex, 1-based
int deg[MAXN][4]; // Number of neighbors of i with color j (j=1..3)
int n, m;
int total_conflicts;

// Calculate current total conflicts based on c and deg
// Since we update deg consistently, we can trust B if we maintain it, 
// but let's verify or just maintain B variable manually to save recomputation time?
// Recomputing B takes O(N), which is fast.
// Actually, B = (Sum_{i} deg[i][c[i]]) / 2.
// We can just use this formula.

void init_degrees() {
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j < 4; ++j) deg[i][j] = 0;
        for (int neighbor : adj[i]) {
            deg[i][c[neighbor]]++;
        }
    }
    // Initial B calculation
    total_conflicts = 0;
    for (int i = 1; i <= n; ++i) {
        total_conflicts += deg[i][c[i]];
    }
    total_conflicts /= 2;
}

bool solve_one_restart() {
    // We'll perform local search
    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 1; i <= n; ++i) {
            int current_color = c[i];
            
            // Try changing to other colors
            int best_change = 0; // minimal change is negative
            int best_new_color = -1;
            
            // Count neighbors with colors
            // Actually we already have deg[i][c]
            // But we need to update total_conflicts logic if we change.
            // If move i from old to new:
            // Change in (Sum of deg[k][c[k]]) is:
            // (deg[i][new_color] - deg[i][old_color]).
            // Why? Because term for i changes.
            // What about neighbors? They update their deg arrays too.
            // But B formula depends on sum of deg[i][c[i]] over all i.
            // So yes, delta is simply deg[i][new_color] - deg[i][old_color].
            // WAIT.
            // If I change c[i], I must update deg[j] for all neighbors j.
            // If I update deg[j], does it affect B?
            // B = Sum_{k} deg[k][c[k]].
            // If I change c[i], deg[i] doesn't change (it's neighbors of i).
            // But deg[j] for neighbors j DOES change because i's color changed relative to them.
            // Does B change due to deg[j]?
            // Yes. If c[j] == c[i], then deg[j][c[j]] includes i.
            // When c[i] changes, if c[i] becomes c[j], it adds to deg[j][c[j]], increasing B.
            // If c[i] becomes != c[j], it decreases B.
            // This means simply checking deg[i][new] - deg[i][old] is NOT sufficient for immediate B update if we maintain B via that sum.
            // We would need to account for ALL affected nodes j.
            // That defeats the purpose of O(1) update.
            // Wait, B = Number of edges (u,v) with c[u]==c[v].
            // When moving u from A to B:
            // For every neighbor v of u:
            //   If c[v] == A: edge (u,v) goes from conflicting to non-conflicting (-1 conflict).
            //   If c[v] == B: edge (u,v) goes from non-conflicting to conflicting (+1 conflict).
            //   If c[v] neither: no change.
            // So for vertex u:
            //   Reduction = count(neighbors with color A).
            //   Increase = count(neighbors with color B).
            //   Net Change = Increase - Reduction.
            // This is exactly what we need to check to improve.
            // This is exactly `deg[u][A] - deg[u][B]`. (Note signs: want net < 0).
            // So `change = deg[u][B] - deg[u][A]`. If change < 0, better.
            
            // So we just compare `deg[i][1]`, `deg[i][2]`, `deg[i][3]`.
            // Let's collect the counts of conflicts for each target color.
            // If current is 1. Options are 2 and 3.
            // Option 2 gain = deg[i][2]. (These edges were 0 conflict, become conflict).
            // Option 2 loss = deg[i][1]. (These edges were conflict, become 0).
            // Actually:
            // Current conflicts involving i is deg[i][c[i]].
            // If switch to c': conflicts involving i becomes deg[i][c'].
            // Wait, this `deg[i]` refers to neighbors of i.
            // If i switches to c', the edges (i, neighbor) where neighbor == c' become conflicts.
            // Where neighbor != c' become non-conflicts.
            // Wait, the definition of `deg[i][c]` is count of neighbors WITH color c.
            // If i switches from X to Y:
            // Neighbors with color X lose the conflict connection (they become non-conflicting).
            // Neighbors with color Y gain the conflict connection (they become conflicting).
            // Neighbors with color Z (Z!=X, Z!=Y) remain non-conflicting.
            // So Change in total conflicts = (count neighbors with Y) - (count neighbors with X).
            // Which is `deg[i][target] - deg[i][current]`.
            // This value is correct assuming we only care about edges incident to i.
            // Since edges are disjoint contributions?
            // Yes, changing color of i only affects edges incident to i.
            // So yes, this calculation is correct.
            
            // We need to actually perform the update.
            // If we pick a move, we must update global state for NEXT iterations.
            // State update:
            //   Update `deg` for i? No, `deg[i]` depends on neighbors' colors. Neighbors didn't change yet.
            //   But wait, i's color is `c[i]`.
            //   `deg[i][c]` counts neighbors with color c.
            //   This value `deg[i][c]` IS CONSTANT regardless of `c[i]`!
            //   It only changes if a NEIGHBOR changes color.
            //   So `deg[i]` does NOT change when `c[i]` changes.
            //   HOWEVER, neighbors' `deg` vectors WILL change.
            //   For each neighbor v: `deg[v][old_c]--`, `deg[v][new_c]++`.
            
            // So the logic holds.
            
            int current_cnt = deg[i][current_color];
            for (int next_c : {1, 2, 3}) {
                if (next_c == current_color) continue;
                int diff = deg[i][next_c] - current_cnt; // diff in conflicts involving i
                // If diff < 0, we reduce conflicts.
                if (diff < 0) {
                    // Is this the best single move?
                    // Greedy choice: pick ANY that reduces.
                    // Usually taking steepest descent or first available.
                    // Let's pick the one with smallest diff (most reduction).
                    // Wait, diff < 0. We want most negative.
                    // Better to pick one that minimizes diff.
                    if (diff < best_change || best_change >= 0) {
                        best_change = diff;
                        best_new_color = next_c;
                    }
                } else {
                    best_change = min(best_change, diff); // Ensure we record best even if no improvement
                    // Actually we only want to move if we can improve.
                }
            }
            // Wait, logic error in loop above.
            // Just pick the best among 1, 2, 3 excluding current.
            // If any is better than others (and better than current?), move.
            // But we are comparing relative to CURRENT conflict contribution.
            // The contribution is `deg[i][current_color]`.
            // New contribution would be `deg[i][new_color]`.
            // Change = `deg[i][new_color] - deg[i][current_color]`.
            // We want Change < 0.
            // Among options < 0, pick the most negative?
            // Or just pick the first one that is < 0? 
            // Picking most negative is better for convergence.
            
            best_new_color = -1;
            int min_diff = 0;
            
            for (int next_c : {1, 2, 3}) {
                 if (next_c == current_color) continue;
                 int d = deg[i][next_c] - current_cnt;
                 if (d < min_diff) {
                     min_diff = d;
                     best_new_color = next_c;
                 }
            }
            
            if (best_new_color != -1 && min_diff < 0) {
                // Make the move
                int old_c = current_color;
                c[i] = best_new_color;
                
                // Update deg[i]? No, deg[i] depends on neighbors. Neighbors haven't changed.
                // So deg[i] remains correct.
                // BUT update neighbors' deg.
                // Also update global conflict count.
                // But we just need to track best score.
                // We can recompute total_conflicts periodically or track it.
                // Given N=1000, tracking is hard with partial updates.
                // Let's track `total_conflicts` explicitly.
                // Wait, earlier I said `deg[i]` doesn't change.
                // If `deg[i]` doesn't change, then `deg[i][c[i]]` changes value index, but not memory?
                // If c[i] changes, `deg[i][c[i]]` accesses different entry.
                // Yes. `deg[i][new_c]` vs `deg[i][old_c]`.
                // The actual values in the array stay same (assuming neighbors haven't moved).
                // So we just read `deg[i][best_new_color]`.
                // But we must update `deg` of neighbors!
                // Wait, if neighbors don't change, their `deg` views of i change.
                // Correct.
                
                for (int neighbor : adj[i]) {
                    deg[neighbor][old_c]--;
                    deg[neighbor][best_new_color]++;
                }
                
                // Update global conflict count:
                // Conflict change was min_diff.
                // total_conflicts += min_diff;
                // Wait, min_diff = deg[i][new] - deg[i][old].
                // This accounts for edges incident to i.
                // Are we counting each edge twice in `total_conflicts`?
                // My initialization logic:
                // total_conflicts = sum(deg[i][c[i]]) / 2.
                // If I use `min_diff` derived from deg array which is symmetric?
                // `deg[i][c[v]]` counts neighbors of i with color c.
                // If I move i, `deg[i]` values (counts) do not change.
                // But `deg[neighbor][i's_color]` changes.
                // So if I simply sum `deg[i][c[i]]` later, it will be consistent.
                // So tracking `total_conflicts` incrementally is tricky if we rely on the sum formula, because `deg` array updates propagate.
                // BUT, notice: `deg[i][c[i]]` is the count of neighbors having same color.
                // This EXACTLY measures the number of monochromatic edges incident to i.
                // Sum of these over all i is 2 * total_conflicts.
                // When i moves:
                // `deg[i][c[i]]` changes value (index).
                // `deg[neighbors][c[neighbor]]` (where neighbor color matches i's OLD color) decreases?
                // Wait.
                // Example: edge (i, v). c[i]=A, c[v]=A.
                // `deg[i][A]` includes v. `deg[v][A]` includes i.
                // Move i to B.
                // `deg[i][A]` does NOT change? No.
                // `deg[i][A]` is count of neighbors with color A. v has color A.
                // So `deg[i][A]` count IS NOT affected by i's move.
                // Wait! `deg` array definition:
                // `deg[u][k]` = number of neighbors of u who have color k.
                // When u moves color from A to B.
                // `deg[u]` array contents remain unchanged?
                // NO. `deg[u]` array stores counts based on NEIGHBORS' colors.
                // Since neighbors' colors didn't change, `deg[u]` array IS UNCHANGED.
                // Wait, why did I think it needed update?
                // Because I thought `deg` was symmetric?
                // Ah, `deg[u][k]` tells me how many neighbors are color k.
                // If I am u, and I change my color.
                // `deg[u]` does not change because my neighbors are still the same people with same colors.
                // HOWEVER, for a neighbor v: `deg[v]` depends on colors of v's neighbors.
                // One of v's neighbors is u.
                // Since u's color changed, `deg[v]` MUST update.
                // Specifically, `deg[v][A]` (where A was u's old color) decreases.
                // `deg[v][B]` (where B is u's new color) increases.
                // So yes, we update neighbors.
                
                // Now, does `deg[i][c[i]]` change for i itself?
                // No, `deg[i]` is static with respect to i's color. It reflects neighbors.
                // BUT, in the formula Sum(deg[u][c[u]]), the term `deg[i][c[i]]` picks the value corresponding to current color.
                // Since c[i] changed, the term we SUM changes!
                // Before: term was `deg[i][A]`.
                // After: term is `deg[i][B]`.
                // Difference is `deg[i][B] - deg[i][A]`.
                // This difference represents the change in monochromatic edges incident to i.
                // Is it correct to just sum this?
                // Yes.
                // What about neighbors?
                // For neighbor v with c[v]=A:
                // Before: `deg[v][A]` included u.
                // After: `deg[v][A]` excludes u.
                // Term for v in sum: `deg[v][c[v]]` (since c[v]=A).
                // Before: contribution `deg[v][A]`. Included 1 (for u).
                // After: contribution `deg[v][A]` (updated, excludes u).
                // Difference -1.
                // Wait, if we rely on updating `deg` for neighbors, the sum formula will work perfectly.
                // We just need to update `deg` correctly.
                
                // So algorithm:
                // 1. `total_conflicts` maintained via formula `sum(deg[u][c[u]]) / 2`.
                //    Wait, computing sum is O(N). Doing it every move is O(N*M/N) = O(M).
                //    Too slow inside inner loop?
                //    We can just track delta.
                //    Delta from moving i:
                //    (term for i) change: `deg[i][B] - deg[i][A]`.
                //    For neighbors v:
                //      If c[v] == A: `deg[v][A]` decreases by 1.
                //        Term `deg[v][c[v]]` was `deg[v][A]`, now `deg[v][A]-1`. Delta -1.
                //      If c[v] == B: `deg[v][B]` increases by 1.
                //        Term `deg[v][c[v]]` was `deg[v][B]`, now `deg[v][B]+1`. Delta +1.
                //      Else: no change.
                //    So total delta in sum = (`deg[i][B] - deg[i][A]`) + (Neighbors A count) - (Neighbors B count).
                //    Wait.
                //    `deg[i][A]` counts neighbors with color A.
                //    So "Neighbors A count" is exactly `deg[i][A]`.
                //    "Neighbors B count" is exactly `deg[i][B]`.
                //    So Delta in sum = `deg[i][B] - deg[i][A]` + `deg[i][A]` - `deg[i][B]` = 0?
                //    Something is wrong.
                //    Ah, `deg[i][A]` is static.
                //    `deg[v][A]` changes.
                //    The sum includes terms for ALL vertices.
                //    For neighbors v with c[v]=A: `deg[v][A]` decreases by 1.
                //    This reduces the SUM by `deg[v][A] - (deg[v][A] - 1)`? No.
                //    Term is `deg[v][c[v]]`. c[v] is fixed.
                //    So if v's `deg[v][c[v]]` decreases by 1, the sum decreases by 1.
                //    Total reduction from all neighbors with A = number of neighbors with A = `deg[i][A]`.
                //    Similarly, increase from all neighbors with B = `deg[i][B]`.
                //    Term change for i itself:
                //    Before: `deg[i][A]`. After: `deg[i][B]`.
                //    Change: `deg[i][B] - deg[i][A]`.
                //    Total Change in Sum = (Term i) + (Sum changes for neighbors).
                //    = `(deg[i][B] - deg[i][A])` + `(Change in deg[v][c[v]] summed over v)`.
                //    Neighbors with color A: their `deg[v][A]` drops by 1. Number of such neighbors is `deg[i][A]`. Total drop = `deg[i][A]`.
                //    Neighbors with color B: their `deg[v][B]` rises by 1. Number of such neighbors is `deg[i][B]`. Total rise = `deg[i][B]`.
                //    Total Sum Change = `deg[i][B] - deg[i][A]` + `deg[i][B] - deg[i][A]` = 2 * (`deg[i][B] - deg[i][A]`).
                //    So `total_conflicts` (which is Sum/2) changes by exactly `deg[i][B] - deg[i][A]`.
                //    So updating `total_conflicts += deg[i][B] - deg[i][A]` is sufficient!
                //    We DO NOT need to update neighbors' terms explicitly for the count, provided we update the `deg` arrays themselves for correctness in FUTURE steps.
                
                // Logic confirmed.
                // 1. Identify best move.
                // 2. Apply move.
                // 3. Update `deg` arrays for neighbors.
                // 4. Update `total_conflicts` by adding `diff`.
                
                // Note: Best move logic uses `deg[i]` which hasn't been updated yet in this step?
                // If we modify `deg` after deciding, the `deg` used for decision is consistent with PRE-move state.
                // But wait, if we do sequential passes, previous moves affect current `deg`.
                // So we MUST update `deg` immediately after committing.
                // My loop:
                // Iterate i.
                // Decide best move based on current `deg` (which includes info from all PREVIOUS moves in this pass).
                // Commit move.
                // Update `deg` for neighbors (this reflects current move).
                // Update `total_conflicts`.
                // Next i uses updated `deg`. This is valid Hill Climbing.
                
                // Optimization:
                // Instead of checking all 1, 2, 3 for every node, check if ANY improves.
                // Since we want to reach a local optimum, finding ANY improvement is good.
                // But greedy (steepest) converges faster usually.
                // With N=1000, iterating all nodes is fast.
            }
        }
    }
    return true;
}
```

One small detail: Constraints $N \le 1000$.
Max conflicts = $M$.
Time complexity per restart:
Inner loop: iterate $i=1..N$.
Inside: check 2 candidates. Constant time.
Update neighbors: proportional to degree sum = $O(M)$.
Wait, updating neighbors is done ONLY IF a move is made.
In early stages of random restarts, many moves happen.
In later stages, few moves.
Worst case: every vertex moves?
Then $O(M)$ update cost.
Number of outer loops (improvement loops): potentially many.
However, each move reduces conflict count by at least 1 (if integer).
Min conflicts $\ge 0$.
So max moves = $M$.
Total complexity $O(M \cdot (\text{something}))$.
If we do $O(M)$ work per move, total $O(M^2)$?
$M \approx 5 \cdot 10^5$. $M^2 = 25 \cdot 10^{10}$. Too slow.
Wait.
We should NOT update neighbors for EVERY potential move or just every move?
If we make a move, we update neighbors.
If we make a move, we pay $O(\deg(i))$.
Sum of $\deg(i)$ over moves = $\sum_{u} \text{moves}_u \times \deg(u)$.
If every node moves many times, it could be heavy.
However, we stop when no move is possible (local optimum).
Does local search converge fast?
Usually yes.
Can we optimize the update?
We really need `deg` for correctness.
Is there a way to avoid $O(M)$ updates?
Notice `deg` array size is small.
Maybe just re-scan `deg`? No.
What if we don't maintain `deg` perfectly but recompute B?
Computing B is $O(N)$.
Checking candidate moves:
We need `deg[i][c]`. If we don't maintain `deg` accurately (or maintain partially), we can't evaluate delta in $O(1)$.
Unless we precalculate.
Recomputing `deg` from scratch costs $O(M)$. Same issue.
Wait, `deg[i][c]` is neighbor count.
If we don't update `deg` for neighbors, it's stale.
Is there a middle ground?
Maybe just run fewer passes?
For CP problems like this, typically $N=1000$ with local search is accepted if constant factor is low.
Also, the number of successful moves is not necessarily equal to $M$. Often converges quickly.
And we can use a timeout or iteration limit.
Let's assume this approach is sufficient for "competitive programmer" context unless specified "Guaranteed optimal".
Usually "minimize" in CP without time limit specified (standard limits) implies finding a good solution, not necessarily optimal for hard instances.
Given the constraints and problem type, randomized local search is the standard technique.
I will implement with a limit on restarts and iterations to prevent TLE.

Let's refine the loop:
Limit total iterations of the outer improvement loop per restart to e.g. 5000.
If no improvement in a sweep (loop over all N), break restart.
Run multiple restarts (e.g. 20).
Keep best result.

Code structure:
Includes.
Global vars.
Random seed.
Main loop.
Output.

One catch: `deg[i][c]` access.
Indices 1, 2, 3. Array size 4. OK.
Adjacency list is 0-indexed or 1-indexed? Problem says 1 to N. Use 1-indexed vector.

Double check the complexity concern.
If graph is dense ($M \approx 500,000$), updating neighbors for every swap is costly.
Swap cost $\sum \deg(u)$.
Is it possible to optimize?
We need `deg[i][color]` counts.
If we just maintain `deg` array, it's accurate.
But updating it takes time.
Actually, we only need `deg[i][c]` to determine move for `i`.
If we update `deg` lazily? No, too complex.
Is there a heuristic?
Since we only care about 3 colors, maybe we can approximate?
No, exactness matters for "minimize".
But practically, most moves happen at the beginning.
Once converged, no moves -> no updates.
Also, we only update `deg` when we commit.
If we don't commit, no update.
So cost is proportional to number of accepted moves.
How many accepted moves?
At most $M$ (since each move reduces B by $\ge 1$).
If $B$ reduces by average 0.5, then $2M$ moves.
Cost $2M \times$ avg_degree.
Avg degree $2M/N$.
Total cost $\approx 2M \times (2M/N) = 4 M^2 / N$.
$5 \cdot 10^5 \times 1000 \approx 5 \cdot 10^8$.
This is slightly high for 1s.
However, typical test cases won't be worst-case dense graphs for long runs.
Also, we can limit max iterations.
If $N=1000$ and we have 1-2 seconds, $10^8$ ops is fine in C++ (simple ops).
Updates involve array lookups and increments. Very fast.
Should be OK.

One more thing: `best_change` logic.
We want to minimize conflicts.
So `diff` should be checked `diff < 0`.
Among `diff < 0`, pick min `diff` (most negative).
Correct.

Edge case: $n=1$. Loop 1 to 1. Adj empty. Moves none. Output 1. Correct.
Case $m=0$. Output 1..1.

I'll wrap it in the requested format.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <ctime>

using namespace std;

// Maximum number of vertices as per constraints
const int MAXN = 1005;

vector<int> adj[MAXN];
int c[MAXN];              // Colors: 1, 2, 3
int deg[MAXN][4];         // deg[u][color] = number of neighbors of u with that color
int n, m;
int best_conflicts = 1000000000;
int best_colors[MAXN];

int calculate_current_conflicts(int current_conflicts) {
    // Not needed to recompute every time, just track delta
    return 0;
}

// Helper to initialize degrees
void init_degrees() {
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j < 4; ++j) deg[i][j] = 0;
        for (int neighbor : adj[i]) {
            deg[i][c[neighbor]]++;
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n >> m)) return 0;

    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    srand(time(0));

    // Run multiple restarts
    int num_restarts = 10; 
    
    // Total conflicts tracker
    int current_total_conflicts = 0;
    
    // Temporary buffer for current run
    // Actually we can reuse global variables with reset
    
    for (int restart = 0; restart < num_restarts; ++restart) {
        // Reset state
        for (int i = 1; i <= n; ++i) {
            adj[i].clear(); // Already cleared from prev? No, adj is persistent.
            // Wait, I should not clear adj inside loop as it's built once.
            // c needs reset. deg needs reset.
            // Actually adj is static for this run.
            // Just reset c and deg.
        }
        
        // Initialize random colors
        for (int i = 1; i <= n; ++i) c[i] = 1 + rand() % 3;
        
        // Build initial degrees
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j < 4; ++j) deg[i][j] = 0;
            for (int neighbor : adj[i]) {
                deg[i][c[neighbor]]++;
            }
        }
        
        // Calculate initial conflicts
        current_total_conflicts = 0;
        for (int i = 1; i <= n; ++i) {
            current_total_conflicts += deg[i][c[i]];
        }
        current_total_conflicts /= 2;
        
        // Local search
        bool improved = true;
        // Limit iterations to avoid TLE on stubborn graphs
        // For N=1000, 500 passes per restart is generous.
        int max_passes = 1000; 
        
        // We need a copy of c to revert? No, just track improvements.
        // Actually, if we improve, we proceed. If not, stop this run.
        // But we might want to restart from a state with 0 moves?
        // If no moves possible for a full pass, we are at local opt.
        
        int pass_count = 0;
        while (improved && pass_count < max_passes) {
            improved = false;
            pass_count++;
            for (int i = 1; i <= n; ++i) {
                int curr_c = c[i];
                
                // Find best move
                int best_move = -1;
                int best_delta = 0; // Positive means increase conflict
                
                // We want to reduce conflict.
                // Change = deg[i][new_c] - deg[i][curr_c]
                // We look for smallest (most negative) change.
                int current_conflicts_inc = deg[i][curr_c];
                
                int min_diff = 1000000000;
                
                for (int next_c : {1, 2, 3}) {
                    if (next_c == curr_c) continue;
                    int diff = deg[i][next_c] - current_conflicts_inc;
                    if (diff < min_diff) {
                        min_diff = diff;
                        best_move = next_c;
                    }
                }
                
                if (min_diff < 0) {
                    // Apply move
                    c[i] = best_move;
                    
                    // Update deg for neighbors
                    // Since c[i] changed from old (curr_c) to new (best_move)
                    // For all neighbors v:
                    // deg[v][curr_c]--, deg[v][best_move]++
                    // Note: accessing adj[i]
                    for (int neighbor : adj[i]) {
                        deg[neighbor][curr_c]--;
                        deg[neighbor][best_move]++;
                    }
                    
                    // Update global conflicts
                    // Each edge incident to i contributes 1 to sum of conficts.
                    // Change is min_diff (calculated based on i's incident edges).
                    // As derived earlier: Total Conflict Change = min_diff.
                    current_total_conflicts += min_diff;
                    
                    improved = true;
                    if (current_total_conflicts <= 0) break; // Shouldn't happen for m>0 but safe
                    
                }
            }
        }
        
        if (current_total_conflicts < best_conflicts) {
            best_conflicts = current_total_conflicts;
            for (int i = 1; i <= n; ++i) best_colors[i] = c[i];
        }
    }
    
    // Output
    for (int i = 1; i <= n; ++i) {
        cout << best_colors[i] << (i == n ? "" : " ");
    }
    cout << "\n";
    
    return 0;
}
```

Wait, `best_conflicts` initialization.
If $m=0$, conflicts=0.
My code initializes `best_conflicts` to infinity.
If no restart improves (all give conflicts > 0, though m=0 gives 0), we need to handle m=0 correctly.
If m=0, `current_total_conflicts` starts at 0. `best_conflicts` updates to 0.
Logic holds.
If $m>0$, conflicts could be initially $M$.
My code updates `best_conflicts` based on improvements found.
One issue: `current_total_conflicts` is computed at start of each restart.
If in a restart we end with conflicts = 5, but global best was 10, we update.
Correct.

Wait, check `min_diff` logic again.
If `deg[i][next_c] - current_conflicts_inc` is negative, it reduces conflict.
Is it guaranteed to be correct?
Yes, `min_diff` corresponds to change in `Sum(deg[u][c[u]]) / 2`?
Wait, derivation earlier: `total_conflicts` change is exactly `diff`?
Let's re-verify:
Change in Sum = `deg[i][B] - deg[i][A]` + `deg[i][A]` (neighbors A lose) + `- deg[i][B]` (neighbors B gain? No.)
Recalculating:
Sum of `deg[u][c[u]]` over all u is $2 \times B$.
When u moves from A to B:
1. Term for u: changes from `deg[u][A]` to `deg[u][B]`. Delta `deg[u][B] - deg[u][A]`.
2. Neighbors v with c[v]=A: Their term `deg[v][A]` decreases by 1 (since u is no longer color A neighbor).
   Number of such neighbors is `deg[u][A]`.
   Total Delta = `deg[u][A] * (-1)`.
3. Neighbors v with c[v]=B: Their term `deg[v][B]` increases by 1 (since u is now color B neighbor).
   Number of such neighbors is `deg[u][B]`.
   Total Delta = `deg[u][B] * (1)`.
   
Total Delta in Sum = `(deg[u][B] - deg[u][A])` + `(-deg[u][A])` + `(+deg[u][B])`
= `2 * deg[u][B] - 2 * deg[u][A]`
= `2 * (deg[u][B] - deg[u][A])`.
So Change in $2B = 2 * (deg[u][B] - deg[u][A])$.
Dividing by 2: Change in $B = deg[u][B] - deg[u][A]$.
So `min_diff` IS the correct delta for `current_total_conflicts`.
My code updates: `current_total_conflicts += min_diff`.
This seems mathematically correct.

Final check on complexity and types.
`int` is sufficient for $M \le 500,000$.
Arrays sized 1005.
Headers ok.
IO optimized.

One minor point: `adj` clear inside restart loop?
Ah, `adj` is populated in main before restarts.
Inside restart loop, I reset `deg` and `c`. `adj` stays as is.
Wait, `adj` contains the edges. They are constant for all restarts.
So clearing `adj` inside restart is WRONG (I commented it out).
But `adj` should persist?
Yes, `adj` is declared outside.
However, `adj` is filled ONCE.
My loop `for (int restart...)` just resets `c`, `deg`, and recomputes everything.
Correct.

One optimization:
The loop `for (int restart...)` repeats `num_restarts` times.
Inside, `max_passes` limits speed.
With 10 restarts and 1000 passes each, $10^4$ passes.
Pass 1000 does $O(N)$ scan?
Wait, `adj` iteration is part of `pass`.
If no move is made in a pass, we stop.
Wait, inside `pass` loop:
`if (min_diff < 0) { ... improved = true; }`
If `improved` is true, we continue passing.
If `improved` is false after scanning all `i`, we break while loop.
So `pass_count` increments only if we complete a scan?
No, `while(improved && pass_count < max_passes)`.
So we run full scan.
Total operations $\approx$ Passes $\times M$.
Wait, in each pass we scan all nodes. If move happens, we update neighbors.
Sum of degrees in one full pass is $2M$.
So one pass is $O(N + M)$.
If we run 1000 passes, total $1000(M+N) \approx 5 \cdot 10^8$.
This might be tight.
But `improved` becomes false quickly.
Average number of swaps is limited by $M$.
Most passes will finish with few/no moves? No, if no moves, we stop immediately.
Wait, if no moves in a pass, we break.
So we spend only a fraction of time if convergence is fast.
However, if it makes a swap near the beginning, cost is paid.
Worst case: makes swap every step, never settles.
But conflict count decreases. Can't oscillate infinitely.
So total accepted swaps $\le M$.
Total cost of updates = Sum(degree) per swap $\approx M^2 / N$.
As calculated before.
If $M=500,000$, $M^2/N \approx 2.5 \cdot 10^{10}$ is bad.
BUT, this is worst case assumption (every swap triggers expensive update).
Most swaps in real graph coloring reduce conflicts significantly or stop early.
Also, randomization helps escape local minima.
And competitive programming problems often have loose limits for heuristics.
I'll stick with this. It's the most standard approach for 3-coloring minimization.

Wait, is there a simpler way?
Maybe bitsets? `vector<int>` vs `bitset<1000>`.
`bitset` speeds up neighbor checks.
But `adj` stores indices.
Using bitsets for adjacency matrix would take $1000^2$ bits = 125KB. Fast.
Operations: row-wise AND/OR.
Conflict check: `row[u] & mask[c]`.
But building masks takes time.
Probably slower than vector scan for sparse graphs.
Problem has $m \le n(n-1)/2$. Could be dense.
With $N=1000$, matrix representation is feasible.
Bitwise operations on 1000 bits (array of 16 ints).
`cnt[u][c]` needs to count set bits.
`__builtin_popcount`.
Moving u from A to B:
If `row[v]` intersects `mask[A]` etc.
It's getting complicated to implement quickly.
Vector scan is safer and simpler.
Also $M$ is up to $5 \cdot 10^5$, which is relatively sparse compared to $1000^2/2 = 5 \cdot 10^5$. Wait.
$1000 \times 999 / 2 \approx 500,000$.
So $M$ is around max density.
Dense graph.
Vector scan is $O(M)$. Matrix operations $O(N)$.
Matrix ops might be better if implemented with bitsets.
But for 3 colors, we just need counts.
Let's stick to vector. The constant factor is very small (array access).
$2.5 \cdot 10^{10}$ is theoretical upper bound but unreachable.
Practically, convergence is logarithmic in number of vertices or similar.
The number of swaps is likely much less than $M$.
Also, we can stop if `current_total_conflicts` hits 0 (optimal).

```cpp
/*
 * Solution for Graph 3-Coloring
 * Strategy: Randomized Local Search (Hill Climbing)
 */
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

const int MAXN = 1005;
vector<int> adj[MAXN];
int c[MAXN];
int deg[MAXN][4];
int n, m;
int best_conflicts = 2000000000;
int best_colors[MAXN];

void init_and_solve() {
    // Read input
    if (scanf("%d %d", &n, &m) != 2) return;
    
    // Clear global structures
    for (int i = 0; i <= n; ++i) {
        adj[i].clear();
        for (int j = 1; j < 4; ++j) deg[i][j] = 0;
    }
    // m can be 0
    for (int i = 0; i < m; ++i) {
        int u, v;
        scanf("%d %d", &u, &v);
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    srand(time(0));
    int current_conflicts = 0;
    
    // Multiple restarts
    // If m=0, conflicts=0, output 1 1 ... 1
    if (m == 0) {
        for (int i = 1; i <= n; ++i) printf("1%c ", i==n ? "" : " "); // Wait, output format: space separated.
        printf("\n");
        return;
    }

    // Limit restarts
    int max_restarts = 20;
    
    // Use a helper array for colors in current run
    // Actually c is global, we overwrite it.
    // Use a lambda or block scope? Globals are fine.

    for (int r = 0; r < max_restarts; ++r) {
        // Initialize colors randomly
        for (int i = 1; i <= n; ++i) c[i] = (rand() % 3) + 1;
        
        // Initialize degrees
        // Since deg depends on current colors c
        // Also reset deg explicitly
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j < 4; ++j) deg[i][j] = 0;
            for (int neighbor : adj[i]) {
                deg[i][c[neighbor]]++;
            }
        }
        
        // Calculate initial total conflicts
        // Formula: sum(deg[i][c[i]]) / 2
        current_conflicts = 0;
        for (int i = 1; i <= n; ++i) {
            current_conflicts += deg[i][c[i]];
        }
        current_conflicts /= 2;
        
        // Hill Climb
        bool improved = true;
        while (improved) {
            improved = false;
            for (int i = 1; i <= n; ++i) {
                int curr = c[i];
                int min_diff = 1000000;
                int best_next = -1;
                
                // We want to move to color X if:
                // (neighbors with X) - (neighbors with curr) < 0
                // And we want the minimum such difference.
                
                // Note: Since we only care about reducing conflicts,
                // we just check if ANY color gives reduction.
                // But picking the BEST color (most reduction) is usually better.
                // However, any move towards improvement works.
                // Let's pick best to converge faster.
                
                int inc_neighbors = deg[i][curr];
                for (int next : {1, 2, 3}) {
                    if (next == curr) continue;
                    int diff = deg[i][next] - inc_neighbors;
                    if (diff < min_diff) {
                        min_diff = diff;
                        best_next = next;
                    }
                }
                
                if (min_diff < 0) {
                    // Perform move
                    c[i] = best_next;
                    
                    // Update global conflict count
                    current_conflicts += min_diff;
                    
                    // Update neighbor degrees
                    for (int neighbor : adj[i]) {
                        deg[neighbor][curr]--;
                        deg[neighbor][best_next]++;
                    }
                    
                    improved = true;
                }
            }
        }
        
        if (current_conflicts < best_conflicts) {
            best_conflicts = current_conflicts;
            for (int i = 1; i <= n; ++i) best_colors[i] = c[i];
        }
    }
    
    // Print result
    for (int i = 1; i <= n; ++i) {
        printf("%d%c", best_colors[i], i == n ? '\n' : ' ');
    }
}

int main() {
    init_and_solve();
    return 0;
}
```
Input reading: `scanf` is faster.
Wait, `scanf` inside `init_and_solve`.
Better to put logic cleanly.
Also, `printf` format string `%d` is good.
Note `min_diff` init. If no better color found (diffs all positive or zero), `min_diff` stays large positive, `best_next` stays -1. Check `min_diff < 0` handles it.
Correct.

Wait, inside loop `for (int next : {1, 2, 3});`. C++11 initializer lists are nice. If compiler is old, might fail.
Competitive programming environments usually support C++11/14/17.
Assuming standard modern environment.
If strict C++98, need explicit loops.
I will use explicit loops to be safe.

Also `best_conflicts` init.
If $m=0$, my special handling prints 1s.
Wait, my special handling logic was added.
Is it redundant?
If $m=0$, `deg` init loop `adj[i]` empty -> `deg` all 0.
`current_conflicts` calc: `sum(0)/2 = 0`.
`best_conflicts` becomes 0.
`c` is random.
Output random numbers? No, 1s is simpler.
Random output is valid since conflicts=0.
Wait, if $m=0$, any assignment has 0 conflicts.
My code outputs whatever `best_colors` ended up as (random).
This is valid.
Special case block is not strictly necessary but cleaner.
I'll include it.

One detail: `best_conflicts` initial value.
If $m>0$, conflicts can be at most $M$.
Safe.

Wait, `time(0)` for seeding.
Some online judges have deterministic time requirements or multiple test files run sequentially, resetting time might not matter, but `srand` with `time` is standard for randomness.

Final check:
The prompt asks for "exactly one line" of output.
`cout` or `printf`.
Format: `c1 c2 ... cn`.
Space separated.
My code does this.

One subtle bug possibility:
`deg[neighbor][curr]--`.
`curr` is the OLD color of `i`.
`best_next` is NEW color.
This logic assumes `deg` tracks neighbors' views.
Yes, `deg[u][c]` is number of neighbors of `u` with color `c`.
If `i` changes from `curr` to `best_next`:
For neighbor `v`:
`v` sees `i` changing color.
So `v` loses a neighbor of color `curr`. `deg[v][curr]--`.
`v` gains a neighbor of color `best_next`. `deg[v][best_next]++`.
This is correct.

Looks solid.

```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstdio>

using namespace std;

const int MAXN = 1005;
vector<int> adj[MAXN];
int c[MAXN];              // Colors: 1, 2, 3
int deg[MAXN][4];         // deg[u][color] = number of neighbors of u with that color
int n, m;
int best_conflicts = 2000000000;
int best_colors[MAXN];

int main() {
    // Optimize I/O operations
    // Since using scanf/printf mix, sync with stdio off
    // Or just use cin/cout with sync off
    // Let's use cin/cout with sync off for safety with types
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n >> m)) return 0;

    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    srand(time(0));

    if (m == 0) {
        for (int i = 1; i <= n; ++i) {
            cout << 1 << (i == n ? "" : " ");
        }
        cout << "\n";
        return 0;
    }

    int current_conflicts;

    // Run local search with random restarts
    int num_restarts = 20;

    for (int r = 0; r < num_restarts; ++r) {
        // Initialize random colors
        for (int i = 1; i <= n; ++i) c[i] = (rand() % 3) + 1;

        // Initialize degrees
        // Reset deg array for each restart
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j < 4; ++j) deg[i][j] = 0;
            for (int neighbor : adj[i]) {
                deg[i][c[neighbor]]++;
            }
        }

        // Calculate initial total conflicts
        // B = (1/2) * sum(deg[u][c[u]])
        current_conflicts = 0;
        for (int i = 1; i <= n; ++i) {
            current_conflicts += deg[i][c[i]];
        }
        current_conflicts /= 2;

        // Hill Climbing
        // Stop if no improvement in a full pass
        bool improved = true;
        while (improved) {
            improved = false;
            for (int i = 1; i <= n; ++i) {
                int curr = c[i];
                int min_diff = 1000000;
                int best_next = -1;
                int curr_inc = deg[i][curr]; // Neighbors of i with color 'curr'

                // Try other colors
                for (int next_val = 1; next_val <= 3; ++next_val) {
                    if (next_val == curr) continue;
                    int diff = deg[i][next_val] - curr_inc;
                    // We want minimal diff (most negative)
                    if (diff < min_diff) {
                        min_diff = diff;
                        best_next = next_val;
                    }
                }

                // If we found a move that reduces conflicts
                if (min_diff < 0) {
                    // Update global conflict count
                    current_conflicts += min_diff;
                    
                    // Update neighbor degrees
                    for (int neighbor : adj[i]) {
                        deg[neighbor][curr]--;
                        deg[neighbor][best_next]++;
                    }
                    
                    // Update current color
                    c[i] = best_next;
                    improved = true;
                }
            }
        }

        // Track best solution
        if (current_conflicts < best_conflicts) {
            best_conflicts = current_conflicts;
            for (int i = 1; i <= n; ++i) best_colors[i] = c[i];
        }
    }

    // Output result
    for (int i = 1; i <= n; ++i) {
        cout << best_colors[i] << (i == n ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
Double check logic for `min_diff`.
If multiple `next_val` give same `min_diff`, order doesn't matter much for greedy.
The code picks the last one encountered if strictly `<`?
No, if `diff < min_diff`. First one is init large. Second updates. Third if smaller updates.
So it picks best.
If tie, keeps existing (first best).
This is fine.

Are there any headers missing?
`vector`, `iostream`, `cstdlib`, `ctime` (for time), `cstdio` (not used since using cin/cout but ok).
Seems complete.

One minor thing: `best_conflicts` initialization.
`2000000000` fits in `int` (limit `2.14e9`).
OK.

What if $n=1000, m=n(n-1)/2$ (complete graph)?
Colors 3. 3-coloring a clique $K_n$ needs $n$ colors.
So conflicts will be high.
We want to partition $V$ into 3 sets $S_1, S_2, S_3$ minimizing edges inside sets.
Ideally sizes $n/3, n/3, n/3$.
Conflicts $\approx 3 \times \binom{n/3}{2} = 3 \times (n^2/9)/2 = n^2/6$.
Total edges $n^2/2$. Ratio $1/3$.
Score $1 - 1/3 = 2/3$.
Heuristic should find balanced partitions.
Code does local optimization which should push towards balance.
With 20 restarts, likely enough.

Final check on time limit.
Typical 1s.
My complexity: $20 \times 1000 \times 1000$ worst case updates?
If $M \approx 500,000$.
Passes might be few.
It should pass.
</think>

```cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

const int MAXN = 1005;
vector<int> adj[MAXN];
int c[MAXN];              // Colors: 1, 2, 3
int deg[MAXN][4];         // deg[u][color] = number of neighbors of u with that color
int n, m;
int best_conflicts = 2000000000;
int best_colors[MAXN];

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n >> m)) return 0;

    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    srand(time(0));

    // If m=0, there are no edges, so any assignment results in 0 conflicts.
    if (m == 0) {
        for (int i = 1; i <= n; ++i) {
            cout << 1 << (i == n ? "" : " ");
        }
        cout << "\n";
        return 0;
    }

    // Track best solution found across restarts
    int best_conflicts_val = 2000000000; // Initialize with a large value

    // Run local search with random restarts
    int num_restarts = 20;

    for (int r = 0; r < num_restarts; ++r) {
        // Initialize random colors
        for (int i = 1; i <= n; ++i) c[i] = (rand() % 3) + 1;

        // Initialize degrees
        // deg[u][color] stores the count of neighbors of u having that color
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j < 4; ++j) deg[i][j] = 0;
            for (int neighbor : adj[i]) {
                deg[i][c[neighbor]]++;
            }
        }

        // Calculate initial total conflicts
        // B = (1/2) * sum(deg[u][c[u]])
        // Because each edge (u,v) is counted twice (once in u, once in v)
        int current_conflicts = 0;
        for (int i = 1; i <= n; ++i) {
            current_conflicts += deg[i][c[i]];
        }
        current_conflicts /= 2;

        // Hill Climbing / Local Search
        // We iteratively move vertices if it reduces the number of conflicting edges.
        bool improved = true;
        while (improved) {
            improved = false;
            for (int i = 1; i <= n; ++i) {
                int curr = c[i];
                int min_diff = 1000000; // Tracks the best improvement (reduction)
                int best_next = -1;
                int curr_inc = deg[i][curr]; // Neighbors of i currently sharing color 'curr'

                // Try changing to other colors (1, 2, 3)
                for (int next_val = 1; next_val <= 3; ++next_val) {
                    if (next_val == curr) continue;
                    // Change in conflicts for vertex i if moved to next_val:
                    // Increases by neighbors with next_val (new conflicts)
                    // Decreases by neighbors with curr (removed conflicts)
                    // delta = deg[i][next_val] - deg[i][curr]
                    int diff = deg[i][next_val] - curr_inc;
                    
                    if (diff < min_diff) {
                        min_diff = diff;
                        best_next = next_val;
                    }
                }

                // If we found a move that strictly reduces conflicts (min_diff < 0)
                if (min_diff < 0) {
                    // Commit the move
                    c[i] = best_next;
                    
                    // Update global conflict count using the delta
                    current_conflicts += min_diff;
                    
                    // Update neighbor degrees: 
                    // Neighbors of i perceive i's color change.
                    // Neighbors with color 'curr' lose a neighbor of color 'curr'.
                    // Neighbors with color 'best_next' gain a neighbor of color 'best_next'.
                    for (int neighbor : adj[i]) {
                        deg[neighbor][curr]--;
                        deg[neighbor][best_next]++;
                    }
                    
                    improved = true;
                }
            }
        }

        // Update global best
        if (current_conflicts < best_conflicts_val) {
            best_conflicts_val = current_conflicts;
            for (int i = 1; i <= n; ++i) best_colors[i] = c[i];
        }
    }

    // Output result
    for (int i = 1; i <= n; ++i) {
        cout << best_colors[i] << (i == n ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=1043)

````
I need to minimize the number of conflicting edges, where a conflict occurs when two connected vertices have the same color.

This is related to finding an independent set or a 3-coloring, but since we can have conflicts, we want to minimize them rather than eliminate all of them.

Key insight: For each vertex, the worst case for one color assignment is connecting it to many vertices with the same color. So I should use information about which colors are already "used" by neighbors.

Let me define used[v][color] = true if vertex v has been assigned color.
When coloring a vertex u, I check its neighbors and count how many have already taken each of the three colors.

Strategy:
- Process vertices in order
- For each vertex u, look at its neighbors' colors
- Count how many distinct colors among neighbors are already taken
- Assign u one of the remaining colors (preferably one with minimum usage)

Why this might work well:
- If a neighbor is colored differently from u, that's good for reducing conflicts
- By checking neighbors first, I can see the actual color distribution in my neighborhood
- When there are few distinct neighbor colors, I have more choices, making it easier to avoid conflicts
- This greedy approach tries to make locally optimal choices that propagate well through the graph

The idea is that minimizing the number of *distinct* colors among neighbors gives us better options to place our current vertex in a color that creates fewer conflicts overall.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    
    vector<vector<int>> adj(n + 1);
    for(int i = 0; i < m; i++){
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    vector<int> c(n + 1, 1);
    
    int used[1001][4];
    memset(used, 0, sizeof(used));
    
    for(int i = 1; i <= n; i++){
        int d[4] = {0, 0, 0, 0};
        
        for(auto& v : adj[i]){
            d[c[v]]++;
        }
        
        int best_color = 1;
        int min_distinct = 1e9;
        
        for(int col = 1; col <= 3; col++){
            int distinct = 0;
            for(int k = 1; k <= 3; k++){
                if(d[k] > 0){
                    distinct++;
                    used[v][k] += dist
```
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n, m;
    cin >> n >> m;
    
    vector<vector<int>> adj(n + 1);
    for(int i = 0; i < m; i++){
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    vector<int> c(n + 1, 1);
    int used[1001][4];
    memset(used, 0, sizeof(used));
    
    for(int i = 1; i <= n; i++){
        int d[4] = {0, 0, 0, 0};
        for(auto &v : adj[i]) {
            d[c[v]]++;
        }
        
        int best_color = 1;
        int min_used = 4;
        for(int col = 1; col <= 3; col++){
            int num_used = 0;
            for(int k = 1; k <= 3; k++){
                if(d[k] > 0 && used[i][k] == 0) num_used++;
            }
            if(num_used < min_used){
                min_used = num_used;
                best_color = col;
            }
        }
        c[i] = best_color;
        for(int k = 1; k <= 3; k++){
            if(d[k] > 0) used[i][k] = 1;
        }
    }
    
    for(int i = 1; i <= n; i++){
        cout << c[i] << (i == n ? "" : " ");
    }
    cout << "\n";
    
    return 0;
}
```
````
