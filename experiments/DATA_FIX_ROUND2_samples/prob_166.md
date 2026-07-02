# 题 166 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The user wants me to solve a competitive programming problem based on an AtCoder contest (AHC034).

**Problem Analysis:**

1.  **Objective:** Level an $N \times N$ grid of heights to all zeros using a dump truck with minimum cost.
2.  **Grid Size:** $N = 20$ is fixed. This means the grid has $400$ cells.
3.  **Initial State:** Heights $h_{i,j}$ are given, sum of heights is 0. Range is roughly -100 to 100.
4.  **Operations & Costs:**
    *   **Load (+d):** Current cell height decreases by $d$. Truck load increases by $d$. Cost: $d$.
    *   **Unload (-d):** Current cell height increases by $d$. Truck load decreases by $d$ (must be non-negative). Cost: $d$.
    *   **Move (U/D/L/R):** Move to adjacent cell. Cost: $100 + \text{current\_load}$.
5.  **Constraints:** Max 100,000 turns.
6.  **Goal:** Minimize total cost. The problem asks for the sequence of operations.

**Key Observations:**

*   Since the total sum of heights is 0, the amount of soil to remove from negative height regions equals the amount to add to positive height regions.
*   Soil can be viewed as "positive mass" or "negative mass". We want to move "excess" soil (where $h > 0$, we need to load it) to "deficit" spots (where $h < 0$, we need to unload into it). Wait, let's re-read carefully.
    *   If $h_{i,j} > 0$: This square has excess height. To make it 0, we must *unload* soil? No, wait.
    *   Let's trace the operation definitions again.
        *   `+d` (Load): Decrease current height $h$ by $d$, increase truck load by $d$. This removes soil from the ground.
        *   `-d` (Unload): Increase current height $h$ by $d$, decrease truck load by $d$. This puts soil onto the ground.
    *   So, if $h_{i,j} > 0$ (too high), we need to remove soil -> Load (`+`).
    *   If $h_{i,j} < 0$ (too low), we need to add soil -> Unload (`-`).
    *   The truck carries soil removed from high areas to put them in low areas.
    *   Cost logic: Moving costs depend on load. Loading/Unloading costs equal the amount moved directly.
    *   Crucial insight: The cost of moving with load $L$ is $100 + L$.
    *   Cost of moving empty ($L=0$) is $100$.
    *   Cost of carrying load $L$ is $100 + L$.
    *   Total Cost = $\sum (\text{Load Amounts}) + \sum (\text{Unload Amounts}) + \sum (\text{Move Costs})$.
    *   Note that $\sum (\text{Load Amounts}) = \sum (\text{Unload Amounts}) = \text{Total Soil Moved}$ because conservation of mass applies (net change in ground is 0). Let this total amount be $V$.
    *   Actually, Load decreases height, so if $h>0$, we load $h$ eventually. If $h<0$, we unload $|h|$ eventually. Total Volume $V = \sum_{h_{i,j}>0} h_{i,j}$.
    *   Total Cost of Load/Unload ops = $\sum |h_{i,j}|$. This part is constant regardless of how we move. It doesn't affect optimization.
    *   Optimization focuses purely on minimizing Movement Costs.
    *   Movement Cost for step $k$: $100 + L_k$.
    *   Here $L_k$ is the load when moving.
    *   We start at $(0,0)$ with $L=0$.
    *   We want to transport soil from sources ($h>0$) to sinks ($h<0$).
    *   Transporting 1 unit of soil from Source to Sink incurs movement cost depending on distance and how much else is being carried.
    *   Specifically, if we carry $L$ units for a move, the cost is $100 + L$. If we carry 0 units, cost is 100.
    *   So, every move costs at least 100. Plus, for every unit of soil transported over one edge, we pay extra 1.
    *   Therefore, Total Cost = $(\sum |h_{i,j}|) + \text{Base Movement Costs} + \text{Extra Transport Cost}$.
    *   Base Movement Cost = $100 \times (\text{Number of Moves})$.
    *   Extra Transport Cost = $\sum (\text{soil units} \times \text{distance traveled})$.
    *   Since $N=20$ is small (400 cells), but max steps are 100,000. This suggests a flow problem or a shortest path problem where we decide which sink each source supplies.
    *   Actually, since the "Base Movement Cost" is incurred for every step taken by the truck regardless of load, and "Extra Transport Cost" is proportional to soil amount $\times$ distance, this structure looks like finding paths to deliver specific amounts of flow.
    *   However, there's a catch: The truck can carry arbitrary amounts up to $10^6$. The constraint $d \leq 10^6$ per operation is likely not binding since max $h_{i,j}$ is 100 and grid is $20 \times 20$. Total volume is small.
    *   We can choose to consolidate flows. If we pick up 5 units from A and 5 units from B, we don't necessarily have to return to A then go to B. But wait, the truck starts at (0,0).
    *   If we visit A, pick up load, move to B, drop off. Then move back or elsewhere.
    *   Every time we pick up load, we add to the truck capacity. Every time we drop load, we reduce it. Moving costs more with higher load.
    *   Ideally, we want to minimize the total flow-weighted distance.
    *   Since all edges effectively cost 1 (extra cost per unit of soil) plus a fixed traversal cost of 100.
    *   The truck is essentially a delivery robot that needs to empty the "Source" piles into "Sink" holes.
    *   Since the truck can hold infinite load (bounded by practical limits), we can think of this as matching Sources to Sinks.
    *   For any path from a Source $S_i$ to Sink $T_j$ with distance $dist(S_i, T_j)$, transporting 1 unit costs:
        *   Pickup: Cost 1 (at S).
        *   Movement: Cost $dist(S_i, T_j) \times (100 + 1)$? No.
        *   Let's refine the cost model.
            *   Suppose we move 1 unit of soil along an edge. The edge traversal cost is $100 + 1 = 101$.
            *   If we move 0 units, cost is 100.
            *   Difference is 1 per unit of soil.
            *   So, transporting $K$ units along a path of length $L$ adds $K \times L$ to the total cost compared to just traversing empty.
            *   Traversal cost is always $100 \times L$.
            *   So, we simply need to pair positive height cells (Sources) with negative height cells (Sinks) such that the sum of flow $\times$ distance is minimized?
            *   Wait, is it exactly that simple? What about the initial position (0,0)?
            *   And the fact that the truck must physically traverse the grid.
            *   Also, we can combine trips. If I pick up 2 units at A, go to B (sink), drop 1, return to C (source), pick up 1...
            *   But notice: Transporting soil from A to B involves picking up at A, moving through edges to B, unloading at B.
            *   Any edge used for transportation contributes $(100+L)$ instead of 100. The difference is $L$.
            *   Total cost = $\sum |h_{i,j}| + \sum_{\text{moves}} 100 + \sum_{\text{transported units } u} \text{distance}(u)$.
            *   Wait, the $\sum_{\text{moves}} 100$ term depends on the number of moves.
            *   Does minimizing soil-distance automatically minimize total moves? Not necessarily.
            *   However, since we just need to satisfy the net demand, and we can revisit nodes, the structure of the tour matters.
            *   Usually in these problems, with small grids, we can simplify the graph.
            *   With $N=20$, maybe we can view this as a Minimum Cost Flow problem? Or simply a bipartite matching?
            *   The problem is finding a set of paths from Sources to Sinks. Each unit of height must travel from its source to a sink. The cost of moving a unit from $S$ to $T$ is effectively the Manhattan distance between them (since $N$ is small, no diagonal, grid distances apply). Wait, is it Manhattan distance?
            *   In a grid, yes, usually Manhattan distance is sufficient if we don't care about obstacles (no obstacles here).
            *   Is it possible to transport multiple units simultaneously? Yes, once loaded, they stick together.
            *   If we load $L$ units at a source and travel distance $D$, we pay $L \times D$ extra over base move costs.
            *   To minimize $\sum L \times D$, we should pair close sources and sinks.
            *   Wait, the base move cost ($100 \times \text{moves}$) is also relevant.
            *   If we treat the grid as a metric space (Manhattan distance), the cost to ship 1 unit from $S$ to $T$ is $|x_S - x_T| + |y_S - y_T|$.
            *   Why? Because for each step towards T, load decreases (if transporting to sink). The cost added per step for load is 1.
            *   Wait, if I have load $L$ and move 1 step, cost is $100+L$. If I had 0 load, cost 100. Difference $L$.
            *   If I transport $L$ units across $k$ steps, extra cost is $L \times k$.
            *   So we just need to match positive height cells to negative height cells to minimize $\sum (\text{flow} \times \text{dist})$.
            *   Wait, the "base cost" of moving ($100 \times \text{moves}$) might encourage us to minimize total moves. Minimizing moves usually implies going directly from current to next target.
            *   Given the constraints and problem type (AtCoder Beginner Contest style), this smells like a Maximum Weight Bipartite Matching or Min-Cost Perfect Matching.
            *   Actually, since we have multiple amounts at each cell, it's a transportation problem (Min Cost Flow).
            *   Nodes: $S_1, \dots, S_m$ (where $h_{i,j} > 0$) and $T_1, \dots, T_p$ (where $h_{i,j} < 0$).
            *   Edge weights = Manhattan distance.
            *   Supply at $S$ = $h_{i,j}$. Demand at $T$ = $|h_{i,j}|$.
            *   Minimize $\sum f_{ij} d(i,j)$.
            *   Since it's on a grid, Manhattan distance satisfies triangle inequality. The optimal matching will decompose into individual transfers.
            *   So we just need to send flow from Sources to Sinks minimizing distance.
            *   Can we assume the truck starts at (0,0) and just teleports or walks efficiently?
            *   The problem says "Initially... at (0,0)".
            *   The truck has to physically move. The "extra cost" calculation assumes the truck visits the source and destination.
            *   Wait, does the base movement cost $100 \times \text{moves}$ imply we pay for "empty" travel?
            *   Yes, every move costs 100 minimum.
            *   So we want to visit all necessary source/sink locations?
            *   Actually, do we need to visit (0,0) first? Yes.
            *   If the optimal matching pairs Source $S_a$ and Sink $T_b$, we must go from $(0,0) \to \dots \to S_a \to \dots \to T_b \to \dots \to \text{next target}$.
            *   Wait, if we pair many sources and sinks, the order of visiting them matters for the $100 \times \text{moves}$ component.
            *   However, note that $100$ is quite large compared to typical $d$. Max $h$ is 100, grid size 20. Distance $\approx 40$.
            *   If we move with load, we pay $100+d$. If empty, 100.
            *   The term $100 \times \text{total\_moves}$ dominates or competes with load costs.
            *   But actually, if we optimize the flow (minimize $L \times dist$), does that implicitly handle the path?
            *   Consider the case where we just transport everything independently.
            *   For a unit at $(r,c)$ with $h>0$ and unit at $(r',c')$ with $h'<0$, min cost to fix these two specifically is transport 1 unit from first to second.
            *   This requires going $(0,0) \to (r,c)$ (pick up), $(r,c) \to (r',c')$ (drop off), $(r',c') \to (0,0)$ (or next target).
            *   Wait, if we finish at $(r',c')$, we might end the game there.
            *   But we can chain deliveries.
            *   Path: $(0,0) \to S_1 \to T_1 \to S_2 \to T_2 \dots$
            *   Moves: $d((0,0), S_1) + d(S_1, T_1) + d(T_1, S_2) + \dots$
            *   Soil cost: $\sum (h \times \text{dist}(S, T))$.
            *   Fixed cost: $100 \times \text{number of steps}$.
            *   Since $N=20$ is very small, maybe we can use this: The total volume of soil is at most $20 \times 20 \times 100 = 40,000$.
            *   The "flow" formulation finds the pairing.
            *   The "path" formulation finds the route.
            *   Since the cost function separates into "Flow Cost" + "Travel Cost", and Travel Cost depends on visiting nodes in some order.
            *   Wait, is the Travel Cost simply determined by the set of visited nodes?
            *   No, it depends on the order.
            *   However, since we can stop at any point, and $N$ is small, maybe a simpler strategy exists.
            *   Wait, the problem is asking for ANY valid solution. The scoring is based on efficiency. The scoring formula uses $cost$ in denominator. Lower cost is better.
            *   The example link shows "Show example". The problem text mentions $h_{i,j}$ values are small integers.
            *   Is it possible to implement a greedy approach or a specific heuristic?
            *   Let's reconsider the cost components.
                1. Load/Unload Cost: Constant $\sum |h_{i,j}|$. Let's call this $C_0$.
                2. Transport Cost: Proportional to $\sum (\text{amount} \times \text{distance})$. Call this $C_{trans}$.
                3. Empty Move Cost: Proportional to $\text{Total Steps}$. Call this $C_{move}$.
            *   Since we want to minimize total cost, and Load/Unload is fixed, we minimize $C_{trans} + C_{move}$.
            *   $C_{trans}$ is minimized by solving the Min Cost Transportation Problem on the grid graph (or complete graph with Manhattan distances).
            *   $C_{move}$ is the length of the path visiting the required sites.
            *   Notice that $C_{trans}$ includes the cost of carrying load over distance.
            *   If we move from $S$ to $T$ with load 100, cost is $100 \times \text{dist} + 100 \times \text{dist} = 200 \times \text{dist}$? No.
            *   Cost of step: $100 + \text{load}$.
            *   Sum of step costs along path of length $L$ with varying loads $l_k$: $\sum (100 + l_k) = 100 L + \sum l_k$.
            *   $\sum l_k$ is exactly the load-weighted distance (Transport Cost).
            *   So Total Cost = $\sum |h_{i,j}| + 100 \times (\text{Total Steps}) + (\text{Transport Cost})$.
            *   Wait, Transport Cost *includes* the cost of moving load. The formula derived earlier was:
                Total Cost = (Load ops cost) + (Unload ops cost) + (Move costs).
                Load ops cost = $\sum$ load amount = $\sum_{h>0} h = \frac{1}{2} \sum |h|$.
                Unload ops cost = $\sum$ unload amount = $\frac{1}{2} \sum |h|$.
                Move costs = $\sum_{steps} (100 + \text{load}) = 100 \times \text{Steps} + \sum_{steps} \text{load}$.
                Total = $\sum |h| + 100 \times \text{Steps} + \sum_{steps} \text{load}$.
            *   $\sum_{steps} \text{load}$ is the Transport Cost (amount $\times$ distance).
            *   So we need to minimize: $\sum |h| + 100 \times \text{Steps} + \text{Transport Cost}$.
            *   Since $\sum |h|$ is constant, we minimize $100 \times \text{Steps} + \text{Transport Cost}$.
            *   This looks like we want to pair sources and sinks efficiently AND visit them in a way that minimizes steps.
            *   However, since we can revisit vertices arbitrarily and the grid is small ($N=20$), we can just find the Min Cost Flow first. That gives us pairs $(S, T)$ with amounts.
            *   This fixes the Transport Cost component optimally (assuming we can travel freely).
            *   Now, how to minimize Steps?
            *   If the flow tells us to send 100 units from $S_A$ to $T_B$, and 50 units from $S_C$ to $T_D$.
            *   We can do this by: Go $Start \to S_A \to T_B \to S_C \to T_D \to \dots$.
            *   Or $Start \to S_A \to S_C \to T_B \to T_D$.
            *   Wait, if we collect at $S_A$ then $S_C$, load is additive. Transporting both to $T_B, T_D$ separately?
            *   Actually, the flow decomposition allows us to choose paths.
            *   Since the grid is a graph, and $N=20$ is tiny, we might even simulate a shortest path search or use a BFS-based approach.
            *   However, the problem asks for a sequence of operations.
            *   Let's check the constraints again. $100,000$ operations max. Grid size $20 \times 20$.
            *   Maybe we don't need full optimization.
            *   Since $N=20$, maybe we can just treat this as finding a path that picks up needed soil and drops it off.
            *   But wait, the Transport Cost is heavily dependent on *how far* we carry load.
            *   Is it possible to simplify the Transport Cost?
            *   Actually, in a grid without obstacles, the shortest path is Manhattan distance.
            *   If we just calculate the Min Cost Flow (Matching), that determines the minimal Transport Cost.
            *   Then we need to construct a tour that respects the flow.
            *   Actually, for the scoring function provided: `round(10^9 * base / (cost + diff))`.
            *   Since `diff` penalizes errors, we MUST clear everything.
            *   So `base` is fixed. We need to minimize `cost`.
            *   `cost` depends on `Steps` and `Transport Cost`.
            *   Minimizing Transport Cost is critical. Minimizing Steps is critical.
            *   Notice $100 \times \text{Steps}$. If we take a direct path from a node to itself, steps=0.
            *   We must visit all $i,j$ where $h_{i,j} \neq 0$.
            *   So we basically need to visit a set of points.
            *   The number of Steps is roughly proportional to the number of connected components of non-zero cells if we could teleport, but we have to walk.
            *   With $N=20$, maybe we can perform a BFS/Dijkstra on the state space $(position, current\_load)$?
            *   State space size: $20 \times 20 \times (\text{load range})$.
            *   Load range: Up to total soil $\approx 40,000$.
            *   State space too big ($10000 \times 40000$). Too complex for standard DP/BFS within time limits unless optimized.
            *   However, do we really need exact optimal matching? The problem is from AtCoder Heuristic Contest (AHC). Usually, these require good heuristics, not necessarily optimal algorithms.
            *   Let's look at the "Story" and "Scoring". It's about leveling the ground.
            *   Often, a simple strategy works well enough.
            *   Strategy Idea:
                1. Identify sources (height > 0) and sinks (height < 0).
                2. Prioritize matching closest pairs?
                3. Or, just process cells row by row or column by column?
            *   Let's consider the specific property: Total height is 0.
            *   A common strategy for these grid level problems with dump trucks is:
                *   Iterate through the grid in some order.
                *   When at a cell with surplus ($h>0$), load it into the truck.
                *   When at a cell with deficit ($h<0$), unload into it.
                *   Carry load between them.
            *   But the direction matters. If I'm at a source, I load. Now I must go to a sink.
            *   If I am at a sink, I must arrive from somewhere with load.
            *   Since the truck starts at (0,0), maybe we can define a priority queue of tasks.
            *   Task: Move 1 unit from $S_i$ to $T_j$. Cost $\propto dist$.
            *   But we carry multiple units.
            *   Since we want to minimize $100 \times \text{Steps} + \text{Transport Cost}$, and Transport Cost is $\sum \text{load} \times \text{dist}$, this is equivalent to minimizing $\sum (100 + \text{load}) \times \text{step\_count}$.
            *   Let's look at the "Sample Example" (though hidden here, the link points to it).
            *   Given this is AHC034 (Heuristic Contest), exact solutions are hard. But $N=20$ is small enough that perhaps we can use local search or simulated annealing or a greedy constructive approach that is quite efficient.
            *   However, usually these contests reward strategies that work well on random cases.
            *   The noise generation suggests somewhat smooth gradients.
            *   Let's try a greedy strategy:
                *   Keep track of the truck's current position.
                *   Look at neighbors or nearby cells.
                *   If current cell has soil ($h>0$), load it. (Wait, if $h>0$, we load).
                *   If current cell has hole ($h<0$), unload. (Wait, if $h<0$, we unload).
                *   Wait, if we are at a cell with $h=0$, we can either do nothing or pass through.
                *   If we have load, we want to move to a cell with $h<0$.
                *   If we have no load, we want to move to a cell with $h>0$.
                *   If we have load and current cell has $h<0$, we should unload?
                    *   Unloading cost is $d$ (amount). This is constant.
                    *   But we also save future travel cost?
                    *   Yes, unloading reduces load, reducing move cost later.
                *   So, Priority Queue approach:
                    *   Store events: (Location, Type, Height).
                    *   Always move to a neighbor that helps reduce the objective.
                *   However, simple greedy might get stuck (local optimum).
                *   Let's refine the objective function locally.
                *   Cost to perform an operation:
                    *   If we move to a cell, cost increases by $100 + \text{load}$.
                    *   If we load/unload, cost increases by amount, changes state.
                *   Wait, the problem statement says: "minimizes the cost".
                *   This suggests we just need a valid plan that is reasonably good.
                *   Common heuristic for this specific problem (Dump Truck Land Leveling) seen in other contexts (like similar LeetCode/CodeForces problems):
                    *   Treat positive cells as sources, negative as sinks.
                    *   Match them greedily based on distance.
                    *   Once matched, follow the path.
                *   However, we need to combine these into a single path starting from (0,0).
                *   If we match $S_1 \to T_1$, $S_2 \to T_2$...
                *   We can create a path $(0,0) \to S_1 \to T_1 \to S_2 \to T_2 \dots$.
                *   This minimizes "backtracking" somewhat if ordered correctly.
                *   Since we don't know the best order, we can sort tasks by coordinate (e.g. reading order).
                *   Is reading order sufficient?
                *   Let's verify with constraints. $N=20$ is small. Reading order covers the grid.
                *   Maybe we just iterate $(0,0)$ to $(N-1, N-1)$.
                *   Algorithm:
                    1. Maintain `truck_pos`, `truck_load`. Initially `(0,0)`, 0.
                    2. Iterate through all cells $(i, j)$ in some order (e.g., row-major).
                    3. If $h[i][j] > 0$: Load $h[i][j]$ units. `truck_load += h[i][j]`. Output `+h`. `h[i][j] -= h[i][j]`.
                       *   Wait, loading happens *now*. Do we immediately try to move to a sink?
                       *   Or do we wait until we encounter a sink?
                       *   Better: Find the nearest sink $h[k][l] < 0$ and move there with the current load?
                       *   If we just process row-major, we might move load unnecessarily long distances if we don't check ahead.
                    4. Optimized Greedy:
                        *   If `truck_load > 0`:
                            *   Search for the nearest neighbor (in BFS distance) with $h < 0$.
                            *   If found, move towards it, dropping load if necessary?
                            *   If multiple sinks, which one? Nearest seems logical.
                            *   What if no sink is available (e.g. all remaining sinks are far)? Still move to nearest.
                        *   If `truck_load == 0`:
                            *   Search for the nearest neighbor with $h > 0$.
                            *   Move towards it, loading if reached.
                        *   Repeat until all $h=0$.
                    5. Is this optimal?
                       *   Cost of moving load $L$ over distance $D$: $L \times D$.
                       *   This matches the "Transport Cost" idea.
                       *   Cost of traversing empty over distance $D$: $100 \times D$.
                       *   By matching sources and sinks greedily, we minimize $\sum L \times D$.
                       *   This heuristic is known to work well for Earth Mover's Distance type problems.
                        *   Wait, the cost is slightly different. If I move empty, I pay 100 per step. If I move load, I pay 100+d per step.
                        *   So moving empty and moving with load both cost 100 per step (fixed), but load adds variable cost.
                        *   So we just need to minimize the sum of distances weighted by load.
                        *   This is exactly matching Sources to Sinks with min weight perfect matching.
                        *   Greedy matching (nearest neighbor) is an approximation.
                        *   Is it allowed to approximate? "score of a submission is the total score for each test case".
                        *   Usually AHC problems accept near-optimal.
                        *   Let's check complexity. $N^2 = 400$ cells.
                        *   Each step reduces total absolute height $\sum |h|$.
                        *   In one iteration of loop above, we might pick up soil and deliver it.
                        *   But finding nearest sink takes $O(N^2)$ or $O(N^2 \log N)$?
                        *   If we do BFS from current pos to find nearest sink/source, it's fast ($400$ edges).
                        *   If we do this repeatedly, say 400 times (visiting each cell once), complexity is fine.
                        *   Wait, we might need to backtrack. e.g. pickup at (1,1), drop at (5,5), now load at (1,1) again?
                        *   Once a source is picked up (height becomes 0), we never load from it again.
                        *   Same for sinks.
                        *   So we visit each active cell at most twice? (Approach to pickup/dropoff).
                        *   Actually, we might zig-zag.
                        *   But total distance isn't huge.
                        *   Max distance per unit of soil is bounded by $N^2$.
                        *   Total units is 40,000.
                        *   Operations limit 100,000.
                        *   40,000 units * distance 40 = 1,600,000 operations?
                        *   Wait, "operations" count includes Load/Unload/Move.
                        *   Constraint: "You can perform a maximum of 100000 turns of operation."
                        *   If we have 40,000 units, and average distance 20, we need ~800,000 moves. This exceeds 100,000.
                        *   **CRITICAL CONSTRAINT**: 100,000 turns.
                        *   Total volume is at most $400 \times 100 = 40,000$.
                        *   If we have to move 40,000 units, and each unit requires a Move (at least 1), we would need 40,000 turns just for movement if we carry 1 unit at a time.
                        *   If we carry full load (up to $10^6$ or 100), we need fewer turns.
                        *   But if we need 1,000,000 units? No, max height is 100. Max total volume is 40,000.
                        *   Ah, max height per square is 100. So total volume is 40,000.
                        *   But we are constrained to 100,000 operations.
                        *   If we have 40,000 units to move, and we move them, we generate operations.
                        *   Each Load operation counts as 1 turn? "Specify an integer d... load d units... This operation ... costs of d."
                        *   Wait, "Turns of operation".
                        *   Input says: "perform a maximum of 100000 turns of operation."
                        *   The output format is $T$ lines. $s_t$ is the operation.
                        *   Does one load/unload/deliver constitute one turn? Yes.
                        *   So we cannot make 1,600,000 moves.
                        *   Wait, if total volume is 40,000, and we carry 1 unit per trip, we need 40,000 Load ops and 40,000 Unload ops. That's 80,000 ops. Plus Moves.
                        *   If distance is 10, we need 400,000 moves. TLE.
                        *   **However**, we can carry multiple units.
                        *   Max load per cell is 100.
                        *   So we can fill the truck at a cell with $h=100$, and carry 100 units.
                        *   We can carry multiple loads across the grid.
                        *   This drastically reduces the number of Move operations.
                        *   With $h$ up to 100, we can aggregate flow.
                        *   So the strategy should definitely be: **Maximize load** before moving.
                        *   i.e., Load everything from a positive cell immediately.
                        *   Then drive to a negative cell to unload everything.
                        *   This ensures we move as few times as possible for the volume transfer.

**Refined Strategy:**
1.  We have a set of positive cells $P$ and negative cells $M$.
2.  We want to pair them up.
3.  Since $N$ is small, we can just compute pairwise distances.
4.  However, we need to respect the constraint of 100,000 turns.
    *   Total volume $V \le 40,000$.
    *   Turns limit $K = 100,000$.
    *   We need to avoid excessive overhead.
    *   Since $V$ is small compared to $K$, but moves cost 100+load, we still want to minimize moves.
    *   If we pair $p \in P$ and $m \in M$, we move from $p$ to $m$.
    *   Path length $L$. Load $h_p$. Cost contribution: Load/Unload ops (constant per unit of volume) + Move ops ($100 + \text{load}$).
    *   Wait, the turn limit is strict.
    *   If we output 200,000 lines, we fail.
    *   But max height is 100. So if we load 100 units at once, we make 1 Load op.
    *   If we travel distance 20, we make 20 Move ops.
    *   If we have 400 cells with value 100, volume = 40,000.
    *   Pairing them up: Suppose 200 sources (val 100) and 200 sinks (val -100).
    *   If we pair them directly: 200 trips.
    *   Trip: 1 Load + $Dist$ Moves + 1 Unload.
    *   Total Ops $\approx 200 \times (1 + Dist + 1)$.
    *   If $Dist \approx 20$, $Ops \approx 200 \times 22 = 4400$.
    *   This is well within 100,000.
    *   So the main goal is to keep trips combined and efficient.
    *   Don't do 400 trips for 40,000 volume. Group them!
    *   Wait, if I have source at $(0,0)$ with $h=50$ and source at $(0,1)$ with $h=50$.
    *   Sink at $(10,10)$ with $h=-100$.
    *   Strategy A: Load 50 at $(0,0)$, move to $(10,10)$, unload 50. Return. Load 50 at $(0,1)$, move, unload.
        *   Cost: $(10+100+50)$ move + load/unload.
    *   Strategy B: Visit $(0,0)$, Load 50. Visit $(0,1)$, Load 50. Total Load 100. Move to $(10,10)$, Unload 100.
        *   This saves the return trip $(10,10) \to (0,1)$.
        *   Instead of returning empty from $(10,10)$ to $(0,1)$ to load next chunk, we go $(0,0) \to (0,1)$ (empty) $\to (10,10)$ (full).
    *   This clearly reduces empty moves.
    *   Since empty moves cost 100 per step, and loaded moves cost $>100$, we want to minimize empty moves especially over long distances, and maximize consolidation.
    *   Consolidation means grouping pickups.
    *   Since we start at $(0,0)$, we should probably just traverse the grid.
    *   **Algorithm Proposal:**
        *   Simulate the grid traversal.
        *   Use a priority queue or sorted list of tasks.
        *   Or simply a "greedy nearest match" that accounts for current location.
        *   Wait, the simplest robust solution for AHC is often a randomized local search or just a good enough greedy.
        *   Given the constraints and nature, I will implement a simulation based approach:
            *   We need to output operations.
            *   We maintain current `(pos, load)`.
            *   We maintain a list of remaining demands (Source/Sink).
            *   Since we want to reduce total "abs sum of heights", we want to clear cells.
            *   But clearing order matters for transport cost.
            *   Let's implement a **Local Search / Optimization loop**?
            *   Time limit 10s (usually) for 100 test cases. Wait, "execution time... limit... stated".
            *   The problem says "execution time limit are also stated in the statement so be aware of the complexity". The statement snippet didn't explicitly give seconds but "There are 150 test cases".
            *   Standard AHC usually runs submissions against many cases.
            *   My code needs to run once per test case? Yes.
            *   Complexity should be low.
            *   Let's go for a **Greedy Heuristic**.
            *   Logic:
                1. List all coordinates with $h \neq 0$.
                2. Compute distances between all pairs? $O(N^4)$. $20^4 = 160,000$. Acceptable.
                3. Match Sources to Sinks to minimize weighted distance?
                   *   Hungarian algorithm? $O(M^3)$. $M \approx 400$. $400^3 = 64,000,000$. Might be slow if repeated 150 times.
                   *   Wait, 150 test cases. $6.4 \times 10^7 \times 150$ is too much.
                   *   Need faster matching.
                4. Simplest matching: Sort sources and sinks by coordinate (e.g., reading order) or proximity?
                   *   Actually, just sorting by distance from start might help.
                   *   Wait, the problem is essentially **Earth Mover's Distance** on a grid.
                   *   Since $N$ is small, we can use **Network Flow**.
                   *   Construct graph: Source node $S$, Sink node $T$. Intermediate nodes = Grid cells.
                   *   Capacity $\infty$ on edges?
                   *   Cost per unit flow = 1 (for the load) + 100?
                   *   No, the 100 is per move. The flow formulation minimizes $\sum \text{load} \times \text{dist}$.
                   *   If we assume we can move freely between layers of flow (which we can't physically due to truck limitations), Min-Cost Max-Flow solves the matching.
                   *   Wait, Min-Cost Flow handles the "pairing" optimally.
                   *   Does it handle the "trucking" aspect?
                   *   Yes, if the cost is linear distance, it finds the optimal assignment.
                   *   Then we reconstruct the path.
                   *   Reconstruction: We have $f(u,v)$ flow from Source $u$ to Sink $v$.
                   *   We can concatenate paths $(0,0) \to u_1 \to v_1 \to u_2 \to v_2 \dots$.
                   *   Order matters. Just following the flow paths sequentially works.
                   *   Wait, if we have a path $S_1 \to T_1$, we do it. Then go back to start?
                   *   No, we can just visit $S_1$, load, go to $T_1$, unload.
                   *   Then where? Start at $T_1$. From $T_1$, go to next source $S_2$?
                   *   We might need to revisit $(0,0)$? No, we don't need to return to $(0,0)$ to refill, because we can refill anywhere ($h>0$).
                   *   So we just need a path that visits all necessary Source/Sink combinations.
                   *   Actually, since we just need to execute the moves, any valid permutation of "Pickup-Delivery pairs" works as long as we connect them.
                   *   But the connection cost (moving empty between pairs) matters.
                   *   Wait, the cost formula $100 \times \text{Steps}$ means we want to minimize total steps.
                   *   So we want a Tour that visits the sequence of Sources and Sinks required by the flow.
                   *   This sounds like a **Path Planning** problem.
                   *   Since we don't need to return to origin, it's a Path Cover.
                   *   Given the complexity and constraints, implementing a full solver might be overkill/risky for TLE if I mess up the details.
                   *   However, a simpler heuristic often works.
                   *   **Strategy**: Iterate through the grid cells. If a cell needs lifting (load), do it. If it needs filling (unload), do it.
                   *   This is a bit naive.
                   *   Let's consider the sample inputs or general behavior.
                   *   Usually, sorting the non-zero cells by distance from $(0,0)$ is a good start.
                   *   Actually, let's write a program that implements a **Min-Cost Matching** approximation using a simple greedy approach (nearest neighbor matching) or K-shortest path.
                   *   Actually, $N=20$ is extremely small. Maybe I can afford a BFS for every decision?
                   *   Wait, I need to generate a string of operations.
                   *   Let's try a **Randomized Local Search** or **Simulated Annealing** on the sequence of moves?
                     *   State: current heights, current truck position.
                     *   Action: Move or Load/Unload.
                     *   Evaluation function: Estimate final cost.
                     *   Hard to evaluate "final cost" without simulating to end (state space explosion).
                   
    *   **Alternative Idea:** Since it's a "leveling" problem, simply processing cells in a sweep-line manner often works.
        *   Iterate $i$ from $0$ to $N-1$, $j$ from $0$ to $N-1$.
        *   Move to $(i,j)$ (using BFS for shortest path).
        *   If $h[i][j] > 0$: Load all $h[i][j]$. (This clears it).
        *   If $h[i][j] < 0$: We must unload $|h[i][j]|$. But truck might not have load.
        *   Wait, if I just iterate and load/unload, I might be left with load that needs to go somewhere.
        *   Better: Maintain a list of tasks.
        *   Tasks = pairs $(Source, Sink, Amount)$.
        *   Generate these pairs via Min Cost Flow.
        *   Since implementing Min Cost Flow in C++ during a contest/hackathon is tedious (requires library or writing Dijkstra/Bellman-Ford), and I need a compact code.
        *   Let's check the constraints and operations again.
        *   100,000 ops.
        *   We can surely implement a BFS to find the shortest path.
        *   Algorithm:
            1.  Calculate Min-Cost Perfect Matching between $PositiveCells$ and $NegativeCells$.
                *   How? Since $N$ is small, we can just try to pair each Positive cell with the *nearest* Negative cell, and update availability.
                *   Is greedy matching optimal? For Earth Mover's Distance on a line/grid with convex cost, yes-ish. But here cost is linear distance.
                *   Actually, linear cost implies we want to minimize $\sum |p-q|$.
                *   On a line, sorting works. On a grid, it's harder.
                *   However, with random inputs (Perlin noise), greedy might be decent.
            2.  Given a matching, we have a set of segments to transport.
            3.  Traverse the grid to visit these segments.
                *   We need to order the segments.
                *   Since we want to minimize empty moves, we should process segments in an order that clusters them spatially.
                *   Clustering is achieved by sorting the segments by their centroid? Or just visiting grid cells $(0..N, 0..N)$.
                *   Since we start at $(0,0)$, we can sort segments by something?
                *   Wait, if we just follow a path through the grid (e.g., snake pattern), we can accumulate loads and dump them.
                *   Snake Pattern: $(0,0) \to (0,1) \to \dots \to (0, N-1) \to (1, N-1) \to \dots \to (N-1, 0)$.
                *   While following this path, whenever we hit a source, we load everything. Whenever we hit a sink, we dump everything.
                *   Does this work?
                *   Example: Source at $(0, 5)$, Sink at $(10, 5)$. Path goes $(0,5) \to \dots \to (10, 5)$.
                *   We load at 5. We keep carrying. We reach 15. Dump.
                *   This seems efficient!
                *   What about Source at $(10, 5)$, Sink at $(0, 5)$?
                *   We encounter Source at 10. We load.
                *   We encounter Sink at 0. We dump. But we haven't reached 0 yet (coming from 0).
                *   We would need to have dumped at 0 before reaching 10.
                *   Wait, if we follow the path, we visit 0 first. Then 10.
                *   If Sink is at 0, we unload. Truck empty. Then at 10, we load. Then we need to go back to 0?
                *   This "Snake" strategy forces long empty returns.
                *   But maybe we can just use the matching to identify the pairs, and then build a path.
                *   Since $N$ is small, finding the path covering the matching nodes is easy?
                *   Wait, we can't easily reorder the snake.
                *   Actually, since the total height is 0, and we start with 0 load.
                *   We can just use a **Queue of Tasks**.
                *   Each task: (Destination Cell, Amount, Priority).
                *   Where to start? (0,0).
                *   Strategy:
                    *   Find the nearest Source to current Position?
                    *   Or find the nearest Sink?
                    *   If we have load, go to nearest Sink.
                    *   If no load, go to nearest Source.
                    *   This is a simple greedy loop.
                    *   Does it fit in 100,000 ops?
                    *   Let's analyze the ops count for this greedy loop.
                    *   Worst case: Alternating Source-Sink.
                    *   Number of Sources $S$. Number of Sinks $K$. $S \approx K$.
                    *   If we pick up at $S_1$, drop at $T_1$, go to $S_2$, drop at $T_2$.
                    *   Total moves $\approx 2 \times (\text{dist}(S_1, T_1) + \text{dist}(T_1, S_2))$.
                    *   Since we match nearest, $S_i \leftrightarrow T_i$ distance is small.
                    *   $\text{dist}(T_1, S_2)$ is likely also small if dense.
                    *   So total ops $\approx O(\text{Num Pairs} \times N)$.
                    *   With 400 cells, $N=20$. Pairs $\approx 100-200$.
                    *   $200 \times 20 = 4000$ moves.
                    *   This is extremely safe within 100,000.
                    *   Even if we backtrack a bit, it won't explode.
                    *   **So, Greedy Loop is the way.**
                    
    *   **Greedy Loop Details:**
        *   State: `curr_x`, `curr_y`, `current_load`.
        *   Initialize at `(0,0)`, load 0.
        *   Loop until `total_abs_height` == 0 (or max iterations).
        *   Check neighbors of `curr`.
        *   Decide action based on `current_load` and neighbor state.
        
        *   **Decision Logic:**
            1.  If `current_load > 0`:
                *   Target: Find neighbor with $h < 0$? No, find the *best* neighbor to move to.
                *   Ideally, move towards a sink.
                *   Which sink? The one with largest $|h|$? Or closest? Closest is safer for local search.
                *   Actually, since we just need *any* sink to drop off load, moving to an adjacent $h<0$ is beneficial.
                *   If adjacent $h<0$ exists:
                    *   Should we dump ALL or PARTIAL?
                    *   Dumping partial leaves load for further travel, paying higher move cost.
                    *   Dumping ALL reduces load to 0, paying only 100 for next move.
                    *   Cost comparison:
                        *   Option A: Move with load $L$ to Next Node. Cost $100+L$.
                        *   Option B: Dump $L$, Move with 0 to Next Node. Cost $100$.
                        *   Difference: Carrying load costs extra. So dumping is always better *if* the node accepts.
                        *   So, if we encounter a sink, dump as much as possible (up to sink capacity).
                    *   Wait, what if the sink is far? We shouldn't move away from it if we can dump closer.
                    *   So, prioritize adjacent Sink.
            2.  If `current_load == 0`:
                *   Target: Find neighbor with $h > 0$ to load.
                *   Priority: Adjacent Source. Load as much as possible.
            
            *   Wait, if we strictly only react to neighbors:
                *   At Source: Load.
                *   At Sink: Unload.
                *   Else: Random?
                *   This might get stuck in loops or inefficient paths.
                *   We need to **plan** moves.
                *   Modified Strategy:
                    *   Maintain a list of "Active Needs".
                    *   While list not empty:
                        *   Find nearest Source/Sink relative to current pos?
                        *   Actually, simpler: **BFS** to find nearest Sink if Load > 0.
                        *   **BFS** to find nearest Source if Load == 0.
                        *   Wait, we might want to visit multiple sources at once before going to sinks.
                        *   e.g. (0,0) is Sink (-10). (0,1) is Source (10).
                        *   Start at (0,0). Need Load.
                        *   Nearest Source is (0,1). Move R. Load 10.
                        *   Now Load=10.
                        *   Nearest Sink? Could be (0,0) itself (but already processed?).
                        *   If we processed (0,0), $h=0$. So no sink.
                        *   We continue looking for sink.
                        *   Say (5,5) is sink (-10). Move R...
                        *   This works.
                    
            *   **Handling Load > 0 vs Empty:**
                *   If `load > 0`: Perform BFS to find nearest neighbor $(nx, ny)$ such that $h[nx][ny] < 0$.
                    *   Wait, just "find any neighbor"? No, BFS from current to nearest sink.
                    *   Wait, what if nearest sink is blocked by sources we haven't cleared? No, sources don't block.
                    *   Wait, we might prefer a *far* sink if the current sink is "full"? No, sinks take whatever.
                    *   Wait, if there are multiple sinks, should we go to the one with largest depth?
                    *   Actually, just the nearest one is safest for move count.
                    *   BUT, we might want to pick up *another* source while traveling?
                    *   E.g. Source A --(load)----> Source B --(carry load)----> Sink.
                    *   If we are at Source A, we load. We leave Source B untouched. We arrive at Sink. Drop load.
                    *   Later we go to Source B.
                    *   This seems valid.
                *   If `load == 0`: Perform BFS to find nearest neighbor $(nx, ny)$ such that $h[nx][ny] > 0$.
                    *   Move there, load.
                
                *   **Refinement:** What if nearest sink is far, but there is a closer sink that is full? Sinks are never "full" conceptually (they just need to become 0). They can accept infinite soil? No, limited by current depth $|h|$. Once $h=0$, it's not a sink anymore.
                *   So we always look for $h < 0$.
                *   Wait, is it possible that taking a slightly longer path allows picking up more load?
                    *   Yes. If we pass through another source on the way, we load there.
                    *   However, our greedy "Nearest Source" if Load==0 might pick up that source.
                    *   So if we are moving to Sink C, and Source B is on the path:
                    *   If we treat it as just "go to C", we ignore B.
                    *   Maybe we should detect if B is on the way?
                    *   Or, simply: If we encounter a Source on the way, we stop and load (since it's a source).
                    *   Wait, the BFS gives us the path. If that path intersects a Source, we could intercept.
                    *   Actually, if `load > 0`, we are committed to delivering. But if we pass a source, we *could* load.
                    *   Loading adds to load, increasing future move cost.
                    *   Is it worth it? Only if the Source is closer than the next Source we'd visit anyway?
                    *   This suggests the BFS logic is sound: if Load > 0, don't deviate to load unless it's on the shortest path?
                    *   Wait, if we load, we increase load cost. If we skip it, we come back later.
                    *   Maybe coming back is cheaper?
                    *   Let's stick to the simplest valid greedy:
                        1.  If Load > 0: Try to move to an adjacent cell with $h < 0$. If no adjacent, pick direction towards nearest $h < 0$.
                        2.  If Load == 0: Try to move to an adjacent cell with $h > 0$. If no adjacent, pick direction towards nearest $h > 0$.
                    *   **Optimization:** When "Pick Direction Towards Nearest...", use BFS to find distance.
                    *   Also, once we land on a cell:
                        *   If $h < 0$ and Load > 0: Unload.
                        *   If $h > 0$ and Load == 0: Load.
                        *   Update grid and state.
                        *   Re-evaluate decision.
                    *   **Special Case:** We start at (0,0).
                    *   Also, what if no sinks/sources exist? Done.
                    *   What if current cell is neither (processed)? Do we just stay?
                    *   We should move to a cell with interest.
                    *   If surrounded by neutral cells (h=0), we are in danger of looping.
                    *   Need to force movement. E.g. Move in a spiral or random?
                    *   Better: Just keep searching for nearest Source or Sink globally.
                    *   If we are at a cell with $h \neq 0$, act immediately (Load/Unload).
                    *   Then determine next move.
                    *   If Load > 0, find nearest Sink.
                    *   If Load == 0, find nearest Source.
                    *   Wait, "Find Nearest Sink" might involve passing through other cells.
                    *   If we move towards it, do we stop at intermediate cells?
                    *   If an intermediate cell is a Source ($h>0$) and we have Load > 0.
                        *   Should we load?
                        *   Adding to load increases move cost.
                        *   It defers the need to revisit the Source.
                        *   Is it ever beneficial?
                        *   Suppose Source A is at (10), Source B at (12). Sink at (20).
                        *   Path: 10 -> 20. B is skipped.
                        *   Later we visit B.
                        *   Cost: Dist(A,B)+Dist(B,Source C)...
                        *   If we stopped at B and loaded, we saved a return trip to B.
                        *   Added load A->B->C.
                        *   Wait, if we are at A. Load. Move to B. Stop at B. Load. Move to C.
                        *   Vs Move A->C (skipping B). Move B->C (return/loop).
                        *   Skipping B avoids immediate loading cost (0 ops), but potentially saves a separate trip from B.
                        *   Since loading costs nothing extra (just updates height), and move cost is $100+L$, increasing $L$ is bad.
                        *   So, generally, **don't load if you have load**, unless it's on your *mandatory* path to a destination that requires high capacity?
                        *   Actually, since we can carry $10^6$, we might as well carry everything.
                        *   Wait, "The value of d must not exceed the amount of soil currently loaded". This refers to unload. For load, just positive integer.
                        *   The limit $d \le 10^6$ is generous.
                        *   Is it ever bad to load?
                        *   Loading adds cost? No, loading cost is 'd'. Unloading cost is 'd'. These sum to fixed volume cost.
                        *   The move cost is the only variable.
                        *   Move cost with load $L$: $100+L$.
                        *   Move cost empty: $100$.
                        *   So increasing $L$ adds $L$ to *every* subsequent move until unloaded.
                        *   If we pick up soil at B while going to C, we save the trip to return to B later (to pick it up) or to visit B's associated sink.
                        *   If B and C share a sink?
                        *   This gets complicated.
                        *   Simplification: Just process grid row-major or similar?
                        *   **Actually, let's look at the "Example" description again.**
                        *   Sample input generation uses Perlin noise.
                        *   Input is small.
                        *   I will implement a **simulation with lookahead/bfs**.
                        
                        *   **Proposed Implementation:**
                            *   Class `Solver`.
                            *   Data structures: `grid[N][N]`, `pos`, `load`, `ops_list`.
                            *   Helper `bfs()` to find nearest cell with $h \neq 0$.
                            *   Main Loop:
                                *   If `grid[pos] != 0`:
                                    *   If `grid[pos] > 0` and `load == 0`:
                                        *   Load `grid[pos]`. `load += grid[pos]`. `grid[pos] = 0`. Record op `+d`.
                                    *   If `grid[pos] < 0` and `load > 0`:
                                        *   Unload `abs(grid[pos])` (cap by `load`).
                                        *   `d = min(abs(grid[pos]), load)`.
                                        *   `grid[pos] += d`. `load -= d`. Record op `-d`.
                                        *   If `load == 0` after unload, continue to move decision.
                                    *   After handling current cell:
                                        *   If `load > 0`:
                                            *   Find nearest `sink` (neighbor with $h < 0$). If none, move towards nearest global sink?
                                            *   Actually, simpler: Find the nearest sink among *all* non-zero cells.
                                            *   Compute path to nearest sink.
                                            *   If `grid[current] == 0` (already handled), and we have load, we should move towards nearest sink.
                                            *   But wait, if we move to neighbor with $h < 0$, we unload.
                                            *   So just BFS to find nearest cell with $h < 0$.
                                        *   If `load == 0`:
                                            *   Find nearest `source` (cell with $h > 0$).
                                            *   BFS to find nearest source.
                                
                                *   **Handling the Path:**
                                    *   Instead of jumping straight to target, step-by-step movement:
                                        *   If Load > 0:
                                            *   Target: Nearest Sink.
                                            *   Move one step towards Target.
                                            *   Record Move op.
                                        *   If Load == 0:
                                            *   Target: Nearest Source.
                                            *   Move one step towards Target.
                                        *   Wait, stepping one by one is slow but correct.
                                        *   BFS gives the shortest path. Follow the path.
                                        
                                *   **Optimization for 100,000 Limit:**
                                    *   If we do many short moves, we might exceed.
                                    *   But $N^2=400$. Shortest path is max 40.
                                    *   Number of "trips" is number of pairs.
                                    *   If we have 200 pairs, total steps ~ 4000.
                                    *   This is fine.
                                    
                                *   **Potential Issue:** What if Nearest Sink is very far, but there is a nearby Source?
                                    *   If Load > 0, we MUST drop load to proceed effectively?
                                    *   No, we *can* carry it.
                                    *   But carrying it costs.
                                    *   If we pick up more load on the way, cost goes up.
                                    *   If we don't pick up, we pay to visit later.
                                    *   Since we can't easily predict optimal, and carrying load is expensive, **prioritizing unloading** is better than carrying.
                                    *   So: If Load > 0, always prioritize moving to Sink.
                                    *   If Load == 0, prioritize moving to Source.
                                    *   What if Nearest Source is (0,1) and Nearest Sink is (0,2)?
                                    *   Start at (0,0) with Load 0.
                                    *   Nearest Source is (0,1). Move to (0,1). Load.
                                    *   Now Load > 0.
                                    *   Nearest Sink might be (0,2). Move to (0,2). Unload.
                                    *   This seems perfectly aligned.
                                    *   What if Sink is (0,20)?
                                    *   We pick up at (0,1). Travel to (0,20).
                                    *   Along the way (say (5,5)), is there a Source?
                                    *   If there is, we encounter it.
                                    *   Our rule: "If Load > 0, prioritize Sink".
                                    *   Does "Prioritize Sink" mean we ignore Sources on the way?
                                    *   If we see a Source on the way, we can Load.
                                    *   Should we?
                                    *   If we load, we increase load, thus increasing move cost for this leg.
                                    *   However, we avoid a separate trip later.
                                    *   This trade-off is subtle.
                                    *   Heuristic: **Don't load if load > 0 unless you have no choice?**
                                    *   Or **Load only if it helps reduce total volume faster?**
                                    *   Wait, actually, if we just process row-major:
                                    *   We visit (0,1), Load.
                                    *   We continue to (0,2)...(0,N).
                                    *   If we have load, we keep carrying.
                                    *   If we reach a Sink, we drop.
                                    *   This maximizes "batching" of transports along the scan line.
                                    *   This is intuitively optimal for grid traversal (scanline).
                                    *   Let's try to incorporate a scanline logic.
                                    
                                    *   **Hybrid Strategy:**
                                        *   Start at (0,0).
                                        *   Iterate rows/cols to visit all cells? No, we can't force order if we have load.
                                        *   We need to dynamically switch.
                                        *   Logic:
                                            *   If Load == 0:
                                                *   Scan grid (BFS) for nearest Source.
                                                *   Move to Source.
                                                *   Load.
                                            *   If Load > 0:
                                                *   Scan grid for nearest Sink.
                                                *   Move to Sink.
                                                *   Unload.
                                            
                                    *   This BFS-based greedy should work well enough for AHC.
                                    *   One concern: Repeated BFS calls in a tight loop.
                                    *   Optimization: Use a Priority Queue of active sources/sinks to avoid scanning whole grid?
                                    *   Actually, since $N$ is small (20), scanning $400$ cells is trivial compared to total ops.
                                    *   Number of sources/sinks <= 400.
                                    *   Each trip reduces the set of active sources/sinks (once a source is emptied or sink filled).
                                    *   But we might not empty them completely.
                                    *   Wait, do we need to empty them completely?
                                    *   Yes, "make height 0".
                                    *   But we can split a cell's need into multiple trips.
                                    *   If Source has 50, we can take 20, leave 30.
                                    *   This complicates things.
                                    *   Better to empty fully? Or just empty as much as truck capacity allows.
                                    *   Capacity is effectively infinite ($10^6$).
                                    *   So just take full height.
                                    *   Thus, a cell is either processed (height becomes 0) or remains.
                                    *   This means each Source/Sink is visited exactly once for the purpose of clearing it.
                                    *   Total Trips $\le 200$.
                                    *   So BFS is called 200 times. Total cost negligible.
                                    
                                    *   **Corner Case:** What if grid becomes all 0s? Stop.
                                    *   What if we are at a cell that was previously a Source (now 0) but we need to pass through?
                                    *   BFS treats all cells as nodes.
                                    *   
    *   **Final Algorithm Choice:**
        1.  Parse input.
        2.  Loop while `abs(h) > 0` somewhere:
            *   If `load > 0`:
                *   Find nearest cell with `h < 0` using BFS.
                *   Move towards it.
                *   Upon reaching it:
                    *   `unload = min(load, abs(h[dest]))`.
                    *   Output `-unload`.
                    *   `h[dest] += unload`. `load -= unload`.
            *   Else (`load == 0`):
                *   Find nearest cell with `h > 0` using BFS.
                *   Move towards it.
                *   Upon reaching it:
                    *   `take = min(0?, h[dest])`. Wait, `h` is positive.
                    *   `load += h[dest]`. `h[dest] -= h[dest]` (set to 0).
                    *   Output `+load`.
        3.  Print ops.

    *   Wait, BFS logic refinement:
        *   We are at `curr`.
        *   We want to find `target`.
        *   BFS explores grid.
        *   Returns distance and path? Or just `target` coords?
        *   We need to move one step at a time.
        *   Inside the loop (for moves), we can check if `target` changed?
        *   Better:
            *   Find target coordinates once per "trip phase".
            *   Move step by step towards target.
            *   During movement, do we encounter other targets?
            *   If we encounter a Source while moving to a Sink (Load > 0), should we stop and load?
                *   This is the tricky part.
                *   If we load, we increase load.
                *   Cost implication:
                    *   Without loading: Move A->B (Load L). Arrive B, Unload. Move B->C (Empty).
                    *   With loading: Move A->X (Load L+m). Arrive X, Unload m? No X is Source.
                    *   If we stop at X, we just pass through.
                    *   If we decide to "pick up", we are effectively treating X as a new source.
                    *   If we pick up X, we extend our load.
                    *   Since we want to minimize empty moves (return trips), picking up X avoids returning to X later.
                    *   The cost of carrying extra load $m$ from A to B is $m \times dist(A,B)$.
                    *   The saving is avoiding a return trip to X.
                    *   Usually $m \times dist < dist(ReturnTrip)$.
                    *   Wait, return trip distance is at least $dist(X, \text{nearest Sink})$.
                    *   So, carrying is better if $dist(A,B) < \text{cost of separate trip}$.
                    *   But since we just implemented "Find Nearest Sink", we commit to B.
                    *   If X is on the path, and we pass it.
                    *   If we skip loading, we pay $m \times dist$ (which is 0 if we don't load). Wait, skipping loading means we don't pay $m \times dist$.
                    *   Skipping loading saves money on the way.
                    *   But forces a trip later.
                    *   Which is cheaper?
                    *   Given the randomness and constraints, **don't pick up additional sources if you already have load.**
                    *   Reason: Load adds to move cost. It is easier to return to a source later (with low load) than to carry extra load over long distances.
                    *   Exception: If the source is right next to the sink.
                    *   But BFS handles shortest paths.
                    *   So sticking to the rule: **Do not pick up sources if Load > 0** is a good heuristic. It keeps load constant, minimizing move penalties. It might increase total turns slightly but avoids "heavy truck" penalty.
                    *   Since load penalty is $L$ per step, and $L$ can be up to 40,000? No, max volume 40,000. If we carry all, $L=40,000$. Move cost 40,000 per step?
                    *   Wait, "Specify an integer d satisfying $0<d\leq 10^6$".
                    *   If we load 100000 at once, and move distance 10, cost is $100 + 100000 \approx 100,000$.
                    *   Wait, Load cost is $d$. Total load cost is constant.
                    *   Move cost is $100+d$.
                    *   If $d$ is huge, move cost is huge.
                    *   **CRITICAL:** Minimizing load amount per move is NOT the goal. Minimizing total load cost + move cost.
                    *   Load cost = $\sum d$. Constant.
                    *   Move cost = $\sum (100+d) = 100 \times Steps + \sum d_{moved}$.
                    *   So we want to minimize $\sum d_{moved}$.
                    *   This is equivalent to minimizing distance traveled *weighted by load*.
                    *   Wait, if I carry 1 unit, cost = $1 \times dist$.
                    *   If I carry 2 units, cost = $2 \times dist$.
                    *   So carrying double is twice as expensive in terms of transport cost.
                    *   Thus, **consolidating loads** (carrying more) is beneficial ONLY IF it reduces the number of steps (turns).
                    *   Wait, "Turns" constraint is 100,000.
                    *   If I carry 1 unit at a time:
                        *   I need Load op, Move op...
                        *   Move op count $\approx \text{total\_volume} \times \text{avg_dist}$.
                        *   With avg dist 10, Vol 2000 -> 20,000 ops. OK.
                        *   Wait, total volume is at most $400 \times 100 = 40,000$.
                        *   If I carry 1 unit at a time, I need 40,000 Load ops and 40,000 Unload ops.
                        *   Plus moves.
                        *   Total ops $\approx 40,000 \times 2 + \dots$. Exceeds 100,000.
                        *   **Therefore, MAXIMIZING LOAD per trip is ESSENTIAL.**
                        *   So, if `load == 0`, we should find a source and take **everything** ($h_{i,j}$).
                        *   If `load > 0`, we should just deliver. Should we pick up more?
                            *   If we pick up more, we carry more.
                            *   Does it help reduce trips?
                            *   Yes, one trip can serve multiple sinks or multiple sources.
                            *   If we pass a Source on the way to a Sink, adding its load to current load combines the trip.
                            *   It eliminates a separate trip to pick up that Source.
                            *   Since we want to reduce moves (due to 100,000 limit), **combining loads is beneficial**.
                            *   So, YES, if `load > 0`, and we encounter a Source, we should **stop and load**?
                            *   Wait, "stop" means interrupt current delivery?
                            *   Interrupting delivery means we go to Sink, deliver *partially*, return to Source, deliver rest? Or deliver fully?
                            *   If we interrupt, we lose the progress of delivery.
                            *   Actually, we can just treat the path as a list of nodes.
                            *   "Go to Nearest Sink". If we hit Source, add load.
                            *   But BFS gives a path. If that path hits a source, we can deviate or just treat it as a waypoint.
                            *   Deviating to a source means moving away from sink? Bad.
                            *   Staying on path is better.
                            *   So: Check if any cell on the shortest path to the Nearest Sink has a Source.
                            *   If so, stop there? No, if we stop, we have to return to sink later (unless we take all load and go back).
                            *   Better: Continue to Sink. Drop load. Pick up more.
                            *   But this splits the flow.
                            *   Since we want to combine, splitting flow to a sink is okay, but combining sources is key.
                            *   However, since we can load unlimited, we should just accumulate as much as possible *before* moving to a sink?
                            *   Yes.
                            *   So, if Load == 0, go to Source, take ALL.
                            *   If Load > 0, find Nearest Sink.
                            *   BUT, before moving to Sink, scan the area?
                            *   Maybe simpler: If Load == 0, just find Nearest Source and take ALL.
                            *   Then repeat.
                            *   Does this guarantee 100,000 limit?
                            *   Yes, because each trip reduces the problem significantly.
                            *   If we have 200 pairs.
                            *   Trip: Load All -> Deliver All.
                            *   If multiple sources are near a sink?
                            *   Strategy: "Scan" for a cluster of sources?
                            *   If we find a Sink. It needs -100.
                            *   Nearest Sources: A(50), B(50).
                            *   Current Pos: (0,0).
                            *   Optimal path: (0,0)->A->B->Sink.
                            *   My algorithm:
                                1. Find Nearest Source (A). Go to A. Load 50.
                                2. Load > 0. Find Nearest Sink. (Is it Sink?)
                                3. Move to Sink.
                                    *   Along the way, pass B.
                                    *   B is a Source (needs 50).
                                    *   My algo says "Don't load if Load > 0".
                                    *   So I pass B.
                                    *   Arrive Sink. Unload 50. B remains (50).
                                    *   Return to B.
                                    *   Go to Sink again. Unload 50.
                                    *   Result: 3 trips.
                                4. Alternative: Load at B too.
                                    *   If I loaded at B, I would carry 100.
                                    *   Cost of moving A->Sink (Load 50) is 50 per step.
                                    *   Cost of moving B->Sink (Load 50) is 50 per step.
                                    *   Total transport cost: $50 \times (dist(A,S) + dist(B,S))$.
                                    *   If I went (0,0)->A->B->S.
                                    *   I pick up 50 at A. Move A->B. Load 50. Now Load 100. Move B->S.
                                    *   Cost: $100 \times dist(B,S)$ (for moving with 100).
                                    *   Wait, comparing:
                                        *   Case 1: A->S (50), B->S (50).
                                            *   $A \to S$ cost: $50 \times d_A$.
                                            *   $B \to S$ cost: $50 \times d_B$.
                                            *   Total: $50(d_A + d_B)$.
                                        *   Case 2: A->B->S.
                                            *   $A \to B$: Load 50. Cost $50 \times d_{AB}$.
                                            *   $B \to S$: Load 100. Cost $100 \times d_{BS}$.
                                            *   Total: $50 d_{AB} + 100 d_{BS}$.
                                    *   Triangle inequality $d_{AS} \le d_{AB} + d_{BS}$.
                                    *   Case 1: $50 d_{AB} + 50 d_{BS} + 50 d_{BS} = 50 d_{AB} + 100 d_{BS}$?
                                        *   Wait, $A \to S$ is direct.
                                        *   $50 d_{AS}$.
                                        *   $50 d_{BS}$.
                                        *   Sum: $50 d_{AS} + 50 d_{BS}$.
                                    *   Case 2: $50 d_{AB} + 100 d_{BS}$.
                                    *   If $d_{AS}$ is short (A is near S), and $d_{AB}$ is long?
                                    *   Generally, direct is cheaper.
                                    *   However, if $d_{AS} \approx d_{AB} + d_{BS}$ (collinear), then Case 2 cost is $50 d_{AB} + 100 d_{BS}$.
                                    *   Case 1 is $50(d_{AB} + d_{BS}) + 50 d_{BS}$.
                                    *   Case 1 is worse by $50 d_{BS}$.
                                    *   WAIT.
                                    *   Case 1 means we do TWO trips from A to S and B to S.
                                        *   Trip 1: A->S. Cost $50 \times d_{AS}$.
                                        *   Trip 2: B->S. Cost $50 \times d_{BS}$.
                                        *   Plus setup moves.
                                    *   Case 2 means ONE trip covering A then B then S.
                                        *   Cost $50 \times d_{AB} + 100 \times d_{BS}$.
                                    *   Comparison: $50 d_{AB} + 100 d_{BS}$ vs $50 d_{AS} + 50 d_{BS}$.
                                    *   Subtract $50 d_{BS}$: Compare $50 d_{AB} + 50 d_{BS}$ vs $50 d_{AS}$.
                                    *   $d_{AS} \le d_{AB} + d_{BS}$.
                                    *   So $50 d_{AS} \le 50 d_{AB} + 50 d_{BS}$.
                                    *   Direct sums are usually smaller or equal.
                                    *   So consolidating load usually increases cost (because of load scaling).
                                    *   EXCEPTION: If "setup moves" are saved.
                                    *   Case 1: Setup A, Setup B.
                                    *   Case 2: Setup A, go to B, Setup B, go to S.
                                    *   If we have to return to Start for both, Case 1 saves nothing.
                                    *   But if we are already at A.
                                    *   So, carrying extra load is BAD for transport cost, but GOOD for trip consolidation.
                                    *   Which factor is stronger?
                                    *   Transport cost is multiplied by volume.
                                    *   Trip consolidation saves empty moves (100/unit step).
                                    *   Transport cost is Load $\times$ dist.
                                    *   If we carry extra 50 units over distance $D$, we add $50 \times D$.
                                    *   If we skip the trip, we save the trip to/from Source.
                                    *   If the trip to Source is distance $L$. Saving it saves $L$.
                                    *   So, if $50 \times D > L \times (\text{some factor})$.
                                    *   It's complex.
                                    *   But given the constraint 100,000 turns, and $N$ small.
                                    *   Maybe the simpler strategy (don't load if load > 0) is safer because it keeps load low, preventing overflow into the 100,000 limit due to heavy moves?
                                    *   Wait, 100,000 is quite high.
                                    *   Total volume 40,000.
                                    *   Even with worst case path length 20 per unit. $40,000 \times 20 = 800,000$.
                                    *   If we don't carry much, we exceed.
                                    *   So we MUST carry as much as possible to meet the "100,000 turns" cap.
                                    *   Actually, the "100,000 turns" is likely a soft limit or strict. Strict.
                                    *   So we MUST combine trips.
                                    *   If we combine trips, we must load multiple sources.
                                    *   So, the strategy should be:
                                        *   When finding nearest Sink (Load > 0), check if any Source is on the way (or very close to the current path).
                                        *   Or, simpler: Just find the *closest* Source to the *current* position?
                                        *   No, that leads to collecting dust (local optima).
                                        *   We need to deliver.
                                        *   Let's try to **sort** the sources by distance from current position?
                                        *   Collect sources?
                                        *   Actually, let's look at the "Turn Limit" again.
                                        *   If I output 200,000 lines, I fail.
                                        *   Max allowed lines 100,000.
                                        *   My previous analysis suggests that carrying load reduces lines (reduces empty moves).
                                        *   Moving empty costs 100. Moving with load costs 100 + Load.
                                        *   The 100 is the dominant term if Load is small.
                                        *   Wait.
                                        *   Empty Move: 100.
                                        *   Full Move: 100 + L.
                                        *   Total moves $\ge \text{Volume} \times \text{Avg Dist}$? No, $\ge \text{Volume}$.
                                        *   If I move 1 unit: Load 1, Move 1 step. Cost 1 (Load) + 101 (Move) = 102.
                                        *   Total Volume = 40,000.
                                        *   Minimum moves required to move volume = Volume? No, we move blocks.
                                        *   We can move 100 units per trip.
                                        *   So trips $\approx 400$.
                                        *   Total moves per trip $\approx 20$.
                                        *   Total moves $\approx 400 \times 20 = 8000$.
                                        *   Wait, why did I estimate 40,000 ops before?
                                        *   Ah, assuming we move 1 unit per op.
                                        *   But we load $d \le 10^6$.
                                        *   We can load 100 units.
                                        *   So we load 1 op per trip.
                                        *   Total ops $\approx 8000$.
                                        *   This is way below 100,000.
                                        *   **Wait.** So we don't need to combine loads aggressively.
                                        *   Even moving 1 unit at a time is feasible?
                                        *   If we move 1 unit at a time, ops = $40,000 \times (\text{Load} + \text{Move})$.
                                        *   Wait, Load cost is included.
                                        *   Number of Ops = Number of Loads + Number of Unloads + Number of Moves.
                                        *   Number of Loads = $\sum |h| = 40,000$ (if we do 1 by 1).
                                        *   Number of Unloads = 40,000.
                                        *   Moves = $40,000 \times \text{Dist}$.
                                        *   Total $\approx 80,000 + \dots$.
                                        *   If Dist > 1, we exceed 100,000.
                                        *   But if we load 100 units at once, we reduce Loads/Unloads by factor 100.
                                        *   Moves remain proportional to volume * dist / 100?
                                        *   No, we carry 100 units. Distance is same.
                                        *   Moves = 200.
                                        *   Total Ops = $400 \text{ (Loads)} + 400 \text{ (Unloads)} + 8000 \text{ (Moves)} \approx 9000$.
                                        *   So **grouping is mandatory** to avoid hitting Turn Limit.
                                        *   Wait, if I do 1 unit at a time:
                                            *   Load 1, Unload 1. 2 ops per unit.
                                            *   40,000 units -> 80,000 ops.
                                            *   Plus Moves.
                                            *   If avg dist 2, Moves = 80,000.
                                            *   Total 160,000 ops -> TLE/WA.
                                        *   So yes, we MUST load large amounts.
                                        *   Strategy: Always load max possible at Source.
                                        *   Always unload max possible at Sink.
                                        *   This minimizes Load/Unload ops.
                                        *   Now about moves.
                                        *   We still want to minimize moves.
                                        *   And avoid carrying unnecessary load?
                                        *   Wait, if we carry 100 units, cost is $100 \times d$.
                                        *   If we do two trips of 50 units, cost is $50 \times d + 50 \times d = 100 \times d$.
                                        *   Cost is identical for transport.
                                        *   However, doing two trips doubles the setup/movement overhead?
                                        *   Actually, doing two trips from A to B means we visit A, go to B, come back to A?
                                        *   If we combine: Visit A, load 100. Visit B. Unload.
                                        *   One trip.
                                        *   Separate: Visit A, load 50. Go to B. Unload 50. Return A. Load 50. Go to B. Unload 50.
                                        *   Return A adds moves.
                                        *   So combining trips reduces moves.
                                        *   **Conclusion:** Always combine.
                                        *   Heuristic: **When moving from Source to Sink, pass through other Sources.**
                                        *   How to implement?
                                        *   If Load == 0: Find nearest Source. Move. Load All.
                                        *   If Load > 0: Find nearest Sink. Move. Unload.
                                        *   Wait, if we encounter another Source on the way, my simple algo ignores it.
                                        *   We need to stop at that Source?
                                        *   If we stop, we interrupt the trip. We must return to the original Sink? Or find another Sink?
                                        *   If we find a nearby Sink for the new Source?
                                        *   Maybe it's better to treat "Current Path" as dynamic.
                                        *   Let's refine the Greedy Loop:
                                            *   We are at `curr`.
                                            *   If `grid[curr]` is Source ($>0$):
                                                *   Load All.
                                                *   Now Load > 0.
                                            *   If `grid[curr]` is Sink ($<0$):
                                                *   Unload as much as possible.
                                                *   Now Load updated.
                                            *   Else:
                                                *   We are in transit.
                                                *   If `load > 0`: Move to nearest Sink.
                                                *   If `load == 0`: Move to nearest Source.
                                            *   This loop continues.
                                            *   **Does this work?**
                                                *   If we have A(Source) -- B(Source) -- S(Sink).
                                                *   We are at A. Load All.
                                                *   We have load.
                                                *   Move to Nearest Sink (S).
                                                *   Path A->B->S.
                                                *   We encounter B.
                                                *   If B is Source, my logic says "Ignore B (since Load > 0)".
                                                *   We pass B. Arrive S. Unload.
                                                *   Now B is still Source.
                                                *   We are at S. Load=0 (assuming we finished).
                                                *   Move to Nearest Source (B).
                                                *   Path S->B.
                                                *   Arrive B. Load All.
                                                *   Move to S (or next Sink).
                                                *   This results in 2 trips.
                                                *   Combined trip (Stop at B, Add Load) is better?
                                                *   As analyzed before, stopping and reloading might not be strictly better in transport cost, but better in trip count.
                                                *   Given the loose Turn Limit (we are safe with combined), maybe the simple logic is fine?
                                                *   Wait, is the Turn Limit tight?
                                                *   100,000.
                                                *   If I have 200 pairs.
                                                *   Ops $\approx 200 \times 20 \approx 4000$.
                                                *   It is VERY safe.
                                                *   So I don't need aggressive combining of sources.
                                                *   Just ensuring I load MAX at sources and UNLOAD MAX at sinks is enough.
                                                *   And ensuring I don't move unnecessarily (e.g. back and forth).
                                                *   My BFS logic "Go to Nearest Source" prevents unnecessary wandering.
                                                *   So I will stick to the simpler logic: **If Load > 0, ignore Sources, only seek Sinks.**

    *   **Implementation Details:**
        *   BFS implementation needs to be correct.
        *   Directions: U, D, L, R.
        *   Grid boundaries.
        *   Output format.
        *   Handling $d$ limits? $d \le 10^6$. Max sum is 40,000. Safe.
        *   Loop termination: when sum of abs(heights) == 0.

    *   **One Edge Case:**
        *   Starting at (0,0).
        *   If (0,0) is Source, we Load.
        *   If (0,0) is Sink, we Unload (from existing load? No, start load 0).
            *   So if (0,0) is Sink, we load from somewhere else.
            *   Wait, if start at Sink, we need to bring soil.
            *   BFS will find nearest Source.
        *   If (0,0) is 0. Move to Source.

    *   **Complexity:**
        *   Outer loop: Iterates until done.
        *   Inner loop: BFS.
        *   In each iteration, we successfully complete a transaction (load/unload).
        *   Transactions = Number of Non-Zero Cells (approx).
        *   Wait, if we load 50 at A, unload 50 at B. Both become 0.
        *   So we reduce active cells by 1 or 2.
        *   Total iterations $\le N^2$.
        *   BFS cost $O(N^2)$.
        *   Total Time $O(N^4)$. $400^2 = 160,000$. Fast.

    *   **Wait**, BFS is run *each turn*? No, we run BFS to find the target, then move step by step.
        *   Step by step movement is just recording ops.
        *   We should NOT run BFS on every single step.
        *   We run BFS once per "trip" (from Source to Sink).
        *   Then we follow the path.
        *   So total ops: $O(N^2 \times \text{Trips})$. Since Trips $\le N^2$. Total $O(N^4)$.
        *   Wait, following path takes moves.
        *   Number of Moves is what we output.
        *   Moves are charged against Turn Limit.
        *   We calculated Moves $\approx 4000$.
        *   So this is perfectly safe.

    *   **One detail**:
        *   When moving from Source to Sink, the path is generated by BFS.
        *   We execute moves.
        *   We check arrival condition.
        *   We perform Load/Unload.
        *   We repeat.
        *   We need to handle the case where we reach the target and update grid.
        *   Grid update affects future searches.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <cmath>
#include <algorithm>

using namespace std;

int N;
int h[22][22];
int tx[22][22], ty[22][22]; // current
int dx[] = {0, 0, 1, -1};
int dy[] = {1, -1, 0, 0};
string dirs[4] = {"R", "L", "D", "U"};

struct State {
    int r, c, load;
};

// BFS to find shortest path from (sr, sc) to (dr, dc)
// Returns vector of coordinates in path, or empty if unreachable (shouldn't happen)
// Also returns distance
vector<pair<int, int>> bfs(int sr, int sc, int tr, int tc) {
    // Use a deque for 0-1 BFS or just regular BFS since edge weights are uniform (1)
    // We need the full path
    vector<pair<int,int>> dist(20 * 20, -1);
    vector<pair<int, int>> parent(20 * 20, {-1, -1});
    
    // Map 2D to 1D
    auto idx = [&](int r, int c) { return r * N + c; };
    
    queue<int> q;
    q.push(idx(sr, sc));
    dist[idx(sr, sc)] = 0;
    parent[idx(sr, sc)] = {-1, -1};
    
    while(!q.empty()) {
        int u = q.front();
        q.pop();
        int r = u / N;
        int c = u % N;
        
        if (r == tr && c == tc) {
            // Reconstruct path
            vector<pair<int,int>> path;
            int curr = u;
            while(parent[curr].first != -1) {
                path.push_back({parent[curr].first + 1, parent[curr].second + 1}); // parent stores prev
                curr = parent[curr].first; 
            }
            reverse(path.begin(), path.end());
            // Actually BFS stores parent of curr. Let's rewrite properly.
            // Wait, parent should store (prev_r, prev_c) for reconstruction.
            break; // Found
        }
        
        // Wait, my parent logic above was confusing. Let's do standard BFS.
        for(int i=0; i<4; ++i) {
            int nr = r + dx[i];
            int nc = c + dy[i];
            if(nr >= 0 && nr < N && nc >= 0 && nc < N) {
                int v = idx(nr, nc);
                if(dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    parent[v] = {r, c};
                    q.push(v);
                }
            }
        }
    }
    // Reconstruct path
    if (idx(tr, tc) != -1 && dist[idx(tr, tc)] != -1) {
        vector<pair<int,int>> path;
        int curr = idx(tr, tc);
        while(curr != idx(sr, sc)) {
            path.push_back({parent[curr].first, parent[curr].second});
            curr = parent[curr].first;
        }
        reverse(path.begin(), path.end());
        return path;
    }
    return {};
}

int main() {
    cin >> N;
    int sum_h = 0;
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            cin >> h[i][j];
            sum_h += abs(h[i][j]);
        }
    }
    
    int cur_r = 0;
    int cur_c = 0;
    int cur_load = 0;
    string result = "";
    // Although we can append strings, printing directly is fine, but we need to output count T first?
    // "Let T be the number of operation turns. ... output s_0 ... s_{T-1}"
    // Wait, the example output format in problem statement:
    // "output to Standard Output in the following format: s_0 ... s_{T-1}"
    // It does NOT ask to print T explicitly as a separate line. Just the lines.
    // Wait, "Then, output to Standard Output in the following format: s_0 ..."
    // So just the sequence.
    
    // Optimization: Store commands and print.
    vector<string> ops;
    
    // Initial check: handle if h[0][0] is source/sink immediately?
    // The loop logic should handle it if we check h[cur_r][cur_c] inside the main loop.
    
    while(sum_h > 0) {
        // Check if current cell needs attention
        bool processed_current = false;
        if(h[cur_r][cur_c] > 0 && cur_load == 0) {
            int amount = h[cur_r][cur_c];
            cur_load += amount;
            h[cur_r][cur_c] = 0;
            ops.push_back(to_string(amount)); // Format +d
            sum_h -= amount; // Wait, sum_h tracks sum of abs(h). Correct.
            processed_current = true;
        } else if(h[cur_r][cur_c] < 0 && cur_load > 0) {
            int amount = -h[cur_r][cur_c];
            int to_unload = min((long long)amount, (long long)cur_load);
            h[cur_r][cur_c] += to_unload;
            cur_load -= to_unload;
            if(h[cur_r][cur_c] != 0) {
                 // sum_h needs update: sum_h = sum(|h|).
                 // Since h decreased in magnitude, sum_h decreases.
                 // Recalc sum_h? Or track delta?
                 // Recalc is O(N^2). Safe to do periodically or at end.
                 // Actually tracking sum_h is safer for loop condition.
            }
            ops.push_back("-" + to_string(to_unload));
            sum_h -= to_unload; // Since |h| reduced by to_unload
            // Wait, h was negative. abs(h) decreased. So sum_h decreases by to_unload.
            processed_current = true;
        }
        
        // If processed, we might still have load or not.
        // If not processed, we are either in transit or idle.
        // If idle (h=0, load=0), we need to find target.
        
        // Determine target
        int target_r, target_c;
        bool target_found = false;
        
        if (!processed_current || (processed_current && cur_load == 0)) {
             // If we just loaded, cur_load > 0.
             // If we just unloaded, cur_load might be > 0 or 0.
             // If cur_load > 0, we want Sink.
             // If cur_load == 0, we want Source.
             
             if(cur_load > 0) {
                 // Find nearest Sink (h < 0)
                 // Optimization: Just check neighbors first.
                 // If not found in neighbors, run BFS.
                 target_found = false;
                 int dist_sink = -1;
                 
                 // Quick check neighbors
                 for(int i=0; i<4; ++i) {
                     int nr = cur_r + dx[i];
                     int nc = cur_c + dy[i];
                     if(nr >= 0 && nr < N && nc >= 0 && nc < N && h[nr][nc] < 0) {
                         target_found = true;
                         target_r = nr;
                         target_c = nc;
                         dist_sink = 1;
                         goto break_search;
                     }
                 }
                 
                 if(!target_found) {
                     // BFS
                     // ... (reuse bfs logic)
                     vector<pair<int, int>> path = bfs(cur_r, cur_c, 0, 0); // Placeholder
                     // Need actual implementation inside here.
                     // Let's refactor bfs to return bool and set coords.
                 }
                 break_search:;
                 
             } else {
                 // Find nearest Source (h > 0)
                 target_found = false;
                 int dist_src = -1;
                 for(int i=0; i<4; ++i) {
                     int nr = cur_r + dx[i];
                     int nc = cur_c + dy[i];
                     if(nr >= 0 && nr < N && nc >= 0 && nc < N && h[nr][nc] > 0) {
                         target_found = true;
                         target_r = nr;
                         target_c = nc;
                         dist_src = 1;
                         goto break_src;
                     }
                 }
                 if(!target_found) {
                     // BFS
                 }
                 break_src:;
             }
        }
        // Wait, I mixed logic.
        // Let's unify.
        
        // Calculate next move.
        // If Load > 0 -> Target Sink.
        // If Load == 0 -> Target Source.
        
        // If current cell is a Source and Load == 0, we just process it (handled above).
        // After processing, if Load > 0, we need Sink.
        // If after processing, Load == 0 (maybe we were at Sink and became empty?), we need Source.
        // Wait, if we were at Source, Load becomes > 0.
        // If we were at Sink, Load decreases.
        // If we are at (0,0) with 0, and h=0, we move.
        
        // Let's clean this logic.
        
        // Main loop condition: sum_h > 0
        
        // 1. Handle current cell interaction if needed
        if(h[cur_r][cur_c] > 0 && cur_load == 0) {
             int amt = h[cur_r][cur_c];
             cur_load += amt;
             h[cur_r][cur_c] = 0;
             ops.push_back("+" + to_string(amt));
             sum_h -= amt; // careful: sum_h tracks total abs. h becomes 0.
                          // Previously abs was amt. New is 0. Diff is amt. Correct.
             continue; // After loading, we should move.
        }
        if(h[cur_r][cur_c] < 0 && cur_load > 0) {
             int amt = -h[cur_r][cur_c];
             int unload = min((long long)amt, (long long)cur_load);
             cur_load -= unload;
             h[cur_r][cur_c] += unload;
             // sum_h decreases by unload.
             // |h_new| = |h_old| - unload.
             // So sum_h -= unload.
             sum_h -= unload;
             ops.push_back("-" + to_string(unload));
             // If h[cur_r][cur_c] is still < 0 (not emptied), we might have load > 0 still (if load > |h|).
             // We are now at a cell with h<0.
             // Should we stop here? Or move to another Sink?
             // If load > 0, we have residual load to dump somewhere else.
             // We should move to nearest Sink.
             // If load == 0, we need to find Source.
             // So we fall through to move logic.
             continue; // Process move logic next
        }
        
        // 2. Move Logic
        int next_move = -1;
        if(cur_load > 0) {
            // Find nearest Sink
            int br = cur_r, bc = cur_c;
            bool found_s = false;
            // BFS for nearest h < 0
            // Implement BFS inline or helper
            // For brevity and correctness, let's write a dedicated BFS
            // BFS queue stores (r, c, dist)
            
            // Optimization: Since we run this frequently, maybe precompute?
            // No, grid changes.
            
            // Inline BFS for nearest Sink
            vector<pair<int,int>> q_vec;
            q_vec.reserve(400);
            q_vec.push_back({cur_r, cur_c});
            // Visited array
            vector<vector<bool>> vis(N, vector<bool>(N, false));
            vis[cur_r][cur_c] = true;
            int qr = cur_r, qc = cur_c; // head
            
            // Standard BFS
            int pr, pc;
            vector<pair<int,int>> path;
            bool found = false;
            
            // Using a simple queue index
            int head = 0;
            while(head < q_vec.size()) {
                int cr = q_vec[head].first;
                int cc = q_vec[head].second;
                head++;
                
                if(cr != cur_r || cc != cur_c) {
                     // Check if sink
                     if(h[cr][cc] < 0) {
                         found = true;
                         target_r = cr;
                         target_c = cc;
                         // Reconstruct path?
                         // Need parent pointers.
                         // Re-implementing BFS with parents
                     }
                }
            }
            // Okay, let's just implement a proper BFS helper function inside the solution class or globals.
        } else {
            // Find nearest Source (h > 0)
        }
    }
    return 0;
}
```

Re-think BFS structure. I'll put it in a helper function.

Also, checking `visited` array needs initialization every time. `vector` allocation is fast enough for $20 \times 20$.

Wait, one edge case in "Move Logic":
After loading (at Source), we continue.
But we are currently AT the Source. We need to leave it.
We need to move to Sink.
BFS will give path from Source to Sink.
If path is length 0? (Source IS Sink? Impossible since h!=0 implies sign diff? No, h is int. Sign diff means opposite signs).
Yes.
So we move 1 step towards target.
Update current position.
Repeat.
Wait, if we are at Source, and we load. `h` becomes 0.
Now Source is gone (h=0).
We move to Sink.
Upon reaching Sink, we unload. `h` increases.
If `h` becomes 0, Sink is gone.
So yes, `h` acts as state.

Wait, optimization:
If we are carrying load, and we arrive at a cell that was a Source (but we passed it, or it's on the path), we should stop?
My logic "If Load > 0, find nearest Sink" means I ignore other Sources.
Is this suboptimal?
If I have Source A, Source B, Sink C.
I am at A. Load 100.
Target B (on path to C)? No, A is Source.
Suppose A and B are close, C is far.
If I carry from A to C, I might pass B.
Should I stop and load at B?
As discussed, stopping and loading reduces trips but increases transport cost.
Given constraints, ignoring B is risky?
No, wait. If I ignore B, I come back later.
Cost of ignoring B:
Trip 1: A -> C (Full load 100).
Trip 2: B -> C (Full load 100).
vs
Trip 1: A -> B (Load 200) -> C.
Wait, if I stop at B, I must unload A's load? No, just continue.
If I pass B without stopping:
Load 100 from A. Move A->B (cost 100*L). Arrive B.
From B, move to C.
Load 100 from B.
Total Load 200.
Wait, I need to distinguish "Carrying Load" vs "Loading at B".
If I am at A with Load 100.
I move to B.
At B, I load 100. Now Load 200.
Move B -> C.
Cost: A->B (with 100). B->C (with 200).
Total transport cost: $100 \times d(A,B) + 200 \times d(B,C)$.
Vs Separate:
A->C (with 100): $100 \times d(A,C)$.
B->C (with 100): $100 \times d(B,C)$.
Difference: $200 d(B,C) - (100 d(A,C) + 100 d(B,C)) = 100 d(B,C) - 100 d(A,C)$.
By triangle inequality $d(A,C) \le d(A,B) + d(B,C)$.
So $100 d(B,C) - 100 d(A,C) \ge 100 d(B,C) - 100 (d(A,B) + d(B,C)) = -100 d(A,B)$.
So separating is better by at most $100 d(A,B)$.
And separating saves the cost of increasing load on B->C segment.
Actually, if I combine, I increase load.
So separating (not loading at B) is almost always better for transport cost.
AND it simplifies logic.
So sticking to **Don't load if Load > 0** is the correct greedy strategy for transport cost minimization.
(Though trip count argument suggested combining, but transport cost penalty is high: multiplying load by distance).
Since the limit is 100,000 ops, and volume is 40,000.
If we don't combine, we do many small trips.
Is it possible to exceed 100,000 with small trips?
If we do 1 unit trips:
Ops = Load (1) + Move (dist).
Sum of loads = 40,000.
If dist=50 (max Manhattan on 20x20 is 40).
Ops $\approx 40,000 \times 50 = 2,000,000$.
EXCEEDS LIMIT.
So we MUST use large loads (up to 100 or more).
With max height 100, we can group sources.
So combining sources (A and B) is NECESSARY to reduce turn count.
Thus, my "Don't load if Load > 0" logic is WRONG because it leads to separate trips for nearby sources.
I SHOULD load nearby sources.
How to implement "Load nearby sources"?
Strategy: **Collect all reachable Sources into one trip.**
Or simpler: If we are at a Source, load. If we encounter another Source, stop and load.
This increases load.
Wait, does this violate the "Load if Load > 0" assumption?
The assumption was "Don't load if Load > 0 to minimize transport cost".
But we established that we MUST combine to satisfy turn limit.
The turn limit is a HARD constraint.
The cost minimization is the objective.
Satisfying turn limit is primary (else WA/TLE).
So I must ensure I don't exceed 100,000 ops.
If I have 200 Sources of size 100.
Separated: 200 trips. Each trip length ~20. Ops ~ 200 * (Load+Move).
Wait, if I combine A and B:
Visit A. Load 100. Move to B. Load 100. Move to Sink.
Ops:
A (Load) + Path A->B (Moves) + B (Load) + Path B->Sink (Moves).
Vs Separate:
A (Load) + Path A->Sink (Moves) + Return A + Load A?? No, Source consumed.
Vs Separate:
A (Load) + Path A->Sink (Moves) + B (Load) + Path B->Sink (Moves).
Note that Separate requires 2 sets of (Load->Sink).
Combined: Requires 1 set of (SourceChain->Sink).
The SourceChain involves A->B.
Separate involves A->Sink AND B->Sink.
If B is on the path from A to Sink, then A->B is part of A->Sink.
So Separate = (A->Sink) + (B->Sink).
Combined = (A->B) + (B->Sink) + [Stop at B and Load].
Cost (Load+Move) for Separate:
Trip A: Load 100. Move 100*d(A,S).
Trip B: Load 100. Move 100*d(B,S).
Total Load ops: 2.
Total Move ops: 100*d(A,S) + 100*d(B,S).
Cost (Load+Move) for Combined (assuming B is on path):
Trip A-B-S:
Move A->B. Carry 100. (Note: A is on path).
Wait, if we start at A. Load 100.
Move A->B. Pass through B? No, B is a source.
If B is a source, we are "on path".
We stop at B. Load 100.
Move B->S. Carry 200.
Total Load ops: 2. (A, B).
Total Move ops:
d(A,B) (with 100) + d(B,S) (with 200).
Comparing:
Sep: $d(A,S) + d(B,S)$. (Assuming Load cost is same, 100*dist? No, transport cost is proportional to load).
Sep Transport Cost: $100 \cdot d(A,S) + 100 \cdot d(B,S)$.
Comb Transport Cost: $100 \cdot d(A,B) + 200 \cdot d(B,S)$.
Diff: Comb - Sep = $100 d(A,B) + 100 d(B,S) - 100 d(A,S)$.
Since $d(A,S) \le d(A,B) + d(B,S)$,
Comb - Sep $\le 100(d(A,B) + d(B,S)) + 100 d(B,S) - 100 d(A,B) - 100 d(B,S)$ ?? No.
$d(A,S)$ is approx $d(A,B) + d(B,S)$ if collinear.
If collinear: $100 d(A,B) + 100 d(B,S) - 100 (d(A,B) + d(B,S)) = 0$.
Wait, this implies transport costs are IDENTICAL if B is on the path and we combine.
What about the setup moves?
Separate: A->S. Then B->S.
We need to go from S to B? Or from A to B to S?
If we visit A, then S, then B, then S.
Path: A -> S -> B -> S.
Moves: $d(A,S) + d(S,B) + d(B,S)$.
Combined: A -> B -> S.
Moves: $d(A,B) + d(B,S)$.
Clearly Combined is shorter by $d(S,B)$!
And Transport Cost is same.
And Load Ops is same.
**So COMBINING is ALWAYS BETTER if we are chaining sources.**
The only reason NOT to combine is if B is NOT on the path to S.
But then we'd have to visit B anyway.
If we visit B on the way, we save the return trip.
So, we should adopt the strategy: **While moving from a Source to a Sink, check if any other Source lies on the shortest path.**
Actually, just simpler:
If we have Load > 0, move to Nearest Sink.
During movement, we move 1 step.
If we encounter a Source cell ($h>0$):
    Load ALL from it.
    (This is safe because we are moving towards a Sink, so we don't deviate much).
Wait, if we Load ALL, we might exceed the optimal load? No, optimal is infinity.
Does loading at B (which is not on path) hurt?
If we divert to B? No, just load if we are "near" or "on path".
But since I can't afford to check "is it on path" dynamically (expensive).
Maybe just **Sort Sources by distance from current Sink?**
No, that's static.
Simplest heuristic that works well:
Use BFS to find Nearest Sink.
Follow path.
If we encounter a Source on the path:
    Check if we should load.
    If we load, we add to `cur_load`.
    We continue to Sink.
    We don't unload (unless h<0).
    So we effectively pick up payload.
    This is safe.
    Wait, if we encounter Source, we should stop there?
    If we don't stop, we pass through.
    If we pass through and load?
    "Pass through" implies we are on the grid.
    If I implement this:
    Loop over moves in BFS path.
    Let `next_step` be the cell.
    Before moving to `next_step`:
       Check if `grid[next_step]` is Source ($h > 0$).
       If yes:
          Load `grid[next_step]`.
          `grid[next_step] = 0`.
          `cur_load += amount`.
          Ops `+d`.
          `sum_h -= amount`.
       Move to `next_step`.
    This logic is perfect. It captures everything.
    Wait, what if we are at a Sink?
    If `grid[next_step]` is Sink ($h < 0$).
       Unload.
       Check `cur_load`. If `cur_load` > `abs(h)`, unload `abs(h)`.
       `cur_load -= amount`.
       `grid[next_step] += amount`.
    So the logic unifies:
    At any step in path:
      Check `grid[next_step]`.
      If `h > 0` (Source): Load.
      If `h < 0` (Sink): Unload.
      (Handle bounds for `cur_load`).
      Move.
    Wait, if we are at Source A, and path goes A->B->S.
    A is start. Load.
    Move to B.
    At B (Source): Load.
    Move to S.
    At S (Sink): Unload.
    This works beautifully.
    It handles batching naturally.
    It only performs "unnecessary" loading if we load at B then go to S.
    But as derived, this is optimal if B is on path.
    What if B is NOT on path? (e.g. Source B is off-path).
    Then BFS will NOT find B.
    So we never encounter B.
    So we don't load B.
    We go A->S.
    Then we search for next target.
    This seems robust.
    The only risk is if we have a choice of multiple Sinks.
    We pick Nearest.
    Is it possible that a farther Sink is better? (To capture a Source in between?)
    If B is Source.
    Path 1 (to Near S): passes A. Misses B.
    Path 2 (to Far S): passes A and B.
    If we pick Path 2, we capture B.
    This is a better heuristic.
    But calculating this requires knowing all paths?
    BFS finds shortest path.
    Shortest path is preferred to minimize transport cost.
    Capturing B is secondary benefit.
    Given constraints, shortest path + on-path loading is excellent.

    **Plan:**
    1. BFS to find Nearest Sink if `cur_load > 0`.
    2. BFS to find Nearest Source if `cur_load == 0`.
    3. Follow path.
    4. Along path, handle interactions (Load/Unload).
    5. Repeat until `sum_h == 0`.

    One small issue: BFS finds a path from `curr` to `target`.
    If `target` is the same cell as `curr` (dist 0)?
    Happens if we are at Source and Load 0? Handled by step 2.
    If we are at Source, Load > 0 (after load). We run BFS to Sink. Dist > 0.
    Path returned.
    Iterate path.
    First node in path might be `curr`?
    BFS path usually excludes start.
    My implementation: `vector` stores `{r, c}`.
    If path contains cells.
    Check if cell is Source or Sink.
    Load/Unload.
    Move.
    Update `cur_r`, `cur_c`.
    
    Wait, if I load at `curr` (A), then `curr` becomes 0.
    Then I move to next cell.
    If next cell is B (Source). Load.
    This is consistent.

    One catch: The problem says `specify integer d... load d`.
    I should load the ENTIRE height of the cell.
    The height is `h[i][j]`.
    If I load, I reduce `h[i][j]` to 0.
    This is optimal.
    Partial loading is rarely optimal.
    So I will always load full amount.
    Same for unload.
    
    Final check on constraints.
    Max turns 100,000.
    With $N=20$, BFS is cheap.
    Ops will be roughly proportional to sum of $|h| / \text{capacity}$.
    Max capacity $10^6$. We treat it as infinite.
    Actually we load $h_{i,j} \le 100$.
    If we have many small cells (value 100).
    We might visit all of them.
    If we visit 400 cells, and each trip is distance 20.
    Ops = $400 \times 20 + 400 \approx 8400$.
    Very safe.

```cpp
/*
    Competitive Programming Solution
    Problem: AtCoder Heuristic Contest AHC034 - Leveling
*/

#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <cmath>
#include <algorithm>
#include <cstring>

using namespace std;

int N;
int h[22][22];
long long sum_abs_h;
int cur_r, cur_c;
int cur_load;

int dx[] = {0, 0, 1, -1};
int dy[] = {1, -1, 0, 0};

struct Point {
    int r, c;
};

// BFS to find shortest path from (start_r, start_c) to (target_r, target_c)
// Returns path as vector of Points. 
// If target is not found or unreachable, returns empty vector.
vector<Point> get_path(int start_r, int start_c, int target_r, int target_c) {
    // If same position, return empty or handle specially
    if(start_r == target_r && start_c == target_c) return {};
    
    vector<Point> parent(22 * 22, {-1, -1});
    int dist_map[22][22];
    memset(dist_map, -1, sizeof(dist_map));
    
    queue<int> q; // Stores encoded index r*N + c
    int start_idx = start_r * N + start_c;
    q.push(start_idx);
    dist_map[start_r][start_c] = 0;
    parent[start_idx] = {-1, -1};
    
    vector<Point> path;
    while(!q.empty()){
        int curr_idx = q.front();
        q.pop();
        int r = curr_idx / N;
        int c = curr_idx % N;
        
        if(r == target_r && c == target_c) {
            // Reconstruct
            int p = curr_idx;
            while(p != -1 && parent[p] != {-1, -1}) {
                path.push_back({parent[p].first, parent[p].second}); // Wait, parent stores index?
                // No, parent should store the *coordinate* or index.
                // Let's store index in parent.
                p = parent[curr_idx].first; // Wait, parent vector size mismatch.
                // parent should be indexed by 0..N*N-1
                // I initialized parent as vector<Point>. That's fine.
            }
            reverse(path.begin(), path.end());
            return path;
        }
        
        for(int i=0; i<4; ++i){
            int nr = r + dx[i];
            int nc = c + dy[i];
            if(nr >= 0 && nr < N && nc >= 0 && nc < N) {
                int nidx = nr * N + nc;
                if(dist_map[nr][nc] == -1) {
                    dist_map[nr][nc] = dist_map[r][c] + 1;
                    parent[nidx] = {-1, -1}; // Need to fix parent storage.
                    // parent stores coordinates for easier access?
                    // parent[nidx] = {r, c};
                    q.push(nidx);
                }
            }
        }
    }
    return {};
}
```
Wait, my BFS reconstruction logic was messy in thought.
Let's write clean BFS.
Parent array `pair<int, int>` mapping from `index` to `parent_index`.
Wait, `parent` vector of indices is easier. `parent[index]` gives previous index.
Then reconstruction is trivial.

Correct BFS logic:
```cpp
vector<pair<int, int>> bfs(int sr, int sc, int tr, int tc) {
    vector<int> parent_idx(22 * 22, -1);
    int dist_idx[22 * 22];
    // memset not safe with large stack if not static/global? N=22 is small. Stack ok.
    // But let's use vector.
    for(int i=0; i<(N*N); ++i) dist_idx[i] = -1;
    
    queue<int> q;
    q.push(sr * N + sc);
    dist_idx[sr * N + sc] = 0;
    parent_idx[sr * N + sc] = -1;
    
    int target_idx = tr * N + tc;
    
    while(!q.empty()){
        int u = q.front();
        q.pop();
        int r = u / N;
        int c = u % N;
        
        if(r == tr && c == tc) {
            // Reconstruct
            vector<pair<int, int>> res;
            int curr = u;
            while(curr != sr * N + sc) {
                res.push_back({parent_idx[curr], curr}); // Wait, storing coordinates or just indices?
                // Let's store parent index.
                // But I need to convert index to coord?
                // Actually res should be coordinates.
                // parent_idx[curr] is the index of prev cell.
                // Wait, I need to reconstruct path from target backwards to start.
                // But target is not guaranteed to be in queue yet? Yes it is when checked.
                // But wait, if I check `if(r == tr && c == tc)` here, `u` is target.
                // So I reconstruct from u.
                
                // Let's just store coordinates in parent array to save conversion later.
                // But `parent_idx` array name implies index.
                // Let's use `int prev_r[22*22]`, `int prev_c[22*22]`.
                curr = parent_u[curr] ? : -1;
            }
            reverse(res.begin(), res.end());
            return res;
        }
    }
    // Wait, simpler: standard reconstruction.
    return {}; // placeholder
}
```
Actually, I'll implement a compact BFS function inside the class or global.

Also, `vector<pair<int, int>>` to store path.
Wait, `vector` resize inside loop is bad.
Just use `static` or allocate once? No, small enough.

Wait, `get_path` returns a path excluding start node (since start is `sr, sc`).
My `while` loop checks `r==tr...`. `u` is the target node index.
Reconstruction loop: `curr = u`. `prev = parent[curr]`. `res.push_back({r, c})`. Wait, `u` is target.
I need to push target node.
Wait, my logic:
```cpp
while(curr != start_node) {
    res.push_back({curr_r, curr_c}); // current is child of parent?
    // No, parent stores previous.
    // Let's trace:
    // q pops u.
    // children v pushed. parent[v] = u.
    // Found target t.
    // path starts at parent[t], then parent[parent[t]], ... until start.
    // This gives path from start->...->t reversed.
}
```
Correct.
Also need `vector<pair<int,int>> parent_coords(22*22)`.
Initialize `parent_coords` with -1/-1.

Wait, `22*22` is 484.
Inside loop, check `if (r == tr && c == tc)`.
Then reconstruct.
This logic assumes `tr, tc` is in queue. It is.
Wait, `if(r==tr...)` is inside the pop loop.
So `u` is the target.
Reconstruction starts from `u`.
Go to `parent[u]`, etc.
Until we hit start node (where parent is -1 or we stop).
Add nodes to `path`.
Reverse.
Return `path`.

Wait, if `tr, tc` is not found? (Impossible in connected grid).
If `start == target`, return empty.

One more thing: The problem requires `s_0`... output.
I'll store strings and print at end.

Input parsing:
N fixed 20? No, "In all test cases, N is fixed at 20".
But input format gives N.
Wait, "In all test cases, N is fixed at 20."
But I should read N anyway.

Turn counting:
Loop limit check? No, I can run until sum_h=0.
But to prevent infinite loops (bug), add safety counter.

Correctness of `sum_abs_h` update:
When `h[i][j]` changes from `x` to `y`.
Change in `sum_abs_h` = `abs(y) - abs(x)`.
If `x > 0`, `y` usually 0 (if fully loaded). Change `-x`.
If `x < 0`, `y` usually 0 (if fully unloaded). Change `-(-x) = x`. (Since x negative, change is positive? No, sum abs decreases).
If `x < 0` (e.g. -5), unloading 3 makes it -2.
Abs change: |-2| - |-5| = 2 - 5 = -3.
So `sum_abs_h` decreases by 3.
Correct.

Double check turn count logic again.
If I output 100,000 strings, it's OK.
Memory limit? Standard.
Time limit? Standard (likely 1-2s per case or cumulative).
Since I do BFS (O(N^2)) per trip.
Number of trips $\le$ Sum of non-zero heights?
Actually, trips correspond to clearing a Source or Sink?
If I visit a Source (h>0) -> Load.
It is cleared.
If I visit a Sink (h<0) -> Unload.
It might still have h<0.
Wait, if I unload partially.
Then it remains a Sink.
This implies I might visit it multiple times.
But I only unload if I have load.
If I find nearest Sink, I unload as much as possible (up to load and sink depth).
If `cur_load` is exhausted, I stop there (load becomes 0) and then find Source.
If I still have load (sink not deep enough), I continue to another Sink.
Wait, if I have load > 0 and current cell is Sink.
I unload.
If I still have load, I continue to nearest Sink?
My code will check `h[cur] < 0`.
If I just arrived and unloaded, `h` might be 0 or < 0.
If `h < 0` and `cur_load > 0`.
Should I unload again?
Yes, `cur_load > 0` triggers "Find Nearest Sink".
If `curr` is a sink (h < 0), the BFS for nearest Sink will find `curr` itself (dist 0).
So I will stay there?
Infinite loop?
Ah, BFS for Nearest Sink will return empty path if `dist == 0`.
My logic: `if(start == target) return {}`.
Then `next_step` loop won't run.
I need to detect this.
If `path` is empty, and `h[cur] < 0` and `load > 0`.
Then we are ALREADY at the sink.
Action: Unload MORE.
So update `h`, `load`, add op, continue.
Then `h` increases (becomes less negative).
Eventually `h` becomes 0.
Then `h[cur]` is not sink.
So BFS will find new target.
This handles partial unloading.
Wait, if `h` becomes 0. `cur_load` might still be > 0 (if we didn't unload full load).
Then we need to move to new Sink.
If `load` becomes 0, we look for Source.
This seems logically consistent.
Wait, "Find Nearest Sink" when `load > 0`.
If `curr` is Sink (and `h<0`), we are already there.
So we should Unload.
My loop structure:
1. Check `h[cur] > 0 && load == 0` -> Load.
2. Check `h[cur] < 0 && load > 0` -> Unload.
3. If neither or after above, determine target.
   If `load > 0` -> Target Sink.
   If `load == 0` -> Target Source.
   (If `load > 0` and `h[cur] < 0`, target found is `curr`. Path empty).
   If path empty, stay and process? No, process "Unload" happens in step 2.
   Wait, if step 2 happened, we might still have `load > 0`.
   Then we enter step 3.
   If `load > 0`, target is `curr` (Sink).
   Path is empty.
   Loop for moves won't run.
   We go to next iteration.
   In next iteration, `h[cur]` might still be `< 0`.
   We repeat step 2.
   This effectively dumps all soil.
   Is this optimal?
   It avoids extra moves. It unloads max possible.
   But wait, if I unload 5 units, I might prefer to carry 5 units further?
   No, carrying load costs more.
   Unloading early is better.
   So yes, just dump until `load` empty or `sink` filled.
   Then continue loop.

Wait, one case:
`h[cur] > 0` and `load > 0`.
Should I load? No.
Should I unload? No.
Should I move?
If `load > 0`, move to Sink.
If `load == 0`, move to Source.
If `cur` is Source but `load > 0`, we pass through.

Final check on BFS.
If `dist` is 0, return empty.
Wait, if `start == target`, `parent` is uninitialized?
Need to handle start==target separately or check `dist` before entering.

One edge case: `sum_h` loop.
If grid becomes all 0. Break.

One optimization: `sum_abs_h` calculation.
Doing it every step is slow?
Summing 400 ints is negligible.
Wait, I only need `sum_h` to terminate.
Can I track it manually?
Yes. `sum_h` variable.
When `h` changes by `delta`. `abs(delta)` decreases sum?
If `h` decreases from `x` to `y` (unloading). `h` increases algebraically.
If `h < 0`, `h` becomes `h + delta`.
Change in `|h|`:
New abs is `|h + delta|`. Old is `|h|`.
Decrease = `|h| - |h + delta|`.
Just subtract this from `sum_h`.
Wait, `delta` comes from `cur_load`.
`load` decreases.
`h` increases.
So `sum_h` decreases.
Similarly for loading: `h` decreases.
`|h|` decreases.
So `sum_h` decreases.
Tracking `sum_h` is easy.

Wait, max value of `d`.
`0 < d <= 10^6`.
Since max `h` is 100, we never use max $d$.
Unless we accumulate load > 100? Yes.
So `d` can be larger than 100.
My code uses `to_string(amt)` where `amt` is `h` or `cur_load`.
Safe.

Wait, one detail:
The problem says "load d units ... decreases height ... by d".
If `h` becomes negative?
Yes, "The height can become negative."
But we want "height of every square 0".
My strategy forces height to 0.
Wait, if I load 100 into `h=-50`.
Height becomes -150? No.
If I Load at `h=-50` (Sink), it's invalid.
Load is only at `h>
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=1250)

````
I need to solve this problem where I need to level an N x N grid of land with a dump truck, minimizing the total cost.

Key observations:
1. Total height is 0, so soil that needs to be added must come from places with excess soil
2. Loading costs d per unit and unloading costs d per unit (both same cost)
3. Moving costs 100 + d units (where d is current load) - this is the expensive part!

The trick here is that moving costs depend on how much soil you're carrying. This means:
- Carrying more soil while moving is more expensive
- We should try to keep the truck light when moving between far locations
- The fixed cost of moving (the 100) means we should minimize unnecessary moves

Strategy:
1. First pass: Remove all negative heights (hills) by loading soil and returning it somewhere
2. Move to each hill location and unload to raise it up
3. Since total height is 0, what goes down in hills comes back as elevation elsewhere

Optimization:
- Instead of removing all soil then reusing it, remove soil from hills only
- Keep some positive soil nearby to avoid moving the truck empty across the whole grid
- When returning soil, unload partially rather than completely - keep the truck at a minimum load to reduce move costs

Since N=20 is small and heights are ≤100, a simple greedy approach works well.
</think>

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

const int N = 20;
int h[N][N];
int curr[20] = {0}; // Track remaining positive heights needing elevation
int cnt[20] = {0};  // Count of cells needing specific positive heights
int ans[100005];    // Operations (store for output)
int T = 0;          // Number of operations

bool has_neg = false;

void out(int x, const char* s) { ans[T++] += x, printf("%s %d\n", s, x); }
void out(char c) { ans[T++], printf("%c\n", c); }

void solve() {
	for (int i = 0; i < N * N; ++i)
		curr[i / N], cnt[i % N],
			h[i / N][i % N] > 0 ? curr[i / N]++, cnt[i % N] : 
				h[i / N][i % N] < 0 && !has_neg, has_neg = true, h[i / N][i % N] *= -1;

	while (T < 100000) {
		if (!has_neg) break;
		
		int loc = -1;
		for (int i = 0; i < N * N; ++i) if (curr[i]) { loc = i; break; }
		if (loc == -1) return;
		
		int row = loc / N, col = loc % N;
		int prev_r = -1, prev_c = -1, min_dist = N + 1;
		
		for (int r = 0; r < N; ++r)
			for (int c = 0; c < N; ++c) {
				if (h[r][c] >= curr[col]) break;
				int dist = abs(r - row) + abs(c - col);
				if (dist < min_dist || (dist == min_dist && h[r][c] < prev_r))
					min_dist = dist, prev_r = r, prev_c = c;
			}
		
		if (prev_r == -1) { prev_r = row; prev_c = col; }
		for (int r = prev_r, dr = r == row ? 1 : -1; r != row; )
			r += dr, out(r == row ? (dr > 0 ? 'D' : 'U') : 'R',
				r == row ? dr > 0 ? 'D' : 'U' : 'L');
		
		for (int i = 0; i < h[prev_r][prev_c]; ++i) { out(+1, "+"); }
		out(-min_dist, "-"), out("+");
		
		for (int r = 0; r < N; ++r)
			for (int c = 0; c < N; ++c)
				h[r][c] -= i;

		int val = h[row][col];
		if (val > curr[col] && val <= min_dist + 1) val = min_dist + 1;
		else val = curr[col];
		
		val--, h[row][col] += val;
		for (int i = 0; i < val; ++i) out(+1, "+");
		
		if (h[row][col] >= curr[col]) { curr[col]--; } else { curr[col]++; }
		
		for (int i = 0; i < N * N; ++i) {
			if (curr[i] == 0 && h[i / N][i % N] < 0) curr[i] = -h[i / N][i % N], h[i / N][i % N] = 0;
			while (cnt[i]--) { h[i / N][i % N]--; }
		}
	}
}

int main() {
	for (int j = 0; j < N * N; ++j) scanf("%d", &h[j / N][j % N]);
	solve();
}
```
````
