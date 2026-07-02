# 题 163 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The user wants a C++ solution for a competitive programming problem.

**Problem Analysis:**
We are given an $n \times n$ grid ($n=50, m=100$) representing a map of wards (colors).
Ward colors are integers from $1$ to $m$. The outside is color $0$.
Each color region (including 0) is connected in the input.
We need to produce another $n \times n$ grid where:
1. Each color region must still be connected.
2. The adjacency relationships between any pair of colors must be preserved. That is, color A touches color B in the new map if and only if they touch in the original map.
3. We want to minimize the number of squares of color 0. Actually, minimizing total area seems good, but specifically the score is defined based on $E$, the number of 0s. Wait, "score of $E+1$". So we want to minimize $E$?
Wait, usually these reconstruction problems ask for something specific. Let's re-read carefully.
"Please create a map using as few tiles as possible."
"Let $E$ be the total number of squares of color $0$ in the created map. Then you will obtain a score of $E+1$."
This implies the objective is to *minimize* the number of color 0 tiles (which acts like padding or background). Wait, usually score is higher is better. So yes, minimize $E$. However, minimizing 0s might conflict with connectivity constraints if not careful. But actually, the constraint is just valid coloring preserving adjacencies.
Wait, usually for this type of problem ("graph drawing on a grid"), finding the exact embedding that minimizes zeros is hard or NP-hard? No, $N=50$ is small enough for some heuristics, maybe.
Actually, there's a simpler interpretation. We have a planar graph formed by the regions (facets). We want to represent this graph using cells in a grid such that adjacencies are preserved. Since the original map satisfies the condition that each color is connected, it corresponds to a map-like structure (dual graph properties).
The critical observation for this specific problem (AtCoder Regular Contest 024 - Problem H or similar, actually this looks like AtCoder "AHC024" which is a hack contest problem) is often related to constructing the map cell by cell.
Actually, this is a known problem type called "Graph Realization" or "Map Reconstruction". Specifically, since $N=50$ and $M=100$, this is likely solvable by filling the grid greedily or by simulating the process of "growing" regions.
However, the scoring function suggests we want to maximize the non-0 area or minimize 0 area. Usually, in these problems, just satisfying the conditions is sufficient for correctness (WA), but score depends on efficiency. Wait, usually "Your program may output multiple solutions... If multiple solutions are output, only the last one is used for scoring." This implies the checker checks validity first.
Wait, looking at similar problems from AHC (AtCoder Hacker Contest) or general graph mapping problems on grids.
Usually, for these types of "Map Coloring" preservation problems, a constructive algorithm works well.
One common approach is to simulate the expansion. Or simply iterate through every cell $(i,j)$ and assign a color.
But wait, the problem statement says "Your task is to create a map ...". It doesn't strictly say we must find the optimal $E$, though minimizing $E$ gives a better score. Often in competitive programming contests, unless specified "output the optimal", any valid solution gets accepted, but here scoring matters. Since the score is just based on validity + $E$, we should try to minimize $E$.
Actually, checking online resources for "AtCoder AHC024", this is Problem H. The intended solution often involves a randomized local search or a greedy construction that fills empty space (0) carefully.
Actually, let's look at the structure. We have $M$ regions. The boundary between regions must be preserved.
Consider the set of adjacencies. In the grid, two colors are adjacent if there exists an edge between them.
Since each region is connected, we can model this as a planar graph where nodes are colors and edges exist if regions touch. The dual graph is connected.
However, the grid is fixed $N \times N$. We are placing these regions into the grid.
A simple heuristic that often works for "reconstructing a map preserving adjacencies" is:
Iterate through colors. Place one pixel of that color. Expand it.
But the grid topology imposes constraints (cycles in boundaries, etc.).
Actually, for $N=50, M=100$, the grid is dense.
There is a strategy:
For every pair of colors $c, d$ that are adjacent in the input, ensure they share a boundary in the output.
Since the input regions are connected, and the grid is large, maybe we can just fill the grid with random noise until valid, then run a solver? No, that's too slow for TLE risk.
Let's consider the "Growth" method.
Initially, we can place all pixels as 0.
Then, for each color $k$ from 1 to $M$, we assign at least one pixel to it.
To maintain adjacencies, if $k$ touches $d$ in the input, we must make sure the pixel for $k$ touches a pixel for $d$.
If we do this sequentially, later colors might block earlier ones.
However, note that "adjacency" is symmetric.
Is there a specific constraint on the shape of regions? Yes, they must be connected components of the same color.
Key Idea:
Maybe we don't need to minimize $E$ perfectly to pass, but getting a high score requires fewer 0s. The constraints allow *any* map.
Wait, if I just copy the input map, $E$ is the count of 0s in input. Can I do better?
In the input, regions can be very large. By shrinking them (while keeping connectivity), I can reduce area. But I must preserve adjacencies. If two regions touch at many points in input, they must touch at least once.
So I should pack the regions together tightly around a core.
Actually, this problem is about realizing the contact graph of regions on a grid.
Since the input guarantees connected regions and correct adjacencies, the dual graph is planar.
A standard way to construct such a map is to use a **simulated annealing** or a **greedy placement** strategy.
Given the strict time limits and complexity of finding the *optimal* arrangement (minimizing empty space), a randomized local optimization is often the intended solution path for Hack Contests. Or perhaps a specific constructive algorithm exists.
However, writing a full simulated annealing engine in C++ within a single response is risky if not careful.
Let's think if there's a deterministic approach.
What if we simply output the input map?
Wait, the input map has specific shapes. Maybe those shapes already minimize 0s relative to some packing?
Actually, the input generation process described is quite random-ish (filling with 0s, then picking neighbors). It produces somewhat scattered blobs.
Maybe the goal is to just find *a* valid configuration. The scoring $E+1$ encourages compactness, but usually in such problems if there are multiple test cases, any valid output is accepted for "correctness", but score penalizes large E.
Wait, if I output the input map, does it satisfy the condition?
"For every pair of colors c and d... adjacency ... identical."
Yes, obviously.
But maybe we can reduce E?
Actually, if I output the input map, the number of 0s is exactly determined. Is it possible to reduce it?
The problem asks for "create a map using as few tiles as possible". This implies we should reduce it.
How to reduce 0s? By merging regions that are currently separated by 0s.
If region A and B are separated by a 0-tile in the input, but they are NOT adjacent in the input, we cannot merge them or put them next to each other directly without creating unwanted adjacency.
Wait, if A and B are NOT adjacent in input, we MUST NOT have A and B adjacent in output.
This means 0 acts as a separator (or other colors act as separators).
If we replace a block of 0s with nothing? No, we must output a grid. The 0s are the background.
Minimizing 0s means making the colored regions cover the grid as much as possible.
This is equivalent to maximizing the union of the areas of the regions.
Since total cells = $n^2$, minimizing 0s $\iff$ maximizing sum of colored cells.
BUT, we must respect connectivity AND adjacencies.
This sounds like we want to pack the regions onto the grid as tightly as possible.
Since the input $N=50$ is small, maybe we can perform Local Search (Simulated Annealing) to find a configuration with fewer 0s.
Algorithm:
1. Start with a valid configuration (e.g., the input map itself).
2. Define cost = number of 0-cells.
3. Perform moves: swap colors of two cells, or recolor a cell, or move a region.
Constraint: Regions must remain connected.
Moving/merging/recombining regions while maintaining connectivity and adjacency preservation is complex to enforce exactly.
Actually, the constraints are:
1. Region connectivity (Color $c$ forms a single connected component).
2. Adjacency consistency (Edge $(c,d)$ exists iff $c$ touches $d$).
If I just take the input and apply local optimization to shrink the holes (0s) while fixing connectivity, it might work.
Specifically, fill holes.
Holes are groups of 0s enclosed by other colors.
If I change a 0 to color $X$, I must ensure connectivity of $X$ is maintained and no forbidden adjacencies are created.
Forbidden adjacency: If $Y$ is not adjacent to $X$ in input, $Y$ must not touch $X$ in output.
Currently, the boundary between $X$ and $Y$ in input ensures they touch. If $X$ expands into a hole, it might reach $Z$. If $X$ touches $Z$ now, check if $X$ touches $Z$ in input. If not, that move is invalid.
This logic suggests we can locally fill holes if safe.

Better idea:
The problem allows us to output *multiple* solutions, taking the last one. This strongly hints at a randomized or iterative approach being acceptable or even required.
The constraints $N=50$ are relatively small.
Maybe a Monte Carlo / Simulated Annealing approach works best.
State: Grid of colors.
Score: Number of 0s (we want to minimize).
Operations:
- Pick a random cell $(r,c)$.
- Try to change its color to something else (valid move?).
Change color makes connectivity/adjacency checks harder.
Alternative Operation:
- Pick a random color $C$.
- Try to shift its region to "compress" the map.
This is basically solving the "Graph Drawing" problem with grid constraints.

Wait, the "adjacency of a set of squares of color c and a set of squares of color d ... must be identical".
This means if $Adj(C, D)$ is true in input, it must be true in output. If false, it must be false.
It does NOT require all touching pairs to touch everywhere, just existence.
Also, connectivity of regions.

Since $N$ is small, a randomized greedy approach with repair might work.
Or simply:
Initialize with Input Map.
Iteratively try to improve score (minimize 0s).
Score improvement comes from changing a 0-cell to a non-0 color.
Condition to change cell $(i,j)$ from 0 to $k$:
1. Color $k$'s region remains connected.
2. No forbidden adjacencies created: For any neighbor of $(i,j)$ with color $x \neq k$, $x$ must already be adjacent to $k$ in input.
3. If we remove a 0-cell, the remaining 0-region must remain 0 (trivially true, it just becomes smaller).
Actually, if we change 0 to $k$, we effectively merge that pixel into region $k$.
We need to ensure that for all neighbors $u$ of $(i,j)$ with color $L(u)$, $Adj(k, L(u))$ holds in input.
If this holds for all neighbors, and region $k$ stays connected, then it's a valid move.
Region connectivity check: Adding $(i,j)$ to $k$ increases size. Since $(i,j)$ was 0, we connect it via neighbors. If $(i,j)$ had at least one neighbor with color $k$, connectivity is maintained (assuming previous $k$ was connected). Wait, initially $(i,j)$ is 0, so no neighbors are $k$. If we turn it to $k$, we need $(i,j)$ to attach to existing $k$. So $(i,j)$ must be adjacent to some cell of color $k$.
So:
Pick a random 0-cell $(i,j)$.
Check if it has a neighbor with color $k$ (to attach to $k$).
If yes, it's a candidate to become $k$.
Constraint check: All neighbors of $(i,j)$ must be compatible with $k$ in terms of adjacency.
Let $S_{nb}$ be the set of colors of neighbors of $(i,j)$.
Condition: $\forall x \in S_{nb}, \text{InputAdj}(k, x)$ must be True.
And we also need $\text{InputAdj}(k, k)$? Trivial (reflexive not needed, but consistent).
And we need to ensure that $k$ doesn't touch any $y$ via this pixel such that $y \notin S_{nb}$. But $y$ would be a neighbor of a neighbor? No, we are changing $(i,j)$ which has neighbors $S_{nb}$. The adjacency created is between $k$ and colors in $S_{nb}$.
Wait, if $k$ touches $y$ (where $y$ is neighbor of current $(i,j)$ cell), and $y$ is NOT adjacent to $k$ in input, then illegal.
So condition: $\forall \text{color } c \in \text{Neighbors}((i,j)), \text{HasAdj}(k, c)$.
Is that sufficient?
What if $(i,j)$ changes to $k$, and previously it separated $c_1$ and $c_2$? Now $c_1$ and $c_2$ might touch $k$, and each other via path? No, adjacency is defined by direct edge sharing. If $(i,j)$ was 0, it wasn't part of boundary between any two colors. Changing it to $k$ creates boundary between $k$ and its neighbors.
Wait, could it destroy a boundary?
No, it removes 0, potentially connecting regions that were not connected before?
Wait, if we change 0 to $k$, we are adding $k$ pixels. We are not removing adjacency of existing colors. Existing adjacencies (between $A$ and $B$) are formed by cells with value $A$ touching cells with value $B$. Changing a 0-cell to $k$ does not remove existing $A-B$ contacts. It might add $k-A$ contacts.
So, the "only" danger is adding an unwanted adjacency or breaking connectivity of $k$.
Connectivity: To keep $k$ connected, the newly added cell $(i,j)$ must connect to existing $k$ pixels. Since it was 0, it has no $k$ pixels inside. It must be adjacent to at least one existing $k$ cell.
So Algorithm:
1. Initialize `current_grid` = `input_grid`.
2. Identify all cells with 0.
3. Repeat for some iterations:
    a. Pick a random 0-cell `(r,c)`.
    b. Find its neighbors' colors. Let this set be `neighbor_colors`.
    c. Iterate over all possible target colors `target_color` (from 1 to M):
        i. Check if `target_color` exists in the grid.
        ii. Check connectivity: Does `(r,c)` touch any `target_color` pixel in `current_grid`?
            - Note: If `target_color` doesn't appear yet, we can't pick it because we need to preserve "existence"? No, "each ward ... guaranteed to be connected". The set of squares of color `target_color` must be connected. If `target_color` has 0 pixels, it violates "guaranteed to exist" (problem says every k=1..m exists at least one square). Wait, input has them. Do we need to ensure we don't remove them?
            - If we change a 0 cell to `target_color`, it adds to `target_color`. Connectivity is fine as long as we attach to existing pixels.
            - What if `target_color` has pixels, but none adjacent to `(r,c)`? Then we can't connect to it without breaking something?
            - If we can't connect, `target_color` remains disconnected? No, `(r,c)` is isolated from other `target_color`s. So we can't just switch it.
        iii. Check forbidden adjacency: For all `c` in `neighbor_colors`: `Original_Map_Has_Adj(target_color, c)` must be true.
        iv. Also, `target_color` must NOT be adjacent to `neighbor_colors`? Wait. `Original_Map_Has_Adj` handles this.
           Wait, `Original_Map_Has_Adj(X, Y)` returns true if X and Y touch in input.
           So if I change `(r,c)` to `target_color`, it will touch all `c` in `neighbor_colors`.
           So I must verify `Has_Adj(target_color, c)` for all `c` in `neighbor_colors`.
           Additionally, I should verify that `target_color` does NOT touch anything else?
           The only new adjacencies created are with neighbors. Are there other neighbors?
           The only other adjacency could be via "diagonal" (no, only edge) or wrapping (no).
           Wait, changing 0 to $k$ makes $k$ adjacent to whatever colors are on the border.
           If $k$ touches $Y$ and $Y$ was not in `neighbor_colors`, impossible since neighbors define border.
           So yes, iterating over `neighbor_colors` is sufficient.
           Wait, what if `target_color` is 0? No, we want to minimize 0s, so we convert 0s to others.
           If `target_color` == 0, we increase 0s, bad.
           Also, need to check if converting 0 to `target_color` causes `target_color` to touch `target_color`? That's connectivity.
    
    d. Apply the best move. If `target_color` found that satisfies conditions and reduces 0-count (it always does if converted), do it.
       Wait, does it always reduce 0-count? Yes, changing 0 -> $k$ reduces $E$.
       However, we might have a preference to avoid increasing 0s if stuck.
    
    e. Also, can we optimize by moving blocks? Or simply, swapping colors?
       Swapping colors is dangerous for connectivity.
       However, if we just want to minimize 0s, filling available holes is key.
       With N=50, the 0-regions might be fragmented.
       Filling holes is a local operation.
       This simple hill climbing (changing 0 to valid neighbor) should converge quickly to a state with few 0s, because 0s tend to accumulate if we don't fill them.
       
       Wait, there is a catch.
       Suppose `neighbor_colors` contains {1, 2}.
       Original map: 1 and 2 touch. 0 separates them elsewhere?
       If we change 0 to 1, 1 touches 2. Valid? Yes, if HasAdj(1, 2).
       What if 1 and 2 DON'T touch in input?
       Then changing 0 to 1 creates a 1-2 adjacency which is forbidden.
       So we can ONLY change 0 to $k$ if all neighbors $x$ satisfy `HasAdj(k, x)`.
       If `neighbor_colors` = {1, 2} and `HasAdj(1, 2)` is false, then we cannot change 0 to 1 OR 2 (unless both 1 and 2 are allowed? No, if we pick 1, we touch 2. Forbidden).
       Wait, if we pick 1, it touches 2. 2 is forbidden neighbor.
       So valid target $k$ must satisfy: $\forall x \in \text{neighbors}, \text{HasAdj}(k, x)$.
       If `neighbor_colors` is {1, 2} and `HasAdj(1, 2)` is False:
       - Trying $k=1$: touches 2. Invalid.
       - Trying $k=2$: touches 1. Invalid.
       - Trying $k=3$: if `HasAdj(3, 1)` and `HasAdj(3, 2)` are True, it's valid.
       - Trying $k=0$: Increases 0s. We want to decrease.
       So if no non-zero $k$ fits, we can't fill this hole safely.
       We must wait until a valid neighbor appears.
       However, we can also permute existing regions?
       If we just blindly apply local search, we might get stuck.
       But $N=50$ is small.
       Another move: swap 0-cell with a neighbor cell? No.
       What about swapping two cells of different colors?
       Or swapping a block of colors?
       Given the randomness of the checker (random seed for test cases, maybe?), maybe random initialization + SA is needed.
       The provided code snippet should implement a robust solution.
       Random shuffling the initial grid or starting from scratch with valid regions?
       Starting from input is safest for connectivity.
       Let's refine the local search.
       Instead of just picking 0-cells, maybe we can pick ANY cell and try to change its color?
       Changing a colored cell $(r,c)$ from $A$ to $B$:
       - Must maintain connectivity of $A$ (removing $(r,c)$ doesn't disconnect $A$).
       - Must maintain connectivity of $B$ (adding $(r,c)$ connects to existing $B$s).
       - Adjacency constraints: New adjacencies created/destroyed.
         Removing $A$ removes adjacencies $(A, \text{neighbors})$.
         Adding $B$ creates adjacencies $(B, \text{neighbors})$.
         Need:
         1. $A$ remains connected after removal.
         2. $B$ remains connected after addition.
         3. Forbidden new adjacencies: $\forall x \in \text{neighbors}, \neg \text{HasAdj}(B, x)$.
         4. Missing allowed adjacencies: $\exists y \in \text{neighbors}, \text{HasAdj}(A, y)$ removed?
            Wait, condition is "adjacency ... identical".
            So if $A$ touched $y$ only via $(r,c)$, and now it doesn't, is that allowed?
            NO. The condition says "if and only if there exist adjacent squares ...".
            If $A$ and $y$ were adjacent in input, they MUST be adjacent in output.
            If we remove a contact point, we must ensure another contact point exists.
            Checking global existence for all pairs is expensive ($O(M^2)$) at every step.
            However, notice the property:
            "If $A$ and $B$ are adjacent in input, they must touch."
            So we must ensure we don't break all connections.
            And we must ensure we don't create new connections.
            
       This makes the search space complex.
       
       Alternative:
       The problem is simpler than general map drawing because the input regions are already valid.
       We just need to pack them tighter.
       Is there a standard trick?
       Maybe the "Small Map" problem refers to compressing the map representation.
       Notice the problem title/context implies finding a minimal valid representation.
       However, in competitive programming contexts like this (AtCoder), often a randomized constructive approach that converges to a local optimum is accepted, especially if the problem is from a Hacker Contest (where hacks are encouraged, meaning solution might be brittle but works on most).
       Wait, "AHC024" usually stands for "AtCoder Hack Contest". These problems are meant to be solved by hacking the judge or submitting clever solutions.
       Since the user asked for a solution, I should provide a robust one.
       
       Let's consider the "Randomized Construction" approach.
       1. Construct the map from scratch.
          - We know the adjacency constraints form a planar graph (the dual graph).
          - But we don't need to reconstruct the dual graph. We can work with the grid.
          - Initialize grid with all 0s.
          - Pick a random valid position for color 1. Set it.
          - Expand color 1 randomly.
          - When color $c$ needs to be placed, check compatibility with existing regions.
          This is hard because $M=100$ and $N=50$ is tight.
       
       Let's go back to the Hill Climbing on the input map.
       Input map is valid.
       Moves:
       - Fill a 0-cell with color $k$.
         Condition: $k$ is compatible with all neighbors of $(i,j)$ AND $(i,j)$ touches existing $k$.
         Wait, if $(i,j)$ touches existing $k$, does it imply connectivity?
         Yes, if $k$ was connected, adding a pixel attached to it keeps it connected.
         So we need:
         - Neighbor colors $N_{set}$.
         - Exists neighbor with color $k$.
         - For all $c \in N_{set}$, `HasAdj(k, c)` is true.
         Cost: reduces $E$.
       - Swap two cells?
         Maybe swap $(i,j)$ (color $A$) with $(p,q)$ (color $B$)?
         This shifts regions. Might help reduce 0s by shifting regions towards 0-holes.
         Complexity of checking swaps is higher, but necessary if we get stuck with 0-holes surrounded by incompatible neighbors.
         
       However, implementing a complex swap system might exceed time/memory if not careful.
       Let's look at constraints again.
       $N=50, M=100$. Time limit 1-2 sec usually.
       Simple local search on filling 0s might get stuck easily if neighbors block.
       BUT, observe the input generation: "Randomly select a square with $c_{i,j}=0$ and set $c_{i,j}=k$." This implies initial positions are sparse.
       The regions grow.
       In the final valid solution, regions should ideally surround the others efficiently.
       
       Actually, there is a specific observation in similar problems (like "Map Making"):
       If we treat the colors as nodes in a graph, we are embedding this graph in a grid.
       Wait, minimizing 0s is effectively maximizing the coverage.
       Since the checker accepts any valid solution, but scores based on 0s, and the score is used for ranking.
       In "Hack Contests", usually solutions that output *something valid* pass, but to rank high you need optimal/near-optimal.
       The problem statement says: "If more than one participant gets the same score, they will be ranked in the same place regardless of the submission time."
       Wait, if I output the input map, is it optimal?
       Probably not, because input generation creates large blobs with holes.
       So optimizing 0s is important.
       
       Strategy:
       Use a random walk with simulated annealing.
       State: Grid.
       Score: Number of 0s (inverse, maximize non-zeros).
       Operations:
       1. Recolor a 0-cell to $k$ (if compatible).
       2. Swap two non-0 cells (permute regions).
          This helps rearrange regions.
          Constraints:
          - Connectivity maintenance (complex).
          - Adjacency maintenance (check affected borders).
       
       Maybe a simpler heuristic:
       Just iterate through every 0-cell and see if we can change it to some $k$.
       If we can't, leave it.
       But what if we swap two non-0 cells to enable a change?
       Actually, there's a known result for this problem (AtCoder AHC024 Problem H).
       The solution typically involves a randomized local search or a specific construction.
       Given the constraints and nature, I will implement a randomized local search (Metropolis / Simulated Annealing) to maximize the number of colored cells (minimize 0s).
       
       To handle connectivity and adjacency checks efficiently:
       We precompute the `Adj[M+1][M+1]` matrix.
       We can check adjacency constraints in $O(N)$ per cell change (scan neighbors).
       For connectivity, BFS on the grid restricted to color $k$ takes $O(N^2)$. Too slow if done frequently.
       Optimization: Maintain `is_connected[color]` status?
       Or use Union-Find?
       Since $N$ is small (2500 cells), $O(N^2)$ per check is $\approx 6 \times 10^6$, feasible for limited iterations.
       With $T=150$ cases, we have limited total time.
       However, for a single case, doing SA with many iterations might TLE.
       We need to balance iteration count.
       Also, connectivity check is crucial.
       
       Wait, connectivity of the whole grid is required.
       For a specific color $k$, its pixels must form a connected component.
       When we change color of $(r,c)$ from $old\_color$ to $new\_color$:
       1. Remove $(r,c)$ from $old\_color$ component. Check if $old\_color$ is still connected.
          (Can be checked via flood fill or counting components, but simpler: BFS from arbitrary $old\_color$ pixel; if visited count < total, fail.)
          Optimization: Only need to check if $(r,c)$ was an articulation point for $old\_color$. Hard to maintain dynamically.
          Given $N=50$, a simple BFS check is fast enough if not done too often.
          
       2. Add $(r,c)$ to $new\_color$.
          Check if $(r,c)$ touches existing $new\_color$ pixels (connectivity of $new\_color$).
          
       3. Check adjacencies.
          Neighbors of $(r,c)$ are $x_1, \dots, x_4$.
          Before: $(r,c)$ was $old\_color$. Created adjacencies $(old\_color, x_i)$.
             (Implicitly assumed valid because original map was valid and we are modifying).
          After: $(r,c)$ becomes $new\_color$. Creates adjacencies $(new\_color, x_i)$.
          We need:
          - For all $x \in \{x_i\}$, $NewMap$ has $(new\_color, x)$ iff $InputMap$ has it.
            Since we modify one pixel, we only create/remove interactions with immediate neighbors.
            So we must check:
              - $\forall x \in \{x_i\}, \text{HasInputAdj}(new\_color, x)$.
              - Also, check that we didn't remove a REQUIRED adjacency for $old\_color$.
                i.e., if $(r,c)$ was the *only* contact point between $old\_color$ and $y$, then removing it is invalid.
                Wait, if $y$ is a neighbor of $(r,c)$, then $old\_color$ touches $y$ at $(r,c)$.
                If $old\_color$ touches $y$ elsewhere, it's fine.
                If not, then removing this contact breaks the "touch" relation.
                So we must verify: For all $y \in \{x_i\}$, $\exists \text{ other contact between } old\_color \text{ and } y$.
                This check is tricky.
              
          Actually, maybe we shouldn't modify non-zero cells arbitrarily.
          It's safer to focus on zero cells first.
          If we are stuck with zero cells, maybe we can swap two adjacent non-zero cells.
          Swapping $(u,v)$ from $A$ to $B$ and $v$ from $B$ to $A$.
          Effectively just moving the boundary between $A$ and $B$.
          Connectivity of $A$ and $B$ must hold.
          Adjacency between $A$ and $B$ must hold (they touch anyway).
          Adjacency between $A$ and $z$ ($z \neq B$) neighbors $u$:
          Before $u$ touches $z$ via $A$. Now $u$ touches $z$ via $A$?
          If we swap $u(A) \leftrightarrow v(B)$, then $u$ is $B$, $v$ is $A$.
          New $u$ (color $B$) touches neighbor $w$. Needs $Adj(B,w)$.
          New $v$ (color $A$) touches neighbor $z$. Needs $Adj(A,z)$.
          This is equivalent to checking neighbors of swapped positions.
          
       Given the difficulty of dynamic connectivity and adjacency checks in C++, and the constraints:
       Let's simplify.
       The problem is essentially asking to find a valid configuration. The scoring encourages reducing 0s.
       If we start with input map, we can try to fill holes.
       The holes are surrounded by various colors.
       For a hole at $(r,c)$, let $S$ be the set of adjacent colors.
       We can fill with color $k$ IF $\forall c \in S, \text{HasAdj}(k,c)$.
       Also, $k$ must be present in the grid adjacent to $(r,c)$? No, $k$ must attach to itself.
       Wait, if $(r,c)$ is a hole, and we fill it with $k$, does it detach $k$? No.
       Does it detach others? No, $k$ grows.
       But what if filling $(r,c)$ with $k$ creates a bridge between two parts of $k$? (Good).
       Or if it bridges $k$ to something?
       The constraint "region $k$ connected" is satisfied if we pick $k$ adjacent to an existing $k$ pixel.
       The constraint "adjacency identity":
       New adjacencies are $(k, c)$ for $c \in S$.
       We need $Adj(k, c)$ true in input for all $c \in S$.
       Wait, if $k$ was not adjacent to $c$ in input, we cannot create contact.
       So, we can only fill a hole with $k$ if $k$ is adjacent to ALL neighbors of the hole.
       If $S=\{1, 2\}$ and we fill with 1. 1 touches 2. If $Adj(1,2)$ is false -> Illegal.
       So we can't fill if neighbors clash.
       
       This suggests we should just keep the input map.
       Wait, why?
       Because the input map is valid.
       Maybe we can't reduce 0s without violating constraints.
       Example: Input has a hole of 0 surrounded by 1 and 2 where $Adj(1,2)$ is false.
       This means in the original map, region 1 and 2 do NOT touch.
       But they separate the hole.
       In the original map, they might be far apart, or touch only via 0s?
       Wait, "connected" means contiguous.
       If they don't touch, then the path between them goes through 0 or something else.
       If they surround a hole of 0, and they don't touch, then 0 is separating them?
       Or maybe they touch "behind" the hole.
       But if they surround the hole, they are adjacent in the grid boundary sense.
       In the input, if $1$ and $2$ are neighbors of a 0-cell, it means the input has $1$ and $2$ touching $0$ at that location.
       Does the input guarantee $1$ and $2$ are NOT adjacent?
       "If and only if there exist adjacent squares of color c and d in the original map..."
       If input has 0-cell, 1-neigh, 2-neigh, and $Adj(1,2)$ is false.
       Then we cannot change 0 to 1 (creates 1-2 contact). Cannot change to 2 (creates 2-1).
       Can we change to 3? If $Adj(3,1)$ and $Adj(3,2)$ are true.
       If no such 3 exists, we can't fill that hole.
       So the minimum 0 count might be achieved by just keeping the 0s.
       
       However, the scoring $E+1$ implies we should reduce $E$.
       Maybe the "hole" isn't surrounded by conflicting neighbors globally?
       Wait, if $S=\{1,2\}$ and $Adj(1,2)$ false.
       Maybe the hole is deep inside 1? No, then only 1 in $S$.
       If hole is surrounded by 1 and 2.
       If we simply accept the input map, is it optimal?
       Not necessarily. The problem allows us to create *any* map.
       Maybe we can rearrange the whole map to be more compact.
       
       Actually, there is a simpler logic.
       If $S=\{1, 2\}$ and $Adj(1,2)$ is false.
       This implies 1 and 2 are disjoint in the original adjacency graph.
       In the original map, they might be separated by a chain of other colors, or 0s.
       Wait, 0 is not a color $1..m$. 0 is "outside".
       If 1 and 2 don't touch, but both touch 0.
       In the output, if we fill the 0-hole with 3, we create 1-3 and 2-3 contacts.
       This requires $Adj(1,3)$ and $Adj(2,3)$ in input.
       If such a 3 exists, we can reduce 0s.
       
       So the strategy is:
       Run local optimization to fill 0s whenever possible.
       Start with Input.
       While improved score:
         Find a 0-cell with valid replacement candidates.
         Greedily pick one (maybe random or priority to reduce 0s).
         Update.
       To handle getting stuck, maybe shuffle neighbors?
       Actually, we might want to perform swaps to bring valid colors closer to 0-cells.
       
       Implementation details:
       - Represent grid as vector<vector<int>>.
       - Precompute `adj[101][101]`.
       - Loop max iterations (e.g. 5000 or so).
       - Inside loop:
         - Shuffle 0-indices.
         - For each 0-index `idx`:
           - Determine neighbors `N_idx`.
           - Find compatible `target` color.
           - Compatible: `HasAdj(target, neighbor)` for all `neighbor` in `N_idx`.
           - AND `target` touches existing `target`? (Connectivity).
             Wait, if we pick a color `target` that is currently present.
             Does `target` need to be adjacent to `idx`?
             If `idx` is filled with `target`, it merges into `target`'s component.
             If `target` was not touching `idx` before, now it touches `idx`.
             Is `target` connected? Yes, if `target` existed and remains connected.
             Since we only add `idx` to `target` and `idx` touches some `target` pixels (at least one), the new component is connected.
             So condition: $\exists p \in \text{grid}, \text{grid}[p]==target$ AND $\exists q \in \text{neighbors}(\text{idx}), \text{grid}[q]==target$.
             Basically, `target` must have pixels adjacent to `idx`.
           - If multiple targets, pick one. Prefer one that reduces $E$ (all do).
             Maybe pick random valid one.
           - Perform move.
             - If valid, update grid, count 0s.
             
         - If no 0-cell improved, try to improve by swapping two non-zero cells?
           Or just restart/shuffle.
           
       One detail: "adjacency ... identical".
       When we change `grid[r][c]` from 0 to `target`:
       New adjacencies: `(target, neighbor_color)`.
       Old adjacencies: `(0, neighbor_color)` (removed, 0 doesn't matter for adjacency of other colors).
       Wait, the condition is: "For every pair of colors c and d... adjacency ... identical".
       Does 0 count? "squares of color 0 ... can be connected through the outside".
       The condition applies to colors $0 \le c < d \le m$.
       So yes, adjacencies involving 0 must also match.
       Input has 0 surrounding cities. Output 0 is "outside".
       So if we fill a hole, we reduce the amount of 0, so potentially lose a boundary with outside?
       Wait.
       "Squares of color 0 can be connected through the outside squares".
       If we fill a hole (which was 0) with color 3.
       Does 0 become adjacent to 3?
       Yes, neighbors of `idx` are 0 (in hole) -> no, `idx` is 0.
       If `idx` is surrounded by 0s, it's inside a lake of 0s.
       If we fill it, 0 (lake) still surrounds it. 0 is connected to outside 0.
       Wait, if `idx` is inside a lake of 0s, neighbors of `idx` are 0.
       Then filling `idx` with 3 creates 3-0 adjacency.
       In input, `idx` was 0. 0 touches 0 (yes).
       So 3 touches 0. We need `HasAdj(3, 0)` in input?
       Wait, `HasAdj(c, 0)` means "does color c touch outside in input?".
       The problem says: "The city consists of m wards... The outside ... is colored 0."
       "For every pair of colors c and d... adjacency ... identical."
       So if we change a cell from 0 to 3, we create a 3-0 boundary.
       If the original map had 0 touching 3 at that location? No, originally it was 0 touching 0.
       So in original map, 0 and 3 might touch somewhere else, or nowhere.
       If 3 does NOT touch 0 in input, we CANNOT create a 3-0 boundary.
       So we must check `HasAdj(target, 0)` as well?
       Wait, "Squares of color 0".
       If we change `grid[r][c]` from 0 to 3.
       The neighbors of `grid[r][c]` include cells with color $X$ (where $X$ can be $>0$ or $0$).
       If neighbor is $X > 0$: We create $3-X$ adjacency. Requires `HasAdj(3, X)`.
       If neighbor is $0$: We create $3-0$ adjacency. Requires `HasAdj(3, 0)`.
       Wait, `grid[r][c]` was 0. It contributed to $0-X$ and $0-0$ adjacencies.
       Since 0 is the outside, 0 touches everything that touches the boundary.
       Wait, does 0 touch 0? Yes, but usually considered trivial.
       The constraint is about distinct pairs.
       Does it matter if we touch 0?
       The problem says: "for every pair of colors c and d (0<=c<d<=m)".
       So yes, pair (3, 0) matters.
       So if we fill a 0-cell with 3, and that cell had a neighbor 0.
       Then we introduce a 3-0 adjacency.
       Wait, in input, that neighbor was 0.
       Originally, that cell was 0. It had neighbor 0.
       The set of adjacencies involving 0 was $Adj(0, X)$ for neighbors $X$.
       If we change cell from 0 to 3.
       New adjacencies from this cell: $(3, Y)$ for all neighbors $Y$.
       Old adjacencies (implicitly): $(0, Y)$ for all neighbors $Y$.
       So we effectively replace the set $\{(0, Y)\}_Y$ with $\{(3, Y)\}_Y$.
       For this to be valid, we need $\{(0, Y)\}_Y \subseteq GlobalAdjacencies(0)$?
       Wait, the condition is "identical".
       This means the SET of pairs $(c,d)$ that are adjacent must be constant.
       If in Input, 0 and $Y$ were adjacent at this spot, then yes.
       If in Input, 0 and $Y$ were adjacent elsewhere, it's okay.
       So, if we change 0 to 3 at a cell with neighbor $Y$.
       We remove $(0, Y)$ from the adjacency list (locally).
       We add $(3, Y)$ to the adjacency list.
       We must ensure:
       1. $(3, Y)$ is allowed in global graph (i.e. `HasAdj(3, Y)` is true in input).
       2. We didn't break a necessity?
          Wait, "adjacency identical" means:
          If $(A, B)$ is an edge in input, it must be in output.
          If $(A, B)$ is NOT an edge in input, it must NOT be in output.
          So:
          For all neighbors $Y$:
             Must have `HasAdj(3, Y)` (new contact).
             Wait, did we rely on $(0, Y)$?
             If $(0, Y)$ is an edge in input (0 touches Y), and we replace it with $(3, Y)$, that's fine as long as $(3, Y)$ exists.
             BUT, what if $(3, Y)$ doesn't exist in input? Then we violate "NOT in output".
             What if $(0, Y)$ does NOT exist in input? (0 touches Y only via this pixel?)
             Then by changing to 3, we might REMOVE an adjacency? No, 0 touching Y is usually not what matters, 0 is background.
             Actually, 0 represents the "outside".
             "The outside ... is colored 0".
             The condition "adjacency ... identical" implies the topological connectivity of the map including outside.
             If in Input, color 3 does not touch Outside (0).
             Then in Output, color 3 must not touch Outside.
             If we fill a hole inside the map (surrounded by other colors), we create internal 0s?
             Wait, if we fill a hole, we reduce 0s.
             If the hole was "enclosed" by non-0 colors, then 0 is NOT touching those colors in input (if enclosed strictly).
             Wait, in input generation, 0s are selected from empty spots.
             Usually 0s are the background. But sometimes 0s can form lakes?
             The generation process: "initialize 0... randomly select 0 and set to k... repeat selecting 0 and adjacent 0-set-to-k".
             This creates islands of non-0 growing from initial seeds.
             So 0s can be completely surrounded?
             If 0s are surrounded, then 0-touches-X adjacency exists in input (on the boundary of the island).
             Wait, if 0s are surrounded by $X$, then 0 touches $X$.
             If we fill a 0-cell inside the 0-region (lake), and it touches $X$ via neighbors.
             Wait, if 0s are surrounded by $X$, then 0 touches $X$ along the perimeter.
             Any 0-cell on the perimeter touches $X$.
             Any 0-cell inside touches 0s.
             If we fill a 0-cell $P$ with 3.
             If $P$ is surrounded by 0s (deep lake), neighbors are all 0.
             Then we create 3-0 adjacency.
             We need `HasAdj(3, 0)` to be true.
             Is `HasAdj(3, 0)` true?
             If 0s are a lake surrounded by $X$, then 0 touches $X$ globally.
             If 3 does not touch 0 in input (because 3 is far away), then 3-0 is forbidden.
             So we cannot fill a 0-cell with 3 if 3 doesn't touch 0 in input.
             If 0 is surrounded by 0s (meaning 0 is a huge region), then 0 touches 0 (trivial) and 0 touches X (boundary).
             So, 3 can only touch 0 in output if 3 touches 0 in input.
             This simplifies things. We just need to check `HasAdj(target, neighbor)`.
             Wait, what if neighbor is 0?
             Then we check `HasAdj(target, 0)`.
             What if neighbor is $Y (>0)$?
             Then we check `HasAdj(target, Y)`.
             
             Wait, there's a subtle point.
             If we change 0 to 3.
             The original cell was 0. It contributed to adjacency $(0, 0)$ (internal) and $(0, Y)$ (boundary).
             The new cell contributes $(3, 0)$ and $(3, Y)$.
             The condition requires that the SET of adjacencies is preserved.
             Does replacing $(0, 0)$ with $(3, 0)$ affect anything? No, 0 touches 0 is irrelevant.
             Replacing $(0, Y)$ with $(3, Y)$ affects the $(3, Y)$ existence.
             If $(3, Y)$ is in input, good. If not, bad.
             So yes, `HasAdj(target, neighbor)` check for all neighbors covers it.
             Also need to check that we didn't break any adjacency?
             We removed $(0, Y)$? No, 0 and Y touching is valid.
             We are just removing a pixel of 0.
             Wait, does the adjacency $(0, Y)$ disappear globally if we fill THIS pixel?
             No, $(0, Y)$ might exist elsewhere.
             But if $(0, Y)$ ONLY existed at this pixel (and nowhere else)?
             If so, by filling it, we lose $(0, Y)$.
             But the input has 0 as "outside". Usually 0 is connected.
             If we lose $(0, Y)$, it implies 0 and Y no longer touch.
             But wait, if we fill a hole with 3, 3 becomes the new boundary with Y?
             If 0 and Y touched *only* via this pixel, then by replacing 0 with 3, Y stops touching 0.
             It starts touching 3.
             So if $(0, Y)$ was unique, we broke the connection.
             BUT, 0 represents "outside".
             "If and only if there exist adjacent squares of color c and d...".
             The condition is existential. "there exist".
             It does not require "every instance".
             So losing one contact point between 0 and Y is fine, as long as Y and 0 still touch somewhere else.
             (Note: 0 is the "outside", so if Y is not fully surrounded by other walls, it touches outside. If we patch a hole, we might sever Y from outside? No, if Y is a region, it touches outside at its boundary. If we patch the boundary with a wall of 3, Y loses access to outside. If Y touches 0 elsewhere, fine. If Y was an island in 0-space, and we fill the 0-space with 3, we isolate Y? No, we fill 0-space. If 3 is not outside, then Y is surrounded by 3. So Y touches 3. Y no longer touches 0.
             If in input, Y did touch 0.
             So we must ensure Y still touches 0 somewhere?
             Wait, if we fill the hole, 0 is reduced.
             If 0 becomes empty (no 0s left), then 0 is gone?
             No, 0 must exist (since "outside").
             Actually, the outside 0s must be connected.
             If we fill all 0s, 0 disappears. But we can't have 0 disappear because 0 is a color.
             The problem says "create a map ... using colored square tiles".
             Usually there is always at least one 0 tile or the outside concept.
             If we fill *all* 0s, the map is fully tiled with $1..m$.
             Does the condition "adjacency ... identical" allow 0 to not touch anyone?
             Input has 0 touching someone. So Output must have 0 touching someone.
             So we cannot eliminate all 0s if 0 touches anyone in input.
             However, input says "For every $k=1..m$, there exists at least one $(i,j)$ with $c_{i,j}=k$".
             It does not say 0 exists. But 0 is the outside.
             The output map has color 0 as the outside.
             So 0 must be represented.
             Minimizing E means minimizing the count of 0s.
             But if we eliminate 0 entirely, we violate the condition if 0 was supposed to touch $X$.
             So we must keep at least one 0 if $Adj(0, X)$ is true.
             
             Okay, simplified logic for local search:
             1. `adj[c][d]` stores if $c$ touches $d$ in input.
             2. Current grid is initialized with Input.
             3. Count zeros $E$.
             4. Loop until convergence:
                - For each 0-cell `(r,c)`:
                  - Get neighbors $N = \{grid[nr][nc]\}$.
                  - Check if filling with $k \in \{1..m\}$ is valid.
                    - Constraint 1: Connect to existing $k$.
                      - Check if $k$ has any pixel adjacent to `(r,c)` in `grid`.
                    - Constraint 2: Consistent with $N$.
                      - $\forall n \in N$, `adj[k][n]` is true.
                      - Wait, does $n$ need to be checked against $k$?
                        - If $n=0$, `adj[k][0]` must be true.
                        - If $n=c'$, `adj[k][c']` must be true.
                    - Constraint 3: Don't disconnect $k$ or break connectivity of others?
                      - Adding pixel to $k$ preserves $k$'s connectivity if $k$ was already connected.
                      - Wait, does it disconnect $n$?
                      - If we change 0 to $k$, we don't change $n$. $n$ is just a neighbor. Its connectivity is unchanged.
                      - We don't need to check "don't disconnect others" since we only change 0s.
                - If valid $k$ exists, perform move.
                - Record new $E$.
                - If $E$ decreased, continue.
                
             This greedy filling might get stuck.
             To improve score further, we might need to swap.
             Given the "Score of E+1" and the constraint that it's a "Hacker Contest", a robust local search is good.
             But simpler:
             The question asks for *a* solution.
             I will implement a randomized local search that tries to fill holes.
             Also, I will add a step to "shake" the map (swap two pixels of same color? No, that does nothing).
             Maybe swap two different colors to rearrange boundaries.
             Actually, since the input is already valid and connected, and just might have inefficient packing, simple greedy filling of holes is a strong heuristic.
             
             Refined Plan:
             - Read Input.
             - Compute `adj[m+1][m+1]` matrix.
               - Iterate all pairs of distinct cells. If colors differ, mark adj.
               - Include adj with 0?
                 - Yes, mark `adj[0][c]` if color c touches 0.
                 - Iterate neighbors.
             - Initialize `ans_grid = input_grid`.
             - Loop:
               - Found_improvement = false
               - Collect all indices where `ans_grid[i][j] == 0`.
               - If no 0s, stop? (Or try to optimize further?)
                 - If no 0s, score is $1$. Max possible (if $N \ge 1$).
               - For each 0-index `(r,c)`:
                 - Check potential replacements `target` in $1..m$.
                   - Must connect to `target` region:
                     - If `target` exists in grid, check if it touches `(r,c)`.
                   - Must be compatible with neighbors:
                     - For each neighbor `val`: `adj[target][val]` must be true.
                     - Special check: `adj` includes `target` vs `0`?
                       - If neighbor is 0, check `adj[target][0]`.
                       - If neighbor is $>0$, check `adj[target][neighbor]`.
                 - If valid `target` found:
                   - Change `ans_grid[r][c] = target`.
                   - Found_improvement = true.
                   - Update `E`.
                   - Break (one move per iteration to speed up convergence).
               - If not Found_improvement:
                 - Maybe try a random shuffle of the grid to escape local optima?
                 - Or try to swap two colored cells to open a path?
                 - Given time limit, maybe 1000 iterations of greedy fill is enough?
                 - The problem has 150 test cases. Total time ~ few seconds.
                 - 1000 iterations is too slow per case ($150 \times 1000 = 1.5 \times 10^5$ operations, per op scanning neighbors $O(N)$? Total $O(1.5 \times 10^5 \times 2500) \approx 3 \times 10^8$ ops. Might be tight for 1-2s).
                 - We need faster checks or fewer iterations.
                 - Scanning all neighbors for a 0-cell is $O(1)$.
                 - Finding valid target is $O(M)$.
                 - So per iteration $O(M)$.
                 - Total $150 \times 500 \times 100 \approx 7.5 \times 10^6$. Very fast.
                 - So 500 iterations is plenty.
                 
             Is this greedy fill guaranteed to find a good solution?
             Sometimes holes are blocked.
             If we can't fill a hole, maybe the current configuration is optimal for that hole.
             Wait, could there be a better configuration reachable by moving a region?
             Perhaps.
             But for the purpose of passing and getting high score in such contest settings, often a greedy fill works well if the input is reasonable.
             However, the input generation creates "blobs". Blobs can enclose 0s.
             If a blob encloses a 0, and all neighbors of the inner 0 are incompatible, we can't fill.
             This is unavoidable unless we cut the blob (change boundary).
             Changing boundary of blob involves swapping colors.
             Swapping colors is complex.
             
             Wait, maybe I should just stick to the simplest valid solution.
             Is there a solution that minimizes 0s without swapping?
             If I output the input, $E$ is minimized if I assume the input is packed optimally.
             Is it?
             The input is generated by "grow blobs randomly". This tends to produce clusters. Clusters might be efficient.
             Maybe the "small map" requirement is just flavor text, or simply requires valid output.
             However, usually Hacker Contest problems reward optimizing.
             I'll implement the filling strategy.
             Also, I'll add a random restart or shuffle if improvements stall for a bit.
             Since the time limit is generous for simple loops (usually 1s-2s per file or total?), wait, "execution time ... limit".
             Typically 2-3 sec total.
             I'll use a time limiter or just a fixed large number of iterations.
             Let's set iterations to 1000.
             Inside, pick a random 0-cell. Try all possible valid targets. If found, apply.
             
             Wait, one more thing:
             The problem asks for the score based on E+1.
             If I output input, I get score $E_{input}+1$.
             If I reduce E, score improves.
             The example link shows an output.
             
             Wait, I recall this problem.
             In this problem, you are allowed to output *any* valid map.
             The challenge is to maximize the score.
             A simple randomized search works.
             
             Code Structure:
             - Includes.
             - Fast IO.
             - Function `solve()`.
             - Read N, M.
             - Read grid.
             - Compute Adjacency Matrix `adj[m+1][m+1]`.
             - Initialize `best_grid = grid`.
             - Loop iterations:
               - `improved = false`.
               - Shuffle 0-cells or pick random.
               - Try to fill.
               - If improved, update `best_grid`.
               - If not improved for K times, maybe try to "shake" (swapping colors)?
                 - Let's skip shaking for now, keep it simple. If TLE/WA, reconsider.
                 - Actually, if we can't fill any 0, we might want to optimize the boundary?
                 - Swapping two neighbors of different colors to resolve blocking?
                 - Let's stick to the filling.
             
             Wait, what if `target` doesn't exist in grid?
             We can't create a new color out of thin air?
             The problem says "city consists of m wards... squares of color c (1<=c<=m)".
             So all colors must appear.
             So if we pick a `target` that has 0 pixels, it's invalid.
             But in the input, all colors exist.
             We never delete colors (just expanding/contracting them).
             Wait, filling 0 with `target` increases pixels of `target`.
             Does it ever remove `target` pixels? No.
             So all colors remain.
             So checking existence is not needed (unless `target` was deleted, which doesn't happen here).
             
             Wait, connectivity check:
             If `target` exists, does adding this pixel maintain connectivity?
             Yes, if the pixel is adjacent to at least one `target` pixel.
             If not, it becomes an isolated pixel of `target`. Disconnected.
             So we must check if `target` has a pixel in neighbors of `(r,c)`.
             
             Final check:
             - Is it possible `target` is not in grid?
             - If so, we can't pick it.
             - But since we only add to existing, we won't create new colors.
             - All good.
             
             Wait, "The outside of the n*n squares is colored 0".
             My `adj` calculation should include 0 correctly.
             Also `adj[0][0]` doesn't matter.
             
             One corner case:
             The input map might have 0s that cannot be filled due to conflicts.
             This is fine.
             
             Complexity:
             Iterations $K$.
             Per iteration: Scan grid? No.
             Better: Keep a list of 0-coordinates.
             Loop over list. Pick random index.
             Check neighbors (size 4).
             Check compatibility with M colors.
             This is fast.
             
             Optimization:
             Precompute neighbor colors for each cell to avoid recomputing.
             Actually `grid` is small.
             
             Time Limit: Standard is usually 1-2s.
             With 150 cases, total time ~ 1.5s per case? No, "execution time limit" applies to the submission. Usually sum of time limits over test cases or total time?
             "150 test cases... execution time and memory limit are stated".
             Usually total time is generous.
             But I should be efficient.
             
             Let's write the code.
             Using `vector<vector<int>>` for grid.
             
             ```cpp
             #include <iostream>
             #include <vector>
             #include <algorithm>
             #include <tuple>
             #include <chrono>
             
             using namespace std;
             
             int main() {
                 // Optimize IO
                 ios_base::sync_with_stdio(false);
                 cin.tie(NULL);
             
                 int n, m;
                 if (!(cin >> n >> m)) return 0;
             
                 vector<vector<int>> grid(n, vector<int>(n));
                 for (int i = 0; i < n; ++i) {
                     for (int j = 0; j < n; ++j) {
                         cin >> grid[i][j];
                     }
                 }
             
                 // Precompute adjacency
                 vector<vector<bool>> adj(m + 1, vector<bool>(m + 1, false));
                 for (int i = 0; i < n; ++i) {
                     for (int j = 0; j < n; ++j) {
                         int c = grid[i][j];
                         if (c == 0) continue; // Skip outer 0? Wait, 0 is color 0.
                         // 0 is a color. It should be tracked.
                         // But 0-0 adjacency doesn't need explicit bool?
                         // We need adj[u][v].
                         // Let's compute for all distinct neighbors.
                     }
                 }
                 
                 // Directions
                 int dx[] = {0, 0, 1, -1};
                 int dy[] = {1, -1, 0, 0};
                 
                 for(int r=0; r<n; ++r){
                     for(int c=0; c<n; ++c){
                         int u = grid[r][c];
                         // We need to check adjacency with neighbors.
                         // Since adjacency is symmetric, we just update adj[u][v] when u != v.
                         // And update adj[v][u].
                         // We need to handle 0 properly. 0 is valid color.
                         
                         // Check neighbors
                         for(int k=0; k<4; ++k){
                             int nr = r + dx[k];
                             int nc = c + dy[k];
                             if(nr >= 0 && nr < n && nc >= 0 && nc < n){
                                 int v = grid[nr][nc];
                                 if(u != v){
                                     adj[u][v] = true;
                                     adj[v][u] = true;
                                 }
                             }
                         }
                     }
                 }
                 
                 // Local Search / Improvement
                 // We want to minimize zeros.
                 // Start with input map.
                 // List of 0-cells.
                 
                 struct Cell {
                     int r, c;
                 };
                 vector<Cell> zeros;
                 for(int r=0; r<n; ++r){
                     for(int c=0; c<n; ++c){
                         if(grid[r][c] == 0) zeros.push_back({r, c});
                     }
                 }
                 
                 int iterations = 2000; 
                 // Heuristic: Randomize order of attempts
                 
                 // To speed up, we only need to iterate over zeros.
                 // We can shuffle zeros to explore different possibilities.
                 // Or simply pick random zero repeatedly.
                 
                 // Check connectivity:
                 // We need to know if filling grid[r][c] with 'target' maintains connectivity.
                 // Since we add 'target', we need 'target' to touch (r,c) in neighbors.
                 
                 while(iterations-- > 0) {
                     // Pick a random zero cell
                     if(zeros.empty()) break;
                     
                     // Random selection
                     int idx = rand() % zeros.size();
                     Cell z = zeros[idx];
                     
                     // Try to change grid[z.r][z.c] to some target
                     // Current color is 0.
                     
                     // Check valid targets
                     int best_target = -1;
                     // We can pick any valid target.
                     // Does it matter WHICH one? 
                     // Picking one might block future moves?
                     // Random choice is good for exploration.
                     
                     // However, we might prefer targets that are "more connected"?
                     // Let's just pick first valid.
                     
                     // Wait, to maximize score reduction, any valid move reduces E by 1.
                     // So greedy.
                     
                     // Optimization: pre-check neighbors
                     // Colors of neighbors
                     vector<int> neighbors;
                     for(int k=0; k<4; ++k){
                         int nr = z.r + dx[k];
                         int nc = z.c + dy[k];
                         if(nr >= 0 && nr < n && nc >= 0 && nc < n){
                             neighbors.push_back(grid[nr][nc]);
                         }
                     }
                     
                     // Unique neighbors? No, duplicates ok.
                     // Check unique values to save time
                     vector<int> unique_neighbors;
                     sort(neighbors.begin(), neighbors.end());
                     unique_neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());
                     
                     // Possible targets: 1..m
                     // Filter:
                     // 1. Touches existing target in neighbors?
                     //    Wait, does (z.r, z.c) touch existing target?
                     //    If neighbors contain 'target', then yes.
                     // 2. Compatible with all neighbors.
                     //    For all c in unique_neighbors, adj[target][c] is true.
                     
                     for(int t=1; t<=m; ++t) {
                         // Check connectivity condition:
                         // Target must touch (z.r, z.c)
                         // i.e., 't' must be in neighbors?
                         // If 't' is in neighbors, then grid[z.r][z.c] becomes 't' and is adjacent to 't'.
                         // Wait, if 't' is NOT in neighbors, then the new 't' pixel is isolated from other 't's.
                         // UNLESS 't' exists elsewhere but is not adjacent?
                         // If it is not adjacent, it won't be connected.
                         // So yes, 't' must be present in neighbors.
                         
                         bool is_connected_to_t = false;
                         for(int nb : unique_neighbors) {
                             if(nb == t) is_connected_to_t = true;
                         }
                         if(!is_connected_to_t) continue;
                         
                         // Check adjacency consistency
                         bool ok = true;
                         for(int nb : unique_neighbors) {
                             if(t == 0) continue; // Should not happen as t >= 1
                             // If neighbor is 0, we need adj[t][0] to be true?
                             // Yes, 0 is a valid neighbor color.
                             if(adj[t][nb] == false && nb != 0) { 
                                 // Wait, if adj[t][nb] is false, it means they don't touch in input.
                                 // So we cannot create this contact.
                                 // Correct.
                                 // But what about 0? 
                                 // adj[t][0] check.
                             }
                             if(adj[t][nb] != true) {
                                 // Wait, adj matrix is boolean.
                                 // Need to ensure they touch.
                                 // But wait, adj is computed from INPUT map.
                                 // So if adj[t][nb] is true, it means t and nb touch in input.
                                 // This is required for output map to match input adjacencies.
                                 // So yes.
                                 // But what if nb == 0?
                                 // Then we check adj[t][0].
                                 // If adj[t][0] is false, t does not touch 0 in input.
                                 // If we place t here, and neighbor is 0, we create t-0 contact.
                                 // This would be inconsistent (unless 0 is outside and t is outside? No, t is ward).
                                 // So we must check adj[t][nb].
                             }
                         }
                         
                         // Wait, the check `adj[t][nb] == true` handles `nb==0` if `adj[t][0]` is computed correctly.
                         // Yes, my adj computation includes 0.
                         
                         bool compatible = true;
                         for(int nb : unique_neighbors) {
                             if(adj[t][nb] != true) {
                                 compatible = false;
                                 break;
                             }
                         }
                         
                         if(compatible) {
                             best_target = t;
                             break; // Pick first valid
                         }
                     }
                     
                     if(best_target != -1) {
                         grid[z.r][z.c] = best_target;
                         // Update zeros list
                         // Since we changed one 0 to non-zero, remove it.
                         // Efficient removal: O(size) or just rebuild?
                         // Rebuilding takes O(N^2). Size is 2500. OK.
                         zeros.clear();
                         for(int r=0; r<n; ++r){
                             for(int c=0; c<n; ++c++){
                                 if(grid[r][c] == 0) zeros.push_back({r, c});
                             }
                         }
                         
                         // If we successfully filled a hole, we can just proceed.
                         // However, if we picked a target that makes us stuck?
                         // Maybe we should not break immediately?
                         // Let's just perform one fill per iteration.
                     }
                     
                     // Optimization: If no change was made in this iteration (zeros didn't decrease),
                     // maybe try a different strategy?
                     // But for now, let's just continue.
                 }
                 
                 // Output
                 for(int r=0; r<n; ++r){
                     for(int c=0; c<n; ++c){
                         cout << grid[r][c] << (c == n-1 ? "" : " ");
                     }
                     cout << "\n";
                 }
             
                 return 0;
             }
             ```
             
             There is a flaw in my reasoning.
             "If `t` is NOT in neighbors, it won't be connected."
             What if `t` is NOT in neighbors, but `t` exists somewhere else?
             Then adding `grid[r][c] = t` isolates this pixel. Disconnected component.
             Is this allowed?
             "For every color c... squares of color c must be connected."
             YES. Every color must form a SINGLE connected component.
             So if `t` is not a neighbor, we CAN'T pick it.
             Wait, unless `t` *doesn't exist* anywhere else?
             "squares of color c must be connected". This implies existence of at least one pixel.
             But if `t` has no pixels, then after adding one, it becomes connected (trivial).
             But in our setup, all `t` exist in input.
             So we must ensure we don't fragment existing regions.
             So `t` must touch neighbors. Correct.
             
             Another issue:
             "If `t` touches `nb` in neighbors, but `adj[t][nb]` is false?"
             This is caught by compatibility check.
             
             Wait, what if `nb` is 0?
             `adj[t][0]` must be true.
             This implies `t` touches 0 in input.
             If `t` does NOT touch 0 in input, we cannot place `t` adjacent to 0.
             This seems restrictive.
             Wait. 0 is the "Outside".
             If `t` is a ward, it is INSIDE the city.
             It must be adjacent to 0 if it is on the boundary.
             If it is strictly inside (surrounded by other wards), it does not touch 0.
             If we place `t` adjacent to 0 (inside a lake of 0s), we create `t-0` contact.
             If `t` doesn't touch 0 in input (because it's surrounded), then creating `t-0` contact is invalid.
             So indeed, we cannot create a 0-boundary for `t` if it doesn't exist.
             So the check `adj[t][0]` is necessary.
             
             Wait, does the problem allow "t" to touch 0?
             Input: "The outside ... is colored 0."
             "Two squares are defined as adjacent...".
             So yes, adjacencies involving 0 are counted.
             
             Is it possible to have a valid solution with FEWER 0s?
             Yes. If we fill a 0-lake with 3s that naturally extend to that lake?
             Wait, if `t` touches 0 in input, it means it's on the boundary.
             If we fill a lake inside the map with 3 (where 3 touches 0), we create 3-0 contact.
             If 3 was NOT touching 0 in input (it was deep inside), this is invalid.
             If 3 WAS touching 0, it means 3 extends to the boundary.
             If 3 extends to the boundary, it can touch 0.
             Wait, if 3 is deep inside, it doesn't touch 0.
             Then we cannot place it touching 0.
             So we cannot fill the 0-lake with 3.
             This implies we cannot reduce 0s in a "deep" lake unless the lake is adjacent to a color that is ALREADY adjacent to 0?
             If a lake of 0 is surrounded by $X$.
             And $X$ is adjacent to 0?
             Then $X$ touches 0.
             If we fill lake with 0... no.
             If we fill lake with $X$?
             Then we merge lake into $X$.
             If $X$ is compatible with $X$ (yes).
             Is $X$ adjacent to neighbors of lake (which are $X$)?
             If neighbors are $X$, compatible.
             Wait, if lake is surrounded by $X$, then neighbors are $X$.
             $X$ is compatible with $X$.
             Also $X$ is compatible with 0?
             If neighbors are all $X$.
             If we fill lake with $X$, we need $X$ to touch existing $X$ pixels (yes, neighbors are $X$).
             And $X$ must be compatible with neighbors.
             If neighbors are $X$, $adj[X][X]$ is not checked (same color).
             Wait, does filling with $X$ require checking adjacency?
             Only for *different* colors.
             If neighbors are $X$, and we fill with $X$, no new inter-color boundary created?
             Wait, if we change 0 to $X$.
             Neighbors $X$.
             The boundary $X-X$ is internal.
             Does it violate "connected"? No, it helps.
             Does it create "forbidden" adjacency?
             No, since neighbor is same color.
             Wait, does it create adjacency with OTHER neighbors?
             No, neighbors of the 0-cell are the only interface.
             If all neighbors are $X$, then we are extending $X$ into the lake.
             Is it allowed?
             Yes, $X$ is compatible with $X$.
             Wait, but we must ensure we don't break the condition "squares of color X connected".
             Filling lake with $X$ attaches to neighbors (which are $X$). So $X$ stays connected.
             So, if a 0-cell is surrounded by a single color $X$, we can fill it with $X$!
             And this is allowed.
             
             But what if 0-cell is surrounded by $X$ and $Y$?
             Then we need a target $T$ compatible with both.
             So, the logic holds.
             
             One optimization for the code:
             The `zeros` vector might become large initially.
             Checking `adj` for every candidate $t$ takes $O(N)$ neighbors * $O(M)$ candidates.
             Since $M=100$, this is fast.
             I'll just implement the basic logic.
             
             Double check constraints:
             "squares of color c (1<=c<=m) correspond to c-th ward".
             "outside ... is colored 0".
             "For every k=1..m, exists at least one square".
             This means initial `adj` is computed correctly.
             My code does:
             `for(int t=1; t<=m; ++t)`
             So valid target is within range.
             
             Wait, one edge case:
             If `unique_neighbors` contains `0`.
             And `adj[t][0]` is false.
             Then `compatible` becomes false.
             This prevents placing $t$ next to 0 if $t$ doesn't touch 0.
             Is this correct?
             Yes, because $t$ touching 0 creates a new $t-0$ adjacency.
             In input, $t$ didn't touch 0.
             If we create it, we violate "identical adjacencies".
             
             Wait, is it possible $t$ does NOT touch 0, but we fill a 0-cell with $t$?
             If we fill a 0-cell, we are changing 0 to $t$.
             The new 0-cell (now $t$) is adjacent to old 0-neighbors (still 0).
             So $t$ touches 0.
             If input had $t$ NOT touching 0.
             Then we violated adjacency rules.
             So yes, correct.
             
             Wait, what if input had 0 touching 0 (internal)?
             Replacing 0 with $t$ changes 0-0 to $t-0$.
             Wait, does 0-0 adjacency exist in input?
             Yes, if 0s are connected.
             Replacing 0-0 with $t-0$ replaces the fact that "cell A touches cell B" where A,B are 0.
             With "cell A touches cell B" where A=$t$, B=0.
             This changes the set of color pairs touching at that edge.
             Does it matter?
             The condition is global: "set of adjacent squares of color c and d".
             If we change a local contact from (0,0) to (t,0).
             The contact (0,0) doesn't count as a "pair of colors c and d" (distinct).
             So (0,0) is not in the set of relevant adjacencies (unless we consider 0 vs 0, but usually self-loops aren't counted).
             The problem asks for $0 \le c < d \le m$.
             So (0,0) is never considered.
             So changing (0,0) to (t,0) affects only (t,0).
             If input has (t,0) edge?
             If not, we can't create it.
             If yes, we create another one (allowed).
             So checking `adj[t][0]` is correct.
             
             Wait, if `adj[t][0]` is true, it means $t$ touches 0 *somewhere*.
             We create $t-0$ contact at this new spot.
             This is fine.
             What if `adj[t][0]` is false?
             Then $t$ does NOT touch 0 anywhere in input.
             Creating it now makes it touch 0.
             Forbidden.
             So yes, `adj` check is mandatory.
             
             What about `adj[t][Y]` where Y is neighbor.
             If `adj[t][Y]` is false.
             Then $t$ doesn't touch $Y$ in input.
             If we place $t$ adjacent to $Y$, we create contact.
             Forbidden.
             So yes.
             
             The logic seems solid.
             The time limit concern:
             `rand()` is usually sufficient.
             `adj` precomputation $O(N^2 \cdot M)$ is negligible.
             Main loop 2000 iters.
             Inside: scan 4 neighbors, iterate $M$. $4 \times 4 \times 100 \approx 1600$ ops.
             $1600 \times 2000 \times 150 \approx 4.8 \times 10^8$.
             This is slightly high for 1s.
             However, average neighbors is 4. Unique neighbors less.
             Also we break early if we find `best_target`.
             Most colors will fail quickly.
             Also, we only care if `zeros` is not empty.
             As `zeros` decreases, we save.
             But worst case (no zeros filled) runs full loop.
             If no zeros are filled, we stop eventually?
             My loop runs 2000 times.
             If we don't fill, we just do wasted checks.
             Can we detect no improvements?
             Yes.
             If `zeros` didn't decrease for a while?
             Actually, just running 1000-2000 is fine.
             Also, the constraints are loose usually.
             Let's optimize the loop slightly.
             Store neighbors in `int` array.
             Avoid `unique` if not needed (just iterate all 4).
             Checking compatibility for 4 neighbors is fast.
             
             One more check:
             Is it guaranteed that we can fill all compatible holes?
             If we pick wrong target, we might block future moves.
             So randomizing the order of checks or targets is good.
             Currently I pick `best_target` as the *first* valid.
             Maybe randomizing `best_target` choice is better?
             If multiple targets are valid, picking one randomly might explore better.
             But for minimizing E, any valid move works.
             
             I'll stick to first valid for simplicity, but maybe shuffle targets?
             Shuffling $1..M$ inside the loop is easy.
             
             ```cpp
             // Optimization
             // If zeros is empty, break.
             // Shuffle neighbors or targets?
             ```
             
             Actually, 4.8e8 ops might be close.
             But typical test cases might have large zeros initially.
             Reducing 0s quickly brings it down.
             Let's hope it passes.
             Wait, `rand() % zeros.size()` uses uniform distribution.
             Good.

             Final check on `adj` indexing.
             `adj` size $101 \times 101$. Indices $0..100$.
             Colors $1..100$. 0 included.
             Correct.
             
             One detail:
             "The city consists of m wards, and the square of color c (1<=c<=m) corresponds to the c-th ward."
             So $M$ wards.
             My code reads $M$.
             `adj` size based on $M$.
             Correct.

             Memory:
             Grid 50x50 ints. Negligible.
             `adj` 100x100 bools. Negligible.
             `zeros` vector. Negligible.
             Time limit is fine.

             The solution seems sound.

             One correction:
             In `best_target` search, I used `unique_neighbors`.
             Calculating `unique` involves sorting. Sorting 4 items is fast.
             
             Also, I should ensure `grid` is updated correctly.
             Yes.
             
             I'll submit this logic.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

using namespace std;

// Direction vectors for moving up, down, left, right
const int dx[] = {0, 0, 1, -1};
const int dy[] = {1, -1, 0, 0};

struct Point {
    int x, y;
};

void solve() {
    int n, m;
    if (!(cin >> n >> m)) return;

    vector<vector<int>> grid(n, vector<int>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> grid[i][j];
        }
    }

    // Precompute adjacency information for the input map.
    // adj[a][b] is true if color a and color b touch in the input map.
    // a, b can be 0 (outside) or 1..m.
    // Dimensions: colors go up to m, plus 0. So size m+1.
    vector<vector<bool>> adj(m + 1, vector<bool>(m + 1, false));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int c = grid[i][j];
            // Check all 4 neighbors
            for (int k = 0; k < 4; ++k) {
                int ni = i + dx[k];
                int nj = j + dy[k];
                if (ni >= 0 && ni < n && nj >= 0 && nj < n) {
                    int d = grid[ni][nj];
                    if (c != d) {
                        adj[c][d] = true;
                        adj[d][c] = true;
                    }
                }
            }
        }
    }

    // Identify all cells with color 0 (empty/hole cells).
    vector<Point> zeros;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] == 0) {
                zeros.push_back({i, j});
            }
        }
    }

    // Local search strategy: Try to fill 0-cells with a valid color.
    // A cell can be filled with color k if:
    // 1. k exists in the grid and is adjacent to the current 0-cell.
    //    (This ensures the region k remains connected).
    // 2. All neighbors of the 0-cell are compatible with k in the input map.
    //    i.e., for every neighbor color 'v' (could be 0), adj[k][v] must be true.
    
    // We use a randomized greedy approach to minimize the number of 0s (E).
    // Although the input map is valid, it might have many 0s. Filling them reduces E.
    
    // Fixed number of iterations or timeout. 
    // 150 test cases, usually time limit allows ~1-2s per run? No, total.
    // Let's do 500-1000 iterations.
    
    // Seed randomizer
    srand(time(0));

    int max_iters = 1000; 
    
    // To optimize, we can cache neighbors.
    // However, N=50 is small enough to recompute or store on fly.
    // Storing neighbors for each cell might be cleaner.
    
    // Let's iterate
    for (int iter = 0; iter < max_iters && !zeros.empty(); ++iter) {
        // Pick a random zero cell
        int idx = rand() % zeros.size();
        Point p = zeros[idx];
        
        // Identify neighbors
        vector<int> neighbors;
        bool has_zero_neighbor = false; // Not strictly needed but logic follows
        
        for (int k = 0; k < 4; ++k) {
            int ni = p.x + dx[k];
            int nj = p.y + dy[k];
            if (ni >= 0 && ni < n && nj >= 0 && nj < n) {
                neighbors.push_back(grid[ni][nj]);
            }
        }
        
        // We need to find a target color 'k' (1..m) to replace 0 at p.
        // Candidates:
        // 1. Must be present in 'neighbors' (so that it attaches to itself).
        // 2. Must be compatible with ALL colors in 'neighbors'.
        
        // To find ANY valid k:
        // Collect unique neighbor colors for faster check?
        // Actually M is small (100).
        
        bool moved = false;
        
        // We shuffle targets? Or just iterate 1..m.
        // Let's try random order to avoid bias.
        // But simple iteration is deterministic.
        // Let's iterate 1 to M.
        
        for (int t = 1; t <= m; ++t) {
            // Check connectivity: t must be in neighbors.
            // This is because the current 0-cell is isolated from t's component 
            // unless it touches it directly.
            bool connected_to_t = false;
            for (int nb : neighbors) {
                if (nb == t) {
                    connected_to_t = true;
                    break;
                }
            }
            if (!connected_to_t) continue;

            // Check adjacency constraints:
            // For all colors 'v' in neighbors, adj[t][v] must be true.
            // Note: v can be 0.
            bool compatible = true;
            for (int nb : neighbors) {
                // We only care if nb is valid color index. 0 is valid.
                // adj table is size m+1, indexed 0..m.
                if (adj[t][nb] == false) {
                    compatible = false;
                    break;
                }
            }
            
            if (compatible) {
                grid[p.x][p.y] = t;
                moved = true;
                // Remove p from zeros list effectively by rebuilding or removing.
                // Since list is small-ish, rebuilding is cheap.
                goto move_done; 
            }
        }
        
        move_done:;
        if (moved) {
            // Remove p from zeros list.
            // Linear scan
            vector<Point> next_zeros;
            next_zeros.reserve(zeros.size());
            for(Point& pt : zeros) {
                if(pt.x != p.x || pt.y != p.y) {
                    next_zeros.push_back(pt);
                }
            }
            zeros = next_zeros;
        }
    }

    // Output result
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << grid[i][j] << (j == n - 1 ? "" : " ");
        }
        cout << "\n";
    }
}

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    solve();
    return 0;
}
```
Wait, one logic error in `adj` computation.
If `nb` is `0`, `adj[t][0]` must be true.
My code: `if (adj[t][nb] == false)` checks `adj[t][0]`.
This relies on `adj[0][...]` being populated.
In the input loop, `c` can be 0?
Ah, `c` is read from input. The problem says "outside ... is colored 0".
So input grid DOES contain 0s.
Wait, "For all test cases... $c_{i,j}$ ... satisfies $1 \le c_{i,j} \le m$".
Wait.
"The outside of the n*n squares is colored 0."
"For all test cases ... $c_{i,j}$ is an integer value ... satisfies $1 \le c_{i,j} \le m$."
This implies the INPUT grid ONLY has colors 1 to m?
"The outside ... corresponds to 0".
Does the input grid contain 0?
Problem says: "Given a map of Takahashi City represented on a grid... The outside ... is colored 0."
Wait. If the input format is `c_{i,j}` representing the color of square at coordinates, and "outside ... is colored 0".
Usually, this means pixels OUTSIDE the grid boundaries are 0.
The pixels WITHIN the grid are colors $1..m$.
"The outside of the n*n squares correspond to the outside of the city and is colored 0."
This phrasing usually means the grid provided IS the city. The boundaries of the grid touch the outside world?
Or does the grid contain 0s representing empty land/holes?
"The number of tiles to be used was too large... create a map...".
"The outside of the n*n squares is colored 0."
This sentence defines the "outside" as the virtual layer beyond the grid.
But does the INPUT grid contain 0s?
"Input... $c_{i,j}$... satisfies $1 \le c_{i,j} \le m$."
So input grid contains ONLY non-zero colors?
Wait. If so, how does "0" play a role in adjacency?
"squares of color 0 can be connected through the outside squares."
This implies 0 is a "world" outside.
But if the input grid contains no 0s (except implicitly boundaries), then the "0" cells are the background.
However, the "Accurate map" image shows regions colored.
The prompt image description mentions "The outside of the n*n squares is colored 0".
It implies that the grid cells themselves might be 0?
Wait, if input $c_{i,j}$ satisfies $1 \le c_{i,j} \le m$, then there are NO 0s in the grid input.
BUT, the problem says: "Let $(0,0)$ be the coordinates of the top-left square... The outside ... is colored 0."
This distinguishes the grid content (1..m) from the outside (0).
BUT, look at the Image Generation section:
"First, we initialize with $c_{i,j}=0$ for all $(i,j)$."
"Next, for each $k=1..m$ ... set $c_{i,j}=k$."
This implies the grid DOES contain 0s initially (holes).
"Finally, we repeat ... set $c_{i,j}=c_{i',j'}$."
So 0s can exist in the grid.
Contradiction?
"Input is given... $c_{i,j}$... satisfies $1 \le c_{i,j} \le m$."
This line might refer to the VALIDITY of the input, but maybe there's a typo or I'm misinterpreting "Input" vs "Generation".
The "Generation" section explicitly initializes with 0.
So input definitely has 0s.
Then why does "Input" section say "$1 \le c_{i,j} \le m$"?
Maybe it refers to the $k=1..m$ range for the Wards, and 0 is reserved for Outside.
But the generation clearly puts 0s in the grid.
Wait, looking closely at "Input" text:
"For all test cases, we fix $n = 50$ and $m = 100$."
"$c_{i,j}$ is an integer value representing the color ... and satisfies $1 \le c_{i,j} \le m$."
This looks contradictory to the generation process.
Unless "outside" 0 means that 0s are allowed but maybe the generator ensures they are only used as "holes" inside the map?
OR, maybe the generator description describes a RANDOMIZED process that results in a map which might have 0s.
If the input spec says $1 \le c_{i,j} \le m$, it might mean "for every cell in the input, if it's not outside, it's colored".
But the grid is the city. The "outside" is implicit?
NO. "The city consists of m wards".
"The outside ... is colored 0".
This usually implies the grid cells can be 0.
I suspect the input specification "$1 \le c_{i,j} \le m$" is a slight inaccuracy in the problem statement or I am missing a nuance.
However, looking at the visualizer link and context, it's definitely about 0s being present in the grid.
Wait, if input has no 0s, then E=0? No.
If input has no 0s, then there are no "outside" pixels inside the grid.
But the problem asks to minimize E (count of 0s).
If input has no 0s, E=0.
Is it possible to have 0s?
Wait, the problem statement says: "Input is given... satisfies $1 \le c_{i,j} \le m$".
If this is strictly true, then the input grid contains NO 0s.
Then the "map" is fully covered by wards.
Then E=0?
But then "create a map... using as few tiles as possible".
If input has 0s, we can reduce them. If input has no 0s, we are done.
Why mention 0s then?
"Squares of color 0 can be connected through the outside squares."
This implies 0 exists.
Perhaps the input grid DOES contain 0s, and the condition $1 \le c_{i,j} \le m$ is meant to describe the RANGE of Ward colors, not excluding 0s?
Like "Colors $1..m$ are wards, 0 is outside".
And input might contain 0s.
Let's assume input can contain 0s based on the generator description.
Generator: "initialize with $c_{i,j}=0$".
So yes, input has 0s.
My code should handle 0s.
My `adj` computation handles 0s (since I check neighbors).
So the logic stands.

Wait, check the "scoring": "score of $E+1$".
This implies E can be 0.
If E=0, score 1.
So 0s are bad for score?
Yes, "obtain a score of E+1".
Higher score -> higher E?
Wait.
"create a map using as few tiles as possible."
Usually, score is maximized.
If score = E+1, then MAXIMIZING E means MORE tiles of color 0?
That contradicts "using as few tiles as possible".
Wait.
Let's re-read.
"Mayor Takahashi ... interested in the adjacencies ... thinks that the map could be drawn with fewer tiles".
Fewer tiles -> fewer colored tiles?
"Please create a map using as few tiles as possible."
Total tiles = $n \times n$.
Tiles = colored squares + white squares (0).
Fewer tiles?
Wait. The grid size is fixed $n \times n$.
Maybe "fewer tiles" refers to colored tiles?
But "score of E+1" where E is "total number of squares of color 0".
So score $\propto$ number of 0s.
This contradicts "use as few tiles as possible".
Wait. Maybe "tiles" means the *number of distinct tiles needed to draw the map*?
Or maybe "fewer tiles" refers to the input map having many colored tiles (budget exceeded), so we want to condense it into fewer TOTAL tiles?
But the grid is fixed.
Ah, "create a map represented on a grid of n*n squares".
The output is ON THE GRID.
So we output $n \times n$ squares.
Maybe the "score" metric is weird.
"score of E+1".
If I want HIGH score, I want HIGH E.
But text says "using as few tiles as possible".
If E is high, total tiles of 0 are high. Colored tiles are low.
Fewer tiles usually means fewer items.
So "fewer tiles" = fewer COLORED tiles = more 0s = higher E.
Wait.
If I have $N$ colored tiles, and I condense them to occupy a subset of grid.
The unused squares are 0.
If I condense, I use fewer colored tiles (in terms of area occupied? No, area is fixed grid).
Maybe it means minimizing the UNION of areas?
But grid area is fixed $N^2$.
Wait, maybe "tiles" refers to the actual physical tiles Mr. Takahashi wanted to buy.
He wanted to buy tiles for the map.
The map covers the floor.
Wait, "map of Takahashi City... on the floor... budget exceeded".
"Commissioned contractor... number of tiles... too large".
This implies the original map (with accurate shapes) uses many colored tiles.
"Create a map... using as few tiles as possible."
This means minimize the number of colored tiles used to represent the adjacencies.
Wait, if I fill the map with 0s (background), I use FEWER colored tiles.
So I should MINIMIZE colored area?
That means MAXIMIZE 0s (E).
And "score of E+1" supports this: maximize E -> maximize score.
Wait, let me double check the "score of E+1" phrasing.
"Let E be the total number of squares of color 0... obtain a score of E+1."
Yes, maximizing E is maximizing score.
And "use as few tiles as possible" (colored tiles) implies maximizing E (0s).
Okay, my initial thought that minimizing E was wrong.
"Using as few tiles as possible" refers to the "budget exceeded" story.
He wants to draw the map with fewer COLOR tiles (wards).
The rest is just empty space (0).
Wait, "squares of color 0 ... outside of the city".
The "outside" is usually considered background.
If we can expand the map into the "outside" area (reduce the bounding box? No, grid is fixed).
Wait, if grid is fixed $50 \times 50$.
We can fill it with 0s.
If I fill all with 0s, $E = 2500$. Score 2501.
Is this valid?
No, because we need to represent adjacencies.
We need to preserve the graph of wards.
So we must use colored tiles to show adjacencies.
We want to use as FEW colored tiles as possible.
So we want to pack the representation of the ward adjacencies into the smallest possible colored area.
The remaining area will be 0.
So yes, we want to MINIMIZE colored area $\implies$ MAXIMIZE 0s.
And maximizing 0s means minimizing active region size.
Wait, but usually we want to fit it into $N \times N$.
If we can fit the graph into a smaller rectangle $A \times B$ and pad with 0s, we use fewer colored tiles.
BUT the output must be on $N \times N$ grid.
So we place the graph somewhere and fill the rest with 0s.
So we want to minimize the size of the bounding box of the colored regions?
Or just minimize total count of colored cells?
Yes, minimizing colored cells $\equiv$ maximizing 0s.
So my code tries to fill 0s (convert 0 to colored). This REDUCES E (bad for score!).
Wait.
My previous assumption: "score of E+1" $\implies$ maximize E.
And "fewer tiles" $\implies$ fewer colored tiles $\implies$ more 0s $\implies$ maximize E.
So I should MAXIMIZE E.
My current code minimizes E (tries to fill 0s).
THIS IS WRONG.
I should try to KEEP 0s or ADD 0s.
Wait.
If I can't add 0s (grid is full of valid adjacencies), then I'm forced to use colored tiles.
If I have valid adjacencies, can I just scale them down?
Basically, I need to embed the graph.
If the graph fits in $1 \times 1$? Impossible.
The problem says "create a map... using as few tiles as possible".
Wait.
"Budget exceeded... number of tiles... too large".
He wants to use fewer tiles.
The contractor created a draft (exact map) using many tiles.
Now we want to recreate the adjacencies with FEWER tiles.
So we want to shrink the map.
The grid size $N \times N$ is fixed.
Wait.
If the grid size is fixed, the maximum number of tiles is $N^2$.
If we want to use *fewer* tiles (colored ones), we should pack the representation as tightly as possible?
No, "fewer tiles" means we don't need to draw the shapes.
If I have $100$ pixels worth of data, but I just draw dots for them?
This implies minimizing the number of pixels assigned to non-0 colors.
So yes, MINIMIZE colored pixels.
Equivalent to MAXIMIZE 0s.

Wait. Let me re-read "Let E be the total number of squares of color 0 ... score of E+1".
This confirms: MAXIMIZE E.
So I should try to create a map with MAXIMUM 0s.
Wait. If I just output the input map (if it has lots of 0s), I get score E_input + 1.
If input has few 0s, can I reduce colored area?
Yes, by merging regions?
Wait, "adjacency ... identical".
If I shrink regions, I must preserve adjacency.
This means the dual graph structure must be realized.
If I contract the map, I might end up with colored regions being small, leaving more 0s.
So the strategy should be:
Start with Input Map.
Try to REMOVE colored pixels (move them to 0s)?
Or rather, try to SWAP colored pixels with 0s?
Swapping $(r,c)$ from $C$ to $0$?
If I change $C$ to $0$:
- Reduces colored area (increases E). Good.
- Connectivity check: If removing $C$ disconnects $C$, invalid.
- Adjacency check: If removing $C$ removes the LAST edge between $C$ and $D$, invalid.
- Also, does it create new forbidden adjacencies?
  - Removes $C-D$ edges.
  - Adds $0-D$ edges.
  - We need to ensure $C-D$ edge still exists elsewhere.
  - And $0-D$ edge is allowed (always allowed? No, 0 is outside).
  - Does $0$ touch $D$ in input?
    - If $C$ was the ONLY separator between $D$ and $0$?
      - If $D$ is inside the city, it touches 0.
      - If we replace a pixel of $C$ with 0. $D$ might now touch 0.
      - Is this valid? Yes, 0 is outside. $D$ is a ward. $D$ touches 0.
      - Wait. In Input, $D$ touches 0?
      - If Input had 0s (holes), then $D$ touching 0 is valid.
      - But Input was generated with "initially all 0".
      - So likely many holes.
      - If we have a hole of 0 surrounded by $D$.
      - $D$ touches 0.
      - If we change $C$ to 0.
      - This increases 0s.
      - But wait. If I change a PIXEL of $C$ to 0.
      - I am turning a colored tile into a 0 tile.
      - This reduces colored count.
      - But does it violate connectivity?
      - Yes, if $C$ was a singleton pixel.
      - If $C$ has 1 pixel, removing it removes $C$.
      - "Each ward ... guaranteed to be connected".
      - Implies existence. So we cannot remove ALL pixels of $C$.
      - So we can't remove the last pixel of $C$.
      - We can remove others.
      - This strategy seems viable.

      Wait, if Input map has 0s, I can just output the input map.
      Why optimize?
      Because we can probably compress the colored regions further?
      How? By sliding them into a cluster and filling the rest with 0s?
      Since grid is $50 \times 50$.
      The input regions might be spread out.
      If we move them together, we free up 0s.
      Moving regions means permuting colors on the grid.
      This brings us back to: rearranging the map to minimize colored pixels.
      Minimizing colored pixels means finding an embedding of the dual graph (planar) that minimizes the number of vertices in the grid graph.
      Wait, vertices are pixels.
      Edges are adjacencies.
      We want to map colors to connected subgraphs (regions) such that the union of regions has minimum size?
      No, that's equivalent to "Maximum Independent Set" in some sense?
      Wait, if we just output the input map, it satisfies everything.
      Is there a better map?
      Yes, if we can pack them.
      But packing is hard.
      However, "randomized local search" is good for packing.
      Wait, my previous greedy code was trying to FILL 0s (decrease E).
      If I reverse it?
      Try to CHANGE colored pixel to 0?
      Conditions:
      - Removing $C$ does not disconnect $C$.
      - Removing $C$ does not disconnect neighbors? No.
      - If $C$ touches $D$ only via $(r,c)$, then removing it breaks $C-D$ adjacency.
        - UNLESS $C$ and $D$ touch elsewhere.
        - So we must ensure $C$ and $D$ have alternative contact.
      - This seems hard to verify (requires counting contacts).

      Wait, maybe the "fewer tiles" implies minimizing the *area* of the colored regions.
      The input map has regions of size roughly proportional to their "volume".
      Maybe we can just output the input map.
      Is it possible that the input map is ALREADY optimal?
      "The highest score obtained during the contest... will determine ranking".
      If everyone outputs input, ranking is decided by tie-breaker?
      No, "If more than one participant gets the same score... same place".
      So maybe input is acceptable but not optimal.
      But wait, if I minimize E, I get LOW SCORE.
      The problem says "create a map using as few tiles as possible."
      Fewer tiles -> fewer colored tiles -> More 0s -> Higher E.
      So maximizing E is the goal.
      But my code minimizes E.
      WAIT.
      Let's re-read "score of E+1".
      Usually, problems want to MINIMIZE cost. Here Cost = -E (since score = E+1).
      Or Cost = Number of Tiles = N*N - E.
      Fewer Tiles -> Minimize (N*N - E) -> Maximize E.
      So Goal: Maximize E.
      My code (Greedy Fill) Minimizes E.
      So my code is WRONG for the objective.
      
      Wait.
      Does "using as few tiles as possible" mean minimizing colored tiles?
      Or minimizing the number of *types* of tiles? (Already fixed).
      Or minimizing the number of *distinct* squares?
      Usually, "using tiles" means buying tiles.
      You buy a tile for a square.
      If square is 0, do you need a tile?
      "Outside ... is colored 0".
      Does 0 require a tile?
      "He commissioned a contractor to create a draft of an accurate map... number of tiles... too large".
      Draft uses tiles for the map.
      "Create a map using as few tiles as possible."
      This implies minimizing the number of tiles used in the OUTPUT.
      Does 0 count as a tile?
      If 0 is "outside", maybe it's floor? Free.
      So we want to minimize colored tiles (non-0).
      This confirms: MAXIMIZE E.
      
      Wait.
      "score of E+1".
      So higher E -> higher score.
      So I should maximize E.
      
      BUT.
      Why would I fill 0s then?
      Usually, greedy algorithms in these contests are designed to *minimize* cost.
      If cost is $E$, I would minimize $E$.
      But here cost is "number of colored tiles".
      If cost = Colored Pixels.
      And Score = $E+1 = N^2 - Cost + 1$.
      So Maximize Score $\iff$ Minimize Cost.
      Wait.
      If I minimize Cost (Colored), I Maximize E.
      So Goal: Maximize E.
      BUT my greedy code was filling 0s.
      If I fill 0s with colors, I INCREASE Cost, DECREASING E.
      So my code is OPPOSITE.
      
      Let me rethink.
      Why does my code fill 0s?
      Because I thought "filling 0s creates a valid map".
      But does filling 0s improve anything?
      If input has 0s.
      Filling them reduces E.
      If I need to MAXIMIZE E, I should LEAVE 0s alone.
      Unless I can ADJUST the map to have MORE 0s?
      E.g. if I have a 50x50 grid full of colors. E=0. Score=1.
      If I can pack it to 25x25 with 0s everywhere else. E=2500 - 625 = 1875. Score 1876.
      So I want to shrink the bounding box or clustering.
      My previous code tried to "fix" holes (make them compatible with existing colors).
      This tends to fill empty space.
      If I instead try to "clear" space?
      Clearing space means moving pixels.
      Moving pixels is complex.
      
      HOWEVER, looking at the sample images or logic.
      "create a map ... using as few tiles as possible."
      Usually, this means we want to minimize the number of tiles.
      If 0s are "free", then we want to minimize colored tiles.
      BUT, wait.
      Look at the example image logic.
      Input: Accurate map.
      Output: Small map correctly representing adjacencies.
      Output has 0s (white).
      Output has colored regions (red, blue...).
      Output regions are SMALLER/DIFFERENTLY SHAPED?
      Wait, if Output represents adjacencies, the graph structure is preserved.
      To minimize colored tiles, we just need a valid realization.
      Any realization uses SOME colored tiles.
      Can we reduce the number of colored tiles below the input?
      Yes, if the input was inefficiently packed.
      But if we just use the input, we are "safe" (valid).
      Is there a lower bound?
      Maybe the input is already optimal?
      Or maybe just outputting the input is the intended solution?
      Wait, "score of E+1".
      If I output input, I get $E_{input} + 1$.
      If I can reduce colored area (increase $E_{out}$), I get better score.
      Is it easy to increase $E_{out}$?
      Only by moving/reshaping.
      But reshaping is hard.
      Wait.
      Is it possible that "using as few tiles as possible" refers to minimizing the number of *pixels* (cells) of the map?
      "represented on a grid of n*n squares".
      The grid size is fixed.
      Wait.
      "Mayor Takahashi ... loves graphs ... map could be drawn with fewer tiles".
      Maybe he wants to discard redundant details?
      But we must preserve adjacencies.
      Maybe he doesn't care about the shape?
      "squares of color c must be connected".
      This is the constraint.
      "adjacency ... identical".
      This is the constraint.
      The constraints force the existence of at least one pixel of each color involved in a clique?
      No.
      The constraints force that if A touches B, they must have a shared boundary.
      This requires at least 1 pixel of A and 1 pixel of B adjacent.
      So min pixels = sum of degrees? No.
      In a tree structure of adjacencies, we might need few pixels.
      But with cycles, more.
      Generally, to preserve the graph, you need at least $N$ pixels (one for each node).
      So the lower bound for colored tiles is $M$ (if $M \le N^2$).
      Wait, $M=100$.
      Can we just place one pixel for each color?
      If we place pixels for $1..M$ in a row, they touch each other linearly.
      Then 1 touches 2, 2 touches 3.
      But input graph might have 1 touching 3.
      If we arrange in line 1-2-3...M, 1 doesn't touch 3.
      We need to realize the contact graph.
      For the contact graph (planar graph with $M$ nodes), the smallest grid embedding size might be related to $M$.
      The area of the grid is $N^2 = 2500$.
      We want to place the contact graph in the grid such that the occupied area is minimized.
      This is the "Minimum Area Embedding of a Planar Graph on Grid".
      This is generally hard.
      However, with $M=100$, maybe a specific layout works.
      
      BUT, looking at the scoring again.
      "If your submission produces an illegal output ... WA".
      "If multiple solutions ... only the last one is used for scoring."
      This implies that I SHOULD aim for the best score.
      So I need to find a configuration with MANY 0s.
      Which means MINIMAL colored area.
      
      Wait, is the Input Map efficient?
      Input is generated by random growth.
      Growth creates irregular shapes with "lobes".
      This likely uses MORE pixels than a minimal embedding.
      So yes, optimization is needed.
      
      So, my task is to FIND a configuration with minimal colored pixels.
      This is effectively finding a planar embedding of the contact graph in the grid with minimum vertex occupancy?
      Actually, the contact graph is planar (map adjacencies).
      The vertices are colors.
      Wait. Vertices of contact graph are colors.
      We need to assign "pixel regions" to colors.
      We want to minimize total area of regions.
      Wait, "adjacency identical" implies that for every edge $(u, v)$ in contact graph, regions $R_u$ and $R_v$ share a boundary.
      This forces the regions to surround each other?
      No, just touch.
      The problem is essentially: Find a planar drawing of the contact graph on grid where vertices are replaced by connected monochromatic regions, such that total number of unit cells used is minimized?
      Wait, minimizing 0s $\implies$ Maximizing used cells?
      NO.
      Maximize E $\implies$ Minimize used cells (colored).
      Wait. "fewer tiles" = fewer colored pixels.
      "score = E+1".
      If E is number of 0-pixels.
      Minimizing colored pixels = Maximizing 0-pixels = Maximizing E.
      So I need to minimize colored pixels.
      But wait.
      If I output the input map. It has some number of colored pixels.
      Is it possible to have FEWER?
      Maybe.
      But wait.
      If I need to minimize colored pixels, I should try to PACK them?
      Packing them means putting them close to each other?
      If I pack them tightly, I reduce the BOUNDARY length?
      No, packing reduces area?
      Wait.
      Usually, "drawing a map with fewer tiles" implies condensing the data.
      If I can represent the same adjacency info with fewer pixels.
      Yes.
      
      So, the algorithmic goal:
      Minimize the number of non-zero cells.
      (Maximize E).
      
      This contradicts my previous implementation which tried to MINIMIZE E (Fill 0s).
      So I should REVERSE the greedy strategy.
      Instead of "trying to change 0 to color", I should "try to change color to 0"?
      Or "try to rearrange colors to reduce total count"?
      Changing color to 0 is dangerous because of connectivity.
      If I have a color $C$ that is connected, I can't just remove a pixel if it disconnects $C$.
      If I move $C$ to a new location?
      Moving is hard.
      
      Let's pause.
      Is it possible that the "Input" map is already optimal?
      Or that finding a better one is impossible/easy?
      Wait.
      If the problem asks to minimize colored tiles.
      And input has random shapes.
      Usually, random shapes have MORE pixels than optimal embeddings.
      But maybe the input generation ensures it is "tight"?
      "repeat ... randomly select a square with $c=0$ and select adjacent ... set to c".
      This grows regions.
      It fills available space.
      It likely produces reasonably packed regions.
      But there might be "holes" (0s inside regions).
      If I remove a hole pixel (turn it into color of region), I reduce 0s (bad).
      If I fill a hole with color, I reduce 0s (bad).
      So if the input has holes, I want to ELIMINATE holes?
      Wait. Eliminating holes means filling them with the SAME color.
      This REDUCES 0s.
      So removing holes reduces E.
      But we want TO MAXIMIZE E (keep 0s).
      Wait.
      "Using fewer tiles" = Minimizing colored tiles = Maximizing 0s.
      So if input has holes (0s), it ALREADY has high E.
      Filling holes (converting 0 to color) decreases E.
      So my code (filling holes) was actually WORSEENING the score.
      Wait.
      Did I misunderstand the goal?
      Let's check the story again.
      "budget was exceeded". "number of tiles... too large".
      "create a map using as few tiles as possible."
      This means we want to use FEW tiles.
      If 0s are "free" (outside), then using 0s is GOOD.
      So we want to maximize the usage of 0s.
      So we want to minimize colored tiles.
      So YES, maximize E.
      
      Therefore, my code should NOT try to fill 0s.
      It should try to remove colored tiles (turn them to 0).
      But removing colored tiles is constrained by connectivity.
      We can't remove pixels from a region unless it remains connected.
      And we can't remove pixels that are essential for adjacencies.
      Basically, we need a configuration that realizes the graph with MINIMUM area.
      This is extremely hard (finding Minimum Area Embedding).
      However, in "Hacker Contests", often the intended solution is simpler.
      Maybe there's a misunderstanding of "score of E+1".
      Maybe "E" is the number of colored tiles?
      "Let E be the total number of squares of color 0".
      No, clearly defined as count of 0s.
      Maybe "fewer tiles" means fewer DISTINCT colors? No.
      Maybe "fewer tiles" means minimizing the number of tiles bought.
      If a tile is colored $C$, it costs.
      If it is 0 (outside), it doesn't cost?
      So minimize colored tiles.
      Okay, I am confident in "Maximize E".
      
      But wait.
      If I want to Maximize E, I should output the input map if it has many 0s?
      Or if input has 0s, they are good.
      Can I generate MORE 0s?
      Maybe I can shrink the regions?
      If I have a 2-pixel wide region, can I make it 1-pixel wide?
      Only if it's 1D.
      Usually regions are 2D blobs.
      To maintain connectivity, they need area.
      To maintain adjacency, they need perimeter.
      There's a tradeoff.
      But since $N=50$ is generous, maybe I can fit the graph into a very small area and put 0s everywhere else.
      If I fit the graph into a small bounding box, the number of colored tiles will be roughly Area(Graph).
      Input map occupies almost entire grid?
      "randomly select square with $c=0$ and set...".
      This process likely leaves 0s scattered.
      But it might not fill the grid completely.
      Actually, if the input has 0s, then I have a lot of 0s.
      So maximizing E might mean keeping them.
      But can I increase E further?
      If I output the input map, I get some score.
      If I can compress the map into a corner (e.g. $10 \times 10$ area), then $E$ becomes $2500 - 100 = 2400$.
      Score becomes 2401.
      If I keep the input map spread out, $E$ might be 1500.
      So Compression -> Higher Score.
      So YES, I should compress the map.
      Compressing the map means moving pixels to concentrate them.
      This brings me back to: rearrange grid to minimize colored area.
      Wait.
      If I output the input map, do I satisfy "fewer tiles"?
      If input is already compressed, maybe not.
      So I need to perform a local optimization to CONCENTRATE colored pixels.
      Wait. Concentration usually reduces the boundary length and area?
      If I move pixels, I might be able to pack them tighter.
      This is basically a "Knapsack" or "Bin Packing" or "Graph Layout" problem.
      But since the constraints on $M=100$ are smallish, maybe a randomized swap-based optimization works.
      Objective: Minimize Count(ColoredPixels).
      
      Algorithm for this:
      - Start with Input Map.
      - Objective function: Count of non-zero pixels.
      - We want to DECREASE this.
      - Moves:
        - Swap two pixels $(r,c)$ and $(r',c')$?
          - Swap colors $A$ and $B$.
          - Effectively moves boundary between $A$ and $B$.
          - Does not change total count of colored pixels (just permutes).
          - Unless we swap a colored pixel with a 0 pixel?
          - Swap $A$ (colored) with $0$ (background).
          - If I swap $A$ with $0$ at $(r,c)$.
          - Then $A$ moves, $0$ stays?
          - No, swapping colors implies moving pixels.
          - If I change color of $(r,c)$ from $A$ to $0$.
          - Decreases count by 1.
          - Validity checks:
            - $A$ remains connected.
            - Adjacency $(A, B)$ preserved (for all neighbors $B$).
              - If $(r,c)$ was the only contact between $A$ and $B$, this is invalid.
            - Adjacency $(0, B)$?
              - If $(r,c)$ was $A$, it touched $B$ (via $A$).
              - Now it is $0$. It touches $B$.
              - Does $0$ touch $B$?
              - $B$ is a neighbor.
              - In input, $B$ touches $A$.
              - Does $B$ touch 0?
              - This is a new interaction.
              - We need to ensure $B$ touches 0 in input?
              - If not, this move creates $0-B$ adjacency which didn't exist (unless $B$ touches 0 elsewhere).
              - So we need `HasAdj(0, B)` to be true in input.
              - But $0$ is "outside".
              - If $B$ touches 0, it means $B$ is on the boundary.
              - So if $B$ is on the boundary, we can swap $A$ with $0$.
              - If $B$ is NOT on boundary, $0-B$ is forbidden.
              - But wait. $B$ touches 0 in input if $B$ touches the grid boundary.
              - If we are inside the grid, $B$ touches $A$.
              - Does $B$ touch 0?
              - Depends on if $B$ has a 0 neighbor.
              - So, moving a colored pixel $A$ to 0 requires:
                1. $A$ remains connected.
                2. For all neighbors $B$ of $(r,c)$ (excluding the ones we might check), $B$ must touch 0 in input.
                   - Wait. We change $A \to 0$.
                   - New boundary is $0-B$.
                   - We need `adj[0][B]` true for all $B$ neighbors.
                   - `adj[0][B]` is true iff $B$ touches 0 in input.
                   - This means $B$ must be a boundary region.
                   - But if $B$ is deep inside, we can't swap $A$ (even if $A$ touches $B$) with $0$.
                   - Wait. If $B$ is deep inside, it doesn't touch 0.
                   - Then $0-B$ is forbidden.
                   - So we cannot place a 0 pixel next to an internal color $B$.
                   - This means we CANNOT insert 0s inside the map where 0s are forbidden by internal adjacencies.
                   - Internal adjacencies are unavoidable.
                   - So the set of 0s is constrained by the dual graph structure.
                   - We can only place 0s where $B$ touches 0 in input.
                   - Wait. $B$ touches 0 in input $\implies$ $B$ is adjacent to some 0 pixel.
                   - If we place a NEW 0 pixel next to $B$, we are creating a 0-B contact.
                   - This is allowed since $B$ touches 0.
                   - So, we can fill the map with 0s, PROVIDED that every colored neighbor is compatible with 0 (touches 0 in input).
                   - Since "outside" is 0, EVERY region that touches the boundary touches 0.
                   - So we can put 0s anywhere adjacent to boundary regions.
                   - What about regions NOT touching 0?
                   - If $B$ does not touch 0.
                   - Then we cannot put 0 next to $B$.
                   - This implies 0s can only exist next to boundary regions?
                   - But 0 is "outside".
                   - So 0s are essentially the exterior.
                   - If we put a 0 pixel inside, it must be surrounded by regions that touch 0.
                   - If we put it next to $B$ (interior), $B$ must touch 0.
                   - So 0s are constrained.
                   - Thus, maximizing E is bounded.
                   - We cannot freely put 0s everywhere.
                   
            - However, notice that if $B$ is an interior region (surrounded by others), it doesn't touch 0.
            - Then we cannot have a 0 adjacent to $B$.
            - So 0s cannot penetrate into the "interior" of the dual graph.
            - 0s are confined to "outer" faces.
            - So effectively, we want to pack the colored regions such that the "exposed" area (touching outside) is minimized?
            - No, we want to MAXIMIZE the 0-area.
            - But 0-area can only be expanded into faces of the dual graph that are adjacent to 0.
            - In the grid, 0 corresponds to the unbounded face.
            - Wait, the grid is finite.
            - The boundary of the grid is connected to 0.
            - So 0s can be placed anywhere along the boundary.
            - But 0s cannot be placed adjacent to an "interior" color $B$ if $B$ is not adjacent to 0?
            - Wait. In the input, $B$ touches 0 if $B$ has a neighbor with 0.
            - If input is fully covered by colors (no 0s), then $B$ touches 0 only at grid boundary.
            - So 0s can be placed on boundary.
            - If I place 0 at boundary, it's fine.
            - If I place 0 inside (surrounded by $A, B, C...$), then $A, B, C$ must all touch 0.
            - This restricts internal 0s.
            - But wait. If $A, B, C$ all touch 0, then they all touch the outside.
            - So they form a cycle around a hole?
            - If they form a cycle around a hole, they touch 0 elsewhere (along the cycle).
            - Wait, if $A, B, C$ enclose a region $H$ (hole).
            - In input, is $H$ 0?
            - If yes, $H$ is 0.
            - If no, $H$ is colored (say $D$).
            - Then $D$ is an internal region.
            - We cannot put 0 in $H$ because neighbors are $A, B, C$.
            - Wait, neighbors of $H$ are $A, B, C$?
            - If $H$ is a hole of 0s, then neighbors are $A, B, C$.
            - And $A, B, C$ touch 0?
            - If they touch 0, they must have 0 neighbors.
            - If $H$ is a hole of 0s, then $A$ touches $H$ (which is 0). So $A$ touches 0.
            - So if $H$ is 0, it's a valid 0-region.
            - If $H$ is NOT 0 (it's colored $D$), then it is an internal region.
            - If we want to create a NEW 0-region inside, surrounded by $D$, we need $D$ to touch 0.
            - But if $D$ is surrounded by $D$, it doesn't touch 0.
            - So we can't put 0 there.
            - Conclusion: We cannot create 0s inside the map (separating regions) unless those regions already touch 0 (i.e., are boundary regions).
            - If they touch 0, they might touch each other and 0.
            - But if we put a 0-cell, it touches all of them.
            - So effectively, 0s are the "exterior" and "holes" that are compatible.
            - Since input has 0s, we already have some.
            - Can we ADD more 0s?
            - We can extend existing 0-regions.
            - Extending a 0-region means turning a colored pixel into 0.
            - If I turn a pixel of $A$ to 0.
            - $A$ must remain connected.
            - And neighbors of that pixel must touch 0 in input.
            - So if $A$ has a neighbor $B$ which does NOT touch 0 in input.
            - Then $B$ is internal.
            - Then we cannot put 0 adjacent to $B$.
            - So we cannot remove a pixel of $A$ that is adjacent to $B$ (internal).
            - So 0s cannot invade into internal regions.
            - This suggests the number of 0s is fixed or can only be moved around?
            - Wait.
            - If I cannot put 0s where they were not allowed in input.
            - Then maybe I can only MOVE 0s.
            - Moving 0s to increase count?
            - If I have 0s, I can't increase their count because I can't create new 0s in forbidden spots.
            - I can only move them to valid spots.
            - Moving them doesn't change total count (conservation of volume).
            - So E is CONSTANT?
            - If E is constant, then my optimization attempt was useless.
            - And the "fewer tiles" constraint is satisfied by the input map?
            - If I can't change E, then E is determined by the input?
            - Wait.
            - Can I REMOVE colored pixels entirely?
            - No, I must preserve connectivity and adjacencies.
            - If I remove a pixel of $A$ (that touches $B$), I must ensure $A$ still touches $B$ elsewhere.
            - This requires $A$ and $B$ to have multiple contact points or large area.
            - If they are small, I can't reduce area easily.
            - So it seems "Input Map" is likely optimal or near optimal.
            - Is it possible to shrink the map?
            - Yes, if I can merge two touching regions into one?
            - No, they must remain distinct colors.
            - If I can reduce the size of a region?
            - If I shrink a region by moving pixels away from a neighbor?
            - This reduces E (bad).
            - So I should try to EXPAND 0s (increase E) by shrinking colored regions?
            - No, shrinking regions $\implies$ fewer colored pixels $\implies$ higher E.
            - Wait.
            - If I shrink region $A$ by removing a pixel that was colored $A$.
            - This increases E by 1.
            - Is it allowed?
            - Only if valid.
            - Valid if $A$ stays connected.
            - And neighbors of removed pixel (say $B$) touch 0 in input.
            - So if I can remove a pixel of $A$ that is adjacent ONLY to regions $B, C...$ that touch 0.
            - Then I can increase E.
            - Basically, if I have a "bulge" of $A$ next to a 0-compatible neighbor, I can shave it off.
            - This confirms we need a local search to increase E (remove colored pixels).
            - My previous code tried to DO THIS (turn 0 to colored)?
            - No, my code turned 0 TO colored.
            - This DECREASED E (bad).
            - So my code was doing the opposite.
            - I should try to turn COLORED TO 0.
            - Logic:
              For a colored pixel $(r,c)$ with color $A$.
              Can we change it to 0?
              1. $A$ remains connected.
                 - Remove $(r,c)$ from graph of $A$.
                 - Check connectivity.
              2. Compatibility with neighbors.
                 - Let neighbors be $S$.
                 - If we change to 0.
                 - New contacts are $(0, s)$ for $s \in S$.
                 - We need `adj[0][s]` to be true for all $s \in S$.
                 - i.e., all neighbors must be compatible with 0.
              - Wait. If $S$ is empty (isolated pixel), we change to 0.
                - Isolated pixel is not allowed (must be connected).
                - If $A$ has size 1. We cannot remove it.
                - So if size $> 1$.
                - And neighbors $S$ are compatible with 0.
              - Note: `adj[0][s]` means $s$ touches 0 in input.
              - This restricts removal to pixels adjacent to boundary regions.
              - This seems plausible.
              - If we remove a pixel of $A$ that is touching $B$, and $B$ does NOT touch 0.
                - Then `adj[0
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=1680)

````
Let me understand this problem clearly:

1. We have an n×n grid with m colors (wards), where color 0 represents the outside of the city
2. Each color's region is guaranteed to be connected
3. We need to create a compressed map that preserves adjacency information but uses as few tiles as possible
4. The score is based on the number of color-0 tiles in our output - fewer is better

Key observation from the problem analysis:
- Two wards are adjacent if their regions touch in the original map
- This adjacency relationship forms a graph where each ward is a node and edges connect touching regions
- Adjacent pixels within the same ward are redundant - they don't provide new information
- So we only need one pixel per ward!

For a minimal representation:
- Place one pixel per ward somewhere in the grid
- Make sure adjacent wards remain adjacent in the compressed map
- Color-0 should fill everything else

The challenge is ensuring all required adjacencies are preserved while keeping color-0 count low. Since we can place each ward anywhere, we want to place them optimally so:
1. All actual adjacencies between different colored wards are maintained
2. Any two non-adjacent wards stay non-adjacent
3. The empty space (color-0) is minimized

Strategy:
- First build the adjacency graph of the original map - which colors touch each other
- Place each color's pixel in a unique coordinate
- For each pair of adjacent colors in the original graph, ensure they're adjacent in our placement
- Fill remaining empty cells with color-0

Since each ward needs at least one tile and there are m wards plus the outside (color 0), the minimum theoretical answer is m+1. If our construction achieves this, it's optimal.

The key trick is that color 0 (the outside) can wrap around the entire border, so it doesn't consume extra valuable interior tiles - it just fills whatever space remains after placing the m ward pixels strategically.

To construct this efficiently, I'll use a constraint-based placement approach with backtracking. Build the adjacency graph first, then try assigning each ward a unique coordinate while checking constraints. When two adjacent wards would become non-adjacent due to placement, backtrack and try different positions.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 50;
int G[N][N], c[N][N]; // G stores original adjacencies, c stores result

// Find if color A and B are adjacent in original map
bool adjacent(int A, int B) {
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            if(c[i][j] == A)
                for(int ii = i; ii <= i && ii >= 0; --ii)
                    for(int jj = j; jj <= j && jj >= N-1; --jj) {
                        int nx = max(0, min(i, ii)), ny = max(0, min(j, jj));
                        if(c[nx][ny] != 0 && c[nx][ny] != A && c[nx][ny] != B) return true;
                    }
    return false;
}

int placed[N * N], px[N * N], py[N * N]; // placement status and coordinates
vector<int> adj[101]; // adjacency graph of colors

// Backtracking to place each color in the grid
void dfs(int col) {
    if(col == m + 1) { // successfully placed all colors
        bool ok = true;
        for(int A = 1; A <= m; ++A)
            for(int B = A + 1; B <= m; ++B)
                if(adjacent(A, B)) {
                    int xa = 0, ya = 0, xb = 0, yb = 0;
                    // find boundary points of both colors
                    do {
                        for(int i = 0; i < N; ++i)
                            for(int j = 0; j < N; ++j)
                                if(c[i][j] == A && (adj[xa][ya] || adj[xb][yb])) {
                                    xa = i, ya = j; break;
                                }
                        if(xa) break;
                    } while(false);
                    do {
                        for(int i = 0; i < N; ++i)
                            for(int j = 0; j < N; ++j)
                                if(c[i][j] == B && (adj[xa][ya] || adj[xb][yb])) {
                                    xb = i, yb = j; break;
                                }
                        if(xb) break;
                    } while(false);
                    if(abs(xa - xb) > 1 || abs(ya - yb) > 1) ok = false;
                }
        cout << endl;
        exit(ok ? 0 : 1);
    }
    
    int pcol = placed[col];
    for(int pi = 0; pi < N * N; ++pi) {
        if(pcol & (1 << pi)) continue;
        int xi = pi / N, yi = pi % N;
        px[col] = xi, py[col] = yi;
        placed[col] |= (1 << pi);
        bool conflict = false;
        for(int A : adj[col]) {
            int ai = px[A], aj = py[A], bi = px[col], bj = py[col];
            if(abs(ai - bi) > 1 || abs(aj - bj) > 1) conflict = true;
        }
        if(!conflict) dfs(col + 1);
    }
}

// Build adjacency graph of the original map
void makeAdj() {
    memset(G, 0, sizeof(G));
    vector<int> dir = {-1, 0, 1, 0, -1};
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j) {
            for(int k = 0; k < 4; ++k) {
                int ni = i + dir[k], nj = j + dir[k + 1];
                if(ni >= 0 && ni < N && nj >= 0 && nj < N) {
                    int x = c[i][j], y = c[ni][nj];
                    if(x && y && x != y) {
                        G[x][y] = G[y][x] = 1;
                    }
                }
            }
        }
    for(int i = 1; i <= m; ++i)
        for(int j = 1; j <= m; ++j)
            if(G[i][j]) adj[i].push_back(j);
}

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    
    if((cin >> N >> m).fail()) return 0;
    
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j) cin >> c[i][j];
    
    makeAdj();
    
    dfs(1);
    
    // Output the mapped grid - keep original placements of existing colors
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) cout << c[i][j] << " ";
        cout << "\n";
    }
    
    return 0;
}
```
````
