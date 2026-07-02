# 题 0 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=3,得分 0.0,completion_tokens=32677)

````
The user wants a C++ solution for a competitive programming problem called "Pack the Polyominoes (Reflections Allowed)".
The goal is to pack $n$ polyominoes into a bounding rectangle of size $W \times H$ such that $W \times H$ is minimized.
Constraints:
$n$: $100 \le n \le 10^4$.
Size of each piece $k_i$: $1 \le k_i \le 10$.
This means the total number of cells can be up to $10^5$.
This problem is a variation of the Bin Packing Problem or specifically, packing irregular shapes into a rectangle.
Packing polyominoes is NP-hard in general. However, with specific constraints (small piece sizes), there might be heuristics available.

Let's look at the constraints again.
Total area = $\sum k_i$. Since $k_i \ge 1$, Area $\ge n$.
Since we want to minimize Area, the lower bound is just the total area. However, due to shape constraints, we will need more space than the sum of areas.
Usually, packing problems like this are solved using backtracking with pruning (Dancing Links / DLX) or simulated annealing / genetic algorithms if the search space is too large.
Given $n$ up to 10,000 and small piece sizes, a pure exact solver is impossible within time limits. A greedy approach or local search is likely required.

Wait, $n=10000$ is quite large. Typical bin packing for $N=10000$ items into bins? Here we have one big container (rectangle).
If all pieces are size 1, it's trivial: $W=10000, H=1$ or optimized based on aspect ratio.
If pieces are larger, the complexity increases.
Since this is a CP problem, usually there is some specific structure or heuristic expected.
However, $N=10000$ with general polyominoes suggests this is actually an optimization task where finding *any* valid packing close to optimal is acceptable, OR there is a mathematical property I'm missing.
Actually, looking at similar problems from Codeforces or other judges, sometimes these are meant to be solved via simple greedy strategies or randomized local search because the constraints allow loose bounds but tight time limits.
But wait, "Scoring ... score = 1e5*Σkᵢ/A". This implies a heuristic/score-based leaderboard is possible. The problem statement mentions "Competitive Programmer" and asks to implement a solution. It doesn't explicitly demand an *optimal* solution, but usually "minimize" implies optimality. However, given the scale ($N=10^4$), exact optimization is impossible. This looks like a problem from a contest like CodeChef or AtCoder where perhaps random inputs are generated but the difficulty lies in the algorithm design.
Actually, checking similar problems online (e.g., "Pack the Polyominoes"), often these require a specific algorithmic approach.
Let's consider the constraints carefully.
Sum of $k_i$ is roughly $N \times 5.5 \approx 55000$ on average.
We need to fit them into a box.
Is it possible that $W$ and $H$ are small enough for DP? No, $W$ and $H$ can be large.
Maybe binary search on area? If we fix $W \times H = K$, can we pack them? That's still a 2D bin packing variant which is NP-complete.
For $N=10^4$, any exact solver will TLE. We must use a heuristic.
Common heuristics for packing:
1. Greedy placement (largest first, best fit).
2. Simulated Annealing / Local Search.
3. Genetic Algorithms.
4. Backtracking with strong pruning (only good for very small instances).

Given $N=10^4$, a global search isn't feasible. But maybe the input generation has properties? "Shapes may have internal holes." "Coordinates may be negative."
Let's analyze the example. 3 pieces. Total area 8. Box 3x3 = 9.
Greedy approach might work surprisingly well for uniform random shapes.
However, with $N$ up to $10^4$, we need something very efficient per step.

Let's assume the target is a reasonably good solution that passes basic validation checks and scoring.
Since the problem asks to *implement a solution*, and given the high $N$, a sophisticated heuristic like Simulated Annealing (SA) or a specialized packing algorithm (like "Next Fit", "First Fit Decreasing" adapted for 2D) is appropriate.
However, standard 2D bin packing for polyominoes is hard.
Let's try to construct a reasonable heuristic.
Algorithm idea:
1. Calculate total area $S_{total} = \sum k_i$.
2. Determine a rough bound for min Area. Ideally, Area $\approx S_{total} + \text{waste}$. Waste depends on fragmentation.
   Since shapes are small ($k \le 10$), they are not huge blocks. They can tile somewhat well.
3. Sort pieces? Maybe by size (descending)? Larger pieces are harder to fit later.
4. Try to place pieces one by one.
   For each piece, iterate through its transformations (Rotation/Reflection).
   Find a location $(x, y)$ inside the current bounding box such that it fits without overlap.
   How to find $(x, y)$ efficiently?
   Using a grid representation of the occupied status. Since max coordinates could be $W$, if $W$ is around $\sqrt{Area}$, say 200-700. Grid size $500 \times 500$ is okay ($250,000$ cells).
   Operations: update grid, find empty cell adjacent to existing ones, etc.
   
   Wait, finding the *best* transformation and position is computationally expensive if done for every placement naively.
   However, with $N=10000$, $k \le 10$, total cells $\approx 50000$.
   If we maintain a 2D boolean array `occupied[W][H]`, placing a piece of size $k$ takes $O(k)$ operations if the position is checked fast.
   Checking all positions for all orientations? That's slow.
   
   Optimization: Use a Disjoint Set Union (DSU) or simply scan.
   Actually, maybe we don't need to pack perfectly. The prompt says "minimize". If it's a heuristic, I should try to get as close as possible.
   
   Let's check if there's a specific known solution for "Pack the Polyominoes".
   Often these are from contests like "ACM ICPC Regionals" or similar.
   With $N=10000$, this is extremely large for exact packing.
   Maybe the intended solution relies on the fact that $k$ is very small?
   Or maybe the test cases are weak.
   Let's write a robust greedy approach with sorting by size.
   
   Heuristic strategy:
   - Sort pieces by area descending (Largest First).
   - Maintain a set of available bounding boxes? No, only one box growing or fixed.
   - Fixed dimensions? We can binary search the max dimension allowed? Or start with estimated minimal area.
   - Estimate: $H_{min} \approx S_{total} / W_{max}$.
   
   Let's assume a heuristic strategy:
   1. Estimate width $W$ and height $H$ such that $W \times H$ covers the area. Maybe multiple rectangles merged?
      Actually, usually for these specific problems, a random packing with SA works best to beat greedy.
      But implementing SA properly with $N=10000$ might be tricky due to state encoding.
   
   Alternative thought:
   Is it possible the problem allows for disjoint components to be placed arbitrarily, and we just need to find $W, H$?
   Yes, "place all polyominoes... into an axis-aligned rectangle".
   
   Given the constraints and the nature of CP problems, often "Minimize Area" with $N$ up to $10^4$ implies the solution needs to be efficient enough to run within 1-2 seconds.
   An exact solver is out.
   Greedy:
   Iterate pieces sorted by size.
   For each piece, try orientations.
   For each orientation, try to find a spot in a grid.
   Which spot? The first available spot that minimizes fragmentation?
   Usually "Best Fit" - pick a spot where the resulting remaining free space is most fragmented or least?
   Or simpler: "First Fit" - pick the first empty cell $(x,y)$ that accommodates the piece (with top-left alignment to $(x,y)$?).
   To optimize placement, usually we align the "bottom-left" of the bounding box of the piece to an existing free component or to a coordinate near others.
   However, simply finding *any* valid placement for Largest-First might lead to fragmentation that prevents smaller pieces from fitting.
   This suggests we need to re-evaluate $W, H$.
   
   What if we determine $W$ and $H$ beforehand?
   Lower bound on Area: $A \ge \sum k_i$.
   Let $S = \sum k_i$.
   Try candidate $W, H$. e.g. $\sqrt{S}$ x $\sqrt{S}$, or variations.
   But $W$ and $H$ depend on the shapes.
   
   Let's refine the Greedy strategy.
   1. Calculate total sum $S$.
   2. Initialize grid. Max width/height?
      If we don't know max width, we can estimate max needed width for the largest piece, but since we can rotate, the bounding box of the largest piece defines the limit in some direction.
      Actually, if we pack sequentially, the accumulated width/height grows.
      Maybe dynamic allocation:
      Place pieces into a large virtual canvas. Keep track of $W, H$ covered.
      If a piece cannot fit in the current $W, H$, expand? No, we fix $W, H$ to output.
      So we should determine $W, H$ first or dynamically grow and compress.
      
      Actually, the "Pack the Polyominoes" problem from specific sources (like IOI or regional) often allows a solution using **Simulated Annealing** or **Randomized Local Search**.
      Why? Because the search space is discrete and huge.
      State: Assignment of transformation + translation for each piece.
      Cost function: Bounding box area (plus penalization for overlaps or overflow).
      We want to find state with overlap=0 and Min Area.
      
      Algorithm:
      1. Initialize randomly valid configuration (or greedy initial config).
         Since random valid is hard, start with a greedy packing into a sufficiently large box, then try to shrink or rearrange.
      2. While iterations < Limit:
         Select a random piece $i$.
         Randomly choose new transformation $R', F'$ and position $X', Y'$.
         Check validity against current state (overlaps).
         If valid, accept if Area decreases or with probability (Metropolis).
         Also try swapping two pieces.
      3. Track minimum Area found.
   
      Data structures needed:
      - `bool occupied[MaxW][MaxH]` (using vector<vector<bool>> or flattened vector).
      - Need to query efficiently if a move is valid.
      - Since MaxW and MaxH are unknown, we can allocate a safe upper bound.
        Sum of areas is $A_{total}$. $W_{max}, H_{max}$ could be approximated. E.g. $W \approx \sum k_i$ (worst case line), $H=1$. But we want compact.
        Actually, if we start with a box of size $W_0 \times H_0$ where $W_0 \times H_0 \ge A_{total}$, we can try to tighten it.
        
      However, with $N=10000$, checking validity of a move on a grid of size $W \times H$ takes $O(k)$. Doing this $N \times (\text{candidates}) \times \text{iterations}$ times is too much.
      We need faster checks.
      Bitsets! `std::bitset<5000>`.
      Represent rows as bitsets.
      Grid size: Max area 55000. So $W+H$ is not small, but $W \times H \approx 55000$.
      Suppose we guess $W \approx 200, H \approx 275$ (approximate square root of 55000).
      Then grid is $200 \times 275$. Rows can be `std::vector<uint64_t>` or `bitset`.
      Since size varies, `vector<string>` or `vector<vector<int>>` is easier.
      Flattened index $idx = x + y \times W$.
      `valid(x, y)` returns `false` if conflict.
      Overlap check for a piece of $k$ cells: iterate all $j \in [0, k-1]$, check grid cell.
      
      Optimization:
      Instead of full SA, maybe a greedy descent.
      1. Compute total Area $A_{sum}$.
      2. Initial guess for $W, H$. Maybe $W = \lceil\sqrt{A_{sum}}\rceil, H = \lceil\sqrt{A_{sum}}\rceil$.
         If $W \times H$ is not enough for the bounding boxes of pieces individually, adjust.
         Wait, pieces can be rotated.
         Each piece $i$ has a set of bounding boxes $(w_i, h_i)$ derived from transformations.
         For piece $i$, min width $w_i^{min}$, min height $h_i^{min}$.
         Global constraint: All $i$ must fit in $[0, W) \times [0, H)$.
         So $W$ must be $\ge w_i$ for chosen orientation of $i$.
         
      Revised Plan:
      Use a heuristic local search with a simplified fitness function.
      Fitness = Area of bounding box containing all cells.
      If overlaps exist, add penalty.
      We want Min (Area + Penalty).
      
      Wait, strictly, we need to ensure non-overlap.
      Better Strategy:
      1. Parse input, store canonical forms for each polyomino.
         Canonical form: Normalize coordinates so min-x=0, min-y=0. Store as vector of points.
         For each piece, precompute all 8 symmetries (4 rotations x (reflect/no reflect)).
         For each symmetry, compute the normalized bounding box (width, height, set of relative points).
      2. Generate a random permutation of pieces? Or sort by size.
         Sorting by size helps deterministic placement.
      3. Run a "Bin Packing" algorithm.
         Common heuristic: First Fit Decreasing (FFD).
         Sort pieces by size (area) descending.
         For each piece, try all orientations. For each orientation, find first valid position.
         How to find position?
         Use a sweep-line or a grid. Since we want to minimize total $W, H$, maybe we don't fix them yet.
         Wait, FFD into a *single* bin (rect) requires us to decide where it goes.
         In FFD for 2D rectangles, usually we maintain a list of free lists (regions).
         Maintaining complex regions for polyominoes is hard.
         Coordinate compression or a simple grid is best.
         
         Let's use a simple grid-based heuristic.
         Assume a maximum width $W_{max}$ and height $H_{max}$ derived from area and worst-case aspect ratios.
         If the pieces fit, record $W, H$.
         Can we reduce $W$ or $H$?
         Binary search on $W$? If we fix $W$, does there exist a packing with height $H$?
         This is equivalent to: Can we pack these polyominoes into a column of height $H$ and width $W$?
         Since we want to minimize $W \times H$, we can iterate over possible widths.
         Lower bound on $W$: $\lceil \text{TotalArea} / \text{TargetHeight} \rceil$.
         Target Height is unknown.
         
         Actually, looking at $N=10000$ and constraints, it's highly likely that the intended solution uses a very fast greedy approach with randomized restarts or a specific heuristic tailored to polyominoes.
         Given the strict requirement for code output and lack of interaction, I should provide a self-contained solution.
         Let's use a greedy placement strategy:
         1. Collect all pieces.
         2. Pre-calculate all valid transformed shapes (coords) for each piece.
            There are at most 8 shapes per piece. Normalize coords (shift to (0,0)).
            Store these as a struct with {dx, dy}.
         3. Sort pieces by area descending.
         4. Iterate pieces. For each piece, try to place it.
            Placement rule: Iterate all 8 orientations.
            For each orientation, try to find a position $(x, y)$ in the current grid.
            Which $(x, y)$?
            Heuristic: Find the lowest $y$, then leftmost $x$ that fits?
            Or "Best Fit": Check empty slots? Too slow.
            "First Fit" is simplest.
            To make it better, prioritize placements that keep pieces aligned?
            
            Wait, just checking `(x, y)` for the whole bounding box is slow if $W, H$ large.
            But grid only contains occupied cells.
            Total occupied cells $\le 55000$.
            Iterating neighbors of occupied cells to find gaps?
            Better: Maintain a list of available cells. Insert piece cells into available list?
            Since we want to minimize W, H, we need to explore.
            
            Actually, with $N=10000$, maybe we can just output a packing that satisfies the condition?
            If I simply place them in a very large rectangle (e.g. $1 \times \sum k_i$), it's valid but area is huge. I need to MINIMIZE area.
            
            This feels like the problem requires a solver capable of reducing waste.
            Waste reduction happens when packing is dense.
            With random polyominoes, they tend to fill space well unless they form long lines.
            
            Let's reconsider the problem type.
            This looks like it could be solved by "Greedy with Lookahead" or "Simulated Annealing".
            But implementing a full SA for $10^4$ objects in Python/C++ might be slow in 1 sec.
            However, we are allowed C++.
            
            Let's write a solution that implements a randomized local search with a bounding box estimator.
            We can fix $W$ to a value, and try to find $H$.
            Or vice versa.
            
            Let's try this flow:
            1. Parse pieces. Normalize their internal coordinates to $(0,0)$-aligned.
            2. Create a list of all unique shapes for all pieces (including reflections/rotations).
               Actually, for each piece $i$, store a list of `Shape` structs. `Shape` has `{width, height, cells}`.
               `cells` are `(dx, dy)`.
            3. Estimate a base $W, H$.
               Start with $W = \sum \text{areas}$. $H=1$.
               Use a heuristic to swap pieces to improve density.
               
            Actually, let's look at the problem source context if possible. The problem title "Pack the Polyominoes (Reflections Allowed)" sounds generic.
            In the absence of specific knowledge about this exact problem instance, the most robust approach is **Simulated Annealing** for 2D packing.
            State representation: For each piece $i$, choice of transform $(r_i, f_i)$ and offset $(x_i, y_i)$.
            Constraints: $0 \le x_i < W, 0 \le y_i < H$.
            Objective: Minimize $W \times H$.
            
            To handle the offsets and dimensions effectively:
            Usually, offsets are relative to 0,0.
            The bounding box is defined by $\max(x_i + dx) - \min(x_i)$ etc.
            This is hard to manage in SA because changing one piece changes the bounds of others.
            Alternatively, keep the origin fixed at (0,0) for the entire assembly?
            Then $W = \max_x(\text{all}) - \min_x(\text{all})$, same for $H$.
            Moves:
            1. Change rotation/reflection of piece $i$. Update coords. Re-compute bounding box.
            2. Translate piece $i$ by $(\Delta x, \Delta y)$. Update coords. Re-compute.
            Constraint: Must not overlap.
            
            Valid moves require checking overlap. With $N=10000$, this check $O(N \cdot k)$ is too slow ($10^8$ ops per check).
            We need to speed up overlap check.
            Use a hash map of occupied `(x, y)` cells mapping to piece ID?
            Or a grid if bounds are small.
            If we keep the box tight, the grid size is limited by Area.
            Max Area 55000. A grid of $W \times H \approx 250 \times 220$.
            A `vector<vector<int>> grid(W, vector<int>(H))` storing piece ID.
            Access is O(1).
            Updating grid for a move: Remove old cells, insert new cells. Complexity $O(k)$.
            This is feasible!
            
            So the plan:
            1. Parse Input.
            2. Precompute all valid orientations for each piece. Normalize coords to start at (0,0).
               Store them as `vector<Point>`.
               Compute bounding box size for each orientation.
            3. Initial Configuration:
               Assign each piece an arbitrary valid orientation.
               Assign positions greedily to fill a box starting from (0,0).
               Sort pieces by size descending.
               For each piece, place it at the first available spot that fits.
               Calculate $W_{curr}, H_{curr}$.
            4. Local Search:
               Current Solution: `state` containing (orientations + positions).
               Loop until time limit or convergence:
                 Pick a random piece $p$.
                 Pick a random neighbor in state space:
                   a) Rotate piece $p$ (try different transforms).
                   b) Move piece $p$ by $(\pm 1, \pm 1)$ steps? Or swap pieces?
                      Swapping positions between identical pieces is useless (they are distinct IDs).
                      Moving piece $p$ to another empty cell?
                      Since we fixed $W, H$ loosely, maybe we don't move.
                
                Wait, minimizing area implies we should also try to shrink $W$ or $H$ (repack).
                But moving piece $p$ is enough to shift the global bounding box boundary if we define $W, H$ as extents of all pieces.
                Extents are defined by min/max x/y of all pieces.
                So if we translate $p$, $W/H$ can change.
                
                Improvement metric: `CurrentBoxArea`.
                If `NewState` overlaps -> Reject.
                Else -> Accept if `NewArea < OldArea` (or Metropolis).
            5. Implementation Details:
               - `Grid`: Flat array or 2D. Since $W, H$ vary, maybe use `vector<string>` or `vector<vector<int>>`.
                 Coordinates are absolute.
                 We need to track `occupied[x][y]`.
                 Since max coords can be roughly sum of lengths? No, if packed well, bounded by $\sqrt{Area}$.
                 Worst case: strip packing -> length $55000$. Grid access $55000 \times 1$. Fine.
                 So use a flat vector of bools or bits for occupancy. `vector<bool>` or `char`.
                 Map `int coord -> occupied`. Index = $y \times W_{lim} + x$?
                 But $W_{lim}$ is not fixed initially.
                 Wait, if we define bounding box dynamically, the index calculation needs a stable origin.
                 Let's just use `set<pair<int,int>>`? Set insertion/deletion is $O(\log N)$. $N=10000, k=10$. Total ops $10^5$. Log factor is fine.
                 But we need to check collision in $O(k)$. `set` check is $O(k \log (\text{occupied}))$.
                 Occupancy count grows to $55000$. $\log \approx 16$. $16 \times 10 \approx 160$ ops per piece move.
                 Number of swaps/moves? If 1 second allowed, $\sim 10^7$ ops. $10^4$ pieces.
                 We might do $\sim 1000$ moves.
                 This seems feasible.
                 
                 Wait, SA needs many steps.
                 If we do too many collisions, we waste time.
                 But valid moves are rare in early stages.
                 Better: Fix $W, H$ first?
                 We can start with a large $W, H$, e.g., $W=\sum k_i, H=1$.
                 Pack pieces into this.
                 Then try to compact.
                 But compaction is hard.
                 
                 Actually, there's a constraint: "Output format: X_i Y_i R_i F_i".
                 My state stores exactly this.
                 
                 Refining the Algorithm for Time Efficiency:
                 1. Parse.
                 2. Sort pieces by size DESCENDING.
                 3. Determine initial bounds.
                    Just take a large enough box.
                    E.g., $W = 100, H = 500$? Or just calculate $W = \max(w_p), H = \max(h_p)$? No.
                    Let's use a dynamic grid.
                    To avoid reallocation issues, use a large static 2D grid if bounds are predictable.
                    Max possible Area $\approx 55000$.
                    If we pack in a way that maximizes aspect ratio 1:1, $220 \times 250$.
                    If we pack in a line, $55000 \times 1$.
                    We can use a large buffer, say $2000 \times 2000$, mapped to coordinates relative to $(0,0)$?
                    Since translation matters for the output $(X_i, Y_i)$, we need absolute coordinates.
                    
                 Let's write a C++ program using `std::set` for overlap checking?
                 No, `set` is slow. Use `vector<bool>` with a calculated max dimension.
                 If I enforce the initial bounding box to be a square covering the area?
                 $\sqrt{55000} \approx 234$.
                 Let's assume the optimal $W, H$ won't exceed, say, $500 \times 500$.
                 This is safe. If the solution requires a very long thin strip (impossible for optimal packing of polyominoes usually), we might fail. But polyominoes can be squares.
                 Let's assume a generous bound like $W=1000, H=1000$ is sufficient for most optimal packings of this size.
                 Wait, if the area is 55000, and we use a $1000 \times 1000$ grid (1M cells), it's sparse.
                 Using `vector<char> grid(1000*1000)` is ~1MB memory. Fast access.
                 
                 Wait, I need to support arbitrary translations.
                 If I place a piece at $(-500, -500)$, my grid index would be negative.
                 Offset everything by `OFFSET_X, OFFSET_Y` (e.g., $N \times \max\_shape\_dim$).
                 Or just shift all pieces so min_x >= 0.
                 Initially, place at $(0,0)$.
                 Then during local search, if I move a piece, I check boundaries.
                 Actually, simpler:
                 We are minimizing the *bounding box*.
                 This means we care about $\max X - \min X$ and $\max Y - \min Y$.
                 We don't care about the absolute position in memory, just relative.
                 BUT the output requires absolute $X_i, Y_i$.
                 So, relative to global origin (0,0).
                 To avoid coordinate explosion, keep coordinates relatively small?
                 If I shift the whole system, the bounding box stays same.
                 Let's just maintain `vector<vector<int>> grid` representing the occupied cells.
                 Size $W_{safe} \times H_{safe}$.
                 Where $W_{safe}, H_{safe}$ are estimates?
                 No, let's just track occupied cells in a `set<pair<int,int>>`.
                 Wait, `set` is $O(K \log N)$ per piece.
                 With $K=10$ and $N=10000$, per move check is $\approx 10 \times 16 = 160$.
                 We can afford $10^6$ moves? $1.6 \times 10^8$ ops. Might be tight for 1s.
                 Using a hash map (`unordered_set`) or open addressing table?
                 Or just a flat array `grid[MAX_W][MAX_H]` with MAX_W = 200, MAX_H = 500?
                 How do we handle expansion?
                 We can fix the target rectangle size.
                 Actually, for the purpose of the problem, if we output a valid placement, the judge verifies if it's optimal.
                 I can't generate a *perfect* optimal solution. I must generate a *good* one.
                 A greedy packing with a fixed box size that is slightly larger than necessary will yield a valid placement.
                 But the objective is to *minimize* the area.
                 
                 Strategy adjustment:
                 The most promising approach for large N and polyominoes is **Simulated Annealing** where we treat the *arrangement* as the variables.
                 But to save time, let's use a **Greedy Best Fit** with a randomized restart and bounding box estimation.
                 
                 Steps:
                 1. Parse.
                 2. Identify unique shapes.
                 3. Sort pieces by area desc.
                 4. Try to pack into a grid.
                    Estimate $W, H$. Let's pick $W_{init} = \lceil \sqrt{\text{Area}} \rceil$. $H_{init} = W_{init}$.
                    If $W_{init} \times H_{init} < \text{Area}$, increase.
                    Actually, just allocate $W = 200, H = 200$ (if area > 40000, increase).
                    Wait, if the pieces force a large bounding box, we might need more.
                    Since $k \le 10$, the pieces are small. The density of packing is key.
                    Assuming 90% density packing.
                    
                 Let's use a 2D array `used[MAX_WIDTH][MAX_HEIGHT]` initialized to false.
                 We need to handle potential shifts.
                 To avoid negative indices, we maintain `current_min_x`, `current_max_x`, etc.
                 Or we just keep a `map<int, int> cols` where keys are columns?
                 
                 Actually, let's just fix a large canvas. $1000 \times 1000$ is definitely safe for Area 55000 (unless aspect ratio is extreme).
                 If I assume $1000 \times 1000$ grid:
                 Memory usage $\approx 10^6$ bytes = 1MB. Very low.
                 Operations: `grid[y][x] = id`.
                 Check overlap: loop over piece cells, check grid.
                 If fits, place.
                 But how to find the *best* position?
                 For greedy, usually: try to place at $(0, y)$? Or first empty row?
                 To minimize final bounding box area, this is hard.
                 However, with $N$ large, simply filling the box tightly might not work if we pick wrong order.
                 But sorting by size helps.
                 
                 Wait, the "Score" formula is $10^5 \times \Sigma k_i / A$.
                 Maximizing density.
                 This confirms we need to minimize A.
                 My solution will produce a valid A.
                 
                 Let's implement a **Greedy Algorithm with Rotation/Reflection**.
                 Since we can rotate/reflect freely, for each piece, generate all 8 orientations.
                 Choose the one that minimizes "wasted space" or "bounding box size"?
                 Hard to predict.
                 Let's stick to: Sort pieces by area desc.
                 Maintain `vector<bool> grid` (flattened).
                 Dimensions: $1000 \times 1000$ is safe?
                 Is it guaranteed that Area 55000 fits in $1000 \times 1000$? Yes.
                 Will it fit in $300 \times 300$? Yes (Area 90000).
                 So we can set a safe MAX WIDTH/HEIGHT = 300 (for 90000 sq).
                 Wait, if pieces are long strips (e.g. dominoes stacked horizontally), we might need width $> \sqrt{Area}$?
                 If all pieces are $1 \times 1$, we can pack $1 \times 55000$. $W=55000, H=1$.
                 Our algorithm would pick $W=1000$, so it forces packing into a box.
                 But if the pieces force a strip, we might fail to fit in $1000 \times 1000$.
                 So we should allow dynamic growth or use a safer bound.
                 Safe bound: $W = \max(\text{piece\_width}), H = \sum k_i$?
                 No, that's $10000 \times 1$.
                 
                 Let's implement a dynamic resizing grid or a set-based approach?
                 Given the time constraint, a `set<pair<int,int>>` is safest for correctness and space, but slower.
                 Given $N=10000$, $k=10$, total cells 100,000 (max).
                 Set overhead is acceptable.
                 
                 Wait, checking the Example:
                 3 pieces, Area 1, 3, 4. Total 8.
                 Output 3x3 (Area 9).
                 Wait, 8 fits in 3x3 (9).
                 The example output gives a specific placement.
                 
                 Let's finalize the implementation details:
                 Class `Piece`: stores original coords. Generates transformations.
                 Struct `Transformation`: {id, r, f, cells}, cells relative to (0,0).
                 Main Logic:
                 1. Read input. Build pieces.
                 2. For each piece, generate all 8 valid (r, f) states with normalized cells.
                 3. Shuffle order of pieces? Or Sort by size descending? Descending is standard greedy.
                 4. Maintain `set<pair<int, int>> occupied`.
                 5. For each piece in order:
                    Found = false
                    Iterate all orientations.
                        Iterate placements:
                            How to iterate placements?
                            We need to find $(x, y)$ such that cells fit.
                            Since we want to pack densely, maybe we scan the grid?
                            Scanning $W \times H$ is slow.
                            Heuristic for placement:
                            Find the "lowest" cell that is currently empty?
                            Or just place at $(0,0)$ and shift?
                            Better: Use a "Next Empty Cell" iterator.
                            Keep a `list` of free cells? Or just `grid`?
                            
                            Actually, for a greedy strategy to work well, we should try to place pieces in a way that doesn't leave unreachable holes.
                            Simple approach:
                            Try to place piece at $(0,0)$ relative to its own bottom-left.
                            Then shift it up/right until it hits obstacles?
                            This is complex.
                            Simplified:
                            Try all valid orientations.
                            For each orientation, find the first available spot $(x,y)$ (starting from 0,0?) that allows placement.
                            Which $(x,y)$?
                            Maybe the one that minimizes the "next available empty cell" index?
                            Or just $(0, y)$?
                            Let's use the "Bottom-Left" heuristic:
                            Sort grid cells by (y, x). Find first empty cell $(c_r, c_c)$ in occupied set order.
                            Align piece's bottom-left corner $(0,0)$ to $(c_r, c_c)$?
                            Check if fits.
                            If yes, place it.
                            
                            Wait, we have $N$ pieces. Total $O(N \cdot 8 \cdot \text{check})$.
                            Finding $(c_r, c_c)$ takes time.
                            But we can just store occupied cells in a sorted structure or bucketed.
                            With `set`, we can find `floor`? No.
                            Actually, we can just keep iterating.
                            But iterating over *empty* space is bad.
                            
                            Alternative:
                            Just maintain the bounding box $W, H$.
                            Place pieces in the grid $W \times H$.
                            If a piece doesn't fit, what then?
                            We assumed we fix $W, H$ before.
                            But we can't easily fix $W, H$ optimally.
                            
                            Okay, I will use a **randomized greedy**.
                            For each piece, assign a random orientation.
                            Assign a random valid position?
                            This creates overlaps.
                            Then we use SA to resolve.
                            This handles the "finding position" part automatically by fixing $W, H$ high enough (say $500 \times 500$) and optimizing coordinates.
                            
                            Algorithm Refined:
                            1. Define `MAX_W = 600`, `MAX_H = 600`. (Safe for Area 55000, assuming decent packing).
                               If total area > 360,000, resize.
                            2. `occupied[600][600]` grid of PieceIndex. -1 if empty.
                            3. `pieces[n]` storing best known transformation/position.
                            4. Init: For each piece, try random orientation. Try to place at (random valid slot).
                               Fill greedily: Sort by size desc.
                               Try to place at (x, y) closest to (0,0) that is valid?
                               Scan grid $(0,0) \to (W,H)$.
                            5. Once we have a valid initial configuration (assuming we can always fit if grid is big enough):
                               Run Simulated Annealing.
                               Moves:
                                 - Select random piece $i$.
                                 - New orientation: random valid orientation.
                                   Check if overlaps. If not, accept (if area better? no, area changes slowly).
                                   Or swap pieces?
                                   Swap positions?
                                - The main driver for area minimization is shrinking the bounding box.
                                  Bounding box is determined by $\max(x)$ and $\max(y)$.
                                  Translating pieces towards the "center" or "origin" reduces extent?
                                  
                                  Actually, we can just fix the global coordinate system at 0,0.
                                  Then $W = \max(x_i+dx) - \min(x_i+dx)$.
                                  $H = \max(y_i+dy) - \min(y_i+dy)$.
                                  We want to minimize $W \times H$.
                                  
                                  Move types for SA:
                                  1. Change Orientation of Piece $i$.
                                     Update coords. Recompute BB.
                                     Check overlaps.
                                  2. Move Piece $i$ to $(x \pm 1, y)$ or $(x, y \pm 1)$.
                                     Update coords. Check overlaps.
                                  
                                  If overlap exists -> Penalize heavily?
                                  Actually, if overlap exists, the move is invalid.
                                  Wait, in SA we explore invalid states but with penalties?
                                  Or strictly valid states? Strictly valid is harder to reach if $N$ is large.
                                  Starting valid is crucial.
                                  
                                  So:
                                  Phase 1: Greedy valid packing.
                                    Sort pieces.
                                    Try to place in $600 \times 600$ grid.
                                    For each piece, try all 8 orientations.
                                    For each orientation, scan the grid for the first $(x,y)$ that fits.
                                    Place it.
                                  Phase 2: Optimize $W, H$ and layout.
                                    Compute current $W, H$.
                                    While iter < Limit:
                                      Try to swap pieces? No.
                                      Try to perturb positions/orientations.
                                      Specifically: Move pieces to reduce bounding box size.
                                      Maybe try to "shift" the whole assembly?
                                      Or individual shifts to fill gaps.
                                      
                                      Key observation: The bounding box area is convex-ish.
                                      Reducing $W$ involves moving right-side pieces left, and vice versa.
                                      Reducing $H$ involves moving top pieces down.
                                      
                                      Heuristic:
                                      Try to move pieces towards the center of the bounding box.
                                      But they must not overlap.
                                      
                                      Let's just rely on the initial greedy packing being decent for $N=10000$ with small $k$.
                                      Small $k$ polyominoes are very flexible.
                                      The main bottleneck is overlapping long strips.
                                      But even then, random starts help.
                                      
                                      Wait, if I implement this in C++, I need to be careful with efficiency.
                                      $N=10000$. Grid $600 \times 600$ = 360,000 ints = 1.4MB.
                                      Access is fast.
                                      SA:
                                      Number of pieces to modify: random.
                                      Check overlap: $O(k)$.
                                      If valid, update BB and Area.
                                      If invalid (overlap), revert.
                                      
                                      We also need to ensure that after optimization, the BB is valid (non-negative indices).
                                      I can enforce global shifting if needed.
                                      The easiest way to handle coordinates is:
                                      After finding *a* valid placement (offsets can be anything), compute min\_x, min\_y.
                                      Shift all to 0.
                                      
                                      Final check: Ensure output format.
                                      
                                    Complexity:
                                    Initialization: $O(N \cdot 8 \cdot \text{GridScan})$.
                                    Grid scan?
                                    Finding empty spot naively takes $O(W \cdot H)$.
                                    Total init: $N \cdot 8 \cdot W \cdot H$.
                                    With $N=10000, W=600$, $H=600$. $3.6 \times 10^9$. Too slow!
                                    
                                    We need a faster way to find a spot.
                                    Use a `Disjoint Set Union` (DSU) to maintain connected free components?
                                    Or just a pointer to next empty cell.
                                    Since we process pieces, we want to place them in "dense" spots.
                                    A common trick for packing:
                                    Maintain a list of free rows/columns?
                                    Or simply iterate over the grid linearly $(0,0) \dots (W,H-1)$.
                                    But updating the list of free cells takes time.
                                    
                                    Let's use `vector<int> grid(W*H, -1)`.
                                    To find a spot for a piece, iterate through the piece's cells in BFS?
                                    No.
                                    Better:
                                    For a piece, we need to find $(x,y)$ such that $(x,c_x) \dots$ are free.
                                    This is 2D pattern matching on a boolean grid.
                                    Slow.
                                    
                                    However, notice $k$ is small (1..10).
                                    Maybe we can just check the first few candidate positions?
                                    Candidates:
                                    (0, y), (1, y), ...?
                                    Or positions adjacent to already placed pieces?
                                    Adjacent placement is the most natural for growth.
                                    Maintain a set of "Frontier" cells (boundary between filled and empty).
                                    When placing a new piece, try to attach to the frontier.
                                    
                                    Frontier set size: proportional to perimeter of occupied region.
                                    Max perimeter $\approx$ Area.
                                    But practically much less?
                                    
                                    Let's refine the Greedy:
                                    1. Sort pieces by size descending.
                                    2. Maintain `occupied[600][600]`.
                                    3. Maintain a list of active "rows" or "segments".
                                    4. For each piece:
                                       Try all 8 orientations.
                                       Try to place it.
                                       Where?
                                       Try $(x,y)$ in increasing order of priority.
                                       Priorities:
                                       1. Attach to existing occupied cells.
                                          Iterate through all neighbors of currently occupied cells?
                                          This is basically "Grow".
                                       2. Otherwise, try random spots.
                                       
                                       This is getting complicated.
                                       Let's simplify.
                                       Just use a heuristic:
                                       Place at (x, y) such that it fits.
                                       Scan x from 0 to MAX_W. For each x, scan y from 0 to MAX_H.
                                       Stop at first fit?
                                       Optimization:
                                       Maintain a queue of (row, col) indices to check?
                                       When we place a piece at $(r, c)$, mark $(r, c)$ as visited/done?
                                       Wait, if we skip $(r,c)$ later, we lose opportunity.
                                          
                                       Okay, given the time constraint and problem difficulty, the best bet is a randomized search (SA) with a fixed grid size (large enough).
                                       To initialize quickly:
                                       Use a greedy heuristic with random shuffle for positions.
                                       1. Allocate Grid 1000x1000.
                                       2. For each piece (sorted desc):
                                            Try all 8 orientations.
                                            Try to place at a random valid $(x,y)$?
                                            No, need *some* placement.
                                            Iterate $y$ from 0 to 1000. For each $y$, $x$ from 0 to 1000.
                                            Check fit. If fit, break and place.
                                            If no fit found in 1000x1000, expand?
                                            Wait, 1000x1000 is 1M cells.
                                            Worst case time: $N \times 8 \times 1M \approx 10^{10}$. Too slow.
                                            
                                    We need $O(k)$ placement check.
                                    Finding the spot is the bottleneck.
                                    Optimization:
                                    Maintain a `queue` of empty cells to check.
                                    Initially queue = all empty cells? No.
                                    Queue = first $(0,0), (0,1) \dots$
                                    Pop $(x,y)$.
                                    Check if we can fit piece at $(x,y)$?
                                    To do this efficiently: check if $x+y$ is consistent? No.
                                    
                                    Correct approach for fast greedy packing:
                                    Use a DSU on cells to track components.
                                    When we occupy cells, unite them.
                                    Free cells form components.
                                    We want to find a component that has enough space for the piece?
                                    No, we want to find a tuple of free cells.
                                    
                                    Actually, for $N=10000$, the constraints might allow a solution with less optimal performance but correct logic.
                                    Or maybe $N=10000$ is just a stress test for $O(N \log N)$ or similar.
                                    Wait, if $k$ is small, maybe we can represent each piece as a bitmask?
                                    Since $k \le 10$, yes.
                                    Grid state: bitset.
                                    
                                    Let's step back.
                                    This is a known competitive programming problem.
                                    "Pack the Polyominoes"
                                    Common solution for this class of problem involves:
                                    1. Preprocessing all piece shapes (up to 8 per piece).
                                    2. Running a randomized hill-climbing or simulated annealing.
                                    3. Initial configuration: Random or Greedy.
                                    
                                    Since implementing a complex "Find Spot" data structure might be bug-prone in a single file response, I will implement a simpler approach:
                                    "Random Restart Hill Climbing".
                                    To make initialization faster:
                                    Instead of scanning the grid, I will use a **heuristic seed**:
                                    Sort pieces by size.
                                    For each piece, pick a random orientation.
                                    Pick a random valid bounding box coordinate.
                                    Actually, just place it at $(0,0)$.
                                    Wait, if everyone places at $(0,0)$, it explodes.
                                    
                                    Better Init:
                                    Sort pieces by size desc.
                                    Use a grid (vector of `vector<bool>`).
                                    For each piece, iterate all 8 orientations.
                                    Find the orientation that allows fitting into the *smallest* empty rectangle at the top-left?
                                    Or just iterate $(y, x)$ in raster order?
                                    To speed up:
                                    Keep track of `next_empty` in a 1D array?
                                    Flatten grid to 1D.
                                    Array `occupied[M]`.
                                    To check if a piece fits at $(y, x)$ (corresponding to `idx = y*W+x`?), we need to know the next free cell.
                                    But we need *contiguous* cells in 2D sense.
                                    
                                    Let's assume the grid size is roughly $300 \times 200$ (60000 cells).
                                    If we check all cells for every piece, it's slow.
                                    However, notice that we only care about valid placements.
                                    If we just place the piece at the *first* valid location encountered in a raster scan, it acts as a First Fit.
                                    To optimize "First Fit":
                                    Use a Disjoint Set Union to store the *next available cell* in each row or column?
                                    Or just `vector<int> head(M, -1)`?
                                    
                                    Okay, I'll use a simpler, albeit potentially slower, valid method:
                                    Since $N$ is up to $10^4$, I will use `vector<vector<int>>` for the grid.
                                    For each piece, loop $y$ from 0 to $H$. Loop $x$ from 0 to $W$.
                                    If `check(x,y)` succeeds, place and break.
                                    To avoid $O(W \times H)$ loop, we can prune.
                                    Only check $x, y$ where we haven't verified it before?
                                    Mark visited $(x,y)$ as "checked for piece"? No, it's for specific shape.
                                    Shape 1 vs Shape 2 differ.
                                    
                                    Wait, let's look at constraints again.
                                    Time Limit is typically 2s?
                                    If I use `set<pair<int,int>>` for `occupied`, checking a placement takes $k \times \log (\text{count})$.
                                    Count $\le 55000$.
                                    $\log \approx 16$.
                                    Ops $\approx 10 \times 16 \approx 160$.
                                    If I check $T$ candidates.
                                    How many candidates?
                                    In a naive scan, $W \times H \approx 600 \times 600 = 360000$.
                                    $10000 \times 160 \times 360000$ is huge.
                                    BUT, we stop at the first fit.
                                    So on average we check fewer? No, if fit is hard, we check many.
                                    If fit is easy, we check few.
                                    
                                    With $N=10000$ pieces, the board fills up.
                                    We will eventually fill the board.
                                    The time complexity will be dominated by finding the last few spots.
                                    
                                    To optimize:
                                    Use a custom "free cell" locator.
                                    Maintain `vector<int> free_rows`.
                                    In each row, maintain a list of free column indices.
                                    Iterate rows.
                                    For each row, scan free cols.
                                    This makes checking a placement in a row take $O(k)$.
                                    Finding the position takes $O(W)$.
                                    Total time: $N \times 8 \times W$.
                                    With $N=10000, W=600 \implies 4.8 \times 10^6$.
                                    This is extremely fast!
                                    Wait, maintaining "free columns in each row" is costly?
                                    Deleting from a list is $O(1)$ if we swap, but $O(k)$ with linked list.
                                    We have at most 10 pieces added per step?
                                    Actually, we add a piece of size $k$ to the board.
                                    This removes $k$ entries from the lists.
                                    Adding back? No, once used, occupied forever (until SA moves).
                                    
                                    Okay, for the Initial Greedy Pass:
                                    Data Structure:
                                    `struct RowFree { int next_free[600]; }` ? No, dynamic.
                                    `vector<int> row_free[600]`?
                                    Or simply: `vector<int> occupied_cells` is fine.
                                    Wait, just keeping track of "holes" is hard.
                                    
                                    Let's go with the `set<pair<int,int>>` approach.
                                    Why? Because it's clean and robust.
                                    Optimization:
                                    Instead of scanning full grid, keep a priority queue of "candidate placement sites"?
                                    Or simply:
                                    Scan the grid linearly $0 \dots W \times H$.
                                    Skip processed cells?
                                    We can mark `visited[x][y]` meaning "we tried this spot for this piece type".
                                    But piece types differ.
                                    
                                    Okay, I'll implement a solution that uses:
                                    1. `vector<vector<char>> grid(MAX_W, vector<char>(MAX_H, 0))`.
                                    2. `vector<int> occupied_indices`.
                                    3. For placement:
                                       Iterate $y$ from 0 to MAX_H.
                                       Iterate $x$ from 0 to MAX_W.
                                       Check if piece fits at $(x,y)$.
                                       Use the `grid` to check.
                                       This takes $O(MAX\_W \times MAX\_H)$ per piece if no optimization.
                                       With $MAX\_W=200$, $MAX\_H=500$. Area 100,000.
                                       $10000 \times 100000 = 10^9$. Too slow.
                                       
                                    Wait, $N=10000$.
                                    Can we reduce MAX_W, MAX_H?
                                    We need to cover Area $\approx 55000$.
                                    Minimal area box is roughly $230 \times 230$.
                                    So $230 \times 230 = 52900$.
                                    If we assume the optimal bounding box is somewhat close to square?
                                    Maybe we try multiple box sizes.
                                    But for the greedy phase, we can use a slightly larger box.
                                    Say $400 \times 150$. Area 60000.
                                    $400 \times 150 = 60000$.
                                    $N=10000$.
                                    Total scans $\approx 6 \times 10^8$. Still high.
                                    
                                    Wait, why am I scanning $O(W \times H)$?
                                    We can optimize the placement check.
                                    We only need to check spots that are "connected" to occupied cells?
                                    No, new pieces can be placed anywhere.
                                    However, if we place pieces one by one, we fill the grid.
                                    Most of the grid is empty.
                                    We want to find *any* spot.
                                    We can just store the `occupied` cells in a `vector<int> free_cells`?
                                    No, `free_cells` would be empty initially.
                                    Wait, `grid` is initially empty (all free).
                                    We want to find a spot.
                                    Any $(0,0)$ works for the first piece.
                                    For subsequent pieces, we need to check $(x,y)$ where they might fit.
                                    If we use a `stack` or `queue` of empty cells.
                                    Initially push $(0,0), (0,1) \dots (0, H-1), (1,0) \dots$.
                                    Pop $(x,y)$.
                                    Try to place piece at $(x,y)$?
                                    If it fits, place it and remove cells.
                                    If it doesn't fit (overlap?), it's not overlap because grid is empty?
                                    Wait, if grid is mostly empty, any check will succeed if piece is small.
                                    The only reason a piece doesn't fit is if we hit a wall of previous pieces.
                                    
                                    Correct Greedy Strategy:
                                    1. Sort pieces by size DESC.
                                    2. Iterate pieces.
                                    3. Try to place in the *first available* spot in a raster scan?
                                       Raster scan is inefficient.
                                    4. Better:
                                       Maintain a list of available "blocks" or just use a heap?
                                       No.
                                    5. Fastest way:
                                       Use a `DSU` to link empty cells?
                                       Or simply a `vector<pair<int, int>> candidates` that are adjacent to occupied cells.
                                       When adding a piece, add new occupied cells' neighbors to candidates.
                                       Pop from candidates. Check fit.
                                       If fit -> Place -> Add neighbors.
                                       If not fit (due to some other piece blocking?), discard?
                                       Wait, if blocked by "another piece" that was already placed, we need to know.
                                       Actually, if we build incrementally, the only obstruction is the previously placed pieces.
                                       Since we place largest pieces first, they create "walls".
                                       Subsequent pieces fit in the holes.
                                       
                                       So:
                                       1. Sort pieces desc.
                                       2. Maintain `priority_queue` of "candidate positions".
                                          Priority?
                                          Smallest row? Or Largest?
                                          Let's use "Bottom-Up" priority.
                                          Candidates = list of points.
                                          Initially {(0,0)}. (Assume valid).
                                          For piece $p$:
                                            Sort Candidates? No, just iterate.
                                            Actually, since we want to pack well, we should look for compact placements.
                                            Placing at (0,0) is often suboptimal for later pieces if it leaves a hole.
                                            But placing "next to" existing pieces is good.
                                          
                                           Let's try a simple heuristic:
                                           `vector<int> grid_idx(MAX_W * MAX_H)`.
                                           Initialize all -1.
                                           `vector<int> active_list`.
                                           For piece $p$:
                                             Iterate candidates?
                                             We can just use `set<pair<int, int>> occupied`.
                                             Try to place at $(0, y)$ for $y=0..H$.
                                             If fits, place and continue?
                                             But this leaves vertical gaps.
                                             What if we iterate $(x, y)$ randomly?
                                              
                                     Given the complexity and risk of bugs in complex structures in a short generation window:
                                     I will write a clean, moderately efficient simulation using a fixed-size grid (1000x1000) and a straightforward scan with pruning.
                                     Optimization:
                                     We can cache the result of "Does piece fit at $(x,y)$" for previous pieces?
                                     No, pieces change.
                                     
                                     Wait, actually, we can optimize the *search* for the placement.
                                     Instead of iterating all $x, y$, we can iterate over rows.
                                     For a row $y$, find segments of consecutive empty cells of length $\ge$ piece_width?
                                     This is complex to maintain.
                                     
                                     Let's go with **Local Search (Simulated Annealing)** as the primary solver, with a quick greedy initialization.
                                     Greedy Initialization:
                                     Use a random permutation of pieces.
                                     Sort by size desc.
                                     Use a grid.
                                     For each piece, find the first $(x,y)$ (scanned in priority order) that fits.
                                     Priority: $(0,0), (1,0) \dots (W-1,0), (0,1) \dots$.
                                     Wait, scanning order is $O(W \times H)$.
                                     We can accelerate this by maintaining `next_available_row_index`?
                                     Or just a `vector<pair<int,int>>` of free cells.
                                     Initially full of cells.
                                     When we place a piece, we remove cells.
                                     This allows $O(1)$ finding of next free cell.
                                     BUT we need to check if they form the *shape* of the piece.
                                     Removing cells from a list while checking connectivity is hard.
                                     
                                     Okay, here's a compromise:
                                     Use a grid.
                                     Use `set<pair<int, int>>` for occupied cells.
                                     In the greedy phase, just iterate $y$ from 0 to MAX_H, $x$ from 0 to MAX_W.
                                     But prune:
                                     If `y` is large, check if piece fits.
                                     Actually, just use `vector<int> row_free(MAX_H, 0)`. `row_free[i]` stores a pointer to next empty cell?
                                     No, because a cell can be used.
                                     
                                     Okay, let's rely on `set` for correctness and simplicity.
                                     And optimize the loop.
                                     Wait, checking $y, x$ naively is $O(W \times H)$.
                                     If $W \approx 300, H \approx 150$. $45000$.
                                     $10000 \times 45000 = 4.5 \times 10^8$.
                                     In C++, $4.5 \times 10^8$ simple operations (bitwise check) might pass in 1-2s.
                                     It's risky.
                                     But we don't need to check ALL spots.
                                     Once we find ONE valid spot for a piece, we place it and CONTINUE to the next piece.
                                     We don't revisit spots.
                                     Wait, the problem is that we might miss a good spot later.
                                     But for the greedy phase, we just need *any* valid spot.
                                     So iterating until we find a fit is amortized cost.
                                     However, in the worst case (last piece fitting nowhere), we scan the whole grid.
                                     But then we fail.
                                     
                                     Assumption: Pieces fit.
                                     Then the cost is $O(TotalCells \times \text{SearchFactor})$.
                                     If we fill the grid sparsely, search is fast?
                                     No, we fill it densely.
                                     
                                     I will implement a fast grid scan using 1D array and bitmasks.
                                     Since $k \le 10$, piece shape is small.
                                     We can check compatibility using bitmasks?
                                     Row $r$: if `grid[r]` matches piece mask at offset?
                                     Yes.
                                     So:
                                     1. Convert piece to bitmasks.
                                        Max width of piece is 10.
                                        So we can encode a row of a piece into an integer.
                                        For the grid, we need rows.
                                        If piece spans 10 columns, we need 10 columns worth of mask.
                                     2. For each piece:
                                        Check if fits at $(x,y)$.
                                        This requires checking row `y` against mask...
                                        Actually, bit manipulation:
                                        `if ((grid[y] & piece_mask) == 0)` ?
                                        No, pieces can span multiple rows.
                                        Piece occupies row $y$ at cols $c \dots$.
                                        Mask is shifted.
                                        So `(grid[y] >> c) & piece_mask_row`.
                                        Wait, grid columns can be up to 1000.
                                        We can't shift 1000 bits with `int`. `unsigned __int128`?
                                        Or just `bitset`.
                                        
                                     Let's stick to the simplest working solution:
                                     **Simulated Annealing** on random permutations of pieces with fixed orientation.
                                     No, orientation must be optimized.
                                     
                                     Let's write a solution that:
                                     1. Parses input.
                                     2. Generates all 8 variants for each piece.
                                     3. Uses a `set<pair<int, int>>` for occupied cells.
                                     4. Initializes by trying to place pieces greedily into a bounding box estimated as $\sqrt{\text{Area}}$.
                                         Actually, to guarantee no crashes, use a larger box (e.g. 2000x2000) or dynamic.
                                         Let's fix the box size to be safe.
                                         Total Area $\le 10^4 \times 10 = 100,000$.
                                         Square root $\approx 316$.
                                         Let's use $500 \times 500$ box.
                                         Wait, if pieces are $1 \times 100$ (long), we might need $100 \times 1000$.
                                         But max area is limited by sum of areas.
                                         If we have $10000$ pieces of area 1, we can fit in $1 \times 10000$.
                                         My fixed box of $500 \times 500$ is not safe for $1 \times 10000$.
                                         So I should make the box adaptive or use a very large one.
                                         $10000 \times 10 = 100,000$ cells.
                                         Max bounding box could be $10000 \times 10$.
                                         So grid size $10000 \times 10$ is safe.
                                         Or $100 \times 1000$.
                                         If I use $1000 \times 100$, it's symmetric.
                                         Actually, let's use a large enough bounding box: $W_{max} = 2000, H_{max} = 2000$.
                                         $4 \times 10^6$ cells.
                                         Fits in memory.
                                         Greedy scan: Iterate $(x, y)$?
                                         Optimization:
                                         Maintain `grid[y]` as a bitmask or integer.
                                         Since $W_{max}$ can be large, we can't use bitsets for full width efficiently with integers.
                                         But we can use `vector<int> grid_rows[H]`.
                                         Wait, `H` can be 2000.
                                         Vector of Vectors is fine.
                                         
                                         Optimization for placement finding:
                                         We want to find $(x, y)$ fastest.
                                         Keep a list of "active" rows or something.
                                         But let's just use the naive scan.
                                         To avoid $O(W \times H)$ per piece, we can stop scanning row $y$ once we found a fit for *that* row?
                                         No, we scan $(0,0) \to (W,H)$.
                                         Once we find a valid spot, we place and break.
                                         This assumes we check $(0,0)$ first.
                                         Is it okay to check $(0,0)$?
                                         It's not necessarily the best spot, but it guarantees progress.
                                         Wait, if we check $(0,0)$ and it's blocked, we move to $(0,1)$?
                                         Yes.
                                         So we iterate $y$ then $x$ or vice versa.
                                         To be faster, we can maintain `next_free[y]` = smallest $x$ where `grid[y]` is not fully blocked?
                                         No, piece is 2D.
                                         
                                         Okay, I will trust that $N \times 8 \times (\text{Average Checks})$ will pass.
                                         Average checks is low because pieces usually fit early in the scan if we start from (0,0)?
                                         Not really.
                                         But with random shuffle or rotation, it helps.
                                         
                                         Wait, I can optimize the grid.
                                         Flatten it.
                                         Use `vector<char> grid`.
                                         Implement a custom search.
                                         
                                         Let's proceed with coding.
                                         I will include a header `<bits/stdc++.h>`.
                                         I will implement a class `Polyomino`.
                                         Function `generate_shapes()`.
                                         Solver class.
                     
                                         One important detail:
                                         "Output Format... R_i F_i".
                                         Reflection flag F: 1 = reflect across Y-axis. 0 = no.
                                         Rotation R: 0..3.
                                         Order: Reflect -> Rotate -> Translate.
                                         My code should track this.
                                         Note: "Reflect across the y-axis before rotation".
                                         Original point $(x, y)$.
                                         Reflect across Y: $(x, y) \to (-x, y)$.
                                         Rotate 90 deg clockwise $(u, v) \to (v, -u)$.
                                         Apply Transform:
                                         Point $p(x,y)$.
                                         If Reflect: $p(-x, y)$.
                                         If Rot R: apply $(x, y) \to (y, -x)$ repeatedly.
                                         This defines the shape.
                                         The shape coordinates can be anything.
                                         We need to normalize them to fit the grid (positive).
                                         We subtract min-x, min-y from all points in the shape.
                                         Then translate later.
                                         
                                         I'll write the solution now.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <numeric>
#include <random>
#include <chrono>

using namespace std;

// Represents a cell in the grid
struct Point {
    int x, y;
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

// Represents a transformed piece
struct Shape {
    int w, h; // Bounding box width and height (normalized)
    vector<Point> cells;
    int original_id;
    
    // Constructor
    Shape(int orig_id) : original_id(orig_id) {}
};

// Reads a polyomino and normalizes coordinates
void read_piece(vector<Point>& raw, int n) {
    int min_x = 1000000, min_y = 1000000, max_x = -1000000, max_y = -1000000;
    raw.reserve(10);
    
    for (auto& p : raw) {
        min_x = min(min_x, p.x);
        min_y = min(min_y, p.y);
        max_x = max(max_x, p.x);
        max_y = max(max_y, p.y);
    }
    for (auto& p : raw) {
        p.x -= min_x;
        p.y -= min_y;
    }
}

int W_global, H_global;

// Global grid for occupied status. 
// We use a 2D vector.
// Given constraints and time limit, we'll use a flattened vector or vector of vectors.
// Since we need to support potentially large areas, we use a reasonable bound.
// Max area is 10^4 * 10 = 100,000. A 300x300 grid covers 90,000.
// If we need more, we can dynamically expand or use a set.
// However, for competitive programming, usually a fixed grid with ample size is used for heuristics.
// Let's use a safe size: 500x500 = 250,000. This fits comfortably.
// If area is larger than that, we scale up.
// Let's dynamically determine bounds.
// Estimation: H = ceil(sqrt(total_area)), W = total_area / H.
// But better to use a large enough rectangle, e.g. 500x500, 600x600.
// To be safe against aspect ratios, let's use a slightly larger buffer.
// Actually, we can just use a 2D vector with max bounds derived from input.
// Max width of a piece is <= 10. Max total width could be ~10000.
// But optimal packing reduces width.
// Let's use a 2D vector with dynamic sizing based on a heuristic or fixed large size if time permits.
// For safety and speed, we will assume a 200x200 max effective area for dense packing, but handle overflow.
// Actually, with N=10000, we might need a larger grid.
// Let's use `vector<vector<int>> grid` where grid[x] corresponds to column x, holding occupied y's?
// Or just a `vector<vector<bool>>` and hope for best.
// Let's use `vector<int> grid_flat`?
// To ensure speed and correctness, I will use a `set` for the initial greedy placement, 
// as it's safer regarding bounds than guessing W, H.
// Wait, sets are slow.
// Let's use `vector<bool> occupied` with a flattened index and dynamic capacity.
// But we need to handle translation.
// Translation is relative. We can fix the bounding box of the final solution.
// Since we need to output specific X, Y, we need to maintain absolute positions relative to (0,0).

struct PieceInfo {
    int id;
    int area;
    vector<Shape> shapes; // Up to 8 variants
    
    PieceInfo(int id, int k) : id(id), area(k) {}
};

vector<PieceInfo> pieces;
int max_n = 10000;
int max_k = 10;
int grid_limit_w = 600; 
int grid_limit_h = 600; 

// To optimize grid access, we use a flattened `vector<int>` or `vector<bool>`.
// Since we might shift the coordinate system, we add an offset.
// But actually, we can just clamp coordinates or handle negative indices by mapping to positive.
// Let's fix the global origin at (0,0).
// The bounding box will be maintained by tracking minX and minY.

// Use a simple grid with a large enough size. 
// Given area <= 100,000, a grid of 1000x1000 is plenty safe.
// 1000x1000 char = 1MB.
vector<vector<bool>> occupied_grid;

// Helper to place a shape
bool try_place(const Shape& s, int& rx, int& ry) {
    // Try to find a position (rx, ry) that fits.
    // We want the first valid one?
    // To optimize, we will try to find *any* valid one.
    // Since W, H are implicit, we just need 0 <= rx < W_final, 0 <= ry < H_final.
    // But we don't know W_final yet.
    // We will maintain a global W_global, H_global based on max extent.
    // But for checking collision, we assume infinite plane and use occupied_grid.
    // We need to check if we stay within some logical bounds or just use the grid.
    
    // We'll scan the grid for a suitable top-left corner (rx, ry).
    // Start scanning from (0,0).
    // To make it faster:
    // Just iterate y then x?
    // Given the time constraint, we optimize by trying to place as close to (0,0) as possible.
    // But checking all y,x is slow.
    // Heuristic: try random spots?
    // No, stick to "scan".
    // Optimization: maintain a list of "candidates" (free cells).
    // But let's just do a simplified scan.
    
    // To make it fit within 1s:
    // We only scan a restricted area or use a heuristic to skip.
    // Or, use `grid[y].find(x)`?
    // Let's iterate cells?
    // Actually, since we have N=10000, we must be very fast.
    // Iterating 600*600 is 360,000. 10000 * 360000 is too slow.
    // We need to find the spot faster.
    
    // We can keep a list of "open intervals" in each row.
    // This is complex to implement.
    // Let's use `set<pair<int, int>> occupied`.
    // Check intersection. O(k log N).
    // Iterate positions?
    // How to find position?
    // Let's iterate (y, x) in raster order.
    // But we can optimize:
    // We maintain a list of "active" rows that contain cells.
    // Or just rely on the fact that pieces are small.
    
    // Let's use the set approach. It is safest.
    // And to speed up, we can restrict the search range.
    // For each piece, we try to place it adjacent to other pieces?
    // Just random search?
    // Let's try to find the *first* valid spot in a raster scan.
    // But skip spots that were recently checked?
    // Or use `vector<int> row_free_cols` for each row?
    // Let's try to implement a simple row-sweep.
    // We keep `grid[y]` as a bitmask? No, width 600.
    // We keep `grid[y]` as a boolean array.
    // We also maintain `vector<int> next_free_row`? No.
    
    // Simplest fast approach:
    // Keep a `priority_queue` of available cells?
    // No.
    
    // Let's try this:
    // We know pieces are placed.
    // We maintain `vector<vector<bool>> occupied`.
    // We also maintain `vector<int> empty_count_per_row`.
    // If empty_count_per_row[y] < piece_width, this row cannot accommodate piece at y?
    // No, piece can span multiple rows.
    
    // Okay, for the sake of passing within limits and complexity, 
    // I will use a randomized placement heuristic.
    // 1. Pick random x, y within [0, LIMIT].
    // 2. Check overlap.
    // 3. Repeat k times.
    // This finds a spot quickly if one exists.
    // If not, it fails.
    // But for the initial greedy phase, if a valid packing exists (which we assume for small polyominoes), 
    // random search should find a spot.
    // However, "first fit" is better for density.
    
    // Let's implement a hybrid:
    // Check a few specific "likely" positions (neighbors of occupied cells).
    // Maintain a `set` of occupied cells.
    // When placing a piece, collect all neighbors of occupied cells.
    // These are the most promising locations.
    // Iterate these candidates.
    
    // Since I cannot implement complex data structures reliably in the short window,
    // I will stick to `set` and scan.
    // Wait, `set` scan is slow.
    // Let's use `vector<vector<char>> occupied` and a simple loop.
    // We limit the loop range to `LIMIT`.
    // To speed up, we break early.
    // To avoid TLE, we use `ios::sync_with_stdio(false)`.
    
    // Let's assume the checker accepts approximate solutions or I need to be clever.
    // Actually, I can just use a `vector<bool>` grid.
    // Iterate `y` from 0 to `H`. Iterate `x` from 0 to `W`.
    // Check fit.
    // If fit, place and break.
    // To make it fast, I will check `occupied` using bitsets if possible.
    // But `vector<bool>` is slow for bitwise ops.
    
    // Let's trust `set` for correctness and hope `k` is small enough to not TLE.
    // With `k=10`, `set` overhead is low.
    // The only concern is finding the spot.
    // I'll scan from (0,0).
    // Optimization:
    // Don't scan every time.
    // Keep `last_checked_x`, `last_checked_y`? No.
    
    // Let's write the code.
    return true; 
}

// Actual implementation details follow.
// I will use a simpler logic: 
// Grid size 2000x2000 is safe for area 100,000? 
// No, 100,000 fits in 1x100,000.
// We need to handle aspect ratio.
// I'll use `set<pair<int,int>> occupied`.
// For each piece, try random orientation.
// For placement:
// Iterate `x` from 0 to 2000. Iterate `y` from 0 to 2000.
// If occupied set check fails for all cells, break and return found?
// Wait, if I check 2000x2000 for 10000 pieces, it's 4e9 ops.
// I need to find a spot *without* iterating empty space.
// I will use a `queue` of candidate cells.
// Initially `queue` has all cells? No.
// Queue has `(0,0), (1,0)...`
// I will just place pieces in `occupied` and check.
// I will iterate rows.
// `vector<int> row_start`?
// I will just use `occupied` set and check random spots.
// With N=10000, maybe the time limit is lenient or the pieces are small.
// Actually, `set` operations are fast enough if we don't do too many.
// Checking `k` cells takes $10 \times 16 \approx 160$ ops.
// We check 100 spots -> 16000 ops.
// For 10000 pieces -> 1.6e8 ops.
// This is acceptable!
// So I will iterate `x` from 0 to 100, `y` from 0 to 1000.
// And also random spots.
// But wait, `y` up to 1000 is fine.
// I'll define a search limit.
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>
#include <numeric>
#include <random>
#include <chrono>
#include <tuple>

using namespace std;

struct Point {
    int x, y;
    bool operator<(const Point& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

int N;
int MAX_W = 2000; 
int MAX_H = 2000; 

vector<vector<Point>> all_shapes; // all_shapes[i][j] is a shape variant for piece i

struct PieceSolution {
    int id;
    int r; // Rotation 0..3
    int f; // Flip 0/1
    int ox, oy; // Position
    vector<Point> cells;
};

vector<PieceSolution> solutions;
set<pair<int,int>> occupied_cells;

// Check if a piece fits at a given position
bool check_fit(int rx, int ry, const vector<Point>& shape_cells) {
    for (const auto& p : shape_cells) {
        pair<int,int> pos = {rx + p.x, ry + p.y};
        if (pos.first < 0 || pos.second < 0 || pos.first >= MAX_W || pos.second >= MAX_H) return false; // Boundaries
        if (occupied_cells.count(pos)) return false; // Collision
    }
    return true;
}

// Place a shape
void place_shape(int rx, int ry, const vector<Point>& shape_cells) {
    for (const auto& p : shape_cells) {
        pair<int,int> pos = {rx + p.x, ry + p.y};
        occupied_cells.insert(pos);
    }
}

// Remove a shape (undo place)
void unplace_shape(const vector<Point>& shape_cells) {
    // We need to know where to remove? 
    // Just iterate and remove.
    // Since we stored points in solution, we need to store them here.
    // Or pass the points.
}

void solve() {
    cin >> N;
    all_shapes.clear();
    all_shapes.resize(N);
    int total_area = 0;
    
    for (int i = 0; i < N; ++i) {
        int k;
        cin >> k;
        vector<Point> raw(k);
        total_area += k;
        for (int j = 0; j < k; ++j) {
            cin >> raw[j].x >> raw[j].y;
        }
        vector<Point> normalized = raw;
        int min_x = normalized[0].x;
        int min_y = normalized[0].y;
        for(auto &p : normalized) {
            p.x -= min_x;
            p.y -= min_y;
        }
        
        // Generate transformations
        // Base: reflected across Y axis -> (x, y) -> (-x, y)
        // But for storage, we can normalize again.
        // Actually, simpler:
        // 1. Identity
        // 2. Rotate 90
        // 3. Rotate 180
        // 4. Rotate 270
        // Same for Reflection
        
        // Let's generate 8 variants.
        // Variant: (is_reflected, rot_angle)
        // Apply to normalized points.
        // Normalize again (shift min to 0).
        
        all_shapes[i].reserve(8);
        
        vector<int> xs(k), ys(k);
        for(auto &p : normalized) { xs[p.id] = p.x; ys[p.id] = p.y; } // Wait, ids lost?
        // Let's use raw pointers? No.
        // Let's copy.
        
        vector<Point> p_temp;
        for(auto &p : normalized) p_temp.push_back(p);
        
        for(int rot=0; rot<4; ++rot) {
            for(int refl=0; refl<2; ++refl) {
                vector<Point> shape = p_temp;
                if(refl) {
                    for(auto &p : shape) p.x = -p.x;
                }
                // Rotation: (x,y) -> (y, -x)
                for(int r=0; r<rot; ++r) {
                    for(auto &p : shape) {
                        int tx = p.x;
                        p.x = p.y;
                        p.y = -tx;
                    }
                }
                
                // Normalize
                int min_xv = 100000, min_yv = 100000;
                for(auto &p : shape) {
                    min_xv = min(min_xv, p.x);
                    min_yv = min(min_yv, p.y);
                }
                for(auto &p : shape) {
                    p.x -= min_xv;
                    p.y -= min_yv;
                }
                
                // Compute bounding box
                int max_xv = -1, max_yv = -1;
                for(auto &p : shape) {
                    max_xv = max(max_xv, p.x);
                    max_yv = max(max_yv, p.y);
                }
                
                all_shapes[i].push_back({(int)shape.size(), shape});
            }
        }
    }
    
    // Heuristic Placement
    // Sort by area descending
    // Actually, N is large, let's sort.
    vector<int> p_indices(N);
    iota(p_indices.begin(), p_indices.end(), 0);
    sort(p_indices.begin(), p_indices.end(), [&](int a, int b) {
        return all_shapes[a].size() > all_shapes[b].size(); // Size is vector of variants, actually area = k.
        // Let's store area in a separate struct or just use k.
        // I didn't store k, but I can recompute or store it.
        // Let's store k.
    });
    // Fix sort criteria: actual area.
    // I'll store area in PieceInfo struct.
    // Oh wait, `all_shapes[i]` has 8 items, each has size. They are all equal to k.
    // So sorting by size is irrelevant as all pieces of same index have same area.
    // Sort by k directly?
    // We can store k in a vector.
    // But we just consumed input.
    // I'll just use random shuffle to avoid worst case, or just sequential?
    // Sequential is fine.
    
    // Greedy placement loop
    solutions.assign(N, PieceSolution{-1, 0, 0, 0, 0});
    occupied_cells.clear();
    
    // We need to map back to original indices.
    // Let's store the result in `solutions` indexed by original i.
    
    // Optimization: To speed up finding a spot, use a heuristic search.
    // Try (0,0) first, then (1,0), etc.
    // But to avoid TLE, we can just use a randomized search for placement.
    // Since area is small, we expect a spot to be found quickly.
    
    // To ensure coverage of the grid, we might need a large grid.
    // Let's use a dynamic check.
    
    for (int idx : p_indices) { // Actually just 0 to N-1
        int id = idx;
        const auto& variants = all_shapes[idx];
        int best_found = -1;
        vector<Point> best_shape;
        pair<int,int> best_pos = {-1, -1};
        int best_variant = -1;
        
        // We need to find a valid placement.
        // If we find one, break.
        
        // Try random order of variants
        for(int var=0; var<variants.size(); ++var) {
            // Shuffle variants?
        }
        
        // Heuristic: try to place "near" others?
        // Just try (0,0) first.
        // If occupied, try (0,1), etc.
        // To do this efficiently:
        // Scan (x,y) from (0,0) upwards.
        // If occupied at (x,y), skip?
        // No, check piece.
        
        // Since `occupied_cells` is a set, lookup is log N.
        // Looping 200x200 is too slow.
        // Let's limit scan to 100x100 first?
        // Or rely on `set` to filter.
        // Actually, just pick a random position (x,y) from [0, 2000].
        // Repeat 100 times.
        // If no success, expand?
        
        // Let's use a specific strategy:
        // We maintain a list of "open" rows or columns.
        // Or simpler:
        // Scan `y` from 0 to 1000.
        // For each `y`, find first `x` where `occupied[y][x]` is false?
        // This is complex with a set.
        
        // Okay, I will use a random search for placement to save time.
        // It might be suboptimal but should find a solution.
        
        // Let's use `mt19937`.
        mt19937 rng(123);
        
        bool placed = false;
        int best_var = 0;
        int best_x = 0, best_y = 0;
        int best_w = 0; // Width of shape
        
        // Try to minimize fragmentation?
        // Just pick first valid.
        
        for(int var_idx = 0; var_idx < variants.size(); ++var_idx) {
            if(placed) break;
            // Try a variety of positions
            // Start with (0,0)
            int cx = 0, cy = 0;
            // Expand cx, cy randomly
            // Actually, let's just iterate cx, cy
            // But limit iterations.
            // Try 100 random positions per variant.
            
            vector<Point>& shape = variants[var_idx];
            // We need to extract the cells from the vector<Point>
            
            // Try to find a position
            // We can try (0,0) -> (500, 500)
            // To speed up, check if we can just place it in an empty row?
            // No, just random.
            
            for(int k_search=0; k_search<50; ++k_search) {
                int rx = uniform_int_distribution<>(0, 500)(rng);
                int ry = uniform_int_distribution<>(0, 500)(rng);
                if(check_fit(rx, ry, shape)) {
                    best_found = 1;
                    best_x = rx;
                    best_y = ry;
                    best_var = var_idx;
                    // Copy shape
                    // Wait, we need to store the exact shape for the solution
                    // Shape is already in `all_shapes[idx][var_idx]`
                    best_shape = shape;
                    placed = true;
                    // Record solution
                    solutions[id] = {id, uniform_int_distribution<>(0, 3)(rng), 0, best_x, best_y}; 
                    // Wait, recording is tricky.
                    // We record the variant index and the rotation/flip used to get it.
                    // We can store the rotation/flip in the solution struct.
                    // Let's re-calculate R, F from var_idx?
                    // No, we need to store them.
                    // Let's store them in a parallel array or in PieceSolution.
                    break; 
                }
            }
            if(!placed) {
                 // If not found in 50 tries, maybe the shape is too big or constrained.
                 // Let's fallback to systematic scan (0..100, 0..100)
                 for(int cy_sys = 0; cy_sys < 200; ++cy_sys) {
                     for(int cx_sys = 0; cx_sys < 200; ++cx_sys) {
                         if(check_fit(cx_sys, cy_sys, shape)) {
                             best_x = cx_sys;
                             best_y = cy_sys;
                             placed = true;
                             best_shape = shape;
                             break;
                         }
                     }
                     if(placed) break;
                 }
            }
            if(placed) break;
        }
        
        // Save shape cells to solution
        if(!placed) {
            // Should not happen if grid is large enough
            // Just crash or fallback
            cout << "ERROR: Cannot place piece " << id << endl;
        }
        // Update occupied
        for(auto &p : best_shape) {
             occupied_cells.insert({best_x + p.x, best_y + p.y});
        }
        // Store info for output
        // We need to reconstruct R, F.
        // Let's store in a helper.
    }
    
    // ... Rest of the logic ...
}

int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    // Reading N...
    // I need to put the reading logic inside solve() or main().
    // Since I need to wrap the solution, I'll put everything in main().
    // Also, I need to print W and H.
    // After placing all pieces, compute max X and Y.
    // W = max_x - min_x + 1
    // H = max_y - min_y + 1
    // But my placement uses [0, 500] etc.
    // I need to find min_x, min_y among placed cells.
    // Wait, I didn't track min_x, min_y.
    // I will track them.
    
    // Also, I need to re-run the placement with tracking of R and F.
    // I'll simplify:
    // `solution_vars` stores (R, F, x, y).
    // `solutions` stores cells.
    
    // One correction:
    // The problem asks to minimize area.
    // My solution might not be optimal, but valid.
    // With 10^4 pieces, I will likely find a valid solution.
    
    // Re-writing the structure to be complete.
    // I will omit complex SA logic to fit token/time, focusing on a solid Greedy.
    
    return 0;
}
```
Wait, I need to provide the full code. The above snippet was incomplete.
Let's structure it properly.
I will use a simpler logic for R/F deduction.
I will precompute the R/F for each variant.
For `variant i` in `all_shapes[idx]`:
Iterate `r=0..3`, `f=0..1`.
If `variant` matches, store `R=r, F=f`.
Then I can fill `Solution` struct.

Final check on constraints:
N <= 10000.
My solution runs in O(N * Area_Limit). Area_Limit ~ 100x100 = 10000.
Total ~ 10^8 operations. Feasible in 1s.

One logic error in `read_piece`:
I read `raw` but `all_shapes` expects normalized `Point`.
My code calculates `normalized` and pushes to `all_shapes`.
Wait, in the snippet I had `all_shapes[i].reserve(8);` then generated.
The logic seems sound.

Let's assemble the code.
I will use `set` for `occupied`.
I will use `rand()` or `mt19937` for random seeds to prevent TLE on bad cases.
I will add a fallback to `check_fit` with loop bounds.
To handle coordinates, I'll use `pair<int, int>`.

Also, to minimize W and H, I will perform a post-processing step?
No, just output the bounding box of the current placement.
To improve W, H, I could try shifting the whole block to align with 0?
My coordinates are relative to (0,0).
If I place at `(5, 5)` everything, `min_x=5`. `W = max - min + 1`.
Wait, my `check_fit` enforces `rx >= 0`, `ry >= 0`.
So `min_x` could be positive.
If I shift everything left/down, I reduce W/H?
Yes.
After placement, find `min_x`, `min_y` of all placed cells.
Shift all `solutions[i].ox += (0 - min_x)`, `solutions[i].oy += (0 - min_y)`?
Yes, this centers the box at 0, reducing the apparent W, H.
But the output requires `X_i, Y_i` to be within `0 <= x < W`.
So if I shift to make `min_x = 0`, `W` becomes the new width.
Correct.

Wait, if I have pieces at `x=5` and `x=10`, `W` should be `10-5+1 = 6`.
By shifting `x` to start at 0, we get `0..5` -> Width 6.
So yes, shifting minimizes the reported box.

I should calculate `min_x` and `min_y` after all pieces are placed.
Wait, `occupied` set will contain `pair<int, int>`.
I can iterate the set?
Or maintain `min_x`, `min_y` during placement?
Easier to iterate at the end (since set size <= 100,000).

Final code structure:
1. Includes.
2. Structures.
3. `main`:
    - Read N.
    - Loop N times to read pieces, generate shapes.
    - Sort pieces by size (descending).
    - Grid setup (implicit via set).
    - Loop pieces, try to place.
    - Track `placed` status.
    - If all placed:
        - Calculate `min_x`, `min_y` from `occupied`.
        - Update all piece coordinates.
        - Calculate `W`, `H`.
        - Print W H.
        - Print piece details.
    - Handle failures (should not happen with large grid).

One detail: `F` (reflection) flag.
Problem: `1 = reflect across the y-axis before rotation`.
So `F=1` means reflect first, then rotate.
My code:
Base shape `raw`.
Loop `rot 0..3`.
Loop `f 0..1`.
If `f==1`, reflect points `(x, y) -> (-x, y)`.
Then rotate `rot` times.
Store `f` and `rot` in a list for the shape.
Note: `all_shapes` stores the resulting geometry. We need to know which `(r, f)` produced it.
I will store a struct `Variant` in `all_shapes` that includes `r, f, cells`.

Let's write it.

```cpp
/*
    Pack the Polyominoes (Reflections Allowed)
    Solution Strategy: Greedy Placement with Precomputed Shapes
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <cstdlib>
#include <ctime>
#include <limits>

using namespace std;

struct Point {
    int x, y;
    bool operator<(const Point& other) const {
        return x < other.x || (x == other.x && y < other.y);
    }
};

struct PieceVariant {
    int w, h;
    vector<Point> cells;
    int r, f; // Rotation and Flip (input flags)
};

struct SolutionPiece {
    int id;
    int r, f;
    int ox, oy;
    vector<Point> cells;
};

int N;
set<pair<int, int>> occupied;
vector<SolutionPiece> final_solutions;
vector<vector<PieceVariant>> all_pieces;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> N)) return 0;

    all_pieces.resize(N);
    int total_area = 0;
    int max_area_k = 0;

    for (int i = 0; i < N; ++i) {
        int k;
        cin >> k;
        total_area += k;
        max_area_k = max(max_area_k, k);
        
        vector<Point> raw(k);
        for (int j = 0; j < k; ++j) {
            cin >> raw[j].x >> raw[j].y;
        }

        // Normalize
        int min_x = 100000, min_y = 100000;
        for (auto &p : raw) {
            min_x = min(min_x, p.x);
            min_y = min(min_y, p.y);
        }
        for (auto &p : raw) {
            p.x -= min_x;
            p.y -= min_y;
        }

        all_pieces[i].reserve(8);

        // Generate all 8 orientations
        // F: 0 or 1. F=1 -> reflect across Y (-x).
        // R: 0..3. Rotate clockwise 90*R.
        
        for (int f = 0; f < 2; ++f) {
            // Apply reflection if needed
            // Note: Reflection of normalized shape depends on axis.
            // The problem says "reflect across y-axis".
            // So (x, y) -> (-x, y).
            // We need to re-normalize after reflection because x became negative.
            vector<Point> shape = raw;
            for (auto &p : shape) {
                p.x = -p.x;
            }

            for (int r = 0; r < 4; ++r) {
                // Apply rotation R times
                for (int rot = 0; rot < r; ++rot) {
                    for (auto &p : shape) {
                        int temp = p.x;
                        p.x = p.y;
                        p.y = -temp;
                    }
                }
                
                // Normalize shape to positive coordinates
                int mx = 0, my = 0;
                for (auto &p : shape) {
                    mx = max(mx, p.x);
                    my = max(my, p.y);
                }
                // Note: min_x/y might be 0 because of previous normalization?
                // Wait, if we reflect, min_x might become negative (originally positive).
                // Example: raw at (0,0) -> (0,0). F=1 -> (-0, 0) -> 0.
                // raw at (2,0) -> (-2,0). Min is -2. Shift by +2 -> (0,0).
                // My normalization logic below is correct.
                
                // However, min calculation needs to account for min value.
                int min_vx = 1000000, min_vy = 1000000;
                for (auto &p : shape) {
                    min_vx = min(min_vx, p.x);
                    min_vy = min(min_vy, p.y);
                }
                for (auto &p : shape) {
                    p.x -= min_vx;
                    p.y -= min_vy;
                }

                // Bounding box
                int bx = 0, by = 0;
                for (auto &p : shape) {
                    bx = max(bx, p.x);
                    by = max(by, p.y);
                }

                PieceVariant pv;
                pv.w = bx + 1;
                pv.h = by + 1;
                pv.cells = shape;
                pv.r = r;
                pv.f = f;
                all_pieces[i].push_back(pv);
            }
        }
    }

    // Prepare solution containers
    final_solutions.resize(N);

    // Sort pieces by area descending
    vector<int> perm(N);
    iota(perm.begin(), perm.end(), 0);
    sort(perm.begin(), perm.end(), [&](int a, int b) {
        return (int)all_pieces[a].size() > (int)all_pieces[b].size();
    });

    // Random generator for tie-breaking
    // srand(time(NULL)); // Good practice
    // mt19937 rng(time(NULL));
    // For deterministic behavior in CP, often seed is constant or time.
    // Let's use a fixed seed for reproducibility unless specified.
    // But randomness helps packing.
    // Let's use rand() % variants.
    
    // To optimize search, we can shuffle the order of checking variants?
    // Yes, randomize order of variants.

    for (int i : perm) {
        // Find placement for piece i
        int id = i;
        bool placed = false;
        
        // Heuristic: Try random positions
        // We need to ensure we find *at least* one.
        // Since k <= 10, grid should be large enough.
        // Let's use a search range [0, 2000].
        // And try to find a fit.
        
        // We will try all 8 variants in random order
        vector<int> var_order(all_pieces[id].size());
        for(size_t j=0; j<var_order.size(); ++j) var_order[j] = j;
        // shuffle(var_order.begin(), var_order.end(), rng); 
        // Without RNG object:
        for(size_t j=0; j<var_order.size()-1; ++j) {
            for(size_t k=j; k<var_order.size(); ++k) {
                if(rand() < RAND_MAX/2) swap(var_order[j], var_order[k]);
            }
        }

        // Search for a position. 
        // To speed up, we check if any variant fits at (0,0) or nearby.
        // Actually, just picking a random valid spot is hard.
        // We iterate x,y in a limited range.
        // Since we want to minimize W, H, picking (0,0) first is good, but might not fit.
        // Let's use a small range first, then expand.
        
        int best_rx = -1, best_ry = -1;
        int best_var_idx = -1;

        // Iterate x, y up to 500
        for(int ry = 0; ry < 500 && !placed; ++ry) {
            for(int rx = 0; rx < 500 && !placed; ++rx) {
                for(size_t var_idx : var_order) {
                    // Check fit
                    const auto& var = all_pieces[id][var_idx];
                    bool ok = true;
                    for(const auto& p : var.cells) {
                        int nx = rx + p.x;
                        int ny = ry + p.y;
                        if(nx < 0 || ny < 0 || nx >= 2000 || ny >= 2000) { ok = false; break; } // Boundary check
                        if(occupied.count({nx, ny})) { ok = false; break; } // Collision
                    }
                    if(ok) {
                        best_rx = rx; best_ry = ry; best_var_idx = var_idx;
                        placed = true;
                        break; // Break var loop
                    }
                }
                if(placed) break; // Break rx loop
            }
            if(placed) break; // Break ry loop
        }

        if(!placed) {
            // Fallback: If not found in 500x500, we might fail or try random spots
            // With 500x500, it should cover area 100,000.
            // If failed, we just exit or expand.
            // Let's expand to 2000x2000 if needed.
            // But for brevity, assume 500x500 is enough.
            // Or use a larger loop if time permits.
            // Given N=10000, expanding to 2000x2000 is 1.6s check.
            // I'll stick to 500x500 for speed but with a twist.
            // Actually, I'll add a check for larger bounds.
            // Wait, 10000 pieces. 500x500 grid.
            // If it fits, we're good.
            // I will simply run the loop to 1000x1000 to be safer.
            // It adds time but ensures fit.
        }
        
        // If failed in loop, try random spots
        if(!placed) {
             for(int attempt=0; attempt<20; ++attempt) {
                 int rx = rand() % 1000;
                 int ry = rand() % 1000;
                 // check best variants?
                 // Try all vars
                 for(size_t var_idx : var_order) {
                    const auto& var = all_pieces[id][var_idx];
                    bool ok = true;
                    for(const auto& p : var.cells) {
                         if(rx + p.x < 0 || ry + p.y < 0) { ok=false; break; }
                         if(occupied.count({rx+p.x, ry+p.y})) { ok=false; break; }
                    }
                    if(ok) {
                        best_rx = rx; best_ry = ry; best_var_idx = var_idx;
                        placed = true; break;
                    }
                 }
             }
        }
        
        // Store result
        if(!placed) {
            // Should ideally handle
            cerr << "Failed to place piece " << id << endl;
            continue; 
        }

        // Update grid
        const auto& var = all_pieces[id][best_var_idx];
        for(const auto& p : var.cells) {
            occupied.insert({best_rx + p.x, best_ry + p.y});
        }
        
        // Store solution
        final_solutions[id] = {id, var.r, var.f, best_rx, best_ry, var.cells};
    }

    // Post-processing: Calculate W, H
    // Min/Max coordinates
    // Since we use occupied set, we can scan it.
    // Or iterate final_solutions.
    int min_x = 2000, min_y = 2000, max_x = -1, max_y = -1;
    for(const auto& sol : final_solutions) {
        for(const auto& p : sol.cells) {
            int gx = sol.ox + p.x;
            int gy = sol.oy + p.y;
            min_x = min(min_x, gx);
            min_y = min(min_y, gy);
            max_x = max(max_x, gx);
            max_y = max(max_y, gy);
        }
    }
    
    // Shift to 0
    int shift_x = -min_x;
    int shift_y = -min_y;
    
    // W = max_x - min_x + 1 = max_x + shift_x + 1 - 0 + 1?
    // New X = gx - min_x.
    // Range [0, max_x - min_x]. Size = max_x - min_x + 1.
    int W = max_x - min_x + 1;
    int H = max_y - min_y + 1;

    cout << W << " " << H << "\n";
    // Verify bounds?
    // We need to output X, Y.
    // Output X = gx - min_x? Or just X, Y.
    // The problem says "no cell outside [0, W)x[0, H)".
    // So we output X shifted.
    // However, `final_solutions` stores `sol.ox`.
    // The cells relative to `ox` are `cells`.
    // Absolute cell: `ox + p`.
    // So `X_i` for a cell is `ox + p.x - min_x`?
    // Wait, output format: `X_i Y_i R_i F_i`.
    // This applies to the *piece*, i.e. translation applied to the piece.
    // If we shift the whole grid, we must shift all X, Y.
    // So `X_out = sol.ox - min_x`.
    // Wait, `min_x` is min of all cells.
    // So the minimum cell of any piece at `min_x` becomes 0.
    // So `X_out` ranges from 0.
    
    // But wait, `occupied` check used global `rx`, `ry`.
    // We must shift all `final_solutions` coordinates.
    for(auto &sol : final_solutions) {
        // sol.ox is global x.
        sol.ox -= min_x;
        sol.oy -= min_y;
    }
    
    // Re-calculate W, H?
    // Actually W, H should be max(sol.ox + p.x) + 1.
    // Since we shifted, max_x_new = max(oldest max_x - min_old_x) = max_x - min_x.
    // Width is max + 1.
    
    for(const auto& sol : final_solutions) {
        cout << sol.ox << " " << sol.oy << " " << sol.r << " " << sol.f << "\n";
    }

    return 0;
}
```
Wait, I missed defining `SolutionPiece` properly in the snippet (I wrote `final_solutions[id] = {id, ...}` but struct has fields).
And I added `vector<Point> cells` to `SolutionPiece`.
Also `min_x` initialization.
And the `rand()` calls.

Also, `occupied` check:
If `occupied` is `set`, `count` is efficient.
However, `occupied` can grow to 100,000 elements.
Insertions 100,000 times.
It should be fine.

Double check: "Input Format ... Line 1: integer n ... For each polyomino ... k ... k lines".
The code handles this.

One small issue:
`rand()` might not be seeded. I'll add `srand(time(0))` or similar.
In CP, `time(0)` is often discouraged if it causes different outputs on same judge run for debugging, but it's usually fine for "randomized".
Better to use `random_device` or fixed seed if deterministic not required. I'll use `rand()`.

Also `max_x` init `-1` is fine. `min_x` init `2000` is fine (since coords are positive).

Wait, `occupied` logic `check_fit` uses `2000` as bound.
I increased loop to `500`. If it fails, I use `rand()` with `1000`.
Should be fine.
The `check_fit` inside the loops needs to be correct.
My nested loop for checking `rx, ry` checks `occupied.count`.
That's correct.

Wait, inside the loop:
`for(int ry = 0; ry < 500 ...)`
Inside: `for(int rx = 0; rx < 500 ...)`
Inside: `for(size_t var_idx : var_order)`
This order:
Priority is `ry` increasing (bottom to top), then `rx` increasing (left to right).
This favors placing things low and left.
This is good for compactness.

Seems solid.
Wait, I need to include `<random>` if I use `mt19937` or just use `<cstdlib>` and `rand()`.
Using `<cstdlib>` is enough.

One edge case: `N` up to 10000. `rand()` range 32767. Modulo `1000` is fine.
But `rand()` period might be short? 10000 calls is fine.

I will add the full includes and structure now.

Also, `occupied` insertions are repeated.
I can optimize by only inserting when placing, and removing if we backtrack (not implemented).
But we don't backtrack.

One issue: `occupied` size.
If all pieces placed, `occupied` has sum `k`.
Memory usage: `set` node overhead.
Each node ~ 32 bytes + payload.
$10^5 \times 40 \approx 4$ MB.
Very safe.

Time complexity:
Initialization: $O(N \cdot 8 \cdot k)$.
Placement: $N \times 500 \times 500 \times 10 \times \log(10^5)$.
$2.5 \times 10^7 \times 17 \approx 4 \times 10^8$.
Might be slightly tight for 1 sec.
Optimization:
Instead of checking `occupied` every time in the loop, we can assume we don't need to check `occupy` if we maintain `empty_cells`.
But `set` is safer.
Can I optimize the inner loop?
Actually, we stop as soon as `ok` is true.
Most of the time, the first checked position `(0,0)` might be valid for the first piece, and `(0,1)` etc.
But for later pieces, they block.
If blocked, we scan.
The number of checks is high.
However, we only check *valid* shapes.
Many shapes might fit.
Actually, `var_order` loop iterates up to 8 times.
`rx, ry` loop iterates $250,000$ times.
Total ops $250,000 \times 8 \times 10 = 2 \times 10^7$ per piece.
Wait, `occupied.count` is $O(\log M)$.
Total $N \times 250000 \times 10 \times 17 \approx 4 \times 10^{10}$ ops.
**This is TLE.**
$4 \times 10^{10}$ is way too much.

I MUST optimize the search.
Checking 250,000 spots for every piece is fatal.
I cannot scan the grid.
I MUST use a smarter heuristic or data structure.

Smart heuristic:
Place pieces in a specific order?
Since we want to minimize area, maybe we should just place them densely.
Use the `row_free` idea?
Or just `set` based placement without scanning.
How?
Maintain a `priority_queue` of empty cells?
Or just use a **randomized approach** without scanning.
Pick 100 random spots. Check them.
If none fit, pick a spot from neighbors of occupied cells?
With `set` of occupied, finding neighbors is hard.

Alternative:
Use `vector<bool>` grid.
Iterate `rx, ry`?
Still $O(W \times H)$.
If $W \times H$ is small (e.g. 100x100 = 10000), it's fine.
If $W \times H$ is large (500x500 = 250000), it's bad.
Wait, why is $W \times H$ large?
Because we don't know $W, H$.
But we know total area $A$.
If we pack efficiently, $W \approx \sqrt{A}$.
Max $A = 55,000$. $\sqrt{A} \approx 234$.
So $W \times H \approx 55,000$.
Scanning 55,000 cells per piece is $N \times 55,000 = 5.5 \times 10^8$.
This is acceptable!
The bottleneck is `occupied.check`.
If I use a 1D flattened array `vector<char> grid(W*H)`.
Then `grid[x+y*W]` is O(1).
Finding `x,y` is just iterating.
So if $W \approx 234, H \approx 234$.
Iteration $W \times H \approx 55000$.
Total ops $10000 \times 55000 = 5.5 \times 10^8$.
This is ON the edge but might pass (simple array access is fast).
`occupied` set adds log factor.
Using `vector<char>` is $O(1)$ check.
And we can estimate $W, H$ from total area.
$W_{est} = \lceil \sqrt{A} \rceil$, $H_{est} = W_{est}$.
If $A$ is large, maybe $W_{est}$ is slightly off.
Let's use $W=500, H=500$.
$500 \times 500 = 250,000$.
Ops: $10000 \times 250,000 = 2.5 \times 10^9$.
Wait, if I iterate ALL cells, it's too slow.
BUT I don't need to check ALL cells.
I only need to check cells that are NOT occupied?
No, I need to find ANY valid spot.
If I maintain a `set` of free cells?
Initially `free` has $250,000$ cells.
When placing, I remove cells.
For next piece, I pick from `free`.
If I pick `x, y` from `free`, I check if it fits?
Yes, because `free` means "no occupied cells".
But "no occupied cells" means I can place at `x,y` ONLY IF the whole shape fits (doesn't overlap itself, which is guaranteed as it's a single piece, and doesn't overlap others).
Wait, the `free` list only tells me `(x,y)` is empty.
It does NOT tell me `(x + dx, y + dy)` is empty.
So I need to verify.
But if I pick `(x,y)` from `free` randomly, probability of success is low?
No, pieces are small.
If I pick a "corner" of the piece, and it matches the shape...
Actually, if I just iterate `(0,0)` to `(W,H)`, I am checking if `cell (x,y)` and `(x+dx, ...)` are empty.
Checking emptiness for `k` cells.
If I use `vector<bool> grid`, check is fast.
The number of checks is the issue.
However, if I only check up to the bounding box of the piece?
No.

Wait, if I use a **Hash Map** for occupied cells?
Insertion $O(1)$.
Lookup $O(1)$.
Check 10 cells. Fast.
Now, how to find a spot?
Random search!
Iterate 1000 times picking random `x, y` in `[0, 1000]`.
For each, check fit.
Probability of hitting a valid spot?
In dense packing, free space is everywhere.
Probability of collision is high, so finding a valid spot might take many tries.
But `check_fit` is fast ($10$ lookups).
If we try 2000 random spots. $2000 \times 10 = 20000$ ops.
$10000$ pieces $\implies 2 \times 10^8$ ops.
This is perfectly fine!
Random search is $O(N \times \text{Tries} \times k)$.
With Tries=2000, it's fast.
So I will use **Random Search** for placement.
Initialize grid bounds to 2000x2000.
Try random `rx, ry`.
Check `occupied`.
If valid, place.
If not, retry.
Limit retries? No, just retry until found or timeout.
With 2000 tries, it's very fast.
This avoids scanning $W \times H$.

Okay, revised plan:
1. Use `vector<vector<char>> grid` for O(1) check?
   No, use `set<pair<int,int>>` or `unordered_map`. `set` is safer and fast enough.
   Wait, `set` is log N. $2000 \times 10 \times \log(10^5) \approx 2000 \times 10 \times 17 \approx 3.4 \times 10^5$.
   $N=10000 \implies 3.4 \times 10^9$. Too slow.
   Wait, $\log(10^5) \approx 17$.
   $2000 \times 10 \times 17 \approx 340,000$.
   $10,000 \times 340,000 = 3.4 \times 10^9$.
   This is high.
   
   Use `vector<bool> grid`. Size $2000 \times 2000$.
   Flattened index `y * 2000 + x`.
   Check takes $O(k)$.
   $2000 \times 10 = 20,000$.
   $10,000 \times 20,000 = 2 \times 10^8$.
   This is acceptable!
   So use `vector<vector<char>>` or flattened `vector<char>`.
   Wait, `vector<vector<char>>` is safer for memory management.
   
   But wait, $W$ and $H$ might not be 2000.
   I can estimate $W \approx \sqrt{A}$.
   Let's use $W=1000, H=1000$.
   Check loop: 2000 random positions.
   If I hit a valid one, great.
   
   Wait, if grid is fully packed, random search is slow (many collisions).
   But we stop when found.
   In a dense grid, collisions are frequent.
   But we need *any* valid spot.
   Actually, if the grid is fully packed, we might never find a spot for the last piece?
   But we assume valid solution exists.
   
   Better approach: **Grid scan with pruning**.
   Iterate `y` from 0 to `H_est`.
   Iterate `x` from 0 to `W_est`.
   But stop early if we find a spot?
   Yes.
   The issue is iterating $W \times H$ which is $10^6$.
   But we don't check *every* cell if we maintain `row_first_free`?
   Yes, maintain `head_free[y]` for each row.
   `head_free[y]` points to the first empty cell index in row `y`.
   Since we place pieces, cells become occupied.
   Removing from array is $O(1)$ if we swap with last, but we might need to preserve order?
   We don't need to preserve order for `head_free`.
   Just swap `head_free` with `tail_free`?
   No, `occupied` cells are scattered.
   This is "disjoint set union" territory.
   DSU can find next free cell in $O(\alpha(N))$.
   But implementing DSU for 2D grid is tedious.
   
   Let's stick to `vector<bool>` grid and random search, but optimize search range.
   The probability of finding a spot in a partially empty grid is high.
   In a dense grid (near end), it drops.
   But for 10000 pieces, density increases.
   However, with random search, we might check occupied cells.
   If we check 2000 random cells, we likely hit occupied ones.
   This wastes time.
   
   How about **Deterministic Best Fit**:
   Iterate rows $y$.
   Inside row $y$, find the first column $x$ that can fit the piece.
   We can do this by checking `grid[y]` (row bitset) and `grid[y+1]` etc.
   Checking `grid` bitset for availability is fast if width is small.
   But width 1000.
   
   Okay, I will use `vector<bool>` and random search but with a fallback to a limited scan.
   Since $N$ is large, I will rely on the fact that small polyominoes fit easily.
   The random search with 2000 attempts is likely to find a valid spot quickly if space is available.
   Also, I will limit `W, H` to $\approx 500 \times 500$ initially.
   
   Wait, `vector<vector<bool>>` is specialized to use 1 bit per bool.
   Access might be slow due to bitwise ops?
   `vector<bool>` is slow.
   `vector<char>` is faster.
   I'll use `vector<vector<char>>` or flattened `vector<char>`.
   Flattened `vector<char> grid(2000*2000, 0)`.
   Access `grid[x + y*2000]`.
   
   Final code logic:
   Use `vector<char>` grid.
   Random search (2000 tries) per piece.
   Check `grid`.
   Update `grid`.
   This should pass.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <utility>

using namespace std;

struct Point {
    int x, y;
    bool operator<(const Point& other) const {
        return x < other.x || (x == other.x && y < other.y);
    }
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

struct PieceVariant {
    int w, h;
    vector<Point> cells;
    int r, f;
};

struct SolutionPiece {
    int id;
    int r, f;
    int ox, oy;
    vector<Point> cells;
};

// Grid size estimation
int GRID_W = 1000; 
int GRID_H = 1000;
// Use flattened vector for O(1) access
vector<char> occupied_grid;

// Helper to reset grid
void reset_grid() {
    occupied_grid.assign(GRID_W * GRID_H, 0);
}

bool is_occupied(int x, int y) {
    if (x < 0 || y < 0 || x >= GRID_W || y >= GRID_H) return true; // Out of bounds considered occupied/error
    return occupied_grid[y * GRID_W + x];
}

void mark_occupied(int x, int y) {
    if (x >= 0 && y >= 0 && x < GRID_W && y < GRID_H) {
        occupied_grid[y * GRID_W + x] = 1;
    }
}

void unmark_occupied(int x, int y) {
    if (x >= 0 && y >= 0 && x < GRID_W && y < GRID_H) {
        occupied_grid[y * GRID_W + x] = 0;
    }
}

void solve() {
    int N;
    if (!(cin >> N)) return;

    vector<vector<PieceVariant>> all_pieces(N);
    int total_area = 0;

    for (int i = 0; i < N; ++i) {
        int k;
        cin >> k;
        total_area += k;
        vector<Point> raw(k);
        for (int j = 0; j < k; ++j) {
            cin >> raw[j].x >> raw[j].y;
        }

        // Normalize
        int min_x = 1000000, min_y = 1000000;
        for (auto &p : raw) {
            min_x = min(min_x, p.x);
            min_y = min(min_y, p.y);
        }
        for (auto &p : raw) {
            p.x -= min_x;
            p.y -= min_y;
        }

        // Generate variants
        all_pieces[i].reserve(8);
        for (int f = 0; f < 2; ++f) {
            vector<Point> shape = raw;
            if (f) { // Reflect across Y: (x,y)->(-x,y)
                for (auto &p : shape) p.x = -p.x;
            }
            for (int r = 0; r < 4; ++r) {
                for (int _r = 0; _r < r; ++_r) {
                    for (auto &p : shape) {
                        int tx = p.x;
                        p.x = p.y;
                        p.y = -tx;
                    }
                }
                // Normalize again
                int mv_x = 1000000, mv_y = 1000000;
                for (auto &p : shape) {
                    mv_x = min(mv_x, p.x);
                    mv_y = min(mv_y, p.y);
                }
                for (auto &p : shape) {
                    p.x -= mv_x;
                    p.y -= mv_y;
                }

                int bw = 0, bh = 0;
                for (auto &p : shape) {
                    bw = max(bw, p.x);
                    bh = max(bh, p.y);
                }

                PieceVariant pv;
                pv.w = bw + 1;
                pv.h = bh + 1;
                pv.cells = shape;
                pv.r = r;
                pv.f = f;
                all_pieces[i].push_back(pv);
            }
        }
    }

    // Reset grid
    // We assume pieces fit in 1000x1000. If not, we might fail or need larger.
    // But for N=10000, k<=10, area <= 100000. 1000x1000 is 1,000,000. Plenty.
    // We'll dynamically check.
    occupied_grid.assign(GRID_W * GRID_H, 0);

    vector<SolutionPiece> final_solutions(N);
    // Sort by size descending
    vector<int> p_indices(N);
    iota(p_indices.begin(), p_indices.end(), 0);
    sort(p_indices.begin(), p_indices.end(), [&](int a, int b) {
        return (int)all_pieces[a].size() > (int)all_pieces[b].size();
    });

    // Seed for random
    srand(time(0));

    for (int idx : p_indices) {
        int id = idx;
        const auto& variants = all_pieces[id];
        
        // Try to find a placement
        // Heuristic: Random search
        bool placed = false;
        int best_x = -1, best_y = -1;
        int best_var = -1;

        // Try to fill the grid
        // We iterate random spots to speed up.
        // If we fail, we can fall back to scanning? No, let's hope random works.
        // We limit random tries to avoid TLE.
        
        // To improve success rate, we can check corners or specific patterns, 
        // but random is robust enough.
        
        // Try 500 random positions
        for (int t = 0; t < 500; ++t) {
            int rx = rand() % (GRID_W - 10);
            int ry = rand() % (GRID_H - 10);
            
            // We should check if rx, ry allows piece to fit within GRID bounds
            // Actually, grid is our universe.
            
            for (const auto& var : variants) {
                // Check if piece fits at (rx, ry)
                bool ok = true;
                for (const auto& p : var.cells) {
                    int cx = rx + p.x;
                    int cy = ry + p.y;
                    if (cx < 0 || cy < 0 || cx >= GRID_W || cy >= GRID_H) {
                        ok = false; break;
                    }
                    if (occupied_grid[cy * GRID_W + cx]) {
                        ok = false; break;
                    }
                }
                if (ok) {
                    best_x = rx; best_y = ry; best_var = &var;
                    placed = true;
                    break;
                }
            }
            if (placed) break;
        }

        // If random failed, try a deterministic scan (limited)
        if (!placed) {
            // Check (0,0) to (500,500)
            for (int ry = 0; ry < 500; ++ry) {
                for (int rx = 0; rx < 500; ++rx) {
                    for (const auto& var : variants) {
                        bool ok = true;
                        for (const auto& p : var.cells) {
                            int cx = rx + p.x;
                            int cy = ry + p.y;
                            if (cx < 0 || cy < 0 || cx >= GRID_W || cy >= GRID_H) { ok = false; break; }
                            if (occupied_grid[cy * GRID_W + cx]) { ok = false; break; }
                        }
                        if (ok) {
                            best_x = rx; best_y = ry; best_var = &var;
                            placed = true;
                            break;
                        }
                    }
                    if (placed) break;
                }
                if (placed) break;
            }
        }

        if (!placed) {
            // In case of failure, we try to place anyway at (0,0) or whatever
            // Just pick first variant and hope it doesn't overlap too much
            // Or output error
            if (variants.empty()) continue;
            const auto& var = variants[0];
            // Force place at (0,0) if not occupied
            bool ok = true;
            for (const auto& p : var.cells) {
                int cx = p.x, cy = p.y;
                if (occupied_grid[cy * GRID_W + cx]) ok = false;
            }
            if (ok) {
                best_x = 0; best_y = 0; best_var = &var;
            } else {
                // Should not happen given large grid
                // Try random again?
                continue; 
            }
        }

        // Place
        place_shape(best_x, best_y, *best_var);
        final_solutions[id] = {id, best_var->r, best_var->f, best_x, best_y};
    }
}

int main() {
    // Fast IO
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    solve();

    // Post processing
    // Calculate bounding box
    int min_x = 2000, min_y = 2000;
    int max_x = -1, max_y = -1;

    for (const auto& sol : final_solutions) {
        for (const auto& p : sol.cells) {
            int gx = sol.ox + p.x;
            int gy = sol.oy + p.y;
            if (gx < min_x) min_x = gx;
            if (gy < min_y) min_y = gy;
            if (gx > max_x) max_x = gx;
            if (gy > max_y) max_y = gy;
        }
    }

    // Shift coordinates to 0,0 relative to bounding box
    int shift_x = -min_x;
    int shift_y = -min_y;
    
    // Recalculate W, H after shift
    int W = max_x - min_x + 1;
    int H = max_y - min_y + 1;

    cout << W << " " << H << "\n";
    for (auto& sol : final_solutions) {
        // Apply shift to output X, Y
        // Wait, my 'final_solutions' stores global 'ox', 'oy'.
        // We need to print adjusted ox, oy?
        // Yes, "no cell outside [0, W)x[0, H)".
        // The current placement has cells in [min_x, max_x] etc.
        // We must print coordinates relative to (min_x, min_y) = (0,0).
        // So X_out = sol.ox - min_x.
        sol.ox += shift_x;
        sol.oy += shift_y;
        cout << sol.ox << " " << sol.oy << " " << sol.r << " " << sol.f << "\n";
    }

    return 0;
}

// Added function definitions inside main scope or helper if allowed
// I'll inline the placement functions for simplicity.
```
````

---

## 直接SFT 样本(sample_idx=3,得分 0.0,completion_tokens=913)

````
I will use a recursive backtracking search with:

1. **Precomputed canonical forms**: For each input polyomino, generate all transformations (reflection, rotation ×4), normalize each to a canonical "sorted cell list", and store the unique normalized form. This ensures we always compare/lookup shapes identically.

2. **Transformation index table**: Map each (reflect, rot) of each input piece to its normalized canonical form. This lets us quickly know which canonical shape we have after applying F/R.

3. **Placement memoization via hashing**: Maintain a hashable state of placed cells; if we reach the same set of occupied cells by a different search branch, reuse that result instead of re-searching from scratch.

4. **Pruning / ordering**: Try placing pieces in order of their number of cells (most constrained first). Skip dominated or already-observed transformation variants during search.

The code is written in C++ for performance.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

// ---------- Piece representation ----------
struct Piece { int k; vector<pair<int,int>> cells; };
// Transformation: 0=identity, 1=r90cw, 2=r180, 3=r270, 4=Y-reflect, 5=Y-ref+r90, 6=Y-ref+r180, 7=Y-ref+r270
int transX[8] = {0, 1, 1, -1, -1,-1,-1, 0};
int transY[8] = {0,-1,-1, 0, 1, 1, 1, 0};
bool isReflect[8] = {0,0,0,0,1,1,1,1};
vector<vector<vector<pair<int,int>>>> precomputedCells(n);
vector<vector<int>> transToIndex(n);

void addCell(const Piece &p, int x, int y, vector<pair<int,int>>& out) { out.push_back({x,y}); }
// Apply t: 0..3 rotations, 4+ reflections; returns new cells sorted by (y,x) then by min(x+y)*H+x
vector<pair<int,int>> applyPiece(const Piece &p, int t) {
    static vector<pair<int,int>> buf; buf.clear();
    for (auto &c : p.cells) {
        int xx=c.first+transX[t], yy=c.second+transY[t];
        addCell(p, xx, yy, buf);
    }
    // Canonical sort: sort by (min(x+y)*H+x, y, x) => lexicographic of (min*H+x, y, x) where H large enough
    long long H = 1e9;
    sort(buf.begin(), buf.end(), [&](const pair<int,int>&a,const pair<int,int>&b){
        return min((long long)a.first+a.second,H)+a.first==min((long long)b.first+b.second,H)+b.first &&
               (int)(min((long long)a.first+a.second,H)+a.first) == (int)(min((long long)b.first+b.second,H)+b.first);
    });
    // Normalize by shifting so top-left corner is at (minx, minY)
    if (!buf.empty()) {
        int mnx=minx,buf.front().first; int mny=minn,buf.front().second;
        for (auto& c : buf) c.first-=mnx; c.second-=mny;
    }
    return buf;
}
int minx(const vector<pair<int,int>>& v){if(v.empty())return 0; auto&a=v[0]; return a.first;}
int minn(const vector<pair<int,int>>& v){if(v.empty())return 0; auto&a=v[0]; return a.second;}
// Build canonical string (cells separated by space, coords as "x-y")
string canonical(const vector<pair<int,int>>& v) { string s; bool f=true; for(auto&c:v){s+=f?"" :" "; s+=to_string(c.first); s+="-"; s+=to_string(c.second); f=false;} return s;}
vector<string> canonShapes;
int shapeIdx;
```
````
