# 题 171 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=2,得分 0.0,completion_tokens=24137)

````
The problem asks us to find a sequence of moves in an $N \times N$ grid ($N=20$) to visit a sequence of $M$ target squares $(i_k, j_k)$ for $k=1 \dots M-1$.
We start at $(i_0, j_0)$.
Wait, the indices in the problem statement say $i_0, j_0$ is the starting position and we must visit $(i_1, j_1), \dots, (i_{M-1}, j_{M-1})$. The input says $M$ coordinates follow. Let's check the constraints.
"In all test cases, $N = 20$ and $M = 40$ are fixed."
Input format:
$N$ $M$
$i_0$ $j_0$
...
$i_{M-1}$ $j_{M-1}$

Actually, usually "visit target squares" means $i_1 \dots i_{M-1}$. But wait, if $M=40$, then there are 40 lines of coordinates following $N \ M$.
Let's re-read carefully: "You must visit the target squares $(i_1, j_1), \dots, (i_{M-1}, j_{M-1})$".
But the input gives $M$ lines of coordinates after $N \ M$. The first one is $(i_0, j_0)$. So there are $M$ points given in total, but the task is to visit $M-1$ targets?
Or does it mean we visit $i_1 \dots i_{M-1}$ given $i_0$ is start?
Actually, looking at typical problems of this sort (e.g., AHC - AtCoder Hitachi Challenge or similar grid navigation), usually all points are important.
However, the text says: "Visit the specified target squares $(i_1, j_1), \dots, (i_{M-1}, j_{M-1})$". And "Initially ... no blocks inside".
Wait, if $i_0, j_0$ is the starting position, it is not a target we need to visit *after* start unless specified.
However, looking at the scoring:
$m$ is number of target squares successfully visited.
Target squares are likely $i_1, \dots, i_{M-1}$? Or are they all provided in the input list except the starting one?
Let's assume the list of coordinates in input represents $P_0, P_1, \dots, P_{M-1}$. We start at $P_0$. We must visit $P_1, P_2, \dots, P_{M-1}$ in that order.
Wait, let me double check the scoring logic.
"If $m < M-1$, $m+1$" -> This suggests we want to visit as many as possible up to $M-1$.
If we can visit all $M-1$ targets, score depends on $T$.
This implies the set of targets is size $M-1$. Since there are $M$ coordinates provided in input (excluding maybe one?), actually the input says:
"$i_0$ $j_0$" (start)
"$\vdots$"
"$i_{M-1}$ $j_{M-1}$"
So there are $M$ lines of coordinates. The problem says "target squares $(i_1, j_1), \dots, (i_{M-1}, j_{M-1})$".
This means the targets are indices $1$ to $M-1$. There are exactly $M-1$ targets.
The starting point is index 0.
So $M=40$ means 39 targets? That seems small. Maybe I should check if $M$ is the count of targets plus start, or just count of targets.
"Input ... $i_0$ $j_0$, ..., $i_{M-1}$ $j_{M-1}$". This means there are $M$ coordinate pairs.
If targets are $1$ to $M-1$, there are $M-1$ targets.
However, often in such grid problems (like sliding puzzle or Sokoban variants), all points are waypoints.
Let's assume the goal is to visit points $1, 2, \dots, M-1$ sequentially starting from $0$.
Constraints: $N=20$, Max actions $2NM = 2 \times 20 \times 40 = 1600$.
This is quite loose compared to $O(M^2)$ or even $O(N^2 M)$ pathfinding if simple BFS/A*.
However, "slide" moves across empty space. "Alter" places/removes blocks.
This is effectively a variation of "Sliding Block Puzzle" or similar.
Since blocks are dynamic (Alter action), the state space is large.
But wait, we are given $2NM$ budget. $2NM$ for $N=20, M=40$ is $1600$.
For each segment between target $k$ and target $k+1$, we have some budget.
Since the targets are generated randomly (shuffled coordinates), the map is empty initially. We just need to navigate through an empty grid.
Wait, "Alter" allows placing blocks. Why place blocks? To stop a slide? Or block enemies? There are no enemies.
Only blocks are obstacles. Initially none.
To visit a square by "Slide", you must stop ON it. Sliding stops only when hitting a block or boundary?
"Continue sliding ... until you hit a block". Boundary is outside the rink (blocks).
So if we slide into the boundary, we stop at the edge.
Can we visit a square by Slide? Yes.
Can we visit by Move? Yes.
If the grid is empty, Sliding will go all the way to the wall or existing block.
Since there are no blocks initially, any Slide goes to the edge of the board.
Why do we need Alter? Maybe to create short-cuts or to stop early?
Or maybe the random generation implies something else?
"Randomly shuffle the coordinates ... assign them sequentially".
So the order is random. $P_0 \to P_1 \to \dots \to P_{M-1}$.
Since the board is empty, to go from $A$ to $B$:
If we use Move, distance is Manhattan distance. Cost $|dx| + |dy|$.
If we use Slide, we move until wall/block.
If we don't place blocks, Slides are long.
If we place blocks (using Alter), we can make a square a block, then slide against it? No, if we want to stop on B, we need a block adjacent to B in direction of travel.
Wait, "Alter: Place a block on the adjacent square ... if it does not already contain one; otherwise remove".
"Place a block on the adjacent square in the specified direction".
This means if I am at $x$, looking East, I look at $(x, y+1)$. If empty, I put a block there.
This changes the map for future turns.
Since we have a strict limit $2NM = 1600$, and we need to visit 39 points.
Average steps per point $\approx 40$.
Manhattan distance between random points in $20 \times 20$ is around $10$.
$10 \times 40 = 400$. Total steps $< 1600$.
We probably don't need to use "Alter" if the grid is empty and we can just move step-by-step.
Wait, "Move" is allowed. $T \le 2NM$.
If I just move Manhattantly, max distance is $2N = 40$.
Between targets, dist $\le 2N$.
Total targets $M=40$. Steps $\le 40 \times 40 = 1600$.
It fits exactly in budget $2NM = 1600$ if we never use Slides/Alters?
Wait, sliding might reduce steps significantly.
If I'm at top left, target at bottom right.
Move 1 by 1 takes 38 steps.
Slide Right then Down takes fewer steps? No, Slide goes to wall.
If I Slide Right, I hit wall. Then I Slide Down to wall.
I end up at bottom right corner, far beyond target.
Then I need to come back.
This sounds like standard movement optimization where we might overshoot and correct.
But the constraint says "at most $2NM$ actions".
With Manhattan movement, we need $dist(P_0, P_1) + dist(P_1, P_2) + \dots$.
Average distance between random points in 20x20 grid is $E[|X_i - X_j| + |Y_i - Y_j|]$.
$E[|diff|] \approx 1/3 \cdot N \approx 6.6$. Sum is $13$.
$13 \times 39 \approx 507$.
This is well within 1600.
So a simple greedy strategy using 'M' (Move) might be enough?
Wait, "Visit all target squares ... using as few turns as possible."
Score is $M+2NM - T$. Minimize $T$.
Is there a reason to use 'S' (Slide)?
If I am at $(0,0)$ and target is $(1,1)$.
Move R, D takes 2 steps.
Slide R goes to $(19, 0)$. Too far.
If I could stop at $(1,1)$, Slide R would require a block at $(1,0)$? No, block at $(2,0)$?
If I place block at $(2,0)$, then Slide R stops at $(1,0)$? No, starts at $(0,0)$, moves to $(1,0)$, tries $(2,0)$ which has block, stops at $(1,0)$.
If target was $(1,0)$, cost is 2 (1 Alter, 1 Slide). Move R is 1 step. Move is better.
So why use slides?
Maybe to save time when targets are aligned with walls or blocks?
But initially no blocks.
So pure Move strategy costs Manhattan Distance sum.
$M=40$, $N=20$.
Sum of distances can be worst case $38 \times 40 \approx 1520$.
Still less than 1600.
Wait, if distances are large. e.g. snake-like path.
Max total distance is bounded by $2 \times (\text{number of steps to cover grid})$.
If targets are arranged such that sum of distances is close to 1600?
Suppose $P_0 \to P_1$ takes 40 steps. $P_1 \to P_2$ takes 40 steps...
There are 39 segments.
If every segment takes $2N$ steps?
Worst case: Start $(0,0)$. Target $(19, 0)$. Dist 19.
Next $(19, 19)$. Dist 38.
It is possible to accumulate more steps.
But we have a hard cap of $2NM = 1600$.
If pure Move exceeds 1600, we might fail or get lower score.
Wait, "If $m=M-1$, score is $M+2NM-T$".
We definitely want to minimize $T$.
Does "Slide" help?
Consider $N=20$. From $(0,0)$ to $(10,0)$.
Move: 10 steps.
Slide R: lands at $(19,0)$. Then Move L 9 times to $(10,0)$. Total $1 + 9 = 10$. Same.
What about diagonals?
$(0,0) \to (1,1)$.
Move: R, D (2 steps).
Slide: R (to 19), D (to 19). Total 2. Then move back (-18, -18). Much longer.
So Slides seem bad here unless there are blocks.
Why "Alter"?
Perhaps to stop a slide?
If I place a block, I can make a target reachable by Slide.
Example: Target at $(10, 10)$. Current $(0,0)$.
If I place block at $(11, 10)$?
Then Slide R to $(19,0)$, then Slide D?
No, to land exactly on $(10,10)$ via Slide.
Suppose I slide R from $(0,0)$. I hit wall at $(19,0)$.
I want to stop at $(10,10)$.
If I place blocks along the row to control slide?
But Altering takes time (1 step per block).
Moves take 1 step per cell.
Moving 10 cells is 10 steps.
Blocking 1 cell + moving to block is same or worse?
Wait, if I'm at $(0,0)$ and target $(19,0)$.
Slide R: 1 step (hits wall). Stops at $(19,0)$.
Move R 19 times: 19 steps.
Using Slide saves huge amount of time when going to wall!
Also going to corners?
Since initial blocks are 0, Slides are extremely effective for reaching edges/corners.
Strategy: Go towards target. If target is near edge and aligns with movement, Slide.
Otherwise Move.
Since we have plenty of budget (up to 1600), and random permutation might give long paths, we should optimize locally.
For each step $k$ from $1$ to $M-1$:
Current pos $C = P_{k-1}$. Target $T = P_k$.
We want to reach $T$ with min cost (and ensure not visiting other required points, or rather, just ignore others since order matters).
Wait, "If you slide over a target square without stopping on it, it is **not** considered visited."
So we must strictly stop on $T$.
With empty grid, Slide only stops at wall or block.
Since no blocks initially, Slide stops at wall.
So we can only reach $T$ via Slide if $T$ lies on the line from $C$ to wall (in specific direction).
Otherwise, we MUST use Move.
If we use Move, we pay Manhattan distance.
If we use Slide, we pay 1 step (plus potential post-slide correction).
Correction might cost extra.
Actually, if $T$ is on boundary and we can Slide directly to it (e.g. $C$ same row, $T$ closer to wall than $C$, slide away from wall? No, slide goes to wall).
So we can Slide to $T$ only if $T$ is further from $C$ than the wall is?
Wait, slide goes until hit block OR boundary.
Boundary is outside.
So if $C=(0,0)$ and we slide R, we land at $(N-1, 0)$.
If Target $T = (N-1, 0)$, we visit it in 1 step.
If $T = (N-2, 0)$, we pass it, land at $(N-1, 0)$. Not visited.
Can we use Alter?
If we put block at $(N-2, 0)$, then Slide R from $(0,0)$ hits block at $(N-2, 0)$ and stops there?
Wait, if block is at $B$, slide stops AT $B$? Or BEFORE $B$?
"Until you hit a block." Usually means you move ONTO the cell, try to move NEXT.
If next cell has block, you stay on current cell?
Problem says: "continue sliding ... until you hit a block."
Usually in these games (like Sokoban or this AHCS problem), sliding stops on the square BEFORE the block, or ON the block?
"Place a block on the adjacent square...".
Standard interpretation: You cannot enter a square with a block.
So if you slide and encounter a block, you stop before entering it.
So if $C=(0,0)$ and Target $T=(N-2, 0)$. Wall at $N-1$.
Without block, Slide goes to $N-1$. Visits $N-1$. Not $N-2$.
If we place block at $T=(N-2, 0)$, Slide R will stop at $T$ (if coming from left).
Cost: 1 Alter (place block at $T$) + 1 Slide.
Move cost: $N-2$.
For $N=20$, $N-2 = 18$. Save time!
So yes, Alter is useful to stop Slides precisely at targets.
But wait, "It is also allowed to place a block on a current or future target square; however, you must remove the block in order to visit that square."
If we place block on Target $T$ and stop there via Slide, did we visit it?
"You must remove the block in order to visit that square."
If we stop on a square with a block, is it considered visited?
Text: "You may not specify a square outside... It is also allowed to place a block on a current or future target square; however, you must remove the block in order to visit that square."
This implies if you stop on a square with a block (even by Slide), you haven't visited it properly?
"Stop on it after a Slide" counts as visited IF you are on it.
But the text says "remove the block ... to visit".
So, strategy:
1. Place block adjacent to target (or on target?)
Wait, if I place block ON target, I am standing on target?
Start at $C$. Move/Slide to place block?
If I am at $C$, and target is $T$.
Case 1: $T$ is reachable by straight Slide to Wall. $T$ is on the path.
   If I don't alter, I land past $T$.
   If I place block at $T$ (requires being adjacent to $T$), I need to move to $T'$ adjacent to $T$.
   This seems complicated.
   
Case 2: Move directly to $T$.
   Always valid. Cost = Manhattan Distance.
   With $N=20$, max dist is 38.
   Sum of dists for 39 targets $\approx 500-800$.
   Budget 1600.
   Is it possible to exceed 1600?
   $2NM = 1600$.
   If path is very jagged, e.g. zig-zag.
   In $20 \times 20$, max path length without revisiting is bounded.
   But we can revisit.
   However, with random permutation, average distance is small.
   But worst case is possible?
   Wait, random permutation of ALL coordinates.
   Start is random. Targets are random distinct spots.
   Average distance is low.
   Is it possible to construct a case where Manhattan distance > 1600?
   Max dist between any two cells is 38.
   $38 \times 39 \approx 1482$.
   This is close to 1600.
   So pure Move strategy is risky if the path winds.
   But with Slides, we can optimize.
   
Algorithm Idea:
For each target $T_k$:
We are at $P_{k-1}$.
We want to reach $T_k$.
Option 1: Pure Move. Cost $d(P_{k-1}, T_k)$.
Option 2: Use Slide to shorten travel.
   This requires alignment with walls/blocks.
   Or placing blocks.
   Given $N=20$ and limited time, maybe simple heuristic is best.
   Greedy approach:
   Try to reach $T_k$.
   Prioritize actions that get us closer with least cost.
   Actions: Move (R, L, U, D).
   Actions: Slide (R, L, U, D).
   Actions: Alter (R, L, U, D) - only if helpful.
   
Actually, since we have a generous budget (1600 vs ~500-800 avg), we don't strictly *need* Slides unless necessary to fit. But Slides reduce steps.
Wait, "You will obtain ... $M+2NM-T$". We want minimal $T$.
So we SHOULD minimize $T$.
This means Slides ARE beneficial.
And since "randomly shuffle ... assign", there is no hidden structure, just geometry.
With random targets, it's unlikely that many are aligned with rows/cols perfectly to benefit from Slides, BUT we can force alignment or use walls.
Wait, walls are fixed.
Coordinates are uniform $0..19$.
Probability of $T$ having same row as $C$ is $1/20$.
With $N=20$, chance is low.
But maybe we can chain slides?
Actually, with $M=40$ and $N=20$, maybe simple BFS on grid states is too slow?
State: $(r, c, blocks, visited\_mask)$? Blocks count is too high.
Visited mask is $2^{39}$. Impossible.
Must be incremental planning.
Since we need to visit specific points in order, and "passing over" doesn't count.
This simplifies things. We treat previous targets as irrelevant obstacles (we can walk on them now, they are empty).
Wait, we removed blocks previously? "Remove the existing block" is allowed.
Actually, "remove" means setting to empty.
We can maintain blocks to optimize, but clearing them is good.
But since we start empty, why leave blocks?
Only reason: to stop slides.
Given the budget $2NM$, and max possible path $\approx 1482$, we are somewhat safe with pure Moves, but Slides are faster.
However, implementing a full planner for Slides might be complex.
Let's check if we can just use BFS for shortest path from $P_{current}$ to $P_{next}$ considering Slides as special moves.
Grid size $20 \times 20 = 400$.
Moves: 4 directions.
From $(r,c)$, we can go to:
- Neighbors: $(r', c')$ via 'M'.
- End of Slide in 4 directions: $(r'', c'')$ via 'S'.
- (Maybe Alter?) Alter modifies state, so not a direct transition in graph search unless state includes blocks.
But notice: Altering is only needed to Stop a slide early.
Stopping a slide early is equivalent to having a block adjacent.
If we decide to slide from $A$ to $B$, and $B$ is not blocked, we fly past.
Unless we place block.
But we can't place block arbitrarily in BFS without changing state.
However, note that we are allowed $2NM$ steps.
Is it possible to reach ANY target from any other target in reasonable steps?
Yes, Manhattan distance.
Can we optimize?
Since $N$ is small (20), maybe we can precalculate optimal moves between points using Slides.
For any pair $(u, v)$, what is min moves ignoring Alters (since Alters take time to clear)?
Wait, if we use Alter to stop slide, we add 1 step.
Path: Move to adjacent, Place Block, Slide. Cost $\approx 1+1=2$ vs Moving.
If dist > 2, Slide+Bloc is better.
But placing block leaves a block. Do we keep it?
If it helps for subsequent moves, good.
But generally, we might want clean grid to maximize flexibility.
Actually, simplest winning strategy for these contests (AHc) is often to stick to Moves and Slides to walls.
Since $M=40$ is small, we can afford some lookahead.
Greedy Strategy:
At each step, we are at $curr$, target is $target$.
Calculate distance if we Move: $D_{move} = |r_u - r_v| + |c_u - c_v|$.
Calculate distance if we Slide towards wall?
If we slide towards wall, we land at wall. If wall is $v$ or $w$, cost 1 + backtrack.
Backtracking is Move.
Wait, if we slide from $u$ to $wall$, we pass $target$.
If we don't stop, we miss it.
If we stop, we must have placed block.
So:
1. Try to reach $target$ by Move. (Always works). Cost $d$.
2. Check if we can reach $target$ by Slide.
   Condition: $u$ and $v$ share row or col. AND $v$ is between $u$ and wall? NO.
   Slide goes to wall. So $v$ must be "behind" wall relative to $u$?
   If $u$ at 0, wall at 19. Slide goes to 19.
   If $v$ at 19. Slide works.
   If $v$ at 10. Slide passes.
   To land on $v$ via Slide, we need block at $v+1$.
   So we can place block at $v+1$.
   Sequence: Move to $v+1$? No, we are at $u$.
   Move to neighbor of $v$ (say $v+1$) is not efficient if we are far.
   Actually, better: Move towards $v$ to get adjacent, then place block? No, block stops slide *incoming*.
   If we slide FROM $u$ towards $v$.
   We need block at $pos$ such that Slide(u, dir) stops at $v$.
   This requires block at $v$ (impossible to visit) or block at $next(v)$?
   Wait, if block is at $next(v)$, Slide stops at $v$.
   But we must visit $v$ (empty). So we must remove block later.
   So Plan: Slide(u, dir) -> stop at $v$. (Requires block at $next(v)$).
   Cost: 1 Slide (if aligned) + 1 Alter (place block) + Visit (move out?).
   Actually, "stop at $v$" means we land on $v$.
   Does landing on $v$ count as visited? Yes.
   Do we need to clear block immediately?
   "you must remove the block in order to visit that square."
   This phrasing is tricky.
   "place a block on ... adjacent ... if it does not already contain one; otherwise, remove."
   "allowed to place ... on a current or future target square; however, you must remove the block in order to visit that square."
   This usually implies: You CANNOT step onto a target square if it has a block.
   Wait, "Slide ... until you hit a block".
   If you hit a block, you STOP before it.
   If $v$ has a block, Slide cannot reach $v$ (you stop at $v-1$).
   Unless $v$ IS the stop location?
   If I am at $u$, looking at $v$, and there is a block at $v+1$. I slide, pass $u$, $v$, hit block at $v+1$. I stop at $v$.
   I am ON $v$. $v$ is empty (block is at $v+1$).
   So I visited $v$.
   Later, if I need to go to $v$ again, block must be removed.
   But here we just visit.
   So, to use Slide to visit $v$:
   1. $u$ and $v$ must be aligned.
   2. $v$ must be between $u$ and some obstruction (Wall or existing Block).
      If no existing block, we must Create an Obstruction at $v+1$ (relative to motion).
      This requires us to be able to place a block at $v+1$.
      Placement requires moving adjacent to $v+1$.
      If $v+1$ is adjacent to $u$? No.
      If $v+1$ is somewhere far, we have to Move there.
      
   Actually, simpler:
   Just use Move. If Move is too long, use Slides.
   Since $N=20$ and random coords, average Move dist is small.
   $2NM$ budget is tight but generous enough for random paths.
   Wait, worst case random permutation could produce sum of distances $> 1600$?
   Max dist is 38. $39 \times 38 = 1482$.
   So pure Move is safe in terms of budget?
   Yes! $1482 < 1600$.
   Wait, is it really $M=40$ targets?
   Input: $N \ M$, then $M$ lines.
   Indices $0$ to $M-1$.
   Start at $0$. Visit $1 \dots M-1$.
   Number of transitions = $M-1 = 39$.
   Max dist $\approx 38$.
   Max total steps $\approx 1482$.
   Limit $2NM = 2 \times 20 \times 40 = 1600$.
   $1482 < 1600$.
   So Pure Move (Manhattan) is guaranteed to succeed without exceeding limit?
   What if path overlaps? Doesn't matter, budget is on actions.
   Overlapping path is fine.
   So, simple Manhattan Move strategy might work perfectly and yield decent score?
   Score: $M + 2NM - T$.
   If we use $T \approx 1482$, score $\approx 40 + 1600 - 1482 \approx 158$.
   If we can improve $T$ significantly, score increases.
   Can we reduce $T$?
   Yes, using Slides.
   For example, going from $(0,0)$ to $(19,0)$ takes 19 Moves.
   Using Slide takes 1 Action.
   Huge gain.
   Even if targets are random, probability that a target shares a row/col with current is roughly $1/20$.
   But we can choose direction.
   If $P_{prev} = (0,0)$ and $P_{curr} = (1, 1)$.
   Manhattan: R, D (2).
   Slide R: Lands $(19, 0)$. Then Move L 18, D 1. Total $1 + 18 + 1 = 20$. Worse.
   So Slides are only good if aligned with boundaries or existing blocks.
   With no blocks, Slides are only good if aligned with Wall.
   Since $N=20$ is small, maybe we can use the fact that we can place blocks to create shortcuts.
   But placing blocks takes time.
   If we place a block at $(19,0)$ to allow Slide to stop at $(19,0)$ from $(0,0)$?
   We are at $(0,0)$. Target $(19,0)$.
   Slide R stops at $(19,0)$ automatically because Wall is at 20 (outside) and boundary acts as wall?
   Wait, boundary is "covered with blocks".
   So Slide R from $(0,0)$ goes to $(19,0)$.
   So for ANY target on boundary, Slide R/L/U/D takes 1 step (aligned).
   This is much faster than 19 Moves.
   For internal targets?
   Say Target $(10, 10)$.
   Path: Slide R to $(19, 0)$, Slide D to $(19, 10)$. Now at $(19, 10)$.
   Need to get to $(10, 10)$. Move U 9 times.
   Total: 2 (Slides) + 9 (Moves) = 11 steps.
   Vs 20 moves.
   Savings!
   So "Corner routing" is better.
   Go to corner, traverse, exit?
   Generally, for any target $(r, c)$, we can reach it from $(r_c, c_c)$ by:
   Option A: Direct Moves. Cost $|r_r - r_c| + |c_c - c_c|$.
   Option B: Via Corner $(0,0)$ or others?
     E.g. $P_{old} \to (0,0)$ (slides/moves) $\to P_{new}$.
     Cost is high if $P_{old}$ far from corner.
     
   Best heuristic:
   For each step $k$:
   Find shortest path to $P_k$ given we are at $P_{k-1}$ and grid is empty.
   Since $N$ is small, run BFS on $20 \times 20$ grid.
   But BFS state must include whether we used Slides optimally.
   However, since we are in "empty grid" mode (except for our placements), and we clear everything ideally.
   Wait, do we need to maintain state of blocks?
   Yes. But if we don't place blocks, state is Empty.
   Is it ever optimal to keep a block?
   Only if we use it for subsequent targets.
   But removing takes steps. Keeping takes 0 steps (but restricts movement).
   Given the budget is tight-ish, keeping blocks might be risky if they impede.
   However, with "Corner Routing" (Slides to wall), we don't need blocks.
   So, Hypothesis: Optimal strategy for Empty Grid involves Slides to Walls.
   We can just perform BFS where:
   Edge $u \to v$ exists if:
     1. $v$ is adjacent to $u$ (Move).
     2. $v$ is on ray $u \to \text{Wall}$ (Slide). Note: This ends at Wall intersection.
        Wait, Slide stops at first obstacle.
        If grid is empty, Slide stops at Wall (boundary).
        If we place block, it stops there.
        Assuming we don't use Alter (costly), we just consider Slides to Walls.
        
   Wait, if we can reach the target $T$ by Slide to Wall and then Walk back?
   Or multiple Slides.
   This looks like finding shortest path on a graph where nodes are cells.
   Edges:
   - Move neighbors: Weight 1.
   - Slide: Weight 1 (goes to wall intersection).
   
   Wait, Slide weight is 1.
   Move weight is 1.
   If we model Slide as moving to specific "Wall Cell" in 1 step.
   Then BFS finds shortest path in weighted graph (all weights 1).
   This BFS would give a plan.
   After we execute the plan (sequence of Ms and Ss), we arrive at $T$.
   But wait, once we use Slide, we are at a specific cell (corner-ish).
   Can we Slide in intermediate direction?
   Yes, from any cell.
   So graph nodes are $(r,c)$.
   Transitions:
   - $M(r,c) \to$ Adjacent (4 neighbors). Cost 1.
   - $S(r,c) \to$ End-of-Slide(r,c, Dir).
     Where Dir $\in \{U,D,L,R\}$.
     End-of-Slide is determined by scanning ray.
     Cost 1.
     
   Is this sufficient?
   This assumes we never place blocks (Weight 0).
   Placing a block costs 1 action and adds a node-edge.
   Might be worth adding to BFS?
   Probably not worth complexity. Random targets likely don't benefit from static blocks.
   Also, we need to output the ACTUAL sequence of actions.
   BFS will give us path.
   Does BFS guarantee validity?
   Yes, if we simulate moves correctly.
   Wait, if we Slide R from $(0,0)$, we land at $(19,0)$.
   This is valid move.
   Then from $(19,0)$ we Slide D $\to (19,19)$. Valid.
   This covers the "L-shape" strategy.
   
   One detail: "Visit ... in order".
   So simply solve for $P_1$ given $P_0$, then $P_2$ given $P_1$, etc.
   Since we need to visit $P_1$ exactly, we stop there.
   Do we need to worry about "visited" status of future targets?
   "Even if you pass over a later target ... not considered visited."
   So just find shortest path $P_{k-1} \to P_k$ minimizing steps.
   Wait, passing over $P_{future}$ is fine, doesn't hurt.
   So Independent subproblems!
   
   Wait, what about $P_0$?
   Input gives $M$ coordinates. $P_0$ is start. $P_1 \dots P_{M-1}$ are targets.
   Wait, does the list include start?
   Input: "$N \ M$, $i_0 \ j_0$, ... $i_{M-1} \ j_{M-1}$".
   Yes, $P_0$ is given.
   Wait, problem says "visit target squares $(i_1, j_1) \dots$".
   Does it mean $P_0$ is NOT a target?
   "Initially ... no blocks."
   Usually "visit" means stepping on.
   If we start at $P_0$, do we need to "visit" it?
   Usually not.
   But maybe $P_0$ is just a start point.
   Okay, so loop $k$ from 1 to $M-1$.
   Find path $P_{k-1} \to P_k$.
   Concatenate.
   
   Algorithm:
   Function `solve(start_x, start_y, target_x, target_y, current_blocks)`:
     BFS from $(start_x, start_y)$.
     State: $(x, y)$.
     Queue stores `(x, y, path_string)`.
     Visited array `[20][20]` stores min steps or visited boolean.
     Wait, if we use BFS, we find shortest path in terms of #actions.
     Since all actions cost 1, unweighted BFS works.
     Transitions from $(x,y)$:
       1. `M U`, `M D`, `M L`, `M R` -> neighbor cells.
       2. `S U`, `S D`, `S L`, `S R` -> compute destination based on ray.
          Destination calculation: Scan from $(x,y)$ in direction until hitting bounds or block.
          If bounds -> last valid cell.
          If block -> last valid cell before block? Or ON block?
          "Continue sliding ... until you hit a block."
          "You cannot move into a square containing a block."
          So you stop on the cell *before* the block.
          
     Wait, BFS might explore many states. $400$ nodes. Each node has 4 Moves + 4 Slides.
     Wait, Slides are not edges in the sense of moving to arbitrary cell. They move to specific cells.
     So degree is at most 8.
     BFS is fast. $400 \times 8$ operations.
     
     Wait, do we need to handle "Alter"?
     Maybe not. If BFS finds solution with only M/S, great.
     If BFS fails? BFS always finds shortest path on graph. Graph is fully connected (via Moves).
     So solution always exists.
     
     What about optimizing path using blocks?
     Maybe BFS with block placement is too deep.
     With $N=20$, pure Move/Slide should be fast enough.
     
     Wait, checking the limit.
     Is it possible that for some configurations, pure M/S path is too long?
     Consider start $(0,0)$, target $(10, 10)$.
     Path using Slides:
     $(0,0) \xrightarrow{S R} (19,0) \xrightarrow{S D} (19,19) \xrightarrow{S L} (10,19) \xrightarrow{S U} (10,10)$?
     No, $(10,19) \xrightarrow{S U} (10,0)$. Misses $(10,10)$.
     Wait, Slide goes to wall.
     So to land on $(10,10)$, we need a block or wall.
     Without blocks, we can only land on walls/boundary.
     So to reach $(10,10)$ exactly without blocks, we must use Moves to adjust.
     E.g. Get close to $(10,10)$ and Move into it.
     Or Slide to $(10, 0)$ (row 10, col 0) then Move R to 10.
     Can we Slide to Row 10?
     From $(0,0)$. Slide R $\to (19,0)$. Slide U $\to (0,0)$.
     Basically, Slides snap us to boundaries.
     Moves bring us to exact coords.
     So Strategy:
     1. Use Slides to get "close" to target (minimize distance component-wise to target, or move to a boundary row/col matching target).
     2. Use Moves to finish.
     
     Wait, if I use Slide to $(19, 0)$. I am on row 19.
     To get to $(10, 10)$, I need to change row.
     So from $(19,0)$, Slide D? No, wall is down.
     Slide U $\to (0,0)$.
     I can Move R to $(19, 10)$.
     Then Slide U $\to (0, 10)$.
     Then Move D to $(10, 10)$.
     Path:
     S R (to 19,0)
     R x 19 (to 19,10) -> 19 moves.
     S U (to 0,10)
     D x 10 (to 10,10).
     Total: $1 + 19 + 1 + 10 = 31$ steps.
     Direct Move: 20.
     Direct Move is better.
     Slides are slower if they require many Moves afterwards.
     But if target is $(19, 10)$.
     S R (19,0) -> S D (no, 19,0 is max y).
     From $(0,0)$, S R $\to (19,0)$.
     Then M R 10 $\to (19,10)$.
     Total 11.
     Move: 19 + 0 = 19.
     Here Slide saves 8 steps.
     So we should prefer Slide if it reduces total Moves needed.
     
     Generalized approach:
     Since $N$ is small, we can actually simulate the grid.
     We need to implement the BFS/Search.
     However, since we know we want to minimize Total Actions, and we have a sequence of independent tasks ($P_{k-1} \to P_k$).
     We can just write a function `find_path(u, v)` that returns string of commands.
     
     Wait, we need to output the code to solve this.
     The "score" rewards minimizing $T$.
     So `find_path` MUST return shortest path.
     BFS works for unweighted graphs.
     Is the state graph unweighted? Yes, each action is 1 turn.
     So `std::queue` BFS is perfect.
     
     Implementation details:
     `int dr[] = {-1, 1, 0, 0}; int dc[] = {0, 0, -1, 1};`
     `dirs = {'U', 'D', 'L', 'R'}`
     BFS state: `pair<int,int>`
     Map `dist[20][20]` initialized to infinity.
     Queue `q`.
     Push start. `dist[start] = 0`.
     While q not empty:
       pop `curr`.
       if `curr == target`: return reconstruct(path).
       // Moves
       for d in 0..3:
         nxt = curr + dr[d]dc[d]
         if valid(nxt) && !has_block(nxt):
           if dist[nxt] > dist[curr] + 1:
             dist[nxt] = dist[curr] + 1
             parent[nxt] = ...
             q.push(nxt)
             
       // Slides
       for d in 0..3:
         // Simulate slide
         nx = curr.x, ny = curr.y
         while true:
           nx += dr[d], ny += dc[d]
           if invalid(nx, ny) || has_block(nx, ny): break
         // landed at prev valid pos
         // The problem says: "hit a block ... until you hit a block".
         // Wait, if I start at (0,0) and slide R.
         // I check (1,0). If ok, I move there.
         // Loop continues until block/boundary.
         // The resulting cell is the last valid one.
         // BUT wait, is the result cell valid?
         // "All squares outside ... covered with blocks".
         // So (20, 0) is blocked.
         // So loop breaks when trying to enter (20,0).
         // Last valid cell is (19,0).
         
         // Wait, what if block is at (1,0)?
         // Start (0,0). Look at (1,0). Blocked. Break immediately?
         // If blocked, I don't move?
         // "Move: Move one square ... Cannot move into ... block".
         // "Slide: Continue sliding ... until you hit a block."
         // If immediately blocked, Slide length 0? Or invalid?
         // Usually implies no displacement.
         // But effectively useless.
         
         if valid(next_pos) && !has_block(next_pos) [Wait, next_pos is result]:
            // Is result valid? Yes, it's a free cell.
            // Check if we improved dist[next].
            if dist[next] > dist[curr] + 1:
              ...
              
     Complexity: $20 \times 20 = 400$ states. 4 Moves, 4 Slides per state.
     Total edges $\approx 3200$. BFS is instant.
     
     Constraints:
     $N=20$.
     Time limit? Standard competitive programming (usually 2s).
     $M=40$.
     $40 \times \text{Fast BFS} \ll 2s$.
     
     Wait, what about "Alter"?
     If pure M/S is optimal, we don't need it.
     Could M/S be suboptimal?
     Suppose target $(10, 10)$. Current $(0,0)$.
     We need to stop at $(10, 10)$.
     As seen, without blocks, we can't "Stop" at $(10, 10)$ using Slide.
     Because Slide snaps to Wall/Block.
     So if we use Slide, we MUST land on $(10, 10)$ exactly.
     This means $(10, 10)$ must be a Wall/Block or we stop before it.
     Since $(10, 10)$ is empty, we can only stop there if:
     1. It IS the boundary (unlikely for general random points).
     2. There is a Block adjacent to it.
     
     If we rely purely on M/S, we can reach $(10, 10)$ by Moves.
     Or by Slides to nearby walls + Moves.
     Example: Reach $(19, 10)$ via Slide D (from 19,0? No, from 0,0 slide R then M R...).
     Basically, BFS naturally handles "Slide to Wall" as a move.
     Does it handle "Slide to stop at $(10,10)$" using a phantom block?
     If BFS treats Slide as "Jump to Wall", then it misses the possibility of "Create Block and Jump".
     But creating a block takes 1 step.
     Is it possible that "Create Block" makes path shorter?
     Example: $(0,0) \to (10,0)$.
     Path 1: Moves. Cost 10.
     Path 2: Place block at $(11,0)$ (if reachable), Slide.
     Wait, to place block at $(11,0)$, we need to be at $(10,0)$ or $(11,0)$?
     "Place a block on the adjacent square ... if it does not already contain one".
     To place block at $(11,0)$, we need to be at $(11,0)$ or $(10,0)$ (adjacent).
     Wait, $(11,0)$ is not adjacent to $(10,0)$? Yes it is (if N>11).
     Wait, if I am at $(0,0)$.
     I need block at $(11,0)$.
     I must move to $(10,0)$ (cost 10) then Alter (cost 1).
     Then Slide (cost 1).
     Total $10+1+1 = 12$.
     Move was 10.
     Worse.
     So Alter is rarely beneficial for a single leg if target is accessible by Move.
     
     What about long distance?
     $(0,0) \to (19, 19)$.
     Move: 38.
     Slide R $\to (19,0)$. Slide D $\to (19,19)$. Cost 2.
     BFS will find this.
     
     Are there cases where "Alter" helps avoid long Moves?
     Suppose target is $(1, 1)$.
     Moves: R, D. Cost 2.
     Can we do better? No.
     
     Suppose $(0,0) \to (10, 10)$.
     Moves 20.
     Can we do better with blocks?
     We need to stop at $(10, 10)$.
     If we put block at $(10, 11)$, we Slide R from $(10, 0)$?
     Wait, to Slide R, we need to be at $(10, 0)$.
     Path: Slide D to $(10,0)$ (if possible? No, from 0,0 Slide D -> (19,0)).
     From $(0,0)$ Slide R $\to (19,0)$.
     Then Move U $\to (10,0)$?
     Then M R $\to (10,10)$.
     Moves involved: $1 + 9 + 10 = 20$.
     Same.
     
     So it seems pure M/S BFS is sufficient and robust.
     Wait, "Pass over later targets".
     My assumption: independent legs.
     Is it possible that visiting $P_k$ takes me through $P_{k+1}$ and it's counted?
     No, problem says "it is not considered visited".
     So independent.
     
     So the plan:
     Read N, M.
     Read $P_0 \dots P_{M-1}$.
     Loop $k = 1 \dots M-1$:
       Solve BFS from $P_{k-1}$ to $P_k$.
       Output path.
     Concatenate all outputs.
     
     Wait, "You must visit the target squares ... in the given order."
     So sequence of actions must achieve $P_0 \to P_1 \to \dots \to P_{M-1}$.
     Correct.
     
     Double check: "You will need to visit it again when its turn ... arrives."
     Yes.
     
     One edge case: "If you slide over a target square ... not considered visited."
     This is consistent with independent legs logic.
     
     Is there any global benefit to blocking?
     Maybe if a block persists helps future moves?
     But we don't know future moves?
     Actually, since we process sequentially, we know all future targets.
     However, maintaining a block takes 1 Action (Alter) to place and 1 Action (Alter) to remove (or Move past it).
     Placing block takes 1 action.
     If we can save $> 1$ action in the future by leaving a block?
     Example: Target $T_1$ at $(0, 10)$, Target $T_2$ at $(10, 0)$.
     Currently at $(0,0)$.
     Leg 1: $(0,0) \to (0,10)$.
     Option A: Move R 10. (Cost 10).
     Option B: Block?
     If we leave block at $(0, 11)$ (to slide R to 10?), no.
     Maybe block at $(0, 9)$?
     If we block at $(0, 9)$, we can slide R $\to (0,8)$? No, blocked at 9. Stop at 8.
     If we need to be at $(0, 10)$.
     If we place block at $(0, 11)$. We slide R from $(0,0)$? No, we need to be at $(0,0)$ looking R.
     Block at 11 means we pass 0..10. Stop at 10.
     So we land on $(0, 10)$!
     Wait.
     If I place block at $(0, 11)$.
     Then Slide R.
     Path: $(0,0) \to \dots \to (0, 10)$.
     Cost: 1 (Slide).
     Wait, to place block at $(0, 11)$, I must be adjacent.
     How to get to $(0, 11)$? Move R 11 times?
     Cost 11 + 1 (Alter) + 1 (Slide).
     Total 13.
     Move was 10.
     Not worth it.
     
     What if I am already at $(0, 10)$? (Start was there).
     Next target $T_2$ at $(0, 11)$.
     If I need to go $(0, 10) \to (0, 11)$.
     Move: 1.
     Slide: Requires block at 12.
     From 10, go to 12 (2 steps), Place (1), Slide (1). Total 4.
     Move 1 is better.
     
     It seems placing blocks is never worth it for $N=20, M=40$.
     The budget is sufficient.
     
     So, final algorithm:
     Run BFS for each segment.
     Use `priority_queue` or `queue`? Unweighted BFS needs `queue`.
     Reconstruct path using parent pointers.
     
     Wait, BFS finds shortest path in *steps*?
     Yes.
     Wait, what if `S` is preferred over `M`?
     Both cost 1.
     BFS will find `S` if it leads to target.
     However, my simulation of `S` in BFS:
     `while(true)` logic.
     This correctly identifies the destination of Slide.
     
     Let's refine Slide simulation.
     `Slide(r, c, dir)`:
       `tr = r, tc = c`
       Loop:
         `nr = tr + dr[dir]`
         `nc = tc + dc[dir]`
         If `valid(nr, nc)` AND `grid[nr][nc] == EMPTY`:
            `tr = nr`, `tc = nc`
         Else:
            Break (Hit boundary or block).
       Return `(tr, tc)`
     
     Wait, "valid" includes boundaries $0 \dots N-1$.
     "Blocks" are obstacles.
     Initially grid is empty.
     But wait, "All squares outside the $N \times N$ area are covered with blocks".
     So `valid` checks $0 \le n < N$.
     If we hit boundary, we stop at $N-1$ (if moving outward)?
     If moving Outward from $N-1$, we check $N$ (invalid). Stop at $N-1$.
     So logic holds.
     
     One catch: "alter" is available.
     Since I decided not to use it, I'll ignore it.
     Wait, what if the judge expects a specific behavior regarding "Alter"?
     Problem text: "Scoring ... $m=M-1$".
     The score formula penalizes $T$.
     Minimizing $T$ is key.
     If "Alter" provides a shortcut, I should use it.
     But given analysis, it seems "Alter" costs too much overhead to offset.
     However, I should verify if there's ANY case where $T$ reduction by "Alter" is significant.
     Suppose $N=100$. Then Slide is very powerful.
     With $N=20$, Manhattan is cheap.
     Also "slide ... until you hit a block".
     If we have a block at $X$, Slide passes $X$? No, stops before.
     This allows precise stopping.
     But again, cost of setup.
     Maybe if $N$ was larger, or targets were dense.
     With $N=20$ and $M=40$, simple path is likely optimal.
     
     Wait, one thing.
     "Output format: $a_t \ d_t$".
     Example:
     M R
     S D
     
     Code structure:
     Includes, main.
     BFS function taking start, end, N, grid.
     Main reads input, loops targets, calls BFS, prints path.
     
     Resetting grid:
     Since no blocks are kept, `grid` is always all zeros (EMPTY).
     
     Corner cases:
     $M=1$?
     Input says $M$ lines. $N \ M$.
     If $M=1$? No targets to visit.
     Loop $1 \dots 0$ won't execute. Output empty.
     Is $M \ge 2$?
     Input says "take first M coordinates".
     Usually $M \ge 1$.
     "visit target squares ... $(i_1, \dots)$".
     If $M=1$, loop is empty. Output nothing.
     Sample says "Show example".
     Sample might have $M>1$.
     If sample has $M=40$, loop runs 39 times.
     
     Wait, what about `M=1`?
     Problem statement: "initial position $(i_0, j_0)$ ... visit $(i_1) \dots (i_{M-1})$".
     If $M=1$, range is empty.
     But input format says $i_0 \dots i_{M-1}$.
     If $M=1$, only $i_0$ given.
     Output empty string.
     Correct.
     
     What about time complexity?
     40 iterations of BFS.
     BFS visits 400 states.
     Fast.
     
     Memory:
     Arrays of size 20x20. Negligible.
     
     Wait, check constraints again.
     $N=20, M=40$.
     Budget $2NM = 1600$.
     Avg moves $\approx 500$.
     Safe.
     
     One minor optimization:
     In BFS, we can store `path` directly in queue?
     String concatenation might be slow/memory heavy?
     $1600$ chars is small.
     But pushing strings in queue is fine.
     Better: Store `parent` pointers (direction taken) and reconstruct at end.
     `struct Node { int x, y, id; }`
     `vector<Node> parent_map(20, 20)` -> need to distinguish directions.
     `map<pair<int,int>, int> dist` or `int dist[20][20]`.
     `int from_dir[20][20]`?
     Moves: $d \in \{0..7\}$. $0..3$ Move, $4..7$ Slide.
     If multiple ways to reach $(x,y)$, BFS ensures first is shortest.
     Since unweighted, first visit is optimal.
     Wait, different paths might have same length. Any is fine.
     Reconstruct: trace back from target to start.
     
     Let's verify `dir` values.
     0: M U
     1: M D
     2: M L
     3: M R
     4: S U
     5: S D
     6: S L
     7: S R
     
     Directions arrays.
     
     Implementation details:
     `vector<string> path_list`
     Inside loop, `string path = bfs()`. `cout << path << endl` (space separated? No, newlines).
     Output format:
     $a_t \ d_t$
     Per line.
     
     Example trace:
     Input:
     4 3
     0 0
     0 1
     1 0
     Start 0,0. Target 0,1.
     BFS finds: M R.
     Path:
     M R
     End at (0,1). Mark visited.
     Next target 1,0.
     BFS from (0,1) to (1,0).
     Moves: M D, M L.
     Path:
     M D
     M L
     Total:
     M R
     M D
     M L
     Total steps 3.
     Score $M + 2NM - T = 3 + 24 - 3 = 24$. (Assuming M=3 targets? No M=3 means 3 coords, 2 targets? No, M is input M.
     Wait, problem says "m be number of target squares successfully visited."
     If we visit all $M-1$, score $M + 2NM - T$.
     With $M=40$, $2NM = 1600$.
     
     Wait, the example link shows:
     Input:
     15 40
     (some coords)
     (Sample input usually matches constraints).
     
     Let's double check $M$ meaning in input vs problem description.
     Problem: "Input ... $i_0 \dots i_{M-1}$".
     Scoring: "$M-1$ targets".
     This confirms $M$ points are given, start is first, targets are rest.
     
     Wait, one edge case.
     If $N$ is small, say $N=1$.
     Coordinates $(0,0)$. Start $(0,0)$.
     Targets?
     "coordinates distinct".
     So $M$ distinct points. $N \times N$ max points.
     If $N=20$, $400$ points. $M=40$ is valid subset.
     
     One detail:
     "If you slide over a target square without stopping on it, it is **not** considered visited."
     This implies we can safely pass over $P_{k+1}$ while targeting $P_k$.
     It's just ignored.
     So my logic holds.
     
     Final check on "Alter".
     Suppose $P_{k-1} = (0,0)$. $P_k = (0, 10)$.
     If I place a block at $(0, 9)$ (requires getting there).
     From $(0,0)$, I can Slide R to $(19,0)$.
     Move U 0.
     Wait, if I place block at $(19, 10)$ (to stop at $(19, 9)$?).
     Actually, if I place block at $(0, 9)$, I can Slide R from $(0, 8)$? No.
     To use block at $(0, 9)$ to stop Slide R: I must be at $(0, 0)$, aim R.
     But Slide R goes all the way to Wall (19, 0) if nothing blocks it.
     To stop at $(0, 9)$, I need block at $(0, 10)$.
     Wait, if I have block at $(0, 10)$.
     Slide R from $(0,0)$ will pass 0..9. Stop at 9?
     "Continues sliding ... until you hit a block".
     So if block is at $(0, 10)$, I am at $(0, 9)$.
     I cannot be at $(0, 10)$ to place block if I want to stop at $(0, 9)$?
     Wait, target is $(0, 9)$.
     I need to stop at $(0, 9)$.
     I need block at $(0, 10)$ or Wall at $(0, 10)$.
     Wall is at 20.
     If block at 10.
     Path: Move to $(0, 0)$ (already there). Aim R.
     Result: $(0, 9)$.
     Cost: 1 (Slide).
     But to create block at $(0, 10)$, I need to be adjacent.
     I am at $(0,0)$.
     I need to place block at $(0, 10)$.
     I need to move to $(0, 9)$ or $(0, 11)$?
     Wait, "Place a block on the adjacent square ...".
     To place at $(0, 10)$, I need to be at $(0, 9)$ or $(0, 11)$ or $(1, 10)$ or $(-1, 10)$?
     Wait, "specified direction".
     "Choose one of four cardinal directions".
     If I want to place block at $(0, 10)$, I can issue command `A R` from $(0, 9)$.
     Command `A R` from $(0, 0)$ affects $(0, 1)$.
     So to place block at $(0, 10)$, I need to issue `A R` from $(0, 9)$.
     This implies I must traverse $(0, 0) \to (0, 9)$ to place block.
     Cost: 9 moves. Place block (1). Slide (1).
     Total $9+1+1 = 11$.
     Move was 9.
     So `A` is worse.
     
     What if I want to stop Slide at $(0, 9)$ but coming from Left?
     I must be at $(0, 0)$.
     If I use Move R 9 times, I am at $(0, 9)$.
     I visited it.
     I don't need Slide.
     
     What if I want to Slide to reach $(0, 9)$ from $(0, -10)$?
     Implies crossing boundaries?
     
     Conclusion: `A` is likely redundant given budget.
     The only reason to use `A` is to enable sliding where `M` is too long.
     But `M` is never "too long" in terms of budget, only in terms of efficiency.
     And sliding efficiency (saving $>1$ steps) seems to require `A` only if we can place block easily.
     But placing block requires traversal.
     Traversal cost cancels out saving.
     Unless we reuse the block?
     If we have block at $(0, 10)$, we can Slide R from $(0,0)$ to $(0,9)$ (Cost 1).
     Then Move R to $(0,10)$ (needs remove block or jump? No, move onto target with block is impossible).
     If we stop at $(0,9)$ via Slide. Target reached.
     Next target $(0, 10)$.
     We are at $(0, 9)$.
     We need to visit $(0, 10)$.
     Block is at $(0, 10)$.
     Move R is blocked.
     We must Remove block (A R).
     Then Move R.
     Total for leg 1: Slide (1).
     Leg 2: Remove (1), Move (1).
     Total combined: 3 steps for $(0,0) \to (0,10)$.
     Wait, if we didn't use block:
     $(0,0) \to (0,10)$ is Move R (10 steps).
     Wait, my comparison above was wrong.
     Case 1: Pure Move. $(0,0) \to (0,10)$: 10 steps.
     Case 2: Place Block.
       1. Traverse to $(0,9)$ (9 steps).
       2. Place Block at $(0,10)$ (1 step).
       3. Slide R (1 step) -> lands at $(0,9)$.
       4. Target is $(0,10)$?
       Wait, if target is $(0,9)$. We reached it.
       If target is $(0,10)$. We placed block at $(0,10)$. We are at $(0,9)$.
       We cannot move to $(0,10)$ (blocked).
       We must Remove Block (1 step).
       Then Move R (1 step).
       Total: $9 + 1 + 1 + 1 + 1 = 13$.
       Worse.
       
       What if Target was $(0,19)$?
       Move: 19.
       With block:
       Traverse to $(0, 18)$ (18 steps).
       Place block at $(0, 19)$ (1 step).
       Slide R?
       If block at $(0, 19)$, Slide R from $(0,0)$ stops at $(0, 18)$?
       No, Slide R goes to Wall.
       We need block at $(0, 20)$? Outside.
       Or block at $(0, 20)$ is wall.
       So we don't need block at 19 to stop at 18.
       We need block at 19 to stop at 18.
       Wait, to stop at $(0, 19)$, we need block at $(0, 20)$ (impossible) or Wall.
       If Wall is at 20. Slide R reaches 19.
       So Target $(0, 19)$ reachable by 1 Slide!
       Wait. Wall is "outside".
       So Slide R from $(0,0)$ travels $0 \to 1 \dots \to 19$.
       Checks $(20, 0)$. Blocked.
       Stops at $(19, 0)$.
       So yes, sliding to corner/edge costs 1.
       Target $(0, 19)$ is on edge.
       So for edge targets, Slide is optimal.
       For internal targets $(10, 10)$.
       Slide R $\to (19, 0)$.
       Then M D 10 $\to (19, 10)$.
       Then M L 9 $\to (10, 10)$.
       Total $1 + 10 + 9 = 20$.
       Move is 20.
       Equivalent.
       
       So it seems BFS covering Slides and Moves is optimal.
       
     Final Logic:
     Use BFS to find path for each target $k=1..M-1$.
     The BFS uses `M` and `S` moves.
     `S` move is simulated by finding the furthest cell in direction.
     `grid` is all empty.
     Since we need to output actions, we store path.
     
     Wait, `queue` stores `pair<int, int>` is enough for BFS, but we need to track path.
     Using `parent[x][y]` storing `{prev_x, prev_y, direction}`.
     `char action[20][20]`? No, just char for action.
     `pair<int, int> p_parent[20][20]` stores coordinates.
     `char dir_type[20][20]` stores 'M'/'S' and direction.
     Actually, direction is encoded in `dir`.
     
     Wait, we have 8 directions.
     Store `byte parent_dir[20][20]`.
     0..3 Move UDLR.
     4..7 Slide UDLR.
     Also need `dist[20][20]` to mark visited.
     
     Since memory is reset for every leg, declare arrays inside `solve` function or use vectors.
     Vectors `vector<vector<int>> visited(N, vector<int>(N, -1))`.
     `vector<vector<pair<int,int>>> parent(N, vector<pair<int,int>>(N, {-1, -1}))`.
     `vector<vector<char>> move_type(N, vector<char>(N, 0))`.
     
     BFS initialization: `visited[start.r][start.c] = 0`.
     `Q.push({start.r, start.c})`.
     
     Reconstruction:
     While `curr != start`:
       `p = parent[curr.r][curr.c]`.
       `type = move_type[curr.r][curr.c]`.
       `dir_idx = type - type_offset`? No, encode better.
       Maybe `struct State { int r, c, pr, pc, d; }`.
       `vector<State> parent[20][20]`.
       Wait, vector inside array is heavy?
       `vector<State> q` (store State) but we need history.
       Standard BFS: `dist` array, `from` array (store index in queue or direction).
       Let's just store `path` in queue?
       `queue<string> q`?
       No, copying string is inefficient (though small).
       Better: `queue<pair<int, int>> q`.
       `vector<char> parent_action[N][N]`. Stores 'M' or 'S'.
       `vector<char> parent_dir[N][N]`. Stores direction char.
       `int dist[N][N]`.
       `int dist_init[r][c]`.
       Wait, `dist` is enough to check visited.
       Parent info stored separately.
       When expanding `u`, for each action leading to `v`:
         `if (dist[v] == INF)`
           `dist[v] = dist[u] + 1`
           `parent_action[v] = action`
           `parent_dir[v] = dir`
           `q.push(v)`
     
     This works perfectly.
     
     Wait, multiple actions might lead to `v`?
     BFS guarantees shortest path found first.
     So overwrite is not needed, just `if (dist[v] == -1)`.
     
     One subtle point:
     Is it possible `S` direction from `u` leads to `v` which was visited by `M` from another `u'`?
     BFS processes layer by layer.
     If we reach `v` via `S`, and `v` was visited via `M` in same or earlier layer?
     If `M` visited `v` in layer $L$, and `S` visits `v` in layer $L$.
     Order of processing determines parent.
     Since `S` and `M` have same weight, any parent is fine.
     
     Wait, we need to make sure `S` simulates correctly.
     Simulation:
     `for dir in 0..3`:
       `nx = u.x, ny = u.y`
       `while valid(nx+dr[dir], ny+dc[dir])` (check bounds AND `grid[nx+dr][ny+dc]`? `grid` empty).
       Wait, check bound check.
       `nrx = nx + dr`, `nrc = ny + dc`.
       If `nrx >= N || nrx < 0 || ...` -> Hit boundary.
       Else `nrc` is valid cell.
       Actually loop:
         `nx += dr`
         `ny += dc`
         `if (is_outside || is_blocked) break`
         `nx += dr` ??
       Correct logic:
       Start pos `(r,c)`.
       Look `(r+dr, c+dc)`.
       If invalid or blocked, slide distance 0?
       Wait, slide goes to `(r,c)` if blocked immediately? Or no move?
       Assume no move if blocked immediately (or move to wall if no block).
       But practically, `S` is only useful if it lands somewhere.
       If blocked immediately, `S` is useless (cost 1, move 0).
       BFS ignores `v=u`.
       If valid next is open, slide continues.
       Loop:
         `r += dr`
         `c += dc`
         `if (!inside(r, c) || grid[r][c]) break;`
         `// update curr to r, c`
       Wait, inside check handles boundary.
       `if (inside) continue loop` ?
       No, logic:
       `r` starts at `curr.r`.
       Loop:
         `nr = r + dr`
         `nc = c + dc`
         If `!isValid(nr, nc)` break.
         If `grid[nr][nc]` break.
         `r = nr`
         `c = nc`
       End Loop.
       New pos `(r, c)`.
       Add to queue if `!visited`.
       
     Wait, `grid` is static empty.
     So just check bounds.
     
     Double check: "Place a block ... if it does not already contain one; otherwise remove."
     Since we don't use it, `grid` is constant.
     
     Wait, "All squares outside the $N \times N$ area are covered with blocks".
     So `isValid` checks $0 \le r < N$.
     If $N=20$, indices $0..19$.
     Check $r=N$ or $r=-1$.
     
     Complexity:
     $N^2 = 400$. BFS is very fast.
     Doing this 40 times. Total ops $\approx 16000$. Trivial.
     
     Include headers: `<iostream>`, `<vector>`, `<queue>`, `<tuple>`.
     
```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <tuple>

using namespace std;

int N, M;
struct Point {
    int r, c;
    bool operator==(const Point& other) const {
        return r == other.r && c == other.c;
    }
};

// Directions: 0: U, 1: D, 2: L, 3: R
// Move types: 0-3 are M, 4-7 are S
int dr[] = {-1, 1, 0, 0};
int dc[] = {0, 0, -1, 1};
char d_char[] = {'U', 'D', 'L', 'R'};
char m_char[] = {'M', 'M', 'M', 'M'};
char s_char[] = {'S', 'S', 'S', 'S'};

void solve(Point start, Point target, int n) {
    if (start == target) {
        return; 
    }
    
    // BFS state: r, c
    // We need to store path info.
    // To save memory/time, we can use vectors for parents.
    // Since N is small (20), local arrays or vectors are fine.
    
    // Parent tracking
    // parent[r][c] stores tuple: {prev_r, prev_c, move_type_idx}
    // move_type_idx 0..3 -> Move U..R
    // move_type_idx 4..7 -> Slide U..R
    
    // Initialize parent with -1
    vector<vector<tuple<int, int, int>>> parent(n, vector<tuple<int, int, int>>(n, {-1, -1, -1}));
    vector<vector<bool>> visited(n, vector<bool>(n, false));
    
    queue<Point> q;
    q.push(start);
    visited[start.r][start.c] = true;
    
    // BFS
    while(!q.empty()) {
        Point u = q.front();
        q.pop();
        
        if (u == target) {
            // Reached target
            break;
        }
        
        // Try 4 Moves
        for (int i = 0; i < 4; ++i) {
            Point v = {u.r + dr[i], u.c + dc[i]};
            if (v.r >= 0 && v.r < n && v.c >= 0 && v.c < n && !visited[v.r][v.c]) {
                visited[v.r][v.c] = true;
                parent[v.r][v.c] = {u.r, u.c, i + 4}; // Move is 0-3, so i+4? No, let's map to 0-7
                // Let's define 0-3 Move, 4-7 Slide.
                // Actually, I mapped i=0..3 to Moves. So index is i.
                // But wait, Move logic:
                // My mapping:
                // Move U -> idx 0
                // Slide U -> idx 4
                // So parent stores {pr, pc, idx}
                // But Move indices are 0..3, Slide are 4..7.
                
                // Wait, if I use i=0..3 for both, how to distinguish?
                // Let's use separate flags.
                // Or just 0..3 Move, 4..7 Slide.
                // My loop 0..3 corresponds to Move directions.
                // Let's use index 0..3 for Move, 4..7 for Slide.
                // So for Move loop i=0..3: push {i}.
                // Then for Slide loop i=0..3: push {i+4}.
                // Wait, Slide simulation logic is different.
                // If I push {u.r, u.c, i+4}, I record that I took a Slide in dir i.
                
                // Re-structure loop:
                // Loop k=0..7
                // if k < 4: Move
                // else: Slide
                
                // Correction inside loop below.
                
                parent[v.r][v.c] = {u.r, u.c, i}; // 0-3 are Move
                q.push(v);
            }
        }
        
        // Try 4 Slides
        for (int i = 0; i < 4; ++i) {
            int nr = u.r;
            int nc = u.c;
            // Simulate slide
            while (true) {
                int tr = nr + dr[i];
                int tc = nc + dc[i];
                
                // Check bounds and blocks
                // All outside are blocked.
                if (tr < 0 || tr >= n || tc < 0 || tc >= n) {
                    // Hit boundary. Stay at nr, nc.
                    break;
                }
                // Assuming no internal blocks (always true as we don't use Alter)
                // Check if we moved
                // Wait, logic: "Continue sliding ... until you hit a block."
                // If next is valid (inside), we slide there.
                // If next is blocked (outside or internal block), we stop BEFORE.
                // So loop continues as long as tr, tc is VALID.
                // Update nr, nc to tr, tc.
                
                // Wait, what if we start at (0,0) and slide Up?
                // dr = -1. tr = -1. Invalid.
                // Break. Stays at (0,0).
                // So effectively, Slide doesn't move anywhere.
                // We can ignore such transitions.
                
                if (tr >= 0 && tr < n && tc >= 0 && tc < n) {
                    nr = tr;
                    nc = tc;
                } else {
                    break;
                }
            }
            
            // Result pos is (nr, nc)
            // If result == u (didn't move), skip
            if (nr != u.r || nc != u.c) {
                 if (!visited[nr][nc]) {
                    visited[nr][nc] = true;
                    parent[nr][nc] = {u.r, u.c, i + 4}; // Slide indices 4-7
                    q.push({nr, nc});
                }
            }
        }
    }
    
    if (visited[target.r][target.c]) {
        // Reconstruct path
        Point curr = target;
        string res = "";
        // We need to print in correct order (last action first).
        // Using reverse.
        // But output requires line breaks.
        
        // Let's collect actions first.
        // Since we need to print: action direction.
        // Let's store in vector<string>.
        vector<string> actions;
        
        while (!(curr.r == start.r && curr.c == start.c)) {
            auto [pr, pc, pd] = parent[curr.r][curr.c];
            int move_type = pd;
            char a;
            char d;
            
            if (move_type < 4) {
                // Move
                a = 'M';
                d = d_char[move_type];
            } else {
                // Slide
                a = 'S';
                d = d_char[move_type - 4];
            }
            actions.push_back(string(1, a) + " " + string(1, d));
            
            curr = {pr, pc};
        }
        // Reverse actions to get chronological order
        for(int i = actions.size() - 1; i >= 0; --i) {
            cout << actions[i] << "\n";
        }
    } else {
        // Should not happen with M moves
    }
}

int main() {
    cin >> N >> M;
    vector<Point> pts(M);
    for (int i = 0; i < M; ++i) {
        cin >> pts[i].r >> pts[i].c;
    }
    
    // Start at pts[0]
    // Visit pts[1]...pts[M-1]
    
    Point current = pts[0];
    // N is global or passed? Global is easier in CP.
    // Wait, N is 20.
    
    // If M=1, loop doesn't run, correct.
    
    for (int i = 1; i < M; ++i) {
        solve(current, pts[i], N);
        // update current position to target
        // solve function doesn't update it, but logic implies we stopped at target.
        // However, does "stop on it" mean we are physically located there?
        // Yes.
        // Wait, "visit" means stop on it.
        // The simulation ensures we reach (r,c).
        // But wait, if we use Slide R from (0,0) to (19,0), we end at (19,0).
        // If target is (19,0), we are there.
        // If target is (18,0), we passed it? No, we went to 19.
        // My BFS only accepts `visited[target]` if we reach exactly `target`.
        // So BFS guarantees `curr == target` at end.
        // Thus we are positioned at `pts[i]`.
        current = pts[i];
    }
    
    return 0;
}
```

Wait, I missed one thing.
In BFS, I used `parent` and `visited`.
I didn't handle the case where `slide` goes to same spot (loop condition `nr != u`).
That is handled.

One edge case: "Move: Move one square... You cannot move into a square containing a block."
My BFS assumes no blocks.
Is it possible that a target square is blocked initially?
"Initially, there are no blocks inside the N x N area."
Correct.

Wait, `parent` initialization `{-1, -1, -1}`.
Tuple construction works? `tuple` requires `<tuple>`. Yes included.
`auto [pr, pc, pd] = ...` is structured binding (C++17).
If environment is old, use `get<0>`.
The prompt doesn't specify version. C++17 is standard now.
But safer to use `get` or manual access just in case. I'll stick to structured binding, usually supported in modern judges.
Wait, `make_tuple` is safer.
Actually, `struct Node { int pr, pc, pd; };` is safer and faster.

Revisiting `solve`:
I defined `Point` struct.
Need to use `tuple` or custom struct. Custom struct is cleaner.

Also, the problem statement says:
"You must visit the target squares in the given order."
"Even if you pass over a later target square ... not considered visited."
This is consistent.

Is there any case where we need to visit intermediate points?
No.

Double check the complexity of path reconstruction.
Length is proportional to distance. Max 1600. Trivial.

Final verification of sample or logic.
Logic seems sound. BFS minimizes steps.
Max steps $1600$.
Memory usage: `vector` of vectors `20x20`. Very small.
Execution time: $40 \times \text{BFS}(400)$. Very fast.

One small detail:
"The coordinates $(i_k, j_k)$ ... $0 \leq i_k, j_k \leq N-1$".
My code assumes `pts` are $0..N-1$. Correct.

Wait, output format:
`a_t d_t`
`M R`
Space separated.
My code: `string(1, a) + " " + string(1, d)`. Correct.

Wait, `cin >> N >> M;`
$N, M$ are integers.
Loop $i=1 \dots M-1$.
If $M=0$, loop doesn't run.
If $M=1$, loop doesn't run.
If $M=40$, runs 39 times.
Looks correct.

Let's double check the sliding logic again.
`tr = nr + dr[i]; tc = nc + dc[i];`
If `tr` out of bounds, break.
Else `nr = tr`.
Wait, this loop advances `nr` by 1 repeatedly.
Example: Start $(0,0)$, Slide Right (R, `dr=0`, `dc=1`).
`tr=0, tc=1`. Valid. `nr=0, nc=1`.
Next iter: `tr=1, tc=2`. Valid. `nr=1, nc=2`.
...
Stops at `tr=N, tc=something`?
Bounds check: `tr < 0 || tr >= n ...`.
So if `tr = n` (20), break.
Last valid was `tr = n-1` (19).
Correct.
The code logic:
`if (tr >= 0 && tr < n && tc >= 0 && tc < n) { nr = tr; nc = tc; } else { break; }`
Wait, I wrote `break` before assignment.
Correct.
Wait, `if (tr >= 0 ...)` check.
If invalid, break.
If valid, update `nr, nc`.
Loop repeats.
Correct.
What if `tr, tc` valid, but it's a BLOCK?
My code assumes `grid` empty.
`// Check bounds and blocks ... if (tr >= 0 ...)`.
Wait, inside `solve`, I commented `// Assuming no internal blocks`.
Since I don't modify grid, this is correct.
But I should explicitly trust "grid empty".

Wait, is there any case where we *must* stop early?
As discussed, `A` isn't needed for budget.
Is there a case where `S` lands on `Target`?
Yes.
Is there a case where `M` lands on `Target`?
Yes.
BFS finds min steps.
Seems solid.

One small bug in my BFS logic variable name `d_char` vs `d` in reconstruction.
`m_char` used nowhere.
Just use `d_char` and `m_char` arrays.
Actually, `move_type` logic:
`if (move_type < 4) { a='M'; d=d_char[move_type]; }`
`else { a='S'; d=d_char[move_type - 4]; }`
This maps 0->U, 1->D, 2->L, 3->R. Correct.

The solution looks complete.
I'll wrap it in `cpp`.

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>

using namespace std;

int N, M;

struct Point {
    int r, c;
    bool operator==(const Point& other) const {
        return r == other.r && c == other.c;
    }
};

struct ParentInfo {
    int pr, pc, move_type; // move_type: 0-3 Move, 4-7 Slide
};

// Directions: 0: U, 1: D, 2: L, 3: R
int dr[] = {-1, 1, 0, 0};
int dc[] = {0, 0, -1, 1};
char d_char[] = {'U', 'D', 'L', 'R'};

void solve(Point start, Point target, int n) {
    if (start == target) return;
    
    // BFS Initialization
    // parent[r][c] stores info to reconstruct path
    // dist is not strictly needed if we use parent to detect cycle? No, need visited.
    vector<vector<bool>> visited(n, vector<bool>(n, false));
    vector<vector<ParentInfo>> parent(n, vector<ParentInfo>(n));
    queue<Point> q;
    
    q.push(start);
    visited[start.r][start.c] = true;
    parent[start.r][start.c] = { -1, -1, -1 }; // marker for start
    
    while(!q.empty()) {
        Point u = q.front();
        q.pop();
        
        if (u == target) break;
        
        // Try Moves (indices 0-3)
        for (int i = 0; i < 4; ++i) {
            int vr = u.r + dr[i];
            int vc = u.c + dc[i];
            
            if (vr >= 0 && vr < n && vc >= 0 && vc < n && !visited[vr][vc]) {
                visited[vr][vc] = true;
                parent[vr][vc] = {u.r, u.c, i}; // Move
                q.push({vr, vc});
            }
        }
        
        // Try Slides (indices 4-7)
        for (int i = 0; i < 4; ++i) {
            int nr = u.r;
            int nc = u.c;
            // Simulate slide
            while (true) {
                int tr = nr + dr[i];
                int tc = nc + dc[i];
                
                // Check bounds
                if (tr < 0 || tr >= n || tc < 0 || tc >= n) {
                    break; // Hit outside (which is blocked)
                }
                // Since grid is empty (conceptually), just continue.
                // If we assumed empty, valid cell is just inside bounds.
                nr = tr;
                nc = tc;
            }
            
            // Check if slide moved anywhere
            if (nr != u.r || nc != u.c) {
                if (!visited[nr][nc]) {
                    visited[nr][nc] = true;
                    parent[nr][nc] = {u.r, u.c, i + 4}; // Slide
                    q.push({nr, nc});
                }
            }
        }
    }
    
    if (visited[target.r][target.c]) {
        // Reconstruct path
        vector<string> actions;
        Point curr = target;
        
        // Backtrack to start
        while (!(curr.r == start.r && curr.c == start.c)) {
            ParentInfo p = parent[curr.r][curr.c];
            char a, d;
            if (p.move_type < 4) {
                a = 'M';
                d = d_char[p.move_type];
            } else {
                a = 'S';
                d = d_char[p.move_type - 4];
            }
            actions.push_back(a + " " + d);
            curr = {p.pr, p.pc};
        }
        
        // Print in reverse order
        for (int i = (int)actions.size() - 1; i >= 0; --i) {
            cout << actions[i] << "\n";
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cin >> N >> M;
    
    vector<Point> targets(M);
    // The input gives M coordinates including start
    // We read start at index 0
    // Then targets are 1 to M-1
    
    // But wait, problem says "Input ... $i_0 \ j_0$, ... $i_{M-1} \ j_{M-1}$"
    // So M lines follow. First is start. Rest M-1 are targets.
    // Wait, problem says "Initially ... no blocks ... start at $(i_0, j_0)$ and must visit ... $(i_1, j_1) \dots (i_{M-1}, j_{M-1})$".
    // Input format: $N \ M$, then $M$ lines.
    // So reading loop:
    for (int i = 0; i < M; ++i) {
        cin >> targets[i].r >> targets[i].c;
    }
    
    Point current = targets[0];
    // Iterate targets 1 to M-1
    for (int i = 1; i < M; ++i) {
        solve(current, targets[i], N);
        current = targets[i];
    }
    
    return 0;
}
```
Wait, `solve` logic `parent[start]` is `-1,-1,-1`.
Backtrack condition `curr.r == start.r` handles it.
However, if `visited[target]` is true but target == start?
Code `if (start == target) return;` handles it.
Also `actions` loop terminates correctly.

One check: "Input Generation ... randomly shuffle ... take first M".
If $M=40$, we have 40 points.
My code reads M points.
`targets` vector size M.
Loops $i=1$ to $M-1$.
Correct.

Wait, check limits again. $2NM = 2 \times 20 \times 40 = 1600$.
The output length is total steps.
If BFS finds shortest path, sum of steps $\le 1600$.
Is it guaranteed?
Worst case path length in grid graph is bounded?
Not necessarily, can wind around.
But Manhattan distance is upper bound on shortest path using M/S (since S is essentially a move with cost 1).
Actually S can be longer distance but shorter time.
Wait, S can traverse many cells in 1 turn.
Manhattan distance is lower bound on steps if we only move.
With Slides, we can travel more distance in 1 step.
So BFS path length $\le$ BFS path length using only Moves (since S is valid edge).
Max steps using only Moves is bounded by $2 \times (\text{something})$.
Actually, max path visiting $K$ vertices can be large?
No, we visit in order $P_{k-1} \to P_k$.
The distance $P_{k-1} \to P_k$ is $\le 2N = 40$.
Sum $\le 39 \times 40 = 1560$.
So total steps $\le 1560$.
Fits in 1600.
So logic holds.

One tiny thing:
`actions.push_back(a + " " + d);`
This creates a string.
`cout` prints it.
Should be fine.

I will formulate the response now.
</think>

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;

int N, M;

struct Point {
    int r, c;
    bool operator==(const Point& other) const {
        return r == other.r && c == other.c;
    }
};

struct ParentInfo {
    int pr, pc, move_type; // move_type: 0-3 Move (U,D,L,R), 4-7 Slide (U,D,L,R)
};

// Directions: 0: U, 1: D, 2: L, 3: R
int dr[] = {-1, 1, 0, 0};
int dc[] = {0, 0, -1, 1};
char d_char[] = {'U', 'D', 'L', 'R'};

// BFS to find the shortest sequence of actions from start to target
void solve(Point start, Point target, int n) {
    if (start == target) return;
    
    // parent[r][c] stores info to reconstruct path
    // Using vectors for dynamic allocation (though N is small)
    vector<vector<bool>> visited(n, vector<bool>(n, false));
    vector<vector<ParentInfo>> parent(n, vector<ParentInfo>(n));
    queue<Point> q;
    
    q.push(start);
    visited[start.r][start.c] = true;
    parent[start.r][start.c] = { -1, -1, -1 }; 
    
    while(!q.empty()) {
        Point u = q.front();
        q.pop();
        
        if (u == target) break;
        
        // Try Moves (indices 0-3 correspond to U, D, L, R)
        for (int i = 0; i < 4; ++i) {
            int vr = u.r + dr[i];
            int vc = u.c + dc[i];
            
            if (vr >= 0 && vr < n && vc >= 0 && vc < n && !visited[vr][vc]) {
                visited[vr][vc] = true;
                parent[vr][vc] = {u.r, u.c, i}; 
                q.push({vr, vc});
            }
        }
        
        // Try Slides (indices 4-7)
        for (int i = 0; i < 4; ++i) {
            int nr = u.r;
            int nc = u.c;
            
            // Simulate slide until boundary (or hypothetical block, but grid is empty)
            while (true) {
                int tr = nr + dr[i];
                int tc = nc + dc[i];
                
                // Check bounds. Squares outside N x N are blocked.
                if (tr < 0 || tr >= n || tc < 0 || tc >= n) {
                    break; 
                }
                // Assuming grid is empty inside N x N
                nr = tr;
                nc = tc;
            }
            
            // Check if slide resulted in a different position
            if (nr != u.r || nc != u.c) {
                if (!visited[nr][nc]) {
                    visited[nr][nc] = true;
                    parent[nr][nc] = {u.r, u.c, i + 4}; // Slide indices 4-7
                    q.push({nr, nc});
                }
            }
        }
    }
    
    // If target is reachable (guaranteed by connectivity)
    if (visited[target.r][target.c]) {
        vector<string> actions;
        Point curr = target;
        
        // Backtrack to start to collect actions
        while (!(curr.r == start.r && curr.c == start.c)) {
            ParentInfo p = parent[curr.r][curr.c];
            char a, d;
            if (p.move_type < 4) {
                a = 'M';
                d = d_char[p.move_type];
            } else {
                a = 'S';
                d = d_char[p.move_type - 4];
            }
            actions.push_back(a + " " + d);
            curr = {p.pr, p.pc};
        }
        
        // Print actions in correct order (reverse of backtracking)
        for (int i = (int)actions.size() - 1; i >= 0; --i) {
            cout << actions[i] << "\n";
        }
    }
}

int main() {
    // Optimization for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    if (!(cin >> N >> M)) return 0;
    
    vector<Point> targets(M);
    // Read all M coordinates
    for (int i = 0; i < M; ++i) {
        cin >> targets[i].r >> targets[i].c;
    }
    
    Point current = targets[0];
    // Visit targets in order 1 to M-1
    for (int i = 1; i < M; ++i) {
        solve(current, targets[i], N);
        current = targets[i];
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=1198)

````
Let me start by creating a minimal solution to establish a baseline and test it on the sample case. I'll implement a BFS approach to find the shortest path from the initial position to the first target square, handling sliding mechanics and block placement/removal as we go.
</think>

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include <cmath>

using namespace std;

int N = 20;

struct Position {
    int i, j;
    
    bool operator<(const Position& other) const {
        if (i != other.i) return i < other.i;
        return j < other.j;
    }
    
    bool operator==(const Position& other) const {
        return i == other.i && j == other.j;
    }
};

enum Action {
    MOVE,
    SLIDE,
    ALTER
};

enum Direction {
    UP,
    DOWN,
    LEFT,
    RIGHT
};

string actionToString(Action action) {
    switch(action) {
        case MOVE: return "M";
        case SLIDE: return "S";
        case ALTER: return "A";
        default: return "";
    }
}

string directionToString(Direction dir) {
    switch(dir) {
        case UP: return "U";
        case DOWN: return "D";
        case LEFT: return "L";
        case RIGHT: return "R";
        default: return "";
    }
}

pair<Action, Direction> parseMove(string move) {
    char a = move[0];
    char d = move[1];
    
    Action action = MOVE;
    if (a == 'S') action = SLIDE;
    if (a == 'A') action = ALTER;
    
    Direction dir = RIGHT;
    if (d == 'U') dir = UP;
    else if (d == 'D') dir = DOWN;
    else if (d == 'L') dir = LEFT;
    else if (d == 'R') dir = RIGHT;
    
    return {action, dir};
}

int main() {
    cin >> N;
    
    map<int, vector<int>> blocks;
    for (int i = 0; i < N; i++) {
        blocks[i] = vector<int>(N, -1);
    }
    
    vector<Position> targets;
    
    int x = 0;
    int y = 0;
    
    Position pos = {x, y};
    string line;
    getline(cin, line);
    
    cin >> x >> y;
    pos = {x, y};
    targets.push_back(pos);
    
    for (size_t i = 1; i < targets.size(); i++) {
        cin >> x >> y;
        targets.push_back({x, y});
    }
    
    cout << actionToString(MOVE) << directionToString(RIGHT) << endl;
    
    while (!targets.empty()) {
        auto [p, t] = targets.back();
        targets.pop_back();
        
        if (blocks[p.i].at(p.j) == -1) {
            continue;
        }
        
        queue<pair<Position, vector<pair<Action, Direction>>>> q;
        q.push({{p.i, p.j}, {{SLIDE, RIGHT}}});
        
        set<pair<Position, pair<Action, Direction>>> visited;
        visited.insert({{{p.i, p.j}}, {SLIDE, RIGHT}});
        
        size_t minLen = SIZE_MAX;
        vector<pair<Action, Direction>> bestPath;
        
        vector<Position> currentPathPos;
        while (!q.empty()) {
            auto curr = q.front();
            q.pop();
            
            if (curr.second.size() > minLen) {
                continue;
            }
            
            currentPathPos.clear();
            for (auto action : curr.first) {
                currentPathPos.push_back(action.first);
            }
            
            for (Action action : curr.second) {
                int r = curr.first.r;
                int c = curr.first.c;
                
                switch(action.direction) {
                    case UP: c--; break;
                    case DOWN: c++; break;
                    case LEFT: r--; break;
                    case RIGHT: r++; break;
                }
                
                if (r >= 0 && r < N && c >= 0 && c < N) {
                    if (blocks[r][c] == -1) {
                        if (currentPathPos.size() + 1 <= minLen) {
                            minLen = currentPathPos.size() + 1;
                            bestPath = currentPathPos;
                        }
                    }
                    
                    Position nextPos = {r, c};
                    string move = actionToString(action.action);
                    move += directionToString(action.direction);
                    string m = move;
                    q.push({m, bestPath});
                }
            }
        }
        
        if (bestPath.size() > minLen) {
            cout << endl;
            cout << minLen << endl;
            for (auto [p, d] : bestPath) {
                cout << actionToString(p) << directionToString(d) << endl;
            }
            break;
        }
    }
}
```
````
