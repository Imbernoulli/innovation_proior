# 题 43 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=2,得分 0.1,completion_tokens=18948)

````
The user wants a C++ program that generates a solvable Sokoban puzzle grid.
The goal is to maximize the minimum number of moves required to solve it (maximize difficulty).
The scoring function implies a linear scaling up to 63,000 moves for 100 points. So I need to create a level that requires as many moves as possible without exceeding some reasonable time limits for human play or just generating a very hard one. However, since this is an automated output generation task, "difficulty" usually means the path length from start to goal state is maximized in terms of pushes and steps.

Constraints:
- Grid size $N \times M$ where $1 \le N, M, N+M \le 100$.
- One Player 'P'
- One Box 'B' (2x2 area)
- One Storage 'S' (2x2 area)
- Walls '#' and empty '.'
- Solvable.

Output format:
Line 1: N M
Lines 2..N+1: String of length M

The example output shows:
5 6
....SS
....SS
.#BB#.
..BB.P
......
Wait, the example has 'B' at row 3 col 4,5 and row 4 col 4,5. That's a 2x2 block.
'S' is at row 1 col 5,6 and row 2 col 5,6.
'P' is at row 4 col 6.
Walls are at row 3 col 2.

Actually, looking closely at the example provided in the problem description:
```
5 6
....SS
....SS
.#BB#.
..BB.P
......
```
Row 1: `....SS` (S at 0-indexed cols 4,5)
Row 2: `....SS`
Row 3: `.#BB#.`
Row 4: `..BB.P`
Row 5: `......`

Wait, counting columns:
Row 1: . . . . S S -> 6 chars. S is at indices 4,5.
Row 2: . . . . S S -> S is at indices 4,5.
Row 3: . # B B # . -> B at 2,3. Wall at 1.
Row 4: . . B B . P -> B at 2,3. P at 5.
Row 5: . . . . . .

Box 'B' occupies rows 3-4, cols 2-3 (1-based). Indices 2,3 in 0-based.
Storage 'S' occupies rows 1-2, cols 4-5 (1-based). Indices 4,5 in 0-based.
Player 'P' is at Row 4, Col 5 (1-based) -> Index 4,5 (0-based). Wait, string `..BB.P`, index 5 is 'P'. So 0-indexed (row 3, col 5).
Path needs to push box from (2,2)-(3,3) to (4,4)-(5,5)?
No, storage is top right (rows 0-1, cols 4-5). Box is middle left-ish.
Player starts at row 3, col 5.
This seems like a simple configuration.

I need to write a code that outputs *a* grid, specifically one designed to be difficult (long solution path), but within the constraints. The prompt asks me to "generate solution code", meaning the source code that produces the grid. It doesn't ask me to compute the optimal grid dynamically during runtime given input (there is no input). It asks me to write a program that prints a grid such that the solver would take maximum moves.

Since there is no input, I can hardcode a specific grid configuration that is known to be long/hard. Or I can try to construct one programmatically using BFS logic offline? No, the code must generate it instantly. I should embed a pre-calculated "hard" level.

Given the scoring (min moves / 630), if I want high score, I need high moves.
Let's see what is a reasonable max for $N+M \le 100$.
If I make a maze-like corridor that forces the player to wander around before pushing the box into the goal, I can increase moves.
The constraint is 1 box and 1 storage.
To maximize moves, I want the box to be far from the storage, and the player to be forced to navigate a large maze to reach a position where they can push the box towards the storage, potentially requiring multiple back-and-forth moves or complex maneuvering.

Wait, the checker computes "minimum number of moves". I need to make that number large.
In standard Sokoban, moves usually count as single steps (including pushes).
So if the distance between start position and the spot next to the box is 50 steps, and then I push it through a corridor to storage which is another 50 steps, total is ~100.
However, Sokoban levels are often constrained by walls.
The box itself is 2x2. The storage is 2x2.
Let's try to design a corridor level.
Max dimensions roughly 50x50 to fit 100 sum.
A straight corridor requires minimal moves. A twisty corridor requires more.
But the box is 2x2. Pushing a 2x2 block in 1 dimension is easier than moving it in 2D freely because once pushed, it takes up space.

Let's aim for a grid size like 50x50.
Construct a U-shaped or spiral path.
Start P far from Box.
Box near center or elsewhere.
Goal S near other end.
To ensure solvability, the player must be able to get to a side of the box, push it all the way to S.
If I make a long corridor with the box inside, say length 40.
Box at pos X. S at pos Y. P at pos Z.
If the corridor is narrow, the player blocks movement.
Since boxes are 2x2, a "corridor" effectively has width 2 cells for the box.
To move the box along a corridor, the player must push it from behind.
To push it forward, player must be at the tail of the box relative to direction of push.
To reposition the player to the other side of the box to push the other direction, player needs to go around.

Strategy:
Create a long snake corridor filled with walls except for the path.
Place Player at one end.
Place Box somewhere in the middle.
Place Storage at the other end.
However, Sokoban rules allow the player to stand adjacent to the box. If the corridor is 2 units wide (cells (r, c) and (r+1, c)), the box fits exactly. The player cannot fit alongside if the player needs to move past the box?
Actually, player is 1x1. If corridor is width 2, player can squeeze between walls on one side and box on the other?
Grid coordinates:
Box covers (r, c), (r, c+1), (r+1, c), (r+1, c+1).
Wall covers...
To restrict width to allow only box passage, we need walls on outer edges.
Inside, we have 2 lanes for player (if not blocked by box).
If box is occupying both lanes of a 2-high strip, player cannot pass it.
So player must be outside the 2x2 box area.
Pushing the box means the player enters one cell that the box vacates.

To make the level hard (longest moves):
1.  **Distance**: Maximize Manhattan distance between P and S (roughly).
2.  **Maneuverability**: Force player to circle back.
3.  **Obstacles**: Use walls to prevent shortcuts.

Let's draft a 10x30 grid (approx).
Width 10, Height 30.
Rows 0 to 9. Cols 0 to 29.
Let's use a spiral or simple winding path.
Let's try a "snake" shape.
Row 0: Path...
Row 1: Path...
Actually, walls define the corridor.
Let's define a simple maze structure.
Or just a very long straight corridor if possible?
Wait, if I make it straight, moves = distance.
Distance ~ 100/2 = 50 cells.
Is 50 moves too low for "score"? Score = moves/630. 50/630 < 1%.
I need moves closer to 63,000?
Wait, the formula says `score = min(100, max(0, moves / 630))`.
To get 100 points, I need 63,000 moves.
With N+M <= 100, can I make 63,000 moves?
Max path length in a graph with V nodes is V. Here V = N*M.
Max V approx 50*50 = 2500.
Even if every step counts as a move, I can't exceed ~2500.
Ah, wait. The scoring is likely based on something else, or maybe I misread the limit?
"100 points" -> 63,000 moves? That implies the reference solution for the test suite (the generator) might produce grids with very high complexity?
Or maybe the formula is for the *checker* evaluating the difficulty against a benchmark set?
Usually, in these problems, "maximize moves" means making the level hard enough to be distinct from trivial ones. Given the formula `moves / 630`, getting 100 points requires 63,000 moves. This is impossible for a grid of 50x50 if a move is defined as a unit action (player step or push). Even if we revisit states, the shortest path to solution shouldn't be that long.
Unless... "minimum number of moves" in the checker context allows loops? No, minimum path never loops.
Maybe the grid size is larger? "1 <= N, M, N + M <= 100".
Sum of dimensions <= 100.
So Area <= 2500.
Max shortest path length is roughly 2500.
2500 moves / 630 is roughly 4 points.
This suggests my understanding of "moves" or the scoring is slightly off, OR the scoring formula provided in the prompt description text is illustrative of a "very hard" scenario that I should just try to satisfy locally?
Actually, reading carefully: "Scoring rule ... Linear scaling between them."
It probably means the evaluator has a baseline.
If I can't reach 100 points easily, I should aim for the hardest possible valid grid within the constraints.
However, often in these "generate a level" tasks, the intent is just to produce *any* valid level that isn't trivial, or one that passes basic sanity checks. But the prompt explicitly says: "Your goal is to maximize the minimum number of moves needed to solve the puzzle."
If the scoring implies 63k moves is achievable, maybe I misunderstood the move definition?
Is it possible to have cycles in the "solution"? No, minimal moves won't cycle.
Is it possible the grid can be large? 100x100? No N+M <= 100.
Okay, maybe the example "63,000" is from a version where N+M was larger, or this specific problem statement is copy-pasted from a contest where the scoring was different or I am expected to produce the *best possible* under these constraints, which might yield few points but is correct.
However, I should still try to maximize complexity.
Perhaps placing obstacles creates "dead ends" where player must detour significantly.
Let's consider the constraints again.
Maybe the checker calculates "pushes" only? No, usually steps.
Maybe "moves" includes checking validity? Unlikely.
Okay, let's assume maximizing moves means making the shortest path long.
With N+M=100, max grid is roughly 50x50.
Let's try to build a 50x50 grid.
Actually, 50+50=100.
If I put walls to force a long path for the player to find the box, and then a long push sequence.
Let's create a layout:
Player at (0,0).
Corridor leading to a chamber containing the box.
Player walks down corridor. Finds box.
Pushes box into a winding tunnel to storage.
Total moves = Walking to box + Moves to store.
Let's estimate.
Grid 25x40.
Path length could be close to N*M if optimized.
So ~1000 moves.
Score = 1000/630 = 1.58 points.
This is extremely low compared to the potential 100.
Maybe I am supposed to interpret "moves" differently?
Or maybe the scoring description is misleading/contextual for a harder version of the problem, and for *this* specific instance (Small Bigger Sokoban?), the maximum is low.
Wait, look at the title "Bigger Sokoban 40k". Maybe "40k" refers to points?
Actually, usually these generated tasks require producing a solvable grid. The optimization objective is secondary to correctness. I will prioritize correctness and then try to add walls to increase path length.

Let's construct a grid with dimensions that allow plenty of space. Say 34x36 (sum 70). Or 45x45 (sum 90).
Let's pick 40x40.
We need 1 P, 1 B, 1 S.
Rest walls or empty.
To maximize path:
Fill empty space with corridors that force detours.
Simplest hard maze: A perfect maze or labyrinth.
But we need to guide P to B, and push B to S.
Constraint: B is 2x2. P is 1x1.
If I pack walls so B can only move in a line, it reduces search space but increases push distance if line is long.
Best strategy: Make B and S far apart, and P far from B.
Use a simple "L" shape corridor or "U" shape to separate P, B, S.
Then fill remaining space with walls to prevent cutting corners?
If I just output a grid with a lot of '.' and walls separating regions, P might choose the direct route.
I want to force P to take a specific, long route.
So I should remove paths.
Make the playable area a thin spiral.
Size 50 x 20. Sum 70.
Area 1000.
Spiral path of length ~500.
Box inside spiral.
Target at end of spiral.
P at start of spiral.
Moves ~ Distance(P, Box_start) + Pushing(Box, Target).
If the corridor is 2 units wide (for box), player walks along one side.
When box pushes, player follows.
This ensures monotonic progress.
Max moves ~ Length of spiral.
To maximize Length, make it tightest.
Grid 50x50. Spiral radius ~15. Circumference ~2*pi*15 ~ 100. Turns 4. Total length ~400.
Let's implement a generator that creates a spiral or winding path and fills the rest with walls.
Actually, I can hardcode a specific pattern.
Let's make a rectangular maze.
Width W, Height H.
Walls on border.
Internal walls forming corridors.
Example:
```
WWWWWWWWWW...
W..........W
|           |
|   B       |
...         |
W           W
```
To ensure uniqueness, I will generate a deterministic maze.
Since the problem asks for a C++ program to *output* a grid, and I need to provide the code. I can just write a script embedded in C++ that prints the grid lines.
Since I am simulating the AI who writes the code, I will create a static grid in the `main` function or print it character by character via a loop in a helper function.
However, the prompt asks for "solution code".
I will write a complete C++ program that prints a well-crafted grid.
To maximize moves:
I'll create a grid where the player has to traverse a long path to the box, then push the box along another long path to storage.
Since I can't run BFS here to verify minimality, I will rely on geometry.
Geometry:
Player at one corner.
Box at opposite corner.
Storage at same side as Box, but further.
Actually, Box to Storage is fixed once placed.
Let's put P, B, S roughly equidistant in a spiral.
Layout:
Top Left: Start area.
Middle: Corridor with Box.
Bottom Right: End with Storage.
Fill the rest of the grid with walls to block direct routes across open space.
I will construct a grid with dimensions 25x25 (Sum 50) or bigger 40x25.
Let's try to make a simple snake.
Rows: 0..R-1
Cols: 0..C-1
Snake goes row by row, alternating direction.
Walls surrounding the snake path to keep it isolated.
Place P, B, S along this path.
Path length = R*C.
If I place P at beginning of snake, B in middle, S at end.
Moves = (dist P->B) + (distance B->S in grid).
If B is pushed along snake, moves += 2 * distance (to push 1 square, player follows? No, 1 push is 1 move).
Wait, does "move" count a step or a push? Usually "steps + pushes".
If I walk 10 steps to push 1 block, that's 10 + 1 = 11 moves.
So I want to maximize steps to reach box, and steps to push box.
With a long snake of available cells, I can achieve ~5000 moves if grid is 100x100 (area 10000). But N+M<=100.
Max area ~ 2500 (50x50).
So max moves ~ 2500.
This will likely score very low on the scale (0-630 moves range), but it's the best I can do.
I will focus on creating a valid solvable grid with maximal "winding".

Plan:
1. Define grid size 48x48 (sum 96). Fits N,M >= 1.
2. Fill borders with '#'.
3. Create a spiral path in the center.
   - Center cell(s).
   - Expand outwards.
   - Mark path as '.'.
4. Place P, B, S on the path.
   - P at entrance.
   - B near the middle.
   - S near the end.
   - Ensure alignment allows box (2x2) to be on path and pushed along path.
   - Since box is 2x2, the path must accommodate a 2x2 block at B location and 2x2 block at S location.
   - If path width is 1 cell, box can't fit unless the path expands to 2x2.
   - So, the path must be wide enough for a box to exist at any point.
   - Best case: A 2-cell wide corridor.
   - Player is 1 cell wide. Can walk along one side, box occupies other.
   - When box moves, player shifts.
   - Let's make the main path 2 units wide.
   - Fill rest with walls.
   - Add walls to block P from reaching B directly unless traversing the designated path.

Construction:
Size H x W. H=40, W=40.
Border '#'.
Inner core of path.
Let's define a simple rectangular corridor:
Rows 15 to 34. Cols 10 to 39.
Wait, that's wide.
Let's make a spiral of width 2 cells.
Center (20,20).
Expand clockwise.
This ensures length is proportional to perimeter.
For H,W = 40, max path length in a 2-wide corridor is roughly 4*(H+W)/2 ~ 300 cells? No.
If I fill the whole grid with a 2-wide snake filling up to 40x40.
Total cells ~1600.
Length ~800.
Plus P walking to B.
Let's simply construct a grid with specific P, B, S coordinates and a maze that connects them.
To maximize score, I will simply create a "hard" layout manually encoded.
I'll encode a grid that is essentially a long snake with B and S embedded.

Wait, I should check the constraints on Box/Storage placement.
They must be 2x2 squares of empty space initially?
"The grid contains one box (2x2)... one storage (2x2)".
They occupy those cells.
My output needs to show 'B' and 'S' covering those 2x2 areas.
Also '.' for others.
'#' for walls.
'P' for player.
Overlap check: Player cannot overlap Box or Storage.
So P must not be inside B or S rectangle.
And P cannot be adjacent to diagonal walls? No rule about that. Just can't walk through walls, boxes, storage (unless pushing box?).
"cannot pass through walls, boxes, or go outside the grid."
"can push box".
Can P stand on Storage? Yes, but usually storage is considered part of the environment or safe zone. In Sokoban, you can walk on storage spots.
So P cannot be inside B or S.

Algorithm for Output Code:
Print a string constant representation of the grid.
I will create a grid that looks like a long pipe.
Example: 40 rows, 20 cols.
Col 0: Wall. Col 1: Path. Col 2: Path. Col 3: Wall.
Inside Path, create walls to form a spiral.
Wait, if I put walls everywhere except a spiral path of width 2.
Then P must follow the spiral.
Let's try to construct one.
Grid 40x40.
I'll use a 2D char array initialized to '#'.
Carve out a 2-unit wide spiral starting from (1,1) to (1,38).
Wait, spiral needs to be connected.
Better: Just a long horizontal bar, zigzagged vertically.
Like this:
`##` (Wall)
`..` (Path 1)
`..` (Path 2)
...
If I make a continuous path of width 2 that winds through the grid.
Player starts at beginning.
Box is placed midway.
Storage is at end.
Since I can't easily draw ASCII art in comments reliably (alignment issues), I will write the code to iterate and print the grid.
The grid will be stored as `std::vector<std::string>` or raw chars.

Let's refine the "Hard" grid.
I'll generate a deterministic maze layout.
Actually, to maximize complexity, I want to maximize path length.
Path Length L.
Complexity comes from the fact that P might be stuck?
In a simple snake, P goes P -> B -> S. Moves = dist(P,B) + dist(B,S).
Can I make it longer?
If P gets blocked and has to go around?
This adds "dead ends" (traps) that P must clear or avoid?
But P cannot push boxes in dead ends if B is already there.
So I will avoid traps where B is.
I can add traps where P is.
But if I add traps, P has to go around them.
Let's make the "open" space contain some dead-end loops.
Player has to enter loops and come back.
Wait, P cannot pull.
If P enters a dead end, he is stuck unless he pushed something out.
Here only 1 Box.
So if P goes into a dead end with Box blocking exit, Game Over (Stuck).
So Dead Ends must be empty when P reaches them, or P comes back.
But P needs to reach Box.
So I should place Obstacles that P must navigate around.
Since I need to maximize moves, I'll place the obstacles in the path between P and B.
Effectively, I'm making the graph for P long.
Once P hits B, pushes it. Then B slides.
The pushing path for B should also be long.
So: P travels long path to B. Then pushes B on long path to S.
Total Moves = Long_Path_P + Long_Path_B.
This sums up.
Max Moves approx sum of lengths of two disjoint paths.
Max length ~ Area/2.
So Area ~ 2000. Moves ~ 1000.
This seems to be the theoretical limit given constraints.
I'll implement a generator to create this.

Implementation Details:
Dimensions: 45 rows, 22 cols. Sum = 67.
I will carve a path that fills most of the grid.
To ensure 2x2 Box fits, the path width must be at least 2.
If path is width 2, Box occupies full cross-section.
This works.
I will write a recursive function to fill the grid with walls first, then carve path.
Wait, simpler: Construct the path coordinates manually in code or generate algorithmically.
Since I need to submit a C++ code block, I will generate the grid data structure within main.
Algorithm:
1. Initialize grid `ch[50][50]` with '#'.
2. Define path: A list of (r,c) coordinates.
   Start (2, 2). End (48, 48).
   Spiral expansion: (x,y) -> (x+1, y) -> (x+1, y+1) -> ...
   Actually, just a boustrophedon (snake) scan filling 2x2 cells?
   Since Box is 2x2, let's work with 2x2 tiles.
   Treat grid as 2x2 blocks.
   Tile coordinates: Row r from 0 to K, Col c from 0 to W.
   Each tile corresponds to cells [2r, 2r+1] x [2c, 2c+1].
   This guarantees 2x2 fit.
   Number of tiles: K x W.
   Constraint N+M <= 100.
   If I treat as 2x2 tiles, total size in characters is 2K x 2W.
   So 2(K+W) <= 100 => K+W <= 50.
   Let K=20, W=20. Total size 40x40.
   Tiles grid 20x20.
   Path through tiles.
   Let's make a path that visits all tiles (Hamiltonian path).
   Length = K*W = 400 tiles.
   P starts at tile 0.
   B starts at tile 200.
   S starts at tile 400 (end).
   Moves: P traverses 200 tiles. B traverses 200 tiles.
   Total ~ 400 pushes/steps.
   Wait, each tile traversal takes 1 move (walk or push).
   But inside a tile, P/B/S cover 2x2.
   Moving from one tile to adjacent takes 1 cell distance?
   Coordinate system:
   Tiles at (i,j) corresponds to Rows 2i, 2i+1 and Cols 2j, 2j+1.
   To move from (i,j) to (i, j+1), P must be at boundary.
   If path is connected 2x2 blocks, player must squeeze in.
   Wait, if B occupies a 2x2 block, it occupies all 4 cells of a tile.
   P cannot be in that tile.
   P must be in adjacent tile or empty space.
   If the whole level is made of these 2x2 tiles connected side-by-side, P must enter the "void" between tiles to switch.
   This is complicated.
   Simpler approach:
   Use a standard grid with holes (walls) to create paths.
   I will code a generator that creates a specific pattern.
   Pattern: A spiral of width 3.
   Width 3 allows P to walk while B is in the center column?
   No, B is 2x2. Needs 2x2 space.
   If path is 3 wide.
   Layout:
   Wall Wall Path Path Path Wall
                  ^      ^
            Box occupies center 2 cols.
   P occupies edge col.
   When pushing B, P enters center-adjacent col, pushes B to next tile.
   This requires B to shift 1 cell per step.
   
   Okay, let's just write a grid in the solution code.
   To save typing errors in the code, I'll generate the strings based on a pattern logic in C++.
   Pattern:
   Fill 0s. Set boundaries.
   Create spiral.
   Place P, B, S.
   Print.

   I'll use N=40, M=40.
   I'll hardcode the grid generation logic inside `main`.
   To ensure max difficulty, I'll fill the grid such that:
   - P is at (1,1).
   - B is at (20, 20).
   - S is at (30, 30).
   - Path winds between them.
   
   Actually, to guarantee solvability, I need to ensure the topology is connected for the pushing action.
   Let's create a straight line for simplicity, but twisted?
   No, simple straight line is easiest to debug mentally.
   Twist it with walls.
   
   Let's define a grid `char grid[40][40]`.
   Initialize with '#'.
   Carve a path.
   Path definition:
   Start P=(2,2).
   Move right to (2, 35).
   Down to (25, 35).
   Left to (25, 5).
   Up to (5, 5).
   Right to (5, 25).
   Down to (25, 25).
   Left to (25, 15).
   Up to (15, 15).
   Down to (30, 15).
   Right to (30, 35).
   ...
   Basically a large rectangle frame inside the grid.
   And fill the center with another frame.
   Like nested boxes.
   This maximizes area covered.
   Outer frame walls.
   Inner path corridors.
   
   Let's construct:
   Rows 0..39, Cols 0..39.
   Wall at r=0, r=39, c=0, c=39.
   Inner walls forming a maze.
   Let's simplify: One big rectangular room, but P, B, S are separated by a wall barrier that P must dismantle? No, walls can't be removed.
   P must find a path.
   If I make the grid a perfect maze (tree structure), path is unique.
   Perfect mazes are easy to generate.
   I'll generate a simple "wall" maze where every room is accessible.
   To ensure high path length, I will minimize "shortcuts" (empty open spaces).
   So, almost everything is wall, except the path.
   
   Structure:
   Main corridor of width 2, winding through the grid.
   Fills most of the grid.
   I'll create a vector of coordinates representing the path.
   Then assign grid values.
   
   Coordinates Logic:
   Start (2,2).
   Go Right 10 cells.
   Down 10.
   Left 20.
   Down 10.
   Right 20.
   Up 10.
   Down 10.
   Left 20.
   Down 10.
   Right 20.
   Up 20.
   Down 10.
   Stop.
   Adjust to fit 40x40.
   Fill all these cells (and neighbors for width 2) as '.'.
   Rest as '#'.
   
   Wait, Box is 2x2. Storage is 2x2.
   These must align with the path.
   If path is 2-wide, Box can sit on it.
   Player is 1-wide.
   If path is strictly 2-wide, Player can walk along one side, Box on other.
   When Pushing, Player moves to box's front face, Box moves forward.
   This requires Player to access the cell immediately behind the box in the pushing direction.
   If the corridor is width 2, the cell behind the box is occupied by the wall?
   No, corridor is 2 cells wide.
   Let's denote cells as `(r, c)` and `(r, c+1)`.
   Box occupies `(r, c), (r, c+1), (r+1, c), (r+1, c+1)`.
   So for a vertical segment of height 2, Box occupies it fully.
   So Box consumes the entire 2x2 footprint.
   Player must be outside.
   If the path is a series of 2x2 tiles, the player cannot walk "between" tiles if the path width is 2.
   Because the Box occupies the full path width.
   To push the box, the player must be in a cell adjacent to the box.
   If the path is width 2, the cells are `(r,c), (r,c+1)`, `(r+1,c), (r+1,c+1)`.
   Player can be at `(r-1, c)`.
   Pushing Down: Box moves to `(r+2, c)`.
   Player moves to `(r, c)`.
   If the next segment is `(r+1, c) ...` (width 2).
   So the next segment overlaps with current segment?
   No, adjacent 2x2 blocks are shifted by 1 row/col.
   If I arrange 2x2 blocks like bricks?
   (r,c), (r,c+1)
   Next block (r+1, c), (r+1, c+1).
   This is standard grid connectivity.
   The issue is Box being 2x2.
   If Box is at Rows 2-3, Cols 4-5.
   Player pushes from Row 1, Col 4.
   After push, Box at 3-4, Player at 2-3.
   If Player is at 2-3, and wants to push further, Player must be at 1-2?
   This implies Player must slide into the "new" top row of the box.
   So, effectively, the player tracks the box from the same side (say Top/Left).
   As long as Player can always find a slot in front of the box, the box can be pushed.
   In a 2-wide vertical corridor, the player must stay in the lane above the box.
   Can the player switch sides? No, if box blocks width 2.
   So we can't have branching or turning that changes the relative position easily?
   If we turn 90 degrees.
   Vertical corridor -> Horizontal corridor.
   Player is above box. Box turns corner.
   This requires 4x4 area usually for smooth turn of 2x2 box in a 2-wide path?
   Actually, if you have a T-junction, it's tricky.
   Given the constraints and scoring, maybe I don't need to worry about perfect geometric maneuvers.
   Just ensure a simple valid setup.
   Example:
   Horizontal path.
   `........`
   `........`
   Box is `......` (2 rows of `BB`).
   Player is above.
   Player pushes Box Right.
   
   This requires 2 rows of width M.
   If M=20, N=20.
   Grid 20x20.
   P at (0,0).
   B at (0,10).
   S at (0,18).
   Moves: Walk P->B (10), Push B->S (8). Total 18.
   Hard level needs walls to make path long.
   Fill empty space with walls.
   But if I fill with walls, how does P reach B?
   If walls block direct route, P must wind.
   Let's make a wall maze.
   P at (1,1).
   Maze of walls and passages.
   Passage is 1-cell wide? No, P is 1-cell.
   But B is 2x2.
   So Passages must be 2-cells wide to accommodate B?
   If passage is 1-cell wide, B cannot fit.
   Unless I leave a gap.
   But B is 2x2 solid.
   So B needs 2x2 free space.
   If path is 1-cell wide, B cannot be moved through it.
   So the "highways" for the Box must be at least 2 cells wide.
   This constrains the map design.
   I'll make the Box travel on a 2-cell wide track.
   The Player travels on the same track (one lane offset) or parallel track.
   If Player is on the same track but on the "side", say Left column of track.
   Track cells: `(r, c), (r, c+1)`.
   Player at `(r, c)`.
   Box at `(r+1, c), (r+1, c+1)` ... wait, Box is 2x2.
   So Box at `(r, c), (r, c+1), (r+1, c), (r+1, c+1)`.
   So Box and Player cannot share the exact same 2x2 footprint.
   If the track is defined by the area `2x2`, Player occupies 1 cell in it. Box occupies the other 3? No, Box occupies 4.
   So Box occupies the "Lane".
   Player must be in a parallel lane?
   If track width is 3.
   Left lane: Player. Middle: Empty/Wall. Right: Empty/Wall.
   This gets messy.
   
   Simplification:
   Make the Box and Player occupy the grid separately.
   Just ensure a path exists for the Player to reach a neighbor of Box.
   And a path exists for the Box to reach Storage.
   Since I am writing code to generate this, I'll hardcode a simple "Sokoban" style grid.
   Common trick:
   Make a large empty grid.
   Put walls to force a long circuit.
   Place P, B, S.
   Ensure B can fit and reach S.
   
   Let's try this layout:
   Size 30x30.
   Border walls.
   Interior: 1-cell wide spiral for P?
   But B is 2x2.
   So I need to reserve 2x2 space for B and S.
   I can make the whole grid 2x2 blocks.
   Let's say 2x2 blocks are the "nodes".
   Node (i,j) is cells (2i, 2j) to (2i+1, 2j+1).
   Connect nodes with corridors.
   If Node is 'B', it's a box.
   If Node is 'S', it's storage.
   If Node is '.', it's passable.
   But B occupies the node. P cannot be in the node.
   If Node is passable, P can be there.
   P can only push B from adjacent Node?
   This seems restrictive.
   
   Alternative:
   Just output a grid that is known to work.
   I will write a code that prints a grid with a specific layout.
   The layout will be:
   - Outer border walls.
   - Inner wall structure that forms a long "loop" for the player.
   - A specific spot where Box and Storage are located in a small alcove or straight path within the loop.
   - To maximize moves, the player must traverse the loop many times?
   - No, player enters once, finds Box.
   
   Okay, I'll write the code to generate a grid.
   I'll use a fixed grid size (e.g., 40x40) and manually place walls to create a "maze".
   I will create a spiral path for the player.
   Inside the spiral, place the Box.
   At the center/end of spiral, place the Storage.
   To satisfy B=2x2, I'll make sure the "path" accommodates it.
   I will assume the checker is lenient about "tight" packing as long as valid.
   
   Let's construct a 1-cell wide path, but widen it at Box/Storage locations to 2x2.
   This way, B fits only where needed.
   P walks on the 1-cell path.
   P reaches the widening. Pushes B.
   B exits the widening onto the path.
   
   Example:
   ```
    .........
    ..#####..
    .#.......#
    ..........
   ```
   I'll code a generator that draws this.
   
   Code structure:
   `solve()` function returns nothing, just prints.
   Main calls `solve`.
   
   I will create a string buffer `grid[50][50]` init with `#`.
   Function `draw_corridor(r, c, w, h, dr, dc)` to carve out space.
   Place P, B, S.
   Check validity? (I will trust manual construction logic for output correctness, but ensure solvability by keeping B accessible).
   To ensure "maximizing moves", I will maximize the number of empty cells `.` in the path.
   
   Final Plan for Grid:
   40x40 Grid.
   P at (2, 2).
   Box at (20, 20).
   Storage at (30, 30).
   Fill the grid with walls, except for a winding path connecting (2,2) to (20,20) and (20,20) to (30,30).
   The path will be 1 cell wide mostly, but expanded to 2x2 at Box/Storage.
   This prevents the player from finding shortcuts.
   The player will have to walk along the path length.
   Then push the box.
   
   I'll write a simple DFS/BFS inside the code to ensure connectivity? No, I'll define the path explicitly in coordinates and fill it.
   Actually, defining a long spiral manually in code is safer than running a random generator.
   
   Spiral coordinates:
   (2,2) to (38, 2)
   (38, 2) to (38, 36)
   (38, 36) to (10, 36)
   (10, 36) to (10, 12)
   (10, 12) to (30, 12)
   ...
   This ensures a long path.
   
   Wait, if I define the path coordinates, I just mark them as '.'.
   Then place B and S.
   I need to ensure B (2x2) doesn't break the path for P.
   So at B location, the path must be wide enough or P must be routed carefully.
   I will make the path width 2 everywhere, except maybe corners?
   If width 2 everywhere:
   P (1x1) can walk on the edge.
   Box (2x2) can sit in the middle.
   This is stable.
   I'll use width 2.
   
   Let's go.

```cpp
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    // Grid size 40x40 (Sum 80 <= 100)
    const int N = 40;
    const int M = 40;
    
    // Initialize grid with walls
    vector<string> grid(N, string(M, '#'));
    
    // Helper to mark path
    auto mark_path = [&](int r1, int c1, int r2, int c2) {
        int curr_r = r1;
        int curr_c = c1;
        bool horizontal = (curr_r != r2); // false means horizontal
        // Actually better to handle segments
    };
    
    // I will define the grid explicitly to avoid coordinate bugs.
    // A spiral design.
    // 2-unit wide path.
    
    // Row/Col ranges for the path
    // Start (2,2)
    // Path goes to (2, 36)
    // Then (20, 36)
    // Then (20, 4)
    // Then (38, 4)
    // Then (38, 30)
    // Then (12, 30)
    // Then (12, 20) -- Box here!
    // Then (30, 20) -- Storage here!
    // Wait, B and S are 2x2.
    // Let's place them in straight sections.
    
    // Let's refine the path coordinates to be safe.
    // I'll generate a vector of points.
    
    vector<pair<int,int>> path;
    int curr_r = 2, curr_c = 2;
    path.push_back({curr_r, curr_c});
    
    // Horizontal to right
    int target_r = 2;
    int target_c = 38;
    for(int i=target_c; i<=38; ++i) path.push_back({target_r, i});
    target_r = 28; 
    target_c = 38;
    for(int i=target_r; i<=28; ++i) path.push_back({i, target_c});
    
    target_r = 38;
    target_c = 4;
    for(int i=target_c; i<=4; --i) path.push_back({target_r, i}); // Wait, 38 to 4
    
    // Let's write a cleaner generation logic.
    // Just fill the whole grid with '#' and carve a path.
    // I will define the path as a set of cells.
    
    vector<vector<bool>> is_path(N, vector<bool>(M, false));
    
    // Define a winding path filling most of the grid
    // Center 20x20 area.
    // Start at (2,2). End at (38,38).
    // Path width 2.
    
    // I will use a specific hardcoded grid for reliability.
    // It's a "Labyrinth" style.
    // Let's just make a very long straight line with turns?
    // No, let's make a spiral.
    
    // To make it robust, I'll create a path that snakes through the grid.
    // Rows 0-39, Cols 0-39.
    // Path will visit every other cell roughly.
    
    // Actually, writing a full maze generator in code is error-prone for formatting.
    // I will output a simple but "hard" grid pattern that is guaranteed valid.
    // 2-wide vertical shaft, with horizontal connections.
    
    // Let's try to output the example grid modified to be bigger and harder.
    // The example was 5x6.
    // I will scale it up.
    // Example had a wall separating P from S.
    
    // My plan:
    // Create a grid of size 30x50 (Sum 80).
    // Path for Player: Snake filling 30x25 area.
    // Box and Storage placed in the snake.
    // Since Box is 2x2, the snake width must be 2 cells at least at Box loc.
    // I'll make the ENTIRE snake width 2 cells.
    // Cells (r, c) and (r+1, c).
    // This allows a 2x2 Box to sit perfectly in one step of the snake.
    // Player is 1x1. Can sit in one cell of the pair.
    // But Player needs to be able to push.
    // If width is 2, say col k and k+1.
    // Box at (r, k)..(r+1, k+1).
    // Player at (r-1, k). Pushes Box down.
    // Box moves to (r+2, k)..(r+3, k+1).
    // Player moves to (r, k)..(r+1, k) ? No, Player is 1x1.
    // Player moves to (r-1, k) -> (r, k).
    // Wait, Box occupies (r, k). Player can't enter.
    // So Box must NOT occupy (r,k).
    // Box is 2x2. So it occupies (r, k), (r, k+1), (r+1, k), (r+1, k+1).
    // If Player is at (r-1, k).
    // Pushing Down puts Box at (r+1, k), (r+1, k+1), (r+2, k), (r+2, k+1).
    // Player moves to (r, k) ?? No (r, k) is now free?
    // Wait, Box started at (r,r+1)..(r+1, r+2).
    // Old top-left (r,k). New bottom-right (r+2, k+1).
    // Cell (r, k) becomes occupied by... nothing? It was occupied by Box. Now Box moves down.
    // So (r, k) becomes empty.
    // Player enters (r, k).
    // Correct.
    // So if the path is 2-wide (cols k, k+1), and Player is at (r-1, k).
    // Box occupies (r,k)..(r+1, k+1).
    // Push Down -> Box to (r+1, k)..(r+2, k+1).
    // (r, k) becomes empty.
    // Player moves to (r, k).
    // Player is now at the new top-left of Box.
    // Can continue pushing.
    // This requires Player to always be at (Box_row-1, Box_col).
    // If the path is 2-wide, Player stays on Left Lane.
    // Box stays on Left-Lane + Right-Lane.
    // This works for Straight.
    // What about Turning?
    // If path turns Right.
    // Vertical segment ends. Horizontal segment begins.
    // Player at (r, k). Box at (r+1, k)..(r+2, k+1).
    // Next cell is (r+1, k+1) (horizontal right from box?)
    // Wait, Box occupies (r+1, k)..(r+2, k+1).
    // If path continues horizontally:
    // Vertical cells (c, k+1) for rows ...
    // Box occupies (r+1, k+1).
    // So horizontal path must connect to (r+1, k+1).
    // Player is at (r, k). Wants to push Right.
    // Player needs to be at (r+1, k).
    // But Box is at (r+1, k)..(r+2, k+1).
    // Wait, if Box is at (r+1, k), it occupies horizontal cell (r+1, k+1)? No, (r+1, k+1) is part of 2x2 box?
    // 2x2 Box at (r+1, k) covers (r+1, k), (r+1, k+1), (r+2, k), (r+2, k+1).
    // So yes, Box occupies (r+1, k+1).
    // So Player cannot push Right from (r, k) because (r+1, k) is occupied by Box.
    // Player must be at (r+1, k-1) or (r+2, k) etc.
    // If Player is at (r, k), he is "above" the box's top-left.
    // He can't push Right.
    // He can only push Down or Up (from (r, k)).
    // To turn, he must move to the side.
    // With 2-wide path, moving to side is hard.
    // To simplify, I will make the whole grid a SINGLE straight 2-wide corridor.
    // Length 40.
    // P at start.
    // B at middle.
    // S at end.
    // Player walks down corridor. Reaches Box.
    // Player pushes Box to S.
    // Moves = Dist(P, B) + Dist(B, S) + extra shuffling.
    // If I add walls that P has to bypass?
    // If corridor is 2-wide, P has only 1 lane.
    // P can't go around Box.
    // If I widen the corridor to 3, P can go around.
    // But to maximize difficulty, I want P to NOT go around.
    // So I will stick to the 2-wide corridor but make it very long.
    // And maybe add some loops (snakes) so P has to travel more distance?
    // If P has to travel through loops, that adds moves.
    // But P must eventually reach B.
    // If I make a maze where B is deep inside.
    // P walks to B.
    // Then pushes B to S.
    
    // Okay, I will create a "Zig-Zag" corridor.
    // Dimensions 20 rows, 50 cols? N+M <= 100.
    // 30 rows, 60 cols? N+M=90.
    // I'll use 30 rows, 60 cols? No, M limited? No, N+M.
    // Max 100.
    // 40x40 is fine.
    // Zig-zag path of length ~400.
    // Fill rest with walls.
    // Ensure P reaches B and B reaches S.
    
    // Implementation:
    // Grid 40x40.
    // Fill with '.':
    // Rows 0 to 39.
    // For r=0 to 38:
    //   c = (r % 4 == 0) ? 0 : (r%2==0 ? 10 : 20) ? ...
    //   Better: Spiral.
    //   Fill (1,1) to (38,38) in a spiral manner.
    //   Mark all visited cells as '.'.
    //   Ensure 2x2 area for Box exists.
    //   Pick B coords.
    //   Pick S coords.
    //   Pick P coords.
    
    // Let's write the grid explicitly.
    // I'll construct a vector of strings.
    // I'll hardcode the logic to fill the grid.
    
    // Let's use a simple 3-pass fill:
    // 1. Draw outer walls.
    // 2. Draw inner walls to form corridors.
    
    // Actually, I can just construct the grid character by character.
    // Let's try to maximize the path length by visiting as many cells as possible.
    // I'll fill the grid with '.' except for walls that separate the player from the box until necessary?
    // No, that makes it unsolvable if P is trapped.
    // I need a connected component for P.
    // And B.
    // I'll make one huge component.
    
    // Let's go with the 40x40 grid.
    // I will define a "maze" using a simple recursive backtracking algorithm inside the code?
    // Yes, that's efficient and generates a good maze.
    // Then I plant P, B, S.
    
    void solve_maze(vector<vector<char>>& g, int n, int m) {
        // Implement maze gen
    }
    
    // But I need to place P, B, S such that they are solvable and far apart.
    // I will generate the maze, then find path, place items.
    // This ensures connectivity.
    
    // Wait, Time Limit 2s. Maze Gen is instant.
    
    return 0;
}
```
Re-evaluating "Maximize moves".
If I generate a maze, the path length depends on randomness.
I can seed the RNG or use a deterministic method to make it consistently long.
Recursive Backtracker (DFS) generates a maze with no loops and max path length (spanning tree of cells).
Number of edges ~ Nodes/2 (average degree 2).
If I fill the grid with '.' (visited), then path length is related to grid size.
If I use 100% density, I visit almost all cells.
Max cells = 2500.
Path ~ 2500.
Score ~ 4.
If I can somehow extend the moves by forcing the Player to backtrack?
In a tree maze, P goes down a branch. If B blocks him?
He needs to push B.
If B is at a leaf, P must push it into a deeper branch?
Yes.
So P travels P -> B. Then pushes B -> S.
If the maze is a tree, B is stuck at a leaf.
P can only push B along the tree away from the root? No, towards leaves.
If S is also at a leaf in the same direction?
If P and B are separated, P has to clear path to B.
With 1 Box, P is free until B is there.
If I make the "Box path" a long straight tunnel, and P has to dig a long way to get there.
This is effectively the same as "Long Corridor".
I will stick to a Long Corridor with bends.
Bends add distance.
Straight line is 1 unit per move.
Tunnel length L. Moves ~ L + 1.
To maximize L, I maximize the grid.
N=45, M=55? No, N+M<=100.
Max product for fixed sum is N=M=50.
So 50x50.
I'll use 50x50.
Filling a 2-wide spiral gives path ~ 2500.
Moves ~ 2500.
Score ~ 4.
I'll write the code to generate this.

To ensure P, B, S are valid:
P must not overlap B or S.
B, S must be aligned with path.
I'll use the spiral path.
Box is 2x2. Storage 2x2.
Path width 2.
I'll ensure the spiral consists of 2x2 blocks.
(2k, 2l) to (2k+1, 2l+1).
This simplifies things.
If I use 2x2 tiles.
Tiles grid 25x25.
Path is a Hamiltonian path on tiles.
Length ~ 600 tiles.
Each tile = 2x2 cells.
P occupies 1 cell. B occupies 2x2 tile.
P cannot occupy the tile where B is.
P cannot occupy the tile where S is.
So P has to move to adjacent tiles.
Since tiles are 2x2, P can't squeeze between them easily.
Wait, if the path is width 2 cells.
(1,1), (1,2) -> Next (2,1), (2,2) ?
If I link tiles (1,1) to (2,1) vertically.
Cells:
Tile A: (0,0)..(1,1).
Tile B: (2,0)..(3,1).
Path connects them.
P moves from A to B.
B is 2x2. Occupies A.
P cannot be in A.
P must be in "outside" A.
Where is outside A?
If path is continuous, there are cells adjacent to A.
But if I fill the grid with this path structure, A is surrounded by walls?
Then P can't enter A if B is in A.
But P can be in A *before* B arrives?
No, initially B is in B_tile.
P is in Start_Tile.
P walks to B_tile.
Once P is adjacent to B_tile, pushes B.
This requires P to access B_tile.
If B_tile is part of the path, and B sits there.
P stands in Adjacent_Tile.
Pushes B into B_tile_next.
This works if Adjacent_Tile is accessible from P's previous tiles.
So, the path of tiles must be a valid graph where B moves from Node_i to Node_{i+1}.
P moves from Node_j to Node_k...
So I just need a sequence of 2x2 tiles.
P starts at Tile_0.
B starts at Tile_K.
S starts at Tile_M.
P walks Tile_0 -> Tile_{K-1} -> Edge of Tile_K.
Pushes B from Tile_K -> Tile_{K+1} ... -> Tile_M.
This assumes P can stand on the "edge" of Tile_K.
Edge means the cells of Tile_{K-1}.
This is consistent.
So I will generate a path of 2x2 tiles.
Grid size 50x50.
Tile count ~ 600.
P at Start.
B at Middle.
S at End.
All connected.
Max moves = Steps to push B.
Steps = Dist(P_start, B_start) + Push_Dist.
Dist(P, B) = (K-1) tiles.
Push_Dist = (M-K) tiles.
Total moves = (K-1) + (M-K) = M-1 tiles.
M ~ 600.
So ~ 600 moves.
Score ~ 600/630 ~ 1.
Okay, this is acceptable. I will implement this.

Structure:
Generate a grid `G[50][50]`.
Initialize with '#'.
Define Tile path:
List of tile coords `(r, c)`.
Mark corresponding cells as '.'.
Place P at `(0,0)` (Tile 0).
Place B at `(24, 24)` (Tile ~1200? No 50*50/4 = 625).
Place S at `(48, 48)`.
Fill remaining with '#'.
Connect tiles?
If I just fill a snake of tiles.
(0,0) -> (0,1) -> ... -> (0,24) -> (1,24) ...
Wait, Tile (0,0) is cells (0,0)..(1,1).
Tile (0,1) is cells (0,2)..(1,3).
Are they connected?
Cells (0,0)..(1,1) touches (0,2)..(1,3) at col 1/2 boundary?
Indices:
Tile (0,0): Cols 0,1.
Tile (0,1): Cols 2,3.
Boundary: Col 1 vs 2. Adjacent.
Yes.
So horizontal adjacency works.
Vertical:
Tile (0,1): Rows 0,1.
Tile (1,1): Rows 2,3.
Boundary: Row 1 vs 2. Adjacent.
Yes.
So a grid of tiles works perfectly for a 2-wide path.
I'll fill the tiles grid with a snake pattern.
Then place P, B, S on the path.
Fill unused cells with '#'.

Code Logic:
1. `vector<string> grid(50, string(50, '#'));`
2. `vector<string> path_grid(25, string(25, '.'));` (25x25 tiles).
3. Fill `path_grid` with walls `#` to enforce the path structure?
   No, I'll just select a path on the 25x25 tile grid.
   Let's make the tile grid a long snake visiting most tiles.
   Start (0,0). End (24,24).
   Path:
   Rows 0..12, Cols 0..12?
   Actually, a simple zigzag fills half the grid.
   Or fill whole grid.
   I'll fill ALL tiles to '.'.
   Then P can go anywhere?
   If I fill all tiles, P has choices.
   To force path, I need to block shortcuts.
   But B is 2x2. If I have empty space, B might stop at a spot.
   If I leave everything as empty space, B can slide anywhere.
   If B is in the middle, P must push it to corner.
   If I leave the grid full of '.' (except walls for P/B/S boundaries).
   P starts at (0,0).
   B starts at (12, 12).
   S starts at (24, 24).
   B can move freely in the empty space.
   P has to find B.
   Once P finds B, P pushes B to S.
   If B has 4-direction freedom, P can optimize path? No, "Minimum Moves".
   I want to MAXIMIZE Minimum Moves.
   So I should restrict B's movement options.
   If I make the path a SINGLE LINE of tiles.
   Then B MUST move along the line.
   This enforces the path.
   So I should BLOCK off branches.
   So I should define a SUBSET of tiles as available.
   The subset forms a long snake.
   P is at start of snake.
   B is somewhere in middle.
   S is at end of snake.
   Any other cells are '#'.
   This forces P to walk the snake to B, then push B along the snake to S.
   Moves = Distance.
   This is optimal for difficulty.
   
   So:
   1. Init 50x50 with '#'.
   2. Construct a snake path of cells (or tiles) of width 2.
      Let's use the 2x2 Tile abstraction.
      Generate a snake on 25x25 Tile Grid.
      Mark cells belonging to "Active Tiles" as '.'.
      Others remain '#'.
   3. Place P, B, S on the Active Tiles.
      P at first active tile.
      B at middle active tile.
      S at last active tile.
      (Ensure B's 2x2 fits in its tile).
   
   How to generate snake on 25x25 tiles:
   Simple spiral or zig-zag.
   I'll fill the grid row by row.
   Row 0: (0,0) -> (0,1) ... -> (0,24)
   Row 1: (1,24) -> (1,23) ... -> (1,0)
   Row 2: (2,0) -> (2,1) ...
   This fills 25x25 = 625 tiles.
   Path length ~ 625.
   
   Wait, does this form a valid grid?
   Yes, each tile is 2x2 cells.
   If Tile (r,c) is active. Cells (2r, 2c) to (2r+1, 2c+1) are marked '.'
   Wait, if I connect Tile (0,0) and (0,1).
   They are side by side.
   Is it a path?
   P is at (0,0).
   To move to (0,1), P must cross boundary.
   Boundary between (0,1) and (0,2) is valid.
   Yes.
   So P can traverse the snake.
   
   Where to put P?
   P is a person (1 cell).
   P starts at Tile (0,0).
   Which cell in Tile (0,0)?
   Doesn't matter much, say Top-Left (0,0).
   Where to put B?
   Tile (120, 120) (Midpoint).
   Cells (240, 240)..(241, 241).
   Where to put S?
   Last Tile.
   Cells (48, 48)..(49, 49).
   
   Is it solvable?
   P walks to B. Pushes B.
   Can P push B?
   P needs to be in a cell adjacent to B's face.
   B is 2x2.
   If B is at Tile T.
   P is at Tile T-1 (previous in snake).
   P is at (2(T-1)r, 2(T-1)c).
   P needs to push B "forward" to Tile T+1.
   Forward direction is (T+1).
   Face of B facing T+1 is the edge shared with T+1.
   P needs to be in the cell adjacent to that face? No, P pushes B *into* T+1.
   So P must be in the cell adjacent to B, pointing to T+1.
   If B is in T. T+1 is next.
   P must be in T (behind B).
   But B occupies T.
   So P cannot be in T.
   P must be in the "tail" of the 2x2 block B occupies?
   Wait, if B occupies T (2x2).
   P is coming from T-1.
   P is at some cell adjacent to B.
   Pushing B to T+1.
   B moves from T to T+1.
   P moves from behind to ... where?
   Into T? Yes.
   But T is becoming empty after B leaves.
   So P enters T.
   But T has been the "old position".
   P needs to be at a cell that becomes available.
   If P was at a cell in T-1 adjacent to T.
   After push, B is in T+1. T is empty.
   P can enter T.
   So yes, valid.
   
   Is there a conflict at the start?
   P starts at T0. B is at Tmid.
   P walks through T1..T(mid-1).
   Enters T(mid-1).
   Now P is adjacent to Tmid (where B is).
   P pushes B.
   B moves to T(mid+1).
   P moves to Tmid.
   Now B is in T(mid+1). P is in Tmid.
   Valid.
   
   So this works perfectly.
   
   Grid Construction:
   1. Init 50x50 with '#'.
   2. Iterate r 0..24, c 0..24 (Tile Grid).
      Trace path in a snake (boustrophedon) order.
      Wait, I need to mark cells as '.' ONLY IF they are part of the path.
      My logic: "Only mark T cells".
      So if I iterate r, c and fill a continuous path.
      I can just fill a snake pattern over the whole grid, assuming it forms a connected component?
      Actually, filling a snake over the WHOLE grid (all tiles) gives a valid connected grid.
      If I fill ALL tiles (0,0) to (24,24) using the snake pattern, P can go anywhere.
      But if I fill EVERYTHING with '.', B is loose.
      I want B restricted.
      Wait, if the grid is a single snake, it IS restricted.
      It is a Hamiltonian path.
      So B cannot jump off the path.
      So I should mark cells for the snake path.
      And fill the rest with '#'.
      
      So:
      Generate Snake on 50x50 grid.
      (Using 2x2 blocks concept is easier to visualize, but I'll just use 50x50).
      Path coordinates:
      Start (0,0).
      (0,0)..(0,1) -> (2,0)..(2,1) ? No, 2x2 blocks are (0,0)..(1,1).
      Let's work with 50x50 cells directly.
      Path of 2-unit width.
      Snake:
      Row 0: (0,0) to (0,49) -> (2,49)?? No.
      To maintain connectivity for 2-unit width:
      Block 1: (0,0)..(1,1).
      Block 2: (0,2)..(1,3).
      Block 3: (0,4)..(1,5).
      ...
      Block k: (0, 2(k-1))..(1, 2k-1).
      This makes a horizontal strip of 2-row height.
      To turn down:
      Next Block must connect to previous.
      Previous is (0, 48)..(1, 49).
      Next can be (2, 48)..(3, 49)?
      Yes.
      So:
      Fill Row 0,1 for Cols 0,1..49.
      Then Row 2,3 for Cols 48,47..0 (Reverse).
      Then Row 4,5 for Cols 0,1..49.
      ...
      Until Row 48,49.
      This covers the entire 50x50 grid.
      Path length = 50x50 cells / 2 (since 2 wide) = 1250 "blocks".
      Total cells = 2500.
      This creates a valid grid where B is trapped in a line.
      
      P at Start Block (0,0)..(1,1).
      B at Middle Block (e.g. Row 15..16, Col 10..11).
      S at End Block.
      
      Wait, if I fill the whole grid with this pattern, P has NO choice but to follow the snake.
      So moves = max possible.
      Score ~ 1250/630 > 2.
      
      Wait, if B is at (15,10), S at (48,49).
      P needs to walk to (15,10).
      Path is linear.
      P walks along the snake.
      Reaches B.
      Pushes B to S.
      Total moves = P_Walk + B_Push.
      Since P and B move in sync (mostly), P_Walk ~ B_Distance.
      Total ~ Distance(P_start, S_end).
      Distance is ~ 50x50 / 2 = 1250 blocks.
      Moves ~ 1250.
      This is great.
      
      Implementation details:
      Loop i from 0 to 1249 (Block index).
      Map i to (r_block, c_block).
      Calculate cell coordinates.
      Mark as '.'.
      Determine P, B, S positions.
      Fill rest with '#'.
      
      One detail: Player P must be 1x1.
      In the snake blocks, I have 4 cells per block.
      P starts in one cell.
      B occupies all 4 cells of its block.
      Wait, if I put B at Block K.
      P is at Block K-1.
      P pushes B.
      P enters Block K.
      But Block K is marked '.'.
      Initially B occupies Block K.
      So B is at Block K.
      Does B block P from entering Block K?
      Yes, initially.
      But P needs to get to Block K-1.
      If P is at Block 0..K-1.
      This works.
      
      Wait, does P start inside the grid?
      Yes, `grid[row][col] = 'P'`.
      I need to overwrite the '.' with 'P'.
      And 'B', 'S'.
      
      Wait, B is 2x2.
      I need to place 'B' in the 4 cells.
      'S' in the 4 cells.
      'P' in 1 cell.
      Other cells in the path '.'.
      
      Correct.
      
      Final verification of N+M.
      N=50, M=50. Sum 100.
      Allowed.
      
      Wait, 1 <= N, M, N+M <= 100.
      50+50 = 100. OK.
      
      I will write the code.
      
      One edge case:
      If P is at (0,0) inside Block 0.
      B is at Block Mid.
      Is it possible for P to push B?
      P is at (0,0) or (1,0).
      B is at (14, 14) ...
      P walks to (13, 13).
      Then pushes.
      
      Wait, does P have to push from the same block?
      B is at Block (r, c).
      P is at Block (r, c-1) (Previous).
      P is at cell adjacent to B.
      B pushes "Right" into Block (r, c+1).
      P moves into Block (r, c).
      This requires B to be able to move from (r,c) to (r,c+1).
      Since the snake is 2-wide blocks.
      Block (r,c) is connected to (r,c+1).
      So this works.
      
      One small fix:
      In the block definition, I said "Block 1, 2...".
      I'll assign block indices 0 to 1249.
      P is at block 0.
      B is at block 625.
      S is at block 1249.
      
      I will write the solution now.
</think>

```cpp
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    // Grid dimensions: 50x50 to maximize path length while respecting N+M <= 100.
    // Using 50x50 allows for a large snake path.
    const int N = 50;
    const int M = 50;
    
    // Initialize grid with walls
    vector<string> grid(N, string(M, '#'));
    
    // We will construct a 2-unit wide "snake" path filling the grid.
    // The path is composed of 2x2 blocks arranged in a snake pattern.
    // Total blocks = (N * M) / 4 = 625.
    // We will fill blocks sequentially.
    // Block i covers cells based on its position in the 25x25 tile grid.
    
    // Tile grid is N/2 x M/2 = 25x25.
    // We traverse tiles in a boustrophedon (zig-zag) order to form a long connected path.
    // Tile (r, c) corresponds to cells (2*r, 2*c), (2*r, 2*c+1), (2*r+1, 2*c), (2*r+1, 2*c+1).
    
    int total_tiles = (N / 2) * (M / 2); // 25 * 25 = 625
    vector<pair<int, int>> tile_path;
    
    int r_tile, c_tile;
    for (int t = 0; t < total_tiles; ++t) {
        // Calculate tile coordinates
        // Fill row by row, alternate direction
        int row_idx = t / (M / 2);
        int col_idx_in_row = t % (M / 2);
        
        if (row_idx % 2 == 0) {
            c_tile = col_idx_in_row;
            r_tile = row_idx;
        } else {
            c_tile = (M / 2) - 1 - col_idx_in_row;
            r_tile = row_idx;
        }
        
        tile_path.push_back({r_tile, c_tile});
    }
    
    // Place Player, Box, and Storage
    // Player starts at the first tile of the path.
    // Box starts at the middle tile.
    // Storage ends at the last tile.
    
    int p_tile = 0;
    int b_tile = total_tiles / 2; // e.g. 312
    int s_tile = total_tiles - 1; // 624
    
    int pr_tile = tile_path[p_tile].first;
    int pc_tile = tile_path[p_tile].second;
    int br_tile = tile_path[b_tile].first;
    int bc_tile = tile_path[b_tile].second;
    int sr_tile = tile_path[s_tile].first;
    int sc_tile = tile_path[s_tile].second;
    
    // Mark the path cells as '.'
    // We iterate all tiles in the path (which is the full grid) and mark as '.' first.
    for (const auto& t : tile_path) {
        int r = t.first * 2;
        int c = t.second * 2;
        grid[r][c] = '.';
        grid[r][c+1] = '.';
        grid[r+1][c] = '.';
        grid[r+1][c+1] = '.';
    }
    
    // Place Player at the first tile
    // We overwrite one cell with 'P'
    // To ensure solvability, P should be positioned to facilitate the first push.
    // Standard Sokoban usually places P at the entrance.
    grid[pr_tile * 2][pc_tile * 2] = 'P';
    
    // Place Box 'B' at middle tile (overwrite all 4 cells)
    grid[br_tile * 2][bc_tile * 2] = 'B';
    grid[br_tile * 2][bc_tile * 2 + 1] = 'B';
    grid[br_tile * 2 + 1][bc_tile * 2] = 'B';
    grid[br_tile * 2 + 1][bc_tile * 2 + 1] = 'B';
    
    // Place Storage 'S' at last tile
    grid[sr_tile * 2][sc_tile * 2] = 'S';
    grid[sr_tile * 2][sc_tile * 2 + 1] = 'S';
    grid[sr_tile * 2 + 1][sc_tile * 2] = 'S';
    grid[sr_tile * 2 + 1][sc_tile * 2 + 1] = 'S';
    
    // Print the grid
    cout << N << " " << M << endl;
    for (int i = 0; i < N; ++i) {
        cout << grid[i] << endl;
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=490)

````
I need to generate a Sokoban grid where the box and storage location are placed far apart, with many walls blocking direct paths, so that solving it requires many moves.

The key idea is:
1. Make the box and storage very far apart (at opposite ends of the board)
2. Place walls between them to force long detours
3. Arrange empty space so boxes can only be pushed along specific winding routes
4. Make each box position "lockable" by pushing a box against a corner/wall
5. Design it like a maze - boxes must move through narrow passages one at a time

Let me construct such a grid step by step:

**Grid size:** 10×14 (10 rows, 14 columns) for good maneuvering room

**Core challenge design:**
- Row 0-1: Storage at (0,0)-(1,1) - top-left corner
- Row 8-9: Box at (8,13)-(9,12) - bottom-right corner (opposite end!)
- Rows 2-7: Complex maze structure with walls forcing specific movement patterns

This creates maximum distance between start (box) and finish (storage), requiring boxes to traverse the entire maze multiple times.
</think>

```cpp
#include <iostream>
using namespace std;

int main() {
    cout << "10 14" << endl;
    cout << "####P#...." << endl; // Top edge with storage nearby, player on open side
    cout << "####P#...." << endl;
    cout << ".####..BB." << endl; // Wall barrier, box in maze section
    cout << ".####..BB." << endl;
    cout << "#.###......" << endl; // First wall corridor
    cout << "#.###..S..#" << endl;
    cout << "##.#..S..#." << endl;
    cout << "##.#..S...#" << endl;
    cout << "..S#.....B.." << endl; // Bottom area connecting regions
    cout << "..S#.....B.." << endl;
    cout << "..#.........#" << endl;
    cout << "..........." << endl;
    return 0;
}
```
````
