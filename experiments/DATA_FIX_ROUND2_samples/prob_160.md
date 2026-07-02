# 题 160 — base均分 2.7 vs 直接SFT均分 0.0(掉点 2.7)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=2,得分 13.6,completion_tokens=21886)

````
The user wants a C++ solution for the "AtCoder Beginner Contest 015" (specifically the AHC015 problem "Trick or Treat").
This is an interactive problem where I need to simulate the process and output a direction (`F`, `B`, `L`, `R`) after each candy placement.
There are $100$ candies to be placed in a $10 \times 10$ grid.
For each step $t=1 \dots 99$, I receive:
1. The flavor of the next candy (known at start).
2. The index of the empty cell where it will be placed ($p_t$). The cells are numbered row by row, then filling remaining rows? No, usually row-major order based on the empty cells. Wait, let's check the problem description carefully.
"The box is initially empty, and $100$ pieces of candy will be placed in order."
"In the $t$-th process... a single integer $p_t$... Let us number the empty cells from $1$ to $101-t$ in front-to-back and left-to-right priority..."
Wait, "front-to-back" usually means rows 0 to 9, and "left-to-right" means columns 0 to 9. So standard raster scan order (top-left to bottom-right) but skipping filled cells.
After placing the candy, I must output a move direction.

Goal: Maximize connectivity of same flavors. Score depends on $\sum n_i^2 / \sum d_j^2$. This means maximizing large connected components of identical flavors. Basically, try to keep candies of the same type together.
The constraints are tight for 100 test cases (total score sums up), but since this is a local contest (AHC), I can write a heuristic.

Strategy ideas:
1. Greedy approach: After placing a candy, tilt in a way that pushes existing candies into corners or keeps them grouped with new ones. Or tilt to create space for future candies of the same flavor.
2. Lookahead simulation: Since we only have 100 steps total for one testcase (and actually 200 testcases in real judging, but locally usually fewer or just 1 large test), and N=100 is small, we can afford some lookahead. However, we cannot predict future placements $p_{t+1}$ until step $t+1$. But we know the sequence of flavors $f_1 \dots f_{100}$ *in advance* before the interaction starts. We don't know the exact positions $p_t$ until they come in.
Wait, the problem says: "First, the flavor of each piece of candy is given... After reading the above information, repeat the following process 100 times."
So I have the full flavor array `f` beforehand.
I get `p_t` interactively. I need to decide move.
Since I know `f`, when I place a candy of flavor X at `(r, c)`, I should check if there are other X's nearby.
If there are, maybe tilt towards them. If not, maybe tilt to group them later.

Let's look at the mechanics of moving.
"Tilt forward (F): top moves down? Left moves right? Usually F means Forward (away from viewer/Up -> Down?), B means Backward (Down -> Up?).
Actually, standard conventions:
Grid coordinates $(i, j)$ where $i$ is row (0..9, top-down), $j$ is col (0..9, left-right).
F: Shift everything "forward". In a gravity context, usually F means pushing things in the +row direction (down)? Or -row (up)?
Let's check the image description or standard interpretations.
Usually, "Forward" means away from you. If screen is the front, forward might mean "into the screen" which could be increasing row index (if row 0 is front-most).
However, looking at typical physics problems or AtCoder grids:
Row 0 is usually top. Row 9 is bottom.
Tilting "Forward": Imagine holding a tray. Tilt it so gravity pulls items "forward".
If row 0 is "front", items move to higher indices?
Wait, let's assume standard coordinate system $(y, x)$ or $(r, c)$.
Usually $r=0$ is top, $r=9$ is bottom.
"Front" is usually towards $r=0$. "Back" is towards $r=9$.
Or is $r=0$ "back"?
Let's infer from logic. If I put a candy near $r=0$ and tilt "Front", does it fall off? The board stops it?
"When you tilt the box... until it reaches the edge or hits another candy."
Candy doesn't fall out; it stays inside. It just slides.
Common convention in such problems:
F: Move in direction of positive y (or row index increase). i.e., Top -> Bottom.
B: Move in direction of negative y (or row index decrease).
L: Move in direction of negative x (col index decrease). Left -> Right? No, Left means sliding "Left". So Col -> Col-1.
R: Move in direction of positive x (col index increase).
Let's verify with the provided image link description (though I can't see images).
Example explanation: "tilt the box forward ... state shown in the right figure."
Without seeing the image, I should rely on common sense or search results if available, but I am an AI.
Let's check the input format description again.
Numbering empty cells: "front-to-back and left-to-right priority".
If row 0 is front, then numbering goes $r=0, c=0$ to $r=0, c=9$, then $r=1, \dots$.
Standard "forward" usually implies movement in the direction "front". If front is $r=0$, maybe it moves to lower indices?
But physically, if you tilt a box "forward", things slide "forward".
If the box is on a table, front is towards me. Tilting forward means the back goes up, front goes down. Gravity pulls things towards the front.
So things slide from $r=9$ to $r=0$.
If "backward", things slide from $r=0$ to $r=9$.
If "left", things slide from $r=9$ (right side?) No. Left usually means direction $x=9 \to x=0$.
If "right", things slide $x=0 \to x=9$.

Wait, let's reconsider the grid indexing.
Input generator says "front-to-back".
In 3D visualization (like the web tool), Front usually means close to camera.
If row 0 is close, row 9 is far.
Tilt Forward -> Things slide towards the front (row 0).
Tilt Backward -> Things slide towards the back (row 9).
Tilt Left -> Things slide towards the left (col 0).
Tilt Right -> Things slide towards the right (col 9).

Let's try to simulate this.
I need to store the board state.
Board $10 \times 10$. Initially all empty.
Step 1: Read $f_1$.
Wait, no. First read full flavor array.
Then loop $t$ from 1 to 100.
Read $p_t$.
Find the cell corresponding to rank $p_t$ among currently empty cells.
Place candy of flavor $f_t$ there.
Perform heuristic choice of move.
Output move.
Flush.

Heuristic:
We want to maximize $\sum n_k^2$. This encourages keeping clusters of same flavor.
When we place a new candy of flavor $X$ at $(r, c)$, we should try to connect it to existing candies of flavor $X$.
Also, we should consider that tilting moves *all* candies.
Maybe we should push existing candies of flavor $X$ closer to $(r, c)$.
If we tilt 'Right', candies move Right. If there are existing $X$s to the right, maybe they align better.
But also, we want to clear space or create paths?
Actually, the most effective strategy for this specific problem (which was a popular AHC problem) is often related to maintaining "stacks" or pushing candies to the sides.
Specifically, many solutions use a simple rule: always move in the direction that brings the current candy in contact with an existing cluster of the same flavor, or if multiple choices, pick one arbitrarily.
Even simpler:
Just maintain connectivity.
If current flavor is X, check if there are adjacent X's in the final grid? No, we can't control adjacency perfectly.
But we can influence relative positions.
A common strategy found online for this specific problem (AHK015 Trick or Treat):
After placing the candy, check the neighbors. Try to push existing candies of the same flavor to touch the new one.
Since we only move *once*, the change is immediate.
Consider the 4 directions. For each direction $D \in \{F, B, L, R\}$:
Simulate the tilt (move all candies in direction $D$ until blocked).
Check the resulting configuration score or potential.
Since $N=100$, simulating full connectivity calculation (O(N log N) or O(N)) is cheap enough per step.
Total complexity per test case: $100 \times (\text{grid updates})$. Grid updates are fast. Score calc is BFS/DFS over 100 nodes. Very fast.
So I can simply evaluate the 4 possible moves and pick the best one based on the current state?
But wait, the current state includes where the *new* candy is placed.
And importantly, the *future* candies are fixed in flavor, but unknown position.
Does looking ahead help?
Actually, since we want to maximize $\sum n_i^2$, we primarily care about the current configuration because earlier decisions constrain future possibilities.
Is it strictly optimal to just greedily maximize the score *after* the move?
Let's check.
Score = round($10^6 \times \frac{\sum n_i^2}{\sum d_i^2}$).
$\sum d_i$ is constant for the whole game? Yes, because we receive exactly $f_1 \dots f_{100}$. So $\sum d_i^2$ is constant throughout the process!
So maximizing the score is equivalent to maximizing $\sum n_i^2$.
This quantity is additive: $\sum n_i^2$.
Each time we add a node to a component of size $S$, the term becomes $S+1$, contribution $(S+1)^2 - S^2 = 2S + 1$.
So adding a candy increases the score by connecting it to a component of size $S$. To maximize growth, we want to connect to the largest possible component of the same flavor.
However, we can also increase score by merging components.
Basically, the goal is to form large blocks.

Greedy Strategy:
Maximize $\sum n_i^2$ immediately after the placement and move.
To do this, we can compute the score for all 4 tilt options and pick the maximum.
Is there any risk? Maybe picking a locally optimal move prevents a better merge later?
Given 100 steps and 100 test cases, strict global optimization is impossible (NP-hard-ish due to path constraints). Heuristic is required.
Local optimization is a strong heuristic.
Since we can compute the state quickly, let's implement the "Evaluate all 4 moves" strategy.
Wait, "Evaluating all 4 moves" requires simulating the *entire* board state after 1 move.
Board state update involves shifting all candies.
Candy movement: iterate through the direction vector. For each row/col line parallel to the move axis, shift candies.
Example: Tilt Forward (assuming F is +row direction? Or -row? I'll guess).
Assume F is row index + 1 (Down).
For each column $j$: shift candies in column $j$ downwards.
Wait, F affects rows. "Forward" usually means along the Y-axis.
If F corresponds to increasing row index.
For each column, collect candies, count empty spots at bottom, fill.
Wait, F/B/L/R affect the whole board.
If F means "push everything in +r direction" (downwards), then for each column, the candies slide down.
If they hit the wall ($r=9$), they stop.
Yes.

So algorithm:
1. Parse flavor list.
2. Loop $t$ from 1 to 99. (Last step 100 has no move).
3. Read $p_t$. Determine cell $(r, c)$. Place $f_t$.
4. Calculate current board config.
5. Evaluate F, B, L, R.
   - For each dir, apply the shift to a copy of the board.
   - Compute $\sum n_i^2$ (connected components of same flavor).
6. Pick dir with max score. Output char.
7. Update main board with chosen move.

Wait, checking the example solution link text... "Show example".
It lists outputs.
Example Input: `2 2 1 3 ...`
Outputs: `R B B R F R F R R F ...`
I can't reverse engineer the exact logic without running it, but evaluating score seems robust enough for competitive programming heuristics on small grids.
Complexity:
Outer loop 100.
Inside: 4 simulations.
Each simulation:
  Update board: $O(100)$ (iterating all cells).
  Check connectivity: BFS/DFS $O(100)$.
  Total ops $\approx 100 \times 4 \times 200 \approx 80,000$.
  With 100 test cases, total $\approx 8 \times 10^6$.
  Time limit is likely generous enough (usually 2s or more for C++).
  Wait, 200 test cases in one run? "There are 200 test cases, and the score of a submission is the total score".
  Ah, "You will be given a problem statement... implement a solution...". The judge runs my code against multiple inputs. My code handles *one* interaction stream.
  So for a single test case (one flavor file), I run 100 steps.
  Wait, the judge might call my program once per test case.
  Or does it send all 200 cases in one go?
  Usually in AtCoder Interactive, it calls the program separately or sends inputs sequentially. The prompt says "There are 200 test cases".
  My code needs to handle one test case execution.
  The efficiency concern is usually if I solve *many* test cases.
  But here, for *each* testcase, I spend ~0.05s roughly. So 200 testcases is fine.

Coordinate Mapping:
$p_t$ is rank of empty cell (1-based).
Empty cells identified by iterating grid:
$r=0 \dots 9$
  $c=0 \dots 9$
    If empty, assign ID++.

Direction Vectors:
Need to define F, B, L, R correctly.
Based on "front-to-back and left-to-right priority" for numbering:
Front is likely row 0. Back is row 9.
Left is col 0. Right is col 9.
Typically, "Forward" tilt makes things slide Forward.
If Front is row 0, Forward slide means decreasing row index?
Or increasing?
If I hold a tray and tilt it "Forward" (away from body), the front end goes down. Gravity pulls objects towards the front end.
So objects move towards the front face (Face closer to me).
If row 0 is "Front", objects move towards row 0. (Decreasing r).
If row 0 is "Back", objects move towards row 0. (Decreasing r).
What if row 0 is "Front"? Then they move towards $r=0$.
Let's guess F: $r \to r-1$, B: $r \to r+1$.
L: $c \to c-1$, R: $c \to c+1$.
However, "Forward" might mean $r \to r+1$ (into the screen).
Let's look for standard conventions or try to deduce from problem phrasing "front-to-back".
"Front-to-back" numbering suggests Front is first (index 0).
So $r=0$ is Front.
If I tilt "Forward" (away from me), gravity pulls towards Front.
So objects move towards Front ($r=0$).
So F: $r \gets r-1$.
If I tilt "Backward" (towards me), gravity pulls towards Back ($r=9$).
So B: $r \gets r+1$.
L: $c \gets c-1$.
R: $c \gets c+1$.
Wait, does the problem specify direction?
"output a single character to Standard Output to indicate which direction to tilt the box... representing forward, backward, left, and right by `F`, `B`, `L`, and `R`".
Usually in 2D grid problems:
F/B is vertical. L/R is horizontal.
If I assume F decreases row index and R increases col index.
I should probably verify this or make the evaluator robust.
Since I have to submit code that works on the judge, and I can't talk to it, I have to guess or follow conventions.
Most visualizers use:
Y-axis points down (r=0 at top).
X-axis points right (c=0 at left).
Tilt "Forward": Pushes things Down (increasing y/r).
Why? Because "Forward" from the camera perspective (viewing from top) might mean "Forward" into the scene?
Actually, let's think about the phrase "front-to-back".
Usually means the dimension where 0 is front, 9 is back.
If "Forward" movement is towards "Front", then $r \to r-1$.
If "Forward" movement is "into the depth" (from $r=0$ to $r=9$?), then $r \to r+1$.
Actually, in physics terms, "Forward" is usually the direction of motion of the vehicle/box?
Wait, this is a static box on a table.
Let's consider the phrase: "tilt the box forward".
Imagine the box is like a tray. The "front" is the side facing you.
If you tilt the front edge down, gravity pulls things towards you (towards front).
So towards $r=0$ (if $r=0$ is front).
So $r \to r-1$.
Let's try to find if there's a similar problem. AtCoder "Trick or Treat".
Searching memory/knowledge base...
Many solutions use:
F: $r$ increases?
B: $r$ decreases?
Let's analyze the Example.
The example input shows $p_1=3$ -> Output `R`.
If $p_1=3$, the first 3 empty cells are occupied?
Initially board is empty. Empty cells 1..100.
Cell 1 is $(0,0)$. Cell 2 is $(0,1)$. Cell 3 is $(0,2)$.
So 1st candy placed at $(0,2)$. Flavor $f_1=2$.
Output `R`.
Tilt Right moves things $c \to c+1$ (usually).
If I tilt R, item at $(0,2)$ moves to $(0,3)$ (since 0-9 are cols).
This doesn't tell me much about direction mapping unless I know the score.
But I know $f_1=2$.
If I want to maximize clustering later, maybe moving right puts it on $(0,3)$ to meet other 2s?
If I assumed wrong direction (e.g. R means $c \to c-1$), the result would be different.
Actually, the scoring function suggests grouping.
The safest bet for "Forward" in these grid problems (especially AtCoder's own examples often imply standard screen coords):
Row 0 is Top. Row 9 is Bottom.
Col 0 is Left. Col 9 is Right.
"Tilt Forward" usually means gravity acts towards Top? Or Bottom?
Often in "gravity" games, "Drop" moves down. "Left" moves left.
But "Tilt Box Forward" sounds like physical action.
Let's look at the image filename `b639c75d_2.png`.
I cannot see it.
However, typically in 3D views:
Top face is Front?
Let's try to infer from common coding standards.
If I write a simulator, I'll define F as "Push towards row 0 (decreasing)".
Wait, standard convention: "Forward" means "Away from the observer".
If Observer looks at top of board (z-axis up).
"Front" is the face closest to the "camera".
Usually Camera looks from Top ($Z>0$).
So "Front" is the face with smaller $Z$? No.
Front is usually $Y=0$ (in 3D).
This is confusing.
Let's stick to the simplest assumption:
F: Moves in +r direction (Down/Into Screen).
B: Moves in -r direction (Up).
L: Moves in -c direction (Left).
R: Moves in +c direction (Right).
Wait, if F is Down, then "front-to-back" numbering (0..9) going from Front to Back.
If Front is $r=0$, Back is $r=9$.
If I tilt "Forward" (towards me), items slide Back ($r \to r+1$).
Wait, if I hold a tray, Front is bottom edge. Tilt forward means front edge down, items slide towards me (Front edge).
So towards $r=0$.
If Front is $r=0$.
So F: $r \to r-1$.
B: $r \to r+1$.
Let's try to verify with "front-to-back priority".
Empty cell 1 is at $r=0, c=0$ (Front-Left corner).
Empty cell 10 is $r=0, c=9$.
Empty cell 11 is $r=1, c=0$ (Behind cell 1).
Okay, let's assume this is correct.
So if I tilt "Forward", gravity pulls to Front ($r=0$). So $r \gets \max(r, \dots) - 1$? No, just shifts.
Shift logic:
For a move vector $(\Delta r, \Delta c)$, apply gravity.
If $\Delta r > 0$:
Iterate from max_r down to min_r (to avoid jumping over gaps in loop).
Collect non-empty. Fill from min_r + gap_count?
Wait, gravity fills holes.
Correct logic for moving in direction D:
Identify the lines of cells that move in that direction.
For F (assume $\Delta r = -1$):
Move items towards $r=0$.
We iterate $r$ from 0 to 9.
In each column $c$, we shift items "up" ($r \to r-1$).
The last spot to determine is $r=9$. If item at $r=9$ moves to 8, what happens at 9? Empty.
Wait, "until it reaches the edge or hits another candy."
If I tilt "Forward", things pile up at the front ($r=0$).
So items slide towards $r=0$.
Algorithm for shifting items towards direction $(-dr, -dc)$:
Iterate starting from the target edge backwards? No.
If moving towards $r=0$:
Process $r=0$: Items at $r=0$ stay.
Process $r=9$: Item at $r=9$ checks if can move to $r=8$.
Better:
Shift all valid items in the column $c$ towards $r=0$.
Start from $r=9$ down to 0.
Current pos $curr$. Next target $curr - 1$.
If $r - dr$ is free and within bounds (assuming we shift continuously), wait, "simultaneously".
But effectively, items slide until blocked.
So, count how many free slots are in the direction of motion (between current item and the wall/blocked item).
Fill that slot.
Actually, it's easier:
For each column:
  Extract list of occupied cells sorted by index.
  Insert empty cells.
  Count distance from the 'wall'.
Wait, the wall is at $r=0$ (Front).
So the stack builds up against $r=0$.
Wait, if items move towards $r=0$, they accumulate at $r=0$.
So we pack them to $r=0$.
Indices: $0, 1, \dots, k-1$ become occupied.
Wait, if I shift everyone towards $r=0$, the empty spaces are pushed towards $r=9$.
So yes, compact towards 0.
Same for others.

Okay, let's refine the heuristic.
The score is purely based on the sum of squares of component sizes.
Since $\sum d_i^2$ is constant, we just maximize $\sum n_i^2$.
Is it possible that we need to preserve potential connectivity for future moves?
Yes, but local maximization is a good baseline.
Also, we might want to break large components to make smaller ones? No, $n^2$ is convex. Larger is better.
So merging is key.
One optimization: If there's a cluster of the *same* flavor very close, maybe try to move towards it.
But simulating score covers this.

One detail: $100$ pieces. Grid $10 \times 10$.
Flavors $1, 2, 3$.
The simulation is fast.
I'll implement the class `Solver`.
Input handling:
Use `cin` to read $f_t$.
Wait, $f_t$ is given *before* the loop.
So read all $f[100]$.
Loop $t=0 \dots 99$:
  Read $p_t$.
  Map $p_t$ to $(r,c)$.
  Board[r][c] = $f[t]$.
  Evaluate moves.
  Choose best.
  Print.

Corner case: Last move $t=100$.
Prompt says: "Since nothing happens at the 100th tilt, you may skip the output."
So loop $t=0 \dots 98$. Read $p_t$, place. For $t=0 \dots 99$, output move?
"There are 100 pieces... In the $t$-th process ($1\leq t\leq 100$)... output a single character... Since nothing happens at the 100th tilt, you may skip the output."
Wait, the input $p_t$ is given for $t=1 \dots 100$.
But we output a move *after* receiving candy and *before* next step?
No, "repeat the following process 100 times."
"read $p_t$ ... output ... to indicate which direction to tilt".
This implies the move affects the current state, possibly for the next placement?
Or does it matter?
"If you tilt the box, each piece of candy moves...".
This movement changes the positions for future placements.
Does the 100th candy placement happen *after* the 99th tilt?
Sequence:
Start.
Step 1: Receive candy 1. Decide tilt 1. Tilt.
Step 2: Receive candy 2. Decide tilt 2. Tilt.
...
Step 99: Receive candy 99. Decide tilt 99. Tilt.
Step 100: Receive candy 100.
The problem says "you may skip the output" for 100th.
This means we place candy 100. We don't need to output a tilt because there's no next step?
Wait, "receive one piece... move... eventually wants to make sure...".
The score is calculated "after receiving 100 pieces of candy".
So the final configuration depends on the 99th tilt?
Yes.
So we perform 99 tilts.
We receive $p_{100}$, place candy 100. Done.
Wait, does $p_{100}$ depend on tilt 99? Yes, $p_{100}$ is index among *current* empty cells.
The set of empty cells changes after every tilt.
So we MUST output the direction for $t=1 \dots 99$.
For $t=100$, we read $p_{100}$, place candy, finish. No output needed (as per note).
Wait, the prompt says "Repeat the following process 100 times." inside which it says "After reading $p_t$ ... output".
And "Note that $p_{t+1}$ will not be given until you output the $t$-th direction."
This implies for $t=100$, we read $p_{100}$, output (nothing/skip), and then judge finishes.
Wait, if I don't output for $t=100$, does the judge read EOF?
"you may skip the output".
Okay.

So loop $t$ from 1 to 100.
Read $p_t$.
If $t < 100$:
  Find empty cell.
  Place candy.
  Evaluate moves.
  Output best move.
Else:
  Find empty cell.
  Place candy.
  Done.

Now, direction implementation details.
Assumption:
F: $r \to r-1$ (Towards Front, $r=0$).
B: $r \to r+1$ (Towards Back, $r=9$).
L: $c \to c-1$ (Towards Left, $c=0$).
R: $c \to c+1$ (Towards Right, $c=9$).

Let's double check this assumption.
If F is "Forward" (towards front), and front is $r=0$, then $r$ decreases.
If F was $r$ increases, it would be "Backward" in terms of "front-to-back" flow?
Actually, if $r=0$ is Front.
Gravity pulls things to the Low point.
If tilted forward, low point is Front.
So things go to $r=0$.
Seems consistent.

Implementation Details:
Data structure: `vector<vector<int>> board(10, vector<int>(10, 0));` 0 for empty.
Helper `get_coords(rank)` returns `{r, c}`.
To find `rank`: Iterate $r, c$. Maintain counter.
Helper `apply_move(dir, board)`: Returns new board state.
Helper `score(board)`: Returns $\sum n_i^2$.
Inside loop:
  Copy board.
  Apply F, calculate score.
  Apply B, calculate score.
  Apply L, calculate score.
  Apply R, calculate score.
  Pick max.
  
Wait, calculating score takes BFS.
With 100 pieces, BFS is instant.
Copying board takes negligible time.
So total cost is extremely low.

Wait, check the constraints on $p_t$.
$1 \le p_t \le 101 - t$.
This ensures $p_t$ refers to a valid empty cell.
Rank is assigned in "front-to-back and left-to-right priority".
My iteration order ($r=0..9, c=0..9$) matches this.

One tricky part: Flavors are integers 1..3.
$d_i$ is total number of pieces of flavor $i$.
Score formula: $\mathrm{round}(10^6 \frac{\sum n_i^2}{\sum d_i^2})$.
Since $\sum d_i^2$ is constant, we just compare $\sum n_i^2$.

Wait, does the problem allow the judge to give me different seeds?
Yes, `seed=0` in link but generally random.
My code must adapt to whatever seed.
I just read the input.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;

// Represents the grid state
int board[10][10];
// Pre-read flavors
int f[105];

// Function to compute the score based on connectivity
long long get_score(int n_flavors, int total_candies[]) {
    // Since total counts are needed for normalization, but denominator is constant
    // we can just return sum of squares of component sizes.
    // We pass the board to traverse.
    // But the function signature here is just returning the squared sum?
    // Actually, the score definition uses d_i.
    // Let's assume d_i is known globally or passed.
    // But wait, d_i changes as we add candies.
    // But d_i is fixed for the entire game (we know total counts of 1,2,3 from input).
    // Actually, let's re-read: "let d_i be the total number of pieces of candy of flavor i".
    // This refers to the final state. Since we know the full sequence f[1..100], d_i is constant.
    // So maximizing sum(n_i^2) is sufficient.
    
    // BFS/DFS to find connected components
    int visited[10][10] = {0};
    long long sum_sq = 0;
    
    for(int i=0; i<10; ++i) {
        for(int j=0; j<10; ++j) {
            if(board[i][j] != 0 && !visited[i][j]) {
                int flavor = board[i][j];
                int cnt = 0;
                queue<pair<int,int>> q;
                q.push({i,j});
                visited[i][j] = 1;
                cnt++;
                while(!q.empty()) {
                    auto curr = q.front();
                    q.pop();
                    int cx = curr.first;
                    int cy = curr.second;
                    
                    int di[] = {0, 0, 1, -1};
                    int dj[] = {1, -1, 0, 0};
                    for(int k=0; k<4; ++k) {
                        int nx = cx + di[k];
                        int ny = cy + dj[k];
                        if(nx >= 0 && nx < 10 && ny >= 0 && ny < 10) {
                            if(board[nx][ny] == flavor && !visited[nx][ny]) {
                                visited[nx][ny] = 1;
                                cnt++;
                                q.push({nx, ny});
                            }
                        }
                    }
                }
                sum_sq += (long long)cnt * cnt;
            }
        }
    }
    return sum_sq;
}

// Global counts of flavors to normalize score if needed (though unnecessary for comparison)
int d_counts[4]; // 1, 2, 3

// Move simulation
// Directions: 0=F, 1=B, 2=L, 3=R
void apply_tilt(vector<vector<int>>& b, int dir, vector<vector<int>>& result) {
    // Result initialized with b
    result = b; 
    int dr[4] = {-1, 1, 0, 0}; // F=-1(r), B=+1(r), L=0, R=0
    int dc[4] = {0, 0, -1, 1};
    
    // Note on direction assumption:
    // F (Forward) -> Row decreases (towards 0).
    // B (Backward) -> Row increases (towards 9).
    // L (Left) -> Col decreases (towards 0).
    // R (Right) -> Col increases (towards 9).
    
    // Optimization: We need to shift candies towards the target direction.
    // For F: target is r=0. Shift items towards r=0.
    // Logic: Iterate from r=0 to r=9. Collect items in col c.
    // No, "shift simultaneously". It implies gravity.
    // Correct simulation:
    // For each row/col line parallel to the walls perpendicular to gravity.
    // E.g. for F/B (changing r), iterate over columns.
    // Inside each column, items shift towards r=0 (or r=9).
    // We can simulate this by counting empty spaces at the leading edge.
    // Or just repeatedly attempt to move. Since max distance is 9, repeated check is safe.
    // But simpler: just pack them.
    // If moving to r=0 (F):
    //   In each column, extract occupied indices.
    //   These indices will shift to occupy 0, 1, ..., count-1.
    //   Wait, this is valid ONLY if the move clears the path.
    //   Yes, tilting removes all friction. They slide until stopped.
    //   Stopped by wall (edge) or other candy.
    //   So essentially they stack up against the "front" wall.
    //   For F (towards r=0), items at r=0 stay? No, r=0 is the wall.
    //   Wait, if I tilt F (front edge down), gravity pulls TO THE FRONT.
    //   So they press against the front wall (r=0).
    //   So indices 0..k-1 become occupied.
    //   Wait, if I had items at 9, 8...
    //   They move to 0, 1...
    //   So effectively, we compress all candies in that column towards the target boundary.
    //   Boundary for F is r=0.
    //   Boundary for B is r=9.
    //   Boundary for L is c=0.
    //   Boundary for R is c=9.
    
    int target_row_min = 0, target_row_max = 9;
    int target_col_min = 0, target_col_max = 9;
    
    if (dir == 0) { // F
        // Pack towards r=0
        for (int c = 0; c < 10; ++c) {
            vector<int> occupied;
            for (int r = 0; r < 10; ++r) {
                if (result[r][c] != 0) occupied.push_back(result[r][c]);
            }
            // Clear column
            for (int r = 0; r < 10; ++r) result[r][c] = 0;
            // Re-fill from r=0 upwards
            for (int k = 0; k < occupied.size(); ++k) result[k][c] = occupied[k];
        }
    } else if (dir == 1) { // B
        // Pack towards r=9
        for (int c = 0; c < 10; ++c) {
            vector<int> occupied;
            for (int r = 9; r >= 0; --r) {
                if (result[r][c] != 0) occupied.push_back(result[r][c]);
            }
            // Clear column
            for (int r = 0; r < 10; ++r) result[r][c] = 0;
            // Re-fill from r=9 downwards
            for (int k = 0; k < occupied.size(); ++k) result[9-k][c] = occupied[k];
        }
    } else if (dir == 2) { // L
        // Pack towards c=0
        for (int r = 0; r < 10; ++r) {
            vector<int> occupied;
            for (int c = 0; c < 10; ++c) {
                if (result[r][c] != 0) occupied.push_back(result[r][c]);
            }
            // Clear row
            for (int c = 0; c < 10; ++c) result[r][c] = 0;
            // Re-fill from c=0 rightwards
            for (int k = 0; k < occupied.size(); ++k) result[r][k] = occupied[k];
        }
    } else if (dir == 3) { // R
        // Pack towards c=9
        for (int r = 0; r < 10; ++r) {
            vector<int> occupied;
            for (int c = 9; c >= 0; --c) {
                if (result[r][c] != 0) occupied.push_back(result[r][c]);
            }
            // Clear row
            for (int c = 0; c < 10; ++c) result[r][c] = 0;
            // Re-fill from c=9 leftwards
            for (int k = 0; k < occupied.size(); ++k) result[r][9-k] = occupied[k];
        }
    }
}

int main() {
    // Optimize I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n = 100;
    // Read flavors
    for(int i=0; i<n; ++i) {
        cin >> f[i+1];
    }
    
    // Initialize board
    for(int i=0; i<10; ++i)
        for(int j=0; j<10; ++j)
            board[i][j] = 0;
            
    int empty_cnt = 100;
    
    // Calculate initial flavor counts
    // Not strictly needed for greedy choice of move if we use raw sum_sq,
    // but technically score uses normalized value.
    // Since norm is constant, raw sum_sq maximizes score.
    
    // We can precalculate denominator if we want, but it's constant.
    // Let's compute d_i for correctness.
    int total_d[4] = {0};
    for(int i=1; i<=n; ++i) total_d[f[i]]++;
    // Denom = sum(total_d[i]^2).
    
    // Main loop
    for(int t=1; t<=100; ++t) {
        int pt;
        cin >> pt;
        
        // Find the cell for rank pt
        int r_found = -1, c_found = -1;
        int rank = 0;
        for(int r=0; r<10; ++r) {
            for(int c=0; c<10; ++c) {
                if(board[r][c] == 0) {
                    if(rank + 1 == pt) {
                        r_found = r; c_found = c;
                    }
                    rank++;
                }
            }
        }
        
        // Place candy
        board[r_found][c_found] = f[t];
        
        if(t == 100) {
            // Final step, no output needed
            continue;
        }
        
        // Simulate moves
        long long max_score = -1;
        char best_move = 'F';
        
        struct MoveRes { long long score; char dir; };
        MoveRes moves[4];
        string dirs = "FBRL";
        
        // Simulation
        // We need to copy board for each simulation.
        // Since we modify 'board' after choosing, we need to save the board.
        // Actually we can just copy board into temp buffers.
        vector<vector<int>> temp_b;
        
        // Current board is updated already.
        // Wait, applying move changes positions.
        // We need to simulate move on current board.
        
        for(int d=0; d<4; ++d) {
            temp_b.resize(10, vector<int>(10));
            for(auto& row : temp_b) row.assign(10, 0);
            for(int i=0; i<10; ++i)
                for(int j=0; j<10; ++j)
                    temp_b[i][j] = board[i][j];
            
            apply_tilt(temp_b, d, temp_b); // Pass ref
            moves[d].score = get_score(10, temp_b); // Wait, get_score needs board passed?
            // My get_score implementation takes board directly?
            // I need to change get_score signature or make it take the buffer.
            // Let's adjust get_score.
        }
        
        // Let's refactor get_score to accept vector<vector<int>> board_ref
        // Or just inline it or use a lambda.
    }
    return 0;
}
```

Refining `get_score` and the main loop structure.
Also, I need to be careful with variable scoping.

Also, regarding "Apply Move":
If I tilt F (towards r=0), the candies shift.
If I have `[2, 1, 0, 3]` in a row.
Tilt L (towards c=0): `2` stays at 0, `1` stays at 1. `0` at 2 (empty). `3` at 3 moves to 2.
Result `[2, 1, 3, 0]`.
Correct.
My logic for applying tilt was "pack items to the wall".
Does this match "until it reaches the edge or hits another candy"?
Yes, if gravity is applied, they settle at the "lowest" energy configuration, which is packed against the wall in the direction of pull.
Wait, "simultaneously" vs sequential.
Physics: If I have two blocks with a gap, and gravity pulls both. They accelerate. Frictionless. They land on floor at same time.
If I have `[C1, gap, C2]`. Both move.
If `[C1, C2, gap]`. Gap is behind C1?
If pulling towards left (c=0).
C1 is at 0. C2 is at 2.
C1 is stuck at wall (c=0). C2 moves to 1.
Result `[C1, C2, 0]`.
My logic: `occupied` list for row. `[C1, C2]`. Re-fill `C1` at 0, `C2` at 1. Matches.
If `[gap, C1, C2]`.
C1 moves to 0. C2 moves to 1.
Result `[C1, C2, 0]`. Matches.
So "packing" is equivalent to the result of the simultaneous slide.
Because order of arrival doesn't matter for final resting positions in gravity on grid.

One small issue: `moves` initialization and score retrieval.
`apply_tilt` modifies `temp_b`.
I need to capture `temp_b` score.
The structure of the code in `main` is slightly messy. I'll clean it up.

Another thought: Is it possible that `F` corresponds to `+r` instead of `-r`?
If I submit and fail, I'd know.
But I can't resubmit here.
Let's consider the wording "front-to-back".
If I look at a matrix printed on paper.
Row 0 is top. Row 9 is bottom.
"Front" is usually top (near hand).
"Back" is bottom (far hand).
Tilt "Forward" -> Hand moves down? No, usually "Forward" means "Away from me".
In 3D, "Forward" is -Z or +Z depending on view.
But in "tilting a box", Forward usually implies the front face goes DOWN.
If front face goes down, gravity pulls things towards the front face.
So towards Row 0.
So F -> `r` decreases.
This seems the most plausible interpretation of "Front-to-back" numbering (1st is Front).

Let's check the example input provided in the problem statement (partially visible).
Input: `2 2 1 3 ...`
Example Output: `R B B R F ...`
Let's trace manually if possible? Hard.
But I can check if `R` makes sense for `2`.
Wait, if `f[1]=2`. Placed at $(0,2)$.
Output `R`.
If `R` means `c` increases.
Candy moves to $(0,3)$.
If `R` meant `c` decreases, candy moves to $(0,1)$.
Without knowing the rest of the state or score, it's hard to be 100% sure.
However, typically `L`/`R`/`F`/`B` map to standard compass directions.
`R`: Right. `F`: Forward.
In screen coordinates:
Right -> +X
Forward -> -Y (up) OR +Y (down)?
If Y is down, "Forward" might mean +Y (into screen).
If Y is up, "Forward" might mean -Y (out of screen).
In matrix/grid problems, usually (0,0) is top-left.
So Y is down.
If Forward means "away from user", and user looks from top (so sees top face), then "Forward" is into the screen -> +Y (down).
So F -> `r` increases.
Contradicts previous "Front-to-back" logic.
Which clue is stronger?
"Front-to-back and left-to-right priority".
If I say "Front is 0". Then "Back is 9".
If "Forward" means "Towards Front", then `r` decreases.
If "Forward" means "Towards the Front face", but "Forward" usually describes the direction *of motion*.
Motion "Forward" in English means towards the front.
So towards $r=0$.
Let's stick with F: `r` decreases.
Wait, let's reconsider "Screen coordinates".
Usually in games (like Tetris), "Forward" isn't used. "Down" is used.
In "Trick or Treat", the box is a cube? "Square box".
Visualizer link mentions "Web version" showing animations.
If I recall similar problems, usually `F` pushes items to the *bottom* of the screen (index 9).
Wait, why?
Because `Forward` means "Push things in the direction of 'Forward'".
If `F` is associated with `r`, and `r` goes `0->9`.
Maybe `0` is Back and `9` is Front?
Then "front-to-back" numbers from 9 down to 0? No, usually 0-indexed arrays are 0..9.
If 0 is Front, then "Forward" moves to 0.
If 9 is Front, "Forward" moves to 9.
Let's assume the numbering defines the geometry.
1 is Front-Left. 101 is Back-Right.
So Row 0 is Front. Row 9 is Back.
If I tilt "Forward", I tilt the box so the "Front" side drops.
Items slide towards the Front side.
So to Row 0.
Okay, I'm fairly confident F means Row 0.
Wait, if I hold a tray and tilt forward (away from me), the front edge goes down.
Gravity pulls things towards the front edge.
If "Front" is Row 0. Then things go to Row 0.
Matches.
But what about "Front" being the side "Closest to me"?
Yes, Row 0.
So F -> $r \to r-1$.
And R -> $c \to c+1$.
L -> $c \to c-1$.
B -> $r \to r+1$.

Wait, I should confirm the direction characters.
Problem: "representing forward, backward, left, and right by `F`, `B`, `L`, and `R`, respectively".
This maps words to chars.
Standard:
F -> Front (Row 0)
B -> Back (Row 9)
L -> Left (Col 0)
R -> Right (Col 9)

Wait, if I tilt *Right*, the Right side goes down. Gravity pulls things to the Right side.
So things move to High Col Index (Col 9).
If I tilt *Left*, Left side goes down. Things move to Col 0.
This matches my logic.
Wait, what if "Tilt Right" means rotate box to the right?
No, "tilt... forward, backward, left, or right... box moves in that direction".
So Tilt Right -> Motion Right.
So R -> Move to Col 9.
L -> Move to Col 0.
F -> Move to Front (Row 0).
B -> Move to Back (Row 9).
This seems solid.

Final check on logic.
Code structure:
- Includes.
- Fast IO.
- Read `f`.
- Loop 100 times.
- Read `p_t`.
- Identify location.
- Place candy.
- If t==100 exit loop (no output).
- Else:
  - Construct copies.
  - Apply F,B,L,R.
  - Calc score.
  - Pick best.
  - Output.
  - `board` is NOT updated by the copy, only simulation.
  - BUT wait! The simulation is just a prediction.
  - Does the actual tilt happen automatically?
  - "By writing a program to determine the direction to tilt the box."
  - And "you must tilt the box...".
  - This implies the judge (interaction) expects me to choose a direction.
  - Does the judge AUTOMATICALLY tilt the box according to my choice?
  - Yes. "Since nothing happens at the 100th tilt...".
  - So I choose a direction, the judge performs the tilt on their internal state, and gives me the next `p`.
  - I DO NOT need to update my `board` state after choosing. The judge maintains the state.
  - WAIT. This is crucial.
  - If I don't track the state of the board on MY SIDE, how can I simulate the moves and predict the outcome of the next steps?
  - I MUST update my internal representation of the board state to match what the judge does.
  - So after I choose a direction, I must apply THAT tilt to MY `board` for the NEXT step.
  - Otherwise my predictions for the future will be wrong (based on stale data).
  - The prompt says "Since you cannot change the order... and it will be chosen uniformly at random among the empty cells."
  - The `p_t` depends on the *current* empty cells.
  - The empty cells depend on the *previous* tilts.
  - So YES, I must maintain the state accurately.
  - Wait, but the problem says "p_t is generated by rand...".
  - Wait. "Each $p_t$ is generated by rand(1, 101-t)."
  - "generated by" -> Is it truly random?
  - "On the other hand, we do not know in advance to which cell each candy will be placed, and it will be chosen uniformly at random among the empty cells."
  - AND "You cannot change the order...".
  - This implies the *position* is determined randomly, independent of the tilt?
  - NO. "chosen uniformly at random among the empty cells."
  - Wait, if the position is RANDOM, then my tilt decisions *don't affect* where the candy lands?
  - If $p_t$ is chosen uniformly at random from empty cells *regardless* of the current board configuration?
  - No, "among the empty cells". The set of empty cells IS determined by previous moves.
  - But WHICH ONE is picked?
  - "randomly".
  - This means the judge picks a cell. I don't know which one.
  - BUT, the judge provides $p_t$ to me.
  - So I learn the location AFTER the fact.
  - However, for the FUTURE steps ($t+1$), the judge needs to know the board state to generate $p_{t+1}$?
  - Or is $p_t$ just random?
  - If $p_t$ is purely random uniform, then the set of *possible* next states is fixed.
  - Wait. "Let us number the empty cells from 1 to 101-t... Then the t-th piece of candy is placed in the p_t-th empty cell."
  - The numbering depends on the empty cells.
  - Empty cells depend on previous placements and TILTS.
  - So the set of empty cells depends on the history of TILTS.
  - So my choice of tilt DOES affect the configuration of empty cells for the NEXT turn.
  - Which affects the randomness distribution of $p_{t+1}$?
  - Wait. $p_{t+1}$ is a random index $1 \dots K$.
  - So the *value* of $p_{t+1}$ is just a number.
  - But its *meaning* (which physical cell it corresponds to) depends on the board state.
  - So my knowledge of board state is necessary to interpret $p_{t+1}$ and plan for $t+2$.
  - BUT, $p_t$ is given to me at step $t$.
  - Wait. The problem asks me to "determine the direction to tilt".
  - If the position is random, does my tilt affect the *probability* of getting a good arrangement later?
  - Yes, it determines the *configuration* of the board.
  - Even if I pick $p_t$ randomly (it's given), the board orientation (clustering) depends on previous tilts.
  - Wait. The "random" nature of $p_t$ is determined by the judge *after* I tilt?
  - "On the other hand, we do not know in advance...".
  - "generate solution code".
  - The judge generates $f_t$ and $p_t$.
  - If I provide the direction $D_t$, does the judge use my $D_t$ to update the board, then generate $p_{t+1}$ based on *that* board?
  - Yes.
  - So my board state must be updated.
  - Wait, the example output `Show example` is linked.
  - It's an interactive problem.
  - The judge waits for my output. Then proceeds.
  - So yes, I must maintain state and update it.
  - But wait. The prompt says "$p_t$ is generated by `rand`".
  - And "we do not know in advance to which cell...".
  - This means $p_t$ is NOT determined by my tilt $D_{t-1}$. It's just random selection.
  - So for step $t$, $p_t$ is a random variable from the set of empty cells.
  - My choice of $D_{t-1}$ determines the geometric arrangement of those empty cells (and the candies).
  - So even though I don't control *where* the candy lands (that's random), I control *where* the candies are relative to each other, which helps future connections.
  - Okay. So yes, updating board state is mandatory.

Wait, the example link mentions `output=R%.0D%0AB%0D...`
It suggests I output chars.
Okay.
Wait, there is a catch.
"The score for the test case is ..."
"There are 200 test cases".
So I need to solve for EACH test case.
The input stream contains 1 test case per run.
Or maybe the input stream contains 1 test case, and I output 100 chars.
Wait. "First, the flavor of each piece... Given from Standard Input".
"After reading the above information, repeat the following process 100 times."
This structure implies ONE run of the program processes ONE test case.
The "200 test cases" comment refers to the judging system running my solution 200 times.
So my code reads one set of $f$, loops 100 times.
Wait, does it loop 100 times per testcase? Yes.
Okay, my logic holds.
Wait, "Note that $p_{t+1}$ will not be given until you output the $t$-th direction."
This confirms interactive nature.

Is there any dependency I missed?
No.

Double check: "Note that $p_{t+1}$ will not be given until you output the $t$-th direction."
This means for step 1, I output direction for step 1. THEN I get $p_2$?
No.
"In the $t$-th process ($1\leq t\leq 100$)... read $p_t$... output direction".
So for $t=1$, I read $p_1$, place candy. Output $D_1$.
Then for $t=2$, I read $p_2$...
Wait.
Is $p_1$ given before I do anything?
"First, the flavor... After reading... repeat process 100 times."
"process 100 times":
  Read $p_t$.
  Place.
  Output $D_t$ (if not last).
So I read $p_t$ first.
Wait.
If I place candy $t$, it's at a random spot.
Then I tilt.
This changes the board for step $t+1$.
Wait. Does the placement of $t$ happen *after* tilt $t-1$?
Yes.
Does $p_t$ account for tilt $t-1$?
The set of empty cells is defined by board state.
Board state depends on tilt $t-1$.
Wait, but I don't know the random spot $p_t$ UNTIL it's given.
So I have to wait for $p_t$ input.
Then I know where candy $t$ is.
Then I decide tilt $t$.
Then I output tilt $t$.
This determines the state for $t+1$.
Correct.

Wait. The "score calculation" for step $t$ shouldn't be affected by tilt $t$?
Wait. The score is defined on the FINAL state after 100 pieces.
But intermediate moves affect where pieces are.
My greedy strategy maximizes $\sum n_i^2$ immediately.
Since $n_i$ depends on final arrangement, and arrangement depends on history, greedy is a decent proxy.
But wait.
At step $t$, I output $D_t$.
The actual board state *on the judge side* is updated.
I must mirror this update in my code.
So:
Read $p_t$.
Get cell $(r,c)$.
Place $f_t$.
Choose $D \in \{F,B,L,R\}$ based on score simulation of $D$.
Print $D$.
Update local board using $D$.
Repeat.
At $t=100$, read $p_{100}$, place $f_{100}$. Finish.
Wait. Do I need to update board at $t=100$?
No, I stop.
But wait, for $t=100$, I read $p_{100}$.
The empty cells at this moment are the result of tilt 99.
So my board state (after tilt 99) must be consistent with the judge's empty cells.
Since I maintain the board consistently (by applying tilt $t$ after step $t$), my empty cells list should match the judge's list when I process $t=100$.
Wait. The $p_t$ given is an index into the list of *currently* empty cells.
My list of empty cells is derived from my board.
Is it guaranteed to match?
Yes, because I simulated the exact same operations.
One subtlety: When I "simulate" a move (tilt), I modify the board.
But when I make a decision at step $t$, I haven't decided yet.
Wait.
Order of events:
1. Judge has Board state (initially empty).
2. Step $t$:
   - Judge calculates set of empty cells on its board.
   - Randomly selects index $p_t$.
   - Sends $p_t$ to me.
   - Receives $f_t$ (pre-loaded).
   - Places candy at cell $(r,c)$ corresponding to $p_t$.
   - Asks me to choose Direction $D_t$.
   - Applies $D_t$ to its Board.
   - Updates empty cells.
3. Repeat for $t=2 \dots 99$.
4. $t=100$:
   - Send $p_{100}$.
   - Place candy.
   - Judge stops.

My Code Order:
1. Init my board (empty).
2. Loop $t=1 \dots 100$.
   - Read $p_t$.
   - Find $(r,c)$ in my board.
   - Place $f_t$.
   - If $t < 100$:
     - Simulate 4 moves on my board (copy current state).
     - Pick best $D$.
     - Print $D$.
     - Apply $D$ to my board.
   - Else ($t=100$):
     - Just consume input/place.
Wait. I need to be careful.
When I pick $D$ based on "simulated board", I am predicting where the candy $t$ lands and how it interacts.
Is it possible that my prediction of the board state *after* placing $f_t$ is wrong?
No, because $f_t$ is placed at $(r,c)$ which is derived from $p_t$.
Wait. The judge places $f_t$ at $(r,c)$.
Then I tilt.
So my board must reflect:
- $f_t$ placed at $(r,c)$.
- Then shifted by $D$.
Correct.
My simulation code does exactly this.
It takes current board (with $f_t$ at $(r,c)$), applies tilt, calculates score.
So my logic is sound.

One minor thing:
"Note that $p_{t+1}$ will not be given until you output the $t$-th direction."
Wait. This implies for $t=100$, I don't need to output.
Does the judge give $p_{100}$? Yes.
Do I need to read $p_{100}$?
Yes, "repeat the following process 100 times".
In the loop:
  Read $p_t$.
  Place $f_t$.
  If $t<100$ output $D$.
  Else continue.
So for $t=100$, I read $p_{100}$ and place candy.
Wait, the prompt says "Since nothing happens at the 100th tilt, you may skip the output."
Does this mean I don't *have* to read $p_{100}$?
The loop says "repeat 100 times".
In the $t$-th process, $p_t$ is given.
So I must read $p_{100}$.
If I don't read $p_{100}$, the stream has garbage or next judge data (if any).
Reading $p_{100}$ is safer.

Let's check the timing.
If I output after reading $p_t$ but before reading $p_{t+1}$, it's synchronous.
Interactive problems usually require flushing stdout.
I'll use `cout << endl` or `flush` which `endl` does.
Fast IO sync off might interfere with `cin/cout` buffering?
Better to turn sync on or use manual flush. `cin.tie(NULL)` is ok with `endl` flushing.
`endl` flushes. `\n` might not.
I will use `endl` to be safe.

Wait, input format:
Line 1: $f_1 \dots f_{100}$.
Next 100 tokens: $p_t$.
Actually "repeat the following process 100 times" -> in each step read $p_t$.
Since it says "standard input", it might be all in one stream.
So reading $p_t$ is straightforward.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace std;

// Globals
int board[10][10];
int f[105];

long long get_component_sum_squares(int** vis, int cur_board[10][10]) {
    // We need to pass the board.
    // Let's rewrite to work with local var or reference
    return 0; // Placeholder
}

// Helper to calculate score for a board configuration
// board is 10x10 grid. 0=empty.
long long calculate_score(int b[10][10]) {
    long long sum_sq = 0;
    bool visited[10][10] = {false};
    int dr[] = {0, 0, 1, -1};
    int dc[] = {1, -1, 0, 0};
    
    for(int i=0; i<10; ++i) {
        for(int j=0; j<10; ++j) {
            if(b[i][j] != 0 && !visited[i][j]) {
                int color = b[i][j];
                int sz = 0;
                queue<pair<int,int>> q;
                q.push({i,j});
                visited[i][j] = true;
                sz++;
                
                while(!q.empty()) {
                    pair<int,int> u = q.front(); q.pop();
                    int ni = u.first, nj = u.second;
                    
                    for(int k=0; k<4; ++k) {
                        int ni2 = ni + dr[k];
                        int nj2 = nj + dc[k];
                        if(ni2>=0 && ni2<10 && nj2>=0 && nj2<10) {
                            if(!visited[ni2][nj2] && b[ni2][nj2] == color) {
                                visited[ni2][nj2] = true;
                                sz++;
                                q.push({ni2, nj2});
                            }
                        }
                    }
                }
                sum_sq += (long long)sz * sz;
            }
        }
    }
    return sum_sq;
}

// Apply tilt to a temporary board
// Returns void, modifies temp_board
void tilt_board(int temp_board[10][10], int dir) {
    // dir: 0=F, 1=B, 2=L, 3=R
    // F: towards r=0
    // B: towards r=9
    // L: towards c=0
    // R: towards c=9
    
    if (dir == 0) { // F: towards r=0
        for(int c=0; c<10; ++c) {
            vector<int> cols[10];
            vector<bool> present(10); // check? No, just iterate
            int r_occ = 0;
            for(int r=0; r<10; ++r) {
                if(temp_board[r][c] != 0) {
                    // store?
                }
            }
            // Packing logic:
            // Collect items
            vector<int> items;
            for(int r=0; r<10; ++r) {
                if(temp_board[r][c] != 0) items.push_back(temp_board[r][c]);
            }
            // Shift to r=0..items.size()-1
            // Wait, items are collected in order of appearance (r=0, then r=1...)
            // If I have [2, 1] at r=0, r=1.
            // Items: {2, 1}.
            // Target r=0 is first.
            // So temp_board[0][c] = 2, temp_board[1][c] = 1.
            // Correct.
            for(int r=0; r<10; ++r) temp_board[r][c] = 0;
            for(int k=0; k<items.size(); ++k) {
                temp_board[k][c] = items[k];
            }
        }
    } else if (dir == 1) { // B: towards r=9
        for(int c=0; c<10; ++c) {
            vector<int> items;
            for(int r=9; r>=0; --r) { // Collect from bottom
                if(temp_board[r][c] != 0) items.push_back(temp_board[r][c]);
            }
            for(int r=0; r<10; ++r) temp_board[r][c] = 0;
            for(int k=0; k<items.size(); ++k) {
                // Put at r=9-k
                temp_board[9-k][c] = items[k];
            }
        }
    } else if (dir == 2) { // L: towards c=0
        for(int r=0; r<10; ++r) {
            vector<int> items;
            for(int c=0; c<10; ++c) {
                if(temp_board[r][c] != 0) items.push_back(temp_board[r][c]);
            }
            for(int c=0; c<10; ++c) temp_board[r][c] = 0;
            for(int k=0; k<items.size(); ++k) {
                temp_board[r][k] = items[k];
            }
        }
    } else if (dir == 3) { // R: towards c=9
        for(int r=0; r<10; ++r) {
            vector<int> items;
            for(int c=9; c>=0; --c) {
                if(temp_board[r][c] != 0) items.push_back(temp_board[r][c]);
            }
            for(int c=0; c<10; ++c) temp_board[r][c] = 0;
            for(int k=0; k<items.size(); ++k) {
                temp_board[r][9-k] = items[k];
            }
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n = 100;
    for(int i=0; i<n; ++i) {
        cin >> f[i+1];
    }
    
    // Initialize board to 0
    for(int i=0; i<10; ++i)
        for(int j=0; j<10; ++j)
            board[i][j] = 0;
            
    for(int t=1; t<=100; ++t) {
        int pt;
        cin >> pt;
        
        // Identify position
        int r, c;
        int cnt = 0;
        for(int i=0; i<10; ++i) {
            for(int j=0; j<10; ++j) {
                if(board[i][j] == 0) {
                    cnt++;
                    if(cnt == pt) {
                        r = i; c = j;
                    }
                }
            }
        }
        
        // Place candy
        board[r][c] = f[t];
        
        if(t == 100) continue;
        
        // Evaluate moves
        long long max_score = -1;
        char best_dir = 'F';
        
        // Temp buffers for simulation
        // We can reuse one buffer but copying is safer/easier
        int temp[10][10];
        
        // Directions: F, B, L, R -> 0, 1, 2, 3
        // Corresponding output chars: 'F', 'B', 'L', 'R'
        // Wait, I mapped F=0, B=1, L=2, R=3
        // Need to map index to char.
        
        int dir_chars[] = {'F', 'B', 'L', 'R'};
        int dir_indices[4] = {0, 1, 2, 3}; 
        
        // We need to simulate on a deep copy
        // Let's just make a copy of current board
        int current_copy[10][10];
        for(int i=0; i<10; ++i) {
            for(int j=0; j<10; ++j) {
                current_copy[i][j] = board[i][j];
            }
        }
        
        // Try all 4
        for(int d=0; d<4; ++d) {
            // Copy current state to sim board
            for(int i=0; i<10; ++i)
                for(int j=0; j<10; ++j)
                    current_copy[i][j] = board[i][j]; // Using global 'board' which is updated
            
            // Apply tilt
            tilt_board(current_copy, d);
            
            // Calculate score
            long long s = calculate_score(current_copy);
            
            if(s > max_score) {
                max_score = s;
                best_dir = dir_chars[d];
            }
        }
        
        cout << best_dir << "\n"; // Use \n for speed, but ensure flush?
        // endl flushes. With sync_with_stdio(false), endl still flushes?
        // "Otherwise, the submission might be judged as TLE."
        // Usually \n is fine if buffer is flushed.
        // To be safe: cout << best_dir << endl; 
        // But endl is slow. Given 100 lines, it's fine.
        // Wait, strict flushing: "must be followed by a new line, and you have to flush".
        // endl flushes.
        
        // Now update the real board with the chosen direction?
        // Wait, did the judge tilt?
        // "By writing a program to determine the direction to tilt the box."
        // The judge tilts based on my direction.
        // So I should update my board to match the judge's state for the next step.
        
        // Apply best_dir to 'board'
        // Find index
        int idx = 0;
        if(best_dir == 'F') { idx = 0; } // Assuming mapping F->0 etc?
        else if(best_dir == 'B') idx = 1;
        else if(best_dir == 'L') idx = 2;
        else if(best_dir == 'R') idx = 3;
        
        tilt_board(board, idx);
    }
    
    return 0;
}
```

Re-checking the mapping between chars and my internal logic.
Chars: F, B, L, R.
Mapping: F(0), B(1), L(2), R(3).
Logic inside `tilt_board`:
Dir 0 (F): $r \to 0$. (Packs to $r=0$).
Dir 1 (B): $r \to 9$. (Packs to $r=9$).
Dir 2 (L): $c \to 0$. (Packs to $c=0$).
Dir 3 (R): $c \to 9$. (Packs to $c=9$).
This matches "Tilt Forward -> Moves Forward".
Wait. If I tilt Forward, the box tip moves away from me.
The gravity direction is towards the front.
Front is Row 0.
So items move towards Row 0.
So Dir 0 (F) packs to $r=0$.
This seems correct.
Wait, is `F` definitely Row 0?
If `F` was Row 9, then items pack to $r=9$.
But `front-to-back` numbering:
Usually Front is $r=0$ (closer to player in 3D view).
So F moves to $r=0$.
Let's stick with this. It's the most logical for "Front".

Wait, the example input again.
$f_1 = 2$. $p_1 = 3$.
$c_1=(0,0), c_2=(0,1), c_3=(0,2)$.
Placed at $(0,2)$.
Output R.
If R is $c \to 9$ (pack to 9).
Items in row 0: `2` at col 2.
If I pack to 9, `2` moves to col 9?
If row 0 is `[0,0,2,0...]`.
Pack to 9: `[0,0,0,0...2]`. `2` moves to 9.
Result row 0: `[..., 2]`.
Is this good?
If I tilt R, maybe I'm trying to put it in the corner?
Or maybe `R` means something else.
But without knowing the judge's answer key, it's hard to debug.
But wait.
What if R means "Rotate Clockwise"? No, it says "Tilt... forward, backward...".
The problem says "representing forward, backward, left, and right by `F`, `B`, `L`, and `R`, respectively".
And "box moves in that direction".
So it's translation of the contents.
Okay, I'll stick with translation.

One more check: `ios_base::sync_with_stdio(false); cin.tie(NULL);`
Does `endl` work properly with `NULL` tied stream?
Yes, `endl` explicitly flushes the stream regardless of tie status (actually tie only affects unbuffered writes when input is read, but flush works).
But `cout << "\n"` is preferred for speed.
However, strict requirement: "flush".
`endl` does it. I'll use `endl`.
Actually, for 100 lines it doesn't matter.

Wait, `current_copy` initialization inside loop.
Inside loop I declare `int current_copy[10][10];` uninitialized.
I need to initialize it from `board`.
My code does:
```cpp
        int current_copy[10][10];
        for(int i=0; i<10; ++i) {
            for(int j=0; j<10; ++j) {
                current_copy[i][j] = board[i][j];
            }
        }
```
This is done AFTER placing the new candy.
So `board` has the new candy. `current_copy` has the new candy.
Then simulate tilt on copy.
Then pick best.
Then apply best tilt to `board`.
This logic preserves the state update.
Looks correct.

One edge case: `best_dir` initialization.
If all scores are equal?
Pick 'F' (first one). That's fine.

Complexity:
100 steps.
Inside: 4 copies + 4 BFS.
Grid size 100.
Operations $\approx 100 \times 4 \times 100 \approx 40,000$.
Total ops minimal. Memory minimal.
This solution is efficient enough.

Wait, "Note that $p_{t+1}$ will not be given until you output the $t$-th direction."
My code outputs inside the loop.
For $t=100$, it continues (skips output).
Wait.
If I don't output for $t=100$, the judge will just finish.
But wait.
The judge waits for my output BEFORE giving $p_{t+1}$.
For $t=100$, there is no $p_{101}$.
So output is not needed.
Wait. "In the $t$-th process ($1\leq t\leq 100$)... output".
"Since nothing happens at the 100th tilt, you may skip the output."
This confirms.

Wait, the input $f$ is 1..100.
Array `f` is size 105.
$f[1]$ is first candy.
Input reading `cin >> f[i+1]` for `i=0..99`.
So `f[1]` is first. Correct.

One final verification on Direction Mapping.
Some sources for AHC problems suggest:
F: Top -> Down (r increases).
B: Bottom -> Top (r decreases).
If that were the case:
F (Forward) -> Into screen (Down).
B (Backward) -> Out of screen (Up).
This is the "Camera View" convention (Forward means further away, which is Down in screen).
But "Front" means $r=0$.
"Front-to-back" numbering.
If "Front" is $r=0$.
Does "Tilt Forward" mean "Tilt towards Front" ($r=0$) or "Tilt towards Depth" ($r=9$)?
In physics "Forward" usually means direction of travel.
In a static box, it usually means the side labeled "Front".
If side "Front" is $r=0$. Then Tilt Forward means "Pull towards Front" ($r=0$).
However, if I assume "Screen Coordinates" (Row 0 is Top).
Usually "Forward" in games is "Up" or "Down"?
In 2D plane, Forward usually means decreasing Y?
If Y is "Depth".
If I stand in front of the box, "Forward" is towards the box?
This is ambiguous.
But let's check the constraint "front-to-back".
If $p_t$ numbers "front-to-back".
So $p=1$ is Front-most.
If I tilt "Forward", gravity pulls things towards the Front.
So things go to $r=0$ (Front).
So F -> $r \to r-1$.
This seems the most consistent interpretation of the "Front-to-back" hint.
If "Front" was the far side (Row 9), then "Front-to-back" numbering would start from 9? No, usually 1 is Front.
So Front is Row 0.
So F pulls to Row 0.

Wait, what if `F` pulls to Row 9?
Then F -> $r \to r+1$.
Is there any reason to suspect this?
Some visualizers show "Forward" as "Forward into the scene".
If the scene is a 3D box. Front face is at $z=0$ (close to camera). Back face at $z=10$.
Gravitation pulls things "Forward" (closer to camera, towards $z=0$).
In projection, $z$ maps to Y?
If $z$ decreases (goes towards camera), does it correspond to $r$ decreasing?
Usually, Top row ($r=0$) corresponds to Closest part (Front).
Bottom row ($r=9$) corresponds to Farthest part (Back).
So Forward (towards Front) means towards $r=0$.
So my assumption holds.

Final check on syntax errors.
`vector<int> cols[10];` declaration inside loop?
Inside `tilt_board`, vectors are created dynamically.
Using `vector<int>` is fine.
Headers included.
`pair` used in BFS, need `<utility>`? Included in `<vector>` or `<iostream>` usually, but `<utility>` is safer. I used `<queue>` which might include it.
Added `<utility>` implicitly via `vector`/`algorithm` or explicit.
Actually `pair` is in `<utility>`.
I should add `#include <utility>`.

```cpp
#include <utility>
```
And `tuple`? No.

One detail: `best_dir` is char. `dir_chars` is array of char.
`idx` logic:
If `best_dir` is 'F', `idx`=0.
My loop iterates `d` from 0 to 3.
So `dir_chars[0]` is 'F'.
Logic matches.

Code looks ready.
</think>

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>
#include <utility>

using namespace std;

// Global variables to store board and input flavors
int board[10][10];
int f[105];

// Function to calculate the score based on connectivity
// Returns the sum of squares of connected component sizes
long long calculate_score(int b[10][10]) {
    long long sum_sq = 0;
    bool visited[10][10] = {false};
    int dr[] = {0, 0, 1, -1};
    int dc[] = {1, -1, 0, 0};
    
    for(int i=0; i<10; ++i) {
        for(int j=0; j<10; ++j) {
            if(b[i][j] != 0 && !visited[i][j]) {
                int color = b[i][j];
                int sz = 0;
                queue<pair<int,int>> q;
                q.push({i,j});
                visited[i][j] = true;
                sz++;
                
                while(!q.empty()) {
                    pair<int,int> u = q.front(); q.pop();
                    int ni = u.first;
                    int nj = u.second;
                    
                    for(int k=0; k<4; ++k) {
                        int ni2 = ni + dr[k];
                        int nj2 = nj + dc[k];
                        if(ni2>=0 && ni2<10 && nj2>=0 && nj2<10) {
                            if(!visited[ni2][nj2] && b[ni2][nj2] == color) {
                                visited[ni2][nj2] = true;
                                sz++;
                                q.push({ni2, nj2});
                            }
                        }
                    }
                }
                sum_sq += (long long)sz * sz;
            }
        }
    }
    return sum_sq;
}

// Function to simulate tilting the board
// Modifies the board passed by reference
void tilt_board(int board[10][10], int dir) {
    // Directions mapping:
    // 0 = F (Forward): Towards Row 0 (Front)
    // 1 = B (Backward): Towards Row 9 (Back)
    // 2 = L (Left): Towards Col 0 (Left)
    // 3 = R (Right): Towards Col 9 (Right)
    // Assumption based on "Front-to-back" numbering (Row 0 is Front).
    
    if (dir == 0) { // F: Pack towards r=0
        for(int c=0; c<10; ++c) {
            vector<int> items;
            for(int r=0; r<10; ++r) {
                if(board[r][c] != 0) items.push_back(board[r][c]);
            }
            // Clear column and refill from r=0
            for(int r=0; r<10; ++r) board[r][c] = 0;
            for(int k=0; k<items.size(); ++k) board[k][c] = items[k];
        }
    } else if (dir == 1) { // B: Pack towards r=9
        for(int c=0; c<10; ++c) {
            vector<int> items;
            for(int r=9; r>=0; --r) {
                if(board[r][c] != 0) items.push_back(board[r][c]);
            }
            for(int r=0; r<10; ++r) board[r][c] = 0;
            for(int k=0; k<items.size(); ++k) board[9-k][c] = items[k];
        }
    } else if (dir == 2) { // L: Pack towards c=0
        for(int r=0; r<10; ++r) {
            vector<int> items;
            for(int c=0; c<10; ++c) {
                if(board[r][c] != 0) items.push_back(board[r][c]);
            }
            for(int c=0; c<10; ++c) board[r][c] = 0;
            for(int k=0; k<items.size(); ++k) board[r][k] = items[k];
        }
    } else if (dir == 3) { // R: Pack towards c=9
        for(int r=0; r<10; ++r) {
            vector<int> items;
            for(int c=9; c>=0; --c) {
                if(board[r][c] != 0) items.push_back(board[r][c]);
            }
            for(int c=0; c<10; ++c) board[r][c] = 0;
            for(int k=0; k<items.size(); ++k) board[r][9-k] = items[k];
        }
    }
}

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int n = 100;
    // Read the flavor of each candy
    for(int i=0; i<n; ++i) {
        cin >> f[i+1];
    }
    
    // Initialize board (empty grid)
    for(int i=0; i<10; ++i) {
        for(int j=0; j<10; ++j) {
            board[i][j] = 0;
        }
    }
    
    // Process each step
    for(int t=1; t<=100; ++t) {
        int pt;
        cin >> pt;
        
        // Find the coordinates of the pt-th empty cell
        int r_target = -1, c_target = -1;
        int rank = 0;
        for(int i=0; i<10; ++i) {
            for(int j=0; j<10; ++j) {
                if(board[i][j] == 0) {
                    rank++;
                    if(rank == pt) {
                        r_target = i;
                        c_target = j;
                    }
                }
            }
        }
        
        // Place the candy
        board[r_target][c_target] = f[t];
        
        // Skip output and update for the last step (t=100)
        if(t == 100) continue;
        
        // Greedy evaluation: try all 4 directions, pick the one maximizing connectivity score
        char best_move = 'F';
        long long max_score = -1;
        
        int dir_char_map[] = {'F', 'B', 'L', 'R'};
        int dir_index_map[4] = {0, 1, 2, 3};
        
        // Make a copy of the current board state for simulation
        int sim_board[10][10];
        for(int i=0; i<10; ++i)
            for(int j=0; j<10; ++j)
                sim_board[i][j] = board[i][j];
        
        // Evaluate F, B, L, R
        for(int d=0; d<4; ++d) {
            // Deep copy board to simulate
            for(int i=0; i<10; ++i)
                for(int j=0; j<10; ++j)
                    sim_board[i][j] = board[i][j];
            
            // Apply tilt
            tilt_board(sim_board, d);
            
            // Calculate score
            long long score = calculate_score(sim_board);
            
            if(score > max_score) {
                max_score = score;
                best_move = dir_char_map[d];
            }
        }
        
        // Output the best move
        cout << best_move << endl;
        
        // Update our internal board state to match the judge's state after the move
        // We need to apply the chosen direction to our board
        // Mapping back direction char to index for tilt_board function
        int apply_idx = 0;
        switch(best_move) {
            case 'F': apply_idx = 0; break;
            case 'B': apply_idx = 1; break;
            case 'L': apply_idx = 2; break;
            case 'R': apply_idx = 3; break;
            default: apply_idx = 0; break; // Should not happen
        }
        tilt_board(board, apply_idx);
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=1482)

````
Let me analyze this problem:

1. We have a 10x10 grid where we receive candies one at a time (100 times)
2. Each candy has a flavor (1, 2, or 3)
3. After each candy is received, the position `p_t` of where it lands becomes known, and then we must choose a tilt direction (F/B/L/R)
4. When tilting, all candies move in that direction until they hit an edge or another candy
5. Score = round(10^6 × Σ(n_i²) / Σ(d_i²)) where n_i are component sizes and d_i are total candies of flavor i

The goal is to maximize the score by choosing good tilt directions.

Key observations:
- Tilting moves ALL candies toward the direction
- This changes the relative positions of different flavors
- To get high score, we want large connected components of same-flavor candies adjacent together
- Flavors with more candies will contribute more to the denominator, so getting them clustered is especially important

Strategy:
1. Count total candies of each flavor
2. Calculate potential benefit of each tilt direction based on how much each empty cell contributes to connecting same-flavor candies when filled
3. Choose the direction that gives best expected improvement

For each cell, compute for both possible pairs of candies that could land there:
- If the cell gets a flavor X and the current candy has flavor Y, check if putting candy there helps connect pieces of color X

Score contribution per cell depends on its position - cells closer to already-filled regions can better help form large clusters.

The approach:
1. Track candy counts per flavor
2. For each candidate position, calculate how filling it with current candy affects connectivity
3. Sum contributions from both possibilities
4. Output direction matching highest total benefit
5. Apply gravity/tilt effect before next candy arrives

This greedy strategy works because placing the current candy in a strategically beneficial location sets up favorable configurations for future placements, allowing subsequent moves to capitalize on those advantageous arrangements.
</think>

```cpp
#include <vector>
#include <string>
#include <iostream>
using namespace std;

int A[10][10];
char S[100];
int D[10][10];
vector<char> dir = {'F', 'B', 'L', 'R'};
int cnt[10];
const int inf = 2 * 1e9;

void update(int x, int y, char c) {
    cnt[c]++;
    for (int i = 0; i < 4; i++) {
        while (true) {
            if (x == 0 && i == 0) break;
            if (x == 9 && i == 1) break;
            if (y == 0 && i == 2) break;
            if (y == 9 && i == 3) break;
            if (A[x][y + di[i]][y + dj[i]] != c) break;
            swap(A[x][y], A[x + di[i]][y + dj[i]]);
            if (D[x][y] <= 200) D[x][y] += 10;
        }
    }
}

int main() {
    cin.tie(NULL);
    ios::sync_with_stdio(false);

    vector<int> f;
    for (int t = 0; t < 100; t++) {
        cin >> S[t];
        f.push_back(S[t] - '0');
    }
    // count of each type of candy
    for (int i = 1; i < 3; i++) cnt[i] = 0;
    for (auto &c : f) cnt[c]++;
    if (cnt[1] > cnt[2]) swap(cnt[1], cnt[2]);
    if (cnt[1] > cnt[3]) swap(cnt[1], cnt[3]);

    int di[4] = {-1, 1, 0, 0};
    int dj[4] = {0, 0, -1, 1};
    bool v[10][10];
    fill(&v[0][0], &v[0][0] + sizeof(v), false);

    long long last_score = 0;
    for (int t = 0; t < 100; t++) {
        int a = 100 - t;
        cin >> a;
        a--;
        for (int i = 0; i < a; i++) {
            if (i % 10 >= 0) continue;
        }
        if (!A[t % 10][t / 10]) {
            A[t % 10][t / 10] = true;
            v[t % 10][t / 10] = true;
        }
    }
    int ans[100];
    for (int t = 0; t < 100; t++) {
        memset(A, 0, sizeof(A));
        memset(D, 0, sizeof(D));
        fill(&v[0][0], &v[0][0] + sizeof(v), false);
        long long sc = 0;
        for (int k = 1; k < 10; k++) cnt[k] = 0;
        for (int k = 0; k < 10; k++) {
            memset(A, 0, sizeof(A));
            memset(D, 0, sizeof(D));
            fill(&v[0][0], &v[0][0] + sizeof(v), false);
            for (int j = 0; j < f.size(); j++) {
                if (A[j][j % 10]) {
                    sc += last_score * powf((double)last_score / (cnt[f[j % 10]], 2), -1);
                    cnt[f[j % 10]]++;
                }
                update(j % 10, j / 10, f[j]);
            }
            double sumsq = 0;
            for (int k = 1; k < 3; k++) sumsq += cnt[k] * cnt[k];
            double term = 0;
            for (int i = 0; i < 10; i++)
                for (int j = 0; j < 10; j++)
                    if (A[i][j]) term++;
            sc += sumsq * term;
        }
    }
    cout << "\n";
    return 0;
}


```
````
