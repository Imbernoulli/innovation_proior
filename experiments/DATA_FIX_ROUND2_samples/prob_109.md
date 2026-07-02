# 题 109 — base均分 73.1 vs 直接SFT均分 0.4(掉点 72.7)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=4,得分 94.4,completion_tokens=17499)

````
The user wants a C++ solution to the Knight's Tour problem.
The goal is to find the longest path of a knight on an $N \times N$ chessboard starting from a given position $(r, c)$, visiting every square at most once.
If it's possible to visit all squares, the length should be $N^2$. If not, we need to output the maximum number of visited squares.
Wait, usually the Knight's Tour problem asks for a Hamiltonian path (visiting every square exactly once). The problem statement says "visit every square exactly once (no revisiting)". But then it says "output the longest possible path".
This implies finding the longest path in the graph where nodes are squares and edges represent valid knight moves.
Constraints: $6 \le N \le 666$.
Time Limit: 1 second.
Memory Limit: 128 MB.

An $N \times N$ board with $N=666$ means roughly $443,556$ squares.
Finding the longest path in a general graph is NP-hard. For $N \approx 600$, standard backtracking is impossible ($O(k^{N^2})$).
However, for the Knight's Tour specifically, on large boards, almost all squares can be visited. In fact, on any rectangular board $m \times n$ where $\min(m,n) \ge 5$ or something similar, a knight's tour exists (Hamiltonian path).
Specifically, Schwenk's theorem conditions apply for closed tours, but for open tours (path), it's easier.
Actually, for $N \ge 6$, a Knight's Tour visiting all $N^2$ squares usually exists if parity allows (which it does for $N^2$ even if $N$ is odd? Wait. Total squares must be reachable).
If $N$ is even, $N^2$ is even. Start square has some color (black/white). Each move changes color. To visit all $N^2$ squares, if $N^2$ is even, we start on one color and end on the other color (if $N^2$ is even, path length is $N^2$, so we have $N^2-1$ steps. Even steps means same color as start? No. Start at step 1 (odd). Step 2 is different. ... Step $k$ is color of $k$. If $k=N^2$ (even), final square has opposite color to start). So bipartite matching constraints apply.
But actually, it is known that for $N \ge 5$ (or maybe slightly higher, definitely $\ge 6$), a knight's tour (Hamiltonian path) exists between *any* pair of squares of opposite colors? Or just existence of a path of length $N^2$?
Schwenk (1991) proved conditions for closed tours. For open tours, a path covering all vertices exists for all $m, n \ge 5$ except some specific small cases.
Wait, let's check the constraint $N \ge 6$.
If a Knight's Tour exists, we need to output it.
Since $N$ goes up to 666, we cannot use Warnsdorff's rule blindly if we want to guarantee finding a path if one exists, but Warnsdorff's heuristic is very strong for Knights.
Actually, there is a constructive approach or a randomized DFS with heuristics (Warnsdorff's rule) that works well enough for large grids.
Given the time limit (1s) and $N=666$, an $O(N^2)$ construction would be ideal.
There are known algorithms for Knight's Tours on large boards that run in linear time $O(N^2)$.
One common strategy is using a pattern-based construction or Warnsdorff's heuristic with optimization.
Since this is a competitive programming context, typically Warnsdorff's heuristic is expected to work, possibly with a slight tweak like preferring fewer exits first.
For $N=666$, simple recursion might stack overflow or TLE due to overhead. Iterative implementation or BFS might be better.
However, finding the *longest* path suggests we might not be able to visit all $N^2$ squares?
"Note that this problem doesn't require a full complete path. It only asks you to find the longest path possible".
Usually, for $N \ge 6$, a full path of length $N^2$ exists unless parity constraints forbid it.
Parity check: Board is bipartite. Let's say $(r, c)$ is sum $S = r+c$. A move changes sum by $\pm 1 \pm 1$, which is always even change? No.
Knight moves: $(\pm 1, \pm 2)$ or $(\pm 2, \pm 1)$.
Change in row + col: $\Delta r + \Delta c = (\pm 1) + (\pm 2) = \mp 1, \pm 1, \pm 3, -3$ etc?
Let's calculate color parity. Usually defined by $(r+c) \pmod 2$.
Move $(x, y)$ to $(x \pm 1, y \pm 2)$. Change in parity: $(x \pm 1 + y \pm 2) - (x+y) = \pm 1 \pm 2$. Parity of sum is $1+2 = 3 \equiv 1 \pmod 2$.
So every move flips the color of the square.
To visit $K$ squares, we make $K-1$ moves.
If $K=N^2$ (total squares), we visit all squares.
If $N$ is even, $N^2$ is even. We start at some square. After $N^2-1$ (odd) moves, we end on the opposite color. This is possible since half squares are one color, half are the other.
If $N$ is odd, $N^2$ is odd. Half squares roughly $N^2/2$. One color has $(N^2+1)/2$, other $(N^2-1)/2$.
If start color matches the majority color, we can end on the minority color? No.
Path visits squares $v_1, v_2, \dots, v_k$.
$v_i$ has color $C(v_1)^{(i-1)\%2}$.
Sequence of colors alternates.
Number of squares of Color A and Color B visited must differ by at most 1.
Total squares available:
Color A: $\lceil N^2/2 \rceil$, Color B: $\lfloor N^2/2 \rfloor$ (assuming standard coloring).
Start at $r, c$. If $(r+c)$ is even, we need $\ge$ count(A) squares?
We can visit all $N^2$ squares if and only if the counts of colors allow alternating sequence covering all squares.
If $N$ is even, counts are equal $N^2/2, N^2/2$. Path length $N^2$ (squares) -> $N^2-1$ steps. Colors: $A, B, A, B \dots A, B$. Equal number of A and B. This works.
If $N$ is odd, total squares $N^2$ (odd). Counts: $\frac{N^2+1}{2}$ of one color (say Black), $\frac{N^2-1}{2}$ of White.
If we start on White, sequence: W, B, W, B... last is W (since odd length). Count W = Count B + 1.
Available: Count B > Count W. So we can't visit all if we start on White.
If we start on Black, sequence starts B, ends B. Count B = Count W + 1. Available fits.
So for $N$ odd, if $(r+c)$ is even (majority color?), we might be fine.
Let's define majority color carefully.
$(1,1) \to 2$ (Even). Corners $(1,N)$ etc.
On $N \times N$ with $N$ odd:
Number of black squares (e.g., $r+c$ even): $\frac{N^2+1}{2}$.
Number of white squares ($r+c$ odd): $\frac{N^2-1}{2}$.
If start is $r,c$.
If $(r+c)$ is even, start is Black. Max path is $N^2$.
If $(r+c)$ is odd, start is White. Max path is at most $N^2-1$ (actually we will get stuck earlier or hit parity mismatch). Specifically, we can't visit all because we'd need to visit one more White than Black, but we don't have enough White squares. We could visit all Blacks and all Whites? No, we need alternating. So we visit max $2 \times (\text{count White}) + 1$ = $2 \times \frac{N^2-1}{2} + 1 = N^2 - 1$ squares?
Wait, if we start on White, we need W, B, W, ..., B, W. Total $k$ squares. If $k=N^2$, we need one more W than B. But we only have one less W than B.
So if $N$ is odd and start is White, we can't visit all $N^2$. The longest path length is $N^2 - 1$? Or less? Actually it might be $N^2 - 1$ if we leave out one Black square? No, if we start White, we alternate W, B, W...
Max path length depends on availability.
If start is White, we need W, B, W, B, ..., W. Number of W = Number of B + 1.
Available: $B = W_{avail} + 1$.
So max path uses all B and all W? That would mean W used = B used + 1.
But we only have $W_{avail}$. If we use all B, we use $B_{avail}$ W's? No.
In a path W-B-W-B-W, #W = #B + 1.
If we have $W_{avail}$ whites and $B_{avail}$ blacks.
Max path size is $2 \min(W_{avail}, B_{avail}) + 1$ if start W?
Here $W_{avail} = (N^2-1)/2$, $B_{avail} = (N^2+1)/2$.
Min is $W_{avail}$. So max size $2 W_{avail} + 1 = N^2 - 1 + 1 = N^2$.
Wait.
Available: W = 16 (for example), B = 17.
Path starts W. W-B-W-B-...
Max length if limited by B: We can use 16 B's. We can use 17 W's.
Path: W(1)-B(1)-...-W(16)-B(16)-W(17)?
Need W after B. Yes.
Wait, if we start W:
1(W)
2(B)
3(W)
...
$k$.
If $k$ is odd, last is W. #W = (k+1)/2, #B = (k-1)/2. Diff is 1.
If we use all 16 B's, #B = 16. Then #W = 17. Total 33.
If $N=6$, $36$. Even. 18W, 18B. Start W. Need 18W, 18B. OK.
If $N=7$, 49. 25W, 24B.
If start W: Need 25W, 24B. Matches avail. Length 49.
If start B: Need 24W, 25B. We have 25W. Can we use 24W? Yes. Last is B. Matches avail. Length 49.
Wait, my parity calculation was wrong.
$(1,1)$ sum 2 (even). Let's call it Type 0.
$(1,2)$ sum 3 (odd). Type 1.
Move changes type.
Board sizes:
Even N: Equal numbers of Type 0 and Type 1. Any start works for length $N^2$.
Odd N: $(N^2+1)/2$ of Type 0 (corners are 0), $(N^2-1)/2$ of Type 1.
So we have more Type 0.
If start is Type 0, we alternate 0, 1, 0, 1... End on 0 (if len odd) or 1 (if len even).
To visit all $N^2$ (odd), we need len $N^2$ (odd). Sequence 0, 1, ..., 0. Count 0 = Count 1 + 1. Matches avail.
If start is Type 1, sequence 1, 0, 1, ..., 1. Count 1 = Count 0 + 1. But avail: Count 0 > Count 1. We don't have enough Type 1.
So for Odd N, if start is Type 1, we cannot visit all squares. Max path length is constrained.
How many? We have $K_0 = (N^2+1)/2$, $K_1 = (N^2-1)/2$.
Start 1. Path: 1, 0, 1, 0, ..., 1.
Length $L$. #1 = $(L+1)/2$, #0 = $(L-1)/2$.
Constraint: #0 $\le K_0$, #1 $\le K_1$.
Max #1 we can use is $K_1$.
So $(L+1)/2 \le K_1 \implies L+1 \le 2 K_1 = N^2 - 1 \implies L \le N^2 - 2$.
So if start is Type 1 on odd $N$, max path length is $N^2 - 1$? Wait.
$L = N^2 - 2 \implies$ #1 = $(N^2-1)/2 = K_1$. #0 = $(N^2-3)/2$.
Is it possible to have #0 less than $K_0$? Yes.
Can we do $L = N^2 - 1$? #1 = $N^2/2$. Not integer since $N^2$ odd.
If $L = N^2 - 2$ (even), path starts 1, ends 1. #1 = $(L+1)/2$ no, $L$ even $\implies$ pairs (1,0)...(1,0) ? No.
1, 0, ..., 0? No, start 1, next 0. Pairs of (1,0).
If $L$ even, #1 = $L/2$, #0 = $L/2$.
If $L = N^2 - 1$ (even). #1 = $(N^2-1)/2 = K_1$. #0 = $(N^2-1)/2 < K_0$.
This seems possible. We just leave out one Type 0.
So for Odd N:
Start Type 0 (Majority): Possible to visit all $N^2$.
Start Type 1 (Minority): Possible to visit $N^2 - 1$ squares.
So basically we can almost always achieve near-optimal.
For $N$ even, always $N^2$.
For $N$ odd, start $(r,c)$. Check parity.
Wait, actually, I recall a result that Knight's tour is possible for all $N \ge 5$ except $1 \times k$ etc. But here $N \ge 6$.
So we can assume for $N$ even, we find full tour.
For $N$ odd, if parity bad, we miss 1.
But wait, sample output gives $N=6$, output length 36. Full tour.
So the task is effectively to construct a tour.

Given the strict time limit and large $N$, recursive backtracking with simple pruning (Warnsdorff's) is risky because worst case still bad, though practically fast. However, for $N=666$, recursion depth 443k is huge memory usage (stack). Iterative approach needed.
Better yet, there are specific constructive algorithms for Knight's Tour.
One very efficient constructive algorithm for Knight's Tour on a grid involves decomposing the board into cycles or paths.
Or simply using the Warnsdorff's heuristic in a loop with shuffling or priority queue.
Since we need to output the path, constructing it greedily with Warnsdorff's rule is the standard way.
With $N=666$, simple Warnsdorff works very fast in practice.
However, to ensure it doesn't get stuck (rare but possible), we should handle it.
Given the 1-second limit, we need to be careful.
Also, memory limit 128MB. Storing $443556 \times 2$ integers is fine (~3.5MB). Array `visited[667][667]` takes $667^2 \times 1$ byte ~450KB.
Input reading: $O(1)$.
Output: $O(N^2)$.
Algorithm:
Warnsdorff's Heuristic:
At each step, among all legal unvisited neighbors of current square, pick the one with the fewest onward legal moves.
Tie-breaking can affect success rate.
For very large boards, this is extremely effective.
Since we just need *a* longest path, not necessarily unique, greedy should work.
Wait, "longest possible path". If Warnsdorff finds a path of length $< N^2$ when $N^2$ is possible, that's suboptimal.
But for $N \ge 6$, Warnsdorff almost never fails to find a full tour if one exists (except for specific parities/start points which we analyzed).
Actually, for the case where full tour is impossible (Odd N, wrong parity), we need to maximize.
But standard Warnsdorff might also fail in those cases?
Actually, for Odd N with correct parity (Start Majority), full tour exists.
For Odd N with wrong parity (Start Minority), we can visit $N^2-1$.
Does Warnsdorff find $N^2-1$?
It should try to extend as much as possible.
Maybe we need to implement a slight variation or just rely on randomness/quality.
Since this is C++, I can use `std::vector`, arrays.
Iterative version to avoid stack overflow.

Algorithm plan:
1. Read N, r, c. Adjust to 0-indexed.
2. Create `visited[N][N]` initialized to false.
3. Store current square `(curr_r, curr_c)`. Mark visited. Add to path.
4. Loop until no moves or we want to stop? We want to stop when no moves left OR we reached max possible.
   Max possible is $N^2$ or $N^2-1$ based on parity.
   If $N$ is even, target $N^2$.
   If $N$ is odd:
     Sum start $S = r+c$. If $N \% 2 == 1$:
       Majority color count: $\frac{N^2+1}{2}$. (Color of (0,0)).
       Minority: $\frac{N^2-1}{2}$.
       If $(r+c) \% 2 == (0+0) \% 2$, Start is Majority. Target $N^2$.
       Else Start is Minority. Target $N^2-1$.
5. Inside the loop:
   Find all valid neighbors.
   If none, break.
   For each neighbor, count its valid unvisited neighbors (degree).
   Sort neighbors by degree ascending (Warnsdorff).
   Pick best, mark visited, move to it. Add to path.
   Repeat.
   Optimization: Tie breaking matters. Maybe shuffle or pick random? Or deterministic (first/smallest coords). Deterministic is safer for reproducibility but might get stuck on rare configurations. Randomized tie-breaking is robust.
   Given 1 second, sorting neighbors is $O(\text{deg}) \log(\text{deg}) \approx O(1)$.
   Since degree is at most 8.
   
   Data structures:
   `dx = {1, 2, 2, 1, -1, -2, -2, -1}`
   `dy = {2, 1, -1, -2, -2, -1, 1, 2}`
   Neighbors array of structs/indices.
   
   Stack size for recursion vs vector push_back for iterative. Vector is better.
   Total operations $\approx N^2 \times 8 \times \log(8)$. Very small constant.
   $666^2 \approx 440,000$. Operations $\approx 3.5 \times 10^6$. Very fast.
   
   Wait, counting neighbors requires iterating 8 moves and checking bounds/visited. $8 \times 440,000 \approx 3.5 \times 10^6$.
   Sorting is negligible.
   Memory: `int visited[667][667]` -> bitset or bool. Bool is 1 byte per element. $450$ KB. Perfectly fine.
   
   Corner cases: $N=6$. Smallest constraint.
   Large $N=666$.
   Is it guaranteed that Warnsdorff works?
   For $N \ge 5$, yes, usually.
   Sometimes Warnsdorff gets trapped. A robust fix is to include the original start square in consideration if dead end occurs? No.
   With $N$ up to 666, the probability of getting stuck is non-zero but low.
   To be safe, if we can't find a path of expected length, we might backtrack. But pure DFS backtracking is too slow.
   There is a randomized approach: Try Warnsdorff. If length < Max, restart with seed shuffle.
   With 1 second, we probably can afford a few retries if failure happens.
   However, for competitive programming solutions, usually a single pass of Warnsdorff is sufficient for these constraints.
   Let's stick to Warnsdorff with random shuffle for ties.
   Wait, standard Warnsdorff often fails on $8 \times 8$ boards without modification, but for larger boards it's better.
   Given the problem asks for "longest possible", implies it might not be $N^2$?
   But description says "try to visit every square exactly once". And "find longest".
   So if $N^2$ is not possible, find max.
   With parity analysis above, if $N$ odd and start parity bad, max is $N^2-1$.
   Does Warnsdorff reach $N^2-1$ in that case?
   Probably.
   
   Wait, I should check if there's a simpler constructive pattern.
   "Knight's tour constructive patterns" exist.
   E.g. split board into strips and fill them.
   But Warnsdorff is simpler to implement.
   However, there's a risk of TLE if the heuristic fails and we need to backtrack, or just take too long due to inefficiency.
   But with $N=666$, pure Warnsdorff is extremely fast ($O(N^2)$).
   
   One detail:
   The output format.
   First line: length $l$.
   Then $l$ lines with coordinates.
   Do not print extra empty line.
   
   Implementation details:
   Use `vector<pair<int,int>> path`.
   `bool visited[666][666]`. Flattened `visited[666*666]` might be slightly faster/cachier? Or `vector<vector<bool>>`.
   Raw array is fastest. `bool visited[670][670]`.
   Move offsets `dr`, `dc`.
   Function `count_moves(r, c)` iterates 8 moves, checks validity and !visited.
   Main loop runs until `path.size() == expected_max`.
   Expected max:
   If $N \% 2 == 0$, $N*N$.
   If $N \% 2 != 0$:
     If `(r+c)%2 == (0+0)%2`, $N*N$.
     Else $N*N - 1$. (Assuming logic holds).
   
   Actually, is it always possible to achieve $N^2-1$ if start parity is bad on odd board?
   For $N \times N$ with $N$ odd, the graph is disconnected bipartite? No.
   It is connected.
   But bipartite components.
   If start color is minority, max path length is bounded by $2 \times \text{minority\_size} + 1$ (start on minority) -> $2 \times ((N^2-1)/2) + 1 = N^2$. Wait.
   Earlier I said start on minority (Type 1):
   Avail Type 1: $K_1 = (N^2-1)/2$. Avail Type 0: $K_0 = (N^2+1)/2$.
   Path: Type 1, Type 0, Type 1, ...
   If we end on Type 1 (max usage of Type 1).
   Number of Type 1 visited = $k$. Number of Type 0 visited = $k-1$.
   We can use at most $K_1$ Type 1s.
   So $k = K_1$.
   Total visited = $2 k - 1 = 2 (N^2-1)/2 - 1 = N^2 - 2$?
   Wait.
   Example $N=1$. $K_1 = 0, K_0 = 1$. Start Type 1? Impossible (no Type 1 square).
   Example $N=3$. 9 squares. 5 Type 0, 4 Type 1.
   Start Type 1. Max usage of Type 1 is 4.
   Path: T1, T0, T1, T0, T1, T0, T1, T0, T1. (Length 9).
   Visited T1: 5. But avail T1: 4. So impossible.
   Max usage of T1 is 4.
   Pattern: T1, T0, T1, T0, T1, T0, T1. (Length 7).
   Used T1: 4. Used T0: 3.
   Remaining T0: $5-3=2$. Remaining T1: 0.
   Can we add? No, next must be T0. Current last T1.
   Wait. Path T1-T0-T1-T0-T1-T0-T1. Ends T1. Next needs T0.
   Can we continue? We have T0 available.
   T1-T0-T1-T0-T1-T0-T1-T0-T1 ?
   This would require 5 T1s.
   So max length is indeed $2 \times 4 + 1 = 9$? No, that requires 5 T1s.
   So if start Type 1, we can't reach length 9.
   Max length where #T1 $\le 4$ and #T0 $\le 5$.
   Start T1.
   Seq: T1, T0, T1, T0, T1, T0, T1, T0, T1 (needs 5 T1). Impossible.
   Seq: T1, T0, T1, T0, T1, T0, T1, T0 (ends T0). #T1=4, #T0=4. Valid.
   Length 8.
   Is it possible to do length 9? No, #T1 would be 5.
   Is it possible to do length 7? #T1=4, #T0=3. Yes.
   Is length 8 achievable? T1..T0 (4 T1, 4 T0). Yes.
   So if Start is Minority on odd N, max length is $N^2 - 1$.
   Wait, $N=3 \implies N^2=9$. $N^2-1=8$.
   Logic holds.
   So `expected_max` calculation:
   $N$ even: $N^2$.
   $N$ odd:
     Start Parity (using sum index 0..N-1):
     If $(r+c) \% 2 == (0+0) \% 2$: (Majority) $N^2$.
     Else: (Minority) $N^2 - 1$.
     
     Note: For $N=3$, corners (0,0), (0,2), (2,0), (2,2) are Type 0. Center (1,1) is Type 0. Total 5.
     Edges are Type 1. Total 4.
     Correct.
   
   Wait, is $N^2 - 1$ always possible?
   For $N \ge 6$, usually yes.
   
   Code structure:
   Include headers.
   Main function.
   Fast I/O.
   Input parsing.
   Logic.
   Output.

   One optimization:
   When counting neighbors, check `!visited[nr][nc]`.
   Also check `nr >= 0 && nr < N && nc >= 0 && nc < N`.
   
   Randomization:
   Shuffle directions? Or just stable sort?
   Shuffling directions makes it less likely to follow a "bad" path early on.
   I will generate the 8 directions once.
   Inside the loop, I collect valid neighbors.
   If multiple have same degree, I can shuffle their order.
   Since degree check is fast, calculating degree for all neighbors is fast.
   Sorting small array of size <= 8 is trivial.
   Using `std::sort` with custom comparator on indices.
   Comparing degrees. If equal, maybe random swap.
   Or just rely on deterministic sort (smaller index first).
   Deterministic is fine.
   
   Wait, is it possible Warnsdorff fails?
   On some grids, yes.
   But for $N \ge 6$, it is generally considered robust enough for CP.
   However, to be absolutely safe against TLE on tricky cases (unlikely for $N \ge 6$ but possible):
   If we find we got stuck before expected max, we can re-run with different tie-breakings.
   Given 1s, maybe retry a few times.
   Actually, implementing a robust Warnsdorff with random shuffle is safer.
   
   Wait, I should confirm the memory limit.
   128 MB.
   My arrays will take ~1-2 MB. Safe.
   
   What if $N=666$?
   $N^2 \approx 4.4 \times 10^5$.
   Time limit 1s.
   Outputting $4.4 \times 10^5$ lines.
   `cout` with `\n` is slow. Use `printf` or buffer.
   Better use `\n` instead of `endl`.
   
   Let's refine the "expected max".
   Is it strictly $N^2-1$ for bad parity on odd?
   Or could it be $N^2$?
   Let's double check.
   Bipartite graph $G=(V, E)$ with partitions $V_1, V_2$.
   $|V| = |V_1| + |V_2| = N^2$.
   If $N$ odd, $|V_1| = \frac{N^2+1}{2}$, $|V_2| = \frac{N^2-1}{2}$.
   Suppose start $u \in V_2$.
   Path $v_1=u, v_2, \dots, v_k$.
   $v_i \in V_2$ iff $i$ is odd.
   $v_i \in V_1$ iff $i$ is even.
   To visit all nodes, $k=N^2$.
   Number of $V_2$ nodes visited: $\lceil k/2 \rceil = (N^2+1)/2$.
   Number of $V_1$ nodes visited: $\lfloor k/2 \rfloor = (N^2-1)/2$.
   Requirement: $|V_2| \ge (N^2+1)/2$.
   But $|V_2| = (N^2-1)/2$.
   Contradiction.
   So $k=N^2$ is impossible.
   Max $k$ such that required counts $\le$ available counts.
   Req: $\lceil k/2 \rceil \le (N^2-1)/2$.
   Let $k$ be even, $k=2m$. $\lceil m \rceil = m \le (N^2-1)/2 \implies m \le (N^2-1)/2$.
   $k \le N^2-1$.
   Let $k$ be odd, $k=2m+1$. $\lceil m + 0.5 \rceil = m+1 \le (N^2-1)/2$.
   $m \le (N^2-3)/2$. $k \le N^2-2$.
   Wait, $N^2$ is odd. $N^2-1$ is even.
   If $k=N^2-1$, req $V_2$ is $(N^2)/2$? No.
   Start $V_2$ (minority).
   $v_1 \in V_2$. $v_2 \in V_1$. ... $v_{N^2-1} \in V_1$ (since $N^2-1$ even, $N^2-1 = 2m \implies$ ends on $V_1$?).
   Indices: 1 (V2), 2 (V1), ..., 2m (V1).
   Count $V_2 = m$. Count $V_1 = m$.
   $m = (N^2-1)/2$.
   Req $V_2 = (N^2-1)/2$. Available $(N^2-1)/2$. OK.
   Req $V_1 = (N^2-1)/2$. Available $(N^2+1)/2$. OK.
   So $N^2-1$ is theoretically possible parity-wise.
   
   Wait, what about $N^2-2$?
   If $k=N^2-2$. Start $V_2$.
   $N^2-2$ is odd.
   End $V_2$.
   Count $V_2 = (k+1)/2 = (N^2-1)/2$.
   Count $V_1 = (k-1)/2 = (N^2-3)/2$.
   Both $\le$ available.
   So $N^2-1$ is the upper bound for even length path?
   Wait, if $k$ is even, path length even.
   $k=N^2-1$ is even.
   So we can have a path of length $N^2-1$.
   
   Okay, so `target = N*N` if start is Majority, else `N*N - 1`.
   
   Wait, I must consider the case where $N$ is even.
   $|V_1| = |V_2| = N^2/2$.
   Start $V_1$. Path length $N^2$ (odd).
   Count $V_1 = (N^2+1)/2$.
   Available $N^2/2$.
   $(N^2+1)/2 > N^2/2$.
   So start $V_1$ on even board, length $N^2$ impossible?
   Wait. $N^2$ is even.
   Path length $N^2$ squares means $N^2-1$ edges.
   Start $v_1$.
   If $N^2$ is even. Sequence $V_1, V_2, V_1, V_2, \dots, V_2$.
   Indices $1, 2, \dots, N^2$.
   $V_1$ at $1, 3, \dots, N^2-1$. (Count $(N^2)/2$).
   $V_2$ at $2, 4, \dots, N^2$. (Count $N^2/2$).
   Matches available $N^2/2$.
   So for Even $N$, both parities allow length $N^2$.
   Wait, $N^2$ squares total.
   If $N$ even, $N^2$ is even. Half black, half white.
   Path covers all.
   If start black. B W B W ... B W.
   B count $N^2/2$. W count $N^2/2$. OK.
   If start white. W B W B ... W B.
   W count $N^2/2$. B count $N^2/2$. OK.
   So for Even N, target is $N^2$ regardless of start.
   
   Summary:
   Target = $N*N$.
   Exception: If $N$ is Odd AND Start is Minority.
   Target = $N*N - 1$.
   
   Wait, is there any exception where $N \ge 6$ prevents reaching target?
   It is known that for $m, n \ge 5$ (and product even? no, just connectivity), Knight's tour exists.
   Wait, "Knight's Tour" usually means Closed Tour (cycle).
   Open tour is always possible if Closed tour is possible or close to it?
   Actually, Schwenk's theorem is about closed tours.
   For open tours (Hamiltonian Path), existence is broader.
   It is a known fact that for $m \times n$ with $m, n \ge 5$, a Knight's tour exists (Hamiltonian path) between ANY two squares of different colors?
   Wait, parity constraint is necessary.
   Is it sufficient?
   Most sources say yes for sufficiently large boards.
   Given $N \ge 6$, we should assume it works.
   
   One edge case: $N=6$.
   Sample input $N=6, 1, 1$. Output 36.
   $N$ even. 1, 1 is corner (1+2=3? No 1+1=2).
   Wait, board indexing 1..N.
   (1, 1) -> sum 2 (Even).
   If $N=6$, $N^2=36$. Even. Parity balance. Target 36.
   Sample output length 36. Correct.
   
   Another edge case: $N$ odd. Say $N=7$.
   Total 49.
   (1, 1) -> sum 2. Majority (type 0, count 25).
   (1, 2) -> sum 3. Minority (type 1, count 24).
   Start (1, 2). Target 48.
   
   Algorithm looks solid.
   
   ```cpp
   #include <iostream>
   #include <vector>
   #include <algorithm>
   #include <random>
   #include <numeric>
   
   using namespace std;
   
   int dr[] = {-2, -2, -1, 1, 2, 2, 1, -1};
   int dc[] = {1, -1, 2, 2, 1, -1, -2, -2};
   
   int N;
   int r, c;
   vector<vector<bool>> visited;
   vector<pair<int, int>> path;
   
   // Helper to count valid moves from (tr, tc)
   int countMoves(int tr, int tc) {
       int cnt = 0;
       for (int i = 0; i < 8; ++i) {
           int nr = tr + dr[i];
           int nc = tc + dc[i];
           if (nr >= 0 && nr < N && nc >= 0 && nc < N && !visited[nr][nc]) {
               cnt++;
           }
       }
       return cnt;
   }
   
   struct Neighbor {
       int r, c, deg;
       bool operator<(const Neighbor& other) const {
           if (deg != other.deg) return deg < other.deg;
           // Tie-break with random or coordinate?
           // Using coordinate is deterministic. 
           // To reduce traps, randomizing tie-break might help.
           // But for speed and determinism, coordinate is usually okay.
           // Let's add a static seed random for tie-break if needed?
           // Actually, purely randomizing directions initially helps.
           // Here we compare sorted neighbors.
           // Just use deg.
           return true; // Fallback, handled by stable_sort or unique keys
       }
   };
   
   int main() {
       ios_base::sync_with_stdio(false);
       cin.tie(NULL);
   
       if (!(cin >> N)) return 0;
       cin >> r >> c;
   
       r--; c--; // 0-indexed
   
       visited.assign(N, vector<bool>(N, false));
       path.reserve(N * N);
   
       visited[r][c] = true;
       path.push_back({r, c});
   
       int max_len = N * N;
       if (N % 2 != 0) {
           // For odd N, if start is minority, we lose 1.
           // Determine minority/majority based on (r+c)%2.
           // On 0-indexed board:
           // (0,0) is sum 0 (Majority).
           // (N-1)*(N-1) ...
           // If N is odd, Majority count = (N*N+1)/2. Minority = (N*N-1)/2.
           // Majorities are at (i, j) where i+j is even.
           // Minorities where i+j is odd.
           // Check current (r,c).
           if ((r + c) % 2 != 0) {
               max_len = N * N - 1;
           }
       }
   
       while (path.size() < max_len) {
           int cr = path.back().first;
           int cc = path.back().second;
   
           vector<Neighbor> neighbors;
           for (int i = 0; i < 8; ++i) {
               int nr = cr + dr[i];
               int nc = cc + dc[i];
               if (nr >= 0 && nr < N && nc >= 0 && nc < N && !visited[nr][nc]) {
                   neighbors.push_back({nr, nc, countMoves(nr, nc)});
               }
           }
   
           if (neighbors.empty()) {
               break; 
           }
           
           // Sort by degree (fewer options first)
           sort(neighbors.begin(), neighbors.end(), [](const Neighbor& a, const Neighbor& b) {
               return a.deg < b.deg;
           });
           
           // Pick first
           const auto& best = neighbors.front();
           visited[best.r][best.c] = true;
           path.push_back({best.r, best.c});
       }
   
       cout << path.size() << "\n";
       for (const auto& p : path) {
           cout << p.first + 1 << " " << p.second + 1 << "\n";
       }
   
       return 0;
   }
   ```
   One concern: `countMoves` recalculates degrees every time.
   Inside loop:
   `countMoves` called $8 \times N^2$ times.
   Total ops: $8 N^2 \times 8 = 64 N^2$.
   $64 \times 440,000 \approx 28,000,000$.
   Plus sorting $8 \log 8 \approx 6 \times 440,000$.
   Total ops around $3 \times 10^7$.
   Time limit 1.0s is tight but usually $10^8$ ops per second.
   Should pass.
   
   Wait, `countMoves` iterates 8 neighbors.
   Is there redundancy?
   We iterate `cr, cc`. For each, we scan 8 neighbors. For each neighbor, we call `countMoves` (scans 8 neighbors).
   Complexity $O(8 \cdot 8 \cdot N^2)$. Constant factor 64.
   This is acceptable.
   
   Potential improvement: Precompute degrees? No, visited changes.
   Lazy update? Overhead.
   Direct computation is simplest.
   
   Is the tie breaking critical?
   Sometimes `sort` with `<` operator is unstable.
   For Warnsdorff, any order of equal degree is fine mostly.
   
   What about `r, c` constraints? $6 \le N$.
   So no need to worry about small boards where tour doesn't exist.
   
   Double check the parity logic.
   If $N=7$. Total 49.
   Squares $(0,0), (0,2) \dots$ (even sum) are Majority. Count 25.
   Squares $(0,1), (0,3) \dots$ (odd sum) are Minority. Count 24.
   Start $(r,c)$.
   If $(r+c)$ is odd, start is Minority.
   Then we need to alternate M, Ma, M, Ma...
   Wait, moves always flip parity.
   Path: Start (M) -> Ma -> M -> Ma ...
   Wait.
   Moves:
   1. Start (Minority, count 24).
   2. Next (Majority, count 25).
   3. Next (Minority).
   ...
   If we visit $L$ squares.
   #Minority visited = $\lceil L/2 \rceil$.
   #Majority visited = $\lfloor L/2 \rfloor$.
   Constraint: #Minority $\le 24$, #Majority $\le 25$.
   $\lceil L/2 \rceil \le 24 \implies L/2 \le 24$ (if L even) or $(L+1)/2 \le 24$ (if L odd).
   If $L=48$ (even): $\lceil 24 \rceil = 24 \le 24$. OK.
   If $L=49$ (odd): $\lceil 24.5 \rceil = 25 \not\le 24$. Impossible.
   So max $L=48 = N^2 - 1$.
   Matches logic.
   
   Start (Majority).
   Path: Start (Ma) -> Mi -> Ma -> ...
   #Majority visited = $\lceil L/2 \rceil$.
   #Minority visited = $\lfloor L/2 \rfloor$.
   Constraint: $\lceil L/2 \rceil \le 25$, $\lfloor L/2 \rfloor \le 24$.
   If $L=49$: $\lceil 49/2 \rceil = 25 \le 25$. OK. $\lfloor 49/2 \rfloor = 24 \le 24$. OK.
   So $N^2$ possible.
   
   Logic confirmed.
   
   Final check on includes. `<vector>`, `<iostream>`, `<algorithm>`, `<utility>` (pair).
   
   One subtle thing:
   The problem says "don't have an extra empty line at the end".
   `cout << p.first + 1 << " " << p.second + 1 << "\n";` prints newlines.
   The loop ends. The last line has a newline character.
   Usually this is allowed or expected. "Don't have an extra empty line" means don't print `\n\n` or blank lines.
   Standard printing with `\n` at end of each line is correct.
   
   Also "make sure to not have an extra empty line after printing out the final move."
   Just means don't print trailing whitespace/newline beyond the last data.
   `cout << ... << "\n"` is standard.
   
   Memory limit 128MB.
   Vectors take dynamic allocation. `vector<vector<bool>>` is specialized, uses bits.
   $N^2$ bits. $440,000 / 8 \approx 55$ KB.
   Much less than 450KB.
   
   Wait, `visited.assign(N, vector<bool>(N, false))` creates separate vectors.
   Heap fragmentation?
   Better use raw pointer or flattened vector or static array.
   Static array `bool visited[670][670]` is safest for stack/global memory.
   If declared inside main, it's on stack (too big for stack?). 450KB on stack is usually OK (default stack 1MB-8MB).
   But Global is safer.
   Let's use global `bool visited[667][667]`.
   Initialize with loop.
   
   Also `path` vector stores `pair<int,int>`.
   $440,000 \times 8$ bytes $\approx 3.5$ MB. Fine.
   
   Optimization:
   Use `int` array `visited` (bitmask) to be cache friendly?
   Boolean array is byte aligned. `vector<char>` or `bool[]`.
   `visited[n][m]` with `char` avoids vector overhead.
   But `vector<vector<char>>` is fine.
   Actually `visited` array access is key.
   
   Let's write code.
   
   One detail:
   `countMoves` accesses `visited`.
   Pass `visited` or make it global. Global is easier.
   
   The prompt says "You are a competitive programmer... implement solution".
   I will produce clean, commented code inside cpp block.
   
   Wait, is `N*N` potentially exceeding `int` range?
   $666^2 = 443556$. Fits in int.
   
   One case: `neighbors.empty()` check inside loop.
   If we can't find a neighbor, we stop.
   Then we check `path.size()`.
   Wait, loop condition `while(path.size() < max_len)`.
   If `neighbors.empty()`, `break`.
   Then output size.
   This handles cases where we get stuck before `max_len`.
   
   Wait, if we get stuck, `path.size()` will be whatever it reached.
   Is it possible to improve?
   Since we can't easily backtrack efficiently in 1s for $N=666$, if we get stuck, we hope it didn't happen on optimal path (which is highly likely with Warnsdorff).
   
   Re-checking constraints:
   Time limit 1 sec.
   For $N=666$, pure Warnsdorff is $O(N^2)$.
   Operations count $\approx 3 \times 10^7$.
   This is very safe for 1 sec (usually $10^8$ ops/sec).
   
   Double check `ios_base::sync_with_stdio(false); cin.tie(NULL);`.
   
   Final check:
   Start coordinates 1-indexed in input.
   Convert to 0-indexed for processing.
   Convert back to 1-indexed for output.
   
   Wait, `max_len` calculation again.
   If $N$ is odd, `start_parity` check.
   `if ((r + c) % 2 != 0)` -> Minority.
   This assumes $(0,0)$ is Majority.
   Let's verify.
   $N=3$. (0,0) is 0. Count of evens: 5. Odds: 4.
   Evens: (0,0), (0,2), (1,1), (2,0), (2,2).
   (0,0) is sum 0. Even. Correct.
   If input $r=1, c=2$ (1-indexed).
   0-indexed $r=0, c=1$.
   Sum $0+1=1$ (Odd).
   Minority.
   Logic holds.
   
   Wait, is it possible for $N$ odd, start Minority, we can visit $N^2$ squares?
   No, proved by parity argument earlier.
   So `max_len = N*N - 1` is the theoretical upper bound.
   Is it always reachable?
   For $N \ge 5$, yes.
   
   One tiny optimization:
   Instead of `sort`, use partial sort or bucket sort?
   Degree is integer 0..8.
   We can use buckets or just sort. `std::sort` on 8 elements is super fast.
   
   Wait, `visited` initialization.
   `memset` works for `bool` array.
   
   ```cpp
   #include <iostream>
   #include <vector>
   #include <algorithm>
   #include <cstdio>

   using namespace std;

   int N;
   int r_start, c_start;
   bool visited[670][670];
   int dr[] = {-2, -2, -1, 1, 2, 2, 1, -1};
   int dc[] = {1, -1, 2, 2, 1, -1, -2, -2};

   void solve() {
       int r, c;
       cin >> N;
       cin >> r >> c;
       r--; c--;

       // Initialize visited
       for(int i=0; i<N; ++i)
           for(int j=0; j<N; ++j)
               visited[i][j] = false;

       vector<pair<int, int>> path;
       path.reserve(N * N);

       visited[r][c] = true;
       path.push_back({r, c});

       int max_path_len = N * N;
       
       // Parity check for N odd
       if (N % 2 != 0) {
           // (0,0) is majority color (even sum). 
           // Count even sum cells: ceil(N*N/2). Count odd sum: floor(N*N/2).
           // If start is odd sum (minority), max len is N*N - 1.
           if ((r + c) % 2 != 0) {
               max_path_len = N * N - 1;
           }
       }

       int path_idx = 1;
       while (path_idx < max_path_len) {
           int cur_r = path.back().first;
           int cur_c = path.back().second;

           // Collect neighbors
           struct Node { int r, c, deg; };
           vector<Node> candidates;
           candidates.reserve(8);

           for (int i = 0; i < 8; ++i) {
               int nr = cur_r + dr[i];
               int nc = cur_c + dc[i];

               if (nr >= 0 && nr < N && nc >= 0 && nc < N && !visited[nr][nc]) {
                   int deg = 0;
                   // Calculate degree
                   for (int j = 0; j < 8; ++j) {
                       int nnr = nr + dr[j];
                       int nnc = nc + dc[j];
                       if (nnr >= 0 && nnr < N && nnc >= 0 && nnc < N && !visited[nnr][nnc]) {
                           deg++;
                       }
                   }
                   candidates.push_back({nr, nc, deg});
               }
           }

           if (candidates.empty()) break;

           // Sort by degree
           sort(candidates.begin(), candidates.end(), [](const Node& a, const Node& b) {
               return a.deg < b.deg;
           });

           // Take best
           visited[candidates[0].r][candidates[0].c] = true;
           path.push_back({candidates[0].r, candidates[0].c});
           path_idx++;
       }

       printf("%d\n", (int)path.size());
       for (const auto& p : path) {
           printf("%d %d\n", p.first + 1, p.second + 1);
       }
   }

   int main() {
       solve();
       return 0;
   }
   ```
   Using `printf` for output is safer/faster than `cout` with `endl`.
   The memory usage for `visited` is small (global).
   The time complexity is fine.
   Wait, `visited` is global array. If I put `solve` inside `main` or `solve` called by `main`.
   I'll put everything in `main`.
   
   Wait, variable shadowing?
   `visited` array is global.
   `candidates` vector inside loop is reallocated.
   Since loop runs $N^2$ times, allocation overhead might be significant?
   $440,000$ allocations.
   `candidates` capacity is 8. Small vector allocation.
   Might be better to declare `candidates` outside and `clear` it.
   `vector<Node> candidates; candidates.reserve(8);` outside loop.
   Inside: `candidates.clear();`
   This saves heap allocs.
   
   Also `printf` requires `<cstdio>`.
   
   Check header inclusion.
   
   One more check: `max_path_len` type.
   $666^2 = 443556$. Fits in `int`.
   
   Wait, `max_path_len` calculation:
   If $N$ is even, $N^2$ is even. Parity balance is perfect.
   If $N$ is odd, $N^2$ is odd.
   My logic: start Minority -> $N^2-1$.
   Is it possible start Majority -> $N^2$?
   Yes, if start Majority.
   Is it possible start Minority -> $N^2$? No.
   Is it possible start Minority -> $N^2-1$? Yes.
   So the code logic is correct.
   
   Wait, if `candidates` is cleared inside loop, vector destructor isn't called, just internal buffer reused?
   `clear()` keeps capacity. `reserve()` ensures it.
   Declaring `candidates` outside loop.
   
   Final check:
   The problem says "When outputting the answer, make sure to not have an extra empty line at the end of the output after printing out the final move."
   Using `printf(..."\n")` puts a newline after the last line.
   This is standard. An "extra empty line" usually means `printf("\n\n")` or just nothing if you wanted no newline.
   Competitive programming platforms expect newline at end of last line.
   I will keep `\n`.
   
   One optimization on `deg` calculation:
   Can be done without inner loop? No.
   But notice `visited` check.
   
   Wait, is `candidates` struct definition inside function okay? Yes (C++11).
   Or use tuple/pair. Struct is cleaner.
   
   Let's refine the solution.
   
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>

using namespace std;

// Globals to avoid stack overflow and repeated allocation
int N;
bool visited[670][670];
int dr[] = {-2, -2, -1, 1, 2, 2, 1, -1};
int dc[] = {1, -1, 2, 2, 1, -1, -2, -2};
struct Node {
    int r, c, deg;
};
vector<Node> candidates; // Moved outside loop

int main() {
    // Optimize I/O operations
    // Although printf is used, syncing might be good to ensure consistency or not?
    // Mixing cin/scanf/printf. cin is not used much. Just cin for input.
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> N)) return 0;
    int r, c;
    cin >> r >> c;

    // 1-based to 0-based
    r--; c--;

    // Initialize visited array
    // Since N <= 666, we only need to clear up to N x N
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            visited[i][j] = false;
        }
    }

    vector<pair<int, int>> path;
    path.reserve(N * N);

    visited[r][c] = true;
    path.push_back({r, c});

    int max_len = N * N;
    // Parity constraint for Knight's Tour on odd sized boards
    // (0,0) is color 0 (sum even).
    // For N odd, count(color 0) = (N*N+1)/2, count(color 1) = (N*N-1)/2.
    // Start color determines max length.
    if (N % 2 != 0) {
        // If start is color 1 (odd sum), max length is N*N - 1
        if ((r + c) % 2 != 0) {
            max_len = N * N - 1;
        }
    }

    int current_idx = 0;
    while ((int)path.size() < max_len) {
        int u_r = path[current_idx].first;
        int u_c = path[current_idx].second;
        current_idx++;

        candidates.clear();
        for (int i = 0; i < 8; ++i) {
            int nr = u_r + dr[i];
            int nc = u_c + dc[i];
            if (nr >= 0 && nr < N && nc >= 0 && nc < N && !visited[nr][nc]) {
                int deg = 0;
                // Calculate degree (number of valid moves from neighbor)
                for (int j = 0; j < 8; ++j) {
                    int nnr = nr + dr[j];
                    int nnc = nc + dc[j];
                    if (nnr >= 0 && nnr < N && nnc >= 0 && nnc < N && !visited[nnr][nnc]) {
                        deg++;
                    }
                }
                candidates.push_back({nr, nc, deg});
            }
        }

        if (candidates.empty()) {
            break;
        }

        // Warnsdorff's heuristic: sort by degree ascending
        sort(candidates.begin(), candidates.end(), [](const Node& a, const Node& b) {
            return a.deg < b.deg;
        });

        // Visit the node with lowest degree
        visited[candidates[0].r][candidates[0].c] = true;
        path.push_back({candidates[0].r, candidates[0].c});
    }

    printf("%d\n", (int)path.size());
    for (const auto& p : path) {
        printf("%d %d\n", p.first + 1, p.second + 1);
    }

    return 0;
}
```
   One minor issue:
   `candidates` declaration outside `main` but inside `solve` scope?
   If I declare it globally `vector<Node> candidates;`.
   Need `#include <vector>`.
   Also `candidates.clear()` is fine.
   Wait, if I declare `candidates` globally, it persists across calls? Only one test case.
   But `main` is called once.
   Okay, putting it globally is fine.
   But inside main is also fine.
   I'll keep it inside main but reserved.
   
   Double check:
   `if (N % 2 != 0)` -> `(r+c)%2 != 0`.
   Wait, `r` and `c` are 0-indexed now.
   Correct.
   
   Wait, `current_idx` logic is redundant if I use `back()`.
   `current_idx` increments.
   `path.back()` works too.
   But since I loop on `path.size()`, I need `back`.
   `int u_r = path.back().first;`
   
   Wait, `path` grows. `back()` is safe.
   
   One thing:
   `candidates` vector.
   `vector<Node> candidates;` outside `while`.
   `candidates.clear();` inside `while`.
   
   Looks good.

   Wait, I used `printf` and `cin`.
   Mixing streams:
   `cin` writes to `cinbuf`. `printf` reads from stdin.
   If I disable sync, they might interleave incorrectly?
   If I only use `cin` for input and `printf` for output, it is usually fine as long as I don't switch back.
   But `cin` buffers. `printf` flushes?
   Best to use `scanf` or consistent `cin`/`cout`.
   I will use `cin` for input and `cout` for output (with `\n` and sync off).
   Or `scanf`/`printf`.
   Since $N$ is small (just 3 ints), I/O speed of input is not bottleneck. Output is bottleneck.
   `cout` with `sync_with_stdio(false)` is as fast as `printf`.
   I will use `cout` to be safe and consistent with C++ idioms.
   Actually `printf` is often preferred for massive output.
   But with `sync_with_stdio(false)`, mixing is dangerous if I don't flush properly?
   Actually, mixing formatted I/O (scanf/printf) with stream I/O (cin/cout) is undefined behavior if sync is off.
   If sync is ON, it's buffered but safe.
   If I turn sync OFF, mixing is bad.
   So either:
   1. Keep sync ON, use `cin`/`cout` or `scanf`/`printf`.
   2. Turn sync OFF, use only `cin`/`cout` OR only `scanf`/`printf`.
   
   Since `printf` is faster for formatting integers, I will use `scanf`/`printf`.
   Or `cin`/`cout` with sync off.
   Given the amount of output ($4 \times 10^5$ lines), `cout` with `\n` is slower than `printf` but might pass.
   400k integers is not huge.
   I'll stick to `cout` with sync off. It's safer for mixing if I'm lazy, but technically `printf` + `sync_off` + `cout` = bad.
   I will just use `scanf` and `printf` completely to be safe and fast.
   `scanf("%d", &N)`
   `scanf("%d%d", &r, &c)`
   `printf("%d\n", ...)`
   
   Also, `candidates` needs to be reset properly.
   
   Wait, `scanf` is C-style.
   I'll use `cin` with sync disabled for input, and `cout` with `\n` for output.
   Wait, `printf` is generally faster for bulk output?
   Actually `cout` with `\n` is quite optimized.
   But `printf` avoids overhead of C++ object construction for each print?
   I'll use `cout` because I already wrote C++ code logic.
   To avoid mixing, I will use ONLY `cout`.
   With `sync_with_stdio(false)`, `cin` and `cout` are decoupled.
   
   Revised code using only `cin`/`cout`.
   
```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int N;
bool visited[670][670];
int dr[] = {-2, -2, -1, 1, 2, 2, 1, -1};
int dc[] = {1, -1, 2, 2, 1, -1, -2, -2};
struct Node {
    int r, c, deg;
};
vector<Node> candidates; // Defined globally or outside loop

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> N)) return 0;
    int r, c;
    cin >> r >> c;
    r--; c--; // 0-indexed

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            visited[i][j] = false;
        }
    }

    vector<pair<int, int>> path;
    path.reserve(N * N);

    visited[r][c] = true;
    path.push_back({r, c});

    int max_len = N * N;
    if (N % 2 != 0) {
        // For N odd, check parity. (0,0) is majority color.
        // Majority: sum even. Minority: sum odd.
        // If start is Minority (sum odd), max_len decreases by 1.
        if ((r + c) % 2 != 0) {
            max_len = N * N - 1;
        }
    }

    // Pre-reserve candidates
    candidates.reserve(8);

    while (path.size() < (size_t)max_len) {
        int u_r = path.back().first;
        int u_c = path.back().second;

        candidates.clear();
        for (int i = 0; i < 8; ++i) {
            int nr = u_r + dr[i];
            int nc = u_c + dc[i];
            
            if (nr >= 0 && nr < N && nc >= 0 && nc < N && !visited[nr][nc]) {
                int deg = 0;
                for (int j = 0; j < 8; ++j) {
                    int nnr = nr + dr[j];
                    int nnc = nc + dc[j];
                    if (nnr >= 0 && nnr < N && nnc >= 0 && nnc < N && !visited[nnr][nnc]) {
                        deg++;
                    }
                }
                candidates.push_back({nr, nc, deg});
            }
        }

        if (candidates.empty()) {
            break;
        }

        // Sort neighbors by degree
        sort(candidates.begin(), candidates.end(), [](const Node& a, const Node& b) {
            return a.deg < b.deg;
        });

        // Move to neighbor with lowest degree
        visited[candidates[0].r][candidates[0].c] = true;
        path.push_back({candidates[0].r, candidates[0].c});
    }

    cout << path.size() << "\n";
    for (const auto& p : path) {
        cout << p.first + 1 << " " << p.second + 1 << "\n";
    }

    return 0;
}
```
   Seems good.
   Memory:
   Global `visited`: ~670*670 bytes = 448900 bytes = ~450 KB.
   Global `candidates`: minimal.
   `path`: 440,000 pairs of ints = 3.5 MB.
   Total ~4 MB. Well within 128 MB.
   Time: ~30M ops. 1s is plenty.
   
   Final check on logic.
   If `N=666`, `path.size()` comparison `(size_t)max_len` cast to avoid signed/unsigned warning?
   `max_len` is `int`. `path.size()` is `size_t`.
   Comparison is fine but warning might occur. Cast is safer.
   
   One more thing:
   The sample output has `36` then `1 1` ...
   My code outputs `path.size()` first.
   Correct.
   
   Ready to wrap.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Global variables to minimize stack usage and overhead
int N;
bool visited[670][670]; // 666*666 fits easily here. Max N=666.
int dr[] = {-2, -2, -1, 1, 2, 2, 1, -1};
int dc[] = {1, -1, 2, 2, 1, -1, -2, -2};

struct Node {
    int r, c, deg;
};

// Candidates vector reused in the loop to avoid reallocation
vector<Node> candidates;

int main() {
    // Optimize I/O operations for performance
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> N)) return 0;
    int r, c;
    cin >> r >> c;
    
    // Convert 1-based coordinates to 0-based
    r--; c--; 

    // Initialize visited array
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            visited[i][j] = false;
        }
    }

    vector<pair<int, int>> path;
    path.reserve(N * N);

    visited[r][c] = true;
    path.push_back({r, c});

    int max_len = N * N;
    // For odd N, bipartite property limits the path length based on start square color.
    // (0,0) corresponds to 'color 0' (even sum of coords).
    // On an NxN board with N odd, 'color 0' count is (N*N+1)/2, 'color 1' count is (N*N-1)/2.
    // If we start on 'color 1' (minority), we cannot visit all squares (parity mismatch).
    // Maximum path length becomes N*N - 1.
    if (N % 2 != 0) {
        if ((r + c) % 2 != 0) {
            max_len = N * N - 1;
        }
    }

    candidates.reserve(8);

    // Iterate to build the path using Warnsdorff's heuristic
    while (path.size() < (size_t)max_len) {
        int u_r = path.back().first;
        int u_c = path.back().second;

        candidates.clear();
        // Evaluate all valid neighbors
        for (int i = 0; i < 8; ++i) {
            int nr = u_r + dr[i];
            int nc = u_c + dc[i];
            
            // Check bounds and if visited
            if (nr >= 0 && nr < N && nc >= 0 && nc < N && !visited[nr][nc]) {
                int deg = 0;
                // Count unvisited neighbors for the potential next step (degree heuristic)
                for (int j = 0; j < 8; ++j) {
                    int nnr = nr + dr[j];
                    int nnc = nc + dc[j];
                    if (nnr >= 0 && nnr < N && nnc >= 0 && nnc < N && !visited[nnr][nnc]) {
                        deg++;
                    }
                }
                candidates.push_back({nr, nc, deg});
            }
        }

        // If no moves possible, stop
        if (candidates.empty()) {
            break;
        }

        // Sort by degree ascending (Warnsdorff's rule)
        sort(candidates.begin(), candidates.end(), [](const Node& a, const Node& b) {
            return a.deg < b.deg;
        });

        // Move to the neighbor with fewest onward moves
        visited[candidates[0].r][candidates[0].c] = true;
        path.push_back({candidates[0].r, candidates[0].c});
    }

    // Output results
    cout << path.size() << "\n";
    for (const auto& p : path) {
        cout << p.first + 1 << " " << p.second + 1 << "\n";
    }

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=3,得分 0.0,completion_tokens=1718)

````
I need to solve the knight's tour problem - finding the longest path a knight can take on an N×N chessboard starting from a given position without revisiting any square.

Since we want the longest path (ideally visiting all squares), this is essentially finding a Hamiltonian path for the knight's graph. For N × N boards where N ≥ 6, it's well-known that the knight can always visit every square exactly once when N is even (and has solutions for many odd cases too), so the optimal answer should be N × N - 1 visits plus 1 start = N² squares visited.

The standard approach here is Warnsdorff's rule combined with backtracking:
- From each position, try moves to squares that have the fewest available onward moves
- This greedy heuristic dramatically reduces search space while still finding near-optimal solutions
- For maximum length guarantee, combine with depth-first search with pruning based on current path length vs best found

Let me write efficient C++ code using Warnsdorff's rule for the knight's tour. The key optimizations are:
1. Precompute move offsets and board boundaries checking
2. Use Warnsdorff's ranking heuristic to order moves by their future move availability
3. Backtrack if needed to find longer paths, but rely mostly on the heuristic
4. Track and record the best complete path found

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int n;
struct Point { int r, c; };
const int dr[] = {-2,-1,1,2,2,1,-1,-2};
const int dc[] = {1,2,2,1,-1,-2,-2,-1};
vector<vector<bool>> visited;
Point path[5000];
int best_len = 0;

bool valid(int r, int c) { return r >= 1 && r <= n && c >= 1 && c <= n; }

int count_moves(Point p) {
    int cnt = 0;
    for(int i = 0; i < 8; i++) {
        int nr = p.r + dr[i], nc = p.c + dc[i];
        if(valid(nr,nc) && !visited[nr][nc]) cnt++;
    }
    return cnt;
}

void backtrack(int u, vector<Point>& cur_path) {
    if(cur_path.size() == (size_t)n*n) {
        if((int)cur_path.size()-1 > best_len) {
            best_len = (int)cur_path.size();
            for(int k=0;k<best_len;k++) path[k] = cur_path[k];
        }
        return;
    }
    Point p = cur_path.back();
    for(int i=0;i<8;i++){
        int nr=p.r+dr[i],nc=p.c+dc[i];
        if(valid(nr,nc)&&!visited[nr][nc]){
            visited[nr][nc]=true;cur_path.push_back({nr,nc});
            backtrack(u+1,cur_path);
            cur_path.pop_back();visited[nr][nc]=false;
        }
    }
}

int main(){cin>>n;
visited.assign(n+1,vector<bool>(n+1,false));int sr,sc;cin>>sr>>sc;
path[0]={sr,sc};visited[sr][sc]=true;backtrack(0,{path});cout<<best_len+1<<"\n";
for(int i=0;i<=best_len;i++) cout<<path[i].r<<" "<<path[i].c<<"\n";
return 0;}
```

This should work but for N up to 666, pure backtracking with the heuristic may still be slow in worst cases. I'll add iterative deepening and stronger pruning since the target path length is usually N².
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int n;
vector<vector<char>> board;
int bestLen;
char boardCopy[700][700];

struct Move {
    int tr, tc, rank;
};

bool isValid(int r, int c) { return r >= 1 && r <= n && c >= 1 && c <= n; }

void dfs(int idx, vector<int>& path) {
    if (idx > bestLen) {
        bestLen = idx;
        for (int i = 0; i <= idx; i++) boardCopy[path[i]] = 'X';
        return;
    }

    // If max possible squares visited doesn't beat best, prune
    long long remainingCells = (long long)n * n - (long long)(idx + 1);
    if (idx > 0 && idx + 1 + remainingCells / 3 < bestLen) return;

    int r = path[idx - 1] >> 15, c = path[idx - 1] & 0x7FFF;
    if (idx == 0) {
        for (int i = 0; i <= idx; i++) boardCopy[path[i]] = 'X';
        return;
    }

    vector<Move> nextMoves;
    static const int dr[] = {-2,-1,1,2,2,1,-1,-2};
    static const int dc[] = {1,2,2,1,-1,-2,-2,-1};
    
    for (int i = 0; i < 8; i++) {
        int nr = r + dr[i];
        int nc = c + dc[i];
        int pos = (nr << 15) | nc;
        if (!isValid(nr, nc) || board[pos] != '0') continue;
        
        int moveRank = 0;
        for (int j = 0; j < 8; j++) {
            int nnr = nr + dr[j];
            int nnc = nc + dc[j];
            if (isValid(nnr, nnc) && board[(nnr << 15) | nnc] == '0')
                moveRank++;
        }
        nextMoves.push_back({nr, nc, moveRank});
    }

    sort(nextMoves.begin(), nextMoves.end(), [](const Move& a, const Move& b){ return a.rank < b.rank; });
    for (const auto& mv : nextMoves) {
        int npos = (mv.tr << 15) | mv.tc;
        board[npos] = 'X';
        path.push_back(npos);
        dfs(idx + 1, path);
        path.pop_back();
        board[npos] = '0';
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n;
    int sr, sc; cin >> sr >> sc;
    int startPos = (sr << 15) | sc;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            board[(i << 15) | j] = '0';
        }
    }

    board[startPos] = 'X';
    vector<int> path;
    path.push_back(startPos);

    bestLen = 0;
    dfs(0, path);

    cout << bestLen + 1 << "\n";
    for (int i = 0; i <= bestLen; i++) {
        int row = boardCopy[path[i]] & 0x7FFF; 
        int col = boardCopy[path[i]] >> 15; 
        cout << row << " " << col << "\n";
    }
    return 0;
}
```
````
