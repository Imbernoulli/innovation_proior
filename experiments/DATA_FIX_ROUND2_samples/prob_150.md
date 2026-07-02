# 题 150 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The user wants a solution to an AtCoder Heuristic Contest (AHC) problem.
Specifically, this is AHC 004 Problem B (implied by the context of N=20, M strings from a toroidal matrix). Wait, looking at the story, it mentions "alien cells... torus-shaped material". This looks like a specific problem. The text says $N$ is fixed to 20. $M$ is between 400 and 800. Strings are subsequences taken horizontally or vertically from a random $N \times N$ matrix generated with characters 'A' through 'H'.
The goal is to reconstruct this matrix (or find one that matches the given constraints) to maximize the score.
Actually, reading the scoring carefully:
"If $c < M$, $\mathrm{round}(10^8\times \frac{c}{M})$. If $c=M$, $\mathrm{round}(10^8\times \frac{2 N^2}{2 N^2-d})$."
This implies we want to match all $M$ strings ($c=M$). If we do, minimizing '.' ($d$) gives a better score because the term $\frac{2N^2}{2N^2-d}$ increases as $d$ decreases. Wait, no. $2N^2 / (2N^2 - d)$. Since $d$ is in the denominator, smaller $d$ means larger denominator? No. $2N^2 - d$ is in the denominator. As $d$ increases (more dots), the denominator decreases, so the fraction increases. So more dots give higher score?
Let's check the logic. "score ... round($10^8 \times \frac{2 N^2}{2 N^2 - d}$)".
If $d=0$, score = $10^8$.
If $d=N^2$ (all dots), score = $10^8 \times \frac{2 N^2}{N^2} = 2 \cdot 10^8$.
So yes, matching all strings and using empty spots ('.') to increase score is good. But actually, if we use '.', it counts as `.` in the output, which doesn't contribute to the character space usually, but here '.' is just a placeholder.
Wait, the input generation says the original matrix was fully filled with 'A'-'H'. We need to recover *an* $N \times N$ matrix. If we can find the exact matrix used to generate inputs, then $c=M$ and $d=0$, score $10^8$. If we can't, maybe we leave some cells as '.'.
But wait, if we put '.', does it affect whether a string is a subsequence? Yes. If a string has 'A', and our cell has '.', it won't match unless 'A' == '.' which is false. So '.' acts as a wildcard only if it wasn't there? No, "." indicates an empty cell in the output. The problem asks to output a matrix consisting of 'A'-'H' or '.'. The definition of subsequence requires $b_p = a_{i, ...}$. So '.' cannot match 'A'-'H'. So effectively, '.' makes it harder to satisfy the constraints for non-dot strings, but helps the score if $c=M$. However, since the input strings are generated from a fully populated matrix (using 'A'-'H'), we should probably try to fill the matrix with 'A'-'H' to satisfy $c=M$. Using '.' would likely reduce $c$ unless those positions were not needed. But since inputs are contiguous segments of length 2-12, finding the exact grid is likely the primary goal.

Problem specifics:
$N=20$.
$M \approx 400-800$.
Strings length 2-12.
Characters 'A'-'H' (8 types).
Input strings are generated from a random matrix $A$ where each entry is uniform random 'A'-'H'.
Since the generation process picks random rows/cols and substrings, it provides local constraints on the matrix structure.
This looks like a constraint satisfaction problem or a grid filling problem. Since $N=20$, iterating over all possibilities is impossible ($8^{400}$). But we have specific constraints.
Each string $s_i$ tells us that in the target grid, there exists a horizontal or vertical alignment (wrapping around) where characters match $s_i$. Note that the problem says "contains as many... as possible as subsequences". It does NOT say "the strings must be found EXACTLY once". Just that they exist somewhere.
However, since the strings are substrings of the true underlying matrix, they definitely exist in the true matrix. So the "true" matrix satisfies $c=M$.
Our goal is to approximate this true matrix.

Strategy:
Since $N$ is small (20), but the state space is large, heuristics are expected (AHC).
We need to assign characters to cells $(r, c)$ for $r, c \in [0, 19]$.
Constraints: For each string $s_k$, there exist $r, c, dir, len$ such that $s_k$ matches the segment.
Actually, the problem statement says: "Given $M$ strings... find a matrix... contains as many... as possible".
Wait, does it imply we don't know $N$ in the reconstruction part?
"N is fixed to 20 throughout all the test cases." - Okay, we know $N=20$.
The difficulty is determining which character goes to which cell.
Since strings are short (max 12), and generated from the true matrix, most cells in the true matrix are covered by at least one string instance.
Actually, the generation process iterates $M$ times. Each time picks a random start, direction, and length. With $M=600$ and length avg 8, we cover about $600 \times 8 = 4800$ cells-worth of info, but with overlap. Given $N^2=400$, this covers the whole grid multiple times.
So, for most pairs $(r,c)$, we have strong evidence for its value from the strings that cover it.
For example, if string "ABC" starts at row 5 col 0 going horizontal, then Grid[5][0]='A', Grid[5][1]='B', Grid[5][2]='C'.
If another string implies Grid[5][0]='D', there is a conflict.
Since the inputs are generated from a single valid matrix, there should be no conflicts if we just vote.
However, the input strings might be overlapping. E.g., String 1 covers (0,0) to (0,2). String 2 covers (0,1) to (0,3). They might agree on the overlap. Or they might conflict if the generator made mistakes? No, "We first generate an NxN matrix... then generate M strings... from that matrix". So all strings are consistent with a single matrix.
Thus, we can simply count how often each position is constrained by each string.
However, we don't know WHICH offset in the string corresponds to which position in the grid.
Wait, "Given M strings ... find a matrix ... contains as many ... as possible".
If the strings come from the matrix, they must appear in the matrix.
So if we find a matrix where every string appears, $c=M$.
To find such a matrix, we can treat this as finding a placement for each string.
Or simpler: Count frequency of patterns.
With $N=20$, the number of possible horizontal placements is $20 \times 20 = 400$. Vertical also 400. Total 800 positions per length? No.
There are $N \times N = 400$ starting cells.
In each cell, there are $N$ horizontal strings starting there, and $N$ vertical strings.
So roughly $400 \times 2 \times N \approx 16000$ possible substrings of length up to $N$ could theoretically exist.
But we only have $M$ observed strings.
Wait, the strings are fixed in the input. We need to place them.
Actually, it is likely that the intended solution is simply to construct the grid that matches these observations as best as possible. Since they were generated from a real grid, they are consistent with that real grid.
The issue is that we don't know the relative positions of the strings.
String $S_1$ could be at any of the 400 start positions (horizontal or vertical) and any shift (offset).
But wait, the strings are just sequences.
For a specific string $S_k$, we know its content. We want to fit it into the grid.
Actually, since it's generated from a specific grid, there IS a correct configuration.
Can we just assume the strings are placed randomly and try to optimize?
Maybe we can model this as a grid fill problem where we decide $Cell[i][j] \in \{A..H\}$.
Constraint: If we place string $S_k$ at $(r, c)$, then $Grid[r][c+k] == S_k[k]$.
Since we want to satisfy all strings, we should pick the configuration that satisfies all.
This sounds like Maximum Constraint Satisfaction.
But since the strings come from a valid grid, satisfying all is possible.
Why isn't it trivial? Because for each string $S_k$, we don't know its location. There are too many locations.
Wait, if we fix the locations of all strings, we can merge the constraints.
Is it necessary to fix locations?
Consider two strings. Maybe they share a substring.
Example: "ABCD" and "BCDE". If we place them adjacent, we get "ABCDE" in the grid.
We can use dynamic programming or backtracking? With $N=20$, backtracking is hard.
However, the observation is that strings are generated randomly from the matrix.
Most cells will be "covered" by at least one string passing through it.
Also, the grid is periodic (torus).
Maybe we can build candidate matrices based on the strings.
Actually, let's look at the density. $M \approx 600$. Length $\approx 7$. $M \times L \approx 4200$. $N^2 = 400$. Average coverage is $> 10$.
This means almost every cell is part of a known sequence.
Let's maintain a probability distribution or a consensus for each cell $(i, j)$.
But we need to align the strings.
Actually, since we know the strings are substrings of the final matrix, and we want to maximize matches, maybe we can just assume the strings are placed according to their occurrences in the true matrix.
But we don't know the true matrix.
Wait, this is an AHC problem. The solution doesn't need to be perfect (AC is not guaranteed by score, though getting AC implies highest score potentially? No, score depends on $c$. To win, you need $c=M$. But wait, the problem says "If $c<M$... score is proportional to $c/M$. If $c=M$... bonus score for low dot usage". Wait, maximizing score means getting $c=M$.
The scoring function rewards $c=M$ heavily compared to partial matches?
$\mathrm{round}(10^8\times \frac{c}{M})$. Max if $c=M-1$ is $\approx 10^8$. If $c=M$, it's $\ge 10^8$.
Wait, $\frac{2 N^2}{2 N^2-d} > 1$. So if $c=M$, score is $> 10^8$.
Max possible score for $c=M$ is $2 \cdot 10^8$ (if $d=N^2$, i.e. all dots).
But we want $c=M$ primarily.
So we must find a matrix containing ALL strings.

How to find it?
We can view this as a 2D tiling problem where we insert strings.
Since strings overlap significantly, we can try to deduce the grid.
Key idea: The strings are generated from a matrix. The matrix is consistent.
Any two strings that intersect in the matrix must agree on the overlap.
We can try to build a graph or map of "likely intersections".
Or simpler: Use the fact that strings are short.
We can maintain a partial grid. When adding a string, we check consistency. If inconsistent, backtrack?
State space is too big.
However, notice that the grid size is small ($20 \times 20$).
And strings are relatively long (length $\ge 2$).
Also, we have multiple strings.
Maybe we can use a randomized approach:
Initialize grid with unknowns.
Iterate $K$ times:
Pick a string $S_i$. Try to place it at a random valid position (row, col, orientation) that is consistent with current grid.
Update grid?
Or rather, use "majority vote".
For each position $(r, c)$ in the grid, consider all strings $S_i$ and all their placements.
If string $S_i$ placed at $(r, c)$ implies $G[r][c] = X$, then increment count for $X$.
Since the true strings are consistent, for the "true" placement of $S_i$, the vote will favor the true character.
The question is: How do we find the "true" placement?
If we simply sum votes for each $(r, c)$ across all possible placements of all strings, and pick the majority character, what happens?
Suppose the true character at $(r, c)$ is 'A'.
Many strings pass through $(r, c)$ in the true configuration, requiring 'A'.
Other placements of strings that don't pass through $(r, c)$ don't constrain it (or they might constrain it elsewhere).
Wait, if we iterate over all $N^2$ placements for each string, we add constraints.
For a specific cell $(u, v)$, suppose in the true matrix string $S_k$ starts at $(r, c)$ and passes through $(u, v)$. Then $S_k$ constrains $(u, v)$.
What if we consider ALL possible placements of $S_k$ in the grid?
$S_k$ could theoretically be placed anywhere.
But only ONE placement (the true one) is "correct".
If we blindly take the intersection of all possible placements, we might get noise.
BUT, we are told that the output matrix must contain the strings.
This means for each $S_k$, there MUST be SOME placement.
This suggests we can treat it as a constraint satisfaction problem.
Variables: Character for each of 400 cells.
Domain: {A..H}.
Constraints: For each string $S_k$, exists placement compatible with variable values.
This is equivalent to: For each $k$, $\exists (r, c, dir)$ such that $\forall p, Cell[r + (p \cdot dir)][(c+p)\%N] = S_k[p]$.
This is hard to optimize directly due to existential quantifier.

Alternative view:
The problem states $N$ is small, $M$ is reasonable.
Maybe we can guess the "grid" by assuming the strings are aligned with the grid structure.
Wait, is it possible to solve this by "matching"?
The strings are substrings of the torus.
If we collect all strings, and look for frequent overlaps.
E.g., suffix of $S_a$ matches prefix of $S_b$.
This allows us to chain strings together.
If we build a long chain of strings $S_{i_1}, S_{i_2}, \ldots$ overlapping, we can form a longer path.
Eventually we want to form cycles covering the grid.
Since the grid is $20 \times 20$, the diameter is small.
Maybe we can perform a local search:
Start with a random matrix.
Evaluate score (how many strings contained).
If less than $M$, try to improve.
Improvement strategy:
If a string is NOT contained, find a position to insert it.
Inserting it means updating characters in the grid.
If we update characters, we might break other strings.
But since we have high confidence that a valid matrix exists, breaking few while fixing one is a heuristic.
This looks like a SAT solver or MaxSAT, or simply simulated annealing.
Simulated Annealing / Local Search seems appropriate for AHC.
State: The $20 \times 20$ grid.
Moves: Change a cell's value, or swap values.
Cost: Number of satisfied strings.
But cost calculation involves checking if all $M$ strings exist. Checking existence is easy ($O(M \cdot N)$).
Optimization: We want to maximize satisfied strings.
Initial State: Can we start with something?
Maybe initialize with the mode of characters?
Or just random? Random might have low $c$.
Better initial state: Try to satisfy as many as possible greedily.
Greedy strategy: Sort strings?
Sort by length? Shorter strings are easier to satisfy. Or longer strings?
Longer strings are more restrictive.
Heuristic: Place longest strings first.
1. Identify candidate positions for longest string $S_1$. There are $400$ starts $\times 2$ dirs $\times 2$ shifts (actually just shifts along the line).
Wait, for a string of length $L$, and grid size $N$, how many placements?
For horizontal: $N$ rows, $N$ starting cols. So $N^2$ placements. Same for vertical.
Total $2 N^2 = 800$ placements.
For $S_1$, we pick one placement. Assign chars.
Then $S_2$. Pick a placement consistent with $S_1$'s assigned chars.
Conflict resolution: If $S_2$'s required char differs from $S_1$'s implied char, we have a contradiction.
Since a solution exists (the ground truth), conflicts must be resolvable by picking the right placement.
However, we don't know the placement.
But we know that for the TRUE grid, there is a valid assignment of placements for all strings.
Since we want to FIND such an assignment.
This is basically "Exact Cover" or similar?
Wait, if we can satisfy all strings, great.
Let's reconsider the "consensus" idea.
Every string $S_k$ comes from a unique source string in the ground truth matrix.
Let's denote the occurrence of $S_k$ as being at row $R_k$, col $C_k$, direction $D_k$ (in the ground truth).
If we could find these $(R_k, C_k, D_k)$, we could just fill the grid.
How to find these?
Notice that for any string $S_k$, it appears at exactly one location in the ground truth (unless duplicated, but even then, one is sufficient).
Actually, since strings are generated randomly, duplicates are unlikely but possible.
However, we don't care about uniqueness. Just existence.
Key insight: The grid consists of rows and columns that are formed by concatenating parts of these strings.
Let's extract all cyclic shifts of all strings.
Wait, a string "ABCDE" appearing at (0,0) is same as appearing at (0,0).
But "ABCDE" appearing at (0,0) vs "ABCDE" appearing at (0,1) are different.
Let's compute a hash or signature for each string?
Actually, we can just collect the set of all substrings of length 1 (chars), 2, etc. from all strings.
Wait, the characters are just symbols.
Let's go back to the idea of solving for the grid.
Since $N=20$ is small, maybe we can use the property that strings are generated from a full grid.
This implies that for every row $i$, the row string $Grid[i]$ must contain all horizontal strings starting in that row? Not necessarily. Only those selected by the generator.
But we have many strings.
Let's count the frequency of characters at each position $(i, j)$ assuming all possible alignments of all strings.
Wait, if we assume that the "truth" is hidden among all possible alignments, the "consistency" filter might work.
If we consider all pairs of strings $(S_a, S_b)$, and try to align them.
If suffix of $S_a$ matches prefix of $S_b$ (of same length), we can overlay them.
If we build a chain of strings $S_{p_1}, S_{p_2}, \dots$, we can infer a larger pattern.
The lengths are small (up to 12).
We can try to find "longest common extensions".
Given $N=20$, any valid grid will have period $N$ in both dimensions.
The "alignment" of two strings $A$ and $B$ in the grid defines their relative displacement.
Let's define a graph where nodes are strings. Edges represent compatibility.
Or better: Construct a "supergraph" of potential grid rows/columns.
There are only 20 rows and 20 columns.
Each string $S_k$ contributes to one row and one column? No, it lies entirely within one row (if horizontal) or one column (if vertical).
Crucial observation:
In the true matrix, each string $S_k$ is either a horizontal segment of some row $r$, or a vertical segment of some column $c$.
It occupies a set of cells.
If it is horizontal, it belongs to row $r$. If vertical, column $c$.
We don't know for sure which strings are horizontal/vertical or their coordinates.
But we know the set of available rows is $\{0, \dots, 19\}$ and columns $\{0, \dots, 19\}$.
Let's hypothesize: Most strings are consistent with a row-string or column-string decomposition.
Since we want to satisfy all, we can try to assign each string to a (row, col_start) tuple or (col, row_start) tuple.
There are $2 \cdot 20 \cdot 20 = 800$ slots for string starts.
Wait, $N=20$. $N$ rows. $N$ columns.
Horizontal strings live in rows. There are $N$ rows. In each row, there are $N$ possible start positions.
So there are $N^2$ horizontal slots. Similarly $N^2$ vertical slots.
Total slots = $2 N^2 = 800$.
We have $M$ strings. $M \le 800$.
So roughly 1 slot per string.
Some slots might be shared if strings overlap, but each string occupies a distinct set of indices? No, strings can share cells (overlap).
Wait, if strings overlap, they occupy the same cells.
If we can assign each string $S_k$ to a unique "placement" (Row $r$, Start Col $c$), we are done?
Wait, if $S_1$ and $S_2$ both map to Row 0, Start Col 0?
That implies they are the same string?
The problem says "subsequences". $s_i$ is a subsequence of $a$.
In the input, $s_i$ are just strings.
They are extracted from $a$. So they correspond to specific substrings of $a$.
Since $a$ is a valid torus, $S_k$ corresponds to $a[r][(c)..]$ or $a[..][c]$.
Since $M \approx 600$ and $N^2=400$, many strings might share the same starting position in the grid (e.g. same substring).
But we just need to cover the requirements.
Actually, we can simplify the problem:
Find a set of 20 horizontal strings $H_0, \dots, H_{19}$ and 20 vertical strings $V_0, \dots, V_{19}$?
No. We don't know the full rows and columns. We only know partial segments.
However, since $N=20$, the number of constraints is tight.
Maybe we can use a SAT solver or a simple randomized local search.
Given the constraints ($N=20$, $M \le 800$), and the nature of AHC, a randomized greedy approach often works.

Algorithm Proposal:
1. Represent the grid as $20 \times 20$. Initialize with 0/empty.
2. Maintain a data structure to track which strings are satisfied.
3. Iteratively pick an unsatisfied string.
4. Try to place it in the grid.
   - Iterate over all possible placements (Row 0..19, Col 0..19, Direction Horizontal/Vertical).
   - Check if placement is consistent with currently filled cells.
     - Consistency means: if a cell in the placement already has a character assigned, it must match the string character.
   - Among consistent placements, pick one that "fits best" or "contributes most" to future fits?
   - Or just pick any consistent placement. If none exists, flip a random cell and retry? Or change a conflicting cell to resolve conflict?
   - Actually, changing a cell might break a previously satisfied string.
   - Heuristic: Minimize disruption. Score = (count of satisfied strings) - penalty for changes.

Refined Algorithm:
Use Simulated Annealing / Hill Climbing.
State: Grid (char array).
Objective Function: $Score = (\text{number of satisfied strings})$.
Initial State: Fill grid randomly with 'A'..'H'.
Loop until convergence or timeout:
  Best state = Current state.
  Loop K times:
    Propose a move.
    Move type:
      1. Change a random cell value. (Uniformly from 'A'..'H').
      2. Insert a string? Hard to define directly as a move.
      Better: Pick a random cell $(r,c)$ and a random neighbor relation?
      Actually, changing cells is standard.
    Calculate delta in objective.
    If improvement, accept. Else with probability $e^{-\Delta/T}$ accept.
    Update temperature.

Is checking objective expensive? $M \times N$ checks.
$600 \times 400 = 240,000$ operations. Per iteration.
If we do 1000 iterations, that's $2.4 \times 10^8$, might be tight for 1 sec?
Wait, execution time limit is usually generous for C++ (2s?).
Checking if a string is present takes $O(N)$ time (length of string). Sum of lengths is $M \times 7$.
So checking is $O(\sum |s_i|) \approx 800 \times 7 \approx 5600$ ops.
This is very fast.
So evaluating one full state is ~6k ops.
We can afford millions of evaluations.
Wait, is "simulated annealing" effective for finding a global optimum (satifying all)?
With $N=20$, maybe there are many local optima.
But since a valid solution exists (ground truth), it should be reachable.
Is the space connected? Yes, we can change any character.
However, we might get stuck in a configuration where no moves improve the score locally.
But we want to reach $c=M$.
Maybe we can prioritize satisfying strings.
Instead of pure SA, use a constructive approach combined with repairs.

Better Constructive Approach:
1. Parse all strings.
2. Maintain a frequency table for each cell $(i,j)$ of the character that supports it?
   No, cells are coupled.
3. Let's try to deduce the rows/cols.
   Since strings are substrings, if we concatenate them correctly, we get rows/cols.
   Consider all pairs of strings $S_i, S_j$.
   Try to find an offset $(dr, dc)$ such that they overlap consistently.
   But this is complex.

Let's stick to the Local Search on the Grid.
However, instead of random moves, let's use the information in the strings.
We can calculate the probability that cell $(i, j)$ is character $X$.
For each string $S_k$ and each possible placement $P$ (that places $S_k$ somewhere):
   If placement $P$ is compatible with the "votes" so far (or just generally):
   Add weight to characters in the covered cells.
   Since we want to support ALL strings, any placement that is valid for $S_k$ is a "witness".
   If we sum the "support" for each grid position from all possible witnesses, and pick the majority, we get a candidate grid.
   Is this enough?
   Suppose the true grid has 'A' at (0,0). String 1 (which uses (0,0)) is placed at some offset.
   If we only look at one placement of $S_k$, we add 'A' at (0,0).
   But we don't know the placement. We have to consider all placements.
   However, most placements of a string $S_k$ will conflict with the true matrix.
   Only the "correct" placements (and possibly others that accidentally match) are compatible.
   Wait, if we consider ALL placements of $S_k$, we are essentially asking "Could $S_k$ be at (0,0)?".
   If it can be at (0,0) without contradicting itself, we cast a vote for (0,0) for $S_k$'s content?
   This is a bit circular.

Let's use the "Majority Vote" on the True Matrix hypothesis.
For each string $s$, we know it must match the grid at *some* location.
Let's try to determine the location.
Actually, the problem is symmetric.
Let's try to match the strings to the grid using a bipartite matching or max flow?
No, grid is not just edges.
Actually, given the small grid size, maybe we can use **Genetic Algorithms** or **Memetic Algorithms**.
Or simpler: **Iterative Repair**.
1. Initialize grid with random characters.
2. Identify unsatisfied strings.
3. For each unsatisfied string, try to modify the grid to satisfy it.
   Modification: Change one character in the grid.
   Try all modifications that might fix the string?
   For string $S$, it fails if no placement is valid.
   We need to create a valid placement.
   This means ensuring that for some $(r, c, dir)$, $Grid[...]$ matches $S$.
   So we select a target $(r, c, dir)$ that currently has mismatches.
   To make it valid, we must change the mismatching cells to match $S$.
   Changing one cell might break other strings.
   We pick the placement that causes minimum damage (fewest existing satisfied strings broken, or minimal number of changes).
   Since we want to maximize satisfied strings, this is like a greedy repair.

   Wait, if we change a cell, we might satisfy the new string but lose old ones.
   Since we start from scratch, maybe we should just find the BEST configuration.
   
   Wait, I missed a crucial point. The scoring is based on $c=M$.
   We absolutely need $c=M$.
   The "random" initialization might never reach $c=M$.
   However, since the inputs come from a specific valid matrix, maybe we can recover that specific matrix.
   The inputs are "contiguous vertically or horizontally".
   Let's collect all substrings of length $\ge 2$ from the input strings.
   Actually, we can use a Trie or suffix structure.
   Also, we know $N=20$.
   What if we try to reconstruct row by row?
   If we know row 0, can we derive row 1?
   Not really.
   
   Let's go back to the idea that strings are substrings of the grid.
   For a specific string $s$, there are limited ways it can align with the grid.
   Let's assume that for each string $s$, there is a set of "candidate" placements in the grid.
   Since the strings are generated from the grid, they MUST be consistent with the grid.
   This means if we find a grid that satisfies all strings, it is a solution.
   
   Let's try a hybrid approach:
   1. **Frequency Analysis**: Build a histogram of character frequencies for every possible relative distance pair?
      No, simpler: Just collect the set of all characters appearing in the grid? No.
   
   2. **Constraint Propagation**:
      Treat the grid as variables $x_{i,j}$.
      Constraints: $\bigvee_{placements} (\text{match})$.
      This is a disjunction.
      However, with $N=20$, we can try to satisfy the strongest constraints first (longest strings).
      
      Let's use the following algorithm:
      **Initialization**: Set grid cells to '?' (unknown).
      **Iteration**:
        Find a string $s_k$ that is NOT satisfied (requires placing).
        Look for a valid placement for $s_k$ in the current grid (cells with '?').
        If found, fill those cells.
        If not found (mismatches with known cells), we must change some known cells to accommodate $s_k$.
        Which cells to change? Those involved in $s_k$ that are mismatched.
        To minimize damage, pick the mismatched cell that affects the FEWEST number of *other* currently satisfied strings (or total strings).
        Update the grid. Repeat.
        
        To implement "damage":
        Precompute a table `impact[cell_index][new_char]` -> list of strings affected/broken?
        Too much memory/time.
        Simplified: Just keep track of satisfied strings. When changing a cell, recalculate satisfied status efficiently?
        Recalculating all strings takes time but is feasible ($5600$ ops).
        
      This looks promising. It's a Greedy Descent with Backtracking/Repair.
      However, pure Greedy can loop or get stuck.
      But since we just need to reach $c=M$, and we know a solution exists, and the solution space isn't too wild (strings are local), this might converge.
      
      One optimization:
      Instead of picking an arbitrary unsatisfied string, pick the one that is "most constrained" or "hardest to place".
      Or simply iterate all strings in order.
      
      Refined Repair Strategy:
      1. Start with random grid or grid derived from 'votes'.
         Deriving from votes:
         For each $(r, c)$, look at all possible ways strings could pass through $(r, c)$.
         Since we don't know positions, we consider ALL possible alignments of strings with the grid.
         For string $s_i$ at length $L$:
           It covers cells $s_i[0..L-1]$.
           It can be placed starting at any $(r, c)$.
           For each placement $(r, c)$, for each index $p$, cell $(r+dp, c+pp)$ gets candidate char $s_i[p]$.
           Wait, this assumes $s_i$ IS at $(r, c)$.
           But we don't know if it is.
           However, since it MUST be somewhere, maybe we can count votes.
           For each $(r, c)$, we collect all chars suggested by ANY valid string placement passing through $(r, c)$.
           This is dangerous because most placements are wrong.
           
      Let's stick to **Constructive + Conflict Resolution**.
      
      Step 1: Initialize grid with random characters.
      Step 2: Calculate $c$ (satisfied strings).
      Step 3: While $c < M$ and steps < LIMIT:
        Pick an unsatisfied string $s_i$.
        Find a placement for $s_i$ that maximizes the number of satisfied strings overall (including re-satisfying others)?
        Or simply: Find the "best" placement for $s_i$ given the current grid state.
        Best means:
          1. If consistent with all pre-filled cells, keep it.
          2. If conflicts, choose a conflict that costs least?
             Cost = number of currently satisfied strings that would become unsatisfied if we change the overlapping cell.
             Wait, if we change a cell, we fix $s_i$ (potentially) but might hurt $s_j$ (which matched that cell before).
             Since we need to fix $s_i$ eventually, and $s_j$ might be fixed later or never.
             Prioritize fixing $s_i$.
             So we want to find a placement for $s_i$ such that the overlap with existing filled cells is minimized?
             Or we overwrite.
             Since we want to satisfy ALL strings, eventually all cells might be fixed.
             The "true" grid has NO conflicts.
             So we just need to drive the state towards the true grid.
             If we encounter a conflict, it means we picked a bad random start, OR we need to flip the bit.
             
      Let's try this: **Local Search with Temperature**.
      1. Start with Random Grid.
      2. Loop:
         - Pick a random string $s_i$.
         - Find all placements of $s_i$ that are consistent with the CURRENT grid.
         - If there is at least one consistent placement:
            - Select one (randomly?) to reinforce?
            - Or, pick the one that agrees with the MOST other characters?
              This sounds like we are voting again.
            - Better: Pick the one that minimizes the Hamming distance to the CURRENT grid (if we treat '?' as neutral, but here grid is filled).
              Wait, if grid is filled, any mismatch is an error.
              If we find a consistent placement, we don't change anything.
         - If there are NO consistent placements for $s_i$:
            - We MUST change some cells to satisfy $s_i$.
            - This suggests the current grid is far from the true one.
            - We need to modify the grid.
            - We should change a cell that participates in $s_i$'s optimal (least conflicting) placement.
            - Specifically: Pick the placement of $s_i$ that has the minimum Hamming distance to the current grid (counting mismatches).
              Wait, Hamming distance on the grid implies the string is shifted relative to grid?
              No, Hamming distance between the string $s_i$ and the grid projection.
              Let's iterate all placements.
              For a placement $(r, c, dir)$, calculate mismatch count: number of cells $(r', c')$ covered where $Grid[r'][c'] \neq s[p]$.
              Pick placement with MIN mismatch count.
              Let this be $k$.
              If $k=0$, we are good (consistent).
              If $k > 0$, we have a conflict.
              We should resolve the conflict.
              How? By modifying the grid to match this placement.
              But we might destroy satisfied strings.
              Modify the grid at the mismatching positions to match $s_i$.
              This increases $c$ by 1 (maybe more if we created a match for another string) but decreases $c$ for others that relied on those cells.
              Net change can be calculated.
              
      So the operator is:
      For a chosen string $s_i$:
        Compute all $2N^2$ placements.
        Score each placement: (Matches - Mismatches? No).
        We want to find a placement that is close to the current grid.
        Actually, since we want to MAXIMIZE total satisfied strings, we can evaluate the impact.
        Impact of switching to placement $P$ for $s_i$:
          Old state: $s_i$ contributes 0 to score.
          New state: $s_i$ contributes 1 to score IF it becomes valid.
            Wait, if we force $s_i$ to match via modifications, it IS valid now.
          But modifications affect other strings $s_j$.
          For each cell $(r', c')$ changed from $v_{old}$ to $v_{new}$:
            For each string $s_j$ that covers $(r', c')$, check if $s_j$ still matches.
            If yes, no loss.
            If no, loss of 1 (unless multiple cells cause one string to fail, we shouldn't double count loss).
        This is computationally heavy but $M$ is small.
        $M \le 800$. $N^2 = 400$.
        Evaluating impact for one string $s_i$:
          Iterate all placements (800).
          For each, simulate changes (at most length $L \le 12$).
          Check impact on all other strings.
          Pick best placement.
        Do this for all unsatisfied strings (or random subset).
        Update grid.
        
      This looks like a solid greedy hill climbing.
      We need to handle "unsatisfied" properly.
      Initially, many strings might be unsatisfied.
      We sort strings by length (descending)? Or just random.
      Long strings are more constraining. Fix them first.
      
      Implementation details:
      - Grid size $20 \times 20$.
      - Strings storage.
      - Helper function `check_score(grid)` -> returns vector<bool> or score.
      - Loop:
        - Find unsatisfied strings.
        - If none, done.
        - For each unsatisfied string, try to improve.
        - Actually, we want to globally optimize.
        - Simplest: Pick an unsatisfied string, try to find the BEST modification of the grid to satisfy it.
          The "BEST" modification is the one that minimizes the drop in satisfied strings among others, while gaining $s_i$.
          Or simply: Iterate all possible modifications to the grid that make $s_i$ satisfied.
          Modifications involve changing characters in the grid to match $s_i$ at some placement.
          Wait, to satisfy $s_i$, we must ensure $\exists$ placement such that $Grid[...]$ matches $s_i$.
          So we need to choose a placement and ensure $Grid$ matches.
          The "cost" of choosing a placement is the number of currently satisfied strings that will be broken.
          We want to minimize cost.
          So for each unsatisfied $s_i$:
             Iterate all possible placements.
             For each placement, compute "Cost" = (Number of satisfied strings that would be broken by forcing this alignment).
               - "Breaking" means for a string $s_j$ that currently contributes to score, after forcing alignment of $s_i$, it no longer contributes.
               - Note: Forcing alignment means overriding grid characters.
               - Wait, if $s_j$ is satisfied by multiple placements, and we change one cell, it might still be satisfied by another placement.
                 We should check if $s_j$ remains satisfied after the hypothetical override.
             Pick placement with lowest Cost.
             Apply the override (overwrite grid cells).
             Repeat until all satisfied or timeout.
             
      Optimization:
      Overwriting multiple cells at once might be needed.
      The "override" sets the characters at the placement points.
      Does it matter WHICH characters?
      $s_i$ dictates the characters. We just set them.
      Wait, if a cell was already set correctly by $s_i$ (no conflict), we don't need to change it.
      We only change cells where $Grid \neq s_i$.
      If we have to change cells, we do so.
      
      Complexity:
      Unsatisfied strings: up to $M$.
      Per string:
        Placements: $2 N^2 \approx 800$.
        Cost calculation:
          Iterate all other strings $M \approx 600$.
          Check if satisfied.
          This check involves scanning the grid.
          Scanning takes $O(N)$ (string length).
          Total cost calc per string: $800 \times 600 \times 10 \approx 4.8 \times 10^6$.
          If we do this for 50 unsatisfied strings: $2.4 \times 10^8$.
          This is a bit high for a second, but acceptable in C++.
          Wait, we can optimize the check.
          Instead of recomputing score from scratch for all strings, we can incrementalize.
          But $M$ is small enough that we can afford full scan.
          
      Initial State:
      Should we start with random? Or with a heuristic guess?
      Random is risky.
      Can we start with a grid that maximizes agreement with "partial" info?
      E.g. For each cell $(i, j)$, consider all strings and all their placements. If string $s$ passes through $(i, j)$ with char $c$, increment count[c].
      If a cell is not covered by any string, it stays random.
      This gives a prior.
      Let's use this as Initialization.
      
      Wait, the strings are generated from a SINGLE valid matrix.
      This implies that the "prior" based on all strings SHOULD strongly correlate with the truth.
      However, as established, we don't know placements.
      But wait! The strings ARE generated from the matrix.
      This means there IS a placement for each string.
      If we consider ALL possible placements of ALL strings, and look at the aggregate statistics...
      Does the "Truth" stand out?
      In the true configuration, for every string $s$, there is exactly one placement (or few) that matches the actual grid content perfectly (Hamming dist 0).
      Any WRONG placement will have Hamming dist > 0.
      If we simply look for the placement of each string that has minimum Hamming distance to the grid...
      And repeat iteratively.
      This is exactly the iterative correction described above.
      
      Algorithm Summary:
      1. Read Input.
      2. Initialize Grid: Randomly fill or 'Mode' fill.
         Mode fill: For each cell $(r, c)$, count candidates from all strings considering ALL possible placements?
         No, this assumes we know which placement.
         Let's stick to Random Initialization with a fallback to 'Mode' if we can calculate it easily.
         Actually, for AHC 004, solutions typically use randomized hill climbing.
         So:
         Grid $G[N][N]$. Init with random 'A'-'H'.
         Loop iterations:
           If all strings satisfied, return Grid.
           Pick a random unsatisfied string $S$.
           Try to improve grid to satisfy $S$.
             For each placement of $S$ (800 options):
               Calculate penalty:
                 Simulate changing cells to match $S$.
                 Penalty = (Current satisfied strings - New satisfied strings).
               Pick placement with min penalty.
               Apply changes.
               Update grid.
           Keep track of best Grid seen.
           Accept/Reject moves?
           Since we just want to fix $S$, we just apply the best move (Steepest descent on penalty).
           Wait, steepest ascent on "Satisfied Count".
           If penalty is positive (satisfaction drops), do we still accept?
           We MUST accept, because we didn't satisfy $S$. We increased score by +1 (for $S$) - Penalty.
           Ideally Penalty < 1.
           If Penalty >= 1, we lose progress on other strings.
           But eventually we need to fix everything.
           So we always apply the move with smallest penalty (most likely to preserve other strings).
           After applying move, we continue.
           
      Wait, we should probably run multiple epochs or restarts.
      With $N=20$, maybe we need 1000-2000 iterations of trying to fix unsatisfied strings.
      
      Wait, the scoring function is tricky.
      "If $c=M$, $\mathrm{round}(10^8\times \frac{2 N^2}{2 N^2-d})$."
      We need $c=M$ first.
      Minimizing $d$ (number of dots) improves score if $c=M$.
      If we use '.' in the output, does it help?
      The problem says output `.`.
      My grid initialization uses 'A'-'H'.
      If we leave a cell as '.', it's empty.
      Does it allow more strings? No, '.' breaks string matches.
      But if a cell is truly irrelevant (not needed for any string match), maybe we can set it to '.' to save score?
      Wait, $2 N^2 / (2 N^2 - d)$.
      If we have $N^2$ characters, $d=0$, score $10^8$.
      If we have $N^2-d$ characters.
      The formula implies we get MORE points for having fewer dots.
      Example: $N=20, N^2=400$.
      $d=0 \implies 10^8$.
      $d=400 \implies 10^8 \times \frac{800}{400} = 2 \cdot 10^8$.
      So using dots INCREASES score?
      Why? "If $c=M$, $\mathrm{round}(10^8\times \frac{2 N^2}{2 N^2-d})$."
      This formula seems to reward having fewer "occupied" cells?
      Wait. Usually score is $10^8$ max. Here it can be $2 \cdot 10^8$.
      Ah, maybe it penalizes dense packing? Or maybe it's just an arbitrary scale.
      Anyway, my goal is $c=M$.
      If I produce a valid matrix (fully filled with A-H), I get $d=0$ (assuming I didn't use '.' explicitly).
      Wait, the problem says output `.`.
      If I output 'A', then $d$ decreases.
      So maximizing 'filled' cells is GOOD if $c=M$.
      Wait, $\frac{2 N^2}{2 N^2-d}$. Denom is smaller when $d$ is larger.
      So fraction is larger when $d$ is larger.
      So YES, using `.` increases the score factor.
      So we should prefer `.` over characters if $c=M$.
      Wait, why would that be? "Empty" cells are allowed.
      It's like a compression ratio?
      Regardless, $c=M$ is the prerequisite.
      Since the inputs come from a fully populated matrix, we should probably populate the grid with A-H to match them.
      If I put `.`, it will likely violate a string that expects 'A'.
      Unless the string has a gap? No, strings are contiguous.
      So '.' will hurt $c$.
      Thus, to get $c=M$, we must avoid '.' unless a string allows it (none do).
      So for $c=M$, we should use minimal '.' (0).
      So my focus should be on filling A-H to satisfy all strings.
      
      One more detail: "Note that if the index is greater than or equal to N, we take the remainder divided by N".
      This confirms Torus.
      
      Complexity check again:
      Outer loop: e.g. 2000 iterations.
      Inner: Pick unsatisfied string (worst case M).
      Inner-inner: 800 placements.
      Check: 600 strings.
      Check cost: Scan grid.
      This is quite slow. $2000 \times 600 \times 800 \times 10 \approx 10^{10}$ ops.
      Too slow.
      Need to optimize.
      Optimization:
      Don't check ALL other strings.
      Only check strings that cover the modified cells.
      Precompute which strings cover which cells?
      Map cell -> list of strings covering it.
      When changing cell, re-check ONLY those strings.
      Updating status of strings is efficient.
      Maintaining a count of satisfied strings.
      
      Optimized Logic:
      Data structures:
      `satisfied_count` = integer.
      `is_satisfied[M]` = bool.
      `covers[N][N]` = vector<int> of strings indices covering this cell.
      `string_cells` = for each string, list of (cell_idx, char_req) relative to placement.
      
      Function `check_string(idx, placement)`:
        Verify if string idx is satisfied by placement against current grid.
        If yes, mark satisfied.
        Return bool.
      
      Algorithm:
      1. Initialize Grid.
      2. Compute initial `covers`.
      3. Compute initial `is_satisfied` for all strings.
         (Brute force check all strings against grid).
         Mark satisfied ones. Update `satisfied_count`.
      4. Loop while `satisfied_count < M`:
         Pick worst unsatisfied string? Or random.
         Let's pick a string that is furthest from being satisfied?
         Actually, just picking any unsatisfied string is fine.
         Let it be `target_str`.
         
         Best_Placement = None
         Min_Damage = Infinity
         
         Iterate all 800 placements for `target_str`.
           Damage = 0
           For each cell `(r,c)` changed to `new_char` by this placement:
             // Check if this change hurts OTHER satisfied strings
             For each `other_str` in `covers[r][c]`:
                If `other_str != target_str` AND `is_satisfied[other_str]`:
                   Check if `other_str` is still satisfied with this change?
                   Wait, `other_str` might be satisfied by MULTIPLE placements.
                   If we rely on the fact that we haven't checked `other_str` thoroughly?
                   We need to know if `other_str` is still satisfied.
                   To check this, we need to see if `other_str`'s constraints are met.
                   This requires scanning the whole `other_str` in the MODIFIED grid.
                   
                   Is it cheaper to scan?
                   Length of string is small (<= 12).
                   Cost to check one other string is small.
                   Number of other strings covering `(r,c)` is small?
                   Since strings are random and sparse-ish (but M is high), average density is $600/400 = 1.5$.
                   A cell might be covered by a few strings.
                   Say 5 strings on average.
                   So checking 5 strings per cell.
                   Placement has length <= 12. Cells <= 12.
                   So cost per placement is $12 \times 5$ ops (checking satisfaction).
                   Total cost per `target_str` selection: $800 \times 12 \times 5 \approx 48000$ ops.
                   If we loop this 50 times (per epoch): $2.4 \times 10^6$.
                   Total iterations: 100? Total time ~ 10^8.
                   Feasible.
           
           Note on "still satisfied":
           We need a way to check this quickly.
           Maybe store the "constraints" for each string?
           A string $S$ is satisfied if $\exists$ placement $P$ s.t. $Grid$ matches.
           If we modify grid, we invalidate $P$ if it conflicts, but $S$ might be satisfied by $P'$.
           So checking if $S$ is satisfied is just calling `matches(S, Grid)`.
           This takes $O(N)$ (checking all placements of $S$).
           Wait, `matches` checks if $S$ appears anywhere.
           We can optimize:
           If we only changed one cell, `matches` for a string covering that cell needs recheck.
           We can check if `matches` fails specifically due to this cell.
           But `matches` scans the whole grid? No, just checks if $S$ occurs.
           We can just run `matches` on the modified grid for the affected strings.
           Affected strings are those covering the modified cells.
           For each such string, re-run `matches` logic.
           If it becomes false, `Damage` increments.
           
           Wait, we need to be careful with "Damage".
           If `matches` fails for `other_str`, `Damage++`.
           But `target_str` itself becomes satisfied (+1 to count).
           So Net Change = $1 - \text{Damage}$.
           We want to maximize Net Change (minimize Damage).
           
           Is `matches` fast?
           For a string $S$, we check if it exists in Grid.
           Since $S$ is short, checking all $N^2$ placements takes $O(N^2)$.
           Wait, $N^2=400$.
           Doing this for each affected string inside the placement loop is slow.
           $800 \text{ placements} \times 10 \text{ strings/cell} \times 400 \text{ checks}$. Too slow.
           
           Wait, we don't need to re-run FULL `matches` logic.
           We know `other_str` WAS satisfied.
           Why did it satisfy? Because there was a placement $P_{old}$ that worked.
           Did we touch cells in $P_{old}$?
           If we didn't touch ANY cell in $P_{old}$, then `other_str` is still satisfied.
           If we touched cells in $P_{old}$, we need to check if they still match.
           But `other_str` might have multiple working placements.
           If it had only one, we check if it's broken.
           If it had multiple, it might survive.
           To be safe, we assume worst case? No, that's bad.
           Actually, we can just verify `matches` by checking the specific placement that made it satisfied?
           We don't store that.
           BUT we can assume that if a string is satisfied, it's "weakly" satisfied unless it's "uniquely" satisfied by a set of cells we are destroying.
           Let's just compute the check `matches` from scratch but optimized?
           Actually, $S$ is short.
           To check if $S$ exists in Grid, we scan $2N^2$ positions.
           If we only changed a few cells, maybe we can prune?
           But checking 2N^2 positions is fast (800 checks).
           Doing this for 10 strings is 8000 ops.
           Inside the placement loop (800 iterations) -> $8000 \times 800 \approx 6.4 \times 10^6$.
           Still a bit heavy if repeated 100 times.
           $6 \times 10^8$ ops.
           C++ can handle $\approx 10^8$ ops per sec easily, sometimes $5 \cdot 10^8$.
           Might be TLE.
           
           We need to optimize `matches`.
           Precompute `matches`? No, grid changes.
           Wait, we don't need to check ALL strings.
           We only check strings that cover the MODIFIED cells.
           Number of modified cells is length of string $\le 12$.
           For each modified cell, iterate over `covers[r][c]`.
           Collect a set of `other_strings`.
           For EACH `other_string` in this set:
             Check if `matches(other_string)` is true in the modified grid.
             Since `matches` is called often, we need to be fast.
             Optimization:
             `matches(s)` checks all placements.
             If $s$ is satisfied by placement $P_{prev}$, and we didn't change cells in $P_{prev}$, then it's still satisfied.
             If we changed cells in $P_{prev}$, we need to check.
             Can we track which placement satisfies each string?
             No, multiple.
             But we can maintain a flag `has_any_valid_placement`.
             For each string, we can precalculate its "valid placements" count?
             No, depends on grid.
             
             Let's try a different approach.
             We don't need to check ALL other strings.
             Most strings are disjoint? No.
             But checking `matches` on a string of length 12 takes very little time.
             Wait, we are doing this check 800 times (for placements).
             Total 640,000 checks.
             Each check iterates 800 placements? No.
             `matches` iterates 800 placements.
             $6.4 \times 10^5 \times 800 \approx 5 \cdot 10^8$.
             Still high.
             BUT, we don't check ALL strings.
             We only check strings that overlap with the modified region.
             Since modified region is small (len $\le 12$), there are few strings overlapping it.
             How many strings cover a single cell?
             In random generation, $M/N^2 \approx 1.5$.
             So on average 1.5 strings per cell.
             Wait, strings are long.
             But they are distributed.
             Wait, `covers` stores list of strings passing through $(r,c)$.
             Size of `covers[r][c]`?
             On average, if strings are uniformly placed, probability a string covers $(r,c)$ is $L/N$ (roughly).
             Expected number = $M \times L / N$.
             $600 \times 7 / 20 = 210$?
             That's huge.
             Every cell is covered by MANY strings.
             Because strings wrap around and $M$ is large.
             If every cell is covered by 200 strings, then checking "modified region" is expensive (checking 12 cells * 200 strings).
             That defeats the purpose.
             
             Wait. If every cell is covered by 200 strings, then changing a cell affects 200 strings.
             We need a way to NOT check 200 strings fully.
             Notice that if a string $S$ is satisfied, it has AT LEAST ONE valid placement.
             Most strings have MANY valid placements (due to randomness/duplicates).
             Actually, since we are constructing the grid to MATCH the strings, we are likely converging to a state where strings have valid placements.
             Maybe we can just use a simple heuristic:
             Assume that if a string $S$ was satisfied before, it is likely to remain satisfied UNLESS we specifically break the ONLY placement that satisfied it.
             BUT, we don't know the placement.
             
             Alternative idea:
             Since $N=20$ is small, maybe we can solve it using **Backtracking**?
             No, $N=20$.
             Wait, the inputs are generated from a specific matrix.
             Let's revisit the **Consensus** idea.
             For each cell $(i,j)$, what is the most frequent character among all possible valid configurations?
             Valid config = a matrix where all strings appear.
             Since we want to find THE matrix that produced the strings (or similar),
             Let's consider that each string $s_k$ corresponds to a segment in the grid.
             Let's assign each string $s_k$ a "position" $(r, c, dir)$.
             We need to choose one position for each string.
             This is the "Maximum Weight Perfect Matching" or similar?
             No, it's not a matching problem. It's a "Set Packing" on a grid?
             Actually, we have $M$ items (strings). We need to pack them into the grid (positions).
             Items are lines of length $L$.
             The grid has capacity 1 (one char per cell).
             Conflicts occur if two items require different chars at the same cell.
             We want to pack all items.
             Wait, items are strings. Two items $S_a, S_b$ can overlap if they agree on the intersection.
             This looks like finding a "Supersequence" or extending a set of intervals.
             Since we need to fit ALL strings, this is exactly the problem:
             Find a labeling of the grid vertices such that each string $s_k$ is realized as a subsequence.
             
             This is equivalent to: For each string $s_k$, select a valid placement $p_k \in P_k$ (set of valid geometric placements).
             Such that $\forall u \in Grid$, $\forall k, l$: if $u \in p_k$ and $u \in p_l$, then $s_k[u \text{ relative}] = s_l[u \text{ relative}]$.
             If we find such a selection, we are done.
             This is a CSP (Constraint Satisfaction Problem).
             Variables: For each string $k$, choice of placement $p_k \in \{0, 1, \dots, 800\}$. (Index of placement).
             Constraints: Compatibility.
             However, this ignores the "grid" consistency (transitive closure).
             Compatibility is transitive.
             If $S_a$ and $S_b$ agree on overlap, they define consistent chars.
             If $S_c$ also agrees with $S_a$ and $S_b$, it's consistent.
             If we find assignments for all strings, we can just read off the grid (taking union of constraints).
             Are there multiple valid grids?
             Probably yes, but any valid one works.
             
             So, can we solve this CSP?
             Variables: $k \in \{1..M\}$. Domain: 800 placements.
             We want to select ONE domain value for each variable such that all pairwise constraints are met?
             Wait, constraints are only defined on overlaps.
             If $S_a$ and $S_b$ DO NOT overlap, there is no direct constraint.
             BUT, they might overlap transitively with $S_c$.
             Actually, if we enforce consistency, we implicitly enforce the grid structure.
             If we select placements $P_k$ for all $k$, we generate a set of required characters.
             If the set of requirements is consistent (no contradictions), we have a solution.
             Contradiction example: $P_a$ puts 'A' at $x$, $P_b$ puts 'B' at $x$.
             If we avoid such pairs, we are good.
             Since $M$ is up to 800 and $N$ is 20, brute force is impossible.
             But we can use **Backtracking with Forward Checking** (DLZ) or **MRV (Minimum Remaining Values)**.
             Or just a randomized hill-climbing on the choice of placements.
             
             Wait, if we choose placements, do we need to fill the grid?
             Yes.
             Let's refine the "Placement Selection" heuristic.
             We need to select $p_1, \dots, p_M$.
             Let's initialize each $p_k$ randomly.
             Check consistency.
             If inconsistent:
               Find conflicting pair $(a, b)$.
               Try to resolve.
               Resolve: Change $p_a$ or $p_b$ (or both) to remove conflict.
               Prefer change that resolves conflict and minimizes new conflicts.
             
             This is a standard "Graph Coloring" or "Constraint Solving" setup.
             Since the conflict graph is dense (many strings cross paths), we need a good solver.
             However, $N=20$ suggests grid locality.
             
             Wait, the previous "Grid Fill" approach was effectively exploring this.
             The "Placement Selection" view is more structured.
             Given the constraints of time, maybe the Grid Filling (Simulated Annealing on Grid) is safer/implementable.
             Let's refine the "Check Damage" part to make it fast.
             We only check strings that cover the modified cells.
             But we saw that `covers` is large?
             Wait, is `covers` really large?
             $M \approx 600$. $N^2 = 400$.
             If strings are random, average overlap is small.
             Wait, $L \approx 7$.
             Number of string-start positions is $800$.
             For a random string start, it covers 7 cells.
             Total cell-coverage = $600 \times 7 = 4200$.
             Average coverage per cell = $4200 / 400 = 10.5$.
             So each cell is covered by ~10 strings.
             This is manageable! 10 strings is not 200.
             My previous estimate of 200 assumed uniform distribution of STARTS, but maybe overlaps are less frequent or I overestimated.
             Even with 10 strings, it's fast.
             $12 \text{ cells} \times 10 \text{ strings} = 120$ checks.
             Each check `matches` is slow?
             `matches` checks all $2 N^2$ placements?
             Yes, checking if a string exists in the grid requires checking all placements.
             $2 N^2 = 800$ placements.
             Inside `matches`: checking one placement takes $O(L)$.
             So `matches` cost $\approx 800 \times L$.
             Wait, checking one placement means verifying characters.
             Verifying $S$ against Grid at $(r,c)$ takes comparing $L$ chars.
             $L \approx 7$.
             So `matches` $\approx 5600$ ops.
             Total cost for one placement check in repair:
             10 affected strings $\times$ 5600 ops $\approx 56000$.
             Plus we loop 800 placements.
             $800 \times 56000 \approx 4.4 \times 10^7$.
             Doing this for 100 unsatisfied strings $\approx 4 \times 10^9$. Too slow.
             
             We need to speed up `matches`.
             Notice that `matches` is called for strings that were ALREADY satisfied.
             If we don't change their relevant cells, they stay satisfied.
             Which cells are relevant? All cells in their placement.
             If we modified grid cells, we need to check if any satisfied string relies on those cells.
             But if a string was satisfied, it means there existed a placement $P_{sat}$ matching.
             If $P_{sat}$ does NOT use any of the modified cells, it is still valid.
             If it DOES use modified cells, we must check if the other cells in $P_{sat}$ still match?
             Wait, if $P_{sat}$ uses a modified cell, and we changed it, then $P_{sat}$ is broken (unless we coincidentally changed it back to the same value? No, we assume it broke).
             So if any cell in $P_{sat}$ was modified, $P_{sat}$ is invalid.
             But maybe there is ANOTHER valid placement $P_{sat}'$?
             So, for a string $S$ to be considered still satisfied, we must check:
             Does $S$ have a valid placement that does NOT overlap the modified region (checked by seeing if any cell in overlap is unchanged)?
             Or if it does overlap, check if it matches.
             This requires knowing which placements were valid initially.
             We can store `valid_placements[S]` list of indices?
             Storing all valid placements for all strings?
             In the initial step, we can find all strings that are satisfied in the initial grid.
             Store the set of valid placements for each satisfied string?
             Maybe just the ONE placement that satisfied it (arbitrary).
             If we assume it's UNIQUE, we might be wrong.
             BUT, if a string is satisfied by $P_1$ and $P_2$.
             And we change a cell that is in $P_1$ but not $P_2$.
             Then $P_1$ is broken, but $P_2$ survives. String remains satisfied.
             If we assume it relies on $P_1$, we think it breaks. That is an overestimate of damage.
             Wait, we want to MINIMIZE damage. Underestimating damage is bad (we accept bad moves).
             So we should be careful.
             Actually, checking all valid placements for a string is expensive.
             
             Wait, let's look at the problem again.
             $N=20$.
             Is it possible to solve this by simple **Greedy Construction**?
             Sort strings by length descending.
             For each string, place it in the grid.
             Where?
             We need to place it somewhere consistent.
             Initially, no constraints. We can place anywhere.
             But subsequent strings might be restricted.
             If we place $S_1$ (longest) at $(r, c)$, it forces values.
             Then $S_2$. It must match $S_1$ where they overlap.
             If conflict, backtracking?
             Since $M$ is large, we can't backtrack deep.
             But we can maintain a pool of "candidates".
             However, the grid is finite ($20 \times 20$).
             If we fill it up, we might block others.
             
             What if we use the "Consensus" method from the beginning?
             It creates a grid.
             Then we check how many strings are satisfied.
             If not all, we tweak the grid.
             The problem is efficiency.
             
             Let's try a different perspective.
             The strings are random substrings of the GRID.
             This means the GRID is composed of parts of these strings.
             Let's extract all "patterns" from the strings.
             Count the frequency of each substring (of length 1 to 12) in the set of strings?
             No, in the grid?
             We don't know the grid.
             But we know the strings.
             If we take all strings $S_i$.
             And we look for common substrings?
             This leads to de Bruijn graphs or similar.
             
             Let's go back to the **Simulated Annealing on Grid** but optimize the evaluation.
             Instead of checking ALL strings for every move, we can cache scores.
             Or use a "heuristic cost".
             
             Another Idea: **Iterative Deepening on Placement Assignment.**
             We have $M$ strings.
             Each must be placed in one of 800 slots.
             Actually, we can treat the "Slot" as a cell coordinate?
             No, a string spans multiple cells.
             
             Wait! The strings are generated from a matrix.
             This means if we sort the strings and try to align them.
             For any two strings $S_i, S_j$.
             If they overlap in the grid, they must agree.
             If they DON'T overlap, they can be anywhere.
             We can try to group strings into clusters that are compatible.
             Eventually these clusters should tile the grid.
             Since the grid is $20 \times 20$, the area is 400.
             If we have many strings, they must tile it densely.
             We can use **Dynamic Programming** on the grid?
             Or **Maximum Flow**?
             
             Let's write a script to simulate the problem locally (conceptually) to understand constraints better.
             But I cannot run code.
             However, I can reason.
             $M$ strings, length $\approx 7$.
             Total length $\approx 4200$.
             Grid Area 400.
             Coverage $\approx 10$.
             This means every cell is covered by ~10 strings.
             This is a "dense" constraint system.
             However, since the strings are from a VALID grid, there exists a solution with 0 conflicts.
             This means the constraints are NOT contradictory.
             
             What if we perform a **Local Search on the "Assignment" of strings to grid cells?**
             No, grid cells are not independent for strings.
             
             Let's stick to **Simulated Annealing** with optimizations.
             The bottleneck is checking satisfied strings.
             We can maintain the state of each string.
             For each string, maintain a list of active placements?
             No.
             Just maintaining `is_satisfied` boolean for each string is enough.
             And `score` counter.
             When we change the grid (change a few cells), we only need to update `is_satisfied` for strings that interact with changed cells.
             We identified that ~10 strings interact per cell.
             So 10 strings per 12 cells $\approx 120$ strings to check.
             For each such string, we need to check if it is still satisfied.
             How to check efficiently?
             We need to know if *any* placement works.
             This requires a quick `matches` check.
             If `matches` is slow, we need optimization.
             
             Optimization for `matches`:
             Pre-calculate a bitmask or trie?
             Since $N$ is small, maybe just simple bitwise logic?
             Or just realize that for random strings, collisions are rare, so a string might be satisfied by only 1 placement in the grid.
             If a string has length $L$, there are 800 placements.
             Most will mismatch.
             The one that matches (if any) has 0 mismatches.
             Others have $>0$.
             So if we find ONE placement with 0 mismatches, it is satisfied.
             To check quickly:
             For each affected string, try to find a matching placement in the modified grid.
             Can we restrict the search space?
             If we modified cells near the start of a placement...
             No.
             However, if we know the string was satisfied BEFORE, and we only changed a few cells.
             If the number of changed cells is small (e.g. 1), and the string is satisfied by multiple placements, it's likely still satisfied.
             If it is satisfied by a SINGLE placement, and we broke that placement (changed 1 cell), then it is UNSATISFIED.
             If satisfied by multiple, we might still have one left.
             So we can store `count_valid_placements`?
             No, that's dynamic.
             
             Actually, checking if a string is satisfied is checking if it appears as a subsequence.
             Wait, the definition of subsequence in the problem is:
             "exists $(i, j)$ satisfying..."
             This is exactly finding the string in the grid.
             
             Let's use the property:
             We only change cells to FIX specific strings.
             Maybe we don't need to check ALL strings.
             Let's just trust that we are moving towards the solution.
             If we use a simple heuristic:
             "Change cells to fix $S_i$".
             Does it break $S_j$?
             If yes, is it worth it?
             If $S_j$ was very important (satisfied), and we break it.
             But maybe $S_j$ can be fixed later.
             Since $M$ is high, and coverage is high, it's probable that changing a cell to fix $S_i$ breaks $S_j$ which also wanted that cell for $S_j$.
             This is "local collision".
             We can simply pick the placement for $S_i$ that causes minimum damage.
             Minimum damage = number of currently satisfied strings that cease to be satisfied.
             
             Wait, calculating "number of satisfied strings that cease" is expensive.
             Is there a proxy?
             Proxy: Count of satisfied strings that use the modified cells?
             Yes. If a string uses a modified cell, it HAS a chance to break.
             If it uses multiple modified cells, higher chance.
             If it uses one cell, and we change it, we need to verify.
             But verification is the bottleneck.
             
             Can we optimize `matches`?
             We can implement `matches` using a rolling hash?
             Grid $N \times N$.
             Hash all $2 N^2$ horizontal/vertical substrings?
             Computing hashes for all substrings takes $O(N^3)$.
             With $N=20$, $N^3 = 8000$. Very fast.
             So we can pre-hash the entire grid into a table of hashes.
             `GridHash[row][col][dir][shift]`?
             Actually, for each row, we have a string of length $N$.
             We can compute hashes for all substrings of row $r$.
             Since it's torus, string is cyclic.
             So each row is a string $R_r$. We need all cyclic substrings of length $L$.
             We can compute this.
             Similarly for columns.
             We maintain a lookup table: `Map<HashValue, List<StringIndices>>`.
             For a query string $S$, we compute its hash.
             If `HashValue` exists in map, and the map contains an index of $S$, then $S$ is present.
             WAIT.
             Hash collisions?
             With 8 characters, collisions are possible but unlikely if we use a large modulus.
             Also need to distinguish between identical strings.
             But $M$ strings are distinct? Not necessarily.
             Input strings can be duplicates.
             If we use Hash, we map Hash to list of indices.
             Checking existence is $O(1)$ (or $O(k)$ where $k$ is count of indices with that hash).
             Since strings are short, hashing is fast.
             
             This reduces `matches` complexity to $O(L)$ (hash computation).
             Precomputing grid hashes: $O(N^3)$ or $O(N^2 \cdot N)$.
             Updating grid: Changing one cell updates hashes of nearby segments.
             Updating hash table: Delete old hashes, add new hashes.
             Since $L$ is small, this is fast.
             
             Algorithm with Hash Table:
             1. Initialize Grid.
             2. Compute Grid Hashes:
                For each row $r$:
                  Concatenate row $r$ to itself to handle torus ($R+r[0..N-1]$).
                  Sliding window of length $1..N$. Compute hashes.
                  Store in `HashTable`. `HashTable[h]` contains `vector<int>` (indices of strings? No, indices of PLACEMENTS).
                  Actually, we need to know which STRING indices match.
                  Better:
                  `HashTable` maps `Pair<StringId, Hash>` to boolean? No.
                  We need to check if string $S_k$ matches grid.
                  So we check `HashTable[S_k.Hash]`.
                  If non-empty, $S_k$ is satisfied.
                  BUT, multiple strings can have same hash?
                  If $S_k$ and $S_m$ have same hash (collision or duplicate content), we just need ONE match.
                  So `HashTable[H]` stores "Has Match".
                  Wait, if $S_k$ is in `HashTable[H]`, then $S_k$ exists.
                  Wait, `HashTable[H]` should be computed from grid.
                  For grid, we compute all substring hashes.
                  `AllSubstrHashes` is a set of integers.
                  Then for each $S_k$, check if `H(S_k)` is in `AllSubstrHashes`.
                  Wait, `AllSubstrHashes` size is $N \times N \times N$ (total substrings).
                  $400 \times 20 = 8000$.
                  So we can maintain `AllSubstrHashes` as a `std::bitset` or `unordered_set`.
                  Wait, we need to track WHICH strings are satisfied.
                  We can pre-compute `TargetHashes` for all $M$ strings.
                  `TargetHashes` = list of hashes of strings.
                  Then `score` = count of unique strings whose hash is in `AllSubstrHashes`.
                  Wait, if multiple copies of a string exist, does it matter?
                  We just need existence.
                  BUT, what if collision occurs between a valid string and a noise?
                  If $S_k$ has hash $H$.
                  Grid produces hash $H$ from a DIFFERENT string placement (noise).
                  Then we falsely think $S_k$ is present.
                  With base $\ge 256$, mod prime $\approx 10^9$, collision prob is negligible.
                  
                  So, Evaluation:
                  Compute `AllSubstrHashes`.
                  Score = $\sum_{k=1}^M (1 \text{ if } TargetHashes[k] \in AllSubstrHashes \text{ else } 0)$.
                  
                  Cost:
                  `AllSubstrHashes` construction/update.
                  Grid size $N=20$.
                  For each cell update, how many substring hashes change?
                  A cell $(r,c)$ affects substrings passing through it.
                  Horizontal: rows $r$, starting at $0..N-1$.
                  Since it's a torus, every substring of length $L$ ending at or after $r$ and starting at or before $c$ involves $(r,c)$.
                  Number of such substrings is $O(L \times N)$.
                  With $N=20, L \approx 7$, it's small.
                  So updating is fast.
                  
                  However, the `AllSubstrHashes` must track presence of strings?
                  Wait, if $S_k$ hash is in set, it means there is SOME substring matching it.
                  That is exactly what we need!
                  
                  Wait, does `AllSubstrHashes` distinguish between different string contents?
                  No, it just knows "substring with hash H exists".
                  If we have $S_1$ and $S_2$ with same content (duplicate inputs), we need to count both?
                  Yes.
                  We need to know how many $S_k$ are satisfied.
                  So we maintain `bool satisfied[M]`.
                  When updating grid:
                    Update `AllSubstrHashes` set.
                    For each $S_k$ whose hash is $H$:
                      Check if $H \in AllSubstrHashes$.
                      If yes and not `satisfied[k]`, mark satisfied.
                      If no and `satisfied[k]`, mark unsatisfied.
                    Update `satisfied_count`.
                    This avoids full scan.
                  
                  Complexity:
                  Update set: $O(N \cdot L)$.
                  Check updates: $O(M)$ in worst case (if many hashes changed? No, only hashes involving modified cells change).
                  Modified cells are only those in the placement we are testing.
                  So only $O(L)$ hashes change.
                  So we only need to check strings that match these NEW/OFFSET hashes?
                  No, we need to check if $TargetHashes$ match the set.
                  But if the SET changes, we might affect status of many strings?
                  If we ADD a hash $H$ to the set.
                  Then any $S_k$ with $Hash=S_k$ might become satisfied.
                  If we REMOVE a hash $H$, any $S_k$ might become unsatisfied.
                  How many strings have the SAME hash?
                  Depends on duplicates.
                  Worst case $M$ strings have same hash.
                  But $M \approx 600$. Checking 600 is fast.
                  
                  Wait, computing `AllSubstrHashes` naively takes $O(N^3)$ initially.
                  $8000$ entries.
                  Update is fast.
                  Checking `satisfied` is $O(M)$?
                  Wait, we need to check if `TargetHashes` are present.
                  Since we maintain the set `GridHashes`.
                  `TargetHashes` is static.
                  For each $S_k$, check `if (GridHashes.count(H[k]))`.
                  If we change grid slightly, `GridHashes` changes.
                  We must re-evaluate `count`.
                  Recomputing count takes $O(M)$.
                  Is this fast enough?
                  If we loop 800 placements.
                  Total checks $800 \times 600 = 480,000$.
                  $O(1)$ lookup.
                  Total ops $\approx 5 \cdot 10^5$.
                  Per string selection.
                  Per outer loop iteration.
                  If we iterate 100 times -> $5 \cdot 10^7$.
                  Total time well within limits.
                  
                  Wait, this approach requires us to UPDATE `GridHashes` for EVERY tested placement?
                  NO!
                  We only test placements to find the BEST one.
                  We simulate the changes virtually.
                  But we can't rebuild `GridHashes` 800 times for each cell change?
                  Building `GridHashes` takes $O(N^3)$?
                  $N=20 \implies N^3 = 8000$.
                  If we rebuild it 800 times -> $6.4 \times 10^6$.
                  Times 100 strings -> $6.4 \times 10^8$.
                  This is borderline but likely OK for 2 seconds.
                  Actually $N^3$ is small.
                  $20^3 = 8000$.
                  Building a map/set of 8000 hashes is fast.
                  So rebuilding is fine.
                  
                  Wait, `AllSubstrHashes` stores hashes of all possible substrings of the grid.
                  Total count = $N^2 \times 2 \times N$ (shifts).
                  $400 \times 40 = 16000$ substrings?
                  Wait, for each cell start $(r,c)$, we have horizontal length $1..N$.
                  So $N$ substrings per start.
                  Starts: $N \times N = 400$.
                  So $400 \times 20 = 8000$ horizontal substrings.
                  Same for vertical. Total 16000.
                  Inserting 16000 hashes into a `std::unordered_set` or `std::bitset` is fast.
                  
                  So the plan:
                  1. Generate random grid.
                  2. Implement `get_all_hashes(grid)`: returns set of hashes of all horizontal/vertical substrings.
                  3. Loop `iterations`:
                     If all strings satisfied, return grid.
                     Pick unsatisfied string $S$.
                     Best_Place = Null. Min_Breaks = Inf.
                     
                     // We need to calculate the "Breaks" metric.
                     // Breaks = Number of currently satisfied strings that will be unsatisfied if we switch to this placement.
                     // Since we are changing cells to match $S$, we must recompute `get_all_hashes` for the modified grid.
                     // Rebuilding `get_all_hashes` takes constant small time (16000 inserts).
                     // Then we can re-evaluate `score` easily?
                     // Wait, re-evaluating `score` requires checking if $S$ is satisfied and other strings.
                     // We can check:
                     //   New_Score = count(strings whose hash is in New_Hashes).
                     //   Current_Score = old score.
                     //   Gain = New_Score - Current_Score.
                     //   We want to maximize Gain (preferably > 0).
                     //   Since we MUST satisfy $S$, Gain $\ge 1$ usually.
                     //   But we want to minimize reduction in other strings.
                     //   So maximize `New_Score`.
                     //   
                     // Wait, is it enough to maximize `New_Score`?
                     // Yes. `New_Score` accounts for $S$ becoming satisfied and other changes.
                     // However, this greedy strategy (fixing $S$ by picking best placement to maximize total score) is valid.
                     
                     // Optimization:
                     // We can compute `New_Score` by taking current `satisfied_count` and adjusting for affected strings.
                     // Affected strings are those that share a hash with changed cells?
                     // No. Hash collisions are rare.
                     // A change to the grid only affects `AllSubstrHashes` (by removing/adding hashes).
                     // If a hash is removed, any string relying on that hash (only one string ID, or group of IDs) might drop.
                     // If added, any string might gain.
                     // Since we don't know which string ID corresponds to the hash (except if hash is unique to $S_k$),
                     // it's safer to just count:
                     // `satisfied_count` = sum(`HashSet.count(TargetHash[k])`).
                     // Wait, this requires storing mapping from Hash to StringID?
                     // If multiple strings have same hash, `count` increases by multiplicity.
                     // So `current_score` = sum over $k$ of (1 if `GridHashSet` contains `TargetHash[k]`).
                     // Calculating this is $O(M)$. Fast.
                     
                     // So for each candidate placement (800):
                     //   Temporarily apply modification to a copy of grid?
                     //   Rebuild HashSet ($O(N^3)$).
                     //   Calculate Score ($O(M)$).
                     //   Compare with current score.
                     //   Pick placement with max Score.
                     //   If Score > Current Score (or $\ge$ if tie? Tie breaks?), apply to main grid.
                     
                     // Complexity:
                     // 800 placements * ( $O(N^3) + O(M)$ ).
                     // $N^3 \approx 16000$.
                     // $M \approx 600$.
                     // Per placement: 17000 ops.
                     // Total per step: $800 \times 17000 \approx 1.36 \times 10^7$.
                     // Number of steps?
                     // If we process each unsatisfied string once. Up to 600 strings.
                     // $600 \times 1.36 \times 10^7 \approx 8 \times 10^9$. Too slow.
                     // We need to limit iterations or reduce overhead.
                     
                     // Reduction:
                     // We don't need to REBUILD `HashSet` for every placement.
                     // We just need to compute `New_Score`.
                     // `New_Score` = (Count of satisfied strings) - (Lost strings) + (Found strings).
                     // Lost strings: Strings whose hash was in `OldSet` but NOT in `NewSet`.
                     // Found strings: Strings whose hash was NOT in `OldSet` but IS in `NewSet`.
                     // Most strings will not change status.
                     // We can identify which hashes change in `Set`.
                     // A placement modifies cells. This affects substrings passing through these cells.
                     // Number of affected substrings is small ($N \times L \approx 400$).
                     // Let the set of OLD_HASHES_removed be $H_{out}$.
                     // Set of NEW_HASHES_added be $H_{in}$.
                     // Then `New_Score` = `Old_Score` 
                     // - (count of $S_k$ s.t. `TargetHash[k]` $\in H_{out}$)
                     // + (count of $S_k$ s.t. `TargetHash[k]` $\in H_{in}$).
                     // Wait, a hash might be in both?
                     // No, `H_{out}` contains hashes that ceased to exist in the grid.
                     // `H_{in}` contains hashes that started appearing.
                     // Intersection should be handled carefully.
                     // Basically, for each hash $h$, count how many strings with $TargetHash=h$ are dependent on it.
                     // Wait, `TargetHash` might map to multiple strings?
                     // Yes.
                     // If `OldSet` has hash $h$, and `NewSet` has $h$ (from other sources).
                     // Then we lose nothing.
                     // We only lose if `NewSet` does NOT have $h$ AND `OldSet` did.
                     // So we need to know if $h$ is "essential" for $S_k$.
                     // But since we don't know if other instances exist, we can assume we lose $S_k$ if $h \in H_{out}$ and $h \notin H_{in}$?
                     // Actually, if $S_k$ has hash $h$.
                     // If $h$ is present in `NewSet`, $S_k$ is satisfied.
                     // If $h$ is absent, $S_k$ is not.
                     // So `Change in Score` = `Score(NewSet)` - `Score(OldSet)`.
                     // `Score(Set)` is $\sum_{k} 1_{TargetHash[k] \in Set}$.
                     // We can compute this efficiently if we track dependencies.
                     // But simpler: Just iterate all strings?
                     // Wait, if we don't rebuild `HashSet`, how do we know `NewSet` content?
                     // We know $H_{out}$ and $H_{in}$.
                     // So `NewSet` = `OldSet` $\cup$ $H_{in}$ $\setminus$ $H_{out}$.
                     // `Score(NewSet)` = $Score(OldSet \cup H_{in} \setminus H_{out})$.
                     // This is equal to $Score(OldSet) + | \{k : TargetHash[k] \in H_{in} \setminus OldSet \} | - | \{k : TargetHash[k] \in H_{out} \setminus (OldSet \cap H_{in}) \text{ wait no} \}|$.
                     // Let's define contribution.
                     // $S_k$ is satisfied if $T[k] \in Set$.
                     // If $T[k] \in OldSet$:
                     //   If $T[k] \in H_{out}$:
                     //     If $T[k] \in H_{in}$: Stays satisfied (moved from old to new instance).
                     //     Else: Becomes unsatisfied. (-1)
                     //   Else:
                     //     If $T[k] \in H_{in}$: Stays satisfied.
                     //     Else: Stays satisfied.
                     // If $T[k] \notin OldSet$:
                     //   If $T[k] \in H_{in}$: Becomes satisfied. (+1)
                     //   Else: Remains unsatisfied.
                     
                     // Wait, "If $T[k] \in OldSet$" condition is redundant.
                     // Just check:
                     // Change = 0.
                     // For each string $k$ with $T[k]$:
                     //   If $T[k] \in H_{out}$ and $T[k] \notin H_{in}$ and $T[k] \notin OldSet$: Impossible, $H_{out} \subset OldSet$.
                     //   Case 1: $T[k] \in H_{out}$ and $T[k] \notin H_{in}$.
                     //     If $T[k]$ was in $OldSet$ (it is), and is NOT in $NewSet$ ($H_{out}$ removed, not in $H_{in}$).
                     //     But is $T[k]$ in $OldSet \setminus H_{out}$?
                     //     If $OldSet$ contains other instances of $T[k]$, then $T[k] \notin H_{out}$ is false?
                     //     Wait, $H_{out}$ is the set of hashes that were REMOVED from `OldSet`.
                     //     No, $H_{out}$ is the set of hashes contributed by the affected substrings.
                     //     But these hashes might be replicated elsewhere.
                     //     Removing one instance of hash $h$ does not remove $h$ from `OldSet` if it appeared twice.
                     //     This is the catch. We don't know multiplicity.
                     
                     // Solution:
                     // Use `map<Hash, int> freq_in_old`.
                     // When changing cells, decrement `freq_in_old` for $H_{out}$.
                     // Increment `freq_in_new` for $H_{in}$.
                     // Wait, we need to know if the hash is STILL present in `NewSet`.
                     // `Present(h)` = (`OldFreq(h)` > 0) ? No.
                     // `OldFreq(h)` includes contributions from unaffected substrings too.
                     // We need to know global frequency.
                     // Global Frequency = Affected Substring Freq + Static Substring Freq.
                     // Static Freq can be precalculated.
                     // Affected Freq:
                     // For each modified cell, we compute the substring hashes contributed by that cell.
                     // Since cells are modified, these hashes change.
                     // Old contribution: Remove from Affected Freq.
                     // New contribution: Add to Affected Freq.
                     // NewGlobalFreq(h) = StaticFreq(h) + NewAffectedFreq(h).
                     // We can maintain `StaticFreq` (computed once from a dummy grid or just precomputed).
                     // Actually, `StaticFreq` depends on grid.
                     // So we just maintain `GlobalFreq` for all hashes.
                     // Initial pass: Build `GlobalFreq`.
                     // Step:
                     //   Identify affected hashes (those that change).
                     //   Update `GlobalFreq` for these hashes.
                     //   Recalculate score for affected strings?
                     //   Wait, if `GlobalFreq[T[k]]` > 0, $S_k$ satisfied.
                     //   So if we update `GlobalFreq`, we can immediately check status of strings with $T[k]=h$.
                     //   But $T[k]$ is unique per string ID? No.
                     //   So we need a `List<string_indices_for_hash[hash]`.
                     //   When `GlobalFreq[h]` changes from 1->0, decrement score.
                     //   From 0->1, increment score.
                     //   (Only relevant if `GlobalFreq[h]` transitions across boundary).
                     //   Wait, if `GlobalFreq` drops from 5 to 4, score is fine.
                     //   If it drops to 0, score decreases.
                     //   So we just need to track `GlobalFreq`.
                     //   
                     //   Algorithm for Candidate Evaluation:
                     //     1. Identify all affected substring hashes (Horizontal and Vertical passing through changed cells).
                     //     2. Subtract their contributions from `GlobalFreq` (temporarily).
                     //     3. Compute `Candidate_Score`.
                     //        Note: We need to know `GlobalFreq` before removal.
                     //        Actually, we can just subtract `Contribution(Old)`, add `Contribution(New)`, and update `GlobalFreq`.
                     //        Score = $\sum_k 1_{GlobalFreq[T[k]] > 0}$.
                     //        We can recompute this sum over the subset of strings that depend on changed hashes?
                     //        Strings depending on changed hash $h$ are those with $T[k]=h$.
                     //        For each such string, check new status.
                     //        Also strings with $T[k] \neq h$ are unaffected.
                     //        So we only need to update `score` for strings with $T[k] \in AffectedHashes$.
                     //     4. Restore `GlobalFreq`.
                     //     
                     //   Cost:
                     //     $|AffectedHashes|$ is roughly $N \cdot L \approx 400$.
                     //     Check status for each string with $T[k] \in AffectedHashes$.
                     //     Let $U$ be the union of strings covering the changed cells.
                     //     $|U|$ is approx $N \cdot L \times (\text{avg strings covering})$.
                     //     Wait, $T[k]$ is specific string. We need to iterate all strings $S$ with $T[S] \in AffectedHashes$.
                     //     This is the same set $U$ (since only strings covering these cells produce these hashes? No, any string CAN produce these hashes if aligned).
                     //     Actually, we care about `TargetHashes` of all strings.
                     //     So we iterate $M$ strings?
                     //     Wait, $AffecetedHashes$ are just hashes of substrings in the grid.
                     //     $M$ strings have their own hashes.
                     //     We need to check if `TargetHash[k]` is in $AffectedHashes$? No.
                     //     We need to update `GlobalFreq` for $AffectedHashes$.
                     //     Then check if $TargetHash[k]$ is affected.
                     //     This means `GlobalFreq[T[k]]` changes value.
                     //     This happens only if $T[k] \in AffectedHashes$ (specifically if the change crosses threshold).
                     //     Since $|AffectedHashes|$ is small (~400), we can check which strings have $T[k]$ in this set.
                     //     This is just looking up in `TargetHashMap`.
                     //     
                     //   Final check:
                     //     We only change `GlobalFreq` for hashes in $AffectedHashes$.
                     //     For each $h \in AffectedHashes$:
                     //       Update count.
                     //       If count becomes 0, decrement score by `cnt_of_strings_with_hash[h]`.
                     //       If count becomes 1, increment score by `cnt...`?
                     //       Wait, if count goes 0 -> 1, we GAIN `cnt` satisfied strings.
                     //       If count goes 1 -> 0, we LOSE `cnt` satisfied strings.
                     //       Otherwise score unchanged.
                     //       We need to do this efficiently.
                     //     But we need to know the `cnt_of_strings_with_hash[h]`.
                     //     Precompute `num_strings_per_hash[h]`.
                     //     
                     //     This is extremely fast.
                     //     $|AffectedHashes| \approx 400$.
                     //     Update `GlobalFreq` and `score` takes negligible time.
                     //     Re-building `GlobalFreq` is not needed.
                     //     Just subtract/insert affected hashes.
                     
                     //   So Cost per placement:
                     //     Compute affected hashes ($O(N \cdot L)$).
                     //     For each changed cell, we have a list of affected hashes.
                     //     Iterate $k \in \{ \text{all substrings at cell} \}$.
                     //     Update `GlobalFreq`.
                     //     Check if `score` changes.
                     //     Finally, we compare `current_score` with `candidate_score`.
                     //     But wait, we don't need to re-evaluate `score` from scratch.
                     //     We maintain `score` incrementally.
                     //     When modifying grid (permanent change), we permanently update `GlobalFreq` and `score`.
                     //     When simulating (testing placement), we temporarily modify `GlobalFreq`.
                     //     If `GlobalFreq` is updated, we check `score` change.
                     //     Wait, `score` is a scalar.
                     //     We need to know the exact `score` for the candidate placement.
                     //     `Score_new = Score_old + Delta_Score`.
                     //     Delta_Score depends on changes to `GlobalFreq`.
                     //     Specifically, for each $h \in AffectedHashes$:
                     //       `Delta_Freq = NewFreq[h] - OldFreq[h]`.
                     //       If `OldFreq[h] == 0` and `NewFreq[h] > 0`: `Gain += num_strings_per_hash[h]`.
                     //       If `OldFreq[h] > 0` and `NewFreq[h] == 0`: `Loss -= num_strings_per_hash[h]`.
                     //     Wait, this logic is flawed if `OldFreq` was > 1.
                     //     If `OldFreq` went from 1 to 0, loss.
                     //     If `OldFreq` went from 2 to 1, no loss.
                     //     If `OldFreq` went from 0 to 1, gain.
                     //     This assumes we only touch counts.
                     //     But `OldFreq` tracks global count.
                     //     So `GlobalFreq[h]` decreases by `RemovedContribution`.
                     //     Increases by `AddedContribution`.
                     //     We check boundary crossings.
                     //     This is $O(|AffectedHashes|)$.
                     //     Total complexity per placement: $O(N \cdot L)$.
                     //     This is very fast.
                     
                     //     Wait, we need to be careful about overlapping cells.
                     //     The placement specifies multiple cells.
                     //     We must update `GlobalFreq` for the WHOLE placement simultaneously.
                     //     So:
                     //       Save current state of `GlobalFreq` for affected hashes?
                     //       Actually, just iterate cells in placement.
                     //       For each cell, we know which hashes it contributes.
                     //       Wait, a cell contributes to hashes starting at various offsets.
                     //       If a placement covers cells $(r,c)$, it determines hashes of substrings.
                     //       Wait, "affected hashes" are hashes of SUBSTRINGS passing through these cells.
                     //       Yes.
                     //       So:
                     //         1. Collect all hashes $h$ contributed by the NEW characters in the placement.
                     //         2. Collect all hashes $h'$ contributed by the OLD characters in the placement.
                     //         3. `GlobalFreq[h]--` for each $h'$ (decrement).
                     //         4. Update score based on decrements.
                     //         5. `GlobalFreq[h]++` for each $h$ (increment).
                     //         6. Update score based on increments.
                     //         7. Restore score/state.
                     
                     //     This is robust.
                     
                     //   Wait, does `AffectedHashes` include hashes of substrings of length $>1$?
                     //   Yes, any substring passing through modified cell is affected.
                     //   For a cell, there are $20 \times 2$ substrings starting at different shifts?
                     //   Actually, for a cell $(r,c)$:
                     //   Horizontal:
                     //     It is part of substring starting at $(r, c-k)$ of length $k+1$.
                     //     $k$ ranges from $0$ to $N-1$ (since torus).
                     //     So $N$ horizontal substrings.
                     //   Vertical:
                     //     Similarly $N$ vertical substrings.
                     //   Total $2N$ hashes per cell.
                     //   With $L$ modified cells, total hashes $\approx 2N \cdot L \approx 40 \times 12 = 480$.
                     //   Manageable.
                     
                     //   So the algorithm is viable.

      Double Check:
      - $N=20$.
      - Time Limit: 2s? Usually 2s for AHC is generous or tight.
      - Operations per test case.
      - We have 100 test cases? "Input Generation... 100 test cases".
      - But the provided code is run ONCE per submission?
      - Wait, AHC problems usually have a fixed time budget for ALL test cases combined.
      - "If you get a result other than AC for one or more test cases...".
      - Wait, standard AHC format: 100 seeds, you submit code. Code runs on all 100 seeds within a time limit (usually 2s or so total).
      - Or is it 2s per seed?
      - "The highest score obtained during the contest time will determine the final ranking".
      - Usually time limit is for the whole batch or per run?
      - Standard CP is per run. But here it says "Input is given from Standard Input".
      - So we receive one seed at runtime.
      - So time limit is per test case.
      - With $N=20$, 100 iterations of "Pick string" loop.
      - Each loop: $800 \times 500 \approx 400,000$ ops.
      - Total $4 \times 10^7$ ops per string fix.
      - If we fix 600 strings, $2.4 \times 10^{10}$ ops. Too slow.
      - We need to optimize further.
      - Wait, we don't need to check ALL placements.
      - Just a few?
      - Or maybe we only fix strings one by one, but reuse the grid?
      - Also, most strings will be satisfied early.
      - Once a string is satisfied, we don't need to touch it?
      - But fixing one might break another.
      - So we can't ignore satisfied strings completely.
      - However, note that `GlobalFreq` tracking handles breaking satisfactions implicitly.
      - But we still need to pick a placement.
      - If we just pick RANDOM placement that fixes $S$, we might break things.
      - But maybe we can pick placement that adds minimal cost?
      - Cost = Loss of satisfied strings.
      - Loss comes from `GlobalFreq` hitting 0.
      - So we want to pick placement with MAX score (min loss).
      - This is exactly what the simulation does.
      - Can we speed up?
      - We only need to find ONE good placement.
      - Maybe sampling 10 placements is enough?
      - If we fix strings one by one, maybe the grid converges.
      - But we have $M$ strings.
      - Wait, $M$ strings cover the grid.
      - If we fix strings greedily, do we succeed?
      - With "Min Damage", yes.
      - How to speed up?
      - Notice that most strings will be satisfied "naturally" if we follow the majority vote?
      - Maybe we can fill the grid FIRST using Majority Vote?
      - If we fill grid with consensus, we might already satisfy many strings.
      - Then we use the solver for remaining.
      - Or better:
      - Run the "fill with consensus" to get an initial grid.
      - Check how many satisfied.
      - If few, the consensus is noisy (because strings came from random grid, but we don't know alignment).
      - Wait, the "Majority Vote" on grid cells using ALL possible alignments is actually the BEST estimator for the random matrix?
      - Yes.
      - Let's do this.
      - 1. Count char frequencies for each cell $(r,c)$ considering all possible alignments of all strings.
         - For each string $S_k$ and each of its 800 placements:
           - For each cell $(u,v)$ covered, increment count[char][r][c].
      - 2. After processing all strings, fill $Grid[r][c]$ with majority character.
      - 3. Calculate initial score.
      - 4. Then use "Repair/Improve" phase with the efficient hash-frequency check.
      - This reduces the number of unsatisfied strings drastically?
      - Probably.
      - With random grid, initial score is low.
      - With Majority Vote, score should be decent.
      
      Let's proceed with this strategy.
      
      Data Structures:
      - `int grid[20][20]`
      - `int counts[8][20][20]` for majority vote.
      - `vector<string> queries`.
      - `unordered_map<long long, int> string_hash_counts` (to check how many strings have same hash).
      - `unordered_map<long long, int> global_freq`.
      
      Hash implementation:
      - Polynomial rolling hash. Base 131, Mod 10^9+7.
      
      Corner Cases:
      - Strings length 2 to 12.
      - $N=20$.
      
      Refining the Repair Step:
      - We can run the repair loop for a fixed number of iterations (e.g., 100).
      - In each iteration, pick the WORST unsatisfied string.
      - Test all 800 placements?
      - If 800 is too many, sample top 20 candidates?
      - Actually, 800 is not THAT many if the inner loop is fast.
      - The inner loop involves updating 480 hashes.
      - Using `unordered_map` adds log factor or overhead.
      - Use direct arrays?
      - Hash values range up to $10^9$. Direct array impossible.
      - `unordered_map` is okay. Or just sort vectors of affected hashes.
      - Actually, the number of affected hashes is small. Sorting is faster than map.
      - Affected hashes list: `vector<long long> affected_old`, `vector<long long> affected_new`.
      - Sort both.
      - Merge to find intersection and differences?
      - Then update `global_freq`.
      - Since `global_freq` is sparse, sorting is efficient.
      
      Also, `num_strings_per_hash` needs to be stored.
      
      Wait, the majority vote initialization.
      We need to compute `counts[r][c]`.
      For each string, 800 placements.
      For each placement, 12 cells.
      Total operations: $600 \times 800 \times 12 \approx 5.7 \times 10^6$.
      Very fast.
      
      So:
      Phase 1: Consensus.
      Phase 2: Repair.
      
      One catch: The consensus might have ties. Random choice.
      
      Let's code this.
      
      Note: $N=20$. Characters 'A'..'H'. Map to 0..7.
      Dot '.' is not used in initialization (problem asks to use A-H).
      
```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <tuple>

using namespace std;

// Constants
const int N = 20;
const int CHAR_MAP = 8;
long long BASE = 131;
long long MOD = 1000000007;

// Globals
int grid[N][N];
int char_counts[N][N];
int num_strings_per_hash; // Not really needed if we store strings per hash
// Instead, we need to check score.
// To check score efficiently, we need:
// For each hash h, how many strings have hash h?
struct HashStats {
    long long h;
    int count;
};
// We will use a map: hash -> count_of_strings
unordered_map<long long, int> hash_to_string_count;

// String data
struct StringData {
    int id;
    string s;
    long long h;
};
vector<StringData> strings;

// Temporary for majority vote
int temp_counts[N][N];

// Precompute hash powers
long long pow_base[N + 1];

void init_pow() {
    pow_base[0] = 1;
    for(int i=1; i<=N; ++i) pow_base[i] = (pow_base[i-1] * BASE) % MOD;
}

long long get_substring_hash(const string& s, int idx, int len) {
    long long h = 0;
    for(int i=0; i<len; ++i) {
        h = (h * BASE + s[idx+i]) % MOD;
    }
    return h;
}

// For grid
// Get hash of a horizontal substring starting at (r, c) with length len
long long get_h_hash(int r, int c, int len) {
    // To handle torus easily, we can extend string or use modulo
    // But we need to map chars to long longs
    long long h = 0;
    for(int i=0; i<len; ++i) {
        h = (h * BASE + grid[(r + i % N)])? No, grid is char.
        // Access char
        // Let's convert grid to 0..255 or similar
    }
    // This is slow to loop. Precompute hashes?
    // No, for repair, we need to compute affected hashes dynamically.
    // The number of affected hashes is small.
    // But we can optimize by pre-converting grid to integer values.
    return 0; 
}

// Let's refactor to use integer grid representation
int val_grid[N][N]; // 0-255

// Function to compute hash from integer grid
long long compute_hash_from_chars(char* buf, int len) {
    long long h = 0;
    for (int i = 0; i < len; i++) {
        h = (h * BASE + buf[i]) % MOD;
    }
    return h;
}

// Optimization: Convert grid to byte array for hashing
// Actually, we can just use the char value.
// But wait, accessing grid in loop is slow if not careful?
// N=20, loop is tiny. It's fine.

long long get_h_hash_val(int r, int c, int len) {
    long long h = 0;
    for (int k = 0; k < len; k++) {
        h = (h * BASE + val_grid[(r + k) % N][c]) % MOD; // Wait, c is fixed, r varies.
        // Wait, I wrote `(r+k)%N`. If I'm traversing row r, `row` index is constant `r`.
        // Column index varies? No, horizontal substring.
        // `val_grid[row][col]`.
        // Horizontal: row r, columns c, c+1...
    }
    // Correct loop for horizontal:
    // for k from 0 to len-1: col = (c + k) % N
    // h = h * BASE + val_grid[r][col]
    // Wait, earlier I said row is fixed. Yes.
    return 0;
}

long long get_h_hash(int r, int c, int len) {
    long long h = 0;
    for (int k = 0; k < len; k++) {
        int rr = r;
        int cc = (c + k) % N;
        h = (h * BASE + val_grid[rr][cc]) % MOD;
    }
    return h;
}

long long get_v_hash(int c, int r, int len) {
    long long h = 0;
    for (int k = 0; k < len; k++) {
        int rr = (r + k) % N;
        int cc = c;
        h = (h * BASE + val_grid[rr][cc]) % MOD;
    }
    return h;
}

// Mapping char to int
int c_to_int(char c) {
    if (c == 'A') return 0;
    if (c == 'B') return 1;
    if (c == 'C') return 2;
    if (c == 'D') return 3;
    if (c == 'E') return 4;
    if (c == 'F') return 5;
    if (c == 'G') return 6;
    if (c == 'H') return 7;
    return 0;
}

void compute_initial_consensus() {
    // Reset counts
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) temp_counts[i][j] = 0;
    }
    
    // Process each string
    for(auto& s_data : strings) {
        const string& s = s_data.s;
        int len = s.length();
        
        // Try all 800 placements
        // Horizontal
        for(int r=0; r<N; r++) {
            for(int c=0; c<N; c++) {
                // Check validity? No, all valid on torus
                for(int k=0; k<len; k++) {
                    int rr = r;
                    int cc = (c + k) % N;
                    int char_code = c_to_int(s[k]);
                    temp_counts[rr][cc] += char_code; // Accumulate
                }
            }
        }
        // Vertical
        for(int r=0; r<N; r++) {
            for(int c=0; c<N; c++) {
                for(int k=0; k<len; k++) {
                    int rr = (r + k) % N;
                    int cc = c;
                    int char_code = c_to_int(s[k]);
                    temp_counts[rr][cc] += char_code;
                }
            }
        }
    }
    
    // Fill grid
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            int max_freq = -1;
            int best_char = 0;
            // Iterate over all 8 chars
            for(int k=0; k<8; k++) {
                int current_sum = 0;
                // Wait, we accumulated sums in temp_counts?
                // I used temp_counts[rr][cc] += char_code.
                // So temp_counts[r][c] stores sum of codes.
                // We need to find which code appeared most often?
                // No, it's weighted by frequency.
                // Wait, simply: most frequent char is the winner.
                // But temp_counts stores SUM of codes.
                // We need to count occurrences.
                // Let's change to int counts[N][N][8].
            }
        }
    }
    // ...
}
// Wait, let's fix the consensus accumulation.
int counts[20][20][8];
// ...
```
One small detail: `BASE` power computation. `BASE` 131. `MOD` 1e9+7.
Be careful with `c_to_int`.
`val_grid` should store `char` values.
When updating grid during repair, update `val_grid`.
Recomputing hashes is fast.

Let's refine the repair loop.
We need `score` and `satisfied` status.
Maintain `satisfied[M]` and `score`.
Also need `hash_to_string_count` and `global_freq`.
`hash_to_string_count` maps `hash -> count`.
`global_freq` maps `hash -> count`.
Wait, `global_freq` is actually just a count of hashes present in the grid.
We need `global_freq[h]`.
Actually, to update score:
Iterate over all strings `k`. If `hash_to_string_count[T[k] > 0]` check if `global_freq[T[k]] > 0`.
We can precompute `hash_to_string_count` from the initial strings.
And maintain `global_freq` (map).
Updating `global_freq` involves removing/adding hashes.
Since we need to iterate affected hashes, we collect them.

Final check on time complexity:
Inside repair loop:
We have `M` strings.
In one repair step (picking a string $S$), we iterate 800 placements.
For each placement, we calculate `delta_score`.
Calculating `delta_score` requires identifying affected hashes.
For a placement of length $L$, cells $\approx L$.
Hashes affected per cell $\approx 2N \approx 40$.
Total hashes $\approx 40L \approx 500$.
We iterate these 500 hashes.
For each hash, update `global_freq`.
Check `if global_freq[h] == 0` and we were about to decrement, we might lose a satisfied string?
Wait, we need to track `satisfied_count`.
`satisfied_count` is derived from `hash_to_string_count` and `global_freq`.
Actually, `satisfied_count` = $\sum_{k=0}^{M-1} (1 \text{ if } global\_freq[strings[k].h] > 0)$.
If `global_freq` changes from 1->0 for some $h$, then any string $k$ with $hash[k]==h$ becomes unsatisfied.
If `global_freq` changes from 0->1, any string $k$ with $hash[k]==h$ becomes satisfied.
So we need `grouped_strings_by_hash`.
`vector<int> strings_by_hash[HASH_RANGE]?` No, use `map<long long, vector<int>>`.
When `global_freq[h]` hits 0, add -size of vector to `score_delta`.
When `global_freq[h]` hits 1, add +size.
Wait, we need to be careful.
If `global_freq` goes $1 \to 0$, we lose the strings.
If `global_freq` goes $0 \to 1$, we gain strings.
If it goes $2 \to 1$, no change.
So we need to track `global_freq`.

Correct logic:
Initialize `global_freq` from grid.
Calculate initial `score` by iterating `M` strings.
Initialize `satisfied_count`.
Loop repair:
  Pick unsatisfied string $S$.
  Best Placement: null.
  Min Loss = INF.
  
  Loop placements (800):
    Calculate affected hashes (list).
    Calculate `temp_global_freq` updates?
    Wait, we need `loss` = (Count of satisfied strings lost).
    Loss comes from hashes that will disappear from `global_freq` (transition $1 \to 0$).
    But we don't know the new frequency without simulating.
    So:
    1. Identify `affected_old_hashes` (hashes that will decrease count).
       These are hashes contributed by the current grid content in the placement area.
    2. Identify `affected_new_hashes` (hashes that will increase count).
       These are hashes contributed by the new grid content.
    3. Temporarily apply changes to `global_freq`.
       Decrement for `affected_old`.
       Check for $h$ where `old_freq[h] == 1` (before decrement). If so, `loss += hash_to_string_count[h]`.
       Increment for `affected_new`.
       Check for $h$ where `new_freq[h] == 1` (after increment). If so, `gain += hash_to_string_count[h]`.
       Wait, net delta = gain - loss.
       New Score = Old Score + Net Delta.
       If New Score > Old Score (or minimal loss), this is a candidate.
       We want to maximize New Score.
    4. Restore changes.
    5. Keep best placement.
    6. Apply best placement.
    7. Update grid, permanently update `global_freq` and `score`.
    8. If no improvement possible (score $\le$ current), stop?
       No, maybe we need to change unsatisfied string even if it hurts.
       But if we prioritize fixing, usually New Score $\ge$ Old Score because we add +1 for $S$ being satisfied.
       Unless we lose many others.
       If we lose many, maybe try another placement?
       If ALL placements result in Net Loss, we are stuck?
       Wait, we MUST satisfy $S$.
       So we must change the grid to accommodate $S$?
       Not necessarily. We just pick the best compromise.
       But to maximize score, we want best compromise.
       
    Wait, `hash_to_string_count` is the number of strings with that hash.
    Is it possible to have multiple strings with same hash?
    Yes.
    So we use `vector<int> strings_by_hash[h]`.
    
    Wait, we need to be able to update `global_freq` for all affected hashes.
    The number of affected hashes is $\approx 400$.
    Iterating 400 items per placement is fast enough.
    $800 \times 400 = 320,000$ ops.
    If we do this 10 times (since only 10 strings left unsatisfied), it's $3 \times 10^6$.
    This is fine.
    
    Wait, we need to implement "Get affected hashes for a placement".
    This requires iterating cells in the placement and generating hashes.
    Generating a hash takes $O(L)$.
    $L \le 12$.
    So per cell $\approx 12$ ops.
    $40 \times 12 = 480$ ops per placement.
    $800 \times 480 = 384,000$.
    Plus overhead.
    Total operations $\approx 10^6$.
    Very fast.
    
    We need a function `get_affected_hashes(placement)` returning pairs `{old, new}`?
    Actually, we need to know which hashes are removed and which added.
    We can compute `grid` hashes for the placement area in `old_grid` and `new_grid` separately?
    No, we just need the hashes of substrings that pass through the modified cells.
    Wait, the modified cells are replaced by NEW chars.
    Substrings passing through them change.
    So we need to list hashes of substrings passing through each modified cell in BOTH configurations (old and new).
    This is the correct way.
    
    Let's structure this.
    
    ```cpp
    struct Placement {
        int type; // 0: horiz, 1: vert
        int r, c; // start
    };
    
    void solve() {
        // ... read input ...
        // ... init pow ...
        
        // Consensus
        // ... fill grid ...
        
        // Init globals
        // Compute initial global_freq and score
        
        // Repair loop
        // while score < M:
        //    pick unsatisfied S
        //    best_plac = null, best_gain = -INF
        //    iterate placements
        //        compute affected
        //        calc gain
        //        if gain > best_gain: update
        //    if no improvement?
        //       // Even if no improvement (score drops), we should still apply?
        //       // Or pick best anyway.
        //       // Applying best ensures we progress towards satisfying S.
        //       // If gain < 0, we still accept it because we gained +1 from S (eventually).
        //       // Wait, my gain logic already includes +1 for S.
        //       // If S was unsatisfied, and now satisfied, gain reflects that.
        //       // But we also lost others.
        //       // If net change is negative, we are worse off.
        //       // But we must fix S.
        //       // So we just pick the placement that maximizes score.
        //       // This is greedy hill climbing.
        //       // If we are stuck, maybe random perturbation?
        //       // Let's just pick best.
        //    apply placement
    }
    ```
    
    Wait, `score` calculation needs to be precise.
    When picking placement, we assume S becomes satisfied.
    Wait, does it?
    We check if placement is valid? No, we FORCE it to be valid.
    But checking if it's valid requires checking if grid matches S.
    But since we are simulating the change, we can check if `hash` of S (which we know) is in `new_global_freq`?
    Yes.
    Wait, `S.hash` is in `hash_to_string_count`.
    If `new_global_freq[S.hash] > 0`, then satisfied.
    So we can check `score` correctly.
    Wait, we don't know if `S` is satisfied just by looking at its hash in `global_freq`.
    Yes, that IS the definition of satisfied.
    So we just need to calculate `new_score` based on `new_global_freq`.
    The `delta` calculation handles transitions $0 \leftrightarrow 1$.
    But wait, we need to handle the case where `S` is NOT satisfied before (so its hash count was 0 in `global_freq`?)
    No, `global_freq` tracks how many HASHES of SUBSTRINGS are in the grid.
    It does NOT track if `S.hash` is in grid.
    If `global_freq[S.hash] > 0`, then `S` is present in grid.
    So `S` is satisfied.
    Wait, if `global_freq` doesn't have `S.hash`, then `S` is NOT in grid.
    Correct.
    So `S` is satisfied iff `global_freq[S.hash] > 0`.
    
    So when we pick a placement, we are essentially setting `grid` to match `S`'s placement.
    This means `S.hash` WILL be in `new_global_freq` (since the placement generates it).
    So `global_freq[S.hash]` will be incremented.
    Thus `S` will be satisfied.
    So `gain` will include `+1` (or more if multiple strings share hash).
    Wait, do we need to check if `S` was satisfied?
    If `S` was satisfied, we wouldn't pick it.
    So `S` was unsatisfied $\implies global\_freq[S.hash] == 0$.
    So transition $0 \to 1$ for `S.hash` is guaranteed.
    Wait, multiple strings might share `S.hash`.
    So `gain` includes `hash_to_string_count[S.hash]`.
    
    So logic holds.
    
    Implementation details:
    Need to efficiently get list of hashes for a cell.
    Since $N=20$, precompute `vector<long long> cell_hashes[N][N][type][len_offset]`?
    No, `cell_hashes` depends on the VALUES in the grid.
    We can't precompute values.
    But we can generate them on the fly.
    Function `get_cell_contrib(r, c, len)` returns list of hashes.
    
    Since we need to check `global_freq` for these hashes, we can just iterate and update.
    Since map operations are costly, we can use `vector<pair<long long, int>> changes`.
    Then iterate changes.
    Or just a temporary map. `vector` is faster.
    
    Wait, `changes` list needs to handle duplicates.
    So `multiset` or just sort `vector`.
    Actually, we can accumulate counts in a map `diff[hash] = delta`.
    Then iterate `diff`.
    Since affected hashes count is small (400), map is fine.
    
    Let's code.
</think>

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <set>

using namespace std;

// N is fixed to 20
const int N = 20;
const long long BASE = 131;
const long long MOD = 1000000007;

// Character mapping
int char_val(char c) {
    if (c == 'A') return 0;
    if (c == 'B') return 1;
    if (c == 'C') return 2;
    if (c == 'D') return 3;
    if (c == 'E') return 4;
    if (c == 'F') return 5;
    if (c == 'G') return 6;
    if (c == 'H') return 7;
    return 0;
}

// Global variables
int grid_val[N][N]; // Stores integer values 0-7
vector<pair<int, string>> strings;
vector<long long> str_hashes;
map<long long, int> hash_freq; // Tracks frequency of hashes in grid
map<long long, int> hash_string_count; // Tracks how many strings have this hash
int satisfied_count = 0;
int num_strings = 0;

long long pow_base[N + 1];

void precompute_powers() {
    pow_base[0] = 1;
    for (int i = 1; i <= N; ++i) {
        pow_base[i] = (pow_base[i - 1] * BASE) % MOD;
    }
}

long long compute_hash(char* chars, int len, int row, int col, int is_horizontal) {
    // Optimized computation
    // If we access grid_val[row][col] etc.
    // For simplicity in this loop, we access directly.
    long long h = 0;
    // Note: For vertical, row changes. For horizontal, col changes.
    // We handle generic access here.
    // But we need to loop properly.
    // Let's rewrite to be generic.
    return 0; // Placeholder
}

// Compute hash of horizontal substring starting at (r, c) of length len
inline long long get_h(int r, int c, int len) {
    long long h = 0;
    for (int k = 0; k < len; ++k) {
        h = (h * BASE + grid_val[r][(c + k) % N]) % MOD;
    }
    return h;
}

// Compute hash of vertical substring starting at (r, c) of length len
inline long long get_v(int c, int r, int len) {
    long long h = 0;
    for (int k = 0; k < len; ++k) {
        h = (h * BASE + grid_val[(r + k) % N][c]) % MOD;
    }
    return h;
}

// Helper to get hashes from grid cells
// Returns a multiset/hash count map of hashes that will be affected if a cell (r,c) is changed
// But actually, changing a cell changes ALL substrings passing through it.
// We can just compute hashes for affected substrings.
// However, we need to differentiate old hashes from new hashes to know what removes/adds.
// We can just collect ALL hashes in the affected area (both old and new) and compute deltas.
// To do this efficiently:
// For a placement, iterate all cells involved.
// For each cell, compute hashes of all substrings passing through it (Horizontal and Vertical).
// We will accumulate counts of hashes for "OLD" placement and "NEW" placement.
// Then diff them.
// To avoid iterating all substrings for each cell (which is 2*N per cell, ~40), 
// we can just loop 40 times. N=20 -> 40 substrings per cell.
// Total per placement (len ~12): 480 hashes.
// This is small.

int get_char_idx(const string& s, char c) {
    return c - 'A';
}

// Precompute char to 0-7 map for strings
vector<vector<int>> string_char_codes;
vector<int> string_lengths;

void run_consensus(vector<string>& input_strings) {
    int counts[N][N][8];
    for(int i=0; i<N; ++i) for(int j=0; j<N; ++j) for(int k=0; k<8; ++k) counts[i][j][k] = 0;

    for(const string& s : input_strings) {
        int len = s.length();
        int cur_len = len;
        
        // Horizontal
        for(int r=0; r<N; ++r) {
            for(int c=0; c<N; ++c) {
                for(int k=0; k<len; ++k) {
                    int rr = r;
                    int cc = (c + k) % N;
                    counts[rr][cc][char_val(s[k])]++;
                }
            }
        }
        // Vertical
        for(int r=0; r<N; ++r) {
            for(int c=0; c<N; ++c) {
                for(int k=0; k<len; ++k) {
                    int rr = (r + k) % N;
                    int cc = c;
                    counts[rr][cc][char_val(s[k])]++;
                }
            }
        }
    }

    // Fill grid
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            int max_freq = -1;
            int best_char = 0;
            for(int k=0; k<8; ++k) {
                if(counts[i][j][k] > max_freq) {
                    max_freq = counts[i][j][k];
                    best_char = k;
                }
            }
            grid_val[i][j] = best_char;
        }
    }
}

// Main solution
void solve() {
    int n_input, m_input;
    cin >> n_input >> m_input;
    vector<string> raw_strings(m_input);
    for(int i=0; i<m_input; ++i) cin >> raw_strings[i];

    // Precompute powers
    precompute_powers();

    // Prepare strings
    string_char_codes.resize(m_input);
    string_lengths.resize(m_input);
    for(int i=0; i<m_input; ++i) {
        string_lengths[i] = raw_strings[i].length();
        string_char_codes[i].resize(raw_strings[i].length());
        for(int j=0; j<raw_strings[i].length(); ++j) {
            string_char_codes[i][j] = char_val(raw_strings[i][j]);
        }
    }

    // Consensus Phase
    // Copy raw_strings for consensus
    run_consensus(raw_strings);

    // Reset global state
    hash_freq.clear();
    hash_string_count.clear();
    satisfied_count = 
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=441)

````
This is the AtCoder Heuristic Contest (AHC004), a grid-based puzzle where we need to generate an $N \times N$ matrix that contains as many of $M$ given strings as possible, where strings can appear as horizontal or vertical contiguous subsequences on a torus (wrapping around edges).

Let me understand the structure:
- We have exactly 8 characters: `A` through `H`
- The grid is $20 \times 20 = 400$ cells
- Each string has length between 2 and 12
- There are 400-800 strings to match

The key insight here is recognizing the symmetry in how these strings are generated from a uniform random matrix. Let's analyze what happens when we extract substrings from a randomly filled grid:

For character counting on a horizontal subsequence of length $k$: each character appears with probability $\approx k/8$. With $k$ values between 2-12, we see characters with frequencies ranging from about $0.25$ (for $k=2$) up to $\approx 1.5$ per string.

With 400-800 strings of average length ~7, the total count for each character across all strings roughly equals its frequency on the grid multiplied by total strings:

$$\text{total char X count} = (\text{occurrences of X in } 400 \text{ cells}) \times M / 8$$

Since the grid fills uniformly, every character should appear in roughly the same total amount! But actually, there's an even more elegant observation here—consider the grid modulo some number...

Wait, let me think differently about what information the strings give us.

Let me write out the string generation mathematically. For a horizontally extracted string of length $k$ starting at position $j$:

$$s_m = (a_{i,j}, a_{i,(j+1)\bmod N}, \ldots, a_{i,(j+k-1)\bmod N})$$
````
