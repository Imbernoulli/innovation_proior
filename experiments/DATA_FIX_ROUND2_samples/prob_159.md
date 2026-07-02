# 题 159 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The user wants a C++ solution for a competitive programming problem called "RectJoin".

**Problem Analysis:**

1.  **Game Mechanics:**
    *   We have an $N \times N$ grid (points from $(0,0)$ to $(N-1, N-1)$).
    *   Initially, $M$ dots are given.
    *   An operation consists of choosing a new dot at $p_1$ and three existing dots at $p_2, p_3, p_4$.
    *   These four points must form a rectangle (axis-aligned or 45-degree tilted).
    *   $p_1$ is the point we place a dot at. The other three ($p_2, p_3, p_4$) already have dots.
    *   Condition 2: No *other* dots can lie on the perimeter of this rectangle except $p_2, p_3, p_4$.
    *   Condition 3: The perimeter cannot share a segment of positive length with any previously drawn rectangle perimeter. (Intersection at vertices is allowed).
    *   After placing a dot at $p_1$, the rectangle edges are "drawn" (implicitly tracked by condition 3).
    *   Goal: Maximize score. Score depends on $\sum_{(x,y) \in Q} w(x,y)$ where $Q$ is the set of all dots.
    *   $w(x,y) = (x-c)^2 + (y-c)^2 + 1$, where $c = (N-1)/2$.
    *   Higher weight points are further from the center. To maximize the sum of weights, we want to place dots as far from the center as possible. This means filling the boundary and then moving inwards, prioritizing corners.

2.  **Input Constraints:**
    *   $N$ is odd, between 31 and 61.
    *   $M$ is between $N$ and $N^2/12$.
    *   Initial dots are in a central region $\lfloor N/4 \rfloor \le x, y \le \lfloor 3N/4 \rfloor$.
    *   Time limit: usually tight for these types of constructive games. Execution time is mentioned to be sensitive.
    *   Number of test cases: 125 per system test.
    *   Total $M$ operations? Usually limited by state changes. Since we add a dot each op, max $K \approx N^2 - M$. But likely much smaller because constraints restrict valid moves. However, looking at the example, $K=20$. With $N=31$, maybe $K$ can be larger.

3.  **Strategy:**
    *   We need to find valid rectangles using the rule.
    *   A rectangle has sides parallel to axes or diagonal.
    *   Axis-aligned rectangle defined by min/max x and y of its vertices. Let vertices be $(x_1, y_1), (x_1, y_2), (x_2, y_1), (x_2, y_2)$. If $p_1$ is the new dot, it must be one of the corners not yet having a dot. The other three corners must have dots. Also no dots on the segments between them (excluding endpoints).
    *   45-degree rectangle: Vertices like $(x, y), (x+k, y-k), (x+k/2, y-k/2+...?)$ No, integer coordinates. A 45-degree square or rectangle implies side slopes are $\pm 1$. So vertices are $(x, y), (x+a, y-a), (x+b, y-b+a)$... wait.
        *   Vector for side: $(\Delta x, \Delta y)$. For 45 degrees, $|\Delta x| = |\Delta y|$.
        *   So sides are vertical/horizontal or slope $\pm 1$.
        *   Wait, condition says "parallel to the axis or inclined at 45 degrees". This usually means squares or rectangles whose diagonals are axis-aligned? Or sides?
        *   "forms a rectangle that is parallel to the axis or inclined at 45 degrees".
        *   Looking at examples in typical geometry problems or AtCoder archives: usually "inclined at 45 degrees" means the sides make 45 degree angles with axes. So vectors like $(u, u)$ and $(-u, u)$.
        *   So vertices would be $(x, y)$, $(x+u, y+u)$, $(x, y+2u)$, $(x-u, y+u)$ forming a square. Or more general rectangle? A rectangle with angle 45 deg means adjacent sides have slopes $1$ and $-1$. Yes, that makes a square rotated by 45 degrees.
        *   Wait, could it be a general rectangle with sides having slopes 0, infinity, 1, -1? Yes. That's the union of axis-aligned and 45-degree rotated rectangles.
        *   Wait, does "inclined at 45 degrees" apply to the whole rectangle shape or just orientation? Usually means rotation.
        *   Let's check the example output.
            *   Input: `9 20 ...` (Wait, example text is garbled, let's look at the provided text).
            *   Actually, I don't need to reverse engineer exact geometry from text description if I understand the standard interpretation.
            *   Standard interpretation:
                *   Type 1: Axis aligned. Points $(x_A, y_A), (x_B, y_A), (x_B, y_B), (x_A, y_B)$. Sides along grid lines.
                *   Type 2: Diagonal. Points $(x, y), (x+k, y+k), (x, y+2k), (x-k, y+k)$? No, that's a square. What about non-square rectangle? A rectangle with sides having slopes 1 and -1 must be a square. Because dot product of direction vectors $(1, 1)$ and $(-1, 1)$ is 0. So yes, 45-degree rectangles are actually squares (possibly rotated).
                *   So we are placing dots on vertices of axis-aligned rectangles OR diagonal-squares.
    *   Scoring: We want to add dots at large radii first.
    *   Greedy approach seems plausible. Always pick the available move that adds a dot at a coordinate $(x,y)$ maximizing $w(x,y)$ (distance squared).
    *   Is greedy optimal? The score is linear combination of individual $w(p)$. Since adding a dot at $p$ enables future moves, it might act as a catalyst. However, the constraint is that we must use existing dots. Placing a dot at a "good" position is generally better than a "bad" one, but sometimes bad positions might unlock good ones.
    *   Given the strict time limit and number of inputs, a heuristic search or greedy with some lookahead is needed. Or simply finding the best local move and taking it.
    *   Wait, the constraint "no other dots on the perimeter" suggests we want "empty" rectangles. As we fill the grid, empty rectangles become scarce near the center (where initial dots are dense) and abundant near the border.
    *   Since we start with dots near the center, we are surrounded by empty space near the border. We can form rectangles connecting center dots to border empty spots.
    *   Actually, to maximize $S_Q / S$, we want to include high-weight points. High weight = distance from center.
    *   So the strategy should be: Find all currently valid rectangles. Filter those that contain at least one currently undotted point (which will be $p_1$). Choose the one where the new dot $p_1$ has the maximum $w(p_1)$. Place dot, mark edge as blocked. Repeat until no moves.
    *   Is this strictly optimal? Maybe blocking an edge prevents a higher value dot later. But blocking an edge only happens when we use a specific rectangle. Using a rectangle consumes 3 existing dots to place 1 new one and draws 4 edges.
    *   Edges block future rectangles. Specifically "does not share a common segment of positive length". So edges can touch (share vertex), but cannot overlap.
    *   So drawing an edge commits us to not using that edge again.
    *   This looks like finding a maximal matching or packing, but weighted.
    *   Given the small $N$ (up to 61), $O(N^2)$ or even $O(N^3)$ per step might be acceptable if steps aren't too many. However, we can't afford $K \times N^4$. We need efficient checks.
    *   Maximum $K$: Roughly filling the grid. $N^2 \approx 3600$. If we do full scan, it's slow.
    *   Optimization: Maintain sets of valid rectangles. When adding a dot or drawing an edge, update valid rectangles.
    *   However, maintaining dynamic validity is complex.
    *   Alternative View: The problem asks for *a* sequence of moves to maximize the final set of dots (specifically weighted sum).
    *   Actually, since we stop when no more moves are possible, and every move increases the sum of weights (assuming we add unique dots), we just want to reach a state where the set of added dots $Q_{added}$ is maximized in the sense of dominant partial ordering (coordinate-wise)? No, just sum of weights.
    *   Heuristic: Prioritize adding dots far from center.
    *   Algorithm idea:
        1.  Identify all potential moves (rectangles with 3 corners dotted, 1 corner empty, empty perimeter inside).
        2.  Sort potential moves by weight of the missing corner descending.
        3.  Try to execute the highest weight move.
        4.  Does executing it block future high-value moves?
        5.  If we simply pick the highest weight available move repeatedly, it might lead to TLE due to suboptimal choices blocking others. But given $N \le 61$, maybe a randomized restart or simulated annealing is needed? Or just pure greedy is expected to pass?
        6.  Wait, typical AtCoder constructive game problems often accept greedy with careful state management or simple heuristics if the constraints aren't adversarial.
        7.  Wait, there is a constraint "system test contains 125 inputs". We have ~2 seconds for one execution? If we run 125 times locally, that would take too long. Usually total time is shared or per case is generous. The text says "Execution time may vary slightly... submissions very close to TLE may result in TLE". It implies efficiency matters.
        8.  Since $N$ is small, we can precalculate distances.
        9.  Checking if a specific rectangle is valid takes $O(\text{perimeter})$. Perimeter length is roughly $N$.
        10. Number of potential rectangles is roughly $O(N^2)$ (since defined by two opposite points or three points).
        11. Iterating over all potential rectangles at every step might be $O(K \cdot N^2)$. If $K \approx N^2$, that is $O(N^4) \approx 10^7$ ops. With 125 inputs, total ops $\approx 10^9$. Might be tight but feasible for C++.
        12. Can we optimize validity check?
            *   Precompute "dottedness" grid.
            *   Store "blocked segments".
            *   A rectangle is defined by $(x_1, y_1), (x_2, y_2)$ etc.
            *   Axis aligned: defined by top-left and bottom-right. Or just min/max x/y.
            *   Diagonal square: defined by center and size? Or vertices.
            *   Let's classify rectangles.
                *   Case A: Axis aligned. Defined by bounding box.
                *   Case B: Diagonal square. Vertices $(x,y), (x+a, y+a), (x+2a, y+2a)$? No.
                    *   Vector $(a,b)$. If axis parallel, $|a|=0$ or $|b|=0$.
                    *   If 45 deg, $|a|=|b|$.
                    *   So vertices of 45-deg rect must satisfy that midpoints of opposite sides are integer coords? Not necessarily.
                    *   Integer coordinates constraint: Vertices are grid points.
                    *   Side vectors must be multiples of $(1,0), (0,1), (1,1), (-1,1)$ etc.
                    *   A rectangle has orthogonal adjacent sides.
                    *   Orthogonal to $(1,0)$ is $(0,1)$. (Axis aligned).
                    *   Orthogonal to $(1,1)$ is $(1,-1)$. (45 degree rotated).
                    *   So any such rectangle is composed of sides with slopes 0, inf, 1, -1.
                    *   If a rectangle has sides with slope 1 and -1, it is a square (as established earlier).
                    *   If a rectangle has sides with slope 0 and inf, it is axis aligned.
                    *   Mixed? e.g. one side slope 0, next slope 1? Then not a rectangle (angle 45). Rectangle requires 90 degree angles.
                    *   So only two types of rectangles exist:
                        1.  Axis-aligned (slopes 0 and inf).
                        2.  Square rotated 45 degrees (slopes 1 and -1).
                        3.  Could we have a rectangle with sides slope 1 and 0? No, angle is 45.
                    *   So we only check these two types.

    *   **Type 1: Axis-Aligned Rectangles**
        *   Vertices: $(x_L, y_B), (x_R, y_B), (x_R, y_T), (x_L, y_T)$.
        *   Missing one vertex, say $(x_L, y_T)$. Others $(x_R, y_T), (x_R, y_B), (x_L, y_B)$ must be dotted.
        *   Check perimeter emptiness.
        *   Check blocked edges.
    
    *   **Type 2: Rotated Squares**
        *   Center must be integer or half-integer?
        *   Vertices $(x, y), (x+s, y-s), (x+s, y+s)$... wait.
        *   Vertices of a square with 45 deg rotation. Let side vector be $(s, s)$. Other side $(-s, s)$.
        *   Vertices: $P_1 = (x, y)$. $P_2 = (x+s, y+s)$. $P_3 = (x+s-s, y+s+s) = (x, y+2s)$. $P_4 = (x-s, y+s)$.
        *   Wait, $(x-s, y+s)$ to $(x, y)$ is vector $(s, -s)$, length $\sqrt{2}s$.
        *   $(x, y+2s)$ to $(x-s, y+s)$ is vector $(-s, -s)$, length $\sqrt{2}s$.
        *   So vertices are $(x, y), (x+s, y+s), (x, y+2s), (x-s, y+s)$.
        *   Note that vertices must be integers. If $s$ is integer, they are.
        *   Wait, are these the only integer-coordinate rotated squares?
        *   What if center is $(X_c, Y_c)$? Vertices would be $(X_c, Y_c \pm \sqrt{R}), (X_c \pm \sqrt{R}, Y_c)$. For integer vertices, $R$ must be integer square?
        *   Actually, simpler to iterate centers and sizes.
        *   Alternatively, just iterate on vertices. Pick 3 dotted vertices, check if they form 2 corners of a valid rectangle, and calculate the 4th.
        *   Since $M$ is relatively small initially but grows, iterating over all triplets is $O(M^3)$ which is bad.
        *   However, we can iterate over pairs of points that could form a side or diagonal?
        *   Better: Iterate over all pairs of points $(A, B)$ that are candidates for corners.
        *   Or just iterate over potential missing points $p_1$. For each empty spot $p_1$, look for $p_2, p_3, p_4$ around it.
        *   Radius of search? Since initial dots are central, new dots spread out.
        *   The maximum coordinate difference is $N$.
        *   If we check neighbors within range $[-N, N]$, it's fine.
        *   Actually, a rectangle is determined by $p_1$.
            *   Axis aligned: Need $(x_1 \pm dx, y_1 \pm dy)$ pattern. Specifically, if $p_1 = (x, y)$, other corners are $(x, y+dy), (x-dx, y+dy), (x-dx, y)$ or similar.
            *   We can check neighbors.
        *   Validity check needs to be fast.
    
    *   **Data Structures:**
        *   Grid `dot[N][N]`.
        *   `blocked_edge` map/set. Edge is a segment between $(x1, y1)$ and $(x2, y2)$. Normalize so min < max.
        *   `is_valid(x1, y1, x2, y2)` checks if segment $(x1, y1)-(x2, y2)$ is occupied.
        *   Function `get_candidates()` returns list of valid moves sorted by score.
        *   Since performance is critical, maybe we don't generate *all* candidates.
        *   We can maintain a priority queue of candidate moves?
        *   Moves change only when we add a dot or block an edge.
        *   When adding dot at $P$: All rectangles involving $P$ become potentially valid (if others are dotted). Also rectangles where $P$ was the *missing* dot are now impossible (wait, we place dot at $P$, so $P$ is no longer missing).
        *   Actually, we need to track which rectangles could be played.
        *   There are not *that* many rectangles.
            *   Max number of disjoint segments?
            *   Maybe just re-evaluate valid moves periodically? No, too slow.
            *   Just run a greedy loop. In each step:
                1. Iterate over all empty cells $(x,y)$.
                2. Check if there exists a valid rectangle having $(x,y)$ as the new vertex.
                   Since $N \le 61$, checking all possible rectangles for a specific $(x,y)$ is cheap?
                   How many rectangles contain $(x,y)$ as a corner?
                   Axis aligned: depends on neighbors. Max $N^2/4$ combinations of $dx, dy$.
                   Diagonal: depends on neighbors.
                   Summing up over all empty cells is too much work per turn.
        *   Optimization:
            *   Instead of scanning all empty cells, look for the cell $(x,y)$ with max weight.
            *   Then check if it can be part of a valid rectangle.
            *   But $(x,y)$ might not be the *best* cell if it blocks nothing.
            *   Actually, we want to find *the* best valid move. The "value" of a move is $w(p_1)$.
            *   So we want to find $p_1$ with max $w(p_1)$ that allows a valid rectangle.
            *   Is it sufficient to just iterate all empty $p_1$ in decreasing order of $w(p_1)$ and return the first valid one found?
            *   Problem: Picking a move blocks edges. A lower value move might block *fewer* useful edges or enable better ones?
            *   Constraint 3 says edges cannot share segments.
            *   Blocking an edge prevents that edge from being used again.
            *   Usually, blocking edges is unavoidable.
            *   However, we want to prioritize high weight. Let's assume greedy works.
            *   Greedy Strategy: In each step, consider the empty point $(x,y)$ with highest $w(x,y)$. If no rectangle can be formed with $(x,y)$ as the new vertex, skip it. Continue searching among empty points with high $w(x,y)$. Pick the first valid move.
            *   Wait, is it guaranteed that the highest weight point is always the one we want to fill?
            *   Example: Point A (weight 100) can only be filled using an edge E that is crucial for point B (weight 101). Point B needs edge E. If we fill A first, we consume E? No, we consume the *rectangle*. The rectangle consists of edges. If we play a rectangle for A, we draw edges. If those edges are required for B, B becomes unreachable (or rather, the rectangle for B won't be valid because edges are drawn/blocked).
            *   If edge E is a perimeter segment. Both rectangles need E. One uses vertices {v1, v2, v3, v4} for A. Another {u1, u2, u3, u4} for B. They share a side.
            *   If they share a side, we can only play one of them (unless they differ elsewhere, but sharing a segment is banned).
            *   So we have a conflict. We prefer A or B based on $w$.
            *   If both require the same edge, playing the higher weight one consumes it.
            *   If A uses a subset of edges of B (e.g. A is a sub-rectangle?), no, rectangles must be distinct.
            *   Conflict is sharing a perimeter segment.
            *   So we want to pick a valid rectangle corresponding to highest weight empty vertex.
            *   This sounds like Weighted Independent Set on Interval Graphs? Too hard.
            *   Given constraints, likely simple Greedy (Pick highest weight valid move) is intended.
            *   Also, notice "no other dots on the perimeter". This constraint forces us to focus on sparse regions (edges of the current shape). Since we start dense in middle, we expand outwards. This naturally aligns with weight distribution (center low weight, boundary high weight).
            *   So high weight points are at boundary. Empty space is at boundary.
            *   So most high weight points are available.
            *   We just need to ensure we find *some* valid move for the highest weight available point, or the second highest, etc.

4.  **Implementation Details:**
    *   $N \le 61$. Coordinates $0..N-1$.
    *   Priority Queue of candidate moves.
    *   A candidate move is a tuple `(weight_of_new_dot, p1_idx, p2, p3, p4, rectangle_edges_info)`.
    *   Or just store `(w(p1), p1, p2, p3, p4)`.
    *   Generating candidates:
        *   Iterate over all pairs of points $(A, B)$ in grid? No.
        *   We know valid moves involve existing dots.
        *   Let's iterate over all empty points $P_{target}$ (ordered by decreasing weight).
        *   For each $P_{target}$, try to find 3 dots $D_1, D_2, D_3$ such that they form a rectangle with $P_{target}$.
        *   How to find $D_i$ efficiently?
            *   Axis Aligned: $P_{target}=(x,y)$. Possible partners $(x', y')$ where $x'=x$ or $y'=y$?
                *   If $P_{target}$ is corner, the other 3 corners must have same x/y coords.
                *   Say $P_{target}$ is $(x,y)$. Other corners must be $(x, y+\delta_y), (x-\delta_x, y+\delta_y), (x-\delta_x, y)$.
                *   Wait, if $P_{target}$ is $(x,y)$, and it is bottom-left. Then $(x, y+h), (x+w, y+h), (x+w, y)$.
                *   We check these 3 spots. If they are dots, AND perimeter empty, AND edges free -> Valid move.
                *   There are $O(N^2)$ pairs $(w,h)$. Total $O(N^4)$ checks per empty point. Too slow if many empty points.
                *   However, $N$ is small. $N^4 \approx 1.3 \times 10^7$. Number of empty points $\approx 3000$. Total ops $10^{10}$ per step? NO.
                *   We don't need to check *every* pair $(w,h)$.
                *   We can scan neighbors?
                *   Actually, a valid rectangle is defined by its perimeter.
                *   The perimeter has 4 sides. Each side has midpoint or corners.
                *   Key observation: To place a dot at $P$, we need 3 existing dots.
                *   So we can just iterate over all triples of existing dots? No, $M \approx 500-1000$. Triples $10^9$.
                *   Better: Iterate over all *existing* dots as potential $P_2$.
                *   Fix one existing dot $D$ (could be $P_2, P_3, P_4$).
                *   Find pairs $(D_1, D_2)$ such that $D, D_1, D_2$ are corners of a rectangle. Calculate 4th.
                *   This is still complicated.
    
    *   **Optimization: Candidate Generation**
        *   Since we want to maximize weight of *new* point, let's sort all empty points by weight.
        *   For the top-most empty point $P$, we check if it can be part of ANY valid rectangle.
        *   To do this, we don't need to check all possible rectangles. We can derive necessary conditions.
        *   If $P$ is $(x,y)$ and we need a rectangle.
        *   Possibilities for rectangle type (A or B).
        *   If Type A: We need either:
            1.  $(x, y+d), (x-w, y+d), (x-w, y)$ to be dots. (And perimeter checks).
            2.  $(x+d, y), (x+d, y-w), (x, y-w)$ to be dots.
            3.  Symmetric cases.
            Basically $P$ is one corner. We need to find the other 3.
            The other 3 corners must be aligned with axes relative to $P$.
            So we look for $x' > x, y' > y$ such that $(x, y')$ is dot, $(x', y')$ is dot, $(x', y)$ is dot.
            Since we need to verify "no other dots on perimeter", we can just check the bounding box.
            If all 3 exist, we verify perimeter.
            Since grid is small, for fixed $P$, we can iterate all valid $(x', y')$ in $O(N^2)$.
            Wait, checking all 3rd quadrant neighbors takes $O(N^2)$. Doing this for each empty point (approx 3000) is $O(N^4) \approx 3000 \times 60^2 \approx 10^7$. This is perfectly fine for ONE step.
            Wait, 3000 steps $\times 10^7 = 3 \times 10^{10}$. TOO SLOW.
            We need to avoid iterating empty points fully or reducing complexity.
        
        *   **Refinement:**
            *   Don't iterate empty points blindly.
            *   Only check $P$ if it is the "newest" target.
            *   Instead of iterating empty points, iterate over existing dots?
            *   Actually, we can simply maintain a list of valid rectangles.
            *   Update the list dynamically.
            *   Initially compute ALL valid rectangles. (There aren't too many valid rectangles on an empty grid? Maybe $N^4$?).
            *   Storing $10^6$ rectangles is memory heavy but ok.
            *   Wait, valid rectangles change. "No dots on perimeter" makes them invalid quickly as we fill grid.
            *   Dynamic generation is key.
            *   Let's reconsider $N \le 61$. The time limit is usually 2-4 sec.
            *   A brute force scan per step is risky.
            *   Can we reduce the search space for neighbors of $P$?
            *   Notice that for a valid rectangle, the midpoint of the diagonal must be integer or half-integer.
            *   More importantly, for an axis-aligned rectangle defined by $(x,y)$ and $(x', y')$, the perimeter consists of segments on lines $x=x_P, x=x'_P, y=y_P, y=y'_P$.
            *   The condition "no dots on perimeter" is the bottleneck.
            *   We can optimize the check:
                *   Use bitmasks or boolean arrays for rows/cols.
                *   Row `row[y]` stores indices of dots.
                *   Col `col[x]` stores indices of dots.
            *   This helps quickly find if a row/col segment is clear?
            *   Actually, we need to check if any dot exists strictly between corners.
            *   We can precompute "next dot" for each cell to left/right/up/down.
            *   If `next_dot_right(x, y)` > `min_x`, then dot exists?
            *   Yes.
            *   This makes perimeter check $O(1)$.
            *   So the cost is dominated by finding the rectangle configuration.
            *   Finding a valid rectangle with $P$ as a corner:
                *   Axis aligned:
                    *   Search for $P'$ in row $x$ ($y \neq y_P$) and col $y_P$ ($x \neq x_P$).
                    *   Specifically, we need 3 dots.
                    *   Suppose $P=(x,y)$. We need $Q_1$ in row $x$ or col $y$.
                    *   Actually, if $P$ is corner, say TL. We need TR $(x, y')$, BR $(x', y')$, BL $(x', y)$.
                    *   We can iterate $y'$ in range, if $(x, y')$ is dot. Then we need $(x', y')$ and $(x', y)$.
                    *   This requires checking intersection of column $x'$ and row $y'$.
                    *   This is basically intersecting the "cross" centered at $P$.
            *   With $N=61$, iterating $x', y'$ is $O(N^2)$. Still slow if repeated for all $P$.
            *   BUT, notice that we only care about $P$ that has a chance to be the optimal next move.
            *   If the optimal solution is greedy, we just need *one* valid move for the highest weight $P$ (or next, etc.).
            *   Can we speed up "Find valid rectangle for $P$"?
            *   Notice that if $(x, y')$ is a dot, and $(x', y)$ is a dot, we need $(x', y')$ to be a dot.
            *   This is like finding a "corner" completion.
            *   Maybe just iterate all valid *rectangles* formed by existing dots and see if the 4th corner is empty.
            *   How many such rectangles?
            *   We can iterate over all pairs of rows $y_a, y_b$. And columns $x_a, x_b$.
            *   If dots at $(x_a, y_a), (x_a, y_b), (x_b, y_a)$ are present, check if $(x_b, y_b)$ is empty.
            *   This checks axis aligned.
            *   Cost: $O(N^4)$.
            *   Wait, iterating over existing dots:
            *   Take all existing dots.
            *   Group them by row/column?
            *   Map `row[y]` -> vector of x's.
            *   Iterate pairs of rows $y_1, y_2$.
            *   For each $x$ in `row[y1]`, check if $x$ in `row[y2]`.
            *   If both $(x, y_1)$ and $(x, y_2)$ are dots, and we have another $x'$ such that $(x', y_1)$ and $(x', y_2)$ are dots?
            *   Wait, if we pick two rows, and two columns, that defines a rectangle.
            *   If 3 corners exist and 1 doesn't, that's a candidate.
            *   If 4 exist, ignore (already filled).
            *   We want to find 3 existing + 1 empty.
            *   Algorithm:
                *   Store rows, cols.
                *   Iterate over all pairs of rows $r1, r2$.
                *   Intersection of $r1, r2$ gives x-coords where both rows have dots.
                *   Let these be $C = x_1, x_2, ...$
                *   If $|C| \ge 2$, we have potential rectangles between $x_a, x_b$ in rows $r1, r2$.
                *   For every pair $x_a, x_b \in C$:
                    *   Check 3 corners:
                        *   $(x_a, r1), (x_b, r1), (x_a, r2)$ -- need $(x_b, r2)$ empty.
                        *   $(x_a, r1), (x_b, r1), (x_b, r2)$ -- need $(x_a, r2)$ empty.
                        *   ... (4 combos)
                    *   This identifies all axis-aligned candidates in $O(N^3 \log M)$ or $O(N^3)$ if hash maps.
                    *   $N^3 = 200,000$. Very fast.
            *   Now what about 45-degree?
                *   Vertices $(x, y), (x+k, y+k), (x, y+2k), (x-k, y+k)$? No.
                *   45 degree vertices satisfy $(x+y) == const$ or $(x-y) == const$?
                *   Let's parameterize.
                *   Square of side $S\sqrt{2}$.
                *   Vertices: $(x, y), (x+S, y-S), (x, y-2S), (x-S, y+S)$.
                *   Actually, center must be at integer coordinates for all vertices to be integers?
                *   Center $(xc, yc)$. Vertices $(xc \pm S, yc \mp S)$.
                *   $xc, yc$ must be half-integers?
                *   If $S$ is integer, and $xc, yc$ half-integer, vertices are integers.
                *   Example $S=1$, center $(0.5, 0.5) \to (1,0), (1,1), (0,1), (0,0)$? No.
                *   $(0.5+1, 0.5-1) = (1.5, -0.5)$ - not integer.
                *   If vertices are integers, their center is midpoint. Midpoint of $(x_1, y_1)$ and $(x_2, y_2)$ must be integer or half-integer.
                *   Let's just iterate over all pairs of existing dots.
                *   If we treat them as diagonal ends of a 45 deg square.
                *   Two vertices $V_1, V_2$ form a diagonal. Center is midpoint.
                *   Wait, 45 deg square has diagonals perpendicular to sides? No.
                *   Diagonals are axis aligned (vertical/horizontal).
                *   Proof: Sides slope $\pm 1$. Product of slopes $-1$.
                *   Diagonal connects $(x,y)$ to $(x', y')$. Slope $\frac{y'-y}{x'-x}$.
                *   If sides slope $\pm 1$, diagonals slope 0 or $\infty$.
                *   So diagonals are axis aligned segments.
                *   This simplifies everything!
                *   Type 2 (Rotated) rectangles are actually squares with *axis-aligned diagonals*.
                *   So they are defined by a horizontal diameter and vertical diameter?
                *   Yes.
                *   Let the "diagonal endpoints" be $(x_L, y_C)$ and $(x_R, y_C)$. Horizontal segment.
                *   The other diagonal endpoints $(x_M, y_D)$ and $(x_M, y_U)$ must exist for a rectangle?
                *   Wait, a square with 45 deg rotation has vertices at $(x_L, y), (x_R, y) \dots$ NO.
                *   Let's visualize. A diamond. Vertices Up, Down, Left, Right.
                *   Top $(x, y+H)$, Bottom $(x, y-H)$.
                *   Left $(x-W, y)$, Right $(x+W, y)$.
                *   It forms a square iff $W=H$.
                *   Vertices: $(x-W, y), (x, y+W), (x+W, y), (x, y-W)$.
                *   This is a 45-deg square.
                *   The condition "vertices are dots" implies:
                    *   Dot at $(x, y+W)$
                    *   Dot at $(x+W, y)$
                    *   Dot at $(x, y-W)$
                    *   Dot at $(x-W, y)$
                *   We need to pick one of these to be empty (the new dot).
                *   So we search for "diamonds" with 3 vertices dotted.
                *   This is equivalent to checking points $(x,y)$ and offsets.
                *   We can iterate over all "centers" $(x, y)$ (must be integer since $N$ is integer grid? No, vertices are integer, so center $(x,y)$ can be integer or half-integer? No, if vertices are integers, sum/diff are even/odd).
                *   If vertices are integer, then $(x_{top} + x_{bot})/2 = x_{left}$. So $x$ coord of left is avg of top/bot.
                *   If top/bot share x-coordinate (vertical diagonal), left/right share y-coordinate (horizontal diagonal).
                *   So we just iterate over all vertical segments (pairs with same x) and horizontal segments (pairs with same y) in grid.
                *   Wait, we iterate over all potential *corners*?
                *   Let's use the structure: Iterate over all "cross" shapes centered at $(cx, cy)$ with radius $r$ (where $r$ is distance to vertex).
                *   Actually, simpler: Iterate over all empty points $P_{target}$. For each $P_{target}$, check if it completes a cross shape with 3 existing dots.
                *   Axis aligned: Check $O(N^2)$ neighbors.
                *   Rotated:
                    *   If $P_{target}$ is Top vertex. We need Left $(x-r, y), Right (x+r, y), Bottom (x, y-2r)$.
                    *   We can iterate $r$ from 1 to $N/2$.
                    *   Check if 3 other vertices are dots.
                    *   Complexity: $O(N^3)$ per empty point? No, $r$ up to $N$.
                    *   Actually, sum of $r$ over $N^2$ points is $O(N^4)$. Again too slow.
                *   Optimization for Rotated:
                    *   Iterate over all pairs of points $(A, B)$ with same Y (horizontal).
                    *   Midpoint is $((A.x+B.x)/2, A.y)$.
                    *   If $A.x + B.x$ is even (integer midpoint), or odd (half-integer), let $x_c = (A.x+B.x)/2, y_c = A.y$.
                    *   Radius $r = |A.x - x_c|$.
                    *   We need vertical pair at $x_c$: Top $(x_c, y_c+r)$, Bot $(x_c, y_c-r)$.
                    *   Check existence.
                    *   This finds diamonds using existing dots.
                    *   For each such diamond, if 3 are dots, 1 is empty -> candidate.
                    *   Number of horizontal pairs is $O(M^2)$ or $O(N^3)$. $N^3$ is fine.
                    *   We just need to store `valid_moves`.
    
    *   **Combined Strategy:**
        1.  Data structures:
            *   `grid[N][N]`: bool (has dot).
            *   `row_dots[N]`: vector of x.
            *   `col_dots[N]`: vector of y.
            *   `active_rects`: List of potential moves?
            *   We want to find the BEST move.
            *   Let's just maintain a pool of candidates.
            *   Or, simpler:
                Since $N$ is small, at each step:
                1. Generate all *potential* valid moves.
                2. Sort them by $w(p_{new})$.
                3. Pick the top one that is NOT blocked by edges.
                4. If blocked, move to next.
                5. But "Blocked by edges" check is expensive if we do it naively?
                   We can check edges in $O(1)$ or amortized.
                   Wait, "shares a common segment of positive length".
                   Segments are straight lines. We can identify them by endpoints.
                   Hash map `set<pair<tuple>>` to track drawn segments.
        2.  Generating Candidates:
            *   Iterate all empty points $P$.
            *   Try to complete an Axis-Aligned Rectangle (check $O(N^2)$ neighbors or $O(N^2)$ total scan).
            *   Try to complete a 45-degree Square (check neighbors or iterate pairs).
            *   Total complexity per step: $O(N^4)$.
            *   If we do this $K$ times, it might TLE.
            *   Is $K$ large?
            *   Usually, number of operations is related to filling the grid.
            *   If we optimize candidate generation, it could be $O(N^3)$.
            *   $O(N^3)$ per step. $N=60$. $216,000$.
            *   If $K \approx 1000$, total $2 \times 10^8$. Feasible in 1-2s.
            *   If $K \approx 3000$, might be tight.
            *   However, as grid fills, valid rectangles drop drastically.
            *   So average case might be much faster.
            *   The bottleneck is early/mid game when grid is sparse.
            *   Also, note that $N \le 61$ and 2000 test cases (system).
            *   Wait, "System test contains 125 inputs for each of N=31...61". Total 125 cases per seed? No, total 125 inputs.
            *   So average $N \approx 40$. $K \approx$ ?
            *   If we assume worst case $K \approx 2000$, and $N=61$, we need faster than $O(N^4)$.
            *   We can prune the search.
            *   Instead of iterating all empty points, iterate all valid configurations of 3 dots.
            *   Iterate over all pairs of dots $(A, B)$ that share a row or col.
            *   If same row, define base. Look for third dot.
            *   Actually, let's implement a "Check Move for $P$" function efficiently.
            *   Or better: "Generate Candidates" function.
            *   Let's stick to iterating empty points $P$. But limit $r$ search?
            *   Actually, observe: The grid starts empty-ish outside a center ball.
            *   Most valid rectangles have small dimensions?
            *   Not necessarily.
            *   Wait, "perimeter empty" constraint is strong.
            *   A large rectangle has long perimeter. More chance to hit an existing dot.
            *   So likely small rectangles dominate or boundary ones.
            *   Anyway, implementing full check is safer than pruning logic that might miss solutions.
            *   Can we check empty points in increasing order of distance to center? No, we want max weight (distance).
            *   So we check outermost empty points first.
            *   Wait, is it better to iterate *existing* dots and find compatible empty corners?
            *   Existing dots are few ($M$). New dots grow.
            *   If we iterate over existing dots $D$, we can find neighbors.
            *   Let's refine the "Generate Candidates" complexity.
            *   Approach:
                1.  Create a list of "Potential Moves".
                2.  Each Potential Move is represented by a specific rectangle configuration (type A or B) and the index of the empty corner to fill.
                3.  We can generate these by iterating over all subsets of size 3 from existing dots? No, $O(M^3)$.
                4.  Better:
                    *   Iterate over all $y$ rows. Extract dots in row $y$.
                    *   Iterate over pairs of dots $(x_1, y), (x_2, y)$ (same row).
                    *   This pair is a side.
                    *   Look for another row $y'$.
                    *   Intersect $y'$ with $x_1, x_2$.
                    *   If intersections exist, we have a candidate rectangle.
                    *   We also need to handle 45-degree squares.
                    *   Iterate over all $y$ rows.
                    *   Iterate pairs $(x_1, y), (x_2, y)$.
                    *   Treat as horizontal diagonal? Or base?
                    *   For 45-deg square: Base $(x_1, y)-(x_2, y)$ is horizontal chord. Center $x_c = (x_1+x_2)/2, y_c=y$.
                    *   If $x_c$ is valid (half int or int).
                    *   Vertical chord at $x_c$. Length equal to horizontal?
                    *   If we use horizontal segment as diagonal? No, in a 45-deg square, vertices are $(x_c \pm r, y_c)$ and $(x_c, y_c \pm r)$.
                    *   So yes, a horizontal segment between $(x_1, y)$ and $(x_2, y)$ corresponds to two vertices.
                    *   Wait, in the diamond configuration, vertices are Left, Right, Top, Bottom.
                    *   Left $(x_c-r, y_c)$, Right $(x_c+r, y_c)$.
                    *   Top $(x_c, y_c+r)$, Bottom $(x_c, y_c-r)$.
                    *   So if we have Left and Right as dots, we check if Top and/or Bottom are dots.
                    *   If we have Top and Left/Bot, we check Right.
                    *   So:
                        *   Case 1: Top, Bottom are dots. (Vertical chord). Check for Left/Right.
                            *   Iterate vertical chords in the grid (pairs of dots in same column).
                            *   For each chord (Top, Bot), calculate center $x_c, y_c$. Radius $r$.
                            *   Check Left $(x_c-r, y_c)$ and Right $(x_c+r, y_c)$.
                            *   Combinations:
                                *   If Left is dot, Right is empty -> Candidate (add Right).
                                *   If Right is dot, Left is empty -> Candidate (add Left).
                                *   If neither -> no cand.
                        *   Case 2: Top, Right, Bot are dots? No, we just need 3 dots.
                        *   So we just iterate ALL valid geometric configurations (axis aligned rects, 45 deg diamonds) that have 3 dots and 1 empty.
                        *   Since we need to maximize score, we can collect all such configs, compute score for the empty corner, sort, and pick.
                        *   How to find such configs efficiently?
                        *   The number of axis aligned rects is bounded by $N^2 \times (\text{number of rows} \times \text{number of cols})$.
                        *   Actually, we can just iterate over all possible "bottom-left" $(x,y)$ and "width $w$, height $h$".
                            *   If dots exist at $(x,y+w), (x,y+h), (x+w, y+h)$ and $(x+w, y)$ is empty.
                            *   Wait, $(x,y)$ is empty, but it could be any corner.
                            *   So we just need to check if for any rect, 3 are dots, 1 is empty.
                            *   This is $O(N^4)$. $60^4 \approx 1.3 \times 10^7$.
                            *   This is perfectly fine per step IF $K$ is not huge.
                            *   $K$ max operations? If we fill every cell, $K \approx N^2$. $3600 \times 1.3 \times 10^7 \approx 4.6 \times 10^{10}$. Too slow.
                            *   BUT, most rectangles are invalid due to "dots on perimeter" constraint.
                            *   Most rectangles are invalid because of existing dots?
                            *   In early stages, perimeter constraint is loose. But valid moves are few? No, lots of empty space.
                            *   Actually, the number of valid moves is often small.
                            *   Can we optimize?
                            *   Yes, iterate over *only* existing dots.
                            *   Since we need 3 existing dots.
                            *   If we pick 3 existing dots, check if they form 3 corners of a valid rectangle.
                            *   Number of triples is $M^3$. $M$ starts small (e.g. 100). $100^3 = 10^6$.
                            *   As game progresses $M$ grows.
                            *   But we only need to find ONE valid move at each step.
                            *   We can stop once we find valid moves.
                            *   Wait, we need to find the BEST move.
                            *   So we might need to scan many triples to find all valid moves?
                            *   If we just scan all valid moves and sort.
                            *   The set of valid moves is likely sparse.
                            *   Let's go with: Iterate over all pairs of existing dots.
                                *   If they share a row/col, they form a side.
                                    *   For side $(x_1, y)-(x_2, y)$.
                                    *   Look for rows $y'$ such that both $(x_1, y')$ and $(x_2, y')$ are dots? Or one is dot?
                                    *   This generates rectangles.
                                *   If they are diagonal (axis aligned): e.g. $(x, y)-(x', y')$.
                                    *   For 45 deg square: Vertices must be $(x_c \pm r, y_c)$ etc.
                                    *   If $(x, y)$ and $(x', y')$ are opposite vertices (e.g. Left and Right), then $x_c = (x+x')/2, y_c=y=y'$.
                                    *   We check Top/Bot.
                            *   Algorithm Refined:
                                1. Collect all candidate moves into a `vector<Move> candidates`.
                                2. Iterate through all pairs of existing dots $(D_1, D_2)$ with $D_1 < D_2$.
                                   *   If $D_1.y == D_2.y$: (Horizontal segment).
                                       *   Midpoint $(mx, my)$. $r = |D_1.x - mx|$.
                                       *   Check vertical segment at $mx$.
                                       *   If $mx$ is half-int (requires $D_1.x+D_2.x$ even/odd? No, $mx = (x1+x2)/2$. If $x1, x2$ both odd/even, $mx$ int. If mixed, $mx$ half).
                                       *   Wait, for 45 deg square, vertices $(x_1, y), (x_2, y)$ imply $x_c = (x_1+x_2)/2, y_c=y$.
                                       *   Other vertices must be $(x_c, y \pm r)$.
                                       *   Check if these are grid points (integers).
                                           *   If $y \pm r$ is integer. Since $y$ int, $r$ must be int.
                                           *   So $x_1, x_2$ must have same parity.
                                           *   If so, $r = (|x_1-x_2|)/2$. Check $(mx, y-r)$ and $(mx, y+r)$ existence.
                                        *   If $x_1 \equiv x_2 \pmod 2$:
                                            *   $mx, my$ are integer coords.
                                            *   Check if $(mx, y-r)$ and $(mx, y+r)$ are dots/empty.
                                            *   Form rectangles.
                                   *   If $D_1.x == D_2.x$: (Vertical segment). Same logic.
                                   *   Else: They are not axis-aligned side. Could they be 45 deg diagonal?
                                       *   No, we assumed 45 deg squares have axis-aligned diagonals.
                                       *   Wait, earlier I concluded: 45 degree square vertices are $(x_r, y), (x, y_r)...$
                                       *   Diagonals connect $(x_r, y)$ to $(x, y_r)$? No.
                                       *   Diamond vertices: $L, R, U, D$.
                                       *   $L=(x-L.x, y), R=(x+R.x, y)$ share $y$.
                                       *   $U=(x, y+U.y), D=(x, y-D.y)$ share $x$.
                                       *   So diagonals are indeed vertical and horizontal segments.
                                       *   So we cover all cases by checking axis-aligned segments (vertical or horizontal).
                                       *   Wait, what about Axis-Aligned Rectangles?
                                           *   Vertices $(x_L, y_B), (x_R, y_B), (x_R, y_T), (x_L, y_T)$.
                                           *   Sides are horizontal $(x_L, y_B)-(x_R, y_B)$ etc.
                                           *   This is covered by "Horizontal Segment" check.
                                           *   If we find horizontal segment, we look for matching points.
                                               *   Case A: Matching points form a rectangle.
                                                   *   If we have $D_1, D_2$ on row $y_1$.
                                                   *   We look for row $y_2$.
                                                   *   If $D_1, D_2$ also in row $y_2$ -> already processed (duplicate).
                                                   *   If we have a single dot $D_3$ in row $y_2$ matching $D_1$ or $D_2$.
                                                   *   If we have both -> completed rect.
                                                   *   If we have none -> no match.
                                                   *   So we need to check for the *existence* of matching points.
                                                *   Wait, iterating all pairs of existing dots is $O(M^2)$.
                                                *   For each pair, finding the 3rd/4th dot takes query time.
                                                *   We need to check ALL potential completions.
                                                *   This effectively generates rectangles.
                                3.  Complexity: $O(M^2 \cdot N)$. Since $M \le N^2/12 \approx 300$ (for $N=60$, $N^2/12 \approx 300$).
                                    *   Wait, $M$ grows. Final $M$ could be $N^2$.
                                    *   But we stop generating when grid is full.
                                    *   If $M$ is large, $O(M^2)$ is too big.
                                    *   BUT, notice "perimeter empty" condition.
                                    *   If $M$ is large, most rectangles are invalid.
                                    *   Maybe just iterate empty points $P$.
                                    *   For each empty point $P$, find 3 neighbors that form a rectangle.
                                    *   Since we need *any* valid rectangle with $P$ as corner.
                                    *   If $P$ is corner of axis aligned rect: We need $P=(x,y)$. Other corners $(x', y), (x, y'), (x', y')$.
                                        *   We need $(x', y)$ dot, $(x, y')$ dot.
                                        *   If we have these, check $(x', y')$.
                                        *   This requires iterating over $x'$ and $y'$.
                                        *   We can use the `row_dots` and `col_dots` structures.
                                        *   `row_dots[y]` has list of x-coords.
                                        *   `col_dots[x]` has list of y-coords.
                                        *   For $P=(x,y)$:
                                            *   We need a partner in row $y$: $x_{partner} \in row\_dots[y], x_{partner} \neq x$.
                                            *   We need a partner in col $x$: $y_{partner} \in col\_dots[x], y_{partner} \neq y$.
                                            *   So iterate $x' \in row\_dots[y]$ and $y' \in col\_dots[x]$.
                                            *   Check if $(x', y')$ is dot.
                                            *   If so, we have 3 dots $(x, y')$, $(x', y)$, $(x', y')$. $P=(x,y)$ is empty.
                                            *   Check perimeter.
                                            *   This is $O(|row\_dots[y]| \cdot |col\_dots[x]|)$.
                                            *   Worst case $O(N^2)$. Sum over all empty $P$ is $O(N^4)$.
                                            *   Still high.
                                            *   BUT, we can iterate *sorted* empty points. Stop when no moves found? No.
                                            *   Optimization:
                                                *   Don't iterate all pairs $(x', y')$.
                                                *   Check if *any* valid config exists.
                                                *   Actually, maybe just iterating valid rectangles via pairs of existing dots is better if $M$ is small.
                                                *   If $M$ is large, maybe iterate empty points.
                                                *   Hybrid?
                                                *   Actually, we only need to find ONE valid move to advance.
                                                *   Wait, the goal is max score. We might need to skip low scoring moves if they block high scoring ones.
                                                *   So we ideally want to find the absolute best move.
                                                *   If we just greedily pick best available, it's safe.
                                                *   To find best available:
                                                    *   We need to evaluate potential moves.
                                                    *   If we sort empty points by weight.
                                                    *   Pick top empty point $P$.
                                                    *   Check if it participates in any valid move.
                                                    *   If yes, great! Pick the best one associated with $P$ (if multiple exist for $P$, pick highest weight? No, $w(P)$ is fixed).
                                                    *   If not, discard $P$, move to next.
                                                    *   This avoids checking ALL empty points.
                                                    *   Wait, checking if $P$ participates in valid move:
                                                        *   Check all possible rectangles having $P$ as corner.
                                                        *   This is $O(N^2)$ checks (neighbors).
                                                        *   Doing this for every empty point is $O(E \cdot N^2) \approx O(N^4)$.
                                                        *   Is it?
                                                        *   Yes.
                                                    *   Can we optimize "Check if $P$ is corner of valid rectangle"?
                                                        *   We need to check if there exists $D_1, D_2, D_3$ completing it.
                                                        *   If $P$ is empty, and we check specific neighbor pair $(x_{part}, y_{part})$.
                                                        *   We can use data structures to quickly check if valid rectangle exists anchored at $P$.
                                                        *   Or, we can just check *all* valid rectangles.
                                                        *   If we iterate all valid rectangles, we can store them.
                                                        *   Valid rectangles can be identified by their "defining features".
                                                        *   For axis aligned, defined by $(x_L, y_B)$ and $(x_R, y_T)$.
                                                            *   We can find such rects by iterating over all pairs of existing dots in same row (or col).
                                                            *   Wait, if $M$ is large, iterating pairs is slow.
                                                            *   But we only need *candidates*.
                                                            *   If a rectangle has 3 dots, it is a candidate.
                                                            *   If we maintain a set of "Active Rows/Cols" (rows/cols with $\ge 1$ dots).
                                                            *   Actually, the "Scan Neighbors" approach is likely robust enough given the density of valid rectangles decreases rapidly.
                                                            *   And checking empty points is better.
                                                            *   Why? Because we stop as soon as we find a valid move for the current best empty point.
                                                            *   If no valid move, we move to next.
                                                            *   So on average we do fewer checks.
                                                            *   Also $N=61$ is quite small.
    *   **Algorithm Plan:**
        1.  Initialize `grid`.
        2.  Precompute `weights` (or calc on fly).
        3.  Loop while true:
            *   Find the empty point $P$ with max $w(P)$.
            *   Try to find valid moves involving $P$.
                *   Iterate over all possible rectangle configurations where $P$ is a corner.
                *   Axis aligned:
                    *   Iterate $x' \in [0, N-1], x' \neq x$.
                    *   Iterate $y' \in [0, N-1], y' \neq y$.
                    *   Determine corners. E.g. if $P=(x,y)$ is Top-Left.
                    *   Need TR $(x, y')$ and BR $(x', y')$ and BL $(x', y)$.
                    *   Check if these are dots.
                    *   If so, check perimeter emptiness.
                    *   Check edge conflicts.
                    *   If valid, record move with score $w(P)$.
                *   Diagonal (45 deg):
                    *   $P$ is a vertex. e.g. Top.
                    *   Need Center $(x_c, y_c)$.
                    *   If $P=(x,y)$ is Top, then $y = y_c + r$. $y_c = y-r$.
                    *   We need $r > 0$.
                    *   Check Left $(x-r, y)$, Right $(x+r, y)$, Bottom $(x, y-2r)$.
                    *   Iterate $r$.
                    *   Check dots. Check perimeter. Check conflicts.
                    *   If valid, record move with score $w(P)$.
            *   If valid moves found, pick the one with HIGHEST score. (Here score is $w(P)$ if unique, but if multiple rects share $P$, score is same).
            *   If NO valid moves for $P$:
                *   Remove $P$ from consideration (effectively done).
                *   Move to next best empty point.
                *   Wait, this is greedy. Is it optimal?
                *   It is greedy on the set of empty points.
                *   Within the set of empty points with same weight (unlikely), any is fine.
                *   Is it possible that a lower weight point $Q$ could trigger a higher weight move later?
                *   Placing a dot at $Q$ creates new constraints (blocks edges).
                *   Usually we want to create *more* freedom or at least not block.
                *   Since we fill from outside in, creating boundary dots helps fill interior?
                *   Wait, interior dots are needed to form interior rectangles? No, interior rectangles require existing dots.
                *   The "no other dots on perimeter" suggests we need sparse areas.
                *   Boundary is sparse. Interior is dense.
                *   So valid rectangles are mostly on boundary.
                *   Moving outward (high weight) places dots on boundary.
                *   Once placed, they can serve as bases for inner rectangles.
                *   So the order seems natural: Fill boundary with high weight points first.
                *   Thus, the greedy strategy of picking the highest weight available empty point that can participate in *any* valid move is solid.
            *   **Edge Conflict Check**:
                *   A segment is defined by endpoints.
                *   For axis aligned: segments on grid lines.
                *   For diagonal: segments are slanted.
                *   Slanted segments are unique to that rectangle.
                *   Storage: `std::set<std::string>` or `map<pair<Point, Point>, bool>`.
                *   Canonical representation: `min(p1, p2) < max(p1, p2)`.
                *   Since $N$ small, string "x1:y1,x2:y2" is okay.
            *   **Perimeter Check**:
                *   Check every point on perimeter.
                *   Since rectangles are small-ish, maybe iterate points.
                *   Optimized: Use the "next dot" pointer array for each row/col and diagonal.
                *   Actually, simply iterating points on segment is fast enough for $N=61$. Segment length $\le 61$. Total perimeter length $\le 4N$.
                *   $4 \times 60 = 240$ points. Fast.

        4.  Wait, what if no moves exist? Terminate.

        5.  One Detail: The problem says "choose a grid point $p_{i,1}$... and three grid points $p_{i,2}, p_{i,3}, p_{i,4}$ containing dots".
            *   We iterate $P_{target}$ (empty).
            *   We find 3 dots $D_1, D_2, D_3$ forming a rectangle with $P_{target}$.
            *   We need to ensure no *other* dots on perimeter.
            *   Also ensure perimeter doesn't share segment with drawn.

    *   **Refined Algorithm:**
        1.  Read input. Mark `grid[dot] = true`.
        2.  `used_edges` map.
        3.  Loop:
            *   Build list of empty points.
            *   Sort by weight desc.
            *   For each empty point $P$:
                *   Generate potential moves involving $P$.
                    *   Check all configurations (4 orientations for AA, 4 for Diag? No, just symmetry).
                    *   Check validity:
                        *   Are 3 corners dots?
                        *   Is perimeter empty of dots?
                        *   Is perimeter clear of edges?
                *   Keep track of best move found for $P$.
                    *   Wait, if multiple moves for $P$, any is fine?
                    *   Wait, if multiple valid moves for $P$ exist, they share the same new dot $P$.
                    *   Different moves correspond to different rectangles.
                    *   Which rectangle to pick?
                    *   Both add dot $P$. Score is same.
                    *   However, picking one blocks edges.
                    *   We should pick the one that blocks "least" or "best"?
                    *   Hard to predict. Just pick the first found or arbitrary?
                    *   Actually, if we find a valid move for $P$, it is a valid contribution to the score.
                    *   Is it possible that one rectangle blocks a better future move?
                    *   Yes.
                    *   Maybe we should prioritize moves that share edges with the "best" future moves?
                    *   But future moves are unknown.
                    *   Given the difficulty, maybe we assume greedy on P is sufficient. If multiple rects for P, try all?
                    *   Actually, if $P$ is corner of multiple valid rects.
                        *   Example: $P$ is corner of Rect A and Rect B.
                        *   Both use $P$ and other dots.
                        *   Rect A uses edges $E_{A1}, E_{A2}, \dots$
                        *   Rect B uses $E_{B1}, \dots$
                        *   Picking A blocks $E_{A}$. Picking B blocks $E_{B}$.
                        *   If we pick A, we lose potential Rect A'. If we pick B, lose B'.
                        *   If A and B are disjoint in terms of other corners, they use same perimeter?
                        *   No, if they share P, they share edges incident to P?
                        *   Yes, if P is a corner, edges incident to P are part of rectangle perimeter.
                        *   So picking A blocks edges connected to P. Picking B blocks edges connected to P.
                        *   Likely they share at least one edge.
                        *   So we can just pick any valid one. It reduces options for P.
                        *   Actually, to be safe, maybe iterate all valid moves for all *empty* points.
                        *   Collect all valid moves in a global list.
                        *   Sort by weight of the new dot.
                        *   Pick top move.
                        *   Check if edges are blocked? (Iterate sorted list, check conflicts).
                        *   If blocked, skip.
                        *   This ensures we pick the globally best valid move available.
                        *   This handles the choice between rects better.
                        *   And we don't commit to P too early (though P dictates the move).
                        *   Sorting by $w(new\_dot)$ is the primary objective.
            *   Wait, collecting *all* valid moves is $O(N^4)$.
            *   We can filter:
                *   Generate moves.
                *   Add to vector.
                *   Sort by score.
                *   Iterate to pick best non-conflicting.
            *   This seems safest.
            *   Complexity: $O(N^4)$ per iteration.
            *   With $N=61$, $N^4 \approx 1.3 \times 10^7$.
            *   If iterations $K \approx 3000$, total $3 \times 10^{10}$.
            *   This is definitely TLE.
            *   We MUST avoid generating all candidates every time.
            *   We only need to check if there is a valid move for the *highest weight empty point*.
            *   If yes, pick it.
            *   If multiple, pick one (maybe random or check edge count?).
            *   Why check all? Because picking highest P might be suboptimal if it blocks a higher weight *future* point?
            *   But all current empty points have weights.
            *   If we process empty points in descending order of weight, we are considering "best available dot".
            *   The only reason to skip P is if it's not part of *any* valid rectangle.
            *   Why would it block a higher score? Higher score comes from *new* dots.
            *   We haven't reached higher weight empty points? Wait, we sorted empty points by weight.
            *   So all subsequent empty points have lower weight.
            *   So blocking a move for a lower weight point is never worse than taking a move for higher weight point (current P).
            *   So Greedy on P (Highest Weight Empty Point) is justified.
            *   If multiple moves for $P$, which one?
                *   They all give same immediate gain $w(P)$.
                *   They block different edges.
                *   Blocking edges affects future moves.
                *   Future moves will be for points with weight $\le w(P)$.
                *   We prefer to leave the grid flexible for these lower weight points?
                *   Maybe picking the move with shortest perimeter or something?
                *   Or maybe just picking the first one is fine.
                *   Or maybe picking the one that uses "inner" vs "outer" boundaries.
                *   Let's stick to just picking the first valid one found for $P$ if we iterate systematically.
            *   Actually, wait. The edges are "shared".
            *   If we use an edge, it's gone.
            *   If edge is shared between two valid rectangles $R_1, R_2$.
            *   $R_1$ targets $P_1$. $R_2$ targets $P_2$.
            *   If $P_1$ has weight 100, $P_2$ has weight 90.
            *   If $R_1$ and $R_2$ share an edge, we can only play one.
            *   If we play $R_1$, we get 100 points. $R_2$ becomes invalid. Loss = 90. Net gain 100.
            *   If we play $R_2$, we get 90. $R_1$ becomes invalid.
            *   Obviously play $R_1$.
            *   So sorting by weight of target point is the correct global priority.
            *   So algorithm:
                1. Identify all empty points. Sort by weight.
                2. Iterate $P$ in sorted list.
                3. For this $P$, find *all* valid rectangles (checking all configs).
                4. Among these rectangles, pick the "best" one.
                    *   Wait, if we have a list of valid rectangles for $P$, do we need to choose?
                    *   They are all valid with respect to $P$.
                    *   They differ by the other 3 corners.
                    *   They all use edges incident to $P$.
                    *   They also use other edges.
                    *   Do they conflict with each other?
                    *   Yes, they might share edges (e.g. same $P$ but different $R$).
                    *   Wait, if they share edges, we can't do both.
                    *   But we only do one move now.
                    *   Which one?
                    *   Probably the one with minimal "damage" to future possibilities?
                    *   Damage = consuming edges.
                    *   Edges consumed: perimeter of rectangle.
                    *   Smaller perimeter?
                    *   Or edges that are unlikely to be reused?
                    *   Edges near boundary are more valuable?
                    *   Let's guess: Minimize perimeter length? Or maximize?
                    *   Usually small rectangles allow filling more granularly?
                    *   Actually, for competitive programming, usually "simple" is best.
                    *   Let's just pick the first valid one we find? No, deterministic order matters.
                    *   Maybe prioritize axis aligned over diagonal?
                    *   Actually, let's keep the list of candidate moves for $P$, and sort them by some secondary criterion (e.g. min perimeter length).
                    *   Or just iterate $P$ and try to find *any* valid move.
                    *   Wait, if we have multiple valid moves for $P$, they compete for the same "slot" of operation.
                    *   We execute ONE operation.
                    *   If we pick one, the others become unavailable (perimeter shares).
                    *   Does it matter? Yes, if one blocks a path to a high value node.
                    *   But all current nodes have $\le w(P)$.
                    *   So blocking a node with $w < w(P)$ is inevitable if we fill $P$.
                    *   However, we might prevent reaching $P$ if we picked someone else?
                    *   No, we are processing $P$ specifically.
                    *   The decision is which rectangle to build for $P$.
                    *   Hypothesis: Just pick the first one found.
            *   Wait, what if we skipped $P$ because it had no valid move, but picking a lower weight point $Q$ unlocks $P$?
            *   Unlocking?
            *   $P$ needs 3 existing dots.
            *   Adding $Q$ adds a dot.
            *   So if $Q$ is part of a rectangle with $P$, maybe.
            *   But $P$ requires 3 existing dots. Currently none exist (except maybe 0, 1, 2).
            *   If $P$ needs 3 dots, and currently only 2 exist. We need to create the 3rd.
            *   Creating the 3rd dot means playing a move that uses $P$? No, $P$ is the new dot.
            *   So we need to find a way to get the 3rd dot.
            *   The 3rd dot must be placed using another move.
            *   So we need a chain of moves.
            *   If $P$ cannot be filled yet (insufficient dots), it will never be filled?
            *   Unless we fill it later? No, if we don't play a move targeting $P$, $P$ remains empty forever.
            *   Wait, $P$ could be $P_{i,2}$ (existing dot) for a later move?
            *   No, we are deciding *where to put the NEW dot*.
            *   So if we skip $P$ as a target, it stays empty unless it becomes a support for some OTHER rectangle?
            *   Wait, the condition is "choose $p_{i,1}$ not containing a dot".
            *   If we choose to play a move targeting $Q$, $Q$ gets a dot.
            *   If $Q$ happens to be needed for a rectangle targeting $P$ later?
            *   Wait, if $Q$ is a vertex of rectangle for $P$.
            *   If $Q$ is empty, we can't use it.
            *   If we target $Q$, $Q$ becomes a dot.
            *   Then later we can target $P$.
            *   This is the dependency.
            *   So if we prioritize $P$ (high weight) but $P$ cannot be targeted (needs dots), we should look for ways to provide those dots.
            *   i.e. Target the dots needed for $P$!
            *   This implies we should not just look at highest weight *empty* point.
            *   We should look at highest weight *potential* move.
            *   But finding potential moves requires knowing which points are empty.
            *   The dependency graph is: $P$ depends on $\{D_1, D_2, D_3\}$.
            *   If all $\{D_1, D_2, D_3\}$ are dots, we can target $P$.
            *   If not, we must target one of $\{D_1, D_2, D_3\}$ (if it is empty) to make it a dot.
            *   So, we should pick the move that fills a "critical" missing dot?
            *   This is getting complicated.
            *   BUT, notice the input: Initial dots are in center.
            *   Valid rectangles tend to form around the cluster of dots.
            *   As we add dots, the cluster expands.
            *   It is highly probable that for the outermost empty point $P$, we can find a valid rectangle.
            *   Because the "source" of dots (center) is fixed.
            *   The constraint "no other dots on perimeter" means we look for gaps.
            *   Gaps are most numerous at boundary.
            *   Boundary points are high weight.
            *   So $P$ (boundary) is likely to have 3 neighbors that are dots (formed by previous expansion or initial seeds).
            *   Example: Initial dots form a square. $P$ is a corner extending it.
            *   Wait, $P$ is empty. If we have 3 dots forming L-shape with corner at empty spot, we can fill it.
            *   This is basic connectivity.
            *   Given the problem type, a simple greedy "Fill highest weight valid spot" is extremely likely to be the intended solution.
            *   We just need to check if valid spot exists.
            *   If we iterate $P$ from highest weight.
            *   If valid move exists for $P$: Pick it.
            *   If not: Move to next.
            *   Wait, "If not" might mean we missed a dependency?
            *   No, if we can't target $P$ directly, it means we can't place a dot at $P$.
            *   So $P$ remains empty.
            *   This is acceptable.
            *   The question is: is there a better $Q$ (lower weight) such that targeting $Q$ eventually helps us target $P$?
            *   Targeting $Q$ puts a dot at $Q$.
            *   If $Q$ is needed for $P$, yes.
            *   If $Q$ is NOT needed for $P$, but we target $Q$, does it help?
            *   It adds score $w(Q)$.
            *   So targeting $Q$ is always good immediately.
            *   Does it hurt $P$?
            *   Targeting $Q$ might block a move for $P$ (by sharing perimeter edge).
            *   If $Q$ and $P$ are "compatible" (can both be targeted independently), good.
            *   If they conflict (share an edge needed for both), we pick the higher weight one.
            *   So the strategy holds: Always try to target the highest weight empty point that is validly reachable. If multiple, pick one.
            *   What if $P$ is not reachable yet (needs 3 dots, only 2 exist)?
            *   Then $P$ is a "ghost" for now.
            *   We skip it.
            *   Later, after some ops, $P$ might become reachable.
            *   So we re-check $P$?
            *   Yes. In next step, $P$ might be reachable.
            *   So the loop: "While moves possible":
                *   Scan all empty points sorted by weight.
                *   Find first one with valid move.
                *   Execute.
                *   Repeat.
            *   This is $O(\text{Steps} \cdot (\text{EmptyPoints} \cdot \text{Cost}))$.
            *   If Steps $\approx N^2$. Empty $\approx N^2$. Cost $\approx N^2$. Total $O(N^6)$?
            *   $61^6 \approx 5 \times 10^{10}$. TLE.
            *   We need to avoid rescanning ALL empty points.
            *   Notice that $P$ stops being reachable? No, we only disable it once we fill it.
            *   If we fill $Q$ (lower weight), maybe $P$ becomes reachable.
            *   But $Q$ is lower weight.
            *   Wait, if $P$ is reachable, it's high priority.
            *   We should keep a set of "candidate empty points".
            *   But finding candidates is the issue.
            *   Let's optimize "Check if $P$ has valid move".
                *   For a specific $P$, checking all rectangles takes $O(N^2)$.
                *   If we do this for all $P$, it's slow.
                *   But how many $P$ are valid candidates?
                *   Likely few.
                *   Most empty $P$ have no valid rectangle (surrounded by empty space or blocked).
                *   Wait, if $P$ is isolated empty, it has no valid move (needs 3 dots).
                *   If $P$ is close to dots, it might have one.
                *   We can optimize:
                    *   Iterate $P$. If $P$ is too far from any dot, skip.
                    *   Distance threshold?
                    *   Or simply, iterate over valid rectangles generated from *dots*.
                    *   Generate all valid rectangles from existing dots.
                    *   Filter those with 3 dots.
                    *   Get the 4th corner $P$.
                    *   If $P$ is empty, record it as candidate move for $P$.
                    *   Add to list.
                    *   Repeat until no new valid rectangles.
                    *   Pick best from list.
                    *   Execute move.
                    *   Update grid.
                    *   Regenerate candidates (incremental or full).
            *   Full regeneration:
                *   Start with empty grid.
                *   Compute all valid rects (from $M$ dots).
                *   List of candidates (rects with 3 dots).
                *   Sort by weight of 4th corner.
                *   If no candidates, terminate.
                *   Else, pick top.
                *   Check conflicts.
                *   Execute.
                *   Now grid has $M+1$ dots.
                *   Some existing candidates might be invalid (edge blocked or perimeter changed).
                *   New candidates might appear.
                *   To manage this efficiently:
                    *   Re-run the full generation is $O(N^4)$ which is risky.
                    *   Incremental generation:
                        *   Maintain list of rectangles.
                        *   When dot added at $P_{new}$:
                            *   Remove invalid rectangles.
                            *   Insert new rectangles formed by $P_{new}$.
                            *   Update candidates.
                    *   This seems viable.
                    *   Complexity:
                        *   Number of rectangles $\approx N^4$. (Too many).
                        *   Wait, "valid" rectangles are few?
                        *   No, *potential* rectangles are many.
                        *   But valid ones (with empty perimeter) are rare?
                        *   Yes, likely.
                        *   But we need to check them.
                        *   Checking perimeter is fast.
                        *   But generating them?
                        *   Generating all rectangles from $M$ dots is $O(N^4)$ or $O(M^2 N)$?
                        *   Axis aligned: iterate all pairs of rows $O(N^2)$. For each row, check pairs of dots.
                        *   Let's use the property:
                            *   Axis aligned rects are defined by 2 rows and 2 cols.
                            *   Iterate all pairs of rows $(r1, r2)$.
                            *   Intersect cols. If $\ge 2$ common cols, we have $\binom{C}{2}$ rects.
                            *   Check validity.
                            *   45 deg: Defined by vertical chord (or horizontal).
                            *   Iterate all pairs of columns $(c1, c2)$.
                            *   Intersect rows.
                        *   Complexity of generating ALL rectangles:
                            *   Iterate rows $O(N^2)$.
                            *   For each row pair, intersect dot lists (size $M/N$).
                            *   If list size $k$, generate $k^2$ rects.
                            *   Total $\sum k^2$. In worst case $k \approx M$, $\sum M^2/N^2 \approx N \cdot (N^2)^2 / N^2 = N^3$.
                            *   Wait, $M \approx N^2/12$.
                            *   So $k \approx N$. $\sum N^2 \approx N^2 \cdot N = N^3$.
                            *   $N^3 = 200,000$.
                            *   This is FAST.
                            *   So we can regenerate all potential rectangles in $O(N^3)$ (actually $O(M^2)$ or $O(N^3)$ depending on density).
                            *   Wait, 45 deg squares are harder?
                            *   Defined by 1 row and 2 columns? Or 1 vertical line segment.
                            *   Iterate all vertical segments (pair of dots in same col).
                            *   Check if horizontal chord exists.
                            *   This is also fast.
                        *   So Algorithm:
                            1.  Maintain current set of valid rectangles `RectList`.
                            2.  Also maintain `ValidMoves` (subset of `RectList` where perimeter is empty).
                            3.  Initially populate `RectList` by scanning all grid pairs.
                            4.  While `ValidMoves` not empty:
                                *   Sort `ValidMoves` by weight of 4th corner.
                                *   Find top move.
                                *   Check if edges blocked.
                                *   If blocked, remove from `ValidMoves`? No, just skip.
                                *   If valid, execute.
                                *   When execute:
                                    *   Add dot.
                                    *   Draw edges.
                                    *   Remove invalid rects (blocked by edges).
                                    *   Add newly formed rects (those utilizing the new dot and other existing dots).
                                    *   Re-evaluate perimeter emptiness.
                            5.  Updating `RectList` and `ValidMoves` efficiently.
                                *   Removing blocked rects: Any rect whose perimeter intersects drawn edges is invalid.
                                *   Can mark them.
                                *   New rects: We need to find rectangles involving the new dot $P$.
                                *   Search for partners $D_1, D_2, D_3$ for $P$.
                                *   This is $O(N^2)$ per new dot.
                                *   This fits.
    
    *   **Summary of Implementation:**
        1.  Classify rectangles.
            *   Type 1 (Axis): $(x_1, y_1), (x_2, y_1), (x_2, y_2), (x_1, y_2)$.
            *   Type 2 (Diag): $(x_1, y_1), (x_2, y_2)$ (diag), $(x_3, y_3), (x_4, y_4)$ (other diag).
            *   Simpler: Iterate over all grid points as center and radius? No.
            *   Best: Iterate over pairs of rows/cols.
                *   For Axis: Iterate $r1, r2$. Get common $x$s.
                    *   Let common x list be $L$.
                    *   Iterate $x_a, x_b \in L$.
                    *   If all 4 corners are dots, it's filled.
                    *   If 3 are dots, 1 empty -> Candidate.
                    *   If 2 dots, 2 empty -> Ignore (need 3).
                    *   If <2 dots -> Ignore.
                    *   Wait, this finds *all* axis aligned rectangles with corners on existing dots.
                    *   Then check perimeter emptiness.
                *   For Diag: Iterate $c1, c2$. Get common $y$s.
                    *   Let common y list be $L$.
                    *   Iterate $y_a, y_b \in L$.
                    *   Vertices $(y_a, c1), (y_a, c2), (y_b, c1), (y_b, c2)$? No, for square with axis diagonals.
                    *   Wait, my previous derivation: Diagonal vertices must share x or y?
                        *   Square with sides slope 1/-1.
                        *   Diagonals are axis aligned.
                        *   Vertices are $(x_c \pm r, y_c)$ and $(x_c, y_c \pm r)$.
                        *   So pairs of vertices share $y$ (if horizontal pair) or $x$ (if vertical pair).
                        *   So we look for existing dots forming these pairs.
                        *   Iterate all vertical pairs $(x, y_1), (x, y_2)$ in same col.
                        *   Let this be the vertical diagonal.
                        *   Center $x_c = x, y_c = (y_1+y_2)/2$.
                        *   Radius $r = |y_1-y_2|/2$.
                        *   If $y_1, y_2$ same parity, $y_c$ int.
                        *   Check horizontal pair at $y_c$: $(x_c-r, y_c), (x_c+r, y_c)$.
                        *   Check counts of dots.
                        *   If valid candidate.
                        *   Also check horizontal pairs (symmetric).
                            *   Wait, horizontal pair covers same squares?
                            *   If vertical diag exists, square exists.
                            *   So just one check is enough.
    
        2.  Structure:
            *   `struct Rect { Point p[4]; int type; };`
            *   `set<pair<Point, Point>> used_edges;`
            *   `vector<Rect> candidates;`
            *   Loop:
                *   Identify valid moves (3 dots + empty 1).
                *   Sort by score.
                *   Pick best unblocked.
                *   Add dot, mark edges.
                *   Prune candidates.
                *   Add new candidates using new dot.
    
        3.  Performance:
            *   Pruning: Check if rect blocked by edges. $O(1)$ or $O(N)$ if naive.
            *   Re-scanning: We can maintain a boolean flag for each candidate.
            *   If blocked, invalidate.
            *   We only need to keep candidates with 3 existing dots (and 1 empty).
            *   When dot added, candidates needing that dot become valid (if not already) or invalid.
            *   Wait, if a rect needs 3 dots, and we add the 4th, it becomes fully dotted -> ignore.
            *   If it needed 4 dots and had 3, now it needs 3 (still).
            *   Wait, we only care about rects with exactly 3 dots (1 empty).
            *   When we add dot, some candidates change status.
            *   Actually, easier to just re-generate all candidates?
                *   Since $O(N^3)$ is fast ($2 \cdot 10^5$), and we do it $K$ times.
                *   If $K \approx 3000$, $6 \cdot 10^8$. Might be OK.
                *   But checking edges is $O(N^2)$ if we iterate segments.
                *   Can we make it $O(1)$?
                *   Store valid rects.
                *   Sort them.
                *   Check if blocked.
                *   Blocked check: for each edge of rect, check `used_edges`.
                *   Number of edges = 4. Constant.
                *   So validation is fast.
            *   Regenerating $O(N^3)$ is the main cost.
            *   $2 \cdot 10^5$ ops per step.
            *   Max steps? If we fill every cell, 3000 steps.
            *   $6 \cdot 10^8$ ops $\approx 0.6$ seconds.
            *   With 2000 test cases?
            *   Wait, system test is TOTAL score.
            *   The submission runs ONCE against all test cases? No.
            *   "System test contains 125 inputs... The final ranking...".
            *   "The system test will be performed only for the last submission...".
            *   This means the judge runs your code multiple times.
            *   But the constraint says "Time limit may vary... submissions very close to TLE may result in TLE".
            *   It implies a single execution has a time limit (likely 2-4s).
            *   Wait, usually for 2000 test cases, the judge processes them.
            *   Is it one run processing all inputs?
            *   "Input is given... Output...". This implies one execution per test case file?
            *   Usually AtCoder judges run your program once per test case file (if single file) or batch.
            *   "Provisional test: 50... System test: 2000."
            *   "If your submission produces illegal output... for some test cases...".
            *   So the program is executed separately for each test case.
            *   So time limit applies to EACH run.
            *   So we have plenty of time (e.g. 2-4s) for one run.
            *   $6 \cdot 10^8$ ops might be tight for 2s.
            *   C++ can do $\approx 10^8$ ops per second comfortably, maybe $10^9$ in peak.
            *   $6 \cdot 10^8$ is risky.
            *   We need to optimize regenerating candidates.
            *   Avoid re-scanning all rects.
            *   Only check affected rects.
            *   When dot at $P$ is added:
                *   Rectangles touching $P$ might change validity.
                *   Rectangles NOT touching $P$ stay same (regarding availability of dots).
                *   However, edges change.
                *   Edges drawn might block rects not touching $P$?
                *   Rectangles are blocked if *any* segment of perimeter matches an existing edge.
                *   The edges we draw are perimeter of the *new* rectangle.
                *   So any rectangle overlapping with these edges is blocked.
                *   Since edges are on specific grid lines (AA) or slant lines (Diag).
                *   Overlap check is easy.
            *   So we can maintain a set of active rectangles.
            *   When $P$ added:
                *   Find all rectangles that *include* $P$ as a vertex.
                    *   How to find efficiently?
                    *   We can iterate neighbors of $P$.
                    *   Check all $N^2$ potential partners? No.
                    *   Use adjacency lists `adj[x][y]` storing IDs of rectangles using this point.
                    *   If we have $ID$ of rects.
                *   For each rect containing $P$:
                    *   If it now has 4 dots (was 3), it's no longer a candidate for "new dot".
                    *   If it had <3 dots, now maybe 3?
                        *   Wait, we only add dot at $P$.
                        *   So if rect needed $P$ (empty), now it has dot.
                        *   Check other 3 dots. If all present, it becomes a valid rectangle?
                        *   Wait, we need a rectangle with *exactly* 3 dots (1 empty) to be a candidate for generating a NEW dot.
                        *   If it had 2 dots before, adding $P$ makes 3 dots. If remaining empty corners = 1, we have a valid candidate (new dot is one of the empty ones).
                        *   Wait, if it becomes fully full (4 dots), it's not a candidate.
                        *   So we check if it's valid now.
                *   Check for new candidates created by $P$.
                    *   Iterate $Q \in \text{Neighbors}(P)$? No.
                    *   We need to check if any rect was "blocked" by $P$ previously?
                    *   Actually, simpler:
                        *   Just keep a list of all currently valid rectangles (perimeter empty, 3 dots).
                        *   Whenever we fill $P$:
                            1.  Remove any rect where $P$ was the empty corner. (Now fully full, removed).
                            2.  Add $P$ to any rect where it was missing (but now we only care about rect with 3 dots).
                                *   Actually, if $P$ becomes a dot, we might create a situation where a rect has 3 dots (including $P$) and 1 empty.
                                *   If it had 2 dots, $P$ is added. Now check if 3rd dot exists.
                                *   Wait, if it has 2 dots, and $P$ becomes 3rd. Where is 4th?
                                *   If 4th is empty, then we have a valid candidate!
                            3.  Check for edge conflicts.
                                *   The edges we drew (perimeter of $P$'s rect) are now blocked.
                                *   Any other candidate rect sharing these edges must be discarded.
                                *   We can store `rect_id` list for each edge?
                                *   Or just store the list of candidate rectangles and re-filter.
                                *   Filtering cost: Iterate all candidates.
                                *   Number of candidates?
                                *   If grid is empty-ish, candidates $\approx N^2$.
                                *   If grid full, candidates $\approx 0$.
                                *   Average maybe small.
                                *   Let's store candidates in a `vector`.
                                *   Re-sort/re-filter is costly.
                                *   But candidates with blocked edges are removed.
                                *   We can iterate candidates, check validity.
                                *   If invalid, remove.
                                *   If valid, keep.
                                *   This is efficient enough.
            
            4.  **Data Structures:**
                *   `rect_list`: `vector<Rect>`.
                *   `dot_grid`: `bool[N][N]`.
                *   `edge_map`: `map<pair<Point, Point>, bool>`.
                *   `candidates`: filtered subset of `rect_list`.
                *   When updating:
                    1.  Scan `rect_list`.
                    2.  If rect has 4 dots, remove.
                    3.  If rect has $\le 2$ dots, maybe still irrelevant (we need 3).
                    4.  If rect has 3 dots (1 empty), it is a candidate.
                        *   Check perimeter empty.
                        *   Check edge conflicts.
                        *   If valid, add to `candidates`.
                    5.  Sort `candidates` by weight of empty corner.
                    6.  Pick best.
                *   Wait, scanning `rect_list` every time is bad if `rect_list` is large.
                *   But we only generate rects using *existing* dots.
                *   The number of rects with 3 dots is bounded.
                *   How many axis-aligned rects can exist? $N^2$ roughly?
                *   Actually, if all cells are dots, all rects are full.
                *   We only care about partially filled.
                *   So we can just iterate all pairs of rows/cols once?
                *   No, the set of rects is static (defined by dot coords).
                *   Wait, "perimeter empty" depends on dot placements.
                *   "edge conflict" depends on edges.
                *   The topology (who is corner of who) is static based on grid.
                *   So the universe of *possible* rectangles is constant?
                *   Yes. A rectangle is a geometric object.
                *   Whether it is valid depends on dot presence and edge usage.
                *   So we can pre-calculate ALL axis-aligned and diagonal squares that fit in the grid?
                *   Yes! There are $O(N^4)$ such geometric rectangles.
                *   We can enumerate them all. Store in a big list.
                *   Check validity dynamically.
                *   Since $N \le 61$, $N^4 \approx 1.3 \times 10^7$.
                *   Enumerating $1.3 \times 10^7$ objects is fine upfront (takes ~100-200ms).
                *   Then each step we check validity against current state.
                *   Checking validity:
                    *   Count dots in 4 corners.
                    *   Check perimeter for dots.
                    *   Check edges in `edge_map`.
                *   Optimization: Store `rect_index` for each dot?
                *   When dot added, update counts.
                *   This seems best.
                *   Algorithm:
                    1.  Generate all geometric rectangles (AA and Diag).
                        *   AA: Loops $x_1, x_2, y_1, y_2$. Check width/height $\ge 1$.
                            *   Wait, $N=61$, loops $\approx 60^4/4 \approx 1.3 \times 10^7$.
                            *   We can iterate efficiently.
                        *   Diag: Loops $x_c, y_c, r$. $r \ge 1$.
                            *   Loop $x_c, y_c, r$. $O(N^3)$.
                            *   Check bounds.
                    2.  Store in `vector<RectInfo>`.
                        *   `RectInfo`: `{int id; Point c1, c2, c3, c4;}`
                    3.  Sort by `weight(empty_corner)`.
                        *   Wait, empty corner changes!
                        *   We can store multiple entries? No.
                        *   Just store the rect and its potential corners.
                        *   Or just rely on runtime sorting.
                        *   Runtime sorting over a set of size $10^7$ is too slow.
                        *   But number of valid candidates is small.
                        *   We need to pick the best valid move.
                        *   So we can just scan the list for best valid move? $10^7 \times K$ is TLE.
                    4.  Backtrack:
                        *   We can group rects by their potential "new corner".
                        *   Map `Point -> vector<int> rect_ids`.
                        *   We only care about moves for *empty* points.
                        *   So we iterate empty points.
                        *   For each empty point, look at `rect_ids[P]`.
                        *   Check if they are valid.
                        *   Pick best.
                        *   This brings back $O(N^4)$ per step?
                        *   Number of empty points $\approx N^2$.
                        *   Each point has few rects (avg $O(N^2)$).
                        *   So $O(N^4)$.
                        *   Can we speed up "Check if valid"?
                            *   We need to check dot counts (fast), perimeter (fast), edges (fast).
                            *   The overhead is the lookup.
                        *   Wait, if we just iterate all rects and check validity, we find all valid moves.
                        *   But we want to pick the one with MAX weight of the new corner.
                        *   So we need to find max over all valid rects.
                        *   If we scan all $10^7$ rects, it takes ~100ms.
                        *   If we do this $K$ times ($K \approx 3000$), total $3 \times 10^{10}$ ops. Too slow.
                    5.  Conclusion:
                        *   We need to only scan "relevant" rectangles.
                        *   Relevant rectangles are those involving the *currently empty points*.
                        *   But we don't know which empty point is relevant.
                        *   However, we know the *target* empty point determines the weight.
                        *   So we iterate empty points $P$ in decreasing order of weight.
                        *   For first $P$, we check all rects involving $P$.
                        *   If any valid, pick the one with max $w(P)$ (which is constant) or tie-break?
                        *   Tie-break: minimal perimeter? Or first.
                        *   Wait, do we need to check *all* rects involving $P$?
                        *   Yes, to pick the best one (maybe one uses more edges which are less restrictive? or vice versa).
                        *   How many rects involve $P$?
                            *   For AA: $P$ is corner. Depends on other corners.
                            *   Number of rects is $\approx (\text{# dots in row}) \times (\text{# dots in col})$.
                            *   Average $\approx (N^2/M) \times (N^2/M)$? No.
                            *   Avg is $N$.
                            *   So $\approx N^2$ rects.
                            *   $N^2 = 3600$.
                            *   So for each empty point, checking $\approx 3600$ rects is OK.
                            *   But number of empty points is large initially.
                            *   If we check ALL empty points until we find a valid move.
                            *   In early game, most empty points are far from dots.
                            *   If $P$ is far from dots, it has 0 valid rects.
                            *   So we skip it.
                            *   But skipping involves checking neighbors?
                            *   Or just iterating valid rects?
                            *   Actually, we can optimize:
                                *   Maintain adjacency `adj[P]` list of rects involving $P$.
                                *   When we determine $P$ is empty, look at `adj[P]`.
                                *   Check validity.
                                *   If valid, pick.
                                *   If not, continue to next empty $P$ (or update $P$ status?).
                                *   Wait, if $P$ has valid rects, we take them.
                                *   Do we take *only* the first valid one?
                                *   If we find a valid one for $P$, should we check if another valid one for $P$ is better? (Tie break).
                                *   If valid, pick best.
                                *   Then update grid, remove $P$ from empty.
                                *   Next iteration, $P$ is filled. `adj` lists updated.
                        *   So we iterate *all empty points*?
                        *   Wait, if $P$ has NO valid rect, we check it.
                        *   Checking if $P$ has valid rect involves checking `adj[P]`.
                        *   Size of `adj[P]` is $O(N^2)$ worst case.
                        *   If we have $N^2$ empty points, total $O(N^4)$.
                        *   But `adj[P]` for empty points usually small or zero if isolated.
                        *   If $P$ is isolated, `adj[P]` is empty or small.
                        *   If $P$ is dense (center), `adj[P]` is large. But center points are not empty (filled with initial dots).
                        *   So empty points are on boundary/periphery.
                        *   Periphery has fewer dots nearby.
                        *   So `adj[P]` should be small.
                        *   So this optimization is sound.
                        
                    6.  **Final Plan:**
                        *   Precompute all geometric rectangles.
                            *   AA: Loop $x1, x2, y1, y2$. (Filter: valid grid).
                            *   Diag: Loop $xc, yc, r$. (Filter: valid grid).
                        *   Assign ID to each rect.
                        *   Build `adj[id]` = list of 4 corner points? No, `adj[pt]` = list of rect IDs where pt is a corner.
                            *   We can do this once.
                            *   $O(N^4)$ setup. Acceptable.
                            *   Wait, constructing `adj[pt]` for all $N^2$ pts.
                            *   Rect count $\approx 2 \cdot 10^6$.
                            *   Total entries in `adj` = $4 \times 2 \cdot 10^6 \approx 8 \cdot 10^6$.
                            *   Memory $\approx 30-40$ MB. Fine.
                        *   Loop:
                            *   Find empty points. Sort by weight desc.
                            *   Iterate sorted empty points.
                            *   For current $P$:
                                *   Get `candidates = adj[P]`.
                                *   Check validity of each candidate.
                                *   If valid, select best one (pick any? or based on tie breaker like min perimeter).
                                *   If no valid candidates for $P$, continue to next $P$.
                                *   If found a candidate, execute move.
                                *   Update:
                                    *   Mark $P$ as filled.
                                    *   Add edges to `used_edges`.
                                    *   The move we picked removes $P$ from the set of empty points.
                                    *   Wait, picking a move for $P$ makes $P$ filled.
                                    *   Does it enable other moves? Yes.
                                    *   Should we re-evaluate?
                                    *   We can just continue the loop.
                                    *   But we processed $P$. If $P$ is filled, we don't need to look at it again.
                                    *   The loop continues with remaining empty points (or same list if we didn't remove it? No, remove it).
                                    *   But wait. Picking a move for $P$ consumes edges.
                                    *   This might invalidate candidates for *other* empty points $Q$ (with $w(Q) \le w(P)$).
                                    *   If we invalidated $Q$'s candidate, we shouldn't waste time trying to pick it again?
                                    *   Actually, we are scanning empty points.
                                    *   If we picked a move for $P$, we stop and go to next step?
                                    *   Or do we continue searching for *another* move in the *same* step?
                                    *   Problem says: "repeat operations". We perform ONE op, then next.
                                    *   So after picking move for $P$, we restart the search for next op.
                                    *   Restarting search means re-evaluating empty points.
                                    *   Since $N^2$ empty points check is costly if done naively.
                                    *   Optimization: We maintain a list of "promising" empty points.
                                    *   Or just realize that $P$ is processed.
                                    *   Wait, if we pick $P$, we use up $P$.
                                    *   The next best move could be for some $Q$ (lower weight).
                                    *   We should scan again.
                                    *   But do we rescan *all* empty points?
                                    *   Yes, because blocking edges might have made some $Q$ invalid.
                                    *   But $Q$ might become valid (if $P$ was needed).
                                    *   So we need to check `adj[Q]`.
                                    *   To avoid rescanning all $N^2$ empty points, notice that usually only a few empty points have `adj[Q]` not empty.
                                    *   Most empty points are isolated.
                                    *   So we can maintain a list of "non-isolated" empty points.
                                    *   When a move is made, update list.
                                    *   When an empty point $Q$'s `adj` list becomes non-empty (due to a new dot enabling a rect), add to list.
                                    *   Initially, check all empty points. If `adj[P]` not empty, add to `available_targets`.
                                    *   Sort `available_targets` by weight? No, re-sort or maintain heap.
                                    *   Pop best $P$.
                                    *   Check `adj[P]`. If valid, execute.
                                    *   If not valid (edges blocked or something), maybe check again?
                                    *   If not valid, can we just discard $P$?
                                    *   Wait, if `adj[P]` had a valid move, we took it.
                                    *   If `adj[P]` became empty (due to perimeter dot placement? No, adding dot fills perimeter).
                                    *   Or `adj[P]` became invalid due to edge blocking?
                                    *   If `adj[P]` became invalid, we should remove $P$.
                                    *   If `adj[P]` had candidates but we couldn't pick any?
                                    *   Then we just discard $P$ (since we can't make a move at $P$).
                                    *   So, strategy:
                                        *   Maintain heap of `available_empty_points` (indices of dots).
                                        *   Populate heap with all empty points $P$ where `adj[P]` is not empty.
                                        *   Loop:
                                            *   If heap empty, terminate.
                                            *   Peek/Pop $P$ (best weight).
                                            *   Check `adj[P]` for validity.
                                            *   If valid candidates exist:
                                                *   Pick one.
                                                *   Execute (fill $P$, add edges).
                                                *   Remove $P$ from grid (it's no longer empty).
                                                *   Update state.
                                                *   Continue loop (replenish heap).
                                            *   Else (no valid candidates):
                                                *   Is $P$ permanently dead?
                                                *   Yes, because $P$ needs 3 existing dots. If they are there now but blocked by edge, they might be unfilled.
                                                *   If edge is blocked, it's permanent.
                                                *   So discard $P$ from heap.
                                                *   Wait, could $P$ become valid later?
                                                *   If edge blocking $P$'s rectangle is removed?
                                                *   Edges are only ADDED. Never removed.
                                                *   So if a move becomes invalid due to edge conflict, it stays invalid forever.
                                                *   However, a move might be valid initially, we skip it because we checked lower weight ones?
                                                *   No, we process in decreasing weight.
                                                *   If we check $P$ and find NO valid candidates:
                                                    *   Either `adj[P]` was empty (initially).
                                                    *   Or `adj[P]` was populated but all candidates invalid (blocked).
                                                    *   In either case, we can never play at $P$ in future (since we only add edges).
                                                    *   So remove $P$ permanently.
                                                *   So logic holds.
                
    *   **Refinement on Initialization:**
        *   Generate `rect_list`.
        *   Compute `weight(empty_corner)` for each rect.
        *   Wait, `empty_corner` varies.
        *   For each rect $R$, it has 4 corners.
        *   If 4 dots initially, valid? No (need empty).
        *   If $\le 2$ dots, no empty corner (needs 3).
        *   If 3 dots, we have 1 empty corner. That is a candidate move.
        *   We can associate this rect with the empty corner $P$.
        *   Add $P$ to `adj[R]`? No, `adj[P].push_back(R)`.
        *   This builds the graph of dependencies.
        *   If $P$ is empty and `adj[P]` has entries:
            *   For each $R \in adj[P]$, check if 3 corners are dots.
                *   If yes (they must be), check perimeter empty.
                *   If valid, we have a candidate.
            *   We want the one with max $w(P)$.
            *   Actually, all rects in `adj[P]` have $P$ as the target empty point.
            *   Wait, $P$ could be a corner in multiple roles?
                *   If rect has 3 dots, and $P$ is the 4th.
                *   Is it possible for $P$ to be one of the 3 dots?
                *   No, we iterate empty points. $P$ is empty.
                *   So $P$ is *always* the target.
            *   Wait, what if rect $R$ has 3 dots $D_1, D_2, D_3$ and empty $P_1, P_2$ (2 empty)?
            *   Then $R$ is not a candidate yet. We need to fill one of them.
            *   But if $R$ has 2 empty, we can't use it directly to place a dot at $P_1$.
            *   We need a rect with exactly 1 empty.
            *   So we only consider rects with exactly 1 empty corner.
            *   When we have 3 dots, 1 empty.
            *   Wait, initially $M$ dots.
            *   Many rects might have 0, 1, 2, 3, 4 dots.
            *   Those with 1 empty are our potential moves.
            *   Those with 0, 2, 4 are not moves yet.
            *   But those with 0/2 dots might become 1 empty later.
            *   If they become 1 empty (say at $Q$), we add $Q$ to `adj[Q]` or update.
            *   Wait, tracking "becomes valid" states.
            *   We need to re-scan rects that changed dot count.
            *   When $D$ becomes dot:
                *   For each rect containing $D$:
                    *   Update dot count.
                    *   If count goes from 2 to 3: Now 1 empty.
                        *   Identify empty corner $P$.
                        *   Check if $P$ is already empty. Yes.
                        *   Add $R$ to `candidates[P]`. (Or `adj[P]`).
                        *   Check if $P$ is in `heap`. If not, push.
                    *   If count goes from 3 to 4: Now 0 empty.
                        *   Remove $R$ from `candidates[P]`. (If $P$ was the target).
                    *   If count 1->2: Still not ready.
            *   We need `candidates[P]` to store rects where $P$ is the unique empty corner.
            *   Initially, populate `candidates[P]` for all such rects.
            *   Heap stores $P$.
            *   Priority Queue order: `w(P)`.
            *   Logic:
                *   PQ pop $P$.
                *   Check rects in `candidates[P]`.
                *   Wait, `candidates[P]` might contain rects that became invalid (blocked edges, etc).
                *   We need to filter.
                *   If we filter and find valid rects:
                    *   Pick best.
                    *   Execute.
                *   If no valid rects:
                    *   Discard $P$ (pop).
                    *   Note: `candidates[P]` might be non-empty but all invalid.
                    *   If $P$ is discarded, we don't care anymore.
                    *   Is it possible $P$ reappears?
                        *   No, $P$ is empty but no valid move.
                        *   Edges accumulate. Invalid moves stay invalid.
                        *   So safe to discard.
                *   Wait, we might discard $P$ prematurely if edges didn't cause invalidity but we just didn't check?
                *   No, we check `candidates[P]` completely.
            *   What about rects becoming valid later?
                *   Handled by `Update` step (when dot added).
            *   What about rects losing validity?
                *   Handled when checking `candidates[P]` (we validate on demand).
            *   Wait, if $P$ is popped and we check `candidates[P]`.
                *   If valid exists, pick it.
                *   If not valid, we discard $P$.
                *   Wait, is it possible $P$ was valid but we chose a lower weight point $Q$ earlier?
                    *   No, we prioritize $P$ (max weight).
                    *   If $P$ is valid, we pick it.
                    *   Wait, what if multiple rects in `candidates[P]`?
                    *   We pick one. Does the choice affect future?
                    *   Assuming greedy on $P$ is fine.
            *   Wait, if we discard $P$ because all rects invalid.
                *   Could a future move open a rect for $P$?
                *   No, opening requires *adding* a dot to a missing corner.
                *   But $P$ is the target (4th dot).
                *   To open a rect for $P$, we need to fill 3 other corners.
                *   But `candidates[P]` already contains rects where 3 corners are filled.
                *   If they are valid, $P$ is playable NOW.
                *   If they are INVALID (blocked edges), they stay blocked.
                *   So if `candidates[P]` is empty or all invalid, $P$ is dead.
            *   Wait, what about rects that were NOT in `candidates[P]` but will become valid?
                *   These are rects with <3 dots.
                *   They are not in `candidates` yet.
                *   When we fill the 3rd dot, we add to `candidates[P]`.
                *   So we cover all future potential moves.
            *   Seems correct.

    *   **Complexity check:**
        *   Init: $O(N^4)$.
        *   Updates:
            *   Add dot: iterate `adj[D]`.
                *   How many rects contain $D$? $O(N^2)$.
                *   Total updates per dot: $O(N^2)$.
                *   Total updates for $K$ dots: $O(K N^2)$.
                *   $K \approx 3000$. $3000 \times 3600 \approx 10^7$.
                *   Inside update: check if valid rect (perimeter empty, edge free).
                *   This is fast ($O(1)$ amortized with bitsets or similar).
            *   PQ operations: Logarithmic.
            *   Total Ops $\approx 10^7$.
            *   Extremely safe.
    *   **Corner case:**
        *   $P$ is popped. `candidates[P]` has rects.
        *   Some valid. We pick one.
        *   Wait, if multiple valid, we pick one.
        *   Is it possible that picking one blocks another for $P$?
        *   Yes.
        *   Should we check all?
        *   If `candidates[P]` has multiple valid rects, we should probably pick one that minimizes impact?
        *   Or just pick the one that appears first?
        *   Given the constraint, maybe just pick any.
        *   Wait, if we pick $R_1$, we block edges.
        *   $R_2$ might also be blocked by edges of $R_1$.
        *   So we essentially select a maximal set.
        *   Since we want max score, and all give $w(P)$, maybe we want to maximize number of future moves?
        *   Hard.
        *   Let's trust the simplest: pick any valid.
        *   Wait, if picking one prevents a better score later?
        *   Later scores are $\le w(P)$.
        *   So preventing a future move with score $w(Q) \le w(P)$ is a trade-off.
        *   We gained $w(P)$. We lost $w(Q)$.
        *   If $w(P)$ is significantly larger, it's worth it.
        *   Here all $P$ have different weights (mostly).
        *   So we prioritize current $P$.
        *   Between choices for $P$:
            *   Choice A blocks edges $E_A$. Choice B blocks $E_B$.
            *   We should choose the one that blocks *least valuable* edges?
            *   Edges with potential to support high weight $Q$'s.
            *   Hard to predict.
            *   Usually random/greedy works.

    *   **Implementation details:**
        *   $N \le 61$.
        *   `Point` struct.
        *   `Rect` struct: 4 corners.
        *   `adj` vector `vector<vector<int>>`.
        *   `candidates` is not stored explicitly for each point?
        *   Actually `adj[P]` stores rects where $P$ is a corner.
        *   When we check $P$, we look at `adj[P]`.
        *   Filter: `rect` has $P$ as the unique empty corner.
        *   This requires tracking "how many dots in rect".
        *   We can compute this on the fly: `count_dots(rect)`.
        *   If count is 3, check if $P$ is the empty one (check if other 3 are dots).
        *   Wait, if count is 3, and $P$ is a corner.
        *   If $P$ is empty, then other 3 must be dots.
        *   So if `count_dots(rect) == 3 && !grid[P]`, then $P$ is the target.
        *   Wait, what if rect has 3 dots, but $P$ is NOT empty (contradiction)?
        *   No, if `!grid[P]`, it IS empty.
        *   So simply: `adj[P]` contains rects where $P$ is corner.
        *   Filter for `count_dots(rect) == 3`.
        *   Check validity.
        *   If `count_dots(rect) != 3`, it is not a candidate for $P$ yet.
            *   It might become candidate for $P$ later if it gets to 3 dots and $P$ is empty.
            *   But since we iterate $P$ from heap (only those that became valid), we don't need to check rects that aren't valid yet.
            *   Actually, when adding dot $D$, we might convert a rect from 2 dots to 3 dots.
            *   Then it becomes a candidate for the remaining empty corner.
            *   That empty corner might be $P$.
            *   So we need to notify `adj[P]`.
            *   This means maintaining "status" of rects?
            *   Or just lazy evaluation.
            *   When processing $P$, iterate `adj[P]`. Check if valid.
            *   If valid, good.
            *   If not, maybe it was waiting for dots?
            *   If `count_dots(rect) < 3`, it's not ready.
            *   We skip it?
            *   Wait, we want to find *at least one* valid move.
            *   If all rects in `adj[P]` are incomplete (count < 3), then we cannot move at $P$.
            *   So $P$ is effectively dead until its neighbors are filled.
            *   Since $P$ is the highest weight empty point.
            *   If we can't move at $P$, we pop it and it's gone (because no future moves will enable it unless we fill neighbors).
            *   Wait, filling neighbors might happen later!
            *   If we pop $P$ because no move found, we might fill a neighbor $Q$ in the future (which has lower weight).
            *   This enables $P$.
            *   So we should NOT discard $P$ immediately?
            *   Or, we should only discard $P$ if *all* rects requiring $P$ are blocked by edges or incomplete but can NEVER be completed?
            *   No, incomplete rects can be completed.
            *   So $P$ should remain in heap?
            *   But heap logic: we want max weight.
            *   If $P$ (weight 100) has no move now.
            *   Next best $Q$ (weight 90) has a move.
            *   We play $Q$. $Q$ is filled.
            *   This might enable $P$.
            *   So we need to add $P$ back to heap when enabled?
            *   Or just re-add $P$ whenever its neighbors change?
            *   This sounds like standard "lazy propagation".
            *   If we don't play at $P$ now, we mark it as "pending"?
            *   Actually, if we play $Q$, $Q$ becomes a dot.
            *   This triggers updates for rects involving $Q$.
            *   If a rect becomes 3-dot, we identify target $P$.
            *   If $P$ was pending, we might add it?
            *   Yes.
            *   So we maintain a set of "empty points".
            *   We extract best.
            *   If we find a move, we play it.
            *   If we DON'T find a move (all rects incomplete/blocked).
            *   If rects incomplete -> we might be able to play later.
            *   If rects blocked -> we never can.
            *   Distinguish?
            *   Check if `count_dots < 3`. If so, it's incomplete.
            *   We can't decide if we should play later without filling neighbors.
            *   So if incomplete, we just skip $P$ for now?
            *   Wait, if we skip $P$ (remove from PQ), we hope to add it back.
            *   But we can't easily add back without scanning.
            *   Better: Keep $P$ in PQ. But PQ pops only if it's top.
            *   If we pop $P$ and find no move (because incomplete), we can't discard it immediately if it might be re-enabled.
            *   BUT, if we play $Q$ (lower weight) to enable $P$, $P$ gets moved up.
            *   So we should NOT discard $P$ if it was incomplete.
            *   But if it was incomplete, why didn't we find a move?
            *   Because no rect was 3-complete.
            *   So we effectively wasted time scanning `adj[P]` (finding nothing).
            *   This suggests we should only consider `P` if it has *at least one* potential 3-complete rect.
            *   Wait, how do we know if there is one?
            *   We check `adj[P]`. If all are incomplete, $P$ has no valid moves *now*.
            *   Should we discard?
            *   If we discard, we lose opportunity.
            *   But notice: $P$ is empty.
            *   For $P$ to be filled, we need a rect with $P$ as 4th corner.
            *   Such a rect exists in `adj[P]`.
            *   If all are incomplete, it means no neighbors exist.
            *   If neighbors are not dots, they will become dots via other moves.
            *   Those other moves will come from `candidates[neighbor]`.
            *   So eventually neighbors become dots.
            *   So `adj[P]` might become valid.
            *   This implies we should treat incomplete $P$ as "potentially valid later".
            *   But how to manage PQ?
            *   If we play a move $Q$, we re-evaluate.
            *   Maybe we can just NOT remove $P$ from `candidates` list, but just ignore it in PQ?
            *   Or add it to PQ again?
            *   Actually, notice that if $P$ is incomplete, we can't play it.
            *   But we also can't prioritize it over $Q$.
            *   So, if $P$ has no valid moves (incomplete or blocked), we can't pick it.
            *   We pick best available from rest.
            *   So, effectively, we maintain a list of all empty points.
            *   In each step:
                *   Find subset of empty points $S$ where at least one rect is 3-complete.
                *   From $S$, pick max weight.
                *   If $S$ is empty, terminate? No, might become non-empty later.
                *   Wait, if $S$ empty, it means NO empty point has a 3-complete rectangle.
                *   This implies we are stuck?
                *   Yes, if we can't find a move anywhere, we are stuck.
                *   But we assumed $P$ is incomplete, not blocked.
                *   If $P$ is incomplete, we need to fill its neighbors.
                *   If NO point has a 3-complete rectangle, then NO point can be filled next.
                *   So game over.
                *   Therefore, if we scan and find no 3-complete rects for *any* empty point, we are done.
                *   Wait, do we scan ALL empty points?
                *   If we scan all, we spend $O(N^2)$ time.
                *   If we do this frequently, it's slow.
                *   But if no 3-complete rects, we stop.
                *   So we only stop when no moves left.
                *   If moves exist, we pick best.
                *   The bottleneck is finding "moves exist".
                *   We can maintain a counter `total_pending_moves`.
                *   If `total_pending_moves > 0`, pick best.
                *   When a dot added: update `total_pending_moves`.
                *   This avoids scanning empty points.
                *   We just need to track which empty points are "active" (have at least one 3-complete rect).
                *   Let `active_targets` be a priority queue of empty points that have $\ge 1$ valid candidate rect.
                *   Wait, "valid candidate" includes "perimeter empty".
                *   This is dynamic.
                *   Simplification: "Potential candidate" (3 dots, 1 empty).
                *   Let's track `active` = set of empty points with $\ge 1$ potential candidate.
                *   Sort `active` by weight.
                *   Step:
                    *   Pick top $P$ from `active`.
                    *   Check validity of ALL candidates for $P$.
                    *   If any valid (perimeter/edge ok), pick it and proceed.
                    *   If none valid (either blocked or incomplete - wait, active implies at least one incomplete candidate? No, active means *exists* 3-complete candidate).
                    *   If it is 3-complete, check perimeter/edge.
                    *   If blocked, remove it from consideration?
                    *   If blocked, can it be unblocked? No.
                    *   So if all candidates for $P$ are blocked, remove $P$ from `active`.
                    *   If some valid, pick one.
                *   But wait, if $P$ has 3-complete candidate $R$. $R$ is blocked.
                *   Then $P$ has NO valid moves.
                *   We must remove $P$.
                *   If $P$ has NO 3-complete candidates (e.g. neighbors missing), then $P$ is not in `active`.
                *   When neighbors filled, $P$ enters `active`.
                *   Wait, how to efficiently detect "neighbors filled"?
                *   When dot $D$ added:
                    *   Iterate rects $R$ containing $D$.
                    *   If $R$ transitions to 3-complete (was 2).
                    *   Identify empty corner $P_{new}$.
                    *   If $P_{new}$ is empty.
                    *   Add $P_{new}$ to `active`.
                    *   Also, if $R$ transitions to 4-complete, ignore.
                *   This handles "neighbors filled".
                *   What about "blocked"?
                    *   If $P$ was in `active`, but its candidates became blocked (due to edges).
                    *   Wait, if we pick a move for $Q$, we draw edges.
                    *   These edges might block $P$'s candidates.
                    *   We can lazily check validity.
                    *   If $P$ popped from `active`, check all candidates.
                    *   If valid found, good.
                    *   If not valid (and blocked), remove $P$.
                    *   Note: If $P$ is not in `active`, it waits.
                    *   So this covers all cases.
                *   One detail: When adding dot $D$, we also invalidate edges for existing candidates.
                    *   Wait, if $R$ was 3-complete and valid, and now blocked by new edges?
                    *   The edges are drawn by the NEW move.
                    *   So we must invalidate candidates for ALL active points.
                    *   If we invalidate many candidates, checking them is costly.
                    *   Better: When edges drawn, check candidates.
                    *   But candidates are associated with `active` points.
                    *   We can mark $P$ as "checked".
                    *   If $P$ remains in `active` but all its candidates are blocked, remove it.
                    *   How to know all are blocked?
                    *   We can iterate `candidates[P]` when popping $P$.
                    *   If all invalid, pop and remove $P$.
                    *   If valid, process.
                    *   What if $P$'s candidates were invalidated *before* $P$ was popped?
                    *   Then when $P$ is popped, it sees them invalid.
                    *   Correct.
                    *   But how to update `active` efficiently when edges are drawn?
                    *   Drawing edges invalidates specific candidates.
                    *   We can't update `active` easily without scanning.
                    *   BUT: `active` points are few.
                    *   Maybe just re-pull $P$ if needed?
                    *   Actually, if we draw edges, they don't automatically make `active` invalid.
                    *   We just check validity when popping.
                    *   If $P$ was added to `active` due to some $R$ becoming 3-complete.
                    *   Then $R$ is 3-complete.
                    *   We check $R$ when popping $P$.
                    *   If $R$ is blocked, we remove $P$.
                    *   This works!
                    *   We only need to ensure `active` contains *potential* candidates (3 dots).
                    *   When popping $P$, we validate.
                    *   If valid found: execute.
                    *   If not valid (blocked): remove $P$ (since blocked forever).
                    *   Is it possible $P$ is not valid now but becomes valid?
                    *   No, validity requires empty perimeter/edges.
                    *   Blocks are cumulative.
                    *   Peremptory holes (empty perimeter) can never increase (dots only added).
                    *   So if blocked, always blocked.
                    *   Wait, if blocked by edge?
                    *   Edge is drawn. Edge blocks forever.
                    *   So yes.
                    *   Is it possible $P$ has no valid move because `adj[P]` contains only incomplete rects?
                    *   Then $P$ is not in `active`.
                    *   We wait until neighbors filled.
                    *   Correct.
    
    *   **Algorithm Recap:**
        1.  Generate all rects.
        2.  `vector<int> adj[MAXN*N]`. For each rect $R$, store in `adj[corners[i]]`.
        3.  `set<int> active_pts`. Stores index of empty points that have at least one `R` where `dots(R)==3`.
            *   Wait, `adj` stores all corners.
            *   So if $R$ has 3 dots, let empty be $P$. Add $P$ to `active`.
            *   Wait, multiple $R$ might point to same $P$.
            *   Just maintain a boolean flag `has_candidate[P]`.
        4.  Priority Queue `pq` of `Pair(weight, index)`.
            *   Only push `index` if `has_candidate[index]` becomes true.
            *   Since weights static, `weight` is computed.
        5.  Loop:
            *   While `!pq.empty()`:
                *   Pop $(P_{idx}, P_{val})$.
                *   Check candidates in `adj[P_{idx}]`.
                    *   Filter: `dots(R) == 3`.
                    *   Check validity (empty perimeter, no edge conflict).
                    *   If any valid:
                        *   Pick one (e.g. first valid).
                        *   Execute move.
                        *   Break loop (continue to next step).
                    *   If no valid:
                        *   $
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=1252)

````
Let me analyze this problem:

1. We're playing a game on an N×N grid
2. Initially M dots are placed at random positions in the center region (floor(N/4) to floor(3N/4))
3. We can repeatedly create rectangles parallel to axes or 45° angled, where:
   - The rectangle must have exactly 3 corners with existing dots and 1 corner without
   - No other dots on the perimeter (except the 3 corner dots)
   - Perimeter doesn't share segments with already drawn rectangles
4. When we make such a move, we place a dot at the empty corner and draw the rectangle
5. Score = round(10^6 × (N²/M) × (sum of weights of points with dots) / S), where weight(x,y) = (x-c)² + (y-c)² + 1, c = (N-1)/2

The key insight is that we want to maximize the weighted sum while spending as few moves as possible (since it's divided by M). Weight favors points near the edges/corners since they're furthest from the center.

Strategy approach:
1. Try different rectangle sizes systematically
2. For each size, try all orientations (axis-aligned and 45°)
3. Greedily pick the best available rectangle at each step
4. Restart when getting stuck (no valid rectangles found)

For axis-aligned rectangles of size s×s, the pattern is predictable based on which side has an existing dot. Let me implement a checker and generator for both cases.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int N, M;
bool dot[8][8];
int x0, y0, sx, sy;
vector<tuple<int,int,int,int,int,int>> ops;
vector<vector<tuple<int,int,int,int,int,int>>> history[N*N];

void reset() {
    memset(dot, 0, sizeof(dot));
    ops.clear();
    history.clear();
    for(int i=0;i<N*N+1;i++) history[i].clear();
}

struct Rect { int x1,y1,x2,y2; };

Rect find_rect(bool check_inside, int px,int py, int sx,int sy) {
    Rect R = {-1,-1,-1,-1};
    int d = max(sx,sy);
    vector<pair<int,int>> pts{make_pair(px,py)};
    
    if(!dot[sx*(py-d)+sy] || !check_inside && sx*(py-d)+sy >= N) return R;
    pts.push_back(make_pair(sx*(py-d)+sy+(sx-1),(px-d)+sy));
    int px4=pts.back().first+(d-sy);
    pts.push_back(make_pair(px4,pts.back().second));
    pts.push_back(pts[2]);
    pts.push_back(pts[0]);
    
    bool ok=true;
    int p1=pts[0].first, p2=pts[1].first, p3=pts[2].first, p4=pts[3].first;
    int q1=pts[0].second, q2=pts[1].second, q3=pts[2].second, q4=pts[3].second;
    if(p1<p2) swap(p1,p2); if(q1<q2) swap(q1,q2);
    if(p2<p3) swap(p2,p3); if(q2<q3) swap(q2,q3);
    if(p3<p4) swap(p3,p4); if(q3<q4) swap(q3,q4);
    
    for(int k=0;k<4*3;k+=3) {
        bool b=false;
        for(int i=k;i<k+3&&b;i++) b |= dot[p[i]]*(i!=3);
        if(b) continue;
        for(int j=0;j<4;j++) for(int t=1;t<=d;t++) {
            if(t==d && !check_inside && (p[j]>N-1||q[j+t]<0)) continue;
            double v=(p[j]-t+p[(j+1)%4])*(q[j]+q[(j+1)%4])-p[j]*(q[(j+1)%4]+q[j]-t)+(q[j]-t+q[(j+1)%4])*p[(j+1)%4];
            if(fabs(v)>0.001) { b=!b; break; }
        }
        if(!b) return {(p1-t,q1),(p2,(q2)-(t>0?t:0))};
    }
    return R;
}

int main() {
    cin >> N >> M;
    int c=(N-1)/2;
    for(int i=0;i<M;i++) { int x,y; cin>>x>>y; dot[x*y+(N-x-1)]=(c>=x&&c+y<=M); }
    x0=c;y0=c;sx=sy=N/2;
    
    double sum_w=S=0;
    long long n_sq=M*=N,N=N*N;
    for(int i=0;i<N;i++) for(int j=0;j<N;j++) {
        long long w=(long long)(i+c-i)*(i+c-i)+(long long)(j+c-j)*(j+c-j)+1;
        sum_w+=w*S+=w;if(dot[i*j+(N-i-1)])n_sq++;
    }
    cout<<"round("<<1000000.0*n_sq/S<<")"<<"\n";
    return 0;
}
```
````
