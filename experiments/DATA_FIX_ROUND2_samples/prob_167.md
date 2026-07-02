# 题 167 — base均分 1.7 vs 直接SFT均分 0.0(掉点 1.7)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 8.6,completion_tokens=27516)

````
The user wants a C++ solution for a competitive programming problem.
The problem asks to construct an orthogonal polygon (edges parallel to axes) that maximizes $a - b$, where $a$ is the number of mackerels inside and $b$ is the number of sardines inside. There are $N=5000$ of each type. The score is $\max(0, a - b + 1)$.

Constraints:
- Polygon vertices $m \le 1000$.
- Edge length sum $\le 4 \times 10^5$.
- Coordinates integers $0 \le x, y \le 10^5$.
- No self-intersection.
- Points on edges are considered inside.

This looks like a specific type of contest problem where you need to find *one* good configuration. The test cases are generated randomly with clusters.
Wait, usually in these types of "AtCoder Heuristic Contest" (AHC) style problems or similar open-ended optimization problems, simple heuristics often work better than complex algorithms because the constraints allow for a lot of freedom but the geometry is restricted.
However, this looks like a specific problem statement from AHC039 (AtCoder Heuristic Contest). Let's check the problem constraints again.
N = 5000. Time limit isn't explicitly stated here but typical is 2s. Memory is standard.
We have 150 test cases. This implies efficiency matters somewhat, but since N is fixed at 5000 for all, maybe we can afford O(N log N) or even O(N) per test case if the constant is small. But the goal is to maximize points across 150 runs? No, "If more than one participant gets the same score..." implies a contest system. Wait, the prompt says "There are 150 test cases". Usually, in AHC, you run your code once against many test files.
Actually, looking at the scoring: "score of a submission is the total score for each test case." So I get one shot to solve it? Or do I submit repeatedly? It's likely an auto-judge for a contest setting. The code must run within time limits. Since there are 150 test cases, if I run a slow simulation per test case, I might exceed TLE.
However, N=5000 is moderate.
Let's analyze the objective function: Maximize (Mackerels Inside) - (Sardines Inside) + 1.
Basically, we want to enclose regions that are rich in Mackerels and poor in Sardines.
Since the coordinates are up to $10^5$, the grid is large. But N is only 5000 points.
An orthogonal polygon can be constructed by drawing a long horizontal line and a long vertical line, making a shape.
Actually, the simplest shapes are rectangles. But the polygon doesn't have to be convex or simple. It just needs to be non-self-intersecting and orthogonal.
Since points on edges count as inside, boundaries matter.
But usually, for "enclosing points", the Voronoi diagram or clustering properties come into play. However, the constraint on edge length ($4 \times 10^5$) and vertices (1000) suggests we can draw a reasonably complex shape, but not cover the whole plane.
Actually, the maximum possible perimeter is bounded, but area isn't bounded directly, though it's limited by coordinates.
Key observation: If we just pick a very thin rectangle around a cluster of mackerels that has no sardines, we get high score.
Given the random generation, mackerels are clustered. Sardines are also clustered.
Ideally, we want to find a path (or boundary) that surrounds high-density mackerel areas while avoiding sardine areas.
Since $N=5000$, sorting points takes little time.
Maybe we can iterate through some candidate bounding boxes.
However, finding the optimal boundary is NP-hard in general (maximum weight simple polygon). But with orthogonal constraints and specific generation method, maybe we can just output a specific shape.
The problem statement mentions "Output vertices... Output multiple solutions... only the last one is used". This is weird for a single submission context unless it's a heuristic contest where you just output one solution and hope it's good enough. But wait, "Your program may output multiple solutions." usually means the judge checks all outputs? No, "only the last one is used for scoring". This suggests we should just output one valid polygon. Wait, "multiple solutions" might refer to generating different polygons if we loop?
Ah, this looks like AHC039 Problem K "Fishing Ground".
In AHC contests, you submit your program and it runs against hidden inputs (test cases).
Since this is a heuristic contest (implied by "Story", "Heuristic Contest" context), exact optimal solutions aren't required, just a good heuristic.
However, for this specific prompt, I need to provide a solution code.
Let's look at the structure of the points. They are generated with Gaussian distributions centered at $(cx_i, cy_i)$.
Mackerels and Sardines are independent sets of such clusters.
Since they share the same generation algorithm (just different random seeds implicitly or sequentially?), they will form similar cluster structures.
Mackerels appear first, then Sardines. The generation uses global randomness.
To maximize (Mackers - Sardines), we want to target a region containing many Mack but few Sards.
Since both are generated with $w_i$ weights and normal distributions, the spatial distribution is the same type (Gaussian blobs).
The most efficient way to enclose points with an orthogonal polygon under length constraints is usually to make the polygon very large (to include many points) OR very local (to exclude others).
Given the coordinate range $10^5$ and max perimeter $4 \cdot 10^5$, the average radius can be $10^5$ (square size) or something like that.
Actually, if we connect all points in a sorted order of X or Y coordinates, we can form a polygon enclosing everything. But that might enclose too many Sardines.
We want to select a subset of points (ideally a cluster) and surround them.
Or, since points on edges count, if we pass between two clusters, we separate them.
Wait, if the polygon is a closed loop, it defines an interior.
The constraint $4 \cdot 10^5$ is generous. $1000$ vertices is also generous.
What is a robust strategy?
1. Identify dense clusters of mackerels.
2. Avoid clusters of sardines.
Since mackerels and sardines are generated independently with the same process, the correlation depends on the relative seed states.
Actually, a very strong heuristic for this problem (often solved in such contests) is to just pick a large rectangular region that balances the counts.
But since we don't know the exact locations until runtime, we must read them.
Strategy:
Calculate density of mackerels vs sardines at various scales.
Or simply, construct a "spiral" or "box" around a promising area.
Given the constraints, we can probably build a "snake" path that visits every mackerel? No, that would take length proportional to N. Distance between points could be large. Summing distances could exceed $4 \cdot 10^5$.
Average distance between points could be high if they are uniform. But they are clustered.
If points are clustered, nearest neighbors are close.
If we sort mackerels by distance from centroid, and try to surround the top $K$ closest ones.
But surrounding them with minimal perimeter is hard. Convex hull is hard for orthogonal.
A simpler idea:
Consider the set of all Mackerel coordinates. Sort them by x-coordinate.
Sort them by y-coordinate.
Construct a bounding box of all mackerels? That might contain sardines.
What if we iterate over all unique X and Y coordinates of the points? There are at most $2N$ such lines.
This forms a grid graph. We can define the polygon by selecting a cycle in this grid graph.
This is equivalent to finding a cycle in a planar bipartite graph?
Given the time limit and complexity, maybe we don't need to be perfect.
Just pick a point (maybe a mackerel) and grow a polygon around it trying to add more mackerels and reject sardines.
Better yet, given the generation mechanism, the clusters are somewhat isolated.
Finding the largest connected component of mackerels separated from sardines by empty space?
Or just a bounding box around the centroid of all mackerels?
Let's consider the constraint: edges parallel to axes.
If we simply output a very large rectangle covering everything, $a=N, b=N$, score $1$. Not great.
If we cover nothing, score 0.
We want $a > b$.
So we want to find a region with $a/b > 1$.
Since both are generated similarly, the ratio of local densities should be roughly 1 globally. Locally it varies.
Some patches might have many mack and few sards.
A "brute force" approach to find such a patch:
Iterate through all pairs of distinct x-coordinates (from input points) and y-coordinates.
Check the rectangle defined by these grid lines. Count m/s. Pick best.
Number of unique X coords is $\le 2N$. Number of Y coords $\le 2N$.
Checking all rectangles is $O(N^3)$ or $O(N^4)$ which is too slow.
We need faster.
We can use a 2D prefix sum array if coordinates were compressed to small values.
Coordinates are up to $10^5$. We can compress them or use a bitset / boolean array since $10^5 \times 10^5$ is too big for memory/initialization.
Wait, $10^5$ is small enough for coordinate compression? Yes, we can map points to ranks.
Max rank is $2N = 10000$.
With coordinate compression, we have a grid of size $10000 \times 10000$. Still too big to allocate.
However, we only care about intervals formed by the points' coordinates.
The number of elementary cells in the arrangement of the points is $O(N^2)$. Too many to enumerate all sub-rectangles.
But maybe we don't need to check all.
Is there a greedy strategy?
Pick a point $P$. Try to find a small rectangle enclosing $P$ with max $a-b$.
Expand rectangle?
Since this is AHC (Heuristic Contest), usually the solutions involve randomized search or local optimization.
However, I am asked to implement a deterministic solution (no "seed" provided in the prompt text except I should assume standard behavior).
Wait, the prompt provides Input Generation details. The judge will run my code with a specific random state? No, usually AHC judges run a generator based on a seed, then feed it to your solver. But the prompt doesn't give me the seed to reproduce the generator locally perfectly (though it describes the logic).
I need to write a solver that works generally.

Let's rethink the constraints. $N=5000$. Vertices $\le 1000$. Edge length $\le 4 \times 10^5$.
Points are integer.
A rectangle has 4 vertices, length $\le 400,000$. A large square of side 100,000 has perimeter 400,000.
So we can definitely fit a bounding box of the entire space.
Since the density of mackerels and sardines is equal on average, we want to find a spot where mackerel density > sardine density.
Given the clustering, the "local" ratio fluctuates.
A simple Monte Carlo approach:
Generate a random center $(X, Y)$. Generate a random radius $R$.
Construct axis-aligned square/rectangle. Check count.
Repeat several times. Keep best.
Given $N=5000$, checking counts inside a rectangle is fast ($O(N)$).
With 150 test cases, we need to be fast. $O(T \cdot (\text{iterations}) \cdot N)$.
If iterations is 1000, $1000 \times 5000 = 5 \times 10^6$, times 150 is $7.5 \times 10^8$. Might be tight for 2s?
Usually 10^8 ops/sec.
Optimization: Coordinate compression + 2D prefix sums.
Map unique x-coords to $1..2N$, y-coords to $1..2N$.
Total points $2N$.
Grid size $2N \times 2N$. Construct prefix sum table `Count[x][y]` for mackerels and `CountS[x][y]` for sardines.
This allows querying any rectangle in $O(1)$ after $O(N^2)$ preprocessing.
Preprocessing takes $O(N^2)$ space?
$2N = 10000$. Grid size $10000 \times 10000$ is $10^8$ ints. $400$ MB.
Memory limit? Typically 256MB or 512MB. 400MB might be too much or risky.
However, $N=5000$ suggests $O(N^2)$ might be acceptable for $N=5000$ if operations are simple. But $O(N^2)$ time per test case?
Wait, if we do prefix sums on compressed coordinates, the grid is $M_x \times M_y$ where $M \le 2N$.
Building the grid takes $O(M^2)$ which is $O(N^2)$.
For $N=5000$, $N^2 = 25 \times 10^6$.
Doing this 150 times is $3.75 \times 10^9$ ops. Too slow.
Wait, do we really need 150 queries? The prompt says "There are 150 test cases, and the score of a submission is the total score for each test case." This implies I submit ONE program. It runs against all 150 cases.
So my program reads one case, solves it, writes output, loops? No, "Input is given from Standard Input". Usually, this means the judge feeds input via stdin.
Does the judge run my binary 150 times? Or feed one file with 150 cases?
Standard format usually implies one execution per test case.
"There are 150 test cases" usually means the platform runs your solution on 150 different inputs and sums scores.
So my program runs ONCE per test case.
Thus, $O(N^2)$ pre-processing per test case is fine! $25 \times 10^6$ ops per case.
Wait, memory? $10000 \times 10000$ int array is 400MB. Limit might be tight.
If memory is tight, we cannot use full grid.
But do we need to check *all* rectangles?
Maybe we can just check a small subset of "interesting" rectangles.
Interesting rectangles are those formed by adjacent grid lines?
Or simply: Randomly sample centers and expand to boundaries.
Or use the property that points are clustered.
We can compute the mean position of Mackerels and Sardines.
Find a region with high difference in potential counts.
Since this is an AHC-like problem, a randomized approach that tries to find a "good" rectangle is expected.
Also, the constraints on vertices and edge length allow non-rectangular shapes, but a rectangle is sufficient to capture "interior".
Wait, the problem says "polygon". Rectangle is a polygon.
Vertices: 4. Edges: 4. Length $\le 4 \times 10^5$.
If we output a rectangle, it satisfies conditions easily as long as coordinates are $\le 10^5$.
Max coordinate is $10^5$. A rectangle with corners $(0,0)$ and $(100000, 100000)$ has perimeter $400000$. Exactly the limit.
Area is huge, likely captures many sardines.
But if we shift it to avoid sardines?
How to find such a spot quickly?
Algorithm:
1. Read points. Store Mack in vector, Sards in vector.
2. If vectors are empty, fail (but N=5000).
3. We want to maximize $CountM(R) - CountS(R)$ subject to valid $R$.
Since we can make any valid rectangle, we just need to pick the best rectangle within bounds.
Since $N$ is small enough, maybe we can iterate over all pairs of mackerels?
No.
Let's use a heuristic.
Consider all unique X coordinates of Mackerels and all unique Y coordinates of Mackerels.
Try to form a bounding box using one Mack and another Mack, but adjusted to avoid Sards.
A simple brute force over "candidate bounding boxes" based on point coordinates:
Candidate X boundaries: For every pair of mackerels $(x_a, x_b)$ with $|x_a - x_b| < \text{threshold}$?
Actually, let's just iterate all combinations of two mackerels and two sardines?
Too many.

Alternative Idea:
The distribution is Gaussian. The highest probability density is at the centers.
Calculate the centroids of all clusters (approximated by points).
But we don't know clusters.
Let's try a grid-based sampling approach without full prefix sums.
Discretize space?
Since $N=5000$, maybe we can check a small number of random rectangles.
Generate 1000 random rectangles?
Random center $(cx, cy)$.
Random width $W, H$.
Check intersection.
Update best.
How to generate centers efficiently?
Maybe pick a mackerel as the target.
Then iterate its neighbors.
Actually, the simplest high-quality solution for this specific problem (AHC039) was likely to output a large rectangle encompassing a specific region found by scanning or simply "best corner".
Looking at online resources for AHC039 (simulated thought):
The problem is often solved by:
1. Identifying "dense" regions of Mackerels.
2. Trying to exclude Sardines.
Since coordinates are integers, we can check rectangles defined by pairs of X and Y coordinates from the input points.
To speed up counting without $O(N^2)$ space/time:
For a rectangle defined by $x_1, x_2, y_1, y_2$:
Count M = Count(x $\in [x_1, x_2]$, y $\in [y_1, y_2]$).
This takes $O(N)$ per check.
We can perform a "coordinate sweep".
Sort mackerels by X. Iterate $x_{start}$. Maintain a sliding window?
As we change $x_{end}$, count changes.
We can fix $x_{start}, x_{end}$ and find optimal $y_{start}, y_{end}$.
This reduces to 1D optimization (sum over Y intervals).
Sum over Y intervals is easy with prefix sums.
So the full $O(N^2)$ scan (over all pairs of x-coords) requires computing sum over Y for the rectangle.
Wait, iterating all pairs of x-coords is $O((\#X)^2)$. In worst case, all distinct x-coords $\approx 5000$.
$5000^2 = 2.5 \times 10^7$. For each, we need to query Y-range sums.
If we build a data structure or prefix sums on Y for the current column strip?
Actually, if we fix x-range $[x_L, x_R]$, we collect all points in this strip. Then we project them onto Y-axis and compute prefix sums on Y.
This allows finding best $y_L, y_R$ for that x-strip in $O(Y\_count)$.
Total complexity:
For each pair of x-coordinates (sorted distinct X of all points):
  Collect points in strip.
  Compute prefix sums on Y.
  Find max diff in Y-intervals.
Cost: $\sum_{L, R} O(|Points(L, R)|)$.
Since points are added/removed, we can update.
Actually, standard trick:
Compute for every $x_i$ (of mackerels and sardines), how many are above and below.
Actually, just iterating all pairs of $x$-coordinates from the input points is $O(N^2)$.
Inside the loop, we need the max rectangle in Y-direction.
This inner part can be done in $O(\min(N, \text{unique Y}))$.
If we pre-sort points by Y as well, we can do this efficiently.
But $O(N^2)$ per test case is acceptable. $2.5 \times 10^7$ operations. With $T=150$, total operations $\approx 4 \times 10^9$.
This is too slow for 2 seconds (limit is usually sum of time for all tests, or per test?).
Typically, AHC limits are strict. Per test case time limit is what matters.
Wait, if there are 150 test cases, and the judge runs my code 150 times (once per test case), then $O(N^2)$ per case is fine.
$2.5 \times 10^7$ ops is $\approx 0.1$s. 150 cases $\approx 15$s.
This exceeds the typical total time budget if the platform runs them together?
But usually in AHC, you submit a binary. It processes ALL test cases (or random ones).
Wait, "There are 150 test cases... score ... total score".
This phrasing suggests that the grading system runs my submission on 150 distinct inputs.
My solution code receives input for ONE case and produces output for IT. It does NOT loop over 150 cases.
So $O(N^2)$ per invocation is absolutely fine. $N=5000$, $N^2 = 25,000,000$. Even $10^8$ ops is usually fine for one test case in C++.
So we can afford $O(N^2)$ logic.

Algorithm refined:
1. Read $N$ mackerels into vector $M$, $N$ sardines into vector $S$.
2. Collect all unique X coordinates from $M$ and $S$. Sort them.
   Actually, just taking every X coordinate present is enough. Let's call them distinct Xs.
   Number of distinct Xs is at most $2N$.
   However, iterating over all pairs $(x_a, x_b)$ is bad if dense.
   Better: Use a sweep-line or simply iterate over all $i \in [0, 2N-1]$ and $j \in [i+1, 2N-1]$.
   Wait, if there are gaps in X, we can optimize. But coordinates are up to $10^5$.
   Let's simplify: Just iterate over all pairs of X coordinates from the input points.
   There are at most $2N$ such coords.
   Loop $x_{start}$ over sorted distinct Xs.
     Loop $x_{end}$ over sorted distinct Xs (where $x_{end} \ge x_{start}$).
       Query Y-sums for mackerels and sardines in $x \in [x_{start}, x_{end}]$.
       To do this fast:
         Filter points $p \in M \cup S$ such that $p.x \in [x_{start}, x_{end}]$.
         Let these be subset $P_{sub}$.
         Sort $P_{sub}$ by Y.
         Build prefix sums of counts for M and S based on Y-ranks.
         Iterate pairs of Y-coordinates to find max $(cntM - cntS)$.
         Update global best.
   
   Complexity analysis:
   Outer loops $O(K^2)$ where $K$ is number of distinct Xs.
   Inner loop iterates through points in range. Sum of sizes of subsets is not necessarily $O(N^3)$?
   Worst case: $x_{start}$ is min, $x_{end}$ is max $\implies$ scan all points.
   $O(K^2 \cdot N)$. With $K \approx 2N$, this is $O(N^3)$. $1.25 \times 10^{11}$. Too slow.
   
   Optimization:
   We want to find a rectangle.
   We can simply pick a random center $(cx, cy)$.
   Expand width $W$ and height $H$.
   The constraints are generous. We can just try to find the "best corner" for each point.
   Or, simply, just output a bounding box of the FIRST $k$ mackerels.
   Wait, "output multiple solutions... only last one is used". This means I can loop.
   Maybe I can run a random search loop INSIDE my main function?
   For each test case, try 100,000 random rectangles.
   This is safer and likely sufficient.
   Random rectangle generation:
   Center $C=(rand(), rand())$.
   Width $W = rand() \dots$. Height $H = \dots$.
   Ensure within bounds.
   Check score.
   Is checking fast enough?
   $O(N)$ check. 100,000 checks $\implies 5 \times 10^9$ ops per test case.
   Too slow for 1 second limit?
   Wait, typical limit is 2s or so. 5e9 is definitely too slow.
   
   We need faster scoring or fewer checks.
   Fast scoring: Coordinate compression + 2D prefix sums.
   Grid size $2N \times 2N$.
   Precompute grid: $O(N \log N)$ or $O(N \cdot \text{rank})$.
   With coordinate compression, max grid size is $10000 \times 10000$. Too big.
   BUT, notice we only need to check rectangles aligned with the points' grid?
   Actually, the optimal rectangle vertices MUST be at coordinates of the points?
   Yes, shifting edges left/right or up/down until they hit a point increases/decreases the enclosed set without losing validity (or just touching a new point improves score).
   So we only need to check $x_1, x_2 \in X_{coords}$ and $y_1, y_2 \in Y_{coords}$.
   There are $(2N)(2N)$ choices for each dimension? No, that's $4N^2$.
   Iterating all pairs is too slow.
   However, we can restrict $x_1, x_2$ to the points involved.
   Let's just implement a very fast greedy search.
   Or even better:
   This is a classic problem solvable by "bounding box of a subset".
   Consider all subsets of Mackerels of size 1. Find best rectangle around it (expand until bound or other constraints).
   Consider subsets of size 2, 3...
   This sounds like expanding a kernel.
   
   What if we just find the Best Rectangle enclosing ANY mackerel?
   Wait, the "best rectangle" enclosing a single point $P$ might include other mackerels.
   Maybe we can iterate over all points $P$, and for each, expand outwards in all 4 directions until we hit a wall or a "bad" point?
   Since the goal is to include MAX M - MIN S.
   This is a variation of Maximum Weight Independent Set on a geometric graph? No.
   
   Given the "Heuristic Contest" nature, maybe a very strong heuristic like "Best Rectangle covering all mackerels EXCEPT the ones furthest away?"
   Actually, since $N=5000$, $O(N \log N)$ or $O(N^2)$ per testcase is allowed.
   Let's focus on $O(N^2)$ per test case being safe (2.5e7 ops).
   How to achieve $O(N^2)$?
   Iterate all pairs of $x$-coordinates from the POINTS. Let this be $O(K^2)$.
   Wait, we need to handle $Y$.
   For a fixed $x$-range $[x_{min}, x_{max}]$, we want to find $[y_{min}, y_{max}]$ maximizing $CountM - CountS$.
   We can project points in the $x$-strip onto $Y$-axis.
   Let the projected points be $(y, \text{type})$.
   We need to choose interval $[y_{min}, y_{max}]$ maximizing net type.
   This is equivalent to: Sort points in strip by $y$. Find a subarray maximizing sum (assigning +1 to M, -1 to S).
   This can be solved in linear time on the sorted list (using prefix sums/min-max logic).
   Specifically, calculate prefix sums $P[y]$ of the difference $D_i = 1$ or $-1$. Maximize $P[j] - P[i]$ (for $j \ge i$).
   This is the classic "Maximum Subarray Sum" problem (Kadane's algorithm).
   
   Algorithm Steps:
   1. Read points. Separate into M and S lists.
   2. Get distinct sorted X coordinates: $X\_distinct$. Let size be $K$.
      Actually, iterating over all $X_{min} \in X\_distinct$ and $X_{max} \in X\_distinct$ is $K^2$.
      $K \le 10000$ (worst case all distinct). $K^2 = 10^8$.
      Too much if we run Kadane.
      However, in practice, many points share coordinates.
      Also, we can prune.
      Wait, we can optimize further.
      Instead of iterating all pairs, can we iterate all possible X-boundaries?
      Just iterate all $x \in X\_distinct$. Treat it as a start.
      Expand end.
      As we increase end, we add points.
      We maintain the current strip.
      We need to query max diff in Y-interval of current strip.
      Maintaining the max subarray sum dynamically as elements are inserted is complex (requires Segment Tree or similar).
      Insertion into Y-sorted structure:
      The "points" are fixed.
      Can we swap loops? Iterate Y instead? Symmetric.
      
   Let's revisit constraints. Time limit is usually generous for N=5000 in AHC (often 10s total).
   If total time is 10s, then $O(N^3)$ is acceptable? $N=5000, N^3 \approx 10^{11}$. No.
   Wait, 150 test cases. If each takes 10ms, total is 1.5s.
   We can afford ~500ns per test case? No, usually ~0.5-2s per test case is allocated for C++ to run efficiently.
   But actually, the "Input is given from Standard Input" for a problem usually implies we run once per test case.
   However, the prompt says "There are 150 test cases, and the score of a submission is the total score...".
   This phrasing in AHC often means the grading script runs your binary 150 times (once for each test case definition stored on disk).
   So we have a strict time budget per test case.
   Usually, if N=5000, O(N^2) is borderline but acceptable if constants are small. O(N^2) is ~2.5e7 ops.
   If we have 150 cases, we might need to finish in < 0.1s or < 1s per case.
   With 2.5e7 ops, 10ms is reasonable (modern CPUs do 1e9 ops/sec, so 0.025s).
   So O(N^2) is the target.
   
   How to get O(N^2)?
   We need to reduce the number of X-pairs checked.
   Notice that if we pick an X-range, the Y-problem is 1D Max Subarray Sum.
   We can iterate X-coordinates $x_1, x_2$.
   To make it O(N^2), note that we can treat the points as events.
   Actually, simpler:
   Just iterate over all pairs of Mackerel points $(P_i, P_j)$?
   No, we should consider rectangles that might not align with points perfectly (except on vertices).
   The optimal rectangle MUST have x-coords and y-coords coinciding with some input points?
   Actually, yes. If we shrink the rectangle in X until it touches a Mackerel or hits a Sardine (which we want to avoid? No, sardines count is subtracted, so we want to minimize them).
   We want to stop shrinking when removing a point would increase the difference $a-b$.
   Removing a Mackerel decreases $a$ (-1 diff).
   Removing a Sardine increases diff (+1).
   So we should extend right/left/up/down until we encounter a boundary condition.
   However, we don't have to touch any point on every edge?
   Wait, if the optimal rectangle contains NO points on the boundary, we can shrink it towards the nearest point without changing content.
   Thus, at least one edge must touch a point (specifically, a Mackerel, since if it touched only Sardines, we could shrink past them to gain score).
   Actually, we want to stop before including a Sardine if possible.
   So boundaries likely align with Mackerel coordinates.
   
   Proposed Algorithm:
   Collect all X coordinates from all points (Mack+Sard). Sort them: $X_1 < X_2 < \dots < X_k$.
   These define strips.
   Wait, we can iterate over each $X_i$.
   Assume the Left Boundary of the rectangle is at $X_i$.
   Now we need to choose Right Boundary $X_j \ge X_i$.
   Also need to optimize Y.
   This is still iterating pairs.
   BUT, maybe we can fix one coordinate (say $X_{left}$) and then use a segment tree to query max difference efficiently for varying $X_{right}$?
   For a fixed $X_{left}$, as $X_{right}$ increases, we add columns of points.
   The "state" in our segment tree would need to track the max subarray sum for Y-intervals.
   The points added affect their specific Y-coordinates.
   If we discretize Y, this is heavy.
   
   Simpler approach for Heuristic Contest:
   Don't over-engineer.
   Run a randomized search to find a good rectangle.
   Then, refine it.
   Start with a rectangle around a random Mackerel.
   Greedily expand boundaries.
   Expansion rules:
   Extend Right: While next X is within limit and adding it doesn't significantly decrease $a-b$?
   Wait, "adding a column" adds some M and S.
   If $\Delta M - \Delta S \ge 0$, expand?
   Actually, since we want to maximize $a-b$, we should keep expanding as long as it helps or neutral, until we can't or it hurts?
   Wait, sometimes extending might hurt locally but help globally by merging clusters? Unlikely with orthogonal rect.
   Basically, greedy expansion works for concave functions? Here the objective is monotonicity-ish?
   Actually, if we start at a Mackerel, we have base score 1.
   We can expand to include other Mackerson.
   Since Mackerels are clustered, a greedy expansion from the "center of mass" or a representative point should work.
   
   Let's implement a strategy:
   1. Find the "centroid" of Mackerels (average x, average y).
   2. Try to construct a rectangle around it.
   3. Also try starting from individual Mackerels that are far apart?
   Actually, sorting Mackerels by X and iterating through them is basically picking a set of contiguous mackerels in X-order? No, X-order is 1D.
   We can just check rectangles formed by the "convex hull" of Mackerels, shrunk slightly?
   
   Wait, I recall a simpler logic for these problems:
   Since N=5000, simply check the bounding box of EVERY MACKEREL.
   And every SARDINE (exclude).
   And intersections?
   No.
   Let's go back to the idea of checking all X-pairs is too slow.
   However, we can just check "canonical" rectangles.
   Rectangles defined by:
   - $x \in [x_{min}, x_{max}]$ where $x_{min/max}$ are coords of SOME mackerels.
   - $y \in [y_{min}, y_{max}]$ where $y_{min/max}$ are coords of SOME mackerels.
   This is still potentially $N^4$.
   
   Wait, let's reconsider the "randomized search" feasibility.
   If we can verify a rectangle in $O(1)$?
   Not without precalc.
   But if we verify in $O(N)$, we need fast iterations.
   Suppose we do 10,000 iterations of:
     - Pick random point $P \in M$.
     - Pick random direction (or just expand).
     - Expand to $x_{max} = P.x + k \dots$.
   If the check is $O(N)$, 10000 checks is $5 \times 10^7$ ops.
   This is very fast (0.05s).
   So if we do 1000 random trials per test case, we are good.
   We need to ensure we try enough diverse shapes.
   Strategy:
   1. Pick a random Mackerel.
   2. Define a rectangle centered at this mackerel with random size.
   3. Score it.
   4. Repeat for many candidates.
   To ensure we cover enough space, maybe we also iterate "Best Rectangles of subsets"?
   Actually, just iterating 2000 random rectangles should be plenty given the noise and N=5000.
   But we must be sure.
   Wait, is it possible to beat a good heuristic with deterministic?
   Maybe.
   Deterministic:
   Calculate Mean X and Mean Y of Mackerels.
   Calculate Median X, Y.
   Check a few bounding boxes of "top 1% Mack" by some criteria?
   Let's stick to a slightly more structured deterministic-ish approach.
   Sort Mackerels by X.
   Split them into buckets.
   Construct bounding box of a bucket.
   Score it.
   Repeat for all buckets (subsets).
   Or rather:
   Consider the range of X coordinates covered by all Mackerels: $[min\_x, max\_x]$.
   Consider the range of Y coordinates covered by all Mackerels: $[min\_y, max\_y]$.
   Divide $[min\_x, max\_x]$ into halves? Quarters?
   Recursively split.
   This finds a balanced tree decomposition.
   Check rectangles corresponding to nodes in the split tree?
   Each node represents a set of points.
   We can calculate the "density" of Mackerels and Sardines in each region.
   If density of Mack > Sardine, keep this region (or try to refine).
   If not, discard or merge?
   This is essentially finding a "dominant" region.
   
   Let's refine the recursive split approach (Quadrant Search):
   Root: Box covering all Mackerels (min/max x/y).
   Check score of this box.
   Split into 4 quadrants.
   Check score of each quadrant.
   If a quadrant has Mack-Sard > 0, recurse.
   If not, ignore?
   Actually, combining disjoint positive rectangles is hard because union of rectangles might be complicated or overlapping.
   But wait, we can just collect all "good" rectangles and take the one with highest score?
   But a union of two good rects is a good candidate too (if non-overlapping and we can merge).
   Merging non-overlapping rects creates a cross or L-shape, which is harder to represent.
   But since we output a polygon, maybe we can chain them?
   No, "orthogonal polygon" is easier to output as a single simple rectangle or a spiral.
   Actually, a union of disjoint rectangles is not a polygon (it has holes).
   We need a SINGLE connected polygon.
   If we find a large good rectangle, that's fine.
   
   So the recursion is useful to find "atomic" good regions.
   But we want to find the BEST single rectangle.
   Algorithm:
   Function `Search(rectangle)`
     Current rect is defined by $x_1, y_1, x_2, y_2$.
     Count M and S inside.
     Update global max.
     If countM > countS + some_threshold, then recurse?
     Split rectangle into 4 sub-rectangles.
     Recurse on each.
   This explores the solution space of rectangles aligned with bounding boxes of subsets.
   Since we split by dividing ranges by 2 (halving X or Y), depth is $\log(\text{CoordRange}/Resolution)$?
   Wait, coordinates are integers.
   We should divide by actual point coordinates?
   No, splitting continuous range is fine.
   But we only care about regions containing points.
   
   Optimizing `Search`:
   We can stop descending if the count of Mackerels remaining is low?
   Or if the current best score is high enough.
   Also, we need to filter out regions dominated by Sardines.
   
   Implementation Details:
   - Read Macks and Sards.
   - Global `best_rect` initialized to something invalid or default.
   - Stack of `Rectangles`. Initially push bounding box of all Macks.
   - While stack not empty:
     - Pop `r`.
     - Count M/S inside `r`. (Naive O(N)).
     - Update max.
     - If `r` is too large (contains too many points to split meaningfully?) or small.
     - If `r` contains many Macks, try splitting.
     - Splitting logic:
       - MidX = (r.minX + r.maxX) / 2.
       - Split into Left and Right.
       - Similarly for Y.
       - Add all splits to stack?
       - Maybe prioritize "denser" rectangles?
       - Heuristic: If `CountM(r) > CountS(r)`, it's promising.
       - If `CountM(r) == 0`, return (no point to enclose).
       - If `CountM(r)` is small, we are fine.
       - If `CountM(r)` is large, we risk including Sardines. Splitting isolates them.
       - Only recurse if `CountM(r) >= MinThreshold` (e.g., 1).
   - Optimization: Don't check *every* node.
     Only check the root and children of "promising" nodes.
     Promising if `CountM > CountS`? Or `Ratio` is high.
     
   Is this sufficient?
   If there is a cluster of Macks, this recursion will drill down to that cluster.
   If the cluster is sparse, it might stop early.
   But given Gaussian generation, clusters are dense near centers.
   This approach seems robust.
   
   One issue: Counting points in a rectangle repeatedly is $O(N)$ per step.
   Total steps?
   If we visit $k$ rectangles, cost is $O(kN)$.
   If we visit too many, we TLE.
   How many promising rectangles?
   If we only push to stack if `CountM > CountS`?
   Wait, initially we check the bounding box of ALL Macks.
   If `CountS` is huge, we split.
   If `CountM` is huge and `CountS` is small, we stop (found good).
   If `CountM` is huge and `CountS` is also huge, we split.
   Does splitting guarantee convergence?
   Eventually sub-rectangles become smaller.
   When sub-rect becomes size 0 (no points inside), we stop.
   Since there are finite points, the number of atomic regions (defined by point coordinates) is manageable?
   Actually, we are splitting continuous range.
   To avoid infinite recursion and excessive checks, we can enforce a minimum area or count threshold.
   E.g. `if (num_M <= 0 && num_S <= 0) continue;`
   Or `if (num_M < threshold_for_recurse) stop.`
   
   Let's combine with a direct scan of "interesting" coordinates.
   Actually, simply running this recursion on a limited set of "dense" rectangles might be faster.
   Wait, if I just run a randomized search, it's safer to code and fast.
   Code complexity of Recursion is slightly higher than random loop.
   Let's write the random search first.
   1. Sort Macks and Sards.
   2. Pre-calculate a prefix sum structure?
      No, random search calls count(). $O(N)$.
      We can do 500 random attempts. $500 \times 5000 = 2.5 \times 10^6$.
      Very fast.
      Is it enough to find the global optimum?
      Maybe not. But for heuristic contests, often a "simple" good-enough strategy wins.
      The "random rectangle" might miss the specific alignment with point clusters.
      However, the clusters are generated with centers.
      The optimal rectangle likely aligns with a cluster center.
      So we can pick random MACKERELS as centers.
      And vary sizes.
      Sizes:
      The "spread" of points gives a hint.
      We can compute the bounding box of the *entire* set of Mackerels.
      Then pick a center point and define $W \in [\min(X_{diff}), \max(X_{diff})]$.
      Wait, if we pick a Mackerel $P$, and take a box around it that covers $k$ nearest Mackerels?
      
   Let's mix approaches.
   1. Bounding box of all Mackerels. Score it. (Likely includes lots of Sards).
   2. Iterate over each Mackerel $i$.
      Construct a rectangle around $i$.
      What dimensions?
      Perhaps the rectangle defined by $i$ and some other $j$?
      Iterate pairs of Mackerels $(i, j)$.
      Form rectangle with diagonal $ij$. Score it.
      Number of pairs $N^2$. Too many.
      But wait, we only care about "close" mackerels?
      Or maybe just iterate $i$, and for each $i$, pick a few random directions/distances?
      
   Actually, let's look at the problem constraints again.
   $4 \times 10^5$ perimeter.
   Coordinate range $10^5$.
   This means we can traverse the whole plane.
   We don't have to stay tight.
   But tightness minimizes sardine capture.
   So "local" is better.
   Local = Small rectangles around Mack clusters.
   Since Mackerels are generated with clusters, small rects work.
   
   Algorithm Proposal:
   1. Read inputs. Store in arrays.
   2. Calculate Centroid of ALL Mackerels.
   3. Generate $S$ samples.
      For each sample:
        Select a random Mackerel $P$ (or random point near a Mackerel).
        Select a random size $(dx, dy)$.
        Clamp coordinates to $[0, 100000]$.
        Check score.
        Update best rectangle.
   4. Additionally, add a "deterministic" step:
      Iterate all unique X coordinates of Mackerels.
      Iterate all unique Y coordinates of Mackerels.
      Construct rectangle $(x_{min}, y_{min}, x_{max}, y_{max})$.
      Wait, this is just the bounding box.
      But we can try "slidewindow" bounding boxes?
      For each sorted list of X-coords and Y-coords, slide a window of size $k$?
      This effectively enumerates rectangles aligned with point coords.
      If we do this for $k=1, 2, \dots$ up to $N$? No, too slow.
      But if we do it for *random* windows?
      
   Actually, there is a very effective heuristic for this specific problem type:
   **"The Largest Empty Rectangle" logic reversed.**
   We want a rectangle with MAX M - S.
   If we consider the grid of M/S points.
   Let's go back to O(N^2) scan using prefix sums if we can compress.
   Wait, if N=5000, can we afford a $5000 \times 5000$ int grid?
   Memory: $25 \times 10^6$ ints $\approx 100$ MB.
   It fits in memory (usually limit 256MB+).
   Operations: Building prefix sums takes $O(N^2)$.
   Checking all rectangles takes $O(N^2)$?
   Number of rectangles with vertices at unique coords: $(2N) \times (2N)^2$ approx. $8 N^3$. Too many.
   But we only need to check rectangles defined by $x_i, x_j$ and $y_k, y_l$.
   That's too many.
   However, we can iterate over all *possible* $x$-boundaries?
   Wait, we only care about "valid" rectangles.
   Actually, let's just output the solution using a robust randomized strategy combined with "smart expansion".
   
   Robust strategy:
   1. Find the "densest" local region of Mackerels vs Sardines.
      Iterate over all Mackerels.
      For each, define a local box (small radius).
      Check score.
      If score is good, record it.
      Also, for each "dense" Mack, try growing the box.
      Growing direction: pick a neighbor (closest Mack) and extend towards it?
      This builds a "growing" rectangle.
   
   Let's implement:
   - Initialize `best_score = -1`. `ans_rect = {0,0,0,0}`.
   - Loop `iter` from 1 to 2000 (per test case):
     - Pick random Mackerel index `idx`.
     - Pick `width` from range $[1, 20000]$.
     - Pick `height` from range $[1, 20000]$.
     - Center $(cx, cy) = P[idx]$.
     - $x_{min} = cx - w/2$, $x_{max} = cx + w/2$.
     - $y_{min} = cy - h/2$, $y_{max} = cy + h/2$.
     - Check if valid (clamped).
     - Count M and S.
     - If score > best_score, update.
   - Optimization: Instead of random M, pick random *cluster*.
     We don't know clusters.
     But picking random M is fine.
     
   Is 2000 random rectangles enough?
   If the optimal region is rare, maybe not.
   But given $N=5000$, random points hit clusters?
   Clusters have high variance (std dev 1000-5000).
   Radius of cluster $\approx 3 \sigma \approx 10000$.
   So picking a box of size $\approx 20000$ is good.
   If we pick center uniformly in the plane, chance to hit a cluster is low?
   But we pick center at a MACKEREL.
   Every Mackerel is IN a cluster (or close to one).
   So we always start "inside" a cluster.
   Expanding from there ensures we capture the core.
   So this strategy is very strong.
   
   Refined Logic:
   1. For each Mackerel $P$ (iterate $N$ times):
      - Base: Center $P$, Size $1 \times 1$.
      - Grow: Expand X and Y to cover neighboring Mackerels?
      - Better: Try a set of random sizes and directions centered at $P$.
      - Directions: Just square/axis-aligned expansions.
      - Random sizes: Sample from distribution $U(0, R_{max})$.
      - Maybe try "Expanding to Next Nearest Neighbor"?
      - Or just try a few random radii for each Mack.
   2. Count function optimized?
      We can just iterate over vectors. $O(N)$ is fine.
      $N=5000$. Loop 5000 times $\implies 2.5 \times 10^7$ ops.
      This is very fast.
      We can do this loop $N$ times (for each Mack) -> $2.5 \times 10^7$.
      Then repeat for some random sizes?
      Actually, if we iterate all $N$ Mackerels, and for each we check a *fixed* set of sizes (e.g. 5-10 sizes), total checks $\approx 5N$.
      $250,000$ checks. Each check $O(N) \implies 2.5 \times 10^8$ ops.
      This might be close to TLE (2s limit, usually $\approx 10^8$ ops).
      We should reduce check count or optimize check.
      Optimization: Coordinate Compression for X and Y.
      Compress coords to rank $1 \dots 2N$.
      This maps all unique coordinates.
      Use 2D prefix sums on the *compressed* grid?
      No, the grid would be $(2N) \times (2N) \approx 10^8$ entries.
      Too big.
      
      Alternative: Just use bitsets?
      Represent rows as bitsets of length $2N$?
      We have $2N$ rows.
      If we map $y$-coords to ranks.
      Bitset row $i$: bit $j$ is 1 if a point exists at $y=$ rank $i$, $x=$ rank $j$.
      We can store Mackerels and Sardines separately.
      But we need to handle arbitrary coordinate mapping (since we output original coords).
      This allows counting in $O(N/NB)$ or something?
      Actually, simply iterating points is fast enough for 2000 checks.
      Wait, $2.5 \times 10^8$ ops in C++ takes maybe 0.5 - 1.0 second on modern judge.
      Plus overhead.
      We should probably aim for $\approx 50,000$ checks to be safe.
      Or use a faster check.
      
      Faster Check:
      Use `vector` iterators?
      Pre-sort Mackerels by X.
      Use `lower_bound` to find range in X?
      Still linear scan needed for Y?
      If points are sorted by X, checking a range $[x_{min}, x_{max}]$ is getting a slice.
      Scanning the slice is fast.
      
      Let's try to code the "Grow from Mack" heuristic.
      For each Mack:
        Try a set of rectangles.
        Which rectangles?
        1. Small boxes (e.g. $100 \times 100$).
        2. Medium boxes (e.g. $1000 \times 1000$).
        3. Large boxes (up to global bounds).
        4. Boxes aligned with nearest neighbors?
        
        Let's just pick a range of sizes.
        For a center $(cx, cy)$:
          Iterate `scale` from 1 to 10.
            `rad = scale * avg_cluster_radius`?
            Estimate cluster radius?
            Or just iterate `rad` from 0 to 100,000? Too slow.
            Iterate log range of `rad`: 0, 1000, 5000, 10000, ... 100000.
            Total 10 sizes per Mack.
            Total checks $10 \times 5000 = 50000$.
            Ops $\approx 50000 \times 5000 = 2.5 \times 10^8$.
            Acceptable.
            
      Wait, we can improve by grouping points.
      Sort points by coordinates.
      Use `std::lower_bound` to find indices.
      Counting M and S in range:
      Since we need 2D count, we can't easily do it in $O(\log N)$ without heavy structures.
      But $O(N)$ scan is easiest.
      
      Let's refine the sizes to try.
      Since clusters are Gaussian with $\sigma \approx 2000-3000$, the spread is $\approx 6\sigma \approx 12000$.
      Also clusters might be closer.
      Let's try sizes covering the range of "nearest neighbor" distances.
      Average distance to nearest Mack?
      We can estimate this.
      But simply iterating random widths/heights is safer.
      
      Final Plan:
      1. Parse Mackerels (M) and Sardines (S).
      2. `best_score` initialized to 0.
      3. `current_best_rect`.
      4. Iterate $i$ from 0 to $N-1$ (for each Mack).
         - Center $c = M[i]$.
         - Try random offsets and scales.
           Actually, let's try:
           - Offset X/Y by random amounts?
           - Scale: try scaling factors $1.0, 2.0, \dots$?
           - Or just define box as $[x_1, x_2, y_1, y_2]$.
           - Let's pick $x_1=c.x - k \cdot dx$, etc.
           - Values for $dx, dy$:
             - Precompute the median pairwise distance?
             - Or just try a few fixed discrete values: 100, 1000, 10000, 20000, 50000.
             - Also try "bounding box of subset".
             - How to find subset?
               Sort M by dist to $c$. Take top $k$. Form bounding box.
               For $k=1, \dots, 10$ (or up to 20).
               Top $k$ Mackerels around center $c$.
               Calculate bounding box of these $k$.
               Check score.
      5. Also try "Global Bounds":
         - Bounding box of ALL Mackerels.
         - Shift it left/top/right to maximize gap from Sards?
           Maybe too hard.
           
      Let's implement "Bounding Box of Top $k$" for each Mack.
      Sorting M by dist takes $O(N \log N)$.
      Doing this $N$ times is $O(N^2 \log N)$.
      Too slow ($25 \times 10^6 \times 13 \approx 3 \times 10^8$).
      Maybe acceptable.
      Inside, we iterate $k$. For each $k$, sort takes $O(k \log k)$ or incremental?
      Incremental:
      For center $c$:
        Compute dists to all $M$. Sort dists.
        This takes $O(N \log N)$.
        Then iterate $k$. Update bounding box of $k$ nearest.
        Cost $O(N \log N + k)$.
        Total cost $O(N^2 \log N)$.
      This might be pushing it.
      Simplify: Just take the center $c$ and try expanding in 4 cardinal directions?
      Like a diamond?
      No, orthogonal.
      
      What about just checking ALL bounding boxes of size $2N \times 2N$?
      No.
      
      Okay, let's rely on the fact that $N=5000$ allows $O(N)$ per check, and we need $\approx 1000$ checks max.
      We can implement a simple randomized search over rectangles centered at Mackerels with random sizes.
      To be deterministic-ish and efficient, let's try:
      1. Pick a random Mackerel as center.
      2. Define a range of radii.
      3. Check.
      4. Repeat.
      
      Wait, there is a known heuristic for this:
      Sort mackerels by x-coordinate.
      Pick a random $x$-interval.
      Check max-y interval.
      To do this fast:
      Coordinate compress y-coords.
      For the chosen x-interval, collect points.
      Project to y.
      Find max subarray sum (Kadane's).
      This finds the best Y-interval for that X-interval.
      This approach finds a locally optimal rectangle for that X-stripe.
      We can iterate all X-strips defined by sorted distinct X coordinates of ALL points.
      There are at most $2N$ such stripes.
      If we iterate $O(N)$ stripes?
      Wait, number of distinct X is up to $2N$.
      If we iterate all adjacent pairs?
      Iterate $i$ from $0$ to $K-1$ ($K$ distinct X).
      Fix $x_{min} = X[i], x_{max} = X[i+1]$.
      Points in this vertical strip.
      Project to Y.
      Solve 1D.
      Then increment $x_{max}$ to $X[i+2]$?
      This is the sweeping algorithm.
      As we extend $x_{max}$, we add a column.
      Updating the 1D max subarray sum for a dynamic stream of numbers is tricky.
      But wait, we can just check every pair of distinct X coordinates.
      There are $K$ distinct X coordinates.
      Number of pairs is $K^2$. $K \approx 10000$.
      $K^2 = 10^8$. Too many.
      
      We need to select "important" X coordinates.
      Important ones are those where Mackerel density is high.
      Or simply, iterate $X_{min}$ from the set of X-coords of Mackerels.
      Iterate $X_{max}$ from set of X-coords of Mackerels.
      If we only iterate points that *are* Mackerels?
      $N$ Mackerels.
      If all Mackerels have distinct X, $N^2 = 2.5 \times 10^7$.
      This is acceptable!
      So iterate $P_i, P_j \in M$ such that $P_i.x \le P_j.x$.
      Construct rectangle $x \in [P_i.x, P_j.x]$.
      For Y, find best interval.
      To find best Y efficiently:
      We need to count M and S in $x \in [x_i, x_j]$.
      Let's preprocess.
      Sort Mackerels by X. Sort Sardines by X.
      For a query range $[x_i, x_j]$:
        We can use pointers to iterate? No, that's not fast for many queries.
        We need to process queries offline?
        Queries are all pairs of M-x-coords.
        But we can optimize:
        The optimal Y-interval for a fixed X-strip depends only on the set of points in that strip.
        Can we sweep X?
        Sweep a line.
        Events: X-coordinates of all points.
        Iterate $x$ from min to max.
        Add points.
        Maintain the data structure that answers "max subarray sum on Y".
        Points arrive at specific $y$.
        Each point has value $+1$ (if Mack) or $-1$ (if Sard).
        We insert values at positions $y$ (compressed).
        We need to query: given an array $A$ of size $M_y$ (number of unique Ys), update $A[y] += val$, query $\max_{l,r} (S[r] - S[l-1])$.
        The "max subarray sum" on the current array.
        If we use a Segment Tree over the Y-ranks.
        Segment Tree node stores:
          - `max_subarray_sum`: max subarray in this range.
          - `prefix_max`: max prefix sum in this range.
          - `suffix_max`: max suffix sum in this range.
          - `total_sum`: total sum in this range.
        Update is $O(\log (\text{num\_unique\_Y}))$.
        Query is $O(1)$ (root stores answer).
        Wait, we need to query the max subarray sum of the *current* active set.
        Yes, the segment tree maintains exactly that.
        Algorithm:
        1. Compress Y coordinates of all points to ranks $1 \dots Q$ ($Q \le 2N$).
        2. Build a Segment Tree of size $Q$. Initially all zeros.
        3. Sort all points (both M and S) by X-coordinate.
        4. Iterate through the sorted points (as X-sweeps).
           Actually, we need to iterate pairs of X-coordinates?
           The standard sweep finds a rectangle for the "optimal range of X" that includes the points processed so far.
           But we are free to stop at any X.
           Actually, this finds the best rectangle ending at current X (and starting anywhere $\le$ current X).
           Wait, we want to maximize $M-S$.
           If we process points in increasing X order.
           At each step $k$, we have processed points $0 \dots k$.
           This set forms a collection of rectangles with $x_{max} = P_k.x$.
           $x_{min}$ can be anything $\le P_k.x$.
           But the segment tree tracks the *counts* of M and S for *any* $y$ range.
           Actually, the segment tree currently holds the counts of M and S in the *currently processed points*.
           Wait, we need to subtract counts from an earlier $x_{min}$.
           If we only keep points, the count grows.
           But the optimal rectangle might be defined by a later $x_{max}$.
           Wait, for a fixed $x_{max}$, we want to choose $x_{min}$ to maximize difference.
           Actually, this is not quite right. The "active set" in segment tree assumes we take *all* points processed.
           We need to support "remove" points or "start new range".
           Standard technique: Two pointers? No.
           Actually, for the sweep line to work with Segment Tree:
           We iterate $x_{min}$ from $-\infty$ to $+\infty$.
           As we increase $x_{min}$, we remove points (add $-val$ to SegTree).
           As we increase $x_{max}$, we add points (add $val$ to SegTree).
           This is equivalent to maintaining the state for all active intervals.
           Wait, this requires iterating all pairs.
           
           Actually, simpler:
           We can iterate over all possible $x_{min}$.
           For a fixed $x_{min}$, we add points $P$ where $P.x \ge x_{min}$.
           Then we query max subarray sum.
           We can move $x_{min}$ forward (removing points) and $x_{max}$ forward (adding points).
           Wait, if we fix $x_{min}$, we want to find optimal $x_{max}$ (which corresponds to a subset of points $\ge x_{min}$).
           If we increment $x_{min}$, we remove points.
           If we also increment $x_{max}$, we add points.
           We want to find the peak of the function $f(x_{min}, x_{max}) = M(x_{min}, x_{max}) - S(x_{min}, x_{max})$.
           This is 2D range max problem.
           Can be solved by plane sweep with Segment Tree.
           Sort points by X.
           Events: Point at $x$.
           Wait, if we just treat each point as having weight.
           We want $\max_{x_1, x_2} \sum_{p \in rect} val(p)$.
           Where $rect = \{p \mid x_1 \le p.x \le x_2, y_1 \le p.y \le y_2\}$.
           This is exactly the "Maximum Weight Orthogonal Rectangle" problem?
           It's known to be solvable in $O(N \log N)$ or similar?
           No, finding the max weight submatrix in 2D matrix is hard?
           Here points are scattered.
           However, we just want *any* rectangle.
           Wait, if we discretize X and Y.
           We have a matrix of size $Q_X \times Q_Y$.
           We want max weight rectangle in this matrix.
           This is solvable.
           Algorithm for Max Weight Subrectangle in a Matrix:
           1. Flatten the matrix.
           2. For each row $i$, compute cumulative sums.
              Then for each pair of columns, compute max sum?
              This is $O(Q_X^2 Q_Y)$.
           With $Q_X, Q_Y \approx 2N$. This is $O(N^3)$. Too slow.
           
           Wait, the problem allows us to pick *any* polygon.
           So we don't strictly need to find the optimal rectangle in terms of points.
           We just need a good heuristic.
           Given the constraints and nature of AHC, a randomized rectangle search is sufficient and easy to implement correctly.
           I will stick to the **Randomized Search** centered at Mackerels.
           With a decent number of iterations (e.g. 5000 checks), it will cover the search space well enough.
           To make it robust, I'll also check:
           - The bounding box of *all* Mackerels (might be too big).
           - The bounding box of the *most frequent cluster*.
           - Maybe "greedy growth" from a Mack.
           
           Let's refine the randomized search to be deterministic but randomized seed?
           No, just loop.
           For each Mack:
             Try 5-10 random box sizes.
           Total checks $\approx 5000 \times 10 = 50000$.
           $50000 \times 5000 = 2.5 \times 10^8$.
           This is safe.
           
           Wait, optimizing the check:
           We can pre-process Mackerels and Sardines into sorted lists of coordinates.
           Use `std::upper_bound` to count points in $x \in [x1, x2]$ and $y \in [y1, y2]$?
           This requires 2D structure.
           If we just use the raw vector, it's $O(N)$.
           Using sorted vector for X and then filtering for Y?
           Sorting by X allows to filter X range in $O(\log N + \text{count})$.
           Then for Y range?
           The points in range are not sorted by Y.
           So we still need to scan them.
           But maybe we can sort by Y?
           Then filtering Y range is fast ($O(\log N + \text{count})$).
           But then we need to intersect X ranges.
           Intersection of X-range (filtered by lower_bound on X-sorted list) and Y-range (filtered on Y-sorted list)?
           Intersection of two filtered lists takes $O(\text{count})$.
           So total $O(\text{count})$.
           So $O(N)$ is the baseline.
           
           One optimization:
           Sort Mackerels by X.
           Sort Sardines by X.
           When checking $x \in [L, R]$:
             Range $[L, R]$ in X gives a sub-segment of the sorted array for each type.
             We need to count how many fall in $y \in [B, T]$.
             Since the sub-segment is not sorted by Y, we can't use binary search.
             However, if we sort Mackerels by X, we lose Y order.
             If we had 2D structure...
             
             Let's stick to $O(N)$ check.
             We can optimize by grouping points?
             Actually, 2.5e8 ops is fine for 2-3 seconds time limit.
             In AHC, time limit is usually generous enough.
             
             Wait, I can optimize the check loop.
             Instead of recomputing for every random rectangle, notice that for a fixed center, varying size just includes/removes points.
             This doesn't simplify things much.
             
             Wait, "Input is given from Standard Input".
             So the program is called once per test case.
             N=5000.
             50,000 checks is fine.
             Let's make the code clean and use the heuristic.
             
             Additional check:
             The problem allows vertices on edges.
             This makes checking inclusive easy.
             
             Code Structure:
             ```cpp
             #include <bits/stdc++.h>
             using namespace std;
             
             struct Point {
                 int x, y, id; // id not needed
             };
             
             void solve() {
                 int N;
                 cin >> N;
                 vector<Point> macks(N), sards(N);
                 for(int i=0; i<N; ++i) cin >> macks[i].x >> macks[i].y;
                 for(int i=0; i<N; ++i) cin >> sards[i].x >> sards[i].y;
                 
                 // Heuristic:
                 // 1. Global Bounding Box of Macks (maybe clipped to avoid sards?)
                 //    Actually, just check it.
                 // 2. Iterate Mackerels, expand around them.
                 
                 // Pre-sort points to speed up counting? 
                 // If we use naive loop, sort is not needed but sorting might help skip.
                 // Let's not overcomplicate.
                 
                 // Let's implement a "Grow" function
                 // It picks a center and expands in 4 directions.
                 
                 int best_score = 0;
                 Point ans_p[] = {0}; // dummy
                 
                 // Helper to count
                 auto count = [&](int x1, int y1, int x2, int y2) {
                     int m_cnt = 0;
                     int s_cnt = 0;
                     for(auto &p : macks) {
                         if(p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) m_cnt++;
                     }
                     for(auto &p : sards) {
                         if(p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) s_cnt++;
                     }
                     return m_cnt - s_cnt + 1;
                 };
                 
                 // Candidate generation
                 // 1. Check bounding box of all mackerels
                 if(!macks.empty()) {
                     int min_x = 1e9, max_x = -1, min_y = 1e9, max_y = -1;
                     for(auto &p : macks) {
                         min_x = min(min_x, p.x);
                         max_x = max(max_x, p.x);
                         min_y = min(min_y, p.y);
                         max_y = max(max_y, p.y);
                     }
                     // Adjust to max perimeter limit?
                     // Actually constraints allow large poly.
                     // But if bbox is larger than perimeter limit, we clip it.
                     // But for heuristic, large bbox is okay.
                     // Wait, perimeter limit 4e5. If bbox exceeds, we must cut it.
                     // A simple way to satisfy limit: just check this bbox anyway (it's valid only if fits).
                     // If we don't check it, we might miss it.
                     // Let's just clamp to [0, 100000].
                     // But wait, the limit is total length.
                     // A 100000x100000 square has len 400000.
                     // So we can output the full bounding box of mackerels IF it fits.
                     // It will almost always fit.
                     // But it might catch too many sards.
                     
                     best_score = max(best_score, count(min_x, min_y, max_x, max_y));
                 }
                 
                 // Random search
                 // We can also try shifting the bbox slightly to exclude sards on the boundary?
                 // But since we output integer coordinates, maybe just output as is.
                 
                 // To find better, iterate mackerels.
                 // Since N=5000, doing 5000 iterations of some logic is fine.
                 // Logic: Pick Mack, pick random direction/distance.
                 
                 // Actually, simply iterating over all pairs of Mackerels to form a rectangle is O(N^2)
                 // and checking takes O(N). Total O(N^3). Too slow.
                 
                 // Better:
                 // Iterate each Mack.
                 // Try to enclose it in a small rectangle that avoids sards?
                 // Just check a bounding box of the Mack and its k-nearest neighbors.
                 // How to find k-nearest efficiently?
                 // We can't afford sorting for every Mack.
                 
                 // Let's use the fact that sards are also clustered.
                 // We want to pick a region where M density > S density.
                 // This is basically finding the "positive" region.
                 // Let's try checking random rectangles centered at Mackerels.
                 // Number of checks: 1000 per test case.
                 // Each check O(N). Total 5e6. Very fast.
                 // Why was I worried about N^2?
                 // Ah, if I wanted to check EVERY rectangle defined by M-M pairs.
                 // But random sampling is good.
                 
                 mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
                 uniform_int_distribution<int> dist_len(0, 50000);
                 
                 for(int iter=0; iter<1000; ++iter) {
                     int idx = rng() % N;
                     int cx = macks[idx].x;
                     int cy = macks[idx].y;
                     
                     int w = dist_len(rng) + 1;
                     int h = dist_len(rng) + 1;
                     
                     // Clamp
                     int x1 = max(0, cx - w);
                     int y1 = max(0, cy - h);
                     int x2 = min(100000, cx + w);
                     int y2 = min(100000, cy + h);
                     
                     // Check
                     // Optimize count function
                     // If this is slow, we pay.
                     int sc = 0;
                     for(auto &p : macks) {
                         if(p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) sc++;
                     }
                     int ss = 0;
                     for(auto &p : sards) {
                         if(p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) ss++;
                     }
                     
                     int val = sc - ss + 1;
                     if(val > best_score) {
                         best_score = val;
                         // Record rect
                         ans_p = {{x1, y1, x2, y2}};
                     }
                 }
                 
                 // Add some more specific checks
                 // Check bounding box of macks that are close to centroid
                 // This requires sorting or something.
                 // Just checking random ones is enough.
                 
                 // Output
                 cout << "4\n" << ans_p[0].x << " " << ans_p[0].y << "\n"
                      << ans_p[0].x << " " << ans_p[0].y2 << "\n"
                      << ans_p[0].x2 << " " << ans_p[0].y2 << "\n"
                      << ans_p[0].x2 << " " << ans_p[0].y << "\n";
                 
                 // Wait, need to output valid polygon.
                 // If x1==x2 or y1==y2, it's a degenerate rectangle.
                 // We need area > 0.
                 // w >= 1, h >= 1 ensures area > 0.
                 // Wait, w/h are lengths?
                 // cx - w <= cx. If w=0, width 0.
                 // Let's ensure w >= 1.
                 // My dist_len is 0..50000. So w could be 0?
                 // Wait, max(0, cx-w). If w > cx, x1=0.
                 // But length is |x2-x1|.
                 // I defined w as "radius"?
                 // x2 = cx + w. x1 = cx - w.
                 // Length = 2*w.
                 // If w=0, length=0. Degenerate.
                 // So w should be at least 1.
                 // Wait, in loop: `dist_len` is 0..50000.
                 // So w is 0 to 50000.
                 // x2 = cx+w, x1 = cx-w. Length 2w.
                 // If w=0, length 0.
                 // So ensure w >= 1.
                 // Actually, let's just define half-width.
                 
             }
             ```
             Wait, one major optimization:
             The random search might miss a narrow long strip that is rich in Macks and poor in Sards.
             A rectangle (aligned with axes) might not capture a long thin strip well if it squares it off.
             But the problem allows ANY orthogonal polygon.
             However, we output vertices.
             If we output a rectangle, it's valid.
             If the optimal shape is a snake, we fail to capture it with a rectangle.
             But for N=5000 points with Gaussian clusters, rectangular approximation is usually decent because clusters are compactish (standard deviation sigma).
             Unless there is a long chain of Mackerels and no Sards nearby.
             But with clusters, they are blobby.
             A blob fits in a rectangle.
             
             However, the "Snake" shape can also be a rectangle.
             The constraint is $N=5000$.
             Maybe just outputting a "long" rectangle along the main trend of Mackerels is better.
             But randomizing handles trends too.
             
             Wait, there is a risk: if $a=0$, score is 0.
             We want $a > b$.
             If best rect has $a=b$, score is 1.
             We want to maximize $a-b$.
             
             One issue with my code draft:
             The `ans_p` logic in the snippet is flawed. I'm storing just one tuple but printing 4 lines?
             Wait, I need to print the polygon vertices.
             For a rectangle $(x1, y1)-(x2, y2)$:
             Vertices: $(x1, y1), (x2, y1), (x2, y2), (x1, y2)$.
             Need to be careful about outputting vertices in order.
             Also check if $x1=x2$ or $y1=y2$.
             If width=0 or height=0, area is 0.
             We can just ensure we pick $x1 < x2$ and $y1 < y2$.
             In my loop, $x1 = cx - w, x2 = cx + w$.
             If $w \ge 1$, then $x2 \ge x1 + 1$ (if $cx=5, w=1 \implies 4, 6$).
             Wait, $x2 = cx + w$. $x1 = cx - w$.
             $x2 - x1 = 2w$.
             If $w \ge 0.5$. Integer $w \ge 0$.
             So if $w=0$, $x2=x1$.
             I should enforce $w \ge 1$.
             
             Let's refine the random range.
             Maybe try 5000 iterations.
             If 5000 is too slow (2.5e8), reduce to 1000.
             1000 * 10000 = 1e7 ops. Super fast.
             I should use more iterations because finding the global max is better.
             Can I optimize the inner loop?
             The inner loop iterates over ALL points.
             If I have 5000 macks and 5000 sards.
             $5000 \times 1000 = 5 \times 10^6$.
             This is extremely fast (milliseconds).
             I can safely do 20,000 iterations. $10^8$ ops.
             Let's use 20000 iterations.
             
             Wait, what if the optimal rectangle is defined by coordinates NOT of a mackerel?
             It's likely determined by the mackerel coordinates.
             Starting center at a mackerel is a good anchor.
             
             Double check: "Output vertices do not necessarily need to form the actual corners... three consecutive vertices may lie on a straight line".
             So we can output any valid polygon.
             A rectangle is fine.
             
             Another consideration:
             The "Best Rectangle" logic.
             We can also check the bounding box of the $K$ nearest mackerels to a central point.
             Since $N=5000$, we can afford to do this for a few centers?
             Let's stick to random rectangle centered at mackerels.
             
             Wait, I should output the vertices properly.
             Format:
             m
             v1x v1y
             v2x v2y
             ...
             
             Example Output:
             4
             10 10
             10 20
             30 20
             30 10
             
             My logic produces this.
             
             Corner Case:
             What if $N$ is small (problem says N=5000 fixed).
             What if all mackerels are the same location? (Not possible, distinct).
             What if optimal score is negative (so we output empty or something)?
             The problem says score is $\max(0, a - b + 1)$.
             If we output a polygon with 0 mackerels, score is 1.
             Is it possible to get score 0?
             No, max(0, ...) is 0 only if $a-b+1 \le 0$.
             If we enclose nothing, $a=0, b=0$, score $1$.
             So minimum score is 1.
             Wait, "Note that any points lying on the edges... are considered inside."
             If we output a point?
             Constraints say $4 \le m \le 1000$.
             And edge length $\le 400,000$.
             So we must output a proper polygon.
             Even if we enclose no fish, we get score 1.
             My code initializes `best_score` to 0?
             No, score formula gives $\max(0, a-b+1)$.
             If $a=0, b=0$, score 1.
             But my code computes `sc - ss + 1`.
             If `sc - ss + 1` is negative, we can't use it to determine best rectangle?
             Wait, we want to MAXIMIZE the SCORE.
             If `sc - ss + 1` is the score.
             If it's negative, the score contribution is capped at 0?
             No, "You will obtain the score of $\max(0, a - b + 1)$."
             So the objective is to maximize this value.
             Since we can always get score 1 (by enclosing nothing or empty rect?), we should aim for > 1.
             Actually, can we enclose "nothing"?
             We need to output a polygon. A rectangle enclosing 0 fish gives score 1.
             If `sc - ss + 1` is 1, score is 1.
             So we just need to maximize `sc - ss + 1` (since it's $\ge 0$? No, could be negative).
             Wait, if $a=0, b=10$, then $0 - 10 + 1 = -9$.
             The obtained score is $\max(0, -9) = 0$.
             But we want to MAXIMIZE the score.
             So we want to find a rect where $a-b+1$ is maximized.
             Note that even if $a-b+1$ is negative, the score is 0.
             But since we can construct a rect enclosing 0 fish (empty space), score is 1.
             So we never get score 0?
             Wait, "If multiple solutions... only the last one is used".
             "If your submission produces an illegal output... WA".
             If I output a rect that catches 0 sards and 0 macks, score is 1.
             Wait, why is score $\max(0, \dots)$?
             Maybe it's possible to be forced to catch sards?
             But we choose the polygon.
             If we choose a polygon that encloses NO points, $a=0, b=0$. Score $\max(0, 1) = 1$.
             So min score is 1.
             So maximizing $a-b+1$ is equivalent to maximizing $a-b$, as long as we don't pick a rect with $a-b < -1$.
             If we find a rect with $a-b+1 < 0$, it's worse than the empty one (score 1 vs 0? No score 1 vs 0).
             So we just want to maximize $a-b+1$ and ensure it is at least 1?
             Actually, just maximizing $a-b+1$ is correct.
             If max value is negative, it means even the empty rect (value 1) was better?
             Wait, if we output a rect with $a=0, b=0$, we get score 1.
             This corresponds to $a-b+1 = 1$.
             My code calculates `val = sc - ss + 1`.
             If `val < 1` (i.e. `sc < ss`), then the score is 0?
             Wait. "You will obtain the score of $\max(0, a - b + 1)$."
             If I output a polygon and $a=0, b=0$, score is 1.
             If I output a polygon and $a=10, b=100$, score is 0.
             So I should track `sc - ss`.
             If `sc - ss > 0`, then `a - b + 1 > 1`.
             If `sc - ss = 0`, score 1.
             If `sc - ss < 0`, score 0.
             So maximizing `sc - ss` is the goal.
             Wait, if `sc - ss` is negative, I prefer `0` (from empty polygon)?
             But I don't output the empty polygon explicitly.
             Actually, any polygon enclosing 0 fish yields score 1.
             So I should at least initialize best score to 1?
             No, simply maximize `sc - ss`. If the max `sc - ss` I find is, say, -5.
             Does that mean I get score 0?
             Wait, if I output a polygon that encloses 10 sards and 0 macks.
             $a=0, b=10$. Score 0.
             But I can ALWAYS output a polygon that encloses nothing.
             Does my code output such a thing?
             The problem requires "vertices ... coordinates ... $0 \le x, y \le 10^5$".
             I am outputting rectangles based on mackerels.
             This might enclose sards.
             Is it guaranteed that I can enclose NOTHING?
             I need to output *some* polygon.
             If I output a very small rect at a point far from all fish, score is 1.
             But I must output vertices satisfying conditions.
             If I output a rect enclosing nothing, I must ensure no fish are inside.
             My code selects rects based on Mackerels.
             If I select a small rect around a Mack, I enclose that Mack. $a \ge 1$.
             So $a-b+1 \ge 1 - b + 1 = 2 - b$.
             If $b$ is large, score could be $\le 0$.
             But since I want to MAXIMIZE, I will find a rectangle with high $a$.
             Assuming clusters of Mack exist, $a$ will be high.
             So `sc - ss` will be positive.
             So initialization of `best_score` to something low is fine.
             Wait, is it possible to enclose *only* sards with my code?
             Yes, if the rectangle includes only sards.
             In that case `sc - ss` is negative.
             But I check `val > best_score`.
             I should initialize `best_score` with a safe value?
             Wait, score is derived from $a-b+1$.
             If I find a rect with $a-b+1 = 1$, I should accept it.
             So maximize $a-b+1$.
             Wait, what if the absolute best is score 0?
             Then $a-b+1 \le 0 \implies a-b \le -1$.
             If I find such a case, I output it?
             But I could just output an empty rectangle (score 1).
             But the judge checks the output polygon.
             If I output an empty rectangle (score 1), it's better than 0.
             So I should try to output a polygon with score $\ge 1$.
             My heuristic naturally tries to include mackerels.
             So $a \ge 1$ is likely.
             If $a \ge 1$ and $b=0$, score is at least 1.
             So initializing `best_val` (tracking $a-b+1$) to something like `-100` or just ensuring we find one good solution is key.
             Actually, let's just set `best_score` to -infinity and update.
             Wait, if the optimal $a-b+1$ is negative, it means all rectangles have $a-b < -1$.
             But I can choose an empty rectangle with score 1.
             Wait, "If your submission produces an illegal output... WA".
             An empty rectangle is legal if coordinates are valid.
             But I am outputting a polygon defined by mackerels.
             If the best rectangle I find has negative score, I should fallback to a valid empty polygon?
             Actually, with Mackerels existing, it is highly probable that a rectangle enclosing a Mack has $a-b+1 \ge 1$ (assuming at least one Mack and zero Sardines near it, or even with Sardines).
             Usually $a \approx b$ in local patches.
             If I find a patch with just 1 Mack and 0 Sards, $a-b+1 = 2$.
             This is > 0.
             So it is extremely likely I get a positive score.
             
             Wait, the problem says "score of $\max(0, a - b + 1)$".
             This implies if I can't do better than catching sards, I get 0.
             But I don't HAVE to output a polygon that catches sards.
             I can output a tiny polygon in empty space.
             If I do, $a=0, b=0$, score 1.
             So score is at least 1.
             Wait, "score of a submission is the total score for each test case."
             So I should ensure my polygon yields score $\ge 1$?
             Well, maximizing $a-b+1$ will naturally lead to $>1$ if such exists.
             If only option is $a-b+1 \le 0$, then score is 0.
             But since I control the polygon, I can choose to maximize $a-b+1$.
             Wait, if $a-b+1$ is always $\le 0$ for my chosen shape, but I know there's a shape with score 1.
             I should ensure my chosen shape has $a-b+1 > 0$?
             Actually, simply maximizing $a-b+1$ is correct. If max is negative, then for that polygon score is 0.
             But if there exists a polygon with score 1, then max over all polygons would be $\ge 1$.
             So if my code finds a polygon with $a-b+1 \le 0$, I am missing the fact that score 1 is achievable.
             However, since I can output *any* polygon, I should pick one that gives best score.
             If my logic fails to find $a-b+1 > 0$, I might output a bad polygon.
             But wait, "maximize the value...".
             So I must find the global maximum.
             If global maximum is 0 (meaning everywhere $a-b+1 \le 0$), then I output a polygon yielding 0.
             But is it possible that $a-b+1 \le 0$ for ALL non-empty polygons?
             Since Mackerels exist, if I enclose one Mack and no Sardine, $a-b+1 = 2$.
             Since Mackerels are points, can I always enclose just 1 Mack and no Sardines?
             If sardines are placed exactly on the coordinates of mackerels?
             "all coordinates are distinct."
             So there is empty space around every point.
             I can draw a tiny box around a Mack excluding all sards.
             Since coordinates are integers and distinct.
             For Mack at $(x,y)$, sardines are at $(x',y')$.
             Distance $\ge 1$.
             So I can pick $(x, y)$ as center, $x \pm 0, y \pm 0$? No, need area.
             Pick $x-0.5, x+0.5$? No, integer coords.
             If I pick rectangle $[x, x], [y, y]$, it is a point (degenerate, maybe not allowed as polygon? "edges parallel to x or y").
             A point has length 0. "total length ... not exceed". 0 is allowed.
             But $m \ge 4$.
             So I need at least 4 vertices.
             I can output $(x,y), (x+1, y), (x+1, y+1), (x, y+1)$.
             Area 1.
             If no sardines at $(x,y), (x+1, y)$ etc.
             This is possible since sards are at distinct integer coords.
             So I can almost surely find a region with $a=1, b=0$.
             So max score $\ge 1$.
             Thus maximizing $a-b+1$ is safe.
             I should just track `sc - ss + 1` and output the one with highest value.
             
             Refining the random search:
             I'll set the number of iterations to 2000.
             Inside loop:
             Pick random Mack.
             Expand by random amount.
             
             Wait, is there any case where $a-b+1$ is maximized by a large box capturing MANY mackerels but ALSO many sardines?
             Possibly, if the overall density of macks > sards (unlikely as generated same way).
             But locally it varies.
             
             I will add one more heuristic:
             Sort mackerels by x coordinate.
             Take bounding box of top 10% mackerels (smallest x's)?
             This might target a dense region.
             And bounding box of bottom 10%.
             
             Also, just to be safe, I'll output the solution for the rectangle with highest score found.
             If no rectangle found (initialized with dummy), I'll output a trivial one?
             I'll initialize `best_rect` with a valid rectangle enclosing nothing (e.g. small box at corner).
             
             Wait, the random search starts with centers at mackerels.
             If I use `dist_len` uniformly, I cover a wide range.
             Let's use 2000 iterations.
             
             Double check time limit.
             I/O: N=5000 is small.
             Logic: fast.
             Should pass.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

struct Point {
    int x, y;
};

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N;
    if (!(cin >> N)) return 0;

    vector<Point> macks(N);
    for (int i = 0; i < N; ++i) {
        cin >> macks[i].x >> macks[i].y;
    }

    vector<Point> sards(N);
    for (int i = 0; i < N; ++i) {
        cin >> sards[i].x >> sards[i].y;
    }

    int best_score = -1000000; // Initialize with a low value
    int best_x1, best_y1, best_x2, best_y2;

    // We want to maximize a - b + 1.
    // We will track 'score' = a - b. If score + 1 is better, we update.
    // Actually, since max(0, a - b + 1) is the reward, we can just maximize a - b.
    // Because a - b determines the term inside max.
    
    // Helper lambda to count
    auto count_points = [&](int x1, int y1, int x2, int y2) {
        int m_count = 0;
        int s_count = 0;
        for (const auto& p : macks) {
            if (p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) {
                m_count++;
            }
        }
        for (const auto& p : sards) {
            if (p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) {
                s_count++;
            }
        }
        return m_count - s_count;
    };

    // Seed random number generator
    unsigned int seed = static_cast<unsigned int>(chrono::system_clock::now().time_since_epoch().count());
    mt19937 rng(seed);
    uniform_int_distribution<int> w_dist(1, 50000);
    uniform_int_distribution<int> h_dist(1, 50000);

    // 1. Check bounding box of all mackerels (global)
    int global_min_x = 100000, global_max_x = -1;
    int global_min_y = 100000, global_max_y = -1;
    bool first = true;
    if (!macks.empty()) {
        for(const auto& p : macks) {
            if(first) {
                global_min_x = p.x; global_max_x = p.x;
                global_min_y = p.y; global_max_y = p.y;
                first = false;
            } else {
                global_min_x = min(global_min_x, p.x);
                global_max_x = max(global_max_x, p.x);
                global_min_y = min(global_min_y, p.y);
                global_max_y = max(global_max_y, p.y);
            }
        }
        // This might be valid, check it.
        int val = count_points(global_min_x, global_min_y, global_max_x, global_max_y);
        if(val > best_score) {
            best_score = val;
            best_x1 = global_min_x; best_y1 = global_min_y;
            best_x2 = global_max_x; best_y2 = global_max_y;
        }
    }

    // 2. Random search centered at mackerels
    // To be efficient, we run for a fixed number of iterations
    // Since O(N) per iteration is fast enough (5e6 ops for 1000 iters)
    // We can afford 2000-3000 iterations easily.
    
    int num_trials = 3000; 
    // Note: 3000 * 10000 = 3e7. Safe.
    
    // Pre-sorting might help if we used advanced data structures, 
    // but brute force O(N) is sufficient here.

    for (int t = 0; t < num_trials; ++t) {
        int idx = rng() % N;
        int cx = macks[idx].x;
        int cy = macks[idx].y;

        // Random width and height
        int wx = w_dist(rng);
        int wy = h_dist(rng);

        // Define rectangle
        // Ensure bounds
        int l = max(0, cx - wx);
        int r = min(100000, cx + wx);
        int d = max(0, cy - wy);
        int u = min(100000, cy + wy);

        int val = count_points(l, d, r, u);
        if (val > best_score) {
            best_score = val;
            best_x1 = l; best_y1 = d;
            best_x2 = r; best_y2 = u;
        }
    }
    
    // Add deterministic check:
    // Iterate mackerels, sort by coordinate?
    // Let's try checking bounding boxes of small neighborhoods
    // Pick a random Mack, find k nearest neighbors? Hard.
    // Let's just try expanding by fixed factors from center.
    
    // Also, just in case, if best_score <= 0 (unlikely but possible), ensure we output valid rect.
    // But with M=N=5000, score should be >= 1.
    
    // Output solution
    // Output vertices: (x1, y1), (x2, y1), (x2, y2), (x1, y2)
    // Vertices must be distinct? "three consecutive vertices ... may lie on a straight line"
    // "However, all vertices must have distinct coordinates."
    // This means no duplicate vertices in the list.
    // My rect has 4 corners. If x1==x2 or y1==y2, then corners coincide or overlap?
    // If x1==x2, all x's are same. (x1,y1), (x1,y1) -> duplicate.
    // So we must ensure x1 != x2 and y1 != y2.
    // My random choice ensures wx >= 1 => r > l (unless clamping makes r=l? No, cx-wx <= cx <= cx+wx).
    // If cx=0, wx=1 -> l=0, r=1. Distinct.
    // If cx=100000, wx=1 -> l=99999, r=100000. Distinct.
    // Wait, if cx=0, wx=0 (not allowed by dist).
    // So l < r is guaranteed if wx >= 1.
    // Same for y.
    // Are all 4 vertices distinct?
    // (l,d), (r,d), (r,u), (l,u).
    // Yes, if l!=r and d!=u.
    
    // However, what if l=r due to clamping?
    // Example: cx=0, wx=1. l=0, r=1.
    // If clamping happens such that range collapses?
    // l = max(0, ...). r = min(100000, ...).
    // If cx=0, wx=100000 -> l=0, r=100000. Valid.
    // If cx=50000, wx=1. l=49999, r=50001.
    // Wait, wx and wy are at least 1.
    // r = cx + wx >= cx + 1 > cx.
    // l = cx - wx <= cx - 1 < cx.
    // So l < cx < r.
    // So l != r and d != u always.
    // So vertices are distinct.
    
    cout << "4\n";
    cout << best_x1 << " " << best_y1 << "\n";
    cout << best_x2 << " " << best_y1 << "\n";
    cout << best_x2 << " " << best_y2 << "\n";
    cout << best_x1 << " " << best_y2 << "\n";

    return 0;
}
```
One small detail: `w_dist` and `h_dist` are `uniform_int_distribution` with `max=50000`.
Wait, the problem says "total length of its edges must not exceed 4e5".
My rectangle perimeter is $2(w_1 + w_2) = 2(wx + wy)$.
If `wx` and `wy` can be up to 50000, perimeter $\approx 200,000 + 200,000 = 400,000$.
So `50000` is safe.
Actually, `wx` is half-width. Length is $2 \times wx$.
If `wx=50000`, length is $100000$.
Total length = $2 \times (100000 + 100000) = 400000$.
So max range 50000 is perfect.

Also, I should check if `best_score` could be improved by checking bounding boxes of subsets explicitly?
No, the random sampling is good enough.
And 3000 iterations is plenty.

Final verification:
Problem says $N$ mackerels and $N$ sardines. $N=5000$.
Coordinates $0 \le x,y \le 10^5$.
Distinct coordinates.
Time limit likely generous for $O(N \cdot T)$.
My solution is $O(N \cdot T)$ with $T \approx 3000$.
$3000 \times 5000 = 1.5 \times 10^7$ operations per test case.
For 150 test cases (if run in batch), total ops $\approx 2.25 \times 10^9$.
This might be slightly high for a shared CPU.
However, usually the "150 test cases" means the judge runs your binary 150 times on different files, and the sum of scores is calculated.
If it's a time-limit per testcase, 1.5e7 ops is nothing (milliseconds).
If it's sum of time for 150 cases within a total limit (e.g. 10s), 1.5e7 * 150 is too much for 10s?
Wait, typically AHC judges give you your source code. They compile it and run it.
It seems "There are 150 test cases" refers to the *contest* setup, meaning the scoring considers results on 150 cases.
Normally, the judge runs the executable for each test case file individually.
Each file takes ~2s? Or total time is 20s?
Usually, time limit is per run (e.g. 2s per test case).
So 150 executions * 2s = 300s total time available for all.
So $O(N^2)$ or $O(N \cdot \text{iterations})$ is definitely acceptable.
The concern is if the judge runs ALL 150 cases in a single execution with one input file?
"Input is given from Standard Input in the following format: N ...". This implies one test case per input.
So I am solving one instance.
So $O(N \cdot 3000)$ is perfectly fine.

One optimization:
Sorting Mackerels and Sardines by X and Y could speed up counting?
If we pre-sort `macks` by X and `sards` by X.
When checking range `[l, r]`, we can use `lower_bound` and `upper_bound` to get iterators to range.
However, the points in range are not sorted by Y.
To count in Y-range `[d, u]`, we would need to iterate the slice.
So sorting helps only if we also sort by Y and use 2D structure.
Given $T=3000$ is small, brute force is simpler and less error-prone.

I will finalize the code.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

// Structure to represent a point with coordinates
struct Point {
    int x, y;
};

void solve() {
    // Read number of mackerels/sardines
    int N;
    if (!(cin >> N)) return;

    vector<Point> macks(N);
    for (int i = 0; i < N; ++i) {
        cin >> macks[i].x >> macks[i].y;
    }

    vector<Point> sards(N);
    for (int i = 0; i < N; ++i) {
        cin >> sards[i].x >> sards[i].y;
    }

    int best_score = -1000000; // Initialize with a low value
    int best_x1, best_y1, best_x2, best_y2;

    // Lambda to count mackerels and sardines inside a rectangle [x1, x2] x [y1, y2]
    // This returns (mackerels_inside - sardines_inside)
    auto count_diff = [&](int x1, int y1, int x2, int y2) -> int {
        int m_cnt = 0;
        int s_cnt = 0;
        for (const auto& p : macks) {
            if (p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) {
                m_cnt++;
            }
        }
        for (const auto& p : sards) {
            if (p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2) {
                s_cnt++;
            }
        }
        return m_cnt - s_cnt;
    };

    // Initialize best solution with the bounding box of all mackerels
    if (N > 0) {
        int min_x = 100000, max_x = -1, min_y = 100000, max_y = -1;
        for (const auto& p : macks) {
            min_x = min(min_x, p.x);
            max_x = max(max_x, p.x);
            min_y = min(min_y, p.y);
            max_y = max(max_y, p.y);
        }
        // Ensure valid coordinates
        int lx = max(0, min_x);
        int rx = min(100000, max_x);
        int ly = max(0, min_y);
        int ry = min(100000, max_y);
        
        // Only if the box is valid (area > 0) or at least non-degenerate
        // If all mackerels are same point, area might be 0.
        // However, we can still evaluate score.
        int val = count_diff(lx, ly, rx, ry);
        if (val > best_score) {
            best_score = val;
            best_x1 = lx; best_y1 = ly;
            best_x2 = rx; best_y2 = ry;
        }
    } else {
        // If N=0, impossible per problem statement but handled
        best_x1 = 0; best_y1 = 0; best_x2 = 1; best_y2 = 1;
        // score = -(-1) = 1? Actually a=0, b=0 -> score 1.
        // count_diff returns 0. best_score init is -1e6.
        // If no mackerels, we should output something valid.
    }

    // Randomized search to find a better rectangle
    // We center rectangles around mackerels and vary the size.
    // The problem guarantees N=5000, so O(N) per check is efficient enough.
    // We run ~3000 iterations.
    
    unsigned int seed = static_cast<unsigned int>(chrono::system_clock::now().time_since_epoch().count());
    mt19937 rng(seed);
    uniform_int_distribution<int> w_dist(1, 50000); // Random half-width
    uniform_int_distribution<int> h_dist(1, 50000); // Random half-height

    // Total length constraint: 4e5. 
    // Rectangle perimeter = 2*(w + h). 
    // If we choose half-width w <= 50000, full width W <= 100000.
    // Max perimeter = 2*(100000 + 100000) = 400000. 
    // This fits perfectly.
    
    for (int t = 0; t < 3000; ++t) {
        int idx = rng() % N;
        int cx = macks[idx].x;
        int cy = macks[idx].y;

        int wx = w_dist(rng);
        int wy = h_dist(rng);

        int lx = max(0, cx - wx);
        int rx = min(100000, cx + wx);
        int ly = max(0, cy - wy);
        int ry = min(100000, cy + wy);

        // Check if the rectangle degenerates to a point or line (lx > rx or ly > ry)
        // With wx >= 1 and clamping to [0, 100000], it's possible if cx is at edge and wx is large.
        // e.g. cx=0, wx=1 -> lx=0, rx=1.
        // lx <= rx is guaranteed since cx is between cx-wx and cx+wx? No.
        // lx = cx - wx, rx = cx + wx.
        // Since wx >= 1, lx <= cx - 1 < cx + 1 <= rx.
        // So lx < rx holds unless clamping squeezes them?
        // If cx=0, lx=0, rx=min(100000, wx). Since wx>=1, rx>=1. lx < rx.
        // So lx < rx is guaranteed.
        // Similarly ly < ry.
        
        int val = count_diff(lx, ly, rx, ry);
        if (val > best_score) {
            best_score = val;
            best_x1 = lx; best_y1 = ly;
            best_x2 = rx; best_y2 = ry;
        }
    }
    
    // Determine vertices for the output rectangle
    // Order: (x1,y1) -> (x2,y1) -> (x2,y2) -> (x1,y2)
    cout << "4\n";
    cout << best_x1 << " " << best_y1 << "\n";
    cout << best_x2 << " " << best_y1 << "\n";
    cout << best_x2 << " " << best_y2 << "\n";
    cout << best_x1 << " " << best_y2 << "\n";

    return;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    solve();
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=1015)

````
I need to construct an orthogonal polygon (edges parallel to axes) that maximizes mackerel - sardine points inside, where points on edges count as inside.

Key insight: The score is `max(0, mackerels_inside - sardines_inside + 1)`, so I want to maximize the net mackerels minus sardines with at least one extra point added.

A key geometric observation: If two points share the same x-coordinate and are close vertically, OR if two points share the same y-coordinate and are close horizontally, then a small rectangle or line can potentially include both while keeping the cost low.

The approach uses **vertical segments** as the primary building block:
1. Group fish by x-coordinate (same vertical lines)
2. Within each x-column, find pairs of adjacent sorted positions (one fish up, one down)
3. Connect them with vertical segments covering the gap between them
4. For each valid segment, calculate potential net gain and greedily select the best ones
5. After selecting horizontal segments, add a large square to ensure base score of at least 1

Let me implement this:
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    cin.tie(NULL);
    ios::sync_with_stdio(false);

    const int MAXX = 100005;
    vector<pair<int,int>> mac[MAXX], sar[MAXX]; // mac[y][x] stores all fish at x-coordinate
    for (int i = 0; i < 5000; ++i) { cin >> X >> Y; mac[X].push_back({Y, 0}); }
    for (int i = 0; i < 5000; ++i) { cin >> X >> Y; sar[X].push_back({Y, 1}); }

    for (auto& v : mac) sort(v.begin(), v.end());
    for (auto& v : sar) sort(v.begin(), v.end());

    long long max_score = 0, max_net = INT_MIN;

    for (int x = 0; x < MAXX; ++x) {
        if (mac[x].empty() && sar[x].empty()) continue;

        int n = (int)(mac[x].size() + sar[x].size());
        vector<tuple<int,int,int>> segs(n-1);
        for (int i = 0; i < n-1; ++i) {
            double dy = abs((double)segs[i].get<1>() - segs[i+1].get<1>()) * 0.5 - 0.5;
            double mx_gain = dy * mac[x].size() - 9e10 / dy;
            double sc_gain = dy * sar[x].size() - 9e10 / dy;
            int m_count = round(mx_gain);
            int s_count = round(sc_gain);
            if (m_count < 0 || s_count < 0) m_count = s_count = 0;
            segs[i] = make_tuple(i*4+1, (int)m_count, (int)s_count);
        }

        string dir = "asc";
        auto [best_seg, best_m, best_s] = greedy_max_segs(segs, mac[x], sar[x]);
        if (best_seg >= 0 && best_m > 0 && best_s > 0) {
            int p[8] = {x,y,y,h,x,low,y,y};
            cout << (best_seg == 0 ? 2 : 4) << endl;
            cout << x << ' ' << y++ << '\n';
            cout << h << ' ' << y << '\n';
            cout << x << ' ' << yhigh << '\n';
            cout << low << ' ' << yhigh << '\n';
            cout << x << ' ' << y << '\n';
        } else if (best_seg == 0 && best_m > 0) {
            cout << 2 << endl;
            cout << x << ' ' << y << '\n';
            cout << h << ' ' << yhigh << '\n';
        } else {
            cout << 4 << endl;
            cout << x << ' ' << yhigh << '\n';
            cout << x+1 << ' ' << yhigh << '\n';
            cout << x+1 << ' ' << yhigh-1 << '\n';
            cout << x << ' ' << yhigh-1 << '\n';
        }
    }
}
```
````
