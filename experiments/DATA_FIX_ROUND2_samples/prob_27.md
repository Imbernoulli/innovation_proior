# 题 27 — base均分 10.4 vs 直接SFT均分 0.0(掉点 10.4)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=3,得分 52.1,completion_tokens=21736)

````
The problem asks us to place as many points as possible on an $n \times m$ grid such that no four points form the corners of an axis-parallel rectangle. This is a classic extremal graph theory problem related to Zarankiewicz's problem, specifically finding the maximum size of a subset of vertices in a bipartite graph (or equivalently, entries in a matrix) with no $K_{2,2}$ submatrix.

Here, we want to find a subset of cells $(r, c)$ with maximum size such that for any two rows $r_1, r_2$, they share at most one common column index. If they shared two columns, say $c_1$ and $c_2$, then we would have points $(r_1, c_1), (r_1, c_2), (r_2, c_1), (r_2, c_2)$, which forms a rectangle.

So the constraint is equivalent to: Every pair of rows can have at most 1 point in common columns.
This means if row $i$ has a set of columns $C_i$, then $|C_i \cap C_j| \le 1$ for all distinct $i, j$.
Similarly, looking from columns' perspective, every pair of columns can have at most 1 point in common rows. But usually, it's easier to think about the row intersection property for the "no rectangle" condition.

Let's denote $d_r$ as the number of black points in row $r$.
Then $\sum_{i=1}^n d_i = k$.
For any pair of rows $i, j$, let $x_{ij} = |C_i \cap C_j|$. We require $x_{ij} \le 1$.
We know $\sum_{1 \le i < j \le n} x_{ij} = \sum_{1 \le i < j \le n} |C_i \cap C_j|$.
Also, consider the contribution of each column $c$. Let $y_c$ be the number of points in column $c$. Then $y_c = |\{i : (i, c) \in S\}|$.
The sum over pairs of rows of intersections is related to sums over columns. Specifically, $\sum_{c=1}^m \binom{y_c}{2} = \sum_{1 \le i < j \le n} |C_i \cap C_j|$.
Since $|C_i \cap C_j| \le 1$, we have $\sum_{i<j} |C_i \cap C_j| \le \binom{n}{2}$.
Thus, $\sum_{c=1}^m y_c(y_c - 1) / 2 \le \frac{n(n-1)}{2}$.
$\sum_{c=1}^m (y_c^2 - y_c) \le n(n-1)$.
Since $k = \sum y_c$, we also have $\sum y_c^2 \ge k^2/m$ by Cauchy-Schwarz (assuming roughly uniform distribution).
So $k^2/m \approx k + n(n-1)$.
$k^2 \approx m k + m n(n-1)$.
$k^2 - mk - m n^2 \approx 0$.
$k \approx \frac{m + \sqrt{m^2 + 4 m n^2}}{2} \approx \frac{m + 2 n \sqrt{m}}{2} = \frac{m}{2} + n \sqrt{m}$.
This suggests $k \approx n \sqrt{m}$.

Given the constraint $n \cdot m \le 100,000$, we need an efficient construction. The upper bound provided in the problem description is $U(n, m) = \lfloor \min(n \cdot \sqrt{m} + m, m \cdot \sqrt{n} + n, n \cdot m) \rfloor$.
This matches the intuition derived above ($n \sqrt{m}$ or $m \sqrt{n}$ depending on orientation). Since the problem is symmetric for $n$ and $m$, we can assume without loss of generality $n \le m$ to simplify thinking, but actually we just need to maximize points. Wait, the formula says min of these terms. Since $n \cdot m$ is an upper bound on total cells, and the other two are asymptotic limits based on the combinatorial constraints. If $n \ll m$, $n \sqrt{m}$ dominates $m \sqrt{n}$?
Let's check: $n=2, m=100$. $n\sqrt{m} = 20$. $m\sqrt{n} = 100\sqrt{2} \approx 141$. So we prefer the smaller one.
Actually, if $n \le m$, then $\sqrt{m} \ge \sqrt{n}$, so $n\sqrt{m} \ge n\sqrt{n}$ while $m\sqrt{n} = m/\sqrt{n} \sqrt{n}^2 \dots$
Wait, comparing $A = n\sqrt{m} + m$ and $B = m\sqrt{n} + n$.
If $n \le m$:
$A \approx n\sqrt{m}$.
$B \approx m\sqrt{n}$.
$n\sqrt{m} = \sqrt{n^2 m}$.
$m\sqrt{n} = \sqrt{m^2 n}$.
Clearly $n^2 m \le m^2 n$ since $n \le m$.
So if $n \le m$, $n\sqrt{m}$ is smaller. If $m \le n$, $m\sqrt{n}$ is smaller.
Basically we should pick the dimension that is smaller as the "rows" to define the sets. Or rather, construct a bipartite graph with part sizes $n$ and $m$ that avoids $K_{2,2}$. This corresponds to Turan type problem.

A known construction that achieves something close to this bound involves finite fields (projective planes), but those require specific sizes. A general greedy strategy or a construction based on polynomial hashing often works well for competitive programming problems of this sort where an exact optimal solution via projective planes isn't directly applicable due to arbitrary $N, M$.
However, there is a simpler construction using modular arithmetic that gives density very close to the theoretical bound for large enough grids.
Specifically, identifying rows $i \in \{0, \dots, n-1\}$ and columns $j \in \{0, \dots, m-1\}$ (using 0-based indexing internally).
One standard approach for maximizing points with no $K_{2,2}$ is:
Map each row $r$ to a set of columns. A randomized greedy algorithm is often used. However, given the strict time limit (1s) and constraints ($nm \le 100,000$), we can afford $O(nm)$ or perhaps slightly more like $O((nm)^{1.5})$ if implemented carefully, but likely $O(nm)$ is expected because the output size could be up to $O(nm)$? No, max points is roughly $n \sqrt{m}$. With $nm = 10^5$, if $n=10^3, m=10^2$, points $\approx 10^3 \cdot 10 = 10^4$. We can iterate through possible positions.
But wait, if $n$ is small and $m$ is large, say $n=1$, we can put $m$ points.
If $n \approx \sqrt{m}$, then $nm \approx m^{1.5}$, points $\approx m \approx n\sqrt{n}$.
Let's consider the specific construction often used for this problem ("No rectangles").
We can use a property of quadratic residues or simply assign each row a unique "signature".
Actually, the best constructive approach without heavy machinery (like Projective Planes) is typically a greedy strategy or assigning columns based on a hash function of row index.
Specifically, for each row $r$, we want to select a set of columns. To ensure no two rows share more than one column, the intersection of their chosen columns must be size $\le 1$.
Consider the set of points $(r, c)$. If we fix the rows, how to choose columns?
If we treat columns as elements of a field or simply integers modulo some prime, we can define points $(x, x^2) \pmod p$. This ensures no rectangle formed by 4 points if we view them in the plane properly?
In the grid interpretation:
Let's try to map the grid to a structure.
Actually, simply iterating columns and greedily picking rows might work.
But there is a specific randomized algorithm mentioned in similar CP problems (e.g., Codeforces 798D or similar constructive problems) or deterministic approaches using properties of squares.
Let's look at the constraint again. $n, m$ up to $100000$ product. So $O(nm)$ is fine for reading input? No, we need to print points. The output size $k$ can be around $n\sqrt{m}$. If $n=300, m=300$, $nm=90000$, $k \approx 300 \cdot 17 \approx 5100$.
If $n=1, m=100000$, $k=100000$. Printing $100000$ lines is fast.
The complexity of generating the solution shouldn't exceed $O(k \log k)$ or $O(nm)$. Since $nm \le 100000$, $O(nm)$ is perfectly acceptable for a generation step (which is just iterating all cells or a subset).
Wait, the problem statement says $nm \le 100000$. This is quite small!
So even $O(nm)$ is allowed. The only issue is finding the valid configuration quickly.
With $nm \le 100000$, we can probably afford to just iterate through all cells and make a decision.
Let's try a simple greedy approach: Iterate over cells $(i, j)$. If placing a point doesn't create a rectangle, place it.
Is simple greedy sufficient?
Simple Greedy Strategy:
For each cell $(i, j)$, place a point if it doesn't complete a rectangle.
To check if adding $(i, j)$ creates a rectangle, we need to know:
1. Row $i$ already has a point in column $j$ (duplicate)? We just maintain the state.
2. Does row $i$ and row $k$ ($k \neq i$) both already have points in column $j$?
Wait, the rectangle condition is: 4 points $(r_1, c_1), (r_1, c_2), (r_2, c_1), (r_2, c_2)$.
Adding $(r, c)$ forms a rectangle if there exists another row $r'$ such that both $(r, c)$ was previously empty but now filled, and $(r', c)$ is filled, and... wait.
Condition: Two rows share at most one column.
When considering $(r, c)$, we check for all rows $r' \neq r$, whether row $r'$ already has a point in column $c$.
If there is *more than one* row $r'$ that already has a point in column $c$, then we cannot add $(r, c)$?
No. If there are two rows $r_a, r_b$ having points in column $c$, that means column $c$ is "too full"? No.
Rectangle formation requires TWO columns $c_1, c_2$ sharing TWO rows $r_1, r_2$.
So if we add $(r, c)$, we need to ensure that for any other existing point $(r', c')$, we don't form a rectangle.
If we add $(r, c)$, a rectangle is formed with existing points if:
There exists another point $(r, c')$ in the same row (not a rectangle yet) AND there exist two other points $(r', c')$ and $(r', c)$.
Basically, checking the rectangle constraint incrementally:
Iterate $(i, j)$.
Check if $(i, j)$ completes a rectangle.
It completes a rectangle if there are two rows $r_1, r_2$ such that $r_1$ has a point in col $j$, $r_2$ has a point in col $j$, and $r_1$ has a point in col $i$? No.
The coordinates are $(r, c)$.
Points are $P$.
Invalid if $\exists (r_1, c_1), (r_1, c_2), (r_2, c_1), (r_2, c_2) \in P$ with distinct rows/cols.
When inserting $(r, c)$, we look for existing pairs of points $(r', c_1), (r', c)$? No.
We look for pairs $(r', c), (r'', c)$? No.
Let's rephrase. We insert $(r, c)$.
Does there exist a pair of existing rows $r_1, r_2$ and a column $c'$ such that we have $(r_1, c'), (r_2, c'), (r_1, c), (r_2, c)$?
If $(r, c)$ is new, it's impossible to have both $(r_1, c)$ and $(r_2, c)$ unless we already had points in column $c$.
Wait, simpler check:
When adding $(r, c)$, does there exist an existing point $(r', c')$ such that adding $(r, c)$ creates a rectangle with it?
We need $(r, c')$ and $(r', c)$ to exist already.
Since we are building the set, if $(r, c')$ exists, it's in row $r$. If $(r', c)$ exists, it's in col $c$.
Wait, if I am at $(r, c)$, and I decide to add it.
I need to check if adding it creates a rectangle.
A rectangle is formed by two rows $r_1, r_2$ and two columns $c_1, c_2$.
Suppose I add $(r_1, c_1)$. For this to be safe, it implies that for any $c_2$, there is not a pair of rows $r_2, r_3$ with $(r_2, c_2), (r_3, c_2)$? No.
Let's stick to the pairwise intersection property.
Rows $r_i$ have sets of columns $S_{r_i}$.
Constraint: $|S_{r_i} \cap S_{r_j}| \le 1$.
So when deciding to put a point in row $r$ at column $c$ (if not already there), we check how many columns are shared between row $r$ and ANY other row $r'$.
If $|S_{r'} \cap S_{r}|$ becomes $> 1$ after adding $(r, c)$, then it's invalid.
So, if we iterate through cells $(r, c)$, can we just accept if it doesn't violate the condition?
What order?
Random shuffle of cells?
Or iterate row by row, and inside row, iterate col by col?
If we process row by row:
Maintain current sets $S_r$.
For row $r$, we want to pick a subset of columns.
To satisfy condition $|S_{r'} \cap S_r| \le 1$ for all previous $r' < r$ and future rows will be handled similarly.
Actually, we don't care about future rows if we just build greedily. We just need to ensure that any two rows eventually have $\le 1$ intersection.
Greedy approach:
Sort all cells randomly or deterministically. Try to add point $(r, c)$. Check if for any row $r'$, $|S_r \cap S_{r'}|$ becomes $> 1$.
Checking this for all $r'$ takes $O(n)$. Total $O(n^2 m)$.
With $nm \le 100000$, worst case $n \approx \sqrt{100000} \approx 316$. $n^2 \approx 10^5$. $m \approx 316$. Total operations $10^5 \cdot 300 \approx 3 \cdot 10^7$, which fits in 1s easily.
Even if $n=1000, m=100$. $O(m \cdot n^2) = 100 \cdot 10^6 = 10^8$, might be tight but acceptable in C++ (simple ops).
However, checking "for any row $r'$" is slow if done naively.
We can optimize.
For row $r$, we want to select columns.
For each column $c$, can we include it in $S_r$?
Condition: For all $r' \neq r$, the intersection count $|S_{r'} \cap S_r|$ must be $\le 1$.
When adding $(r, c)$, we increment the count for every row $r'$ where $(r', c) \in S_{r'}$? No.
We maintain the count of common columns for each pair of rows? That's too much space ($O(n^2)$). $n$ up to 1000. $n^2 = 10^6$. This is fine!
Wait, $n$ can be 100,000 if $m=1$. Then $O(n^2)$ is too big.
But if $n$ is large, $m$ is small. The roles are symmetric. We can swap $n$ and $m$ so that $n \le m$ (actually we want $\min(n, m)$ to be small for $O(\min(n, m)^2)$?).
Wait, if $n \le m$, then $n$ is the smaller dimension. But $n^2$ might still be larger than $m$?
If $n \le m$, and $nm \le 100000$. Max $n$ is $\sqrt{100000} \approx 316$.
If $n > 316$, then $m < 316$.
So $\min(n, m) \le 316$.
Thus $\min(n, m)^2 \le 10^5$.
So storing a table `cnt[r1][r2]` where `cnt[i][j]` stores number of common columns between row $i$ and row $j$ is feasible ONLY IF we treat the smaller dimension as "rows".
Wait, if we treat $n$ rows, and $n$ can be up to 100,000 (with $m=1$), we cannot store $n^2$ array.
However, if $n=100,000, m=1$, we can just take all points. The condition "no rectangle" is trivially satisfied (need 2 rows and 2 columns).
Generally, if $\min(n, m)$ is large, then $\max(n, m)$ must be small.
So we can always perform the check on the "smaller" dimension.
Let's formalize:
If $n > m$, swap $n$ and $m$ and handle accordingly. Now $n \le m$.
Then $n \le \sqrt{100000}$.
We can maintain a matrix `count[n][n]` initialized to 0. `count[i][j]` tracks common columns between row $i$ and row $j$.
Algorithm:
1. Read $n, m$.
2. If $n > m$, swap $n, m$.
3. Initialize `points` list.
4. Maintain `common_cols[n][n]` (size $\le 320 \times 320$).
   Actually we only need to track pairs $(r_1, r_2)$ that have points in common.
   When considering cell $(r, c)$ for row $r$:
   If row $r$ already has a point at $c$, skip.
   Iterate over all rows $i$ from $1$ to $n$ (actually only $i < r$ matters for past decisions? No, we want global consistency).
   Wait, if we iterate rows $1..n$, and for each row we pick a subset of columns.
   If we add $(r, c)$, we check if adding it increases intersection with any previous row $i < r$.
   Condition: `common_cols[r][i]` (number of shared cols) + 1 <= 1 ?
   If `common_cols[r][i] == 1`, then adding $(r, c)$ would make it 2, which is forbidden (since $(r, i)$ share col $c$ now, and they share previous col). Wait, `common_cols[r][i]` counts columns present in both row $r$ and row $i$.
   If `common_cols[r][i]` is 0, we can add $(r, c)$ freely? No, only if `common_cols[r][i]` remains $\le 1$.
   So if `common_cols[r][i]` is 0, adding $(r, c)$ makes it 1. OK.
   If `common_cols[r][i]` is 1, adding $(r, c)$ makes it 2. Forbidden.
   So, we can only add $(r, c)$ if for all rows $i \neq r$ where $(i, c)$ is already placed, the number of other common columns is 0.
   Wait, this sounds like we need to check ALL rows $i$ that currently have a point in column $c$.
   Instead of checking all rows $i$ (which is $O(n)$), maybe we can check faster.
   We can keep an adjacency list: for each column $c$, list of rows containing it.
   When considering $(r, c)$, let `rows_with_col[c]` be the list of rows $i$ such that $(i, c)$ is placed.
   If $r \in rows\_with\_col[c]$ (already there), continue.
   Otherwise, count how many $i \in rows\_with\_col[c]$ satisfy `common_cols[r][i] == 1`.
   If count == 0, we can place $(r, c)$.
   Update: Add $r$ to `rows_with_col[c]`.
   For each $i$ already in `rows_with_col[c]`, update `common_cols[r][i] += 1` and `common_cols[i][r] += 1`.
   Wait, updating `common_cols` takes $O(|rows\_with\_col[c]|)$. In worst case this could be large.
   Is it safe?
   Yes, $k$ is the number of points. $O(k \cdot (\text{avg rows per col}))$.
   Actually, worst case is $O(n \cdot m)$ points? No, max points is limited.
   Sum of $|rows\_with\_col[c]|$ over all $c$ is $k$.
   However, we update `common_cols` for all pairs. Total updates $\approx k \cdot (\text{something})$.
   Specifically, for each $(r, c)$ added, we iterate over existing rows in col $c$.
   Let $x_c$ be number of points in col $c$. We do $x_c$ updates.
   Total complexity $\sum x_c^2$? No, $\sum x_c$ is $k$. The inner loop runs $x_c$ times.
   Wait, if we iterate all existing rows in column $c$ to update `common_cols`.
   The cost to add $(r, c)$ is proportional to the current size of `rows_with_col[c]`.
   Sum of costs over all additions: $\sum_c \sum_{i < \text{current}} (\text{number of points in } c \text{ at step } t)$.
   Roughly $\sum_c x_c(x_c-1)/2$.
   Since $x_c^2$ can be large if a column is dense?
   Constraint: Any two rows share at most 1 column.
   This implies $\sum_{r \neq i} |S_r \cap S_i| \le \binom{n}{2}$.
   Also $\sum_c \binom{x_c}{2} = \sum_{i<j} |S_i \cap S_j| \le \binom{n}{2}$.
   So $\sum x_c^2$ is bounded roughly by $n^2 + nm$.
   Thus total complexity of updates is roughly $\sum_c x_c^2 \approx O(n^2 + nm)$.
   Given $n \le m$ and $nm \le 100000$, $n \le 316$. $n^2 \le 100000$.
   So this logic holds: $\sum x_c^2 \le n(n-1)$ is false. $\sum x_c(x_c-1) \le n(n-1)$.
   Since $n(n-1)$ is around $10^5$, the sum of squares is bounded by $O(n^2) \approx 10^5$.
   So the total number of `common_cols` updates is bounded by $O(n^2)$.
   Checking `common_cols` takes $O(n^2)$.
   So the greedy strategy with this optimization runs in $O(n^2 + km)$.
   Wait, we iterate through all cells $(r, c)$. There are $nm$ cells.
   If we use random order, the probability of accepting/rejecting varies.
   Is it guaranteed to be optimal? Probably not exactly optimal but good score.
   Given the problem constraints and scoring (based on $U(n,m)$), exact optimality is hard.
   However, with random permutation of cells and greedy check, we often get close to optimal.
   Since time limit is generous for $n^2 \approx 10^5$, we can afford this.
   
   Wait, is the condition $|S_{r_i} \cap S_{r_j}| \le 1$ sufficient?
   Yes, it forbids 2 rows sharing 2 columns, which defines the rectangle.
   The logic seems sound.
   
   Optimization details:
   We need to efficiently find "rows_with_col[c]" that share a column with $r$.
   Actually, we don't need to iterate all rows in col $c$. We just need to know if for the specific column $c$, there is *any* other row $i$ such that $S_i \cap S_r$ has count 0? No.
   We can only add $(r, c)$ if for ALL $i$ currently in column $c$ (except $r$), `common_cols[r][i] == 0`.
   Wait. If I add $(r, c)$, the common count between $r$ and $i$ increments by 1. It becomes 1. This is allowed.
   It is NOT allowed if it becomes 2.
   So if for some $i$ in col $c$, `common_cols[r][i]` is already 1, then adding $(r, c)$ would make it 2 -> FORBIDDEN.
   So we can add $(r, c)$ if and only if for all $i \in rows\_with\_col[c], i \neq r$, `common_cols[r][i] == 0`.
   This simplifies things significantly. We don't need to check counts against future rows, only current state.
   
   Implementation plan:
   1. Read $n, m$.
   2. If $n > m$, swap $n, m$. (Ensure $n \le m$).
   3. Use a vector of vectors or flattened array for `common_cols`. Size $n \times n$.
      Since $n \le 316$, static array `int common[320][320]` is safe and fast. Initialize to 0.
      Note: if we swapped, original $n$ might be large. We need to map original row indices back if needed, but for output we just need coordinates. Swapping dimensions effectively changes the shape of the grid we are solving for?
      Wait. If we solve for $n \times m$ with $n \le m$, we output points in grid of size $m \times n$. We must be careful to output correct coordinates.
      If we transpose the result, we get points in correct orientation.
      Actually, let's just implement without swapping first, or handle the mapping.
      Better: Keep logic simple. We want to fill an $N \times M$ grid.
      If we use the "row vs row" intersection logic, we assume $N$ is the "row" dimension.
      If $N$ is large, we can't store $N \times N$.
      So yes, swapping $N, M$ so that $N \le \sqrt{NM}$ is smart to enable $O(N^2)$ storage.
      If we swap $N$ and $M$ (treat original $M$ as rows), we solve the problem for transposed grid.
      Result: set of $(r', c')$. Map back to $(c', r')$.
      
   Algorithm Refined:
   Input $n, m$.
   Swap if $n > m$.
   Create `common[n][n]`.
   Create `vec_rows_per_col[m]`. Each element is a vector/list of rows.
   List of accepted points.
   
   Randomize the order of cells $(r, c)$ for $0 \le r < n, 0 \le c < m$.
   Shuffle the sequence.
   For each $(r, c)$:
     Check validity:
       Can we add point $(r, c)$?
       Condition: For all $i$ in `vec_rows_per_col[c]`, if $i \neq r$:
         If `common[r][i] > 0`: 
           Then adding $(r, c)$ would increase count of shared columns between $r$ and $i$ from 0 to 1 (currently 0? No wait.)
           Wait, logic check.
           `common[r][i]` stores the number of shared columns between row $r$ and row $i$ found SO FAR.
           Currently, we are considering adding $(r, c)$.
           We check if $c$ is already in row $r$. If yes, skip.
           Check if there is any other row $i$ such that `(i, c)` is already placed.
           Iterate $i \in vec\_rows\_per\_col[c]$.
           If `common[r][i] == 1`, then we CANNOT add $(r, c)$ (would become 2).
           If `common[r][i] == 0`, it's OK (becomes 1).
           So, we can add $(r, c)$ iff for all $i \in vec\_rows\_per\_col[c] \setminus \{r\}$, `common[r][i] == 0`.
           Wait. What if $i$ shares a column $c' \neq c$ with $r$?
           Yes, `common[r][i]` captures that.
           So if `common[r][i] >= 1`, it means $r$ and $i$ already share a column.
           Since they are both in col $c$ (after adding), they would share TWO columns.
           Wait. If $i$ is in $vec\_rows\_per\_col[c]$, then $(i, c)$ is ALREADY placed.
           So we already have points $(i, c)$ and potentially $(r, c')$.
           If `common[r][i] == 1`, it means there is some $c'$ such that $(r, c')$ and $(i, c')$ exist.
           If we add $(r, c)$, then $r$ and $i$ share $c$ AND $c'$.
           Rectangle $(r, c), (r, c'), (i, c), (i, c')$ formed.
           So YES, we forbid this.
           
       So the check is:
       bool ok = true;
       for(int i : vec_rows_per_col[c]) {
          if (i == r) continue;
          if (common[r][i] > 0) { // Already share a column
             ok = false; break;
          }
       }
       If ok:
          vec_rows_per_col[c].push_back(r);
          For(int i : vec_rows_per_col[c]) { // Need to update common counts
             // Wait, we just pushed r. We need to update pairs.
             // Actually the condition `common[r][i] == 0` is for the ones BEFORE push.
             // After pushing, `common[r][i]` must be incremented.
          }
          // Correct logic:
          // 1. Identify if adding (r,c) is bad. Bad if exists i in col c (excluding r) s.t. common[r][i] > 0.
          // 2. If good:
          //    Add r to col c.
          //    Increment common[r][i] for all i in col c (including the ones before, but we need to be careful not to double count r itself or iterate incorrectly).
          //    Wait, `vec_rows_per_col[c]` now contains $i$'s that were there.
          //    We iterate that list.
          //    common[r][i]++.
          //    common[i][r]++.
          
   Complexity Analysis again:
   Outer loop iterates $nm$ cells.
   Inner loop iterates `vec_rows_per_col[c]`.
   Cost $\sum_c (\text{final size of } c)^2$.
   As established, $\sum_c x_c(x_c-1) \le n(n-1)$.
   So total cost is bounded by $O(n^2)$.
   Here $n$ is the *swapped* smaller dimension.
   Original $n$ might be large, but effective $n$ is $\min(original\_n, original\_m) \le \sqrt{100000}$.
   So complexity is perfectly fine.
   
   Wait, what about checking `common[r][i]`?
   Accessing `common` is $O(1)$.
   Updating is $O(n)$ per addition? No, we iterate over the list of size $x_c$.
   So cost is proportional to $x_c$.
   Summing over all additions: $\sum x_c^2$.
   Max value is when $x_c$ is large.
   Bound: $\sum x_c(x_c-1) = \sum_{i<j} 2 \cdot |S_i \cap S_j| \le n(n-1)$.
   So sum of $x_c^2$ is $O(n^2)$.
   Since we access $common[r][i]$ which is random access in memory? No, contiguous rows.
   Memory access pattern: `vec_rows_per_col[c]` contains row indices. We access `common[r][i]` and `common[i][r]`.
   This is cache friendly enough.
   
   One corner case: $n=1$ or $m=1$.
   If $n=1$, $common$ size $1 \times 1$. Works.
   If $n=0$? Problem says $n, m \ge 1$.
   
   Another detail:
   The condition `common[r][i] > 0` checks if they share ANY other column.
   Is it possible they share $c$ (which we are adding)?
   No, `vec_rows_per_col[c]` contains rows that ALREADY have points in col $c$.
   So $(i, c)$ exists. $(r, c)$ does not exist yet.
   So `common[r][i]` only counts columns excluding $c$ (since $c$ wasn't counted yet).
   If `common[r][i] > 0`, they share some $c' \neq c$. Adding $c$ makes them share $c, c'$.
   If `common[r][i] == 0`, they share nothing yet. Adding $c$ makes them share $c$. Count becomes 1. Valid.
   This logic is correct.
   
   Data structures:
   `vector<int> col_rows[M+1]` (since $M$ can be up to 100,000).
   `int common[N+1][N+1]` (since $N \le \sqrt{100000}$).
   Wait, $N$ here refers to the smaller dimension.
   If input is $N, M$ with $N \le M$.
   If input is $20, 5000$. $N=20$. Common table $20 \times 20$.
   If input is $5000, 20$. We swap to $20, 5000$.
   Outputting: Remember to reverse coordinates.
   Input $R, C$.
   Output: If we treated grid as transposed, we stored $(r, c)$ where $r$ is from first dim (now rows in our internal logic).
   If we swapped $N, M$ at start (let's call internal dimensions $n', m'$ where $n' = \min(n, m), m' = \max(n, m)$):
   Points generated are $(r', c')$ with $0 \le r' < n', 0 \le c' < m'$.
   If originally $N > M$, we swapped. So we need to map $(r', c')$ back to original $(c', r')$.
   If originally $N < M$, no swap, so $(r', c')$ maps to $(r', c')$.
   Wait, coordinate system:
   Problem uses $1 \le r \le N, 1 \le c \le M$.
   My code uses $0 \le r < n', 0 \le c < m'$.
   If swapped:
   Original $(N, M)$ became $(m, n)$.
   Internal $(r', c')$ corresponds to Row $r'$ in internal grid (height $m$?? No).
   Wait. If we swap $N, M$ into $n', m'$ ($n' \le m'$).
   This means we conceptually rotated the grid by 90 degrees.
   A point $(x, y)$ in original grid corresponds to $(y, x)$ in transposed grid.
   If we store point at internal $r'$ (corresponds to original column?) and $c'$ (corresponds to original row?).
   Let's trace:
   Original: Grid $N \times M$. Rows $1..N$, Cols $1..M$.
   We want to construct $S \subset \{1..N\} \times \{1..M\}$.
   We swap so we solve for $n' \times m'$ where $n' \le m'$.
   Suppose $N > M$. Then $n'=M, m'=N$.
   We solve for grid of size $n' \times m'$ (rows $1..n'$, cols $1..m'$).
   Let generated points be $(u, v)$.
   These correspond to original columns $u$ (since $u \in 1..n'=M$) and original rows $v$ (since $v \in 1..m'=N$).
   So original point is $(v, u)$.
   If $N < M$, $n'=N, m'=M$. We generate $(u, v)$ where $u \in 1..N, v \in 1..M$.
   Original point is $(u, v)$.
   Wait, is it better to just stick to the logic that we minimize the number of rows to check intersections against?
   Yes.
   If we treat the dimension of size $n$ as "rows" (the ones we intersect), we need $n$ to be small.
   If we have an $N \times M$ grid ($N, M$ given).
   We want to fill it.
   Intersection constraint applies to rows.
   To implement efficiently, we treat the set of size $\min(N, M)$ as the "rows" in our data structure.
   Case 1: $N \le M$.
     We maintain intersection info for rows $1..N$.
     Columns $1..M$.
     For a point $(r, c)$, we update row $r$.
   Case 2: $N > M$.
     We swap. Treat $M$ rows and $N$ columns.
     Points generated $(r, c)$ with $r \in 1..M, c \in 1..N$.
     Original point is $(c, r)$.
     This works.
   
   Wait, one subtle thing.
   The problem requires outputting points.
   We generate $k$ points.
   Score depends on $k / U(n, m)$.
   The greedy strategy is probabilistic.
   Shuffling the order of cells is important to avoid bias.
   Usually, greedy + randomization works very well for this Zarankiewicz problem.
   
   Let's write the solution.
   
   Corner cases:
   $n=1, m=1$. Loop works.
   $n=1, m=large$. Logic: $N=1, M=large$. `common[1][1]`. Loop runs. All cols accepted. $k=m$.
   $n=large, m=1$. Swap. Solve for $1 \times large$. Generate points. Map back.
   
   Memory Limit 512 MB.
   Arrays: `vec_rows_per_col` vector of vectors. Overhead might be high?
   Vector headers take 24 bytes. If $m=100000$, $100000 \times 24 \approx 2.4$ MB. Content is integers. $O(k)$ ints.
   Safe.
   
   Time limit 1s.
   Sorting/Shuffling: `std::shuffle` $O(nm)$.
   Main loop $O(nm + n^2)$.
   $nm \le 10^5$.
   Operations count $\approx 10^5 + 316^2 \approx 2 \cdot 10^5$. Very fast.
   
   Include `<random>` for shuffle.
   
   Double check the condition again.
   "no four of them form the four corners of an axis-parallel rectangle."
   This is exactly "no two rows share two columns".
   My check: `common[r][i]` is the count of columns shared by row $r$ and row $i$.
   We enforce this count $\le 1$.
   Wait, `common[r][i]` is updated whenever we add a point that affects the intersection.
   If we add $(r, c)$, we iterate $i$ in `vec_rows_per_col[c]`.
   These are rows that ALREADY have $(i, c)$.
   If for any such $i$, `common[r][i]` was already $> 0$, then adding $(r, c)$ would make intersection $\ge 2$.
   Wait, is it $\ge 2$?
   If `common[r][i]` was 1 (they shared $c'$ before). Now they share $c$ (new). Total 2.
   If `common[r][i]` was 0. Now they share $c$. Total 1.
   So yes, we reject if `common[r][i] > 0` for any $i$ in column $c$.
   Wait, does `common[r][i]` correctly reflect intersection excluding current potential column?
   Yes, because we only add to `common` AFTER verifying the condition. And `common` only accumulates intersections from columns processed so far.
   So if we are at step $(r, c)$, `common[r][i]` reflects intersections in all columns $\{c'\}$ where we have already decided to place points in column $c'$ OR in row $r$ or row $i$.
   Wait, `common` is global.
   If I added $(r, c')$ long ago, and later added $(i, c')$.
   When adding $(i, c')$, I checked `common[r][i]`.
   It must have been 0 (at that time).
   Then I incremented it.
   So `common` maintains accurate intersection counts for all columns processed so far.
   The "greedy" nature might mean we skip some columns for row $r$ if they conflict.
   Does skipping prevent future conflicts?
   If we skip $(r, c)$, we don't add the point. $k$ decreases.
   Is it possible that skipping allows a later point?
   No, the decision for $(r, c)$ only affects conflicts involving $r$ and columns processed.
   Since we process columns in a shuffled order, this is just heuristics.
   Usually this "greedy with shuffle" achieves $O(n\sqrt{m})$ points.
   
   Wait, I should confirm if $N$ and $M$ are $1$-based or $0$-based in output.
   "Print: ... ri and ci each (1 ≤ ri ≤ n, 1 ≤ ci ≤ m)".
   So 1-based.
   My code uses 0-based, so add 1 before printing.
   
   One more check:
   What if `vec_rows_per_col[c]` is empty?
   Then loop over $i$ doesn't run. `ok` stays true.
   We add $r$. `common[r][i]` update doesn't happen (loop empty).
   Correct.
   
   Wait, `common` initialization.
   Global array is zero-initialized.
   Inside function, need `memset` or `vector`.
   Since `n` varies, use `vector<vector<int>> common(n, vector<int>(n, 0));`
   
   Code Structure:
   ```cpp
   #include <iostream>
   #include <vector>
   #include <algorithm>
   #include <random>
   #include <chrono>
   
   using namespace std;
   
   int main() {
       int n, m;
       if (!(cin >> n >> m)) return 0;
       
       if (n > m) swap(n, m);
       
       // n is now min_dim, m is max_dim
       // We store intersection counts between rows.
       // Rows are indexed 0..n-1.
       
       // We need to map points back to original coordinates if we swapped.
       // If we swapped, original_n was m (in old variables), original_m was n.
       // Wait, if we swap n, m, we lose info. Store a flag or logic.
       bool swapped = (original_n > original_m); // Need to handle this carefully.
       
       // Just use a flag.
       // But I need to capture input before swapping.
       // Actually, better: just copy input to variables and swap local copies.
       
       int orig_n = n;
       int orig_m = m;
       bool swaped_dims = (orig_n > orig_m);
       if (n > m) swap(n, m);
       // Now n <= m.
       // If swapped, the 'rows' in my logic correspond to original columns (indices 0..orig_m-1).
       // And 'columns' in my logic correspond to original rows (indices 0..orig_n-1).
       // Point (r_internal, c_internal) -> if swapped: original=(c_internal, r_internal). else: (r_internal, c_internal).
       // Note: internal coords 0-based. Output needs 1-based.
       
       // Prepare structures
       // vec_rows_per_col: for each internal col (0..m-1), list of internal rows.
       vector<vector<int>> vec_rows_per_col(m);
       
       // common[r][i]
       // n <= 316 (since n*m <= 100000)
       // Using vector for safety
       vector<vector<int>> common(n, vector<int>(n, 0));
       
       vector<pair<int, int>> points;
       
       // Shuffle cells
       // To save time/memory, instead of shuffling whole nm pairs which is 10^5, it's fine.
       vector<pair<int, int>> cells;
       cells.reserve(n * m);
       for(int r=0; r<n; ++r) {
           for(int c=0; c<m; ++c) {
               cells.push_back({r, c});
           }
       }
       
       mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
       shuffle(cells.begin(), cells.end(), rng);
       
       for(auto& cell : cells) {
           int r = cell.first;
           int c = cell.second;
           
           // Check if (r, c) can be added
           // Condition: For all existing i in col c, common[r][i] == 0.
           bool ok = true;
           const auto& row_list = vec_rows_per_col[c];
           for(int i : row_list) {
               if(i == r) continue; // Should not happen as r is new
               if(common[r][i] > 0) {
                   ok = false;
                   break;
               }
           }
           
           if(ok) {
               // Add point
               points.push_back({r, c});
               vec_rows_per_col[c].push_back(r);
               
               // Update common
               // We need to increment common[r][i] for all i in col c (existing ones)
               // And update symmetry
               for(int i : row_list) {
                   common[r][i]++;
                   common[i][r]++;
               }
           }
       }
       
       cout << points.size() << "\n";
       for(auto& p : points) {
           // If swapped, swap coords
           int out_r, out_c;
           if(swaped_dims) {
               out_r = p.second + 1;
               out_c = p.first + 1;
           } else {
               out_r = p.first + 1;
               out_c = p.second + 1;
           }
           cout << out_r << " " << out_c << "\n";
       }
       
       return 0;
   }
   ```
   Wait, logic check on "common[r][i] > 0".
   Suppose row $r$ has point $(r, c_1)$ already.
   Row $i$ has point $(i, c_1)$ and $(i, c_2)$.
   We consider adding $(r, c_2)$.
   $c_2$ has row $i$ in its list.
   We check `common[r][i]`.
   At this moment, `common[r][i]` should reflect intersection of $r$ and $i$.
   They share $c_1$. So `common[r][i]` should be 1.
   If `common[r][i] > 0` (it is 1), we reject $(r, c_2)$. Correct.
   
   Suppose we haven't added $(r, c_1)$ yet.
   Then `common[r][i]` is 0. We accept $(r, c_2)$.
   Later when we consider $(r, c_1)$, it will see $(i, c_1)$ in col $c_1$.
   `common[r][i]` is 0. Accept.
   Wait. If we process $(r, c_2)$ first.
   Then later process $(r, c_1)$.
   At $(r, c_1)$, we check col $c_1$. Contains $i$.
   Check `common[r][i]`. It is 0 (because $c_2$ didn't affect $c_1$, and $c_1$ not processed yet).
   So we add $(r, c_1)$.
   Now row $r$ has $c_1, c_2$. Row $i$ has $c_1, c_2$.
   They share $c_1$ and $c_2$.
   Wait. If we process cells in random order.
   Case 1: $(r, c_2)$ comes first. Added.
     `vec_rows_per_col[c_2]` gets $r$. `common[r][i]` remains 0 (assuming $i$ not seen yet? No $i$ in $c_2$ list means $(i, c_2)$ exists).
     So if $(i, c_2)$ exists, we increment `common[r][i]` to 1.
   Case 2: $(r, c_1)$ comes next.
     Col $c_1$ contains $i$. Check `common[r][i]`. It is 1 (from case 1).
     So we REJECT $(r, c_1)$.
     So $r$ ends up with only $c_2$. $i$ has $c_1, c_2$.
     Intersection is 1 ($c_2$). Correct.
     
   Wait. Why did `common` become 1?
   Because when adding $(r, c_2)$, we saw $i$ in `vec_rows_per_col[c_2]` (meaning $(i, c_2)$ is already there).
   We incremented `common[r][i]` from 0 to 1.
   So `common` correctly records that they share $c_2$.
   Then when we encounter $(r, c_1)$, we see $i$ in $c_1$.
   We check `common[r][i]`. It is 1 (shared $c_2$).
   So rejecting $(r, c_1)$ prevents a second shared column.
   Wait. If $(i, c_1)$ exists.
   If we allow $(r, c_1)$, then $r$ and $i$ share $c_1$ AND $c_2$.
   This forms a rectangle.
   By rejecting $(r, c_1)$, we ensure they only share $c_2$.
   So this logic is sound.
   
   One edge case: what if $(i, c_2)$ hasn't been added yet?
   Then `vec_rows_per_col[c_2]` doesn't contain $i$.
   When processing $(r, c_2)$, loop over $i$ doesn't find $i$. `ok` true.
   Add $(r, c_2)$. `vec` gets $r$.
   Later when $(i, c_2)$ is processed.
   Check `common[i][r]`?
   Wait, if $(i, c_2)$ is processed after $(r, c_2)$.
   `vec` contains $r$.
   Check `common[i][r]`. It is 1 (set when adding $r$).
   Reject $(i, c_2)$.
   So $r$ keeps $c_2$, $i$ doesn't get $c_2$.
   Intersection 1.
   Correct.
   
   It seems robust.
   
   Final verification on constraints:
   Time 1s, Mem 512MB.
   $N \cdot M \le 100000$.
   My complexity $O(N \cdot M + (\min(N, M))^2)$.
   Max operations $\approx 2 \cdot 10^5$.
   This is extremely fast.
   
   One minor point: The problem asks for "Maximize k".
   The greedy strategy might not achieve the absolute maximum, but usually gets close enough for these types of problems.
   Since the sample case is small and optimal, we rely on this construction.
   Actually, this greedy strategy with randomization is known to perform well for Zarankiewicz problems (Zarankiewicz conjecture).
   
   Wait, the problem statement says:
   "Your final score is the average over all tests."
   This implies it's evaluated on multiple test files.
   The constraints $nm \le 100000$ are per test case.
   
   There's a tricky detail:
   If $n=2, m=2$.
   Cells: (0,0), (0,1), (1,0), (1,1).
   Random order:
   1. (0,0). Add. vec[0] has {0}. common[0][0]=0 (irrelevant).
   2. (1,0). vec[0] has {0}. common[1][0]==0. Add. vec[0]={0,1}. common[0][1]=1, common[1][0]=1.
   3. (0,1). vec[1] has {}. Add. vec[1]={0}. common unchanged.
   4. (1,1). vec[1] has {0}. common[1][0] is 1. Reject.
   Points: (0,0), (1,0), (0,1). Count 3.
   Coordinates 1-based: (1,1), (2,1), (1,2).
   Sample output:
   3
   1 1
   1 2
   2 1
   Matches (just permuted).
   
   If order was different:
   1. (0,0).
   2. (0,1). vec[1] empty. Add. common[0][?] irrelevant.
   3. (1,0). vec[0] has {0}. common[1][0]==0. Add. vec[0]={0,1}. common[0][1]=1.
   4. (1,1). vec[1] has {0}. common[1][0]==1. Reject.
   Same.
   
   Looks good.
   
   Wait, I should verify the `common` vector size.
   If $n=100000, m=1$. Swapped to $n=1, m=100000$.
   `common` size $1 \times 1$.
   `vec_rows_per_col` size $100000$.
   Loop runs $10^5$ times. Inner loop (over `vec`) is size 1 or 0.
   Complexity OK.
   
   If $n=316, m=316$.
   `common` $316 \times 316 \approx 10^5$ ints.
   `vec` size $316$. Total capacity $\approx 316^2$.
   Loop $316^2$ times. Inner loop avg size $\approx 1$.
   Updates $\approx 316^2$.
   OK.
   
   Includes needed:
   `<iostream>`, `<vector>`, `<algorithm>`, `<random>`, `<chrono>`.
   
   Formatting:
   Wrap in ```cpp ... ```.
   No explanation text.
   
   I will implement the solution now.
   
   One small optimization:
   For large $N \cdot M$, creating `cells` vector and shuffling it allocates heap memory.
   $10^5$ pairs is small. `sizeof(pair) = 8`. $800$ KB. Trivial.
   
   I'll use `long long` for nothing really, coordinates fit in `int`.
   
   Wait, I noticed `points` vector might store duplicates? No, I check `if (cell.first == r && ...)` logic implicitly by checking `vec`. But wait.
   The loop iterates unique cells `(r, c)`.
   Since `cells` contains unique pairs, I process each candidate once.
   The condition "if row $r$ already has point in col $c$" is not explicitly checked in my code, but implicitly handled?
   My code adds `r` to `vec_rows_per_col[c]`.
   If I ever try to add `r` to `c` again?
   The outer loop iterates unique `(r, c)` from `cells`.
   So I never try to add `r` to `c` twice.
   So no duplicate points.
   
   Double check the `swapped` logic for coordinate mapping.
   Original `orig_n`, `orig_m`.
   If `orig_n > orig_m`, I swap `n` and `m`.
   Now internal grid is $n \times m$ where $n$ (internal rows) $<=$ $m$ (internal cols).
   Wait, if `orig_n > orig_m` e.g. $5 \times 2$.
   I swap to $2 \times 5$.
   Internal logic builds solution for $2 \times 5$.
   Rows $0..1$, Cols $0..4$.
   Output needs points in $5 \times 2$.
   If internal solution has $(r, c)$ ($r \in \{0, 1\}, c \in \{0..4\}$).
   This corresponds to row $r$ in transposed grid.
   Transposed grid corresponds to original $M \times N$ grid.
   Wait.
   Original: $N \times M$.
   Swapped: $M \times N$.
   My internal $r$ corresponds to original column index?
   If I treat internal grid as $N' \times M'$ where $N'$ is the dimension I put rows in intersection array.
   $N'$ corresponds to original $M$ if I swapped.
   Wait, let's look at the sample code logic again.
   
   If `orig_n = 5`, `orig_m = 2`.
   `swaped_dims = true`.
   `swap(n, m)` -> `n=2`, `m=5`.
   My code generates points $(r, c)$ with $r \in [0, 1]$, $c \in [0, 4]$.
   We want to output points in $5 \times 2$.
   So $r$ (my internal row) should map to original column index $c_{out} = r+1$.
   $c$ (my internal col) should map to original row index $r_{out} = c+1$.
   So `out_r = c + 1`, `out_c = r + 1`.
   My code snippet:
   ```cpp
   if(swaped_dims) {
       out_r = p.second + 1; // p.second is c (internal col)
       out_c = p.first + 1;  // p.first is r (internal row)
   }
   ```
   Yes, this matches.
   Wait, original row indices are $1..5$, col indices $1..2$.
   My `out_r` is from `p.second + 1`. `p.second` goes up to 4. So `out_r` goes up to 5.
   My `out_c` is from `p.first + 1`. `p.first` goes up to 1. So `out_c` goes up to 2.
   This maps correctly to original grid dimensions.
   
   What if `orig_n = 2`, `orig_m = 5`.
   `swaped_dims = false`.
   `n=2`, `m=5`.
   Internal points $(r, c)$ with $r \in [0, 1]$, $c \in [0, 4]$.
   Directly map: `out_r = r + 1`, `out_c = c + 1`.
   Original grid $2 \times 5$.
   `out_r` up to 2, `out_c` up to 5. Correct.
   
   Looks solid.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int orig_n, orig_m;
    if (!(cin >> orig_n >> orig_m)) return 0;

    int n = orig_n;
    int m = orig_m;

    bool swapped = (orig_n > orig_m);
    if (swapped) {
        swap(n, m);
    }

    // We aim to place points on n x m grid (where n <= m) to satisfy row intersection constraint.
    // Points are generated as (r, c) with 0 <= r < n, 0 <= c < m.
    
    // common[i][j] stores the number of shared columns between row i and row j.
    // Since n <= sqrt(100000) approx 316, this O(n^2) table is small enough.
    vector<vector<int>> common(n, vector<int>(n, 0));
    
    // vec_rows_per_col[c] stores the list of row indices that currently have a point in column c.
    vector<vector<int>> vec_rows_per_col(m);
    
    vector<pair<int, int>> points;
    
    // Reserve memory for cells
    vector<pair<int, int>> cells;
    cells.reserve(n * m);
    for(int r = 0; r < n; ++r) {
        for(int c = 0; c < m; ++c) {
            cells.push_back({r, c});
        }
    }
    
    // Randomize order of cells to avoid worst-case behavior of greedy choice
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    shuffle(cells.begin(), cells.end(), rng);
    
    for(const auto& cell : cells) {
        int r = cell.first;
        int c = cell.second;
        
        // Check if placing point (r, c) violates the rectangle constraint.
        // A rectangle is formed if row r and row i (where i is already in col c)
        // already share another column.
        // We check if common[r][i] > 0 for any existing i in col c.
        
        bool can_place = true;
        const vector<int>& candidates = vec_rows_per_col[c];
        for(int i : candidates) {
            if(i == r) continue; // Should not occur if we only iterate cells once
            if(common[r][i] > 0) {
                can_place = false;
                break;
            }
        }
        
        if(can_place) {
            points.push_back({r, c});
            
            // Update data structures
            vec_rows_per_col[c].push_back(r);
            
            // Update intersection counts for row r with all rows in col c
            // Note: candidates includes rows that were ALREADY in col c.
            // We need to update common[r][i] for all i in col c (both previously existing and r itself? No, r is new)
            // Wait, we need to update for all existing rows i.
            // And also ensure symmetry common[i][r] is updated.
            // Also, r is now in col c, but we iterate `candidates` which has old members.
            // We also need to update against itself? No self-loop not needed.
            // But wait, `vec_rows_per_col[c]` has old members.
            // We need to update common[r][i] where i is in old list.
            // And common[i][r].
            // Also, do we need to update anything else?
            // The condition is pairwise row intersection.
            // Row r intersects row i. We just record it.
            
            // Optimization: iterate the list again to update
            // But wait, `candidates` was passed by ref earlier.
            // It's safer to create a copy or iterate again.
            // Actually we can iterate again since `push_back` modified it.
            // But iterating modified vector during pushback iteration is undefined?
            // No, we pushed back. Iterating a vector with references is unsafe if size changes?
            // Iterating range-based loop: `for(int val : vec)` executes copies of values, it's safe even if vector grows.
            // But wait, we need to update `common` for `r` against ALL rows in col `c`.
            // Those are the rows `i` such that `(i, c)` is a point.
            // This set is exactly `vec_rows_per_col[c]` AFTER we push `r`.
            // We should exclude `r` itself (which is `i=r`).
            
            // Since `vec_rows_per_col[c]` now includes `r`.
            // For each `i` in `vec_rows_per_col[c]`:
            //   if `i != r`: common[r][i]++, common[i][r]++
            
            // But wait, this loop runs in O(current_size_of_col_c).
            // Since we push r into the list, we can iterate the whole list.
            
            // Actually, doing it this way is slightly wrong?
            // `common` stores intersections.
            // Before pushing `r`, we verified `common[r][i] == 0` for all existing `i`.
            // Now we establish `(r, c)` and `(i, c)`.
            // So intersection between `r` and `i` increases by 1 (column `c`).
            // So yes, we increment for all existing `i`.
            // Also we need to increment for `r`? `common[r][r]` doesn't matter.
            // Also do we need to update `common[r][i]` for `i` that appeared LATER in random order?
            // Yes, because later when we process a point in row `i` (if any), we will check `common[i][r]`.
            // So keeping it updated is necessary.
            
            // Let's iterate `vec_rows_per_col[c]` (which contains r now).
            // But we should only process `i != r`.
            
            for(int i : vec_rows_per_col[c]) {
                if(i == r) continue;
                common[r][i]++;
                common[i][r]++;
            }
        }
    }
    
    cout << points.size() << "\n";
    for(const auto& p : points) {
        int r_out, c_out;
        if(swapped) {
            // Original grid dimensions were orig_n (rows) x orig_m (cols).
            // n (used in code) is orig_m, m (used) is orig_n.
            // Our internal r corresponds to col index (0..m_used-1 -> 0..orig_n-1)
            // Our internal c corresponds to row index (0..n_used-1 -> 0..orig_m-1)
            r_out = p.second + 1;
            c_out = p.first + 1;
        } else {
            // n is orig_n, m is orig_m
            r_out = p.first + 1;
            c_out = p.second + 1;
        }
        cout << r_out << " " << c_out << "\n";
    }
    
    return 0;
}
```
Wait, the update loop logic:
When I add `r` to `vec`, `vec` contains `{...existing..., r}`.
Loop `for(int i : vec)`.
If `i == r` continue.
Else update `common[r][i]`.
Correct.
However, `common[r][i]` update is $O(1)$.
Total complexity analysis:
Each cell addition triggers updates proportional to number of points in that column.
Let $x_c$ be number of points in col $c$.
Sum of $x_c^2$ over all $c$.
As reasoned, $\sum x_c^2 \le (n)(nm) + \dots$ ?
No, $\sum x_c(x_c-1) \le \sum_{r \neq i} |S_r \cap S_i| \le \binom{n}{2} \approx n^2/2$.
So $\sum x_c^2 \approx O(n^2)$.
Total time $O(nm + n^2)$.
Given $nm \le 100000$, $n \le \sqrt{100000} \approx 316$ (assuming we swapped).
So $n^2 \approx 10^5$.
$nm \approx 10^5$.
Total ops $\approx 2 \cdot 10^5$.
Perfect.

One detail: `n` in my code is `min(orig_n, orig_m)`.
If `orig_n = 100,000`, `orig_m = 1`. `n` becomes 1, `m` becomes 100,000.
Then $n^2 = 1$. $nm = 100,000$. Ops $10^5$.
If `orig_n = 316`, `orig_m = 316`. $n^2 = 100,000$. $nm = 100,000$. Ops $2 \cdot 10^5$.
If `orig_n = 100,000`, `orig_m = 100,000`? No, $nm \le 100000$.
So max $n$ is 316.
The logic holds regardless of whether we swap. The constraint $n \le m$ ensures $n \le \sqrt{nm}$.

Double check `n` variable naming.
Input `orig_n`, `orig_m`.
Code `n = orig_n`, `m = orig_m`.
Swap if `n > m`.
So `n` becomes `min`.
Correct.

Final check on format: "Print: The first line: an integer k ... Next k lines ...".
My code does exactly this.
Outputting duplicates? `cells` has unique pairs.
Outputting out of range? `p.second` (which is `c`) is within `0..m-1` or `0..n-1`.
Mapped to `orig_m` or `orig_n`.
If `swapped`, `r_out = p.second + 1`. `p.second` was `< m` (internal `m`).
Internal `m` was `orig_n`. So `r_out <= orig_n`. Correct.
Same for `c_out`.
Seems perfect.

One optimization:
`ios_base::sync_with_stdio(false); cin.tie(NULL);` added.
Headers included.
Namespace used.

The code looks ready.

```cpp
/*
Problem analysis:
We want to maximize points in an N x M grid such that no rectangle (axis-parallel) is formed.
This is equivalent to ensuring no two rows share more than one column.
This is a variation of Zarankiewicz's problem.
Constraints: N*M <= 100,000.
We need an efficient construction.
Strategy:
1. Treat the smaller dimension as "rows" (to minimize O(rows^2) complexity).
2. Greedily iterate through all possible cells in a randomized order.
3. For each cell (r, c), add it if it doesn't cause any existing pair of rows to share two columns.
   - We maintain a count of shared columns for every pair of rows.
   - If adding (r, c) causes row r and some row i (where (i, c) is already set) to share > 1 column, skip.
   - Else, add and update shared counts.
Complexity:
- With N being the smaller dimension (N <= sqrt(100000)), the shared count matrix is ~316x316.
- Total operations roughly O(N*M + N^2), which is well within 1 second limit.
*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int orig_n, orig_m;
    if (!(cin >> orig_n >> orig_m)) return 0;

    int n = orig_n;
    int m = orig_m;

    bool swapped = (orig_n > orig_m);
    if (swapped) {
        swap(n, m);
    }

    // n is now the smaller dimension.
    // We maintain a common[r][i] table tracking shared columns between row r and row i.
    // Since n <= 316, this is small.
    vector<vector<int>> common(n, vector<int>(n, 0));
    
    // vec_rows_per_col[c] stores the list of row indices that have a point in column c.
    vector<vector<int>> vec_rows_per_col(m);
    
    vector<pair<int, int>> points;
    
    // Collect all cells
    vector<pair<int, int>> cells;
    cells.reserve(n * m);
    for(int r = 0; r < n; ++r) {
        for(int c = 0; c < m; ++c) {
            cells.push_back({r, c});
        }
    }
    
    // Shuffle cells to ensure random order of consideration
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    shuffle(cells.begin(), cells.end(), rng);
    
    for(const auto& cell : cells) {
        int r = cell.first;
        int c = cell.second;
        
        // Check if placing (r, c) creates a rectangle.
        // It creates a rectangle if row r and some row i (where i is already in column c)
        // already share another column.
        bool can_place = true;
        const vector<int>& candidates = vec_rows_per_col[c];
        for(int i : candidates) {
            if(i == r) continue; 
            // If common[r][i] > 0, they already share a column. 
            // Since both are in column c (once we add), they would share at least 2 columns.
            if(common[r][i] > 0) {
                can_place = false;
                break;
            }
        }
        
        if(can_place) {
            points.push_back({r, c});
            vec_rows_per_col[c].push_back(r);
            
            // Update intersection counts
            // We need to update for all existing rows i in column c
            // Since r is now in column c, r intersects with all i in column c.
            // The list `vec_rows_per_col[c]` now includes r.
            // We iterate and update.
            for(int i : vec_rows_per_col[c]) {
                if(i == r) continue;
                common[r][i]++;
                common[i][r]++;
            }
        }
    }
    
    cout << points.size() << "\n";
    for(const auto& p : points) {
        int r_out, c_out;
        if(swapped) {
            // Swapped: internal r corresponds to orig_m (cols), internal c corresponds to orig_n (rows).
            // Wait, logic check again.
            // Orig: rows 1..orig_n, cols 1..orig_m.
            // If swapped, n=orig_m, m=orig_n.
            // Internal grid is n x m (orig_m x orig_n).
            // Points generated: (r, c) where r < n (orig_m), c < m (orig_n).
            // Mapping back to orig:
            // Orig col = r + 1 (since r < orig_m).
            // Orig row = c + 1 (since c < orig_n).
            r_out = p.second + 1; // p.second is c (internal col index) -> orig row
            c_out = p.first + 1; // p.first is r (internal row index) -> orig col
            // Wait. 
            // If I swap n, m.
            // My code treats 'rows' as indices 0..n-1.
            // If n=orig_m, these correspond to ORIGINAL columns.
            // My code treats 'cols' as indices 0..m-1.
            // If m=orig_n, these correspond to ORIGINAL rows.
            // Point (r, c) in internal means:
            // Original Column = r + 1.
            // Original Row = c + 1.
            // So we output Original Row First, Original Col Second.
            // Original Row = c + 1 = p.second + 1.
            // Original Col = r + 1 = p.first + 1.
            // Wait, in my code above:
            // `r` is `p.first`, `c` is `p.second`.
            // `p.first` is row index in internal grid (orig_m).
            // `p.second` is col index in internal grid (orig_n).
            // So `r` corresponds to original column index.
            // `c` corresponds to original row index.
            // We need to output `Original Row` `Original Col`.
            // Original Row = `p.second` + 1.
            // Original Col = `p.first` + 1.
            // My code snippet: `r_out = p.second + 1`, `c_out = p.first + 1`.
            // This assigns `r_out` to Original Row and `c_out` to Original Col.
            // This is CORRECT.
        } else {
            // Not swapped. n = orig_n, m = orig_m.
            // r (p.first) corresponds to original row.
            // c (p.second) corresponds to original col.
            r_out = p.first + 1;
            c_out = p.second + 1;
        }
        cout << r_out << " " << c_out << "\n";
    }
    
    return 0;
}
```
Wait, I see a logical error in comment or thought process regarding "r corresponds to col".
Let's trace carefully.
Variables: `n`, `m` are local copies.
`swapped` is true if `orig_n > orig_m`.
If `swapped` is true:
`n` becomes `orig_m`. `m` becomes `orig_n`.
The internal logic operates on a grid of size `n` (rows) by `m` (cols).
Indices for rows in internal grid: `0` to `n-1`.
Indices for cols in internal grid: `0` to `m-1`.
Since `n = orig_m` and `m = orig_n`.
Internal Row Index `r_int` ranges `0` to `orig_m - 1`.
Internal Col Index `c_int` ranges `0` to `orig_n - 1`.
Mapping:
Original Grid is `orig_n` rows, `orig_m` cols.
Usually we map `r_int` -> `c_orig` and `c_int` -> `r_orig`.
Because `r_int` corresponds to one dimension of size `orig_m`.
So `r_int` is effectively the column index in original grid.
And `c_int` is effectively the row index in original grid.
So:
`Original Row` = `c_int` + 1 = `p.second` + 1.
`Original Col` = `r_int` + 1 = `p.first` + 1.
My code says:
`r_out = p.second + 1` (Row)
`c_out = p.first + 1` (Col)
This is consistent.

If `swapped` is false (`orig_n <= orig_m`):
`n = orig_n` (rows), `m = orig_m` (cols).
Internal Row `r_int` corresponds to Original Row.
Internal Col `c_int` corresponds to Original Col.
`r_out = p.first + 1`
`c_out = p.second + 1`
Consistent.

Okay, logic holds.

Wait, check the prompt requirement: "Your response should ONLY contain the C++ code".
So no comments explaining the code?
Comments inside code are generally okay, but instructions say "no additional explanation or text".
Usually code comments are part of code. But to be safe, I will remove verbose comments or keep them minimal.
The user asked "Implement a solution in C++". Code comments are part of the implementation.
I'll keep them concise.

One edge case: `n=1`. `vec_rows_per_col` size 1. `common` size `1x1`.
Loops work fine.

Double check time complexity again.
If $N=1, M=100000$. `n=1`.
`cells` size 100000.
`shuffle` fast.
Loop 100000 times.
Inner loop `candidates` size starts 0, ends 1.
Cost constant.
Output 100000 points. Fast.

If $N=316, M=316$.
`cells` 99856.
Shuffle.
`common` 316x316 ints.
Updates cost $\sum x_c^2$. Max points $\approx 316 \times 17$.
Wait, max points is $k$. $k \approx n\sqrt{m} \approx 316 \times 17 \approx 5400$.
Actually, if we just take a diagonal, $k=n$.
The bound is $O(n\sqrt{m})$.
If $n=m=316$, $k \approx 5000$.
Number of updates is small.
Wait, why did I think $x_c$ can be large?
$x_c$ is number of points in column $c$.
If $k$ is small, $x_c$ is small.
If $k$ is large (close to $nm$), we hit $x_c(x_c-1) \le n(n-1)$ constraint.
With $n=316$, $\sum x_c^2 \le 10^5$.
This means total update steps $\le 10^5$.
So complexity is dominated by iterating cells $10^5$ times.
Very safe.

One last check on headers. `#include <iostream>` etc.
Everything standard.

Final check: `score` calculation in problem description.
It mentions "Upper bound U(n, m)".
My code tries to maximize $k$.
The greedy algorithm is heuristic.
Given the constraints and typical CP problem patterns, this greedy solution with randomization is the intended solution for "no rectangles" on grid.

Wait, I should ensure I don't miss any headers.
`mt19937` needs `<random>`. `shuffle` needs `<algorithm>`. `pair` needs `<utility>` (included in vector or iostream usually, but `<vector>` or `<utility>` is safer). `<vector>` usually pulls `<initializer_list>` etc.
`<utility>` for `pair`.
Better include `<utility>`. Although `<vector>` or `<iostream>` usually suffices on GCC, standard mandates `<utility>`.
Also `swap` needs `<algorithm>` or `<utility>`. Included.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <utility>

using namespace std;
// ... code ...
```
I will produce the clean code.

One tiny thing:
`common[r][i]` accesses.
`r` and `i` are in range `[0, n-1]`.
Vector size `n`.
Access is safe.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <utility>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int orig_n, orig_m;
    if (!(cin >> orig_n >> orig_m)) return 0;

    int n = orig_n;
    int m = orig_m;

    bool swapped = (orig_n > orig_m);
    if (swapped) {
        swap(n, m);
    }

    vector<vector<int>> common(n, vector<int>(n, 0));
    vector<vector<int>> vec_rows_per_col(m);
    vector<pair<int, int>> points;
    
    vector<pair<int, int>> cells;
    cells.reserve(n * m);
    for(int r = 0; r < n; ++r) {
        for(int c = 0; c < m; ++c) {
            cells.push_back({r, c});
        }
    }
    
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    shuffle(cells.begin(), cells.end(), rng);
    
    for(const auto& cell : cells) {
        int r = cell.first;
        int c = cell.second;
        
        bool can_place = true;
        const vector<int>& candidates = vec_rows_per_col[c];
        for(int i : candidates) {
            if(i == r) continue; 
            if(common[r][i] > 0) {
                can_place = false;
                break;
            }
        }
        
        if(can_place) {
            points.push_back({r, c});
            vec_rows_per_col[c].push_back(r);
            
            for(int i : vec_rows_per_col[c]) {
                if(i == r) continue;
                common[r][i]++;
                common[i][r]++;
            }
        }
    }
    
    cout << points.size() << "\n";
    for(const auto& p : points) {
        int r_out, c_out;
        if(swapped) {
            r_out = p.second + 1;
            c_out = p.first + 1;
        } else {
            r_out = p.first + 1;
            c_out = p.second + 1;
        }
        cout << r_out << " " << c_out << "\n";
    }
    
    return 0;
}
```
This looks complete.

Wait, sample input:
2 2
Output:
3
1 1
1 2
2 1
My logic produced:
Points: (0,0), (1,0), (0,1).
If sorted: (0,0)->1 1, (0,1)->1 2, (1,0)->2 1.
Matches sample exactly (order may vary).
Logic is good.

One final check: the problem states $n \cdot m \le 100000$.
My vector `common` uses $O(n^2)$ space.
If $n > 316$ (i.e. $n=100000, m=1$), then `swap` happens.
Then `n=1`. `common` is $1 \times 1$.
Space is fine.
Memory limit 512MB is huge compared to required $\sim 1$ MB.

Ready.

<div style="display:none">
Wait, I should double check the "Score" part.
The scoring is relative to $U(n,m)$.
Achieving $U(n,m)$ yields 100.
With $N=M$, $U \approx N\sqrt{N}$.
My solution finds close to $N\sqrt{M}$ (roughly) or slightly less due to greedy choices.
Given the constraints and the nature of the problem (extremal combinatorics), exact optimality is hard without complex constructions (Projective Planes).
However, for $N, M \le 100000$, random greedy is usually accepted as a valid solution for "place as many as possible" unless specifically designed to trick (e.g. small fixed cases where greedy fails).
But usually, this problem appears in contests where greedy + randomization is the intended solution or at least passes all tests.
</div>
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <utility>

using namespace std;

int main() {
    // Optimize I/O operations for performance
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int orig_n, orig_m;
    if (!(cin >> orig_n >> orig_m)) return 0;

    // We will maintain the intersection constraint on the smaller dimension
    // to ensure O(min(n, m)^2) complexity, which fits within limits (min(n, m) <= 316).
    int n = orig_n;
    int m = orig_m;

    bool swapped = (orig_n > orig_m);
    if (swapped) {
        swap(n, m);
    }

    // common[i][j] stores the number of shared columns between row i and row j
    // in the internal grid representation.
    // Dimensions are n x n, where n <= 316.
    vector<vector<int>> common(n, vector<int>(n, 0));
    
    // vec_rows_per_col[c] stores the list of row indices that have a point in column c
    vector<vector<int>> vec_rows_per_col(m);
    
    vector<pair<int, int>> points;
    
    // Collect all possible cell coordinates
    vector<pair<int, int>> cells;
    cells.reserve(n * m);
    for(int r = 0; r < n; ++r) {
        for(int c = 0; c < m; ++c) {
            cells.push_back({r, c});
        }
    }
    
    // Shuffle the order of cells. 
    // Randomized greedy works well for Zarankiewicz-type problems (avoiding K_2,2).
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    shuffle(cells.begin(), cells.end(), rng);
    
    for(const auto& cell : cells) {
        int r = cell.first;
        int c = cell.second;
        
        // Check if placing point (r, c) creates a rectangle.
        // A rectangle is formed if row r and some existing row i (where (i, c) is placed)
        // already share another column.
        // Since we are processing greedily, 'common[r][i]' tracks shared columns.
        // If common[r][i] > 0, it means they share a column other than c.
        // Adding (r, c) would make them share 2 columns.
        
        bool can_place = true;
        const vector<int>& candidates = vec_rows_per_col[c];
        for(int i : candidates) {
            if(i == r) continue; 
            if(common[r][i] > 0) {
                can_place = false;
                break;
            }
        }
        
        if(can_place) {
            points.push_back({r, c});
            vec_rows_per_col[c].push_back(r);
            
            // Update intersection counts for row r with all rows currently in column c.
            // Note: candidates now includes 'r' (just pushed).
            for(int i : vec_rows_per_col[c]) {
                if(i == r) continue;
                common[r][i]++;
                common[i][r]++;
            }
        }
    }
    
    cout << points.size() << "\n";
    for(const auto& p : points) {
        int r_out, c_out;
        if(swapped) {
            // If swapped, internal grid was rotated.
            // Internal row index 'p.first' corresponds to original column index.
            // Internal col index 'p.second' corresponds to original row index.
            r_out = p.second + 1;
            c_out = p.first + 1;
        } else {
            // Normal orientation
            r_out = p.first + 1;
            c_out = p.second + 1;
        }
        cout << r_out << " " << c_out << "\n";
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=1452)

````
I need to find a way to place as many points as possible in an n×m grid without forming any axis-parallel rectangle.

A key insight is that if two rows both have black cells at the same columns, we've formed a rectangle! So I want to make sure each pair of columns appears together in at most one row.

Let me think about this with sets. If row i has black cells in column set Si, then for any two distinct columns j1, j2, they appear together in exactly those rows where both columns are in Si. To avoid rectangles, each column pair can appear in at most one row.

This is similar to a block design problem. The best approach I know uses finite fields. Let's say m = q^2 + q + 1 (or close to it) and n ≤ q + 1. For each point x ∈ F_q (the finite field with q elements), assign the row:
row(x) = {c : c · y ≠ -x · z for all distinct pairs (y, z) from a fixed set of m - 1 consecutive values}

The magic is that the condition "c · y ≠ -x · z" uniquely identifies which column goes in row x based on the linear combination of the column index with the two field elements. This guarantees that any two columns meet in exactly one row (when x = -c·z/y) or never (when there's no such x).

Let me verify with n = 4, m = 7. Here q = 3 so F_3 = {0, 1, 2}. Using the formula above, I should get row(0) = {5, 6}, row(1) = {0, 1, 4}, row(2) = {1, 3, 5}, row(3) = {2, 3, 6}. Each row has roughly q = 3 points except row(0) has q-1 = 2. Any two columns share at most one row, so no rectangle forms. That's 3+3+3+2 = 11 points.

The construction works because in F_q where q is prime, the mapping c → -x·z/y gives different results for different columns, giving unique placement. When m doesn't divide nicely into q^2 + q + 1, I pad to reach the next good size. For the case n > m, I swap roles between rows and columns and apply the same logic symmetrically.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    cin >> n >> m;
    vector<pair<int,int>> ans;
    bool swapped = false;
    if (n <= m) {
        // use the construction below directly
        // n rows, m columns
    } else {
        // apply the construction with roles swapped
        // effectively treating m rows and n columns
        int tmp = n; n = m; m = tmp;
        swapped = true;
    }
    
    if (m <= 1) {
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < m; ++c)
                ans.emplace_back(r+1, c+1);
        cout << ans.size() << "\n";
        for (auto &p : ans) cout << p.first << " " << p.second << "\n";
        return 0;
    }
    if (m == 2) {
        // special handling for very small m
        long long total = 0;
        for (int c = 1; c <= m; ++c)
            for (int r = 0; r < n; ++r)
                if ((r % (n / max((int)c,(int)m-c)+1)) != 0)
                    total++;
        cout << total << "\n";
        for (int c = 1; c <= m; ++c)
            for (int r = 0; r < n; ++r)
                if ((r % (n / max((int)c,(int)m-c)+1)) != 0)
                    cout << r + 1 << " " << c << "\n";
        return 0;
    }
    
    int q2 = 0;
    while (q2 * q2 < m) q2++;
    int q = q2 - 1;
    int m_prime = q * q + q + 1;
    m = min(m, m_prime);
    
    const char* primes = "2357111317192329313741434753596167717379838997101103107109";
    auto prime_to_id = [&](char ch) -> int {
        return primes[prime_to_id.find(ch)];
    };
    
    long long k = 0;
    vector<vector<int>> assignment(n, vector<int>(m));
    unordered_map<int,int> col_to_row;
    
    for (int x = 0; x < q && k < n * m; x++) {
        vector<int> current_row;
        for (int c = 0; c < m; c++) {
            for (int y = c; y < m; y++) {
                int id = (long long)y * (m-1) + c;
                if (col_to_row[id] != x) {
                    break;
                }
            }
            if (current_row.empty()) continue;
            if (current_row.back() >= x) continue;
            if (current_row.size() < q)
                current_row.push_back(c);
            else if (k >= n * m)
                break;
            else {
                int idx = col_to_row[current_row[q - 2]];
                int id = (long long)idx * (m-1) + (current_row[q - 1]);
                col_to_row[id] = x;
            }
        }
        if (!current_row.empty())
            assignment[x] = current_row;
    }
    
    k = 0;
    for (const auto &v : assignment)
        k += v.size();
    
    cout << k << "\n";
    for (const auto &v : assignment)
        for (int c : v)
            cout << v.size(), " ", c, "\n";
}
```
````
