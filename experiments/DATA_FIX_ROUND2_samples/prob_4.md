# 题 4 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The user wants a C++ solution for an interactive problem.
Problem: Find the k-th smallest value in an $n \times n$ matrix $a$.
Constraints on $a$: Rows are non-decreasing, columns are non-decreasing. This means the matrix is sorted both row-wise and column-wise.
$n \le 2000$.
Number of QUERY calls allowed $\le 50000$.
$a_{i,j} \le 10^{18}$.

Key observation: The matrix is sorted. We want to find the $k$-th smallest element. Since we can't access arbitrary elements efficiently (QUERY takes time), and we have a limit of $50000$ queries, we need an algorithm that leverages the sorted property.
Wait, $n^2 = 4,000,000$, but we only get $50,000$ queries. This suggests we cannot iterate or binary search over values easily unless we use the structure effectively. Actually, $50000$ is quite small compared to $n=2000$ if we were to query naively, but larger than $O(n)$? No, $O(n) = 2000$. So we can afford linear scan in some dimension.
Actually, the limit is $50000$. $n^2$ is up to $4 \times 10^6$.
We need to find the rank or value efficiently.

This problem is equivalent to finding the $k$-th smallest number in a sorted matrix without reading the whole matrix.
Standard approach for "K-th smallest in sorted matrix": Binary search on the answer (value).
Let's denote $count(v)$ as the number of elements $\le v$. If we can calculate $count(v)$ efficiently, we can binary search on the range of possible values $[min\_val, max\_val]$.
However, we don't know the values of the matrix directly. We can only query specific positions $(x, y)$.
Calculating $count(v)$ requires counting how many $(i, j)$ satisfy $a_{i,j} \le v$.
Since rows and columns are sorted, for a fixed value $v$, we can count elements $\le v$ by iterating through one dimension and using the monotonicity. Specifically, if we pick a row $i$, since it's sorted, we can use `std::lower_bound` or just traverse. But doing this for every row would take $O(n^2)$ time/queries. We only have $50000$ queries.
If we binary search on values, say range $R$, then $count(v)$ involves querying.
But wait, we can't simply guess values arbitrarily because we don't know the distribution.
Is it possible to simulate the merge sort process or something similar?
Or maybe we can binary search on the indices?
Actually, the constraints allow $50000$ queries.
If $n=2000$, $n^2 = 4,000,000$.
If we just do a BFS/DFS on the state space defined by coordinates?
Let's consider the set of all cells. We want the $k$-th smallest.
Since $a_{i,j}$ increases with $i$ and $j$, this looks like finding the shortest path in a DAG or similar?
Actually, there is a known technique for this problem given a limit on queries.
If we want to check if the value at $(r, c)$ is the $k$-th smallest, we need to know how many elements are smaller.
But we don't know the values.
Wait, "find the k-th smallest value". Usually, in interactive problems where you don't know values, you binary search on the answer.
But the "answer" space is large ($10^{18}$). However, the number of distinct values is at most $n^2$.
With $n=2000$, we have $4 \times 10^6$ potential positions. We can't visit them all.
Wait, the limit is 50,000. $n^2$ is much larger.
So we definitely cannot query all.
But we can do something smart.
Consider the problem: find $k$-th smallest in sorted matrix.
Usually, this is solved by picking a pivot, checking its rank, etc.
Here we have the constraint $a_{i,j} \le a_{i+1, j}$ and $a_{i,j} \le a_{i, j+1}$.
This implies $a_{1,1} \le \dots \le a_{n,n}$.
We can view this as a min-priority queue problem (like Dijkstra) starting from $a_{1,1}$, expanding to neighbors $(i+1, j)$ and $(i, j+1)$.
If we maintain a priority queue of candidates, always extracting the smallest candidate, the sequence of extracted elements corresponds to the sorted order of the matrix.
We stop when we extract the $k$-th element.
How many elements do we extract? $k$.
Wait, $k$ could be up to $n^2 = 4 \times 10^6$.
Limit is 50,000.
So we cannot simply run $k$ extractions.
Unless... we can prune or use the structure better.
But wait, if we don't visit nodes, we can't compare their relative order.
Is there any information given about $a_{i,j}$ other than inequalities? No.
This seems to suggest we cannot solve this using $k$-th smallest in general sense if $k$ is large.
However, look at the scoring function: `(50000 - used) / (50000 - n)` if correct.
And `n <= 2000`. `n*n <= 4,000,000`.
The scoring encourages using fewer queries. But we must use at least some? Wait, if `used <= n`, score 1.0.
So we ideally want to find the answer with very few queries.
Maybe we don't need to output the exact $k$-th smallest by simulating extraction?
Is it possible that for certain $k$, we can determine the value more directly?
Wait, the problem statement says "find the k-th smallest value".
With $N \le 2000$ and Query limit 50000, this is basically finding the median (if $k \approx n^2/2$) within very few queries?
Actually, notice the limit 50000 is roughly $25 \cdot n$.
Is there a randomized approach or a divide and conquer approach?
This problem is likely based on "Median of Row-Merged Columns" or similar.
Since each row is sorted and each column is sorted, the matrix is totally ordered in terms of reachability.
Actually, consider the set of cells we need to check.
There is a specialized algorithm for this.
Algorithm Idea:
We are looking for $k$-th smallest.
The matrix has $n^2$ entries.
We can treat this as selecting the $k$-th smallest from a set of $n$ sorted arrays (the rows), or $n$ sorted arrays (columns).
Merging $n$ sorted arrays to find the $k$-th smallest usually takes $O(k + \log(\text{range}) \cdot n)$ or similar, but here comparing elements costs 1 query?
No, we don't know the values.
But wait, we can query ANY $a[i][j]$.
So if we hypothesize a value $X$, we can check how many $a[i][j] \le X$.
To count elements $\le X$: For each row $i$, since $a[i][\cdot]$ is sorted, we want to find index $p_i$ such that $a[i][p_i] > X$ and $a[i][p_i-1] \le X$.
This can be done via binary search on row $i$. Cost per row: $O(\log n)$ queries. Total cost for all rows: $O(n \log n)$ queries.
Then total cost to binary search over values: $O(\log(10^{18}) \cdot n \log n) \approx 60 \cdot 2000 \cdot 11 \approx 1.3 \times 10^6$. Too slow (limit 50000).
Wait, actually we don't know the values. Binary search on values requires knowing a range or finding values.
If we don't know the values, we can't binary search on values unless we pick pivots from the matrix itself.
What if we pick $a[i][i]$ or something as pivot?

Let's reconsider the scoring. It rewards using *less* than 50000. But specifically, if used $\ge n^2$, score is 0.0. Wait, no: `else if used >= n*n: 0.0`.
Wait, if I use $n^2$ queries, I read the whole matrix, then I can trivially find the $k$-th smallest.
$n^2 = 4 \times 10^6$. Limit is $50000$.
So I MUST NOT read the whole matrix.
I must exploit the $O(50000)$ budget.
Notice $50000 \approx 25 \times n$.
So I have a budget of roughly $25 \times n$ queries.
Finding the $k$-th smallest element in a sorted matrix with limited queries.
This sounds like "Find K-th smallest in a sorted matrix" problem but with very strict limits.
Is it possible to binary search on the *coordinates* instead of values?
Let the target cell be $(r, k_{idx}, c)$? No.

Let's look for similar competitive programming problems.
"Matrix K-th Smallest" interactive problem often relates to:
1. Determining if $val(x, y) < val(u, v)$. But we can only get value.
2. Using the sorted property.

Wait, $a_{i,j} \le a_{i,j+1}$ and $a_{i,j} \le a_{i+1, j}$.
This is exactly the condition for "Young Tableau" shape if we ignore specific values.
We can find the $k$-th smallest value using a priority queue logic, but that is $O(k \log (\text{active cells}))$. If $k$ is large, this is too slow.
BUT, maybe we don't need to simulate $k$ steps.
We need to narrow down the value.
Wait, if we can estimate the rank of an element, we can move towards the $k$-th.
Let $C(i, j)$ be the rank of $a[i][j]$ (how many elements are $\le a[i][j]$ excluding itself).
Rank is hard to compute without querying.
However, we can find a pivot.
For example, query diagonal elements $a_{1,1}, a_{2,2}, \dots, a_{n,n}$. $n$ queries.
These define $n$ bounds? Not really.
They are not sufficient.

Let's think about the constraints again.
Query limit 50000.
$n \le 2000$.
Time limit 5s.
Scoring gives bonus if used $\le n$.
But `used <= n` is score 1.0. This implies optimal solutions might exist with very few queries.
Is it possible to do it with $O(n)$ or $O(n \log n)$ or $O(n \sqrt{k})$?
Actually, there is a strategy:
Divide and Conquer on the grid?
Or maybe we can find the $k$-th smallest by maintaining a candidate value?

Let's check if there is a simpler interpretation.
Maybe we can assume the $k$-th smallest lies within some subrectangle?
If we partition the matrix into two halves?
Suppose we query $a[i][j]$ and $a[n-i+1][n-j+1]$?
Wait, if $k$ is small, we can find small values.
If $k$ is large, we find large values.
Actually, with 50000 queries, we can potentially explore the "frontier".
Consider the cells $(1, 1), (2, 1), \dots, (k, 1)$?
Wait, we don't know which ones are small.
However, we know $a_{1,1}$ is the minimum.
So $a_{1,1}$ is the 1st smallest.
Neighbors of $(r,c)$ are $(r+1, c)$ and $(r, c+1)$.
In a heap approach to generate sorted sequence, we extract min, add neighbors.
To find $k$-th, we perform $k$ operations.
With $k$ up to $n^2$, this is bad.
BUT we only need to find the VALUE.
Maybe we can binary search on the RANK? No.

Wait, there is a trick!
We can binary search on the value $V$ IF we can quickly determine count of elements $\le V$.
Count of elements $\le V$: $\sum_{i=1}^n (\text{number of } j \text{ s.t. } a[i][j] \le V)$.
For a fixed row $i$, we want largest $j$ such that $a[i][j] \le V$.
Let's call this $pos(i, V)$.
Then count is $\sum pos(i, V)$.
Since $a[i][j]$ is sorted, we can find $pos(i, V)$ by binary searching the row.
This takes $O(\log n)$ queries per row. Total $O(n \log n)$ queries to find count for a specific $V$.
Now, how to choose $V$?
If we knew $V$, we'd do it in $n \log n$.
$50000 / \log(2000) \approx 50000 / 11 \approx 4500$.
So we have ~4500 rows worth of checks.
We can't afford to do full binary search on $10^{18}$ range.
But wait, the values come from the matrix. There are only $n^2$ distinct values.
We can collect a set of candidate values from the matrix, sort them, and binary search on indices? No.
Wait, if we can identify a subset of cells $S$ such that the $k$-th smallest value is among $S$, and $|S|$ is small?
This leads back to finding $k$-th smallest.
However, note the query limit is generous compared to $n$. 50000 vs 2000.
It allows $O(n \log n)$ complexity.
Is it possible that we can determine the exact rank of $a[i][i]$?
Not really.

Let's rethink. Is there a method that doesn't require full counts?
Maybe a randomized pivot selection?
Pick random cell $(r, c)$. Query $v = a[r][c]$.
Compute how many are $\le v$.
How? Iterate rows. For each row, find first element $> v$.
This takes $O(n \log n)$ queries.
If we are lucky, $count \approx k$, we found it?
Wait, we need exact count.
Suppose we pick a few cells $C_1, \dots, C_m$ with values $v_1, \dots, v_m$.
Sort these values. They split the value range into intervals.
In each interval $[v_p, v_{p+1}]$, the count is consistent? No, intermediate cells might fall in between.
So we must check all cells? No.
We need to select a pivot row/col strategy.

Wait, look at the problem source/style. This is "Matrix k-th Smallest (Interactive)".
Often these problems rely on the fact that with enough queries we can approximate.
But 50000 is the HARD limit.
Also, there is a special case.
What if $k$ is small? Then we just expand from $(1,1)$ until size $k$.
This visits $\approx k$ elements. Each step adds 1 or 2 elements to PQ.
Heap size grows.
Wait, $k$ can be $n^2/2 = 2 \times 10^6$.
We cannot visit $k$ elements.
We must jump to the answer.

What if we use the structure $a_{i,j} \le a_{i+1, j}$ and $a_{i,j} \le a_{i, j+1}$ to implement a "binary search on matrix coordinates"?
Like, we want to find a cell $(x, y)$ such that the number of cells in the rectangle $[1..n] \times [1..y]$ or something is related to $k$? No.
Actually, the rank of $a[x][y]$ is determined by the region $\le a[x][y]$.
Since the matrix is sorted, the region $\{(i, j) \mid a[i][j] \le a[x][y]\}$ forms a Young Diagram shape (monotone path boundary).
Specifically, there exists a path from top-right to bottom-left separating $\le$ and $>$.
This suggests that finding the $k$-th element is equivalent to finding the boundary such that area is $k$.
Since we can query $a[x][y]$, can we navigate to find the element with rank $k$?
This sounds like finding the "cut" in a Young Tableau.
But we can't see the diagram directly.

Wait! I recall a problem "Interactive: Matrix K-th Smallest" or similar.
There's a technique involving $O(n)$ or $O(n \log n)$ or $O(n \sqrt{n})$ queries.
Given the constraints, $n \log n$ fits perfectly ($2000 \times 11 = 22000 < 50000$).
So the solution should be roughly $O(n \log n)$.
Can we achieve $O(n \log n)$?
Yes, if we can check the rank of a candidate in $O(\log n)$ or $O(n)$.
Wait, computing rank exactly takes $O(n \log n)$ queries if we do binary search per row.
Total queries $\approx (\text{# pivots}) \times n \log n$.
If we have 2 pivots, that's $2 \cdot 22000 = 44000$. Fits.
So if we can reduce the problem to finding if the rank of some pivot is $\ge k$ or something?
But we need to find the value.
Actually, if we can find a value $V$ such that `rank(V) >= k` and `rank(next_smaller_value(V)) < k`, we are good.
But we don't know next smaller.
However, notice the values are integers.
Is it possible to do binary search on the VALUES?
We need an initial range $[L, R]$.
Can we determine the range of values present in the matrix?
Min value is $a_{1,1}$, max is $a_{n,n}$.
So query $a_{1,1}$ and $a_{n,n}$. Range is $[v_{min}, v_{max}]$.
Binary search value $Mid$.
Check if `count(Mid) >= k`.
Checking `count(Mid)` costs $O(n \log n)$ queries (using binary search on each row to find how many are $\le Mid$).
So one step of BS on values costs $O(n \log n)$.
How many steps? $\log(10^{18}) \approx 60$.
Total queries $\approx 60 \times n \log n \approx 60 \times 2000 \times 11 \approx 1.3 \times 10^6$.
This exceeds 50000.
So we cannot do full value binary search.

However, we can use the "candidates" idea.
We know the answer is one of the $a_{i,j}$.
Maybe we can select a small set of candidate values and binary search among them?
Wait, if we just sample some values, say $a_{1,1}, a_{1,2}, \dots, a_{n,n}$?
That's $n$ values.
We can check how many elements are $\le a_{i,i}$? That takes $n \log n$ queries. Doing this for $n$ diagonals takes $n^2 \log n$, too slow.
But we only need to find the $k$-th.
Wait, if we binary search on the coordinate space?
Suppose we consider the cell $(x, y)$ whose rank is around $k$.
The rank function $R(x, y) = |\{(u, v) : a[u][v] \le a[x][y]\}|$.
If we fix $x$, $R(x, y)$ increases with $y$.
If we fix $y$, $R(x, y)$ increases with $x$.
We want to find $(x, y)$ such that $R(x, y) \approx k$.
Specifically, we want to find the cell with the $k$-th smallest value.
Let the answer be $(r, c)$. Then $R(r, c) \approx k$ (actually, strictly speaking $R(r, c)$ is rank, so if distinct, $R=r+c-1$?).
Wait, duplicates are allowed.
But essentially we want to find $(r, c)$ minimizing $R$ subject to being the $k$-th element.
Is it possible to locate $(r, c)$ efficiently?
This is a 2D search problem.
We can use a Divide & Conquer strategy.
Divide the rows into two halves $[1, m]$ and $[m+1, n]$.
We want to check if the $k$-th element is in the top half or bottom half.
But the values are not uniform across rows.
However, for any column $j$, $a[m+1][j] \ge a[m][j]$.
Actually, we don't know the relation between rows. We know row $m+1$ elements are $\ge$ row $m$ elements *at same index*.
But cross-row comparison $a[m][j]$ vs $a[m+1][l]$ is unknown except through sorting.
Wait, this is the key.
For any value $v$, the set of indices $(i, j)$ with $a_{i,j} \le v$ is a Young Diagram (Ferrers diagram).
This means there is a monotonic boundary.
Let this boundary be defined by a function $f(j)$ such that $a_{i,j} \le v \iff i \le f(j)$? Or something like that.
Actually, since columns are sorted, $a_{i,j} \le v \implies a_{i',j} \le v$ for $i' < i$.
So for each column $j$, let $h(j)$ be the largest index $i$ such that $a_{i,j} \le v$. Then $h(j)$ is non-increasing with respect to $j$?
Wait. $a_{i,j} \le a_{i, j+1}$.
If $a_{i,j} \le v$, does it imply $a_{i, j+1} \le v$? No.
Wait, $a_{i,j} \le a_{i+1, j}$ (column sorted) and $a_{i,j} \le a_{i, j+1}$ (row sorted).
Let's check the boundary property.
Consider the set $S_v = \{(i, j) \mid a_{i,j} \le v\}$.
Since rows and columns are sorted, if $(i, j) \in S_v$, then all $(i', j')$ with $i' \le i$ and $j' \le j$ are also in $S_v$ (since $a_{i',j'} \le a_{i,j} \le v$).
Thus $S_v$ is a "lower-left justified" set.
So for each column $j$, let $cnt_j$ be the number of elements in col $j$ that are $\le v$.
Since $a_{i,j} \le v \implies a_{i-1, j} \le v$, the valid $i$'s form a prefix $1 \dots cnt_j$.
Also, since $a_{i,j} \le a_{i, j+1}$, if $a_{i,j} \le v$, then $a_{i-1, j+1}$ might be anything?
Wait. If $(i, j) \in S_v$, does it constrain $(i, j+1)$?
If $a_{i,j} \le v$, it implies $a_{i, j+1} \ge a_{i,j}$. It doesn't help upper bound.
However, if $a_{i,j} > v$, then $a_{i, j-1}$ could be $\le v$.
The crucial property is:
For each row $i$, there is a threshold $x_i$ such that $a_{i, j} \le v$ for $j \le x_i$ and $a_{i, j} > v$ for $j > x_i$.
Also $a_{i,j} \le a_{i+1, j}$ implies if $a_{i,j} \le v$, then $a_{i+1, j} \ge a_{i,j}$. No direct constraint on $x_{i+1}$ vs $x_i$.
But wait, we established earlier that counting elements $\le v$ costs $n \log n$.
And we need to do it fast.
If we fix the value $v$, finding count is easy ($n \log n$).
If we fix a row $i$, finding $x_i$ such that $a_{i, x_i} \approx v$ is easy (binary search $x_i$).
Wait, $a_{i, x_i}$ depends on $v$.
If we don't know $v$, we can't find $x_i$.

Let's try a different perspective.
We want to find $k$-th smallest.
Maybe we can iterate through the matrix using a Min-Heap but pruning aggressively?
With $n=2000$, $k$-th element is likely near center.
If we start with $(1,1)$, neighbors are $(1,2), (2,1)$.
Add them to heap. Pop min.
But we only have 50000 queries.
If $k$ is small, we win.
If $k$ is large, we lose.
Is it possible to use $O(n)$ queries to find the $k$-th value?
Consider the following recursive strategy (Divide and Conquer):
Function `solve(top_row, bot_row, left_col, right_col, k_limit)`:
We want to find the $k$-th smallest in the submatrix.
But submatrices inherit sorted properties.
We need to eliminate regions.
Maybe we can check $a[mid][mid]$?
If $a[mid][mid]$ is the pivot.
How many elements are $\le a[mid][mid]$?
We can count them.
Let $c = count(a[mid][mid])$.
If $k \le c$, the $k$-th smallest is $\le a[mid][mid]$.
If $k > c$, the $k$-th smallest is $> a[mid][mid]$.
But wait, if we find $c$ accurately, we can discard the elements that are either clearly smaller or clearly larger.
However, to find $c$ accurately, we spend $n \log n$ queries.
If we do this once, we spend $\approx 22000$ queries.
We have plenty left.
So we can do this ONE time?
But if $k$ falls in the left part, we recurse?
Recursion depth could be $\log(n^2)$ or $\log n$?
If we recurse, the submatrix shrinks.
Wait, if we go left (values smaller than pivot), we restrict the search space to rows/cols where values are smaller?
Since the matrix is globally sorted, if we decide values $> a[mid][mid]$ are excluded, we exclude the upper-right triangle?
Wait, $a_{i,j} \le a_{mid, mid}$ does NOT imply $(i, j)$ is in a rectangle.
Example: $a_{1, n}$ might be $\le a_{mid, mid}$ even if $a_{mid, mid}$ is large.
Wait, if $a_{mid, mid}$ is small, then $a_{1, n}$ could be larger.
The property $a_{i,j} \le a_{i+1, j}$ and $a_{i,j} \le a_{i, j+1}$ defines that if $(i, j)$ is "small", then $(i-1, j-1)$ is smaller.
Conversely, if $(i, j)$ is "large", then $(i+1, j+1)$ is larger.
So the set of elements $\le P$ forms a Young diagram shape.
The boundary separates $\le P$ and $> P$.
If we compute $count(P)$, we know the area of this shape.
If $count(P) \ge k$, the $k$-th smallest is in the $\le P$ region.
Wait, $count(P)$ includes $P$ itself. So if $k \le count(P)$, the $k$-th element is $\le P$.
We need to find a tighter value.
But simply knowing "it is $\le P$" isn't enough unless we have a candidate that is closer.
Actually, if $count(P) \ge k$, we know the answer is somewhere in the subregion defined by the boundary?
Wait, the subregion is the shape itself. We still need to find the $k$-th in that shape.
But the shape is irregular.
However, notice we spent $O(n \log n)$ queries to get $count(P)$.
Can we reduce the query cost?
Actually, if we pick a point $(r, c)$ such that $r \cdot c \approx k$? Or something heuristic?
If we pick $a_{1, k}$, then we know $a_{1, 1} \dots a_{1, k}$ are $\le a_{1, k}$.
But there are other rows.
Actually, $a_{i, j}$ values are bounded.
Maybe we can iterate on rows and maintain current candidates?
Actually, consider the following approach:
We can find the $k$-th smallest value in $O(n \log n)$ time if we use binary search over the matrix?
Wait, $n \log n$ is the query budget.
Actually, $n \log n \approx 22000$.
So we can perform $O(n \log n)$ work.
What if we just perform one check:
Choose a pivot $P = a_{x, y}$. Check $count(P)$.
If $count(P) \approx k$, we have found it? No, count is rank.
If $count(P)$ tells us how many elements are $\le P$.
If we find $P$ such that $count(P) \ge k$ and we haven't queried others, can we conclude $P$ is the answer?
Only if there are NO elements strictly smaller than $P$ that we missed?
No, $count(P)$ accounts for all elements $\le P$.
If $count(P) == k$, then there are $k$ elements $\le P$.
Does this mean $P$ is the $k$-th smallest?
It means $P$ could be the $k$-th or smaller (e.g. multiple copies of the $k$-th value).
So $P$ is a valid candidate.
But wait, $P$ is $a_{x,y}$. If there are multiple elements equal to $P$, we just need *any* instance of the value.
But wait, if $count(P) > k$, then $k$-th smallest is strictly smaller than $P$.
So we must lower our search.
If $count(P) < k$, we must raise our search.
So we can use this to narrow down.
But finding $P$ such that $count(P) \approx k$ requires checking many $P$'s.
Each check costs $O(n \log n)$ queries?
No, if we just query $a_{x,y}$, we know its value.
Counting takes $O(n \log n)$.
If we do this multiple times, we exceed budget.
Wait, maybe we don't need to count exactly?
Or maybe we can count faster?
If we query $a_{x, y}$, can we estimate count?
For each row $i$, find number of $a_{i, j} \le a_{x, y}$.
If we use binary search on each row, it is $O(n \log n)$.
Is there a way to do it faster?
What if we pick a specific $P$?
Maybe we can construct the answer by merging rows?
Let's revisit the idea: $O(n)$ queries to find $k$-th?
Wait, 50000 is huge.
Wait, $n=2000$, $n \log n \approx 22000$.
If we can do binary search on the *value* but optimize the check.
Optimization:
Instead of full binary search on each row, maybe we can parallelize?
No, sequential queries.
But we can skip some checks?
Wait, what if we don't binary search on value, but on coordinates?
We want to find $(r, c)$ such that its rank is $k$.
Wait, this is equivalent to finding a cell $(r, c)$ such that the number of cells $(u, v)$ with $a_{u,v} \le a_{r,c}$ is $\ge k$ and $(r,c)$ is the minimal such cell?
Actually, the set of elements $\le v$ forms a monotone region.
The $k$-th smallest value $V$ corresponds to a cutoff.
We can try to find the cutoff line.
This feels like finding a cut in a binary search tree?
But we don't have a BST.
Let's look at the constraint again. 50000 queries.
We can afford $O(n)$ queries per test case if it works.
Is there an $O(n)$ solution?
For $N=1$ matrix is $1\times 1$.
For $N=2$, $2\times 2$.
Maybe we can maintain a set of "candidate values" and eliminate them?
Let's try the "binary search on value" again but smarter.
If we query $a_{1,1}$ (min) and $a_{n,n}$ (max).
Range $[L, R]$.
If we pick $Mid = (L+R)/2$.
If we can calculate $count(Mid)$ fast...
But $Mid$ is arbitrary.
Wait, if $count(Mid)$ calculation is expensive.
Is there a way to avoid calculating count?
Yes, maybe we can use the fact that $k$ is known.
If we just greedily collect $k$ elements?
Start with $(1,1)$. Add to list.
Current smallest is $a_{1,1}$. Candidates for next smallest are neighbors in PQ.
If we only keep $k$ elements in PQ?
When we pop an element $(r, c)$ with value $v$, we push $(r+1, c)$ and $(r, c+1)$ if they are not visited.
If we only do this for $k$ pops.
Cost: $k \log (\text{heap size})$. Heap size $\le k$.
Cost $\approx k \log k$.
If $k$ is close to $n^2$, this is $n^2 \log n \approx 4 \times 10^6 \times 11 \approx 4.4 \times 10^7$. Too slow.
If $k$ is small, it works.
What if $k$ is large?
The problem asks for $k$-th smallest.
Maybe we can assume $k$ is small? No, $k$ can be anything up to $n^2$.
Wait, look at the limits again.
Query limit 50000.
Memory 1024 MB.
Time 5 sec.
Is there a strategy where we query $a_{1,k}$?
No, indices are $1 \dots n$.
What if we query $a_{1, n}$?
Wait, there is a property.
For any $x$, the elements $a_{1,x}, a_{2,x-1}, \dots, a_x, 1$ ? No.
Consider the main diagonal $a_{i, i}$.
Is it true that $a_{i, i}$ divides the matrix?
Not really.

Wait, check the scoring again.
`(50000 - used) / (50000 - n)`.
If used $\le n$, score 1.
So ideal is $\le 2000$ queries.
But we must be able to handle worst case.
Can we do it with $O(n)$ queries?
Yes, if we can find a pivot that splits the count exactly?
Wait, we have 50000 queries.
$50000 \approx 25 \cdot n$.
So we have 25 times budget per row?
If we can process each row with 25 queries?
Or $25$ rows fully processed?
Wait, if we process each row with $\approx 5$ queries, total $10000$.
How to check rank with 5 queries?
Impossible to get exact rank in 5 queries for a row of 2000 elements.
So we cannot check rank of an arbitrary value $v$ by scanning.
Wait, we need to select the value $v$ itself.
Maybe we can pick a subset of values to check?
If we select $M$ values $v_1 < v_2 < \dots < v_M$.
Then we can binary search on these values.
But selecting them requires knowing some structure.
We can sample $a_{i, j}$?
Try $a_{1, 1}, a_{1, 2}, \dots, a_{1, k}$?
If $k \le n$, then $a_{1, k}$ is an upper bound?
No, $a_{1, k}$ is the $k$-th smallest in row 1.
There are $k$ elements $\le a_{1, k}$ in row 1.
In other rows, elements might be smaller.
So total elements $\le a_{1, k}$ is $\ge k$.
This gives an upper bound.
Similarly, $a_{k, n}$?
If we pick $a_{i, j}$ such that we hope it is near the $k$-th value.
Maybe the value is around $a_{1, 1}$ shifted by $k/n$?
Consider $a_{x, y}$.
If we pick $x, y$ such that we expect rank $\approx k$.
We don't know the distribution.
But we know $a_{i, j} \le a_{i, j+1}$.
What if we query the diagonal?
Query $a_{1, 1}, a_{2, 2}, \dots, a_{n, n}$. (Cost $n$).
Let these values be $d_1 < d_2 < \dots < d_n$.
These values split the real line.
Also for each $i$, we know $a_{i, j} \le a_{i, i+1}$ etc.
Actually, $a_{1, 1} \le a_{2, 1} \le \dots \le a_{n, 1}$.
And $a_{i, 1} \le a_{i, 2} \le \dots \le a_{i, n}$.
The diagonal elements are not independent.
But $a_{i, i} \le a_{i+1, i+1}$.
Is it possible to use binary search on the *indices* of the diagonals?
No.

Let's pause and think about standard problems.
"Find k-th smallest in sorted matrix with $O(n \log n)$ queries."
Wait, the constraint $50000$ for $N=2000$ implies we can do something slightly slower than $O(n)$, maybe $O(n \log n)$ or $O(\sqrt{n^2})$?
No, $n \log n \approx 22000$. $2.5 n \log n \approx 55000$.
So we have a budget for something around $2.5 n \log n$.
Or maybe $O(n \cdot \text{something small})$.
Wait, is it possible to do this in $O(n \log k)$?
Wait, if we just use the Min-Heap strategy but stop early.
Wait, the heap strategy gives us the exact values.
If $k$ is large, we do too much.
But maybe we don't need to output $k$-th smallest if it is unique?
No, we need the value.

Let's look at the interaction model again.
We can ask `QUERY x y`.
The response is `v`.
The scoring penalizes queries.
There is a possibility that the intended solution uses **binary search on the value** but with a twist.
What if we pick a pivot value $P$ and determine the count of elements $\le P$ efficiently?
If we can determine count in $O(n)$ queries (or $O(\text{something})$), then binary search over values is feasible.
How to count elements $\le P$ in $O(n)$ queries?
Since matrix is sorted, in each row $i$, the position of $P$ can be found.
But we don't know $P$ precisely?
Wait, if we pick $P = a[i][j]$.
We want to find count of elements $\le a[i][j]$.
This requires scanning/matching against $a[i][j]$.
For row $i$, since $a[i, \cdot]$ is sorted, we can use `upper_bound` (binary search).
Cost per row: $\log n$ queries.
Total cost: $n \log n$ queries.
So checking one pivot value $a[i][j]$ takes $O(n \log n)$ queries.
This matches our budget if we do it very few times.
Say 1-2 times.
If we check $a_{mid, mid}$, and suppose count $\ge k$, we know the $k$-th smallest is $\le a_{mid, mid}$.
But we can't just say "it is $a_{mid, mid}$". It might be smaller.
But if $count(a_{mid, mid}) = k$ exactly, then we found the value?
Not necessarily. There might be $a_{mid, mid} + 1$ (next integer) etc.
But the matrix contains values. If we find a value $V$ such that rank($V$) $\ge k$, and we also know that there are no smaller values with rank $\ge k$?
Actually, if we find $V$ such that $rank(V) \ge k$ and for some other value $U < V$ (maybe $U$ is also in matrix), $rank(U) < k$.
But we only found $rank(V)$. We didn't check all $U$.
However, if we find $V$ such that $rank(V) \ge k$ and $rank(\text{previous in matrix}) < k$?
We can find previous values easily?
Wait, $V = a_{mid, mid}$. The "previous in matrix" would be another $a_{i,j}$?
This suggests we need to compare values.
Comparison of arbitrary matrix cells is hard (we can get their values, so we can compare).
So, if we pick a set of candidate values $S = \{a_{i,j} \mid \text{some subset}\}$.
Sort $S$.
Check ranks for each element in $S$.
The one with $rank \ge k$ and predecessor $rank < k$ is the answer.
How many candidates?
We can't query all $a_{i,j}$.
But we can query $a_{1, 1}, a_{2, 2}, \dots, a_{n, n}$. (Cost $n$).
Or $a_{1, 1}, \dots, a_{n, n}$ plus off-diagonals.
Maybe binary search on the index of the diagonal?
Let's try to reduce the candidate set.
Notice that $a_{i, j} \le a_{i+1, j}$ and $a_{i, j} \le a_{i, j+1}$.
This implies $a_{i, j} \le a_{i+1, j+1}$.
Also $a_{1, 1} \le a_{2, 2} \le \dots \le a_{n, n}$.
Wait, this chain holds.
But what about $a_{1, n}$? $a_{1, n} \ge a_{1, 1}$? Yes.
$a_{1, n} \ge a_{2, 2}$? Yes, since $a_{2, 2} \le a_{n, n}$? No.
Actually, $a_{i, j}$ is not monotonic along diagonals of slope 1.
Wait, $a_{i,j}$ vs $a_{i-1, j+1}$?
$a_{i-1, j+1} \le a_{i, j+1}$ (col sorted). And $a_{i, j} \le a_{i, j+1}$.
So $a_{i-1, j+1}$ and $a_{i, j}$ relation is unknown?
Wait, if we increase $i$ (down) and decrease $j$ (left), values decrease.
So $a_{1, n} \le a_{2, n} \le \dots \le a_{n, n}$.
$a_{1, 1} \le a_{2, 1} \le \dots \le a_{n, 1}$.
Basically, the matrix is sorted in "grid" sense.
Is it true that if we pick $a_{n, n}$, it is the maximum? Yes.
$a_{1, 1}$ is minimum.
The question is how to narrow down.
What if we maintain a set of possible candidates and reduce it?
Maybe we can use the budget $O(n \log n)$ to do ONE complete "check" of a pivot?
Wait, $n \log n \approx 22000$.
We have 50000 queries.
So we have a lot of slack.
We can perhaps check a few pivots.
What if we check $a_{1, 1}$? Rank is at least 1.
If $k=1$, answer is $a_{1, 1}$. Done (1 query).
What if we check $a_{i, i}$?
Can we determine if $k$-th element is $< a_{i, i}$?
If we can compute rank($a_{i, i}$), say $R$.
If $R < k$, then $k$-th element $\ge a_{i, i}$.
If $R \ge k$, then $k$-th element $\le a_{i, i}$.
If we do this for several $a_{i, i}$, we can bracket the value.
Suppose we find $a_{i_1, i_1}$ and $a_{i_2, i_2}$ such that $count(a_{i_1, i_1}) < k$ and $count(a_{i_2, i_2}) \ge k$.
Then the value lies in $(a_{i_1, i_1}, a_{i_2, i_2}]$.
But we need to identify the exact value.
The values in the matrix are discrete.
But we don't know them.
Wait, if $count(a_{i_2, i_2}) \ge k$ and $count(a_{i_1, i_1}) < k$, then the $k$-th smallest is $\le a_{i_2, i_2}$ and $> a_{i_1, i_1}$.
Wait, strictly greater?
If $a_{i_1, i_1} < a_{i_2, i_2}$, then yes.
If there are no values strictly between them in the matrix, then $a_{i_2, i_2}$ could be the $k$-th value?
Not necessarily. There could be other values in between.
However, if we just have a tight range of values from the matrix.
What if we collect a set of values $V_{set}$ such that the gaps are small?
Or rather, we use the structure to select a representative subset.
Since we have 50000 queries, we can perhaps perform a binary search on the index $i$ of $a_{i, i}$?
No, we need to check rank of each pivot. Checking rank takes $O(n \log n)$.
If we do it $\log n$ times, total cost is $O(n \log^2 n)$.
$\log(2000) \approx 11$.
$11 \times 11 \times 2000 \approx 242,000$. Too high.
We need $O(n \log n)$ total.
This means we can afford ONLY $O(1)$ rank checks if each check is $O(n \log n)$.
Wait, checking rank of ONE pivot takes $O(n \log n)$ queries.
If we can do it with $O(n)$ queries, we could do $\log n$ times.
Can we check rank of $P=a[i][j]$ in $O(n)$ queries?
For a fixed row $r$, we want to find number of elements $\le P$.
Since $a_{r, c} \le P \iff c \le \text{threshold}$.
We can find threshold by binary search? No, that's $\log n$.
Wait, if we don't do binary search, we can't.
Wait, if we can just query $a[r][j]$ sequentially?
No, $n$ queries per row. Total $n^2$.
So we must use binary search (or interpolation).
But wait!
Is it possible that we don't need to check *all* rows for the pivot?
Suppose we query the pivot $P$.
We want to know how many elements in the matrix are $\le P$.
We know that for any row $r$, the elements $\le P$ form a prefix $1 \dots c_r$.
The boundary moves monotonically?
Wait, $a_{r, c} \le P \implies a_{r-1, c} \le P$.
So $c_r$ (the last index in row $r$ with value $\le P$) satisfies $c_1 \ge c_2 \ge \dots \ge c_n$.
This is a monotonicity property!
The boundaries $c_r$ are decreasing with respect to row index $r$.
Proof: If $a_{r, c} \le P$, then $a_{r-1, c} \le a_{r, c} \le P$.
So if row $r$ goes up to $c_r$, row $r-1$ must go at least up to $c_r$.
So $c_{r-1} \ge c_r$.
Similarly, $c_r$ depends on $a_{r, c}$.
So the shape formed by $\{(r, c) \mid a_{r, c} \le P\}$ is "convex" (specifically, columns are prefixes and boundaries are non-increasing).
We want to compute $\sum c_r$.
Since $c_r$ is monotonic, we can find all $c_r$ in $O(n)$ queries using a sweep!
This is exactly the property of counting inversions or similar structures.
We can use the "staircase walk" to find all $c_r$ in $O(n)$ queries.
Algorithm to count elements $\le P$:
1. Initialize current column pointer `col_ptr` = $n$.
2. Iterate $r$ from $1$ to $n$:
   While `col_ptr` > 0 and $a_{r, \text{col_ptr}} > P$:
     `col_ptr`--;
   Add `col_ptr` to sum.
Since `col_ptr` decreases monotonically, the inner loop runs at most $n$ times in total.
The outer loop runs $n$ times.
Total queries: $O(n)$.
Wait, we need to query $a_{r, \text{col_ptr}}$ to check condition.
Initially for $r=1$, we might need to binary search? Or just start with $n$.
Wait, `col_ptr` starts at $n$ for $r=1$? No, we don't know.
But we know $c_1 \le n$.
Actually, if we don't know $c_1$, we can just start at $n$ and decrement.
Wait, for row 1, if $a_{1, n} \le P$, then $c_1=n$. Else $a_{1, n} > P \implies a_{1, n-1}$ might be $\le P$.
So yes, we can maintain `col_ptr` from $n$ downwards.
Wait, for row 1, we can start `col_ptr` at $n$.
Wait, if $a_{1, n} \le P$, then $c_1 = n$. We don't need to decrement.
But we need to verify if $a_{1, n} \le P$ or not.
Wait, we need to know where the threshold is.
Since $a_{r, c}$ are sorted in row, we want max $c$ such that $a_{r, c} \le P$.
We know $c_r \ge c_{r+1}$.
So we can iterate $r$ from 1 to $n$.
Maintain `ptr` such that we search in range `[0, ptr]`.
Wait, since $c_r \ge c_{r+1}$, if we start with `ptr = n` for $r=1$.
Actually, it's better to iterate backwards?
$c_1 \ge c_2 \ge \dots \ge c_n$.
Start with `ptr = n`.
For $r=1$, while $a_{1, ptr} > P$, decrement `ptr`.
Wait, we don't know if $a_{1, n} > P$.
So we start `ptr` at $n$ (assuming safe upper bound) and shrink.
Wait, $a_{1, n}$ might be $\le P$.
If $a_{1, n} \le P$, `ptr` stays $n$. Correct.
Then for $r=2$, we know $c_2 \le c_1$. So we can start search from `ptr`.
While $a_{2, ptr} > P$, decrement `ptr`.
This ensures we stay in valid range for next row.
Since `ptr` never increases, total decrements is $O(n)$.
Total comparisons: $O(n)$.
So counting elements $\le P$ takes $O(n)$ queries!
Wait, checking `while a[r][ptr] > P` takes 1 query per step.
So total queries for `check(P)` is $O(n)$.
This is great.
Budget 50000. $n=2000$.
We have a budget of roughly 25 checks.
So we can do binary search on values!
Wait, what values to binary search on?
We have candidates $a_{i, j}$.
Actually, we can binary search on the *set of cells*?
Wait, if we binary search on the value range $[L, R]$.
We need to pick a pivot value.
If we pick a pivot value $P$ (from matrix), say $a_{mid, mid}$?
No, we need a value that partitions the space.
Actually, the values of interest are the values present in the matrix.
If we do binary search on the *number of elements*?
Suppose we want to find the $k$-th smallest.
We can use the "counting" oracle.
Oracle: Given a value $V$, return count of elements $\le V$. Cost $O(n)$ queries.
We want to find $V$ such that $count(V) \ge k$ and $count(V') < k$ for some $V' < V$?
Or simply find $V$ with $count(V) \ge k$ and $count(V-\epsilon) < k$?
Since values are discrete, we can just find $V$ such that $count(V) \ge k$.
If we find *any* $V$ with $count(V) \ge k$, it means the $k$-th smallest value is $\le V$.
If we find a $V$ with $count(V) < k$, it means the $k$-th smallest value is $> V$.
This allows us to narrow down the value range.
If we binary search on the "index" of the diagonal?
Wait, the values are not dense.
We can construct a sequence of pivot values from the matrix itself?
For example, query $a_{1, 1}$. $a_{n, n}$.
We have range.
Wait, we can't afford to pick arbitrary $V$ from outside.
We must pick $V$ from the matrix?
If we pick $V \notin Matrix$, the count will tell us how many elements are $\le V$.
If $V$ is not in Matrix, it's fine, it just acts as a gap filler.
However, if we pick $V$ that is extremely unlikely to be in Matrix, say mid-range, we might find out it's not the answer.
But we want to minimize queries.
The crucial observation:
We can find the $k$-th smallest value by finding a cell $(r, c)$ such that $count(a_{r, c}) \ge k$ and $a_{r, c}$ is minimized?
Wait, if we find $P = a_{r, c}$ such that $count(P) \ge k$, we know answer $\le P$.
If we also have $P'$ such that $count(P') < k$, we know answer $> P'$.
Eventually we converge to the true answer.
But the values are sparse.
Is there a sequence of values we can check?
We can try to perform a binary search on the *cells*?
Let's define a pivot cell strategy.
We can pick a set of $M$ cells, query them, and binary search the result?
But the values are not ordered in the grid in a simple way (only locally).
Wait, $a_{i, j}$ values are unsorted globally.
But if we pick a specific cell, say $(r, c)$, its rank can be computed in $O(n)$ queries.
If we pick a diagonal cell $(r, r)$, we can check its rank.
If $rank(r, r) < k$, we know the $k$-th smallest value is $> a_{r, r}$.
Since $a_{r, r}$ is small, we look for something larger.
Wait, if $rank(r, r) < k$, we can discard all elements $\le a_{r, r}$?
We don't know the set of elements.
But if $count(a_{r, r}) < k$, it implies there are fewer than $k$ elements $\le a_{r, r}$.
So $k$-th element is strictly greater than $a_{r, r}$? Not necessarily. If duplicate values exist, it might be equal.
But if distinct, yes.
Assuming distinct for simplicity (worst case handles duplicates).
If $rank(r, r) < k$, then we need to search for values $> a_{r, r}$.
If $rank(r, r) \ge k$, we need to search for values $\le a_{r, r}$.
This looks like we can binary search on the "diagonal index" or similar?
But rank computation depends on value.
Wait, if we just use the "check rank" function as a black box.
We can find the optimal cell?
Wait, we have 50000 queries. $O(n)$ per check.
Max checks $\approx 25$.
So we can do at most $\approx 25$ "checks".
Checking one value takes $2000$ queries.
So we can check 25 values.
How do we pick 25 values?
We can pick cells $(i, j)$.
Which ones?
If we check $(r, c)$ and find it's "too big", we discard the region?
Wait, if we check $(r, c)$ and it's "too big" (rank $\ge k$), we know the $k$-th value is $\le a_{r, c}$.
Can we discard a large portion of the matrix?
If $rank(r, c) \ge k$, we know we are confined to the set of cells $\le a_{r, c}$.
This set forms a Young Diagram.
But identifying this set requires knowing the boundary.
But wait, the set of cells is implicit.
The value we want is somewhere in this region.
Can we restrict our search to a subrectangle?
The region $\le a_{r, c}$ is NOT necessarily a rectangle.
But maybe close to a rectangle?
Actually, if we pick $a_{mid, mid}$ and it turns out $rank(mid, mid) \ge k$, then the $k$-th element is $\le a_{mid, mid}$.
Can we say the $k$-th element is in the subrectangle $1..mid \times 1..mid$?
No. Elements like $a_{1, n}$ could be $\le a_{mid, mid}$?
Wait, if $a_{mid, mid}$ is the value.
We know $a_{i, j} \le a_{mid, mid}$ implies $(i, j)$ is in the "lower-left" boundary.
This boundary passes through $(mid, mid)$? No, it passes through $(mid, mid)$ in the sense that $a_{mid, mid}$ is included.
Is it guaranteed that all $(i, j)$ with $i \le mid, j \le mid$ have $a_{i, j} \le a_{mid, mid}$?
Yes, because $a_{i, j} \le a_{mid, mid}$ (transitivity).
Because $a_{i, j} \le a_{mid, j} \le a_{mid, mid}$.
So the square $[1..mid] \times [1..mid]$ is completely contained in the set of elements $\le a_{mid, mid}$.
So if $rank(a_{mid, mid}) \ge k$, we are NOT sure if it's in the square.
Actually, if $rank(a_{mid, mid}) \ge k$, it means there are at least $k$ elements $\le a_{mid, mid}$.
Since the square has $mid^2$ elements and all are $\le a_{mid, mid}$.
If $mid^2 \ge k$, we might not need to check further?
Wait, if $mid^2 \ge k$, does it guarantee the $k$-th element is inside the square?
No. The $k$-th element could be anywhere.
But wait, we know that the square contributes to the count.
Is it possible that the $k$-th element is outside the square?
If $k \le mid^2$, then the $k$-th smallest element among ALL elements is definitely $\le a_{1, 1}$? No.
Wait, the $k$-th smallest element among the whole matrix.
The square $S = [1..mid] \times [1..mid]$ contains $mid^2$ elements. All of these are $\le a_{mid, mid}$.
If $k \le mid^2$, is it possible that the $k$-th smallest value is $> a_{x, y}$ where $(x, y)$ is outside the square?
Suppose there is an element $e \notin S$ such that $e < \text{all elements in } S$.
But min element in $S$ is $a_{1, 1}$ (which is global min). Max is $a_{mid, mid}$.
Any element $e \notin S$?
Consider $e = a_{mid+1, mid}$. We know $a_{mid+1, mid} \ge a_{mid, mid}$.
So elements outside $S$ (with $i > mid$ or $j > mid$) are generally larger.
Specifically, if $i > mid$, then $a_{i, j} \ge a_{mid, j}$.
If $j > mid$, $a_{i, j} \ge a_{i, mid}$.
Is it true that all $e \notin S$ satisfy $e \ge a_{mid, mid}$?
Yes!
Proof: Let $(i, j) \notin S$. Then either $i > mid$ or $j > mid$.
If $i > mid$, $a_{i, j} \ge a_{mid, j} \ge a_{mid, mid}$ (since $a_{mid, j}$ increases with $j$?? No $a_{mid, j} \ge a_{mid, mid}$ for $j \ge mid$).
Wait, if $i > mid$ and $j < mid$?
Then $a_{i, j} \ge a_{mid, j}$. Is $a_{mid, j} \ge a_{mid, mid}$?
Not necessarily. $j < mid \implies a_{mid, j} \le a_{mid, mid}$.
So elements to the right-bottom of the square are $\ge a_{mid, mid}$.
Elements to the left-top are $\le a_{mid, mid}$.
Wait, elements in the quadrant $(1..mid, mid+1..n)$?
Those have index $j > mid$.
Since $a_{i, j} \ge a_{i, mid}$. Also $a_{i, mid} \ge a_{mid, mid}$?
No, $i \le mid$.
Actually, consider any $(i, j) \notin S$.
If $i > mid$: $a_{i, j} \ge a_{mid, j}$.
If $j > mid$: $a_{i, j} \ge a_{mid, mid}$?
Wait. If $i > mid$, $a_{i, j} \ge a_{mid, j}$.
If $j$ can be anything.
Actually, let's simplify.
Is it true that for any $(i, j) \notin [1..mid] \times [1..mid]$, $a_{i, j} \ge a_{mid, mid}$?
Counterexample: $n=3, mid=1$.
Square is $(1,1)$. Element $(2, 1)$?
$a_{2, 1} \ge a_{1, 1}$? Yes. $a_{2, 1} \ge a_{1, 1}$.
Is $a_{2, 1} \ge a_{1, 1}$?
We know $a_{2, 1} \ge a_{1, 1}$ by column property.
Wait, $a_{mid, mid}$ vs $a_{2, 1}$ with $mid=1$. $a_{1, 1}$.
$a_{2, 1} \ge a_{1, 1}$. So yes.
What about $(1, 2)$? $a_{1, 2} \ge a_{1, 1}$.
So for $mid=1$, yes.
What about general $mid$?
Take $(i, j)$ outside.
Case 1: $i > mid$. Then $a_{i, j} \ge a_{mid, j}$.
If $j > mid$, then $a_{mid, j} \ge a_{mid, mid}$. So $a_{i, j} \ge a_{mid, mid}$.
If $j \le mid$, then $a_{i, j}$ vs $a_{mid, mid}$.
We know $a_{i, j} \ge a_{mid, j}$. And $a_{mid, j} \le a_{mid, mid}$.
So no guarantee $a_{i, j} \ge a_{mid, mid}$.
Example:
1 100
1 100
mid=1. Square is $a_{1, 1}=1$.
Outside: $a_{2, 1} = 1$.
$a_{2, 1} \not> a_{1, 1}$.
It is equal.
So if $a_{i, j} > a_{mid, mid}$, then we can be sure it is larger.
But if equal, we can't distinguish.
However, the problem is to find the $k$-th smallest value.
If $count(a_{mid, mid}) \ge k$, then we know the answer is $\le a_{mid, mid}$.
If $count(a_{mid, mid}) < k$, then answer $> a_{mid, mid}$.
Also, note that if we pick $mid$ such that $mid^2 \approx k$.
If $mid^2 < k$, then $rank(a_{mid, mid}) \ge mid^2 < k$.
So $rank(a_{mid, mid})$ is definitely less than $k$.
Thus $k$-th smallest $> a_{mid, mid}$.
If $mid^2 \ge k$, then we don't know.
But actually, if we can estimate the rank, we can refine.
Wait, this logic suggests we can search for the best $mid$.
But we still need to count exactly.

Let's re-evaluate the strategy:
We want to find $k$-th smallest.
Strategy:
Use binary search on the matrix dimensions?
Specifically, consider the diagonal indices.
We want to find a cell $(r, c)$ that "covers" $k$ elements?
Actually, there is a known algorithm for this problem:
The "divide and conquer on the grid" approach.
Function `solve(rows, cols, k)`:
If `rows * cols <= k`? No.
We want to find the $k$-th smallest in the intersection of rows and cols?
Wait, we want to reduce the problem size.
Actually, the property "sorted rows and cols" means we can apply a strategy similar to finding the $k$-th smallest in a 2D array which is sorted.
Algorithm:
Query $a[mid][mid]$ (or similar).
Calculate rank of $a[mid][mid]$.
If rank $\ge k$:
  The $k$-th smallest value is $\le a[mid][mid]$.
  Does this restrict the search space to rows/cols $\le mid$?
  As discussed, no. $a[i, j]$ might be $\le a[mid, mid]$ even if $i > mid$ or $j > mid$ (though usually not strictly larger).
  Wait, if we have a bound $V = a[mid, mid]$, we know that any $(i, j)$ with $i \le mid$ AND $j \le mid$ has $a_{i, j} \le V$.
  What about others?
  Maybe we can restrict search to submatrices that contain the potential candidates?
  If $rank(a[mid, mid]) < k$, then we know $k$-th value $> a[mid, mid]$.
  Can we safely discard the square $[1..mid] \times [1..mid]$?
  Yes! Because for any $(i, j)$ in that square, $a_{i, j} \le a_{mid, mid} < \text{target}$.
  So all elements in that square are smaller than the target.
  Since $rank(a[mid, mid]) < k$, and there are $mid^2$ elements $\le a[mid, mid]$?
  Wait. The rank of $a[mid, mid]$ is the number of elements $\le a[mid, mid]$.
  If $rank(a[mid, mid]) < k$, then the $k$-th smallest element must be strictly larger than $a[mid, mid]$?
  Wait, duplicates make this tricky.
  If $rank(a[mid, mid]) = R$. This means there are $R$ elements $\le a[mid, mid]$.
  If $R < k$, then we need to pick the $(k-R)$-th element among those $> a[mid, mid]$.
  And we can discard all elements $\le a[mid, mid]$.
  Which elements are $\le a[mid, mid]$?
  At least the square $[1..mid] \times [1..mid]$.
  Are there elements outside this square that are $\le a[mid, mid]$?
  Yes, possibly. (Example $a_{2, 1}$ with $mid=1$).
  So we cannot just discard the square. We must know ALL elements $\le a[mid, mid]$.
  But wait, if we are trying to reduce the search space for recursion.
  Actually, this is exactly the D&C logic.
  If $rank(a[mid, mid]) < k$, we can discard the set of elements $\le a[mid, mid]$.
  We just need to know WHICH elements these are.
  But if we don't know the boundary, we can't efficiently exclude them.
  UNLESS we know the boundary is exactly the square?
  No, boundary is monotonic.
  BUT, if we choose $mid$ carefully?
  What if we iterate?
  
Let's try a simpler greedy strategy with the limited budget.
We have $Q=50000$. $N=2000$.
We can afford $\approx 25$ scans.
One scan computes rank of $a[mid, mid]$.
If we scan $a[mid, mid]$, we get $R$.
If $R < k$, we know target $> a[mid, mid]$.
We also know that the elements in the square $1..mid \times 1..mid$ are all $\le a[mid, mid]$.
Are there other elements $\le a[mid, mid]$?
Maybe, but we know their contribution to count is captured in $R$.
If we know that $a[mid, mid]$ is "too small", we need to go higher.
The "highest" we should check in the diagonal is larger.
So we can double $mid$?
If $R < k$, we need a larger value. Try $mid' = \text{something}$.
If $R \ge k$, we need smaller.
Since values on diagonal $a_{i, i}$ are non-decreasing ($a_{i, i} \le a_{i+1, i+1}$), we can binary search on the index $i$ of the diagonal!
Wait, is $a_{i, i}$ sorted?
$a_{i, i} \le a_{i+1, i} \le a_{i+1, i+1}$?
$a_{i, i} \le a_{i+1, i}$ (col sorted).
$a_{i+1, i} \le a_{i+1, i+1}$ (row sorted).
Yes, $a_{1, 1} \le a_{2, 2} \le \dots \le a_{n, n}$.
So the sequence of values along the main diagonal is sorted.
Let's call them $d_1, d_2, \dots, d_n$.
We want to find $k$-th smallest in the whole matrix.
For any $i$, we know $a_{i, i}$ is the value.
If we calculate rank of $a_{i, i}$, say $R_i$.
Note that $R_i$ is NOT necessarily monotonic with $i$?
If $d_i$ increases, the set of elements $\le d_i$ expands.
So $R_i$ must be non-decreasing with $i$.
Because $d_1 \le d_2 \implies \{x \le d_1\} \subseteq \{x \le d_2\} \implies |S_1| \le |S_2|$.
So $rank(d_i)$ is monotonic!
We can binary search for index $idx$ such that $rank(d_{idx}) \approx k$.
Algorithm:
1. Construct candidate set of diagonal values: $a_{1, 1}, \dots, a_{n, n}$.
   Wait, we don't need to query them one by one first if we can compute rank efficiently?
   To binary search on index, we need to evaluate $rank(d_{mid})$.
   Computing $rank(d_{mid})$ takes $O(n)$ queries.
   If we do this $\log n$ times, total queries $n \log n \approx 22000$.
   This is well within 50000!
   
   So, strategy:
   Binary search on index $i \in [1, n]$.
   Maintain range $[L, R]$.
   In step, pick $mid = (L+R)/2$.
   Compute $V = a_{mid, mid}$.
   Compute $cnt = count(V)$. (Takes $O(n)$ queries).
   If $cnt \ge k$, it implies the $k$-th smallest value is $\le V$ (so we need to look for smaller values).
   So we try to restrict search to $i \le mid$?
   Wait, if $cnt \ge k$, it means the $k$-th element is $\le V$.
   Since $d_1 \le \dots \le d_{mid} \le V$, we know $V$ is a valid upper bound.
   But we want to minimize the value.
   Can we say that if $cnt \ge k$, then the answer lies among values $\le V$? Yes.
   But we want to find the specific $k$-th value.
   Wait, if we find a $V$ such that $cnt(V) \ge k$ and $cnt(prev\_d) < k$, then $V$ is the answer?
   Not exactly. It could be that the $k$-th value is $V$ (if duplicates) or something slightly smaller.
   But since $d_i$ are values in the matrix.
   If we find $d_{mid}$ such that $rank(d_{mid}) \ge k$ and $rank(d_{mid-1}) < k$.
   Then the $k$-th smallest value is between $d_{mid-1}$ and $d_{mid}$ (inclusive).
   Actually, since $d_{mid-1} < d_{mid}$ (assuming distinctness, or just strict inequality for values).
   If values are not distinct, we have to handle equality.
   Suppose $d_{mid-1} \le V \le d_{mid}$.
   If we can confirm that $d_{mid}$ is indeed the $k$-th value.
   Wait, if $rank(d_{mid}) \ge k$, and $d_{mid}$ is in the matrix.
   The values in matrix are from the set of all $a_{i, j}$.
   If we binary search, we can narrow down to a range of values.
   Eventually, we might end up with a single value $d_{idx}$ such that $rank(d_{idx}) \ge k$ and $rank(d_{idx-1}) < k$.
   Is $d_{idx}$ necessarily the $k$-th smallest?
   Suppose $d_{idx-1} < V < d_{idx}$ where $V$ is the true $k$-th value.
   Then $V$ must be in the matrix.
   But $d_{idx-1}$ and $d_{idx}$ are consecutive in the diagonal? No, diagonal indices are discrete.
   But values $d_i$ are a subset of matrix values.
   If the true $k$-th value $V$ is NOT in the diagonal, then it is between some $d_u$ and $d_{u+1}$.
   Wait, if $V$ is in matrix but not in diagonal.
   Then we might fail to pinpoint it if we only check diagonal.
   However, we can improve the candidate set.
   Instead of just diagonal, we can check other cells?
   Wait, if $V$ is between $d_u$ and $d_{u+1}$, it means $d_u < V < d_{u+1}$.
   Since $d_u \le a_{u, u}$ and $d_{u+1} \ge a_{u+1, u+1}$? No.
   Wait, $d_i = a_{i, i}$.
   The matrix contains values that are not necessarily on diagonal.
   Example: $a_{1, 2}$ might be the answer. $a_{1, 2}$ vs $a_{2, 2}$?
   $a_{1, 2} \le a_{2, 2}$. $a_{1, 2} \ge a_{1, 1}$.
   It could be that $d_u < a_{1, 2} < d_{u+1}$ is false?
   Actually, $d_u$ and $d_{u+1}$ are just points.
   If we just use the diagonal, we might miss values.
   However, notice the scoring. If we find a value $V$ with $rank(V) \ge k$ and no other values between $prev(V)$ and $V$ in matrix...
   Wait, the diagonal is just ONE path.
   Maybe we can refine the search range.
   If we find a cell $(r, c)$ such that $rank(r, c) \approx k$.
   How to find such a cell efficiently?
   We can use the diagonal binary search to find a *good* pivot cell.
   Specifically, find $i$ such that $rank(a_{i, i}) \approx k$.
   Let's refine this.
   If $rank(a_{i, i}) < k$, we need to go to a larger value.
   The largest values in the matrix are $\ge a_{i, i}$.
   If $rank(a_{i, i}) \ge k$, we go smaller.
   After binary search, we find a cell $C^* = (i, i)$ such that $rank(C^*) \ge k$ and $rank(predecessor) < k$.
   Now, is $C^*$ the $k$-th smallest?
   Suppose true answer is $A$.
   If $A = a_{i, i}$, we found it.
   If $A \neq a_{i, i}$, then $A$ is some other value.
   But wait.
   If $rank(a_{i, i}) \ge k$, it means there are at least $k$ elements $\le a_{i, i}$.
   This implies $A \le a_{i, i}$.
   If $rank(a_{prev}) < k$, then $A > a_{prev}$.
   So $a_{prev} < A \le a_{i, i}$.
   This implies $A$ lies between two diagonal values.
   But can $A$ be significantly different?
   Actually, consider the structure.
   If we find a "tight" range $[V_{low}, V_{high}]$.
   We know $V_{low} = a_{prev}$ and $V_{high} = a_{i, i}$.
   Also we know that for all $(r, c)$ such that $a_{r, c} < V_{high}$, we know it is $\le a_{prev}$? No.
   This is tricky.
   
   Wait, there is a much simpler logic for this specific problem "Matrix K-th Smallest (Interactive)".
   The problem asks for the k-th smallest.
   Maybe we can just query the matrix using the heap approach but optimized?
   Or maybe just binary search on value using the "count" function.
   If we can implement the "count" function efficiently ($O(n)$ queries), we can perform binary search on the value range.
   But the range is $10^{18}$.
   However, we only care about values present in the matrix.
   Can we binary search on the *index* of the diagonal AND the *offset*?
   Wait, if we just want to find the $k$-th smallest value.
   Let's assume the answer is one of the diagonal elements. Why?
   It's not. E.g. $10, 20$
             $11, 21$
   Diagonal: $10, 21$. Median is 15.
   Wait, if $k=2$ (out of 4), we want 11.
   Diagonal gives 10, 21.
   Rank(10)=1 < 2. Rank(21)=4 >= 2.
   We picked 10 (index 1) and 21 (index 2).
   We found $10 < 11 \le 21$.
   So $A \in [10, 21]$.
   If we had a better candidate set?
   What if we include off-diagonal candidates?
   Notice that if we have $a_{r, c}$, we can check its rank.
   Maybe we can select candidates $a_{r, c}$ such that they cover the matrix sufficiently?
   Actually, we can simply maintain a set of candidate values.
   Start with empty set.
   We know $a_{1, 1}$ is min.
   We can query $a_{1, 1}$, get $v_1$. Add to candidates.
   Then we know answer $\ge v_1$.
   We want to find $k$-th.
   Maybe we can check $a_{i, j}$ that are likely to be the $k$-th.
   Like $a_{1, k}$?
   Wait, $a_{1, k}$ might be out of bounds if $k > n$.
   If $k \le n$, query $a_{1, k}$. Its rank is $> k$.
   If $k > n$, query $a_{i, j}$ such that $i \cdot j \approx k$?
   Or $a_{i, \lfloor k/n \rfloor}$?
   Basically, pick cell $(i, j)$ such that expected rank is $k$.
   Since $a_{i, j}$ values increase with $i, j$.
   So $a_{1, 1}$ is 1st. $a_{n, n}$ is $n^2$-th.
   Pick $(i, j)$ such that $i+j$ is related to $k$?
   No, ranks don't correspond to indices linearly.
   However, due to symmetry/sorting, $a_{i, j}$ roughly covers rank $i \times j$?
   No, $a_{i, j} \ge i \times \text{something}$.
   Wait, $a_{i, j} \ge a_{1, j} \ge a_{1, 1}$.
   But we can bound the rank of $a_{i, j}$?
   Lower bound: $rank(a_{i, j}) \ge$ number of $(u, v)$ such that $u \le i, v \le j$. This is $i \cdot j$.
   Upper bound: $rank(a_{i, j}) \ge$ number of $(u, v)$ such that $u \le n, v \le j$? No.
   Upper bound: rank of $a_{i, j}$ is at most...
   Actually, $a_{i, j} \ge$ rank $i \cdot j$ is not always true (due to values overlap).
   But $a_{i, j} \ge a_{i-1, j}$ and $a_{i, j} \ge a_{i, j-1}$.
   This implies that the cell $(i, j)$ is $\ge$ the $i$-th element in col $j$?
   This doesn't give a rank bound.
   Wait, rank of $a_{i, j}$ is definitely $\ge i \cdot j$?
   Let's check.
   If values were strictly increasing, then $a_{1, 1} < a_{1, 2} \dots$
   Then $a_{i, j}$ would be $\ge a_{i, j} \ge a_{i, j-1} \dots \ge a_{i, 1}$.
   Wait, is it true that $a_{i, j} \ge$ the value at rank $i \cdot j$?
   Consider $2 \times 2$. $a_{1, 1} \le a_{1, 2}$, $a_{1, 1} \le a_{2, 1}$. $a_{2, 2} \ge a_{2, 1}, a_{1, 2}$.
   Values: 1 2
          2 2
   $a_{1, 2}=2$. Rank is 3. $1 \cdot 2 = 2$. $2 \ge 2$. True.
   $a_{2, 1}=2$. Rank 3. $2 \cdot 1 = 2$.
   $a_{2, 2}=2$. Rank 3. $2 \cdot 2 = 4$. False.
   So rank $\ge i \cdot j$ is FALSE.
   However, rank $\ge \text{count of } (u, v) \le (i, j)$ is not useful.
   
   Back to the drawing board.
   The $O(n \log n)$ solution using "count" function with diagonal binary search is promising because it respects the query limit.
   But we need to handle the "missing" values between diagonal elements.
   Is it possible to check just the diagonal?
   If we find $a_{i, i}$ such that $rank(a_{i, i}) \approx k$.
   If $rank(a_{i, i}) \ge k$, we suspect answer $\le a_{i, i}$.
   Wait, if $rank(a_{i, i}) \ge k$, does it mean we can restrict search to a submatrix?
   Not necessarily.
   BUT, notice the query limit is 50000.
   If we do binary search on diagonal indices, we perform $O(\log n)$ checks.
   Each check costs $O(n)$. Total $O(n \log n) \approx 22000$.
   This is perfect.
   We find a diagonal element $V_{diag}$ which is an upper bound for the answer (or close).
   Actually, we want the smallest $V$ in diagonal such that $rank(V) \ge k$.
   Let this be $V_{best} = a_{i_{opt}, i_{opt}}$.
   So $rank(V_{best}) \ge k$ and $rank(a_{i_{opt}-1, i_{opt}-1}) < k$ (if exists).
   Now, we know the answer is $\le V_{best}$.
   Also, we know that $V_{best}$ is in the matrix.
   And we know that $rank(V_{best})$ is the count.
   If $rank(V_{best}) == k$, then $V_{best}$ is the answer.
   If $rank(V_{best}) > k$, then the answer is smaller than $V_{best}$.
   Wait, if $rank(V_{best}) > k$, it means there are $> k$ elements $\le V_{best}$.
   So the $k$-th smallest is strictly smaller than $V_{best}$ (unless duplicates, then equal).
   So we need to find a value in the matrix that is smaller than $V_{best}$ but has rank $\ge k$?
   Wait, if $rank(V_{best}) > k$, we want to find a value $V'$ such that $rank(V') \ge k$.
   Since we only checked the diagonal, we might have skipped values.
   But wait, if we find $V_{best}$ is the smallest diagonal element with $rank \ge k$.
   Does it imply that the answer is $V_{best}$?
   Suppose the true $k$-th value is $X < V_{best}$.
   Then $rank(X) \ge k$ (since $X$ is the $k$-th value, count of elements $\le X$ is at least $k$).
   But we chose $V_{best}$ to be the SMALLEST diagonal element with $rank \ge k$.
   This implies for all diagonal elements $d < V_{best}$, $rank(d) < k$.
   If $X$ was in the matrix and $X < V_{best}$, why did we miss it?
   Because we didn't check it.
   However, if $rank(d) < k$ for all $d < V_{best}$ (where $d$ are diagonal elements).
   This means the $k$-th element $X$ is NOT among the diagonal elements (or if it is, we found it).
   Wait, if $X$ is not on diagonal, but $rank(X) \ge k$.
   And we have diagonal elements $d_1 < d_2 < \dots < V_{best}$.
   For all $d_j < V_{best}$, $rank(d_j) < k$.
   But $X < V_{best}$ implies $rank(X) < rank(V_{best})$.
   This is not contradictory.
   But $X$ could be the $k$-th value.
   So we need to check if $V_{best}$ is the true answer.
   It is if there are no other values between $X$ and $V_{best}$ that change the rank?
   Actually, if we find $V_{best}$ with $rank(V_{best}) \ge k$.
   Is it possible that $V_{best}$ is NOT the $k$-th value?
   Yes, if $rank(V_{best}) > k$.
   But if $rank(V_{best}) > k$, then there are more than $k$ elements $\le V_{best}$.
   So $k$-th smallest is $\le$ some element $< V_{best}$.
   If no diagonal element $d < V_{best}$ has rank $\ge k$, then the $k$-th element is NOT a diagonal element?
   Wait, if $rank(d) < k$ for all $d < V_{best}$, then we know that for all $x \in Diagonal, x < V_{best} \implies x \notin \{ \text{elements with rank } \ge k \}$.
   Wait, this doesn't mean $x$ has rank $< k$. It means its rank is $< k$.
   So any element $Y$ with rank $\ge k$ must be $\ge V_{best}$?
   No. If $Y$ is not in diagonal, we don't know.
   Wait. If $rank(Y) \ge k$, does it imply $Y \ge V_{best}$?
   Suppose $Y < V_{best}$. Then $rank(Y) \ge k$ (since $Y$ is $k$-th).
   But we checked all diagonal elements $< V_{best}$ and found their ranks $< k$.
   Does this imply $Y$ must be $\ge V_{best}$?
   Not necessarily.
   However, consider the property.
   If $Y < V_{best}$, does it force $Y$ to be "close" to some diagonal element?
   Maybe we can simply query the matrix to find elements?
   Wait, if $rank(V_{best}) \ge k$, we can check the rank of $V_{best}$ more precisely?
   Wait, $V_{best}$ IS the value we want if its rank equals $k$ (count of elements $\le V_{best}$ equals $k$).
   If count $> k$, we have a problem.
   BUT, maybe we can check if $count(V_{best}) == k$?
   Yes. If $rank(V_{best}) == k$, then $V_{best}$ is the $k$-th smallest value.
   If $rank(V_{best}) > k$, then we need to find a smaller value.
   Can we find a smaller value with rank $\ge k$?
   If we do this recursively?
   No, the set of values is not structured enough to search "backwards".
   
   Wait, look at the constraints again. $n \le 2000$.
   Maybe there's a simpler property.
   $rank(a_{i, j}) \ge i \cdot j$? No.
   $rank(a_{i, j}) \ge i + j - 1$? No.
   
   Let's check if the diagonal strategy works assuming the matrix is "dense" or "random"? No, must work for worst case.
   However, there is one trick:
   If we find $V_{best} = a_{i, i}$ with $rank(V_{best}) \ge k$ and $rank(a_{prev}) < k$.
   Then the $k$-th smallest value is either $V_{best}$ or it is one of the values strictly smaller than $V_{best}$ that has rank $\ge k$.
   But if no diagonal element has rank $\ge k$ smaller than $V_{best}$, then there are no diagonal elements $\ge k$.
   Is it possible that a non-diagonal element is the answer?
   Yes.
   BUT, if the answer is not on the diagonal, then the answer is some $X < V_{best}$.
   Is it possible to find $X$ efficiently?
   Wait, if $rank(a_{i, i}) \ge k$, we know answer $\le a_{i, i}$.
   If $rank(a_{i, i}) > k$, we need something smaller.
   But maybe we can check $a_{i, j}$ for some $j$?
   Specifically, we can try to find the "boundary" for the chosen $V_{best}$.
   The boundary separates $\le V_{best}$ and $> V_{best}$.
   We want to know if we can shift the boundary "up" or "left".
   Actually, the problem reduces to finding the boundary of the region $\le V_{best}$ such that its area is $k$.
   We have $V_{best}$. We computed its count.
   If $rank(V_{best}) \ge k$, it means the region $\le V_{best}$ contains at least $k$ cells.
   We want to remove elements from this region until we have $k$ elements.
   Wait, removing elements means moving the boundary.
   Moving the boundary means decreasing the threshold value?
   No, we are just identifying the $k$-th smallest element.
   If $rank(V_{best}) \ge k$, it means the $k$-th smallest value is in the set of cells $\le V_{best}$.
   Let's call this set $S$. We want to find the $k$-th smallest element in $S$.
   But $S$ is defined by values $\le V_{best}$.
   This doesn't help much because $S$ is large.
   Wait, if we just want to find the $k$-th value.
   Actually, if we just find $a_{i, j}$ such that $rank(a_{i, j}) \approx k$.
   Wait, if we can find $a_{i, j}$ such that $rank(a_{i, j}) = k$. Then we are done.
   If $rank(a_{i, j}) > k$, then $k$-th is smaller.
   If $rank(a_{i, j}) < k$, then $k$-th is larger.
   Since we have limited queries, we cannot check all cells.
   However, the diagonal gives us a monotonic sequence of values $d_1, \dots, d_n$.
   Their ranks $r_1, \dots, r_n$ are monotonic.
   We find index $p$ such that $r_p \ge k$ and $r_{p-1} < k$.
   This implies that for all $j < p$, $d_j$ has rank $< k$.
   This implies that the $k$-th smallest value is $\ge d_{p-1}$?
   Wait, $rank(d_{p-1}) < k$ implies that there are fewer than $k$ elements $\le d_{p-1}$.
   So the $k$-th smallest element is strictly greater than $d_{p-1}$ (unless $d_{p-1}$ equals the $k$-th value and we have ties? No, if rank is count of $\le$, then rank $< k$ means count of $\le$ is $< k$. So there are at least $k$ elements $> d_{p-1}$. So $k$-th element $> d_{p-1}$.)
   So $k$-th element $> d_{p-1}$.
   Also, $rank(d_p) \ge k$ implies $k$-th element $\le d_p$.
   So $d_{p-1} < \text{ans} \le d_p$.
   Since $d_{p-1}$ and $d_p$ are values from the matrix.
   If there are NO values in the matrix strictly between $d_{p-1}$ and $d_p$, then ans must be $d_p$.
   How to ensure no values between them?
   This is not guaranteed.
   BUT, wait.
   If $rank(d_p) \ge k$, then ans $\le d_p$.
   If $rank(d_{p-1}) < k$, then ans $> d_{p-1}$.
   If ans was strictly less than $d_p$, it would be some value $X \in (d_{p-1}, d_p)$.
   But $d_p = a_{p, p}$. $d_{p-1} = a_{p-1, p-1}$.
   Is it possible that all matrix values are $\{ \dots, d_{p-1}, d_p, \dots \}$?
   If the matrix values are dense, this fails.
   But wait, we can check other cells.
   For example, $a_{p, p-1}$.
   We know $a_{p, p-1} \le a_{p, p} = d_p$.
   Also $a_{p, p-1} \ge a_{p-1, p-1} = d_{p-1}$.
   So $a_{p, p-1}$ is in the range $[d_{p-1}, d_p]$.
   We can check $rank(a_{p, p-1})$.
   If $rank(a_{p, p-1}) \ge k$, then ans $\le a_{p, p-1}$.
   Repeat this?
   Basically, we can maintain a "bounding box" of possible answers.
   Initially $L=a_{1,1}, R=a_{n,n}$.
   We check middle of diagonal.
   Based on rank, we shrink the box.
   Wait, we can't just shrink the box of values. We need to pick a new cell to check.
   Strategy:
   Maintain a range of rows $[r_1, r_2]$ and columns $[c_1, c_2]$? No.
   Actually, we can maintain a set of candidate cells.
   Start with $\{(1, n)\}$?
   Wait, the simplest robust approach with the budget is:
   Binary search for index $i$ on the diagonal such that $rank(a_{i, i}) \ge k$.
   Let this be $i_{opt}$.
   Then $a_{i_{opt}, i_{opt}}$ is an upper bound.
   Let's check $a_{i_{opt}, 1}$.
   $a_{i_{opt}, 1}$ is $\le a_{i_{opt}, i_{opt}}$.
   And $a_{1, 1} \le a_{i_{opt}, 1}$.
   If we check $a_{i_{opt}, 1}$, we can refine.
   Actually, we can try to find the exact cell $(r, c)$ such that $rank(r, c) \approx k$.
   Wait, we can iterate on $r$ for the optimal $c$?
   For a fixed row $r$, we want to find $c$ such that $rank(a_{r, c}) \approx k$.
   But we can't query rank of all $a_{r, c}$.
   However, notice that $rank(a_{r, c})$ is roughly increasing with $c$.
   We can find the boundary $c$ for a fixed $r$ using binary search on the row?
   Wait, to find $c$ we need to compare $a_{r, c}$ with a value $V$.
   But we don't have $V$.
   We only have $rank(a_{r, c})$ which depends on all elements.
   
   Let's step back.
   We have $n \log n$ queries available for one rank check.
   Actually, we can do $O(n \log n)$ to find the $k$-th smallest using the Merge-K-sort idea if we could do $O(\log n)$ per row.
   Wait, merge sorted lists to find $k$-th.
   Algorithm:
   1. Pick a row $r$. Query all elements in row $r$. ($n$ queries).
   2. Use this row to filter? No.
   
   Wait, the "count" function is the key.
   We can compute rank of ANY value $V$ in $O(n)$ queries.
   The problem is we don't have values.
   But we CAN construct values from the matrix.
   Let's just maintain a "search space" of values?
   What if we just query the diagonal elements and their ranks?
   We have $\approx 25$ queries budget for full rank check.
   Wait, $50000$ queries. $n=2000$.
   $O(n)$ is $2000$.
   $50000 / 2000 = 25$.
   So we can do 25 rank checks.
   With 25 checks, we can narrow down the value range significantly?
   Wait, if we have 25 values $v_1 < v_2 < \dots < v_{25}$ from the matrix.
   If we sort them and check their ranks.
   If we find $v_j$ such that $rank(v_j) \ge k$ and $rank(v_{j-1}) < k$.
   Then $v_j$ is the candidate.
   Is it the answer?
   Only if $rank(v_j) == k$ or we are confident.
   Wait, if $rank(v_j) > k$, then answer is smaller than $v_j$.
   If we have many values, we can binary search on the set of values.
   We need to collect a good set of values.
   Which values?
   $a_{i, j}$ where $i \cdot j$ or something is related to $k$.
   But we can just pick diagonal elements $a_{1,1}, \dots, a_{n,n}$.
   We can check all $n$ diagonal elements?
   Cost $n \times O(n) = O(n^2) \approx 4 \times 10^6$.
   Too slow.
   But we only need to find one good pivot.
   Actually, if we just do one rank check on $a_{mid, mid}$.
   If $rank(a_{mid, mid}) \ge k$, we know ans $\le a_{mid, mid}$.
   We can then query $a_{mid/2, mid/2}$?
   Wait, $rank(a_{mid/2, mid/2})$ is likely much smaller than $k$.
   If we just follow the diagonal, we find the smallest $a_{i, i}$ such that $rank \ge k$.
   Let this be $V_{ans}$.
   If $rank(V_{ans}) == k$, we are done.
   If $rank(V_{ans}) > k$, we need a value smaller than $V_{ans}$.
   The diagonal strategy gave us a value $d_p$ such that $rank(d_p) \ge k$ but $rank(d_{p-1}) < k$.
   This implies $d_{p-1} < ans \le d_p$.
   If $ans < d_p$, then $ans$ is in $(d_{p-1}, d_p)$.
   We need to check if there are any values in this interval.
   We can check $a_{p, p-1}$?
   $a_{p, p-1}$ is between $d_{p-1}$ and $d_p$ (usually).
   Specifically $a_{p, p-1} \ge a_{p-1, p-1}$ and $a_{p, p-1} \le a_{p, p}$.
   Wait, $a_{p, p-1} \le a_{p, p}$ is obvious.
   $a_{p, p-1} \ge a_{p-1, p-1}$?
   We know $a_{p, p-1} \ge a_{p-1, p-1}$ (since row $p$ is sorted? No. Col $p-1$ is sorted. $a_{p, p-1} \ge a_{p-1, p-1}$).
   Yes. So $a_{p, p-1} \in [d_{p-1}, d_p]$.
   We can check its rank.
   If $rank(a_{p, p-1}) \ge k$, then ans $\le a_{p, p-1}$.
   If $rank(a_{p, p-1}) < k$, then ans $> a_{p, p-1}$.
   This allows us to refine the range to $[a_{p, p-1}, a_{p, p}]$.
   And we can continue to $a_{p, p-2}, a_{p+1, p}$, etc.
   Essentially, we can search the neighborhood of $(p, p)$.
   The number of steps to refine will be limited by the number of "gaps" or simply by the budget.
   But wait, if we have $rank(V_{ans}) > k$, we need to go down.
   We can try $a_{p, p-1}, a_{p, p-2} \dots$ until we find one with rank $\approx k$.
   The number of queries to check neighbors of $(p, p)$?
   We can just check cells $(p, j)$ for various $j$.
   Or $(i, p)$.
   Since we have a budget of 25 queries, and $n=2000$, we can't check many.
   But maybe the diagonal is a very good estimator.
   With 25 queries, we can find the diagonal index $p$ in $O(\log n)$?
   Wait, we can't afford $O(\log n)$ full rank checks if each check takes $O(n)$.
   If each check takes $O(n)$, then $\log n \approx 11$ checks takes $11 \times 2000 = 22000$.
   This leaves 28000 queries.
   So we can do:
   1. Binary Search on diagonal index $i \in [1, n]$.
      Check $a_{mid, mid}$ rank.
      Cost: $11 \times 2000 = 22000$.
      Result: Found $i_{opt}$ such that $rank(a_{i_{opt}, i_{opt}}) \ge k$ and $rank(a_{i_{opt}-1, i_{opt}-1}) < k$.
      Let $V_{low} = a_{i_{opt}-1, i_{opt}-1}$ and $V_{high} = a_{i_{opt}, i_{opt}}$.
   2. Now we know ans $\in [V_{low}, V_{high}]$.
      We want to find the exact value.
      Note that we know $V_{high} = a_{i_{opt}, i_{opt}}$.
      We can check $a_{i_{opt}, i_{opt}-1}$ (cell left of pivot).
      Wait, if $rank(V_{high}) > k$, then ans $< V_{high}$.
      If $rank(V_{high}) = k$, then ans $= V_{high}$.
      Wait, we need to be careful.
      If $rank(V_{high}) \ge k$, it doesn't mean $ans = V_{high}$.
      It means $ans \le V_{high}$.
      If $rank(a_{i_{opt}-1, i_{opt}-1}) < k$, it means $ans > V_{low}$.
      So $V_{low} < ans \le V_{high}$.
      If we find a cell $C$ in the range such that $rank(C) \ge k$ and $C < V_{high}$ (impossible if $C < V_{high}$ and $rank(C) \ge k$ implies we should have picked $C$ as $V_{high}$? No).
      Wait, if $ans < V_{high}$, then $ans$ is some value $X < V_{high}$.
      But we know $X > V_{low}$.
      And we haven't checked any cell with value in $(V_{low}, V_{high})$.
      Wait, the cell $(i_{opt}, i_{opt}-1)$ is $\le a_{i_{opt}, i_{opt}}$ and $\ge a_{i_{opt}-1, i_{opt}-1}$.
      So $a_{i_{opt}, i_{opt}-1}$ is a candidate to bridge the gap.
      Actually, we can iterate:
      Current range of values $[V_A, V_B]$.
      If $rank(V_B) \ge k$, we update $V_B$ to a smaller value.
      We want to find the smallest $V$ in matrix such that $rank(V) \ge k$.
      Since we have $a_{i, j}$ on the diagonal, we used them to bracket.
      We can extend the bracket by including neighbors of the diagonal.
      But we only have ~25 rank checks left.
      If we check $a_{i_{opt}, i_{opt}-1}$, that's 1 check.
      Then $a_{i_{opt}, i_{opt}-2}$...
      How many do we need?
      If the matrix is random, gaps are small.
      But worst case: $V_{low}$ is $1$, $V_{high}$ is $10^{18}$.
      But values in matrix must satisfy sorted constraints.
      The number of elements between $V_{low}$ and $V_{high}$?
      If $rank(V_{high})$ is large and $rank(V_{low})$ is small.
      Wait, $rank(V_{high}) - rank(V_{low})$ tells us how many elements are in $(V_{low}, V_{high}]$.
      Let this difference be $\Delta$.
      If $\Delta$ is large, then $ans$ is not uniquely determined by just diagonal values?
      Wait, if $\Delta = rank(V_{high}) - rank(V_{low}) \ge 1$.
      Then $ans$ is one of the values in that set.
      If we can't find it, we fail.
      But wait, we have enough queries to just check the matrix structure?
      Actually, we can solve this by realizing that we can binary search on the coordinate pair $(i, j)$ in a "tree" of candidates?
      
   Wait, look at the problem constraints.
   Maybe we don't need to worry about "between diagonal elements".
   Actually, the solution might be:
   Just output $a_{i, i}$ if $rank(a_{i, i}) \ge k$?
   No, that's not correct.
   However, notice that we can afford $O(n \log n)$ queries.
   With $n=2000$, $n \log n \approx 22000$.
   We have 50000.
   So we have plenty.
   What if we simply binary search on the value range, but using the "count" function which costs $O(n)$ queries.
   But we need to guide the binary search.
   If we binary search on $i$ of diagonal, we find a cell with rank $\ge k$.
   Let this be $P$. $rank(P) \ge k$.
   Now we know the $k$-th element is $\le P$.
   Can we refine?
   Actually, if we pick the *first* cell (reading row-major) that has rank $\ge k$? No, we can't read row-major.
   However, the query limit is loose.
   Is it possible to check *all* $a_{i, j}$? No.
   But wait, if we check all $a_{i, j}$, it's $n^2 = 4 \times 10^6$.
   Wait, the scoring formula is weird.
   It rewards low usage.
   But penalty is 0 if usage $\ge n^2$.
   This implies we MUST use less than $n^2$ queries.
   
   Actually, maybe the solution is:
   Just use the diagonal binary search, and when $rank(a_{i, i}) \ge k$, assume $a_{i, i}$ is the answer.
   Why?
   If $rank(a_{i, i}) \ge k$, then $a_{i, i}$ is at least the $k$-th smallest.
   Is it possible there is a smaller element?
   Yes.
   But if $rank(a_{i, i}) == k$, then it is EXACTLY the $k$-th (assuming distinct).
   If $rank(a_{i, i}) > k$, then it is strictly larger than the $k$-th element.
   Wait.
   If $rank(a_{i, i}) \ge k$.
   If we just output $a_{i, i}$, we might be wrong.
   But maybe for the problem constraints, there is a property.
   Or maybe we can do better.
   
   Wait, I remember a solution for this problem (Codeforces 1296E? No).
   Actually, there is a problem "K-th Number" on sorted matrix.
   Standard solution is $O(n \log^2 n)$ or $O(n \log n)$.
   Here, we have 50000 queries.
   We can implement the $O(n \log n)$ solution.
   Algorithm:
   Maintain a set of active rows.
   Actually, the simplest implementation is to find the $k$-th smallest by maintaining a set of potential candidates.
   Wait, we can do this:
   Use a priority queue to store "min of row".
   We insert $(val, row\_index, col\_index)$ into PQ.
   Initially insert $a_{r, 1}$ for all $r$? No, $O(n)$ queries.
   Actually, we insert $(a_{r, 1}, r, 1)$.
   Extract min. Add its right neighbor $a_{r, 2}$.
   This gives sorted order.
   To find $k$-th, we need to extract $k$ times.
   If $k$ is large, this is $O(k \log n)$.
   Since $k$ can be $n^2$, this is too slow.
   Wait, but we can optimize the heap.
   We only need to find $k$-th.
   If we maintain the heap of size $k$?
   We can use a Min-Heap of the "smallest of each row"? No.
   The standard algorithm for $k$-th smallest in sorted matrix is using a heap to merge rows.
   Complexity $O(k \log n)$.
   But we can also use binary search on the value.
   Complexity $O(n \log n)$ to check a value.
   We have 50000 queries.
   $n \log n \approx 22000$.
   So we can perform 2-3 checks.
   But to perform binary search on values, we need to know a range.
   Or we need to binary search on the matrix cells?
   Wait, if we can perform one check with high precision.
   Actually, if we use the "count" check, we can binary search on the *matrix cells* using a 2D binary search?
   If we define a predicate $P(r, c)$: is $a_{r, c}$ the $k$-th smallest?
   Hard to define.
   
   Wait, the query limit is 50000.
   The limit is $25 \times n$.
   This strongly suggests an algorithm that is $O(n)$ or $O(n \log n)$ per pass, with a constant number of passes.
   The "Diagonal Binary Search" strategy is $O(n \log n)$ queries (total).
   And it narrows down the answer to a range between two diagonal elements.
   Can we just pick the answer to be the $a_{i, i}$ corresponding to the diagonal index found?
   No, we saw counterexamples.
   However, maybe we can check neighbors to refine.
   If we check $a_{i, i}$ and find $rank < k$, we increment $i$.
   If $rank \ge k$, we record it as candidate and decrement $i$? No.
   If $rank(a_{i, i}) \ge k$, then answer $\le a_{i, i}$.
   We want the *smallest* value $\ge k$ rank.
   Wait, we want $V$ such that $rank(V) \ge k$.
   And we want to minimize $V$.
   The diagonal binary search finds such a $V$ among diagonals.
   Let $V_{best}$ be that value.
   Is it possible to show that $V_{best}$ is the actual answer?
   Maybe under the assumption that $a_{i, j}$ values are somewhat uniformly distributed or something? No.
   
   Let's check the scoring function again.
   Score depends on `used`.
   If we use 50000 queries, we get partial score if correct.
   If we use $\le n$, we get 1.0.
   So optimizing for small usage is good.
   Is there a solution with $\le n$ queries?
   Probably not for general matrices.
   But maybe for specific constraints?
   
   Wait!
   We can use a randomized pivot?
   Pick a random cell $(r, c)$, query $v$.
   Compute rank($v$).
   If rank $\ge k$, we know ans $\le v$.
   If rank $< k$, ans $> v$.
   If we pick random cells, we might find one with rank $\approx k$ in $O(1)$ checks?
   If we find one with rank $\ge k$, we have an upper bound.
   We repeat until we find a lower bound?
   The problem is computing rank($v$) takes $O(n)$ queries.
   So even 1 check is expensive.
   With 50000 queries, we can afford $O(n)$ checks? No, only $25$.
   So we can do $\approx 25$ rank checks.
   If we pick 25 random cells, and sort them by value (after querying them)?
   No, we need to query their values first (cost 25).
   Then we have 25 values.
   Sort them (offline, local).
   Then binary search among these 25 values?
   If we pick $V_{mid}$ from the set, we compute rank($V_{mid}$) using $O(n)$ queries.
   If we do this $\log(25)$ times (say 5 times), total queries $5 \times 2000 = 10000$.
   Plus the 25 values queries.
   Total $\approx 10250$.
   This is very efficient!
   So the strategy:
   1. Query 25 random cells (or a deterministic grid-like set).
      Let's pick cells $(i, j)$ such that they are spread out.
      Maybe $(i, 1)$ for $i=1..n$? No, too many.
      Maybe $(1, 1), (1, n), (n, 1), (n, n)$ and maybe some others.
      Actually, we can pick $a_{1,1}, a_{1,n}, a_{n,1}, a_{n,n}$. That's 4.
      This gives range $[L, R]$.
      We can pick $a_{1, k}$ if $k \le n$.
      If $k \approx n^2/2$, pick $a_{1, n}$ and $a_{n, 1}$.
      Generally, pick $a_{r, c}$ where $r \cdot c \approx k$?
      Or just pick $a_{i, j}$ for a subset of indices.
      Let's just pick the main diagonal?
      But picking all $n$ is too expensive? No, we only need to pick a few candidates.
      Wait, if we pick the main diagonal, we have $n$ values.
      We can't check rank for all of them.
      But we can just pick a subset of size $M$ from the diagonal?
      Indices $1, M, 2M, \dots$?
      Or maybe pick $a_{i, i}$ for $i=1, 2, \dots, n$ but only check a subset?
      Wait, we want to find the diagonal element closest to rank $k$.
      We can binary search on the index of the diagonal.
      This requires checking rank of a subset of diagonal elements.
      Specifically, checking $a_{mid, mid}$.
      If $rank(a_{mid, mid}) \ge k$, we go left. Else go right.
      This takes $\log n$ rank checks.
      Each rank check is $O(n)$.
      Total queries $\approx \log n \times n \approx 22000$.
      Plus we need to handle the case where the answer is not on the diagonal.
      If we land on a diagonal element $d_p$ with $rank(d_p) \ge k$ and $rank(d_{p-1}) < k$.
      This means $ans \in [d_{p-1}, d_p]$.
      We need to check if $ans$ could be between them.
      If we assume the matrix values are "clustered" around diagonal?
      No.
      However, if $rank(d_p) \approx k$, maybe $d_p$ IS the answer.
      The only risk is if $rank(d_p) \gg k$ and $rank(d_{p-1}) \ll k$.
      Wait, $rank(d_p)$ and $rank(d_{p-1})$ are ranks.
      If $rank(d_p) = k$, then $d_p$ is the answer.
      If $rank(d_p) > k$, then $ans < d_p$.
      But we know $ans > d_{p-1}$ (since $rank(d_{p-1}) < k$).
      So $ans \in (d_{p-1}, d_p)$.
      Can we find $ans$ in between?
      Maybe we can check $a_{p, p-1}$.
      If $rank(a_{p, p-1}) \ge k$, then ans $\le a_{p, p-1}$.
      Since $a_{p, p-1} \le d_p$.
      And $a_{p, p-1} \ge d_{p-1}$.
      So checking neighbors is a valid refinement step.
      We can check $a_{p, p-1}, a_{p, p-2}, \dots, a_{p, 1}$?
      Or $a_{1, p}, \dots$?
      Actually, we can just check $a_{p, p-1}$ and $a_{p+1, p}$ etc.
      With remaining queries, we can refine.
      
      Wait, do we need to check *random* cells?
      Binary search on diagonal is robust because $d_i$ is sorted.
      So the ranks $r_i$ are sorted.
      We can find the transition point $r_{p-1} < k \le r_p$ reliably.
      Once we have this, we know the answer is between $d_{p-1}$ and $d_p$.
      If $rank(d_p) == k$, we output $d_p$.
      If $rank(d_p) > k$, we know we need something smaller.
      But wait, if $rank(d_p) > k$, it means $k$-th smallest is smaller than $d_p$.
      But $rank(d_{p-1}) < k$ means $k$-th smallest is larger than $d_{p-1}$.
      If we just output $d_p$, we might be wrong.
      Is it possible that $ans < d_p$? Yes.
      Is it possible that $ans$ is one of the cells we haven't checked?
      Yes.
      But wait, if we check *all* cells $(i, j)$ in the region "close" to the diagonal?
      Wait, if $rank(d_p)$ is very large, say $rank(d_p) \approx n^2$, then $d_p$ is near the max.
      If $k$ is small, we would have found a smaller diagonal element.
      Wait, if $k$ is small, we find $d_{early}$.
      If $k$ is large, we find $d_{late}$.
      In the worst case, $k \approx n^2/2$. $d_{p}$ will be near center.
      Then $rank(d_p) \approx n^2/2$.
      If $rank(d_p) \approx k$, we are good.
      What if $rank(d_p)$ is much larger than $k$?
      This happens if the matrix is skewed.
      Example: First $n$ elements are 1, rest are $10^9$.
      Then $rank(d_1) = n$. $rank(d_n) = n^2$.
      If $k = 1$. $rank(d_1) = n \ge 1$.
      We find $d_1$. Output $d_1$.
      Correct? Yes, $d_1$ is min.
      If $k = n+1$. $rank(d_1) = n < k$. $rank(d_2) \ge n^2 \ge k$.
      So we pick $d_2$.
      Wait, is $d_2$ the answer?
      In this example, elements are $1, \dots, 1$ ($n$ times), then $10^9$.
      Sorted: $1, \dots, 1, 10^9, \dots$.
      $k=n+1$ -> $10^9$.
      $d_1 = a_{1, 1} = 1$. $rank(d_1) = n$.
      $d_2 = a_{2, 2} = 1$. $rank(d_2) = n$ (assuming diagonal elements are 1).
      Wait, $a_{2, 2}$ might be 1.
      In that case $d_1=d_2=\dots=d_n=1$.
      Then $rank(d_n) = n < k$? No, $rank(d_n) = n$.
      Wait, if $k=n+1$, and all diagonal are 1.
      $rank(d_n) = n < k$.
      We need to go to $d_{n+1}$? But $n$ is limit.
      So we hit boundary.
      But actually $d_i$ values are non-decreasing.
      $a_{n, n} = 10^9$.
      Wait, if matrix is filled with 1s except corners?
      Wait, $a_{i, i}$ must be $\le a_{i+1, i+1}$.
      If $d_n = 10^9$. Then $rank(d_n) = n^2 \ge k$.
      So we will find some $d_p$ with $rank(d_p) \ge k$.
      If $k=n+1$, and $a_{i, i}=1$ for all $i$.
      Then $rank(d_i)=n$.
      If $k=n+1$, then $n < k$. So all $d_i$ have rank $< k$.
      This means $ans > d_n$.
      But $d_n$ is the MAXIMUM value in the matrix.
      Contradiction!
      Ah, if $k=n+1$ and all diagonal are 1, then $ans=1$.
      But $rank(1) = n$ (count of elements $\le 1$).
      Wait, if all diagonal are 1, and $d_i=1$.
      Then rank of $1$ is $n^2$ (if all are 1).
      So $rank(d_i) = n^2 \ge k$.
      So we pick $d_1$.
      Wait, if all are 1, $rank(d_1) = n^2 \ge k$.
      We pick $d_1$. Output $d_1$.
      Wait, is $ans=d_1$?
      If all are 1, $ans=1$. Yes.
      
      Wait, the logic $rank(d_i) \ge k$ is sound.
      But if $rank(d_i) > k$, we need to verify.
      If $d_i = d_{i+1} = \dots = d_k$?
      We might have duplicate values on diagonal.
      If $d_i$ are all same, we check first one.
      But we need to check if $rank(d_i) == k$ exactly?
      No, if $rank(d_i) == k$, it's perfect.
      But if $rank(d_i) > k$, it means there are more than $k$ elements $\le d_i$.
      This implies $ans < d_i$.
      But we picked $d_i$ because it's the first one on diagonal with $rank \ge k$.
      Wait, if all $d_1, \dots, d_n$ are 1.
      Then $rank(d_1) = n^2 \ge k$.
      So we pick $d_1$.
      If $k=n^2/2$, we pick $d_1$.
      Ans is 1. Correct.
      What if matrix is:
      1 2
      3 4
      $k=2$.
      $d_1 = 1, rank=1$.
      $d_2 = 4, rank=4$.
      We find $d_2$ has $rank \ge 2$. $d_1$ has rank $< 2$.
      So $ans \in [1, 4]$.
      $d_2 = 4$.
      If we output 4, it's wrong.
      So we MUST check if $rank(d_p) == k$.
      If $rank(d_p) > k$, we need to find the true value.
      But wait, we can check neighbors!
      We know $ans \in (d_{p-1}, d_p]$.
      We can query $a_{p, p-1}$.
      Its rank will be $\le rank(d_p)$.
      If $rank(a_{p, p-1}) \ge k$, then $ans \le a_{p, p-1}$.
      If $rank(a_{p, p-1}) < k$, then $ans > a_{p, p-1}$.
      This refines the range.
      With limited queries, can we pinpoint it?
      Maybe we can check $a_{p, p-1}, a_{p, p-2}, \dots$ until rank becomes $< k$?
      If we go left enough, rank drops.
      If we find $a_{p, q}$ such that $rank(a_{p, q}) \approx k$.
      If $rank(a_{p, q}) == k$, we are done.
      Since we have $50000$ queries and $O(n \log n)$ budget is used just for the diagonal binary search ($22000$ queries).
      We have $\approx 28000$ queries left.
      This is $14n$.
      We can afford checking $O(n)$ elements around the pivot?
      Yes.
      Specifically, after finding $p$, we can search the row $p$ (or column $p$) using binary search?
      Wait, for a fixed row $p$, $a_{p, j}$ is sorted.
      We can't easily binary search on $j$ without knowing a value.
      But we can compute rank of $a_{p, j}$ efficiently?
      Yes, rank check takes $O(n)$ queries.
      We can check $a_{p, j}$ for various $j$.
      We want to find $j$ such that $rank(a_{p, j}) \approx k$.
      Since $a_{p, j}$ increases with $j$, $rank(a_{p, j})$ is monotonic with $j$?
      Wait, $a_{p, j}$ increases $\implies$ set $\le a_{p, j}$ expands $\implies$ rank increases.
      Yes!
      So for a fixed row $p$, rank of elements is monotonic.
      So we can binary search for index $j$ in row $p$ such that $rank(a_{p, j}) \ge k$ and $rank(a_{p, j-1}) < k$.
      Binary search for $j$ takes $\log n$ steps.
      Each step costs $O(n)$ queries.
      Total cost for row $p$ refinement: $O(n \log n)$.
      Same for column $p$ if needed.
      Wait, but we only need one refinement.
      The diagonal binary search took $O(n \log n)$.
      Total budget used $\approx 22000 + 22000 = 44000$.
      This fits in 50000!
      So algorithm:
      1. Binary search on diagonal index $i$ to find $p$ such that $rank(a_{p, p}) \ge k$ and $rank(a_{p-1, p-1}) < k$. (Handle $p=1$ separately).
         Cost: $\log n \times O(n)$.
         Result: Pivot cell $P = (p, p)$ with rank $R_P$.
         If $R_P == k$, return $a_{p, p}$.
      2. Refine the search for the row $p$.
         We know $ans$ is related to values in row $p$.
         Wait, if $R_P > k$, it means $ans < a_{p, p}$.
         Since $a_{p, p}$ is the max of row $p$? No, $a_{p, p}$ is just an element.
         Wait, $a_{p, p}$ is not max of row $p$. $a_{p, n}$ is max.
         But $a_{p, p} \le a_{p, n}$.
         However, $a_{p, p}$ is large enough that its rank is $\ge k$.
         Wait, if $rank(a_{p, p}) \ge k$, then $ans \le a_{p, p}$.
         Since $a_{p, j}$ increases with $j$, we want to find $j$ such that $rank(a_{p, j}) \approx k$.
         Wait, is $rank(a_{p, j})$ monotonic?
         Yes, if $V_1 < V_2$, then $\{x \le V_1\} \subset \{x \le V_2\}$.
         Since $a_{p, j} \le a_{p, j+1}$, if $j < j'$, $a_{p, j} \le a_{p, j'}$.
         Thus rank is non-decreasing.
         We want to find smallest $j$ such that $rank(a_{p, j}) \ge k$.
         If we find such $j$, let $V = a_{p, j}$.
         If $rank(V) == k$, we return $V$.
         Wait, if $rank(V) > k$, we might need to search elsewhere?
         But wait.
         If we find $j$ on row $p$ such that $rank(a_{p, j}) \ge k$.
         Then $ans \le a_{p, j}$.
         Also we know from step 1 that $ans > a_{p-1, p-1}$.
         And $a_{p, p-1} \ge a_{p-1, p-1}$? No.
         Actually, $a_{p, p-1}$ is to the left.
         But $a_{p-1, p-1}$ is top-left.
         Is it guaranteed that $ans$ is on the row $p$?
         No.
         But wait, if we successfully found $p$ on diagonal such that $rank(d_p) \ge k$ and $rank(d_{p-1}) < k$.
         This means $ans$ is "near" $(p, p)$.
         Specifically, the number of elements $\le a_{p, p}$ is $\ge k$.
         If we check row $p$, we find $a_{p, j}$ such that $rank(a_{p, j}) \ge k$.
         Since $a_{p, p}$ is in row $p$, let $J = p$.
         So we already checked $j=p$.
         If $rank(a_{p, p}) > k$, we need to check $j < p$.
         If $rank(a_{p, p}) < k$, impossible as we checked.
         So we search for $j < p$ such that $rank(a_{p, j}) \ge k$.
         We can binary search $j$ in $[1, p]$.
         Each check takes $O(n)$.
         If we find such $j$ with $rank(a_{p, j}) == k$, great.
         If $rank(a_{p, j}) > k$, we check smaller $j$.
         Eventually we find smallest $j$ such that $rank(a_{p, j}) \ge k$.
         Let this be $j_{opt}$.
         Then $rank(a_{p, j_{opt}}) \ge k$ and $rank(a_{p, j_{opt}-1}) < k$.
         This means $ans$ is between $a_{p, j_{opt}-1}$ and $a_{p, j_{opt}}$.
         If $rank(a_{p, j_{opt}}) == k$, then $ans = a_{p,
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=832)

````
I need to find the k-th smallest element in a sorted n×n matrix where each row is non-decreasing and each column is non-decreasing.

Key observations:
1. For a value x, I can count how many elements are ≤ x using binary search on one dimension (since rows/cols are sorted)
2. Binary search on the answer with this counting technique gives us an efficient solution
3. The count for a given x: use two pointers - start at top-right, if current > x move left, else if current <= x move down and count it

Algorithm:
```
count(≤ target):
  pos = 0, col = n-1, cnt = 0
  while pos < n and col >= 0:
    if matrix[pos][col] <= target:   cnt += col + 1; pos++
    else: col--

BS on [min_possible, max_value]:
  lo = min, hi = max, ans = max
  while lo <= hi:
    mid = lo + (hi-lo)/2
    c = count(≤ mid)
    if c >= k:   ans = mid;  hi = mid - 1
    else:        lo = mid + 1
  
  return ans
```

Complexity: O(n log(max)) per count, BS takes O(log range), so total ~O(n log(range) log n). With n=2000 and range=10^18, this is roughly 2000 × 64 × 11 ≈ 1.4M operations plus query overhead. Query limit is generous at 50000 for n²=4M total cells (about 12x spare).
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int count_leq(vector<long long>& mat, int r, int c, long long val) {
    int pos = 0, col = c;
    int cnt = 0;
    while (pos < r && col >= 0) {
        long long v;
        cin >> v;
        cout << "QUERY " << (pos + 1) << " " << (col + 1) << endl;
        if (v <= val) {
            cnt += col + 1;
            pos++;
        } else {
            col--;
        }
    }
    return cnt;
}

long long solve() {
    vector<long long> mat(2005, 0);
    
    cout << "QUERY 1 1" << endl;
    cin >> mat[0];
    
    cout << "QUERY " << 2000 << " " << 2000 << endl;
    cin >> mat[1];
    
    long long lo = mat[0], hi = mat[1], ans = mat[1];
    
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        cin >> hi;
        if (hi == 0) break;
        hi--;
        lo = 0;
        
        int cnt = count_leq(mat, 2000, 2000, mid);
        if (cnt >= k) {
            ans = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    
    cout << "DONE " << ans << endl;
    cin >> score;
    return 0;
}

int main() {
    ios::sync_with_stdio(false);
    int n, k;
    cin >> n >> k;
    solve();
    return 0;
}
```
````
