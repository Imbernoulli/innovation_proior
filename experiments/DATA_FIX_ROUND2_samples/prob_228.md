# 题 228 — base均分 20.0 vs 直接SFT均分 0.0(掉点 20.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=3,得分 100.0,completion_tokens=32768)

````
The problem asks us to count the number of substrings in a binary string such that the number of zeros equals the square of the number of ones.
Let $S$ be the input string of length $N$.
A substring is defined by its start index $i$ and end index $j$, where $0 \le i \le j < N$.
Let $c_0(i, j)$ be the count of '0's in $S[i..j]$.
Let $c_1(i, j)$ be the count of '1's in $S[i..j]$.
We want to find the number of pairs $(i, j)$ such that $c_0(i, j) = (c_1(i, j))^2$.

Let $P_0[k]$ be the prefix sum of '0's up to index $k-1$ (so $P_0[0] = 0$).
Let $P_1[k]$ be the prefix sum of '1's up to index $k-1$ (so $P_1[0] = 0$).
Then $c_0(i, j) = P_0[j+1] - P_0[i]$ and $c_1(i, j) = P_1[j+1] - P_1[i]$.
The condition becomes:
$P_0[j+1] - P_0[i] = (P_1[j+1] - P_1[i])^2$.

Let $u = P_0[j+1] + k \cdot M$ and $v = P_1[j+1] + m \cdot M$ be some coordinates? No, simpler.
Let $x = P_0[j+1]$ and $y = P_1[j+1]$ for the ending position $j+1$.
Let $x' = P_0[i]$ and $y' = P_1[i]$ for the starting position $i$.
We need $x - x' = (y - y')^2$.
Rearranging this, we get $x = (y - y')^2 + x'$ or $x - x' - (y - y')^2 = 0$.
Here $x, x'$ are counts of zeros, $y, y'$ are counts of ones. Note that $x - x'$ is simply the difference in zero counts.
Wait, let's define a new coordinate system to simplify this.
Usually, for problems involving $c_0 = c_1$, we use a transformed coordinate system like $(x+y, x-y)$.
Here the relation involves a square. The maximum length is $2 \times 10^6$.
If $c_1$ is large, say $1000$, then $c_0$ can be up to $10^6$. Since total length is limited, $c_1$ can't be arbitrarily large relative to $N$. specifically $\sqrt{N}$.
Actually, if $c_1 = k$, then $c_0 = k^2$. The length of the substring would be $k^2 + k$.
Since $N \le 2 \cdot 10^6$, $k^2 + k \le 2 \cdot 10^6$. This implies $k \approx \sqrt{2 \cdot 10^6} \approx 1414$.
So the number of ones in any valid substring cannot exceed roughly 1500.

Let's fix the starting position $i$ and iterate over possible values of $c_1$.
For a fixed $i$, if the substring has $k$ ones, it must have $k^2$ zeros.
So the length of the substring must be at least $k + k^2$.
Also, the substring must actually exist with these counts.
We can precompute the positions of 1s and 0s.
Or we can process each ending position $j$ and iterate over possible numbers of ones $k$.
For a specific $j$, suppose the substring ends at $j$. Let the number of ones be $k$. Then the required number of zeros is $k^2$.
So we look for an index $i$ such that in range $(i, j]$ there are exactly $k$ ones and $k^2$ zeros.
This means if we scan from right to left or left to right...
Maybe two pointers or iterating on the value of $k$ is better given the constraint on $k$.

Let's try fixing the number of ones $k$.
For a fixed $k \ge 1$, we want to find pairs $(i, j)$ such that $c_1(i, j) = k$ and $c_0(i, j) = k^2$.
The length of such a substring is $L_k = k + k^2$.
We just need to check how many substrings of length $k^2+k$ have exactly $k$ ones.
However, checking every $k$ and every window might be too slow if we do it naively.
Number of possible $k$'s is small ($\approx 1414$).
Total length $N = 2 \cdot 10^6$.
If we iterate $k$ from 1 to $\approx 1414$:
For each $k$, we consider windows of size $k^2 + k$. We need to know if they contain exactly $k$ ones.
Checking the number of ones in a window takes $O(1)$ with precomputed prefix sums.
So complexity would be $O(N \cdot \sqrt{N})$.
$2 \cdot 10^6 \cdot 1414 \approx 2.8 \cdot 10^9$, which is likely too slow (usually limit is around $10^8$ operations per second).

Can we optimize?
Let $pos[p][t]$ be the position of the $t$-th occurrence of character $p$ ('0' or '1'). Or maybe just store indices.
But wait, for a fixed $k$, we need to count windows of length $K = k^2+k$ with exactly $k$ ones.
The window is $S[l \dots l+K-1]$. Number of ones is $count\_ones(l+K) - count\_ones(l)$.
We want $count\_ones(l+K) - count\_ones(l) = k$.
This looks like finding equal values in a transformed array.
Let $A[t]$ be the prefix sum of ones. We want $A[l+K] - A[l] = k$.
$\implies A[l+K] - k = A[l]$.
Here $K = k^2+k$ depends on $k$. So for each $k$, the offset varies.
Since $k$ changes, we can't easily combine queries. However, since $K$ is determined by $k$, and for each $k$ we iterate over all possible start positions, the naive iteration over $l$ for each $k$ is $O(N \sqrt{N})$.

Is there a way to avoid $O(N \sqrt{N})$?
Let's analyze the constraints on $k$.
If $k=0$, $c_0 = 0$, substring is empty? Usually empty substring is not counted or has 0 length. Problem says "number of substrings", usually non-empty. If $k=0$, $c_1=0$, then $c_0=0$, so "substring" containing only nothing. But typically "substrings" refers to non-empty. If non-empty, $c_1=0 \implies c_0 > 0 \implies 0 = (\text{something}>0)^2$ impossible. If empty, length 0. Usually competitive programming problems imply non-empty. Let's assume non-empty. If empty substring allowed, count += 1 for $k=0$. But let's stick to standard interpretation.
Actually, if $k=0$, $c_0=0, c_1=0$, substring is empty. Let's assume non-empty substrings.
Max $k$ is roughly $\sqrt{N}$.
With $k$ varying, we can iterate over $i$ (start index) and see what $j$ (end index) satisfies the condition.
But for a fixed $i$, valid $j$'s depend on $c_1$.
$c_1(j) = P_1[j+1] - P_1[i]$. $c_0(j) = P_0[j+1] - P_0[i]$.
Condition: $P_0[j+1] - P_0[i] = (P_1[j+1] - P_1[i])^2$.
$P_0[j+1] = (P_1[j+1] - P_1[i])^2 + P_0[i]$.
Let $y = P_1[j+1]$. Let $x = P_0[j+1]$.
Equation: $x = (y - P_1[i])^2 + P_0[i]$.
Here $P_1[i]$ and $P_0[i]$ are constants for a fixed start $i$.
Let $C_1 = P_1[i]$ and $C_0 = P_0[i]$.
We need to find indices $j \ge i$ such that $P_0[j+1] - (P_1[j+1] - C_1)^2 = C_0$.
The term on LHS depends only on $j$. Specifically, $f(t) = P_0[t+1] - (P_1[t+1] - C_1)^2$. Wait, $P_1[i]$ depends on $i$.
Let's rewrite as: $P_0[j+1] - (P_1[j+1])^2 = C_0 - 2 C_1 P_1[j+1] + P_1[j+1]^2 + P_0[i]$. No.

Let's expand: $P_0[j+1] - P_0[i] = P_1[j+1]^2 - 2 P_1[j+1] P_1[i] + P_1[i]^2$.
Rearranging: $P_0[j+1] + 2 P_1[i] P_1[j+1] = P_0[i] + P_1[j+1]^2 + P_1[i]^2$.
This doesn't seem to separate $i$ and $j$ cleanly into one side being independent.

However, notice $N \le 2 \cdot 10^6$.
The time limit is not specified but typical is 1-2s.
$O(N \sqrt{N})$ is definitely risky.
Maybe we can iterate on the *ending* position $j$?
For a fixed $j$, we want to count $i \le j$ such that $P_0[j+1] - P_0[i] = (P_1[j+1] - P_1[i])^2$.
Let $X = P_1[j+1]$ (total ones up to $j$) and $Y = P_0[j+1]$ (total zeros up to $j$).
We want $Y - P_0[i] = (X - P_1[i])^2$.
This is $Y - P_0[i] = X^2 - 2X P_1[i] + P_1[i]^2$.
$P_0[i] - 2X P_1[i] = Y - X^2 - P_1[i]^2$.
We need to count $i \in [0, j]$ satisfying this equation.
Here $X$ and $Y$ are fixed for current $j$. The RHS depends on $P_1[i]$ (and implicitly $P_0[i]$).
Specifically, for each $i$, let $a_i = P_1[i]$ (prefix ones). Then $P_0[i] = i - a_i$. Wait, $i$ is index, prefix sums are up to $i$ (exclusive for substring $S[0..i-1]$ or inclusive? Let's stick to indices $0..N-1$. Prefix sums $P[k]$ stores count in $S[0..k-1]$. So $P[0]=0, P[N]$ total).
Substrings $S[i..j-1]$ correspond to $P[j] - P[i]$.
So for current $j$ (end index in $1..N$), we look for $i \in [0, j]$ such that $(P_0[j] - P_0[i]) = (P_1[j] - P_1[i])^2$.
Let $X = P_1[j], Y = P_0[j]$.
We need $Y - P_0[i] = (X - P_1[i])^2$.
Note that $P_0[i] + P_1[i] = i$. So $P_0[i] = i - P_1[i]$.
Substitute $P_0[i]$:
$Y - (i - P_1[i]) = (X - P_1[i])^2$.
$Y - i + P_1[i] = X^2 - 2X P_1[i] + P_1[i]^2$.
$i - P_1[i] + X^2 - 2X P_1[i] + P_1[i]^2 = Y$? No.
$Y = X^2 - 2X P_1[i] + P_1[i]^2 - P_1[i] + i$.
$Y = X^2 + i + P_1[i]^2 - (2X + 1) P_1[i]$.
Let's rearrange terms depending on $P_1[i]$:
$(2X + 1) P_1[i] - P_1[i]^2 = X^2 + i - Y$.
$P_1[i] ((2X + 1) - P_1[i]) = X^2 + i - Y$.
Here $i$ is also part of the equation because $P_0[i] = i - P_1[i]$.
So we need to solve for $i$.
Notice that $P_1[i]$ is the number of ones in $S[0..i-1]$.
Since $P_1[i]$ increases slowly (at most $i$), we can group $i$'s by their value of $P_1[i]$.
Actually, for a fixed $j$, we calculate target values.
But $i$ appears in the equation linearly too.
$P_1[i]^2 - (2X + 1)P_1[i] + (X^2 + i - Y) = 0$.
This looks quadratic in terms of variables, but coefficients vary with $i$.
This seems hard to query efficiently.

Let's go back to the constraint $c_1 = k$.
$k = c_1(i, j)$. Then $c_0(i, j) = k^2$.
Length $len = k^2 + k$.
We are looking for substrings of length $len$ with exactly $k$ ones.
As noted before, summing over $k$ gives $\sum_{k} N \approx N \sqrt{N}$, which is potentially too slow ($2 \cdot 10^9$ ops).
Is $N \sqrt{N}$ really too slow?
In C++, $10^8$ ops is roughly 1 sec. $2.8 \cdot 10^9$ is 28 seconds. Too slow.
But note that not all $k$ allow a window of size $k^2+k$ inside $N$.
More importantly, we can iterate over $k$ and perform the check efficiently.
For a fixed $k$, we need to count pairs $(i, j)$ such that $j - i + 1 = k^2 + k$ AND $c_1(i, j) = k$.
This is equivalent to counting $i$ such that $S[i \dots i + k^2 + k - 1]$ contains exactly $k$ ones.
This is a standard sliding window problem.
We can compute the number of ones in every window of length $L = k^2+k$.
Let $Ones[x]$ be prefix sum of ones.
Count $i$ such that $Ones[i+L] - Ones[i] == k$.
To do this for ALL $k$ efficiently:
Notice that we only care about $k$ where $k(k+1) \le N$.
Let $M_k = k^2+k$.
We iterate $k$ from 1 while $M_k \le N$.
For each $k$, we want to count indices $i$ where window of length $M_k$ starting at $i$ has $k$ ones.
This requires calculating sliding windows for different lengths.
Standard sliding window approach for a fixed $k$:
Iterate $i$ from $0$ to $N - M_k$. Compute count.
This takes $O(N)$ per $k$. Total $O(N \sqrt{N})$.
Can we do better?
Observe that $M_k$ increases with $k$.
For very small $k$, $M_k$ is small, many windows.
For large $k$, $M_k$ is large, few windows.
The number of valid windows is roughly $N - M_k$.
Summing $(N - M_k)$ over $k \approx \sqrt{N}$ still yields $O(N \sqrt{N})$.

Wait, we can just collect the indices of ones. Let `pos_on` be the list of indices where `1` occurs.
If we fix $k$, we need to select $k$ ones within a range of length $M_k$.
Suppose we choose the first one included to be at index `pos_on[p]`. Then the last one included would be `pos_on[p+k-1]`.
The range covered by these $k$ ones is from `pos_on[p]` to `pos_on[p+k-1]`.
Let this distance be $D_p = pos\_on[p+k-1] - pos\_on[p] + 1$.
For a valid substring to contain exactly these $k$ ones, the substring must start before or at `pos_on[p]` and end after or at `pos_on[p+k-1]`.
Specifically, if the substring has length $M_k$, then:
Start index $S$ and End index $E$ satisfy $E - S + 1 = M_k$.
We need to count $k$ ones in $S \dots E$.
This means if we take the $p$-th one and the $(p+k-1)$-th one (0-indexed among 1s), the distance between them is fixed.
If we have $k$ ones strictly inside the window, say the first one of the $k$ is at index $idx_1$ and the last one is at $idx_k$.
Then $idx_k - idx_1$ must be $\le M_k - 1$ (since length is $M_k$, indices differ by at most $M_k - 1$).
Wait, the number of ones is exactly $k$.
Let the ones be at indices $z_1 < z_2 < \dots < z_m$ in the entire string.
Any substring of length $L=M_k$ starts at $s$ and ends at $s+L-1$.
It contains exactly $k$ ones iff $z_{q+k-1} - z_q < L$ and $z_{q+k} - z_{q-1} \ge L$?
Actually, for a fixed $k$, any block of $k$ consecutive ones (indices in `pos_on`) defines a "core".
Let the chosen $k$ ones be `pos_on[j], pos_on[j+1], ..., pos_on[j+k-1]`.
If the substring covers all of these, does it cover more?
If the substring contains exactly $k$ ones, then there exists a $j$ such that these are the ones, and no other ones are in the range.
Actually, if the substring has length $L$ and contains exactly $k$ ones, let these ones be the ones from index $j$ to $j+k-1$ in the `pos_on` array.
Then the start of the substring $S$ must satisfy:
$S \le pos\_on[j]$
And the end $E = S + L - 1$.
Also the $(j+1)$-th one (or previous) must not be included.
The condition "contains exactly $k$ ones" is satisfied if the substring includes `pos_on[j]...pos_on[j+k-1]` AND excludes `pos_on[j-1]` and `pos_on[j+k]`.
Since length is fixed to $L = k^2+k$:
We need $pos\_on[j+k-1] - pos\_on[j] < L$ (so all $k$ fit) -- actually, since we need $k$ ones, the span of these $k$ ones must be less than $L$. If span $> L$, impossible. If span $\le L$, possible.
Wait, if span $\le L$, then it is possible to place the window of size $L$ to cover these $k$ ones.
Specifically, the valid starting positions $S$ are constrained by:
1. Include `pos_on[j]`: $S \le pos\_on[j]$.
2. Include `pos_on[j+k-1]`: $S + L - 1 \ge pos\_on[j+k-1] \implies S \ge pos\_on[j+k-1] - L + 1$.
3. Exclude `pos_on[j-1]` (if it exists): $S + L - 1 < pos\_on[j-1] \implies S < pos\_on[j-1] - L + 1$.
   Wait, if $pos\_on[j]$ is the first one, we don't have `pos_on[j-1]`. If we include it, count increases. So we must exclude previous ones.
   Actually, simply: the set of ones in $S \dots S+L-1$ corresponds to the intersection of indices in `pos_on` with $[S, S+L-1]$.
   The number of such ones is $count$.
   We need $count = k$.
   Let's consider the range of valid $S$'s for a fixed block of $k$ ones (from $j$ to $j+k-1$).
   For this block to be exactly the ones in the window:
   Start $S$ must be $\le pos\_on[j]$.
   End $E = S+L-1 \ge pos\_on[j+k-1]$.
   And $E < pos\_on[j+k]$ (the next one must be outside).
   And $S > pos\_on[j-1]$ (the previous one must be outside).
   
   Wait, the condition $c_0 = c_1^2$ fixes $L = k^2+k$.
   So we are NOT counting arbitrary windows. We are counting windows of FIXED length $L_k = k^2+k$ having exactly $k$ ones.
   With fixed length $L_k$, the number of ones is determined solely by the start position $S$.
   $count\_ones(S, S+L_k-1) = k$.
   This is exactly the same as "count number of $S$ such that window of length $L_k$ starting at $S$ has $k$ ones".
   
   We established earlier that doing this for each $k$ takes $O(N)$ and total is $O(N\sqrt{N})$.
   Can we optimize the inner loop?
   For a fixed $k$, we are essentially looking for occurrences of a pattern.
   Notice $L_k$ depends on $k$.
   However, we can iterate over the blocks of $k$ ones in `pos_on`.
   Let `pos_on` have size $M$.
   For a specific $k$, we can iterate over all blocks of size $k$ in `pos_on`. There are $M-k+1$ such blocks.
   For each block (indices $j, \dots, j+k-1$ in `pos_on`), let the first one be at $u$ and last at $v$.
   If $v - u + 1 \le L_k$, then these $k$ ones can fit in a window of size $L_k$.
   Specifically, the window must cover $[u, v]$. So $S \le u$ and $S+L_k-1 \ge v$.
   This implies $S \in [v - L_k + 1, u]$.
   Let range be $[L_{min}, R_{max}] = [v - L_k + 1, u]$.
   Any $S$ in this range will include at least these $k$ ones.
   Does it include any *other* ones?
   It might include `pos_on[j-1]` (index of one before $u$) or `pos_on[j+k]` (index after $v$).
   Condition: No other ones in $[S, S+L_k-1]$.
   1. Previous one $prev = pos\_on[j-1]$ (if $j>0$). We need $S > prev$.
   2. Next one $next = pos\_on[j+k]$ (if $j+k < M$). We need $S + L_k - 1 < next \implies S < next - L_k + 1$.
   
   So for a specific $k$, and for each block $j \in [0, M-k]$, the valid $S$'s are:
   $S \in [\max(v - L_k + 1, prev + 1), \min(u, next - L_k)]$.
   We add $\max(0, upper - lower + 1)$ to total answer.
   
   Complexity:
   Outer loop: iterate $k$ from 1 to $\approx 1500$.
   Inner loop: iterate over $M-k+1$ blocks.
   Total complexity: $\sum_{k=1}^{1500} (M - k) \approx M \cdot \frac{\sqrt{N}}{2} + \sum k \approx O(N \sqrt{N})$?
   Wait, $M$ is the number of ones, $M \le N$.
   So yes, it is $O(M \sqrt{N}) \le O(N \sqrt{N})$. Still $2 \cdot 10^9$.
   Is it possible that $M$ is small?
   Worst case $M \approx N$.
   
   Let's re-read constraints. $N \le 2 \cdot 10^6$. Time limit usually 1s or 2s.
   Usually $10^8$ ops is safe. $2 \cdot 10^9$ is TLE.
   We need something closer to $O(N \log N)$ or $O(N)$.
   
   Wait, $c_0 = c_1^2$.
   This equation is symmetric in some sense? No.
   Consider the transformed coordinate system $(x, y)$ where $x = \#0, y = \#1$.
   In 2D grid, we walk from $(0,0)$ to $(x_j, y_j)$. Step is $(1,0)$ for '0', $(0,1)$ for '1'.
   We want to count pairs $(i, j)$ such that $(P_0[j] - P_0[i]) = (P_1[j] - P_1[i])^2$.
   Let $u = P_1[j] - P_1[i]$, $v = P_0[j] - P_0[i]$.
   Condition $v = u^2$.
   The path goes from origin to $(P_0[N], P_1[N])$.
   Each point in the grid represents a prefix state.
   We have a set of points $Pt = \{(P_1[k], P_0[k]) \mid k \in 0..N\}$.
   We want to find pairs $p_1, p_2 \in Pt$ such that $x(p_1)-x(p_2) = k$, $y(p_1)-y(p_2) = k^2$.
   Let $x_1, y_1$ be coords of $p_1$ (current) and $x_2, y_2$ be coords of $p_2$ (previous).
   $x_1 - x_2 = k$, $y_1 - y_2 = k^2$.
   So $y_1 - y_2 = (x_1 - x_2)^2$.
   $y_1 - (x_1 - x_2)^2 = y_2$.
   Or $y_1 - x_1^2 = y_2 - 2 x_1 x_2 + x_2^2$.
   This doesn't factor nicely.
   
   Let's look at the structure again.
   Points $(x, y)$ in plane. We want pairs with vector $(k, k^2)$.
   Notice that $k$ is the horizontal component difference (number of 1s). $k^2$ is vertical (number of 0s).
   Actually, I used $(x,y) = (P_1, P_0)$. $x$ is count of 1s, $y$ is count of 0s.
   Vector is $(\Delta x, \Delta y) = (k, k^2)$.
   So we are looking for pairs of points separated by vector $(k, k^2)$.
   Since $\Delta x$ determines $\Delta y$, we just need to check if for each pair of points, their difference matches $(k, k^2)$ for some $k$.
   
   Iterate over the "difference" parameter $k$.
   We want to count pairs $(p_1, p_2)$ such that $x_1 - x_2 = k$ and $y_1 - y_2 = k^2$.
   This is equivalent to counting pairs $(p_1, p_2)$ such that $x_1 - k = x_2$ and $y_1 - k^2 = y_2$.
   We can iterate over all $k$ from $1$ to $\sqrt{N}$.
   For a fixed $k$, we want to count how many pairs have diff $(k, k^2)$.
   For a fixed $k$, we can iterate through all $p_1$ and check if $p_2 = p_1 - (k, k^2)$ exists in our set of prefix points.
   To do this efficiently:
   Store all visited points $(P_1, P_0)$ in a hash map or a sorted list/structure.
   Map key: $(P_1, P_0)$. Value: count? Actually each prefix is unique because $P_1+P_0$ is distinct? No, $P_1$ increases by 0 or 1, $P_0$ increases by 0 or 1. But at step $k$, $P_1(k) + P_0(k) = k$. So $(P_1, P_0)$ uniquely identifies the prefix index. Indeed, the path is a lattice path with unit steps. No two prefixes can have same $(P_1, P_0)$ unless indices are same.
   So points are unique.
   Algorithm:
   1. Collect all points $S_k = (P_1[k], P_0[k])$ for $k = 0 \dots N$.
   2. Store them in a data structure for fast lookup. Since we need to match $p_2 = p_1 - (k, k^2)$, we can sort the points or use a hash map. Sorting takes $O(N \log N)$. Lookup $O(\log N)$.
   3. Iterate $k$ from 1 to $\approx 1414$.
      For each point $(x, y)$ in $S$ (specifically those with $x \ge k, y \ge k^2$), check if $(x-k, y-k^2)$ exists in $S$.
      If it exists, increment count.
      
      Wait, iterating all points for each $k$ takes $O(N \sqrt{N})$.
      Is it possible to optimize?
      Most points won't match.
      Can we iterate only relevant points?
      Or can we iterate on $k$ and just do the work?
      Wait, $N=2 \cdot 10^6$. $O(N \sqrt{N})$ is bad.
      BUT, note that we are iterating $k$. For a fixed $k$, we iterate all $p_1$.
      Do we really need to iterate all $p_1$?
      We need to count occurrences of $p_2 = p_1 - (k, k^2)$.
      This is equivalent to finding intersections of shifted sets.
      Set $A = \{(x, y)\}$. Set $B = \{(x+k, y+k^2)\}$. Count overlaps.
      Intersection size.
      
      Since coordinates are bounded ($0 \le x, y \le N$).
      Maybe sweep-line?
      Coordinate transformation?
      $y = x^2$ curve.
      The condition is $y_{diff} = (x_{diff})^2$.
      Points lie roughly near diagonal $x=y$ if string is balanced, or axes.
      However, we don't need geometric tricks necessarily.
      
      Is $N \sqrt{N}$ actually acceptable?
      Usually $10^7 - 10^8$ operations per second.
      $2 \cdot 10^6 \cdot 1414 \approx 2.8 \cdot 10^9$.
      Even with simple checks, it's likely to TLE.
      However, note that $P_1[k]$ and $P_0[k]$ are correlated with $k$.
      Specifically $P_1[k] + P_0[k] = k$.
      So the point is $(x, k-x)$.
      So $P_0[k] = k - P_1[k]$.
      Substituting into condition $P_0[j] - P_0[i] = (P_1[j] - P_1[i])^2$:
      $(j - P_1[j]) - (i - P_1[i]) = (P_1[j] - P_1[i])^2$.
      $j - i - (P_1[j] - P_1[i]) = (P_1[j] - P_1[i])^2$.
      Let $\Delta 1 = P_1[j] - P_1[i] = c_1$. Let $\Delta 0 = P_0[j] - P_0[i] = c_0$.
      $\Delta 0 = c_1^2$.
      Length of substring $\ell = j - i + 1 = c_1 + c_0 = c_1 + c_1^2$.
      So $j - i = c_1^2 + c_1$.
      This confirms $j - i = k(k+1)$ is constant for a fixed $k$.
      So for a fixed $k$, we are looking for $i, j$ such that $j - i = k^2+k$ and $P_1[j] - P_1[i] = k$.
      
      So my earlier analysis holds:
      Iterate $k$ from 1 to $\approx 1414$.
      Check how many $i$ exist such that $P_1[i + k^2 + k] - P_1[i] = k$.
      Let $Len_k = k(k+1)$.
      We need to count $i \in [0, N - Len_k]$ such that $P_1[i + Len_k] - P_1[i] = k$.
      Since $Len_k$ is large for large $k$, fewer valid $i$'s.
      For small $k$, many $i$'s.
      Sum of $N$ over all $k$ is $O(N \sqrt{N})$.
      
      Wait, I am assuming that for a fixed $k$, we need to iterate all $i$.
      Can we iterate $i$ once and update answers?
      For a fixed $i$, we need to find $j$ such that $P_1[j] - P_1[i] = k$ and $j - i = k^2 + k$.
      This implies $j = i + k^2 + k$.
      And we need $P_1[i + k^2 + k] - P_1[i] = k$.
      So for each $i$, we can check $k$.
      Possible values of $k$: $P_1[i + k^2 + k] - P_1[i] = k$.
      This implies $P_1[i + k^2 + k] \le P_1[i] + k$.
      Also $j > i$, so $P_1[j] \ge P_1[i]$. Thus $P_1[j] - P_1[i] \ge 0$.
      So $k$ must be a valid count.
      Since $j-i$ grows quadratically, $k$ is limited.
      For each $i$, $k$ is such that $k^2+k \le N - i$.
      So for each $i$, we iterate $k$ starting from 0 up to $\approx \sqrt{N}$.
      Wait, checking $k$ values for each $i$ is also $O(N \sqrt{N})$.
      
      However, observe that $P_1$ array is computed in $O(N)$.
      Can we optimize the check $P_1[i + k(k+1)] - P_1[i] == k$?
      Notice that for a specific difference in indices $\Delta = i_{target} - i$, we have $k$ related to $\Delta$.
      Specifically $\Delta = k^2 + k$.
      So $k$ is roughly $\sqrt{\Delta}$.
      We only need to check this condition if $\Delta = k^2 + k$ for some integer $k$.
      Let's iterate over $\Delta$.
      Valid $\Delta$ are of form $k^2+k$.
      For a fixed $\Delta$, we need to count $i$ such that $P_1[i+\Delta] - P_1[i] = \sqrt{\Delta}(\dots)$? No, $k = (-1 + \sqrt{1+4\Delta})/2$.
      So for a valid $\Delta$, we just check if $P_1[i+\Delta] - P_1[i] == k$.
      This reduces to: For each $k$, iterate all valid $i$ and check.
      Wait, $\sum_{k} (N - (k^2+k))$.
      The dominant term is when $k$ is small.
      $k=1 \implies \Delta = 2$. Check $N$ times.
      $k=1400 \implies \Delta \approx 2 \cdot 10^6$. Check few times.
      Total checks $\approx \sum (N - k^2) \approx N \sqrt{N}$.
      
      So direct simulation is too slow.
      But wait!
      $P_1[x]$ takes values between $0$ and $N$.
      Can we group queries?
      For a fixed $\Delta$, we want to count $i$ such that $P_1[i+\Delta] - P_1[i] == K_{fixed}$.
      Let $Q_\Delta[x] = P_1[x+\Delta] - P_1[x]$. We want to count $x$ where $Q_\Delta[x] = K_{target}$.
      This is simply checking frequency of differences.
      We can precompute $Q_\Delta$? No, too much memory/time.
      
      Let's reconsider the constraints and properties.
      $P_1[j] - P_1[i] = k$.
      Also $j - i = k^2 + k$.
      This means $P_1$ has a "slope" roughly 0 over distance $k^2+k$ but accumulated $k$.
      Slope is $k / (k^2+k) \approx 1/k$. Small slope for large $k$.
      Large density of 0s.
      
      What if we iterate over the position of the LAST '1' in the substring?
      Let the substring be $S[i \dots j]$.
      Suppose it contains exactly $k$ ones.
      Then the number of zeros is $k^2$.
      Total length $k^2 + k$.
      Let the $k$ ones be located at indices $z_1, z_2, \dots, z_k$ relative to $i$.
      The positions in the string are absolute.
      Let the actual indices of ones in $S[i \dots j]$ be $o_1, o_2, \dots, o_k$.
      The number of zeros before $o_1$ is $z_1 - i$ (where $o_1$ is the index of the one, but wait, indices are $i \dots j$).
      Actually, let's use global indices.
      Let $pos$ be the list of indices of '1's.
      For a substring with $k$ ones, these ones correspond to some contiguous subsegment of $pos$.
      Specifically, suppose the ones involved are $pos[p], pos[p+1], \dots, pos[p+k-1]$.
      These $k$ ones determine the minimal bounding box covering them: from $pos[p]$ to $pos[p+k-1]$.
      The distance between first and last one is $d = pos[p+k-1] - pos[p]$.
      Inside this range, we have $k$ ones.
      The number of zeros inside is determined by the length of the segment minus $k$.
      Wait, no. The number of zeros is $d - (k-1)$? No, that's number of slots.
      Zeros in $S[pos[p] \dots pos[p+k-1]]$?
      The number of ones is $k$.
      The positions of these ones are $pos[p], \dots, pos[p+k-1]$.
      Any index between $pos[p]$ and $pos[p+k-1]$ that is NOT in the $pos$ list is a zero.
      So number of zeros in $[pos[p], pos[p+k-1]]$ is $(pos[p+k-1] - pos[p] + 1) - k$.
      Let this be $zeros\_inside$.
      We need total zeros in substring to be $k^2$.
      Let the substring extend from $S$ to $E$. Length $L = S-E+1$. No $E-S+1$.
      Length must be $k^2 + k$.
      Also the number of zeros in $[S, E]$ is $k^2$.
      This means the number of ones is $k$.
      The condition implies that the substring MUST contain these $k$ ones $pos[p] \dots pos[p+k-1]$ and NO OTHER ONES.
      This means:
      1. $S \le pos[p]$
      2. $E \ge pos[p+k-1]$
      3. $S > pos[p-1]$ (if $p>0$)
      4. $E < pos[p+k]$ (if $p+k < |pos|$)
      5. Length $E - S + 1 = k^2 + k$.
      
      Substitute (5): $E = S + k^2 + k - 1$.
      From (1) and (5): $S + k^2 + k - 1 \ge pos[p+k-1] \implies S \ge pos[p+k-1] - k^2 - k + 1$.
      From (2): $S + k^2 + k - 1 \ge pos[p+k-1]$ is same as above.
      From (3): $S \ge pos[p-1] + 1$.
      So combined lower bound for $S$:
      $S \ge \max(pos[p] , pos[p+k-1] - k^2 - k + 1, pos[p-1] + 1)$.
      Wait, condition (1) $S \le pos[p]$.
      Wait, for the substring to include $pos[p]$ and NOT $pos[p-1]$:
      Start $S$ must satisfy $pos[p-1] < S \le pos[p]$.
      Also must cover $pos[p+k-1]$. So $E \ge pos[p+k-1]$.
      $S + k^2 + k - 1 \ge pos[p+k-1] \implies S \ge pos[p+k-1] - k^2 - k + 1$.
      So valid $S$ range:
      Lower bound $L_S = \max(pos[p-1] + 1, pos[p+k-1] - k^2 - k + 1)$.
      Upper bound $R_S = \min(pos[p], pos[p+k] - 1)$? No.
      Upper bound conditions:
      Must not include $pos[p+k]$. So $E < pos[p+k]$.
      $S + k^2 + k - 1 < pos[p+k] \implies S < pos[p+k] - k^2 - k + 1$.
      So $R_S = \min(pos[p], pos[p+k] - k^2 - k)$. (Using integer arithmetic, strict inequality $<$ becomes $\le$).
      Wait, indices are 0-based. $S \le pos[p]$. $S < pos[p+k] - k^2 - k + 1 \implies S \le pos[p+k] - k^2 - k$.
      So valid $S \in [L_S, R_S]$.
      If $L_S \le R_S$, we add $R_S - L_S + 1$ to answer.
      Also need $L_S \ge 0$ (always true if $pos \ge 0$) and $R_S \le N$.
      
      So algorithm refined:
      1. Identify positions of all '1's. Let this list be `ones`. Size $M$.
      2. Iterate over length of ones-block $k$ from $1$ to $M$.
         (Wait, max $k$ is such that $k^2+k \le N$. So $k \le \approx 1500$. Even if $M$ is larger, we can't form blocks of size $k$ if $k^2+k$ exceeds $N$. But wait, block of size $k$ means we need $k$ ones. Is it possible to have valid substring with $k > 1500$?
          Max zeros = $N$. So $k^2 \le N \implies k \le \sqrt{N}$. Yes, $k$ is bounded by $\approx 1500$.)
      3. So outer loop $k$ from $1$ to $\approx 1500$.
         If $k > M$, break (not enough ones).
      4. Iterate over all valid blocks of $k$ ones in `ones`. Indices $j$ from $0$ to $M-k$.
         Block is `ones[j] ... ones[j+k-1]`.
         Calculate required $L_S$.
         $pos[p] = ones[j]$.
         $pos[p+k-1] = ones[j+k-1]$.
         $pos[p-1] = (j==0 ? -1 : ones[j-1])$.
         $pos[p+k] = (j+k==M ? N : ones[j+k])$. (Since end of string is effectively $N$ for exclusion purpose).
         
         Range for $S$:
         $LB = \max(0, ones[j+k-1] - k^2 - k + 1)$, but actually $S$ can be anything $\ge 0$. The constraint is $S$ must exclude $pos[p-1]$. If $pos[p-1] = -1$, then lower bound is 0.
         Actually, if $pos[p-1]$ exists, $S > pos[p-1]$.
         If $pos[p-1]$ doesn't exist ($j=0$), then no restriction from left. $S \ge 0$.
         Similarly $pos[p]$ is the first one. We must include it, so $S \le ones[j]$.
         Also $pos[p+k]$ (if exists) restricts right side: $S < ones[j+k] - k^2 - k$.
         Wait, logic check:
         $S$ determines the substring start.
         Substring ends at $E = S + k^2 + k - 1$.
         Conditions:
         a) $S \le ones[j]$ (must include $j$-th one) -> $S \le ones[j]$.
            Wait, if $S > ones[j]$, then $j$-th one is NOT included. Correct.
            Also if $S \le ones[j]$, does it ensure we include it?
            Yes, provided $E \ge ones[j]$.
            $E \ge ones[j] \implies S + k^2 + k - 1 \ge ones[j]$.
            Since $ones[j] < ones[j+k-1]$ and $k^2+k \ge 2$, and $ones[j+k-1] - ones[j] \ge 0$.
            Usually $E \ge ones[j+k-1] \ge ones[j]$.
            So the condition $E \ge ones[j]$ is dominated by $E \ge ones[j+k-1]$.
            
         b) $E \ge ones[j+k-1]$ (must include last one of block)
            $S \ge ones[j+k-1] - k^2 - k + 1$.
            
         c) Exclude previous one $ones[j-1]$.
            $E < ones[j-1]$. (If $j>0$)
            $S + k^2 + k - 1 < ones[j-1] \implies S < ones[j-1] - k^2 - k + 1$.
            Wait, this implies we can't even reach the ones $j \dots j+k-1$ if they are far from previous one?
            Actually, if $S > ones[j-1]$, then $ones[j-1]$ is strictly before $S$.
            So substring starts after $ones[j-1]$.
            This is correct.
            
         d) Exclude next one $ones[j+k]$.
            $S < ones[j+k] - k^2 - k + 1$. (If $j+k < M$)
            Actually condition is $E < ones[j+k]$.
            $S + k^2 + k - 1 < ones[j+k] \implies S \le ones[j+k] - k^2 - k$.
         
         So $S \in [LB, UB]$.
         $LB = \max(0, ones[j+k-1] - k^2 - k + 1)$.
         If $j>0$, $LB = \max(LB, ones[j-1] + 1)$.
         $UB = \min(N, ones[j], ones[j+k] - k^2 - k)$.
         (Assume $ones[M] = N$).
         Wait, $S$ cannot exceed $N$. But $ones[j]$ is $< N$. So $ones[j]$ is tighter.
         So $UB = \min(ones[j], (j+k < M ? ones[j+k] - k^2 - k : N))$.
         Wait, if $j+k \ge M$, then there is no next one, so no upper bound from that side except $N$.
         Wait, actually if we have no next one, can we stop?
         Yes, we can go until end of string.
         But we must satisfy length constraint?
         If we pick $S$ too large, $E$ might go out of bounds $N-1$.
         So $S \le N - k^2 - k$.
         
         Let's formalize bounds carefully.
         $LB = \max(0, \dots)$.
         Constraints:
         1. $S \ge 0$.
         2. $S \le N - (k^2 + k)$. (Substring fits in string).
         3. $S \le ones[j]$. (Include $j$-th one).
         4. $S \ge ones[j+k-1] - (k^2 + k - 1)$. (Cover $j+k-1$-th one).
         5. If $j > 0$: $S > ones[j-1]$. $\implies S \ge ones[j-1] + 1$.
         6. If $j+k < M$: $S < ones[j+k] - (k^2 + k - 1)$. $\implies S \le ones[j+k] - k^2 - k$.
         
         Combine:
         $L = \max(0LL, (j>0 ? (long long)ones[j-1] + 1 : 0LL), (long long)ones[j+k-1] - k^2 - k + 1LL)$.
         $R = \min((long long)N, (long long)ones[j], (j+k < M ? (long long)ones[j+k] - k^2 - k : N));
         (Also ensure $R \le N - (k^2+k)$ is covered by the fact that ones[j] etc are $< N$. But explicit check safer? Actually if $R > N - k^2 - k$, we must clamp.
         However, if $S > N - k^2 - k$, substring length exceeds $N$, impossible.
         So we take $R = \min(R, (long long)N - k^2 - k)$.
         
         Contribution: $\max(0LL, R - L + 1)$.
         Add to total.
         
         Complexity:
         Outer loop $k$ up to $\approx 1500$.
         Inner loop iterates $M$ times (number of ones).
         Total operations $\sum_{k=1}^{1500} M \approx 1500 \cdot M$.
         If $M \approx N = 2 \cdot 10^6$, then $3 \cdot 10^9$.
         Still slow.
         
         Wait, for large $k$, the range of valid $S$ is very restrictive.
         Specifically $L = \max(\dots, ones[j+k-1] - k^2 - k + 1)$.
         And $R = \min(\dots, ones[j] - \dots)$.
         Notice that as $k$ increases, $k^2+k$ grows quadratically.
         For large $k$, the term $ones[j+k-1] - k^2 - k + 1$ could be negative?
         Or $ones[j]$ vs others.
         Actually, $ones[j]$ is increasing with $j$.
         For a fixed $k$, we iterate $j$.
         The condition $L \le R$ implies $ones[j+k-1] - k^2 - k + 1 \le ones[j]$.
         $\implies ones[j+k-1] - ones[j] \le k^2 + k - 1$.
         If the distance between the first and last one in the block exceeds $k^2+k-1$, then the window cannot cover them.
         Since $dist \approx k \cdot avg\_gap$.
         If gaps are small, $dist \approx k \cdot (const)$.
         $k^2+k$ is quadratic. $k \cdot const$ is linear.
         So for small $avg\_gap$, condition $ones[j+k-1] - ones[j] \le k^2 + k$ is always satisfied for a wide range of $j$.
         However, we have to subtract the contribution of valid ranges.
         Wait, the cost is iterating $j$.
         Can we avoid iterating $j$?
         The contribution depends on $LB - LB$?
         Look at the formula for contribution: $\max(0, R - L + 1)$.
         $L = \max(A_j, B_j, C)$. $R = \min(D_j, E_j, F)$.
         Where $A_j = (j>0 ? ones[j-1]+1 : 0)$. $B_j = ones[j+k-1] - k^2 - k + 1$.
         $C = 0$.
         $D_j = ones[j]$. $E_j = (j+k<M ? ones[j+k] - k^2 - k : N)$. $F = N - k^2 - k$.
         
         Notice $B_j$ depends on $j+k$.
         $D_j$ depends on $j$.
         This is a sliding window minimum/maximum problem?
         Specifically, for fixed $k$, we need $\sum \max(0, \min(D_j, E_j) - \max(A_j, B_j) + 1)$ subject to constraints.
         Given the constraints and monotonicity of `ones`, maybe we can speed up.
         Actually, `ones` is sorted.
         $A_j$ is `ones[j-1]+1`. $D_j$ is `ones[j]`.
         $B_j$ is `ones[j+k-1] - K`. $E_j$ is `ones[j+k] - K`.
         Basically, we want to count valid intervals.
         
         Is there a faster way?
         We are counting pairs $(i, j)$ with indices in `ones`.
         The ones indices define boundaries.
         Wait, we can rewrite the condition:
         We need to pick $k$ consecutive ones in `ones` array, say at indices $j$ to $j+k-1$.
         Let $u = ones[j]$ and $v = ones[j+k-1]$.
         We need $v - u + 1 \le k^2 + k$.
         Wait, if $v - u + 1 \le k^2 + k$, it is possible to cover them.
         If $v - u + 1 > k^2 + k$, it is impossible.
         Let's check the number of zeros.
         The distance between $u$ and $v$ contains $k$ ones and $(v - u + 1) - k$ zeros.
         The remaining zeros needed is $k^2 - ((v - u + 1) - k)$.
         These remaining zeros must be distributed to the left or right of the block $[u, v]$.
         Left of $u$: available space is until $ones[j-1]$. (Or start of string).
         Right of $v$: available space until $ones[j+k]$. (Or end of string).
         Total available zeros on left = $(u - 1) - (ones[j-1] + 1) + 1 = u - ones[j-1] - 1$.
         (Wait, if $j=0$, zeros on left = $u$).
         Actually, zeros before $u$ are indices $< u$ and not in `ones[0..j-1]`.
         Number of ones before $u$ is $j$.
         Number of zeros before $u$ is $u - j$.
         We are using $k$ ones in the block.
         The total zeros required is $k^2$.
         The number of zeros already present in $[u, v]$ is $(v - u + 1) - k$.
         Let $Z_{block} = (v - u + 1) - k$.
         Remaining zeros needed $Z_{rem} = k^2 - Z_{block}$.
         We need to fill $Z_{rem}$ zeros from the outside (left or right).
         Available zeros to the left: $Z_{avail\_left} = (ones[j-1] + 1 - u)$? No.
         Let's use positions.
         String indices $0 \dots N-1$.
         Block of ones occupies $ones[j], \dots, ones[j+k-1]$.
         Substring must cover this range.
         So $S \le ones[j]$ and $E \ge ones[j+k-1]$.
         Zeros in substring come from $[S, E] \setminus \{ones \text{ indices}\}$.
         Total ones in substring is exactly $k$. So no extra ones allowed.
         So $S$ must be greater than $ones[j-1]$ (if exists).
         $E$ must be smaller than $ones[j+k]$ (if exists).
         So we need to find count of $S$ such that:
         $S \ge ones[j-1] + 1$
         $S \le ones[j]$
         $S + L - 1 \ge ones[j+k-1] \implies S \ge ones[j+k-1] - L + 1$.
         $S + L - 1 < ones[j+k] \implies S \le ones[j+k] - L$.
         Where $L = k^2 + k$.
         The condition "exactly $k$ ones" is automatically satisfied if $S$ is in these bounds because $S$ stays strictly between $ones[j-1]$ and $ones[j+k]$ and covers $ones[j] \dots ones[j+k-1]$.
         Why? Because if we include ones $j \dots j+k-1$, and exclude $j-1$ and $j+k$, the number of ones is exactly $k$ (assuming $k \ge 1$).
         So for a fixed block of $k$ ones (defined by $j$), the number of valid start positions $S$ is simply the length of the intersection of intervals.
         Intersection of $[ones[j-1]+1, ones[j]]$ and $[ones[j+k-1]-L+1, ones[j+k]-L]$.
         Let $I_1 = [L_1, R_1] = [ones[j-1]+1, ones[j]]$. (Left boundary constraints)
         Let $I_2 = [L_2, R_2] = [ones[j+k-1]-L+1, ones[j+k]-L]$. (Right boundary constraints, inverted)
         Wait, $S$ must satisfy both. So $S \in I_1 \cap I_2$.
         Also $S \ge 0$ and $S \le N-L$ (to keep substring inside string).
         Wait, $ones[j+k]-L$ handles the right side of string implicitly if we treat $ones[M]=N$.
         Wait, if we use virtual $ones[M] = N$, then $S < N - L + 1 \implies S \le N - L$.
         Also $ones[j]$ is naturally $\le N$.
         So effectively, $S$ must be in $[\max(L_1, L_2), \min(R_1, R_2)]$.
         Provided $\max \le \min$.
         
         So we just need to calculate overlap for each $j$.
         $L_1 = ones[j-1] + 1$ (or 0 if $j=0$).
         $R_1 = ones[j]$.
         $L_2 = ones[j+k-1] - (k^2+k) + 1$.
         $R_2 = ones[j+k] - (k^2+k)$. (using virtual $ones[M]=N$, wait, strict inequality was $S < ones[j+k] - L + 1 \implies S \le ones[j+k] - L$).
         Wait, $E = S + L - 1 < ones[j+k] \implies S < ones[j+k] - L + 1$.
         Integer math: $S \le ones[j+k] - L$.
         Wait, if $ones[j+k]$ is the index of the NEXT one, we want $E < ones[j+k]$.
         Indices are $0 \dots N-1$.
         If $ones[j+k] = idx$, we need $S+L-1 < idx$.
         $S < idx - L + 1$.
         Max integer $S$ is $idx - L$.
         So $R_2 = ones[j+k] - (k^2+k)$.
         Also implicit constraint $R_2 \le N - (k^2+k)$?
         Yes, if $S \le N - L$, then $S \le N - L$.
         $R_2$ could be larger than $N-L$ if $ones[j+k]$ is close to $N$.
         Wait, $ones[M]$ should be set to $N-1$? Or we just enforce $S \le N-L$.
         The "next one" constraint applies only if we don't want to include it.
         If there is no next one ($j+k \ge M$), then no right-side one constraint.
         So we can imagine a virtual one at $N$?
         Wait, substring can end at $N-1$.
         If no next one, we just need $E \le N-1 \implies S \le N-L$.
         So effectively $ones[M] = N$ works for the constraint $S \le ones[M] - L$.
         But wait, if $ones[M] = N$, then $S \le N - L$. Correct.
         What about $L_1$?
         If $j=0$, no previous one. $S \ge 0$.
         So we can set $ones[-1] = -1$.
         Then $L_1 = 0$.
         So we just need $R - L + 1$.
         
         Algorithm complexity again:
         Outer loop $k \in [1, \sqrt{N}]$.
         Inner loop $j \in [0, M-k]$.
         Total iterations $\sum_{k} (M-k) \approx M \sqrt{N}$.
         This is still $O(M \sqrt{N})$.
         Is there any redundancy?
         Notice $L_1$ and $R_1$ depend on $j$.
         $L_2$ and $R_2$ depend on $j$ via $j+k$.
         Let's shift indices.
         $j$ goes from $0$ to $M-k$.
         Terms:
         $U = ones[j-1]+1$ (for $j>0$ else 0).
         $V = ones[j]$.
         $X = ones[j+k-1] - k^2 - k + 1$.
         $Y = ones[j+k] - k^2 - k$. (with virtual $ones[M]=N$).
         Overlap $[ \max(U, X), \min(V, Y) ]$.
         Note that $V$ is non-decreasing with $j$. $X$ involves $j+k-1$, so $X$ is also non-decreasing with $j$ (roughly).
         However, $U$ increases, $X$ increases.
         This looks like counting pairs or intersections of two moving intervals.
         Actually, $V$ and $X$ are functions of $j$. $U$ and $Y$ are functions of $j$ and $j+k$.
         Wait, this is $O(M \sqrt{N})$. Can we optimize to $O(M + N)$ or similar?
         Maybe $O(N)$?
         Observe that $k^2+k$ grows fast.
         For small $k$, we have many valid $j$.
         For large $k$, we have few valid $j$? No, $j$ goes up to $M-k$, so many if $M$ is large.
         Wait, we iterate $k$ up to $\approx 1500$.
         If $M$ is small (e.g., all 0s, $M=0$), loop runs once, $k=1$. Answer 0. Fast.
         If $M=N$ (all 1s), then $k^2+k \approx 2 \cdot 10^6$ only for $k \approx 1414$.
         For $k \approx 1$, $k^2+k=2$. Loop runs $N$ times.
         Total ops $N + (N-2) + \dots \approx N \sqrt{N}$.
         Wait, we can swap loops?
         Instead of iterating $k$ then $j$, iterate $j$ (start of block of ones)?
         But we don't know $k$.
         
         Wait, is $N \sqrt{N}$ really the worst case?
         If $S$ consists of alternating 0s and 1s, $M \approx N/2$.
         Then still $O(N \sqrt{N})$.
         However, we can skip $k$ if $k^2+k > N$.
         Actually, the sum is $\sum_{k=1}^{\sqrt{N}} (M - k + 1)$.
         If $M \approx N$, sum is $N \sqrt{N} - \sum k \approx N \sqrt{N}$.
         This is too much.
         BUT, do we need to check all $j$?
         Condition $L \le R$:
         $\max(U, X) \le \min(V, Y)$.
         $\iff U \le V$ AND $U \le Y$ AND $X \le V$ AND $X \le Y$.
         $U \le V \iff ones[j-1]+1 \le ones[j] \iff$ gap between ones $\ge 1$. Always true (since ones are distinct).
         $X \le Y \iff ones[j+k-1] - L + 1 \le ones[j+k] - L \iff ones[j+k-1] + 1 \le ones[j+k]$. True.
         So intersection is non-empty if $U \le Y$ and $X \le V$.
         1. $U \le Y \implies ones[j-1]+1 \le ones[j+k] - (k^2+k)$.
            $\implies ones[j+k] - ones[j-1] \ge k^2 + k + 1$.
            This means the distance between the one BEFORE the block and the one AFTER the block must be large enough.
         2. $X \le V \implies ones[j+k-1] - (k^2+k) + 1 \le ones[j]$.
            $\implies ones[j+k-1] - ones[j] \le k^2 + k - 1$.
            This means the distance between the FIRST and LAST one in the block must be small enough.
            
         So for each $k$, we count $j$ such that:
         $ones[j] - ones[j+k-1] \le k^2+k-1$ (Block is compact enough to fit in window length wise)
         AND
         $ones[j+k] - ones[j-1] \ge k^2+k+1$ (Surrounding gap is large enough)
         
         Wait, $ones[j+k-1] - ones[j]$ is the spread of the block.
         If spread is large, we can't cover them with length $k^2+k$.
         Wait, $k^2+k$ is the length of the window.
         The span of the block is $ones[j+k-1] - ones[j]$.
         The condition $span \le len-1$ is necessary.
         Wait, if span $\le len-1$, does it mean we can fit?
         Yes, provided we don't hit surrounding ones.
         But we also need $len = k^2+k$.
         So for fixed $k$, we need to count $j$ such that:
         $ones[j+k-1] - ones[j] \le k^2+k-1$
         AND
         $ones[j+k] - ones[j-1] \ge k^2+k+1$.
         (Indices adjusted properly).
         And then add $\min(R, Y) - \max(L, X) + 1$?
         Wait, the contribution is length of intersection.
         The intersection is $[L, R]$.
         $L = \max(U, X)$. $R = \min(V, Y)$.
         If $L \le R$, add $R - L + 1$.
         Since $U < V$ and $X < Y$ (due to distinct ones), $L \le R$ is equivalent to $U \le Y$ and $X \le V$.
         Which is exactly the two inequalities derived above.
         Wait, no.
         $U \le Y$ is necessary.
         $X \le V$ is necessary.
         If these hold, then $\max(U, X) \le \min(V, Y)$ holds.
         So we just need to count how many $j$ satisfy these.
         But we also need to sum $(R - L + 1)$.
         Note that $R - L + 1 = \min(V, Y) - \max(U, X) + 1$.
         $V - U = ones[j] - (ones[j-1]+1) = gap_{j}$.
         $Y - X = (ones[j+k] - L) - (ones[j+k-1] - L + 1) = ones[j+k] - ones[j+k-1] - 1 = gap_{j+k}$.
         Wait, $V-Y$ or $X-U$?
         Let's visualize the interval $[U, V]$ (allowed starts from left perspective) and $[X, Y]$ (allowed starts from right perspective).
         $[U, V] = [ones[j-1]+1, ones[j]]$.
         $[X, Y] = [ones[j+k-1] - k^2 - k + 1, ones[j+k] - k^2 - k]$.
         We need overlap.
         Notice $Y - U = ones[j+k] - k^2 - k - ones[j-1] - 1$.
         $V - X = ones[j] - ones[j+k-1] + k^2 + k - 1$.
         The overlap is $\min(V, Y) - \max(U, X) + 1$.
         If conditions $U \le Y$ and $X \le V$ met, overlap $\ge 1$.
         Can we optimize the summation?
         We need $\sum_{j} (\min(V, Y) - \max(U, X) + 1)$.
         $= \sum_{j} (1 + \min(ones[j], ones[j+k] - L) - \max(ones[j-1]+1, ones[j+k-1] - L + 1))$.
         $= \sum_{j} (1 + \min(ones[j], ones[j+k]) - \max(ones[j-1], ones[j+k-1] - 1) - \min(L, L-1)? )$.
         Wait, let $L = k^2+k$.
         Term: $\min(ones[j], ones[j+k] - L) - \max(ones[j-1], ones[j+k-1] - L + 1)$? No.
         Actually:
         $\min(A, B) - \max(C, D) = \min(A-C, A-D, B-C, B-D)$. Not quite useful.
         Let's rewrite:
         $overlap = \min(ones[j], ones[j+k]-L) - \max(ones[j-1]+1, ones[j+k-1]-L+1) + 1$.
         Let $g_k(j) = ones[j+k] - ones[j-1]$. (Distance between outer ones).
         Let $h_k(j) = ones[j+k-1] - ones[j]$. (Spread of block).
         Then condition $U \le Y$ is $ones[j-1]+1 \le ones[j+k]-L \implies g_k(j) \ge L+1$.
         Condition $X \le V$ is $ones[j+k-1]-L+1 \le ones[j] \implies h_k(j) \le L-1$.
         The contribution is roughly related to $L$?
         If $L$ is very large, $U \le Y$ might fail often.
         Wait, if $L$ is large, the overlap $[U, V] \cap [X, Y]$ is small or empty.
         Actually, $V - U = ones[j] - ones[j-1] - 1$. This is the number of '0's immediately following $ones[j-1]$ and preceding $ones[j]$.
         $Y - X = ones[j+k] - ones[j+k-1] - 1$.
         Let $zeros\_between\_i = ones[i] - ones[i-1] - 1$ (for $ones[-1]=-1$).
         Then $V - U = zeros\_between\_j$.
         $Y - X = zeros\_between\_{j+k}$.
         Wait, no.
         $Y - X = (ones[j+k] - L) - (ones[j+k-1] - L + 1) = ones[j+k] - ones[j+k-1] - 1$. Yes.
         Wait, $Y$ is $ones[j+k]-L$. $X$ is $ones[j+k-1]-L+1$.
         The interval $[X, Y]$ is shifted version of $[X+L, Y+L] = [ones[j+k-1], ones[j+k]-1]$.
         Wait, $[X, Y]$ represents valid $S$. $S$ must be such that $S+L-1$ falls in $[ones[j+k-1], ones[j+k]-1]$.
         No, $S$ must satisfy $ones[j+k-1]-L+1 \le S \le ones[j+k]-L$.
         So $S+L-1$ falls in $[ones[j+k-1], ones[j+k]-1]$.
         And $S$ must fall in $[ones[j-1]+1, ones[j]]$.
         So we are intersecting $[ones[j-1]+1, ones[j]]$ and $[ones[j+k-1]-L+1, ones[j+k]-L]$.
         Note that $[ones[j-1]+1, ones[j]]$ is exactly the set of positions $S$ that start in the gap between $j-1$ and $j$.
         $[ones[j+k-1]-L+1, ones[j+k]-L]$ is the set of positions $S$ such that $S+L-1$ falls in the gap between $j+k-1$ and $j+k$.
         Essentially, $S$ must "land" a zero-gap somewhere?
         No, $S$ is the start of substring. $S$ falls in gap $j-1|j$. $E$ falls in gap $j+k-1|j+k$.
         So $E = S + L - 1$.
         We require $E \ge ones[j+k-1]$ and $E \le ones[j+k]-1$.
         Wait, $E$ must be in $[ones[j+k-1], ones[j+k]-1]$.
         Wait, $E$ must be in gap $j+k-1|j+k$? No, $E$ must be in $[ones[j+k-1], ones[j+k]-1]$?
         If $E < ones[j+k]$, then $E$ can be anywhere before $ones[j+k]$.
         But $E \ge ones[j+k-1]$ is required to include $j+k-1$-th one.
         So $E \in [ones[j+k-1], ones[j+k]-1]$.
         And $S \in [ones[j-1]+1, ones[j]]$.
         So we need $S+L-1 \in [ones[j+k-1], ones[j+k]-1]$ AND $S \in [ones[j-1]+1, ones[j]]$.
         So $S \in [ones[j-1]+1, ones[j]] \cap [ones[j+k-1]-L+1, ones[j+k]-L]$.
         So yes, intersection of these two intervals.
         The length of $[ones[j-1]+1, ones[j]]$ is $Z_{gap, j}$ (number of zeros between $ones[j-1]$ and $ones[j]$).
         Wait, indices are inclusive. $ones[j] - ones[j-1] - 1$.
         Length of $[ones[j+k-1]-L+1, ones[j+k]-L]$ is also $Z_{gap, j+k}$.
         So we need intersection length of an interval of size $Z_{gap, j}$ and an interval of size $Z_{gap, j+k}$ shifted by $L-1$.
         Let $z_j = ones[j] - ones[j-1] - 1$ (size of gap after $j$?). No, gap between $j-1$ and $j$.
         Let $Gap_j = ones[j] - ones[j-1] - 1$.
         Interval 1: $[ones[j-1]+1, ones[j]]$ corresponds to valid start positions within this gap?
         Wait, $S$ must be in gap $j-1$.
         Interval 2: $S+L-1$ must be in gap $j+k$.
         Let $Gap'_x = Gap_{x+1}$? No.
         The gap between $j+k-1$ and $j+k$ has zeros at indices $ones[j+k-1]+1 \dots ones[j+k]-1$.
         If $S+L-1$ lands in this gap, then $S \in [ones[j+k-1]-L+1, ones[j+k]-L]$.
         We want number of $S$ in intersection of Gap$_{j-1}$ (shifted?) No.
         Gap$_{j-1}$ indices are $ones[j-1]+1 \dots ones[j]$.
         Shifted Gap$_{j+k}$ indices are $ones[j+k-1]+1 \dots ones[j+k]-1$ mapped back to $S$.
         Mapping: $E \mapsto E - L + 1$.
         So valid $S$ are those where start of substring is in Gap$_{j-1}$ and end of substring is in Gap$_{j+k}$.
         Wait, start $S$ must be in $[ones[j-1]+1, ones[j]]$.
         End $E = S+L-1$.
         Condition $E \in [ones[j+k-1], ones[j+k]-1]$.
         This implies $S \in [ones[j+k-1]-L+1, ones[j+k]-L]$.
         So we need to count $S$ in Intersection.
         Notice that $L = k^2+k$ is fixed.
         We need $\sum_{j} length(Interval(ones[j-1]+1, ones[j]) \cap Interval(ones[j+k-1]-L+1, ones[j+k]-L))$.
         Let $I_a = [u_a, v_a]$ with length $z_a$.
         Let $I_b = [u_b, v_b]$ with length $z_b$.
         Here $u_a = ones[j-1]+1, v_a = ones[j]$. $u_b = ones[j+k-1]-L+1, v_b = ones[j+k]-L$.
         Note that $v_a - u_a = z_a$. $v_b - u_b = z_b$.
         Actually, $u_b - v_a = ones[j+k-1] - L + 1 - ones[j]$.
         $v_a - u_b = ones[j] - ones[j+k-1] + L - 1$.
         Let $dist = ones[j+k-1] - ones[j]$.
         Then $v_a - u_b = L - 1 - dist$.
         If $dist > L-1$, intersection is empty.
         If $dist \le L-1$, overlap is $v_b - u_a + 1$? No.
         Intersection is $[\max(u_a, u_b), \min(v_a, v_b)]$.
         $\max(u_a, u_b) = \max(ones[j-1]+1, ones[j+k-1]-L+1)$.
         $\min(v_a, v_b) = \min(ones[j], ones[j+k]-L)$.
         
         Let's analyze the complexity again.
         We need to compute this sum efficiently.
         Notice that $u_b$ and $v_b$ are just $u_b$ and $v_b$.
         The intervals are shifting with $j$.
         However, $L$ is constant for the inner loop.
         So we have $\sum_{j} (\min(ones[j], ones[j+k]-L) - \max(ones[j-1]+1, ones[j+k-1]-L+1) + 1)_+$.
         Let $f(x) = ones[x]$.
         Sum is $\sum_{j} (\min(f(j), f(j+k)-L) - \max(f(j-1)+1, f(j+k-1)-L+1) + 1)$.
         Let $g(j) = f(j) - f(j-1) - 1$ (length of gap after $f(j-1)$).
         Actually $f(j) - f(j-1) - 1$ is number of zeros between $f(j-1)$ and $f(j)$.
         Wait, $ones[j]$ is index of $j$-th one (0-indexed).
         Gap $j$ is between $ones[j]$ and $ones[j+1]$?
         Let's denote $z_i$ as number of zeros between $ones[i-1]$ and $ones[i]$.
         Then $z_i = ones[i] - ones[i-1] - 1$.
         The interval $I_a$ is $[ones[j-1]+1, ones[j]]$. Its length is $z_j$.
         Wait, $ones[j-1]+1$ is index of first zero after $j-1$. $ones[j]$ is index of $j$-th one.
         Wait, $S$ can be $ones[j]$. That is start at a 1.
         But we must NOT start before $ones[j-1]$. So $S \ge ones[j-1]+1$.
         Also $S \le ones[j]$.
         So $S$ is in range $[ones[j-1]+1, ones[j]]$.
         This range includes $ones[j]$. It includes zeros.
         Does it include $ones[j]$ itself?
         If $S = ones[j]$, then substring starts at a '1'. The first character is '1'.
         This is allowed. $c_1=1, c_0=0$ is not allowed since $0 \neq 1^2$.
         But general $k$, substring starts with whatever.
         However, we derived conditions based on including $j$-th one.
         If $S=ones[j]$, we include $j$-th one.
         So range $[ones[j-1]+1, ones[j]]$ is correct.
         Length is $ones[j] - (ones[j-1]+1) + 1 = ones[j] - ones[j-1] = z_j + 1$.
         Wait, $z_j$ zeros between. Total items in range is $z_j + 1$.
         Ah, $ones[j]$ is an index. $ones[j-1]$ is index.
         Number of integers in $[A, B]$ is $B - A + 1$.
         Here $A = ones[j-1]+1, B = ones[j]$.
         So length is $ones[j] - ones[j-1]$.
         Let's call this $len\_gap(j)$.
         Similarly for $I_b$, range is $[ones[j+k-1]-L+1, ones[j+k]-L]$.
         Length is $ones[j+k]-L - (ones[j+k-1]-L+1) + 1 = ones[j+k] - ones[j+k-1] = len\_gap(j+k)$.
         Wait, $I_b$ corresponds to $S$ such that $S+L-1$ lands in $[ones[j+k-1], ones[j+k]-1]$.
         The length of $[ones[j+k-1], ones[j+k]-1]$ is $ones[j+k] - ones[j+k-1]$.
         Wait, if $E \in [ones[j+k-1], ones[j+k]-1]$, then $S \in [ones[j+k-1]-L+1, ones[j+k]-L]$.
         The number of integers is same.
         So we are intersecting an interval of size $A_j = ones[j] - ones[j-1]$ and interval of size $B_j = ones[j+k] - ones[j+k-1]$.
         The shift is determined by relative positions of blocks.
         
         Optimization Idea:
         We have $\sum_{j} (\text{overlap of } [u_1, v_1] \text{ and } [u_2, v_2])$.
         $[u_1, v_1]$ moves with $j$. $[u_2, v_2]$ moves with $j+k$.
         Notice that $[u_1, v_1]$ is just $[ones[j-1]+1, ones[j]]$.
         Let's define $Pos[x] = ones[x-1] + 1$.
         Let's define $End[y] = ones[y]$.
         Wait, $[u_1, v_1] = [ones[j-1]+1, ones[j]]$.
         $[u_2, v_2] = [ones[j+k-1]-L+1, ones[j+k]-L]$.
         This is just $[Pos[j+k-1]-L+1, End[j+k]-L]$.
         The condition $L$ is fixed.
         Actually, this is equivalent to finding number of ways to pick two gaps (one at $j-1$, one at $j+k$) such that shifted interval overlaps.
         Wait, the indices in $ones$ array are $0 \dots M-1$.
         $j$ ranges $0 \dots M-k$.
         We access $ones[j-1]$ (virtual -1 is -1) and $ones[j+k]$ (virtual M is N).
         Let's pad $ones$ with virtual elements.
         $ones[-1] = -1$. $ones[M] = N$.
         We need to compute $\sum_{j=0}^{M-k} overlap([ones[j-1]+1, ones[j]], [ones[j+k-1]-L+1, ones[j+k]-L])$.
         This can be rewritten.
         Notice $overlap([a, b], [c, d]) = \min(b, d) - \max(a, c) + 1$ (clamped to 0).
         Here $b-j$ and $d-j$ are functions of $j$.
         Actually, $ones$ is increasing.
         So $ones[j]$ is increasing.
         For fixed $k$, as $j$ increases:
         $Interval_A$ shifts right. $Interval_B$ shifts right.
         The shift of $B$ relative to $A$?
         Center of $A \approx ones[j]$. Center of $B \approx ones[j+k]$.
         Distance between centers $\approx ones[j+k] - ones[j]$.
         If $ones[j+k] - ones[j] \approx O(k \cdot \text{avg\_gap})$.
         The condition for overlap is $ones[j+k-1] - ones[j] \le L-1$.
         And $ones[j-1] - ones[j+k] \le \dots$
         Basically, we need $ones[j+k] - ones[j-1] \ge L+1$.
         So $ones[j+k-1]$ must be "close" to $ones[j]$ (distance $\le L$).
         But $ones[j+k]$ must be "far" from $ones[j-1]$ (distance $\ge L$).
         This implies that $ones[j]$ to $ones[j+k]$ must be contained in length roughly $L$.
         Specifically, span $\approx L$.
         Wait, if span is small, then $ones[j+k] - ones[j-1]$ is small?
         No, $ones[j+k] - ones[j-1] = (ones[j+k] - ones[j+k-1]) + (ones[j+k-1] - ones[j]) + (ones[j] - ones[j-1])$.
         Roughly sum of gaps.
         So we need sum of gaps between $j$ and $j+k-1$ to be $\le L$.
         Wait, $ones[j+k-1] - ones[j]$ is span.
         We need $ones[j+k-1] - ones[j] \le L-1$.
         AND $ones[j+k] - ones[j-1] \ge L+1$.
         This implies we have at most $L$ distance between first and last one of the block, plus margins on sides.
         Actually, this sounds like we are selecting $k$ ones.
         If we just iterate $j$, for each $j$, we check condition and add contribution.
         Since $k$ is small, maybe $k^2$ dominates.
         Wait, if $k$ is small (e.g., 1), $L=2$.
         $ones[j] - ones[j-1] \le 1$. This implies $ones[j] = ones[j-1]+1$.
         i.e. no zeros between them?
         If $k=1$, $ones[j] - ones[j-1] \le 1$.
         And $ones[j+1] - ones[j-1] \ge 3$.
         This implies $ones[j]$ and $ones[j+1]$ are separated by at least 2 zeros?
         Wait, $ones[j+1] - ones[j-1] \ge 3$.
         $ones[j+1] - ones[j] + ones[j] - ones[j-1] \ge 3$.
         $Gap_{j+1} + Gap_{j} + 2 \ge 3 \implies Gap_{j+1} + Gap_{j} \ge 1$.
         So we need adjacent gaps to be sparse? No, sum $\ge 1$. Almost always true.
         
         So for small $k$, the constraints are loose.
         For large $k$, constraints are tight.
         However, the total number of valid $k$'s is small.
         Wait, if $N \le 2 \cdot 10^6$, $M$ can be up to $N$.
         Iterating all $j$ for all $k$ is indeed $O(M \sqrt{N})$.
         Is it possible to implement this in $C++$ fast enough?
         Operations are simple integer comparisons and additions.
         $2 \cdot 10^9$ ops might pass in 2-3 seconds if simple.
         Typical judge allows $10^8$ ops/sec.
         But modern judges (like Codeforces) allow slightly more or have fast CPUs.
         However, $2 \cdot 10^9$ is pushing it.
         Is there a solution faster?
         
         Let's look at the constraints on $k$.
         $c_0 = c_1^2$.
         If we fix the number of zeros $z$ in the substring, then $c_1 = \sqrt{z}$.
         So $z$ must be a perfect square. $z \in \{0, 1, 4, 9, \dots\}$.
         Max $z \le N$.
         So we only need to consider substrings with length $k + k^2$ where $k^2 \le N$.
         Wait, iterating over $k$ is iterating over $z=k^2$.
         So the set of possible lengths is restricted to $L_k = k^2+k$.
         There are only $\approx 1500$ such lengths.
         We need to count substrings of length $L_k$ with $k$ ones.
         This is equivalent to: Count substrings of length $L$ having $k$ ones, summed over all $k$.
         Wait, no. For each valid $k$, we count substrings of length $k^2+k$ with $k$ ones.
         This is exactly what we derived.
         The number of such substrings is the answer.
         So the problem is $\sum_k (\text{count of windows of length } L_k \text{ with } k \text{ ones})$.
         The issue is computing this for all $k$.
         But wait, we can just iterate over all substrings? No, $O(N^2)$.
         We can iterate over $k$ and solve in $O(N)$? No, $O(N \sqrt{N})$.
         Can we solve in $O(N \log N)$ or $O(N)$?
         We need to sum over $k$ the count of windows.
         Window condition: $Ones[i+L_k] - Ones[i] = k$.
         Let $D_L[k][i] = Ones[i+L_k] - Ones[i]$.
         We need $\sum_k \text{count}(D_{L_k}[k] == k)$.
         This looks like we need to count equality $Ones[i+L_k] - Ones[i] - k = 0$.
         Or $Ones[i+L_k] - k = Ones[i]$.
         $L_k$ depends on $k$.
         We can rewrite this as: Find number of pairs $(i, k)$ such that $Ones[i + k(k+1)] - k = Ones[i]$.
         Here $i$ ranges $0 \dots N$, $k$ ranges $0 \dots \sqrt{N}$.
         Let's check indices. $i + k(k+1) \le N$.
         So we need to find pairs $(i, k)$ satisfying equation.
         This looks like we can use a hash map to store values of $Ones[i]$.
         But $Ones[i+L_k]$ depends on $i$ and $k$.
         Let's iterate $k$. For a fixed $k$, we want to count $i$ such that $Ones[i] = Ones[i+L_k] - k$.
         This is equivalent to finding intersections of sequence $A$ and sequence $B$ where $B$ is shifted $A$?
         No. $A[i] = Ones[i]$. $B[i] = Ones[i+L_k] - k$.
         We want count $i$ such that $A[i] = B[i]$.
         This is easy: iterate $i$, check equality.
         Cost $O(N)$ per $k$. Total $O(N \sqrt{N})$.
         
         Wait!
         Look at $Ones[i + k^2+k] - k = Ones[i]$.
         $Ones[x] - k = Ones[x - (k^2+k)]$.
         Let's iterate over $x$ (which is $i + L_k$).
         Then $i = x - L_k$.
         Condition: $Ones[x] - k = Ones[x - k^2 - k]$.
         We need to find pairs $(x, k)$ such that $x - k^2 - k \ge 0$.
         Let $y = x - k^2 - k$.
         Then $x = y + k^2 + k$.
         Equation: $Ones[y + k^2 + k] - Ones[y] = k$.
         This brings us back to $Ones[y + len] - Ones[y] = k$ with $len = k^2+k$.
         
         However, notice that $k$ is uniquely determined by the number of ones?
         No.
         But $k$ is relatively small.
         Let's change perspective.
         Iterate $k$.
         Construct a boolean array `valid[i]`? No.
         
         Wait, is it possible to use bitsets?
         $N \le 2 \cdot 10^6$. `std::bitset` might work if bits were packed.
         But we have random access.
         
         Let's check the constraints and limits again.
         Time limit not explicitly stated but "Execution time and memory limit".
         Assume standard 1s, 256MB.
         $O(N \sqrt{N})$ is risky.
         But $k$ is only up to 1414.
         Average operations per step inside loop:
         Access array, subtract, compare. Very fast.
         Branch prediction: good (linear scan).
         Memory access: sequential (scanning $Ones$). Cache friendly.
         So it's basically $2.8 \cdot 10^9$ cycles.
         On modern CPU (3-4 GHz), this is roughly 1 second if instructions are fast.
         Wait, $2.8 \cdot 10^9$ ops. If each op takes ~1 cycle, it's 1s.
         But loop overhead?
         Actually, $k$ loop runs 1414 times.
         Inside, loop $i$ runs $N$ times.
         Total iterations $2.8 \cdot 10^9$.
         In C++, a simple loop with subtraction and comparison takes very little time.
         $2 \cdot 10^9$ simple integer ops might take 0.5s - 1s.
         Plus IO.
         Maybe it passes.
         We should optimize the code.
         Using `long long` for $ans$.
         Use `vector<int>` for `Ones`. `int` is sufficient (max $N$).
         Use fast IO.
         Reduce modulo or heavy ops.
         Just simple addition/subtraction.
         
         Wait, we can prune.
         If $k^2 + k > N$, stop.
         Also, for a fixed $k$, we only iterate valid $i$ where $i + k^2 + k < N$.
         Wait, $i$ goes from $0$ to $N - k^2 - k$.
         If $k$ is large, loop is short.
         Sum of $(N - k^2 - k)$ for $k=1..\sqrt{N}$.
         $\approx N \sqrt{N} - \sum k^2 \approx 2 \cdot 10^6 \cdot 1414 - \frac{1}{3} 1414^3 \approx 2.8 \cdot 10^9 - 1.2 \cdot 10^9 \approx 1.6 \cdot 10^9$.
         Wait, $\sum_{k=1}^{1414} k^2 \approx \frac{1414^3}{3} \approx \frac{2.8 \cdot 10^9}{3} \approx 10^9$.
         So total iterations $\approx N \sqrt{N} - \frac{N^{1.5}}{3} \approx \frac{2}{3} N \sqrt{N}$.
         Wait, $\sum (N - k^2) \approx N \sqrt{N} - \frac{N^{1.5}}{3} = \frac{2}{3} N \sqrt{N}$.
         $2 \cdot 10^6 \cdot 1414 \cdot 0.66 \approx 1.9 \cdot 10^9$.
         This is high.
         Is there any way to skip?
         If $Ones[i+L_k] - Ones[i]$ is likely to be $k$.
         The average number of ones in substring of length $L_k$ is roughly $\frac{L_k}{N} \times M \approx \frac{k^2+k}{N} M$.
         If density of ones is $D$, expected ones is $D \cdot L_k$.
         We need count to be $k$.
         So we need $D \cdot (k^2+k) \approx k$.
         $D \cdot (k+1) \approx 1 \implies D \approx \frac{1}{k+1}$.
         If actual density $D_{actual}$ is far from $1/(k+1)$, then count of matches will be small.
         But we still have to iterate to check.
         
         Wait, if $M$ (number of ones) is small, loop is faster.
         If $M$ is large, loop is slower.
         But max $N$ is $2 \cdot 10^6$.
         Maybe there's a trick.
         If we use the property $Ones[i+L_k] - Ones[i] = k$.
         This is equivalent to $Ones[i+L_k] - k = Ones[i]$.
         We need to match values.
         We can insert $(Ones[i] \times \text{some\_hash})$ into a set?
         No.
         
         Let's consider the problem statement again.
         Input: 01-string.
         Output: count of substrings.
         Constraints: length $2 \cdot 10^6$.
         
         Is it possible that $N \sqrt{N}$ is accepted?
         Some problems have $O(N \sqrt{N})$ solutions.
         Usually for $N=10^5$ or $2 \cdot 10^5$.
         For $2 \cdot 10^6$, it's usually $O(N \log N)$ or $O(N)$.
         However, $N$ here is up to $2 \cdot 10^6$, which is large.
         Perhaps the constant factor is very small.
         Inside loop:
         `cnt = ones[i + k*k + k] - ones[i];`
         `if (cnt == k) ans++;`
         This is extremely tight loop.
         Compilers will unroll it.
         Memory access is sequential.
         Maybe it will pass.
         
         Let's write the solution with $O(N \sqrt{N})$ but optimized.
         One optimization:
         Precompute $Ones$ array.
         Use `vector<int>`.
         Reserve memory.
         Use fast I/O.
         
         Wait, we can check the number of ones.
         If total ones in string is $T$, then for large $k$, we require $k$ ones.
         If $k > T$, count is 0.
         So $k$ loop only goes up to $\min(\approx 1500, T)$.
         If $T$ is small, it's fast.
         If $T \approx N$, it's slow.
         But if $T$ is small, then for small $k$, the condition $Ones[i+L_k] - Ones[i] = k$ might be rare?
         No, if $T$ is small, $Ones[i]$ is sparse.
         
         Another optimization:
         We can iterate $i$ instead of $k$.
         For each $i$, we iterate $k$ such that $i + k^2 + k \le N$.
         Check condition.
         This avoids recalculating $Ones[i]$.
         We can compute $L_k = k^2 + k$.
         If $L_k > N$, break.
         Check $Ones[i+L_k] - On[e] - k$?
         Wait, we need $Ones[i+L_k] - On[e] - k$? No.
         Condition: $Ones[i+L_k] - Ones[i] = k$.
         Check $Ones[i+L_k] - On[i] - k == 0$.
         This is same complexity.
         
         However, for a fixed $i$, $k$ goes $0 \dots \sqrt{N}$.
         So we compute $k$ loop up to $\approx 1500$.
         Inside, compute $L_k$, access array.
         Total operations same.
         Which order is better?
         If we iterate $k$ first, we access $Ones[i]$ sequentially. Good for cache.
         $i$ is index.
         $Ones[i+L_k]$ jumps by $L_k$.
         If we iterate $i$ then $k$, $Ones[i]$ is accessed repeatedly?
         No, $Ones[i]$ changes every outer loop. $Ones[i+L_k]$ jumps.
         If we iterate $k$ then $i$, for a fixed $k$, we access $Ones[i]$ and $Ones[i+L_k]$ for all $i$.
         $Ones[i]$ is accessed sequentially. $Ones[i+L_k]$ is accessed sequentially (stride $L_k$).
         Stride access $L_k$ might cause cache misses if $L_k$ is large.
         When $L_k$ is large, we do few iterations.
         When $L_k$ is small, we do many iterations, but stride is small, so cache friendly.
         When $L_k \approx 0$ (small), it's fine.
         So iterating $k$ then $i$ is likely best for cache locality when $L_k$ is small (most cases for small $k$).
         For large $k$, $L_k$ is large, so stride is large, but iterations are few.
         So this order is optimal.
         
         Let's refine the upper bound of $k$.
         $k^2 + k \le N$.
         $k(k+1) \le 2 \cdot 10^6$.
         $k \approx 1413$.
         Also $k \le Ones[N]$ (total ones).
         So loop condition: $k \le min((long long)Ones[N], 2000000LL)$.
         While $(k+1)*(k+2) \le N$: increment $k$. (Use careful check to avoid overflow, though $k$ small).
         Or just `while (k*(k+1) <= N && k <= Ones[N])`.
         
         Implementation details:
         Read string.
         Build `pref1` array. `pref1[i]` stores ones in `S[0..i-1]`. Size $N+1$.
         `pref1[0] = 0`.
         Loop $k=1, 2, \dots$
           $L = k*(k+1)$;
           If $L > N$ break.
           If $k > pref1[N]$ break.
           For $i = 0; i + L - 1 < N;$ // Substring indices $S[i \dots i+L-1]$ (length L)
               // Corresponds to prefix sums indices $i$ and $i+L$.
               // Substring starts at $i$ (0-based), ends at $i+L-1$.
               // Prefix count for start is `pref1[i]`.
               // Prefix count for end is `pref1[i+L]`.
               // Wait, problem uses 1-based or 0-based indices?
               // Problem says "substrings". $S[0\dots N-1]$.
               // Substring from index $l$ to $r$ (inclusive) has length $r-l+1$.
               // Number of ones is $P_1[r+1] - P_1[l]$.
               // We use $P_1$ size $N+1$. $P_1[x]$ is ones in $0..x-1$.
               // If we pick window of length $L$ starting at $i$ (0-based index in string):
               // Substring is $S[i \dots i+L-1]$.
               // Prefix sum indices are $i$ and $i+L$.
               // Count = $P_1[i+L] - P_1[i]$.
               // We check if count == k.
           Print ans.
           
         Wait, we need to check if $i+L \le N$.
         So $i$ goes from $0$ to $N - L$.
         Loop: `for (int i = 0; i + L <= N; ++i)`
         Inside: `if (pref1[i+L] - pref1[i] == k) ans++;`
         `ans` is `long long`.
         Be careful with $k=0$.
         Problem: "number of '0's in the substring is equal to the square of the number of '1's."
         If substring has $k$ ones, needs $k^2$ zeros.
         If $k=0$, then $0$ ones and $0$ zeros. Empty substring.
         Usually "substrings" implies non-empty.
         Does the problem allow empty substrings? "01-string".
         Standard interpretation: Non-empty.
         If empty substring is considered, answer includes it?
         But typically substrings of a string are non-empty contiguous parts.
         I will assume non-empty.
         However, if $k=0$ is considered, length is 0.
         My loop for $k$ starts at 1.
         If $k=0$ is valid, length 0.
         I will ignore $k=0$ (empty substring) as per convention.
         Wait, if the string has "0000", and $k=0$, $c_1=0 \implies c_0=0$.
         Substring "0" has $c_1=0, c_0=1 \neq 0$.
         Substring "" has $c_1=0, c_0=0$.
         If we include "", answer is 1.
         Usually not required. I'll stick to $k \ge 1$.
         Wait, what if the string is "1"?
         $k=1 \implies c_1=1, c_0=1^2=1$. Need "01" or "10" or "001001".
         Wait, $c_0=1$. Length 2.
         My code checks windows of size $L=k(k+1)$.
         For $k=1, L=2$.
         If string is "01", window "01" has 1 one, 1 zero. Match.
         So "01" is counted.
         If string is "0", $L=2 > 1$. Loop doesn't run. Count 0. Correct.
         If string is "00", $L=2$. Window "00". 0 ones. $0 \neq 1$. Count 0. Correct.
         If string is "10", $L=2$. Window "10". 1 one, 1 zero. Match.
         
         Double check constraints again.
         $N=2 \cdot 10^6$.
         Time limit.
         If I submit this, will it pass?
         $2 \cdot 10^9$ ops is borderline.
         But notice $L_k$ grows.
         For $k=1, L=2$. Loop runs $2 \cdot 10^6$ times.
         For $k=2, L=6$. Loop runs $2 \cdot 10^6$ times.
         ...
         For $k=1000, L \approx 10^6$. Loop runs $10^6$ times.
         For $k=1400, L \approx 2 \cdot 10^6$. Loop runs few times.
         Total iterations $\approx 1.6 \cdot 10^9$.
         This is high but maybe OK.
         The constant factor is extremely low (load int, read int, cmp, add).
         Let's ensure compiler optimization (-O3).
         Also, using `unsigned char` or `vector<char>` for string, and `vector<int>` for pref.
         Accessing `pref` is fast.
         
         One small optimization:
         `pref1[i+L] - pref1[i] == k`.
         This is `pref1[i+L] - k == pref1[i]`.
         Wait, `pref1[i]` is increasing. `pref1[i+L]` is increasing.
         Can we skip?
         Maybe not easily.
         
         Let's double check if there's an $O(N)$ solution or something better.
         We had $Ones[j] - Ones[i] = k$ and $j-i = k^2+k$.
         This means $Ones[j] - k = Ones[j - (k^2+k)]$.
         Let $k$ be variable.
         We want to match $Ones[x] - k = Ones[y]$ where $x - y = k^2+k$.
         This is matching pairs $(x, y)$ with $x-y = k^2+k$ and $Ones[x] - Ones[y] = k$.
         Rewrite: $Ones[x] - Ones[x - k^2 - k] = k$.
         Let $f_x = Ones[x]$.
         We need $f_x - f_{x - (k^2+k)} = k$.
         Notice that for large $k$, $k^2+k$ is large.
         For a fixed $x$, we can iterate over possible $k$.
         But $x - k^2 - k \ge 0$.
         So for each $x$, we iterate $k$ such that $k^2+k \le x$.
         Max $k \approx \sqrt{x}$.
         Total complexity $\sum_{x=1}^N \sqrt{x} \approx \int \sqrt{x} dx \approx \frac{2}{3} N^{1.5}$.
         Same complexity.
         Wait, iterating $k$ inside loop over $x$ means we do:
         `for x from 0 to N:`
         `  for k from 0 to sqrt(x):`
         `     check`
         This is same number of iterations.
         Which one is faster?
         Nested loop `for k` then `for x` (my proposed solution) accesses memory with stride $L_k$.
         Nested loop `for x` then `for k` accesses memory with stride 0 (re-reading `Ones[x]`).
         But `Ones[x - k^2 - k]` has stride.
         If `k` is small, $k^2+k$ is small, stride is small.
         If `k` is large, `k` loop is small.
         In `for x` loop:
         $k=1$: check $Ones[x] - Ones[x-2] == 1$.
         $k=2$: check $Ones[x] - Ones[x-6] == 2$.
         $k=3$: check $Ones[x] - Ones[x-12] == 3$.
         Strides increase.
         This might be worse for cache than iterating $k$ first because in $k$ loop we traverse the array with stride $L_k$.
         Actually, traversing array with stride $L_k$ is bad if $L_k$ doesn't align.
         But if $L_k$ is small (most iterations), it is good.
         If $L_k$ is large, we do fewer iterations.
         Also, the `k` loop approach iterates $i$ from $0$ to $N-L$.
         This is contiguous memory access for `Ones[i]` and contiguous for `Ones[i+L]`.
         Wait, `Ones[i]` and `Ones[i+L]` are separated by distance $L$.
         If we load `Ones[i]`, `Ones[i+L]` might be far.
         But for $k=1$, $L=2$, contiguous-ish.
         For $k$ large, we do few checks.
         So `k` then `i` loop is probably better.
         
         Let's stick with `k` loop then `i` loop.
         Code structure:
         ```cpp
         #include <iostream>
         #include <vector>
         #include <string>
         #include <algorithm>
         
         using namespace std;
         
         int main() {
             ios_base::sync_with_stdio(false);
             cin.tie(NULL);
             
             string s;
             if (!(cin >> s)) return 0;
             
             int n = s.length();
             vector<int> pref(n + 1, 0);
             for (int i = 0; i < n; ++i) {
                 pref[i + 1] = pref[i] + (s[i] - '0');
             }
             
             long long ans = 0;
             int max_k = 0;
             // Calculate approximate max_k. 
             // k*(k+1) <= n => k approx sqrt(n). 
             // Also k <= pref[n].
             // Since n <= 2*10^6, k <= 1500.
             // pref[n] can be up to 2*10^6.
             
             // We can just loop k.
             for (int k = 1; ; ++k) {
                 long long L = 1LL * k * (k + 1);
                 if (L > n) break; // length exceeds string
                 if (k > pref[n]) break; // need k ones, but string has fewer
                 
                 int limit = n - (int)L;
                 // Accessing pref array. 
                 // To avoid branch misprediction or overhead?
                 // Simple loop is fine.
                 // pref[i+L] - pref[i] == k
                 
                 // Optimization: check pref[L] - pref[0] == k?
                 // No.
                 
                 // We can perform a simple check.
                 // Using pointer arithmetic might be slightly faster than indexing?
                 // Not significantly.
                 
                 // Let's iterate.
                 // Using local variable for L to prevent calculation every time.
                 for (int i = 0; i <= limit; ++i) {
                     if (pref[i + L] - pref[i] == k) {
                         ans++;
                     }
                 }
             }
             
             cout << ans << endl;
             
             return 0;
         }
         ```
         Is there any chance of TLE?
         If $N=2 \cdot 10^6$, `limit` goes from $2 \cdot 10^6$ down to $0$ roughly.
         Total iterations $\approx 1.6 \cdot 10^9$.
         In modern contest environments (e.g. Codeforces), 1.6 billion simple ops might TLE (usually 10^8 is limit).
         However, this is very simple arithmetic.
         Can we optimize the inner check?
         We need to count $i$ such that $pref[i+L] - pref[i] = k$.
         Let $val = pref[i+L] - k$. We need to check if $val == pref[i]$.
         We can iterate $i$.
         Wait, if we just iterate $i$, we have to check condition.
         Can we avoid checking every $i$?
         Notice that for large $L$, most $pref[i+L] - pref[i]$ will be large (if density of ones is uniform).
         If density of ones is $D \approx 0.5$.
         Expected diff is $L/2$.
         We need diff $= k$.
         So we need $L/2 \approx k \implies k(k+1)/2 \approx k \implies k \approx 2$.
         If $k$ is large, $L$ is large. $L/2 \gg k$.
         So we expect very few matches if ones are uniformly distributed.
         If string has very few ones ($pref[n] \ll N$), then $L$ can be large but $pref$ diff is small?
         If $pref[n]$ is small, $k$ loop breaks early.
         So worst case is $pref[n] \approx N/2$.
         Then $L \approx k^2$. Diff $\approx L/2 \approx k^2/2$.
         We need Diff $= k$.
         So $k^2/2 = k \implies k=2$.
         For $k > 4$, expected diff $\gg k$. Matches rare.
         Can we skip regions where diff is guaranteed $> k$?
         $pref[i+L] - pref[i]$ is number of ones in window.
         If $pref[i+L] - pref[i] > k$, we can't increment.
         But we still need to check.
         Since $pref$ is monotonic, maybe we can use two pointers?
         We want to find number of windows of length $L$ with sum $k$.
         Since $L$ is fixed for inner loop.
         Let $W_i = pref[i+L] - pref[i]$. This is the number of ones in window starting at $i$.
         We want count of $i$ where $W_i = k$.
         Since $pref$ changes by 0 or 1, $W_i$ changes by -1, 0, or 1 as $i$ increments.
         (Specifically $W_{i+1} - W_i = pref[i+L+1] - pref[i+1] - (pref[i+L] - pref[i]) = s[i+L] - s[i]$).
         $s[i+L] - s[i]$ is 0 or 1 or -1.
         So $W_i$ is a walk with steps -1, 0, 1.
         We need to find values equal to $k$.
         This can be done by just iterating.
         Is there a way to jump?
         If $W_i$ stays above $k$ for long?
         Probably not helpful.
         
         Wait, we can use the fact that for large $k$, $L$ is large.
         The window size is large.
         If $L \approx k^2$, and number of ones is $k$.
         Then the window must be mostly zeros.
         So we can search for blocks of zeros.
         The window contains $k$ ones.
         The distance between first and last one must be $\le k^2+k-1$.
         Actually, if the window contains $k$ ones, the ones are at indices $p, p+1, \dots, p+k-1$ (in ones array).
         We need distance $ones[p+k-1] - ones[p] \le L-1$.
         Also $ones[p+k] - ones[p] \ge L$?
         Basically, we iterate over $k$ ones in the original string.
         This was my "ones array" method.
         Method: Iterate $k$. Iterate blocks of $k$ ones. Check if block can be contained in window.
         This required iterating $j$ (index of block start).
         Complexity was $O(M \sqrt{N})$ where $M$ is number of ones.
         If $M \approx N$, this is same as $O(N \sqrt{N})$.
         But if $M$ is small, it's faster.
         Wait, $M \approx N$ means string is half 0s half 1s.
         If $M$ is large, $k$ loop runs fully.
         Is the "ones array" method faster in practice?
         Inner loop body is:
         $LB = \dots, RB = \dots$.
         Comparison and addition.
         But wait, in the `k`-then-`i` loop, we access `pref` which is size $N$.
         In `ones` array approach, we access `ones` which is size $M$.
         If $M \approx N$, same.
         However, in `ones` approach, we compute bounds.
         Actually, in `ones` approach, we have a condition `ones[j+k-1] - ones[j] <= k^2+k-1`.
         This condition might filter out many $j$'s quickly?
         Not really.
         
         Wait, if we use `ones` array approach, the inner loop runs $M-k+1$ times.
         Sum is $\sum (M-k) = M\sqrt{N} - N^{1.5}/3 \approx 2/3 M \sqrt{N}$.
         The `pref` array approach runs $N-k(k+1)$ times.
         Sum is $\sum (N - k^2) \approx N \sqrt{N} - N^{1.5}/3$.
         Both are $\approx 1.6 \cdot 10^9$ ops for $N=M=2 \cdot 10^6$.
         The difference is negligible.
         However, the `pref` array approach involves reading random memory locations?
         No, $pref[i]$ and $pref[i+L]$.
         If $L$ is large, random.
         If $L$ is small, sequential.
         As $k$ increases, $L$ increases.
         For small $k$, $L$ is small, so prefetch works.
         For large $k$, $L$ is large, but few iterations.
         So `pref` approach is likely cache-friendly.
         
         Wait, is it possible to optimize the check `pref[i+L] - pref[i] == k`?
         We can rewrite `pref[i+L] - k == pref[i]`.
         If we view this as finding $i$ where value at $i$ matches value at $i+L$ minus shift.
         We can use a map? No.
         
         Wait, I remember seeing this problem or similar before.
         Maybe $N$ is small enough for $O(N \sqrt{N})$?
         Or maybe the test cases are weak?
         Usually, competitive programming problems with $N=2 \cdot 10^6$ require $O(N)$ or $O(N \log N)$.
         Is there an $O(N \log N)$?
         The condition is $Pref[i+L_k] - Pref[i] = k$.
         This is counting occurrences.
         This is equivalent to counting occurrences of value $k$ in the difference array $D_k$.
         $D_k[i] = Pref[i+L_k] - Pref[i]$.
         We need $\sum_k \text{freq}_k(k)$.
         Notice that $L_k = k(k+1)$.
         Also $Pref[i+L_k] - Pref[i]$ is simply the number of ones in substring.
         Let's flip the problem.
         For each substring, let $cnt_0$ be number of zeros, $cnt_1$ be number of ones.
         We need $cnt_0 = cnt_1^2$.
         $cnt_0 + cnt_1 = len$.
         $cnt_1^2 + cnt_1 = len$.
         $cnt_1(cnt_1+1) = len$.
         So $len$ must be of form $k(k+1)$.
         This implies we only care about substrings with length $L_k$.
         So we are partitioning the problem into disjoint sets based on length (modulo existence).
         For a specific length $L$, we count substrings with $k = \sqrt{L - \frac{1 - \dots}{2}}$? No, $k$ is determined by $L$ uniquely (for $L \ge 2$).
         So we just need to count for each $k$ how many windows of size $L_k$ have $k$ ones.
         This is what we are doing.
         Is there a faster way to count windows of specific length $L$ with specific ones count?
         We can use Mo's algorithm? No, too slow.
         We can use a persistent segment tree?
         We want to query range sums.
         We want $\sum [ (Query(1, i, i+L) - Query(0, i, i+L) == k) ? 1 : 0 ]$? No, 0 is implied.
         We need count of ones in range $[i, i+L]$.
         This is just prefix sum difference.
         Is there a way to compute $\sum_{i} [Ones[i+L] - Ones[i] = k]$ faster?
         If we construct a data structure?
         Maybe FFT?
         We want $Ones[i+L] - Ones[i] = k$.
         $Ones[i+L] - k = Ones[i]$.
         Let $A[i] = Ones[i]$.
         We want to match $A[i+L] - k$ with $A[i]$.
         This is correlation?
         If $L$ varies, it's hard.
         
         Wait, I suspect $O(N \sqrt{N})$ is intended or acceptable for C++.
         The constraint is $N \le 2 \cdot 10^6$.
         Time limit 2.0s?
         If time limit is 1.0s, it might be tight.
         However, I should provide the most efficient solution.
         The loop optimization is key.
         
         One trick:
         We can iterate $i$ and update possible answers?
         No.
         
         Wait, look at `ones` array approach again.
         We iterate blocks of $k$ ones.
         Block is $ones[j \dots j+k-1]$.
         Check if `ones[j+k-1] - ones[j] <= L-1` (necessary).
         This condition fails for many $j$ if ones are dense?
         No, if ones are dense, $ones[j+k-1] - ones[j]$ is small (approx $k$).
         So $k \le k^2+k-1$ is always true for $k \ge 1$.
         Wait, if ones are dense, block condition holds.
         If ones are sparse, condition fails?
         If sparse, distance is large.
         But if distance is large ($> L$), we can't cover it.
         So if ones are sparse, we fail often.
         If ones are dense, we succeed often?
         Wait, if ones are dense (many ones in small space), $L$ is huge ($k^2$).
         So we have lots of zeros required.
         But if ones are dense, there are no zeros?
         Contradiction.
         Dense ones $\implies$ low $zeros$.
         $L_k = k^2+k$. Contains $k$ ones. So $k^2$ zeros.
         So we need $k^2$ zeros.
         If string has many ones, it has fewer zeros.
         So it is unlikely to have $k^2$ zeros if $k$ is large.
         So for large $k$, if ones are dense, count should be 0.
         The condition `ones[j+k-1] - ones[j] <= L-1` ensures that we found a block of ones that *could* fit.
         It does not guarantee there are enough zeros.
         The zeros are implicitly handled by the bounds $LB, RB$.
         
         Actually, the `ones` approach has a hidden complexity:
         The number of iterations is $\sum (M-k)$.
         If $M$ is large, this is bad.
         However, if $M$ is large, we are looking for $k^2$ zeros.
         Total length $k^2+k$.
         If $M$ is close to $N$, then string is mostly ones.
         Then $k^2+k$ (required zeros + ones) will be huge, but we only have $k$ ones available?
         Wait, $k$ is the number of ones in substring.
         If $M$ is large, we can pick $k$ ones.
         But we need $k^2$ zeros.
         If $M \approx N$, zeros are few.
         So $k^2$ will likely exceed available zeros.
         Available zeros $Z = N - M$.
         We need $k^2 \le Z$.
         So if $Z < 1$, no solution (unless $k=0$).
         So $k \le \sqrt{Z}$.
         So loop for $k$ only goes up to $\sqrt{Z}$.
         If $Z$ is small, $k$ loop is short.
         So complexity is actually bounded by $\min(\sqrt{N}, \sqrt{Z}) \cdot M$?
         No.
         The number of valid $k$'s is $\min(\sqrt{Z}, \sqrt{M})$?
         No, $k$ ones and $k^2$ zeros.
         So $k \le M$ (available ones) AND $k^2 \le Z$ (available zeros).
         So $k \le \min(M, \sqrt{Z})$.
         Let $K_{max} = \min(M, \sqrt{Z})$.
         Complexity is $\sum_{k=1}^{K_{max}} M$.
         $M \cdot K_{max}$.
         If $M$ is small, fast.
         If $M$ is large, $Z$ is small, so $K_{max}$ is small.
         Specifically $M \approx N \implies Z \approx 0 \implies K_{max} \approx 0$.
         $M \approx N/2 \implies Z \approx N/2 \implies K_{max} \approx \sqrt{N/2}$.
         Complexity $\approx \frac{N}{2} \sqrt{N/2} \approx \frac{N^{1.5}}{2.8}$.
         If $M$ is small, $Z$ is large. $K_{max} \approx \sqrt{Z} \approx \sqrt{N-M}$.
         Complexity $\approx M \sqrt{N-M}$.
         Max of $x(N-x)$ is at $x=N/2$.
         So worst case is when ones and zeros are balanced.
         Then complexity $\approx N \sqrt{N}$.
         This confirms worst case is $O(N \sqrt{N})$.
         But wait, in the `ones` approach, inner loop calculates intersection.
         Is it possible that intersection is empty often?
         Intersection is non-empty if $LB \le RB$.
         $LB \approx ones[j-1] + 1$. $RB \approx ones[j+k] - L$.
         $LB \le RB \iff ones[j-1] - ones[j+k] \le -L+1 \iff ones[j+k] - ones[j-1] \ge L-1$.
         This condition basically says there are enough zeros between the ones $j-1$ and $j+k$.
         Specifically, distance between outer ones must be at least $L$.
         This is equivalent to saying there are $\ge L - (k+1)$ zeros in between?
         Approximation: distance $\approx$ zeros count.
         So we need enough zeros.
         If $Z$ is small, then for large $L$, condition $ones[j+k] - ones[j-1] \ge L$ will rarely hold.
         Because total range of indices is $N$.
         If we require span $\ge L$ between specific ones?
         Actually, for all $j$, this condition might be violated if ones are dense?
         Wait, if ones are dense, $ones[j+k] - ones[j-1]$ is small (approx $k$).
         We need it $\ge k^2+k$.
         If $k \ge 1$, $k^2+k > k$.
         So if ones are dense ($ones[j+k] \approx ones[j-1] + k$), then condition fails immediately.
         So if ones are dense, loop runs fast (condition check fails).
         But wait, the loop body still executes.
         Can we break early?
         If $ones[j+k] - ones[j-1] < k^2+k$, we know nothing?
         No, we need $\ge k^2+k$.
         If $ones[j+k] - ones[j-1] < k^2+k$, it is impossible to fit the window of size $L$ between these ones?
         Wait, if distance between outer ones is small, then all ones are clustered.
         We need to fit window of size $k^2+k$ containing these $k$ ones, AND excluding $j-1$ and $j+k$.
         This means the window must start after $ones[j-1]$ and end before $ones[j+k]$.
         So the width available for window is $ones[j+k] - ones[j-1] - 1$.
         Wait, $ones[j-1]$ is index. $ones[j+k]$ is index.
         Space between them is $ones[j+k] - ones[j-1] - 1$.
         We need to place a window of size $L = k^2+k$ inside this space?
         Wait, we don't need to place window *only* between them.
         We need window to cover $j \dots j+k-1$ and exclude $j-1$ and $j+k$.
         So yes, the entire window must be contained in $(ones[j-1], ones[j+k])$.
         So length of space available is $ones[j+k] - ones[j-1] - 1$.
         We need this to be $\ge L = k^2+k$.
         So if $ones[j+k] - ones[j-1] - 1 < k^2+k$, then no solution for this block.
         Since $ones[j+k] - ones[j-1]$ is sum of gaps.
         If gaps are small (dense ones), this check fails quickly.
         If gaps are large (sparse ones), it succeeds.
         If it fails, we can continue.
         Wait, we still need to sum up valid ranges.
         If condition fails, we add 0.
         But we still iterate $j$.
         Can we break the $j$ loop?
         $ones$ is increasing.
         $ones[j+k] - ones[j-1]$ is increasing with $j$ (monotonically non-decreasing).
         So if for some $j$, $ones[j+k] - ones[j-1]$ is small, it doesn't help to stop?
         Wait, as $j$ increases, $ones[j+k]$ and $ones[j-1]$ increase.
         $ones[j+k] - ones[j-1] = (ones[j+k] - ones[j]) + (ones[j] - ones[j-1])$.
         This is sum of gaps between $j-1$ and $j+k$.
         It fluctuates?
         It tends to increase on average if density is positive.
         Actually, $ones[x]$ is increasing, so $ones[j+k] - ones[j-1] \ge ones[(j+1)+k] - ones[(j+1)-1]$?
         $ones[j+k+1] - ones[j] \ge ones[j+k] - ones[j-1]$?
         Not necessarily. But generally it drifts up.
         Actually, it's the length of segment containing $k+1$ ones (indices $j \dots j+k$).
         This length tends to stay roughly constant or drift slowly.
         So we cannot break early.
         
         Back to $O(N \sqrt{N})$ implementation.
         Given constraints and typical CP limits, this is likely the intended solution or acceptable.
         I will provide the solution with the optimized loop over $k$.
         The `pref` array solution is easier to implement and likely fast enough due to simple ops and prefetching.
         Also it handles "dense ones" case naturally (loops fewer times? No, loop count is $N-L$).
         Wait, if $M$ is small, $L_k$ is large, so $N-L$ is small.
         If $M$ is large (so $Z$ small), then we need $k^2$ zeros.
         If $Z < k^2$, then count is 0.
         So if $M$ is large, max $k$ is small.
         Wait, my loop condition `if (k > pref[n]) break;` stops if $k > M$.
         It does NOT stop if $k^2 > Z$.
         However, if $k^2 > Z$, then condition $pref[i+L] - pref[i] == k$ will never be true?
         Wait, $pref[i+L] - pref[i]$ is count of
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=1216)

````
Let's analyze this problem:

Given a 01-string, we need to find the number of substrings where count('0') = (count('1'))².

For example, if we have a substring like "0011", count('0') = 2, count('1') = 2, and 2 ≠ 2² = 4.
If we have "001", count('0') = 2, count('1') = 1, and 2 = 1² ✓ This is valid.

Key observations:
1. If count('0') = k, then count('1') must be √k for some perfect square k.
2. So possible values are: (count '1' = 0 → count '0' = 0), (count '1' = 1 → count '0' = 1), (count '1' = 2 → count '0' = 4), (count '1' = 3 → count '0' = 9), etc.

Approach:
- For each position in the string, we want to count how many starting positions exist such that the substring from that position has the required property.
- We can use prefix sums: let prefix[i] be the pair (number of 0s, number of 1s) in s[0...i].
- A substring s[l...r] has n zeros and m ones if it means prefix[r].zeros - prefix[l-1].zeros = n AND prefix[r].ones - prefix[l-1].ones = m.
- The condition is: n = m²

Let me try a different approach with better complexity:
- For each ending position r, we want to count starting positions l where the substring satisfies the condition.
- We need: zeros[l..r] = (ones[l..r])²

This can be solved using a map/dictionary for counting, since for small values of ones the corresponding zeros value is relatively small.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    string s;
    cin >> s;
    
    long long answer = 0;
    int prefixZero = 0;
    int prefixOne = 0;
    map<int, int> posMap;
    
    // For substring s[l..r], with a 0 count = Z and 1 count = O,
    // we want: prefixZero[r] - prefixZero[l-1] = O², 
    //            prefixOne[r] - prefixOne[l-1] = O
    
    posMap[0] = 1;
    
    for (char c : s) {
        if (c == '0') {
            prefixZero++;
        } else {
            prefixOne++;
        }
        
        // O ranges from 1 to ~√N
        for (int O = 1; O * O <= prefixOne + prefixZero && O * O <= prefixZero + prefixZero; ++O) {
            int targetZ = O * O;
            
            // For this O value, we want prefixZero[l-1] = prefixZero[r] - O²
            //                and prefixOne[l-1] = prefixOne[r] - O
            
            long long prevZ = (long long)prefixZero - targetZ;
            long long prevO = prefixOne - O;
            
            // Only add if previous counts match O value
            // (since a prefix state with fewer than O ones cannot contribute)
            if (prevZ > 0 && prevO == O) {
                if (posMap.find((int)prevZ) != posMap.end()) {
                    answer += posMap[(int)prevZ];
                }
            }
        }
        
        posMap[prefixZero]++;
    }
    
    cout << answer << endl;
    return 0;
}
```
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    string s;
    cin >> s;
    
    long long answer = 0;
    int prefixZero = 0;
    int prefixOne = 0;
    vector<int> posMap(s.length() + 1, 0);
    
    for (int r = 0; r < (int)s.length(); ++r) {
        if (s[r] == '0') {
            prefixZero++;
        } else {
            prefixOne++;
        }
        
        // O ranges from 1 to roughly sqrt(prefixZero)
        for (int O = 1; O * O <= prefixZero; ++O) {
            int targetZ = O * O;
            
            long long prevZ = (long long)prefixZero - targetZ;
            long long prevO = prefixOne - O;
            
            if (prevZ >= 0 && prevO >= 0 && prevO == O) {
                int prevPos = (int)prevZ;
                if (prevPos <= r && prevPos >= -1) {
                    int count = (prevPos == -1) ? 1 : ((int)(prevPos < 0 ? 0 : 1));
                    answer += count;
                }
            }
        }
    }
    
    cout << answer << "\n";
    return 0;
}
```
````
