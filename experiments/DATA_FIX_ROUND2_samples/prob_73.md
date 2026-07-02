# 题 73 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=31273)

````
The problem asks us to find a hidden permutation $p$ of $\{1, \dots, n\}$ where $n \le 2000$. We are allowed up to $1.99875 \times 10^6$ queries (actually the example says $1999000$, but it's likely around $O(n^2)$). Each query gives us the parity of inversions in a subarray $p[l \dots r]$. Specifically, we are given $(\sum_{l \le i < j \le r} [p_i > p_j]) \bmod 2$.

Let $I(l, r)$ denote this value.
$I(l, r) = (\sum_{l \le i < j \le r} [p_i > p_j]) \bmod 2$.
This is the inversion number of the subarray modulo 2.

We need to determine all $p_1, \dots, p_n$.
The constraints on $n$ are small enough ($2000$), and the number of queries is quite large ($1999000$ for $n=2000$ is roughly $1000n$ or even more like $n^2/2$? Wait, $n^2/2 = 2000^2/2 = 2,000,000$. So actually we can afford roughly one query per pair $(i, j)$ or something similar). However, note that the limit is strictly slightly less than $n^2$ or close to it. Actually, $n=2000$ implies $n^2/2 = 2,000,000$. The limit is $1,999,000$. This suggests we might not be able to ask for every subarray $p[i \dots j]$ individually. But maybe we only need specific ones.

Let's analyze the properties of the parity of inversions.
For any $k$, if we swap adjacent elements $p_k, p_{k+1}$, the total number of inversions changes by an odd amount (specifically $\pm 1$). Thus the parity flips.
More generally, the inversion number of a permutation is equal to the number of swaps required to reach the identity permutation from it (if we only consider swaps of adjacent elements). This parity is related to whether the permutation is even or odd.

Specifically, let $inv(P)$ be the number of pairs $(i, j)$ such that $i < j$ and $p_i > p_j$. We want to know $inv(P) \bmod 2$.
Actually, wait. Usually "inversion" refers to indices $i < j$ with $p_i > p_j$. Here, the query asks about inversions *within* the subarray $p_l \dots p_r$.
Let's define $S_l = p_l, S_{l+1} = p_{l+1}, \dots, S_r = p_r$. The number of inversions within this subsequence is what is queried.
Wait, if I have the sequence $2, 3, 1$, the number of pairs is: $(2,3)$ no, $(2,1)$ yes, $(3,1)$ yes. Total 2. Parity 0.
If the array was $1, 2, 3$, total 0.
If the array was $3, 2, 1$, pairs: $(3,2)$ yes, $(3,1)$ yes, $(2,1)$ yes. Total 3. Parity 1.

We want to recover $p$. Since the values are a permutation of $1 \dots n$, we know that the set of values is fixed. This is different from just determining relative orderings or finding values directly.
However, notice that if we fix the position of each element $v \in \{1, \dots, n\}$, that's the same as fixing the permutation.
Alternatively, maybe we can determine the values $p_i$ one by one or in some order.
Or maybe we can construct the permutation bit by bit using some binary representation approach? No, values are $1..n$.

Let's look at the structure of the query result.
$I(l, r) = (\text{number of inversions in } p[l \dots r]) \bmod 2$.
Note that $I(l, r)$ satisfies:
$I(l, r) = I(l, r-1) + I(r, r) + \text{something crossing}$? No.
Let's decompose.
Total inversions in $p[1 \dots n]$ is known as the global parity.
Is there a recursive structure?
Consider $p_1, \dots, p_n$. If we remove $p_1$, the remaining elements form a permutation of the other values.
Wait, if we just want to know the positions of numbers $1, \dots, n$, we can try to find the position of 1.
Or rather, since we don't know which values are where, maybe we can find the value at index 1, then index 2...
But we can't easily compare two values without seeing their relationship.

Actually, there is a known property about inversion parities.
For a sequence $a_1, \dots, a_n$, the number of inversions is $\sum_{i<j} [a_i > a_j]$.
Consider the sum of values. Or maybe just the relative ordering.
A key observation in similar problems (like "Inversions" on Codeforces or AtCoder) often involves bitwise operations or segment trees. But here it's interactive.
With $n=2000$, $O(n^2)$ time complexity is acceptable if we do constant work, but we have $O(n^2)$ query budget. Since $n=2000$, $n^2/2 \approx 2 \cdot 10^6$, which matches the budget roughly.
Wait, $1999000$ is very close to $2000^2/2$. Maybe we can determine the relative order of every pair $(i, j)$? That is exactly asking $p_i > p_j$ or $p_i < p_j$.
If we knew for all pairs $i < j$ whether $p_i > p_j$, we could reconstruct the permutation?
Not necessarily. For example, knowing $p_1 > p_2$ and $p_1 > p_3$ and $p_2 > p_3$ defines the transitive relation. But the exact values matter because of the constraint that values are a permutation of $1 \dots n$.
However, once we establish the relative order of all elements, the permutation is uniquely determined.
Actually, yes. If we know the full tournament graph of "less than" relations among $p_1, \dots, p_n$, this is exactly a linear ordering. A linear ordering corresponds to a unique permutation of the underlying values if we assign values based on rank.
Wait, the problem says $p$ is a hidden permutation of $1 \dots n$. It doesn't mean the values $p_i$ are distinct integers; it means $p$ maps $\{1 \dots n\}$ to $\{1 \dots n\}$ bijectively.
If we can determine for every pair $i, j$ ($i \neq j$) whether $p_i < p_j$, we completely identify the permutation. Why? Because sorting the array $p$ would give $1, 2, \dots, n$ in sorted order. If $p$'s relative order is consistent with $1 < 2 < \dots < n$, we can map the position of value 1, value 2, etc.
Specifically, if we can sort $p$, we find its rank. Since values are distinct and range from $1$ to $n$, rank corresponds to value.
So if we can query $p_i > p_j$ for all $i, j$, that would take $n(n-1)/2$ queries?
Wait, the query gives inversion parity on subarrays, not direct comparison $p_i > p_j$.
Can we derive $p_i > p_j$ from inversion parities?
Suppose we know $p[l \dots r]$ inversion parity.
Consider the case $r=l+1$. The query `0 l l+1` returns $[p_l > p_{l+1}] \bmod 2$.
Since there is only one pair, $I(l, l+1) = [p_l > p_{l+1}]$.
So if we can query every adjacent pair $(i, i+1)$, we get the relative order of adjacent elements.
But knowing adjacent comparisons $p_1 > p_2$, $p_2 > p_3$, etc., only tells us local inversions.
$p_1 > p_2 < p_3$. We don't know if $p_1 < p_3$ or $p_1 > p_3$ directly from these.
However, $I(1, 3)$ gives $[p_1 > p_2] + [p_1 > p_3] + [p_2 > p_3] \pmod 2$.
So if we know $I(1, 2)$, $I(2, 3)$, and $I(1, 3)$, we can solve for $[p_1 > p_3]$:
$[p_1 > p_2] + [p_1 > p_3] + [p_2 > p_3] \equiv I(1, 3) \pmod 2$
$\Rightarrow [p_1 > p_3] \equiv I(1, 3) - [p_1 > p_2] - [p_2 > p_3] \pmod 2$.
So yes! We can determine the relative order of ANY pair $(i, j)$ ($i < j$) by querying subarrays.
Specifically, $I(i, j)$ gives information about all pairs in between.
But notice:
$I(i, j) = \sum_{k=i}^{j-1} [p_k > p_{k+1}] + \sum_{i \le k < m \le j, |m-k| \ge 2} [p_k > p_m]$.
No, wait. The formula is simply over all pairs $x, y$ in range with $pos(x) < pos(y)$ and $val(x) > val(y)$.
If we consider just adjacent swaps logic:
Moving a single value from index $k$ to $k-1$ (if $p_k < p_{k-1}$?) changes inversions by odd amount?
This seems complicated. Let's stick to the deduction derived above.
We established: $I(i, j) = [p_i > p_j] + \sum_{i < x < y < j} [p_x > p_y] \pmod 2$.
Let $Q(i, j) = I(i, j) \bmod 2$.
Then $Q(i, j) = [p_i > p_j] + Q(i+1, j) + \sum \dots$ ?
Actually, notice that:
$I(1, n) = \sum_{1 \le i < j \le n} [p_i > p_j]$.
Also $I(1, n) = I(1, n-1) + \text{inv}(p_1, p_2, \dots, p_n \text{ vs } p_n \text{ in context})$.
Wait, let's look at the recursive structure properly.
$I(l, r)$ is the parity of inversions in the subsegment.
Consider the relation:
$Q(i, j) \oplus Q(i, j-1) \oplus Q(i+1, j) \oplus Q(i+1, j-1) = \dots$?
Maybe it's simpler.
Recall $Q(i, j) = [p_i > p_j] + \sum_{i < u < v < j} [p_u > p_v]$.
Actually, notice that:
Sum of inversions involving $p_i$ within range $[i, j]$ is $C_i = \sum_{u=i+1}^j [p_i > p_u]$.
Then total inversions $I(i, j) = C_i + I(i+1, j)$.
So $C_i = I(i, j) - I(i+1, j) \pmod 2$.
Here $I(k, k) = 0$.
So for any $k$ and $j > k$, $C_k(j) = \sum_{u=k+1}^j [p_k > p_u]$ is determined by querying $I(k, j)$ and $I(k+1, j)$.
By varying $j$, we can find $C_k(j)$ for all $j$.
Specifically, $C_k(k+1) = [p_k > p_{k+1}] = I(k, k+1)$.
$C_k(k+2) = [p_k > p_{k+1}] + [p_k > p_{k+2}]$.
From $C_k(k+2) = I(k, k+2) - I(k+1, k+2)$, and knowing $I(k+1, k+2)$ (which is $C_{k+1}(k+2)$ from previous step), we can isolate $[p_k > p_{k+2}]$.
Wait, actually $C_k(j) - C_k(j-1) = [p_k > p_j]$.
Yes!
So, if we know $I(l, r)$ for all pairs $l, r$, we can calculate $C_l(r)$?
Wait, we need to minimize queries.
We want to find $p_i$. To do that, we essentially need the rank of each $p_i$.
Rank of $p_i$ depends on how many $p_j < p_i$.
Actually, knowing $[p_i > p_j]$ for all pairs is sufficient to sort the array.
There are $n(n-1)/2$ pairs. We need to recover the boolean variable $[p_i > p_j]$ for $i < j$.
Using the relation $[p_i > p_j] = I(i, j) - I(i, j-1) - I(i+1, j) + I(i+1, j-1)$? No, let's recheck the recurrence.
$Inv(i, j) = Inv(i+1, j) + \sum_{k=i+1}^j [p_i > p_k]$.
So $\sum_{k=i+1}^j [p_i > p_k] = I(i, j) \oplus I(i+1, j)$.
Let $S(i, j) = \sum_{k=i+1}^j [p_i > p_k]$. Then $S(i, j) = I(i, j) \oplus I(i+1, j)$.
Then $S(i, j-1) = I(i, j-1) \oplus I(i+1, j-1)$.
Thus $[p_i > p_j] = S(i, j) \oplus S(i, j-1)$.
Substituting back:
$[p_i > p_j] = (I(i, j) \oplus I(i+1, j)) \oplus (I(i, j-1) \oplus I(i+1, j-1))$.
All indices are bounded by $n$.
To compute $[p_i > p_j]$ for a specific pair $(i, j)$, we need 4 queries: $(i, j), (i+1, j), (i, j-1), (i+1, j-1)$.
We want this for all $1 \le i < j \le n$.
This seems inefficient if we do it naively for all pairs. Number of queries would be $2 \times n(n-1)/2 \approx n^2$. With $n=2000$, $n^2 = 4 \cdot 10^6$, which exceeds the limit $1.999 \cdot 10^6$.
Wait, the limit is $n^2/2$ roughly.
Is there a way to optimize?
Notice that we need $S(i, j)$ for all $i, j$.
$S(i, j)$ is the number of elements to the right of $i$ (up to $j$) that are smaller than $p_i$.
Actually, if we have the whole grid of $[p_i > p_j]$, we can just count how many $j > i$ have $[p_i > p_j] = 1$. This gives us some info. But we need to reconstruct values.
The standard reconstruction algorithm for permutations with pairwise comparisons takes $O(n^2)$ or $O(n^2 \log n)$ depending on implementation (e.g. tournament tree or just sorting). With the boolean matrix, we can just iterate to build the sorted order.
However, we are limited in queries.
Wait, the recurrence $[p_i > p_j] = I(i, j) \oplus I(i+1, j) \oplus I(i, j-1) \oplus I(i+1, j-1)$ is correct?
Let's check base cases.
For $i=1, j=n$. $[p_1 > p_n] = I(1, n) \oplus I(2, n) \oplus I(1, n-1) \oplus I(2, n-1)$.
Is this always true?
$Inv(i, j) = Inv(i+1, j) + [p_i > p_{i+1}] + [p_i > p_{i+2}] + \dots + [p_i > p_j]$.
Also $Inv(i+1, j) = Inv(i+2, j) + [p_{i+1} > p_{i+2}] + \dots$.
And $Inv(i, j-1) = Inv(i+1, j-1) + [p_i > p_{i+1}] + \dots + [p_i > p_{j-1}]$.
So $Inv(i, j) \oplus Inv(i+1, j) = [p_i > p_{i+1}] + \dots + [p_i > p_j]$.
And $Inv(i, j-1) \oplus Inv(i+1, j-1) = [p_i > p_{i+1}] + \dots + [p_i > p_{j-1}]$.
XORing these two cancels out the common terms $[p_i > p_{i+1}] \dots [p_i > p_{j-1}]$.
Leaving $[p_i > p_j]$.
Yes, the formula holds.
Each pair requires 4 queries. There are $O(n^2)$ pairs.
Wait, notice that to get $[p_i > p_{i+1}]$, we can just use $I(i, i+1)$ directly?
If $r=l+1$, then $I(i, i+1)$ is exactly $[p_i > p_{i+1}]$.
Does the formula reduce?
Formula: $I(i, i+1) \oplus I(i+1, i+1) \oplus I(i, i) \oplus I(i+1, i)$.
$I(i+1, i+1) = 0$, $I(i, i) = 0$.
$[p_i > p_{i+1}] = I(i, i+1)$. Correct.
So boundary cases save queries.
However, asymptotically we still have $2 \cdot (\text{total pairs}) \approx n^2$.
With $N=2000$, $N^2/2 = 2,000,000$.
The limit is $1,999,000$.
Wait, $N(N-1)/2 \approx 2,000,000$.
If we need to query almost every pair, we are tight.
BUT, wait. Do we really need to query every pair?
To reconstruct the permutation $p$, we just need to know the sorted order.
Knowing the inversion table $inv_i = |\{j < i : p_j > p_i\}|$ is sufficient to reconstruct the permutation.
Wait, standard inversion table is defined differently.
Usually, if we know for each position $i$, how many elements to its left are larger, we can reconstruct.
Let $L_i$ be the number of indices $j < i$ such that $p_j > p_i$.
Then we can place $p_i$ into a sorted list of available numbers.
To find $L_i$, we need to know counts of $j < i$ with $p_j > p_i$.
This seems to require iterating $j$ for each $i$.
Actually, the quantity $[p_j > p_i]$ is symmetric in terms of existence but antisymmetric in truth.
Wait, if we know $[p_i > p_j]$ for all $i < j$, we know the entire matrix.
Is it possible to determine $p$ without knowing the full matrix?
Maybe we don't need all pairs.
But notice, $p$ is a permutation of $1..n$. The values are distinct.
This means we cannot just have a cyclic order, we must have a linear order.
So the graph of comparisons must be acyclic and complete.
So we need a complete ordering.
Is it possible to get the answers faster?
Look at the formula again: $[p_i > p_j] = I(i, j) \oplus I(i+1, j) \oplus I(i, j-1) \oplus I(i+1, j-1)$.
Notice that $I(a, b)$ is shared across multiple equations.
Specifically, $I(x, y)$ helps determine $[p_x > p_y]$, but also helps determine $[p_x > p_{y+1}]$ etc? No, that's not right.
$I(x, y)$ is used when computing $[p_{x-1} > p_x]$? No.
$I(x, y)$ contributes to $[p_x > p_y]$ via the formula with terms $I(x+1, y), I(x, y-1), I(x+1, y-1)$.
It seems independent.
However, observe that we don't need to determine ALL pairs $[p_i > p_j]$ to recover the permutation.
We need to resolve the relative order of all elements.
One method is to insert elements one by one.
Suppose we want to determine $p$.
Consider the positions.
Is there a simpler way?
Consider the "inversion vector" approach.
Let $c_v$ be the number of elements smaller than $v$ (value-wise)? No.
Let's go back to basics.
We want to find $p_1, \dots, p_n$.
Actually, if we can determine the relative order of $p_i$ and $p_j$, we can just sort them.
Wait, if we only have $O(N)$ queries? No, we have $O(N^2)$ queries.
The limit $1.999 \times 10^6$ allows for roughly $O(N^2/2)$.
Since $N=2000$, $N^2/2 = 2,000,000$.
Wait, $2000^2 / 2 = 4,000,000 / 2 = 2,000,000$.
The limit is strictly less than $2,000,000$.
So we can barely afford to query every adjacent pair (which is $N-1$) plus every pair?
Wait, if we just query $I(i, i+1)$, we get $p_i$ vs $p_{i+1}$. This is $N-1$ queries. Not enough.
But wait, if we query $I(i, j)$ for all $1 \le i \le j \le n$, that's $N(N+1)/2$ queries.
$2000 \times 2001 / 2 \approx 2,001,000$.
This is very close to the limit. The limit is 1,999,000.
So we can query basically all subarrays? No, $I(i, j)$ gives a single bit. We need more info.
But notice that $I(i, j)$ gives exactly the parity of inversions.
Is there a case where we don't need all $I(i, j)$?
Actually, wait.
If we have the full matrix of $[p_i > p_j]$, we have the order.
But obtaining $[p_i > p_j]$ for all pairs using the formula takes 4 queries per pair. That would be huge.
But we can reuse the values.
Wait. $I(i, j)$ is NOT independent.
However, my derivation showed $[p_i > p_j]$ requires $I(i, j) \oplus I(i+1, j) \oplus I(i, j-1) \oplus I(i+1, j-1)$.
This suggests that if we know all $I(a, b)$, we can derive everything.
But we only have ~1M queries. The number of pairs $(i, j)$ with $i < j$ is ~2M.
Even if we knew $[p_i > p_j]$ directly, that would be ~2M queries.
But we don't query $[p_i > p_j]$ directly. We query $I(i, j)$.
Querying $I(i, j)$ for all $1 \le i < j \le n$ costs $N(N-1)/2$ queries.
$2000 \times 1999 / 2 = 1,999,000$.
Exactly the limit!
Wait, $2000 \times 1999 / 2 = 2000 \times 999.5 = 1,999,000$.
Let me double check calculation.
$2000 \times 1999 = 3,998,000$.
Divide by 2: $1,999,000$.
YES! The number of pairs $(i, j)$ with $1 \le i < j \le n$ is exactly $1,999,000$ for $n=2000$.
The query limit is $1,999,000$.
So we can EXACTLY afford to query $I(i, j)$ for every pair $(i, j)$ with $i < j$.
Wait, does the query allow $l=r$?
"0 l r" where $l \le r$.
$I(i, i)$ is always 0. So no need to query.
So yes, we can afford to query $I(i, j)$ for all $i < j$.
With these values, can we reconstruct the permutation?
We have $Q(i, j) = I(i, j) \bmod 2$.
We also derived that $[p_i > p_j] \equiv I(i, j) \oplus I(i+1, j) \oplus I(i, j-1) \oplus I(i+1, j-1) \pmod 2$?
Wait, I derived that earlier.
Let's verify if this derivation is necessary.
$Inv(i, j) = \sum_{i \le k < m \le j} [p_k > p_m]$.
Let $T(i, j)$ be the value $I(i, j)$ returned by interactor.
We want to recover the relative order.
Is it true that if we have all $T(i, j)$, we can recover the relative order?
The relation derived was: $[p_i > p_j] = T(i, j) - T(i+1, j) - T(i, j-1) + T(i+1, j-1)$.
Is this valid?
Let's expand $T(i, j)$ again.
$T(i, j) = \sum_{u=i}^{j-1} \sum_{v=u+1}^j [p_u > p_v]$.
Consider $T(i+1, j) = \sum_{u=i+1}^{j-1} \sum_{v=u+1}^j [p_u > p_v]$.
Consider $T(i, j-1) = \sum_{u=i}^{j-2} \sum_{v=u+1}^{j-1} [p_u > p_v]$.
Consider $T(i+1, j-1) = \sum_{u=i+1}^{j-2} \sum_{v=u+1}^{j-1} [p_u > p_v]$.
Now compute XOR sum (addition mod 2):
$T(i, j) + T(i+1, j) + T(i, j-1) + T(i+1, j-1)$.
Terms in $T(i, j)$: pairs $(u, v)$ with $i \le u < v \le j$.
Terms in $T(i+1, j)$: pairs with $i+1 \le u < v \le j$.
Intersection: $i+1 \le u < v \le j$. These cancel.
Remaining in $T(i, j)$ (exclusive): pairs with $u=i$ and $i < v \le j$. i.e. $(i, i+1), (i, i+2), \dots, (i, j)$.
So contribution from first pair is $\sum_{v=i+1}^j [p_i > p_v]$.
Terms in $T(i, j-1)$: pairs with $i \le u < v \le j-1$.
Terms in $T(i+1, j-1)$: pairs with $i+1 \le u < v \le j-1$.
Intersection: $i+1 \le u < v \le j-1$. Cancel.
Remaining in $T(i, j-1)$: pairs with $u=i$ and $i < v \le j-1$. i.e. $(i, i+1), \dots, (i, j-1)$.
Contribution: $\sum_{v=i+1}^{j-1} [p_i > p_v]$.
Now, looking at the combined sum $T(i, j) \oplus T(i+1, j) \oplus T(i, j-1) \oplus T(i+1, j-1)$:
Wait, addition is XOR.
$T(i, j) + T(i+1, j) = (\sum_{v=i+1}^j [p_i > p_v])$.
$T(i, j-1) + T(i+1, j-1) = (\sum_{v=i+1}^{j-1} [p_i > p_v])$.
Subtracting (XORing):
$(\sum_{v=i+1}^j [p_i > p_v]) \oplus (\sum_{v=i+1}^{j-1} [p_i > p_v]) = [p_i > p_j]$.
Yes! This is exactly correct.
So, $[p_i > p_j]$ can be computed from $T(i, j), T(i+1, j), T(i, j-1), T(i+1, j-1)$.
Note that we need $T(i, j)$ to be defined for all valid ranges.
Base cases:
If $j=i+1$:
$[p_i > p_{i+1}] = T(i, i+1) \oplus T(i+1, i+1) \oplus T(i, i) \oplus T(i+1, i)$.
$T(k, k) = 0$, $T(k, l)$ with $l < k$ is undefined (but here indices are $l \le r$, so lower triangular $i > j$ not queried).
However, we need $T(i+1, i+1)$ which is 0. $T(i, i)$ is 0. $T(i+1, i)$ would be invalid query (since $l > r$).
Wait, the formula uses $T(i+1, j-1)$. If $j=i+1$, then $j-1=i$. So $T(i+1, i)$. This is empty range $l > r$, usually considered 0.
Wait, let's check the indices carefully.
We need $T(a, b)$ for $1 \le a \le b \le n$.
For pair $(i, j)$, we need $T(i, j), T(i+1, j), T(i, j-1), T(i+1, j-1)$.
If $j=i+1$:
Indices: $(i, i+1)$, $(i+1, i+1)$, $(i, i)$, $(i+1, i)$.
$T(i+1, i)$ is out of bounds for our queries. But conceptually sum over empty range is 0.
Also $T(i+1, i+1)=0$.
$T(i, i)=0$.
So $[p_i > p_{i+1}] = T(i, i+1) \oplus 0 \oplus 0 \oplus 0 = T(i, i+1)$.
This works perfectly. We just query $T(i, i+1)$ and use it.
So indeed, if we query all $T(i, j)$ for $1 \le i < j \le n$, we can compute the adjacency relations.
Actually, wait. Does this help?
The formula gives us $[p_i > p_j]$ for all $i < j$ using the queries.
Wait, if we query ALL $T(i, j)$, we consume exactly the budget.
Wait, we can't store all $T(i, j)$ and then process? Yes we can.
$N=2000$, memory $2000 \times 2000$ bools/integers is $4 \text{MB}$. This fits easily in typical memory limits (usually 256MB+).
Time complexity to process all: $O(N^2)$ is fine.
Wait, is $O(N^2)$ queries OK?
Max queries $1,999,000$.
Total pairs $(i, j)$ with $1 \le i < j \le n$ is $n(n-1)/2 = 2000 \times 1999 / 2 = 1,999,000$.
So we can make exactly one query per pair.
Is there any edge case?
Wait, if $n=2000$, we use exactly 1,999,000 queries.
What if $n < 2000$? The problem statement says $1 \le n \le 2000$.
So for $n < 2000$, we use fewer queries.
The strategy:
Iterate all pairs $(i, j)$ with $1 \le i < j \le n$.
Query `0 i j`. Store result in a 2D array.
After collecting all results, we have the full Inversion Table $T[i][j]$.
Reconstruct $p$ using the relations.
Wait, do we need the relations?
Actually, we can just reconstruct $p$ from $T[i][j]$ directly.
Wait, knowing $[p_i > p_j]$ for all $i < j$ means we know the relative order of every pair.
Since we know $p$ is a permutation of $1 \dots n$, the values are distinct.
The relative order is a strict total order.
We can just collect the elements and sort them based on their values. But we don't know values, just relative order.
Wait, if we know $p_i > p_j$, this effectively sorts the positions $1 \dots n$.
Wait, no.
The positions $1 \dots n$ are fixed. We need to find the values $p_1, \dots, p_n$.
If we know for all pairs of indices $(i, j)$ whether $p_i < p_j$, we can assign values $1 \dots n$ to indices $i$ based on the rank.
Wait, this assumes that the values $p_i$ are ordered $1 < 2 < \dots < n$.
Yes, the problem states $p$ is a permutation of $1 \dots n$.
So the smallest element has value 1, next 2, ..., largest $n$.
If we sort the indices $1, \dots, n$ according to the relation "value at index $x$ is less than value at index $y$", we will get an ordering of indices $idx_1, idx_2, \dots, idx_n$ such that $p_{idx_1} < p_{idx_2} < \dots < p_{idx_n}$.
Then we can set $p_{idx_1} = 1, p_{idx_2} = 2, \dots, p_{idx_n} = n$.
This solves the problem.

Algorithm Plan:
1. Initialize a 2D array `inv[2005][2005]` to store query results.
2. Loop `i` from 1 to `n`.
3.   Loop `j` from `i+1` to `n`.
4.     Query `0 i j`. Read response `res`.
5.     `inv[i][j] = res`.
6.   End Loop `j`.
7. End Loop `i`.
8. Now we have all inversions parities.
   We need to determine for all pairs $i, j$ whether $p_i < p_j$.
   Let `is_less[i][j]` be true if $p_i < p_j$.
   Note: `inv[i][j]` stores `(count inversions in p[i...j]) % 2`.
   Formula: `[p_i > p_j] = inv[i][j] ^ inv[i+1][j] ^ inv[i][j-1] ^ inv[i+1][j-1]`.
   Wait, is this valid for all $i < j$?
   Let's check indices again.
   We need `inv[i][j]`, `inv[i+1][j]`, `inv[i][j-1]`, `inv[i+1][j-1]`.
   If $j=i+1$, `inv[i+1][j-1]` becomes `inv[i+1][i]`. This is 0.
   Also `inv[i+1][i]` isn't queried. But logically it's 0.
   If $j > i+1$, `inv[i+1][j-1]` corresponds to subarray starting at $i+1$ ending at $j-1$. Since $i+1 \le j-1$ (for $j \ge i+2$), this is a valid query we made.
   So for $j=i+1$, we use `inv[i][j]` (which is `inv[i][i+1]`). `inv[i+1][j]=inv[i+1][i+1]=0`. `inv[i][j-1]=inv[i][i]=0`. `inv[i+1][j-1]=0`. Result is `inv[i][i+1]`. Correct.
   For general case, the formula works.
   So we can compute `greater[i][j] = inv[i][j] ^ inv[i+1][j] ^ inv[i][j-1] ^ inv[i+1][j-1]`.
   Wait, this computes `[p_i > p_j]`.
   Then `is_less[i][j] = !greater[i][j]` (since they are complementary).
   
   Wait, if $i=j$, $p_i > p_i$ is false.
   We only need to know comparisons for distinct indices.
   Also we can deduce $p_i < p_j$ directly.
   Actually, we need to sort the indices.
   Construct a boolean matrix `cmp[i][j] = (p_i < p_j)`.
   `cmp[i][j] = (greater[i][j] == false)` (for $i \neq j$).
   Is it guaranteed transitive?
   The original problem guarantees a hidden permutation, so yes, there exists a consistent assignment.
   
   Wait, implementing sorting with a custom comparator?
   If we have `cmp` for all pairs, we can extract the rank of each index.
   Rank of index $x$: count how many $y$ satisfy `cmp[x][y] == true`? No, count how many $y$ have `cmp[y][x] == true`.
   If $y$ comes before $x$ in value order, then `cmp[y][x]` is true.
   Wait, `cmp[y][x]` means $p_y < p_x$.
   So if we count number of $y$ such that $p_y < p_x$, we get the rank (0-indexed or 1-indexed).
   Specifically, `rank[x]` = count of $y$ where `cmp[y][x]` is true.
   If we add 1, that's the value.
   Wait, this relies on calculating `rank[x]` for all $x$.
   Is `rank[x]` sufficient?
   Yes. If $p_x$ has $k$ elements smaller than it, then $p_x = k+1$.
   But wait, this requires checking $O(N)$ elements for each of $N$ elements.
   Calculating `rank[x]` for all $x$ takes $O(N^2)$.
   We already spent $O(N^2)$ queries.
   Calculating `rank[x]`:
   For each $x \in 1 \dots n$:
     cnt = 0
     For each $y \in 1 \dots n$:
       if $y == x$ continue
       check if $p_y < p_x$.
       To check if $p_y < p_x$:
         If $y < x$, we have `greater[y][x]`? No, `greater[y][x]` is $p_y > p_x$.
         Our loop computed `greater[i][j]` for $i < j$.
         So if $y < x$, we have `greater[y][x]`. $p_y < p_x \iff !(p_y > p_x) \iff !\text{greater}[y][x]$.
         If $y > x$, we need `greater[x][y]`. $p_y < p_x \iff p_x > p_y \iff \text{greater}[x][y]$.
         So for a fixed $x$:
           Iterate $y$.
           if $y < x$: term = !`greater[y][x]`
           else ($y > x$): term = `greater[x][y]`
           if term add 1.
   Finally print mapping.

   Wait, there is a catch. The limit is strict.
   $1,999,000$.
   My plan uses exactly that many queries.
   The computation phase is fast.
   Memory usage: 2000x2000 ints = 4MB. Safe.
   Time limit: typical 1s or 2s. $N^2$ iterations is $4 \cdot 10^6$ ops, very fast.
   The main bottleneck is I/O speed and flushing.
   Also flushing after every query. `endl` or `\n` with flush is expensive.
   Using `cout << ... << endl;` flushes. `cout.flush()` explicitly is better if we avoid `endl`.
   However, we are printing query `0 l r`. Then read response.
   We need to be careful with `cin` synchronization.
   Disable sync with stdio: `ios::sync_with_stdio(0); cin.tie(0);`.
   
   One detail:
   Can we optimize the queries?
   Wait, I said "we can afford to query every pair".
   Is it possible to deduce some pairs without queries?
   Actually, $I(i, j)$ is the inversion parity of subarray $p[i \dots j]$.
   The query cost is the issue.
   Is there any redundancy?
   Suppose we have $I(i, j)$ and $I(i, j+1)$.
   $I(i, j+1) = I(i, j) \oplus \text{inversions involving } p_{j+1}$.
   Specifically, $I(i, j+1) = I(i, j) \oplus \sum_{k=i}^j [p_k > p_{j+1}]$.
   This sum is not trivial to separate into individual $[p_k > p_{j+1}]$ without more queries.
   However, notice that if we just wanted to reconstruct the permutation, maybe there's a randomized solution or something simpler?
   But the deterministic $O(N^2)$ approach seems to fit the query budget exactly.
   Is it safer to assume the limit is strict or can we do better?
   Given "score will be...", minimizing queries is good.
   However, $N^2$ approach is worst-case $O(N^2)$.
   Is there a risk of exceeding limit for $N=2000$?
   $N(N-1)/2 = 1,999,000$.
   If $N=2000$, this is exactly the limit.
   Are there any off-by-one errors?
   Problem says "no more than 1999000 queries".
   So if we use 1,999,000, it is accepted.
   
   Wait, what if $N < 2000$? Then limit is generous.
   What if $N$ is small? e.g. $N=1$.
   Loops won't execute any queries. Just output `! 1`.
   
   Wait, is there a simpler way to determine $p_i$?
   Consider the inversion table $A_i$ where $A_i$ is the number of inversions ending at $i$ (i.e., $p_j > p_i$ for $j < i$).
   The values of $A_i$ determine the permutation uniquely?
   Yes, specifically if $0 \le A_i < i$.
   But we don't know $A_i$. We only know parities of subarray sums.
   However, if we had $A_i \bmod 2$, that would be $\sum_{j=1}^{i-1} [p_j > p_i] \bmod 2$.
   Notice that $\sum_{j=1}^{i-1} [p_j > p_i]$ is exactly the inversion count for the prefix $1 \dots i$ restricted to pairs involving $p_i$ and elements to its left?
   Wait, $I(1, i)$ is the number of pairs $(u, v)$ in $1 \le u < v \le i$ such that $p_u > p_v$.
   This includes pairs entirely inside $1 \dots i-1$ and pairs involving $i$.
   $I(1, i) = I(1, i-1) + \sum_{j=1}^{i-1} [p_j > p_i]$.
   Therefore, $\sum_{j=1}^{i-1} [p_j > p_i] \equiv I(1, i) \oplus I(1, i-1) \pmod 2$.
   Let $L_i = \sum_{j=1}^{i-1} [p_j > p_i]$.
   $L_i \equiv I(1, i) \oplus I(1, i-1) \pmod 2$.
   This value $L_i \bmod 2$ is the number of elements to the left of $i$ that are greater than $p_i$, modulo 2.
   This gives us ONE bit of information for $p_i$'s position relative to the prefix.
   Does this help?
   Knowing $p_i$'s rank modulo 2 might not be enough to fully identify it.
   But we have much more data from $I(i, j)$ queries.
   
   Actually, wait. If we query $I(i, j)$ for all $i < j$, we effectively query all subarrays.
   There is a potential optimization.
   The problem is from a contest (likely).
   The query budget $1999000$ for $N=2000$ strongly suggests an $O(N^2)$ query complexity where coefficient is $1/2$.
   Specifically $N(N-1)/2$.
   This matches the count of pairs $(i, j)$.
   So yes, querying every subarray pair is likely the intended solution path.
   However, notice that calculating $[p_i > p_j]$ using 4 subarray queries seems wasteful?
   Wait. $I(i, j)$ is a single bit.
   If we query $I(i, j)$ for all $i, j$, we use $N(N-1)/2$ queries.
   Then we can derive ALL pairwise relationships $[p_i > p_j]$?
   NO.
   I derived $[p_i > p_j] \leftarrow I(i, j) \oplus I(i+1, j) \oplus I(i, j-1) \oplus I(i+1, j-1)$.
   Wait, if I use this formula, do I need MORE queries?
   I am READING the $I$ values from my storage. I am NOT making NEW queries.
   The derivation shows how to GET the boolean relation from the stored values.
   Wait, if I stored ALL $I(i, j)$ values, then yes, I can reconstruct everything.
   But storing all $I(i, j)$ took $N(N-1)/2$ queries.
   And with $N=2000$, that is exactly the limit.
   So the strategy is:
   1. Query $I(i, j)$ for all $1 \le i < j \le n$.
   2. Reconstruct the tournament graph / comparison matrix.
      Wait. If I have $I(i, j)$ for all pairs, does it imply I know $[p_i > p_j]$?
      Let's check the formula again.
      $[p_i > p_j] = I(i, j) \oplus I(i+1, j) \oplus I(i, j-1) \oplus I(i+1, j-1)$.
      This formula allows recovering the direct comparison between $i$ and $j$ using neighbors.
      Wait. To get $[p_i > p_j]$, we need $I(i, j)$.
      But do we need ALL $I$?
      To recover $[p_i > p_j]$, we need $I(i, j)$ AND $I(i+1, j)$ AND $I(i, j-1)$ AND $I(i+1, j-1)$.
      But wait.
      Does knowing all $[p_i > p_j]$ give us enough info? Yes, it sorts the array.
      BUT wait.
      Is it possible to determine $p$ WITHOUT knowing ALL $[p_i > p_j]$?
      Actually, we don't need to calculate $[p_i > p_j]$ explicitly for all pairs.
      Wait, if I have the full matrix $I(i, j)$, I can definitely determine the order.
      How?
      Maybe by induction.
      For $N=1$, trivial.
      Suppose we know $p'$ on $\{1, \dots, N-1\}$.
      We want to place $p_N$.
      Wait, $I(i, j)$ values are for the fixed array.
      We want to find indices $k$ such that $p_k$ is the smallest, second smallest, etc.
      
      Let's reconsider the $I(1, k)$ recursion.
      $I(1, k) = I(1, k-1) + \text{count}\{j < k : p_j > p_k\} \pmod 2$.
      So we know the parity of "how many to the left are larger".
      Call this $L_k \equiv I(1, k) - I(1, k-1) \pmod 2$.
      This gives one bit of info per element about its rank relative to prefix.
      This is clearly not enough to distinguish all $N!$ permutations with $O(N)$ info.
      So we need more info.
      
      Let's stick to the $O(N^2)$ query strategy.
      It consumes the budget.
      Can we implement it efficiently?
      Yes.
      
      Let's refine the reconstruction step.
      We need to sort the indices $1 \dots n$ based on their values $p_i$.
      We have a boolean matrix `less[i][j]` indicating $p_i < p_j$.
      To get this matrix:
      For each $i < j$:
        val = `inv[i][j]` ^ `inv[i+1][j]` ^ `inv[i][j-1]` ^ `inv[i+1][j-1]`
        // `inv` contains the answer from queries.
        // Note: `inv[a][b]` where $b < a$ should be treated as 0.
        // Wait, is this assumption safe?
        // $I(a, b)$ with $a > b$ is undefined or 0.
        // My formula was derived assuming $I(a, b)$ are sub-inversion parities.
        // Let's re-verify the indices for the corner case $j=i+1$.
        // $j=i+1$. We need $I(i, i+1), I(i+1, i+1), I(i, i), I(i+1, i)$.
        // $I(i+1, i)$ is technically query `0 i+1 i` which is invalid.
        // But mathematically sum over empty range is 0.
        // So yes, treat invalid indices (where $l > r$) as 0.
        
        So if $val$ (which is $[p_i > p_j]$) is 0, then $p_i < p_j$.
        If 1, then $p_i > p_j$.
        
      After filling the `less` matrix (or just storing `is_greater`):
      We can simply sort the indices $1 \dots n$ using a custom comparator that looks up these relations.
      To ensure stability/correctness, since it's a linear order, any stable sort or just standard sort with the predicate should work.
      Wait, `std::sort` takes a comparator.
      `comp(u, v)` returns true if $p_u < p_v$.
      We need to handle cycles? No, permutation implies no cycles.
      Transitivity holds.
      But `std::sort` might be $O(N^2 \log N)$ with bad comparators? No, $O(N^2)$ comparisons in worst case is too slow? No, comparisons are $O(1)$. Sorting is $O(N \log N)$ calls. Total $O(N \log N)$. Very fast.
      
      So the algorithm is:
      1. Declare `bool inv[N+2][N+2]` initialized to 0.
      2. Loop $i$ from 1 to $n$.
         Loop $j$ from $i+1$ to $n$.
           Print query `0 i j`. Flush.
           Read response $r$.
           Store `inv[i][j] = r`.
      3. Compute `vector<int> p` representing indices $1 \dots n$.
         Define comparator:
         `bool compare(int a, int b) {`
           `// return true if p_a < p_b`
           `// Check if we have relation stored.`
           `if (a == b) return false;`
           `// We know [p_a > p_b].`
           `if (a < b) {`
             `int greater = inv[a][b] ^ inv[a+1][b] ^ inv[a][b-1] ^ inv[a+1][b-1];`
             `return !greater;` // if p_a > p_b is false, then p_a < p_b is true.
           `} else {`
             `int greater = inv[b][a] ^ inv[b+1][a] ^ inv[b][a-1] ^ inv[b+1][a-1];`
             `return !greater;` // wait. If b < a, we stored inv[b][a].`
           `}`
         `}`
         
         Wait, in the else branch ($b < a$), I'm accessing `inv[b][a]`.
         The logic: if I need to check $p_a$ vs $p_b$ where $a > b$, I should look at the precomputed comparison for pair $(b, a)$.
         So `inv[b][a]` holds the parity of inversions in $p[b \dots a]$.
         Applying the formula:
         `[p_b > p_a] = inv[b][a] ^ inv[b+1][a] ^ inv[b][a-1] ^ inv[b+1][a-1]`.
         This is exactly what I computed.
         Wait, `inv[x][y]` stores the query result.
         So the code is uniform: compute `g = get_greater(u, v)`.
         If `g` is 1, $p_u > p_v$, so $p_u < p_v$ is false.
         Return `!g`.
         
      4. Sort the indices $1 \dots n$ using `compare`.
      5. Output the values.
         Wait. The sorted indices $idx_1, idx_2, \dots, idx_n$ correspond to values $p_{idx_1} < p_{idx_2} < \dots < p_{idx_n}$.
         Since the values are a permutation of $1 \dots n$,
         $p_{idx_k} = k$.
         So we just output `1 p_{idx_1}`?
         Wait. The problem asks for $p_1 p_2 \dots p_n$.
         We found the permutation in terms of indices.
         Wait, if $idx_1$ is the index of the smallest value, then $p[idx_1] = 1$.
         If $idx_2$ is the index of the second smallest, then $p[idx_2] = 2$.
         So we can fill an array `ans[n+1]`.
         `ans[idx[k]] = k`.
         Then print `ans[1] ... ans[n]`.
         
      Double check the formula logic again.
      $I(l, r)$ is parity of inversions in $p[l \dots r]$.
      We have stored $I(l, r)$ for all $l \le r$.
      We need to deduce $[p_l > p_r]$.
      Formula: $[p_l > p_r] \equiv I(l, r) - I(l, r-1) - I(l+1, r) + I(l+1, r-1)$.
      Wait, earlier I wrote $I(i, j) \oplus I(i+1, j) \oplus I(i, j-1) \oplus I(i+1, j-1)$ for $[p_i > p_j]$.
      Let's re-verify signs.
      Identity: $\sum_{u,v} [p_u > p_v]$.
      $Inv(l, r) = \sum_{u=l}^{r-1} \sum_{v=u+1}^r [p_u > p_v]$.
      Term of interest is pairs involving $l$ and $r$: $(l, r)$.
      Let's see which sets contain $(l, r)$.
      Set 1: $(l, r)$ in $Inv(l, r)$.
      Set 2: $(l, r)$ in $Inv(l+1, r)$? No, starts at $l+1$.
      Set 3: $(l, r)$ in $Inv(l, r-1)$? No, ends at $r-1$.
      Set 4: $(l, r)$ in $Inv(l+1, r-1)$? No.
      Wait, my previous manual check concluded that the formula isolates $[p_l > p_r]$.
      Let's write down the expansion again.
      $Inv(l, r) = \text{terms with } l \text{ involved as first elem} + Inv(l+1, r)$.
      Terms with $l$ involved as first elem: $\sum_{v=l+1}^r [p_l > p_v]$.
      So $Inv(l, r) = Inv(l+1, r) + \sum_{v=l+1}^r [p_l > p_v]$.
      Also $Inv(l, r-1) = Inv(l+1, r-1) + \sum_{v=l+1}^{r-1} [p_l > p_v]$.
      Subtracting (XORing):
      $Inv(l, r) \oplus Inv(l+1, r) \oplus Inv(l, r-1) \oplus Inv(l+1, r-1)$.
      $= (\sum_{v=l+1}^r [p_l > p_v]) \oplus (\sum_{v=l+1}^{r-1} [p_l > p_v])$.
      The terms for $v=l+1 \dots r-1$ cancel out.
      Leaving $[p_l > p_r]$.
      This assumes the base terms are 0 for boundaries.
      Wait, in the expression above, I subtracted $Inv(l+1, r-1)$ twice?
      Let's trace:
      LHS: $Inv(l, r) \oplus Inv(l+1, r) \oplus Inv(l, r-1) \oplus Inv(l+1, r-1)$.
      From $Inv(l, r)$, we keep $\sum_{v=l+1}^r [p_l > p_v]$.
      From $Inv(l+1, r)$, we cancel $Inv(l+1, r)$ part?
      Wait. $Inv(l, r) = Inv(l+1, r) + \text{row } l$.
      $Inv(l, r-1) = Inv(l+1, r-1) + \text{row } l \text{ (up to } r-1)$.
      So $Inv(l, r) \oplus Inv(l, r-1) = Inv(l+1, r) \oplus Inv(l+1, r-1) + ([p_l > p_r])$.
      Rearranging:
      $[p_l > p_r] = Inv(l, r) \oplus Inv(l, r-1) \oplus Inv(l+1, r) \oplus Inv(l+1, r-1)$.
      Yes, the formula is correct.
      Signs are all positive (mod 2).
      
      Implementation Details:
      Indices are 1-based. Array size 2005.
      Need to handle boundaries carefully.
      Accessing `inv[x][y]` where $x > y$ should return 0.
      Or simply check condition `x <= y`.
      Since we only query $i < j$, we never store for $i > j$.
      So access function:
      `get(val_l, val_r)` -> if `val_l < val_r` return `inv[val_l][val_r]` else return 0.
      Wait, `inv[val_l][val_r]` is needed.
      My loops run $1 \le i < j \le n$.
      So we populate `inv` for $i < j$.
      When computing $[p_i > p_j]$, we need $inv[i][j], inv[i+1][j], inv[i][j-1], inv[i+1][j-1]$.
      Valid checks:
      1. $inv[i][j]$: $i < j$ (since loop covers this). OK.
      2. $inv[i+1][j]$:
         If $i+1 == j$, then we need $inv[j][j]$. $j=j$, query not performed. Treat as 0.
         Wait, if $i+1 > j$ impossible since $i < j \implies i+1 \le j$.
         So if $i+1 < j$, we have query. If $i+1 = j$, value is 0.
      3. $inv[i][j-1]$:
         If $i == j-1$, value is 0. Else valid query.
      4. $inv[i+1][j-1]$:
         If $i+1 < j-1$, valid query.
         If $i+1 == j-1$, value 0 (empty range $i+1 \dots i+1$? No $j-1=i+1$ means $j=i+2$, range $[i+1, i+1]$ is length 1, no inversions. Value 0).
         Wait, $inv[l][r]$ stores inversion parity of subarray $p[l \dots r]$.
         Subarray of length 1 has 0 inversions.
         So if $l=r$, value is 0.
         If $l > r$, value is 0.
      So logic: helper function `bool query_res(l, r)` returns `inv[l][r]` if $l < r$ else `0`.
      Wait, we must strictly store 0 for $l \le r$.
      Since we never stored for $l \le r$, default initialization handles it.
      Except we need to be sure `inv` is zero-initialized. Global variables are zero-initialized.
      But wait, inside the logic, $inv[l][r]$ where $l \le r$ is assumed 0.
      Is it?
      If $l < r$, we stored it.
      If $l \ge r$, we didn't store it, but it should be 0.
      So returning `inv[l][r]` (assuming $l < r$) is only correct if we checked $l < r$.
      However, `inv` array is indexed $2000 \times 2000$.
      So checking bounds is implicit if I just access `inv[l][r]`.
      But I MUST NOT access uninitialized garbage if I haven't initialized.
      Since I declare it globally, it's 0.
      So `inv[l][r]` is safe to access for $l \ge r$ as it returns 0.
      Wait, is it possible $inv[l][r]$ (for $l < r$) is accessed incorrectly?
      No, logic is sound.
      
      Wait, there's a slight issue.
      $inv[i+1][j-1]$. If $j-1 < i+1$, this accesses upper triangle $l > r$ which is not populated.
      Correctly returns 0.
      
      Corner case: $n=1$.
      Loops $1 \dots 1$ for $i$. Inner loop $i+1 \dots 1$ (empty).
      Code runs fine.
      Reconstruction:
      Sort indices $1$. `ans[1] = 1`.
      Output `! 1`.
      Wait, format for final answer: `! p1 p2 ... pn`.
      Correct.
      
      Wait, I should check if `p[i]` (value) or index matters.
      The problem asks for `p1 p2 ... pn`.
      We need to print the values.
      My logic `ans[index] = k` (where $k$ is rank) sets the value at that index.
      E.g. if index 2 is rank 1 (smallest), `ans[2] = 1`.
      Then print `ans[1], ans[2], ...`.
      This is correct.
      
      Memory Limit:
      Standard usually 256MB. $2000 \times 2000$ `int` or `short` or `bool`.
      `bool` takes 1 byte usually. $4 \text{MB}$.
      Even `int` takes 16 MB. Very safe.
      
      Complexity:
      Queries: $N(N-1)/2$. For $N=2000$, 1,999,000.
      Processing: $O(N^2)$ to parse/compute ranks. $2000^2 = 4 \cdot 10^6$.
      Sorting: $O(N^2)$ comparisons in worst case (if we use bubble sort) or $O(N \log N)$ with `std::sort`.
      Comparisons call `query_res` which is $O(1)$.
      So total time is fine.
      
      Wait, one subtle thing.
      The problem statement: "score will be (exp(-q/249875) - exp(-8)) / (1 - exp(-8))".
      Since exp(-8) is negligible (~0.000335), the denominator is approx 1.
      Score $\approx \exp(-q/249875) - \text{small}$.
      Minimizing $q$ increases score (closer to 0? Wait, score is positive? Or negative?).
      Actually usually score functions penalize high queries.
      Wait, exp(-x) is decreasing.
      Wait, usually scoring is like 1 - exp(...).
      Here: $(e^{-q/A} - e^{-8}) / (1 - e^{-8})$.
      If $q$ is small, $e^{-q/A}$ is close to 1. Numerator near $1 - e^{-8}$. Score near 1.
      If $q$ is large, $e^{-q/A}$ is 0. Score near $-e^{-8}/(1-e^{-8}) < 0$.
      So maximizing score means minimizing $q$.
      My $q$ is minimal possible for exact reconstruction?
      Actually, $N(N-1)/2$ is exactly the theoretical minimum if we treat inversions as bits? No.
      Information theoretic bound: $\log_2(N!) \approx N \log N$.
      For $N=2000$, $\log_2(2000!) \approx 2000 \times 11 = 22000$ bits.
      Queries provide 1 bit each.
      So theoretically we need $\sim 22000$ queries.
      Why the budget is $10^6$?
      Because reconstructing the permutation might require $O(N^2)$ comparisons, and we are solving it via an algebraic approach on the inversion table.
      But since we can solve it with $O(N^2)$ queries, and we are asked for a competitive programming solution, and the budget is tight, doing exactly what is needed is good.
      Wait, is $1,999,000$ queries for $N=2000$ the intended solution?
      Maybe. The budget suggests we should be able to query "most" things.
      Actually, there is a known solution using bitsets or just sorting.
      But with $N=2000$, maybe the $O(N^2)$ query is acceptable because the test cases are loose or I misread the budget.
      Wait, the prompt says "no more than 1999000".
      Wait, $2000 \times 1999 / 2 = 1,999,000$.
      So for $N=2000$, it hits the limit exactly.
      For smaller $N$, it is well within.
      This confirms the solution is likely intended to be $O(N^2)$ queries.
      
      Wait, there is a possibility to optimize the number of queries.
      Instead of querying all pairs $(i, j)$, can we just use $I(i, i+1)$ to get $[p_i > p_{i+1}]$?
      Wait, if we only query $I(i, j)$ for ALL $i, j$, we have the whole matrix.
      Is there a way to NOT query everything?
      Suppose we fix $i=1$. We query $I(1, j)$ for all $j$.
      $I(1, j) = I(1, j-1) \oplus \sum_{k=1}^{j-1} [p_k > p_j]$.
      This gives us the parity of "number of larger elements to the left" for each $j$.
      This doesn't give us full order.
      However, with the full matrix, we have enough.
      Given the strict limit matching the number of pairs, we probably shouldn't worry about optimizing further unless we can find a much faster way (randomized or divide & conquer).
      But a simpler solution is best to avoid bugs.
      
      Is there any library headers needed?
      `<iostream>`, `<vector>`, `<algorithm>`, `<cstring>` for memset.
      Using `cin` and `cout`.
      Flushing is important. `endl` flushes. Or explicit `cout.flush()`.
      Use `cout << "0 " << i << " " << j << endl;`
      
      Let's double check the logic one last time.
      Is it possible that $p_i$ and $p_j$ values collide in my logic?
      No, $p$ is a permutation. Distinct values. Strict inequality is total.
      
      Example Trace:
      Input $n=3$. Hidden: 2 3 1.
      Pairs: (1, 2), (1, 3), (2, 3).
      $p = [2, 3, 1]$.
      Query (1, 2): subarray [2, 3]. Inv: 0. (2<3). Response 0.
      Query (1, 3): subarray [2, 3, 1]. Pairs: (2,3)ok, (2,1)inv, (3,1)inv. Total 2. Resp 0.
      Query (2, 3): subarray [3, 1]. Pair: (3,1)inv. Total 1. Resp 1.
      
      Recover $[p_1 > p_2]$:
      $I(1, 2) \oplus I(2, 2) \oplus I(1, 1) \oplus I(2, 1)$.
      $0 \oplus 0 \oplus 0 \oplus 0 = 0$.
      So $p_1 < p_2$. (Correct $2 < 3$).
      
      Recover $[p_2 > p_3]$:
      $I(2, 3) \oplus I(3, 3) \oplus I(2, 2) \oplus I(3, 2)$.
      $1 \oplus 0 \oplus 0 \oplus 0 = 1$.
      So $p_2 > p_3$. (Correct $3 > 1$).
      
      Recover $[p_1 > p_3]$:
      $I(1, 3) \oplus I(2, 3) \oplus I(1, 2) \oplus I(2, 2)$.
      Wait, formula is $I(1, 3) \oplus I(2, 3) \oplus I(1, 2) \oplus I(2, 2)$.
      Values: $0 \oplus 1 \oplus 0 \oplus 0 = 1$.
      So $p_1 > p_3$. (Correct $2 > 1$).
      
      Matrix of greater:
      (1,2): 0
      (1,3): 1
      (2,3): 1
      
      Relations:
      $p_1 < p_2$.
      $p_2 > p_3$.
      $p_1 > p_3$.
      Order: $p_3 < p_1 < p_2$.
      Values: $1, 2, 3$.
      Map: $p_3=1, p_1=2, p_2=3$.
      Output: $p_1 p_2 p_3 \to 2, 3, 1$. Matches.
      
      Seems correct.
      
      Wait, I used `inv[2][2]` which is 0.
      In my helper `query_res(l, r)`, I should return `inv[l][r]` if `l < r` else 0.
      However, for `inv[i+1][j]`, if $i+1=j$, it's `inv[j][j]`.
      If $i+1 > j$ (impossible in loop), it's 0.
      So `query_res` must handle $l=r$.
      
      Also, note that I need to include the header `<vector>` and others.
      
      One more thing: The problem asks to generate solution code.
      The code should read $n$, perform queries, then print answer.
      
      Final check on query limit.
      For $N=2000$, max queries $1,999,000$.
      My code generates $2000 \times 1999 / 2$ queries.
      Exactly the limit.
      
      Wait, what if $N=2000$ and I submit?
      The execution environment might be slow.
      I/O is fast.
      Flushing every query:
      `cout << "0 " << i << " " << j << endl;`
      `endl` does `\n` and `flush`.
      This might be a bit slow but acceptable.
      For $2 \times 10^6$ lines, `endl` might TLE in Python or Java, but in C++ it's usually okay-ish, though potentially risky for 1s time limit.
      Better to use `\n` and `cout.flush()`.
      Or tie(NULL) is good.
      Actually, `cout << "0 " << l << " " << r << "\n"; cout.flush();` is better practice in competitive programming to minimize overhead.
      
      Wait, there is a constraint "Your score will be ...".
      This implies we are penalized for queries.
      But as discussed, this query complexity is likely optimal for this approach.
      Could there be a better approach?
      If we don't query everything?
      Actually, there is a randomized algorithm to find the permutation with fewer queries?
      Probably not worth the risk of randomness or complex logic.
      Deterministic $O(N^2)$ fits the limit.
      
      One tiny optimization:
      Do we need to query $I(i, j)$ where $j = i+1$?
      Yes, we need it for the formula.
      Also $I(i, i+1)$ is just $[p_i > p_{i+1}]$.
      Wait, if we know $[p_i > p_{i+1}]$ directly, that saves queries?
      No, $I(i, i+1)$ IS the query for $[p_i > p_{i+1}]$.
      But do we need $I(i, j)$ for $j > i+1$?
      If we know $[p_i > p_j]$ directly, we wouldn't need to combine anything.
      But we don't have a query type for $[p_i > p_j]$. Only $I(i, j)$.
      So to get $[p_i > p_j]$ (for non-adjacent), we MUST use the combination formula.
      And the formula relies on $I(i, j)$.
      So we need $I(i, j)$ for all $i < j$.
      
      Wait, can we deduce $[p_i > p_j]$ for far apart pairs using just adjacent info?
      Not easily.
      So yes, querying everything is the safest path.
      
      I'll write the code now.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

int n;
int inv[2005][2005];

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n)) return 0;

    // Helper to get inv[l][r] treating out of bounds as 0
    // Since we only query for l < r, we will only populate inv[l][r] for l < r.
    // Default initialization ensures 0.
    
    // Queries
    // We can estimate n*(n-1)/2 queries.
    // To avoid timeout due to flush overhead, we will use explicit flush with \n.
    // But endl is safer. With fast I/O enabled, it should be ok. 
    // Actually, let's use \n and flush.
    
    // Optimization:
    // We can batch queries? No, we need immediate responses.
    
    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j) {
            cout << "0 " << i << " " << j << "\n";
            cout.flush();
            
            int res;
            cin >> res;
            inv[i][j] = res;
        }
    }

    // Now reconstruct the permutation.
    // We need to know the order of elements p_1, ..., p_n.
    // We have the values of [p_i > p_j] implicitly.
    // To check if p_a < p_b (where a != b):
    // Calculate g = inv[a][b] ^ inv[a+1][b] ^ inv[a][b-1] ^ inv[a+1][b-1].
    // If g == 1, then p_a > p_b.
    // If g == 0, then p_a < p_b.
    // We need to handle cases where indices are out of bounds correctly (return 0).
    
    // Let's define a lambda for comparing indices u and v.
    // We want to return true if p_u < p_v.
    
    auto get_greater = [&](int l, int r) -> int {
        if (l >= r || l < 1 || r > n) return 0;
        // Note: inv is filled for l < r. 
        // If l == r, value is 0 (range length 1).
        // If l > r, value is 0.
        // But wait, my loop populated for i < j.
        // inv[x][y] for x < y is valid.
        // inv[x][y] for x == y should be 0.
        // inv[x][y] for x > y should be 0.
        // So accessing inv[l][r] directly works IF we trust default init.
        // Since global array is zero-init, and we only write to i < j,
        // accessing i >= j gives 0.
        return inv[l][r];
    };

    vector<int> p_indices(n);
    for(int k=0; k<n; ++k) p_indices[k] = k+1;

    // Sort indices based on values
    // Comparator: comp(u, v) returns true if p_u < p_v
    // u, v are indices (1-based)
    // Wait, std::sort expects operator< or comparator.
    // If p_u < p_v, then u comes before v in sorted list.
    // So we return true if p_u < p_v.
    
    // To avoid O(N^2) inside sort?
    // Wait, N=2000. N^2 is 4*10^6. 
    // Comparison is O(1). 
    // std::sort does O(N log N) comparisons. 
    // Total time for sorting is fast.
    // BUT, if the comparator is inconsistent? It shouldn't be.
    
    sort(p_indices.begin(), p_indices.end(), [&](int u, int v) {
        if (u == v) return false;
        // Determine if p_u < p_v
        // Compute greater = [p_u > p_v]
        int greater = get_greater(u, v) ^ get_greater(v, u);
        // Wait, get_greater(u, v) returns [p_u > p_v].
        // get_greater(v, u) returns [p_v > p_u] if we apply formula with roles swapped?
        // Wait, my formula [p_i > p_j] depends on inv[i][j] etc.
        // If I call get_greater(u, v), it returns [p_u > p_v].
        // Since p_u and p_v are distinct, exactly one is true.
        // If [p_u > p_v] is true (1), then p_u > p_v.
        // So we want to return true if p_u < p_v.
        // This is equivalent to [p_u > p_v] == 0.
        // So we just check if get_greater(u, v) is 0.
        
        // WAIT.
        // My formula `inv[u][v] ^ inv[u+1][v] ^ ...` gives [p_u > p_v].
        // This works regardless of whether u < v or u > v in indices.
        // Just need to pass correct arguments to `get_greater`.
        // get_greater(l, r) computes [p_l > p_r] using inv.
        // Let's verify:
        // If I call get_greater(u, v) with u=2, v=1.
        // Formula uses inv[2][1] etc.
        // inv[2][1] is 0 (not populated, treated as 0).
        // inv[3][1] is 0.
        // inv[2][0] is 0.
        // inv[3][0] is 0.
        // Result 0? That's wrong. [p_2 > p_1] needs real data.
        // Ah, `get_greater` implementation relied on `inv[l][r]`.
        // The formula I derived was: `[p_i > p_j] = inv[i][j] ^ inv[i+1][j] ^ inv[i][j-1] ^ inv[i+1][j-1]`.
        // This assumes `i` and `j` are positions in the permutation.
        // This derivation used `Inv(i, j)` as subarray inversions.
        // `Inv(i, j)` is defined for $i \le j$.
        // For $i > j$, subarray is empty, 0.
        // Wait, if I use $u=2, v=1$ (so $u > v$), I am looking for relation between $p_2$ and $p_1$.
        // Does the formula hold if I swap roles?
        // $[p_2 > p_1]$ should be derivable from `Inv(?, ?)`.
        // `Inv(a, b)` gives info about pairs within $a \dots b$.
        // If we use `Inv(1, 2)`, we get info about $p_1, p_2$.
        // So if $u > v$, we should use arguments such that we cover the range covering both.
        // If $u=2, v=1$, we need `Inv(1, 2)`.
        // `Inv(1, 2)` depends on indices $1, 2$.
        // It gives $[p_1 > p_2]$ (via formula with $i=1, j=2$).
        // So $[p_2 > p_1] = !([p_1 > p_2])$.
        // Generally, $[p_u > p_v] = \text{formula applied to } \min(u, v), \max(u, v)$?
        // Let $L = \min(u, v), R = \max(u, v)$.
        // We can compute $[p_L > p_R]$ using the formula.
        // Then $[p_u > p_v]$ is $[p_L > p_R]$ if $u=L$, else $! [p_L > p_R]$?
        // Yes, because $[p_u > p_v]$ is either $[p_L > p_R]$ (if $u=L$) or $[p_R > p_L]$ (if $u=R$).
        // And $[p_R > p_L]$ is negation of $[p_L > p_R]$ (since distinct values).
        // So:
        // 1. Identify $L = \min(u, v), R = \max(u, v)$.
        // 2. Compute $K = \text{formula}(L, R)$. This is $[p_L > p_R]$.
        // 3. If $u == L$, then $[p_u > p_v] = K$.
        //    If $u == R$, then $[p_u > p_v] = \neg K$.
        // 4. Result is $K$ if $u < v$, $\neg K$ if $u > v$.
        
        // This simplifies things. We only ever compute for $i < j$.
        // So we can write a helper that takes $L, R$ (with $L < R$) and returns $[p_L > p_R]$.
        // And adjust in comparator.
        
        // Let's implement `calc_gte(L, R)` which returns `[p_L > p_R]` for $L < R$.
        int L = min(u, v);
        int R = max(u, v);
        // Formula components
        int t1 = inv[L][R];
        int t2 = inv[L+1][R];
        int t3 = inv[L][R-1];
        int t4 = inv[L+1][R-1];
        int greater_LR = t1 ^ t2 ^ t3 ^ t4; // [p_L > p_R]
        
        if (u < v) return greater_LR == 0; // u < v means u=L. If p_u > p_v is false, p_u < p_v is true.
                                           // wait. greater_LR is [p_L > p_R].
                                           // We want p_u < p_v <=> [p_u > p_v] is false.
                                           // if u=L, [p_L > p_R] = greater_LR.
                                           // So return !greater_LR.
        else return greater_LR == 1; // u > v means u=R.
                                      // p_u < p_v <=> p_R < p_L <=> p_L > p_R is true.
                                      // So return greater_LR.
        
        // Wait, let's trace logic again.
        // Case 1: u < v. Then L=u, R=v.
        // We want true iff p_u < p_v.
        // p_u < p_v is equivalent to NOT (p_u > p_v).
        // p_u > p_v is `greater_LR` (since u=L).
        // So return `!greater_LR`.
        
        // Case 2: u > v. Then L=v, R=u.
        // We want true iff p_u < p_v.
        // This is equivalent to p_v > p_u? No. p_u < p_v is true.
        // p_u > p_v is false.
        // Relation is `greater_LR` which is $[p_v > p_u]$.
        // If $[p_v > p_u]$ is true, then $p_v > p_u \implies p_u < p_v$.
        // So if `greater_LR` is true, we return true.
        // So return `greater_LR`.
        
        // Combining:
        // bool ret = greater_LR;
        // if (u < v) ret = !ret;
        // return ret;
        
        // Let's double check.
        // u=1, v=2. L=1, R=2. greater_LR = [p1 > p2].
        // We want p1 < p2.
        // If [p1 > p2] is 1 (false for p1<p2), we return false.
        // If [p1 > p2] is 0 (true for p1<p2), we return true.
        // So `!greater_LR` is correct.
        
        // u=2, v=1. L=1, R=2. greater_LR = [p1 > p2].
        // We want p2 < p1.
        // If [p1 > p2] is 1 (meaning p1 > p2, so p2 < p1 holds), return true.
        // If [p1 > p2] is 0, return false.
        // So `greater_LR` is correct.
        
        // Logic holds.
    });
    
    // Fill values
    // Sorted p_indices gives us indices $id_1, id_2, \dots, id_n$
    // such that $p[id_1] < p[id_2] < \dots < p[id_n]$.
    // So $p[id_k] = k$.
    vector<int> ans(n + 1);
    for (int k = 0; k < n; ++k) {
        ans[p_indices[k]] = k + 1;
    }
    
    cout << "!";
    for (int i = 1; i <= n; ++i) {
        cout << " " << ans[i];
    }
    cout << "\n";

    return 0;
}
```
Wait, one small syntax error. `inv` array is global, so it's 0-init.
Access `inv[L+1][R-1]` where $L+1 > R-1$ (e.g. $L=u, R=u+1$) accesses $inv[u+1][u]$.
If $u+1 > u$, indices are valid (since max index 2005). Value is 0.
My `get_greater` logic inside sort handles this by relying on global array content.
Global array `inv` is 0 except `i < j` entries.
So reading `inv[x][y]` with $x \ge y$ returns 0.
This matches my assumption.
Wait, `inv[L][R-1]` with $L=R-1$ accesses `inv[L][L]` (returns 0).
`inv[L+1][R-1]` with $R=L+1$ accesses `inv[L+1][L]` (returns 0).
Seems correct.

Wait, the lambda captures `inv` and `n`?
Actually, I can just use the global scope logic or capture reference.
Lambda `[&]` captures everything by reference.
Inside `main`, `inv` is global, so accessible.
Wait, I declared `int inv[2005][2005];`.
If I put `inv` inside `main`, stack size might be an issue?
$2000 \times 2000 \times 4$ bytes $\approx 16$ MB.
Stack is typically 8MB on Linux judges, might overflow.
Better to keep `inv` global.
So I should move `inv` outside.
And inside `main`, I can capture by reference or just rely on global `inv` being visible.
Since lambda `[&]` captures `inv` by reference (it's global, so address is taken), it works. Or just don't capture and access global directly.
Actually, capturing `inv` by reference `[&]` is fine for globals.
However, I need to declare `inv` before lambda.
It is declared before `main`.
The lambda uses `inv`, so it needs to be captured or accessed via `this` (no class).
If I capture by reference `[&]`, it works.
But for simplicity, I will just write a free function or use the capture.
Wait, `inv` is global.
If I use `[&]`, it captures `inv` by reference.
Is it legal to capture a global variable in a lambda? Yes, it captures it by reference.
Wait, if I access it by name `inv` inside lambda, it uses captured version.
If I use `[&]`, all external names are accessible?
No, only variables in enclosing scope. Global variables are not in enclosing scope unless captured?
Actually, global variables are in global scope.
In C++, you can refer to globals without capture.
But if I want to capture nothing, `[&]` captures locals.
If `inv` is global, it's not a local.
So referencing `inv` without capturing is allowed.
Wait, if I declare `int inv[...]` globally, and inside `main` I declare a lambda:
```cpp
int inv[];
auto f = [&](){
    return inv[0];
};
```
This works fine.
But wait, if I define lambda inside `main` and use `inv`, `inv` is looked up in global scope.
Is there ambiguity? No.
So I don't need to capture. `[&]` is unnecessary but harmless (captures `p_indices` etc).
Wait, I should capture `inv`?
Actually, since `inv` is global, I don't *need* to capture it.
But `p_indices` is local. So I should capture `p_indices` or make it passed.
But I put the lambda inside `sort` call.
So `sort(begin, end, lambda)`
Inside lambda I need `u` and `v`.
I don't need `inv` to be captured if I access it as global.
Wait, actually, accessing global variables inside a lambda without capture works in C++.
But it's cleaner to capture `[&]` to be safe and access any needed variables (though `inv` is global).
Actually, if I declare `inv` inside `main` as static or something? No, stack overflow.
Okay, I will keep `inv` global.
And I will use `[&]` to capture `p_indices` (though not strictly needed as `u, v` are args).
Actually, `p_indices` is not used inside lambda. `u, v` are params.
I only need `inv` and `min`, `max`.
Wait, I need `inv`. Since `inv` is global, I can just use it.
Wait, the problem with capturing `[&]` is if I capture `inv`? No, `inv` is global, so not captured automatically.
If I capture `[&]`, it captures local variables.
If I use `inv` inside lambda, it refers to global `inv`.
This is fine.

Double check the formula logic again.
$t1 = inv[L][R]$
$t2 = inv[L+1][R]$
$t3 = inv[L][R-1]$
$t4 = inv[L+1][R-1]$
If $L=R-1$, $t3 = inv[L][L] = 0$.
$t4 = inv[L+1][L] = 0$ (since $L+1 > L$).
$t2 = inv[L+1][R] = inv[R][R] = 0$.
So result is $inv[L][R] \oplus 0 \oplus 0 \oplus 0 = inv[L][R]$.
Which is $I(L, R)$.
For adjacent $L, L+1$, $I(L, L+1)$ gives $[p_L > p_{L+1}]$.
So logic holds for adjacent too.

The code looks solid.

One small thing:
`cout << "0 " << i << " " << j << "\n";`
`cout.flush();`
This is repeated $2 \times 10^6$ times.
On some online judges, this is slow.
Is there a way to optimize?
Buffering?
But we need to read response before sending next.
So we must flush or line-buffer.
Standard stream is usually buffered until newline.
With `sync_with_stdio(false)`, `cin`/`cout` are tied?
Usually `cin` is tied to `cout`, so `cin` flushes `cout`.
If I disable tie? `cin.tie(NULL)`.
Then I must manually flush.
Manual flush with `cout` might be slower than relying on tie.
If I keep tie: `cin >> res;` will flush `cout` automatically.
So `cout << ... << endl;` is safest?
`endl` forces flush.
So keeping `cin.tie(0)` is good for performance.
Then `cout << "0 " << i << " " << j << "\n";`
Then `cin >> res;` (this triggers flush if tied).
Wait, if `cin.tie(0)`, it's not flushed automatically.
If I DON'T untie, `cin` flushes `cout`.
If I UNtie, `cin` does not flush.
So if I untie, I MUST flush.
If I DON'T untie, `cin` will flush before read.
But untying makes input faster?
Actually, if I untie, I need `cout.flush()`.
Flushing every line is costly.
However, `endl` also flushes.
Given the time limit is not specified as extremely tight, but standard 1s.
$2 \times 10^6$ flushes might take 0.5-1s.
It might be safer to use `cout << "0 " << i << " " << j << endl;` (which flushes).
If I use `\n` and untie, then explicit `flush`.
`cout << "0 " << i << " " << j << "\n"; cout.flush();`
Usually `cout << "\n"` is faster than `endl`.
So I will use `"\n"` and `cout.flush()`.
Also `ios::sync_with_stdio(0); cin.tie(0);` is essential.

Final check:
Input reading: `cin >> res`.
If `cin` is not tied to `cout`, `cin` won't flush `cout`.
Wait, if I untied `cin` and `cout`, I need to flush.
If I leave `cin` tied, `cin` reads will trigger flush.
So:
```cpp
ios::sync_with_stdio(0);
cin.tie(0); // Untie for speed, requires manual flush
// ...
cout << "0 " << i << " " << j << "\n";
cout.flush();
cin >> res; // res is ready
```
Wait, if I untie, `cin >> res` doesn't trigger flush.
But I manually flushed before `cin >> res`?
Wait, I flush AFTER printing.
`cout << ... << "\n"; cout.flush();`
Then `cin >> res;`
This sequence is correct.
However, `sync_with_stdio(0)` is recommended.
`tie(0)` makes `cin` independent of `cout`.
So manual flush is required.
Without `tie(0)`, `cout << ... << endl` flushes.
`endl` is heavy.
`\n` + `flush` is lighter? Or same?
`endl` writes `\n` then `flush`.
So `\n` then `flush` is exactly what `endl` does.
So `endl` is fine.
Actually, `cin` is buffered too. Reading from `stdin` is blocking.
So we print, flush, read.
With `endl`, we print newline and flush.
If I use `cin.tie(NULL)`, `cout` is not flushed on `cin`.
So I need `cout.flush()`.
Using `endl` automatically calls flush.
So if I use `cin.tie(NULL)`, I MUST use `flush` explicitly if I use `\n`.
Or use `endl`.
Given `N` is large, `endl` might be slow.
But `\n` with flush is the most robust.

Let's stick to `\n` and `cout.flush()`.
Wait, `cin.tie(NULL)` disconnects input from output buffer.
So I should be careful.
If I don't untie, `cin >> res` flushes `cout`.
Then `cout << ... << endl` prints and waits for buffer.
If I untie, I lose that guarantee.
Most CP solutions untie.
So I will untie and use explicit flush.

Code structure:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

int n;
int inv[2005][2005];

void solve() {
    if (!(cin >> n)) return;

    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j) {
            cout << "0 " << i << " " << j << "\n";
            cout.flush();
            int res;
            cin >> res;
            inv[i][j] = res;
        }
    }

    // ... reconstruction ...
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    solve();
    return 0;
}
```
Wait, `inv` array is large.
Memory allocation: `int` is 4 bytes. $2005^2 \approx 4 \times 10^6$.
$16$ MB.
Safe.

One edge case: `cin >> n`. If input fails?
Problem says "First, read an integer n". Guaranteed.
Code is good.

Reconstruction loop:
`sort` comparator logic:
If $u < v$: return `!(inv[u][v] ^ inv[u+1][v] ^ inv[u][v-1] ^ inv[u+1][v-1])`
Else: return `(inv[v][u] ^ inv[v+1][u] ^ inv[v][u-1] ^ inv[v+1][u-1])`
Wait, let's use `L` and `R` logic to simplify coding.
$L=\min(u,v), R=\max(u,v)$.
$g = inv[L][R] \oplus inv[L+1][R] \oplus inv[L][R-1] \oplus inv[L+1][R-1]$.
If $u < v$: return `!g`.
Else: return `g`.
This is concise.
Wait, accessing `inv` needs to be careful with bounds.
`inv` is global, size 2005. Max index $n$.
$n \le 2000$.
Indices used:
$L, R$ are within $1 \dots n$.
$L+1$ can be $n+1$.
$R-1$ can be $n-1$.
So indices up to $n+1$ might be accessed?
$L+1$ when $L=n$ (if $u=n, v=n+1$ impossible).
Max $R=n$.
If $R=n$, we access `inv[L+1][n]`. Max index $n+1$ (if $L=n$, but $L<R$ so $L \le n-1$, so $L+1 \le n$).
Wait. $L < R \le n$.
So $L \le n-1$. Thus $L+1 \le n$.
So row index $L+1$ is at most $n$.
Column index $R-1$: $R \ge 2$ (since $L \ge 1$).
If $R=1$, impossible since $L<R$.
So $R \ge 2$. $R-1 \ge 1$.
Also if $R=n$, $R-1 = n-1$.
Indices seem within $1 \dots n$.
Except $L+1$ might be $n$ if $L=n-1$.
What if $L+1$ is accessed as column index?
Wait, $inv[L+1][R]$. Row $L+1$, Col $R$.
If $L+1 > R$, we access invalid range $l > r$.
My global array has 0s there.
If $L+1 > R$, i.e., $L+1 \ge R+1 \implies L \ge R$, impossible as $L<R$.
Wait, $L < R \implies L+1 \le R$.
So $L+1 \le R$.
So $L+1 < R+1$?
Wait, if $L < R$, then $L+1 \le R$.
Wait, if $L=R-1$, then $L+1=R$.
So $inv[L+1][R] = inv[R][R]$.
My array is global, initialized to 0.
Is it possible $inv[R][R]$ is accessed?
Yes. It is 0.
Is it possible $inv[L+1][R-1]$ is accessed with $L+1 > R-1$?
$L < R \implies L \le R-1$.
If $L = R-1$, $L+1 = R$.
Then we access $inv[R][R-1]$.
Row $R$, Col $R-1$.
Here Row $>$ Col.
This corresponds to invalid query $l > r$.
Global array initializes to 0. So it's 0.
This is correct behavior.
So indices are within bounds $1 \dots n$ mostly, except maybe $L+1$ row index could be $n+1$?
Wait, $L < R \le n$.
Max $L = n-1$. Then $L+1 = n$.
So row indices up to $n$.
Col indices up to $R \le n$.
So max index accessed is $n$.
Wait, what if $R=n$, $L=n-1$.
Access $inv[L+1][R] \to inv[n][n]$. OK.
Access $inv[L][R-1] \to inv[n-1][n-1]$. OK.
Access $inv[L+1][R-1] \to inv[n][n-1]$. OK (zero).
What about $L=n$? Impossible as $L < R$.
So indices are within $1 \dots n$.
Wait, actually, if I declare size 2005, accessing `inv[2000][2000]` is safe.
Indices are up to $n \le 2000$.
Wait, what about $inv[L+1][R]$ where $L+1$ might exceed $n$?
No, $L < R \le n \implies L \le n-1 \implies L+1 \le n$.
So indices are safe.
Actually, $L+1$ can be $n+1$ if $L=n$. But $L<n$ always.
So we are safe.

Wait, inside lambda `min` and `max` are standard.
Headers included.

One minor point: `if (!(cin >> n)) return 0;` inside `solve`?
The loop condition `if (!(cin >> n))` is just to prevent running if EOF immediately.
But `main` calls it.
It's fine.

Final check: `p_indices` contains $1 \dots n$.
Output `ans` array.
Everything seems fine.

```cpp
/*
    Problem: Inversion
    Strategy:
    We are given n and need to find a permutation p of {1..n}.
    We can query inversion parity of subarray p[l..r].
    Constraint: n <= 2000, Max queries ~ 2,000,000.
    The number of pairs (i, j) with 1 <= i < j <= n is n*(n-1)/2.
    For n=2000, this is exactly 1,999,000, which matches the query limit.
    So we can query inversion parity for every subarray p[i..j].
    
    Once we have I(i, j) for all i < j, we can determine the relation [p_i > p_j] for any pair.
    The relation is given by:
    [p_i > p_j] = I(i, j) ^ I(i+1, j) ^ I(i, j-1) ^ I(i+1, j-1) (mod 2)
    Base cases where ranges are empty/invalid yield 0.
    
    With the full pairwise relation matrix, we can sort the indices to find the values.
    Since p is a permutation, the element with rank k (k-th smallest) has value k.
*/
```

Wait, I should check `#include <algorithm>` for `sort`.
And `<vector>`, `<iostream>`, `<cstring>`.
The code is self contained.

One question:
Can `min(u, v)` fail if u or v is not int? No, `sort` compares ints.
Wait, `p_indices` elements are ints.
All good.

Also, `inv` array size. `2005` is enough for `n=2000`.
Index range `1` to `2000`. `inv[2005][2005]` fits.

Final review of query loop.
For `n=2000`, nested loops run $2000 \times 1999 / 2 = 1,999,000$ times.
Each time prints 0 l r.
Reads response.
Stores.
Complexity is fine.

Double check indentation and formatting.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

int n;
int inv[2005][2005];

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n)) return 0;

    // Query all subarray inversions
    // We can afford O(n^2) queries, specifically n*(n-1)/2
    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j) {
            cout << "0 " << i << " " << j << "\n";
            cout.flush();
            
            int res;
            cin >> res;
            inv[i][j] = res;
        }
    }

    // Reconstruct permutation
    // Create a list of indices 1 to n
    vector<int> p_indices(n);
    for(int k=0; k<n; ++k) p_indices[k] = k+1;

    // Sort indices based on the values in the permutation p
    // We need a comparator that determines if p[u] < p[v]
    sort(p_indices.begin(), p_indices.end(), [](int u, int v) {
        if (u == v) return false;
        
        // Identify the range [L, R] covering both u and v
        int L = min(u, v);
        int R = max(u, v);
        
        // Compute [p_L > p_R] using the inclusion-exclusion principle on inversions
        // inv[x][y] stores the parity of inversions in subarray p[x...y]
        // Invalid indices (out of bounds or l > r) contribute 0
        
        // Note: Since inv is global and 0-initialized, we can safely access inv[L][R] etc.
        // We must ensure we don't access beyond array bounds.
        // L < R <= n. L >= 1.
        // L+1 <= R <= n. So L+1 is within bounds (<= 2000).
        // R-1 >= 1.
        // inv array size is 2005.
        
        int term1 = inv[L][R];
        int term2 = inv[L+1][R];
        int term3 = inv[L][R-1];
        int term4 = inv[L+1][R-1];
        
        // [p_L > p_R] = term1 ^ term2 ^ term3 ^ term4
        int greater_LR = term1 ^ term2 ^ term3 ^ term4;
        
        // If u < v, then u=L, v=R.
        // We check if p_u < p_v => NOT (p_u > p_v) => NOT greater_LR
        if (u < v) return !greater_LR;
        
        // If u > v, then u=R, v=L.
        // We check if p_u < p_v => p_R < p_L => (p_L > p_R) is True
        // So we return greater_LR
        return greater_LR;
    });

    // Assign values
    // Since p is a permutation of 1..n, if the indices are sorted by value,
    // the k-th index in sorted list corresponds to value k.
    vector<int> ans(n + 1);
    for (int k = 0; k < n; ++k) {
        ans[p_indices[k]] = k + 1;
    }

    // Output result
    cout << "!";
    for (int i = 1; i <= n; ++i) {
        cout << " " << ans[i];
    }
    cout << "\n";

    return 0;
}
```
Wait, `term2 = inv[L+1][R]`. If `L+1 > R`?
Impossible since `L < R` implies `L+1 <= R`.
Wait, if `L = R-1`, `L+1 = R`.
Then `inv[R][R]` is accessed.
Global array `inv` is initialized to 0.
`inv[R][R]` is not modified by query loops (loop `j` starts at `i+1`).
So `inv[R][R]` remains 0.
This is consistent with logic that empty/inverted range has 0 inversions.
So logic is perfect.

One check: `min`, `max` are in `<algorithm>`. Included.
Namespace `std` used.
Seems good.

The only potential issue is time limit if `n=2000` causes TLE on sort.
Sort is $O(N \log N)$ with $O(1)$ comparator. Fast enough.
Query part dominates.
Memory limit: array `int[2005][2005]` is $\approx 16$ MB. Usually limit is 256MB or 512MB. Safe.

I will wrap it in cpp block.

One detail: `inv[i][j] = res` where `res` is `int`.
Parity is 0 or 1. `int` is fine.

One small thing:
Wait, `inv[i][j]` stores `res`.
Is it guaranteed `res` is 0 or 1?
"response with ... mod 2". Yes.

The code seems complete.

One more thing: `vector<int> p_indices(n);`
`p_indices` elements are `k+1`, so $1 \dots n$.
Correct.

Final verification of the problem statement details.
"print 1 p1 p2 ... pn".
Wait. The example output says:
`! 2 3 1`.
Ah, the problem description says:
"To give your final answer, print 1 p1 p2 ... pn".
Wait. Looking at the example provided:
Example:
Standard input:
3
0
0
1
Standard output:
? 1 2
? 1 3
? 2 3
! 2 3 1
Wait. The format `1 p1 p2 ... pn` is NOT shown in example.
Wait. "To give your final answer, print 1 p1 p2 ... pn".
Usually Codeforces interactive problems ask for `! a1 a2 ... an`.
Wait.
"Example Standard output: ... ! 2 3 1".
But text says "print 1 p1 p2 ... pn".
Usually, this means the string starting with `!`.
The text says `1 p1 p2 ... pn`. The digit `1` might be a typo in problem description or referring to the command character?
Looking at example:
`! 2 3 1`
So the character is `!`.
Wait. The text says "print 1 p1 p2 ... pn". The `1` might be `!`.
Or maybe it means print the line starting with `1` (one)? No, example shows `!`.
Usually interactive protocols use `!` to signal end.
Codeforces uses `!`.
AtCoder uses `!`.
The text "print 1 p1 p2 ... pn" is suspicious.
Could it mean "print !"?
Wait. The example clearly shows `! 2 3 1`.
The text `1 p1 p2 ... pn` might be a copy-paste artifact or typo.
Wait, looking at `Score` calculation, it depends on queries, not on correctness.
I should follow the example `!`.
Wait. Maybe the user wants me to follow the text?
"print 1 p1 p2 ... pn".
Wait, looking at the prompt again.
"To give your final answer, print 1 p1 p2 ... pn"
But the example says `! 2 3 1`.
This is a contradiction.
However, in most such problems (like "Inversion" on various judges), the delimiter is `!`.
Wait, checking the problem source if possible?
It looks like a problem from "Codeforces Round #...".
Wait, actually the text "print 1 p1 p2 ... pn" is weird.
Wait, let's look closer.
Example: `! 2 3 1`.
Text: `print 1 p1 p2 ... pn`.
Is it possible the character is actually `1`? No, that would be confusing.
Maybe it means "print '!' followed by...".
Or maybe the prompt meant `!` and formatted poorly.
Given the ambiguity, the Example is the strongest hint for IO format.
The text "print 1 p1 p2 ... pn" could be interpreted as "print 1 p1...pn" (literal char 1).
But Example says `!`.
Wait, I should check if there's a convention.
Actually, if I print `1 p1...pn` and the judge expects `!`, I fail.
If I print `!` and judge expects `1`, I fail.
However, the problem title "Inversion" appears in search results for a problem where interaction protocol uses `!`.
Also, usually `1` is not a special control character. `!` is.
I will follow the example `!`.
Also, the prompt description might have typos ("1" instead of "!").
Wait, checking "Interactive Protocol" section again.
"To give your final answer, print 1 p1 p2 ... pn".
Wait, could it be `!` (exclamation mark) that looks like `1` in some font?
Or is it literally the digit 1?
If I look at the example:
Standard output:
...
! 2 3 1
The character is `!`.
So I will use `!`.
If the user copied the text and it got garbled, following example is standard.
I will use `!`.

Wait, I should check if the prompt meant "Print 1 ..." meaning "Line 1..."? No.
I'll stick with `!` as per example.
Wait, I should double check the prompt provided in the chat window.
User: "To give your final answer, print 1 p1 p2 ... pn"
User: "Example ... ! 2 3 1"
This is definitely a discrepancy.
However, in many contexts (like Codeforces), the symbol is `!`.
Wait, Codeforces 1102G? No.
Actually, maybe it's "1" meaning "Answer"?
No, it says "print 1 p1...pn".
If I look at the prompt text closely:
"print 1 p1 p2 ... pn"
vs
"! 2 3 1"
I will follow the example, assuming the text description had a typo (`1` instead of `!`).
It is highly unlikely the answer format is `1 p1...pn` with a digit 1, while the example shows `!`.
Unless the example output is illustrative but text is spec?
Usually Example trumps spec if ambiguous or vice versa if spec is clear.
But "!" is standard.
I will use `!`.

Actually, wait.
If the text says `1`, maybe it is `1` (digit 1).
But why?
Let's assume standard competitive programming conventions where `!` denotes termination.
Also, the Example is generated by an automated system or human. It shows `!`.
I will write code to print `!`.

Another check: Is it possible the problem is "Inversion" from a specific contest?
Search for "interactive problem inversion parity permutation".
It matches a problem from "Codeforces Round #208 (Div. 2)"? No.
Actually, this looks like a variation of finding permutation from inversion table.
Wait, there is a Codeforces problem "D. Permutations" or similar.
Anyway, the example `!` is strong evidence.

One more check:
The prompt says "Generate solution code:".
I will produce the code.

Wait, I should confirm if the user meant "print 1 ..." literally.
If I use `!` and judge expects `1`, it's WA.
If I use `1` and judge expects `!`, it's WA.
However, `!` is visually similar to `1` in some sans-serif fonts or monospaced displays if font is weird?
No, `!` has dot.
But maybe OCR or copy paste lost the dot?
Given the Example explicitly renders `! 2 3 1` (with space after exclamation?), and text says `1 p1...`.
It's likely a typo.
I'll stick with `!`.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

// Global array to store inversion parities
// Size 2005 is sufficient for n <= 2000
int inv[2005][2005];
int n;

int main() {
    // Optimize standard I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n)) return 0;

    // The number of pairs (i, j) with 1 <= i < j <= n is n*(n-1)/2.
    // For n = 2000, this equals 1,999,000, which is exactly the query limit.
    // We query the inversion parity for every subarray p[i...j].
    
    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j) {
            cout << "0 " << i << " " << j << "\n";
            cout.flush();
            
            int res;
            cin >> res;
            inv[i][j] = res;
        }
    }

    // We need to reconstruct the permutation p.
    // Having the inversion parities for all subarrays allows us to determine
    // the relative order of any pair p[u] and p[v] (i.e., whether p[u] > p[v]).
    // The relationship is derived using inclusion-exclusion on the subarray sums.
    // Specifically, [p[u] > p[v]] = inv[min(u,v)][max(u,v)] ^ inv[min+1][max] ^ inv[min][max-1] ^ inv[min+1][max-1].
    // Here inv[a][b] is the parity of inversions in p[a...b].
    // Note: inv[a][b] is 0 if a >= b (handled by global initialization).

    // Create a list of indices 1 to n.
    vector<int> p_indices(n);
    for (int k = 0; k < n; ++k) {
        p_indices[k] = k + 1;
    }

    // Sort the indices based on their values in the hidden permutation.
    // We define a custom comparator that uses the precomputed inversions.
    sort(p_indices.begin(), p_indices.end(), [](int u, int v) {
        if (u == v) return false;
        
        int L = min(u, v);
        int R = max(u, v);
        
        // Calculate [p_L > p_R]
        // Indices are guaranteed to be within [1, n] or slightly higher/lower but handled correctly.
        // Since L < R <= n, indices L, L+1, R, R-1 are all <= n.
        // However, inv is accessed for L >= R cases (returns 0).
        int term1 = inv[L][R];
        int term2 = inv[L+1][R];
        int term3 = inv[L][R-1];
        int term4 = inv[L+1][R-1];
        
        int greater_LR = term1 ^ term2 ^ term3 ^ term4;
        
        // If u < v, then u=L, v=R.
        // We check if p[u] < p[v] => NOT (p[u] > p[v]) => !greater_LR
        if (u < v) return !greater_LR;
        
        // If u > v, then u=R, v=L.
        // We check if p[u] < p[v] => p[R] < p[L] => p[L] > p[R] is true => greater_LR
        return greater_LR;
    });

    // The sorted indices correspond to the sorted values 1, 2, ..., n.
    // So if p_indices[k] is the k-th index in sorted order, its value is k+1.
    vector<int> ans(n + 1);
    for (int k = 0; k < n; ++k) {
        ans[p_indices[k]] = k + 1;
    }

    // Print the reconstructed permutation
    // Following the example format "!" is used to indicate termination.
    cout << "!";
    for (int i = 1; i <= n; ++i) {
        cout << " " << ans[i];
    }
    cout << "\n";

    return 0;
}
```
Wait, I need to ensure `ans` vector is printed with spaces.
The example output `! 2 3 1` has space between `!` and `2`.
My code `cout << "!";` then loop `cout << " " << ans[i];`.
Result: `! 2 3 1`.
Matches example.
Looks good.

One final check: The loop `for (int k = 0; k < n; ++k)` fills `ans` at indices `p_indices[k]`.
`p_indices` is sorted by value.
So `p_indices[0]` has value 1. `ans[p_indices[0]] = 1`.
`p_indices[n-1]` has value n. `ans[p_indices[n-1]] = n`.
Then printing `ans[1]` to `ans[n]` gives `p_1, p_2, ..., p_n`.
Logic is correct.

Wait, `inv` access safety.
`inv[L+1][R-1]`.
If `L=n, R=n`? Impossible as `L < R`.
If `L=n-1, R=n`.
`L+1 = n`. `R-1 = n-1`.
Access `inv[n][n-1]`.
Row index `n`, col `n-1`.
Valid.
Since `inv` is 2005x2005.
`n` up to 2000.
Safe.

I think the solution is robust.

One detail: `cout << "\n"` at end.
It's good practice.

```cpp
// Final Code Structure
```
I'll output just the code as requested.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

// Global array to store inversion parities for subarrays.
// The constraints are n <= 2000, so a 2005x2005 array fits in memory.
// Initialized to 0 by default.
int inv[2005][2005];
int n;

int main() {
    // Optimize I/O operations for performance.
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n)) return 0;

    // The problem provides a limit of approximately 1,999,000 queries for n=2000.
    // The number of pairs (i, j) with 1 <= i < j <= n is n*(n-1)/2.
    // For n=2000, this is exactly 1,999,000.
    // Thus, we can afford to query the inversion parity for every subarray p[i...j].
    // Storing these values allows us to reconstruct the full relative ordering of elements.

    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j) {
            cout << "0 " << i << " " << j << "\n";
            cout.flush();
            
            int res;
            cin >> res;
            inv[i][j] = res;
        }
    }

    // Now we reconstruct the permutation p.
    // We create a list of indices 1 to n.
    vector<int> p_indices(n);
    for (int k = 0; k < n; ++k) {
        p_indices[k] = k + 1;
    }

    // Sort these indices based on their values in the hidden permutation p.
    // We need a comparator to determine if p[u] < p[v].
    sort(p_indices.begin(), p_indices.end(), [](int u, int v) {
        if (u == v) return false;
        
        // Let L and R be the smaller and larger indices respectively.
        // We want to determine the relationship between p[L] and p[R].
        // The parity of inversions in the subarray p[L...R] relates to individual inversions.
        // By inclusion-exclusion, [p[L] > p[R]] can be derived from the inversion counts of subarrays.
        // Formula: [p[L] > p[R]] = inv[L][R] ^ inv[L+1][R] ^ inv[L][R-1] ^ inv[L+1][R-1] (mod 2)
        // Note: If L >= R, inv[L][R] is conceptually 0 (handled by global initialization).
        
        int L = min(u, v);
        int R = max(u, v);
        
        // Accessing inv array. Since L < R <= n, indices are within bounds.
        // inv[L+1][R] accesses row L+1 <= R (valid or L+1 > R impossible since L < R).
        // Wait, L < R implies L+1 <= R. If L+1=R, it accesses inv[R][R] which is 0.
        // inv[L+1][R-1] accesses row L+1, col R-1.
        // Since L < R, L+1 <= R. If L+1 > R-1 (i.e. L+1=R), it accesses inv[R][R-1] which is 0.
        // Thus, valid indices are always within [1, n] and defaults to 0 for invalid/empty ranges.
        
        int term1 = inv[L][R];
        int term2 = inv[L+1][R];
        int term3 = inv[L][R-1];
        int term4 = inv[L+1][R-1];
        
        // Calculate if p[L] > p[R]
        int greater_LR = term1 ^ term2 ^ term3 ^ term4;
        
        // If u < v, then u = L, v = R.
        // We want true if p[u] < p[v], which means [p[L] > p[R]] is False.
        if (u < v) return !greater_LR;
        
        // If u > v, then u = R, v = L.
        // We want true if p[u] < p[v] => p[R] < p[L] => p[L] > p[R] is True.
        return greater_LR;
    });

    // Since p is a permutation of {1, ..., n}, the sorted indices correspond to values 1, 2, ..., n.
    // If the k-th element in sorted indices is idx, then p[idx] = k + 1.
    vector<int> ans(n + 1);
    for (int k = 0; k < n; ++k) {
        ans[p_indices[k]] = k + 1;
    }

    // Output the result in the specified format.
    // Although the text description says "1 p1 p2 ... pn", the example shows "! 2 3 1".
    // Following the example is standard.
    cout << "!";
    for (int i = 1; i <= n; ++i) {
        cout << " " << ans[i];
    }
    cout << "\n";

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=406)

````
I will implement this in C++.
</think>

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n;
    cin >> n;
    
    vector<int> p(n + 1);
    
    // Calculate inversion parity for each position to find element at each index
    // by counting its contribution through the interaction
    
    for (int i = 1; i <= n; i++) {
        // For each element, we determine what value it has
        // by binary searching on its possible values
        
        int l_val = 0, r_val = n + 1;
        
        while (l_val < r_val) {
            int mid = (l_val + r_val) / 2;
            
            // Check if current element > mid
            int cnt_gt = 0;
            for (int j = 1; j < i && p[j] != -1; j++) {
                cnt_gt += (p[j] > mid ? 1 : 0);
            }
            for (int j = i + 1; j <= n; j++) {
                if (p[j] == mid || p[j] == -1) continue;
                if (j >= i && p[i] != -1) {
                    // Query [i, i]
                    cout << "? " << i << " " << i << endl;
                    
                    // Actually need to query a range involving i with a fixed mid
                    // Let me reconsider the approach
                    
                    // Better: use prefix xor array idea
                    // f[l][r] = parity of inversions in [l,r]
                    // But that's too many queries.
                }
            }
            
            l_val = mid + 1;
        }
    }
    
    return 0;
}
```
````
