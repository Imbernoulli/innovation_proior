# 题 249 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The user wants a C++ solution for an interactive problem called "X-OR".
The goal is to find a hidden permutation $p$ of length $n$ where elements are integers from $0$ to $n-1$.
I can query the value of $(p_i \ | \ p_j)$ for distinct indices $i, j$.
The maximum number of queries allowed is 4300.
$n \le 2048$.
Time limit is 1 second.

Let's analyze the constraints.
$n$ up to 2048 means $O(n^2)$ might be too slow in terms of queries ($2048^2 \approx 4 \times 10^6 > 4300$). So we need a more efficient way than asking all pairs.
The query returns the bitwise OR of two values. Since values are $< n \le 2048$, they fit within 11 bits (since $2^{11} = 2048$).
The maximum value in the array is $n-1$. Since $n \le 2048$, the numbers have at most $\lceil \log_2 2048 \rceil = 11$ bits. Actually $2047$ requires 11 bits. Wait, $n=2048$ means max element is 2047. $2^{11} = 2048$. So elements are $0..2047$. They fit in 11 bits.

Let's look at the properties of XOR vs OR. Usually, with only OR queries, it's harder to distinguish individual bits because $A|B = A + B - (A\&B)$. But we don't get AND or sum directly. However, knowing the set of values helps.
Wait, standard XOR problems often involve parity. Here it is OR.
If I know $A | B$, do I know $A$ and $B$? Not necessarily. If $A=1, B=2$, $A|B=3$. If $A=3, B=0$, $A|B=3$.
But since it's a permutation, every number appears exactly once. This is crucial.

Let's consider small $n$.
If $n$ is large, maybe we can determine the bits one by one? Or maybe we can group numbers based on their bit representations?
Actually, finding a specific value in the array using OR queries with other known values is a common strategy.
Suppose we fix $p_k$ for some index $k$. Then for any other index $j$, querying $p_k \ | \ p_j$ gives us information about bits that are NOT in $p_k$ but are in $p_j$, plus those that are in both.
Specifically, if we had an element with value 0, then $0 | p_j = p_j$. So if there was a 0 in the array, say at index 0, then $p_0 | p_j$ would reveal $p_j$ exactly.
Does 0 exist in the permutation? The problem says "elements from 0 to n-1". So yes, 0 must be in the permutation since it's a permutation of $0..n-1$.
So, if we can find which position holds the value 0, let's call it $z$, then querying $? z i$ for all $i$ will give us $p_z | p_i = 0 | p_i = p_i$.
This determines the entire permutation with $n-1$ queries.
Since $n \le 2048$, $n-1 \approx 2000$, which is well within the limit of 4300.
So the main task is to find the index $z$ such that $p_z = 0$.

How to find the index of 0 using OR queries?
We don't know who is 0. We just know it exists somewhere.
Querying arbitrary pairs might help identify candidates.
However, note that $x | y \ge x$ and $x | y \ge y$.
The minimum possible return value is 0 (only if one of them is 0).
So if we query many pairs, the result is likely to be strictly greater than at least one operand unless one operand is 0.
Wait, if $p_z = 0$, then $p_z | p_j = p_j$.
If we pick some pivot $i$, and query $i$ against everyone else... wait, that doesn't tell us if $i$ is the pivot.
Actually, if we have a candidate for 0, we can verify it.
But how to find it efficiently without iterating all pairs?
Wait, if $n$ is small enough, maybe we can use binary search? No, no ordering property on indices.
But we have 4300 queries. Finding the zero with $\approx \frac{2048}{2}$ queries via pairwise checks might work?
Consider this: if we check a subset of elements against a fixed element $x$, we get $x | p_j$.
If $x$ is large, say $x=n-1$, then $x | p_j = n-1$ for all $j$ because $n-1$ has all bits set up to the necessary range? Not necessarily.
Example: $n=5$, elements $0,1,2,3,4$. Binary: $000, 001, 010, 011, 100$. Max is 100 (4).
If we pick 4, $4 | 0 = 4$, $4 | 1 = 5$ (oops, max element is $n-1=4$). Wait.
Permutation of $0 \dots n-1$. Max value is $n-1$.
Let $M = n-1$.
Does $M$ always OR anything to result in something $\ge M$? Yes.
Does it result in exactly $M$?
If $p_j$ is a submask of $M$, then $M | p_j = M$.
If $p_j$ has a bit not in $M$? Impossible since $p_j < n \le M+1$, so all bits are within the bits of $M$ (assuming $M$ covers the range of values). Wait, $M=n-1$. $2^k$ can be larger than $n-1$. E.g., $n=3$, values $0, 1, 2$. Binary: $0, 1, 10$. $M=2$. $2 | 1 = 3 > 2$.
So querying against $n-1$ (if we knew its index) would return something potentially larger than $n-1$.
Wait, if the result is $> n-1$, that's impossible because values are $< n$.
Wait, $p_i | p_j$ is the bitwise OR.
Is it possible for $p_i | p_j \ge n$?
$p_i < n$ and $p_j < n$.
Yes. Example: $n=3$, values $0, 1, 2$. $1 (01) | 2 (10) = 3 (11)$. 3 is not in the permutation, but is a valid output of a query. The constraint is that $p_k$ are in $0..n-1$. The query returns $p_i | p_j$. This value can be up to roughly $2n$ or whatever the next power of 2 is.
For $n=2048$, values are up to 2047. $p_i | p_j$ can be up to 2047. Why? Because all $p_i$ share the same set of bits available. No, that's false.
For $n=2048$, elements are $0..2047$. All these are representable in 11 bits. The maximum possible OR of two numbers in $0..2047$ is indeed bounded by the smallest power of 2 minus 1 covering the range, which is 2047.
So $p_i | p_j \le 2047$ is guaranteed.
Back to finding 0.
If we find $p_i | p_j == \max(p_i, p_j)$, it implies $p_i \subseteq p_j$ or $p_j \subseteq p_i$ as bitmasks.
This doesn't help much to find 0 specifically.

Let's use the fact that we have plenty of queries (4300 for $N=2048$).
Ratio is roughly 2:1.
Can we perform a tournament or similar structure?
If we pick a random index, say 1. Let's query $? 1 i$ for $i=2..n$.
Let $res_i = p_1 | p_i$.
If $p_1 = 0$, then $res_i = p_i$. We learn the whole permutation. Cost: $n-1$.
If $p_1 \neq 0$, we learn partial info.
We can try to filter indices.
If $res_i == res_j$, does it mean $p_i = p_j$? Yes, because permutation.
But multiple $res_i$ can be equal even if $p_i$ different? No, $p_i$ unique implies $p_1 | p_i$ doesn't uniquely identify $p_i$.
Example: $p_1=1 (01), p_2=2 (10), p_3=3 (11)$.
$p_1 | p_2 = 3$.
$p_1 | p_3 = 3$.
Here $p_2 \neq p_3$, but result is same.
So equality doesn't imply identity.
Inequality implies nothing either (different results don't mean anything special).

Let's refine the strategy to find 0.
We want to find the index $k$ where $p_k=0$.
Since $0$ is the identity element for OR, $0 | x = x$.
Also for any $y > 0$, $y | x > x$ isn't necessarily true ($y|x \ge y$ and $y|x \ge x$).
Actually, if we can find the minimum value in the permutation among a set, does it help?
No, we get OR sums.
Let's observe the properties of $0$. It is the unique element with no bits set.
So for any $k$, $p_k$ has some bits set.
For $k=z$ (where $p_z=0$), $p_z$ has 0 bits set.
If we query $? k l$, we get $p_k | p_l$.
If we fix an index $k$, and iterate $l=1 \dots n-1$.
We get $val_l = p_k | p_{l+1}$.
If $p_k$ is "sparse" (has few bits), it's hard to see. If $p_k$ is 0, we see $p_{l+1}$ directly.
If we pick an index $x$ such that $p_x$ is likely to cover the bits of other numbers?
Or maybe we can narrow down candidates for 0.
Notice that if $p_i$ has high bits set, $p_i | p_j$ will have those bits.
Maybe we can eliminate non-zero candidates?
For each bit $b$ (from 0 to 10), consider the mask $M_b = 2^b$.
If $p_i$ does NOT have bit $b$, then $p_i$ contributes 0 at bit $b$.
However, we don't know the mapping.

Wait, $N \le 2048$. 4300 queries.
$N$ queries allows solving completely if we know 0.
Can we find 0 using fewer queries?
There is a trick for finding 0 in an OR array.
Let's assume we have found one element $u$. We query $? u v$ for all $v$.
We obtain a set of values. One of them corresponds to $p_u | p_v$.
If $p_u$ shares no bits with $p_v$, result is sum. If shared, less.
This approach seems insufficient to pinpoint 0 quickly.
Let's reconsider the example logic.
If we select a random $i$, and ask $i$ against all others, we get $n-1$ values.
Among these $n-1$ results, one value will be $p_i$ itself (corresponding to the query involving the index where $p=0$, i.e., $p_i | 0 = p_i$).
Wait! This is the key.
Let's say $z$ is the index where $p_z = 0$.
If we query $? i j$, and $i \neq z, j \neq z$, result is $p_i | p_j$.
If we query against $i$: $? i z \implies p_i | 0 = p_i$.
So if we query a specific index $i$ against *everyone else*, we will receive a value equal to $p_i$ if and only if the partner is $z$.
Wait, for a specific pair $(i, z)$, the result is $p_i$.
Since $z$ is unique, for a fixed $i$, exactly one query $? i k$ will return $p_i$ (the one where $k=z$).
All other queries $? i k$ ($k \neq z$) will return $p_i | p_k \ge p_i$.
Actually $p_i | p_k \ge p_i$ is always true.
And since $p_k \ge 0$, strict inequality holds iff $p_k$ has some bit that $p_i$ doesn't.
Since $p_k$ can be anything, usually $p_i | p_k > p_i$.
But for the specific case where the other operand is 0, we get $p_i$.
So if we query $i$ against all other $n-1$ indices, we get one response $R$ such that $R = p_i$, and for others $R' \ge p_i$ (and typically $R' > p_i$).
However, knowing $p_i$ doesn't give us $z$. It just tells us $p_i$'s value.
Wait, if we collect the results for a fixed $i$: $\{ q_j = p_i | p_j \}_{j \neq i}$.
One of these $q_j$ is exactly $p_i$. The others are $p_i | p_j$.
If we could identify which $q_j$ is $p_i$, we would know that the corresponding index $j$ is $z$ (the one holding 0).
HOWEVER, simply checking if $q_j == p_i$ requires knowing $p_i$.
But we don't know $p_i$.
Can we deduce $p_i$ from the set $\{q_j\}$?
Not really easily.
But notice: $p_i | p_j \ge p_i$ always.
If we find an index $j$ such that $q_j < p_i | p_j$, it's obvious? No, we don't know $p_i$.
But we do know that $p_i | p_j \ge p_i$.
If $p_i = 0$, then $0 | p_j = p_j$. The minimum of the results for fixed $i$ would be 0 (since one result is 0).
Actually, if $i$ is the index of 0, then all $n-1$ results are $p_j$ for $j \neq i$.
These results are the actual values of the permutation excluding $p_i=0$.
So the set of answers will be $\{0, 1, \dots, n-1\} \setminus \{0\}$.
Wait, if $i=z$, results are $p_z | p_j = 0 | p_j = p_j$.
Since $\{p_k\}$ is a permutation of $0 \dots n-1$, the values $\{p_j : j \neq i\}$ are a permutation of $0 \dots n-1 \setminus \{p_i\}$.
If $p_i=0$, the answers are $\{1, \dots, n-1\}$. Note that 0 won't appear in the answers because 0 is at $i$ and we query $i$ against $j \neq i$.
So if we choose index $i$ such that $p_i=0$, we will receive answers $\{1, \dots, n-1\}$.
Wait, this assumes $p_i=0$ is not returned. Correct, since we query $i$ with $j \neq i$, and $p_i=0$, result is $p_j$. Since $p_j \ge 0$ and $p_i$ is the only 0, if $j \neq i$, then $p_j \neq 0$.
So, if we pick index $i$, get $n-1$ answers $ans_1, \dots, ans_{n-1}$.
If $p_i=0$, the answers are $1, \dots, n-1$. The MINIMUM answer should be 1.
If $p_i \neq 0$, what happens?
One of the answers will be $p_i | p_z = p_i | 0 = p_i$.
All other answers are $p_i | p_k$.
Since $p_k$ is a permutation, there is exactly one answer equal to $p_i$ (corresponding to $j=z$).
The other $n-2$ answers are $> p_i$ (strictly greater)? Not necessarily. $p_i | p_k$ could be equal to $p_i$ if $p_k$ is a submask of $p_i$.
But $p_i | p_k \ge p_i$.
If $p_i \neq 0$, then $p_i \ge 1$.
If we see any answer equal to $0$, it means $p_i | p_j = 0 \implies p_i=0, p_j=0$. Impossible since distinct.
So answers are always $\ge 1$.
Key observation:
If $p_i = 0$, answers are $\{1, \dots, n-1\}$. Minimum is 1.
If $p_i \neq 0$, answers include $p_i$.
Minimum of answers could be 1 (if some $p_k=1$ and $p_i | 1 = 1$).
However, if we just take the minimum of the answers returned for a fixed $i$, let it be $min\_ans$.
If $p_i = 0$, $min\_ans = 1$ (since values are $1..n-1$).
Wait, if $p_i=0$, answers are $p_1, \dots, p_n \setminus \{0\}$. The minimum is indeed 1 (since 0 is removed).
So $min\_ans$ is 1 regardless of $p_i$ being 0 or not?
Example: $p = [1, 0, 2]$. $n=3$.
Try $i=1$ (value 1). Query $? 1 2$ -> $1|0 = 1$. Query $? 1 3$ -> $1|2 = 3$. Answers $\{1, 3\}$. Min = 1.
Try $i=2$ (value 0). Query $? 2 1$ -> $0|1 = 1$. Query $? 2 3$ -> $0|2 = 2$. Answers $\{1, 2\}$. Min = 1.
Try $i=3$ (value 2). Query $? 3 1$ -> $2|1 = 3$. Query $? 3 2$ -> $2|0 = 2$. Answers $\{3, 2\}$. Min = 2.
Here, when $i$ is index of 2 ($p_3=2$), min is 2.
So, if the minimum of the answers for $i$ is 1, it doesn't prove $p_i=0$.
But if the minimum of the answers is 1, $i$ COULD be 0, or $i$ COULD be something that ORs with 1 to give 1.
This means $p_i$ doesn't have bit 0 set? No.
If $p_i=0$, we get all remaining values. Since it's a permutation of $0..n-1$, if $n \ge 2$, 1 must be present somewhere.
So if $p_i=0$, 1 is in the answers. Thus min is 1.
If $p_i \neq 0$, we have $p_i$ in answers. And $p_i | 1$ might be in answers (where 1 is $p_z$ if $z \neq i$, wait).
Let's formalize.
We suspect $i$ is $0$-index.
We query $i$ against $1 \dots n-1$ (excluding $i$).
If $i$ is indeed the 0-index ($p_i=0$):
Answers = $\{ p_k : k \neq i \}$. Since $p$ is perm of $0..n-1$, this set is $\{1, \dots, n-1\}$.
Min = 1. Also, the set of answers contains all numbers from $1$ to $n-1$.
If $i$ is NOT the 0-index ($p_i \neq 0$):
Let $z$ be the true 0-index. $p_z=0$.
Answer for $j=z$ is $p_i | p_z = p_i$.
Other answers are $p_i | p_k$ ($k \neq i, k \neq z$).
So the set of answers is $\{ p_i \} \cup \{ p_i | p_k : k \neq i, z \}$.
Crucially, $p_i \ge 1$.
Also, is it possible that some $p_i | p_k = 1$?
Yes, if $p_i$ is submask of 1 and $p_k$ is submask of 1 (impossible as permutation).
Or $p_i | p_k = 1$ means bits are subset of 1. Since $p_k$ unique, we can have e.g. $p_i=0$ (but we assumed not). $p_i$ must be 1. Then $1 | p_k = 1$ implies $p_k$ submask of 1. Only 0 and 1. If $p_i=1$, then $p_k$ cannot be 1. Must be 0. But we assumed $i \neq z$, so $p_z=0$ exists elsewhere. If $k=z$, we already handled that ($p_i | 0 = 1$).
So, if $i \neq z$, then $p_i$ is in answers.
And since $p_i \ge 1$, the minimum of the answers is $\min( \text{values} )$.
If $p_i$ has bits outside of lower bits, say $p_i=2$, then $p_i | p_k \ge 2$.
Unless there is a $k$ such that $p_i | p_k = 1$. This requires $p_i | p_k \le 1$. Since $p_i \neq 0$, $p_i \in \{1\}$. Then $1 | p_k = 1 \implies p_k \subseteq 1$. $p_k=0$ or $1$. $p_k \neq 1$. So $p_k=0$. This corresponds to $j=z$.
So if $p_i=1$, then answers are $\{ 1 | 0, 1 | p_k \} = \{1, 1|p_k\}$. Min is 1.
If $p_i > 1$, then $p_i$ has bit at pos $\ge 1$ set. $p_i | p_k \ge p_i \ge 2$. So Min $\ge 2$.
So:
If $\min(\{ p_i | p_k \}) == 1$, then it is possible $p_i=0$ or $p_i=1$.
If $\min > 1$, then $p_i \neq 0$ and $p_i \neq 1$.

This looks like we can binary search for 0? Or rather, test candidates.
Actually, if we just check $i$ against everyone. If $\min(ans) > 1$, $i$ is not 0 and not 1.
If $\min(ans) == 1$, $i$ could be 0 or 1.
Since we have 4300 queries, we can afford to check "candidates".
But wait, checking against ALL others takes $O(n)$ queries. Doing this twice is fine.
Strategy:
1. Pick index 0 (using 1-based indexing for protocol).
2. Query $? 0 k$ for all $k \neq 0$. Store results. Find minimum result $m_0$.
3. If $m_0 == 1$, candidate $0$ is either 0-index or 1-index (if $p_0=1$).
   Wait, if $p_0=0$, answers are $1..n-1$, min is 1.
   If $p_0=1$, answers contain $1|0=1$ and others $\ge 1$. Min is 1.
   Is there a way to distinguish?
   If $p_0=0$, the answers are exactly $\{1, 2, \dots, n-1\}$ (sorted).
   If $p_0=1$, the answers are $\{ 1 \} \cup \{ 1 | x : x \neq 0, 1 \}$.
   $1 | x = x$ if $x$ submask of 1 ($x=0$ impossible here). $x=1$ impossible.
   Wait, $1 | x \ge x$.
   Actually, if $p_0=1$, one answer is $1|0=1$. Another is $1|x$.
   If we sort answers, if $p_0=0$, answers are $1, 2, \dots, n-1$.
   If $p_0=1$, answers are $1$ (from 0), and then $1|p_k$.
   For $p_0=1$, we have duplicate values?
   Suppose $p=[0, 1, 2]$. Indices 0, 1, 2.
   Try $i=0$ (val 1). Queries: $1|0=1$, $1|2=3$. Answers $\{1, 3\}$.
   Try $i=1$ (val 0). Queries: $0|1=1$, $0|2=2$. Answers $\{1, 2\}$.
   Sorted sets:
   $i=0 \implies \{1, 3\}$.
   $i=1 \implies \{1, 2\}$.
   If we just compare the set of values, if $p_i=0$, we get full set of remaining.
   If $p_i \neq 0$, we miss at least one value (or rather we have modified ones).
   Specifically, if $p_i=0$, answers = $\{ p_k : k \neq i \}$.
   If $p_i \neq 0$, let $S_i = \{ p_i | p_k : k \neq i \}$.
   Notice that $p_i$ must be in $S_i$ (at query with $z$).
   Also for any $x \neq p_i, p_z$, $p_i | x \ge x$.
   If $p_i > 0$, it seems unlikely we cover all numbers $1..n-1$.
   Actually, if $p_i=0$, $S_i = \{ 1, \dots, n-1 \}$.
   If $p_i=1$, $S_i = \{ 1, 3, 2|1?, 2, 4|1... \}$. Basically $1|p_k$.
   If $p_i=1$, we definitely get 1.
   Can we distinguish $p_i=0$ from $p_i=1$?
   If we query another index?
   
Alternative Idea:
Find 0 by halving.
Split array into two halves $L, R$.
Pick arbitrary pivot $x$. Check $x$ against $y \in L \cup R$. Too many.
Better:
Query index $i$ with a few representatives.
But we have $O(n)$ budget.
Wait, can we just find 0 by iterating?
Iterating all $i$ and doing $O(n)$ checks is $O(n^2)$ -> TLE/MTE.
We need $O(n)$ total queries.
With 4300 queries and $N=2048$, average per element is 2.
If we check $i$ against a subset of size $K$, we can filter candidates.
Consider the property: $p_i | p_z = p_i$. $p_z = 0$.
If we query $i$ against a random $j$, $ans = p_i | p_j \ge p_i$.
Equality holds iff $p_i$ is a superset of $p_j$ (bitwise).
Since 0 is subset of everything, $0 | x = x$.
If we query $z$ (0-index) against $j$, ans $= p_j$.
If we query $j$ against $z$, ans $= p_j$. Same.
So, if we find an index $z$ such that querying it against ANY $j$ yields $p_j$?
But we don't know $p_j$.
Wait, if we find $z$ such that querying $z$ against $j$ returns a result smaller than querying $j$ against $z$? Impossible, commutative.
Result is symmetric.
However, we established that if $p_i=0$, then answers to $\{p_i | p_j\}$ are exactly $\{p_j\}_{j \neq i}$.
This means if we sort the answers, we see a sequence $1, 2, 3, \dots$ ? No, $p_j$ is not sorted.
But the set of values is $\{1, \dots, n-1\}$.
So count of answers equal to $v$ is 1 for each $v \in \{1, \dots, n-1\}$.
If $p_i \neq 0$, then $p_i$ appears in answers.
Does any value missing from answers appear in $\{1, \dots, n-1\}$?
Yes, $p_i$ is the one missing (replaced by $p_i | p_z = p_i$ ? No).
Wait.
Set $S_{true} = \{ p_k : k \neq i \}$. Size $n-1$.
Set $S_{observed} = \{ p_i | p_k : k \neq i \}$. Size $n-1$.
Difference:
$S_{true} = S_{observed} \Delta \{ p_i, \text{some } x \}$.
Actually, let's analyze the values.
If $p_i=0$, $S_{observed} = \{1, \dots, n-1\}$.
If $p_i \neq 0$. $p_i \in S_{observed}$ (at $k=z$).
Are there duplicates?
If $p_i | p_a = p_i | p_b$, does $p_a = p_b$? Not necessarily.
But $p_a, p_b$ are distinct elements of permutation.
Anyway, if $p_i=0$, the answers are a permutation of $1..n-1$.
If $p_i \neq 0$, the answers are NOT a permutation of $1..n-1$. Why?
Because $p_i | p_k \ge p_i \ge 1$.
If $p_i \neq 0$, then $p_i$ is some value $V \ge 1$.
In $S_{observed}$, $V$ comes from $k=z$.
In $S_{true}$, $V$ (if it was present) comes from $k=$ index of $V$.
Wait, if $p_i$ is not 0, does $S_{observed}$ contain all numbers $1..n-1$?
If $n=3$, $0, 1, 2$.
$i$ points to 1. $S_{observed} = \{1|0, 1|2\} = \{1, 3\}$. Missing 2.
$i$ points to 2. $S_{observed} = \{2|0, 2|1\} = \{2, 3\}$. Missing 1.
It seems if $p_i \neq 0$, then $S_{observed}$ will NOT contain all values in $\{1, \dots, n-1\}$.
Proof sketch:
If $p_i \neq 0$, let $V = p_i$.
Then $V \in S_{observed}$ (derived from $z$).
But for $V$ to be in $S_{true} = \{ p_k : k \neq i \}$, we would need $p_k = V$.
But $p_i = V$ is at $i$. Since $i \neq k$, impossible. So $V$ is NOT in $\{ p_k \}$.
So $S_{true}$ does not contain $V$.
Wait, $S_{observed}$ IS derived from $S_{true} \cup \{ p_i \}$ modified.
If $p_i \neq 0$, is $V=p_i$ in $S_{observed}$? Yes, as $p_i | 0$.
Is $V$ in $\{1, \dots, n-1\}$? Yes, since $V < n$.
Is $V$ in $S_{true}$? No, because $V$ is used at $i$.
So if $p_i \neq 0$, $S_{observed}$ contains $V$ but $S_{true}$ doesn't.
Does $S_{true}$ always equal $\{1, \dots, n-1\}$?
Only if $p_i=0$.
Wait, if $p_i \neq 0$, $S_{true}$ is a permutation of $0..n-1$ excluding $p_i$. So $S_{true} = \{0..n-1\} \setminus \{p_i\}$.
Wait, 0 is excluded?
Ah! If $p_i \neq 0$, then 0 is at some $z \neq i$.
So $S_{true}$ contains 0?
Wait, if $i$ is NOT the zero-index. Then $z \neq i$. So $p_z = 0$ is in $\{ p_k : k \neq i \}$.
So $S_{true}$ contains 0.
BUT we are querying $p_i | p_k$.
Since $p_i \neq 0$, $p_i | p_k \ge p_i \ge 1$.
So $0$ can NEVER be in $S_{observed}$.
If $p_i = 0$, then $0 | p_k = p_k$. Since $p_z=0$ is at $i$, $0$ is not in the set of OTHER values $\{ p_k : k \neq i \}$.
So if $p_i=0$, $S_{observed} = \{1, \dots, n-1\}$.
If $p_i \neq 0$, $S_{observed}$ contains NO zeros.
Wait, if $p_i \neq 0$, is 0 possible in answers? No, since $p_i \ge 1$.
So in BOTH cases, 0 is not in answers.
So that distinction fails.

However:
If $p_i = 0$, $S_{observed} = \{1, \dots, n-1\}$.
If $p_i \neq 0$, $S_{observed} = \{ p_i | p_k \}$.
Note that if $p_i \neq 0$, then $p_i \in S_{observed}$.
But also $S_{true}$ (which would be $\{ p_k \}$) contains 0 and does not contain $p_i$.
Wait, $S_{observed}$ never contains 0.
If $p_i = 0$, $S_{observed}$ is a permutation of $1..n-1$. (Does not contain 0).
So the condition "$S_{observed}$ contains 0" is impossible.
So we rely on: Does $S_{observed}$ look like a permutation of $1..n-1$?
If $p_i=0$, YES.
If $p_i \neq 0$, NO.
Why?
If $p_i \neq 0$, let $S_{observed} = \{ r_k \}$.
One of $r_k$ is $p_i$.
Another is $p_i | p_k$.
It turns out $S_{observed}$ will have duplicates or holes relative to $1..n-1$.
Actually, simpler:
If $p_i=0$, then $p_i | p_k = p_k$. Since $\{ p_k \}$ (for $k \neq i$) contains $0$? No, $p_k$ ranges over all except $p_i$.
Since $p_i=0$, the set $\{ p_k \}_{k \neq i}$ is exactly $\{1, \dots, n-1\}$.
So answers are a permutation of $1..n-1$.
If $p_i \neq 0$. The set $\{ p_k \}_{k \neq i}$ contains 0.
The answers are $\{ p_i | 0 \} \cup \{ p_i | p_k : k \neq i, z \}$.
Values: $\{ p_i \} \cup \{ p_i | p_k \dots \}$.
Does this set equal $\{1, \dots, n-1\}$?
If it did, then every value $v \in \{1, \dots, n-1\}$ must be achievable as $p_i | x$.
Is it possible?
Consider $p_i=2$ (binary 10). Target includes 1 (binary 01).
We need $2 | x = 1$. Impossible since $2|x$ has bit 1 set.
So 1 cannot be formed. Thus the set is not $\{1, \dots, n-1\}$.
Generalizing: If $p_i$ has some bit $b$ set, then no $v < 2^b$ with bit $b$ unset can be formed (wait, $p_i | v$ keeps bit $b$ set).
Actually, if bit $b$ is set in $p_i$, then $p_i | x$ will have bit $b$ set.
So if $S_{observed}$ contains only numbers with bit $b$ set, then it misses any number with bit $b$ unset.
But $1, \dots, n-1$ definitely contains numbers with bit $b$ unset (unless $p_i=0$ and we missed something?).
Wait, if $p_i=0$, no bits set. $0 | x = x$. So all values possible.
If $p_i \neq 0$, let $b$ be the highest bit of $p_i$. Then for all $x$, $p_i | x$ has bit $b$ set.
Thus, $S_{observed}$ is a subset of $\{ y : y \text{ has bit } b \text{ set} \}$.
But $\{1, \dots, n-1\}$ contains numbers that do NOT have bit $b$ set (e.g. 0, though not in range, or small numbers).
Specifically, there exists $y \in \{1, \dots, n-1\}$ such that bit $b$ is 0.
For instance, $y=1$ if $b \ge 1$.
So, if $\min(S_{observed}) \neq 1$? No.
Condition: If $p_i \neq 0$, then $S_{observed}$ misses values that don't match $p_i$'s masks.
Therefore, if $S_{observed}$ matches the set $\{1, \dots, n-1\}$ (multiset wise?), then $p_i=0$.
Since checking multiset equality is $O(N)$, we can just collect all $N$ results and sort/check.
Wait, sorting takes time, but number of queries is the bottleneck.
We have budget 4300 for $N=2048$.
One pass of $N$ queries takes 2048.
Checking if $S_{observed} == \{1..n-1\}$ is easy.
Count frequencies of each number $1..n-1$. All should be 1. 0 should not appear. Max value $\le n-1$.
Actually, $p_i | p_j$ can exceed $n-1$ (up to 2047).
If $S_{observed}$ contains any value $\ge n$, then $p_i \neq 0$ (because $\{1..n-1\}$ only goes up to $n-1$).
Wait, this is a strong heuristic.
If $p_i=0$, answers are exactly $1..n-1$. Max is $n-1$.
If $p_i \neq 0$, is it possible max answer $< n$?
Example $n=3$, $p=[1, 2, 0]$. $p_1=1$. Answers $\{1|0, 1|2\} = \{1, 3\}$. Max 3. $n-1=2$. Max > $n-1$.
Example $n=4$, $0..3$. $p_0=0$. Answers $\{1, 2, 3\}$. Max 3.
$p_1=1 (01)$. Answers $1|0=1, 1|2=3, 1|3=3$. Set $\{1, 3, 3\}$. Values $\le 3$.
Here max is 3, which is $n-1$.
So max value check is not sufficient.
What about checking if values are a permutation of $1..n-1$?
If $p_1=1$, answers $\{1, 3, 3\}$. Counts: $1 \to 1$, $3 \to 2$. Not a permutation.
So we can check if answers form a valid permutation of $1..n-1$.
Algorithm:
Iterate $i$ from 0 to $n-1$ (using 1-based indexing $1..n$).
For each $i$, perform $n-1$ queries against $j \neq i$.
Wait, doing this for each $i$ is $O(N^2)$ queries. We can't do that.
We need to identify ONE $i$ that satisfies the condition.
We can stop early if $p_i=0$ is identified.
But worst case $p_0 \neq 0$, we spend $N$ queries.
If we check sequentially, we might waste queries.
BUT, do we need to check all $n-1$?
If we find ONE candidate $i$ that behaves like 0-index.
When we check $i$, we query $n-1$ times.
If $S_{observed} \neq \{1, \dots, n-1\}$, we know $i \neq 0$.
Can we do better than checking $n-1$ times?
If we query $i$ against a random sample?
If $p_i=0$, answers are $\{1, \dots, n-1\}$.
If $p_i \neq 0$, answers are "bad".
If we pick a small sample, say size $K$.
Probability that sample distinguishes?
If $p_i=0$, $p_k$ are random-ish. Sample will have distinct values.
If $p_i \neq 0$, answers might have duplicates or wrong values.
However, worst case we need to be sure.
Given the limits (4300 queries, $N=2048$), ratio is ~2.
We can afford 2 checks.
Maybe we can split $N$ into groups?
Or use the fact that we have $N$ queries roughly.
Actually, is it possible to just pick one index $0$ and run the full scan?
If $p_0 = 0$, we solve it.
If $p_0 \neq 0$, we spent $N$ queries but learned nothing useful yet?
Wait, if $p_0 \neq 0$, we learned $\{ p_0 | p_k \}$.
This tells us relations.
Can we reconstruct the permutation from these?
Maybe not directly.
But notice: 0 is the ONLY element such that $x | 0 = x$.
And for all $x$, $x | 0 = x$.
Is there any other element $y$ with property $x | y = x$ for some $x$? Yes, if $y$ is submask of $x$.
But 0 is submask of ALL elements.
So 0 is a universal submask.
If we treat 0 as a wildcard?
Actually, if we just find 0.
Can we find 0 faster?
Suppose we query $? 0 1$. Result $r_1$.
If $p_0=0$, $r_1 = p_1$.
If $p_1=0$, $r_1 = p_0$.
Since we don't know values, we just get some integer.
If we swap roles, same.
If we check $? i j$.
Consider the property:
If $i$ is 0-index, then $? i k$ gives $p_k$.
If $k$ is 0-index, then $? k i$ gives $p_i$.
So if we query $i$ against $k$, and get result $R$.
If $i$ is 0-index, we learn $p_k$.
If $k$ is 0-index, we learn $p_i$.
If neither, we learn $p_i | p_k$.
This suggests we can use 0 as a "key".
If we have a set of queries, maybe we can identify 0.

Let's rethink. $N \le 2048$. Max queries 4300.
We need to identify the index of 0.
Let's try a randomized approach or divide and conquer.
Divide and Conquer for finding 0:
Partition elements into two sets $A, B$ of size $n/2$.
Can we find 0 in one of them without $O(n)$ queries?
Pick a pivot $p \in A$.
Query $p$ against all $q \in B$.
Wait, this costs $|B|$ queries.
If we do this recursively?
Total cost? $T(n) = n/2 + T(n/2) \approx O(n)$.
Wait, if we pick $p \in A$, query against $B$ ($n/2$ queries).
Check if $p=0$.
If $p=0$, done.
If not, we haven't checked anyone in $B$ fully.
Wait, if $p \neq 0$, and we got answers.
None of the answers $r$ will be "perfect" set?
Actually, if we just want to find 0, we can check candidates.
Candidate testing:
To check if $x$ is 0-index, we need $n-1$ queries.
So we can't check many candidates.
BUT, we can check if $x$ is "compatible" with being 0.
Condition: If $x$ is 0, then for all $y \neq x$, $p_x | p_y = p_y$.
This means answers are $\{ p_y \}$.
This means answers must be all distinct and in range $[1, n-1]$ (actually range is values of perm).
If we query $x$ against $y$, get $val$.
If $x$ is 0, then $val$ MUST be one of the remaining elements.
But we don't know remaining elements.
However, we know that if $x$ is 0, all answers must be distinct?
Yes, because $\{ p_y \}$ are distinct.
If $x \neq 0$, are answers distinct?
$\{ p_x | p_y \}$. Distinctness is not guaranteed.
E.g. $p_x=1, p_y=2 \implies 3$. $p_z=3 \implies 3$.
Duplicates are very likely if $p_x \neq 0$.
So, **distinctness of answers** is a good signature.
If we query $x$ against all $y \neq x$, and get $n-1$ answers.
If all answers are distinct, does that imply $x=0$?
If $x=0$, answers are permutation of others $\implies$ distinct.
If $x \neq 0$, can answers be distinct?
$p_x | p_y = p_x | p_z$.
If $p_x=1$, $1|2=3, 1|4=5, 1|0=1$. Distinct?
Depends on $p$.
However, probability of collision increases with density.
But maybe not reliable.
Another signature:
If $x=0$, $\min(answers) = 1$?
If $p_x=0$, answers are $1..n-1$. Min is 1.
If $p_x \neq 0$. Is it possible min is 1?
Yes, if some $p_y=1$ (which exists for $n \ge 2$). Then $p_x | 1 \ge 1$.
Wait, $p_x | p_z = p_x$.
If $p_x \ge 2$, then $p_x$ itself is an answer. Min answer could be 1 if $p_x | p_y = 1$ for some $y$.
This implies $p_x \le 1$ or $p_y=0$ (handled).
If $p_x \ge 2$, then $p_x | p_y \ge 2$ (since $p_x \ge 2$ and $p_x$ has higher bit).
So, if $p_x \ge 2$, $\min(answers) \ge 2$.
So:
If $\min(ans) == 1$, $p_x$ can be 0 or 1.
If $\min(ans) > 1$, $p_x \neq 0$ AND $p_x \neq 1$.
This filters out most candidates.
How many candidates have $p_x=1$? Only one (since unique).
Candidates with $p_x=0$: one.
So we can just test a few candidates?
No, we need to be sure.
But wait.
If we pick an arbitrary index $idx$.
Run full query against all $j \neq idx$.
Calculate min.
If min $> 1$, then $p_{idx} \neq 0$ and $p_{idx} \neq 1$.
If min $== 1$, then $p_{idx}$ is 0 or 1.
If we find that min $> 1$, we discard $idx$.
If we find min $== 1$, we keep $idx$ as candidate.
Since there is only 1 zero and 1 one.
Most indices will yield min $\ge 2$.
By picking random indices, we can find the zero or one?
But we must guarantee correctness.
With limited queries, can we verify a candidate without full scan?
We established full scan is $O(N)$.
If we use full scan to filter:
Pick index 1. Scan all.
Case A: $p_1=0$. Found.
Case B: $p_1=1$. Found candidate for 0? No, min=1. We know $p_1=1$.
If $p_1=1$, we know where 0 is?
We have answers $\{ 1 | p_j \}$. One answer is $1$. The corresponding $j$ is the 0-index.
So IF we detect $p_1=1$, we can find 0 directly from the query results!
Because $p_1 | p_0 = 1$. Any other $p_1 | p_k \ge 2$ (since $p_k \ge 0$, and if $p_k=0$ result is 1, else $1 | p_k \ge 1$. Wait $1|0=1$. $1|2=3$. If $p_k=1$ impossible. If $p_k=0$ is at 0. If $p_k=1$ impossible.
Wait, if $p_1=1$, answers include 1 (from 0-index).
Do answers include any other 1?
$1 | p_k = 1 \implies p_k \subseteq 1 \implies p_k \in \{0, 1\}$.
Since $p_k$ distinct from $p_1=1$, must be 0.
So EXACTLY ONE answer is 1.
The index corresponding to answer 1 is the 0-index.
So if $p_1=1$, we solved it.
If $p_1=0$, we solved it (we get $p_j$ values, we know $p_1=0$, so we have everything).
If $p_1 \neq 0, 1$.
Then $\min(ans) \ge p_1 \ge 2$? No, could be smaller?
Wait, $ans = p_1 | p_k$. Since $p_1 \ge 2$ (no bits 0,1?), if $p_1$ has bit 1 set, min could be small?
Example $p_1=2 (10)$. $p_k=1 (01)$. $2|1=3$.
Generally $p_1 | p_k \ge p_1$.
So min $\ge 2$.
Wait, earlier I said min $> 1$ implies $p_1 \neq 0$ and $p_1 \neq 1$.
Correct.
So strategy:
1. Pick index 0. Query against all $j$.
2. Analyze answers.
   a. If $min(ans) == 1$:
      Check count of answer equal to 1.
      If count == 1:
         The index corresponding to answer 1 is the zero-index (since $p_0 | p_{zero} = 1$).
         Wait, if $p_0=0$, answers are $1..n-1$. Count of 1 is 1. Zero-index is NOT index 0.
         If $p_0=0$, we have the full permutation. Done.
         So check if the set of answers is exactly $\{1..n-1\}$.
         If yes -> $p_0=0$. Return full perm.
         If count of 1 is 1, but set is not $\{1..n-1\}$ -> $p_0=1$. The index giving 1 is zero-index. Return perm (can be deduced later? Or just store zero-index).
         Actually if $p_0=1$, we know $zero$ is found. We still need to find $p_k$ for all $k$.
         To find $p_k$, we can now use $zero$ as pivot. Query $zero | k$ gives $p_k$.
         Total queries: $N$ (first phase) + $N$ (second phase) = $2N$.
         For $N=2048$, $4096 < 4300$. OK.
      b. If count of 1 != 1:
         Could it be $p_0 \ge 2$?
         If $p_0 \ge 2$, $min(ans) \ge 2$ generally?
         Is it possible $p_0 | p_k = 1$ with $p_0 \ge 2$? No, $p_0 | p_k \ge p_0$.
         So if $min(ans) > 1$, then $p_0 \neq 0$ and $p_0 \neq 1$.
         In this case, index 0 is neither 0 nor 1.
         We discard index 0? No, we spent $N$ queries.
         We need to find 0 and 1.
         Since we know $p_0 \ge 2$, 0 and 1 are somewhere else.
         We can try to find 1.
         How?
         If we know $p_0 \ge 2$, then $p_0$ is not 0 or 1.
         Can we find 1 using $p_0$?
         Maybe not directly.
         But we have many queries left. $4300 - N$.
         With $2N$, we are safe.
         Wait, what if $p_0 \neq 0$ and we need more steps?
         If $p_0$ is random, probability it is 0 is $1/N$. Probability it is 1 is $1/N$.
         Most of the time it's neither.
         If $p_0$ is neither, we wasted $N$ queries.
         Remaining budget $4300 - 2000 = 2300$.
         We still have $N$ indices to check.
         This looks risky. $O(N^2)$ in worst case.
         
         Optimization:
         We don't need to check full $N$ scans for every element.
         However, we identified that if $min(ans) > 1$, $p_0$ is bad candidate.
         But we can use this info to prune.
         Actually, if $min(ans) \ge 2$, we know $p_0$ has no bits 0 or 1?
         No, $p_0$ could have bit 0 or 1 set, but masked by others?
         Wait $ans = p_0 | p_k$. Min over all $k$.
         If $min(ans) > 1$, it means for ALL $k$, $p_0 | p_k > 1$.
         This implies $p_0$ and $p_k$ combined always $> 1$.
         Does it imply $p_0 \ge 2$? Yes.
         Because if $p_0=1$, then $1|0=1$. If $0$ exists (always), min would be 1.
         So if min > 1, $p_0 \neq 0$ and $p_0 \neq 1$.
         So $p_0$ is a "large" number.
         
         Wait, if we find $p_0 \ge 2$.
         Then 0 and 1 must be among the other $n-1$ elements.
         Can we identify them faster?
         We have a set of queries results.
         We have $p_0$ and $p_0 | p_k$ for all $k$.
         This means we know the set of OR-sums with $p_0$.
         Does this help?
         $p_k = (p_0 | p_k)$ if $p_k \subseteq p_0$.
         If $p_k$ is a submask of $p_0$, we recover $p_k$ directly!
         How to check if $p_k$ submask?
         $p_0 | p_k == p_0$.
         So, if we find any $k$ such that $p_0 | p_k == p_0$.
         Then $p_k \subseteq p_0$.
         Since $p_k \ge 0$.
         This gives us direct values.
         But we need 0.
         $0 \subseteq p_0$ always. So $p_0 | 0 = p_0$.
         There is exactly one $k$ (the zero index) where this holds?
         Are there other submasks?
         Yes, any element in permutation that is submask of $p_0$ satisfies $p_0 | p_k = p_0$.
         Since $0$ is the only element with no bits, 0 is one of them.
         But others might exist.
         Example $p_0 = 3 (11)$. Elements $0, 1, 2$.
         $3|0=3, 3|1=3, 3|2=3$. All satisfy.
         So we can't distinguish 0 from 1 or 2 just by this.
         
         Let's use the budget wisely.
         If we fail to find 0 or 1 in first $N$ queries, we have limited budget.
         We need a general solver.
         What if we just maintain the list of indices whose status is unknown?
         Initially all indices.
         Step 1: Pick an index $u$. Query against others.
         Determine if $u=0$, $u=1$, or neither.
         Wait, checking "neither" requires full $N$ query.
         Can we do partial check?
         Check against a few random indices $v_1, v_2, \dots$.
         If $u=0$, $u|v = v$.
         If $u=1$, $u|v \in \{v, v+1?\}$.
         If $u \ge 2$, $u|v \ge 2$.
         If we pick $v$ such that $p_v=0$. $u|v = u$.
         So if we can find a zero, we are golden.
         Wait, we are trying to FIND zero.
         This is circular.
         
         Alternative Strategy:
         The values are in $0..N-1$.
         We can determine the MSB (Most Significant Bit) distribution.
         Split $0..N-1$ into Low and High based on MSB.
         If $p_i$ has MSB $k$, then $p_i \ge 2^k$.
         We can find the set of indices having MSB $k$.
         If we can partition indices by bit values.
         Let's focus on finding 0 first.
         There is a known technique for finding 0 in $O(N)$ queries.
         We can just check $p_0$ against all. If it works, great.
         If not, $p_0 \neq 0$.
         Wait, actually, we can construct a candidate for 0.
         Consider all indices.
         If we maintain a pool of potential 0s.
         Start with all indices.
         Pick two indices $i, j$. Query $i, j$.
         $ans = p_i | p_j$.
         If $ans = p_i$ (meaning $p_j \subseteq p_i$) ... no we don't know $p_i$.
         If $ans < i$? No order.
         
         Wait, I am overthinking.
         If we pick index 0.
         Run queries against $1..N$.
         Collect answers.
         Check if 0 is at 0.
         Condition for 0 at 0: The set of answers is $\{1, 2, ..., N-1\}$.
         If this holds, we win.
         Else, we spend $N$ queries.
         Now we know $p_0 \neq 0$.
         Wait, if $p_0 \neq 0$, we also spend $N$ queries.
         Can we reuse these answers?
         Yes!
         If we know $p_0 | p_j$ for all $j$.
         We want to find 0.
         Let $z$ be 0-index. $p_0 | p_z = p_0$.
         So we are looking for $z$ such that $ans_z = p_0$.
         But we don't know $p_0$.
         However, we know $p_0$ is in the set of answers (at index $z$).
         Also $p_0$ is the value of the query with $z$.
         Let $S$ be the set of answers.
         If $p_0 \neq 0$, then $p_0$ must be present in $S$.
         Moreover, $p_0$ is the unique element $x \in S$ such that... ?
         Actually, if $p_0 \neq 0$, $p_0$ is likely the minimum? No.
         But notice $p_0 | p_k \ge p_0$.
         So the minimum of the answers set $S$ is $p_0$.
         WAIT!
         $p_0 | p_k \ge p_0$.
         This inequality holds for all $k$.
         So $\min(S) = \min_k (p_0 | p_k) \ge p_0$.
         Does equality hold?
         If $k=z$ (where $p_z=0$), $p_0 | 0 = p_0$.
         So $\min(S) = p_0$.
         YES!
         If we find the minimum of the answers to $?$ 0 $k$, that minimum IS $p_0$.
         This is HUGE.
         So, if we pick index 0, query everyone.
         Calculate min of answers. That is $p_0$.
         Let this value be $V$.
         We also know that the index $k$ that produced this minimum $V$ is the index where $p_k=0$.
         So we found the zero-index immediately!
         
         Wait, let's verify.
         Query: $Q_k = p_0 | p_k$.
         We know $p_0 \ge 0$.
         Since $p_k \ge 0$, $Q_k \ge p_0$.
         Equality holds iff $p_k \subseteq p_0$ (bits of $p_k$ are subset of $p_0$).
         Wait, $x | y = x \iff y \subseteq x$.
         So $Q_k = p_0 \iff p_k \subseteq p_0$.
         Since $p_0$ is in the permutation, 0 is in the permutation.
         $0 \subseteq p_0$ is always true.
         So for $k=z$ (where $p_z=0$), $Q_z = p_0$.
         Are there other $k$ such that $p_k \subseteq p_0$?
         Yes, potentially.
         If there are other submasks, $\min(S)$ might come from another $k$.
         However, if there are multiple minima equal to $p_0$, we can't uniquely identify $z$.
         BUT, we know 0 is unique.
         We need to find WHICH one is 0.
         We can iterate through all indices $k$ that gave the minimum value $V = p_0$.
         These are candidates for $z$.
         Among candidates for $z$ (where $p_k \subseteq p_0$), we need to find the one with value 0.
         Wait, if we know $p_0$, we can verify $p_k=0$ by querying $k$ against someone?
         If $p_k=0$, then $p_k | X = X$.
         If $p_k \neq 0$, then $p_k | X > p_k$ usually.
         Actually, if we have multiple candidates, we can distinguish them.
         Candidate $k$ is 0 iff for all other $j$, $p_k | p_j = p_j$.
         Or simpler: $p_k=0$.
         Since we have $V = p_0$.
         We can check if any $p_k$ (candidate) is 0.
         To check if $p_k=0$, we can just check if $p_k$ is submask of 0? No.
         We know 0 has no bits.
         If $p_k$ has no bits, it is 0.
         But we don't know bits of $p_k$ individually.
         However, we know $p_0$'s value $V$.
         Candidates are those $k$ where $p_k \subseteq V$ and $p_0 | p_k = V$.
         Actually, $p_k \subseteq V$ is implied by $p_0 | p_k = V$.
         So the candidates are a subset.
         To confirm $p_k=0$, we can query $k$ against some $u$.
         $p_k | p_u = V | p_k$? No.
         Just query $k$ against a known non-submask?
         If we pick $u$ such that $p_u$ is not submask of $V$?
         Wait, $p_0$ itself is $V$.
         If $p_k=0$, $p_k | p_0 = p_0 = V$.
         If $p_k \neq 0$, then $p_k \subseteq V$ and $p_k \neq 0$.
         So $p_k$ has at least one bit set.
         We can check if $p_k$ has bits set?
         Wait, we need to verify $p_k=0$.
         If we query $k$ against $z'$ (another candidate), we get $p_k | p_{k'}$.
         If both are proper subsets of $V$, result is union.
         Maybe easier:
         Once we have $p_0$ (the min of answers from 0-index), we know the index 0 is one of the indices yielding this min value.
         Let's say indices $C = \{ k \mid p_0 | p_k = p_0 \}$.
         $z \in C$.
         We need to find $z$.
         Since $p_k$ are distinct.
         Also $p_0$ is one of the values. $p_0$ is at index 0.
         So $p_0$ value is $V$.
         Index 0 has value $V$.
         Any other $k \in C$ has $p_k$ such that $p_k \subseteq V$.
         If we can differentiate $p_0$ (value $V$) from $0$ (value 0).
         Value $V$ is large. Value 0 is small.
         But we don't know values, only positions.
         Wait, $p_0$ is at index 0. So index 0 is IN $C$ obviously ($p_0 | p_0$ is undefined query, but $p_0$ value is consistent).
         Actually, we query $0$ against $k$.
         $ans_k = p_0 | p_k$.
         If $ans_k = p_0$, then $p_k \subseteq p_0$.
         This implies $p_k \le p_0$ numerically? No, bitwise.
         But if $p_k \subseteq p_0$, then $p_k$ could be 0.
         Is it possible to have multiple such $k$?
         Yes.
         But we want to find the one with value 0.
         If $p_k = 0$, then $0 \subseteq p_0$.
         If $p_k \neq 0$, then $p_k \ge 1$ (since integer).
         So if we can distinguish $p_k=0$ from others in $C$.
         Note that if $p_k=0$, then $p_k | x = x$.
         If $p_k \neq 0$, $p_k | x \neq x$ for $x$ with some new bit?
         But we don't know $x$.
         Wait, we know $p_0$ (value) is $V$.
         And we know $p_0$ is at index 0.
         So index 0 has value $V$.
         If $V > 0$, then index 0 is NOT the zero index.
         We are looking for $z$ such that $p_z=0$.
         Since $V \ge 0$.
         If $V=0$, then $p_0=0$, so $0$ is at index 0. Done.
         If $V > 0$.
         Then $0$ is at some other index $z \in C$.
         Also 0 is the only value that is NOT $\ge 1$.
         So 0 is the minimal value?
         Yes, but all $p_k \ge 0$.
         In $C$, we have $p_k \subseteq p_0$.
         All elements in $C$ have value $p_k \le p_0$ (value-wise? No bitwise subset implies $\le$ value-wise only if positive powers of 2 align, actually subset implies value $\le$ superset).
         Wait, $a \subseteq b \implies a \le b$. Proof: $b$ has all bits of $a$ plus maybe more. Sum of powers.
         So for all $k \in C$, $p_k \le p_0 = V$.
         So values in $C$ are in $[0, V]$.
         We know $p_z = 0$. So $z \in C$.
         We want to find $z$.
         Since $p_0$ is at index 0, and $p_0 = V$.
         If we check the other indices in $C$.
         If we pick $k \in C, k \neq 0$.
         Query $k$ against 0.
         $Q = p_k | p_0$. Since $p_k \subseteq p_0$, $Q = p_0 = V$.
         This doesn't help.
         But we can query $k$ against other elements.
         Actually, since we know $p_0=V$.
         We can query $k$ against a random index $u$ (not in $C$).
         $Q = p_k | p_u$.
         If $p_k = 0$, $Q = p_u$.
         If $p_k \neq 0$, $Q \ge p_k$.
         Also $p_u$ is likely larger than $p_k$ (since $p_u \ge 1$ usually).
         If $p_k=0$, $Q=p_u$.
         If $p_k \neq 0$, can $p_k | p_u = p_u$? Yes, if $p_k \subseteq p_u$.
         But we don't know.
         
         Actually, simplest check:
         We have candidates $C$. One is 0-index, rest are indices $k$ with $p_k \subseteq p_0$.
         We know 0 has NO bits set.
         Any $p_k \in C$ has bits set?
         If $k=0$, $p_k=V$. $V$ has bits set (since $V \ge 1$).
         If $k=z$, $p_k=0$.
         We need to find the one with 0 bits.
         We can pick $u$ such that $p_u$ has some bits not in $p_k$?
         Or just query $k$ against everyone? No, too expensive.
         Wait, we have $V$.
         $V$ is $p_0$.
         We can check bits of $V$.
         If we query $k$ against a pivot $x$ such that we know something about $x$.
         We don't know $x$.
         
         Wait, we have $N$ queries.
         If we just take the element $z$ that gave min value.
         There might be multiple such elements.
         If $p_0$ has many submasks.
         However, $N \le 2048$.
         Can we iterate through all $k$?
         If we have many candidates, we can use the remaining queries.
         Budget 4300. Used 2000.
         Remaining 2300.
         Candidates are subset of $N$.
         Number of submasks can be small?
         Worst case $p_0 = 2^{11}-1 = 2047$.
         Then ALL $p_k$ are submasks.
         So $C$ could be size $N$.
         In this case we can't distinguish 0 from others easily with few queries?
         But $p_k$ are distinct.
         So 0 is unique.
         If we query $k$ against a random $j$.
         $p_k | p_j = p_j$ ?
         If $p_k=0$, yes.
         If $p_k \neq 0$, usually no.
         With $2300$ queries, we can verify one candidate.
         But we might have many candidates.
         Wait.
         If $p_0=2047$, all answers are 2047.
         $p_0 | p_k = 2047$ for ALL $k$.
         So $C = \{1, \dots, N-1\}$.
         We have no info on which is 0.
         But wait!
         We know $p_0=2047$.
         So $p_k \subseteq 2047$. This is always true.
         So this strategy failed.
         
         Let's go back.
         Strategy refinement:
         Instead of fixing index 0, let's do something smarter.
         We want to find 0.
         0 is the unique element $x$ such that $x | y = y$ for some $y$.
         Actually $0 | y = y$ for ALL $y$.
         This is key.
         If we find a candidate $c$ and test it.
         Test: Query $c$ against $n-1$ random indices?
         Or all?
         If we query $c$ against $j$.
         If $c=0$, answer is $p_j$.
         If $c \neq 0$, answer is $p_c | p_j$.
         If we query $c$ against all $j$, and check if answers are distinct.
         As discussed, distinctness is strong indicator.
         Also range.
         If $p_c=0$, answers are $p_j$. Permutation of $1..n-1$ (excluding $p_c=0$).
         Wait, if $c=0$, answers are $\{ p_k : k \neq c \}$.
         The set $\{ p_k \}$ is a permutation of $0..n-1$. Excluding $p_c=0$.
         So answers are $\{ 1, \dots, n-1 \}$.
         This is extremely restrictive.
         The set of answers must be EXACTLY $\{ 1, \dots, n-1 \}$.
         (Sorted version).
         If we find any $c$ that produces this set, $c$ is 0-index.
         If we check $c$ against $n-1$ indices, cost $N$.
         If $c$ fails, cost $N$.
         We only have 4300 queries.
         We can check roughly 2 indices fully.
         This is a risk. $N$ up to 2048.
         Maybe we can optimize the check.
         To check if set of answers is $\{1, \dots, n-1\}$:
         We can check min, max, and number of answers? No.
         We can check if max is $n-1$.
         If $p_c=0$, max is $n-1$.
         If $p_c \neq 0$.
         Can max be $n-1$?
         Yes.
         Can we verify uniqueness?
         If $p_c \neq 0$, duplicates might occur.
         If we just sort answers, we can check if $ans[i] == i+1$.
         
         Wait, is there a randomized way?
         We can pick a random index $c$. Query $n-1$ answers.
         If valid, done.
         Expected iterations?
         $1/N$. $N=2048$. So expected 2048 queries.
         Budget 4300.
         $2048 < 4300$.
         So randomized strategy is feasible!
         Expected complexity is fine.
         Wait, what is the probability?
         If $p_c=0$, we succeed. Prob $1/N$.
         If we fail, we spent $N$ queries.
         We pick next random index.
         This is basically sampling until success.
         Expected trials $N$.
         Total queries $N \times (\text{cost per trial})$.
         Cost per trial is $N-1$ queries.
         So expected queries $N^2$. That's huge.
         My previous reasoning was wrong.
         I thought 1 trial is cheap.
         But one trial is $O(N)$ queries.
         So we can't do many trials.
         We can do AT MOST 2 full trials.
         So we cannot rely on randomization hoping for 0.
         We need a deterministic $O(N)$ approach.
         Actually, the question is finding 0.
         Can we find 0 in $O(N)$ deterministically?
         Yes.
         My strategy of "check index 0" gives us $p_0$ (value of $p_0$) from the minimum of answers.
         If $p_0 = 0$, we are done.
         If $p_0 \neq 0$, we know $p_0$'s value.
         Let $V = p_0$.
         We know $p_k \subseteq V \implies p_0 | p_k = V$.
         We want to find 0.
         We have identified $p_0$'s value.
         Let's try to reduce the search space.
         We know 0 is one of the $k$ such that $p_0 | p_k = V$.
         This set $C$ can be large.
         But we also know that $0$ has NO bits set.
         And all $k \in C$ have $p_k \subseteq V$.
         We can check specific bits.
         For each bit $b \in [0, 10]$.
         Consider mask $M_b = 2^b$.
         If $p_k$ has bit $b$ set, then $p_k | M_b = p_k$.
         Wait.
         If $p_k$ has bit $b$ clear (0), then $p_k | M_b > p_k$.
         So, if we can test if $p_k$ has bit $b$ set.
         If $p_k$ has bit $b$ set, then for any $j$, $p_j$ doesn't help directly?
         Wait, $p_k | p_j$.
         If $p_k$ has bit $b$, $ans$ has bit $b$.
         If $p_k$ doesn't have bit $b$, $ans$ depends on $p_j$.
         Since we can vary $j$.
         
         Let's simplify.
         We found $p_0 = V$.
         We want to find $z$ such that $p_z = 0$.
         We know $p_z \subseteq V$.
         In fact $p_z = 0$.
         So for any $b$, $p_z$ does not have bit $b$.
         Consider the bit $b=0$ (LSB).
         If $V$ has LSB set (odd).
         Then for $z$, $p_z$ has LSB 0.
         Then $V | p_z = V$ (since $0 | V = V$).
         For other $k$ with $p_k$ odd, $V | p_k$ will be $V | \text{odd} = \text{something}$.
         Wait, $p_0 | p_k = V$ means $p_k \subseteq V$.
         If $p_k$ is odd, then $p_k$ has LSB 1.
         If $p_k$ is even, LSB 0.
         We don't know $p_k$ bits directly.
         However, we know $p_k \subseteq V$.
         This means $p_k$ can have bit $b$ set only if $V$ has bit $b$ set.
         If $V$ has bit $b$ clear, then $p_k$ must have bit $b$ clear.
         This is helpful.
         If $V$ has bit $b$ set.
         Then $p_k$ can be even or odd (if bit $b=0$ is set/clear).
         We want to check if $p_k$ has ANY bit set.
         If $p_k=0$, it has none.
         If $p_k \neq 0$, it has at least one.
         Since $C$ contains only $k$ with $p_k \subseteq V$.
         If we pick a random $b$ such that $V$ has bit $b$ set.
         Check if $p_k$ has bit $b$ set.
         If $p_k=0$, bit $b$ is 0.
         If $p_k \neq 0$, it might be 0 or 1.
         This probabilistic check works.
         Since $p_k$ is uniformly distributed in some sense? No, permutation.
         But if $p_k \neq 0$, probability it has bit $b$ set is related to population of bit $b$.
         For large $N$, many numbers.
         
         Better idea:
         Use the answers $S = \{ p_0 | p_k \}_k$.
         We know $S$ contains $V$ (with multiplicity $|C|$).
         We also know $S$ contains other values.
         Let's pick a value $W \in S \setminus \{V\}$?
         Wait, $S$ is $\{ p_0 | p_k \}$.
         If $p_k$ is not a subset, $p_0 | p_k > p_0$.
         Let's pick $k$ such that $p_0 | p_k > p_0$.
         Let $U = p_0 | p_k$.
         Then $U > p_0$.
         Does this help?
         
         Let's restart with the $O(N)$ construction algorithm known for this problem.
         Problem source: Codeforces 1194F? No. It's X-OR.
         Problem is from "Educational Codeforces Round 89 (Rated for Div. 2)"?
         Problem name "X-OR"? No, maybe "Interaction" problem.
         Actually, this looks like "Problem B" of some round.
         Let's think from scratch.
         The goal is to guess permutation.
         Constraints: $N=2048$, Queries 4300.
         Algorithm:
         1. Pick a random index $i$. Ask against $j=i+1, \dots$.
         Wait, better:
         If we just maintain a list of indices `unknown`. Initially all $0..N-1$.
         We want to find the index with 0.
         Let's maintain a variable `found_zero_idx` initialized to -1.
         If `found_zero_idx` is not found, we try to find it.
         Can we find it by maintaining a set of candidates?
         Start with candidate set `Cands = {all indices}`.
         But we can't afford to query all.
         
         Wait, look at the constraint: 4300 queries.
         If we pick ANY index $i$, and query it against ALL others.
         This is $N$ queries.
         If $i$ happens to be 0, we win.
         If not, we spend $N$.
         Since we have $4300$ queries, we can afford approx 2 passes.
         But what if 0 is at index 1000?
         If we pick 0..500 randomly, we might miss.
         BUT, we can find 0 deterministically in $O(N)$?
         Yes.
         Here is the idea:
         Let's maintain a mask of bits we know for each index.
         Initially, we know nothing.
         But we know $p_k \in 0..N-1$.
         
         Actually, the simplest solution given the constraints is likely:
         Iterate $i$ from $0$ to $n-1$.
         Keep track of $cnt$ queries.
         Query $i$ against a SINGLE element $x$.
         If $p_i | p_x == p_x$ (i.e. $p_i$ submask of $p_x$) -> $p_i$ is small?
         If $p_i | p_x == p_i$ (i.e. $p_x$ submask of $p_i$) -> $p_x$ is small?
         Wait.
         If we find an index $i$ such that $p_i$ is 0.
         Then $p_i | p_x = p_x$ for ALL $x$.
         So if we query $i$ against any $x$, we get $p_x$.
         Wait, we don't know $p_x$.
         But if we have found 0 at index $z$, then querying $z$ against ALL $k$ gives the full permutation.
         This takes $N$ queries.
         So if we find $z$ quickly, we are done.
         So the problem reduces to: **Find the index $z$ such that $p_z = 0$ efficiently.**
         
         Let's check the property again.
         If we query $i$ against $j$.
         $p_i | p_j$.
         If $p_i=0$, result is $p_j$.
         If $p_j=0$, result is $p_i$.
         If neither, result is $p_i | p_j$.
         
         What if we just pick one index $x$, and assume it is 0.
         We query it against others.
         If $x$ is 0, we get correct permutation.
         If $x$ is not 0, we get garbage.
         Can we check validity quickly?
         Validity check:
         The answers must be $\{1, \dots, n-1\}$?
         No, if $x$ is not 0, answers are $\{ p_x | p_k \}$.
         We can check if answers are consistent with some permutation.
         E.g. if answers were $p_x | p_k$.
         If we know $p_x$, we can invert.
         But we don't know $p_x$.
         However, if $x=0$, we know $p_x=0$.
         
         Actually, there is a constructive approach.
         We can find 0 by splitting.
         Suppose we have set of candidates $S$.
         Initially $S = \{0, \dots, n-1\}$.
         Pick pivot $p \in S$.
         Query $p$ against a random $q \notin S$? No.
         Query $p$ against some $q \in S, q \neq p$.
         $res = p_p | p_q$.
         If $res = p_p$ (as value?) No we don't know values.
         But we know $res \ge p_p$ and $res \ge p_q$.
         Also $res = p_p$ if $p_q \subseteq p_p$.
         $res = p_q$ if $p_p \subseteq p_q$.
         If $res$ is "small", maybe?
         
         Wait, here's a solid $O(N)$ solution:
         1. Maintain a mask of bits for the current candidate 0?
            Let's track the MSB of the unknown 0.
            Suppose we suspect 0 is in range $[0, m]$.
            Split into $[0, m/2]$ and $[m/2, m]$.
            Can we test which half contains 0?
            We need to check elements.
            Take two indices $i, j$. Query $i, j$.
            If $i$ is 0, result is $p_j$.
            This doesn't help distinguishing.
            
         Wait, I remember a similar problem.
         If we pick index 0.
         We calculate $min\_ans = \min_k (p_0 | p_k)$.
         As derived, $min\_ans = p_0$.
         So we know $p_0$.
         Let $P_0 = p_0$.
         Now we need to find $z$ where $p_z=0$.
         We know $p_z=0 \implies p_0 | p_z = p_0$.
         So $z$ is a candidate if $p_0 | p_z = p_0$.
         We can identify all such candidates $z$.
         Let $C = \{ k \mid p_0 | p_k = P_0 \}$.
         One of them is 0 (if $P_0 > 0$).
         If $P_0=0$, then $0$ is at index 0.
         So we narrowed down to set $C$.
         How to eliminate candidates?
         If $z \in C$, then $p_z \subseteq P_0$.
         We want to check if $p_z=0$.
         If we can find a candidate $z$ and verify $p_z=0$ in constant time?
         Verification: Query $z$ against some $u$ not in $C$.
         $p_z | p_u = P_u$ (if $z=0$).
         If $p_z \neq 0$, $p_z | p_u$ usually differs from $p_u$ unless $p_z \subseteq p_u$.
         But since $p_z \subseteq P_0$ and $u$ random.
         If $p_z=0$, $p_z | p_u = p_u$.
         If $p_z \neq 0$, $p_z | p_u$ likely $> p_u$ (unless submask).
         Since we want to distinguish 0 from other submasks.
         If $C$ is small, we can check all.
         If $C$ is large (e.g. $P_0=2^{11}-1$), we are stuck.
         But if $P_0$ is large, then almost all $p_k$ are submasks.
         Wait, if $P_0$ is large, say $2^{11}-1$.
         Then ALL $p_k$ are submasks.
         So $C = \{1, \dots, n-1\}$.
         We need to find 0 among them.
         But we know $p_k \subseteq P_0$.
         We don't know bits.
         However, if we query $k$ against $u$.
         $Q = p_k | p_u$.
         If $p_k=0$, $Q=p_u$.
         If $p_k \neq 0$, $Q \ge p_u$.
         So we check if $Q == p_u$? No, we don't know $p_u$.
         But we can pick a reference $ref$.
         Suppose we pick an element $u$ that we suspect is NOT 0.
         Then for $z=0$, $p_z | p_u = p_u$.
         For other $k$, $p_k | p_u > p_u$?
         Actually, if we pick $u$ such that $p_u$ has high value?
         Wait, if $C$ is full, it means $p_k \subseteq P_0$ for all $k$.
         This implies all numbers in permutation are submasks of $P_0$.
         If $P_0$ has all bits set (like 2047), this is trivial.
         But we need to find 0.
         Wait, if $P_0 = 2047$, then $p_k$ can be ANYTHING in $0..2047$.
         So we have NO reduction.
         This means if $P_0$ is large, finding 0 is HARD?
         Wait, if $P_0=2047$, we know $p_0=2047$.
         We queried $0$ against everyone and got 2047 for everyone.
         So we know $p_0 | p_k = 2047$ for all $k$.
         This tells us nothing about $p_k$ except $p_k \subseteq 2047$ (always true).
         Wait, if $p_0 | p_k = 2047$ for all $k$, we lost a lot of info?
         Yes, because $p_0$ was a "cover".
         So picking a pivot with many bits set is BAD.
         We should pick a pivot with FEWER bits set to reduce $C$.
         Ideally we pick index 0.
         If $p_0$ has few bits set, $C$ is small.
         If $p_0$ has many bits set, $C$ is large.
         But $p_0$ is a random element of permutation (since we don't know which).
         So probability of picking a low-value number is roughly uniform.
         But we need a deterministic guarantee.
         Can we force $p_0$ to be small?
         No.
         
         Wait, is there a solution that iterates bits?
         Let's check the bits.
         We know 0 has NO bits.
         So for bit $b$, the element 0 is 0 at bit $b$.
         Other elements might be 0 or 1.
         If we query index $i$ against index $j$.
         If result has bit $b$, it could be from $p_i$ or $p_j$.
         If result doesn't have bit $b$, then NEITHER has it.
         Wait!
         $A | B$. Bit $b$ is 0 iff $A$ has 0 AND $B$ has 0.
         If we find a pair $(i, j)$ such that $p_i | p_j$ has bit $b$ as 0.
         Then $p_i$ has bit $b$ as 0, AND $p_j$ has bit $b$ as 0.
         We can maintain a set of candidates for 0.
         $ZerosCandidates$ initially all indices.
         For each bit $b \in 0..10$:
             For each pair $(i, j)$? No too many pairs.
             We need to filter $ZerosCandidates$.
             Pick a representative $i$.
             Query $i$ against $j$.
             If result bit $b$ is 0, then both have bit $b$ as 0.
             If result bit $b$ is 1, then at least one has it.
             
             If we maintain $Candidates$.
             Pick $i \in Candidates$.
             Pick $j \notin Candidates$?
             If $j$ is known to not be 0?
             Actually, if we maintain $Candidates$ as those that CAN be 0.
             Initially $Candidates = \{0, \dots, n-1\}$.
             Pick $u \in Candidates$.
             We want to eliminate $u$ if it's not 0.
             How?
             If we find an index $v$ that we are confident is NOT 0.
             Then check $u$ against $v$.
             $Ans = p_u | p_v$.
             If $Ans == p_v$, then $p_u \subseteq p_v$. Doesn't eliminate $u$ (could be 0).
             Wait, $0 | p_v = p_v$.
             If $p_u \neq 0$ but submask, still looks like 0 w.r.t $v$.
             If $Ans > p_v$, then $p_v$ does not contain all bits of $u$.
             Wait, we don't know $p_v$.
             
             Backtrack to $O(N)$ solution for X-OR.
             Standard solution:
             Find an element $x$ such that it is 0.
             If $p_0=0$, we are lucky.
             Otherwise, $p_0$ is some value.
             We calculated $p_0$ by min of queries from index 0.
             Let $P_0$ be the value of index 0.
             Then $p_0 | p_k = P_0$ for all $k$ where $p_k \subseteq P_0$.
             If $P_0$ is small, then only few numbers are subsets.
             Actually, $p_k \in 0..N-1$.
             Subsets of $P_0$ are restricted.
             If $P_0$ is e.g. 3 (binary 011), subsets are 0, 1, 2, 3.
             If $P_0$ is 1, subsets are 0, 1.
             If $P_0=0$, subset is 0.
             
             The issue is when $P_0$ is large (covers many bits).
             But we chose index 0 randomly.
             Is it possible to ensure $P_0$ is small?
             No, 0..N-1 is range.
             However, we can iterate indices $i=0, 1, \dots$ and pick the one with MINIMUM $p_i$?
             No, we don't know $p_i$.
             
             Wait. The constraint 4300 queries is generous.
             Maybe we can use randomization on which index we probe.
             Instead of fixed index 0, we cycle through indices?
             Or just pick a random index.
             If it fails (many candidates), try another.
             Wait, failing means we didn't find 0 or 1.
             Actually, finding 0 is the bottleneck.
             If we find $P_0$ (value of index 0).
             Then we can compute the set of possible values for $p_k$: those $v$ such that $v \subseteq P_0$ (if $p_k \subseteq P_0$).
             Wait, if $p_0 | p_k > P_0$, then $p_k$ has a bit NOT in $P_0$.
             So for any $k$, $p_k$ is constrained by $p_0 | p_k$.
             If $p_0 | p_k = P_0$, then $p_k \subseteq P_0$.
             If $p_0 | p_k = Y > P_0$, then $p_k$ contributes extra bits.
             
             Key Insight:
             If we query index $i$ against index $j$.
             We get $p_i | p_j$.
             If we assume $p_i$ is unknown.
             
             Let's use the fact that $0$ is unique.
             We have $O(N)$ budget (4300). $N \le 2048$.
             We can just try to find 0 by binary search on BITSET?
             No.
             
             Wait, look at the scoring. Fewer queries better.
             But limit is 4300.
             Is there an $O(N)$ deterministic solution?
             Yes.
             Algorithm:
             1. Pick index 0. Compute $P_0 = \min_{k \neq 0} (p_0 | p_k)$.
                This takes $N$ queries.
                Identify candidates $C = \{ k \mid p_0 | p_k = P_0 \}$.
                One of them is the 0-index.
                Note that $p_0$ is at index 0.
                So $p_0$ itself is in $C$ conceptually (since $P_0 \subseteq P_0$).
                Actually, $C$ contains indices $k$ where $p_k \subseteq P_0$.
                Since $p_0 = P_0$, index 0 is always in $C$ (or rather satisfies condition).
                If $P_0 = 0$, then $p_0=0$. Index 0 is 0-index.
                If $P_0 > 0$, then $p_0 \neq 0$. So 0-index is some $z \in C$.
                Now we need to pick $z$ from $C$.
                How many elements in $C$?
                $C$ contains all $k$ such that $p_k \subseteq P_0$.
                This depends on $P_0$.
                If $P_0$ is sparse (few bits), $C$ is small.
                If $P_0$ is dense, $C$ is large.
                But $P_0$ is the value of $p_0$.
                Since $p_0$ is from permutation, and values are $0..n-1$.
                Wait! The values $p_k$ are uniform in $0..n-1$.
                $P_0$ is just one of them.
                So $P_0$ can be anything.
                However, we can try to improve $P_0$.
                Wait.
                If we query $0$ against $k$ and get $Y_k$.
                $Y_k = P_0$ if $p_k \subseteq P_0$.
                $Y_k > P_0$ otherwise.
                The value $P_0$ is fixed.
                
                Let's refine finding $z$.
                We need to find $z \in C$ such that $p_z=0$.
                For $z \in C$, $p_z \subseteq P_0$.
                If we pick a bit $b$ such that $P_0$ has bit $b$ set.
                Check if any $k \in C$ has bit $b$ set.
                We can check this by querying $k$ against someone?
                Actually, for each $k \in C$, we have data $p_0 | p_k = P_0$.
                We want to distinguish 0 (value 0) from others (values $>0$).
                0 is the only value that has NO bits set.
                If we check a candidate $k \in C$ against $P_0$ (the value, wait index is 0).
                $p_k | p_0 = P_0$.
                This doesn't help.
                We need to find a $u$ such that $p_u$ has some bit NOT in $p_k$.
                Or just $u$ such that $p_k | p_u > p_u$?
                If $p_k=0$, then $p_k | p_u = p_u$.
                If $p_k \neq 0$, then $p_k | p_u \ge p_k$.
                Also $p_u \ge 0$.
                If $p_u$ is small, say 1.
                Then $p_k | 1$ might be $1$ or larger.
                If $p_k$ has bit 0 set (is odd), $p_k | 1 = p_k$.
                If $p_k$ even, $p_k | 1 = p_k + 1$.
                
                We can iterate through $C$ and check each.
                For each $k \in C$, query against all others? No.
                If $|C|$ is large, we cannot check all.
                
                Is it possible that $C$ is small?
                On average, $P_0$ is around $N/2$.
                Average number of submasks of a number $X$ is roughly $N / \sqrt{X}$? No.
                It's related to Hamming weight.
                If $P_0$ is chosen uniformly, the number of submasks in $0..N-1$ is roughly $\sum_{X} \text{count}(X)$.
                Wait, for $N=2048$, numbers are 11-bit.
                Average submasks of random 11-bit number is $2^{11}/2 = 1024$.
                So $|C|$ is likely large.
                Wait, if $|C|$ is large, this approach is bad.
                Unless...
                Is there a way to reduce $|C|$?
                Maybe pick index 0 such that $P_0$ is small?
                How?
                We don't know $P_0$ beforehand.
                
                BUT!
                Look at the query results.
                If we query $i$ against all $j$, we get answers $A_j$.
                We want $P_i = \min(A_j)$.
                If we pick $i$ such that $P_i$ is minimized over all $i$?
                We want to minimize $P_i = p_i$.
                But $p_i$ is unknown.
                Wait.
                If $p_i$ is 0, then $P_i = 0$.
                If $p_i$ is 1, $P_i = 1$.
                If we find $i$ such that $P_i$ is minimal.
                Can we compare $P_i$ and $P_j$?
                No.
                
                Wait. 4300 queries for 2048.
                This is roughly 2 per element.
                If we just do this:
                Iterate $i$ from $0$ to $N-1$.
                Query $i$ against $i+1$.
                Store $p_i | p_{i+1}$.
                This gives us partial info.
                
                Let's rethink the $O(N)$ approach for 0.
                Maybe the set $C$ is not that bad?
                Wait, if $P_0$ has high bits set, it covers everything.
                But we want to find 0.
                Actually, 0 is the ONLY number that is 0.
                If $C$ contains indices with values $\subseteq P_0$.
                If $P_0$ is dense, it contains many values.
                However, if $P_0$ is dense, then $p_0$ is dense.
                If $p_0$ is dense, it is likely not 0.
                Wait, if $p_0=2047$, then $p_k \subseteq 2047$ is always true.
                So $C=\{all\}$.
                So we can't filter.
                However, if $p_0$ is 2047.
                We have $p_0=2047$.
                Then for any $k$, $p_0 | p_k = 2047$.
                We get NO info about $p_k$.
                We just get 2047 repeatedly.
                So we wasted $N$ queries.
                This suggests we shouldn't start with arbitrary $i$.
                We should try to find an element with SMALL value first.
                How?
                If we query $i$ against $j$, and $p_i | p_j$ is small.
                Then $p_i$ and $p_j$ are small.
                If $p_i | p_j = 1$, then $p_i, p_j \in \{0, 1\}$.
                If $p_i | p_j = 2$, then $p_i, p_j \in \{0, 2\}$.
                Basically small OR implies small operands.
                
                Strategy:
                1. Find the minimum possible value of $p_i | p_j$ over all pairs.
                   Since $p$ is perm of $0..n-1$, $p$ contains 0 and 1.
                   $0 | 1 = 1$.
                   So min possible answer is 1.
                   Can we achieve 1? Yes, with 0 and 1.
                   Can we achieve 0? No, distinct indices.
                   
                   So if we find a pair $(i, j)$ such that $p_i | p_j = 1$.
                   Then $\{p_i, p_j\} = \{0, 1\}$.
                   One is 0, one is 1.
                   So we found a pair containing 0!
                   Then we check which one is 0.
                   We can do this by checking against others?
                   Or just query $i$ against all $k$ and $j$ against all $k$?
                   
                   So, Algorithm:
                   1. Find a pair $(i, j)$ with minimal OR sum.
                      We can pick random pairs.
                      Or just iterate.
                      Pick $i=0$. Query against $j=1 \dots n-1$.
                      Find min OR sum.
                      Let min val be $M$.
                      If $M=1$. Then the pair giving 1 involves 0 and 1.
                      Say $i$ and $j$ give 1.
                      Then $\{p_i, p_j\} = \{0, 1\}$.
                      One of them is 0.
                      We can distinguish by querying $i$ against a random $k$.
                      If $p_i=0$, $p_i | p_k = p_k$.
                      If $p_i=1$, $p_i | p_k \ge 1$.
                      We can check counts?
                      Actually, if we know $\{p_i, p_j\} = \{0, 1\}$.
                      Then we just need to check if $p_i=0$.
                      Query $i$ against $k$. Get $res$.
                      If $p_i=0$, $res = p_k$.
                      If $p_i=1$, $res = 1 | p_k$.
                      If $p_k$ has bit 0 (odd), $1|p_k = p_k$.
                      If $p_k$ has bit 0 (even), $1|p_k = p_k+1$.
                      So if $res$ is odd, inconclusive.
                      If $res$ is even, then $p_i$ must be 0?
                      Wait, $p_k$ can be even. $p_k | 1$ is odd.
                      So if $res$ is odd, could be $p_k$ or $p_k+1$.
                      If $res$ is even, impossible for $p_i=1$ (since $1|x$ has LSB $1 \oplus LSB(x)$).
                      Wait, $1 = 01_2$. $1 \oplus x$.
                      If $x$ is even (00...), $1|x$ ends in 1.
                      If $x$ is odd (01...), $1|x$ ends in 1.
                      So $1 | x$ is ALWAYS odd.
                      Thus, if we query $i$ (candidate 1) against $k$, result MUST be odd.
                      If we query $z$ (candidate 0) against $k$, result is $p_k$. Can be even or odd.
                      So if we get EVEN result from $i$ against $k$, then $i$ cannot be 1.
                      Thus $i$ must be 0.
                      
                      So, Algorithm refined:
                      1. Select two indices $a, b$ that minimize $p_a | p_b$.
                         We can do this by simple scan.
                         Pick random $a$. Query $a$ against all $k$. Find min answer $M$.
                         Let $b$ be the index producing $M$.
                         Check if $M=1$.
                         If $M=1$, then $\{p_a, p_b\} = \{0, 1\}$.
                         We found the pair containing 0.
                         Now we need to check which is 0.
                         We can query $a$ against a third index $c$ (random or specific).
                         If $a$ gives an even result (against $c$), then $a$ is 0.
                         Wait, $a$ gives even result implies $p_a$ does not have LSB set?
                         Wait, if $p_a=1$, $1|p_c$ is odd.
                         So $p_a$ cannot be 1 if we get even result.
                         So if $p_a | p_c$ is even, $p_a \neq 1$.
                         Since we know $p_a \in \{0, 1\}$, $p_a=0$.
                         Is it possible to get even result?
                         Yes, if $p_c$ is even.
                         Does permutation contain evens? Yes, if $n \ge 2$.
                         So with probability $\approx 0.5$ we get even.
                         If we get odd, we can try another $c$.
                         If we try $N/2$ candidates and all odd? Unlikely if we pick random.
                         Actually, if $p_a=0$, $0|p_c = p_c$. $p_c$ can be anything.
                         So finding even result confirms $p_a=0$ (because $p_a=1 \implies$ always odd).
                         If we don't find even result after some tries?
                         We can also query $b$ against $c$.
                         If $p_b=1$, $1|p_c$ is odd.
                         So if $p_b=0$, $0|p_c$ can be even.
                         Wait, if $M=1$, one is 0, one is 1.
                         Query both against $c$.
                         At least one will produce an even result?
                         If $p_c$ is even, then 0-produces-even, 1-produces-odd.
                         So 0 produces even.
                         If $p_c$ is odd, 0-produces-odd, 1-produces-odd. Both odd.
                         So if $p_c$ is odd, both look like 1.
                         So we need to find an index $c$ with even $p_c$.
                         There are $n/2$ such indices.
                         So prob of picking even $c$ is high.
                         
                         Once we find which is 0, say index $z$.
                         Then query $z$ against all $k$ to get full permutation.
                         Cost: $N$ (find pair) + 2 (verify) + $N$ (solve). Total $2N+2$.
                         With $N=2048$, $4098$. Within 4300.
                         
                         What if min OR is not 1?
                         Possible?
                         Values $0..n-1$.
                         Contains 0. So $0|x=x$.
                         Min value is $\min(x)$. Since 1 is in perm (if $n \ge 2$).
                         So $0|1 = 1$.
                         So 1 is achievable.
                         Is it possible min > 1?
                         No, because 0 and 1 exist.
                         So min is ALWAYS 1 (for $n \ge 2$).
                         
                         Wait, what if $n=1$? Constraint $n \ge 3$.
                         So min is 1.
                         So we WILL find a pair with OR sum 1.
                         Wait, is it unique?
                         Pair giving 1 must be $\{0, 1\}$.
                         Are there other pairs?
                         $p_i | p_j = 1 \implies p_i, p_j \le 1 \implies \{p_i, p_j\} = \{0, 1\}$.
                         So unique pair (indices of 0 and 1).
                         So we just need to find the pair with OR sum 1.
                         But there might be pairs with OR sum $>1$.
                         Wait, we need to find the pair with OR sum 1.
                         Can we easily find it?
                         If we query $i$ against all $j$.
                         If we find any $j$ such that $p_i | p_j = 1$, then we are done!
                         We just need to find such pair.
                         Since 0 and 1 exist, there exists such pair.
                         We need to find it fast.
                         Random scan?
                         Or just scan $i$ from $0$ to $N-1$.
                         For each $i$, query against $j=0 \dots N-1$? Too slow.
                         Wait, we have limited queries.
                         Can we find index 0 faster?
                         Actually, finding index 0 is equivalent to finding a pair $\{0, 1\}$.
                         If we query $i$ against a specific $j$ (say $0$).
                         If $p_0 | p_1 = 1$.
                         So we need to find $i$ such that $p_i | p_j = 1$ for SOME $j$.
                         This is hard.
                         
                         Wait, if we query $i$ against a random $j$.
                         Prob of hitting 0 or 1 is small?
                         $2/N$.
                         So $N/2$ tries.
                         Each try costs 1 query.
                         $1000$ queries.
                         We find $i$ such that $\{p_i, p_j\} = \{0, 1\}$.
                         Then verify 0. Cost $O(1)$.
                         Then solve full. Cost $N$.
                         Total $O(N)$.
                         Wait, is it guaranteed to find in $N/2$?
                         If we pick random pairs.
                         Expected queries $N/2 \approx 1024$.
                         Plus verification and solve ($N=2048$).
                         Total $\approx 3072$. Safe!
                         Wait, this assumes we find it quickly.
                         But what if we pick pairs that are NOT 0 and 1?
                         Then we get value $>1$.
                         We keep searching.
                         How to know we found 0 and 1?
                         The OR sum will be 1.
                         Any pair involving 0 and 1 gives 1.
                         Any other pair gives $\ge 1$.
                         Wait, $0|2=2$. $1|2=3$.
                         So if we get 1, we found 0 and 1.
                         Is it possible to get 1 from other pairs?
                         $x | y = 1 \implies x,y \in \{0, 1\}$.
                         Since 0 and 1 are unique in permutation, only indices holding 0 and 1 can produce 1.
                         So finding 1 identifies the pair $\{0, 1\}$.
                         
                         So Algorithm:
                         1. Loop: Pick random index $i$.
                            Query $i$ against random $j$.
                            If result is 1, we found pair. Break.
                         Wait, checking 1 is fast.
                         But we might loop many times?
                         Wait, 0 and 1 are distinct.
                         We just need one pair.
                         If we pick $i$ random, $j$ random.
                         Chance that $\{p_i, p_j\} = \{0, 1\}$?
                         $2/N$ if we order doesn't matter?
                         Number of pairs is $N(N-1)/2$.
                         Number of pairs $(i,j)$ such that $\{p_i, p_j\}=\{0,1\}$ is 2.
                         So prob is $2 / N(N-1)$. Extremely small for $N=2048$.
                         Wait, we can't pick random pair and hope.
                         We need to FIND 0 and 1.
                         Wait, I forgot about the $O(N)$ scan.
                         If we scan $i=0..N-1$.
                         For a fixed $i$, query against ALL $j$.
                         If $i$ is 0, we find 0!
                         Wait, if $i=0$, queries against $j$ give $p_j$.
                         Do we find 1?
                         We find 1 in the answers.
                         If we see 1 in answers.
                         Then $p_j=1$ for some $j$.
                         Then we know $i$ is 0.
                         Because if $i \neq 0$, $p_i | p_j$ would be $\ge 1$.
                         But we would see $p_i$ (at $j=z$).
                         If $p_i > 0$, $p_i$ might be 1?
                         Wait, if $i$ is 0. Answer set is $\{1..n-1\}$. Contains 1.
                         If $i$ is not 0. Answer set $\{ p_i | p_k \}$.
                         Contains 1 iff exists $k$ such that $p_i | p_k = 1$.
                         This implies $p_i=0$ or ($p_i=1$ and $p_k=0$).
                         So if $i$ is NOT 0, but $p_i=1$. Then answer is 1.
                         So if we see 1 in answers, $i$ could be 0 or 1.
                         If we see 1, and $i=1$ (i.e. $p_i=1$).
                         Then we found $i$ is 1.
                         If we see 1, and $i=0$ (i.e. $p_i=0$).
                         Then we found $i$ is 0.
                         So if we pick $i$ and scan all $j$.
                         If answers include 1.
                         Case 1: $i$ is 0. Then $p_i | p_k = p_k$. We see all values $1..n-1$.
                         Case 2: $i$ is 1. Then $p_i | p_k$.
                         $p_i | p_0 = 1$.
                         Other answers $1 | p_k \ge p_k$.
                         We see 1.
                         We need to distinguish $i=0$ from $i=1$.
                         If $i=0$, answers are permutation of $1..n-1$.
                         If $i=1$, answers are $\{1\} \cup \{1|p_k\}$.
                         If we check if answers are permutation of $1..n-1$.
                         If yes -> $i=0$.
                         If no -> $i=1$ (most likely). Or $i$ neither but 1 appeared?
                         Wait, if $i$ neither, can 1 appear?
                         If $p_i | p_k = 1$.
                         Requires $p_k=0$ and $p_i \subseteq 1$ (so $p_i=1$).
                         So if $i \neq 1$, then $p_i \ge 2$. Then $p_i | p_k \ge 2$.
                         So 1 never appears in answers.
                         So:
                         If answers contain 1 -> $i$ is either 0 or 1.
                         If answers do not contain 1 -> $i \neq 0, 1$.
                         So we can filter $i$.
                         We just need to find if 1 is in answers.
                         We can check this quickly!
                         Just check if min answer is 1.
                         Since all answers $\ge 1$.
                         Min answer is 1 iff 1 is in answers.
                         So:
                         Pick $i$.
                         Query all $j$. Compute min.
                         If min > 1, $p_i \neq 0, 1$.
                         If min == 1, $p_i \in \{0, 1\}$.
                         If we encounter $min > 1$, we eliminate $i$.
                         We continue until we find $i$ with $min=1$.
                         Since there is at least one such $i$ (index of 0 and index of 1).
                         Worst case we skip many.
                         Skipping costs $N$ queries.
                         We can only afford 2 skips.
                         BUT, notice $p_i \in \{0, 1\}$ condition.
                         If we find $min=1$, we know $p_i \in \{0, 1\}$.
                         How to decide if 0 or 1?
                         We just check if answers are permutation of $1..n-1$.
                         This requires storing all answers or re-checking.
                         But we already queried them.
                         So:
                         1. Iterate $i$ from $0 \dots N-1$.
                            Query $i$ against all $j \neq i$.
                            Check min.
                            If min > 1: Continue (discard $i$).
                            If min == 1:
                               Check if answers contain ALL numbers $1..N-1$.
                               If yes -> $i=0$. Success.
                               If no -> $i=1$. Success (since $p_i=1$).
                        Complexity:
                        In worst case, we check all $N-2$ indices before finding 0 or 1.
                        Total queries $2N$ approx.
                        $2N = 4096$. Fits in 4300!
                        Wait, is worst case $N$ checks?
                        Yes, if 0 and 1 are at end of iteration order.
                        But $N \le 2048$.
                        $2048 \times 2048$ is too big.
                        Wait, my complexity calculation was:
                        Loop $N$ times. Each iteration $O(N)$ queries. Total $O(N^2)$.
                        We need to stop early.
                        Do we need to iterate ALL?
                        No.
                        Just find ONE index with $p_i \in \{0, 1\}$.
                        Since $p_i \in \{0, 1\}$ is rare (prob 2/N).
                        Most $i$ will have $min > 1$.
                        So we will do $O(N)$ queries for each check? No.
                        Wait, if we query $i$ against ALL $j$, we pay $N$.
                        If we just check min? We need to query everyone to find min.
                        So cost is $N$.
                        So we do $O(N)$ queries per candidate.
                        If we have $N$ candidates, total $O(N^2)$.
                        This is bad.
                        BUT!
                        We only need to find 0 or 1.
                        We don't need to check everyone.
                        Wait, we don't know WHERE 0 or 1 are.
                        Can we check fewer than $N$ candidates?
                        No, we don't know.
                        
                        Wait, the question says "You can ask queries".
                        There are 4300 queries.
                        $N=2048$.
                        We can query $i$ against all $j$ ONCE.
                        Then we know $p_i$ and $p_i | p_j$ for all $j$.
                        Wait.
                        If we query $i$ against ALL $j$.
                        We get answers.
                        If $p_i=0$, answers are permutation of $1..N-1$.
                        If $p_i=1$, answers contain 1.
                        If $p_i \ge 2$, answers $\ge 2$.
                        So we need to find $i$ such that $min(ans)=1$.
                        But to find $min(ans)$ we need all answers.
                        So one full scan is $N$.
                        So we can do at most 2 full scans.
                        If $p_0=0$, done.
                        If $p_0 \ge 2$, we wasted $N$.
                        We need to find 0 or 1 among the rest.
                        But wait!
                        When we scanned $i=0$ and got answers.
                        We saw that for all $j$, $p_0 | p_j \ge p_0$.
                        And we found $min = p_0$.
                        So we learned $p_0$.
                        Let $V = p_0$.
                        If $V > 1$, then $0$ is NOT at $i=0$, and $1$ is NOT at $i=0$.
                        We know $0$ and $1$ are somewhere else.
                        Do we have more info?
                        We have answers $ans_j = p_0 | p_j$.
                        If we find $z$ such that $ans_z = V$ (since $p_0 | 0 = p_0 = V$).
                        Then $z$ is candidate for 0.
                        Are there other candidates?
                        Candidates are $\{ k \mid p_0 | p_k = p_0 \}$.
                        If $p_0$ has many submasks, we have many candidates.
                        But we want to find 0.
                        If $p_0$ is small, few candidates.
                        If $p_0$ is large, many candidates.
                        But if $p_0$ is large, it has many bits.
                        Wait, if $p_0$ is large, say $2047$.
                        Then we get nothing from answers.
                        But we know $p_0=2047$.
                        We still need to find 0.
                        We can try to scan another index $i=1$.
                        But we have 2300 queries left.
                        We can do 1 more full scan.
                        If $p_1=0$ or $p_1=1$, we are good.
                        If not, we failed again.
                        Is it possible $p_0, p_1 \ge 2$? Yes.
                        So this approach is $O(N)$ per failure.
                        We can't afford failures.
                        
                        Wait, I missed the constraint logic.
                        Actually, finding 0 using $O(N)$ queries is solvable.
                        The key is finding 0 without checking all pairs.
                        
                        What if we just pick index 0 and find min?
                        We know $p_0$.
                        If $p_0=0$, done.
                        If $p_0=1$, done (we found 1).
                        Wait, if $p_0=1$, we need to find 0.
                        We know $p_0=1$.
                        We scanned all $j$.
                        We found index $k$ such that $ans_k = 1$.
                        Then $p_k=0$.
                        Because $1 | p_k = 1 \implies p_k=0$.
                        So if $p_0=1$, we identify 0 immediately from the scan results!
                        So if $p_0 \in \{0, 1\}$, we are DONE in one scan.
                        The only problematic case is $p_0 \ge 2$.
                        If $p_0 \ge 2$, we wasted $N$ queries.
                        But we know $p_0$.
                        And we have many candidates for 0?
                        Wait, if $p_0 \ge 2$, can we eliminate candidates?
                        We know 0 is one of the candidates $\{ k \mid p_0 | p_k = p_0 \}$.
                        Since $p_0$ is known, we know its value.
                        Let's check the candidates.
                        If $p_0$ is small, few candidates.
                        If $p_0$ is large, many.
                        But notice that 0 is a specific number.
                        If $p_0$ is large, it has many submasks.
                        BUT, if $p_0$ has many bits set, it implies $p_0$ is "close" to $2^{11}-1$.
                        Actually, we can check candidates one by one?
                        Suppose candidates are $C$.
                        We want to find $z \in C$ such that $p_z=0$.
                        If we pick $k \in C$.
                        Query $k$ against some $u$ not in $C$ (or any random).
                        $Res = p_k | p_u$.
                        If $k=0$, $Res = p_u$.
                        If $k \neq 0$, $Res \ge p_k$.
                        We can try to find $u$ such that $p_u$ is small?
                        Or just pick ANY $u$.
                        If we query $k$ against many $u$, we can check consistency?
                        Actually, if $p_k=0$, then for ALL $u$, $p_k | p_u = p_u$.
                        So answers are copies of $p_u$.
                        If $p_k \neq 0$, then answers will be different.
                        So if we query $k$ against all $u$ and check if answers match $p_u$?
                        No, we don't know $p_u$.
                        
                        Wait, if $p_0$ is small, we are happy.
                        If $p_0$ is large, we are worried.
                        But $p_0$ is random.
                        Actually, 0 and 1 are present.
                        We can try to find them using random queries.
                        With $N=2048$, probability that random index is 0 is $1/2048$.
                        If we scan it (cost 2048), prob of success is 1/2048.
                        Expected trials to hit 0 is 2048.
                        Expected cost $2048 \times 2048$ is too high.
                        
                        Wait. We have 4300 queries.
                        If we do $N$ queries.
                        Maybe we can just check $N/2$ indices partially?
                        No.
                        
                        Let's go back to the idea:
                        We need to find 0.
                        We know $p_0 | p_j \ge p_0$.
                        So $p_0$ is a lower bound for answers of index 0.
                        Also for ANY index $i$, $p_i$ is lower bound for answers.
                        So $\min_i ( \min_{j \neq i} (p_i | p_j) ) = \min_i p_i$.
                        The value of the absolute minimum of all answers (over all pairs) is $\min(0, \text{others}) = 0$?
                        No, $p_i | p_j \ge p_i \vee p_j$.
                        If we take pair $(z, k)$ where $p_z=0$. $0 | p_k = p_k$.
                        Min value over all pairs is $\min_k p_k = 0$? No.
                        If we pair 0 with anything $k$, result is $p_k$.
                        Min of these is $\min_{k} p_k$.
                        Since 0 is in permutation, $0$ is a value.
                        Wait, $p_k$ are values. Min of values is 0.
                        So we can pair 0 with 0? No distinct indices.
                        So pair $(z, y)$ gives $p_y$.
                        Since $p_z=0$ is unique.
                        The values obtained from pairing with 0 are $\{ p_k : k \neq z \} = \{ 1, \dots, n-1 \}$.
                        Min of these is 1.
                        So the MINIMUM possible answer we can EVER get is 1.
                        (Since any pair involves at least one non-zero value, OR sum $\ge 1$).
                        Wait, if $p_k$ contains 1, and we pair with 0, we get 1.
                        So 1 is reachable.
                        If we pick pair $(i, j)$ and get 1.
                        Then one of them is 0 (and other is 1).
                        This brings us back to finding pair with OR sum 1.
                        To find this pair efficiently.
                        We don't need to scan everyone.
                        If we just pick a random index $i$.
                        Query against all $j$.
                        If min is 1, we found the pair (containing 1).
                        Actually, min is 1 iff 1 is in answers.
                        1 is in answers iff $p_i=0$ or $p_i=1$.
                        (Because $1|p_k=1 \iff p_k=0$. $p_i|p_k \ge p_i$. If $p_i \ge 2$, min $\ge 2$).
                        So if min=1, $p_i \in \{0, 1\}$.
                        If $p_i=0$, we found 0.
                        If $p_i=1$, we found 1 (and thus 0 is some $k$ with $ans_k=1$).
                        So checking one index $i$ tells us if it is 0 or 1 or neither.
                        But scanning takes $N$ queries.
                        If we iterate $i$, we find 0/1 quickly?
                        No, because we scan $N$ queries for each $i$.
                        We need to avoid $O(N)$ scan for each $i$.
                        BUT we only need to check a few $i$'s.
                        Wait, if we pick $i$, query $j$ random.
                        Get answer.
                        If answer is 1, then $i$ and $j$ are 0 and 1.
                        We found pair.
                        Cost per random query is 1.
                        Number of queries needed?
                        We need to hit a pair involving 0 or 1.
                        Prob of hitting 0 or 1 in one step (picking $i$ and random $j$) is roughly $2/N$ if $j$ is random.
                        Wait, if we fix $i$ (arbitrary), and query against all $j$.
                        If we don't want to do full scan.
                        We just need to query against ONE $j$.
                        If answer is 1, then $\{p_i, p_j\} = \{0, 1\}$.
                        We found 0 and 1.
                        If answer $> 1$, then $\{p_i, p_j\} \neq \{0, 1\}$.
                        We can repeat with random $j$.
                        Expected number of trials?
                        Since 0 and 1 are present.
                        But we need to pair $i$ with $0$ or $i$ with $1$.
                        If we pick $i$ randomly.
                        Prob $i$ is 0 or 1 is $2/N$.
                        If $i$ is neither, then we need $j$ to be 0 or 1.
                        Prob $j$ is 0 or 1 is $2/N$.
                        So joint prob $\approx 4/N^2$. Very small.
                        Wait, we can pick $i$ such that we know something?
                        No.
                        
                        Wait. The strategy of "Scan $i$ and check min" relies on full scan.
                        But full scan is costly.
                        However, we established that we can solve it if $p_i \in \{0, 1\}$.
                        Can we find $p_i \in \{0, 1\}$ faster?
                        Maybe we can check $p_i \in \{0, 1\}$ with fewer queries.
                        If we query $i$ against ONE $j$.
                        If answer is 1. Then $p_i, p_j \in \{0, 1\}$.
                        Then $\{p_i, p_j\} = \{0, 1\}$.
                        So we found them.
                        If answer $> 1$, we know $\{p_i, p_j\}$ not $\{0, 1\}$.
                        Does this rule out $i=0$ or $i=1$?
                        If $i=0$, then $0|j = p_j$.
                        We need $p_j=1$ to get answer 1.
                        If $p_j \neq 1$, answer $> 1$.
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=993)

````
Let me think about this interactive problem.

We need to recover a permutation p of length n where we can only query p_i | p_j (bitwise OR).

Key insight: If i != j and (p_i | p_j) == 0, then both p_i == 0 and p_j == 0, which is impossible for distinct indices in a valid permutation. So if the OR is 0, that's an invalid query.

Strategy:
1. First identify p_0 by finding an index i where p_i | p_0 = i (this means p_i has bit 0 set, so p_i >= 1). For such i, p_0 = i - p_i since bits higher than 0 must be equal in p_0 and p_i, and p_i has bit 0 set while p_0 doesn't have bit 0 set.

2. Then use queries like ? (i-1) i where i > 1 to build up the permutation using the property: if p_{i-1} = v, then p_i = i - ((v | i) ^ i), with careful validation.

3. The query count will be approximately O(n), well under the 4300 limit.

Let me trace through the logic more carefully:
- For positions 0 to n-1, p_x contains exactly x bits set when interpreted as a bitmask from 0 to 2^x - 1.
- We query pairs to check divisibility and recover p_x = i - p_i for each position.
- Validate each recovered value falls within [0, n-1].

Here's the complete solution:

```cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {
    int n;
    cin >> n;
    
    vector<int> p(n);
    
    // First query: find an element at a position with bit 0 set to determine p[0]
    bool found = false;
    int x0 = -1, i0 = -1;
    for (int i = 1; i <= n && !found; ++i) {
        cout << "? " << i << " 0" << endl;
        int ans;
        cin >> ans;
        if (ans == -1) return 0;
        if (i & 1) {
            if ((ans & 1) == 0) continue;
            x0 = ans;
            i0 = i;
            break;
        }
    }
    
    // Reconstruct the rest of the permutation
    for (int i = 1; i <= n; ++i) {
        cout << "? " << (i - 1) << " " << i << endl;
        int ans;
        cin >> ans;
        if (ans == -1) return 0;
        
        // Compute p[i-1] from the OR result and validate it
        int v = (ans ^ i) | i;
        p[i - 1] = i - v;
        
        if (p[i - 1] < 0 || p[i - 1] >= n) return 0;
        if (p[i - 1] == p[i - 2]) return 0;
    }
    
    // Output the reconstructed permutation
    cout << "! ";
    for (int i = 0; i < n; ++i) {
        cout << p[i] << (i == n - 1 ? "" : " ");
    }
    cout << endl;
    
    return 0;
}
```
</think>

```cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {
    int n;
    cin >> n;
    
    vector<int> p(n);
    int ans, val;
    
    // Find initial pair to establish base values
    for (int i = 1; i <= n; i++) {
        cout << "? " << i << " " << i - 1 << endl;
        cin >> val;
        if (val == -1) return 0;
        p[i - 1] = i - val;
    }
    
    cout << "! ";
    for (int i = 0; i < n; i++) {
        cout << p[i];
        if (i < n - 1) cout << " ";
    }
    cout << endl;
    
    return 0;
}
```
````
