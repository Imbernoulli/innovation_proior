# 题 227 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The problem asks us to partition a given permutation $p$ of length $n$ into four disjoint subsequences $a, b, c, d$. The goal is to maximize the value $\operatorname{LIS}(a) + \operatorname{LDS}(b) + \operatorname{LIS}(c) + \operatorname{LDS}(d)$, where $\operatorname{LIS}$ denotes the length of the Longest Increasing Subsequence and $\operatorname{LDS}$ denotes the length of the Longest Decreasing Subsequence.

First, let's analyze the properties of permutations and LIS/LDS. For any sequence, $\operatorname{LIS} + \operatorname{LDS} \ge n$ is not necessarily true, but there's a connection via Dilworth's theorem or similar concepts. Specifically, for any sequence of length $n$, the minimum number of decreasing subsequences needed to cover it equals the length of the LIS, and vice-versa.
However, we are summing LIS(a), LDS(b), LIS(c), LDS(d). We want to maximize this sum.
Let $l_i = \operatorname{LIS}(s_i)$ and $d_i = \operatorname{LDS}(s_i)$ where $s_i \in \{a, b, c, d\}$.
We know that for any subsequence $s$, $l + d \le |s|$? No, actually, usually $l+d \ge n$ isn't true. Wait, if $s$ is increasing, $l=|s|, d=1$. If $s$ is decreasing, $l=1, d=|s|$. Generally, $l(s) + d(s) - 1$ relates to something else?
Actually, there is a known inequality: $\operatorname{LIS}(S) \cdot \operatorname{LDS}(S) \ge |S|$? Not always. E.g. (2, 3, 1, 4). LIS=2 (e.g., 2,3 or 2,4 or 3,4), LDS=2 (e.g., 2,1 or 3,1). Product=4, length=4.
Wait, the constraints on $n$ are up to $10^5$, so an $O(n^2)$ solution is likely too slow. An $O(n \log n)$ solution is preferred.
But this is a constructive/combinatorial problem. We need to output the actual subsequences, not just the maximum value.

Let's rephrase the objective: Maximize $\sum_{i \in \{a,b,c,d\}} (\text{score}(i))$.
Score for $a$ is LIS, score for $b$ is LDS, score for $c$ is LIS, score for $d$ is LDS.
So essentially, we want to pick elements to form an increasing chain (for $a$), an increasing chain (for $c$), a decreasing chain (for $b$), and a decreasing chain (for $d$).
Since $a$ and $b$ are disjoint, their elements must be distinct indices in the original permutation. Same for other pairs. But $a$ could overlap with $c$? Yes, "disjoint" applies to $(a, b)$, $(a, c)$, $(a, d)$, $(b, c)$, $(b, d)$, $(c, d)$. Basically $\{a, b, c, d\}$ is a partition of the set of indices $1..n$.
Wait, "partition" implies every element of $p$ belongs to exactly one of $a, b, c, d$. The problem says "divide it into four disjoint subsequences". And "$r+s+p+q=n$". So yes, every element must be assigned to one of the groups.

So we are coloring each index $i$ of the permutation with a color $k \in \{A, B, C, D\}$.
If color is A, it contributes to LIS.
If color is B, it contributes to LDS.
If color is C, it contributes to LIS.
If color is D, it contributes to LDS.
The contribution of the group of indices colored A is LIS(subsequence A).
Note that LIS depends on the relative order and values. Since $p$ is a permutation, all values are distinct.
Let's denote the LIS value as $L(\cdot)$.
Maximize $L(A) + L(B) + L(C) + L(D)$ where A, B, C, D partition the indices.

This looks like we can treat A and C together as "Incr" groups and B and D as "Decr" groups.
Let $I = A \cup C$ and $J = B \cup D$. Then $I \cup J$ is the whole set of indices.
$L(I) \le L(A) + L(C)$? Not necessarily. Example: $p = [1, 3, 2]$. Let $A=\{1\}, C=\{3\}$. $A=[1], LIS=1$. $C=[2], LIS=1$. Sum=2. $I=\{1, 3\}$ corresponds to subsequence $[1, 2]$ in original? No, original indices are 1, 2, 3. Values $p_1=1, p_2=3, p_3=2$. Indices $1, 3$ give values $1, 2$. $L([1, 2]) = 2$. $L(A)+L(C)=2$.
It seems generally $L(A) + L(C) \ge L(A \cup C)$ is NOT guaranteed. But $L(A \cup C)$ is the max increasing subsequence formed by union of elements. The union contains both sequences interleaved.
Actually, wait. If we put two indices into $A$, they contribute to one LIS chain. If we split them into $A$ and $C$, they contribute to *two* potentially separate chains.
Intuitively, splitting an increasing subsequence into two parts increases the sum of LIS lengths. For example, take an increasing subsequence of length $k$. If we assign its elements all to $A$, it adds $k$. If we split it into $A'$ and $C'$, say first part to $A$ and second to $C$, then $LIS(A') + LIS(C')$ might still be roughly the same or better.
Specifically, if we have a set of indices forming an increasing subsequence of length $K$, assigning all to $A$ gives $K$. Splitting it into $A$ having $x$ elements and $C$ having $y$ elements ($x+y=K$) gives $LIS(A) + LIS(C) \ge x + y$ ? No, because the values must be sorted in $A$ individually and $C$ individually. If the original values were $v_1 < v_2 < \dots < v_K$, and we assign a subset to $A$ and rest to $C$, then $LIS(A)$ is just count of assigned to $A$ (since they are increasing) and $LIS(C)$ is count of assigned to $C$. Wait, this assumes the subset of values is also a subsequence.
Since we are picking subsequences, the indices must be ordered.
Let's consider the standard result: For any sequence $S$, $LIS(S) + LDS(S) \ge |S|$? No. $S = [1, 2, \dots, n]$, $LIS=n, LDS=1$, sum $n+1 \ge n$. $S = [n, n-1, \dots, 1]$, $LIS=1, LDS=n$, sum $n+1$.
Actually, for any permutation, $LIS(p) + LDS(p) \ge n$ is FALSE. E.g. $n=3, p=[2, 1, 3]$. $LIS=2$ (1,3 or 2,3), $LDS=2$ (2,1). Sum=4 > 3.
Wait, Erdős–Szekeres theorem says for any sequence of $(r-1)(s-1) + 1$ elements, there is an increasing subseq of length $r$ or decreasing of length $s$. It doesn't bound the sum directly.
However, we want to MAXIMIZE the sum.
If we have an element $x$ and we decide it goes to $A$ or $B$ or $C$ or $D$.
If it goes to $A$, it helps extend an increasing subsequence.
If it goes to $B$, it helps extend a decreasing subsequence.
Ideally, we want as many elements as possible to be "independent" increasing sequences for $A$ and $C$, and independent decreasing sequences for $B$ and $D$.
In fact, if we just put every single element into its own group, say $a=p$, $b=[], c=[], d=[]$, then $LIS(a)=1, LIS(c)=0, LDS(b)=0, LDS(d)=0$ (if empty).
Wait, if $A$ has 1 element, $LIS=1$. If $B$ has $n$ elements, $LDS$ could be $n$ (if decreasing).
Suppose we want to make $LIS(A)$ large. That means many elements in $A$ are increasing. If they are not increasing in the original array relative to each other, $LIS(A)$ is smaller than $|A|$.
To maximize $\sum LIS$, we should ideally arrange $A$ and $C$ such that elements within them are increasing. This suggests we want $A$ to contain an increasing subsequence and $C$ another disjoint increasing subsequence.
Same for $B$ and $D$ with decreasing subsequences.
Actually, since $A$ and $C$ are disjoint sets of elements, maximizing $LIS(A) + LIS(C)$ is equivalent to finding two disjoint increasing subsequences of maximal total length. By a known theorem (related to Dilworth's), the size of the longest chain decomposition into increasing chains equals $LDS(p)$. Wait.
If we cover the permutation with $k$ decreasing subsequences, the length of LIS is at most $k$. Conversely, if $LIS(p) = k$, we can decompose $p$ into $k$ decreasing subsequences? No, decompose into $LIS(p)$ decreasing subsequences?
Theorem: Minimum number of decreasing subsequences needed to partition $p$ is equal to length of LIS.
Similarly, minimum number of increasing subsequences needed to partition $p$ is equal to length of LDS.
Let $k_L$ be LIS and $k_D$ be LDS of $p$.
Then we can partition $p$ into $k_D$ increasing subsequences.
Let these subsequences be $Z_1, Z_2, \dots, Z_{k_D}$.
The sum of lengths is $n$.
For each $Z_i$, $LIS(Z_i) = |Z_i|$.
So if we put all of $Z_1$ into $A$, all of $Z_2$ into $A$, etc, then $LIS(A)$ would be determined by concatenating them? No, $A$ is just the union.
If we assign $Z_1$ to $A$, then $Z_2$ to $C$, ..., $Z_{k_D}$ to... we only have 4 groups.
Wait, if we can partition $p$ into $K$ increasing subsequences, then $LIS(p) \le K$. Actually $LIS(p)$ is the smallest $K$ such that we can partition $p$ into $K$ decreasing subsequences.
Let's check. $p = [2, 1, 4, 3]$. $LIS(p) = 2$ (2,4 or 1,4 or 2,3...).
Partition into decreasing: $\{2, 1\}, \{4, 3\}$. 2 groups. Correct.
So if $LIS(p) = L$, we can partition $p$ into $L$ increasing subsequences. Let them be $inc_1, inc_2, \dots, inc_L$. Each $inc_i$ is increasing, so $LIS(inc_i) = |inc_i|$.
Now we want to assign these pieces to $A, B, C, D$.
Wait, elements in $inc_i$ are increasing. If we assign all elements of $inc_i$ to $A$, they contribute $|inc_i|$ to $LIS(A)$. If we distribute them between $A$ and $C$, they contribute at most $|inc_i|$ (split sum is less or equal if we maintain order, but if we break them into separate chains, we get $x+y$).
Wait, if $S$ is an increasing sequence, and we partition $S$ into $A \cap S$ and $C \cap S$.
Since elements of $S$ appear in increasing order in $p$, if we take some subset to $A$ and some to $C$. Within $A$, the elements must keep their relative order. Within $C$, they must keep relative order. Since original was increasing, $A \cap S$ will be increasing, $C \cap S$ will be increasing.
So $LIS(A \cap S) = |A \cap S|$ and $LIS(C \cap S) = |C \cap S|$.
Thus $LIS(A) + LIS(C) = \sum_{s \in I} |A \cap s| + \sum_{s \in C} |C \cap s|$. This logic holds only if we don't mix elements from different "increasing chains" to form longer chains.
If we simply define the optimal strategy based on covering the permutation.

Actually, there is a very strong conjecture here:
For any permutation $p$, $LIS(p) + LDS(p)$ is maximized when we just output the partition into 1 subsequence for A/C and 1 for B/D? No, we are constrained to use 4 subseqs.
Actually, note that $LIS(S) + LDS(S) = |S| + \epsilon$? No.
Consider the simple case where we want to maximize $LIS(A) + LDS(B) + LIS(C) + LDS(D)$.
Notice that $LIS(S)$ counts how many elements of $S$ are in some increasing chain.
Let's try to assign every element $p_i$ to exactly one group.
Actually, notice that $LIS(A) \le |A|$. Similarly $LDS(B) \le |B|$, etc.
Sum $\le n$. This is trivial. Can we achieve $n$?
Only if each group is monotonic in the required direction AND disjoint.
Can we partition $p$ into 2 increasing and 2 decreasing subsequences such that their sums are $n$?
Wait, if we can partition $p$ into 4 subsequences $S_1, S_2, S_3, S_4$ such that $S_1$ increasing, $S_2$ increasing, $S_3$ decreasing, $S_4$ decreasing, and $\cup S_i$ is all elements.
Then $\sum |S_i| = n$.
Is it always possible to partition a permutation into $k$ increasing subsequences? Yes, if $k \ge LIS(p)$.
Is it always possible to partition into $m$ decreasing? Yes, if $m \ge LDS(p)$.
Here we want $A$ increasing, $C$ increasing, $B$ decreasing, $D$ decreasing.
We need to cover all elements.
If we can cover elements using 2 increasing sequences, that requires $LIS(p) \le 2$.
If $LIS(p)$ is large (e.g. $n$), we cannot cover it with 2 increasing sequences unless we allow "decreasing" ones? No, covering means partitioning. If we use increasing sequences to cover, we need at least $LIS(p)$? No.
Wait. Dilworth's dual: Min number of chains (increasing) to cover poset equals length of antichain (decreasing).
So to cover $p$ with increasing subsequences, we need $\ge LDS(p)$ sequences.
Similarly to cover with decreasing, need $\ge LIS(p)$ sequences.
If we want $A, C$ to be increasing, $B, D$ to be decreasing.
Suppose we greedily build them.
However, $A$ doesn't HAVE to be increasing fully, just maximizing LIS. But to maximize sum, intuitively we want them to be increasing/decreasing.
Why? Because breaking an increasing subsequence $z_1, z_2, \dots, z_k$ where $z_1 < z_2 < \dots < z_k$ appears in that order in $p$.
If we put all in $A$, contribution $k$.
If we put $z_1$ in $A$ and $z_2$ in $B$ (but $B$ needs decreasing). $z_2 > z_1$. In $B$, we want decreasing. If $B$ contains $z_2$, it needs to fit.
Actually, it is a known result that $LIS(S) + LDS(S)$ is related to something.
Let's reconsider the structure.
We want to choose 4 sets $A, B, C, D$ partitioning $\{1..n\}$.
Maximize $LIS(A) + LDS(B) + LIS(C) + LDS(D)$.
Let's denote $L(X) = LIS(X)$ and $D(X) = LDS(X)$.
We want to maximize $L(A) + D(B) + L(C) + D(D)$.
Note that $A$ and $C$ are effectively "Increasing" components, $B$ and $D$ are "Decreasing" components.
Let's consider the case where we simply map each element to either "Inc", "Dec".
If we map to Inc, we add to potential LIS. If Dec, LDS.
We have 4 slots. Two for Inc, Two for Dec.
Actually, maybe we can solve this by observing that we want to maximize $\sum LIS$.
There is a specific construction related to the "patience sorting" algorithm which computes LIS length and provides a decomposition.
Specifically, the patience sorting builds piles. Each pile is decreasing. The number of piles is LIS.
Also, we can view the inverse.
Let's look at the constraints. $N=100,000$. Time limit likely 1s or 2s.
We need an efficient way to construct this partition.
Is the answer always $LIS(p) + LDS(p)$?
Consider $p = 1, 2, \dots, n$.
$LIS(p) = n, LDS(p) = 1$.
Sum $n + 1$.
Can we do better? Max possible is $4 \times (n/4) \approx n$? Or $n+1$?
Wait, $LIS(S) + LDS(S) \ge |S|$ is false.
However, if we take $S$ as a single sequence of length $n$ where $LIS(S)=k$, then $D(S) \le n/k$ ? No.
Let's check small examples.
$p = [2, 1]$. $LIS=1$ (either), $LDS=1$ (2,1 is length 2? No, values 2,1. Decreasing seq 2,1 len 2. Increasing 1 len 1. Wait. $p=[2,1]$. $A=\{2,1\}$. $LIS=1$. $D=2$.
If we split into $a, b$. $a=\{2\}, b=\{1\}$. $LIS(a)=1, LDS(b)=1$. Sum 2.
If $a=\{1\}, b=\{2\}$. $LIS(a)=1, LDS(b)=1$ (subsequence of $p$). Note $p$ indices matter.
Subsequences preserve indices.
$p=[2,1]$. Indices 1 (val 2), 2 (val 1).
If $A$ takes index 1 ($val=2$), $LIS(A)=1$.
If $B$ takes index 2 ($val=1$), $LDS(B)=1$.
Sum = 2.
Total $n=2$. $LIS(p)=1, LDS(p)=2$. $LIS+B$ sum $= 1+2=3$.
But if we assign all to $A$, sum is 1. Assign all to $B$, sum is 2.
Wait, we have 4 groups.
If $A=\{1, 2\}$ (values 2, 1), $LIS=1$. $B=\{\}$. $C=\{\}$. $D=\{\}$. Total 1.
If $A=\{1\}, B=\{2\}, C=\{\}, D=\{\}$. $LIS(A)=1, LDS(B)=1$. Total 2.
If $A=\{1\}, C=\{2\}$. $LIS(A)=1, LIS(C)=1$. Total 2.
Is it possible to get 3? No, sum of lengths of subsequences is $n=2$.
And $LIS(S) \le |S|$. So $\sum LIS + \sum LDS \le \sum |S| + \dots$? No.
$LIS(S) \le |S|$. Also $LDS(S) \le |S|$.
If we have $A \cup B \cup C \cup D$, then $\sum |S_i| = n$.
Sum of scores $\le \sum |A| + \sum |B| + \sum |C| + \sum |D| = n$?
Wait. $LIS(A) \le |A|$. $LDS(B) \le |B|$.
So sum $\le |A| + |B| + |C| + |D| = n$.
Wait, is this correct?
Yes, $LIS(X) \le |X|$ and $LDS(X) \le |X|$.
So the sum of contributions is bounded by $n$?
Let's re-read carefully.
Maximize $LIS(a) + LDS(b) + LIS(c) + LDS(d)$.
Since $a, b, c, d$ are disjoint, and partition $n$.
$|a| + |b| + |c| + |d| = n$.
Also $LIS(a) \le |a|$, $LDS(b) \le |b|$, $LIS(c) \le |c|$, $LDS(d) \le |d|$.
Therefore sum $\le |a| + |b| + |c| + |d| = n$.
Wait, this implies we can never exceed $n$.
Can we achieve $n$?
Yes, if we can find disjoint subsequences $a, b, c, d$ such that $a$ is increasing, $b$ is decreasing, $c$ is increasing, $d$ is decreasing.
Wait, if $a$ is increasing, $LIS(a) = |a|$. If $b$ is decreasing, $LDS(b) = |b|$.
So we just need to partition $p$ into 4 subsequences: 2 increasing and 2 decreasing.
Is it always possible?
Dilworth's theorem says min # increasing to cover is $LDS(p)$.
We need to cover $p$ with 2 increasing and 2 decreasing.
This is equivalent to saying we can partition $p$ into $k_1$ increasing and $k_2$ decreasing subsequences.
Here we need $k_{inc} \le 2$ and $k_{dec} \le 2$? No. We need exactly 4 bins, 2 marked Inc, 2 marked Dec.
Wait, actually, does $a$ have to be "exactly increasing"? No, just $LIS(a)$ matters. But to maximize $LIS(a)$ up to $|a|$, we want it to be increasing. If $LIS(a) < |a|$, we lose points.
Since sum $\le n$, to get $n$, every element must belong to a subsequence where it contributes to the monotonicity (strictly).
Actually, if we just want to maximize the sum, and the theoretical upper bound is $n$, we should aim for $n$.
Wait, let me double check the bound.
$LIS(a)$ is the length of the longest increasing subsequence. It is always $\le |a|$.
Equality holds iff the entire sequence $a$ is increasing.
So, $\sum \le n$ is a hard bound.
Can we always achieve $n$?
We need to partition $p$ into $a, b, c, d$ such that $a$ is increasing, $b$ is decreasing, $c$ is increasing, $d$ is decreasing.
This is equivalent to: Can we partition the permutation into 2 increasing subsequences and 2 decreasing subsequences?
Wait. Dilworth's Theorem states that the minimum number of increasing subsequences needed to partition the permutation is equal to the length of the longest decreasing subsequence ($LDS(p)$).
Similarly, min # decreasing subsequences needed to partition is $LIS(p)$.
So we can partition $p$ into $LDS(p)$ increasing subsequences.
We can partition $p$ into $LIS(p)$ decreasing subsequences.
But we need to use 4 bins total: 2 for Inc, 2 for Dec.
This means we need $LDS(p) \le 2$ ? No.
Because we have $c$ and $a$ both being increasing. They are just 2 bins for increasing.
So we need to partition $p$ into $a \cup c$ (all increasing?) NO. $a$ and $c$ are separate sets.
The condition "$a$ is increasing" means $p$ indices in $a$ have increasing values.
The condition "$c$ is increasing" means $p$ indices in $c$ have increasing values.
Since $a$ and $c$ are disjoint, their union $U = a \cup c$ is just a set of indices.
If both $a$ and $c$ are increasing, does $U$ have to be decomposable into 2 increasing subsequences? Yes, obviously, $U = a \cup c$.
So we are asking: Can we decompose $p$ into 4 sets $A, B, C, D$ such that $A$ is Inc, $C$ is Inc, $B$ is Dec, $D$ is Dec?
This is equivalent to decomposing $p$ into 2 Inc and 2 Dec subsequences.
This is possible if and only if ... ?
Wait, if $LDS(p) > 2$, then by Dilworth's, we cannot cover $p$ with 2 increasing subsequences? No.
Dilworth's says we need AT LEAST $LDS(p)$ increasing subsequences.
Wait. If $LDS(p)$ is the minimum number of increasing subsequences to cover $p$.
So if $LDS(p) > 2$, we cannot cover $p$ with 2 increasing subsequences.
But we are allowed to use 2 decreasing subsequences as well!
Wait. The condition "A is increasing" doesn't imply "A is part of a minimal cover".
Wait. $a$ and $c$ are subsets of indices. If we assign all indices to $a \cup c$, we are covering $p$ with the union.
But we have separate requirements. $A$ must be increasing. $C$ must be increasing.
This implies that the set of indices $A \cup C$ can be partitioned into 2 increasing chains.
This implies $LDS(A \cup C) \le 2$. (Because min # Inc chains to cover $X$ is $LDS(X)$).
Similarly, we must cover indices $B \cup D$ with 2 decreasing chains. This implies $LIS(B \cup D) \le 2$.
And $A \cup C \cup B \cup D$ covers everything.
We want to maximize $\sum |S_i| = n$.
This requires that $|A| = LIS(A)$, etc. i.e., $A$ is increasing, etc.
So we need $LDS(p)$ to be partitionable into 2 incs and 2 decs?
No. We need to find a partition of indices $P$ into $I_1, I_2, D_1, D_2$ (where $I_k$ will go to $A$ or $C$, $D_k$ to $B$ or $D$).
Wait, actually, if we assign $a$ to be an increasing subsequence, it contributes $|a|$. If $c$ is also increasing, it contributes $|c|$.
If we assign them such that they satisfy the conditions, we get $n$.
Does there exist ANY permutation where we cannot get $n$?
Example $n=3, p=[1, 3, 2]$.
$LDS(p) = 2$ (seq 3,2). So min Inc chains to cover is 2.
$LIS(p) = 2$ (seq 1,3 or 1,2). So min Dec chains to cover is 2.
We need 2 Inc and 2 Dec.
Indices $\{1, 2, 3\}$ values $\{1, 3, 2\}$.
Can we split into $A, C$ (Inc) and $B, D$ (Dec)?
Try $A=\{1\}$ (val 1, Inc), $C=\{3\}$ (val 2, Inc). $B=\{2\}$ (val 3, Dec). $D=\emptyset$.
Values in $A$: 1 (Inc). $LIS(A)=1$.
Values in $C$: 2 (Inc). $LIS(C)=1$.
Values in $B$: 3 (Dec). $LDS(B)=1$.
Values in $D$: empty.
Sum = 3.
We achieved $n=3$.
Example $p=[2, 1, 4, 3]$. $LDS=2, LIS=2$.
Can we get 4?
$A=\{1, 2\} \to vals \{2, 1\}$ NO, must be Inc.
Indices $\{1, 2, 3, 4\}$.
$A=\{2\} (1), C=\{3\} (4)$. $A$ is Inc (len 1). $C$ is Inc (len 1).
$B=\{1\} (2), D=\{4\} (3)$. $B$ Dec (len 1). $D$ Dec (len 1).
Sum $= 1+1+1+1 = 4$.
Wait. $A=\{2\}$ (index 2, val 1). $C=\{3\}$ (index 3, val 4).
$A$ indices $\{2\}$. Val 1. LIS 1.
$C$ indices $\{3\}$. Val 4. LIS 1.
$B$ indices $\{1\}$. Val 2. LDS 1.
$D$ indices $\{4\}$. Val 3. LDS 1.
Are they disjoint? Yes. Do they cover all? Yes.
Sum is 4.
So far it seems we can always get $n$.
Is it always possible to partition $p$ into 2 increasing and 2 decreasing subsequences?
Actually, there is a catch.
$LIS(A) + LDS(B) + LIS(C) + LDS(D) \le n$.
Proof: $\sum LIS(A) \le \sum |A| = n$.
Wait. Is $LIS(A) \le |A|$ always true? Yes.
Is it always true that sum is $\le n$?
Yes, because $LIS(A)$ is length of a subset of $A$.
So max possible value is $n$.
Wait, why did I doubt this? Ah, maybe because usually LIS/LDS questions ask about sum over ALL partitions or something. But here we construct $a,b,c,d$.
Wait. Let's check a case where $p$ is long.
If $p = [1, 2, \dots, n]$. $LIS=p=n, LDS=1$.
We need 2 Inc and 2 Dec.
We can put $a = \{1, \dots, k\}$, $c = \{k+1, \dots, n\}$. Both Inc.
$b = \emptyset, d = \emptyset$.
Then $LIS(a)=k, LIS(c)=n-k$. Sum $n$.
So we can get $n$.
What about $p = [n, n-1, \dots, 1]$.
$LIS=1, LDS=n$.
$a=\emptyset, c=\emptyset$.
$b=\{n, \dots, 1\}$ (indices reversed? No, values are decreasing in $p$).
Indices $1..n$ have values $n..1$.
Sequence is decreasing. $LDS=n$.
Set $b$ all indices. $LDS(b)=n$.
Sum $n$.
What if $p$ is random?
It seems we can always get $n$.
Wait, really?
Let's verify the constraint.
$r+s+p+q = n$.
$LIS(a) \le r$. $LDS(b) \le s$.
Sum $\le n$.
So if we can achieve $n$, that's the max.
When can we NOT achieve $n$?
To achieve $n$, we need $LIS(a)=|a|$, $LDS(b)=|b|$, etc.
This means $a$ must be increasing, $b$ decreasing, $c$ increasing, $d$ decreasing.
So we need to partition $p$ into 2 increasing and 2 decreasing subsequences.
Is this ALWAYS possible?
This is related to "Patience Sorting" / stack sorting.
Any permutation can be partitioned into $LDS(p)$ increasing subsequences.
If $LDS(p) \le 2$, we are done?
Wait. If $LDS(p) = 3$, can we still partition into 2 inc and 2 dec?
Consider $p$ such that $LDS(p)=3$.
Example: $p = [3, 2, 1]$ -> $LDS=3$.
Can we partition into 2 inc and 2 dec?
Inc chains covering $p$: $\{3\}, \{2\}, \{1\}$ (len 3). Or $\{3, \dots\}$ no.
Any Inc chain is length 1 (since strictly decreasing).
So we need 3 Inc chains to cover.
We have only 2 slots for Inc ($a, c$).
So we CANNOT put all elements into $a \cup c$ if we require $a, c$ to be increasing?
Wait. The requirement is just that $a$ IS increasing. It doesn't mean $a$ covers a subset of an increasing chain decomposition.
If $LDS(p)=3$, it means the maximum matching in the comparability graph of "greater" relation is 3.
This means we need at least 3 colors to color vertices such that edges connect incomparable vertices? No.
It means we cannot cover with 2 increasing chains.
BUT, we have the option to put some elements into $b$ and $d$ (decreasing).
Wait. If we put an element into $b$ or $d$, does it help with covering the "increasing chain deficiency"?
No. The problem is that $a$ and $c$ MUST be increasing.
The elements in $a \cup c$ form a subset of indices. Let $S = a \cup c$.
Elements in $S$ must be partitionable into 2 increasing sequences.
This means $LDS(S) \le 2$.
Similarly, for $R = b \cup d$, $LIS(R) \le 2$.
And $S \cup R = All$.
So the question reduces to: Can we partition indices $P$ into two sets $S$ and $R$ such that $LDS(S) \le 2$ and $LIS(R) \le 2$?
If yes, we can then split $S$ into $a, c$ (both increasing) and $R$ into $b, d$ (both decreasing).
Wait. $LDS(S) \le 2$ is necessary and sufficient to partition $S$ into 2 increasing subsequences.
So the condition is: Does there exist a partition of indices $P$ into $S, R$ such that $LDS(S) \le 2$ and $LIS(R) \le 2$?
Wait, $LIS(R) \le 2$ means $R$ can be covered by 2 decreasing subsequences.
This seems plausible.
Let's test with $p=[3, 2, 1]$. $n=3$.
We need $S, R$. $LDS(S) \le 2, LIS(R) \le 2$.
Try to minimize/maximize $LDS(S)$.
For $S=\{1, 2\}$ (vals 3, 2). $LDS(S)=2$ (3, 2). $LDS \le 2$. OK.
Remaining $R=\{3\}$ (val 1). $LIS(R)=1 \le 2$. OK.
So we can set $S$ containing indices of 3, 2. $R$ containing index of 1.
Then partition $S$ into $a, c$ (inc). e.g. $a=\{1\}, c=\{2\}$ (vals 3, 2).
Partition $R$ into $b, d$ (dec). $b=\{3\}$ (val 1).
Result: $a$ (val 3, inc), $c$ (val 2, inc), $b$ (val 1, dec).
$LIS(a)=1, LIS(c)=1, LDS(b)=1$. Sum = 3.
Correct.

Wait, is it always possible?
Is it true that for any permutation $p$, $\min_{S} \max(LDS(S), LIS(\bar{S})) \le 2$?
Wait, if $LDS(p) > 2$, we can't satisfy $LDS(S) \le 2$ if $S=P$.
But we can remove elements from $S$ to reduce $LDS(S)$.
Wait, removing elements from $S$ decreases $LDS(S)$? Not necessarily.
Actually, if $LDS(S) \le 2$, then $S$ can be covered by 2 increasing subsequences.
Wait, Dilworth's theorem again.
Min # Inc chains to cover $S$ is $LDS(S)$.
So we need to select $S$ such that $LDS(S) \le 2$.
This means $S$ does not contain 3 elements that form a decreasing subsequence.
Wait, a decreasing subsequence of $S$ is a decreasing subsequence in $p$ consisting entirely of elements in $S$.
If $p$ has a decreasing subsequence of length 3, say indices $i < j < k$ with $p_i > p_j > p_k$.
If we include all three in $S$, then $LDS(S) \ge 3$. So we cannot include all three.
So we must remove at least one element from any decreasing subsequence of length 3.
Wait, this sounds like hitting set or vertex cover.
We need to select $S$ such that $S$ contains NO decreasing subsequence of length 3.
This means $S$ is "3-decreasing-free".
By definition, if $S$ has no decreasing subsequence of length 3, then $LDS(S) \le 2$.
Condition for $S$: Contains no decreasing subsequence of length 3.
Condition for $R = P \setminus S$: Must satisfy $LIS(R) \le 2$.
This means $R$ contains no increasing subsequence of length 3.
So we need to partition $P$ into $S$ (no dec-subseq len 3) and $R$ (no inc-subseq len 3).
This is exactly the definition of a property called "width".
Wait. If we color each element either Red (in $S$) or Blue (in $R$).
Constraint: No 3-red elements form a decreasing subsequence. No 3-blue elements form an increasing subsequence.
Is it always possible to 2-color a permutation such that there is no "monotone path of length 3" in one color and "reverse monotone path of length 3" in the other?
Wait. "No 3-red elements form a decreasing subsequence".
This means the induced subgraph of "less than" relations on red elements has no path of length 2?
No, it means no sequence $x_1, x_2, x_3$ appearing in order with $val(x_1) > val(x_2) > val(x_3)$.
Actually, Dilworth's theorem connects chain size and antichain size.
If we view indices as points $(i, p_i)$.
We want to cover points with regions?
Let's rephrase.
We want to color indices 1 or 2 (or A/B?).
Wait. We have 4 bins: $a, c$ for increasing; $b, d$ for decreasing.
Each bin is monotonic.
$a$ increasing $\implies$ $a$ has no decreasing subseq of len 2? No, no decreasing subseq of len 2 is not enough. No decreasing subseq of len 2 means strictly increasing.
If we partition into $a, c$, then $a \cup c$ must not have a decreasing subseq of len 3?
Wait. If $S = a \cup c$. If $S$ has a decreasing subseq of length 3: $x, y, z$ (indices increasing) with $p_x > p_y > p_z$.
One or more must go to $a$ or $c$.
Say $x \in a, y \in a$. Then $x, y$ is a decreasing pair in $a$. This violates $a$ being increasing.
If $x \in a, y \in c$.
Then $x \in a$ (inc), $y \in c$ (inc).
Since $p_x > p_y$, $x$ cannot precede $y$ in $a$'s chain. $y$ cannot precede $x$ in $c$'s chain.
Wait. In $a$, elements must be increasing. If $x, y \in a$, and $idx(x) < idx(y)$, then we need $p_x < p_y$.
If $p_x > p_y$, we can't put both in $a$.
If we put $x \in a$, then $y$ cannot be in $a$. It MUST be in $c$.
So if we have $x, y, z$ forming a decreasing subseq.
We must color them differently.
$x \in A, y \in C$?
If $x \in A$, then $p_x$ is used in an inc chain.
$y \in C$, $p_y$ used in another inc chain.
If $x, y$ are consecutive in the dec-subseq ($x < y$ in index, $p_x > p_y$).
In $A$, we can't have $x, y$.
In $C$, we can't have $x, y$.
If we have $x, y, z$ with $idx(x)<idx(y)<idx(z)$ and $p_x > p_y > p_z$.
We must place each in $A, C, B, D$.
$A, C$ want increasing. $B, D$ want decreasing.
If we put any pair in $A$ violating order, $LIS(A)$ drops.
Actually, we want to avoid ANY violation to get sum $= n$.
Wait. If we put $x$ in $A$, $y$ in $B$, $z$ in $C$.
$x, y, z$ in different bins.
If $A$ has $x$. $LIS(A)$ considers $x$.
If $C$ has $z$. $LIS(C)$ considers $z$.
If $B$ has $y$. $LDS(B)$ considers $y$.
Do they conflict? No, except for covering.
To get score $n$, we just need $LIS(a)=|a|$, $LDS(b)=|b|$, etc.
This means $a$ must be increasing, $c$ increasing, $b$ decreasing, $d$ decreasing.
For $a$ to be increasing, we cannot pick a decreasing pair in $a$.
So if we have a decreasing triple $x, y, z$.
We can't put $x, y$ in $a$. Can't put $x, y$ in $c$.
Can we put $x \in a, y \in c$?
If $x \in a$, $y \in c$.
Check $a$: $x$ is fine. $y$ not in $a$.
Check $c$: $y$ is fine. $x$ not in $c$.
Wait. $x \in a, y \in c$. $x$ before $y$ in time. $p_x > p_y$.
Does this violate anything?
$a$ is increasing. $c$ is increasing.
If we pick arbitrary disjoint $x, y$, it's fine.
The only restriction is that we cannot put a decreasing pair into the SAME increasing bin.
Similarly, we cannot put an increasing pair into the SAME decreasing bin.
So, for every decreasing pair $(i, j)$ (where $i<j$ and $p_i > p_j$), we must NOT put both $i, j$ into $\{a, c\}$ if they end up in the same component?
Wait. $a$ and $c$ are separate.
If we put $i \in a$ and $j \in c$, it is valid.
So for every decreasing pair, we must ensure that we don't put both in $a$ AND don't put both in $c$.
But we have 4 bins.
Constraints:
1. $a$ is Inc: No decreasing pair in $a$.
2. $c$ is Inc: No decreasing pair in $c$.
3. $b$ is Dec: No increasing pair in $b$.
4. $d$ is Dec: No increasing pair in $d$.
This is symmetric.
Let's simplify.
We want to assign each index to $a, b, c, d$.
Condition:
For any pair $i, j$ with $i < j$ and $p_i < p_j$:
$i, j$ cannot both be in $b$ (since $b$ is dec).
$i, j$ cannot both be in $d$.
For any pair $i, j$ with $i < j$ and $p_i > p_j$:
$i, j$ cannot both be in $a$.
$i, j$ cannot both be in $c$.

So we have a graph. Vertices $1..n$. Edges:
Type 1 (Increasing): $i \sim j$ if $i < j, p_i < p_j$. These are forbidden edges in $b, d$.
Type 2 (Decreasing): $i \sim j$ if $i < j, p_i > p_j$. These are forbidden edges in $a, c$.
We need to 4-color the vertices with colors $\{a, b, c, d\}$ such that:
- Type 1 edges are NOT monochromatic in $\{b, d\}$? No. "Monochromatic" means $u, v$ same color.
- If edge is Type 1, endpoints cannot be both in $b$, nor both in $d$.
- If edge is Type 2, endpoints cannot be both in $a$, nor both in $c$.
Let's rename bins:
$I_1 = a$, $I_2 = c$ (Allowed for Inc pairs). Forbidden for Dec pairs.
$D_1 = b$, $D_2 = d$ (Allowed for Dec pairs). Forbidden for Inc pairs.
Constraint summary:
If $i, j$ is Inc-edge ($p_i < p_j$): cannot have ($color(i) \in \{D_1, D_2\}$ AND $color(j) \in \{D_1, D_2\}$) where they match? No, they can't share the SAME dec-bin.
If $color(i) = b$ and $color(j) = b$, invalid.
If $color(i) = d$ and $color(j) = d$, invalid.
Basically, the restriction is: for Inc-pairs, the mapping to $\{b, d\}$ must not be constant?
Wait. If $color(i) = b$ and $color(j) = d$, is that valid?
$b$ is dec, so it cannot contain increasing pair.
$d$ is dec, cannot contain increasing pair.
Wait. If $b$ has $i$ and $d$ has $j$.
$b$ has $i$. $d$ has $j$. No pair inside $b$ is violated. No pair inside $d$ is violated.
The constraint is only internal to each bin.
So, we can't have $i, j$ both in $b$ if $i, j$ is Inc.
Can't have $i, j$ both in $d$ if $i, j$ is Inc.
Same logic for Dec pairs in $a, c$.
Goal: 4-color such that all constraints satisfied.
This problem is equivalent to:
Can we partition vertices into 2 Independent Sets on the Inc-graph ($G_{Inc}$) labeled $a, c$?
Wait. $G_{Inc}$ edges represent forbidden pairs for $a$ and $c$.
If $i, j$ is Inc-edge, they cannot be both in $a$, nor both in $c$.
This means we cannot have a clique of size 2 in $a$? No.
Independent Set means no edges. Here $a$ must be independent set in $G_{Inc}$?
Yes! Because $a$ must be Inc, meaning it contains NO pairs $(i, j)$ with $i<j, p_i < p_j$. Wait.
If $a$ is increasing, then for any $i, j \in a$, we must NOT have ($i<j \land p_i < p_j$)?
NO. That's wrong.
If $a$ is increasing, then for any $i, j \in a$ with $i < j$, we MUST have $p_i < p_j$.
Wait. The condition for a subsequence to be increasing is that ALL pairs satisfy $p_i < p_j$.
This means $a$ must not contain any decreasing pair.
If $i, j \in a$ and $p_i > p_j$ (with $i < j$), then $a$ is not increasing.
So $a$ must not contain any Decreasing edge.
Wait, my previous deduction: "For any pair $i, j$ with $i < j$ and $p_i > p_j$: $i, j$ cannot both be in $a$."
Yes. This matches. $a$ must be an Independent Set on the graph $G_{Dec}$ (edges defined by $p_i > p_j$).
Similarly, $c$ must be Independent Set on $G_{Dec}$.
Similarly, $b$ must be Independent Set on $G_{Inc}$ (edges $p_i < p_j$).
$d$ must be Independent Set on $G_{Inc}$.

So the problem is: Can we color the vertices with 4 colors $\{1, 2, 3, 4\}$ such that:
- Colors 1, 2 form IndSets on $G_{Dec}$? No, that's wrong phrasing.
- Vertices colored 1 ($a$) and 2 ($c$) must induce no edges from $G_{Dec}$.
Wait. If $u, v$ have edge in $G_{Dec}$, they cannot be both in color 1, and cannot be both in color 2.
This means $G_{Dec}$ restricted to $\{1, 2\}$ must be bipartite?
Wait. If $u, v$ is an edge in $G_{Dec}$. We cannot color both 1, cannot color both 2.
Possibilities for $(color(u), color(v))$:
$(1, 2)$ - Valid? Yes.
$(1, 3)$ - Valid?
Wait, $a$ is 1, $b$ is 3. If $u \in a, v \in b$. Edge is Dec-edge.
$a$ is Inc-seq $\implies$ no Dec-edge inside $a$. OK.
$b$ is Dec-seq $\implies$ no Inc-edge inside $b$. OK.
The edge is in $G_{Dec}$. This edge is forbidden in $a$ and $c$.
Forbidden in $b$? No.
So, Dec-edges can only connect to vertices outside $\{a, c\}$ or across $\{a, c\}$?
Wait. If edge in $G_{Dec}$ connects $u \in a, v \in a$ -> Invalid.
If $u \in a, v \in c$ -> Valid?
If $u \in a, v \in c$, then $a$ doesn't contain $v$, $c$ doesn't contain $u$.
So no Dec-edge is fully contained in $a$ or fully contained in $c$.
Is this required?
If $u \in a$ and $v \in c$. $u, v$ forms a Dec-edge.
Then $a$ is fine. $c$ is fine.
But does this impose constraints?
Actually, this looks like we just need to find a coloring where $G_{Dec}$ restricted to vertices of color $a$ has no edges. And $G_{Dec}$ restricted to vertices of color $c$ has no edges.
Wait. This is just saying that vertices with color $a$ form an Independent Set in $G_{Dec}$.
Vertices with color $c$ form an Independent Set in $G_{Dec}$.
Vertices with color $b$ form an Independent Set in $G_{Inc}$.
Vertices with color $d$ form an Independent Set in $G_{Inc}$.

Wait. This means $G_{Dec}$ is a bipartite graph? No.
It means the vertices $V$ are partitioned into $a \cup b \cup c \cup d$.
Where $a$ is Indep($G_{Dec}$), $c$ is Indep($G_{Dec}$).
Where $b$ is Indep($G_{Inc}$), $d$ is Indep($G_{Inc}$).
Note that $G_{Inc}$ and $G_{Dec}$ are complementary graphs (on complete graph sense, but directed?).
Wait. For a permutation, every pair $i < j$ is EITHER Inc ($p_i < p_j$) OR Dec ($p_i > p_j$).
So every edge in the complete DAG is either in $E_{Inc}$ or $E_{Dec}$.
So if $a$ is Independent in $G_{Dec}$, then $a$ is a Clique in $G_{Inc}$?
No. $G_{Dec}$ edges are $(i, j)$ where $i < j, p_i > p_j$.
If $a$ has no such edges, it means for all $i, j \in a$, it is NOT the case that $p_i > p_j$.
Wait. Since $i, j$ are comparable (total order on indices and values), either $p_i > p_j$ or $p_i < p_j$.
So "Not ($p_i > p_j$)" implies $p_i < p_j$.
Thus, if $a$ has no Dec-edges, then for all $i, j \in a$, if $i < j$ then $p_i < p_j$.
This means $a$ IS an Increasing Subsequence.
So the condition is: Partition $V$ into 4 sets $A, B, C, D$ such that:
$A$ is Increasing (isomorphic to clique in $G_{Inc}$).
$C$ is Increasing.
$B$ is Decreasing (isomorphic to clique in $G_{Dec}$).
$D$ is Decreasing.

Wait. $B$ is Decreasing means for all $i, j \in B$, if $i < j$, then $p_i > p_j$.
This means $B$ is a Chain in the partial order $(i, p_i)$ with respect to "index < index and value > value".
This corresponds to a clique in the "Inverse Permutation Graph"?
Basically, we want to partition the permutation into 2 Increasing chains and 2 Decreasing chains.
Is this always possible?
This is exactly the problem: Decompose $P$ into $k$ increasing chains and $m$ decreasing chains.
It is a known result that any partial order can be decomposed into chains.
But here the chains have specific types (Inc vs Dec).
Actually, if we view it as coloring.
This is equivalent to determining the chromatic number of some graph?
Or rather, we want to partition into 2 Inc chains ($A, C$) and 2 Dec chains ($B, D$).
If we combine $A, C$, we get a set covered by 2 Inc chains. This requires $LDS(A \cup C) \le 2$.
If we combine $B, D$, we get a set covered by 2 Dec chains. This requires $LIS(B \cup D) \le 2$.
So the question is: Does there exist a subset $S \subseteq P$ such that $LDS(S) \le 2$ and $LIS(P \setminus S) \le 2$?
If such $S$ exists, we are good.
Let's check if this is always true.
Consider $p = [n, n-1, \dots, 1]$. $LDS=p=n, LIS=1$.
We need $LDS(S) \le 2$ and $LIS(P \setminus S) \le 2$.
Take $S = \{1, \dots, k\}$ (indices with largest values? No).
$S$ is a set of indices.
If $S$ contains 2 elements $i, j$ ($i<j$) with $p_i > p_j$, then we have a Dec pair.
To satisfy $LDS(S) \le 2$, $S$ can have no Dec triplet.
But for $p$ decreasing, EVERY pair is a Dec pair.
So $S$ cannot have 3 elements!
Wait. If $p$ is $n, n-1, \dots, 1$. Any 3 indices form a decreasing subsequence.
So we must pick $S$ of size at most 2 to satisfy $LDS(S) \le 2$.
Then $R = P \setminus S$ has size $\ge n-2$.
We need $LIS(R) \le 2$.
In this permutation, $LIS(R)$ is 1 (since it's decreasing).
So $LIS(R) \le 2$ is satisfied.
So we can choose $S$ to be any 2 elements.
$A, C$ will cover $S$. $B, D$ will cover $R$.
Wait. We need to partition $S$ into 2 Increasing chains.
Since $S$ has size $\le 2$, we can always do this.
Wait, $A$ increasing chain?
If $S$ contains $i, j$ with $i < j$.
In $p$, values are decreasing, so $p_i > p_j$.
An Increasing chain in $S$ would require $p_u < p_v$ for $u < v$.
If $S=\{i, j\}$ with $p_i > p_j$, we can't put both in $A$.
So we must split $S$ into $A, C$.
Say $i \in A, j \in C$. Then $A=\{i\}$ (Inc), $C=\{j\}$ (Inc).
Valid.
So we can cover $S$ with 2 Inc chains (length 1 each).
And cover $R$ with 2 Dec chains. Since $p$ is totally decreasing, any subset is a Dec sequence.
So $B, D$ can be anything.
Thus for $p$ totally decreasing, we can achieve $n$.

Is there any case where we fail?
Maybe $n=3$, $p=[2, 1, 3]$?
$LDS(p)=2$.
Need $S$ with $LDS(S) \le 2$.
All subsets satisfy this.
Need $R$ with $LIS(R) \le 2$.
Subsets of $\{1, 2, 3\}$.
Max LIS is 2. So all $R$ satisfy $LIS(R) \le 2$.
Wait. So for this case, trivial.

Hypothesis: We can always achieve $n$.
Why?
The problem asks us to construct $a, b, c, d$ maximizing sum.
Upper bound is $n$.
If we can always achieve $n$, then we just need to output ANY valid partition into 2 Inc and 2 Dec.
If we can't always achieve $n$, then the max is less than $n$.
Given it's a competitive programming problem, usually $n$ is achievable or close to it.
However, if $n$ is always achievable, the solution is just constructing it.
If not, we need to find a counterexample or a dynamic programming approach.
With $N=10^5$, DP is hard.
Let's try to see if $n$ is always possible.
The condition is: Exists partition $V = S \cup R$ such that $S$ is union of 2 Inc chains, $R$ is union of 2 Dec chains.
This is equivalent to: $S$ is partitionable into 2 Inc $\iff LDS(S) \le 2$.
$R$ is partitionable into 2 Dec $\iff LIS(R) \le 2$.
So we need to split $P$ into $S$ and $R$ minimizing $\max(LDS(S)-1, LIS(R)-1)$?
No. Just existence.
Is it always possible to partition a permutation into $k$ Increasing and $m$ Decreasing subsequences?
This is a variation of the "chain decomposition" problem.
Actually, any permutation can be partitioned into $k$ decreasing subsequences where $k=LIS(p)$.
Also any permutation can be partitioned into $m$ increasing subsequences where $m=LDS(p)$.
Here we have a mix.
Let's consider the specific case where $LIS(p)$ is large.
Say $p$ is identity $1, 2, \dots, n$.
$LIS=n, LDS=1$.
We need $S$ with $LDS(S) \le 2$ (easy, e.g. any 2-chain is okay, but wait. $LDS(S)$ is length of longest dec subseq. Since $p$ is Inc, any subset is Inc, so $LDS(S) = 1 \le 2$).
So we can pick $S=P$. $R=\emptyset$.
Then $LIS(R)=0 \le 2$.
So $S=P$ works. $P$ can be covered by 2 Inc chains?
Wait. $P$ is 1 Inc chain (itself). 1 Inc chain $\le$ 2 Inc chains.
So we can cover $S$ with $A=P, C=\emptyset$.
Cover $R$ with $B=\emptyset, D=\emptyset$.
So $n$ works.

Is there a case where $LDS(S) \le 2$ forces $R$ to have large LIS?
$LDS(S)$ small means $S$ is mostly Increasing.
$LIS(R)$ small means $R$ is mostly Decreasing.
Intuitively, $S$ takes the increasing nature, $R$ takes the decreasing nature.
It seems highly likely that we can always achieve $n$.
Wait. There is a paper "Partitioning a Permutation into two Increasing and two Decreasing Subsequences"? No such title.
But there is a concept of "Chains and Antichains".
Actually, looking at online resources for similar problems, often the answer is related to Dilworth's theorem.
If $LIS(p) \le k$, we can partition into $k$ decreasing.
If $LDS(p) \le m$, partition into $m$ increasing.
If we can partition into $k$ Inc and $m$ Dec?
This is equivalent to: Can we color the vertices of the DAG (representing permutation) with $k+m$ colors?
Actually, it's simpler.
Let's try to construct it greedily.
Sort of patience sorting.
We want to fill 2 piles for Increasing (let's call them Pile A, Pile C).
We want to fill 2 piles for Decreasing (Pile B, Pile D).
We process elements $p_1, \dots, p_n$.
For each $p_i$, we can assign it to A, C, B, D.
Assignment rules:
- To A: must be consistent with increasing order of A. (Top of A < current).
- To C: must be consistent with increasing order of C.
- To B: consistent with decreasing. (Top of B > current).
- To D: consistent with decreasing.
Wait. This is greedy assignment.
Is it possible to get stuck?
Greedy with patience sorting (assigning to pile with largest top, or something) finds the LIS length.
Here we have multiple piles.
Let's formalize the state.
We maintain 4 tops. $t_A, t_C$ for increasing piles. $t_B, t_D$ for decreasing piles.
Actually, for increasing piles $t$ represents the last value added. We need $current > t$.
Wait, for increasing, we append smaller values? No.
Increasing: $v_1 < v_2 < \dots$.
So if we have a pile ending in $top$, we can append $x$ if $x > top$.
To maximize flexibility?
Usually, for LIS, we want to update a pile. Patience sorting: place card on leftmost pile whose top is $\ge$ card.
Here we want to maintain validity.
If we have 2 increasing piles $A, C$.
And 2 decreasing piles $B, D$.
We iterate $i = 1 \dots n$. Element $x = p_i$.
We want to assign $x$ to one of $\{A, B, C, D\}$ such that the invariant holds.
Is it sufficient to just maintain 4 active tops?
No, because elements in the past are fixed.
If we have 4 sequences built so far.
When considering $x$, we need to find a bin where $x$ fits.
If we assign to $A$: need $x > top(A)$.
If we assign to $B$: need $x < top(B)$.
If we assign to $C$: need $x > top(C)$.
If we assign to $D$: need $x < top(D)$.
Wait. Is it that simple?
Suppose we assign $x$ to $A$. Then $A$ becomes longer.
Does adding $x$ prevent future assignments?
Yes. $top(A)$ becomes $x$.
Future elements $y$ must be $> x$ to join $A$.
If we kept $old\_top$, future elements $z$ with $old\_top < z < x$ could have joined $A$, but now can't.
To make it easy to extend, we want the "threshold" to be as small as possible.
For Increasing piles ($A, C$): threshold is $top$. We want small $top$.
For Decreasing piles ($B, D$): threshold is $top$. We need $x < top$. We want large $top$.
So we want $top(A)$ and $top(C)$ to be small.
We want $top(B)$ and $top(D)$ to be large.
Strategy:
Maintain $T_{inc} = \{top(A), top(C)\}$ (sorted).
Maintain $T_{dec} = \{top(B), top(D)\}$ (sorted).
Initially $\{-\infty, -\infty\}$ and $\{+\infty, +\infty\}$.
Incoming $x$.
We want to insert $x$ into one of the 4 piles.
Option 1: Put $x$ into an Increasing pile. This requires $x > top$.
After insertion, that pile's top becomes $x$.
To keep $top$s small, we should pick the pile with the largest valid $top$?
Wait. If we pick the pile with $top=A$, new $top=x$. If we picked $B$ (assuming it became Inc? No, Bin type is fixed).
Wait, once a bin is assigned type (Inc/Dec), it stays that way?
Actually, LIS(a) maximization implies we want $a$ to be increasing.
Does $a$ have to be fully increasing?
As established, to achieve $n$, yes.
If we stick to types, then $A, C$ are always increasing, $B, D$ always decreasing.
So we have 2 stacks growing increasing, 2 stacks growing decreasing.
Algorithm:
Iterate $x$ through $p$.
Find compatible bin.
If multiple compatible, which one?
For Increasing bins, we want to update the one with largest $top$ (so that $top$ becomes $x$, keeping others small).
Wait. If we have tops $t_1 < t_2$. And $x > t_2$.
If we put on $t_1$, new top is $x$. Remaining $t_2$. Set $\{t_1=x, t_2\}$. Max is $x$ (if $x>t_2$).
If we put on $t_2$, new top is $x$. Remaining $t_1$. Set $\{t_1, x\}$. Max is $x$.
So regardless, the max top increases to $x$.
To maintain "small tops", we don't care much, just pick any valid one.
However, for future elements, we want to have available slots with thresholds.
For Increasing: we want to have small thresholds.
For Decreasing: we want to have large thresholds.
If $x$ is compatible with both $A$ and $B$.
Put in $A$: $A$ grows. $B$ remains same.
Put in $B$: $B$ shrinks threshold? No, $B$ top becomes $x$. New threshold $x$. Old was $y > x$.
Since $B$ needs $val < top$, larger $top$ is better (more compatible).
Smaller $top$ is worse.
So for Decreasing, we want to preserve large tops.
Updating a Dec pile changes $top$ from $y$ to $x$ ($x < y$).
The new threshold $x$ is smaller (worse) than $y$.
So putting $x$ in Dec pile degrades our ability to receive further elements.
Putting $x$ in Inc pile improves (keeps $x$ as threshold).
Wait. Inc pile needs $x > top$.
Updating $top$ to $x$ ($x > old\_top$).
New threshold $x$ is larger (worse) than $old\_top$.
So updating INC pile makes it harder to add elements later.
Updating DEC pile makes it harder (smaller threshold).
Wait.
For Inc pile: Need $next > top$. Larger $top$ is bad.
Update $top$ to $x$. $x > old$. Badness increases.
For Dec pile: Need $next < top$. Smaller $top$ is bad.
Update $top$ to $x$. $x < old$. Badness increases.
So both operations increase the difficulty for that pile.
However, we have 2 Inc piles and 2 Dec piles.
We should balance them?
This looks like we just need to find *any* valid assignment that works.
With 4 piles, it's quite generous.
Maybe we can just run a simulation?
Wait, the greedy strategy with fixed order might fail.
But since we want to prove achievability or find algorithm.
Let's look at the constraint again.
Maximize sum of lengths.
If we assume $n$ is achievable, the code just needs to implement the construction.
What if $n$ is NOT always achievable?
Then what is the strategy?
Actually, there is a known result for this exact problem.
Problem is "Split permutation into 4 subsequences".
Actually, it's "Maximize LIS(a) + LDS(b) + LIS(c) + LDS(d)".
Let's check if there's a pattern.
Wait, let's consider the "worst case" scenario.
If we use patience sorting for $LIS(a)$, we generate piles.
If we just use the standard LIS algorithm, we get a decomposition into $k$ decreasing subsequences where $k = LIS(p)$.
Wait. LIS algorithm maintains piles where each pile is Decreasing.
Number of piles is $LIS(p)$.
Let these piles be $P_1, \dots, P_k$. Each is decreasing.
Lengths sum to $n$.
If we assign all of $P_1$ to $b$, all to $d$, etc.
Then $b, d$ are decreasing. $a, c$ are empty?
Sum $LDS(P_i) = \sum |P_i| = n$?
Wait. $LDS(P_i) = |P_i|$ since it is decreasing.
So if we can cover $p$ with Decreasing subsequences, we can assign them to $b, d$.
But we have a limit of 2 decreasing subsequences.
We can cover with $\ge LIS(p)$ decreasing.
If $LIS(p) \le 2$, we can assign to $b, d$. Then sum is $n$.
If $LIS(p) > 2$, we cannot cover all with 2 decreasing.
So we MUST leave some elements for $a, c$ (increasing).
Those elements must be arranged to be increasing in $a, c$.
But wait. If an element is not in $b$ or $d$, it contributes to $LIS(a)$ or $LIS(c)$.
Actually, this suggests that any element NOT contributing to LDS(b) + LDS(d) will contribute to LIS(a) + LIS(c).
Wait. Every element $x$ can contribute at most 1 to the sum.
If we classify $x$ as belonging to Inc-group ($a$ or $c$) or Dec-group ($b$ or $d$).
If $x$ is in $a$ or $c$, we hope it contributes to LIS.
If $x$ is in $b$ or $d$, we hope it contributes to LDS.
Ideally every element is counted.
To count in Inc-group, we need it to fit in an Inc-chain.
To count in Dec-group, it needs to fit in Dec-chain.
This implies we partition elements into Inc-partition and Dec-partition.
If we put $x$ in Inc-partition, it MUST be part of an Inc-chain (or we just hope it is).
Wait. If $x$ is in $a$, it extends some chain.
If we just say "contribute to LIS", we want to cover the whole permutation.
Actually, there is a simpler interpretation:
Every element $x$ has two potential roles: "Be increasing" or "Be decreasing".
If we say "Be increasing", we gain 1 point towards LIS count.
If we say "Be decreasing", we gain 1 point towards LDS count.
However, constraints apply.
If we select a set of elements to be "Increasing", say $S_{inc}$, then we must be able to cover $S_{inc}$ with 2 Inc chains.
If we select $S_{dec}$, we must cover $S_{dec}$ with 2 Dec chains.
Wait. $S_{inc}$ and $S_{dec}$ are disjoint. Union is all elements?
If we assign all to Inc, we need $LDS(P) \le 2$.
If all to Dec, we need $LIS(P) \le 2$.
If mixed, we need $LDS(S_{inc}) \le 2$ and $LIS(S_{dec}) \le 2$.
This confirms my earlier derivation.
So we need to find $S$ such that $LDS(S) \le 2$ and $LIS(P \setminus S) \le 2$.
If we can find such $S$, max score is $n$.
If not, max score $< n$.
Is it possible that no such $S$ exists?
This would mean for all $S$, either $LDS(S) > 2$ or $LIS(P \setminus S) > 2$.
This implies we can't partition into 2 Inc and 2 Dec.
Is there a counterexample?
Let $p$ be such that every subset $S$ has $LDS(S) > 2$ or complement has $LIS > 2$.
Wait. $LDS(S) > 2$ means $S$ contains a decreasing subsequence of length 3.
$LIS(S^c) > 2$ means $S^c$ contains an increasing subsequence of length 3.
This sounds like Ramsey theory.
Ramsey number $R(3, 3) = 6$.
For any graph with 6 vertices, either there is a triangle or an independent set of size 3? No.
Here we talk about sequences.
Permutations are total orders.
The statement "contains no decreasing subsequence of length 3" means width $\le 2$ (in poset terms).
Poset is defined by $i < j$ and $p_i < p_j$? No.
Let's model it properly.
Elements $1..n$ with order relations.
We are looking for a 2-coloring (Red/Blue) of vertices such that:
Red induces no "decreasing triple".
Blue induces no "increasing triple".
Is this always possible for any size $n$?
No. The size of permutation can be large.
However, a "decreasing triple" is a chain of length 3 in the reverse order.
"Increasing triple" is a chain of length 3 in forward order.
So we are coloring a DAG (transitive tournament).
We want to avoid Monochromatic paths of length 3 (3 vertices).
Wait. "Monochromatic" meaning all edges same?
No. "Induced subgraph contains a chain of length 3".
So for Red: No chain of length 3 in Reverse Order ($>$).
For Blue: No chain of length 3 in Forward Order ($<$).
Wait. The order is fixed by the permutation.
Let's visualize.
We have $n$ elements.
For every pair $(u, v)$, they are ordered.
Either $u \prec v$ (increasing) or $v \prec u$ (decreasing).
We want to color vertices Red/Blue.
Red set must not contain $u \prec v \prec w$ such that $v \prec u$ ?
No. $LDS(S) \le 2$ means no $u, v, w \in S$ with $u < v < w$ and $p_u > p_v > p_w$.
This is exactly avoiding a chain of length 3 in the relation "Index < Index" AND "Value > Value".
Let's call this relation $R$.
Similarly $LIS(S) \le 2$ avoids chain of length 3 in relation $L$ (Index < Index, Value < Value).
These are two different partial orders. $R$ is $L$ inverted in terms of values.
Wait. $p$ defines a bijection between indices and values.
Consider the poset $P$ on indices $\{1..n\}$ defined by $i <_p j \iff p_i < p_j$. (This is increasing).
Consider poset $Q$ defined by $i <_q j \iff p_i > p_j$. (This is decreasing).
We want to color vertices $V$ into Red/Blue.
Red set induces no chain of length 3 in $Q$.
Blue set induces no chain of length 3 in $P$.
Is it always possible?
If $n$ is large enough, maybe not.
Consider the Erdos-Szekeres theorem: Any sequence of length $(r-1)(s-1)+1$ contains either Inc chain of length $r$ or Dec chain of length $s$.
Here we are looking at specific colors.
If we put all elements in Red, we need no Dec chain of len 3.
This means $LDS(p) \le 2$.
If we put all in Blue, $LIS(p) \le 2$.
If $LDS(p)$ and $LIS(p)$ are large, we can't do mono-colored.
But we split.
If we can show that for any $p$, $\exists S$ with $LDS(S) \le 2, LIS(S^c) \le 2$.
Actually, we just need to output ANY valid partition.
Is it possible that we simply output a partition based on some heuristic?
Wait. If we just use the algorithm that assigns each element to the "best" bin, does it guarantee optimality?
Wait. There is a solution involving "greedy".
Actually, look at the constraints again. $N \le 100000$.
This suggests $O(N)$ or $O(N \log N)$.
If the answer is $n$, then the greedy assignment should work if implemented correctly.
If not, we need to calculate the value.
But finding the maximum value might be hard.
However, usually these constructive problems have a trick.
Wait. Maybe the answer is simply $LIS(p) + LDS(p)$?
No. We found example where $LIS(a)+LDS(b)+LIS(c)+LDS(d)$ can be $n$.
Example $p=1, 2$. $LIS=2, LDS=1$.
If we output $a=\{1\}, b=\{2\}, c=\emptyset, d=\emptyset$.
$LIS(a)=1, LDS(b)=1$. Sum 2.
$n=2$. $LIS(p)+LDS(p) = 3$.
Sum $\le n$. So $LIS(p)+LDS(p)$ is not the answer. The answer is bounded by $n$.
Wait. My manual trace for $p=[1, 2]$ gave sum 2.
$n=2$. $LIS(p)=2, LDS(p)=1$.
Can we get higher than 2?
Sum $\le n$ proof holds.
Is it possible to get sum $n$?
We established $a, c$ must be Inc, $b, d$ Dec.
So we need to partition $p$ into 2 Inc, 2 Dec.
For $p=[1, 2]$. Inc chains needed: 1 ($p$ itself). Dec chains needed: 2 (e.g. $\{1\}, \{2\}$).
We have 2 Dec bins. So we can put $\{1\}$ to $b$, $\{2\}$ to $d$.
Then $a, c$ empty.
Sum $LDS(b) + LDS(d) = 1 + 1 = 2$.
Or put $\{1, 2\}$ to $a$. $LIS(a)=2$. Sum 2.
So 2 is correct.
What about $p=[2, 1]$. $n=2$.
$LIS=1, LDS=2$.
Dec chains needed 1. Inc chains 2.
Put $\{2, 1\}$ to $b$. $LDS(b)=2$. Sum 2.
So it seems $n$ is always achievable.
Wait. Why would it be impossible?
It seems the problem reduces to: Can we always cover $P$ with 2 Inc and 2 Dec chains?
Theorem: Any permutation $P$ can be partitioned into $k$ increasing subsequences iff $LDS(P) \le k$.
Theorem: Any permutation $P$ can be partitioned into $m$ decreasing subsequences iff $LIS(P) \le m$.
We need partition into 2 Inc AND 2 Dec.
This implies we need to find $S$ such that $LDS(S) \le 2$ and $LIS(P \setminus S) \le 2$.
Wait. If we have a partition into 2 Inc ($S$) and 2 Dec ($R$), then $S$ satisfies condition 1, $R$ satisfies condition 2.
The converse: If such $S$ exists, can we partition $S$ into 2 Inc? Yes, Dilworth.
Can we partition $R$ into 2 Dec? Yes.
So existence of such $S$ is sufficient.
Is it necessary?
Yes, because $a \cup c$ is a set covered by 2 Inc, so $LDS(a \cup c) \le 2$. Let $S = a \cup c$.
Similarly $LIS(b \cup d) \le 2$. Let $R = b \cup d$.
So we definitely need to find such $S$.
Question: Does such $S$ always exist?
Wait. What if $LDS(P) > 2$? Then $S=P$ fails.
We must move some elements to $R$.
Moving element $x$ from $S$ to $R$ might reduce $LDS(S)$?
No, removing elements can only reduce (or stay same) the width of a poset?
Dilworth's: Width = length of longest chain in dual.
Removing elements reduces or keeps length.
So $LDS(S \setminus \{x\}) \le LDS(S)$.
So if we start with $S$ that fails (large $LDS$), we need to remove elements until $LDS(S) \le 2$.
Wait. If we keep removing elements from $S$, $S$ gets smaller. $P \setminus S$ gets larger.
Eventually $P \setminus S$ might become the whole set $P$.
If $LIS(P) \le 2$, then we are done (take $S=\emptyset$).
If $LIS(P) > 2$, we can't take $S=\emptyset$.
So we need to find $S$ such that $LDS(S) \le 2$ AND $LIS(P \setminus S) \le 2$.
This looks like a problem of finding a set with bounded parameters.
Is it guaranteed that we can reach a "balanced" cut?
Or does the minimum of $\max(LDS(S), LIS(S^c))$ over all $S$ drop to $\le 2$?
This quantity $\min_S \max(LDS(S), LIS(S^c))$ is exactly what we need.
Let's call this parameter $\alpha(P)$.
Is $\alpha(P) \le 2$ always true?
If $P$ is identity $1..n$. $LDS(P)=1$. Take $S=P$. $LDS(S)=1 \le 2$. $LIS(P)=n$? No $LIS(P)=n$.
Wait. $LIS(P \setminus S) = LIS(\emptyset) = 0 \le 2$.
So for Identity, $\alpha(P) = 1$. (Max of 1, 0).
Wait. $LDS(S)$ is length of longest decreasing chain in $S$.
For $S=P=1..n$, $LDS(S)=1$.
So condition $LDS(S) \le 2$ is met.
Wait. $LIS(P \setminus S) \le 2$. Here $P \setminus S = \emptyset$.
So condition met.
So for identity, answer is $n$.
For $n..1$. $LDS(P)=n$. $LIS(P)=1$.
Take $S=\emptyset$. $LDS(\emptyset)=0 \le 2$. $LIS(P)=1 \le 2$.
Answer $n$.
For $p$ such that $LDS(p)$ is large AND $LIS(p)$ is large.
Example $p = [3, 2, 1, 6, 5, 4]$. $n=6$.
$LDS(p) = 3$ ($3, 2, 1$).
$LIS(p) = 3$ ($3, 6$? No. $3 < 6$. $2 < 5$? No $2 < 4$? Yes. $1 < 4$.
Seq $3, 6$. Length 2.
Seq $2, 6$. Length 2.
Seq $3, 4$.
Longest increasing is $3, 4$? No.
$p = [3, 2, 1, 6, 5, 4]$.
Inc subseqs: $\{1, 4\}$, $\{1, 5\}$, $\{1, 6\}$, $\{2, 5\}$, $\{2, 6\}$, $\{3, 4\}$, $\{3, 5\}$, $\{3, 6\}$.
Max length 2.
Wait. $LIS(p) = 2$.
$LDS(p) = 3$.
We need $S$ with $LDS(S) \le 2$.
We have $LDS(P)=3$.
If we pick $S=P$, fails.
We need to remove elements from $P$ to kill the decreasing triple.
Dec triple is $(3, 2, 1)$.
Remove 3? Or 2? Or 1?
If we remove 3, remaining $S=\{2, 1, 6, 5, 4\}$. $LDS(S)=2$.
Check $LIS(P \setminus S) = LIS(\{3\}) = 1 \le 2$.
So $S=\{2, 1, 6, 5, 4\}$ works.
$S$ covered by 2 Inc chains.
$P \setminus S$ covered by 2 Dec chains.
So yes, it seems likely that $n$ is always achievable.
Wait. Is there any permutation with $\alpha(P) > 2$?
If $\alpha(P) > 2$, then for every $S$, either $LDS(S) \ge 3$ or $LIS(S^c) \ge 3$.
This means we can't partition into 2 Inc and 2 Dec.
Such a permutation would require $LDS(P) \ge 3$ (so $S=P$ fails) and $LIS(P) \ge 3$ (so $S=\emptyset$ fails).
But even if $LDS, LIS \ge 3$, we might find intermediate $S$.
Actually, the question is: Is there a $p$ with $\min_S \max(LDS(S), LIS(S^c)) > 2$?
This is equivalent to: Does there exist a permutation where every subset $S$ with $LDS(S) \le 2$ has $LIS(S^c) \ge 3$?
If yes, then we can't achieve $n$.
But this problem is solvable in $O(N)$.
Wait, if it's always $n$, the code is just construction.
Construction is simple:
Run Patience Sorting for increasing chains? No.
Actually, there is a known algorithm for partitioning into $k$ increasing and $m$ decreasing.
But here we need specifically 2 and 2.
If we suspect it is always $n$, we should provide a constructive algorithm.
If not, we should compute the optimal.
But computing optimal for general $N$ is hard?
Actually, the problem statement allows generating code.
Maybe the intended solution is simply:
Output $a$ as the result of running the LIS greedy algorithm on $p$ but collecting into piles?
No.
Let's try to code a greedy strategy that tries to maximize the score.
Initialize $a, b, c, d$ as empty lists.
Iterate $x$ in $p$.
If we can append $x$ to $a$ (i.e. $x > a.back()$), we do so?
If we can append to $b$ ($x < b.back()$), we do so?
We need to prioritize.
Heuristic:
Try to extend existing chains.
If we can extend $a$ or $c$ (Inc) or $b$ or $d$ (Dec), which one?
We want to maximize total sum.
Extending an Inc chain costs nothing but might block future?
Actually, since we want to cover all elements (aiming for $n$), we should accept $x$ if possible.
If $x$ cannot be placed in any currently valid position (which shouldn't happen if we plan well), then we create new chains? No, limited to 4.
If $x$ cannot extend any chain in the current 4, we have a problem?
But wait. We can assign $x$ to form a NEW chain (if we have space? No, we fix 4 chains).
Ah, we can't create new chains arbitrarily. We must assign to $a, b, c, d$.
Wait. $a, b, c, d$ are fixed to have capacity.
Actually, $a, b, c, d$ don't have fixed lengths.
They are subsequences.
The constraint is just that they must be disjoint.
Wait. "Divide into 4 disjoint subsequences". This means every element must be IN ONE OF THEM.
So every element $x$ MUST be assigned to $a, b, c,$ or $d$.
It cannot be discarded.
So we MUST assign each $x$.
If we assign $x$ to $a$, it must be that $x$ is compatible with $a$ (i.e. maintains increasing property) OR it breaks the property (and we just live with lower LIS).
But we want to maximize.
So for each $x$, we choose a target bin.
If we assign to $a$, $x$ is added to $a$.
This affects $LIS(a)$.
Actually, if we simply assign every element to $a$, we get $LIS(a) \ge 1$.
But if we assign some to $b$, maybe sum increases.
Since we want to maximize, we should make each element contribute to the length of ITS chain.
That is, if $x$ is assigned to $a$, it should be part of an increasing chain of $a$.
This means for all $y \in a$ after $x$ (by index), $p_y > p_x$.
This implies $a$ consists of an increasing chain.
If $a$ is just a single increasing chain, then $LIS(a) = |a|$.
If $a$ has "drops", then $LIS(a) < |a|$.
But we only have 2 Inc chains ($a, c$).
Wait. If $a$ is increasing, it cannot contain any decreasing pair.
So essentially, we are selecting a subset of elements $A$ which form an Increasing chain.
Another subset $C$ forming Increasing chain.
Subset $B$ forming Decreasing.
Subset $D$ forming Decreasing.
This returns to the Dilworth/Pigeonhole setup.
We are back to: Can we partition $P$ into 2 Inc and 2 Dec chains?
If yes, answer is $n$.
My analysis suggests yes is almost certainly true or the problem relies on a specific construction.
Wait, if I output a partition that fails (not 2 Inc/2 Dec), the sum will be less than $n$.
But since $n$ is the upper bound, if we can achieve $n$, that's perfect.
If we cannot, we should try to maximize the sum.
Let's assume $n$ is achievable.
How to construct?
We can use the property that any poset can be partitioned into $k$ chains where $k$ is the size of max antichain.
Here, Inc-chain is a chain in the order $(i, p_i)$ (value-wise increasing).
Dec-chain is a chain in the order $(i, p_i)$ (value-wise decreasing).
Wait. An Inc chain corresponds to a chain in the poset defined by $u \prec v$ if $idx(u) < idx(v)$ and $p_u < p_v$.
This is just the standard chain in $P$.
Wait. $a$ is a subsequence. $A$ is a chain in the poset $P$.
$c$ is also a chain in $P$.
So we need to cover $P$ with 2 chains in $P$ and 2 chains in $P^{op}$ (reverse order of values).
Wait. $a, c$ are chains in $P$ (standard increasing).
$b, d$ are chains in $P^{op}$ (decreasing).
We need to decompose $P$ into $k=2$ chains of $P$ and $m=2$ chains of $P^{op}$.
Is it possible to decompose a poset into 2 chains and 2 reverse chains?
This sounds like we are decomposing the permutation matrix into Young Tableaux?
Actually, there's a theorem by Greene-Kleitman or similar.
Actually, simpler:
The problem asks for max of $\sum |Chain_i|$ subject to constraints.
Since $\sum |Chain_i| = n$ is the max, and it seems reachable.
The construction for partitioning a permutation into $k$ increasing and $m$ decreasing subsequences is not trivial for general $k, m$.
But for small $k, m$?
Wait. If $k=2, m=2$.
Maybe we can just use a priority queue?
Algorithm idea:
Use a multiset of tails for increasing chains? No.
We need 2 chains in $P$. This is a classic decomposition problem.
We can decompose $P$ into $LDS(P)$ increasing chains.
Let $k = LDS(P)$. If $k \le 2$, we are done for Inc part.
If $k > 2$, we can't cover with 2.
But we can cover with $k$ chains, and assign some chains to $a$, some to $c$? No, we only have 2 slots.
Wait. If $LDS(P) = 3$, we need 3 increasing chains to cover $P$.
This means we have at least 3 disjoint increasing chains.
Can we merge some? No, merging increases decreasing subseqs.
Wait. Dilworth: Min # increasing chains = $LDS(P)$.
So if $LDS(P) = 3$, we MUST use at least 3 increasing chains to cover $P$.
But we only have 2 bins ($a, c$).
This implies we CANNOT cover $P$ with 2 increasing chains if $LDS(P) \ge 3$.
This implies we MUST move some elements to $B, D$.
Elements in $B, D$ do NOT contribute to Inc-cover.
So we select a subset $S \subset P$ such that $LDS(S) \le 2$.
The complement $P \setminus S$ must be coverable by 2 Dec chains $\implies LIS(P \setminus S) \le 2$.
So indeed, the problem IS: Find $S$ such that $LDS(S) \le 2$ and $LIS(S^c) \le 2$.
And then decompose $S$ into 2 Inc, $S^c$ into 2 Dec.
The question is: Does such $S$ always exist?
Wait. I haven't found a counter-example yet.
Let's check the literature.
"Partition a permutation into 2 increasing and 2 decreasing subsequences".
This is equivalent to asking if the permutation has dimension $\le 4$?
No.
Actually, the maximum possible score is $n$.
Why?
Consider the problem from a different angle.
Assign each $p_i$ a value $v_i \in \{1, 2, 3, 4\}$.
We want to maximize $|\{(i, j) : i < j, p_i < p_j, v_i, v_j \in \{1, 2\}\}| + \dots$
No.
Let's assume there is a valid assignment achieving $n$.
The problem asks to output $r, s, p, q$ and the subsequences.
The code just needs to produce *a* solution.
If I write a solution that tries to find such $S$, and outputs a valid partition, it's correct.
Finding $S$ efficiently?
Wait. If $S$ doesn't exist, we have to settle for $<n$.
But given constraints and typical CP style, $n$ is likely achievable.
Let's try to construct a greedy solution.
Wait. If I simply run the following:
1. Compute the decomposition of $P$ into increasing subsequences (via patience sorting on $p^{-1}$? No. Standard patience sorts on values).
Standard patience sorting gives a decomposition into $L$ decreasing subsequences.
Wait. Patience sorting on $P$ (placing element $x$ on first pile with top $\ge x$) creates piles of decreasing elements. Number of piles is $LIS(p)$.
This gives a decomposition into $LIS(p)$ Dec-chains.
If $LIS(p) \le 2$, we put all into $b, d$ (Dec). $a, c$ empty. Score $n$.
If $LIS(p) > 2$, we can't do this.
However, we can use the dual decomposition.
We can decompose $P$ into $LDS(p)$ Increasing chains.
If $LDS(p) \le 2$, put all into $a, c$ (Inc). Score $n$.
If both $LIS(p) > 2$ and $LDS(p) > 2$?
Then neither full Inc nor full Dec works.
We need a mix.
Is it possible that if $LIS, LDS > 2$, we can't achieve $n$?
Let's check small cases.
$p$ such that $LIS \ge 3$ and $LDS \ge 3$.
Smallest such $n$?
By Erdos-Szekeres, $R(3, 3) = 6$.
So for $n=6$, any perm has Inc len 3 or Dec len 3.
If we take $p$ such that $LIS=3$ and $LDS=3$?
Example $p = [2, 1, 4, 3, 6, 5]$.
LIS: $2, 4, 6$ (len 3).
LDS: $2, 1$, $4, 3$, $6, 5$ (len 2).
Wait, LDS of $[2, 1, 4, 3, 6, 5]$?
Subsequence $4, 3$. $6, 5$.
$4, 3, 2$? No 2 is at start. $2, 1$.
Is there length 3?
$4, 3$ is 2. $2, 1$ is 2.
Wait. $2, 1$ then $4, 3$?
Indices $1, 2$ ($2, 1$). Indices $3, 4$ ($4, 3$).
Can we pick $1, 2, 4$ ($2, 1, 3$)? No. $p_2=1, p_4=3$. $1 < 3$ (increasing).
Wait. I want decreasing.
$p = [2, 1, 4, 3, 6, 5]$.
Decreasing:
Pairs $(2,1), (4,3), (6,5)$.
Any triple?
$(4, 3, ?)$. Next available smaller is none.
So LDS is 2.
Wait.
I need an example where $LIS \ge 3$ AND $LDS \ge 3$.
The "canonical" example for $R(3,3)=6$ is a grid permutation.
Like rows sorted, cols reversed?
Matrix $\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}$ flattened?
$1, 2, 3, 4, 5, 6$ -> LIS 6, LDS 1.
We want to shuffle to break both.
Try $p = [2, 1, 4, 3, 6, 5]$ failed (LDS 2).
Try $p = [2, 4, 1, 3, 6, 5]$.
LIS: $2, 4, 6$ (3). $1, 3, 6$ (3). $2, 3, 6$ (3).
LDS: $4, 3$. $4, 1$ (no 1 after 4? yes). $4, 3$. $4, 1$ (index 3).
Wait. $4$ at index 2. $1$ at index 3. $p_2=4, p_3=1$.
$2, 4, 1$. $p_1=2, p_2=4, p_3=1$.
$4 > 1$. $2 > 1$.
Longest decreasing?
$4, 3$. $4, 1$ (len 2).
$6, 5$.
Is there length 3?
$4, 1$ (indices 2, 3).
$2, 1$ (indices 1, 3).
Seems LDS is 2.
Wait. $R(3, 3)$ is for general sequences. For permutations, $R(3, 3)$ guarantees either LIS 3 or LDS 3.
So for $n=5$, it's possible to have $LIS < 3$ and $LDS < 3$.
For $n=6$, at least one is $\ge 3$.
Wait. $R(r, s)$ is length such that one chain $\ge r$ or one antichain $\ge s$.
Antichain in permutation order corresponds to decreasing subsequence.
So $R(3, 3)$ means if $n=6$, then either $LIS \ge 3$ OR $LDS \ge 3$.
It DOES NOT guarantee BOTH.
For example $p=[1, 2, 3, 4, 5, 6]$. $LIS=6, LDS=1$.
To have BOTH $LIS \ge 3$ and $LDS \ge 3$, we need a permutation that avoids "simple" structures.
Example: $p = [2, 3, 4, 1, 5, 6]$?
LIS: $2, 3, 4, 5, 6$ (5).
LDS: $4, 1$ (2).
We need "zigzag".
$p = [3, 2, 1, 6, 5, 4]$.
LIS: $3, 6$ (2).
LDS: $3, 2, 1$ (3).
Still LIS small.
We need high in both.
Try $p = [2, 3, 4, 1, 5, 6]$ -> LIS 5, LDS 2.
Actually, it is known that max min(LIS, LDS) is $\sqrt{n}$.
If we want both $\ge 3$, we need $n \ge 9$?
Wait. $R(3, 3) = 6$. This means if $n=6$, we are forced to have $\ge 3$ in one direction.
But we can have 3 in one and 1 in other.
So if we encounter a case where we have both $\ge 3$, then we CANNOT cover with 2 Inc and 2 Dec?
If $LIS \ge 3$ and $LDS \ge 3$.
We need $LDS(S) \le 2$. So $S$ must miss at least one element from every Dec-chain of length 3?
If there are many Disjoint Dec-chains?
If $LIS(P) \ge 3$, then there exists Inc chain of length 3. This means we have 3 elements $x_1 < x_2 < x_3$ (indices/values).
We need to put them in Dec bins or Inc bins.
Wait. $x_1, x_2, x_3$ is increasing.
We can put them in Dec bins?
If we put in Dec bins, say $x_1 \in b$, $x_2 \in d$.
Dec bins forbid increasing pairs.
Wait. $b, d$ are decreasing.
So if $x_1, x_2$ are in $b$, we need $p_{x_1} > p_{x_2}$?
But $x_1, x_2$ are increasing, so $p_{x_1} < p_{x_2}$.
So $x_1, x_2$ CANNOT be both in $b$ or both in $d$.
So they must be separated.
Similarly for Inc bins $a, c$. They forbid decreasing pairs.
If we have a decreasing chain $y_1, y_2, y_3$.
They cannot be both in $a$ or $c$.
So we need to separate elements of Inc chains into $b, d$ and elements of Dec chains into $a, c$.
Wait.
If we have an Inc chain $I = \{x_1, \dots, x_k\}$ ($k \ge 3$).
We must assign elements of $I$ to bins.
Constraint: No two adjacent in $I$ can go to $b$ together?
Actually, constraint is: $b, d$ cannot contain any increasing pair.
So $b$ cannot contain $\{x_i, x_j\}$ if $i < j$ and $p_{x_i} < p_{x_j}$.
Since all pairs in $I$ are increasing, NO two elements of $I$ can be in $b$.
And NO two elements of $I$ can be in $d$.
So ALL elements of $I$ must be in $a \cup c$.
If $|I| = 3$, we need to put 3 elements into $a$ and $c$.
Constraint on $a, c$: No decreasing pair.
But elements of $I$ are increasing. So any pair is an increasing pair.
Decreasing pairs forbidden.
Increasing pairs are allowed in $a, c$.
So we can put all elements of $I$ into $a \cup c$.
Wait. This is consistent.
We need to put $I$ into $a \cup c$.
$a, c$ must be Increasing. $I$ is Increasing.
So we can put $I$ into $a$ (if we want).
So Inc-chains are happy in $a, c$.
The problem arises with Dec-chains.
If we have a Dec chain $D = \{y_1, \dots, y_m\}$ ($m \ge 3$).
Elements of $D$ form a decreasing sequence.
$a, c$ (Inc bins) forbid decreasing pairs.
So $D$ cannot be in $a \cup c$.
All elements of $D$ must be in $b \cup d$.
$b, d$ (Dec bins) forbid increasing pairs.
Elements of $D$ are decreasing. So no increasing pairs in $D$.
So $D$ can be put into $b \cup d$.
Wait.
So if we have an Inc-chain $I$ (len $\ge 3$), it forces its elements into $a \cup c$.
If we have a Dec-chain $D$ (len $\ge 3$), it forces its elements into $b \cup d$.
Is there a conflict?
If $I$ forces elements to $a \cup c$, and $D$ forces elements to $b \cup d$.
We need $a, c$ to cover $I$.
We need $b, d$ to cover $D$.
This requires that we can partition $V$ into $S_{inc}$ (covered by $a, c$) and $S_{dec}$ (covered by $b, d$).
And we need $S_{inc}$ to be coverable by 2 Inc chains?
No. The condition is:
If $x \in S_{inc}$, it must go to $a$ or $c$.
For $a, c$ to be valid, they must not contain Dec-pairs.
If $S_{inc}$ contains a Dec-triple, can we distribute it into $a, c$ without conflict?
No. If $u, v, w \in S_{inc}$ form a Dec-triple ($u<v<w, p_u>p_v>p_w$).
Then we cannot put $\{u, v\}$ in $a$. Cannot put $\{u, v\}$ in $c$.
We cannot put $\{u, w\}$ in $a$.
We must put each into different bins?
If $u \in a, v \in c, w \in a$?
Then $u, w$ in $a$. Check condition $p_u > p_w$. Violation!
So we cannot put $u, w$ in $a$.
So $u, w$ must be separated.
Actually, if we have a Dec-triple $u, v, w$.
We cannot put ANY PAIR in the same Inc-bin.
So we need to assign colors to $u, v, w$ such that no pair shares the same color?
But we have only 2 Inc-bins ($a, c$).
By Pigeonhole Principle, among 3 elements $u, v, w$, at least 2 must share a bin.
If they share a bin, they form a Dec-pair in that bin.
Which is forbidden.
Conclusion: We CANNOT tolerate any Dec-triple in $S_{inc}$.
Thus, $S_{inc}$ must not contain any decreasing subsequence of length 3.
This implies $LDS(S_{inc}) \le 2$.
Similarly, $S_{dec}$ must not contain any increasing subsequence of length 3.
Implies $LIS(S_{dec}) \le 2$.
This brings us back to:
Can we partition $P$ into $S, T$ such that $LDS(S) \le 2$ and $LIS(T) \le 2$?
If yes, we can achieve score $n$.
If not, we must sacrifice some elements (score $< n$).
Is it possible that no such partition exists?
Consider $p$ constructed by interleaving a long Dec chain and a long Inc chain?
Wait. If $LDS(P) \le 2$, we are done ($S=P$).
If $LIS(P) \le 2$, we are done ($T=P$).
The problematic case is when both $LDS(P) \ge 3$ and $LIS(P) \ge 3$.
Actually, we can just check if $n$ is achievable by solving the decision problem.
But calculating exact partition for $N=100000$ is non-trivial if not always $n$.
However, notice that if $S$ is chosen such that $LDS(S) \le 2$, $S$ must consist of elements that don't form a decreasing triple.
By Dilworth, $S$ can be partitioned into 2 Inc chains.
Similarly $T$ can be partitioned into 2 Dec chains.
So finding $S$ is the core task.
Is it true that $\forall P, \exists S, LDS(S) \le 2 \land LIS(P \setminus S) \le 2$?
This looks like a variation of "Ramsey property".
Actually, for $n$ large enough, the answer to "can we partition into 2 Inc and 2 Dec" is YES.
Wait. The theorem says any poset can be partitioned into $k$ chains if height is small? No.
There is a result: The chromatic number of the comparability graph is the size of the max antichain? No, that's Dilworth.
Actually, for permutations, it is known that $LIS(p) + LDS(p) \ge n$? No.
But maybe there is a stronger condition.
Wait. Let's assume the problem implies we should output $n$.
If I implement a greedy strategy, and it gets stuck, I'll fail.
But maybe there's a construction.
Greedy Strategy for $S$:
We want to identify elements to put in $S$ (Inc-friendly).
Condition: $S$ must contain no Dec-triple.
This means for every decreasing subsequence of length 3 in $P$, we can remove at least one element.
Wait. $S$ is a subset. If $P$ has a Dec-triple, $S$ must miss at least one.
If $P$ has many Dec-triples, maybe intersection is non-empty.
This feels like Hitting Set on hypergraphs (triples).
But we also have $LIS(P \setminus S) \le 2$.
This means $T$ must miss at least one from every Inc-triple.
If $P$ has lots of triples of both kinds.
But usually triples intersect?
In permutations, triples are somewhat correlated.
If $u < v < w$ is Dec ($p_u > p_v > p_w$).
Can it be that $T$ (complement) has an Inc-triple?
Actually, if we pick $S$ to be an interval? Or based on rank?
Consider the construction from a known Codeforces problem.
"Divide into 4 subsequences maximizing..."
Maybe the answer is simply:
Use `std::vector` to store `a`, `b`, `c`, `d`.
Assign each element greedily?
Or better:
Sort the permutation?
If we simply sort the indices such that we get the Longest Increasing Subsequence?
Wait. If we want $a$ to be LIS.
Let's construct $a$ as the Longest Increasing Subsequence of $p$.
Then $b$ as Longest Decreasing Subsequence of remaining?
Let's try this greedy:
1. Extract $a$ as an LIS of $p$.
2. Extract $c$ as an LIS of $p \setminus a$.
3. Extract $b$ as an LDS of $p \setminus (a \cup c)$.
4. $d$ takes the rest.
Does this maximize?
Example $p=[2, 1, 4, 3]$.
LIS: $2, 4$ (len 2). $a=\{2, 4\}$.
Rem: $1, 3$.
LIS of rem: $1, 3$ (len 2). $c=\{1, 3\}$.
Empty.
Sum $LIS(a)+LIS(c) = 4$.
Wait. $a, c$ must be disjoint.
Indices of $a$: indices 1, 3 (values 2, 4).
Indices of $c$: indices 2, 4 (values 1, 3).
$LIS(a)=2, LIS(c)=2$.
$b, d$ empty.
Sum 4. $n=4$.
This works!
Example $p=[3, 2, 1]$.
LIS: $3$ (or 2 or 1). $a=\{3\}$.
Rem: $\{2, 1\}$.
LIS: $\{2\}$. $c=\{2\}$.
Rem: $\{1\}$.
LDS: $\{1\}$. $b=\{1\}$.
Empty $d$.
Sum $1+1+1=3$.
Wait. We got 3.
But we could have put all in $b$ for LDS=3.
Wait. Greedy choice of LIS might hurt.
In $p=[3, 2, 1]$, LIS is 1. $a=\{3\}$.
If we took $a=\{3, 2\}$ (impossible).
Wait. LIS is a sequence.
$p=[3, 2, 1]$.
LIS candidates: $\{3\}, \{2\}, \{1\}$.
If we pick $a=\{3\}$ (first).
Next we find LIS in $\{2, 1\}$. $\{2\}$.
Then LDS in $\{1\}$. $\{1\}$.
Sum 3.
But wait. If we assign $b=\{3, 2, 1\}$, sum is 3.
It seems hard to beat 3 here.
But wait. Can we do better? $LIS(a) + LDS(b) + \dots \le n$.
So 3 is max.
What if $p=[2, 1, 4, 3, 6, 5]$. $n=6$.
LIS: $2, 4, 6$ (len 3). $a=\{2, 4, 6\}$.
Rem: $1, 3, 5$.
LIS: $1, 3, 5$ (len 3). $c=\{1, 3, 5\}$.
Sum 6. $b, d$ empty.
Achieved 6.
Seems greedy works.
But is there a case where taking LIS first hurts?
Suppose $p=[1, 2, 3, 4, 5]$.
LIS takes all. $a=\{1..5\}$. $LIS=5$. Sum 5.
Suppose $p=[5, 4, 3, 2, 1]$.
LIS takes $\{5\}$ (first). $a=\{5\}$.
Rem $\{4, 3, 2, 1\}$.
LIS takes $\{4\}$. $c=\{4\}$.
Rem $\{3, 2, 1\}$.
LDS takes $\{3, 2, 1\}$. $b=\{3, 2, 1\}$.
Sum $1+1+3 = 5$.
Optimal is 5 (put all in $b$).
Wait. My greedy gave 5.
So it seems robust.
Let's refine the greedy.
Iteratively:
1. Find LIS of remaining $P$. Assign to $a$ (or alternate to $c$?).
   Wait. If I assign to $a$, I use 1 "Inc" slot.
   I have 2 Inc slots ($a, c$).
   Should I fill $a$ completely? Or interleave?
   Maybe filling $a$ then $c$ is safe.
   What about $b, d$?
   Fill $b, d$ last.
   Wait. If I fill $a$ with LIS, does it use up "good" elements for $b$?
   Elements in $a$ are increasing.
   If an element $x$ was critical for a long Decreasing chain in $b$, putting it in $a$ removes it from $b$.
   However, $x$ contributes to $a$'s score.
   Is it worth sacrificing $b$'s length to save $a$'s?
   But we already captured $x$ in $a$.
   The trade-off is:
   Option 1: $x \in a$. Gain $+1$ to sum (since it increases length of Inc chain). Cost? Might reduce potential for $b$.
   Option 2: $x \in b$. Gain $+1$ to sum (increases length of Dec chain).
   Since max sum is $n$, and each element contributes at most 1.
   Every element contributes 1 IF IT IS PART OF SOME CHAIN COVERAGE.
   Wait. If $x \in a$, it MUST be part of an increasing chain in $a$ to count.
   If $a$ is not just a chain but a collection of elements where some pairs violate order, then $x$ might not count?
   But if we enforce $a$ is Increasing (as much as possible), we can count it.
   Actually, if we enforce $a$ to be an increasing subsequence, then any element added to $a$ increases LIS length by 1 (since it appends).
   So essentially we are partitioning elements into $A, B, C, D$ such that $A, C$ are increasing subsequences and $B, D$ are decreasing subsequences.
   We want to maximize $|A| + |B| + |C| + |D| = n$.
   This requires that $A \cup B \cup C \cup D = P$ (always true) and each set forms a valid subsequence type.
   So the question is exactly: Can we partition $P$ into 2 Inc and 2 Dec?
   If yes, we get $n$.
   If no, we get less.
   But wait. If we assign $x$ to $A$, we enforce $A$ is increasing.
   If we can't find such a partition, then we can't get $n$.
   But the greedy strategy of "Extract LIS, put in A" ensures $A$ is increasing.
   But does it prevent finding a partition?
   Suppose $p=[3, 1, 2, 4]$.
   LIS: $3, 4$. $a=\{3, 4\}$.
   Rem: $1, 2$.
   LIS: $1, 2$. $c=\{1, 2\}$.
   Sum 4.
   Partition $A=\{3, 4\}$ (Inc), $C=\{1, 2\}$ (Inc).
   Works.
   Suppose $p=[4, 3, 2, 1]$.
   LIS: $4$. $a=\{4\}$.
   Rem: $3, 2, 1$.
   LIS: $3$. $c=\{3\}$.
   Rem: $2, 1$.
   LDS: $2, 1$. $b=\{2, 1\}$.
   Sum $1+1+2=4$.
   Partition works.
   It seems we can always partition into 2 Inc and 2 Dec.
   Why? Because we can just take the whole permutation $P$.
   If $LDS(P) \le 2$, we put $P$ in $A, C$. (Covered by 2 Inc).
   If $LIS(P) \le 2$, we put $P$ in $B, D$. (Covered by 2 Dec).
   If both $> 2$?
   Actually, if both $> 2$, can we split?
   Wait. I am claiming that we can ALWAYS partition into 2 Inc and 2 Dec.
   Is this true?
   The claim is equivalent to saying $\min_S \max(LDS(S), LIS(S^c)) \le 2$.
   Wait. If $LDS(P) \le 2$, take $S=P$. Then $LDS(S) \le 2$, $LIS(S^c) \le 2$.
   If $LIS(P) \le 2$, take $S=\emptyset$.
   If both $> 2$, is there a split?
   Consider $P$ constructed to have $LDS=3, LIS=3$.
   Example $p = [2, 3, 1, 4]$.
   $LDS=2$ (2,1; 3,1). Wait. $2,1$. $3,1$.
   Indices $1, 3$. $p_1=2, p_3=1$.
   $p_2=3, p_3=1$.
   $LIS=2$ (2,3 or 2,4 or 1,4 or 3,4).
   Wait. $n=4$. $LIS, LDS \le 2$ always for $n < 6$?
   No.
   Try to find a counterexample to "Always possible to partition into 2 Inc and 2 Dec".
   This would require a permutation where for every $S$, either $LDS(S) \ge 3$ or $LIS(S^c) \ge 3$.
   If such permutation exists, we can't achieve $n$.
   Actually, for $n=6$, we can construct a permutation with $LIS \ge 3$ and $LDS \ge 3$.
   Example $p = [2, 4, 1, 3, 6, 5]$.
   Let's check properties.
   $LIS$: $2, 4, 6$ (3). $2, 3, 6$ (3).
   $LDS$: $4, 3$ (2). $4, 1$ (2). $4, 1$ is dec. $2, 1$ is dec.
   Are there any Dec len 3?
   $4, 1$. Next is 6, 5 (indices > 4). $1 < 5$. No.
   $3, 1$? $p_4=3$. $p_5=6, p_6=5$. $3 < 5$.
   Wait. Dec seq must have $idx$ increasing, $val$ decreasing.
   $p_1=2, p_3=1$. (len 2).
   $p_2=4, p_3=1$. (len 2).
   $p_2=4, p_4=3$. (len 2).
   Seems LDS=2.
   It seems hard to get both $\ge 3$ with small $n$.
   Actually $R(3, 3) = 6$ says we force at least one to be $\ge 3$. It doesn't say both.
   So for $n=6$, it's possible to have $LIS=6, LDS=1$.
   We need BOTH $\ge 3$ to force a contradiction?
   If we can find a permutation with BOTH $\ge 3$, does that imply impossibility?
   Not necessarily. It just means $S=P$ fails and $S=\emptyset$ fails.
   We need an intermediate $S$.
   If $p$ has $LIS \ge 3$ and $LDS \ge 3$.
   Suppose we pick $S$ to break all Dec triples.
   Does breaking all Dec triples force $LIS(S^c) \ge 3$?
   If $S$ kills all Dec triples, $LDS(S) \le 2$.
   If $LIS(S^c)$ must be large...
   Let's check $p=[3, 2, 1, 6, 5, 4]$.
   $LDS=3$ ($3, 2, 1$).
   $LIS=2$ ($3, 6$).
   Here $LIS(P)=2 \le 2$. So $S=\emptyset$ works.
   We need a case where $LIS(P)$ is large AND $LDS(P)$ is large.
   Such a permutation has "long increasing" and "long decreasing" subsequences.
   Example: Zig-zag?
   $n=8$. $1, 8, 2, 7, 3, 6, 4, 5$.
   LIS: $1, 2, 3, 4, 5$ (5).
   LDS: $8, 7, 6, 5$ (4).
   $LIS \ge 3, LDS \ge 3$.
   Can we partition into 2 Inc and 2 Dec?
   Let's check.
   We need $S$ such that $LDS(S) \le 2$.
   $P$ has Dec chains $8, 7, 6, 5$.
   Elements involved: $\{1, 8\}$ (idx 2). $\{2, 7\}$ (idx 3). $\{3, 6\}$ (idx 5). $\{4, 5\}$ (idx 7? No, $p_7=4, p_8=5$. 5 comes after 4. $4 < 5$).
   Wait.
   $p = [1, 8, 2, 7, 3, 6, 4, 5]$.
   Dec subsequences:
   $8, 7, 6, 5$? No. Indices $2, 3, 5, 8$. $p_2=8, p_3=7, p_5=6, p_8=5$.
   This is a decreasing subsequence of length 4.
   To make $LDS(S) \le 2$, we must hit all Dec subsequences of length 3.
   Specifically, the chain $8, 7, 6, 5$ has length 4.
   We need to remove at least 2 elements from this chain?
   Wait. Dilworth on $S$: if $S$ contains $8, 7, 6, 5$, then $LDS(S) \ge 4 > 2$.
   So $S$ cannot contain all of them.
   Actually, $LDS(S) \le 2$ means $S$ can be covered by 2 Inc chains.
   This means $S$ is a union of 2 chains in the "poset of indices/value".
   Wait. If $S$ contains $8, 7, 6, 5$ (values).
   Indices $2, 3, 5, 8$.
   If all these 4 indices are in $S$.
   Are they comparable in the Inc-poset?
   Inc-poset: $u \prec v \iff idx(u) < idx(v) \land p_u < p_v$.
   For $8, 7, 6, 5$.
   $8$ at idx 2. $7$ at idx 3. $p_2 > p_3$. Not $\prec$.
   $8$ at idx 2. $5$ at idx 8. Not $\prec$.
   $7$ at idx 3. $6$ at idx 5. Not $\prec$.
   $7$ at idx 3. $5$ at idx 8. Not $\prec$.
   $6$ at idx 5. $5$ at idx 8. Not $\prec$.
   None of them form an increasing pair.
   So they form an antichain in the Inc-poset.
   So $LDS(S)$ counts size of antichains in Inc-poset.
   If $S$ contains $\{8, 7, 6, 5\}$, then $LDS(S) \ge 4$.
   So $S$ cannot contain all of them.
   So $S$ must miss at least one element from $\{8, 7, 6, 5\}$.
   In fact, if we want $LDS(S) \le 2$, $S$ cannot contain 3 pairwise incomparable elements (an antichain of size 3).
   The set $\{8, 7, 6, 5\}$ is an antichain.
   We need to remove enough to break all antichains of size 3.
   Actually, if we keep 2 elements from this antichain, $LDS(S)$ might be small?
   Wait. Antichain means no two are connected by Inc-edge.
   So any pair is a Dec-pair.
   If $S$ has $\{u, v, w\}$ from antichain.
   Then $u, v$ is Dec-pair. $v, w$ is Dec-pair. $u, w$ is Dec-pair.
   Wait. $u, v$ is Dec-pair.
   So $S$ has 3 Dec-pairs? No.
   If $S$ has 3 elements from antichain, say $x, y, z$.
   Any pair is a decreasing pair.
   So we have a clique of size 3 in Dec-graph.
   $LDS(S)$ is the length of longest decreasing chain.
   Since they are antichain in Inc-poset, they form a chain in Dec-poset?
   Wait. If they are antichain in Inc, does it mean they are a chain in Dec?
   Not necessarily. Antichain in Inc means for all distinct $u, v$, $\neg(u \prec v)$.
   $\neg(u \prec v)$ means $u$ and $v$ are not comparable in Inc.
   Comparable in Inc means $idx(u)<idx(v) \land val(u)<val(v)$.
   So for all pairs, either $idx(u)<idx(v) \implies val(u)>val(v)$ (Dec) or $idx(u)>idx(v)$ and $val(u)<val(v)$? No.
   If we consider the permutation matrix, Inc is NE direction. Dec is NW.
   If $u, v$ is antichain in Inc, it means they are incomparable.
   Wait. Dilworth: Min Inc chains = Max Dec chain (Antichain in Inc is Dec chain).
   Yes. An antichain in the Inc-poset corresponds EXACTLY to a Decreasing Subsequence.
   So if $S$ contains a Dec-subsequence of length 3 (antichain size 3), then $LDS(S) \ge 3$.
   So $LDS(S) \le 2$ is equivalent to: $S$ contains no antichain of size 3.
   Which is equivalent to: The width of the poset restricted to $S$ is $\le 2$.
   Which is equivalent to: $S$ can be partitioned into 2 Chains (Inc-sequences).
   Okay, this matches my earlier thought.
   
   Now, we have $S$ with width $\le 2$.
   And $S^c$ with height $\le 2$ (LIS $\le 2$).
   Height of $S^c$ in Inc-poset $\le 2$.
   This means $S^c$ contains no chain of size 3.
   
   The question: Can we always split the vertices of a poset (derived from permutation) into $S, T$ such that width($S$) $\le 2$ and height($T$) $\le 2$?
   Wait. Height of $T$ is LIS($T$).
   Width of $S$ is LDS($S$).
   Actually, for any poset $P$, is it true that $\exists$ cut $(S, T)$ with width($S$) $\le 2$ and height($T$) $\le 2$?
   If $P$ is the "zigzag" poset from $n=8$ example:
   $1 \prec 8 \succ 2 \prec 7 \dots$? No.
   Let's check $p=[1, 8, 2, 7, 3, 6, 4, 5]$.
   Inc-edges: $(1, 8)$ NO ($1 < 8$). Yes.
   $(1, 2), (1, 3)... (1, 5)$.
   $(2, 7)$.
   $(3, 6), (3, 5)$.
   $(4, 5)$.
   $(8, ?)$ No outgoing from 8.
   $(7, ?)$ No.
   $(6, ?)$ No.
   So it's basically a tree-like structure or something.
   Antichains (Dec-subs): $\{8, 7, 6, 5\}$ size 4.
   Chains (Inc-subs): $\{1, 2, 3, 4, 5\}$ size 5.
   Width of $P$ is 4. Height is 5.
   We want to split $S$ with width $\le 2$.
   If we remove elements to reduce width to 2.
   Removing elements from $P$.
   If we take $S = \{8, 7, 6, 5\}$. Width is 4.
   If we take $S = \{1\}$. Width is 1.
   If we take $S$ to be empty. Width 0.
   If $S=\emptyset$, then $T=P$. Height of $P$ is 5. Fails ($5 > 2$).
   If $T=\emptyset$, then $S=P$. Width 4. Fails.
   So we need intermediate.
   Try $S = \{ \text{elements that make } LDS \le 2 \}$.
   We need to eliminate all antichains of size 3.
   The large antichain is $\{8, 7, 6, 5\}$.
   To break all size-3 subsets, we need to remove enough.
   Actually, we just need to ensure no subset of size 3 forms antichain.
   In $\{8, 7, 6, 5\}$, any 3 is an antichain.
   So $S$ can contain at most 2 elements from $\{8, 7, 6, 5\}$.
   Suppose we pick $S$ containing $\{8, 7\}$ and NO other from the big set.
   $S$ can contain other elements (like 1, 2, 3, 4) as long as they don't form antichain with $\{8, 7\}$.
   $\{8, 7\}$ is an antichain.
   Can we add 1?
   $1$ is at bottom. $1 \prec 8$. So $1, 8$ comparable. $1 \prec 7$.
   $1, 8, 7$? $1 \prec 8$, $8 \times 7$ (incomparable).
   So $\{1, 8, 7\}$ contains pair $\{8, 7\}$ incomparable.
   Does it contain 3 incomparable?
   No. Only $\{8, 7\}$.
   So $\{1, 8, 7\}$ has antichain size 2.
   Wait. Antichain definition: subset where NO two are comparable.
   In $\{1, 8, 7\}$, 1 is comparable to both. 8, 7 incomparable.
   So max antichain size is 2.
   So width is 2.
   So we can include $1$.
   Can we include 2?
   $2 \prec 7, 2 \prec 6$ (wait, 6 not in set).
   $2 \prec 7$.
   $2 \prec 8$ (No, $p_1=1, p_3=2$. $1 < 3$ and $1 < 2$. $1 \prec 2$.)
   So $2$ is below $7$.
   Wait. $p_3=2$. $p_2=8$. $3 > 2$. $2 < 8$.
   Inc-order is by index. $idx(2) < idx(8)$?
   Indices: $1(1), 2(8), 3(2), 4(7), \dots$
   $idx(2)=3, idx(8)=2$.
   So $idx(8) < idx(2)$ and $8 > 2$. So $8 \times 2$ (in Inc).
   So $\{2, 8\}$ is incomparable.
   So if $S$ has $2$ and $\{8, 7\}$.
   $\{2, 8, 7\}$. Pairs: $(2, 8)$ inc. $(2, 7)$ inc? $idx(2)=3, idx(7)=4$. $3 < 4$. $2 < 7$. Yes.
   $(8, 7)$ incomparable.
   So $\{2, 8, 7\}$ has antichain $\{8, 7\}$. Size 2.
   Can we add 3?
   $3(idx=5)$. $p_5=3$.
   $idx(3)=5$. $8(idx=2)$. $3 < 8$ (values). $idx(2) < idx(5)$. So $8 \succ 3$ in Inc.
   $7(idx=4)$. $
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=1328)

````
Let me understand this problem first. We need to divide a permutation into 4 disjoint subsequences $a, b, c, d$ to maximize $\operatorname{LIS}(a) + \operatorname{LDS}(b) + \operatorname{LIS}(c) + \operatorname{LDS}(d)$.

Key observations:
1. For any subsequence, LIS ≤ length of subsequence, and LDS ≤ length of subsequence.
2. We want each element to contribute as much as possible to the sum.
3. The maximum possible LIS/LDS for a single element is 1 (the element itself).
4. So ideally we'd get every element counted once, giving us score = n/n = 1.

A crucial observation from Dilworth's theorem: In any permutation of length n,
- LIS × LDS ≥ n (for the entire permutation)

Actually, more precisely: if LIS(a) = x, then there are at least n-x elements that don't belong to the LIS of a. Those elements form some decreasing pattern... but let me think differently.

For a single element to contribute:
- To LIS: it needs to be part of an increasing sequence
- To LDS: it needs to be part of a decreasing sequence

The key insight is: in a permutation, what's the minimum LIS of a subsequence containing k consecutive elements in value? Or better - can we guarantee that we can extract certain patterns?

Consider decomposing the permutation using patience sorting / stack decomposition:

Let `LDS[i]` = Longest Decreasing Subsequence ending at position i
Let `LIS[i]` = Longest Increasing Subsequence starting at position i

For any element p[i], its contribution depends on whether it goes to an increasing or decreasing group.


```
If going to LIS, contributes 1; to LDS also 1.
But two elements cannot contribute twice! Each element must go to exactly one group.
```

So the real question becomes: how do I partition n elements across these 4 groups so nothing's double-counted while getting the maximum total?

```
Let me consider which elements are "forced" into LIS versus LDS constraints.
```

Elements with very small values tend toward increasing sequences, while large elements pull toward decreasing ones. If I split based on a threshold—elements below it into LIS, above into LDS—I might capture the natural flow of the permutation. The challenge is finding where that cutoff should fall.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    vector<int> p(n);
    vector<pair<int,int>> pos(n + 1);
    for (int i = 0; i < n; i++) {
        cin >> p[i];
        pos[p[i]] = {i, i}; // (position_index, value_in_pos)
    }
    // pos[v] gives the position index where value v appears

    auto lds = [&](int v) -> int { // LIS starting at value >= v, ending before pos[v]
        int best = 0;
        for (int cand : pos) {
            if (v <= cand.second && cand.first < pos[v].first) {
                best = max(best, cand.second);
            }
        }
        return best + 1;
    };

    auto lis = [&](int v) -> int { // LDS starting at position >= pos[v][0], value > v
        int best = 0;
        for (int cand : pos) {
            if (pos[cand.second] >= pos[v][0] && cand.second > v) {
                best = max(best, pos[cand.second]);
            }
        }
        return best + 1;
    };

    vector<vector<int>> dp(n + 2, vector<int>(4, -1));
    pair<int, int> memo[500005];

    auto solve = [&](int idx, int state, auto&& self) -> int {
        if (state == 7 || state == 6) { // LIS-LIS-LDS-LDS or LIS-LIS-LDS-DOS
            int ret = 0;
            for (int i = 0; i < 4; i++) {
                auto [l, m, s, f] = pos[idx];
                if (s >= pos[s][0] && (idx % 4 == 2 ? pos[s] > f : 1))
                    ret = max(ret, m - s);
            }
            return ret;
        } else if (state == 5 || state == 6) { // LIS-LIS-LIS-LDS/DOS
            int ret = 0;
            auto [l, m, s, f] = pos[idx];
            if ((idx % 4 == 2 ? pos[f] > l : 1)) ret = max(ret, s);
            for (int i = 0; i < 4; i++) {
                if ((idx % 4 == 2 ? pos[l] > f : 1)) ret = max(ret, m - s);
            }
            return ret;
        } else { // LIS-LIS-LDS-DOS
            int ret = 0;
            auto [l, m, s, f] = pos[idx];
            if (m >= pos[m][0] && (idx % 4 == 2 ? pos[f] > m : 1)) ret = max(ret, m - s);
            for (int i = 0; i < 4; i++) {
                auto [xl, xm, xs, xf] = pos[idx];
                if ((xm >= pos[xm][0] && (xs % 4 == 2 ? pos[xf] > xm : 1))) 
                    ret = max(ret, m - s);
            }
            return ret;
        }
    };
    return 0;
}
```
````
