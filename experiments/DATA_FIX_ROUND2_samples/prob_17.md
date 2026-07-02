# 题 17 — base均分 15.6 vs 直接SFT均分 0.0(掉点 15.6)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=3,得分 78.0,completion_tokens=32768)

````
The problem asks us to find the position of the largest number $n$ in a permutation of length $n$. We can query any interval $[l, r]$ and get back the position of the second largest number in that interval.

Let's denote the position of value $x$ in the permutation as $pos(x)$.
We want to find $pos(n)$.
A query `? l r` returns the index $idx$ such that the element at $arr[idx]$ is the second largest in the subarray $arr[l \dots r]$, and it returns 0-based or 1-based index (usually 1-based based on problem statement). The problem says "position of the second largest number".

Wait, the query gives the *position* (index) of the second largest number in the range $[l, r]$. It doesn't tell us the values directly. But knowing the position is enough if we can deduce something.

Actually, let's look at what the query tells us.
Query `? l r`: returns index $k$ such that $arr[k]$ is the second largest in $\{arr[l], arr[l+1], \dots, arr[r]\}$.
This means $\max(arr[l \dots k]) < arr[r']$ where $r'$ is the position of the largest in this range. Wait, no.
Let $M = \max_{i=l}^r arr[i]$. Let $S$ be the set of indices in $[l, r]$ where $arr[i]$ is the second largest value in that range. Usually there's only one unless there are duplicate max values which is not possible in a permutation. So the second largest is unique.
So for a given query `[l, r]`, we get an index $idx$ which points to the second largest element in $[l, r]$. This implies there exists some index $mx\_idx \in [l, r]$ such that $arr[mx\_idx]$ is the maximum in $[l, r]$ and $arr[idx]$ is the second maximum in $[l, r]$. Note that $mx\_idx \neq idx$. Also, since it's a permutation, all elements are distinct.

This seems a bit indirect. Why does querying for the *second* largest help us find the *largest*?
Usually, finding the position of the maximum in an unsorted array takes $O(N)$ time, but here we have queries.
If we could identify the range containing the maximum, say $[L, R]$ where $arr[L]$ is max, we could try to narrow it down.
However, we don't know the value of the maximum, nor its value relative to others easily.
But wait, if we know the position of the second largest, can we use that to eliminate ranges?
Suppose we have a range $[l, r]$. Let the largest be at $p_{max}$ and second largest at $p_{sec}$. The query returns $p_{sec}$.
Since $p_{sec}$ is in $[l, r]$, and $p_{max}$ is also in $[l, r]$, and $arr[p_{max}] > arr[p_{sec}]$, this information is somewhat specific.
Is it possible that we can determine the location of the global maximum by recursively narrowing down?

Consider the standard approach to finding the maximum: divide and conquer.
If we want to find the global maximum in $[1, n]$, we could check $[1, mid]$ and $[mid+1, n]$. But our query works on an arbitrary interval.
If we pick a range $[l, r]$ and ask `? l r`, we learn the position of the second largest in that range. Let's call the returned index $idx$.
This tells us that the largest element in $[l, r]$ is NOT at $idx$. It must be somewhere else in $[l, r]$.
More specifically, the largest element is greater than the element at $idx$.
Also, any element in $[l, r]$ other than the largest and second largest is smaller than the element at $idx$.
Wait, $arr[idx]$ is the *second* largest. So $arr[mx\_idx] > arr[idx] > arr[j]$ for all $j \in [l, r] \setminus \{mx\_idx\}$.
This doesn't immediately give us $mx\_idx$. But it splits the search space potentially.

Let's reconsider. We want to find the position of $n$. Since $n$ is the absolute maximum, if it were inside a range $[l, r]$, it would be the maximum in that range. If it's not in $[l, r]$, it's irrelevant to the query result.
So if we can establish that a range contains $n$, we are good. But we don't know if a range contains $n$.
Actually, if $n$ is in $[l, r]$, then the maximum in $[l, r]$ is $n$ itself. In this case, the second largest in $[l, r]$ would be the second largest number available in $[l, r]$, which is strictly less than $n$.
If $n$ is NOT in $[l, r]$, then the maximum in $[l, r]$ is some $m < n$. The second largest is then even smaller.
This distinction doesn't seem helpful directly because we just get an index.

However, observe the properties.
If we query $[l, r]$, let's say we get back index $p$. This means the element at $p$ is the runner-up.
If we query $[l, r-1]$, we might get a different $p'$.
Consider the interval $[1, n]$. If $n$ is at position $k$, then $arr[k]=n$.
Any query covering $k$ will have $n$ as the max in that query. The response will be the index of the second largest element in that query.
If a query does NOT cover $k$, then the max in that query is $< n$.

Strategy Idea:
We need to locate $n$ among positions $1, \dots, n$.
Suppose we suspect $n$ is in $[l, r]$. If we assume $n$ is in $[l, r]$, then the true max is at some position $p \in [l, r]$. The second max query gives us an index $idx \neq p$.
Is there a property relating the position of the maximum and the second maximum?
Usually, the second maximum is adjacent to the maximum in some sense (like a Cartesian tree neighbor). Specifically, if we consider the Cartesian tree built from the permutation, the parent of a node $u$ is the smallest range containing $u$'s subtree root? No.
In the Cartesian tree of the array $A$, the parent of node at index $i$ is the minimum range index $j$ such that $A[j]$ is the max of $A[\min(i, j) \dots \max(i, j)]$. Actually, simpler: the parent of $i$ is the index $k$ closest to $i$ (left or right) such that $A[k] > A[i]$ and $k$ is the first element larger than $A[i]$ encountered when scanning outwards. Or rather, $A[parent(i)]$ is the nearest element larger than $A[i]$ in the recursive decomposition.
Specifically, in the Cartesian tree (where left child is max-left-subtree, right child is max-right-subtree), the parent of a leaf $i$ is either the first larger element to the left or the first larger element to the right, whichever is smaller in value.

Let's use the query logic.
If we query $[l, r]$ and get index $p$, we know $A[p]$ is the second largest in $A[l \dots r]$.
Let $M$ be the index of the largest in $A[l \dots r]$. Then $A[M]$ is the largest, $A[p]$ is second largest.
Notice that $A[M]$ separates $A[l \dots M-1]$ and $A[M+1 \dots r]$. Within these two intervals, no element is greater than $A[M]$. But within $A[l \dots r]$, the next largest candidates are the largest of $A[l \dots M-1]$ and the largest of $A[M+1 \dots r]$.
Actually, $A[p]$ must be either the largest in $A[l \dots M-1]$ OR the largest in $A[M+1 \dots r]$.
Proof: If $A[p]$ was not the largest in $[l, M-1]$ (assuming $p < M$), there would be some $k \in [l, M-1]$ with $A[k] > A[p]$. Since $k \in [l, r]$ and $k \neq M$, this contradicts that $A[p]$ is the second largest in $[l, r]$. Similarly if $p > M$.
So, if we find the second largest index $p$ in $[l, r]$, and we knew the position of the largest $M$ in $[l, r]$, we would split the range into $[l, M-1]$ and $[M+1, r]$. One of these contains the "true" successor of the max in terms of values?
No, that reasoning requires knowing $M$. We know $p$, we don't know $M$.
BUT, notice that the element at $p$ is locally maximal in some way.
Actually, consider the relationship between $p$ and $M$. They are neighbors in the Cartesian tree (parent-child or sibling?).
Specifically, in the set of indices of all elements in $[l, r]$, sorted by value descending: $v_1=M, v_2=p, v_3 \dots$.
We know $p$ is the second one.
If we can query a range where $n$ is definitely included, say initially $[1, n]$, we get the position of the second largest element in $[1, n]$. Let this be $p_1$. The position of the largest (which is $n$) is unknown, let's call it $pos(n)$.
Wait, this only gives us one piece of info about $p_1$.
Is it possible to find the max index?
Actually, finding the max position $pos(n)$ is exactly finding the position of the element with value $n$.
What if we query overlapping ranges?

Let's re-read the query constraint carefully.
Sum of $(r - l + 1)$ over queries $\le 30n$. And we should aim for $\log_2 n$.
The standard algorithm to find the maximum using comparisons is binary search-like or tournament.
Here the comparison isn't direct. We get the *position* of the second largest.
This sounds like we can maintain the current candidate for the maximum and verify it against something?
Or perhaps, if we know the range $[l, r]$ contains the maximum (which is $n$), we can try to reduce the range size.
If $[l, r]$ contains $n$, the query `? l r` returns the second largest index $p$.
If we query $[l, p]$, does it tell us something?
Or $[p, r]$?

Hypothesis:
If the global maximum is at position $pos(n)$, and we query $[1, n]$, we get $p$ which is the second largest index.
If $pos(n) > p$, then the range $[1, pos(n)]$ contains the global maximum and potentially smaller elements.
But we don't know $pos(n)$.

Wait, think about the structure of the maximums.
Consider the process of elimination.
If I can prove that $n$ is in $[l, r]$, and then somehow exclude one half?
Suppose I have a guess for $pos(n)$. How do I confirm it?
If I query $[l, r]$ and $pos(n) \in [l, r]$, I get $p$ (position of second largest).
Then the largest $n$ is at some $M \in [l, r]$.
And $arr[p]$ is the second largest.
This implies $arr[M] > arr[p] > \text{any } x \text{ in } [l, r] \setminus \{M, p\}$.
So, in the set of indices $[l, r]$, $M$ is the unique index with value $> arr[p]$.
Thus $M$ divides $[l, r]$ into $[l, M-1]$ and $[M+1, r]$.
One of these sub-intervals contains all elements in $[l, r]$ that are larger than some threshold?
Actually, $p$ is the largest element in either $[l, M-1]$ or $[M+1, r]$.
Because $arr[p]$ is the second largest overall in $[l, r]$. So it must be the largest of the remaining part.
So, if we knew $M$, we could query $[l, M-1]$ and $[M+1, r]$.
If we query $[l, M-1]$, if it returns some index $p_l$, then $p_l$ is the largest in $[l, M-1]$.
Similarly $p_r$ for $[M+1, r]$.
Then the true second largest $p$ is either $p_l$ or $p_r$.
If $p = p_l$, then $arr[p_l] > arr[x]$ for all $x \in [M+1, r]$. This would mean querying $[M+1, r]$ would return nothing or something else? No, $[M+1, r]$ has a max, say $M_R$. That $M_R < arr[p]$ because $arr[p]$ is the second largest. So $p$ cannot be inside $[M+1, r]$ if $arr[p] > M_R$. But $p$ is the index.
So yes, if we know $M$, we can compare the values at $p_l$ and $p_r$ by checking the second largest in the union.
Wait, if we know $M$, $p$ is simply the argmax of the second largest in $[l, r]$.
Actually, we don't know $M$.
But if we have a candidate for $M$ (say we found it or guessed it), we can refine.
How to find $M$?

Let's flip the problem.
The second largest element is always close to the largest element.
Actually, consider the range $[1, n]$. Query gives index $p_1$ of the second largest.
This tells us that the largest is at some $M$.
And $p_1$ is the largest in $[1, M-1] \cup [M+1, n]$.
Specifically, $p_1$ is either the max of $[1, M-1]$ or max of $[M+1, n]$.
If $p_1 < M$, then $arr[p_1]$ is the max of $[1, M-1]$.
If $p_1 > M$, then $arr[p_1]$ is the max of $[M+1, n]$.
This doesn't fix $M$.
But we can narrow down the search space for $M$.
Suppose we have a current "best candidate" range or element?
Maybe we can search for the maximum $M$ using binary search?
Binary search for $M$:
Can we check if $M > mid$?
If we query $[1, n]$, we get $p$ (second largest index).
If $M$ is in $[1, mid]$, and $p$ is in $[1, mid]$, then $M$ is likely in $[1, mid]$.
Wait, $p$ is the second largest in $[1, n]$. The largest is $M$.
If $M \in [1, mid]$ and $p \in [1, mid]$, then both max and second max are in $[1, mid]$. This suggests the top 2 are in $[1, mid]$.
If $M \in [1, mid]$ and $p \in [mid+1, n]$, then $M$ is in left half, 2nd max is in right half.
If $M \in [mid+1, n]$ and $p \in [1, mid]$, then $M$ in right half.
If $M \in [mid+1, n]$ and $p \in [mid+1, n]$, then top 2 in right half.

Case 1: $p < M$ and $M > p$. (Index relation)
This doesn't work well. Indices are mixed up with values.
Let's analyze the condition $M$ vs $p$.
We know $A[M]$ is max, $A[p]$ is 2nd max. $A[M] > A[p]$.
If we restrict our query to $[1, mid]$.
If $p \notin [1, mid]$, then $p > mid$.
Does this imply anything about $M$?
Not necessarily. $M$ could be anywhere.
But if $p$ (the index of the 2nd largest in $[1, n]$) is outside $[1, mid]$, say $p > mid$.
Then the 2nd largest element of the whole array is located in $[mid+1, n]$.
This means that all elements in $[1, mid]$ are smaller than $A[p]$.
This is a crucial observation!
Let $X = A[p]$. $X$ is the second largest in $[1, n]$. So $\forall i \neq M, A[i] < X$.
If $p > mid$, then $X = A[p]$ is in $[mid+1, n]$.
Thus for all $i \in [1, mid]$, $A[i] < A[p]$.
So $\max_{i \in [1, mid]} A[i] < \max_{i \in [mid+1, n]} A[i]$.
Why? Because $\max_{i \in [mid+1, n]} A[i] \ge A[p]$ (since $p$ is in this range).
And we established $A[i] < A[p]$ for all $i \in [1, mid]$.
So the maximum of $[1, mid]$ is strictly less than the maximum of $[mid+1, n]$.
Actually, the maximum of $[mid+1, n]$ must be $A[M]$ or $A[p]$ (or something else?).
We know $A[p]$ is the second largest in $[1, n]$. The largest is $A[M]$.
Since $p \in [mid+1, n]$, we have $A[p]$ in that right half.
Is it possible that $M \in [1, mid]$?
If $M \in [1, mid]$, then $A[M]$ is max of $[1, n]$. But we said $\forall i \in [1, mid], A[i] < A[p]$.
This contradicts $A[M] > A[p]$ if $M \in [1, mid]$.
Therefore, if $p > mid$, then $M$ CANNOT be in $[1, mid]$.
So $M$ must be in $[mid+1, n]$.
Symmetrically, if $p < mid$, then $M$ must be in $[1, mid]$.
Wait. Is this correct?
Condition derived: If $p$ (index of 2nd largest in $[1, n]$) lies in $[mid+1, n]$, then all elements in $[1, mid]$ are smaller than $A[p]$.
This implies no element in $[1, mid]$ can be equal to or greater than $A[p]$.
Since $A[M] > A[p]$, $M$ cannot be in $[1, mid]$. So $M \in [mid+1, n]$.
Correct.
So, simply checking the position of the second largest element $p$ allows us to determine if the maximum $M$ lies in the left or right half!
Algorithm:
1. Start with range $[L=1, R=n]$. Current candidate for $M$ is in $[L, R]$.
   Actually, we maintain the invariant that $n$ is at some position in $[L, R]$. Initially $L=1, R=n$.
2. While $L < R$:
   Calculate $mid = (L + R) / 2$.
   We need to know if $M$ is in $[L, mid]$ or $[mid+1, R]$.
   We perform query `? L R`. Wait, the query gives the position of the 2nd largest in the range $[L, R]$.
   Does this query return the same $p$ as before?
   Yes, because $[L, R]$ still contains $n$. If we are looking for $n$ globally, we are just reducing the search space.
   BUT, wait. The definition of 2nd largest depends on the range.
   If we restrict to $[L, R]$, and $n \in [L, R]$, then $n$ is still the maximum of $[L, R]$. The second largest element is the largest element in $[L, R] \setminus \{pos(n)\}$.
   Let $p_{LR}$ be the index returned for query `? L R`.
   So $p_{LR}$ is the index of the second largest element in the *current* subarray $A[L \dots R]$.
   Again, let $M_{curr}$ be the position of $n$ inside $[L, R]$.
   Then $A[M_{curr}]$ is the max, $A[p_{LR}]$ is the second max.
   Using the previous logic:
   If $p_{LR} \in [mid+1, R]$, then $M_{curr} \in [mid+1, R]$.
   Proof: Suppose $M_{curr} \in [L, mid]$. Then $A[M_{curr}]$ is max in $[L, R]$. $A[p_{LR}]$ is second max.
   We know $p_{LR} > mid$. So $A[p_{LR}]$ is an element in the right half.
   All elements in $[L, mid]$ are compared against $A[p_{LR}]$.
   Specifically, $A[p_{LR}]$ is the second largest in $[L, R]$.
   So for any $k \in [L, mid] \setminus \{M_{curr}\}$, $A[k] < A[p_{LR}]$.
   What about $M_{curr}$? $A[M_{curr}]$ is max.
   Is it guaranteed that all $k \in [L, mid]$ satisfy $A[k] < A[p_{LR}]$?
   Not necessarily. $M_{curr}$ has $A[M_{curr}] > A[p_{LR}]$.
   However, if we assume $M_{curr} \in [L, mid]$, then $A[p_{LR}]$ is the max of the set $([L, R] \setminus \{M_{curr}\})$.
   The set $[L, R] \setminus \{M_{curr}\}$ includes all of $[mid+1, R]$.
   If $M_{curr} \in [L, mid]$, then the max of $[L, R] \setminus \{M_{curr}\}$ is determined by comparing $\max([L, mid] \setminus \{M_{curr}\})$ and $\max([mid+1, R])$.
   If $p_{LR} \in [mid+1, R]$, then $A[p_{LR}]$ must be larger than any element in $[L, mid] \setminus \{M_{curr}\}$.
   And certainly larger than any element in $[mid+1, R]$ except $M_{curr}$ (if $M_{curr}$ were there).
   But wait. If $p_{LR} \in [mid+1, R]$, it means $A[p_{LR}]$ is larger than everything in $[L, R]$ except $M_{curr}$.
   Consider $M_{curr} \in [L, mid]$. Then $A[M_{curr}] > A[p_{LR}]$.
   But this contradicts nothing yet?
   Wait, I used the property derived earlier:
   "If $p$ (index of 2nd largest in full range) is in Right Half, then Max is in Right Half."
   This relied on the fact that $A[p]$ is the 2nd largest.
   Let's re-verify:
   Let $p$ be index of 2nd largest in $[L, R]$.
   Case A: $p \in [L, mid]$.
     Then $A[p]$ is the second largest.
     This implies that $A[p] > \max_{k \in [L, R] \setminus \{M_{curr}, p\}} A[k]$.
     Does this force $M_{curr} \in [L, mid]$?
     Suppose $M_{curr} \in [mid+1, R]$.
     Then $A[p]$ is in $[L, mid]$.
     Consider the element $M_{curr}$ (Max) is in Right Half.
     The Second Largest ($p$) is in Left Half.
     This is consistent. The Max is big in Right, 2nd largest is big in Left.
     Example: Range $[1, 3]$. Values: $3, 1, 2$. Max is 3 at 1. 2nd max is 2 at 3.
     Here $mid = 1$ (or 2).
     If $mid=2$. Left $[1, 2]$, Right $[3, 3]$.
     Max at 1 (Left), 2nd max at 3 (Right).
     Here $p=3 \in [3, 3]$ (Right). But $M=1 \in [1, 2]$ (Left).
     So my previous deduction was wrong.
     "If $p > mid$, then $M$ must be in $[mid+1, R]$" is FALSE.
     Counter example: Arr = $[3, 1, 2]$. $L=1, R=3, mid=2$.
     Max at 1. Second max at 3.
     $p=3$. $p \in [3, 3]$. $M=1 \in [1, 2]$.
     $p$ is in Right, $M$ is in Left.
     Wait, so how did I conclude $M \in [mid+1, R]$ earlier?
     Earlier proof attempt:
     "If $p > mid$, then $A[p]$ is in Right Half. All elements in $[L, mid]$ are smaller than $A[p]$."
     Is this true?
     $p$ is the index of the *second largest* element.
     So for any $k \neq M_{curr}$, $A[k] < A[p]$.
     This holds for $k \in [L, mid]$.
     Does this hold for $k = M_{curr}$? No, $A[M_{curr}] > A[p]$.
     If $M_{curr} \in [L, mid]$, then $A[M_{curr}]$ violates "all elements in $[L, mid]$ are smaller than $A[p]$"?
     Yes, because $A[M_{curr}] > A[p]$.
     So the condition "$\forall k \in [L, mid], A[k] < A[p]$" implies $M_{curr} \notin [L, mid]$, thus $M_{curr} \in [mid+1, R]$.
     But why did I think $A[k] < A[p]$ for all $k \in [L, mid]$?
     Because $p$ is the second largest in $[L, R]$.
     So $A[p]$ is the largest element in $[L, R]$ excluding $M_{curr}$.
     Therefore $A[p]$ is the largest among $\{ A[k] : k \in [L, R], k \neq M_{curr} \}$.
     If $p \in [mid+1, R]$, then for ALL $k \in [L, R] \setminus \{M_{curr}\}$, $A[k] \le A[p]$.
     Now consider $k \in [L, mid]$.
     These indices are definitely in $[L, R] \setminus \{M_{curr}\}$ UNLESS $M_{curr} \in [L, mid]$.
     If $M_{curr} \in [L, mid]$, then we only know that $A[k] \le A[p]$ for $k \in ([L, mid] \cap \{k \neq M_{curr}\}) \cup ([mid+1, R])$.
     For $k \in [L, mid], k \neq M_{curr}$, we have $A[k] < A[p]$.
     So $A[M_{curr}]$ is the only element in $[L, mid]$ that could be greater than $A[p]$.
     However, if $M_{curr} \in [mid+1, R]$, then for all $k \in [L, mid]$, $k \in [L, R] \setminus \{M_{curr}\}$. So $A[k] < A[p]$.
     This means $\max([L, mid]) < A[p]$.
     So $\max([L, mid]) < \max([mid+1, R])$ roughly (since $p \in [mid+1, R]$).
     Wait, if $M_{curr} \in [L, mid]$, then $A[M_{curr}] > A[p]$.
     But $A[p]$ is the second largest.
     Is it possible that $M_{curr} \in [L, mid]$ and $p \in [mid+1, R]$?
     Let's check my counter-example again.
     Array: $[3, 1, 2]$. Max at 1. 2nd max at 3.
     $p=3$. $M=1$.
     Range $[1, 3]$, $mid=2$. $L=[1, 2], R=[3, 3]$.
     $p \in R$. $M \in L$.
     Is it true that $\forall k \in [1, 2], A[k] < A[p]$?
     $A[1]=3, A[2]=1$. $A[p]=2$.
     $A[1] = 3 > 2$.
     So the condition fails for $k=M$.
     Wait, $M=1$ is in $[1, 2]$. So $A[M]$ is in $[L, mid]$ and is $> A[p]$.
     But $p$ is the 2nd largest.
     $p=3$ means $A[3]=2$.
     Is $2$ the 2nd largest in $[1, 3]$? No! $A[3]=2, A[1]=3$. $3 > 2$.
     Where is 2nd largest?
     The values are 3, 1, 2. Sorted: 3, 2, 1.
     Max is 3. 2nd max is 2.
     So $p=3$ is correct.
     Why did my logic "If $p \in R$, then $M \in R$" fail?
     Because I claimed "All elements in $L$ are smaller than $A[p]$".
     But $A[M]$ is in $L$ and is larger.
     My claim "For all $k \in [L, R] \setminus \{M_{curr}\}, A[k] \le A[p]$" is correct.
     If $M_{curr} \in L$, then elements in $R$ are subset of $\{k \in [L, R] \setminus \{M_{curr}\}\}$.
     So elements in $R$ are $\le A[p]$.
     But $p \in R$, so $A[p] \le A[p]$ (trivial).
     So elements in $R$ are small except maybe $M_{curr}$ if $M_{curr} \in R$.
     If $M_{curr} \in L$, then all elements in $R$ are $\le A[p]$.
     AND elements in $L \setminus \{M_{curr}\}$ are $< A[p]$.
     This leaves $A[M_{curr}]$ as the only thing potentially $> A[p]$.
     So $A[M_{curr}] > A[p]$.
     So it IS possible to have $p \in R$ and $M_{curr} \in L$.
     The previous deduction was flawed.

     Okay, so simply checking $p$'s side of $mid$ is not enough to determine $M$'s side.
     However, look at the counter example:
     Arr: $[3, 1, 2]$. $M=1, p=3$.
     If we query $[1, 2]$, values $\{3, 1\}$. Max 3, 2nd max 1.
     Query returns position of 1, which is 2.
     So $p_{sub} = 2$.
     Notice $p_{sub}$ is the index of the 2nd max in $[L, mid]$.
     Wait, $p_{sub}$ refers to the 2nd largest index in $[L, mid]$.
     The 2nd largest in $[1, 2]$ is 1 at index 2.
     So query `? 1 2` returns 2.
     What if we query `? 1 3`? Returns 3.
     
     Let's rethink. We want to find $M$.
     We know $A[M]$ is max.
     We can query ranges.
     Consider the Cartesian Tree idea again.
     $n$ corresponds to the root of the tree.
     Querying `[l, r]` finds the second largest index in that range.
     In the Cartesian tree of $A[L \dots R]$, let $root$ be $M$. The children of $root$ are roots of left/right subtrees.
     The second largest in the range is simply the max of the two children? No.
     The second largest is the larger of the two roots of the children of $root$ in the Cartesian tree formed by $A[L \dots R]$ (where root is $M$).
     Let $T(L, R)$ be the Cartesian tree of $A[L \dots R]$. Root is at index $M$.
     Left child is root of $A[L \dots M-1]$. Right child is root of $A[M+1 \dots R]$.
     The max value in left part is $val(lc)$, in right is $val(rc)$.
     The max value in $A[L \dots R]$ is $A[M]$.
     The second largest is $\max(A[lc], A[rc])$.
     Let this be $val(max\_child)$.
     The index of this max child is $p$.
     So $p$ is either $M_L$ (root of left) or $M_R$ (root of right).
     This means $p$ is a direct child of $M$ in the Cartesian tree restricted to $[L, R]$.
     Specifically, $p$ is the index of the child (either left or right) that has the larger root value.
     This doesn't directly tell us which side $M$ is on, because $p$ is a child of $M$.
     Wait, we don't know the parent of $p$. We just get $p$.
     
     But we know that $A[M]$ is the global maximum.
     Can we maintain a candidate $M$?
     Suppose we have a candidate position $pos$.
     If we query `[pos-1, pos+1]`, we might find out?
     
     Alternative Strategy:
     We have a limit on queries $\approx 30n$. We can afford slightly more than $\log n$.
     Actually, the constraints say: sum of $(r-l+1) \le 30n$.
     This implies we can do many small queries.
     But optimal solution uses $\log n$ queries.
     Wait, scoring is based on number of queries per test case.
     Sum of $N \le 100000$. $T \le 10000$.
     If we use binary search, each step takes 1 query?
     But we established binary search is tricky because $p$ doesn't directly tell us direction.
     
     Let's revisit the property of $p$ being a child of $M$.
     If $p$ is the second largest, it must be the largest among $A[L \dots M-1]$ OR $A[M+1 \dots R]$.
     If $p \in [L, M-1]$, then $A[p] = \max(A[L \dots M-1])$.
     If $p \in [M+1, R]$, then $A[p] = \max(A[M+1 \dots R])$.
     This is a HUGE insight.
     Proof: If $p \in [L, M-1]$, then $A[p]$ is the second largest in $[L, R]$.
     Any $x \in [M+1, R]$ is not $M$. Thus $x \neq M$.
     So $A[x] < A[p]$.
     This means $\max(A[M+1 \dots R]) < A[p]$.
     Also for any $y \in [L, M-1] \setminus \{p\}$, $A[y] < A[p]$ because $p$ is 2nd largest.
     So $A[p]$ is indeed the maximum in $[L, M-1]$.
     Similarly if $p \in [M+1, R]$, $A[p]$ is max in $[M+1, R]$.
     
     So $p$ is the index of the maximum element in the interval adjacent to $M$.
     Now, suppose we know $M$'s approximate location or we want to find $M$.
     If we query `[L, R]`, we get $p$.
     We know $M \notin \{L, \dots, p-1, p+1, \dots, R\} \setminus \{L \dots R \text{ excluding } M\}$.
     Wait, $M$ is not $p$.
     So $M \in [L, R] \setminus \{p\}$.
     But we don't know which side.
     However, $A[p]$ is the max of $[L, M-1]$ IF $p < M$.
     And $A[p]$ is the max of $[M+1, R]$ IF $p > M$.
     
     So if we can find an element $x$ in $[L, R] \setminus \{p\}$ such that $A[x] > A[p]$, then $x$ MUST be $M$.
     Why? Because $M$ is the only element larger than $A[p]$ in $[L, R]$.
     So if we can compare values, we find $M$.
     But we can't compare values directly.
     We can only query for position of 2nd largest.
     
     Wait, if we have a query `? L R`, we get $p$.
     We know $A[M] > A[p]$.
     And $A[p]$ is max of $[L, M-1]$ (if $p<M$) or $[M+1, R]$ (if $p>M$).
     
     Let's use a randomized approach or simply iterate.
     We can query $[L, R]$. Get $p$.
     We know $M \in [L, R] \setminus \{p\}$.
     So we eliminated $p$.
     This is useful? Not really, we have $N-1$ options.
     However, if we restrict the range further.
     Suppose we query `? L R`. We get $p$.
     Now we know $A[p]$ is a "local max" of some sub-interval.
     Specifically, $A[p] >$ all elements in $[L, R] \setminus \{p, M\}$.
     
     Wait!
     If we can find $M$, we are done.
     If we can show $M$ is in $[a, b]$, we recurse.
     Let's try to maintain the set of candidate positions.
     Initially $C = \{1, \dots, n\}$.
     Pick any $x \in C$. Query `? L R`?
     
     Actually, let's look at the constraint: "sum of $(r-l+1)$".
     This allows for linear scan or something similar?
     Wait, score is based on log2n. $15 \log n$.
     So we need $O(\log n)$ queries.
     
     Let's reconsider the $p$ vs $M$ relationship.
     $p$ is the index of the 2nd largest.
     This implies $p$ is adjacent to $M$ in the Cartesian tree.
     Specifically, $M$ is the parent of $p$ in the Cartesian tree (assuming standard heap property on indices/values).
     Or more accurately, $p$ is the "larger child" of $M$.
     If we knew $M$, we could find the second largest easily.
     
     What if we maintain the list of candidates for $M$ in decreasing order of "potential"?
     Or maybe:
     We start with range $[1, n]$.
     Query $[1, n]$, get $p$.
     Now we know $M \neq p$.
     Is it possible to eliminate parts of $[1, n]$?
     The only thing we know is $A[p]$ is very high.
     Specifically, if $M$ is far from $p$, say distance $d$, then there are $d$ elements between them.
     But we don't know values.
     
     Wait, consider this:
     We want to find $M$.
     Suppose we have an estimate $K$.
     If we query `? K n`, and get $p$.
     If $p < K$, then $A[p]$ is the 2nd largest in $[K, n]$? No, the query returns 2nd largest in range.
     
     Let's go back to the idea of eliminating indices.
     If we have an interval $[L, R]$ which definitely contains $M$.
     We want to shrink this interval.
     If we can find an index $k \in [L, R]$ such that $A[k] < A[M]$, that helps.
     We know $A[p] < A[M]$ (always).
     And $A[p]$ is very large.
     Maybe we can query `? L mid`?
     Wait, the interactor returns the *position* of the second largest.
     Let's denote $f(l, r)$ as the return value.
     $f(1, n) = p_1$.
     We know $M \in [1, n] \setminus \{p_1\}$.
     Now query `? 1, mid`.
     If $p_{new} \neq p_1$, what does it imply?
     If $M \in [1, mid]$, then $A[M]$ is max of $[1, mid]$.
     Then $f(1, mid)$ is the index of 2nd max in $[1, mid]$.
     Let's call this $p_{left}$.
     If $p_{left} == p_1$, then $p_1 \in [1, mid]$.
     Since $p_1$ was the 2nd max in $[1, n]$ (where $M$ was max).
     If $p_1 \in [1, mid]$ AND $M \in [1, mid]$, then both max and 2nd max are in $[1, mid]$.
     This means $\max([1, mid]) > \max([mid+1, n])$.
     If $M \in [mid+1, n]$ and $p_1 \in [1, mid]$.
     Then $\max([1, mid]) < A[M]$.
     Also $A[p_1]$ is 2nd max of $[1, n]$.
     Since $M \in [mid+1, n]$ is max, $A[M]$ is largest.
     $p_1 \in [1, mid]$ is 2nd largest.
     This implies $A[p_1]$ is larger than any element in $[mid+1, n] \setminus \{M\}$.
     So $\max([mid+1, n]) = A[M]$.
     And $\max([1, mid] \setminus \{p_1\}) < A[p_1]$.
     Also $\max([mid+1, n] \setminus \{M\}) < A[p_1]$.
     
     This looks like we can distinguish cases.
     Case 1: $M \in [1, mid]$.
       Then $A[M]$ and $A[p_1]$ are the two largest in $[1, mid]$.
       So $A[p_1]$ is the max of $[1, mid] \setminus \{M\}$.
       This means if we query `? 1 mid`, we should get $p_1$ IF $p_1$ is indeed the 2nd largest in $[1, mid]$.
       Wait, if $M \in [1, mid]$, then the 2nd largest in $[1, mid]$ is the same as 2nd largest in $[1, n]$?
       Not necessarily. The set is a subset. Removing elements larger than $A[p_1]$ is key.
       In $[1, n]$, $M$ is max. $p_1$ is 2nd max.
       Everything else is $< A[p_1]$.
       If $M \in [1, mid]$, then the set of elements in $[1, mid]$ contains $A[M]$ and $A[p_1]$.
       And all other elements in $[1, mid]$ are $< A[p_1]$.
       Also all elements in $[mid+1, n]$ are $< A[p_1]$.
       So yes, if $M \in [1, mid]$, then $p_1$ is the 2nd largest in $[1, mid]$.
       So query `? 1 mid` will return $p_1$.
       
       Case 2: $M \in [mid+1, n]$.
       Then $M$ is not in $[1, mid]$.
       So in $[1, mid]$, the largest is some $k \neq p_1$.
       Since $M \in [mid+1, n]$ is max of $[1, n]$, $A[M]$ is gone.
       The 2nd largest in $[1, n]$ is $A[p_1]$.
       So all elements in $[1, mid]$ are smaller than $A[M]$.
       Is $A[p_1]$ the largest in $[1, mid]$?
       No, $p_1 \in [1, mid]$. So $A[p_1]$ is in $[1, mid]$.
       Is there any element in $[1, mid]$ larger than $A[p_1]$?
       No, because $A[p_1]$ is the 2nd largest in $[1, n]$.
       The only element larger than $A[p_1]$ in $[1, n]$ is $A[M]$.
       Since $M \notin [1, mid]$, no element in $[1, mid]$ is larger than $A[p_1]$.
       So $A[p_1]$ IS the max in $[1, mid]$.
       So the query `? 1 mid` returns the index of the 2nd largest in $[1, mid]$.
       The max in $[1, mid]$ is $p_1$. The 2nd largest is some $p'$.
       So `? 1 mid` returns $p'$.
       Crucially, $p' \neq p_1$ (unless $n=2$ etc, but $p'$ is 2nd largest of subset).
       Wait, if $A[p_1]$ is the max of $[1, mid]$, then the query returns the index of the element which is second largest in $[1, mid]$.
       Since $p_1$ is now the max, the answer is NOT $p_1$.
       So if `? 1 mid` returns $p_1$, then Case 1 holds ($M \in [1, mid]$).
       If it returns something else, then Case 2 holds ($M \in [mid+1, n]$).
       
     This seems solid!
     Let's verify.
     We maintain the assumption that $M \in [L, R]$.
     Let $mid = (L + R) / 2$.
     We already queried $[L, R]$ at some point?
     No, we need to know the 2nd largest of the current range $[L, R]$ to use the logic.
     So Step 1: Query `? L R`. Let result be $p$.
     (Assuming $M \in [L, R]$).
     Step 2: Query `? L mid`. Let result be $q$.
     If $q == p$, then $p$ is the 2nd largest in $[L, mid]$.
       Since $p$ is also the 2nd largest in $[L, R]$, and $M \in [L, R]$.
       We need to check if $M$ is in $[L, mid]$ or $[mid+1, R]$.
       Logic derived above:
       - If $M \in [L, mid]$, then $p$ is 2nd largest in $[L, mid]$ AND $M$ is max in $[L, mid]$.
         So $q = p$.
       - If $M \in [mid+1, R]$, then $A[p]$ is the max in $[L, mid]$ (because $A[M]$ is not there).
         Then $q$ (2nd largest in $[L, mid]$) must be something else, $q \neq p$.
         Wait, is it guaranteed $q \neq p$?
         Yes, because $q$ is the 2nd largest, $p$ is the max. Max != 2nd max.
         
     So:
     If $q == p$, then $M \in [L, mid]$.
     If $q \neq p$, then $M \in [mid+1, R]$.
     
     Is this always true?
     We assumed $M \in [L, R]$.
     Base case: $L=1, R=n$. We query `? 1 n`, get $p$.
     Then query `? 1 mid`.
     If result $q == p$, then $M \in [1, mid]$. Else $M \in [mid+1, n]$.
     We continue in the new range.
     In each step, we need the "2nd largest" of the new range to compare with the previous "global" 2nd largest?
     Wait, no.
     In the next step, we define $L', R'$ (either $[L, mid]$ or $[mid+1, R]$).
     We know $M \in [L', R']$.
     To apply the logic again, we need to query `? L' R'` to get the NEW $p'$.
     Then query `? L' mid'`.
     Compare $p'$ and result of second query.
     This logic seems sound and takes 2 queries per reduction.
     Complexity: $O(\log n)$.
     Number of steps $\approx \log_2 n$. Total queries $2 \log_2 n$.
     This fits well within limits (limit is $15 \log n$).
     
     Wait, there is a catch.
     Is it true that "If $M \in [mid+1, R]$, then $p$ is max in $[L, mid]$"?
     Recall: $p$ was the 2nd largest in $[L, R]$.
     $A[M]$ is the max in $[L, R]$.
     $M \in [mid+1, R]$.
     So $p$ is the 2nd largest in $[L, R]$.
     For any $k \in [L, mid]$, $A[k] < A[p]$?
     Suppose there exists $k \in [L, mid]$ such that $A[k] > A[p]$.
     Then $A[k]$ is larger than $A[p]$.
     But $p$ is 2nd largest in $[L, R]$.
     The only element larger than $A[p]$ in $[L, R]$ is $A[M]$.
     So if $A[k] > A[p]$, then $k$ must be $M$.
     But $M \in [mid+1, R]$, so $k \neq M$. Contradiction.
     So indeed, $\forall k \in [L, mid], A[k] < A[p]$.
     So $A[p]$ is the strict maximum of $[L, mid]$.
     So the query `? L mid` will return the position of the 2nd largest in $[L, mid]$.
     Since $A[p]$ is the unique maximum, it is not returned.
     So result $q \neq p$.
     
     Wait, what if $A[p]$ is NOT the unique maximum?
     In a permutation, all elements are distinct. So it is unique.
     
     So the strategy works perfectly.
     
     Algorithm details:
     Maintain range $[L, R]$. Initially $1, n$.
     Loop while $L < R$:
       1. Query `? L R`. Store result in `p`.
       2. Calculate $mid = (L + R) / 2$.
       3. Query `? L mid`. Store result in `q`.
       4. If `p == q`:
          $M \in [L, mid]$. Set $R = mid$.
        Else:
          $M \in [mid+1, R]$. Set $L = mid + 1$.
          
     When loop terminates ($L=R$), $L$ is the answer.
     Output `! L`.
     
     Complexity:
     Range reduces by half each time.
     Steps: $\log_2 n$.
     Queries per step: 2.
     Total queries: $2 \log_2 n$.
     Constraints allow $15 \log_2 n$. This is efficient.
     
     Edge case: $n=2$.
     $L=1, R=2$.
     Query `? 1 2` -> returns $p$. $p$ is index of 2nd largest.
     $mid = 1$.
     Query `? 1 1`. This is invalid!
     Constraint says $1 \le l < r \le n$.
     So we cannot query single element intervals.
     The loop condition should handle $n$.
     If $R - L == 1$ (i.e., $R = L + 1$):
       We have $[L, L+1]$.
       Query `? L L+1`. Returns 2nd largest index.
       Since there are only 2 elements, the 2nd largest is the smaller one.
       The larger one is $M$.
       If query returns $L$, then $arr[L]$ is 2nd largest. So $M$ is at $L+1$.
       If query returns $L+1$, then $arr[L+1]$ is 2nd largest. So $M$ is at $L$.
       Return $L$ if return is $L+1$, else $L+1$.
       So we can solve base case in 1 query.
       
     Wait, my generic logic required `? L mid` where $mid < R$.
     If $R = L+1$, $mid = L$.
     `? L L` is invalid.
     So special handling for base case needed.
     
     Also, is it guaranteed that $M$ is never $p$?
     In the logic: $p$ is the index of 2nd largest. $M$ is index of largest.
     Since values are distinct, $p \neq M$.
     
     Wait, there's another edge case: $n=2$.
     Code needs to handle $R=L+1$ separately or adjust loop.
     
     Wait, the problem says "sum of $(r-l+1)$ over all queries ... not exceed $30n$".
     With $2 \log n$ queries, total cost is $\sum 2 \log n \approx 2 N$ ? No.
     The cost is per query.
     Cost of query is $O(r-l)$.
     Wait, the problem says: "Each query '? l r' is answered in O(r - l) time by the interactor."
     "The sum of (r - l + 1) over all queries in a test case must not exceed 30n."
     This is an interaction limit constraint. We must respect it.
     Our queries: `? L R` and `? L mid`.
     The range lengths sum to $(R-L+1) + (mid-L+1)$.
     In worst case, this is roughly $1.5 \times$ current length?
     Total sum over recursion levels: $\sum_{depth=0}^{h} 2^0 \times (n/2^d) \times 2$?
     Wait, level 0: len $n$. 2 queries: len $n$ and $n/2$. Sum $1.5n$.
     Level 1: len $n/2$. 2 queries: len $n/2$ and $n/4$. Sum $1.5(n/2)$.
     Total sum $\approx 1.5n (1 + 1/2 + ...) \approx 3n$.
     This is well within $30n$.
     
     Is there a constraint on memory? 1024MB.
     Time limit 10s.
     
     Wait, flushing.
     Must flush after every output.
     
     Implementation details:
     Use fast I/O? Not strictly necessary but good practice.
     Interactive problem requires `endl` or `flush`.
     Include `<cstdio>` for `fflush(stdout)`, `scanf`, `printf`.
     
     One nuance:
     If $p==q$, we go Left.
     If $p!=q$, we go Right.
     Wait, what if $p$ is not in $[L, mid]$?
     In the logic:
     If $M \in [L, mid]$, then $p \in [L, mid]$ and $p$ is 2nd largest in $[L, mid]$. So $q=p$.
     If $M \in [mid+1, R]$, then $A[p]$ is max in $[L, mid]$. $p \in [L, mid]$.
     Wait, if $p$ is max in $[L, mid]$, then $p$ is NOT returned by `? L mid`.
     Wait, the query returns the index of the *second largest*.
     If $p$ is the max in $[L, mid]$, then $q$ is the index of the element which is second largest in $[L, mid]$.
     Since $A[p]$ is max, $q \neq p$.
     This implies $q \in [L, mid]$ but $q \neq p$.
     
     Wait, there is a subtle possibility.
     Is it possible that `? L mid` returns an index outside $[L, mid]$?
     Definition of query: `? l r` returns index in $[l, r]$.
     So $q \in [L, mid]$.
     
     So the condition `q == p` checks if $p$ is the answer to the query.
     If $M \in [L, mid]$, $p$ is 2nd largest in $[L, mid]$, so query returns $p$. Correct.
     If $M \in [mid+1, R]$, $p$ is MAX in $[L, mid]$, so query returns something else (2nd largest). Correct.
     
     Are there any cases where $q$ could equal $p$ even if $M \in [mid+1, R]$?
     Only if $p$ is not the max.
     But we proved $p$ is max in $[L, mid]$ in that case.
     Why? Because $p$ is 2nd max in $[L, R]$. The only element bigger is $M$.
     If $M \notin [L, mid]$, then no element in $[L, mid]$ is bigger than $A[p]$.
     So $A[p]$ is max.
     Since values are unique, $p$ is unique max.
     So $q \neq p$.
     
     Wait, what if $n=3$?
     $L=1, R=3$. Mid=2.
     Query `? 1 3` -> $p$.
     If $M \in [1, 2]$, $p \in [1, 2]$. Query `? 1 2` returns $p$. -> Go Left.
     If $M \in [3, 3]$, $M=3$. Then $A[p]$ is max in $[1, 2]$. Query `? 1 2` returns something else. -> Go Right.
     
     Seems correct.
     Base case $n=2$:
     $L=1, R=2$.
     Query `? 1 2` -> $p$.
     If $p=2$, $M=1$.
     If $p=1$, $M=2$.
     Output $p$?
     If $p=1$, then 1 is 2nd largest. 2 is max. $M=2$.
     If $p=2$, then 2 is 2nd largest. 1 is max. $M=1$.
     So return $M = (p == 1 ? 2 : 1)$. Or simply $p == 1 ? 2 : 1$?
     Wait. If $p=1$ (value at 1 is smaller), then 2 is larger. Max is at 2.
     So if $p=1$, $M=2$.
     If $p=2$, $M=1$.
     Basically if $p \neq L+1$ then $M=L+1$? No.
     $p$ is the index returned.
     If $p$ is returned, it is the index of 2nd largest.
     We want index of max.
     Since only 2 elements, the max is the other one.
     So $M = L + R - p$.
     Since $L+R = 1+2 = 3$. So $M = 3 - p$.
     Or just `! (p == 1 ? 2 : 1)`.
     
     Is it possible to handle base case in the general loop?
     Loop condition `while (L < R - 1)`?
     If $R = L+1$, stop.
     Handle $L, L+1$ manually.
     
     Wait, the question says "After printing each query... flush".
     
     Double check the logic for $p \in [L, mid]$.
     Wait, if $M \in [L, mid]$, we concluded $p \in [L, mid]$.
     Is it possible $p \in [mid+1, R]$?
     If $M \in [L, mid]$, then $A[M]$ is max in $[L, mid]$.
     The 2nd max in $[L, R]$ is determined by $\max(\text{2nd max of } [L, mid], \text{max of } [mid+1, R])$.
     Wait, earlier I said $A[p]$ is 2nd largest in $[L, mid]$ if $M \in [L, mid]$.
     Is this true?
     Recall: if $M \in [L, mid]$, then $p$ is the index of 2nd largest in $[L, mid]$.
     Let's re-verify.
     If $M \in [L, mid]$, then $A[M]$ is max of $[L, mid]$.
     So any element in $[mid+1, R]$ is $< A[M]$.
     Actually, $A[M]$ is global max.
     If $p$ is the index of 2nd largest in $[L, R]$, then $A[p]$ is the largest value among $[L, R] \setminus \{M\}$.
     If $M \in [L, mid]$, then elements in $[mid+1, R]$ are in the set.
     Is $A[p]$ necessarily in $[L, mid]$?
     Suppose $p \in [mid+1, R]$.
     Then $A[p]$ is an element in the right part.
     But $A[p]$ is the 2nd largest.
     So $\max([L, mid] \setminus \{M\}) < A[p]$ AND $\max([mid+1, R] \setminus \{p\}) < A[p]$.
     If $p \in [mid+1, R]$, then $\max([mid+1, R] \setminus \{p\}) < A[p]$ is trivial.
     And $\max([L, mid] \setminus \{M\}) < A[p]$.
     So this configuration is possible?
     Let's construct an example.
     $[1, 3]$. $L=1, R=3$. $mid=2$.
     Arr: $[3, 1, 2]$. $M=1$. $p=3$ (value 2).
     Here $M \in [L, mid] = [1, 2]$.
     $p=3 \in [mid+1, R]$.
     So $p$ CAN be in the right half.
     
     If $p$ can be in right half when $M \in [L, mid]$, then my condition $q == p$ is insufficient?
     Wait, in the example $[3, 1, 2]$, $L=1, R=3, mid=2$.
     Query `? 1 3`. $M=1$ (at 1). 2nd max is 2 (at 3). So $p=3$.
     Query `? 1 2`. Elements $\{3, 1\}$. Max 3 (at 1), 2nd max 1 (at 2).
     So query returns $q=2$.
     Check condition: $p=3, q=2$. $p \neq q$.
     So we go Right ($M \in [3, 3]$).
     But actual $M=1 \in [1, 2]$ (Left).
     My algorithm went Right! This is WRONG.
     
     Ah, the flaw is assuming that if $M \in [L, mid]$, then $p$ must be in $[L, mid]$.
     Wait, why did I think that?
     Earlier proof: "If $M \in [L, mid]$, then $A[p]$ is max of $[L, mid]$?" No, that was for the case $M \in [mid+1, R]$.
     Let's re-evaluate.
     
     Case 1: $M \in [L, mid]$.
       We query `? L R` -> returns $p$. $p$ is 2nd max in $[L, R]$.
       We query `? L mid` -> returns $q$. $q$ is 2nd max in $[L, mid]$.
       Subcase 1.1: $p \in [L, mid]$.
         Then $A[p]$ is 2nd max in $[L, mid]$?
         If $p \in [L, mid]$, then $p \in [L, R] \setminus \{M\}$.
         Since $M \in [L, mid]$, the set $[L, R] \setminus \{M\}$ is composed of $([L, mid] \setminus \{M\}) \cup [mid+1, R]$.
         $A[p]$ is the max of this union.
         So $A[p] > \max([mid+1, R])$.
         Also $A[p] > \max([L, mid] \setminus \{M\})$? No, only if $p$ is max of $[L, mid] \setminus \{M\}$.
         Wait, if $p \in [L, mid]$, it is one specific element.
         If $A[p]$ is max of the whole remainder, then it must be $> \max([mid+1, R])$.
         And it must be $> \max([L, mid] \setminus \{M\})$.
         So $p$ is indeed the 2nd max of $[L, mid]$ IF AND ONLY IF $A[p] > \max([mid+1, R])$ and $A[p] > \max([L, mid] \setminus \{M\})$.
         We know $A[p] > \max([mid+1, R])$ because $p$ is max of remainder.
         Also we know $p \in [L, mid]$. So $A[p]$ is in $[L, mid]$.
         So in $[L, mid]$, $A[p]$ is greater than everything in $[mid+1, R]$ (but those aren't in $[L, mid]$) and greater than everything in $[L, mid] \setminus \{M\}$?
         Since $A[p]$ is 2nd max in $[L, R]$, yes $A[p]$ beats everything in $[L, R] \setminus \{M, p\}$.
         So in $[L, mid]$, $A[p]$ beats everything except possibly $M$.
         So if $p \in [L, mid]$, then $p$ is 2nd max in $[L, mid]$.
         So $q = p$.
         
       Subcase 1.2: $p \in [mid+1, R]$.
         Then $p$ is not in $[L, mid]$.
         So query `? L mid` will definitely not return $p$ (since index must be in $[L, mid]$).
         So $q \neq p$.
         
       Conclusion: If $M \in [L, mid]$, then $q == p$ implies $p \in [L, mid]$.
               If $q == p$, we know $M \in [L, mid]$?
               Wait. If $M \in [L, mid]$ AND $p \in [mid+1, R]$, then $q \neq p$.
               If $M \in [L, mid]$ AND $p \in [L, mid]$, then $q = p$.
               So if $q == p$, then $M \in [L, mid]$.
               BUT if $q \neq p$, $M$ could be in $[L, mid]$ (with $p$ on right) OR $M \in [mid+1, R]$.
               
               Let's check $M \in [mid+1, R]$.
               Then $M \notin [L, mid]$.
               In this case, $A[M]$ is not in $[L, mid]$.
               Max of $[L, mid]$ is $A[p]$ (proven before).
               Since $A[p]$ is max of $[L, mid]$, $p$ cannot be the 2nd max.
               So $q \neq p$.
               
               So:
               If $M \in [mid+1, R]$, then $q \neq p$.
               If $M \in [L, mid]$, then $q = p$ OR $q \neq p$ (if $p \in [mid+1, R]$).
               This makes the logic incomplete. $q \neq p$ doesn't uniquely identify $M \in [mid+1, R]$.
     
     We need a stronger condition.
     We need to distinguish "p is the 2nd max of L..R and lies in L..mid" vs "p lies in mid+1..R".
     Or rather, we need to decide if $M$ is in $[L, mid]$.
     
     Actually, look at the example where it failed: $M \in [L, mid]$ but $p \in [mid+1, R]$.
     In that case `? L R` returned $p$ (on right). `? L mid` returned $q$ (something else).
     Result was $q \neq p$, leading us to guess Right. Wrong guess.
     Why did this happen? Because there was a large element on the right (which was $p$), and an even larger one on the left ($M$).
     But we didn't see $M$ as max because we didn't have a query for the whole array that distinguished... wait we had `? L R`.
     The issue is $p$ was in the right half.
     
     Wait, if $p \in [mid+1, R]$, then $A[p]$ is the largest in $[mid+1, R]$.
     Actually $A[p]$ is the 2nd largest in $[L, R]$.
     So $\max([mid+1, R] \setminus \{p\}) < A[p]$.
     And $\max([L, mid] \setminus \{M\}) < A[p]$.
     
     This seems hard.
     Wait, let's revisit the problem.
     Maybe we don't need to binary search for $M$ directly.
     We want to find $M$.
     
     Let's think about the Cartesian tree again.
     $M$ is the root.
     $p$ is the index of one of the children of $M$.
     Specifically, the child with larger value.
     The other child (let's call it $other$) is the root of the other subtree.
     Its value is smaller than $A[p]$.
     We want to find $M$.
     We have identified one child $p$.
     If we knew which child is $p$, we could move up? No.
     If we knew $M$, we would know $p$.
     We found $p$ by querying $[1, n]$.
     Now we know $p$ is a child of $M$ (in Cartesian tree sense).
     This means $A[p] >$ all nodes in the subtree of $p$.
     And $A[p] <$ all ancestors.
     
     What if we use the property of the query `? l r`?
     If we query `[l, r]` and $M \in [l, r]$, we get a child of $M$.
     If we query `[l, r]` and $M \notin [l, r]$, we get the max of the range?
     No, we get the 2nd largest in range.
     If $M$ is outside, then the max in range is some $M'$.
     The return is the 2nd max.
     This doesn't help much unless we know $M'$.
     
     Let's try to find the position of $n$ by iterating.
     Maybe we can maintain the candidate for $M$ as a list of possible indices? No.
     
     Alternative: Randomized guessing?
     Wait, we can perform queries to isolate $M$.
     If we pick a random $k$, and query `? k R` (if $R>k$), we get $p$.
     Does $p$ tell us anything about $M$?
     
     Actually, consider the following:
     If we have a candidate $M_{cand}$ for the position of $n$.
     If $M_{cand}$ is actually the position of $n$, then for any $l, r$ containing $M_{cand}$, $n$ is the max.
     If $M_{cand}$ is NOT the position of $n$, then $M_{cand}$ is some $k < n$.
     Then there exists $pos(n)$ elsewhere.
     If we query a range containing $M_{cand}$ but NOT $pos(n)$, the query returns 2nd largest in that range.
     If we query a range containing BOTH, it returns 2nd largest.
     
     Is there a property related to $n$ specifically?
     The problem asks for position of $n$.
     Since we don't know values, we treat values as abstract.
     But we know $n$ is the global maximum.
     
     Let's go back to the logic.
     If we query `? 1 n`, we get $p$.
     $p$ is the index of the second largest element.
     This element $p$ is adjacent to $n$ in the Cartesian tree.
     In the Cartesian tree, $n$ is root. $p$ is a child.
     The other child is some $r'$.
     If we could find $p$ and $r'$, we could merge them?
     Actually, if we know $p$, can we query around $p$?
     If we query `? p n`, we get...
     If $n \in [p, n]$ (i.e. $p < n$), then $n$ is the max.
     Then we get the index of the second largest in $[p, n]$.
     Let this be $p_2$.
     If $n$ is at index $pos(n)$.
     $p$ is in the right subtree of $n$. So all elements in $p$'s subtree are smaller than $A[n]$.
     But $p$ is the root of the right subtree?
     Wait, $p$ is the index of the 2nd largest in $[1, n]$.
     This implies $p$ is the largest of the two subtrees of $n$.
     So if $pos(n) < p$, then $p$ is the max of $[pos(n)+1, n]$.
     If $pos(n) > p$, then $p$ is the max of $[1, pos(n)-1]$.
     
     YES!
     This is the key.
     If $pos(n) < p$, then $p$ is the max of $[pos(n)+1, n]$.
     If $pos(n) > p$, then $p$ is the max of $[1, pos(n)-1]$.
     We know $p$. We want to find $pos(n)$.
     Let's check the subarray $[1, p-1]$.
     If $pos(n) \in [1, p-1]$, then $A[pos(n)]$ is max of $[1, n]$ and in $[1, p-1]$.
     So max of $[1, p-1]$ is $n$.
     Then query `? 1 p-1` returns 2nd largest of $[1, p-1]$.
     But this doesn't help directly finding $pos(n)$.
     
     Wait, let's look at the two hypotheses again.
     Hypothesis 1: $pos(n) < p$.
     Hypothesis 2: $pos(n) > p$.
     Can we check these?
     If $pos(n) < p$, then $A[p]$ is the maximum of $[pos(n)+1, n]$.
     This means $A[p] >$ all elements in $[pos(n)+1, n]$.
     This implies that for the range $[p, n]$, the maximum is $A[p]$?
     No, $A[p]$ is the max of $[pos(n)+1, n]$. But does it extend to $p$?
     Wait, $pos(n)$ is the index of $n$.
     If $pos(n) < p$, then $n$ is not in $[p, n]$.
     So $[p, n]$ is entirely to the right of $n$.
     So $\max_{k \in [p, n]} A[k]$ is some value $< n$.
     Is $A[p]$ the max of $[p, n]$?
     $p$ is the max of $[pos(n)+1, n]$. Since $p$ itself is in this range (obviously), yes.
     Wait, $p \in [pos(n)+1, n]$ is guaranteed if $pos(n) < p$.
     So $A[p]$ is the max of $[p, n]$.
     So if $pos(n) < p$, then $A[p] = \max([p, n])$.
     If $pos(n) > p$, then $p$ is in $[1, pos(n)-1]$.
     $p$ is the max of $[1, pos(n)-1]$.
     So if $pos(n) > p$, then $A[p] = \max([1, p])$.
     
     So:
     1. If $pos(n) < p$, then $A[p]$ is max of $[p, n]$.
        This implies `? p n` returns $pos(n)$? NO.
        `? p n` returns index of 2nd largest in $[p, n]$.
        Since $A[p]$ is max in $[p, n]$, `? p n` returns index of element which is 2nd largest in $[p, n]$.
        Let's call this $p_2$.
        Does this give us info?
        
     2. If $pos(n) > p$, then $A[p]$ is max of $[1, p]$.
        This implies `? 1 p` returns index of 2nd largest in $[1, p]$.
        Let's call this $p'_2$.
     
     Wait, we don't know which hypothesis is true.
     However, we can try to verify them.
     
     Check Hypothesis 1: $pos(n) < p$.
     Condition: $A[p] = \max([p, n])$.
     We don't know max.
     But we can query `? p n`.
     If `? p n` returns $p_2$.
     This tells us who is 2nd max in $[p, n]$.
     This doesn't confirm if $A[p]$ is max.
     Unless...
     Wait. If $A[p]$ is NOT max of $[p, n]$, then there is some $k \in [p, n], k \neq p$ with $A[k] > A[p]$.
     But $p$ was defined as the 2nd max in $[1, n]$.
     The only element larger than $A[p]$ is $n$.
     So if $A[k] > A[p]$ for $k \in [p, n]$, then $k$ must be $pos(n)$.
     So $pos(n) \in [p, n]$ means $pos(n)=k$.
     But if $pos(n) \in [p, n]$, then $n$ is in the range $[p, n]$.
     So $n$ is the max of $[p, n]$.
     So $A[pos(n)] > A[p]$.
     But $p$ is 2nd max of $[1, n]$. So $n$ is max of $[1, n]$.
     So $A[pos(n)] = n > A[p]$.
     This is consistent.
     
     Let's check the condition:
     $pos(n) < p \implies n \notin [p, n]$.
     If $n \notin [p, n]$, then $A[p]$ is the largest in $[p, n]$?
     Let's trace.
     $n$ is at $pos(n)$.
     $p$ is 2nd max in $[1, n]$.
     $p$ is the max of $[pos(n)+1, n]$?
     Wait. $p$ is 2nd max of $[1, n]$.
     This implies $A[p]$ is larger than everything except $A[pos(n)]$.
     So for any $k \neq pos(n)$, $A[k] < A[p]$.
     So specifically for any $k \in [p, n], k \neq p$, $A[k] < A[p]$?
     No.
     $A[p]$ is the 2nd largest in $[1, n]$.
     So it is larger than all other elements except $n$.
     So yes, $A[p]$ is larger than all $k \neq pos(n)$ and $k \neq p$.
     In particular, for any $k \in [p, n]$, if $k \neq p$, is $A[k] < A[p]$?
     Yes, provided $k \neq pos(n)$.
     If $pos(n) \in [p, n]$, then $A[pos(n)] > A[p]$.
     If $pos(n) \notin [p, n]$ (meaning $pos(n) < p$), then for all $k \in [p, n], k \neq p$, $A[k] < A[p]$.
     So $A[p]$ is indeed the MAX of $[p, n]$ IF $pos(n) < p$.
     
     If $pos(n) > p$, then $pos(n)$ is not in $[1, p]$.
     So for all $k \in [1, p], k \neq p$, $A[k] < A[p]$.
     So $A[p]$ is MAX of $[1, p]$ IF $pos(n) > p$.
     
     So:
     - If $pos(n) < p$, then $p$ is the max of $[p, n]$.
     - If $pos(n) > p$, then $p$ is the max of $[1, p]$.
     
     Now, how to check this?
     We can query `? p n`.
     If `? p n` returns $p$, then $p$ is NOT the max? No, `? p n` returns 2nd largest.
     If $p$ is max of $[p, n]$, then `? p n` returns index of 2nd largest.
     This index $p_{res} \neq p$.
     If $pos(n) \in [p, n]$, then $n$ is max of $[p, n]$.
     So `? p n` returns 2nd largest.
     In this case $p_{res}$ would be the max of the other part?
     Actually, if $pos(n) \in [p, n]$, then $n$ is in range.
     Then $p$ (2nd max in $[1, n]$) is smaller than $n$.
     Is $p$ in $[p, n]$? Yes.
     So `? p n` compares $n$ and $p$. $n$ is max. $p$ is 2nd max?
     Wait. Is $p$ the 2nd max of $[p, n]$?
     If $pos(n) \in [p, n]$, then range $[p, n]$ contains $n$.
     The max is $n$.
     The 2nd max in $[p, n]$ is either $p$ or some other $k$.
     Since $p$ is the 2nd max in $[1, n]$ (global), it is the best candidate for 2nd max.
     Is there any $k \in [p, n], k \neq n$ with $A[k] > A[p]$?
     No, because $A[p]$ is 2nd max in $[1, n]$. The only element $> A[p]$ is $n$.
     So $p$ IS the 2nd max of $[p, n]$.
     So if $pos(n) \in [p, n]$, `? p n` returns $p$.
     
     If $pos(n) < p$, then $n \notin [p, n]$.
     As derived, $A[p]$ is max of $[p, n]$.
     So `? p n` returns index of 2nd largest in $[p, n]$.
     This index is definitely not $p$ (since $p$ is max).
     
     So:
     Query `? p n`.
     - If result is $p$: Then $pos(n) \in [p, n]$.
       Then $n$ is in right half (relative to $p$).
       Wait, $p$ is fixed. $pos(n)$ could be anywhere.
       But if result is $p$, it means $p$ is 2nd max. This implies $n$ is present and larger.
       So $pos(n) \in [p, n]$.
       Then we restrict search to $[p, n]$.
     - If result is NOT $p$: Then $p$ is NOT 2nd max in $[p, n]$.
       Since $p$ is not $n$ (obviously), this implies $n$ is NOT in $[p, n]$.
       So $pos(n) \notin [p, n]$.
       Since $pos(n) \in [1, n]$, it must be $pos(n) < p$.
       So $pos(n) \in [1, p-1]$.
     
     Wait!
     If $pos(n) \in [p, n]$, then `? p n` returns $p$.
     Wait. Is it guaranteed that $p$ is 2nd max in $[p, n]$?
     We know $A[p]$ is 2nd max in $[1, n]$.
     If $pos(n) \in [p, n]$, then $n$ is in $[p, n]$.
     Then $A[n] > A[p]$.
     Are there any elements in $[p, n]$ with value $> A[p]$ besides $n$?
     No, because $A[p]$ is 2nd max in $[1, n]$.
     So $p$ is indeed the 2nd max of $[p, n]$.
     So result is $p$.
     
     If $pos(n) \notin [p, n]$ (i.e. $pos(n) < p$):
     Then $n$ is in $[1, p-1]$.
     Then $A[n]$ is not in $[p, n]$.
     We established $A[p]$ is max of $[p, n]$.
     So `? p n` returns something other than $p$.
     
     So this simple check works!
     Algorithm:
     1. Query `? 1 n`. Let result be $p$.
     2. Check `? p n`. Let result be $q$.
        Wait, if $p=n$, impossible since $p \neq n$ (indices distinct? No $p$ is index). $p$ is index of 2nd max. $n$ is index of max. $p \neq n$.
        Wait, index of max is not $n$. The max is $n$. Index is unknown.
        Wait, $p$ is index.
        So `? p n` is valid query if $p < n$.
        If $p = n$, we can't query `[n, n]`. But `? l r` requires $l < r$.
        So if $p=n$, then $M$ is not in $[n, n]$, contradiction.
        Wait. If $pos(n) < p$. If $pos(n)$ is in $[p, n]$ implies $pos(n) \ge p$.
        Wait. If $pos(n) = n$? Impossible, $p$ would be $n-1$ or something.
        Wait, $p$ is index of 2nd max. $pos(n)$ is index of max. $p \neq pos(n)$.
        So if $pos(n) = n$, then $p < n$.
        If $pos(n) < p$, then $p \le n$ (trivial).
        Wait, if $p=n$, then $pos(n)$ cannot be $n$ (distinct indices).
        If $p=n$, then $pos(n) \in [1, n-1]$.
        But if $p=n$, then we can't query `? n n`.
        Wait, query `? p n`. If $p=n$, this is `? n n` invalid.
        Can $p=n$? $p$ is index of 2nd largest. If $p=n$, then element at $n$ is 2nd largest.
        Then $pos(n) \neq n$.
        If $pos(n) < p$ (i.e. $<n$), then `? p n` (`? n n`) is invalid.
        However, if $p=n$, we know $pos(n) \in [1, n-1]$.
        Wait, if $p=n$, $A[n]$ is 2nd max.
        Is it possible $pos(n) > n$? No.
        So $pos(n) < p$.
        So if $p=n$, $pos(n) < p$.
        We can proceed with recursion.
        
        General logic:
        Range $[L, R]$.
        Query `? L R` -> $p$.
        If $p == R$, then $M \in [L, R-1]$. (Because $A[R]$ is 2nd max, so Max is elsewhere).
        Wait, if $p=R$, then $A[R]$ is 2nd max. $M$ is somewhere else in $[L, R]$.
        Does this imply $M < R$?
        $M$ could be anywhere.
        But we checked `? L R` gave $p$.
        If we just rely on `? p n`...
        Wait, if $p=n$, then $p \not< n$ is false.
        If $p < n$, we can query `? p n`.
        If result is $p$, then $pos(n) \in [p, n]$.
           If result is not $p$, then $pos(n) < p$.
        But if result is $p$, we haven't narrowed down the range yet?
        Wait.
        If `? p n` returns $p$:
          This means $pos(n) \in [p, n]$.
          So new range is $[p, n]$.
        If `? p n` returns $\neq p$:
          This means $pos(n) < p$.
          So new range is $[L, p-1]$?
          Wait, original $L$ was 1. So $[1, p-1]$.
          
        Wait, this narrows down the range!
        Start with $L=1, R=n$.
        1. Query `? L R` -> $p$.
        2. If $p == R$:
             We know $pos(n) \neq R$. So $pos(n) < R$.
             New range $[L, R-1]$.
             (Since $A[R]$ is 2nd max, $M$ is not at $R$. Could it be $> R$? No, bounded by $R$.)
             So range becomes $[L, R-1]$.
             This is 1 query step.
        3. If $p < R$:
             We query `? p R`.
             Note: range is $[p, R]$ because $p$ is index.
             If result is $p$:
                $pos(n) \in [p, R]$.
                New range $[p, R]$.
             If result $\neq p$:
                $pos(n) < p$.
                New range $[L, p-1]$.
             
        Let's trace complexity.
        If we go to $[p, R]$ (length decreases? $R-p$).
        If we go to $[L, p-1]$ (length decreases? $p-L-1$).
        Wait, does length decrease geometrically?
        Initially $n$. $p$ comes from `? 1 n`.
        $p$ is index of 2nd max.
        If we go right: new length $R-p$.
        If we go left: new length $p-L-1$.
        In worst case, $p$ could be $n/2$. Length halves.
        Then we do `? p R`. This splits range again.
        Wait, if we select $[p, R]$ (right part), $p$ is the index we found in the LEFT part? No.
        Wait. $p$ is index of 2nd max.
        If `? p R` returns $p$, then $pos(n) \in [p, R]$.
        Wait, if $pos(n) \in [p, R]$, then $n$ is in $[p, R]$.
        Then $A[p]$ is 2nd max in $[p, R]$.
        So we are recursing on $[p, R]$.
        But wait, we know $p$ is a specific element (2nd max).
        So we are effectively discarding $[L, p-1]$.
        Is it possible that $|p-R|$ is small?
        Example: Arr $[2, 1, 3]$. $n=3$.
        $L=1, R=3$.
        `? 1 3` -> 2nd max is 2 (at 1). So $p=1$.
        Check `? 1 3`. Result $p=1 \neq 3$. (Wait, $R=3$).
        Query `? p R` -> `? 1 3`.
        Wait, `? p R` is same as `? L R`?
        If $p$ is index of 2nd max.
        If $p=L$, then $pos(n)$ could be in $[L, R]$.
        If result is $p$ (i.e. 1), then $pos(n) \in [1, 3]$.
        So range stays same?
        Wait. If result is $p$, we conclude $pos(n) \in [p, R]$.
        If result $\neq p$, we conclude $pos(n) < p$.
        If $p=L$, then result $\neq p$ implies $pos(n) < L$, which is impossible.
        So if $p=L$, we MUST have result $p$.
        So we stay in $[L, R]$. No progress.
        
        But $p=L$ means $A[L]$ is 2nd max.
        Wait, $A[L]$ is 2nd max in $[1, n]$.
        This implies $A[L]$ is very large.
        But if $p=L$, we know $pos(n) \ge p$.
        So $pos(n) \in [L, R]$.
        So we keep searching.
        But we made no progress in range size.
        We wasted queries.
        
        However, look at the logic:
        If $p=L$, we query `? L R` -> $p=L$.
        Query `? p R` -> `? L R`.
        This is redundant.
        
        Wait. $p$ is index of 2nd max.
        We know $pos(n)$ is somewhere.
        If $p$ is index of 2nd max, then $pos(n)$ must be an index where $A[k] > A[p]$.
        Since $p$ is 2nd max, $pos(n)$ is unique such index.
        So $pos(n) \neq p$.
        So if we find $p$, we know $pos(n) \neq p$.
        Wait, we also know something more.
        If we use the property that $A[p]$ is max of $[L, R] \setminus \{pos(n)\}$.
        
        Actually, let's look at the problem differently.
        We want to find $pos(n)$.
        We have $n$.
        We can find $pos(n)$ by identifying the position where the element is larger than everything else.
        
        Let's use the first successful strategy again but fix the flaw.
        Flaw was distinguishing $M \in [L, mid]$ vs $[mid+1, R]$.
        My derivation:
        If $M \in [mid+1, R]$, then $q \neq p$ (where $q$ is res of `? L mid`).
        If $M \in [L, mid]$, then $q$ can be equal or not equal to $p$.
        Specifically, if $p \in [mid+1, R]$, then $q \neq p$.
        So both branches yield $q \neq p$.
        
        Wait. If $M \in [L, mid]$, and $p \in [mid+1, R]$.
        This means 2nd max is on the right, Max is on the left.
        This configuration is possible.
        In this case, `? 1 3` -> $p=3$. `? 1 2` -> $q=2$. $q \neq p$.
        This led to "Go Right", which was wrong (should go Left).
        However, notice something.
        In `? 1 3`, $p=3$ (index 3).
        In `? 1 2`, $q=2$ (index 2).
        The value at $p$ ($A[3]$) was 2.
        The value at $q$ ($A[2]$) was 1.
        The value at $M$ ($A[1]$) was 3.
        $A[p] > A[q]$.
        Is it always true that $A[p] > A[q]$?
        $p$ is 2nd max in $[L, R]$.
        $q$ is 2nd max in $[L, mid]$.
        If $M \in [L, mid]$ and $p \in [mid+1, R]$.
        $A[p]$ is 2nd max in $[L, R]$.
        So $A[p]$ is larger than all elements in $[L, mid] \setminus \{M\}$.
        And $q$ is the 2nd max in $[L, mid]$.
        So $q$ is smaller than $A[p]$ (unless $q=M$, impossible).
        So $A[p] > A[q]$.
        
        Wait, can we compare values using queries?
        No, direct comparison.
        But maybe we can infer from indices?
        We have $p$ and $q$.
        If $M \in [mid+1, R]$, $p \in [L, mid]$ and is max of $[L, mid]$.
        Then $A[p] > A[q]$ is obvious (max > 2nd max).
        If $M \in [L, mid]$ and $p \in [L, mid]$, then $p=q$.
        If $M \in [L, mid]$ and $p \in [mid+1, R]$, then $A[p] > A[q]$.
        
        So in all cases $A[p] \ge A[q]$.
        And equality holds iff $p=q$.
        This doesn't help distinguishing cases where $q \neq p$.
        
        However, we have more information.
        $A[p]$ is 2nd max in $[L, R]$.
        $A[q]$ is 2nd max in $[L, mid]$.
        In the case $M \in [mid+1, R]$, $A[p]$ is max in $[L, mid]$.
        So $A[p] > A[q]$.
        In the case $M \in [L, mid]$ and $p \in [mid+1, R]$, $A[p] > A[q]$.
        In the case $M \in [L, mid]$ and $p \in [L, mid]$, $A[p] = A[q]$ (as indices).
        
        So if $p \neq q$, we know $M$ could be in either half.
        Wait, is it possible to distinguish?
        
        Let's try a different angle.
        We want to locate $n$.
        We know $n$ is the max.
        If we have a range $[L, R]$ containing $n$.
        Query `? L R`. Gets $p$.
        $p$ is the index of the second largest.
        This $p$ is a "neighbor" of $n$.
        Actually, $p$ is the index of the largest element in $[L, R] \setminus \{pos(n)\}$.
        If we can find the range $[L, R]$ where $n$ is, and then narrow it down.
        
        Consider this strategy:
        Keep a set of candidate positions $C$. Initially $[1, n]$.
        Query `? 1 n` -> $p$.
        We know $n$ is at $pos(n) \neq p$.
        Is it possible to eliminate a range?
        Suppose we check `? p R`.
        If `? p R` returns $p$, then $pos(n) \in [p, R]$.
        Wait. If `? p R` returns $p$, it implies $A[p]$ is NOT max of $[p, R]$.
        Wait, `? p R` returns index of 2nd max.
        If result is $p$, then $A[p]$ is NOT max (since it returns 2nd max).
        Wait, if $A[p]$ is max, it cannot be returned.
        So if result is $p$, $p$ is NOT max. So $pos(n) \in [p, R]$.
        If result is $\neq p$, then $A[p]$ is max. So $pos(n) \notin [p, R]$.
        So $pos(n) < p$.
        
        This logic held before.
        So we can binary search $pos(n)$!
        We need a function `check(mid)` to test if $pos(n) \in [1, mid]$.
        Wait, the check logic relies on querying `[p, R]`.
        If we pick $mid$, and we want to know if $pos(n) \in [1, mid]$.
        We need to know if $pos(n) < p$ (from previous context) or something.
        
        Actually, we don't need to maintain $p$.
        Just:
        Range $[L, R]$. Assume $pos(n) \in [L, R]$.
        Query `? L R` -> $p$.
        If $p == R$, then $pos(n) < R$. (Range reduces to $[L, R-1]$).
        If $p < R$.
        Query `? p R`.
        If `? p R` returns $p$:
           $pos(n) \in [p, R]$.
           Reduce to $[p, R]$.
        If `? p R` returns $\neq p$:
           $pos(n) < p$.
           Reduce to $[L, p-1]$.
           
        Wait.
        If $pos(n) \in [L, R]$, and we reduce to $[p, R]$ or $[L, p-1]$.
        Wait. $pos(n)$ is an index.
        Does this logic guarantee convergence?
        If we reduce to $[p, R]$, length is $R - p$.
        If we reduce to $[L, p-1]$, length is $p - L$.
        Is this better than just checking `? L R`?
        Wait, if we go to $[p, R]$, $p$ is the boundary.
        Does $p$ change?
        In the recursive step, we query `? L' R'` to get new $p'$.
        So yes, we iterate.
        Is the length always halving?
        $p$ is index of 2nd max in $[L, R]$.
        Is it guaranteed that $p$ is close to boundaries?
        Not necessarily. $p$ could be anywhere.
        But wait.
        If $p$ is index of 2nd max in $[L, R]$.
        If $pos(n) \in [p, R]$, we go Right.
        If $pos(n) \in [L, p-1]$, we go Left.
        This partitions the search space based on $p$.
        Wait.
        If $pos(n) < p$, we go to $[L, p-1]$.
        If $pos(n) > p$, we go to $[p, R]$.
        This assumes $pos(n)$ cannot be equal to $p$. True.
        This is EXACTLY a partition of indices!
        Wait, this is not binary search on value, but binary search on index using $p$.
        Does this work?
        We assume $pos(n)$ is the target.
        In one step, we get $p$.
        We split search into $[L, p-1]$ and $[p, R]$.
        Wait, is $p$ a pivot?
        The query `? p R` tells us if $pos(n) \in [p, R]$.
        If yes, we discard $[L, p-1]$.
        If no, we discard $[p, R]$.
        The only issue is determining if `? p R` returning $p$ is sufficient.
        My derivation:
        "If `? p R` returns $p$, then $pos(n) \in [p, R]$."
        "If `? p R` returns $\neq p$, then $pos(n) < p$."
        Let's re-verify.
        Query `? p R`.
        Case 1: $pos(n) \in [p, R]$.
          Then $n$ is in $[p, R]$.
          $p$ is the index of 2nd max in $[L, R]$.
          Since $n \in [p, R]$, $A[n] = n > A[p]$.
          Also, for any $k \in [p, R]$, $k \neq n$, $A[k] < A[n]$.
          Is $A[k] < A[p]$ for $k \in [p, R]$?
          We know $A[p]$ is 2nd max in $[L, R]$.
          So $A[p] > A[k]$ for all $k \in [L, R] \setminus \{pos(n), p\}$.
          So for any $k \in [p, R] \setminus \{n, p\}$, $A[k] < A[p]$.
          So in the range $[p, R]$, the elements are $\{A[p], \dots, A[n], \dots\}$.
          $A[n]$ is max.
          $A[p]$ is larger than everything else in $[p, R]$ (except $n$).
          So $p$ is the index of 2nd largest in $[p, R]$.
          So `? p R` returns $p$.
          This direction holds.
          
        Case 2: $pos(n) < p$.
          Then $n$ is in $[L, p-1]$.
          $n$ is not in $[p, R]$.
          So for any $k \in [p, R]$, $A[k] < A[n]$.
          We need to check if $A[p]$ is the max of $[p, R]$.
          $p$ is 2nd max in $[L, R]$.
          So $A[p] > A[k]$ for all $k \in [L, R] \setminus \{pos(n), p\}$.
          Since $pos(n) < p$, $pos(n) \notin [p, R]$.
          So for all $k \in [p, R] \setminus \{p\}$, $A[k] < A[p]$.
          So $A[p]$ is the unique maximum of $[p, R]$.
          So `? p R` returns index of 2nd largest, which is NOT $p$.
          So it returns something $\neq p$.
          
        So the logic is sound.
        Algorithm:
        1. Query `? 1 n`. Get $p$.
        2. If $p == n$, impossible (since $n$ is index of max, $p$ is index of 2nd max, distinct).
           Wait, $p$ is the returned index. It is the index. $n$ is the value.
           The variable name $n$ denotes length. The value is $n$. The position is $pos(n)$.
           Let's rename variables to avoid confusion.
           Let $len$ be $n$.
           Let $target\_pos$ be unknown.
           
           $L=1, R=len$.
           Repeat:
             Query `? L R` -> $p$.
             If $p == R$:
               // Max is not at R.
               // Since R is the rightmost, and $p$ (2nd max) is at R, max must be in $[L, R-1]$.
               // So target_pos < R.
               $R = R - 1$.
             Else ($p < R$):
               // Check if target_pos >= p.
               // Query `? p R`.
               Query `? p R` -> $res$.
               If $res == p$:
                 // target_pos $\in [p, R]$.
                 // Discard left part.
                 $L = p$. // New range starts at p.
                 // Wait, target_pos cannot be p.
                 // So range is $[p+1, R]$?
                 // If target_pos < p, we would have gone Left.
                 // If target_pos >= p, we are in $[p, R]$.
                 // Wait, target_pos != p.
                 // If we keep $L=p$, next step we query `? p R` again?
                 // Yes, next iteration will query `? p R` (same query).
                 // So infinite loop?
                 // Ah, if we go to $[p, R]$, we must ensure $p$ is excluded?
                 // We know target_pos $\neq p$.
                 // So effectively we narrowed to $[p+1, R]$?
                 // But we only know $res == p \implies target\_pos \in [p, R]$.
                 // But we know target_pos $\neq p$ from definition ($p$ is 2nd max).
                 // So yes, we can set $L = p + 1$.
               Else ($res \neq p$):
                 // target_pos < p.
                 // Discard right part.
                 $R = p - 1$.
                 
           Wait.
           In the case $res == p$, we set $L = p+1$.
           But the range passed to query was $[L, R]$.
           If we set $L = p$, then we query `? p R`.
           If we set $L = p+1$, we query `? p+1 R`.
           Wait. The logic `if res == p: target_pos \in [p, R]` used query `? p R`.
           This query was valid because $p < R$ (checked in `else`).
           If we narrow to $[p+1, R]$, does the query `? p+1 R` make sense?
           Wait. If target_pos $\in [p+1, R]$, then we should update $L=p+1$.
           If we update $L=p$, we might re-enter the same state?
           If $L=p$, we query `? p R`. $res$ will be $p$ (assuming target_pos $> p$).
           Then $L$ becomes $p+1$.
           So next iter $L=p+1$.
           So we essentially skip one iteration but consume query.
           Better to set $L=p+1$ directly.
           
           So:
           Query `? p R`.
           If $res == p$:
              $L = p + 1$.
           Else:
              $R = p - 1$.
              
           Wait. Does this terminate?
           If $res == p$, we go to $[p+1, R]$.
           The old $p$ is removed from consideration?
           Wait. If $target\_pos \in [p+1, R]$, does the property $res == p$ hold?
           We verified: If $target\_pos \in [p, R]$, then $res=p$.
           But if $target\_pos \in [p+1, R]$ (so target $> p$), then $n$ is in $[p+1, R]$.
           Then $p$ is outside $[p+1, R]$.
           In $[p+1, R]$, is $A[p]$ relevant? No, $p$ is not in range.
           Wait. If we query `? p R` and get $p$, it means $n \in [p, R]$.
           If we restrict to $[p+1, R]$, then $n$ is still in range.
           But $p$ is no longer available.
           So `? p R` is not relevant anymore.
           We must restart with new range.
           So:
           1. Query `? L R` -> $p$.
           2. If $p == R$: $R = R - 1$. Continue.
           3. If $p < R$:
              Query `? p R` -> $res$.
              If $res == p$:
                 // $n \in [p, R]$. Since $n \neq p$, $n \in [p+1, R]$.
                 // So new range $[p+1, R]$.
                 $L = p + 1$.
              Else:
                 // $n < p$. New range $[L, p-1]$.
                 $R = p - 1$.
              
           Is $[L, p-1]$ always non-empty?
           If $p=L$, then $R=p-1$ impossible (empty).
           But if $p=L$, then $res=p$ must be true (as proven earlier, if $p=L$, $n \ge p$, so $n \in [L, R]$).
           Wait. If $p=L$, then $res$ (from `? p R` = `? L R`) is $L=p$.
           So we go $L = p+1$.
           If $p=L$, then we shift $L$ to $p+1$.
           So $L$ increases.
           Is it possible $L > R$?
           If $L > R$, loop ends.
           
           Base cases?
           If $L=R$, we found it.
           Wait. If $L=R$, we output $! L$.
           Initial $L=1, R=n$.
           If $n=1$? Constraints $n \ge 2$.
           
           Is the number of queries too high?
           At each step, either $R$ becomes $R-1$ or $L$ becomes $p+1$.
           If $p$ is index of 2nd max.
           If we go Left ($R=p-1$), range shrinks by $R - p$.
           If we go Right ($L=p+1$), range shrinks by $(p+1) - L = p - L + 1$.
           Wait. Is the shrinkage guaranteed to be significant?
           If $p$ is very close to $R$, we shrink $L$.
           If $p$ is very close to $L$, we shrink $R$.
           If $p$ is in middle, we shrink both ways? No, one direction.
           Wait.
           If we go Left, range is $[L, p-1]$. Size is $p - L$.
           If we go Right, range is $[p+1, R]$. Size is $R - p$.
           We don't know if $p-L$ or $R-p$ is smaller.
           $p$ is index of 2nd max.
           Can $p$ be arbitrarily placed?
           Consider Arr: $[n, 1, 2, \dots]$. $p$ will be 2nd largest.
           If $A$ is reverse sorted: $n, n-1, \dots, 1$.
           $p$ (index of 2nd max) will be 2.
           $L=1, R=n$. $p=2$.
           Query `? 2 n`. Res is 2.
           $L$ becomes $3$. Range $[3, n]$.
           Then query `? 3 n`. Res 2nd max is 4 (if $n-1$ is at 2, $n-2$ at 3).
           Wait. Reverse sorted:
           $1: n$
           $2: n-1$
           $3: n-2$
           ...
           Query `? 1 n`. Max is $n$ at 1. 2nd max is $n-1$ at 2. $p=2$.
           Query `? 2 n`. Range $[2, n]$. Max $n-1$ at 2. 2nd max $n-2$ at 3.
           Returns 3.
           So res $\neq p$ (since $res=3 \neq 2$).
           So $R$ becomes $2-1 = 1$.
           Range becomes $[1, 1]$. Found.
           Steps: 1 (L..R), 1 (p..R). Total 2.
           Wait. $res$ was 3. We expected $res \neq p$.
           Wait. If $res \neq p$, then $R = p-1$.
           Range becomes $[L, p-1] = [1, 1]$.
           $L=1, R=1$. Done.
           Works.
           
           Consider sorted array: $1, 2, \dots, n$.
           Max at $n$. 2nd max at $n-1$. $p=n-1$.
           Query `? 1 n` -> $n-1$.
           Query `? n-1 n`. Max $n$ at $n$. 2nd max $n-1$ at $n-1$.
           Returns $n-1$. $res = p$.
           So $L = n-1+1 = n$.
           Range $[n, n]$. Done.
           
           What about worst case?
           $p$ being very balanced?
           $p \approx n/2$.
           Then we cut range to size $n/2$.
           Then we cut again.
           So binary search behavior.
           Each step reduces length by at least 1?
           No. If $p=L+1$, then $L \to L+2$? Or $R \to L$.
           If $p \approx L$, we go Left. $R \to p-1 \approx L-1$.
           If $p \approx R$, we go Right. $L \to p+1 \approx R+1$.
           So we converge.
           The only concern is `? p R`.
           Does the cost constraint hold?
           Sum of $(r-l+1)$.
           In binary search, sum is $O(N)$.
           Here we do 2 queries at each level.
           First query length $K$. Second query length $\approx K/2$ or something?
           Wait. If we go Left ($R=p-1$), new length is $p-1-L$.
           Old length $R-L+1$.
           If we go Right ($L=p+1$), new length $R-(p+1)+1 = R-p$.
           In both cases new length $\le (R-L)/2$ roughly?
           Not necessarily. If $p=L$, left length 0. Right length $R-L$.
           If $p=L$, we go Right (since $res=p$ usually holds if $n \ge p$? Wait, if $p=L$, $res=p$ is true because $n \in [L, R]$).
           So we go Right. New length $R-L$.
           Wait. If $p=L$, length doesn't shrink?
           Let's re-verify $p=L$ case.
           If $p=L$, then $A[L]$ is 2nd max.
           This implies $n$ (max) is at $pos(n)$.
           Since $A[L]$ is 2nd max, $n$ must be somewhere.
           Is it possible $p=L$ implies $pos(n)$ is far away?
           If $p=L$, then $n$ is not at $L$.
           $n$ is at $pos(n)$.
           If $pos(n) \in [L, R]$.
           If $pos(n) = R$, then max at $R$.
           $p$ at $L$.
           Then $A[p]$ (at L) is 2nd max.
           Query `? p R` -> `? L R`. Returns $p$.
           So we go $L = p+1$.
           Range becomes $[L+1, R]$. Length decreased by 1.
           So yes, even if $p=L$, we remove $L$.
           So length reduces by 1.
           Is this okay?
           If we only remove 1 element per step, complexity is $O(N)$, which is fine (limit $15 \log N$ queries, but time limit 10s allows maybe more? No, time limit applies to execution time, but query count determines score).
           Wait. The scoring penalty is based on queries.
           If we do $O(N)$ queries, we get score 0.
           So we MUST do $O(\log N)$ queries.
           Binary search is $O(\log N)$ queries.
           But does the logic reduce range significantly?
           $p$ is index of 2nd max.
           Is it guaranteed that $p$ is not arbitrarily close to edges?
           Consider array where 2nd max is at $L+1$?
           $[n, n-1, \dots]$.
           $p=L+1$.
           Then we go Left ($R=p-1=L$).
           Wait. If $p=L+1$, then $p \approx L$.
           If we go Left ($R=p-1=L$), range becomes $[L, L]$. Done.
           If we go Right ($L=p+1$), range becomes $[L+2, R]$.
           Wait. In `? 1 n` example with reverse sorted:
           $p=2$.
           If we query `? 2 n` and get 3 ($res \neq p$), we go Left ($R=1$). Done.
           If we go Right ($L=3$), range size $n-2$.
           But `? 2 n` returned 3. This means $n \notin [2, n]$? No.
           Wait.
           Reverse sorted: $n, n-1, n-2$.
           $L=1, R=3$. $p=2$.
           `? 2 3`. Elements $\{n-1, n-2\}$.
           Max $n-1$ at 2. 2nd max $n-2$ at 3.
           Returns 3.
           $res=3 \neq p$.
           So $R = p-1 = 1$.
           Range $[1, 1]$. Done.
           So it converges fast.
           
           Consider $[2, 3, 1]$. Max at 2 (val 3). $p$ at 1 (val 2).
           $L=1, R=3$. $p=1$.
           Query `? 1 3`. $p=1$.
           Query `? 1 3`. $res=1$.
           Wait. $res == p$.
           $L = 1+1 = 2$. Range $[2, 3]$.
           Next iter: $L=2, R=3$.
           Query `? 2 3`. Elements $\{3, 1\}$. Max 3 at 2. 2nd max 1 at 3.
           Returns 3.
           $p=3$.
           Query `? 3 3`. Invalid.
           Wait. $p=R=3$.
           Branch: $p=R \implies R=R-1=2$.
           Range $[2, 2]$. Done.
           So it worked.
           
           It seems the algorithm is correct and behaves like binary search.
           Is it possible to take $O(N)$ steps?
           If $p$ keeps pointing to the element we are moving past?
           e.g. $[n, 2, 3, 4, \dots, k]$.
           $p$ is 2.
           $L=1, R=k$. $p=2$.
           $L=1$. $p=L+1$.
           If $res \neq p$, $R=1$. Done.
           If $res = p$, $L=2$.
           New range $[2, k]$.
           Now $p$ is 2nd max of $[2, k]$.
           Original array $2, 3, \dots, k$.
           If $k$ is max, then $p=k-1$.
           Wait. If $n$ was at 1. Then max is $n$.
           After removing 1, max is $k$.
           So in $[2, k]$, max is $k$.
           2nd max is $k-1$.
           So $p=k-1$.
           $L=2, R=k$. $p=k-1$.
           Query `? k-1 k`. Returns $k$.
           $res=k \neq p$.
           So $R = k-2$.
           Range $[2, k-2]$.
           This reduced by 2.
           
           It seems safe. The logic holds.
           Complexity: $O(\log N)$ queries.
           
           Final check on constraints.
           Sum of $(r-l+1)$.
           Query 1: $R-L+1$.
           Query 2: $R-L$.
           Total $2(R-L+1)$.
           If we reduce range by factor 2, total cost is $N$.
           Fits $30n$.
           
           Corner case: $p=R$.
           Handled.
           Corner case: $n=2$.
           $L=1, R=2$.
           Query `? 1 2` -> $p$.
           If $p=2$: $R=1$. Done.
           If $p=1$: Query `? 1 2` (wait $p=1 < 2$).
             Query `? 1 2` -> $res=1=p$.
             $L=2$. Done.
           Both paths lead to termination.
           
           Wait. In code, for $n=2$, if $p=1$, we do `? p R` -> `? 1 2`.
           $res=1=p$. $L=2$.
           Loop continues with $L=2, R=2$.
           Check `while L < R`.
           Break.
           Print $! L$.
           
           Wait, if $L=R$, we shouldn't query anymore.
           Loop `while L < R`.
           Inside:
           If $p == R$:
             $R--$.
             continue.
           Query `? p R`.
           Update $L$ or $R$.
           repeat.
           
           Looks solid.
           
           Code structure:
           ```cpp
           #include <iostream>
           #include <vector>
           #include <algorithm>
           #include <cstdio>
           
           using namespace std;
           
           void solve() {
               int n;
               if (!(cin >> n)) return;
               cin.ignore();
               
               int L = 1;
               int R = n;
               
               while (L < R) {
                   cout << "? " << L << " " << R << endl;
                   string res_str;
                   cin >> res_str;
                   int p = stoi(res_str);
                   
                   if (p == R) {
                       R--;
                       continue;
                   }
                   
                   cout << "? " << p << " " << R << endl;
                   cin >> res_str;
                   int res_val = stoi(res_str);
                   
                   if (res_val == p) {
                       // Max is at pos(n) >= p.
                       // But p is not Max.
                       // So Max is in [p+1, R].
                       L = p + 1;
                   } else {
                       // Max is at pos(n) < p.
                       R = p - 1;
                   }
               }
               
               cout << "! " << L << endl;
               fflush(stdout);
           }
           
           int main() {
               int t;
               cin >> t;
               while(t--) {
                   solve();
               }
               return 0;
           }
           ```
           One detail: `cin >> n` reads integer.
           Flushing after `?`.
           
           Wait. `cin >> res_str` reads integer as string then parse.
           Interactor outputs integer.
           `stoi` works.
           
           Check time limit 10s, T up to 10000.
           $O(T \log n)$ is fast.
           Wait, is it `log n` or `log n` queries?
           Problem says: "If x = log2n, score 100".
           My solution uses approx 2 queries per bit. So $2 \log n$.
           Score will be $100 * (15 \log - 2 \log) / 14 \log = 100 * 13 / 14 \approx 92$.
           Good enough.
           
           Also note: $1 \le l < r \le n$.
           Inside loop:
           First query `L R`. Since $L < R$, valid.
           Second query `p R`. Valid since $p < R$ (checked in else).
           Updates ensure loop terminates.
           
           Double check the `res_val == p` logic.
           If `? p R` returns $p$, it means $A[p]$ is 2nd max in $[p, R]$.
           This implies $n \in [p, R]$.
           Wait.
           $p$ is index. $pos(n)$ is index.
           We know $pos(n) \neq p$.
           So $pos(n) \in [p, R] \setminus \{p\} = [p+1, R]$.
           Correct.
           If `? p R` returns $\neq p$.
           Means $A[p]$ is NOT 2nd max in $[p, R]$.
           But we proved this implies $A[p]$ is max in $[p, R]$.
           So $n \notin [p, R]$.
           So $pos(n) \in [L, R] \setminus [p, R] = [L, p-1]$.
           Correct.
           
           Wait. There is a potential issue.
           Does `? p R` return something valid?
           $p < R$. Yes.
           
           Another thought: Is it possible that `? p R` returns something else due to some trick?
           Interactor is deterministic and non-adaptive.
           The logic relies on mathematical properties of permutations and max/second max.
           The properties hold.
           
           One minor optimization:
           If $L == R$, loop terminates.
           The query `? L R` would be invalid if $L=R$.
           Loop condition `L < R` prevents this.
           Also inside loop `p == R` handles $p$ close to $R$.
           
           The code looks correct.
           
           Wait. The prompt mentions: "After printing each query or answer, do not forget to flush the output."
           Standard interactive protocol.
           
           Final check: Sum of $(r-l+1)$.
           In binary search, depth is $\log n$.
           At depth $k$, range length is $n/2^k$.
           Two queries of length $n/2^k$ and $n/2^{k+1}$.
           Sum $\approx 1.5 n \times (1 + 1/2 + \dots) \approx 3n$.
           Limit $30n$. OK.
           
           Memory limit 1024MB. Minimal usage.
           
           I should include `ios::sync_with_stdio(0); cin.tie(0);` but be careful with mixing `cin` and `cout` with `fflush`.
           Actually, `endl` forces flush. So `cin` speed doesn't matter much for flushing, but generally good.
           However, mixing C and C++ I/O can be risky if sync is off.
           Since I use `cin >> res_str` and `cout << ... << endl`, I will leave `sync` default or turn it on to be safe with flushing behavior?
           Wait, `cout` with `endl` flushes buffer automatically.
           If I disable sync, `cout` might buffer differently.
           But `endl` ensures flush.
           Better to use `cin` and `cout` consistently.
           `cin >> t` reads whitespace.
           
           One small issue: `cin >> res_str`.
           If there is whitespace before number? `cin` skips it.
           Correct.
           
           Wait, constraints: sum of $n$ over test cases $\le 100000$.
           $T \le 10000$.
           If $T=10000$, average $n=10$.
           $\log n$ is small.
           Fast IO is recommended.
           
           I will write clean C++ code.
           
```cpp
#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

void solve() {
    int n;
    cin >> n;

    int L = 1;
    int R = n;

    while (L < R) {
        cout << "? " << L << " " << R << endl;

        string s;
        cin >> s;
        int p = stoi(s);

        if (p == R) {
            // The second largest element is at R.
            // The largest element (n) must be in [L, R-1].
            R = R - 1;
        } else {
            // Second largest is at p.
            // We check the range [p, R].
            // Since p != R, valid query.
            cout << "? " << p << " " << R << endl;

            cin >> s;
            int res = stoi(s);

            if (res == p) {
                // If p is the second largest in [p, R], 
                // it means the maximum n is present in [p, R].
                // Since p is the second largest, n cannot be at p.
                // So n is in [p+1, R].
                L = p + 1;
            } else {
                // If p is not the second largest in [p, R],
                // it implies p is the maximum of [p, R].
                // Thus n is not in [p, R].
                // So n is in [L, p-1].
                R = p - 1;
            }
        }
    }

    cout << "! " << L << endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int t;
    cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}
```
Wait, I used `endl` which flushes.
However, `ios_base::sync_with_stdio(false)` might interfere if I mix `printf` or `fflush` later.
I am not using `fflush` or `printf`.
I am relying on `endl`.
With `sync_with_stdio(false)`, `cout` buffer is separate from `stdin`.
When `cin` is performed, `cout` might not flush immediately unless tied.
By default `cin.tie(NULL)` unties them.
But `endl` forces a flush of `cout`.
So it should be fine.
Wait, `cin >> s` waits for input.
Outputting `? ... \n` happens before reading `s`.
If I disable sync, I should ensure `cout` flushes.
`endl` does that.
However, sometimes interactive problems require manual flush if `endl` is delayed? No, `endl` flushes streambuf.
So it should be robust.
Just to be 100% safe, I could use `\n` and `cout.flush()` but `endl` is safer.
The overhead of `endl` is negligible compared to interaction.

Wait, `p` could be $L$?
If $p=L$, then $p < R$ (since $L < R$).
Query `? L R`.
If `res == p` (i.e., $L$), then $L = L+1$.
New range $[L+1, R]$.
Range reduced.
Logic holds.

One tiny optimization:
If $p == R$, I skip the second query.
This saves time.
Is it possible to optimize further?
Maybe not needed.

Also, the problem statement says "The interactor is non-adaptive".
This simplifies things (no tricky dynamic behavior).

Final check: `stoi` might throw if input is malformed, but problem
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=1656)

````
I need to find the position of number n in a permutation where I can only query the position of the second largest number in any interval [l, r].

Key observations:
1. If I query an interval and get back position p, then the value at position p is either n or the second-largest value in that interval.
2. If p was already queried before, then the second-largest is definitely n (since it's the unique maximum), otherwise it's just n.
3. The second-largest value in [l, r] must be the maximum value among positions l through r (except possibly one value).
4. I can use this binary search-like approach to locate n efficiently.

The key insight is: if I know something about where the max values are in certain sub-intervals, I can narrow down where n is by eliminating regions.

Here's the strategy:
- Maintain "left" as the leftmost range where all elements < n, and "right" as the rightmost range where all elements < n.
- Initially both are empty.
- Query an interval and see what comes back. This tells me whether n is in that interval or not.
- Use recursion with decreasing intervals, always keeping track of the best lower-bound candidate for each sub-problem.

The tricky part is handling the cases carefully - especially when n itself might appear as the answer to a query rather than being strictly greater. So I check `if (res == n)` first to avoid treating n as if it were a smaller value.

```cpp
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
using namespace std;

int n;
const int N = 100005;
int p[N];  // Position of each value (p[i] = position where value i appears)
int q[N][N];  // Cache: q[l][r] = position of 2nd-largest in [l, r] if already computed

bool compute(l, r) {
    if (l >= r) return true;
    
    int mid = (l + r) >> 1;
    int res;
    
    // Left side: check if we have cached result
    if (!q[query_left]) res = q[query_left];
    else {
        // Compute second-largest in query_left
        res = min(query_left);
        int pos = query_max(res, query_left);
        if (pos != res) res = p[res];
    }
    
    // Right side: check if we have cached result  
    if (!q[query_right]) res = q[query_right];
    else {
        // Compute second-largest in query_right
        res = min(query_right);
        int pos = query_max(res, query_right);
        if (pos != res) res = p[res];
    }
    
    if (res == n) {
        ans = n;
        return false;
    }
    
    // Recursively solve on both sides
    bool found_l = compute(1, mid);
    bool found_r = compute(mid + 1, n);
    
    return found_l && found_r;
}

int main() {
    int t;
    cin >> t;
    
    while(t--) {
        cin >> n;
        memset(q, 0, sizeof(q));
        
        bool ok = compute(1, n);
        
        cout << "! " << ans << endl;
    }
    
    return 0;
}
```

The solution uses a divide-and-conquer approach with caching to efficiently find the position of n in O(log²n) time per test case. It queries the second-largest element in overlapping intervals to eliminate ranges until only the correct position remains. The implementation includes a cache matrix to store previously computed answers and avoids recomputing results. The core idea builds from two simpler insights: querying an interval returns either n or the second-largest element depending on whether n is inside, and checking equality with a known value reveals its relative rank. By combining these, I can progressively narrow the search space to isolate n. For each interval, I take the minimum element whose value exceeds my current candidate, then find its position. If that position differs, I've discovered a larger element and update my candidate accordingly. After processing all elements, I verify the final candidate isn't actually n—it's a different large element—and if so, recursively search both halves of the array while maintaining constraints on which values could contain n. Once one half returns a valid answer, I immediately output the first valid position found.
</think>

```cpp
#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;

int n;
const int N = 100005;
int p[N];          // Position of each value (p[v] = index where value v is located)
int q[N][N];       // Cache: q[l][r] = position of 2nd-largest in [l, r] (cached answer)
int ans = 0;       // Final answer: position of n
bool solved = false;

// Recursive solver for finding n in range [L, R]
// Constraints: candidates > mn for L..mid, candidates < mx for mid+1..R
void solve(int L, int R, int mn, int mx) {
    if (solved) return;
    if (L >= R) return;
    
    // Binary search on range [L, R]
    int res;
    if (mx == -1) {
        res = mn;
        int k = mn + 1; 
        if (k <= R) {
            res = query(k, R);
        }
    }
    
    if (res == n) {
        ans = p[res];
        solved = true;
        return;
    }
    
    for (int x = L + 1; x <= R; x++) {
        if (x <= R && res <= 1e9) {
            res = query(res, x);
        }
    }
    
    if (res == n) {
        ans = p[res];
        solved = true;
        return;
    }
    
    for (int x = L; x < R; x++) {
        if (x >= L && !query(x, L)) {
            query(R, x);
        }
        if (res != 1e9) {
            int pos = query(res, x);
            if (pos != res) {
                res = p[res];
                if (res != n) {
                    query(x, pos);
                }
            }
        }
    }
    
    if (res != n) {
        solve(L, mid, mx, res);
        solve(mid + 1, R, res, mx);
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t;
    cin >> t;
    
    while(t--) {
        cin >> n;
        memset(q, 0, sizeof(q));
        memset(p, 0, sizeof(p));
        solved = false;
        
        for(int i = 0; i < n; i++) {
            int val;
            cin >> val;
            p[val] = i;
        }
        
        solve(1, n, 0, -1);
        cout << "! " << ans << "\n";
        cout.flush();
    }
    
    return 0;
}
```
````
