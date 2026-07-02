# 题 247 — base均分 5.0 vs 直接SFT均分 5.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=2,得分 25.0,completion_tokens=32768)

````
The problem asks us to transform sequence $A$ into sequence $B$ using a specific operation. The operation is: choose indices $(i, j)$ with $1 \le i < j \le N$, set $A_i = A_j - 1$, and $A_j = A_i + 1$ (before the update). Let's analyze this operation carefully.

Let the values before the operation be $a$ and $b$ at indices $i$ and $j$.
After the operation:
$A_i' = b - 1$
$A_j' = a + 1$

Wait, let me double check the operation description.
"Replace $A_i$ with $A_j - 1$ and $A_j$ with $A_i + 1$."
Usually, these are simultaneous updates based on the current state.
So if we have values $x = A_i$ and $y = A_j$.
New $A_i = y - 1$
New $A_j = x + 1$
Sum of new values: $(y - 1) + (x + 1) = x + y$.
The sum of elements in $A$ remains invariant under this operation.
Also, the difference $A_j - A_i$ changes from $y - x$ to $(x+1) - (y-1) = x - y + 2$.
Actually, notice that $A_i + A_j$ is constant. This means $\sum A_k$ is constant.
If $\sum A_k \neq \sum B_k$, it's impossible to reach $B$ from $A$.

Let's look closer at the constraint. $N \le 100$, $A_i, B_i \le 100$.
Since we need to find the minimum number of operations, this sounds like a shortest path problem on a graph where nodes are states of $A$. However, the state space is huge ($100^{100}$), so BFS is impossible directly.
We need a greedy strategy or a constructive approach.

Let's consider what the operation does locally. It takes 1 unit from index $i$ (if $A_i > A_j - 1$, effectively reducing $A_i$ relative to something? No.)
It reduces $A_i$ by 1 and increases $A_j$ by 1?
Let's re-read: "Replace $A_i$ with $A_j - 1$ and $A_j$ with $A_i + 1$".
Original: $A_i, A_j$.
New: $A_j - 1, A_i + 1$.
Wait. Is this correct?
If I have $A_i=5, A_j=2$.
New $A_i = 2-1 = 1$. New $A_j = 5+1 = 6$.
Change for $i$: $-4$. Change for $j$: $+4$.
This looks like we can transfer value between indices, but the transfer amount depends on the values themselves.
Specifically, $\Delta A_i = (A_j - 1) - A_i = A_j - A_i - 1$.
$\Delta A_j = (A_i + 1) - A_j = A_i - A_j + 1 = -(A_j - A_i - 1)$.
So indeed, $\Delta A_i + \Delta A_j = 0$. Sum is preserved.
However, the amount transferred is not arbitrary. It depends on the gap $A_j - A_i$. If $A_j > A_i$, say $A_j = A_i + k$, then $A_i$ becomes $A_i + k - 1$, $A_j$ becomes $A_i + 1$. Wait.
Let $x = A_i, y = A_j$.
$x_{new} = y - 1$
$y_{new} = x + 1$
If $y > x$, then $x_{new} = y - 1 \ge x$. So $A_i$ increases?
Example: $A_i=2, A_j=5$.
$x=2, y=5$.
$x_{new} = 4, y_{new} = 3$.
So $A_i$ goes $2 \to 4$ (+2), $A_j$ goes $5 \to 3$ (-2).
Wait, the formula says replace $A_i$ with $A_j - 1$.
If $A_j=5$, then new $A_i = 4$. Original was 2. Increase by 2.
New $A_j$ is replaced with original $A_i + 1 = 3$. Original was 5. Decrease by 2.
This operation transfers $2(A_j - A_i)/2$ ? No.
The change is $A_j - 1 - A_i$.
This seems like moving mass from $j$ to $i$ only if $A_j > A_i + 1$.
But the condition is just $1 \le i < j \le N$. We can pick any pair.
Is there an invariant other than sum?
Maybe min/max elements?
Consider the example 1:
$A = [2, 2, 1, 4]$
$B = [3, 2, 2, 2]$
Sum $A = 9$. Sum $B = 9$. Possible.
Op 1: 1 4. $i=1, j=4$.
$A_1 = 2, A_4 = 4$.
New $A_1 = 4 - 1 = 3$. New $A_4 = 2 + 1 = 3$.
$A$ becomes $[3, 2, 1, 3]$.
Op 2: 3 4. $i=3, j=4$.
Current $A_3 = 1, A_4 = 3$.
New $A_3 = 3 - 1 = 2$. New $A_4 = 1 + 1 = 2$.
$A$ becomes $[3, 2, 2, 2]$. Matches $B$.
Number of ops = 2.
Output matches Example 1.

Let's trace another one.
Suppose $A=[1, 10], B=[10, 1]$.
Op 1 2:
$A_1 = 10 - 1 = 9$. $A_2 = 1 + 1 = 2$.
$A = [9, 2]$.
Next op 1 2:
$A_1 = 2 - 1 = 1$. $A_2 = 9 + 1 = 10$. Back to start.
It seems oscillating.
Wait, in the first example $A_i=2, A_j=4$, became $4, 3$.
Order matters.
If $A_i > A_j$, say $5, 2$.
New $A_i = 2 - 1 = 1$. New $A_j = 5 + 1 = 6$.
Values become $1, 6$. The smaller element decreased significantly.
Actually, the transformation is:
$(x, y) \to (y-1, x+1)$.
Notice that $x+y$ is constant. Also $y-x$ changes.
Old diff: $y-x$.
New diff: $(x+1) - (y-1) = x - y + 2 = -(y-x) + 2$.
So $D' = -D + 2$.
This suggests that we can toggle the sign of the difference roughly by applying the operation.
If we want to move mass from $j$ to $i$ (decrease $A_j$, increase $A_i$), we need the new $A_i$ to be larger than old $A_i$.
New $A_i = A_j - 1$. Old $A_i = A_i$.
We need $A_j - 1 > A_i \iff A_j > A_i + 1$.
If $A_j > A_i + 1$, one operation swaps roles such that $A_i$ increases by $A_j - 1 - A_i$ and $A_j$ decreases by the same.
Essentially, if we pick indices where we want to move value from high to low, this operation is powerful but weird because it forces the updated values to be $y-1$ and $x+1$.
Wait, is this just sorting?
Or is it related to some property modulo something?
Note that if we apply the operation repeatedly on same pair $(i, j)$, we get:
Start $(x, y)$.
1. $(y-1, x+1)$. Diff changed from $y-x$ to $x-y+2$.
2. Apply again on $(i, j)$.
Input: $u=y-1, v=x+1$. Note $v = x+1, u=y-1$.
Wait, order in input is $i, j$. First $x$ was at $i$, second $y$ at $j$.
Operation on $(i, j)$: $A_i \leftarrow A_j - 1$, $A_j \leftarrow A_i + 1$.
Let's see sequence of diffs $d = A_j - A_i$.
Initially $d = y-x$.
After 1 op:
$A_i' = y-1, A_j' = x+1$.
$d' = A_j' - A_i' = x+1 - (y-1) = x - y + 2 = 2 - (y-x) = 2 - d$.
So $d_{k+1} = 2 - d_k$.
This recurrence $d_{k+1} = 2 - d_k$ has fixed point $d = 2 - d \implies 2d=2 \implies d=1$.
If we start with $d_0$, we go to $2-d_0$.
Then $2 - (2-d_0) = d_0$.
So it cycles with period 2!
$(x, y) \to (y-1, x+1) \to (x, y)$.
So simply applying the same operation twice brings us back to the original configuration.
This means we can't just pump values using one pair. We must involve other indices.
Wait, look at Example 1 again.
Initial: 2 2 1 4. Target: 3 2 2 2.
Ops: (1, 4), then (3, 4).
Indices involved: 1, 4 then 3, 4.
This looks like bubble sort moves but restricted.
In bubble sort adjacent swap $(i, i+1)$ exchanges values. Here we do something else.
However, observe the target state vs current state.
We want to match $A_k = B_k$.
The constraints $N \le 100$ are small enough for maybe some flow or matching, but usually these competitive programming problems about sequences with small $N$ might require simulation or constructive logic. But given "Minimum number of operations", it feels like BFS if state is small. But state isn't small.
Actually, is it possible that the number of operations is always bounded or follows a pattern?
Wait, look at the cycle property again. If we use pair $(i, j)$, the values just flip-flop around the mean. The mean is $\frac{A_i+A_j}{2}$.
$A_i' = \text{mean} - 0.5, A_j' = \text{mean} + 0.5$.
Wait, $A_i+A_j$ is integer. So sum is odd implies half-integers.
If sum is even, they average to integers.
If sum is odd, they become $k+0.5$ and $k+0.5$ shifted.
Wait, the values stay integers.
$A_i' = A_j - 1$. $A_j' = A_i + 1$.
Average of new values: $\frac{A_j - 1 + A_i + 1}{2} = \frac{A_i + A_j}{2}$. Same average.
If we use different pairs, we mix values.
This operation preserves the multiset of values modulo 2?
$A_i + A_j \equiv A_j + A_i \pmod 2$. Sum mod 2 is preserved.
Also $A_i \pmod 2$?
$A_i' = A_j - 1 \equiv A_j + 1 \pmod 2$.
$A_j' = A_i + 1 \equiv A_i + 1 \pmod 2$.
So the parity vector changes. specifically swapping parities of two elements and flipping them both?
No.
$p_i' = p_j + 1 \pmod 2$.
$p_j' = p_i + 1 \pmod 2$.
So $p_i' \oplus p_j' = (p_j+1) \oplus (p_i+1) = p_j \oplus p_i$.
So $p_i \oplus p_j$ is invariant?
Wait, if $p_i, p_j$ are bits.
If $p_i=0, p_j=0 \implies p_i'=1, p_j'=1$. Parity changed for both.
If $p_i=0, p_j=1 \implies p_i'=0, p_j'=0$. Parity changed for both.
If $p_i=1, p_j=1 \implies p_i'=0, p_j'=0$.
Basically $A_i \pmod 2$ flips to $\neg(A_j \pmod 2)$? No.
$A_j - 1$. If $A_j$ is even, $A_j-1$ is odd. If $A_j$ is odd, $A_j-1$ is even.
So $A_i \pmod 2$ becomes opposite of $A_j \pmod 2$.
And $A_j \pmod 2$ becomes opposite of $A_i \pmod 2$.
So the set of parities of the pair $\{i, j\}$ is flipped.
Wait, if $A=\{even, even\}$, after op $\{odd, odd\}$.
If $A=\{even, odd\}$, after op $\{odd, even\}$ (order swapped parity-wise).
If $A=\{odd, odd\}$, after op $\{even, even\}$.
So we can flip the parity pair status.
But wait, we need to reach exact values, not just parities.
Is it possible that we can achieve any configuration with the same sum?
Not necessarily. In Example 2, sum is not equal?
Ex 2:
A: 5 4 4 3 4 2 -> Sum = 22.
B: 5 1 2 3 4 1 -> Sum = 16.
Sums differ -> No.
Wait, Ex 2 output is No. Correct.
Check constraints again. $A_i, B_i \le 100$.
Is there a condition where sums are equal but still No?
Look at Ex 3.
A: 2 4 2 4 3 2 5 -> Sum = 22.
B: 5 4 3 2 5 1 2 -> Sum = 22.
Output Yes.
So sum equality is necessary. Is it sufficient?
In many similar problems ("can we rearrange"), sum equality is often the key if moves preserve sum and allow full mixing. Here moves are restrictive.
But since we can permute indices freely (pick any $i<j$), maybe it behaves like a permutation network?
Actually, notice the transformation $(x, y) \to (y-1, x+1)$.
This can be rewritten as adding $-(y-x-1)$ to $x$ and $+(y-x-1)$ to $y$. No.
It is basically $x_{new} = y-1, y_{new} = x+1$.
This is equivalent to sorting the pair $(x,y)$ in reverse order?
No, if $y > x$, then $y-1 \ge x$ and $x+1 < y$.
If $x=2, y=5 \to 4, 3$. Sorted? Yes $3, 4$.
If $x=5, y=2 \to 1, 6$. Not sorted.
Wait, if $x, y$ are close?
If $|x-y|=1$, say $x=k, y=k+1$.
$x \to k+1-1=k$. $y \to k+1$.
Back to $x, y$. It loops immediately.
So pairs with difference 1 are stable. They absorb no change.
Pairs with larger difference can shift values.
This problem looks very much like sorting with a limited comparator.
However, since $N$ is small (100), maybe we can solve it greedily?
Target $B$. Current $A$.
For each position $k$, if $A_k \neq B_k$, we need to change $A_k$.
We can bring values from other positions to $k$.
Since sum is conserved, $\sum A_k = \sum B_k$ is definitely required.
What about individual constraints?
Maybe we can treat this as matching $A$ values to $B$ values? No, positions matter.
Actually, looking at the scoring rule: "score ... based on number of operations m". We need minimum operations.
Given the limits, maybe we can just run a BFS on the relevant differences? Or greedy simulation?
Wait, $N=100$ allows $O(N^3)$ or $O(N^4)$.
Can we formulate this as a flow problem? No, we need sequence of operations.
Let's reconsider the operation structure.
$A_i \leftarrow A_j - 1, A_j \leftarrow A_i + 1$.
This looks like we are exchanging $A_i$ and $A_j$, but modifying them.
Actually, $A_i$ gets $A_j - 1$ and $A_j$ gets $A_i + 1$.
Note that $A_i$ decreases by 1 iff $A_j = A_i$ (impossible since distinct?) No.
Wait, if $A_i$ decreases, $A_j$ increases. If $A_i$ increases, $A_j$ decreases.
The condition for $A_i$ to increase is $A_j - 1 > A_i \iff A_j > A_i + 1$.
If $A_j > A_i$, $A_i$ increases.
If $A_j < A_i$, $A_i$ decreases.
If $A_j = A_i$, $A_i \to A_i - 1$, $A_j \to A_j + 1$. Values diverge.
Basically, we can take from $j$ and give to $i$ or vice versa?
If $A_j > A_i$, we can move value from $j$ to $i$.
If $A_j < A_i$, we can move value from $i$ to $j$ (since $A_j$ will decrease).
Wait, if $A_j < A_i$, say $A_i=5, A_j=2$.
Op on $(i, j)$ (indices): $A_i \leftarrow 2-1=1$ (dec), $A_j \leftarrow 5+1=6$ (inc).
So if $A_i > A_j$, $A_i$ decr, $A_j$ incr.
So the operation moves 1 unit of "mass" from the larger to the smaller? No.
From $5, 2$ we get $1, 6$. The larger became larger? No $5 \to 1$.
The gap changed from $5-2=3$ to $6-1=5$. Gap increased.
Wait, my analysis of "moving mass" was wrong.
Let's look at the quantity $Q = \sum |A_k - B_k|$. We want to make it 0.
Each operation changes the array.
Does it reduce the distance to $B$?
Maybe not always.
However, note that the problem asks for *minimum* operations.
Since $N$ is small, maybe we can model the state of the system more simply.
Is there any property that restricts valid configurations?
Actually, the constraints on $A_i, B_i$ are small. Max value 100.
But values change during operations.
However, notice that $A_i$ and $A_j$ are exchanged with a shift.
Is it possible that the order of elements doesn't matter?
No, positions are distinct.
Wait, there's a simpler observation.
Look at the sum $\sum A_i$. Constant.
Look at $\sum A_i \cdot i$.
Before: $S = i A_i + j A_j$.
After: $S' = i(A_j - 1) + j(A_i + 1) = i A_j - i + j A_i + j$.
Change $\Delta S = i A_j - i + j A_i + j - i A_i - j A_j = (j-i) A_i + (i-j) A_j + (j-i)$.
$= (j-i)(A_i + 1 - A_j)$.
So $\sum k A_k \pmod{j-i}$ is invariant? Or something?
Not a strong invariant.

Let's step back. Is this solvable by simple simulation?
Constraints: Time 2s, Memory 512MB.
Greedy approach: At each step, pick an operation that reduces $\sum |A_k - B_k|$ by maximum amount? Or fixes one mismatch?
With $N=100$, maybe just fixing mismatches works.
However, changing $A_i$ might mess up $A_i=B_i$ status.
But if we fix $A_i=B_i$ later, it must hold.
Actually, this looks like finding a path in a grid or graph.
Wait, consider the values modulo something.
The operation changes values by $A_j-1-A_i$.
Let's check if we can always satisfy conditions if sums match.
Wait, is there any case where sum matches but impossible?
The constraints $A_i, B_i \le 100$ are quite small.
Maybe the intended solution is actually to simulate a process that sorts $A$ into $B$?
Wait, if $A$ and $B$ are permutations of each other (ignoring shifts?), it would be easier. But here values change magnitude.
Let's check the provided examples again.
Ex 1: $A=[2, 2, 1, 4], B=[3, 2, 2, 2]$.
Diffs: $A-B = [-1, 0, -1, 2]$.
We need to add $1$ to index 1, $1$ to index 3, subtract $2$ from index 4.
Ops used: $(1, 4)$ then $(3, 4)$.
Effect of $(1, 4)$: $A_1 \gets A_4-1$, $A_4 \gets A_1+1$.
Before: $A_1=2, A_4=4$. After: $A_1=3, A_4=3$.
Diffs updated: $A_1 \to 3$ (was $2 \to +1$), $A_4 \to 3$ (was $4 \to -1$).
Remaining Diffs needed: Index 3 needs $+1$, Index 4 needs $-1$.
Effect of $(3, 4)$: $A_3 \gets A_4-1$, $A_4 \gets A_3+1$.
Before: $A_3=1, A_4=3$. After: $A_3=2, A_4=2$.
Diffs updated: $A_3 \to 2$ (was $1 \to +1$), $A_4 \to 2$ (was $3 \to -1$).
Total changes: $A_1: +1$, $A_3: +1$, $A_4: -2$.
This matches requirements exactly.
Notice that the operations were chosen such that $A_i < A_j$ in the pair?
$(1, 4): 2 < 4$. $A_1$ increases. $A_4$ decreases.
$(3, 4): 1 < 3$. $A_3$ increases. $A_4$ decreases.
So we moved value FROM $4$ TO $1$ and FROM $4$ TO $3$.
Since index 4 had excess (+2 relative to B), and 1, 3 had deficit.
The operation $(u, v)$ where $A_u$ should increase and $A_v$ should decrease requires $A_u < A_v$.
In general, if we want to transfer 1 unit from $v$ to $u$, we need $A_u$ to increase and $A_v$ to decrease.
This happens if $A_v > A_u + 1$.
If $A_v = A_u + 1$, then $A_v \to A_u$, $A_u \to A_v-1=A_u$, no wait.
If $A_v = k, A_u = k-1$.
Op: $A_u \leftarrow k-1$, $A_v \leftarrow k-1+1 = k$.
State unchanged!
So we strictly need $A_v > A_u + 1$ to make progress towards transferring mass.
Wait, in Ex 1, initial $A_4=4, A_1=2$. Diff is 2. $4 > 2+1$. Progress made.
Initial $A_4=3, A_3=1$. Diff is 2. $3 > 1+1$. Progress made.
So the condition $A_v > A_u + 1$ is required to change anything.
What if $A_v = A_u$? Then $A_u \to A_v-1 = A_u-1$, $A_v \to A_u+1 = A_u+1$.
Gap increases. Mass moves $A_v \to A_u$? No, $A_v$ increases, $A_u$ decreases.
Wait, if $A_v > A_u$ is false, i.e., $A_v \le A_u$.
Case 1: $A_v = A_u$. Swap values (shifted). $A_u \downarrow, A_v \uparrow$. Mass flows $v \to u$? No $u$ decreases.
Mass flows from $u$ to $v$.
But we wanted $A_u$ to increase?
If we wanted to increase $A_u$ and decrease $A_v$, we must perform op on $(u, v)$ assuming $A_u$ ends up at $A_v-1$ and $A_v$ at $A_u+1$.
Wait, the operation is defined as picking $i, j$. $A_i$ becomes $A_j-1$.
So if we want $A_i$ to increase, we need $A_j - 1 > A_i$.
So $A_j > A_i + 1$.
If $A_i$ needs to decrease, we could pick $(j, i)$? No, indices in operation are ordered $i < j$.
We must pick valid indices $i < j$.
If we want to swap roles, say we have index $u$ and $v$ with $u < v$.
We can apply $(u, v)$ to set $A_u \to A_v-1, A_v \to A_u+1$.
Or we can't apply $(v, u)$.
So direction is fixed by indices. We can only push from $v$ to $u$ if we want to decrease $v$ and increase $u$.
Wait, can we ever increase $A_u$ and decrease $A_v$ if $A_v < A_u + 1$?
Only if we use an intermediate node or multiple steps.
But wait, in Example 1, we used $4 \to 1$ and $4 \to 3$.
Indices: $1, 3, 4$.
Here $1<4$ and $3<4$.
So we treated 4 as source.
Is it possible to have multiple sources?
If $A_k$ has large surplus, it can feed multiple deficits.
Strategy:
Find an index $i$ with $A_i > B_i$ (surplus).
Find an index $j$ with $A_j < B_j$ (deficit).
We want to move mass from $i$ to $j$ (make $A_i$ smaller, $A_j$ larger).
The operation allows us to modify $A_x$ and $A_y$ for $x < y$.
To affect $i$ and $j$, we must pair them. Since indices are distinct, say $x, y \in \{i, j\}$, we can choose pair $(min(i, j), max(i, j))$.
Suppose $i < j$. We operate on $(i, j)$.
$A_i \leftarrow A_j - 1$.
$A_j \leftarrow A_i + 1$.
Wait, if we want to move mass FROM $i$ TO $j$ (i.e. decrease $A_i$, increase $A_j$), we need $A_j$ to increase.
$A_j$ becomes $A_i + 1$. So $A_j$ increases by 1.
$A_i$ becomes $A_j - 1$. So $A_i$ decreases by 1.
So this operation ALWAYS increases $A_{max\_idx}$ by 1 and decreases $A_{min\_idx}$ by 1?
Let's check.
Pair $(i, j)$ with $i < j$.
New $A_i = A_j - 1$.
New $A_j = A_i + 1$.
Change:
$\Delta A_i = (A_j - 1) - A_i = A_j - A_i - 1$.
$\Delta A_j = (A_i + 1) - A_j = A_i - A_j + 1 = -(A_j - A_i - 1)$.
So $A_i$ changes by $A_j - A_i - 1$.
$A_j$ changes by $-(A_j - A_i - 1)$.
Wait, if $A_j > A_i + 1$, then $\Delta A_i > 0$. $A_i$ INCREASES, $A_j$ DECREASES.
This moves mass from $j$ (larger index) to $i$ (smaller index).
If $A_j = A_i + 1$, then $\Delta A_i = 0$. No change.
If $A_j \le A_i$, then $\Delta A_i < 0$. $A_i$ DECREASES, $A_j$ INCREASES.
Wait, if $A_j \le A_i$, then $\Delta A_i \le -1$. $A_i$ decreases. $A_j$ increases.
This moves mass from $i$ (smaller index) to $j$ (larger index).
Wait, I said earlier if $A_j > A_i+1$, $A_i$ increases.
Yes.
So:
1. If $A_j > A_i + 1$, we transfer mass from $j$ to $i$ (amount $A_j - A_i - 1$).
2. If $A_j \le A_i$, we transfer mass from $i$ to $j$ (amount $A_i - A_j + 1$).
Wait, in case 2, amount is $A_i - A_j + 1$.
Note: In case 2, $A_j$ increases.
Wait, does it increase by exactly 1? No.
Example: $A_i=5, A_j=2$. $A_j \le A_i$ holds.
$\Delta A_i = 2 - 5 - 1 = -4$.
$\Delta A_j = 5 - 2 + 1 = 4$.
So $A_j$ increases by 4.
The "unit" concept I thought of initially was based on "move 1 unit". But this move amounts to shifting the whole block of values.
Actually, the operation essentially performs a cyclic shift or something?
Notice $A_i \to A_j - 1$.
This looks like the values are getting pushed towards the boundaries.
Let's analyze the sum of squared errors $\sum (A_k - B_k)^2$. Maybe that helps.
Actually, if we look at the constraints again, $A_i$ can grow arbitrarily large? No, usually problems imply bounds, but with $N=100$ and time 2s, maybe growth is constrained.
Wait, if we keep swapping, values can blow up. But we want minimum operations.
Is there a bound on operations?
Actually, let's look at the constraints on the solution again.
Score calculation implies minimizing $M$.
Maybe there is a deterministic algorithm to find $M$.
Observation:
The operation on $(i, j)$ affects ONLY $A_i$ and $A_j$.
Also notice that $A_j - A_i$ changes by $-2(A_j - A_i - 1)$? No.
Old diff $d = A_j - A_i$.
New diff $d' = A_j' - A_i' = (A_i + 1) - (A_j - 1) = A_i - A_j + 2 = -(A_j - A_i) + 2 = 2 - d$.
So $d_{new} = 2 - d_{old}$.
This means $A_j' - A_i'$ is determined solely by previous difference.
This implies the relation between $A_i$ and $A_j$ is purely a function of their previous relative difference.
Wait, this is very restrictive. If we want to set $A_j - A_i = K$, we can do it in 1 step if we find the right initial $d$.
If $2 - d = K \implies d = 2 - K$.
So if we have difference $d$, one op makes it $2-d$.
Two ops makes it $d$.
So applying the same op twice restores the pair.
So we cannot continuously adjust the difference using the SAME pair.
Therefore, we MUST use different pairs to transport values.
This strongly suggests that values flow through a network of indices.
Think of $A_i$ as capacity of node $i$.
We have edges $(i, j)$ for all $i < j$.
But the edge weights are dynamic.
However, note that the operation $d \to 2-d$ is very symmetric.
Is it possible to think of $A_i$ as something like "number of items"?
Wait, if $N$ is small, maybe we can view this as a flow problem where we move items.
Let's try to map the operation to a standard problem.
If we ignore the condition $d \to 2-d$ and just assume we can change $A_i$ by $x$ and $A_j$ by $-x$, this is trivial.
But we can't.
Actually, let's re-examine $A_j \to A_i + 1$.
If we apply $(i, j)$, we effectively set $A_i = A_j - 1$ and $A_j = A_i + 1$.
If we consider the total sum of $A$, it is constant.
Wait, the constraints $A_i, B_i \le 100$ is small.
Is it possible that we can perform the operations in a specific order to solve it?
Consider the target state $B$.
Can we construct the solution backwards? From $B$ to $A$?
Reverse operation:
Choose $i, j$ ($i<j$).
State $A'$ comes from $A$.
$A'_i = A_j - 1$. $A'_j = A_i + 1$.
We want to recover $A$.
Given $A', B'$...
Reverse op:
If we did op $(i, j)$, then $A_i = A'_j - 1$ and $A_j = A'_i + 1$.
Wait, forward: $A_i^{new} = A_j^{old} - 1$. $A_j^{new} = A_i^{old} + 1$.
Inverse: Given $A^{new}$, find $A^{old}$.
$A_j^{old} = A_i^{new} + 1$.
$A_i^{old} = A_j^{new} - 1$.
So the inverse operation is exactly the same form!
If we reverse time, the operation is:
Pick $i < j$, set $A_i \leftarrow A_j + 1, A_j \leftarrow A_i - 1$ (using old values? No, derived ones).
Wait.
Forward: $A_i \gets A_j - 1, A_j \gets A_i + 1$.
Inverse: $A_i^{old} = A_j^{new} - 1, A_j^{old} = A_i^{new} + 1$.
This is NOT symmetric in terms of values unless we track which came from where.
Wait, if we are at state $B$, and we want to reach $A$ in minimum steps (same as $A$ to $B$),
applying the inverse of the forward op transforms $B$ back to something.
The inverse op looks like: $X_i \leftarrow Y_j + 1, X_j \leftarrow Y_i - 1$.
Let's check.
Forward: $A_i \to A_j-1, A_j \to A_i+1$.
Let's denote state at $t$ as $S_t$.
$S_{t+1}[i] = S_t[j] - 1$
$S_{t+1}[j] = S_t[i] + 1$
So $S_t[i] = S_{t+1}[j] - 1$ and $S_t[j] = S_{t+1}[i] + 1$.
So if we go backwards from $B$ to $A$, we perform the SAME formula but on different variables?
Wait, $S_{t+1}[i] = S_t[j] - 1$.
If we know $S_{t+1}$, we compute candidates for $S_t[i]$ and $S_t[j]$.
$S_t[j] = S_{t+1}[i] + 1$.
$S_t[i] = S_{t+1}[j] - 1$.
This is EXACTLY the same operation!
Let's verify.
Suppose we apply op $(i, j)$ forward.
$A_{new} = [x, y] \to [y-1, x+1]$.
Now apply op $(i, j)$ backward? Or forward on the new state?
If we apply the *same* operation logic (replace $i$ with $j-1$ etc) on $[y-1, x+1]$, we get:
$i \leftarrow (x+1) - 1 = x$.
$j \leftarrow (y-1) + 1 = y$.
So YES! Applying the operation $(i, j)$ on the resulting state restores the original state.
This means the operation is its own inverse!
$Inv(f(x, y)) = f(y-1, x+1)$? No.
$f(x, y) = (y-1, x+1)$.
$f(y-1, x+1) = ((x+1)-1, (y-1)+1) = (x, y)$.
Correct.
So the graph is undirected. The operation is involutory.
So finding min steps from $A$ to $B$ is equivalent to finding min steps from $B$ to $A$.
Also, this confirms that we can just move around.
The crucial question: Is the graph connected?
If yes, and sum is matched, is it always reachable?
The problem statement asks to determine *whether* it is achievable.
If the sum of $A$ equals sum of $B$, is it guaranteed to be Yes?
The example 2 failed on sum.
Is sum equality sufficient?
There might be other invariants.
Let's check parity.
$A_i' = A_j - 1$.
$A_j' = A_i + 1$.
$A_i' \equiv A_j + 1 \pmod 2$.
$A_j' \equiv A_i + 1 \pmod 2$.
Sum $A_k \pmod 2$?
$\sum A_k' = \sum (A_k) + \Delta A_i + \Delta A_j = \sum A_k$.
So total sum mod 2 is preserved.
Parity of sum is preserved. (Already implied by sum equality).
Individual parities?
$P_i' = P_j + 1$.
$P_j' = P_i + 1$.
So $P_i' + P_j' = P_i + P_j + 2 \equiv P_i + P_j \pmod 2$.
The sum of parities is preserved.
Also $P_i' \equiv P_j' + P_j + P_i \dots$?
Basically, if we swap two bits in a string, their XOR sum is invariant.
Here $P_i' = P_j + 1, P_j' = P_i + 1$.
So $P_i' \oplus P_j' = (P_j+1) \oplus (P_i+1) = P_j \oplus P_i$.
So for every pair involved, the XOR sum of their parities is preserved.
This means the "parity profile" can change somewhat.
Actually, $P_i \to \neg P_j, P_j \to \neg P_i$.
So we can replace a pair of bits $(p_i, p_j)$ with $(\neg p_j, \neg p_i)$.
If $p_i=p_j=0 \to 1, 1$.
If $p_i=0, p_j=1 \to 0, 0$.
If $p_i=1, p_j=1 \to 0, 0$.
So we can convert $(0, 0) \leftrightarrow (1, 1)$. And $(0, 1) \leftrightarrow (0, 0)$.
Wait, $(0, 1) \to (0, 0)$? No.
$p_i=0, p_j=1 \implies p_i' = 1+1=0, p_j' = 0+1=1$.
Wait.
If $p_i=0, p_j=1$.
$A_i$ even, $A_j$ odd.
New $A_i = A_j - 1$ (odd - 1 = even). So $p_i' = 0$.
New $A_j = A_i + 1$ (even + 1 = odd). So $p_j' = 1$.
So $(0, 1) \to (0, 1)$.
Wait, parity doesn't change if inputs have different parity!
If $p_i=0, p_j=1$: $A_j$ is odd. $A_j-1$ is even ($p_i'=0$).
$A_i$ is even. $A_i+1$ is odd ($p_j'=1$).
So if parities differ, they remain same.
If parities same:
$(0, 0) \to (1, 1)$.
$(1, 1) \to (0, 0)$.
So we can flip the pair parity IF parities are equal.
This implies we can't arbitrarily change parities. We can only switch types.
But we don't need to fix parities arbitrarily. We need to match $B$.
Is it true that if $\sum A = \sum B$, we can solve it?
Actually, the problem asks "Determine whether the goal is achievable".
It hints that sometimes it is No.
Is there any other invariant?
Consider $A \pmod 1$? No.
Maybe connectivity components.
With $N \le 100$, maybe we can search? No, state space too big.
But maybe the "No" cases are rare or specific.
Is it related to graph connectivity?
If we have just one pair $(i, j)$, we can only toggle parities if $A_i \equiv A_j \pmod 2$.
Wait, this logic holds for ANY $i, j$.
So if we have at least one pair with $A_i \equiv A_j \pmod 2$, we can flip parities.
If ALL pairs have $A_i \not\equiv A_j \pmod 2$, then all $A_k$ must be alternating $0, 1, 0, 1$? No.
If $N$ elements, and for all pairs parities differ?
Impossible for $N \ge 3$.
For $N=2$, $(0, 1)$ or $(1, 0)$.
If parities are all distinct (alternating), we can't change them?
Wait, if $A = (2, 3)$, sum=5.
Apply op: $(3-1, 2+1) = (2, 3)$. Fixed point.
So $(2, 3)$ is a sink. Can't reach $(3, 2)$? Sum(3, 2)=5.
Wait, sum is invariant. $(3, 2)$ sum is 5.
Can we reach $(3, 2)$ from $(2, 3)$?
$A=(2, 3) \to (2, 3)$.
So no.
So we need to be able to change configuration.
If we have $A=(2, 2)$, sum=4.
Op $\to (1, 3)$. Sum=4.
From $(1, 3)$ Op $\to (2, 2)$.
From $(2, 2)$ can we reach $(3, 1)$?
$(2, 2) \to (1, 3)$.
$(1, 3) \to (3, 1)$?
Check $(1, 3)$: $1, 3$. $i=1, j=2$. $A_1=1, A_2=3$.
$A_1' = 3-1 = 2$. $A_2' = 1+1 = 2$.
So $(1, 3) \to (2, 2)$.
So for $N=2$, we have a component $\{(2, 2), (1, 3)\}$ and $\{(3, 2)\}$?
Wait, is $(3, 2)$ isolated?
$A=(3, 2)$. Op $\to (2-1, 3+1) = (1, 4)$. Sum=5.
Ah, sum must be equal.
For sum=5:
States reachable from $(2, 3)$:
$(2, 3) \to (2, 3)$. (Fixed).
Wait, is $(2, 3)$ really fixed?
$A_1=2, A_2=3$.
$A_1' = 3-1 = 2$. $A_2' = 2+1 = 3$. Yes.
So $(2, 3)$ is unreachable from $(3, 2)$?
$(3, 2) \to (1, 4) \to (3, 2)$. Cycle length 2.
$(2, 3)$ is a cycle of length 1.
Are there other states? $(1, 4) \to (3, 2)$.
So for sum=5, states are $\{(2, 3)\}$ and $\{(3, 2), (1, 4)\}$.
So for $N=2$, it's possible that we CANNOT reach $B$ from $A$ even if sums match.
Condition for impossibility in $N=2$:
If $A_1 < A_2$ and $A_1, A_2$ have different parity? No.
Let's check $(1, 4)$. Parity $1, 0$. Diff=3.
$(3, 2)$. Parity $1, 0$. Diff=-1.
$(2, 3)$. Parity $0, 1$. Diff=1.
It seems that parity of difference or something is invariant?
Actually, notice that for $A=(x, y)$, $x+y = S$.
We found $x \to S-y-1$, $y \to S-x+1$ ? No.
Formula: $x \to y-1, y \to x+1$.
If $x+y=S$, then $y = S-x$.
$x' = S-x-1$. $y' = x+1$.
So $x' + y' = S$.
Also $x' = S-1-x$.
So the next value of $x$ is linear function of current $x$.
$x_{k+1} = (S-1) - x_k$.
Then $x_{k+2} = (S-1) - ((S-1) - x_k) = x_k$.
So it alternates $x, S-1-x, x, \dots$
So $x$ never reaches any value except $x$ or $S-1-x$.
Thus for $N=2$, $A$ can only reach states where $A_1 \in \{x, S-1-x\}$.
If $B_1 = B$, we need $B_1 \in \{A_1, S-1-A_1\}$.
If $S$ is fixed.
So for $N=2$, if $B_1 \neq A_1$ AND $B_1 \neq S - 1 - A_1$, then impossible.
Wait, in $(2, 3)$ case, $S=5$. $A_1=2$.
$S-1-A_1 = 5-1-2 = 2$.
So only reachable $A_1$ is 2.
$B=(3, 2)$ has $B_1=3$.
So impossible.
Wait, my example earlier: $(1, 4) \to (3, 2)$.
Here $S=5$. Start $(1, 4)$. $A_1=1$.
$S-1-1 = 3$.
So reachable $A_1 \in \{1, 3\}$.
$B=(3, 2)$ has $A_1=3$. So reachable.
So for $N=2$, condition is $A_1 + B_1 = S-1$ OR $A_1 = B_1$.
Which simplifies to $A_1 + B_1 = S-1$ or $A_1 = B_1$?
No, if $A_1=B_1$ we are done.
So generally, $A_1$ must map to $B_1$ via the orbit.
Orbit is size 2: $\{A_1, S-1-A_1\}$.
Wait, this is extremely specific to $N=2$. For $N>2$, we can mix elements.
With $N \ge 3$, the state space is much larger.
But wait, the operation for $N \ge 3$ involves picking ANY pair.
Wait, is it possible to solve it greedily?
Actually, for $N \ge 3$, maybe the "No" case only happens if sums differ?
Or is there a specific modular invariant?
Let's check if there is any case with sum equal but No for $N \ge 3$.
The problem constraints say $N \le 100$.
Usually if it was "always yes", the problem wouldn't ask to determine.
Wait, let's re-read carefully.
"Determine whether the goal is achievable."
This suggests it's not always achievable.
However, finding the counter-example might be hard.
But implementing a solution that handles it requires either a heuristic or checking conditions.
Wait, there is a known result for this problem?
The operation $x \to y-1, y \to x+1$ is called "swap and shift"?
Actually, let's look at the constraints again.
If we sort $A$ and sort $B$, do they have to match?
No, positions matter.
Wait, let's write a brute force BFS solver for very small $N$ and states to guess the condition.
Wait, I am an AI text model, I cannot run code.
However, I can deduce from properties.
Let's look at the property $A_i \pmod 2$.
In $N=2$, parity difference determines reachability?
$(2, 3) \to 0, 1$.
$(1, 4) \to 1, 0$.
$(3, 2) \to 1, 0$.
Notice that $A_1 \pmod 2$ is invariant if $N=2$ and $A_1 \equiv A_2 \pmod 2$ is FALSE?
In $(2, 3)$, $2 \equiv 0, 3 \equiv 1$.
Op doesn't change parities (as analyzed before).
In $(1, 4)$, $1 \equiv 1, 4 \equiv 0$.
Op doesn't change parities.
So if parities are mixed, they can't be changed.
So for $N=2$, if parities differ, we can't change $A_1$'s parity.
If $A_1$ and $B_1$ have different parity, impossible?
Wait, $A_1=2 (0)$, $B_1=3 (1)$. Impossible.
$A_1=1 (1)$, $B_1=3 (1)$. Possible.
So for $N=2$, we need $A_1 \equiv B_1 \pmod 2$?
Wait, if parities are different ($0, 1$), then neither can change parity.
So $A_1 \pmod 2$ must equal $B_1 \pmod 2$.
Is this true for $N > 2$?
If $A$ has $N$ elements.
We can pick $i, j$. If $A_i \equiv A_j \pmod 2$, then $A_i$ flips parity (to $A_j-1 \equiv A_j+1$), $A_j$ flips parity.
Wait, $A_j-1 \equiv A_j + 1 \pmod 2$.
Yes, $-1 \equiv 1 \pmod 2$.
So if we pick a pair with SAME parity, BOTH flip parity.
If we pick a pair with DIFFERENT parity, NEITHER flips parity.
So the *number* of odd elements modulo 2 is invariant?
Wait.
Same parity pair (E, E): $\to$ (O, O). Number of odds changes by +2.
Same parity pair (O, O): $\to$ (E, E). Number of odds changes by -2.
Different parity pair (E, O): $\to$ (E, O). Number of odds changes by 0.
(Diff parity pair (O, E) -> same).
So parity count changes by multiples of 2.
This implies the parity of the number of odd elements is invariant.
Let $K = \sum (A_i \pmod 2)$.
Then $K \pmod 2$ must be equal for $A$ and $B$.
This is equivalent to $\sum A_i \equiv \sum B_i \pmod 2$.
Since sum equality implies $\sum A_i = \sum B_i$, the parity sum condition is automatically satisfied if sums are equal.
So parity invariant doesn't give extra restriction beyond sum.

Wait, looking at $N=2$ again.
Sum equal $\implies A_1 + A_2 = B_1 + B_2$.
For $(2, 3)$ vs $(3, 2)$, sums are 5.
Parity counts: $(0, 1)$ has 1 odd. $(1, 0)$ has 1 odd.
They match.
So why is $(2, 3) \to (3, 2)$ impossible?
Because the parities are FIXED when they are different.
Wait, my analysis of "different parity pair -> neither flips parity" meant $A_i \pmod 2$ is preserved?
Let's check.
$A_i$ becomes $A_j - 1$.
If $A_i$ is even, $A_j$ is odd.
$A_j - 1$ is even.
So $A_i$ stays even.
If $A_i$ is odd, $A_j$ is even.
$A_j - 1$ is odd.
So $A_i$ stays odd.
So if we have distinct parities, parities are LOCKED.
If we have equal parities, we can flip them.
Wait, for $(2, 3)$, parities are $(0, 1)$. Distinct. Locked.
We cannot change $A_1$ from 0 to 1.
So $A_1 \pmod 2$ is invariant if parities are different.
Wait, but if we can't change parities, then we can't change values across parity classes?
For $N=2$, $A_1$ can only stay even or odd.
If $A_1$ is even, $B_1$ must be even.
In $(2, 3)$, $A_1=2$ (even). $B_1=3$ (odd). Mismatch $\to$ Impossible.
So for $N=2$, we need $A_1 \equiv B_1 \pmod 2$?
Wait, is this true?
Suppose $A=(2, 4)$, $B=(4, 2)$. Sum=6.
$A$: (0, 0). $B$: (0, 0).
Can we reach $B$?
$A \to (3, 3)$. (Flip both to odd).
$A \to (3, 3)$ (since $A_i=A_j$, op gives $A_j-1, A_i+1 \implies 3, 3$).
From $(3, 3)$, op gives $(2, 4)$.
So $(2, 4) \leftrightarrow (3, 3)$.
Can we reach $(4, 2)$?
$(3, 3) \to (2, 4)$ or $(4, 2)$?
From $(3, 3)$, $i=1, j=2$. $A_1=3, A_2=3$.
$A_1' = 3-1 = 2$. $A_2' = 3+1 = 4$. Result $(2, 4)$.
So NO. $(3, 3)$ leads back to $(2, 4)$.
Wait, $N=2$ graph structure.
States with sum $S$:
If $S$ is even: $x + (S-x) = S$.
If we apply op: $x \to S-x-1$.
So $x$ goes to $S-1-x$.
This is a toggle.
So for $N=2$, we have components $\{x, S-1-x\}$.
For $A=(2, 4)$ ($x=2, S=6$). Component $\{2, 3\}$. (Wait $S-1-2 = 3$).
Wait, state $(2, 4)$ corresponds to $x=2$. Next is 3.
So reachable $x \in \{2, 3\}$.
$B=(4, 2)$ corresponds to $x=4$.
$4 \notin \{2, 3\}$.
So impossible.
Wait, $S=6$. $x=4 \implies S-1-4 = 1$. Component $\{4, 1\}$.
Component $\{2, 3\}$ and $\{4, 1\}$ are disjoint.
Wait, sum is 6.
$2+4=6$.
$3+3=6$.
$4+1=6$.
$5+0=6$.
$0+5=6$.
Components:
$x \in \mathbb{Z}_{\ge 0}$.
Map $f(x) = S-1-x$.
Components are $\{x, S-1-x\}$.
Wait, this assumes $x$ is unique identifier.
But we have indices. $A_1=x$.
$A_1$ maps to $S-1-A_1$.
So $A_1$ is confined to $\{A_1^{init}, S-1-A_1^{init}\}$.
This must contain $B_1$.
So $B_1 \in \{A_1^{init}, S-1-A_1^{init}\}$.
This implies $B_1 = A_1$ OR $B_1 + A_1 = S-1$.
Wait, $S-1-A_1$ depends on $A_1$.
Actually, $S = \sum A$.
So if $N=2$, condition is $A_1=B_1$ or $A_1+B_1 = \sum A - 1$.
Wait, this is quite strict.
Is this for $N > 2$?
For $N=3$, we have $(i, j), (i, k), (j, k)$.
Can we connect components?
Yes, we can mix.
Is it likely that for $N \ge 3$, it's always possible if sum matches?
The problem is from AtCoder or similar.
Actually, I recall a problem "Sequence Transformation".
If $N$ is small, maybe it is always possible except for some parity/sum issue?
Wait, if $N \ge 3$, we can change $A_1$ parity without changing $A_2$ parity?
Pick $A_1, A_2$ with diff parity?
No, then $A_1, A_2$ don't change parity.
Pick $A_1, A_2$ with same parity. Both flip.
If we have $N=3$, say parities $0, 0, 0$.
Pick $(1, 2) \to 1, 1, 0$.
Now we have mixed parities.
So we can manipulate parities freely?
Wait, if we have $0, 0, 0$, op on $(1, 2)$ gives $1, 1, 0$.
Op on $(2, 3)$ (values at 2, 3) $\implies 0, 1, 1$?
State: $A_2$ odd, $A_3$ even.
Op $(2, 3)$ on $A_2, A_3$ ($1, 1, 0$)?
$A_2=1$ (odd), $A_3=0$ (even). Diff parity.
Op preserves parities!
So $A_2 \to 0$ (even), $A_3 \to 1$ (odd)?
Wait.
If $A_2=1, A_3=0$.
$A_2' = A_3 - 1 = -1$ (Wait, inputs positive?). Constraints $A_i \ge 1$.
If negative allowed, ok. If not, restricted.
Problem says $A_i, B_i \ge 1$. Operations can produce $\le 0$?
The operation is reversible and we can backtrack. So intermediates can be anything?
But we usually care about reachable states.
If parities are fixed for a pair of diff-parity numbers.
To change parity of a single number, we need a partner with same parity.
If we have any pair with same parity, we can flip both.
If we have no pairs with same parity (all alternating $0, 1, 0, 1 \dots$), then we can't change any parity?
Wait, if we have $0, 1, 0$.
Pairs: $(0, 1)$ (diff), $(1, 0)$ (diff), $(0, 0)$ (same).
So we have a pair with same parity.
When is it impossible to find same parity pair?
Only if parities alternate perfectly?
For $N$ even, $0, 1, 0, 1 \dots$ implies perfect alternation.
If parities are $0, 1, 0, 1 \dots$, then every pair $(i, j)$ has diff parity?
No. Indices $1, 3$ have same parity (index-wise). But $A_1$ might be even, $A_3$ might be even.
Wait, the condition "pair with same parity" refers to values $A_i, A_j$.
If the array contains all distinct values mod 2?
Impossible for $N \ge 3$ to have only 0s and 1s such that no two are equal. By PHP, at least two are same.
So we can always find a pair $(i, j)$ such that $A_i \equiv A_j \pmod 2$.
Applying op $(i, j)$ flips both.
This adds 2 to count of odds (or removes 2).
Wait, flipping two 0s to 1s increases odd count by 2.
Flipping two 1s to 0s decreases odd count by 2.
So the parity of the number of odd elements is invariant (modulo 2).
Wait, I already established this.
But we also established that we can change the parities of specific elements if we act on pairs.
So parities are not rigidly locked for $N \ge 3$.
Wait, is there any constraint forbidding $B$?
It seems for $N \ge 3$, sum equality is the ONLY condition?
Let's check Example 2 again.
$A$: sum 22. $B$: sum 16. Output No.
Matches sum condition.

What if $N \ge 3$ and sums equal?
Is it always Yes?
Actually, looking at competitive programming history, this problem resembles "Make A equal to B".
Often the answer is "Yes if sums equal, provided we can reach any permutation/matching".
But this operation is not a simple swap.
Wait, if $N$ is large enough to mix everything, it's likely Yes.
For $N \ge 3$, the graph is likely connected on the subspace of equal sum.
I suspect the condition is just Sum A == Sum B.
Wait, is there any case where minimal operations matters?
We need to minimize $M$.
Since I need to output operations, I should probably implement a search/BFS.
With $N=100$, BFS is impossible on state space.
But maybe I can just find a sequence of moves to satisfy differences.
If I want to fix $A_i = B_i$, I need to move values in/out of $i$.
If $A_i > B_i$, I need to decrease $A_i$.
If $A_i < B_i$, I need to increase $A_i$.
Decreasing $A_i$ requires $A_j \le A_i$ (since if $A_j > A_i+1$, $A_i$ increases).
Wait, if $A_j > A_i+1$, $A_i$ increases.
If $A_j \le A_i$, $A_i$ decreases (amount $A_i - A_j + 1$).
Wait, check formula: $\Delta A_i = A_j - A_i - 1$.
If $A_j \le A_i$, $\Delta A_i < 0$.
So $A_i$ decreases by $1 + A_i - A_j$.
This can be large.
So we can drain $A_i$ using some $A_j \le A_i$.
Similarly, we can boost $A_i$ using some $A_j > A_i+1$.
This looks like we can flow mass.
If we can move mass freely, we can surely equalize sums.
So the strategy would be:
Iterate $i$ from 1 to $N$. If $A_i \neq B_i$:
   Try to fix $A_i$ by interacting with some $j$.
   Since we want to minimize operations, maybe we just fix one element at a time?
   Fixing one element might disturb others.
   But if we use a temporary buffer or just interact with someone who doesn't need value yet?
   Actually, if $N$ is small, maybe just running a greedy loop works?
   What if we simply sort the operations based on which $j$ helps most?
   Or maybe just a randomized local search?
   But score depends on operations count.
   Wait, if we can fix $A_i$ by using $A_j$ and $A_j$ needs adjustment, we can chain.
   
Let's look at the complexity of "flow".
Maybe it's simpler.
Let's observe the example 3.
It uses 18 operations.
$N=7$.
The solution is long.
Maybe the number of operations is roughly proportional to something.
Could it be that we simply perform operations until convergence?
No, we need minimum.
Actually, if $N$ is small enough, maybe BFS on a reduced state is possible?
Like, how many mismatches?
At most $N$.
If we maintain the list of indices where $A_i \neq B_i$.
This looks like we can solve it.
However, I need a robust algorithm.
What if I use a heuristic search like A* or simulated annealing?
Given 2 seconds, maybe I can run a randomized local search that quickly finds a solution if it exists?
Wait, the problem asks to determine YES/NO.
If sum != sum, output NO.
Else YES (assuming $N \ge 3$).
Then generate solution.
If I assume YES for sum match, I just need to find a solution.
Is there a risk that sum match but NO exists?
Let's check the constraints again.
Constraints: $2 \le N \le 100$.
If $N=2$, we found a case where sum match but NO (unless strict condition met).
Wait, if $N=2$ is handled by BFS/DFS on the small component, and $N \ge 3$ assumed connected.
Actually, the component size for $N=2$ is 2. For $N=3$, is it small?
For $N=3$, state space is roughly bounded by sum. Sum $\le 300$.
Actually, if we just want to solve it, we can treat it as finding a path in a graph.
But the graph is huge.
However, if the graph is "dense" and "mixes well", maybe a simple heuristic works.
Heuristic: Pick $i$ with $A_i \neq B_i$. Find $j$ such that applying $(i, j)$ reduces $\sum |A_k - B_k|$.
If none improves, randomize?
Wait, greedy might fail.
But notice the operation can move large chunks.
Let's refine the "move mass" idea.
We want to transfer $\delta_i = B_i - A_i$ to $i$.
Total $\sum \delta_i = 0$.
We want to perform a set of operations.
Each op on $(i, j)$ changes $A_i \to A_j-1, A_j \to A_i+1$.
Effect on deviation vector:
$\delta_i' = B_i - (A_j - 1) = (B_i - A_i) + 1 - (A_j - A_i) ...$ no.
Let's look at the effect on deviations.
We want $A_i$ to match $B_i$.
If $A_i$ needs to increase (i.e., $A_i < B_i$), we want to apply op where $i$ receives from $j$.
Receiving from $j$ means $A_i$ increases. This happens if $A_j > A_i + 1$.
If $A_i$ needs to decrease ($A_i > B_i$), we want to apply op where $i$ loses to $j$.
Losing to $j$ means $A_i$ decreases. This happens if $A_j \le A_i$.
Wait, if $A_i > B_i$, we want to apply op on $(i, j)$ (with $i < j$) such that $A_i$ decreases?
Op on $(i, j)$ sets $A_i \leftarrow A_j - 1$.
So if we want $A_i$ to decrease, we need $A_j - 1 < A_i \iff A_j < A_i + 1$.
Since $A_j$ is integer, this means $A_j \le A_i$.
So yes, if we have $A_i$ too high and $A_j$ too low (or similar), we can balance.
This looks like we are just moving mass.
Is it possible to greedily fix $A_1$ then $A_2$...?
Fix $A_1$: if $A_1 \neq B_1$, we need to change $A_1$.
We need a helper $j$ ($j \neq 1$).
If $A_1 < B_1$, we need $A_1$ to increase. We need $A_j > A_1 + 1$.
We find such $j$. Perform op $(1, j)$.
This sets $A_1 = A_j - 1$.
If $A_j - 1$ matches $B_1$, great. If not, $A_1$ becomes closer?
But $A_j$ becomes $A_1 + 1$. This modifies $A_j$.
If $A_1 > B_1$, we need $A_1$ to decrease.
We need $A_j \le A_1$.
Perform op $(1, j)$. $A_1$ becomes $A_j - 1$.
Since $A_j \le A_1$, $A_j - 1 < A_1$.
Can we make $A_1$ hit $B_1$ exactly?
Suppose $A_1 = 10, B_1 = 5$.
Need to decrease by 5.
We need $A_j$ such that after op, $A_j-1 = 5 \implies A_j = 6$.
If we have some $A_j = 6$, we apply $(1, j)$. $A_1 \to 5$.
If we don't have $A_j = 6$, maybe we create it?
This suggests a constructive algorithm.
Actually, since $N$ is small, maybe we can just iterate on which pairs to use.
Wait, there is a known algorithm for this specific problem "Minimum number of operations to transform A to B" with this operation.
It turns out that we can treat this as a flow problem or simply a greedy strategy works if we are careful.
However, given the constraints and scoring (penalty for too many ops), we need a good solution.
Wait, is it possible that the optimal strategy is related to sorting?
Or just matching?
Let's assume the following greedy strategy:
While $A \neq B$:
  Find index $i$ such that $A_i \neq B_i$.
  If $A_i > B_i$: We need to decrease $A_i$.
    Find $j \neq i$ to help decrease $A_i$.
    Ideally, we want $A_j \approx A_i$? No, we want $A_j - 1 = B_i$ ideally.
    If no such $A_j$ exists, maybe just pick any $j$?
    Actually, if we just pick any $j$ with $A_j \le A_i$ and $i < j$ (or swap order? No, order fixed), we decrease $A_i$.
    Wait, we can pick any pair $(u, v)$. If we want to decrease $A_i$, we can use pair $(i, j)$ where $i < j$ (so $A_i$ decreases if $A_j \le A_i$) OR pair $(k, i)$ where $k < i$ (so $A_i$ acts as $v$, increasing $A_k$, decreasing $A_i$).
    Wait, op $(k, i)$ with $k < i$: $A_k \leftarrow A_i - 1$, $A_i \leftarrow A_k + 1$.
    So $A_i$ increases!
    So to decrease $A_i$, we MUST use it as the first index in the pair (source of decrease? No target of decrease).
    Wait.
    Op $(i, j)$: $A_i \leftarrow A_j - 1$.
    Op $(k, i)$: $A_k \leftarrow A_i - 1$, $A_i \leftarrow A_k + 1$.
    So $A_i$ decreases ONLY if it is the 'first' element in the pair AND $A_j \le A_i$ (condition for $A_j-1 < A_i$).
    Wait, check formula again.
    $\Delta A_i = A_j - 1 - A_i$.
    If $A_j \le A_i$, $\Delta A_i < 0$.
    So $A_i$ decreases.
    So we must select a pair $(i, j)$ with $i < j$ such that $A_j \le A_i$.
    Wait, what if all available $j > i$ have $A_j > A_i + 1$?
    Then we can't decrease $A_i$ using $j$.
    We can't use $k < i$ because that acts as $A_i$ being second element, causing $A_i$ to increase.
    So if we can't decrease $A_i$ directly, we might need to rearrange.
    This suggests dependency.
    
    Actually, there is a much simpler observation.
    The operation is $(x, y) \to (y-1, x+1)$.
    This preserves $x+y$.
    Also $x \pmod 2$ and $y \pmod 2$ relationship.
    But most importantly, it looks like we can just bubble sort the values to match target?
    Wait, if we want to move value from $u$ to $v$.
    We want $A_u$ to decrease and $A_v$ to increase.
    Assume $u < v$. We want $A_u$ decr, $A_v$ incr.
    This requires pairing $(u, v)$.
    Condition for $A_u$ decr: $A_v \le A_u$.
    Condition for $A_v$ incr: $A_u \ge A_v - 1$.
    Basically we need $A_v \le A_u$.
    If we have $A_u > A_v$, we can decrease $A_u$ and increase $A_v$.
    Wait, if we have $A_u$ large and $A_v$ small.
    We can transfer mass.
    What if $A_u < A_v$? We want $A_u$ decr.
    Then $A_u$ is small, $A_v$ is large.
    Pairing $(u, v)$: $A_u$ will increase (since $A_v > A_u + 1$).
    So we can't use $(u, v)$ to decrease $A_u$ directly.
    We would need an intermediate node $w$.
    Move from $v$ to $w$ (requires $v<w$ and $A_w \le A_v$), then $w$ to $u$?
    
    Okay, this is getting complicated for greedy.
    But wait!
    The problem might be simpler.
    Look at the example outputs. They perform many ops.
    But notice the constraints. $N \le 100$.
    Maybe there's a property that $A_i + i$ is invariant? Or something.
    No.
    
    Let's reconsider the "No" condition.
    If I'm confident that for $N \ge 3$ with equal sum, it is always possible.
    Then I just need a generator.
    What if I use the property that I can swap adjacent elements with cost?
    No.
    
    Actually, what if the solution involves sorting the array $A$ and comparing with sorted $B$?
    No, positions matter.
    
    Wait, I found a reference to a similar problem (Codeforces? AtCoder?).
    Problem "Make Equal".
    If $\sum A \neq \sum B$, impossible.
    If $N=2$, special conditions (checked above).
    If $N \ge 3$, always possible?
    Wait, is it possible that $A_i$ values become negative?
    No, constraints say $1 \le A_i$.
    Does the operation guarantee positive values?
    $A_i' = A_j - 1$. If $A_j = 1$, then $A_i' = 0$.
    The problem doesn't say $A_i$ must stay $\ge 1$ during intermediate steps.
    However, constraints say $A_i$ are inputs. Goal $A=B$ with $B_i \ge 1$.
    Usually competitive programming problems allow negative intermediate values unless specified.
    "All values in input are integers" implies inputs.
    It does not restrict intermediate values.
    So 0 is fine. Even negative.
    This relaxation means connectivity is easier.
    
    If so, for $N \ge 3$, it is highly probable that sum equality is the necessary and sufficient condition.
    Let's assume this is true.
    Now, how to construct solution efficiently?
    We need to minimize $M$.
    With $N=100$, maybe BFS on a simplified state works?
    Like, we just need to correct the values.
    Notice that the operation preserves $\sum A_k$ and $\sum A_k^2$ ? No.
    $x^2 + y^2$ vs $(y-1)^2 + (x+1)^2 = y^2 - 2y + 1 + x^2 + 2x + 1 = x^2 + y^2 + 2(x-y+1)$.
    Not invariant.
    
    Let's check the operation again.
    $A_i \leftarrow A_j - 1$. $A_j \leftarrow A_i + 1$.
    This is equivalent to $A_i \leftarrow A_j - 1$.
    Note that $A_j$ is just a variable holding a value.
    Maybe we can think of this as:
    We have a pool of values. We can reorder/modify.
    But since we want min ops, maybe we should try to use each op to fix at least one index?
    Or fix multiple.
    If $A_i \neq B_i$ and $A_j \neq B_j$.
    Maybe apply $(i, j)$ to make $A_i = B_i$ or $A_j = B_j$?
    
    Wait, the example 1 used 2 ops to fix 2 indices.
    Initial: A=[2, 2, 1, 4], B=[3, 2, 2, 2].
    Indices needing change: 1 (2->3), 3 (1->2), 4 (4->2).
    Ops: (1, 4) -> A=[3, 2, 1, 3]. Fixes 1.
    Remaining issues: 3 (1->2), 4 (3->2).
    Op (3, 4) -> A=[3, 2, 2, 2]. Fixes 3 and 4.
    So 2 ops fixed 3 indices.
    Strategy:
    Iterate $i$ from 1 to $N$. If $A_i \neq B_i$, find $j > i$ to fix it?
    If we just fix $i$ greedily, does it work?
    Suppose we are at $i$. We want $A_i \to B_i$.
    We look for $j > i$ such that applying $(i, j)$ brings $A_i$ to $B_i$.
    Equation: $B_i = A_j - 1 \implies A_j = B_i + 1$.
    If we find such $j$ (where $j>i$ and $A_j = B_i+1$), we can just execute $(i, j)$.
    Then $A_i$ becomes $B_i$.
    What about $A_j$? $A_j$ becomes $A_i + 1$.
    If $A_i$ was the current value at $i$.
    Does this mess up future $j$?
    Yes.
    But if we process $1 \dots N$ sequentially.
    When processing $i$, we might pick a $j > i$.
    Later when we reach $j$, we will have a new value $A_j$.
    Maybe we can fix it then?
    This looks like a standard algorithm for "making array equal".
    Since $N$ is small, maybe this works.
    Complexity $O(N^2)$ per operation? Or just one pass?
    Wait, after updating $A_j$, its value changes.
    If we scan $1 \dots N$, and whenever $A_i \neq B_i$, we try to find a $j$.
    If no $j$ satisfies condition, what to do?
    Maybe we can't fix $A_i$ to $B_i$ in one step.
    Maybe we need multiple steps involving intermediaries.
    But if the graph is connected, maybe we just need to "bubble" values.
    
    Let's look at the scoring again.
    Score is linear with $M$. We need $M \approx M_0$.
    Since we don't know $M_0$, we should aim for a reasonable solution.
    Actually, if I use the sequential fix strategy:
    For $i = 1 \dots N$:
      While $A_i \neq B_i$:
         Find $j > i$.
         We want to change $A_i$ and potentially $A_j$.
         To change $A_i$, we must interact with some $k$.
         Ideally $k < i$ to change $A_i$ without affecting processed indices?
         No, if $k < i$, $A_k$ is already "fixed" (hopefully). If we touch it, we break it.
         So we should only use $k > i$.
         Using $k > i$, we set $A_i \leftarrow A_k - 1$.
         Then $A_i$ becomes $A_k - 1$.
         We want this to be $B_i$ or closer?
         But this consumes $A_k$.
         If we fix $A_i$ by taking $A_k - 1 = B_i$, then $A_k$ increases.
         Wait, if we set $A_i \leftarrow B_i$, then we are done with $i$.
         Cost is 1 op.
         Is it always possible to find such $k$?
         Only if there exists $k > i$ with $A_k = B_i + 1$.
         This is not guaranteed.
         However, if we can't find $k$ with exact value, we might need to modify $A_k$ first?
         But that requires touching $k$.
         Maybe we can perform a series of swaps?
         
    Alternative Idea:
    The problem might be solved by simply trying to sort $A$ such that it matches $B$?
    No.
    
    Let's go back to the constraints and the nature of the problem.
    Maybe the intended solution is to just run a BFS on a "reduced" state?
    But $N=100$ is too big for standard BFS.
    However, look at the memory limit: 512MB.
    Is there any property we missed?
    "If you output Yes, also output a sequence".
    Usually this means we need to construct it.
    The fact that it asks for minimum number suggests it's hard.
    But scoring is based on ops.
    
    Wait! There is a trick.
    If we want to move $A_i$ to $B_i$, we can use any $j$.
    Consider the values $A_i$ and $B_i$.
    Let's consider the multiset of values in $A$ and $B$.
    Wait, sums must match.
    Is it possible that we can just simulate a flow?
    Assign capacities.
    Since we want min ops, this is equivalent to min-cost flow where cost of edge is infinite? No.
    Each operation costs 1.
    But operation moves non-unit amount.
    So 1 operation can move many units.
    This is a HUGE advantage.
    If $A_j \gg A_i$, one op moves huge amount.
    So min ops is likely small.
    
    Algorithm idea:
    Loop while $A \neq B$:
      Find an index $i$ such that $A_i < B_i$ (needs value).
      Find an index $j$ such that $A_j > B_j$ (has extra).
      We want to transfer from $j$ to $i$.
      Transfer direction: $j \to i$ means $A_j$ decr, $A_i$ incr.
      If $j > i$, op is $(i, j)$.
      Effect: $A_i \leftarrow A_j - 1$, $A_j \leftarrow A_i + 1$.
      If we want $A_i$ to increase, we need $A_j > A_i$.
      Also to ensure $A_j$ decreases (transfer out), we need $A_j$ to be large enough?
      Wait, if we use $(i, j)$, $A_j$ decreases iff $A_j > A_i$.
      If $A_j \le A_i$, $A_j$ increases (bad).
      So we need $A_j > A_i$ to have a "valid" transfer from $j$ to $i$.
      Wait, earlier I said if $A_j \le A_i$, $A_i$ decreases.
      Yes.
      So if we need $A_i$ increase and $A_j$ decrease, we MUST have $A_j > A_i$.
      If we have $A_i$ needing increase, and all available $j$ (surplus) have $A_j \le A_i$, we can't fix $i$ by direct transfer from $j$.
      But if $A_j$ is surplus, $A_j$ is "large" typically (relative to B)?
      Not necessarily, just $A_j \neq B_j$.
      
      Wait, we can pick ANY pair.
      What if we pick $k$ such that $A_k$ is "neutral"?
      
    Let's consider the simplest greedy:
    Just iterate through pairs and apply if it reduces error metric $D = \sum |A_k - B_k|$.
    Since $N$ is small, maybe we can just run this loop until convergence.
    Heuristic metric: $\sum (A_k - B_k)^2$.
    Minimize this.
    This is a local search.
    Given 2 seconds, maybe we can improve the local optimum.
    But we need minimum number of operations.
    Does minimizing squared error correspond to min ops?
    Not necessarily. One op might reduce error by 100, or by 1.
    But since $A_j > A_i + 1$ allows large jumps, it should be efficient.
    Wait, the operation $(x, y) \to (y-1, x+1)$ preserves sum.
    If we interpret this as moving points on the hyperplane $\sum x = S$.
    We want to reach $B$.
    Since we can move large distances, we should prioritize large moves.
    Large moves occur when $|A_j - A_i|$ is large.
    Specifically, $A_i$ changes by $A_j - A_i - 1$.
    If $A_j \gg A_i$, change is large positive.
    If $A_i \gg A_j$, change is large negative.
    
    Algorithm:
    Repeat until $A=B$:
      Find pair $(i, j)$ with $i < j$ that maximizes reduction in $\sum |A_k - B_k|$.
      Apply it.
    This is a greedy local search.
    Will it get stuck? Maybe.
    But with sum constraint, the space is compact.
    Also, random restarts could help.
    However, we need to output *the* minimum sequence.
    Greedy might not be optimal.
    But for "minimum length", maybe the number of ops is small?
    In Ex 3, 18 ops for $N=7$.
    This is small.
    If I just run random greedy with restarts?
    The problem asks to determine achievability.
    So first check: $\sum A == \sum B$.
    Also handle $N=2$ special cases properly.
    If $N \ge 3$, assume achievable.
    Then find solution.
    
    How to guarantee minimum?
    Maybe the "min" part is a red herring for this difficulty level?
    Or maybe the number of ops is unique/minimal for a certain strategy?
    Actually, the scoring penalizes suboptimal solutions.
    So I MUST find a short sequence.
    
    Is there a closed form for min ops?
    Maybe it's related to how many inversions?
    Or just matching.
    
    Let's try to code a solver.
    If I can't prove optimality, maybe I can use a randomized hill climbing.
    Since $N$ is small, maybe we can run a BFS on the *graph of differences*.
    Difference state $D = A - B$. We want $D=0$.
    State space is huge.
    But maybe the depth is small?
    Ex 3: depth 18.
    With $N=100$, depth could be larger.
    But maybe we can optimize.
    
    Wait, there is a constraint $A_i, B_i \le 100$.
    Values are small.
    Maybe this is the key.
    Values are small.
    Can we use BFS with pruning?
    Or IDA* (Iterative Deepening A*)?
    Heuristic: $h(state) = 0$ (since we can always do it?).
    Lower bound on ops:
    We need to fix $\sum (A_i - B_i)_+$ deficits.
    Let $D_k = A_k - B_k$. $\sum D_k = 0$.
    Positive $D_k$ are surpluses, negative are deficits.
    Each op $(i, j)$ changes $D_i \leftarrow (A_j - 1) - B_i = (A_j - B_i) - 1 = (A_j - B_j + B_j - B_i) - 1 = D_j + (B_j - B_i) - 1$.
    And $D_j \leftarrow (A_i + 1) - B_j = D_i + (B_i - B_j) + 1$.
    So $\Delta D_i = D_j + (B_j - B_i) - 1 - D_i$? No.
    Wait, let's express change in $D$.
    $A_i^{new} = A_j - 1$.
    $D_i^{new} = A_j^{old} - 1 - B_i = (A_j^{old} - B_j) + (B_j - 1 - B_i) = D_j^{old} + (B_j - 1 - B_i)$.
    Similarly $D_j^{new} = A_i^{old} + 1 - B_j = D_i^{old} + (1 + B_i - B_j)$.
    So $D_i^{new} - D_j^{new} = D_j - D_i + (B_j - 1 - B_i) - (1 + B_i - B_j) = D_j - D_i + 2(B_j - B_i) - 2$.
    This doesn't simplify nicely.
    However, we can define $E = \sum_{k: D_k > 0} D_k$. (Total surplus).
    We want to reach 0.
    Does each operation reduce $E$?
    Not necessarily.
    But intuitively, we want to move mass from positive to negative.
    
    Given the time limit and difficulty, I will implement a heuristic solver.
    1. Check Sum Equality.
    2. If $N=2$, check specific reachability condition (which I'll derive/code).
       Wait, for $N=2$, if sum match, check if $(A_1, A_2)$ can reach $(B_1, B_2)$.
       Reachable if $A_1 = B_1$ OR $A_1 + B_1 = \sum A - 1$? No.
       My derivation:
       $A_1$ reachable from $A_1$ is $\{A_1, \sum A - 1 - A_1\}$.
       So we check if $B_1 \in \{A_1, S-1-A_1\}$.
       This covers $N=2$.
    3. If $N \ge 3$:
       Use a randomized local search / Hill Climbing.
       Metric: Distance $\sum |A_k - B_k|$.
       Also $\sum (A_k - B_k)^2$.
       We want to minimize this.
       Since we need minimum operations, maybe just simple greedy with some randomness is enough.
       Wait, simple greedy might loop.
       We can use Simulated Annealing?
       Or just BFS if we prune.
       Actually, with $N$ up to 100, maybe we can use the small value constraint ($A_i \le 100$).
       Wait, $A_i$ can go outside [1, 100]?
       If yes, then state space is unbounded.
       But if we want min ops, probably values won't wander far.
       
       Let's refine the greedy strategy:
       We want to reduce $\sum |A_k - B_k|$.
       Pick pair $(i, j)$ that reduces this sum the most.
       Break ties randomly.
       Repeat.
       If stuck (no move reduces sum), try to escape (accept worse move with prob? or just random?).
       But we want min ops, so we shouldn't accept worse moves.
       If no improvement, maybe we are at a local optimum where $\sum |A-B|$ is minimal but $A \neq B$.
       This means $\sum |A-B| > 0$.
       In that case, is it possible to proceed?
       Yes, we can trade off.
       
       Actually, maybe there's a better heuristic.
       Sort the array $A$ and $B$? No.
       Just focus on making $A_i$ close to $B_i$.
       
       Let's write a code that implements this local search.
       To ensure it terminates and is fast:
       Max iterations = some large number (e.g. 10000).
       If not solved, output failure? No, must solve.
       Maybe the problem guarantees solution for $N \ge 3$ with sum equal.
       Is there a specific construction that is optimal?
       Wait, if $N \ge 3$, we can always find a solution.
       Is there a constructive solution?
       Sort $A$ indices by value?
       Try to move values to fill deficits.
       Collect all indices with $A_i < B_i$ (Deficits) and $A_j > B_j$ (Surplus).
       While Surplus and Deficits lists not empty:
         Pick $u$ from Deficit (needs inc) and $v$ from Surplus (needs decr).
         We want to move value from $v$ to $u$.
         If we use pair $(u, v)$, we need $u < v$ or $v < u$.
         If $u < v$, we apply $(u, v)$.
           $A_u \leftarrow A_v - 1$.
           $A_v \leftarrow A_u + 1$.
           We hope $A_u$ approaches $B_u$.
           Specifically, if $A_v$ is huge, $A_u$ becomes huge. Good.
           If $A_v$ is small?
           Wait, if we just greedily pair closest?
           
       Actually, this looks like we can treat this as a flow problem on a graph where we have capacity to move values.
       Since operations preserve sum, we can just balance things.
       
       Wait! There's a solution using DFS/BFS on $N$ indices? No.
       
       Let's look at the constraints again.
       $N \le 100$, $A_i \le 100$.
       Maybe the "No" condition is ONLY sum inequality for $N \ge 3$?
       Let's assume YES for sum match, $N \ge 3$.
       And implement the local search.
       
       Wait, I can optimize the local search.
       Instead of iterating all pairs, pick best $u$ and best $v$ (closest to target)?
       Then pick the partner $w$ that helps best?
       
       Another thought:
       Since $N$ is small, maybe I can use a recursive backtracking with memoization?
       Too slow.
       
       Let's try to code a "randomized greedy" solver.
       Initialize `ops` list.
       Loop until $A=B$:
         Iterate over all pairs $(i, j)$ ($i < j$).
         Calculate $\Delta = \text{error}(A) - \text{error}(A')$.
         Where error is e.g. $\sum |A_k - B_k|$.
         If $\Delta < 0$ (better found):
           Store top 3 best improvements.
           If none improve, stop (local optima).
           If multiple improve, pick random among them.
         Execute op.
         Add to history.
       
       To avoid infinite loops, we must ensure monotonic decrease or termination.
       Local optima might trap us.
       But for small $N$ and small values, maybe the landscape is flat-ish?
       Actually, if we reach a state where no op improves error, but error > 0.
       Then we are stuck.
       However, it's unlikely we reach a state with zero improvement but non-zero error?
       Wait, if $A \neq B$, is there always an operation that improves $\sum |A-B|$?
       Suppose $A=[1, 2, 3]$ and $B=[2, 2, 2]$. Error=2.
       Pairs:
       (1, 2): $1 \to 2, 2 \to 1+1=2$?
       Wait, $A=[1, 2, 3]$. Op (1, 2): $A_1=2-1=1, A_2=1+1=2$.
       No change. Error same.
       (2, 3): $2 \to 3-1=2, 3 \to 2+1=3$.
       No change.
       (1, 3): $1 \to 3-1=2, 3 \to 1+1=2$.
       New A: $[2, 2, 2]$. Matches B. Error 0.
       Improvement!
       So if we have a suitable $i, j$, we can fix.
       Are there cases where no op improves error?
       Suppose $A=[2, 2]$, $B=[2, 2]$. Done.
       Suppose $A=[3, 3]$, $B=[4, 2]$. Sum 6.
       Error $|3-4|+|3-2| = 1+1 = 2$.
       Pairs (1, 2).
       $A_1 \leftarrow 3-1=2$. $A_2 \leftarrow 3+1=4$.
       New A: $[2, 4]$.
       Error $|2-4|+|4-2| = 2+2=4$. Worse.
       So (3, 3) $\to$ (2, 4) increases error.
       So for $[3, 3]$ vs $[4, 2]$, greedy fails immediately.
       But we established for $N \ge 3$ it should be possible.
       What about $[3, 3, 3]$ vs $[4, 2, 3]$?
       $A=[3, 3, 3]$. $B=[4, 2, 3]$.
       Need to fix 1 (needs +1), 2 (needs -1).
       Use pair (1, 2): $A_1 \leftarrow 2$ (since $A_2=3 \to 3-1=2$). $A_2 \leftarrow 4$.
       New A: $[2, 4, 3]$.
       Error: $|2-4| + |4-2| + 0 = 4$.
       Still bad.
       Use pair (2, 1)? No, must be ordered.
       Use (2, 3) where $2 < 3$?
       $A_2 \leftarrow 3-1=2$. $A_3 \leftarrow 3+1=4$.
       New A: $[3, 2, 4]$. Matches B? No B is $[4, 2, 3]$.
       Current: $3, 2, 4$. Error $|3-4|+|2-2|+|4-3| = 1+0+1 = 2$.
       Error improved from 2 to 2?
       Wait, initial error was 2. Final is 2. No change.
       Actually, $[3, 3, 3] \to [3, 2, 4]$ is useful?
       Wait, target is $[4, 2, 3]$.
       We want $A_1=4$.
       Maybe we need to cycle values.
       With $N \ge 3$, we can shuffle.
       
       The "greedy improvement" might get stuck in local minima.
       But we can use **Simulated Annealing** or **Random Walk**.
       Given 2 seconds, and $N=100$, maybe a few thousand random attempts are OK.
       However, random walk without memory is slow.
       Better: Randomly pick an operation that reduces error slightly or increases slightly but explores.
       Or just "stochastic hill climbing":
       Always pick op that reduces error.
       If stuck, pick op that increases error but is "less bad" or random?
       Actually, if I just use a heuristic that prioritizes fixing elements?
       
       Let's implement a randomized search with restarts.
       This is robust.
       State representation: vector of ints.
       Cost function: sum(|A[k]-B[k]|).
       Neighbors: All pairs $(i, j)$.
       Step:
         Best neighbor found.
         If neighbors include improving, take best.
         Else if neighbors include neutral? No improvement.
         Else worst neighbor?
         
       Optimization:
       Since $N=100$, calculating all neighbors is $100*100/2 = 5000$ checks.
       Doing this 100 times is 500k ops. Fast.
       So we can afford to check all neighbors at each step.
       The challenge is escaping local minima.
       If stuck (no improving move), we must perturb.
       We can pick the neighbor with the smallest *increase* in cost (least damage).
       Then continue searching for improvement from there.
       This is basically steepest descent allowing uphill moves.
       
       Wait, Example 1:
       Init: 2 2 1 4. B: 3 2 2 2. Cost 4.
       Pairs:
       (1, 4): $2, 4 \to 3, 3$. Cost $|3-3|+|2-2|+|1-2|+|3-2| = 0+0+1+1=2$. Improved.
       (3, 4): $1, 4 \to 3, 2$. (Using original A? No, sequential).
       If we apply (1, 4) -> [3, 2, 1, 3]. Cost 2.
       Next step, from [3, 2, 1, 3].
       (3, 4): $1, 3 \to 2, 2$. A -> [3, 2, 2, 2]. Cost 0.
       Found.
       
       This suggests simple greedy works if lucky.
       With randomness, we can guide it.
       
       I will use a hybrid:
       1. Check Sum. If unequal -> No.
       2. Check N=2 special cases.
       3. If N >= 3 -> Run Local Search.
          Use a timeout/iteration limit.
          Allow occasional worsening moves.
       4. To ensure we output *something* within time.
       
       Wait, if N=2, my condition was:
       Reachable iff $B_1 \in \{A_1, S-1-A_1\}$.
       Let's double check this.
       If $A=[2, 3], B=[3, 2]$. $S=5$.
       $S-1-A_1 = 5-1-2 = 2$.
       $B_1 = 3$.
       $3 \notin \{2, 2\}$. Impossible.
       Correct.
       Wait, if $A=[2, 3], B=[2, 3]$. $B_1=2 \in \{2\}$. OK.
       If $A=[1, 4], B=[3, 2]$. $S=5$. $A_1=1$. Set $\{1, 3\}$. $B_1=3$. OK.
       So this condition seems correct.
       
       Implementation details:
       Use `std::vector<int>`.
       Cost function: `abs(sum(A[i] - B[i]))`.
       Loop for 10000 iterations?
       If found solution, print.
       If not, maybe fail? No, problem implies always solvable if sum matches?
       Wait, if local search fails, what then?
       Maybe the "No" cases for $N \ge 3$ are rare or non-existent?
       Or maybe my heuristic needs to be smarter.
       But for CP, if I output a valid solution that satisfies constraints, I get points.
       If it's not minimum, score decreases.
       But I can't guarantee min easily.
       The prompt asks for "minimum length".
       This implies I should try to find a short one.
       Maybe just finding *any* solution is risky if length is huge.
       However, the problem statement scoring gives partial points for fewer ops.
       So I must try to minimize.
       
       Actually, for $N \ge 3$, the graph diameter is small.
       Is there a better way?
       What if I just sort $A$ and try to match $B$?
       No.
       
       Let's stick to the greedy local search.
       But to improve speed and chance of success:
       Prioritize pairs that reduce cost.
       Also, maybe prioritize fixing $i$ with $A_i < B_i$?
       Wait, in $N=100$, we have many choices.
       
       One more thing:
       Is it possible that the operations commute? No.
       
       Let's write the code.
       I will use a randomized local search with tabu-like mechanism?
       No, keep it simple.
       Random shuffling of the search order at each step.
       
       Wait, I can implement a specific logic:
       Identify indices where $A_i \neq B_i$.
       While there are mismatches:
         If $A_i < B_i$:
           We need to increase $A_i$.
           We need a $j > i$ (or any $j$) such that applying $(i, j)$ helps.
           This requires $A_j > A_i$.
           Wait, if we use $(i, j)$ with $i < j$, $A_i \to A_j - 1$.
           If we want $A_i$ to increase, we need $A_j - 1 > A_i \iff A_j > A_i + 1$.
           So we need a neighbor with larger value.
           Among all such $j$, pick the one that makes $A_i$ closest to $B_i$ or reduces global error most.
           Wait, $A_j$ will become $A_i + 1$.
           We need to balance this side effect.
           Maybe just pick $j$ such that error reduction is maximized.
         If $A_i > B_i$:
           Need to decrease $A_i$.
           We need a neighbor $j$ (could be any $k < i$ or $k > i$) such that $A_i$ decreases.
           If we use $(k, i)$ with $k < i$:
             $A_k \to A_i - 1$ (needs $A_i > A_k + 1$?), $A_i \to A_k + 1$ (increases!).
             Wait, $(k, i)$ with $k<i$ makes $A_i$ increase.
             So to decrease $A_i$, we MUST use it as first index.
             Wait, can we use $j$ such that $A_j \le A_i$ with $i < j$?
             Then $A_i \leftarrow A_j - 1$ (decreases).
             So yes, if $i < j$ and $A_j \le A_i$, we can decrease $A_i$.
             If $A_i > B_i$ (surplus), we need to decrease it.
             We need to find $j > i$ with $A_j \le A_i$ that minimizes error impact.
             
             Wait, if $A_i > B_i$ and we need to decrease it, but all $j > i$ have $A_j > A_i + 1$?
             Then we can't decrease $A_i$ using $j > i$.
             We must use some $k < i$ acting as second index?
             If we use $(k, i)$, $A_i$ increases. Bad.
             So if all $j > i$ are "strong" (much larger), we can't drain $A_i$.
             We might need to borrow from $k < i$ to increase something?
             This suggests we might need to shuffle values.
             
             This manual construction is complex to code correctly in one shot.
             So local search is safer.
             
             One detail: The problem says "output a sequence".
             If I can't find it, I might fail.
             However, for $N \le 100$, maybe I can run the search multiple times with random seeds and pick the best solution (shortest path).
             Wait, "best" solution?
             I need to store visited paths? No, too much memory.
             Just store best found $M$.
             Actually, just finding *a* solution is hard enough.
             Wait, scoring:
             if $m \le m_0$: Full score.
             if $m > 2 * m_0$: 0 score.
             Otherwise linear.
             This implies I should aim for $m \approx m_0$.
             Usually $m_0$ is reasonably small.
             If my local search finds a solution with $m \approx 50-100$, it might be optimal.
             If I get stuck, I'll try again.
             
             Let's finalize the plan:
             1. Input N, A, B.
             2. Check sum equality.
             3. If N=2, check condition.
                If fail, No.
                If success, check if $A=B$. If not, check if one op solves it?
                Wait, if $N=2$ condition passes, how many ops?
                My analysis showed for $N=2$, reachable states are $\{A, A'\}$ where $A'$ is derived by one op.
                If $B$ is not in set, impossible.
                Wait, my analysis assumed only 1 op type.
                Wait, for $N=2$, I said we can't swap values if parities differ.
                If parities are same?
                $A=(2, 4), B=(4, 2)$. Sum 6.
                $A \to (3, 3)$. $B$ is not reachable from $A$ in $N=2$ case?
                Wait, earlier I checked $\{2, 3\}$ and $\{4, 1\}$ components.
                So if $B$ is in different component, impossible.
                So for $N=2$, we can solve it by checking if $B$ is in the same component.
                Actually, just run BFS for $N=2$. States are pairs $(x, y)$ such that $x+y=S$.
                Wait, BFS on states is easy. $x$ can range $1 \dots S-1$ (roughly). $S \le 200$.
                So BFS on $N=2$ is trivial.
                For $N \ge 3$:
                  Assume Yes.
                  Run local search.
                  
             Code structure:
             Class Solver:
               solve() method.
               BFS for N=2.
               Local Search for N>=3.
               
             Wait, for N>=3, if local search fails, can I just output a random sequence?
             No, must make $A=B$.
             If I output garbage, it's WA.
             But if local search works 99% of the time, it's fine.
             Also, I can implement a simple heuristic to fix one element at a time using a temporary storage?
             Wait, can I move $A_i$ to a safe spot?
             For $N \ge 3$, there are "extra" indices.
             If I want to fix index 1.
             Find $j, k$ such that I can swap/adjust?
             Since I have extra nodes, I can probably use them as buffers.
             But coding a generic buffer strategy is tricky.
             Local search is the easiest generic solver.
             
             Let's refine Local Search to be faster and more robust.
             Use a priority queue or simple sorting?
             Maybe iterate through all pairs and pick the one that maximizes reduction in $\sum (A_i - B_i)^2$?
             Squared error penalizes large deviations more.
             
             Let's implement this.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

// Function to calculate sum of absolute differences
long long calc_diff(const vector<int>& A, const vector<int>& B) {
    long long res = 0;
    for (size_t i = 0; i < A.size(); ++i) {
        res += abs(A[i] - B[i]);
    }
    return res;
}

// Function to calculate sum of squares of differences
long long calc_sq_diff(const vector<int>& A, const vector<int>& B) {
    long long res = 0;
    for (size_t i = 0; i < A.size(); ++i) {
        res += (long long)(A[i] - B[i]) * (A[i] - B[i]);
    }
    return res;
}

// Operation function
void apply_op(vector<int>& A, int i, int j, vector<pair<int, int>>& ops) {
    // i < j is required
    // Update A_i <- A_j - 1, A_j <- A_i + 1 (using old values)
    // Wait, problem says replace simultaneously.
    int val_i = A[i];
    int val_j = A[j];
    A[i] = val_j - 1;
    A[j] = val_i + 1;
    ops.push_back({i + 1, j + 1}); // 1-based indexing
}

int main() {
    int N;
    if (!(cin >> N)) return 0;
    vector<int> A(N);
    for (int i = 0; i < N; ++i) cin >> A[i];
    vector<int> B(N);
    for (int i = 0; i < N; ++i) cin >> B[i];

    // Check sum
    long long sumA = accumulate(A.begin(), A.end(), 0LL);
    long long sumB = accumulate(B.begin(), B.end(), 0LL);

    if (sumA != sumB) {
        cout << "No" << endl;
        return 0;
    }

    // N=2 special handling
    if (N == 2) {
        // Since state space is small (sum <= 200), we can just BFS to find if B is reachable from A
        // Or simply check conditions derived.
        // Derived condition: A_1 must map to B_1 under x -> S-1-x.
        // But wait, for N=2, are we guaranteed to find solution if reachable?
        // Yes, graph is finite and connected components.
        // Let's just simulate BFS for safety and correctness.
        
        struct State {
            vector<int> vals;
        };
        
        // Map state to dist?
        // Since N=2, values are bounded. Sum <= 200.
        // Values can be negative? If we allow, space is infinite.
        // But if we stay within bounds near sum/2?
        // Actually, for CP, usually constraints imply staying in range.
        // Let's BFS with a hash map.
        // But to save time/memory, just check reachable set manually.
        // Reachable x1: {A1, S-1-A1}.
        // Wait, is this correct?
        // Earlier derivation: x_next = S - 1 - x_prev.
        // So x toggles between x and S-1-x.
        // So reachable set for A1 is {A1, S-1-A1}.
        // So we just check if B1 is in this set.
        
        // Note: indices 0, 1 in vector.
        int s = sumA;
        bool possible = false;
        // Condition: B1 must be equal to A1 or S-1-A1
        // Wait, S-1-A1 might be invalid? No, integers.
        if (B[0] == A[0] || B[0] == (int)s - 1 - A[0]) {
            // Also we need to ensure A1 and B1 are compatible with A2?
            // If A1 = B1, then A2 must equal B2 (since sums equal).
            if (B[0] == A[0]) possible = true;
            else {
                // B[0] = S - 1 - A[0].
                // Then A[1] = S - A[0].
                // B[1] = S - B[0] = S - (S - 1 - A[0]) = 1 + A[0].
                // But we need to check if B is reachable.
                // Actually, if A1 reaches B1, then A2 reaches B2 automatically?
                // If A -> (y-1, x+1).
                // A_new[0] = A_old[1] - 1.
                // A_new[1] = A_old[0] + 1.
                // If A_old[0] maps to B[0], then A_old[1] must map to B[1].
                // Let's check consistency.
                // If A0 = B0, done.
                // If A0 != B0, we need transition.
                // If B0 = S-1-A0, then we need one op.
                // Does that op produce B?
                // Apply op (0, 1). A'[0] = A[1] - 1.
                // We need A[1] - 1 = B[0] => A[1] = B[0] + 1.
                // Also A'[1] = A[0] + 1.
                // We need A[0] + 1 = B[1].
                // Consistency check:
                // A[0] + A[1] = S. B[0] + B[1] = S.
                // A[1] = S - A[0].
                // Need S - A[0] = B[0] + 1 => S - B[0] - 1 = A[0].
                // And B[0] = S - 1 - A[0]. Matches.
                // Also need B[1] = A[0] + 1.
                // B[1] = S - B[0] = S - (S - 1 - A[0]) = 1 + A[0]. Matches.
                // So if B[0] == S-1-A[0], then B is exactly the result of 1 op.
                // What if B[0] is neither?
                // Then B is not reachable.
                possible = true; 
                // Wait, checking if reachable:
                // It is reachable iff B0 in {A0, S-1-A0}.
                // But wait, I need to handle the case where B[0] == A[0] implies possible=true.
                // And B[0] == S-1-A[0] implies possible=true (distance 1).
                // Is it possible distance is > 1?
                // No, x_next = S-1-x. Two steps restore x.
                // So distance is at most 1.
                // Unless A[0] == B[0] (dist 0).
                // Wait, check logic again.
                // x -> S-1-x -> S-1-(S-1-x) = x.
                // Cycle length 2.
                // So B[0] must be either A[0] or S-1-A[0].
                // BUT we also need to check if the resulting values are correct?
                // If B[0] == A[0], then B[1] == A[1] (sums match), dist 0.
                // If B[0] == S-1-A[0], then B[1] must match A[1]'s transform.
                // A'[1] = A[0] + 1.
                // We need B[1] == A[0] + 1.
                // Since B[0] + B[1] = S = A[0] + A[1].
                // If B[0] = S - 1 - A[0].
                // Then B[1] = S - (S - 1 - A[0]) = A[0] + 1.
                // So it is consistent.
                
                if (possible) {
                    if (B[0] == A[0]) {
                        cout << "Yes" << endl;
                        cout << 0 << endl;
                    } else {
                        cout << "Yes" << endl;
                        cout << "1" << endl;
                        cout << "1 2" << endl;
                    }
                    return 0;
                }
            }
        } else {
            // Wait, if B0 == S-1-A0, possible is true.
            // The check was inside.
            // My logic: if B0 == A0 || B0 == S-1-A0, then possible=true?
            // Actually, is it possible that B0 == S-1-A0 but B is NOT reachable?
            // As shown above, B[1] is automatically consistent.
            // Wait, what if A0 + A1 = S but A1 < 0?
            // Constraints say A_i, B_i >= 1.
            // So valid states only.
            // So yes, condition holds.
            
            // Wait, earlier I said (2, 3) -> (3, 2) impossible.
            // A=[2, 3]. S=5. A0=2.
            // S-1-A0 = 5-1-2 = 2.
            // B=[3, 2]. B0=3.
            // 3 is NOT in {2}. So impossible.
            // My logic: 3 != 2 and 3 != 2. Correct.
            
            // Another example: A=[1, 4]. S=5. A0=1.
            // S-1-A0 = 5-1-1 = 3.
            // B=[3, 2]. B0=3.
            // 3 in {1, 3}. Possible.
            // So logic holds.
            
            // Re-write cleanly:
            int compA[2];
            compA[0] = A[0];
            compA[1] = s - 1 - A[0];
            if (B[0] == compA[0] || B[0] == compA[1]) {
                 if (B[0] == A[0]) {
                     cout << "Yes\n";
                     cout << "0\n";
                 } else {
                     cout << "Yes\n";
                     cout << "1\n";
                     cout << "1 2\n";
                 }
            } else {
                 cout << "No\n";
            }
            return 0;
        }
    }

    // N >= 3
    // Strategy: Local Search with Randomness
    vector<pair<int, int>> ops;
    vector<int> curA = A;
    auto best_sol = ops; // Store ops? No, just ops vector.
    
    // We can perform random restarts if needed.
    // Limit iterations.
    
    // Shuffle initial state? A is fixed.
    // We can add some random noise? No.
    
    // Iteration limit
    int iter_limit = 10000;
    
    // To avoid getting stuck, we use Simulated Annealing parameters implicitly?
    // Or just pick best neighbor. If no better, pick random.
    
    long long current_cost = calc_diff(curA, B);
    vector<pair<int, int>> final_ops;
    
    // Optimization: We want to minimize diff.
    // If stuck, maybe we can perform random moves.
    
    // Let's try to fix elements greedily first?
    // Actually, the random local search is robust.
    
    // Wait, we need to print the ops.
    // Let's keep track of ops.
    // However, storing ops and re-applying every iteration is slow?
    // N=100. Reapplying is O(N^2).
    // 10000 * 5000 ops is too much? 50 million ops. Feasible in 2s.
    // But we should be smarter.
    // Keep current state A.
    
    // Let's use a timer.
    // Reset if no improvement for 100 steps?
    
    // Since time limit is generous, let's use a robust search.
    // Randomized Local Search.
    
    while(iter_limit--) {
        // Find best move
        long long best_gain = -1; // Max reduction. -1 indicates no improvement found yet.
        int best_i = -1, best_j = -1;
        
        // Check all pairs
        // We want to reduce sum(|A_k - B_k|).
        // Precompute deltas?
        // Calculating delta for each pair takes O(1) if we just simulate.
        // Naive simulation takes O(N) to recalc cost.
        // Total cost per step O(N^3).
        // 100^3 = 1,000,000. Times 10000 steps = 10^10. Too slow.
        // Need O(N^2) per step.
        // Delta can be computed in O(1).
        
        // Cost is sum(abs(A[k]-B[k])).
        // Changing A[i], A[j] affects only these two.
        // New cost = Old cost - |A[i]-B[i]| - |A[j]-B[j]| + |(A[j]-1)-B[i]| + |(A[i]+1)-B[j]|.
        
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                int ai = curA[i];
                int aj = curA[j];
                int bi = B[i];
                int bj = B[j];
                
                // Calculate contribution
                // We assume i < j always for the op definition?
                // Problem says choose pair (i, j) with 1 <= i < j <= N.
                // So we only consider pairs (i, j) with i < j.
                // But indices are 1..N.
                // Vector is 0..N-1.
                
                // Simulate
                // We need to maximize gain (reduction).
                // Reduction = (|ai-bi| + |aj-bj|) - (|(aj-1)-bi| + |(ai+1)-bj|).
                
                // If we want to increase ai (when ai < bi):
                // We need aj > ai.
                // If we want to decrease ai (when ai > bi):
                // We need aj <= ai.
                
                // Let's just compute reduction.
                int val_i_new = aj - 1;
                int val_j_new = ai + 1;
                
                int cost_curr = abs(ai - bi) + abs(aj - bj);
                int cost_new = abs(val_i_new - bi) + abs(val_j_new - bj);
                int diff = cost_curr - cost_new;
                
                if (diff > best_gain) {
                    best_gain = diff;
                    best_i = i;
                    best_j = j;
                } else if (diff == best_gain && diff < 0) {
                   // If best_gain is negative, we prefer less negative?
                   // Actually we want to minimize loss.
                   // But we are looking for best.
                }
            }
        }
        
        if (curA == B) break;
        
        if (best_gain > 0) {
            // Greedy step
            int i = best_i;
            int j = best_j;
            curA[i] = curA[j] - 1;
            curA[j] = (int)curA[i] + 1; // Wait, old A[i] is lost?
            // Store old values.
            // Ah, applying op: A_i' = A_j - 1, A_j' = A_i + 1.
            // My code updates sequentially, need to store old.
            // Wait, I didn't store `ai` correctly in loop?
            // I used `ai` and `aj` from `curA`.
            // So `ai` and `aj` are fresh for each pair.
            // But inside loop, I don't update `curA` yet.
            // I only record `best_i`, `best_j`.
            // So I need to read `curA[best_i]` and `curA[best_j]` NOW.
            // But wait, I need to update `curA`.
            
            // To be safe and efficient:
            // Update `curA` directly using stored vars.
            // But `curA` changes.
            // I should do this update inside the main loop only once.
            // But my loop computes diffs.
            // I must be careful not to mutate `curA` inside the loop over pairs.
            // So I need to copy `curA` or just access safely.
            
            // Re-calculate best move properly.
            // Let's just store best move.
        } else if (best_gain < 0) {
            // No move improves.
            // We must pick a move that hurts the least.
            // Or random move.
            // Let's pick the move with max(best_gain) (which is negative, so least negative).
            // Actually, let's pick RANDOM move to explore.
            // Random is better to escape local optima.
            // Wait, if we just pick random, we might loop forever.
            // Let's pick the one with MAX diff (closest to 0).
            
            // Actually, if best_gain == 0, we are in a plateau.
            // Pick random pair.
            // If best_gain < 0, pick the one with highest value (min damage).
            
            // Let's implement "Steepest Descent" for < 0.
            // But for plateau, random.
            // If no move improves, and we are stuck (cost same for all?), pick any?
            // If all moves have gain < 0, it means any move increases error.
            // We MUST take one.
            // If there are moves with gain 0, take those (plateau).
            // If all negative, take max(-ve).
            
            // Let's refine:
            // If best_gain < 0, take max(best_gain).
            // Wait, I initialized best_gain = -infinity? Or 0.
            // If I init with -1, then if all gains < 0, best_gain will be negative.
            // Correct.
            
            // But for random exploration, maybe I should pick completely random if best_gain == 0?
            // If best_gain < 0, maybe we are forced to climb.
            // To prevent climbing high, we should pick minimal loss.
            
            // However, there's a catch.
            // If I blindly follow "max diff", I might go down a slope.
            // If all moves increase error, I take the "cheapest" increase.
            // This is a standard strategy (Gradient ascent on inverse of error? No, just steepest descent).
            
            // Wait, I should reset `final_ops` if I pick a "backwards" move?
            // No, I need the sequence.
            // Wait, if I pick a "bad" move, my path is longer.
            // But I want min ops.
            // If I get stuck, my strategy fails to find min ops.
            // But given N=100, maybe I can find a solution without "bad" moves?
            // Assuming connectivity and no local minima for this specific error metric?
            // Unlikely.
            // But maybe "bad" moves are necessary.
            
            // I will implement a version that prefers improvements.
            // If no improvement, picks random move.
            // Random move probability increases if stuck?
            // Let's just implement: if best_gain <= 0 (meaning <= 0), pick random?
            // No, if best_gain < 0, maybe best_gain is the only choice.
            
            // Let's just use a simple heuristic:
            // Iterate pairs, calculate diff.
            // Sort by diff.
            // If max diff > 0, pick that.
            // Else pick random.
            
            // Wait, sorting 5000 items is fast.
            // But creating vectors for sorting is overhead.
            // Just find max.
        }
    }
}
```

Wait, my logic for "random move if stuck" is weak.
Better: use "Tabu search" or just restart.
Given the time limit and scoring, maybe I can run 5 iterations of random restarts.
In each iteration:
1. Start with `A`.
2. Do local search (greedy) for 5000 steps.
   - Always pick pair that improves cost.
   - If tie, pick random.
   - If no improvement found (all pairs worsen), stop this attempt (assume stuck) or pick "least worst".
     Actually, if no improvement, pick the pair with max cost reduction (min increase).
     If we keep doing this, we might converge to a bad local min.
3. Record the cost reached. Keep the solution with lowest cost and min length?
   Wait, cost is not length.
   Length is number of ops.
   Greedy usually yields decent length.
   
   Let's refine:
   If cost reduces, record move.
   If we reach 0 cost, stop and return ops.
   If we hit limit or stuck, break.
   Repeat restarts.
   
   To avoid TLE, limit steps per restart.
   
   One critical optimization:
   Don't re-evaluate cost from scratch.
   Update cost incrementally.
   Maintain `current_cost`.
   Inside loop, pick `i, j` maximizing `gain`.
   Update `current_cost` and `A`.
   
   Wait, picking best gain might lead to cycles or oscillation?
   Not really, it descends cost.
   
   Let's code this.
   Note: The problem requires *minimum* operations.
   If my greedy fails to find optimal, I might get partial score or 0.
   But it's the best I can do generically.
   
   Wait, what if I sort the target indices by deficit/surplus?
   No, values matter.
   
   Let's stick to greedy hill climbing with random restarts.
   Also, handle `No` case for N=2 properly.
   
   One small correction:
   My condition for N=2 was: `B[0] in {A[0], S-1-A[0]}`.
   Wait, if `B[0] == A[0]`, we output 0 ops.
   If `B[0] == S-1-A[0]`, we output 1 op `(1 2)`.
   Is it possible `B[0]` matches but we need more ops?
   No, because for N=2, max distance is 1.
   Wait, what if `A` is `[100, 100]` and `B` is `[50, 150]`. Sum 200.
   `A[0]=100`. `S-1-A[0] = 199-100 = 99`.
   `B[0] = 50`.
   50 not in {100, 99}. Impossible.
   Wait, is it really impossible?
   $A=(100, 100)$.
   Op: $99, 101$.
   Op: $100, 100$.
   So only $(100, 100)$ and $(99, 101)$ are reachable?
   Wait, what about $(101, 99)$?
   If we start with $(100, 100)$, we can't reach $(101, 99)$ directly?
   $A=(100, 100)$. $i=1, j=2$.
   $A_1 \leftarrow 100-1=99$. $A_2 \leftarrow 100+1=101$.
   So we get $(99, 101)$.
   From $(99, 101)$, apply op.
   $A_1 \leftarrow 101-1=100$. $A_2 \leftarrow 99+1=100$. Back.
   So for $(100, 100)$, reachable states are $\{(100, 100), (99, 101)\}$.
   So yes, unreachable.
   So condition holds.
   However, what if $B[0] == S-1-A[0]$ but indices swapped?
   $B$ could be $(A[1], A[0])$?
   Wait, $B$ is a sequence.
   In $N=2$, if $A=(100, 100)$, $B=(99, 101)$.
   $S=200$. $A[0]=100$. $S-
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=996)

````
I see two critical mistakes in my previous reasoning. Let me correct them:

1. **The operation is NOT a swap**: The problem states $A_i \leftarrow A_j - 1$ and $A_j \leftarrow A_i + 1$. This preserves $A_i + A_j = (A_j - 1) + (A_i + 1) = A_i + A_j$, so the sum of any pair doesn't change, meaning the **total sum** is conserved. So if $\sum A \neq \sum B$, it's impossible. I previously made up arbitrary "swap" examples which were wrong.

2. **The direction matters**: Since the operation only applies when $i < j$ (and specifically the formula uses $A_j - 1$), there's actually an implicit ordering constraint. For $A_0 = 5$ to become $B_0 = 4$, we need $A_i + A_j = 9$ with target values $4+4=8$, requiring $A_i = 3.5$—impossible with integers! Similarly for $A_3 = 4 \to B_3 = 3$, $A_i+A_j = 7$ needs target $3+3=6$, giving $A_i = 2.5$. My previous claim that swaps could handle this was nonsense because these required half-values.

The real question is whether the operation constraints allow reaching the target configuration at all, given that each operation conserves $A_i + A_j$. Since the total sum must match ($\sum A = \sum B$) AND the integer divisibility constraint on pairs must hold, let me verify if Example 1 even satisfies these necessary conditions before claiming it's achievable.
</think>

```cpp
#include <iostream>
#include <vector>
#include <numeric>
using namespace std;

int main() {
    int n;
    cin >> n;
    vector<int> a(n), b(n);
    for (int i = 0; i < n; ++i) cin >> a[i];
    for (int i = 0; i < n; ++i) cin >> b[i];
    
    // Operation: (Ai, Aj) -> (Aj-1, Ai+1), i<j
    // Sum is preserved: (Aj-1)+(Ai+1) = Ai+Aj
    
    auto possible = [&](const vector<int>& A, const vector<int>& B) -> bool {
        // Check sum equality
        long long sum_a = accumulate(A.begin(), A.end(), 0LL);
        long long sum_b = accumulate(B.begin(), B.end(), 0LL);
        if (sum_a != sum_b) return false;
        
        // Greedy with min-heap of differences needed from smallest index
        // At step i, diff[i] = bi - ai (positive means need to add to ai)
        // Operation (i,j): moves +1 from j to i
        // We need net flow d[j] >= diff[j] at each step
        
        priority_queue<int, vector<int>, greater<int>> pq; // heap of indices by current diff
        for (int i = 0; i < n; ++i) 
            pq.push({i, max(0, b[i] - a[i])}); // positive diffs only, indices don't matter
        
        while (pq.size() > 1) {
            // Take two smallest non-zero diffs (most urgent)
            auto& x = pq.top(); pq.pop(); int xi = x.first; int xi_val = x.second;
            auto& y = pq.top(); pq.pop(); int yi = y.first; int yi_val = y.second;
            
            // Can we satisfy both? Need xi_val + yi_val <= remaining capacity of other nodes
            if (xi_val == 0 || yi_val == 0) continue; // skip zeros
            
            // Check if total needed fits within remaining capacity
            int total_needed = xi_val + yi_val;
            int total_remaining = 0;
            for (auto& z : pq) {
                if (z.second > 0) total_remaining += z.second;
            }
            if (total_remaining < total_needed) return false; // not enough room
            
            pq.push({xi, xi_val - yi_val}); // xi gets reduced by yi, but may still be positive
            pq.push({yi, 0}); // yi satisfied, becomes zero
        }
        return true;
    };
    
    cout << (possible(a, b) ? "Yes" : "No") << endl;
}
```
````
