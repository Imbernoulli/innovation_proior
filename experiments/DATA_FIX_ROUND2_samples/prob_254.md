# 题 254 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The user wants a solution for the "Pepe Racing" problem.
**Problem Summary:**
- There are $N^2$ distinct pepes with speeds.
- One query consists of picking $n$ distinct pepes and finding the fastest among them.
- The goal is to identify the set of the top $N^2 - N + 1$ fastest pepes.
- The slowest $N-1$ pepes are indistinguishable from each other (their relative order doesn't matter, just that they are the bottom).
- We need to output these top $N^2 - N + 1$ labels in descending order of speed.
- $t$ test cases ($t \le 10^4$).
- $n \le 20$.
- $\sum n^3 \le 3 \cdot 10^5$.
- Adaptive interactor.

**Key Constraints & Goal:**
- Time limit: 5s.
- Memory limit: 256 MB.
- Number of queries matters for scoring.
- Output the list of IDs sorted by speed (fastest first).
- Since we only care about the *top* $K = N^2 - N + 1$ elements, we don't necessarily need to fully sort all $N^2$ elements, just filter out the worst $N-1$. Actually, wait. If there are $N-1$ slowest, they are indistinguishable. Does it mean their values can be anything? No, they are distinct speeds. It means if we find a set of pepes, we need to report the ones that *could* be the top $K$. But actually, since the interactor is adaptive, we need to find a configuration that is consistent.
- Wait, the output format says "! p_1 ... p_{n^2-n+1}". These must be the labels of the actual fastest pepes.
- So we need to eliminate at least $N-1$ candidates. Or rather, find the subset of size $N^2 - N + 1$ that contains the fastest. Since there are exactly $N^2$ items and we remove the bottom $N-1$, we keep $N^2 - (N-1) = N^2 - N + 1$.
- Effectively, we want to find the "losers". There are $N-1$ losers.
- Each race takes $n$ participants and returns the winner. This means $n-1$ losers per race.

**Strategy:**
We have $M = n^2$ items. We want to identify the top $n^2 - n + 1$. Equivalently, identify a subset of $n-1$ items that are slower than everyone else in the final answer set.
Actually, the easiest way to think about this is finding a tournament structure or elimination.
Consider a standard sorting network approach or tournament selection.

Let's look at the numbers. $N$ is small ($N \le 20$). Total items $M = N^2 \le 400$.
A simple tournament between all pairs? No, groups of $n$.
If we treat the items as nodes in a graph where an edge $(u, v)$ exists if $u$ wins against $v$. But here comparisons are in batches of $n$.
However, notice that if $u$ beats a group containing $v$, then $u$ is faster than $v$.

Since we want to remove the slowest $n-1$ elements, maybe we can construct a specific graph.
Consider a cycle or something?
Actually, $n$ is fixed for all queries.
Let's divide the $n^2$ elements into $n$ groups of size $n$. Let these groups be $G_1, G_2, ..., G_n$.
In the first round, run races on each group $G_i$. Let the winner of $G_i$ be $w_i$.
These winners $w_1, ..., w_n$ are faster than the other $n-1$ elements in their respective groups.
So we know that for any $x \in G_i \setminus \{w_i\}$, speed($x$) < speed($w_i$).
But we don't know the relation between different groups yet.
However, consider that we have eliminated everyone except $\{w_1, ..., w_n\}$.
Wait, we eliminated $n(n-1)$ elements.
The remaining $n$ elements are candidates.
Is it enough?
We need to keep $n^2 - n + 1$ elements.
Currently we have identified $n^2 - n(n-1) = n$ candidates? No.
Total elements $n^2$.
In each group of size $n$, we discard $n-1$ elements because the winner is faster than them.
If we do this for $n$ disjoint groups, we discard $n(n-1)$ elements.
Remaining candidates: $n$.
But we need to keep $n^2 - n + 1$ elements.
$n^2 - n + 1 - n = n^2 - 2n + 1 = (n-1)^2$.
This seems inverted. We want to discard $n-1$ elements, not $n(n-1)$.
Discarding $n(n-1)$ is too many. We might discard valid fast players who lost to someone in the same group but could still be faster than others?
No, if A is the winner of a race against B, then A > B. We know A is strictly faster than B.
If we have a candidate set of size $S$, and we find a winner $W$, $W$ remains, $Others$ remain.
Wait, if we declare the $n^2 - n + 1$ fastest, that implies the remaining $n-1$ are the slowest.
This means every element in the kept set is faster than every element in the discarded set?
Not necessarily pairwise. Just that the minimum of the kept set is greater than the maximum of the discarded set?
Yes. If we identify the set of $n^2 - n + 1$ fastest, let this set be $T$.
Then $\min(T) > \max(S)$ where $S$ is the complement set of size $n-1$.
So we need to find a partition of indices into sets $F$ (kept) and $L$ (lost) such that for all $f \in F, l \in L$, speed(f) > speed(l)?
Actually, the problem asks to order the $F$ part. It doesn't ask us to prove $L$ is slower than $F$. But it implies that if we report a set, it must be correct.
If we pick an arbitrary subset of size $n^2 - n + 1$ and output it, we risk being wrong. We must deduce the true order.

Let's re-read carefully: "Can you order the n^2 - n + 1 fastest pepes?"
Note that the slowest n - 1 pepes are indistinguishable from each other.
So if we correctly identify the set of top $n^2 - n + 1$ people, and sort them internally, we get full points. The bottom $n-1$ don't matter which are them, just that they are not in our answer. And since they are indistinguishable, we don't need to know their relative order or even their labels precisely (though we must exclude them).
Actually, if we include one slow pepe in our answer, we might be right if the interactor allows it?
No, "relative speeds ... may depend on your queries".
Wait, "at any moment there is at least one initial configuration ... consistent".
This is standard adaptive problem text.
So we assume worst-case behavior within consistency.
Basically, we need to prove that no element in our output could be slower than an element not in our output.
This means for every pair $(u, v)$ where $u \in F$ (output) and $v \notin F$ (discarded), we must have proven $u > v$ or something similar?
Actually, if we just find $n-1$ elements and say "everyone else is faster", we are good.
How to find $n-1$ elements that are definitely slower than everyone else?
Or rather, find $n-1$ candidates that cover the "bottom".

Let's try to reduce the problem space.
We have $n^2$ items. We want to find the "winner" among the whole set repeatedly?
Maybe we can build a tournament tree?
With $n$ participants in a race, we get 1 winner.
If we arrange everyone in a grid $n \times n$.
Indices $(r, c)$ for $r, c \in \{1..n\}$.
Group $(r, \text{all } c)$ into $R_r$ (row $r$). Group $(\text{all } r, c)$ into $C_c$ (col $c$).
Race the rows. In race $i$, participants are $\{(i, 1), ..., (i, n)\}$. Winner is $w_i$.
We win $\{w_1, ..., w_n\}$. Everyone else in row $i$ is slower than $w_i$.
Race the columns using winners? No, columns need $n$ members.
We have $n$ winners from rows. They are distinct.
What about the losers in the rows?
For row $i$, we have loser set $L_i = \{ (i, j) | j \neq \text{pos}(w_i) \}$. Size $n-1$.
There are $n$ such rows. Total losers $n(n-1)$.
Winner set $\{w_1, ..., w_n\}$ has size $n$.
We haven't compared rows against each other.
Suppose we compare row winners.
But we have $n$ row winners. We can put them in one big race? No, max size is $n$. So yes, we can put all $n$ winners in one race.
Query: ? $w_1 \dots w_n$.
Result: one winner, say $w_{k}$ is the global winner of these $n$.
So $w_k$ is faster than $w_j$ for all $j \neq k$.
Also $w_k$ was faster than all its column mates in the original row comparison? No, row comparison tells us $w_k$ is faster than other $n-1$ in row $k$.
So now we have established a partial order.
But we need to handle all $n^2$ elements.
This looks like a transitive reduction problem.

Actually, there is a known constructive algorithm for this.
The total number of queries should be related to $O(n^3)$. The time limit allows it.
Sum of $n^3 \le 3 \cdot 10^5$.
Wait, $n \le 20$, so $n^3 \approx 8000$. With sum constraint, we can afford roughly $n^3$ operations per test case.
We can perform more than $n$ races?
Yes.
Let's formalize. We need to select the $n^2 - n + 1$ fastest.
This means excluding $n-1$ slowest.
Think of the $n^2$ positions as a square matrix.
Row $i$: $(i, 1), (i, 2), \dots, (i, n)$.
Col $j$: $(1, j), (2, j), \dots, (n, j)$.

Phase 1: Race all rows.
Queries: $n$ races. One for each row $i=1 \dots n$.
From row $i$, we get a winner $u_i$. All other $n-1$ in row $i$ are slower than $u_i$.
Let $S$ be the set of all items except the losers of row-races.
The losers are $n(n-1)$ items.
The winners are $n$ items: $\{u_1, \dots, u_n\}$.
Now we need to rank the winners to determine the best overall.
Wait, simply winning a row doesn't make someone globally fast.
Example: $n=2$. Pepes 1,2,3,4.
Row 1: {1, 2}. Winner $u_1$. Loser $l_1$.
Row 2: {3, 4}. Winner $u_2$. Loser $l_2$.
Now we race $u_1, u_2$. Winner $w$, loser $l'$.
So $\{w, u_2 \text{ or } u_1, ...\}$.
We know $w$ is faster than $u_1$ and $u_2$?
Yes, $w \in \{u_1, u_2\}$, so $w$ is the fastest among them.
Does $w$ beat everyone?
If $w=u_1$, then $u_1 > u_2$. Also $u_1 > l_1$ (from row 1).
Is $u_2 > l_2$? Yes, from row 2.
So we know $u_1 > l_1, u_1 > u_2 > l_2$.
So $u_1$ is the absolute fastest.
Who else is safe?
$l_2$ is slower than $u_2$ and $u_1$. So $l_2$ is definitely not in top 3?
Wait, we need top $n^2 - n + 1 = 4 - 2 + 1 = 3$ elements.
We need to discard 1 element.
Here $l_2$ is slower than $u_2$ and $u_1$. Is it slower than $l_1$? Not necessarily.
If we discard $l_2$, we keep $\{u_1, u_2, l_1\}$.
Is $\{u_1, u_2, l_1\}$ guaranteed to contain the top 3?
We know $u_1 > u_2 > l_2$. Also $u_1 > l_1$.
Possible orders compatible:
1. $u_1 > l_1 > u_2 > l_2$ -> Top 3: $u_1, l_1, u_2$. Set $\{u_1, u_2, l_1\}$ matches.
2. $u_1 > u_2 > l_1 > l_2$ -> Top 3: $u_1, u_2, l_1$. Matches.
So discarding $l_2$ works?
What if $l_2$ is actually faster than $l_1$?
Wait, we don't have direct evidence $l_2 < l_1$ or $l_2 < u_1$ etc.
We have chain $u_1 > l_1$ and $u_2 > l_2$ and $u_1 > u_2$ (assuming $w=u_1$).
Transitivity: $u_1 > u_2 > l_2$.
Is it possible that $l_2$ is faster than $l_1$? Yes.
Is it possible $l_2$ is faster than $u_2$? No, $u_2$ won row 2.
Is it possible $l_2$ is faster than $u_1$? No, $u_1$ won grand race.
So $l_2 < u_1$ and $l_2 < u_2$.
Could $l_2$ be the fastest? No.
Could $l_2$ be in the top 3?
The candidates are $\{u_1, u_2, l_1\}$. Plus $l_2$.
We need to choose 3 out of 4.
We know $l_2$ loses to both $u_1$ and $u_2$.
This suggests $l_2$ is likely "bad".
Specifically, $l_2$ is beaten by $u_1$ and $u_2$.
So there are at least 2 people faster than $l_2$.
Therefore, $l_2$ cannot be in the top 2.
Since $n=2$, we need top $4-2+1=3$.
Can $l_2$ be 3rd?
Yes, if $u_1, u_2$ are 1st and 2nd.
So we MUST NOT discard $l_2$ blindly.
However, the example output says `! 2 3`.
Input $n=2$. Items 1, 2, 3, 4.
Output 2, 3. Wait, 2 items?
Ah, the example input line says `1` test case. Then `2` for n.
Example input:
```
1
2
2
4
4
3
2
```
Wait, parsing example input:
Line 1: `t = 1`.
Test Case 1: `n = 2`.
Wait, interaction protocol says `After reading the integer n... begin interaction.`
Example Output shows queries.
The Example Input provided in problem description seems weirdly formatted or I'm misinterpreting it.
Usually interactive problems provide answers in the stream.
The "Example Input" block usually shows what stdin contains and "Example Output" shows stdout.
But this includes responses `2`, `4`, `3`, `2`. These must be responses `p` from queries.
Let's trace:
User: `n = 2`.
Query 1: `? 1 2`. Interactor replies `2`. (Means 2 is faster than 1).
Query 2: `? 3 4`. Interactor replies `4`. (4 is faster than 3).
Query 3: `? 2 4`. Interactor replies `3`. (Wait, response is `3`. User asked `{2, 4}`. Who is returned? Fastest is 3?
Impossible, 3 is not in the query `{2, 4}`.
Maybe the example input/output block provided in the prompt is just illustrative and slightly inconsistent or I am misreading it.
"Example Input":
```
1
2
2
4
4
3
2
```
Interpretation:
`t=1`.
Case 1: `n=2`.
Response 1 (`2`): Result of query `? 1 2`. Winner is 2.
Response 2 (`4`): Result of query `? 3 4`. Winner is 4.
Response 3 (`3`): Result of query `? 2 4`. Winner is... 3? But 3 was not queried.
Wait, "After each race, you will only know the fastest pepe of these n pepes."
The input lines `2`, `4`, `3` correspond to responses.
Ah, the third query must involve `3`?
In Example Output: `? 2 4`.
The response is `3`.
This implies `3` was somehow involved? No.
If I ask `? 2 4`, I get either 2 or 4.
Why does the example say `3`?
Wait, maybe the Example Input block includes all interactions for multiple cases?
`1` testcase. `n=2`.
Interaction starts.
My code prints `? 1 2`. Interactor sends back `2`. (Stored in input stream?)
The example input shows `2`. This matches.
Next my code `? 3 4`. Interactor sends back `4`. (Matches second line).
Next my code `? 2 4`. Interactor sends back `3`. (Mismatch! 3 not in {2, 4}).
Wait, is the example output actually: `? 1 2 \n ? 3 4 \n ! 2 3`?
And the inputs are responses?
Let's check the example outputs again.
Output:
```
? 1 2
? 3 4
? 2 4
! 2 3
```
Wait, `! 2 3` lists 2 pepes. For `n=2`, output length is $n^2 - n + 1 = 4 - 2 + 1 = 3$.
So `! 2 3` is WRONG length.
Maybe the example input provided in the prompt description is flawed or I am missing something fundamental.
Ah, looking at Codeforces Problem 1901E? No, "Pepe Racing" sounds like Codeforces Round #913 (Div. 3) Problem E? Or similar.
Found similar problem: Codeforces 1761D? No.
Let's search for "Pepe Racing". It corresponds to Codeforces 1963C? No.
It is likely Codeforces 1966D? No.
Actually, searching for "Pepe Racing" leads to Codeforces Global Round 25 Problem C? Or something.
Found: Codeforces 1796 D? No.
Codeforces 1915 E?
Let's not rely on external search. Let's analyze the logic myself.
The problem is from Codeforces Round 936 (Div. 2), Problem D? Or Div 3 E?
Actually, it is "Pepe's Speeds".
Wait, the constraints are specific. Sum of $n^3$.
This implies an algorithm depending on $n$.
Let's assume the provided example has a typo or I should ignore the exact response values and focus on the protocol.
Wait, if I print `? 2 4` and get `3`, the system would reject me because 3 is not in `{2, 4}`.
Unless... the example output in the prompt description is messed up.
Wait, `? 2 4` -> Response `3`. Maybe it meant `? 2 3 4`?
If $n=2$, a race has 2 peeps. `? x1 x2`.
If example output has `? 2 4`, and response `3`, this is invalid.
Perhaps the Example Input provided in the text is for a DIFFERENT problem or corrupted.
I should trust the problem statement rules over the example data if it conflicts.
Rules:
- Input `n` ($2 \le n \le 20$).
- Query `? x_1 ... x_n`.
- Response `p` ($1 \le p \le n^2$, $p \in \{x_i\}$).
- Output `! p_1 ... p_{n^2-n+1}`.

Okay, let's derive a strategy that fits constraints.
We need to eliminate $n-1$ elements.
We can maintain a set of potential candidates. Initially size $n^2$.
We want to reach size $n^2 - n + 1$.
Strategy Idea:
1. Divide $n^2$ items into $n$ blocks of size $n$. (Rows).
   Run a race for each block. We get $n$ winners.
   Eliminate the $n(n-1)$ losers? No, we can't eliminate them yet.
   We know for each block $B_i$, $winner(B_i)$ is better than others in $B_i$.
   
   Let's denote the blocks as $B_1, \dots, B_n$.
   Winners $w_1, \dots, w_n$.
   
   Now we have $n$ winners. We can race them.
   Since $n$ items fit in a race, we put all $w_i$ in one race.
   Winner is $w_{global}$. Loser is some $w_{loser}$.
   Now $w_{global}$ is faster than all $w_{others}$.
   Specifically $w_{global} > w_k$ for all $k$.
   So $w_{global}$ is faster than every single item in its own row (since it won the row race too, wait).
   Wait, the race on winners uses items $w_1, \dots, w_n$.
   So $w_{global}$ beats everyone else in the winner set.
   Does $w_{global}$ beat others in its block?
   It did participate in the block race where it was the winner.
   Wait, if it won the block race, it is faster than all other $n-1$ items in that block.
   So $w_{global}$ is faster than all $n-1$ items in block $j$ AND all $w_k$ for $k \neq j$.
   Thus $w_{global}$ is faster than all other winners AND all non-winners in block $j$.
   What about non-winners in other blocks $k \neq j$?
   We know $w_{global} > w_k$.
   We know $w_k >$ others in block $k$.
   So by transitivity $w_{global} >$ others in block $k$.
   Conclusion: $w_{global}$ is faster than EVERYTHING EXCEPT possibly... wait.
   If $w_{global}$ is faster than all $w_k$ (for $k \neq j$) and all non-winners in block $j$ and block $k$.
   Then $w_{global}$ is faster than ALL other $n^2 - 1$ items.
   So $w_{global}$ is the unique fastest.
   
   Wait, is this always true?
   Block races were conducted on the original $n$ groups.
   We picked a winner from each group.
   Then we raced the winners.
   The global winner of the winners is indeed the fastest among ALL $n^2$ items.
   Why?
   Let $x$ be any item. $x$ belongs to some block $B_k$.
   $x \in B_k$.
   If $x = w_k$, then in the winners race, $w_k$ lost to $w_{global}$. So $w_{global} > w_k$.
   If $x \neq w_k$, then in block race $B_k$, $w_k$ won against $x$. So $w_k > x$.
   Combined with $w_{global} \ge w_k$, we get $w_{global} \ge w_k > x$.
   Wait, strictly speaking $w_k > x$. So $w_{global} > x$ holds.
   So $w_{global}$ is strictly the fastest.
   
   Okay, so we can identify the single fastest Pepe.
   But we need to order the top $n^2 - n + 1$.
   That means we need to find $n^2 - n$ items that are NOT the slowest.
   Or simply eliminate $n-1$ items.
   
   With the above, we found 1 item that is definitely the fastest.
   We also know relations.
   In the winners race (size $n$), we found the winner and $n-1$ losers.
   For the losers $l_1, \dots, l_{n-1}$ in the winners race:
   They are slower than $w_{global}$.
   But they might be faster than others?
   Actually, the losers in the winners race are the winners of their respective blocks.
   Let's call the race of winners "Race 0".
   Participants: $w_1, \dots, w_n$.
   Winner: $W$.
   Losers: $L_W = \{ w_i \mid w_i \neq W \}$.
   
   We established $W$ is faster than everyone in its block (call it $Block_W$).
   And $W$ is faster than everyone in other blocks' winners $w_k$.
   So $W$ is faster than everyone in other blocks via transitivity ($w_k >$ rest of block).
   So $W$ is the MAX.
   
   What about the other items?
   Consider $L_W$. These are winners of some blocks (where they lost to $W$).
   Consider the losers of those blocks.
   Basically, we have a hierarchy.
   
   This logic finds 1 element.
   Can we generalize?
   Instead of grouping arbitrarily, we can structure it such that we find "layers" of winners.
   If we view the $n^2$ items as a grid.
   Layer 1: The winners of all rows. (Size $n$).
   Layer 2: The winners of these $n$ items. (Size 1).
   This is what I did.
   
   But we need to return a list.
   Wait, the question asks to return the *order*.
   "Order the $n^2 - n + 1$ fastest pepes".
   Descending order.
   So we need to sort the large set.
   Since $n$ is small, maybe we can construct the full order?
   No, $n^2$ can be 400. Sorting 400 elements requires many comparisons.
   We are allowed to use $O(n^3)$ queries.
   Maybe $O(n^2)$ comparisons?
   
   Alternative idea:
   Use the property that we can compare $n$ items.
   This is effectively a comparison oracle that takes $n$ inputs and returns the min.
   We want to sort $n^2$ items.
   With an oracle that sorts $n$ items and gives us the min, we can simulate a standard sorting network or merge sort.
   
   Wait, the "indistinguishable" part means we just need the set of top $N^2 - N + 1$.
   We need to identify the bottom $N-1$.
   
   Let's revisit the "rows and columns" idea.
   Grid $N \times N$.
   Row winners: $R_1, \dots, R_N$.
   Column winners: $C_1, \dots, C_N$.
   To get $R_i$, race row $i$. Cost $N$ races.
   To get $C_j$, race col $j$. Cost $N$ races.
   Total $2N$ races.
   Now we have winners $R_i$ (who beat row losers) and $C_j$ (who beat col losers).
   Are $R_i$ and $C_j$ comparable?
   We know $R_i$ is fast. $C_j$ is fast.
   If we race all $R_i$s? No, we already did that implicitly?
   Wait, if we race all $R_i$ together, we get the best among rows.
   
   Let's think about the set of candidates.
   We have $N^2$ elements. We want to discard $N-1$.
   Suppose we find a subset of size $N^2 - (N-1) = N^2 - N + 1$ that are candidates for the top.
   How to ensure the ones left out are truly slow?
   If we show that each excluded element is slower than at least $N$ elements, then it cannot be in the top $N-1$?
   Wait, if $x$ is slower than $y_1, y_2, \dots, y_N$, then $x$ is at best $(N+1)$-th.
   So if we can prove for each excluded element $z$, there exists a set of $N$ distinct elements all faster than $z$, then $z$ is in the bottom $N$.
   We only need to exclude $N-1$ elements.
   So if we can find $N-1$ elements $z_1, \dots, z_{N-1}$ such that for each $k$, there are at least $N$ elements faster than $z_k$, then these $z_k$ must be among the slowest $N$ elements?
   Actually, if we find $N-1$ distinct elements, each beaten by a *disjoint* set of $N$ elements? No, overlap is fine.
   If element $z$ is beaten by $N$ elements, it ranks $\ge N+1$.
   Since we have $N^2$ elements, the ranks go from 1 to $N^2$.
   We want to pick the top $K = N^2 - N + 1$.
   This leaves the bottom $N-1$ spots (ranks $K+1$ to $N^2$).
   Any element at rank $\le K$ must be *not* beaten by $N$ elements?
   No, rank $K$ element could be beaten by $K-1$ elements.
   Rank $N^2$ element is beaten by $N^2-1$.
   We want to identify the set $Top$.
   Complement is $Bottom$.
   If we identify $n-1$ elements that are guaranteed to be in $Bottom$.
   Condition: An element $x$ is guaranteed to be in $Bottom$ (rank $\ge K+1$) if we can find $N$ distinct elements $y_1, \dots, y_N$ such that $y_i > x$ for all $i$.
   Then $x$ is worse than at least $N$ people. So $x$ cannot be in the top $N-1$ people? No.
   Top $N-1$ people means rank $\le N-1$.
   If $x$ is beaten by $N$ people, then $x$'s rank is at least $N+1$.
   So $x$ is not in the top $N-1$ people?
   Wait, the target set size is $N^2 - N + 1$.
   The excluded set size is $N-1$.
   If an element is beaten by $N$ people, it has rank $\ge N+1$.
   Is it possible that an element in the excluded set has rank $\le N-1$?
   Only if the excluded set overlaps with the "beaten by N" condition?
   Wait. We want to return the set of elements with rank $\le N^2 - N + 1$.
   Equivalent to finding elements with rank $\le N^2 - N + 1$.
   Or equivalently, discarding elements with rank $\ge N^2 - N + 2$.
   i.e., rank $\ge N+1$ (if we discard last $N-1$, indices $N^2-N+2 \dots N^2$).
   Wait, $N^2 - (N-1) = N^2 - N + 1$.
   So we discard the lowest $N-1$ ranked items.
   If we prove an item $x$ has rank $\ge N+1$ (which is $\ge$ lowest $N-1$ only if $N+1 \ge N^2 - N + 2$? No).
   Rank $N+1$ is not necessarily in the excluded set.
   Excluded set is the *slowest* $N-1$.
   These correspond to ranks $N^2 - N + 2, \dots, N^2$.
   Rank $N+1$ is much better.
   So proving $x$ is beaten by $N$ elements puts $x$ at rank $\ge N+1$.
   This doesn't help much because $N+1$ is inside the kept set (unless $N^2 - N + 1 < N+1$).
   $N^2 - N + 1 \ge N+1 \iff N(N-1) \ge N-1 \iff N \ge 2$ (true).
   So rank $N+1$ is kept.
   Wait, rank $\ge N^2 - N + 2$.
   This means $x$ must be beaten by at least $(N^2 - (N^2 - N + 2)) + 1$ elements?
   No. Rank is determined by how many are faster.
   If rank is $R$, there are $R-1$ faster elements.
   We want to exclude elements with rank $> N^2 - N + 1$.
   Min rank to exclude is $N^2 - N + 2$.
   This implies at least $N^2 - N + 1$ elements are faster.
   So we need to prove an element $x$ is slower than $N^2 - N + 1$ elements.
   That's a lot.
   
   Wait, this approach is too strong.
   However, note the problem statement: "slowest n - 1 pepes are indistinguishable from each other".
   This phrasing often implies we don't need to distinguish between the bottom $N-1$ specifically, nor even their values.
   Just that we need to find the top $N^2 - N + 1$ correctly.
   Which means we need to separate the top group from the bottom group.
   The boundary is after the $(N^2 - N + 1)$-th fastest.
   
   Let's reconsider the "winning layers".
   Imagine we have $N$ rows.
   Race each row. Get winners $R_1, \dots, R_N$.
   Race these $N$ winners. Get winner $W_{best}$.
   As derived before, $W_{best}$ is faster than everyone else?
   Let's verify this claim again.
   Items in Row $i$ are $r_{i,1}, \dots, r_{i,n}$. $R_i$ is winner.
   $R_i > r_{i,j}$ for all $j$.
   We race $\{R_1, \dots, R_N\}$. Let winner be $R_k$.
   $R_k > R_j$ for all $j \neq k$.
   Transitivity: $R_k > R_j > r_{j,m}$.
   So $R_k$ is faster than all items in row $k$ (except itself) AND all items in all other rows.
   So $R_k$ is faster than ALL OTHER $N^2 - 1$ items.
   So $R_k$ is the unique 1st place.
   
   Now, consider the set $S = \{R_1, \dots, R_N\} \cup \{ \text{rest} \}$.
   We know $R_k$ is the global max.
   What about the rest?
   We have a winner $R_k$.
   We have a set of candidates $C = \{R_1, \dots, R_N\} \setminus \{R_k\}$.
   For each $j \neq k$, $R_j$ won its row race.
   But lost to $R_k$.
   Is $R_j$ faster than others in row $j$? Yes.
   So $R_j >$ row $j$ losers.
   But $R_k$ is the only one we know is faster than $R_j$.
   Is $R_j$ faster than $R_i$ ($i \neq k, j$)?
   We know $R_k > R_j$ and $R_k > R_i$. But no relation between $R_j$ and $R_i$.
   
   We need to form a set of size $N^2 - N + 1$ that includes the top ones.
   Note that $R_k$ is definitely in.
   What about the losers of row $k$?
   They lost to $R_k$.
   Are they guaranteed to be in top $N^2 - N + 1$?
   Not necessarily.
   Wait, if $N^2 - N + 1$ is very large.
   For $N=2$, we keep 3 elements. Exclude 1.
   Here $R_1, R_2$ are the row winners. $R_k$ (say $R_1$) is global max.
   $R_1$ wins vs $R_2$.
   Losers: $L_1$ (from row 1), $L_2$ (from row 2).
   Order: $R_1 > R_2 > L_2$.
   And $R_1 > L_1$.
   Possible orders:
   1. $R_1 > R_2 > L_1 > L_2$ (Keep $\{R_1, R_2, L_1\}$). Discard $L_2$.
      $L_2$ is slower than $R_1, R_2, L_1$.
      $L_2$ is beaten by 3 elements.
      $N-1 = 1$. We need to discard 1 element.
      The one to discard is $L_2$.
      Is it always $L_2$?
      In this case $L_2$ lost to $R_2$, $R_2$ lost to $R_1$.
      $L_2$ lost to two winners.
      $L_1$ lost to $R_1$.
      $L_1$ is beaten by $R_1$.
      $L_2$ is beaten by $R_2$ and $R_1$.
      Intuitively $L_2$ is "more likely" to be bad.
      
      Can we prove that $L_2$ is the $N^2$-th (or worst) element?
      We know $R_1$ is 1st.
      Remaining spots $2 \dots 4$.
      $L_2$ is beaten by $R_1, R_2$.
      So $L_2$ is $\le$ 2nd? No.
      Beaten by 2 means rank $\ge 3$.
      Since we need to keep 3, rank 3 is okay.
      Wait, we need to keep the FASTEST $N^2 - N + 1$.
      For $N=2$, keep top 3. Discard 4th.
      If $L_2$ is rank 4, we discard it.
      Is $L_2$ rank 4?
      $L_2$ lost to $R_2$. $R_2$ lost to $R_1$.
      If $L_2$ lost to $R_2$, $R_2$ is faster.
      Does $L_2$ lose to $R_1$? Yes.
      Does $L_2$ lose to $L_1$? Unknown.
      If $L_2$ lost to $L_1$, then $L_1 > L_2$.
      Then ranking: $R_1 > R_2 > L_1 > L_2$.
      Top 3 are $\{R_1, R_2, L_1\}$. Discarded $L_2$.
      If $L_1$ lost to $L_2$?
      Then $R_1 > L_2 > R_2 > L_1$?
      Wait, $R_2 > L_2$ is known.
      So this case is impossible.
      So $R_2 > L_2$ always.
      So $L_2$ is worse than $R_2$.
      Also $R_1 > R_2$.
      So $L_2$ is worse than $R_1, R_2$.
      Thus $L_2$ is at best 3rd (rank 3).
      Wait, rank 1 is $R_1$. Rank 2 is $R_2$? No, maybe $L_1 > R_2$.
      But $L_2$ is worse than $R_2$.
      So $R_2$ is rank $\le$ 2.
      $L_2$ is rank $> 2$. So rank $\ge 3$.
      We need to keep top 3.
      So keeping $\{R_1, R_2, L_1\}$ is risky?
      If $L_1$ is actually slower than $L_2$ (e.g. $L_1$ lost to $L_2$), then order is $R_1 > L_2 > R_2 > L_1$.
      Then top 3 are $R_1, L_2, R_2$.
      Our set $\{R_1, R_2, L_1\}$ would be $\{R_1, R_2, L_1\}$. Missing $L_2$.
      This set is incorrect.
      Wait, why did I think $L_2$ is "bad"?
      Because it lost to $R_2$.
      $L_1$ lost to $R_1$.
      Wait, $L_1$ is faster than nothing? $L_1$ lost to $R_1$.
      $L_2$ lost to $R_2$.
      Is $R_1$ faster than $R_2$? Yes.
      So $R_1 > R_2 > L_2$.
      And $R_1 > L_1$.
      The only unknown is between $L_1$ and $L_2$ (and $L_1$ vs $R_2$).
      We know $L_1$ lost to $R_1$.
      $L_2$ lost to $R_2$.
      Since $R_1 > R_2$, and $R_2 > L_2$, $L_2$ is beaten by 2 elements.
      $L_1$ is beaten by 1 element ($R_1$).
      Does this imply $L_2$ is worse?
      If $L_1 > L_2$, then $L_1$ beaten by $\{R_1\}$. $L_2$ beaten by $\{R_1, R_2, L_1\}$.
      $L_2$ beaten by 3. $L_1$ by 1.
      $L_2$ is "heavier" loser.
      But for $N=2$, we can't just guess.
      We need an algorithm.
   
   Strategy:
   We need to eliminate the "worst" $n-1$ items.
   For $n=2$, eliminate 1.
   If we eliminate the losers of the "winner of all".
   For $n=2$:
   Rows $\{1,2\}, \{3,4\}$.
   Winners $u, v$.
   Race $u, v$. Winner $W$, loser $L$.
   $W$ is global max.
   $L$ is loser of winners race.
   Is $L$ the one to discard?
   No, we saw counterexample for just discarding $L$.
   
   Wait, the problem says "Can you order the $n^2 - n + 1$ fastest".
   This implies we MUST find the correct list.
   If we have ambiguity, we are wrong.
   So there must be a deterministic way.
   Maybe I should use $O(N^3)$ queries.
   Since $N \le 20$, $N^3$ is small.
   Can we simulate a tournament?
   If we construct a balanced binary tournament tree where each internal node represents a race of size $n$?
   Wait, tournament reduces 1 item. Here we reduce $n-1$ items per race.
   If we want to find the top $K$, we can keep reducing the set of candidates until size $K$.
   Start with $N^2$. Target size $K = N^2 - N + 1$.
   In each step, we want to reduce the size of the candidate set.
   How?
   Take a group of $n$ candidates. Race them.
   Keep the winner. Discard the other $n-1$.
   Wait, we can only discard the ones we proved are slower.
   In a race of $\{x_1, \dots, x_n\}$, winner is $w$.
   We know $w$ is faster than $x_1, \dots, x_{n-1}$?
   Yes, winner is fastest among them.
   So we know $w > x_i$ for all $i \neq pos(w)$.
   So for the other $n-1$, we have concrete evidence they are slower than $w$.
   But we don't know if they are slower than elements outside the race.
   So we can't discard them globally yet.
   BUT, if we maintain a pool of candidates, and we find a race where we keep 1 and discard $n-1$, that helps.
   
   Let's try a recursive strategy.
   Function `solve(size M, count needed K)`?
   No.
   
   Let's look at the constraint: sum of $n^3$ is limited.
   This suggests complexity like $O(n^3)$ per testcase.
   
   There is a specific trick for this problem.
   Construct a grid $N \times N$.
   Perform row races. Get winners $R_i$.
   Perform column races. Get winners $C_j$.
   Now we have $2N$ items (some might be same).
   Actually, we can define the set of "candidates" as all items.
   Consider the intersection graph.
   
   Another angle:
   We want to select the $N^2 - N + 1$ largest elements.
   This is equivalent to selecting the $N^2$ elements and removing the $N-1$ smallest.
   Since we can remove elements only if we have proof, we need to prove something about the smallest.
   
   Key insight from similar problems (e.g. Codeforces "Pepe Racing"):
   Use the $N$ rows and $N$ columns idea.
   Race rows: Get $R_1, \dots, R_N$.
   Race columns: Get $C_1, \dots, C_N$.
   Note that the items in $R$ are winners of rows.
   The items in $C$ are winners of columns.
   Now, for any element $x$ not in $\{R\} \cup \{C\}$, $x$ is a loser in its row and a loser in its column.
   So there exists $r \in \{R\}$ such that $r > x$ (from row race).
   And there exists $c \in \{C\}$ such that $c > x$ (from column race).
   So non-candidates are "covered" by candidates.
   
   Let's denote $S = \{R_1, \dots, R_N\} \cup \{C_1, \dots, C_N\}$.
   Wait, $|S|$ can be up to $2N$.
   But we only have $N$ row winners and $N$ col winners.
   If $x$ is a row loser, it is slower than some $R_i$.
   If $x$ is a col loser, it is slower than some $C_j$.
   Is every element covered?
   Yes, everyone is in exactly one row and one column.
   So every element is covered by some $R_i$ and some $C_j$.
   
   We know $R_i$ is the max of row $i$.
   $C_j$ is the max of col $j$.
   We need to compare elements in $S$ to find the global ordering or just the cut-off.
   We know $R_i >$ non-$R_i$ in row $i$.
   We know $C_j >$ non-$C_j$ in col $j$.
   
   Can we determine the top $N^2 - N + 1$?
   Consider the set $U = \{R_1, \dots, R_N\} \cup \{C_1, \dots, C_N\}$.
   Actually, maybe we can just take ALL row winners and ALL column winners and race them?
   No, we need to output the result of races.
   
   Let's assume the optimal solution uses the grid structure.
   Algorithm:
   1. Identify row winners $R_1, \dots, R_N$. (Cost $N$ races).
   2. Identify column winners $C_1, \dots, C_N$. (Cost $N$ races).
   Wait, identifying column winners requires running $N$ races involving the column members.
   But column members include $R_i$'s.
   Wait, to run a race on column $j$, we need elements $\{(1,j), \dots, (N,j)\}$.
   Some of these are already used in row races.
   Can we reuse the information?
   We know relative order in rows/cols.
   We don't know $R_i$ vs $R_k$.
   We don't know $R_i$ vs $C_j$.
   
   However, notice that $R_i$ is the best in row $i$. $C_j$ is best in col $j$.
   Consider the set of all "local maxima" $M = \{R_1, \dots, R_N\} \cup \{C_1, \dots, C_N\}$.
   Actually, $R_i$ and $C_j$ might be the same element (intersection $(i,j)$).
   But generally distinct.
   The size of $M$ is at most $2N$.
   
   Claim: The set of elements that are NOT "dominated" by at least $N-1$ other elements in $M$?
   No.
   
   Let's go back to the condition: $N-1$ elements are indistinguishable/slowest.
   This means if we correctly output any $n-1$ elements that are guaranteed to be in the bottom $n-1$, we are good.
   Wait, no. The bottom $n-1$ are indistinguishable.
   It means we don't need to distinguish between the bottom $n-1$.
   We just need to identify that they are bottom.
   So if we find a set $S$ of size $N^2 - N + 1$ such that for any $x \notin S$, $x$ is definitely slower than at least one element in $S$?
   No, that doesn't guarantee $S$ is the top set.
   It guarantees $S$ isn't the bottom set?
   
   Actually, if $x$ is in the bottom $N-1$, it means there are at least $N^2 - (N-1) = N^2 - N + 1$ elements faster than $x$.
   Wait, rank $N^2$ (worst). $N^2-1$ faster.
   Rank $N^2 - N + 2$. $(N^2 - N + 2 - 1) = N^2 - N + 1$ faster.
   So if we can prove $x$ is slower than $N^2 - N + 1$ elements, then $x$ is in the bottom.
   This is very hard to prove for each $x$.
   
   However, notice the pattern in Example Input:
   Queries were:
   `? 1 2` -> winner 2.
   `? 3 4` -> winner 4.
   `? 2 4` -> winner 3? (Wait, maybe 3 is a typo in example, maybe it was `? 2 3`?)
   If the example intended `? 2 4` and response `3`, that's impossible.
   If response was `2`: `2` is faster than `4`.
   Then we have `2 > 1`, `4 > 3`, `2 > 4`.
   Implies `2 > 4 > 3` and `2 > 1`.
   Order: `2 > 4 > 3` (maybe), `2 > 1`.
   Is `1` faster or slower than others?
   If `1` was slowest, we discard it.
   If `3` was slowest, we discard it.
   But we see `! 2 3` output. Length 2. But $N=2$ needs 3.
   The example provided in the prompt is confusing/wrong.
   "Example Output:
    ? 1 2
    ? 3 4
    ? 2 4
    ! 2 3"
    
   Ah, I suspect the example output for `!` command corresponds to `n=2`.
   But $n=2$ needs 3 labels. `! 2 3` has 2.
   Perhaps the example input corresponds to `n` such that $n^2 - n + 1 = 2$?
   $n^2 - n + 1 = 2 \implies n^2 - n - 1 = 0$. No integer sol.
   Maybe $n$ varies in example?
   The example input has `1` testcase.
   Then `2` (n).
   Then `2` (response).
   Then `4` (response).
   Then `4` (query label?).
   Wait.
   Example Input:
   1 (testcases)
   2 (n)
   2 (response to query)
   4 (response to query)
   4 (this is likely a query printed by program?)
   Wait, the example output block shows the program output.
   Example Output:
   ? 1 2
   ? 3 4
   ? 2 4
   ! 2 3
   
   Example Input (stdin):
   1
   2
   2
   4
   4
   3
   2
   
   Parsing this:
   Testcase count 1.
   n = 2.
   Program reads n. Starts interaction.
   Output `? 1 2`.
   Stdin provides response `2`.
   Output `? 3 4`.
   Stdin provides response `4`.
   Output `? 2 4`.
   Stdin provides response `3`.
   Output `! 2 3`.
   End of file.
   
   This implies `3` was the winner of `? 2 4`.
   BUT `3` is NOT in `{2, 4}`.
   This confirms the Example in the prompt text is completely broken or describes a different version.
   However, the problem title "Pepe Racing" matches a specific Codeforces problem.
   Problem Name: "Pepe Racing" -> likely Codeforces 1959E or something.
   Actually, checking online resources for "Pepe Racing" yields "Codeforces Round 956 (Div. 2) - Problem E".
   Problem Statement: "Pepe's Racing".
   Let's check the official tutorial or editorial logic if possible, or just solve it logically.
   
   Wait, if I assume the problem is exactly as stated in the prompt (with valid constraints), I should write code that satisfies it.
   If the example is garbage, I ignore it.
   
   Let's focus on the logic.
   We have $N^2$ items. We need to output $N^2 - N + 1$ items.
   We need to find $N-1$ items to exclude.
   The exclusion set $L$ has size $N-1$.
   Every $x \in L$ must be slower than some $y \notin L$?
   Actually, since $L$ is the set of slowest $N-1$, every $x \in L$ is slower than at least $N^2 - (N-1) = N^2 - N + 1$ elements.
   Wait, if $x \in L$, rank$(x) > N^2 - N + 1$.
   So rank$(x) \ge N^2 - N + 2$.
   Number of faster elements $\ge N^2 - N + 1$.
   So $x$ is slower than at least $N^2 - N + 1$ elements.
   
   This means if we can show $x$ is slower than $K = N^2 - N + 1$ elements, we can exclude $x$.
   But doing this for $N-1$ items is expensive.
   
   Is there a simpler condition?
   Look at the problem: $n \le 20$.
   Maybe we can find the top $n$ elements first?
   If we can find the top $n$ elements, can we proceed?
   
   Let's consider the "cycle" construction.
   Arrange $N^2$ items into a grid.
   Let $w(i, j)$ be the item at row $i$, col $j$.
   If we race rows, we get winners $r_i$.
   If we race columns, we get winners $c_j$.
   
   Let's hypothesize that the set of winners of all row races and column races contains the "elite".
   But we need to be sure.
   
   Wait, I remember a solution for a similar problem (finding top $k$) involves building a DAG.
   Given the constraints $\sum n^3$, we can afford $O(n^3)$ comparisons.
   
   Let's try a generic selection algorithm adapted for batch comparisons.
   We have $M$ elements.
   Randomly sample $n$ elements? No, interactor is adaptive. Must be deterministic.
   
   Let's try to implement the "Elimination" method.
   Maintain a pool of "Candidates". Initially $M$ elements.
   While $|Candidates| > N^2 - N + 1$:
     Pick a subset of Candidates of size $n$.
     Race them.
     Remove $n-1$ "losers".
     Keep 1 winner.
   But which ones to remove?
   We can only remove losers if we are sure they are not needed.
   But we don't know global status.
   However, notice that if we remove $x$, we are saying "x is not in top $K$".
   This requires proving $x$ is slower than at least $K$ elements? No.
   If we want to be sure $x$ is not in top $K$, we must find a set $Y$ of size $M - K + 1$ such that all $y \in Y$ are faster than $x$?
   Wait, if we have a set $Z$ of size $n^2 - (n^2 - n + 1) = n-1$ that we claim are the slowest.
   Then for any $z \in Z$, we need to ensure it cannot be in the top $n^2 - n + 1$.
   This means there exist at least $n^2 - n + 1$ elements faster than $z$.
   If we can construct a set of $n^2 - n + 1$ elements all faster than $z$, then $z$ is excluded.
   
   This looks like finding a "dominating set" of sorts.
   
   Wait, look at the constraints again. $N$ is small.
   Can we just perform $O(N^2)$ races to fully sort the winners of rows and columns?
   
   Let's step back. What if we just find the top $N^2 - N + 1$ using a selection network?
   Consider $N^2$ items.
   Sort them? Too slow.
   Selection?
   Maybe we can just race the winners of rows against the winners of columns?
   
   Let's check if the problem has been solved before. "Pepe Racing" Codeforces 1966C? No.
   Found similar problem "E. Pepe's Speeds".
   Problem 1869E?
   Actually, let's assume this is "Pepe's Speeds".
   
   The strategy for "Pepe's Speeds" is as follows:
   1. Divide the $n^2$ pepes into $n$ groups of $n$ (Rows).
      Query each row. Get $n$ row-winners.
   2. Divide the $n^2$ pepes into $n$ groups of $n$ (Cols).
      Query each col. Get $n$ col-winners.
   3. Wait, querying cols requires accessing items.
      Row $i$ contains $\{(i, 1), ..., (i, n)\}$.
      Col $j$ contains $\{(1, j), ..., (n, j)\}$.
      Total queries so far: $n + n = 2n$.
   4. Now we have a set of $2n$ potential winners?
      No, we have $n$ row winners $R_1...R_n$ and $n$ col winners $C_1...C_n$.
      Note that $R_i$ is faster than others in row $i$.
      $C_j$ is faster than others in col $j$.
   
   Now consider the grid of winners.
   Actually, the optimal strategy uses a "Cycle" or "Knockout".
   But given the time limit and memory, and $N \le 20$.
   Wait, there is a constraint: sum of $N^3$ over test cases is $3 \cdot 10^5$.
   This implies we can use $O(N^3)$ queries.
   $N=20 \implies N^3 = 8000$.
   So we can afford up to 8000 queries.
   Sorting $N^2 = 400$ elements normally takes $\log(400!) \approx 400 \times 9$ comparisons.
   But our comparison is a "group min".
   Comparing 1 item takes $N^2$ ops?
   
   Let's use the following algorithm:
   1. Construct an adjacency matrix representing the relations we discover.
   2. Initialize graph with $n^2$ nodes.
   3. Perform races on rows and columns.
      For each row $i$, race all items. Let $w_i$ be winner.
      Add edges $(w_i, x)$ for all $x$ in row $i$ (direction: $w_i \to x$).
      Same for columns.
   4. Now we have partial order.
   5. Identify the set of "potential candidates" that are NOT dominated by $N$ elements.
   
   Wait, if we just race ALL rows and ALL columns.
   This costs $2n$ queries.
   Then we have $n$ row winners and $n$ col winners.
   Wait, if we race a row, we find the max.
   If we race a col, we find the max.
   Now, notice that for any cell $(i,j)$, it lost to $R_i$ (its row winner) and $C_j$ (its col winner).
   So $(i,j)$ is dominated by $R_i$ and $C_j$.
   If we want to keep an element $x$, we need to be careful.
   
   What if we consider the set of ALL row winners $\mathcal{R}$ and ALL col winners $\mathcal{C}$.
   Union $\mathcal{R} \cup \mathcal{C}$ has size at most $2n$.
   Let's race all elements in $\mathcal{R} \cup \mathcal{C}$?
   Size $\le 2n$. We can split into races of size $n$.
   Say we race $\{R_1, \dots, R_n\}$ -> winner $W_R$.
   And race $\{C_1, \dots, C_n\}$ -> winner $W_C$.
   This gives info about $\mathcal{R}$ vs each other, and $\mathcal{C}$ vs each other.
   But we still don't know relation between $R$ and $C$.
   
   However, note that every element is dominated by SOME element in $\mathcal{R}$ and SOME element in $\mathcal{C}$.
   If we want to find the top $n^2 - n + 1$ elements.
   This is equivalent to finding $n-1$ elements to discard.
   Let's try to identify $n-1$ "weak" elements.
   An element is "weak" if it is dominated by $n$ elements?
   If $x$ is dominated by $n$ elements, it is slower than at least $n$ items.
   But being slower than $n$ items does not exclude it from top $n^2 - n + 1$.
   Wait, $n^2 - n + 1$ is almost everything.
   For $N=2$, keep 3. Dominate by 2 is bad?
   If $x$ dominated by 2, rank $\ge 3$.
   If kept set is top 3, rank 3 is allowed.
   So dominating by $n$ is not enough to exclude.
   We need to dominate by $n^2 - n + 1$.
   
   Wait, there is a simpler interpretation.
   We are looking for the $N^2 - N + 1$ fastest.
   Let's label them $P_1, \dots, P_K$.
   
   Correct solution path for this type of problem:
   1. We have $N^2$ elements.
   2. We want to find a set of size $N^2 - N + 1$.
   3. Notice that $N^2 - N + 1 = (N-1)^2 + (2N - N - 1)$? No.
   4. Consider the graph where we draw edges.
   
   Actually, look at the "Pepe's Speeds" editorial.
   Solution involves constructing a tournament where each layer eliminates $N$ elements.
   Since we need to keep $N^2 - N + 1$, we can discard $N-1$ elements.
   We can discard elements one by one? No.
   Wait, $N$ elements in a race produces 1 winner, eliminating $N-1$.
   If we can perform a sequence of races that systematically eliminates candidates, eventually we are left with the top ones.
   But we must preserve the invariant that the survivors are potentially fast.
   If we eliminate an element, we must be sure it's not needed.
   
   Let's try to find the $N-1$ slowest.
   For $N=2$, discard 1.
   For general $N$, discard $N-1$.
   If we can find a "winner of winner..." structure?
   If we arrange all $N^2$ elements in a grid.
   Row winners: $R_1 \dots R_N$.
   Race these winners: Winner $W$.
   $W$ is the fastest element overall?
   YES.
   Proof:
   Let $X$ be any element. $X$ is in row $k$.
   $R_k$ is the winner of row $k$. So $R_k > X$.
   $W$ is the winner of $\{R_1, \dots, R_N\}$. So $W \ge R_k$.
   (Strictly $W > R_k$ if $R_k$ lost, or $W=R_k$).
   Wait, strictly speaking, $W$ is one of the $R$'s.
   So $W$ is faster than all $R$'s that are not $W$.
   And $W$ is equal to the fastest of $\{R_i\}$.
   If $X$ is not in row $k$ where $W$ came from?
   If $X$ is in row $k$ and $R_k$ is not $W$. Then $W > R_k > X$.
   If $X$ is in row $k$ and $R_k$ IS $W$. Then $W$ is in row $k$, and $W$ won row $k$. So $W > X$.
   So $W$ is strictly faster than EVERYTHING.
   
   So $W$ is definitely rank 1.
   Now we remove $W$. We are left with $N^2 - 1$ items.
   We need to remove $N-2$ more items to leave $N^2 - (N-1) - 1$ items?
   Wait, we need to return $N^2 - N + 1$ items.
   So we need to discard $N-1$ items total.
   We have identified $W$ as rank 1.
   We can't discard $W$.
   So we need to identify $N-2$ items to discard from the remaining $N^2 - 1$.
   
   Wait, this recursive logic works if we can efficiently find the next fastest.
   But finding the next fastest takes time.
   However, note that we know $W$ beats $R_k$ (if $R_k \neq W$).
   So all $R_k \neq W$ are slower than $W$.
   Also all elements in row $k$ (including $W$ if $R_k=W$) are slower than $W$.
   Actually, $W$ dominates row $k$ (as it won row $k$).
   And $W$ dominates other row winners.
   
   This implies that the set of "Row Winners" $\{R_1, \dots, R_N\}$ contains candidates for high ranks.
   The other $N^2 - N$ elements are losers in their rows.
   Each loser $X$ (in row $k$) is slower than $R_k$.
   If $R_k$ is "slow", $X$ is definitely slower.
   If $R_k$ is fast, $X$ might be slow.
   
   Hypothesis: The top $N^2 - N + 1$ elements are formed by taking the set of all "Row Winners" plus "Column Winners"?
   Or maybe just "Row Winners" plus some others?
   There are $N$ row winners.
   We need $N^2 - N + 1$ elements.
   $N^2 - N + 1 - N = N^2 - 2N + 1 = (N-1)^2$.
   So we need $(N-1)^2$ additional elements.
   The number of elements NOT in row winners is $N^2 - N$.
   We need to select $(N-1)^2$ from them?
   
   Actually, there is a known solution for this problem.
   The solution relies on the fact that if we race rows, then race columns, we get enough info.
   Wait, I can construct the full graph.
   
   Let's assume we simply output ALL elements EXCEPT the $N-1$ losers of row races?
   No, we saw counter-example for $N=2$ ($L_2$).
   In $N=2$, we had winners $R_1, R_2$.
   $R_1$ wins grand race.
   $L_1$ is loser of row 1.
   $L_2$ is loser of row 2.
   We kept $R_1, R_2, L_1$. Excluded $L_2$.
   Wait, was $L_2$ really the one to exclude?
   We established $R_1 > L_2$. $R_2 > L_2$.
   Also $R_1 > R_2$.
   So $R_1 > R_2 > L_2$.
   Also $R_1 > L_1$.
   So $L_1$ and $L_2$ are both slower than $R_1$.
   We need to keep 3.
   If we keep $L_2$, we exclude $L_1$.
   If we keep $L_1$, we exclude $L_2$.
   In my trace, $L_2$ seemed "more beaten".
   Is it possible to determine?
   No, because $L_1$ vs $L_2$ is unknown.
   Wait, if $L_1$ vs $L_2$ is unknown, then we cannot decide deterministically without asking?
   If we ask `? L_1 L_2`, we get the answer.
   But we are restricted by number of queries.
   But for $N=2$, we have plenty of queries.
   
   The key is that we don't need to distinguish $L_1$ vs $L_2$ perfectly to solve the problem.
   Wait, if the answer set is ambiguous, the problem is unsolvable.
   But the problem asks "Can you order...".
   So there is a unique set.
   But $L_1$ and $L_2$ are indistinguishable from the perspective of being "slowest".
   Wait, "slowest n-1 pepes are indistinguishable from each other".
   This phrase means IF we find the set of top $N^2 - N + 1$, the identity of the bottom $N-1$ doesn't matter, just that they are not in the top set.
   Wait, if $L_1$ and $L_2$ swap roles, does it change the "top $N^2 - N + 1$ set"?
   Top 3 set could be $\{R_1, R_2, L_1\}$ OR $\{R_1, R_2, L_2\}$.
   If both are valid, we just output one?
   But wait, "indistinguishable from each other" usually refers to the values.
   "The slowest n - 1 pepes are indistinguishable from each other."
   This means in the INTERACTOR'S model, the slowest $N-1$ have identical properties (e.g. same probability distribution, or we don't care about their order).
   But their LABELS are distinct.
   So if there is ambiguity, say we can't tell if $L_1$ is rank 3 or $L_2$ is rank 3, then BOTH configurations are consistent.
   In ONE configuration, $L_1$ is rank 3 (so $L_1$ is in Top 3). In another, $L_2$ is rank 3.
   Wait, if the interactor is adaptive, it chooses the configuration AFTER seeing our queries.
   So if we output a set that depends on our queries, but there exists a configuration consistent with answers where a different set is the top set, we fail.
   We must be robust against ANY consistent configuration.
   
   Therefore, we must find a set that contains the true top $N^2 - N + 1$ in ALL consistent configurations.
   Or, the set must be the SAME across all consistent configurations.
   
   Back to $N=2$.
   Config 1: $R_1 > R_2 > L_1 > L_2$. Top 3: $\{R_1, R_2, L_1\}$.
   Config 2: $R_1 > R_2 > L_2 > L_1$. Top 3: $\{R_1, R_2, L_2\}$.
   Both configs are consistent with $R_1 > R_2$ (grand), $R_1 > L_1$ (row 1), $R_2 > L_2$ (row 2).
   Wait, $R_2 > L_2$ and $R_1 > L_1$.
   Grand race: $R_1 > R_2$.
   Relations consistent:
   1. $R_1 > R_2 > L_2$. $R_1 > L_1$. (No relation $L_1, L_2$).
      Possible: $R_1 > R_2 > L_1 > L_2$.
      Possible: $R_1 > R_2 > L_2 > L_1$.
   2. $R_1 > R_2$. $R_1 > L_1$. $R_2 > L_2$.
      Possible: $R_1 > L_1 > R_2 > L_2$. (Check $R_1 > R_2$, ok. $R_2 > L_2$, ok. $R_1 > L_1$, ok).
      In this case, Top 3 is $\{R_1, L_1, R_2\}$.
      Wait, $L_1$ beats $R_2$.
      But earlier I said $R_2$ wins row 2, so $R_2 >$ others in row 2.
      $L_2$ is other in row 2. So $R_2 > L_2$.
      $L_1$ is other in row 1. $R_1 > L_1$.
      No relation between $L_1$ and $R_2$.
      So $L_1$ COULD be faster than $R_2$.
      So Top 3 could be $\{R_1, L_1, R_2\}$ (if $L_1 > R_2$) or $\{R_1, R_2, L_1\}$ (if $L_1 < R_2$).
      Or $\{R_1, R_2, L_2\}$? No, $R_2 > L_2$. So $L_2$ is worse than $R_2$.
      So $L_2$ can be rank 3 only if $R_2$ is rank 2.
      If $L_2$ is rank 3, then $L_1$ must be rank 4 (slower than $R_1, R_2, L_2$).
      Wait, $L_1$ is slower than $R_1$. $L_2$ is slower than $R_2$.
      If $L_2 > L_1$, then $L_2$ is faster.
      So possible rankings of $L_1, L_2$ vs $R_1, R_2$:
      1. $R_1 > R_2 > L_1 > L_2$ (Rank: $R_1, R_2, L_1, L_2$) -> Top 3: $R_1, R_2, L_1$.
      2. $R_1 > L_1 > R_2 > L_2$ (Rank: $R_1, L_1, R_2, L_2$) -> Top 3: $R_1, L_1, R_2$.
      3. $R_1 > R_2 > L_2 > L_1$ (Rank: $R_1, R_2, L_2, L_1$) -> Top 3: $R_1, R_2, L_2$.
      4. $R_1 > L_2 > R_2 > L_1$ (Rank: $R_1, L_2, R_2, L_1$) -> Top 3: $R_1, L_2, R_2$.
         Check consistency: $R_1 > R_2$ (ok), $R_1 > L_1$ (ok), $R_2 > L_2$ (NO, here $L_2 > R_2$).
         So 4 is invalid.
   
   So valid Top 3 sets: $\{R_1, R_2, L_1\}$, $\{R_1, L_1, R_2\}$, $\{R_1, R_2, L_2\}$.
   Intersection of these sets: $\{R_1, R_2\}$.
   Wait, $\{R_1, R_2, L_1\}$ vs $\{R_1, R_2, L_2\}$.
   Intersection is $\{R_1, R_2\}$.
   This is size 2.
   We need size 3.
   This suggests we cannot just rely on the row winners alone.
   We MUST query $L_1$ vs $L_2$?
   Or query $L_1$ vs $R_2$?
   If we query $L_1$ vs $R_2$.
   If $L_1$ wins: $L_1 > R_2$.
   Then Top 3 must include $L_1$. $R_1, L_1, R_2$?
   $L_2$ is below $R_2$ (since $R_2 > L_2$). So $L_2$ is rank 4.
   So Top 3 is $\{R_1, L_1, R_2\}$.
   If $R_2$ wins: $R_2 > L_1$.
   Then we have $R_1 > R_2 > L_1$.
   And $R_2 > L_2$. $R_1 > L_1$.
   We need to know relative order of $L_1, L_2$.
   If we don't query, we have two possibilities:
   $R_1 > R_2 > L_1 > L_2$ -> Top 3 $\{R_1, R_2, L_1\}$.
   $R_1 > R_2 > L_2 > L_1$ -> Top 3 $\{R_1, R_2, L_2\}$.
   Ambiguity!
   But the problem says "slowest n-1 pepes are indistinguishable".
   This might mean that $L_1$ and $L_2$ are treated as a group.
   If we output $\{R_1, R_2, L_1\}$ in the first case and $\{R_1, R_2, L_2\}$ in the second, is it considered correct?
   NO. "order the n^2 - n + 1 fastest pepes". This implies the set of IDs must be correct for the TRUE underlying permutation.
   The interactor has a hidden permutation. We adapt.
   If there are two consistent permutations giving different top sets, then we cannot solve it uniquely.
   Wait. If the permutations are consistent with answers, does the interactor ensure uniqueness?
   No, the interactor picks one.
   So we must produce the answer that is valid for the chosen one.
   Since we don't know which one, we must resolve the ambiguity.
   
   So we need to resolve $L_1$ vs $L_2$ and $L_1$ vs $R_2$.
   
   Actually, there is a trick.
   We have $N^2$ elements.
   We can simply find the "winner of all" $W$.
   Then remove it.
   We need to repeat this process $N-1$ times to find the $N-1$ fastest to keep?
   No, we need to keep $N^2 - N + 1$ elements.
   We need to eliminate $N-1$ slowest elements.
   Eliminating 1 element (the slowest) takes some cost.
   If we can eliminate the slowest element, we save a slot in the "slow" bin.
   If we eliminate $N-1$ slowest, we are done.
   The cost of eliminating 1 element might be high.
   
   However, we established that with $2N$ queries (Rows + Cols), we can narrow down candidates.
   But we need to identify the specific slow elements.
   
   Let's check the constraints again.
   "Slowest n - 1 pepes are indistinguishable from each other."
   This usually means that we don't need to distinguish between the bottom $N-1$.
   Wait, this means the problem setter doesn't care which one is bottom 1 or bottom 2, as long as the outputted set excludes the "slowest $N-1$ group".
   BUT, "indistinguishable" implies that their identities are swapped in the solution checker?
   No, it means in the problem context (e.g. Codeforces), the checker treats the bottom $N-1$ as a blob.
   But usually, if I submit a solution, it checks if the set of my reported IDs is the same as the set of true top IDs.
   If the bottom $N-1$ are indistinguishable, it means any subset of size $N-1$ drawn from the bottom $N-1$ can replace each other in the ground truth?
   Or does it mean that the interactor generates a scenario where the bottom $N-1$ have arbitrary speeds but we don't know who is who, so we just need to find ANY valid set of size $N^2 - N + 1$?
   Usually, it means we need to find the set that is guaranteed to be the top set regardless of the relative speeds of the bottom $N-1$.
   Essentially, we need to isolate the top $N^2 - N + 1$ from the bottom $N-1$.
   If $L_1$ and $L_2$ are the only slow elements left, and we don't know who is faster, it implies $L_1$ and $L_2$ ARE in the bottom set.
   Wait, in my $N=2$ analysis:
   Valid Top 3 sets were $\{R_1, R_2, L_1\}$, $\{R_1, R_2, L_2\}$, $\{R_1, L_1, R_2\}$.
   Note that $L_1$ appears in 2 sets, $L_2$ in 1. $R_1, R_2$ in 3.
   If the interactor chose the case where $R_1 > R_2 > L_2 > L_1$, then Top 3 is $\{R_1, R_2, L_2\}$.
   If I output $\{R_1, R_2, L_1\}$, I include $L_1$ instead of $L_2$.
   $L_1$ is rank 4. $L_2$ is rank 3.
   My output set has an element of rank 4 ($L_1$).
   So my set is NOT the top 3. It contains the 4th fastest.
   So I failed.
   So we MUST know $L_1$ vs $L_2$ or $L_1$ vs $R_2$.
   
   However, note that $L_1$ is in row 1, $L_2$ in row 2.
   $R_1 > L_1$. $R_2 > L_2$.
   We know $R_1 > R_2$.
   This implies $R_1$ is "very" fast. $R_2$ is fast.
   $L_1, L_2$ are "slow".
   Is it possible that the "slowest $N-1$" are exactly $\{L_1, L_2\}$?
   Wait, there are 2 slowest pepes. $L_1$ and $L_2$ are the only losers.
   So the set of slowest $N-1$ (size 1) must be either $\{L_1\}$ or $\{L_2\}$?
   No, "slowest $N-1$". For $N=2$, slowest 1 pepe.
   So exactly 1 pepe is in the bottom.
   Either $L_1$ is bottom, or $L_2$ is bottom.
   If $L_1$ is bottom (rank 4), then $L_2$ is top 3 (rank 3).
   If $L_2$ is bottom (rank 4), then $L_1$ is top 3 (rank 3).
   So we must find out which one is slower.
   We can race $L_1$ and $L_2$ directly!
   Query `? L_1 L_2`.
   This takes 1 query.
   Then we know who is slower.
   If $L_1$ wins, $L_1$ is rank 3, $L_2$ is rank 4. Keep $L_1$, drop $L_2$.
   If $L_2$ wins, $L_2$ is rank 3, $L_1$ is rank 4. Keep $L_2$, drop $L_1$.
   
   So for $N=2$, total queries:
   Row 1 (1), Row 2 (1).
   Winners race (1).
   Losers race (1).
   Total 4 queries.
   Wait, we also need to output the order.
   If we do `? L_1 L_2` and $L_1$ wins.
   We know $R_1 > L_1$ and $R_1 > R_2 > L_2$.
   Also $R_1 > R_2$ (from winners race).
   Also $R_2 > L_2$ (from row 2 race).
   Also $L_1 > L_2$.
   We need to sort the top 3.
   Candidates: $R_1, R_2, L_1$.
   Known: $R_1 > R_2$.
   $R_1 > L_1$.
   What about $R_2$ vs $L_1$?
   We didn't query this.
   We just know $L_1 > L_2$ and $R_2 > L_2$.
   If $L_1$ wins against $L_2$, we know $L_1 > L_2$.
   But $L_1$ vs $R_2$ is still unknown.
   If we don't know $L_1$ vs $R_2$, can we order them?
   We need to output order.
   So we need to resolve $L_1$ vs $R_2$.
   This adds another query.
   
   Total queries for $N=2$:
   1. `? 1 2` (get $R_1$)
   2. `? 3 4` (get $R_2$)
   3. `? R_1 R_2` (get $R_{best}$, e.g. $R_1$) -> Assume $R_1 > R_2$.
      Now we know $R_1 > R_2$.
   4. `? L_1 L_2` (get relation between losers).
      Wait, $L_1$ is the other guy in row 1. $L_2$ other in row 2.
   5. `? L_1 R_2` (get relation between $L_1$ and $R_2$).
      Wait, $R_2$ is a row winner. $L_1$ is a row loser.
      Maybe we don't need to query all?
      Actually, with 5 queries, we can fully sort 4 elements.
      Cost is low.
      
   Generalizing to $N$:
   We have $N$ rows. Race each.
   Get $R_1, \dots, R_N$.
   Race them. Find $R_{best}$.
   $R_{best}$ is 1st.
   The other $N-1$ winners are candidates for ranks $2 \dots N$.
   The losers of rows are candidates for ranks $\ge N+1$.
   Wait, we need to fill $N^2 - N + 1$ slots.
   We keep $R_{best}$.
   We need to pick $N^2 - N$ more elements.
   Total slots to fill.
   We have $N^2$ elements.
   We want to eliminate $N-1$ elements.
   
   Actually, we can construct a specific algorithm:
   1. Partition elements into $N$ rows.
      Race each row.
      For each row $i$, we have a winner $R_i$ and a set of losers $Loss_i$.
      For all $x \in Loss_i$, we know $R_i > x$.
      So $Loss_i$ elements are "safe" candidates for being removed ONLY IF $R_i$ is one of the "very slow" elements?
      Wait, if we know $R_i > x$, $x$ cannot be faster than $R_i$.
      If we can prove $R_i$ is slow (not in top set), then $x$ is definitely slow.
      So we should prioritize keeping $R_i$ elements.
   
   2. Collect all $R_i$'s.
      Race them in groups of $n$.
      This is essentially sorting the $R_i$'s.
      Since we can race $n$ items, we can find the top $n$ among $R$'s easily.
      We need to select top $n^2 - n + 1$ elements.
      Note that there are $N^2$ elements total.
      We want to eliminate $N-1$ elements.
      This is very few.
      Actually, $N^2$ is large, but $N-1$ is small.
      Most elements (specifically losers of rows) are "behind" the row winners.
      Specifically, for each row $i$, there are $N-1$ losers.
      Total losers $N(N-1)$.
      Total winners $N$.
      Wait, $N(N-1) = N^2 - N$.
      We need to remove $N-1$ elements.
      We can't just remove losers.
      But notice that losers are "covered" by winners.
      If we can identify a set of $N-1$ losers that are guaranteed to be slower than everyone else (or at least slower than everyone not in that set).
      
      Wait, consider the intersection of row winners and column winners.
      Actually, just use the "Grid" approach with random sampling? No.
      
      Let's rethink.
      We want to remove $N-1$ elements.
      Can we just remove the losers of the $N$ races of winners?
      Wait, the races of winners involve the $N$ winners.
      There are $N-1$ losers in such a race.
      Let the race of winners be $Q_{win}$.
      The participants are $R_1, \dots, R_N$.
      The result gives 1 winner $W$, and $N-1$ losers $L_{win}$.
      We know $W > R_j$ for all $j$.
      Wait, we know $W >$ the other participants in $Q_{win}$.
      Does $W$ beat all other row losers?
      Not directly. $W$ beats $R_j$, and $R_j > Loss_j$.
      So yes, $W > Loss_j$ for all $j$.
      So $W$ beats all row losers.
      Thus $W$ beats ALL $N^2 - 1$ other elements.
      So $W$ is the absolute fastest.
      What about the losers of $Q_{win}$?
      These are $N-1$ elements from $\{R_1, \dots, R_N\}$.
      Let $L_{win} = \{R_{j_1}, \dots, R_{j_{N-1}}\}$.
      We know $W > l$ for each $l \in L_{win}$.
      Also each $l$ wins its own row race.
      So $l > Loss_l$.
      So $l$ beats $N-1$ losers.
      We need to determine the bottom $N-1$ elements.
      Currently we have identified $W$ as 1st.
      The set of elements "proven to be faster" than anyone else is $\{W\}$.
      Actually, $W$ is faster than everyone.
      $L_{win}$ elements are faster than their respective row losers.
      Are the row losers candidates for the bottom set?
      The row losers are $N(N-1)$ elements.
      We need to pick $N-1$ from these or from $L_{win}$ or mix.
      Wait, if we pick $N-1$ from $L_{win}$, we keep them? No, we discard them.
      But $L_{win}$ elements are "fast" locally.
      They beat $N-1$ losers.
      If we discard an element $l \in L_{win}$, we are saying it's in the bottom $N-1$.
      Is it?
      $l$ beats $N-1$ elements. So it's not in the bottom $N-1$?
      Wait, if it beats $N-1$ elements, its rank is at most $N^2 - (N-1) = N^2 - N + 1$?
      Wait, rank $k$ means $k-1$ faster.
      If $l$ is faster than $x_1, \dots, x_{N-1}$, then at least $N-1$ elements are slower than $l$.
      This does NOT prevent $l$ from being in the top set.
      In fact, it helps.
      Wait, we need to identify the SET of top $N^2 - N + 1$.
      If we keep $l$, we might keep someone who is actually slow.
      We want to REMOVE the slowest $N-1$.
      If $l$ beats $N-1$ people, $l$ is "relatively fast".
      So we probably shouldn't discard $l$.
      But wait, the bottom $N-1$ are the SLOWEST.
      The ones that beat the LEAST amount of people are good candidates for bottom.
      Row losers $x$ beat NO ONE.
      So $x$ are prime candidates for the bottom set.
      If we pick all $N(N-1)$ losers, that's too many.
      We need to pick $N-1$ losers.
      Which ones?
      Those that are "slowest" among losers?
      Wait, if we pick $x$ and $y$ as bottom, we discard them.
      If $x$ beats nobody, and $y$ beats nobody.
      We need to know who is slower.
      If we just discard ANY $N-1$ losers, is it valid?
      If there is a configuration where a kept loser is faster than a discarded loser, we are in trouble.
      But we know $x$ is slower than its row winner $R_x$.
      If we discard $x$, we claim $x$ is slow.
      If we keep $x$, we claim $x$ is fast.
      If we keep ALL losers, we have size $N(N-1)$.
      We need to keep $N^2 - N + 1$.
      Total items $N^2$.
      $N^2 - (N^2 - N + 1) = N - 1$.
      We need to discard $N-1$ items.
      If we discard $N-1$ items from the LOSERS set (size $N(N-1)$), is that correct?
      Yes, if we assume the "winners" are in the top set.
      Are all winners (row winners) in the top set?
      Row winners $R_i$ beat $N-1$ items each.
      Sum of beats = $N(N-1)$.
      This accounts for everyone else.
      So winners are "very" fast.
      It is extremely likely that all $N$ winners are in the top $N^2 - N + 1$ elements.
      Actually, since $N(N-1)$ losers are beaten by winners, and each winner covers a chunk.
      It turns out that for this problem, the set of row winners plus column winners (or just row winners?) forms the core.
      But we need to fill the gaps.
      Wait, $N^2 - N + 1$ is almost $N^2$.
      We only discard $N-1$.
      So we keep almost everyone.
      The discarded elements must be the "absolute" slowest.
      These are the ones that beat NOBODY.
      Every element beats at least 0 elements.
      The losers of row races beat 0 elements. (Assuming transitivity is not checked, they are local losers).
      There are $N(N-1)$ such losers.
      We need to discard $N-1$ of them.
      Which ones?
      Maybe we can't distinguish them easily without racing.
      
      Let's look at the example again. $N=2$.
      Row 1 loser $L_1$. Row 2 loser $L_2$.
      We discarded 1.
      We kept $\{R_1, R_2, L_1\}$ (in one case).
      Here $L_1$ is kept.
      If we kept both $L_1, L_2$, we would have size 4 (too big).
      So we must discard one loser.
      Which one? The "worst".
      But we don't know.
      However, we need to output the order of top 3.
      So we need to know the order of $L_1$ vs $L_2$ (partially).
      
      Actually, there's a solution that uses $O(N^3)$ comparisons and finds a Hamiltonian path or similar? No.
      
      Let's simplify.
      Since we need to print the top $K$, maybe we can just run a randomized algorithm? No.
      Deterministic.
      
      Is it possible that we can simply output the result of a specific tournament?
      Yes.
      Consider the set of ALL elements.
      If we can find a subgraph of size $N^2 - N + 1$ that contains the max.
      
      Wait! I recall a solution for this specific problem ("Pepe Racing" / "Speeds").
      The solution is to:
      1. Partition into rows.
      2. Race rows. Get $R_1 \dots R_N$.
      3. Race cols. Get $C_1 \dots C_N$.
      4. Note that $R_i$ beats row-loser $L_{ij}$.
      5. Also $C_j$ beats col-loser.
      
      Actually, the simplest working solution for $N=20$ with $5s$ time limit:
      Since we need to keep $N^2 - N + 1$, we can basically keep EVERYONE except a specific set of $N-1$ people.
      Can we just pick $N-1$ people who are "worst"?
      Wait, the problem guarantees $N^2 - N + 1$ can be ordered.
      
      Let's implement a heuristic that works for small $N$ (up to 20).
      Algorithm:
      1. Assign ID $1 \dots N^2$ to positions $(i, j)$.
      2. Race rows: `query(row i)`. Store winner $r[i]$.
      3. Race columns: `query(col j)`. Store winner $c[j]$.
      4. Collect candidates for removal.
         Candidates for removal are likely those that are losers in both row and col races.
         But everyone is loser in row or col (except intersections).
         Intersection $(i, j)$ is $r[i]$ (row winner) and $c[j]$ (col winner) only if it happened to be winner.
         
      Let's assume the row winners are the top $N$? No.
      
      Actually, let's try to code a solution that mimics the example behavior or a known correct pattern.
      Since I don't have access to the editorials, I will synthesize a reasonable approach.
      
      Approach:
      1. Identify $N$ "Strongest" representatives (Row Winners).
      2. Identify $N$ "Strongest" representatives (Col Winners).
      3. The union of these $2N$ elements contains the true top $2N$?
         Actually, every element is dominated by some Row Winner and some Col Winner.
         If we sort the $2N$ winners, we get a hierarchy.
         But this doesn't account for the non-winners.
         
      Wait, if I simply output ALL elements except $N-1$ "random" losers?
      Wrong Answer.
      
      How about this:
      We need to find the top $N^2 - N + 1$ elements.
      Let's construct a directed graph of confirmed inequalities.
      Edges: $u \to v$ if $u$ won against $v$.
      We want to find the set $S$ of size $N^2 - N + 1$ such that for all $u \in S, v \notin S$, we don't necessarily know $u > v$.
      Wait, if we pick $S$ to be the set of "non-discardable" elements.
      The condition is that $S$ contains all $K$-th fastest.
      Actually, if we can construct a spanning tree where we peel off leaves?
      
      Let's try a simple recursive pruning.
      We want to prune $N-1$ elements.
      Pick $n$ elements. Race them.
      If the winner is known to be fast, keep him.
      If he loses, he is slow.
      But we start with $n^2$.
      If we just pick arbitrary $n$ elements, say row 1.
      Winner $w$. Others $n-1$ are slower than $w$.
      Can we discard $n-1$?
      No, because they might be faster than $w$ of other rows.
      Wait, if $w$ is slow, $w$ beats only his peers.
      So $w$ is not necessarily global.
      
      Wait, I missed a simple observation.
      "Slowest n - 1 pepes are indistinguishable".
      This means the relative order of the bottom $n-1$ doesn't matter.
      It means we just need to identify the set of size $n^2 - n + 1$.
      If there are multiple valid sets, any one is accepted? No.
      It means the problem instance has a property where the bottom $n-1$ form a "blob" in terms of speed.
      Wait, if the speeds are distinct, they can't be a blob.
      So it means the interactor ensures that any two solutions differing only by swapping elements in the bottom $n-1$ are equivalent.
      Or simply, we don't need to find the exact permutation of the bottom $n-1$, just the set.
      Wait, usually "indistinguishable" means "don't worry about their order".
      It doesn't mean "don't worry about who is in the bottom set".
      
      Let's code a solution that performs $N$ row races and $N$ col races.
      And then identifies the top candidates.
      Wait, I'll implement the following logic which is plausible:
      1. Query all rows. Record winners $R_i$.
      2. Query all cols. Record winners $C_j$.
      3. Form a set of candidates $U = \bigcup R_i \cup \bigcup C_j$.
      4. Sort $U$ based on results?
         No, we don't have complete info.
      5. Actually, for $N=2$, we used 4 queries.
      For $N=20$, we have budget for $8000$ queries.
      $N^3$ is generous.
      We can just perform a tournament-style elimination.
      
      Algorithm:
      Initialize `active_set` = all elements.
      We want to reduce size to $N^2 - N + 1$.
      Wait, reduction means discarding elements.
      If we have $n$ elements in `active_set`.
      Race them.
      Keep winner. Discard $n-1$ losers?
      We can only discard losers if we are sure.
      But in a general subset, we aren't.
      
      However, we can iterate through rows and columns.
      If we have processed all rows, we have winners.
      If we process all cols, we have winners.
      
      Let's use the property that $N$ is small.
      We can try to find the global ranking using a standard sorting network adapted.
      But comparisons are "find max of $N$".
      Wait, finding max of $N$ is powerful.
      If we can sort $N^2$ elements using $\{ \text{Max}_N \}$ operator.
      
      Actually, the solution might be simpler.
      We want to output the top $N^2 - N + 1$.
      The only elements we DON'T output are the $N-1$ slowest.
      Notice that $N-1$ is small.
      Can we identify $N-1$ elements that are definitely slow?
      For $N=2$, we discarded $L_2$ (one of the losers).
      Wait, is it always one of the losers?
      In $N=2$, $L_1$ was kept (maybe).
      If $L_1$ is kept, then $L_1$ is in the top set.
      If $L_2$ is dropped, $L_2$ is in the bottom set.
      So the dropped element is a loser.
      Are dropped elements ALWAYS losers in some race?
      Probably.
      If we just discard the losers of the "final round" or something.
      
      Let's try a specific construction.
      1. Create $N$ groups of size $N$. (Rows).
      2. Race each group. Get $N$ winners.
      3. Create $N$ groups of size $N$. (Cols).
      4. Race each group. Get $N$ winners.
      5. Now we have a set of winners.
      6. Notice that any element not in the winners set (of either rows or cols) is a loser in at least one race.
         In fact, every element is a loser in its row race (unless it's row winner) AND loser in its col race (unless it's col winner).
      7. If an element is a loser in BOTH row and col races, it is "very weak".
         If it is a winner in at least one race, it is "strong".
         There are $N$ row winners and $N$ col winners.
         Many elements might be neither (if they lose in row AND lose in col).
         Actually, intersection of "Not Row Winner" and "Not Col Winner" are the non-winner elements.
         Count:
         Total $N^2$.
         Row Winners: $N$ elements.
         Col Winners: $N$ elements.
         Intersection $I$?
         At least some elements are both.
         But we know every element $x$ is dominated by $R_{row(x)}$ and $C_{col(x)}$.
         So $x$ is slower than at least 1 element.
         Wait, if $x$ is not a row winner, $x < R_{row(x)}$.
         So $x$ is slower than 1 element.
         If $x$ is not a col winner, $x < C_{col(x)}$.
         If $x$ is neither, $x$ is slower than 2 elements.
         If $x$ is neither, it is definitely "slow".
         How many such elements are there?
         We have $N$ row winners and $N$ col winners.
         The number of elements that are NOT row winners is $N^2 - N$.
         The number of elements that are NOT col winners is $N^2 - N$.
         We want to find $N-1$ elements to discard.
         The "weakest" are those not row winners AND not col winners.
         If we have many such elements, we should discard them.
         But we need to discard exactly $N-1$.
         Is it possible that ALL such elements are among the bottom $N-1$?
         If count of $(\neg R) \land (\neg C)$ is $< N-1$, then we must discard some winners.
         But winners beat losers, so winners are stronger.
         Thus we never discard a winner unless forced.
         Is it possible we are forced?
         If we have fewer than $N-1$ double-lossers.
         Wait, if $x$ is double-loser, it is slower than $R$ and $C$.
         So $x$ is slower than 2 elements.
         If $N=2$, double-losers = $\neg R \land \neg C$.
         Elements $x$ s.t. $x \neq R_1$ and $x \neq R_2$ (wait, $N=2$ rows, 2 winners $R_1, R_2$).
         Actually, row winners are 2 elements. Col winners are 2 elements.
         Total winners $\le 4$.
         Non-winners $\ge 0$.
         If $N=2$, $R_1$ is in row 1. $R_2$ in row 2.
         $C_1$ in col 1. $C_2$ in col 2.
         Suppose $R_1 \neq C_1$, etc.
         Then 4 winners. 0 non-winners.
         But wait, $N^2 = 4$. Winners are 4.
         But we only need to keep $4-2+1 = 3$.
         So we must discard 1.
         If we have 0 double-losers, who do we discard?
         We must discard one of the winners!
         Wait, if $R_1$ is row winner, $R_1$ beats $L_1$.
         Is $R_1$ necessarily in top 3?
         Yes, because $R_1 > L_1$.
         $R_2 > L_2$.
         $R_1$ vs $R_2$.
         $R_1 > R_2$.
         So $R_1$ is 1st. $R_2$ is 2nd? Or $L_1 > R_2$.
         So $R_2$ is likely top 2 or 3.
         $L_1$ is below $R_1$. $L_2$ below $R_2$.
         If $R_1 > L_1 > R_2 > L_2$, top 3 is $R_1, L_1, R_2$. Discard $L_2$.
         Here $L_2$ is not a winner (it's a loser in row 2 and loser in col 2).
         Wait, $L_2$ lost row 2 ($R_2$ won). $L_2$ lost col 2?
         $C_2$ is winner of col 2.
         Col 2 has $\{(1,2), (2,2)\}$.
         $(2,2)$ is $R_2$? No, $(2,2)$ is the intersection of row 2 and col 2.
         So if $R_2$ is at $(2,2)$, then $R_2$ is also in Col 2.
         So $R_2$ is a candidate for $C_2$.
         So if $R_2$ wins Col 2, then $R_2$ is Col Winner.
         So $R_2$ is a "Winner in both races" or just one.
         Basically, every element is either a row winner or a col winner or both.
         Except if an element loses in row and loses in col?
         Wait, if element $x$ is at $(r,c)$.
         It is in row $r$. If it loses, $R_r$ is someone else.
         It is in col $c$. If it loses, $C_c$ is someone else.
         So $x$ is "double loser" if $x \neq R_r$ AND $x \neq C_c$.
         For $N=2$, can we have double losers?
         If $R_1 = (1,1)$, $R_2 = (2,2)$.
         Then row 1 winners $(1,1)$. row 2 winners $(2,2)$.
         Col 1 has $(1,1), (2,1)$. Winner is $C_1$.
         Col 2 has $(1,2), (2,2)$. Winner is $C_2$.
         If $C_1 = (1,1)$, $C_2 = (2,2)$.
         Then all 4 cells are winners.
         Wait, $x$ at $(1,2)$ is in row 1. Loser to $(1,1)$.
         In col 2. Loser to $(2,2)$.
         So $(1,2)$ is double loser.
         So we have double losers.
         We discard $(1,2)$.
         So yes, double losers are the ones to discard.
         How many double losers are there?
         Depends on structure.
         Generally, number of double losers is $N^2 - (\text{size of } (R \cup C))$.
         If $R$ and $C$ are disjoint, $2N$. Double losers $N^2 - 2N$.
         If $R=C$, $N$. Double losers $N^2 - N$.
         Wait, we need to discard $N-1$.
         If number of double losers is large, we pick $N-1$ of them.
         But which ones?
         If we pick any, is it safe?
         No, double losers are all weaker than 2 elements.
         Single losers are weaker than 1 element.
         So single losers are "safer" to keep.
         So we should pick double losers to discard.
         But what if we have only $N-2$ double losers?
         Then we must discard a single loser.
         But wait, if $N^2 - |R \cup C| < N-1$.
         This means $|R \cup C| > N^2 - N + 1$.
         This happens if $|R \cup C|$ is large.
         Max $|R \cup C| \le 2N$.
         $2N > N^2 - N + 1 \iff N^2 - 3N + 1 < 0$.
         For $N=2$, $4 - 6 + 1 = -1 < 0$. True.
         So for $N=2$, $2N = 4$. $N^2 - N + 1 = 3$.
         So we can cover the entire set with winners!
         In this case, we have 0 double losers.
         We have 4 winners (potentially).
         We need to keep 3.
         We discard 1.
         We must discard 1 winner.
         Which one?
         One who is "least dominant".
         Dominance: $R_1$ beats $L_1$. $C_1$ beats $L'_1$.
         $R_1$ wins row 1. $C_1$ wins col 1.
         If $R_1$ and $C_1$ are different.
         $R_1$ beats $(1,2)$. $C_1$ beats $(2,1)$.
         $R_2$ beats $(2,1)$. $C_2$ beats $(1,2)$.
         Wait, in $N=2$ with grid.
         Row 1: $(1,1), (1,2)$. Winner $R_1$.
         Row 2: $(2,1), (2,2)$. Winner $R_2$.
         Col 1: $(1,1), (2,1)$. Winner $C_1$.
         Col 2: $(1,2), (2,2)$. Winner $C_2$.
         Suppose $R_1=(1,1)$. Then $(1,1) > (1,2)$.
         Suppose $R_2=(2,2)$. Then $(2,2) > (2,1)$.
         Suppose $C_1=(1,1)$. Then $(1,1) > (2,1)$.
         Suppose $C_2=(2,2)$. Then $(2,2) > (1,2)$.
         So $(1,2)$ lost to $R_1$ AND $C_2$.
         $(2,1)$ lost to $R_2$ AND $C_1$.
         Both $(1,2)$ and $(2,1)$ are double losers.
         Wait, I thought earlier there were 0 double losers?
         Ah, because $R_1$ is $(1,1)$, $C_1$ is $(1,1)$.
         $R_2$ is $(2,2)$, $C_2$ is $(2,2)$.
         So $|R \cup C| = 2$ (since $R_1=C_1$ and $R_2=C_2$).
         Wait, no. $R$ is the set of row winners.
         $C$ is the set of col winners.
         $R=\{R_1, R_2\}$, $C=\{C_1, C_2\}$.
         If $R_1=(1,1), R_2=(2,2)$, then $|R|=2$.
         If $C_1=(1,1), C_2=(2,2)$, then $C=R$.
         Then $R \cup C = \{(1,1), (2,2)\}$.
         Size 2.
         Total elements 4.
         Double losers are $4-2 = 2$.
         Wait, I calculated 0 double losers before assuming $R$ and $C$ are disjoint? No, I assumed $R_1 \neq C_1$ etc?
         If $R=\{ (1,1), (2,2) \}$ and $C=\{ (1,2), (2,1) \}$?
         Is it possible that $C$ winners are the losers of rows?
         $C_1$ winner of col 1 $\{(1,1), (2,1)\}$.
         $(1,1)$ is $R_1$. $(2,1)$ is loser of row 2.
         If $(2,1) > (1,1)$, then $C_1 = (2,1)$.
         But we established $R_1 > (1,2)$ etc.
         Anyway, the "Double Losers" strategy is solid.
         We need to identify elements that are losers in BOTH row and col races.
         These are definitely "slow".
         We collect them.
         If count $\ge N-1$, we pick $N-1$ double losers to discard.
         What if count $< N-1$?
         This implies $|R \cup C| > N^2 - (N-1)$.
         Wait, $|R \cup C| \le 2N$.
         So $2N > N^2 - N + 1$.
         This holds for small $N$.
         If count of double losers is insufficient, we must discard winners.
         But winners are faster than losers.
         If we have to discard a winner $W$, it means $W$ is slower than someone else we kept.
         Who would that be? A loser?
         But we keep losers too!
         Wait, we want to KEEP $N^2 - N + 1$.
         If we discard $N-1$ losers, we keep the rest.
         The rest includes all row winners and all col winners (mostly).
         Actually, if we discard $N-1$ double losers, we are left with the rest.
         The rest includes $N^2 - (N-1)$.
         Wait, we keep exactly that many.
         So if we identify $N-1$ double losers, we just discard THEM.
         Then we return all other elements.
         This set has size $N^2 - (N-1)$.
         Is it guaranteed to be the correct set?
         Only if the bottom $N-1$ are exactly the double losers.
         This implies that NO double loser is faster than any "winner" (single winner).
         Is it true?
         Double loser $d$. Won in neither row nor col.
         So $d < R_{row(d)}$ and $d < C_{col(d)}$.
         If we keep $R_{row(d)}$ and $C_{col(d)}$, then $d$ is slower than at least 2 kept elements.
         Is it possible for $d$ to be faster than a kept element $k$?
         If $k$ is also a double loser?
         Then we have multiple double losers.
         We need to pick WHICH $N-1$ to discard among them.
         If we have $K_{dl}$ double losers.
         We need to pick $N-1$ to discard.
         We know all are slower than 2 elements.
         Maybe we can sort the double losers?
         Sort them by race.
         If $K_{dl}$ is small, easy.
         If large, maybe we don't need to sort them fully.
         Wait, if we have many double losers, we need to pick the "worst" ones.
         Wait, actually, is it guaranteed that the bottom $N-1$ are exactly the double losers?
         In $N=2$, double losers are $\{(1,2), (2,1)\}$. Bottom 1 must be chosen from these?
         We had cases where $L_1$ (single loser in row 1) was kept?
         Wait, $L_1$ in row 1 is a loser in row race.
         If $L_1$ was $(1,2)$, it is also loser in col 2.
         So it IS a double loser.
         Wait, if $L_1$ is $(1,2)$, then it is loser in row 1.
         If it was winner in col 2, it wouldn't be a double loser.
         But $C_2$ (winner of col 2) was $(2,2)$?
         If $(2,2)$ wins col 2, then $(1,2)$ lost to $(2,2)$.
         So $(1,2)$ lost to $R_1$ AND $C_2$. Double loser.
         So yes, in this config, $(1,2)$ is double loser.
         $(2,1)$ is also double loser (lost to $R_2, C_1$).
         So we have 2 double losers.
         We need to discard 1.
         One of them.
         If we discard $(1,2)$, we keep $(2,1)$.
         Is $(2,1)$ guaranteed to be in top 3?
         It's a double loser, so slower than 2.
         But we keep it.
         If we keep it, it must be $\ge$ top 3?
         Rank 3 is allowed.
         If we discard $(2,1)$, we keep $(1,2)$.
         So we need to break the tie.
         Tie breaker: Race them!
         Query `? (1,2) (2,1)`.
         Winner is better (less slow).
         Wait, we want to DISCARD the WORST one.
         So if `(1,2)` beats `(2,1)`, then `(2,1)` is worse. Discard `(2,1)`.
         If `(2,1)` beats `(1,2)`, discard `(1,2)`.
         So yes, we just need to sort the double losers?
         Actually, if we have $K$ double losers, we need to pick $N-1$ to discard.
         So we keep $K - (N-1)$ double losers.
         We need to discard the "worst" $N-1$ of them.
         So we need to select the $N-1$ weakest among double losers.
         This can be done by comparing them.
         Since $N \le 20$, $K$ can be up to $N^2$.
         We need to find the $N-1$ weakest.
         This is equivalent to finding the $K-(N-1)$ strongest among them.
         We can do this with a standard selection algorithm.
         Since we have limited queries, maybe $O(K \log K)$? No, $K$ is large.
         But we only need to pick $N-1$.
         We can use the group comparison property.
         We have $K$ items. We want to find bottom $N-1$.
         This is selection.
         However, wait.
         We also have "Single Losers".
         Are single losers always better than double losers?
         Single loser $s$. Lost to $R_s$. Won nothing? No, won nothing means lost to someone.
         Wait, single loser means it WON in one race? No, "Not winner in both".
         Single loser means Winner in exactly one race (Row OR Col).
         Wait, everyone is a loser in at least one race.
         Winners are those who won their race.
         Intersection $W_{rc} = R \cap C$.
         $W_r = R \setminus C$. (Won Row, lost Col).
         $W_c = C \setminus R$. (Won Col, lost Row).
         $L = U \setminus (R \cup C)$. (Lost both).
         So $L$ are double losers.
         Are elements in $L$ always slower than elements in $W_r, W_c, W_{rc}$?
         $x \in L \implies x < R_{row(x)}$ and $x < C_{col(x)}$.
         $R_{row(x)}$ is in $R$ (so either $W_r, W_c, W_{rc}$).
         So $x$ is slower than an element in $W \setminus L$.
         So $L$ is strictly "below" $R \cup C$.
         So yes, double losers are the weakest candidates.
         Wait, are elements in $L$ necessarily slower than EACH other?
         No. Two double losers can be incomparable.
         But both are slower than 2 others.
         If we pick $N-1$ from $L$ to discard.
         The remaining elements in $L$ are kept.
         The elements in $W$ are all kept (since we only discard from $L$).
         Wait, total elements = $|L| + |W|$.
         We keep $|W| + (|L| - (N-1))$.
         Total kept = $N^2 - N + 1$.
         This implies $|L| \ge N-1$ is required for this strategy to work.
         If $|L| < N-1$, we have a problem.
         When does $|L| < N-1$ happen?
         $|L| = N^2 - |R \cup C|$.
         Max $|R \cup C| = 2N$.
         So min $|L| = N^2 - 2N$.
         For $N=20$, $400 - 40 = 360 \ge 19$.
         For $N=2$, $4 - 4 = 0$.
         So for $N=2$, $|L|$ can be 0.
         Wait, if $|L|=0$, then $R \cup C$ covers all elements.
         Then we need to discard $N-1$ elements FROM $R \cup C$.
         But $R \cup C$ elements are "winners" in at least one race.
         So they are faster than their opponents.
         Wait, if $|L|=0$, it means every element won either row or col race.
         This means for every cell $(i, j)$, it is either the row winner or col winner.
         This requires specific structure.
         In this case, we have a conflict.
         But wait, we need to discard $N-1$ elements.
         If we discard winners, we must ensure they are not in top set.
         But winners are fast.
         However, maybe the top set includes all $R \cup C$ except one?
         Yes.
         So we need to identify the $N-1$ "weakest" elements among $R \cup C$.
         These are likely the ones that won "weakly"?
         Wait, everyone in $R$ won their row. Everyone in $C$ won their col.
         If $x \in R \cup C$, it beat some opponents.
         If $x \in L$, it beat NONE (strictly, no, it lost).
         Wait, definition of "won".
         $x \in R \implies x$ won row $i$.
         So $x$ beat $n-1$ items.
         If $x \in L$, $x$ lost row $i$ (beat none? No, lost means $R_i$ is someone else).
         Does losing mean beating 0? No, it just means there is someone faster.
         But we don't know if $x$ beat anyone.
         However, we know $x < R_i$ and $x < C_j$.
         So $x$ is bounded by 2 elements.
         Elements in $R \setminus C$ are bounded by 1 (only lost to Col winner).
         Wait, if $x \in R \setminus C$, $x$ won row $i$.
         So $x$ beat $n-1$ elements.
         So $x$ is relatively strong.
         So elements in $L$ are the weakest.
         If $|L| \ge N-1$, we discard $N-1$ elements from $L$.
         Which ones?
         We need to find the $N-1$ weakest in $L$.
         Wait, if $x, y \in L$. $x < R_x, x < C_x$. $y < R_y, y < C_y$.
         Do we know $x < y$? Not necessarily.
         But we can compare them using races.
         We can race all elements of $L$?
         Wait, $|L|$ can be large.
         But we need to discard $N-1$.
         Actually, if we just take the "bottom $N-1$" of the whole set?
         Wait, if we use the algorithm:
         1. Find Row Winners $R$.
         2. Find Col Winners $C$.
         3. If $|R \cup C| < N^2 - N + 1$? No, $R \cup C$ is small ($2N$).
            Wait, we need to KEEP large set.
            So we keep $R \cup C$ PLUS some elements from $L$.
            If $|L| \ge N-1$, we can discard $N-1$ elements from $L$.
            But do we know which ones are the "worst"?
            Actually, we can't easily distinguish $x, y \in L$ without racing.
            However, notice that the problem statement guarantees existence.
            And maybe we can simply output ALL elements EXCEPT the ones that are "losers of the winners race".
            Wait, this sounds like the tournament method.
            
      Let's refine the solution.
      Algorithm:
      1. Race all $N$ rows. Get winners $R_1 \dots R_N$.
      2. Race all $N$ cols. Get winners $C_1 \dots C_N$.
      3. We have $2N$ winners.
      4. Now, perform a "Winner-Winner" race.
         Put all $R_i$'s and $C_j$'s into groups of size $N$ and race them to find the global maximum.
         Wait, we need to sort them to find the cut-off.
         Actually, we can find the top $K$ of $2N$ elements.
         But we also need to include elements from $L$.
         This path is getting complicated.

      Alternative:
      The problem is simple enough that maybe I can just iterate.
      We want to find $N^2 - N + 1$ fastest.
      We can maintain a set of candidates.
      Since $N \le 20$, we can try to find the "worst" $N-1$ elements by iterating.
      For each test case:
        Current set of un-decided elements $S = \{1..n^2\}$.
        We want to remove $n-1$ elements.
        Pick $n$ elements from $S$. Race them.
        Remove the $n-1$ losers?
        If we pick randomly, we might remove a fast element.
        But we can pick strategically.
        What if we pick $n$ elements that are "known to be slow"?
        We can track the "loss history".
        
      Actually, I will implement the solution described in the "Editorial" for this problem (Pepe's Speeds), which I will reconstruct mentally.
      The logic is:
      1. We can simply discard the $n-1$ losers of the row winners race?
         Wait, if we race the $n$ row winners, we get 1 winner and $n-1$ losers.
         The losers are row winners who lost to someone.
         These are "fast" relative to others, but maybe not global top.
         But wait.
         If we race row winners, we get $W$.
         The losers $L_w$ are row winners.
         We also have losers of rows.
         Actually, the solution is:
         The set of $N-1$ elements to discard are exactly the losers of the race of the $N$ row winners.
         WAIT.
         In $N=2$, we discarded $L_2$.
         If we race row winners $\{R_1, R_2\}$, losers is $\{R_2\}$ (if $R_1$ wins).
         Is $R_2$ the same as $L_2$?
         No. $R_2$ is a winner. $L_2$ is a loser.
         But in my example $L_2$ was discarded.
         Wait, $L_2$ was row loser.
         So we discarded a row loser.
         But the rule "discard losers of row winners race" would suggest discarding $R_2$.
         But $R_2$ is a row winner.
         If we discard $R_2$, we might discard the 2nd fastest element.
         Is it possible that $R_2$ is the 2nd fastest?
         Yes.
         If we discard $R_2$, we are left with $\{R_1, L_1, L_2\}$.
         In that case, $L_2$ is faster than $R_2$? No, $R_2 > L_2$.
         So $\{R_1, L_1, L_2\}$ contains $L_2$ (slow) and misses $R_2$ (fast).
         So we must keep $R_2$.
         So the solution "discard losers of row winners race" is INCORRECT.
         
      So we must discard something else.
      Actually, the "slowest $N-1$" are the losers of the races where they participated.
      Wait.
      The correct algorithm:
      1. Race all rows. Keep winners $R_i$. Discard losers $L_i$.
         Wait, we can't discard $L_i$ yet.
      2. We need to find $N-1$ elements to discard.
      3. The correct set to discard is: The $N-1$ elements that are losers in BOTH row race AND col race.
         (Assuming we can determine this set).
         But we saw for $N=2$, $|L|$ (double losers) was 2. We need to discard 1.
         So we pick 1 from $L$.
         But we need to know which one is slower.
         Since $N$ is small, we can compare elements of $L$ using a tournament.
         Wait, we need to order them.
         
      Let's write the solution based on finding double losers and discarding the worst $N-1$.
      To be safe, I will also include a fallback or just ensure I compare all double losers if $|L| < N-1$.
      But for $N=2$, $|L|$ can be small? No, $|L| \ge N^2 - 2N$.
      For $N=2$, $|L| \ge 0$.
      If $|L| < N-1$, we can't rely on $L$.
      Wait, I derived $|L| \ge 0$.
      But wait, if $|L|
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=1532)

````
Let me break down this problem:

1. There are n² pepes with distinct speeds
2. In one race, we can choose exactly n distinct pepes and learn only the fastest among them
3. We need to find the top n² - n + 1 fastest pepes (the bottom n-1 are indistinguishable)
4. The interactor is adaptive

Key observation: Since each race gives us exactly n pepes, and a pepe never races against itself in another position, any pepe participates in some races but we're told it will be the fastest of those races. This means if a pepe wins k races where each win involves n different opponents, it beat at least k(n-1) other pepes. 

So the number of races for pepe i is related to how many faster pepes exist above it. Let's say pepe i has p faster pepes above it. Then pepe i would win approximately (n²-n+1-p)/n races, and the winner would have won all races containing that pepe as the fastest result. So the first pepe in our answer should have roughly ((n^2-n+1)-1)/n = (n^2-n)/n = n-1 races below it. The next should have about n-2 races below it, and so on.

The approach: maintain a list of candidate winners for each expected rank, starting from candidates that could occupy positions 0 through n-1. For each rank k, track which pepe could potentially have k faster pepes above it. We start by initializing candidates: pepe 1 can't have anyone faster above it so go in list 0, pepe 2 could have pepe 1 faster so goes in list 1, and so on up to pepe n going in list n-1. Then process ranks from lowest to highest, testing whether each candidate truly belongs there.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;

        vector<int> cand(n * n);
        for (int i = 0; i < n * n; ++i)
            cand[i] = i + 1;

        // For each pepe, track how many races they have won (i.e., appeared as fastest)
        vector<int> wins(n * n + 1, 0);

        // Process each expected rank of the result from position 1 to n^2-n+1
        for (int pos = 1; pos <= n * n - n + 1; ++pos) {
            vector<int> r = vector<int>(cand.begin(), cand.end());
            sort(r.rbegin(), r.rend());
            r.erase(unique(r.begin(), r.end()), r.end());

            auto check_pepes = [&](vector<int>& p) -> void {
                for (int i : p) {
                    cout << "? " << i << " ";
                    for (int j = i; j <= n * n; j += n) {
                        cout << j + 1 << " ";
                    }
                    cout << "\n";
                    int best;
                    cin >> best;
                    wins[best]++;
                    for (auto &x : p)
                        if (x == best) x /= n;
                }
            };

            vector<int> l = {}, u = {};
            bool possible = true;
            for (int i = 0; i < r.size(); i++) {
                int pe = r[i];
                if (wins[pe] >= pos - 1)
                    l.push_back(pe);
                else if (wins[pe] <= n * n - n - pos + 2)
                    u.push_back(pe);
                else {
                    possible = false;
                    goto done_check;
                }
            }

        done_check:;
            if (!possible || l.empty() || u.empty()) {
                if (!possible) {
                    cout << "! ";
                    for (int i = 1; i <= n * n; i++)
                        cout << i << " \n"[i == n * n];
                } else {
                    if (l.empty()) {
                        l = u;
                        continue;
                    }
                    if (u.back() != n * n - pos + 1) {
                        u.pop_back();
                        continue;
                    }
                    u.insert(u.begin(), n * n - pos + 1);
                }
            }

            if (!l.empty())
                check_pepes(l);
            else
                check_pepes(u);

            vector<vector<int>> s(n * n);
            for (int i = 1; i <= n * n; i++) {
                int cnt = wins[i];
                long long target = (cnt / n);
                if (i > 0 && cnt % n != 0 && wins[i - 1] % n == 0) target -= 1;
                if (target >= cnt / (double)n)
                    target -= 1;
                s[target].push_back(i);
            }

            long long cnt = 0;
            for (int i = 0; i < n * n; i++) {
                auto& v = s[i];
                for (size_t idx = v.size() - 1; idx > 0; --idx) {
                    if (v[idx - 1] != i) {
                        swap(v[idx - 1], v[idx]);
                        cnt++;
                    }
                }
                for (size_t idx = v.size() - 1; idx > 0; --idx) {
                    if (v[idx - 1] == i && v[idx] != 0) {
                        cnt += v[idx] / n;
                        cout << "! ";
                        cout << i << " ";
                        for (size_t j = idx; j + 1 < v.size(); j++)
                            cout << v[j] / n << " ";
                        cout << "\n";
                    }
                }
            }
            if (cnt == 0 && n > 2) {
                for (size_t i = 0; i < v.size() - 1; i++) {
                    cout << "! ";
                    cout << v[i] << " ";
                    for (size_t j = i + 1; j + 1 < v.size(); j++)
                        cout << v[j] / n << " ";
                    cout << "\n";
                }
            } else if (cnt == 0) {
                cout << "! ";
                for (int i = 1; i <= n * n; i++)
                    cout << i << " \n"[i == n * n];
            }
        }
    }
    return 0;
}
```
````
