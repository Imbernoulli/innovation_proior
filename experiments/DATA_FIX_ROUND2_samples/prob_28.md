# 题 28 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The user wants me to solve an interactive problem called "Hacking the Project".
I need to read from standard input and write to standard output.
The problem is about finding $N$ distinct words used by a compiler, all of length 1 to 10.
The interaction protocol allows me to query prefixes $S$. For a prefix $S$, I can request up to $K$ lexicographically smallest words starting with $S$. The service returns these words and counts them (let's say $k$). If $k < K$, it means there are fewer than $K$ words.
The goal is to find all $N$ words such that the sum of $K$ across all queries is minimized. The constraint says if the sum of $K$ exceeds 4000, I get 0 points. Actually, the limit for sum of $K$ per test case seems related to this bound.
Wait, let's re-read carefully. "sum of K in those queries is as less as possible". "If value is greater than 4000, the solution will get 0 points." This implies a strict upper bound on total $K$ usage, or rather, the score depends on minimizing it, but > 4000 guarantees failure.
Actually, usually in such problems, the constraint is that we must stay within limits. Let's check the constraints again. $N \le 1000$. Words length $\le 10$.
Time limit 1s. Memory 512MB.

Let's analyze the properties of the search space.
We have a dictionary of $N$ words. All words start with lowercase English letters 'a'-'z'.
Max word length is 10.
If I query root '' with $K=N$, I would get all $N$ words in sorted order? Wait, does querying an empty string return all words?
The problem statement says: "query S K ... getK ... words starting with prefix S". An empty string is a prefix of every word. So if I query '' $N$, I should get all words.
However, $S$ has length at least 1 ("1 ≤ |S| ≤ 10").
So I cannot query the empty string directly to get everything at once.
But maybe I can query single characters? 'a', 'b', ..., 'z'.
If I know which branches exist in the trie formed by these words, I can drill down.

Let's think about the structure.
All words are nodes in a Trie.
If I ask for "a" $K$, and get $k$ words, then there are at least $k$ words in the subtree of 'a'.
If I ask for "a" 1, and get nothing, then no words start with "a".
Since I need to reconstruct *all* $N$ words, I essentially need to explore the Trie until I have found all leaves representing the words. But the words themselves are what matters.
Wait, since the words are distinct and consist of length 1-10, they correspond to leaves or internal nodes marked as "is_word" in a Trie. But actually, the problem just asks for the set of words.
Crucially, I don't need to output the Trie structure, just the list of words.
Also, the words are given in lexicographical order by the query.
If I query a prefix $P$, the response gives me all matching words sorted.
To minimize sum of $K$, I should be smart.
Is there a way to recover the exact words without knowing $N$ exactly beforehand?
Well, I am given $N$ at the start.
Total words = $N$.
If I query "a" $K$, and get count $k_a$. Then I know exactly $k_a$ words start with "a".
Then I can recursively query "aa", "ab", etc., only for the ones that exist.
This sounds like traversing a Trie.
Since the maximum depth is 10, and alphabet size is 26, the total number of nodes in a full Trie is small? No, $26^{10}$ is huge. But only $N=1000$ words exist.
Most paths won't be taken.
If I use a simple strategy:
Maintain a set of candidate prefixes that *might* contain words.
Initially, the set is { "" }? But query $S$ requires $|S| \ge 1$.
Actually, any word starts with some character.
If I query each char 'a'...'z' with $K=N$, I get the distribution of words among first characters.
Suppose 'a' returns $k_a$ words. Then 'b' returns $k_b$, etc.
Sum of $k_i$ should be $N$. Since words are distinct, $\sum k_i = N$.
So querying $N$ times for single chars costs $26 \times N$ roughly? That might be too much if $N=1000$, total $K = 26000$. Limit is 4000.
So we need a more efficient way. We need to prune branches that have 0 words.
But initially we don't know which branch has 0 words.
Can we query just one word? No, we need prefix.
Maybe we can query shorter strings? No, minimum length is 1.
Wait, if I query "a" $K=1$, and I get something, I know there is at least one word. If I get 0, there are none.
Query cost for "a" $K=1$ is 1. Query "a" $K=2$ costs 2.
To find out how many words start with "a", we could binary search for the count? No, the function `count(prefix)` is monotonic with respect to extending prefix? No. `count("a")` >= `count("ab")`.
But I can't just binary search the length easily because I don't know if a node exists.
Actually, I can maintain a list of current prefixes that contain words.
Initially, I know nothing. I need to identify which branches are valid.
This is effectively searching for a set of words in a tree where edges are labeled 'a'-'z'.
Since max length is 10, the depth is limited.
With $N=1000$, maybe we can afford to query specific paths.
Consider the trie structure. Each word corresponds to a path.
We want to output the words.
Let's try to build the Trie implicitly.
However, we are restricted on $\sum K$.
If I query "a" 1, I get a list of 0 or 1 words.
If I find a word $w$ starting with "a", I still need to check if there are more.
Actually, if I query "a" 1, I get the smallest word starting with "a".
If there are more, I need another query?
No, the query asks for $K$ words starting with $S$. It returns them ALL sorted.
So if I query "a" 1, I get the 1st word. If the system returns $k>1$ words, I have them all!
Wait, "returns... the first K words... If there are only k < K words, the service gives out only k words".
So if I query "a" 1, I get either 1 word or 0 words. If I get 0, no words start with "a". If I get 1, there is at least one word.
Does this tell me how many words start with "a"?
Suppose there are 100 words starting with "a". If I query "a" 1, I get 1 word. I don't know if there are 1 more, 99 more, etc.
Unless I keep asking? No, I cannot ask "how many are there?" without using more $K$.
Wait, I can ask "a" 2, then I get 2nd word if it exists.
So basically, to discover *all* words starting with "a", I can just query "a" with a large enough $K$ to retrieve them all?
But I don't know how many start with "a".
If I query "a" 1000 (since $N \le 1000$), and $k$ words come back, then I know there are $k$ words starting with "a".
Cost is 1000.
If I do this for 'a', 'b', ..., 'z', worst case $26 \times 1000 = 26000$. Too high.
We need to save queries.
Key observation: Total words $N$.
If I query a prefix $S$, and it returns $k$ words. If $k=0$, I stop this branch.
If $k > 0$, I found $k$ words.
Can I optimize?
Actually, notice that words have lengths between 1 and 10.
The total number of words is $N$.
If I just want to collect all words.
Is it better to guess the next character?
Maybe I can perform a traversal.
Let's consider a priority queue or similar structure of potential prefixes.
Or simpler: The words are unique.
Is it possible to determine the count for a prefix without querying that prefix directly? No.
So we must query.
Wait, if I query "a" 1 and get word $W_1$. Do I know there are other words? Not necessarily.
BUT, I know that $W_1$ is the lexicographically first word starting with "a".
If there was another word "abc" coming after "abd" (wait, lexicographical), I would see it.
Actually, if I get a list of $k$ words, I get all of them starting with $S$ that are "next" to each other.
Specifically, they are returned in lexicographical order.
If I query $S$ and get $k$ words, I have consumed $k$ towards the limit? No, I paid $K$.
If I paid $K=1$ and got $k=1$ word, I pay 1 to get 1 word info.
If I paid $K=1000$ and got $k=1000$ words, I pay 1000 to get 1000 words.
It seems paying a larger $K$ upfront is better because it retrieves all children in one go.
Wait, if I query "a" 1000 and get 10 words, I pay 1000. That's terrible.
If I could have queried "a" 10, I pay 10.
So the optimal strategy involves knowing exactly how many words are there before setting $K$.
But how to know?
We know that total words is $N$.
Maybe we can deduce counts from the hierarchy.
Actually, if I know that there are $C$ words starting with $S$, I can query $S$ $C$.
But I don't know $C$.
However, I can query smaller prefixes?
For example, if I query "a" 1 and get "apple", "apricot". Wait, if $K=1$ and there are 2 words, I only get 1 word.
So querying with small $K$ is risky as it doesn't give me all words in the subtree, unless $K$ is at least the subtree size.
But I want to minimize $\sum K$.
$\sum K = \text{cost}$.
Every time I make a query $(S, K)$, I gain information.
Specifically, if $k$ words are returned ($k \le K$), I learn that there are $k$ words with prefix $S$.
And I learn their content.
Wait, if $k < K$, then I know the count is $k$.
If $k = K$, I know there are at least $K$ words.
The goal is to eventually collect all $N$ words.
So eventually for every leaf representing a word, I need to identify it.
Actually, identifying the words themselves is enough.
The most expensive part is guessing $K$ for a subtree that is large.
If I query $S$ with $K_{subtree}$, I pay $K_{subtree}$.
If I query $S$ with $K_{guess}$ and $k_{actual}$ comes back.
If $K_{guess} < k_{actual}$, I missed words.
But wait. If I query "a" $K$, and get $k$ words.
If $k < K$, I know count is $k$. I have retrieved all of them? Yes, because they are returned.
If $k = K$, I know there are at least $K$ words.
So if I query with $K$, and get $k$ words:
1. If $k < K$: I have found all words starting with $S$. Cost $K$. Gain: $k$ words.
2. If $k = K$: I have found $K$ words. There might be more. I must continue deeper or with larger $K$.
   Cost $K$. Gain: $K$ words so far.
   
This looks like binary search on the number of words in a prefix subtree?
No, I need to retrieve the words.
Suppose I want to collect all words in the range [start, end] sorted.
Actually, the simplest approach is to maintain the trie structure conceptually.
Start with root. For each child 'c' in ['a', 'z'], I need to visit.
But I can't iterate 'a' to 'z' blindly because many might be empty.
I need a mechanism to find non-empty children.
Maybe I can just query single words? No, words are unknown.
Wait, is it possible to find the first word of the dictionary overall?
Query "" ? No, min length 1.
Query "a" 1. If result > 0, then "a" is a prefix. The first returned word starts with "a".
Wait, I don't know the order of first letter across branches.
Actually, I can maintain a list of candidates.
Initially, the set of known prefixes is $\{ "a", "b", ..., "z" \}$.
I want to check which of these contain words.
Checking "a": query "a" 1.
If result is 0 words, discard "a".
If result is 1 word $w_1$, then we know count is at least 1. Is it more?
We might need to query "a" 2.
Actually, if I query "a" 1 and get $k$ words.
If $k < 1$, impossible (must be 0 or 1 since $K=1$). So 0.
If $k=1$, then I found 1 word. It could be that there are more.
So for every active prefix $P$, I need to query its count.
But how to determine count efficiently?
If I query $P$ $K=P\_len \cdot N$? No.
Wait, look at the constraints. $N \le 1000$.
If I just use the fact that I want to find *all* words.
Every word is distinct.
If I manage to extract a word $w$, I remove it from consideration?
But words are not extracted one by one easily. They come in batches.
What if I simply query every character with $K=N$?
If the distribution is very skewed (e.g. all words start with 'a'), this is fine (cost $N$).
But if words are spread evenly ('a' through 'z'), cost $26N \approx 26000$. Fails.
So we must avoid querying characters with 0 words.
How to avoid? By querying "smaller" chunks.
But 'a'.. 'z' are disjoint sets of first letters.
Maybe query all 'a'..'z' with $K=1$?
Sum $K = 26 \times 1 = 26$. Very cheap.
Result: for each char, I get 0 or 1 word.
If I get 1 word from 'a', I now know count of 'a' $\ge 1$.
If I get 0, count is 0.
Now for the ones with count $\ge 1$.
Wait, getting 1 word from 'a' with $K=1$ just tells me the first word.
It does NOT tell me if there is a second word starting with 'a'.
But I can repeat the query?
Wait, the state of the server persists? "The jury program is adaptive".
Yes, the database contains the words. My queries consume credits but don't change data.
However, if I query "a" $K=1$ twice, I get the same answer.
To find if there are more words starting with 'a', I can query "a" $K=2$.
If it returns $k=2$, then I need more.
So effectively, for each character $c$ that appeared in the $K=1$ query:
I need to find the exact count.
Let $cnt[c]$ be the count.
I can binary search the count for $c$?
Low = 0, High = $N$.
Check mid: query "$c$" $mid$.
If returns $k < mid$, actual count is $k$. Stop.
If returns $mid$, actual count $\ge mid$.
Wait, if query returns $k$ words, does it mean count is $k$ or count $\ge mid$?
It returns *first* $K$ words.
If $k < K$, then count is definitely $k$. (Because there were fewer than $K$ available).
If $k = K$, then count $\ge K$.
So yes, I can binary search for the exact count of words starting with $c$.
Binary search takes $\log N$ queries.
Each query costs $mid$ in $K$.
Total cost for one character $c$ roughly $\sum_{steps} \frac{High}{2}$.
If I assume the distribution of words is uniform or worst case (all words start with 'a').
If I have to find count for 'a', it could be 1000.
Binary search steps:
1. Check $K=500$. Return 500. Cost 500.
2. Check $K=750$. Return 750. Cost 750.
...
Total cost $\approx 1000 + 750 + \dots \approx 2000$.
Doing this for all 'a'...'z'?
If sum of counts is $N$, maybe total cost is manageable?
If I distribute words such that 'a' has 1, 'b' has 1, ...
Total cost for finding count: $26 \times \log N \times (\text{avg K})$. Avg $K$ around $N/26$?
Wait, if I query with $K$ large, it might hit $K=N$ immediately.
If I query 'a' with $K=1$, cost 1. Get 0 or 1.
If I get 1, I know count $\ge 1$.
If I get 0, count 0.
Then for those with count $\ge 1$, I need to find exact count.
Example: 'a' has 10 words.
BS:
Check 5: get 5 (cost 5). Count $\ge 5$.
Check 8: get 8 (cost 8). Count $\ge 8$.
Check 9: get 9 (cost 9). Count $\ge 9$.
Check 10: get 10 (cost 10). Count $\ge 10$?
Wait, if I query with $K=$ estimated count and get $K$ words, I haven't confirmed it's exact.
But if I query $K=est+1$ and get $K_{ret} = est+1$?
No, if I query $K=X$ and get $X$ words, it could be that count is $X, X+1, \dots$.
So binary search logic holds only if we treat "return $X$" as "count $\ge X$".
But we need to be careful. If I query $K=1000$ and get 1000 words (where total in universe is 1000), I know count $\ge 1000$. Since max is 1000, it is 1000.
But here $N$ is given. So max count for a prefix cannot exceed $N$.
So if I query $c$ with $K=N$, I get all words starting with $c$ (if they exist).
If I use BS to find exact count $M_c$ for each $c$:
Total cost $\sum_c M_c \times \log(\dots)$?
No, the query costs are the parameters $K$ passed.
The total budget is 4000.
If I do this, let's trace worst case.
Suppose $N=1000$, all words start with 'a'.
1. Query 'a' 1. Get 1 word. Cost 1. Count $\ge 1$.
2. BS for 'a':
   Range $[1, 1000]$.
   Mid 500. Query 'a' 500. Get 500. Cost 500.
   ...
   Eventually reach 1000. Cost $\approx 1000$.
   Sum of costs for 'a': $1 + 500 + 750 + \dots \approx 2000$.
   Plus other chars cost 1 each (to confirm 0) -> 25.
   Total $\approx 2000$. Fits!
   What if words are distributed evenly? 'a' 250, 'b' 250...
   For each char:
   1. Check 1. Get 1. Cost 1. (26 times = 26).
   2. BS:
      'a': $N_a=250$.
      Mid 125 -> get 125. (125)
      Mid 190 -> get 190. (190)
      Mid 220 -> get 220. (220)
      Mid 235 -> get 235. (235)
      Mid 245 -> get 245. (245)
      Mid 250 -> get 250. (250)
      Sum $\approx 1 + 125 + 190 + 220 + 235 + 245 + 250 \approx 1200$.
      Times 26 chars $\approx 31000$. Way over limit.
   
So binary searching the count for every single character is bad if $N$ is large and distribution is even.
Wait, my cost analysis for BS is wrong.
In BS, if query($K$) returns $k < K$, then exact count is $k$. We STOP for this branch.
If query($K$) returns $K$, then exact count $\ge K$. We CONTINUE.
Wait!
If I query 'a' 1, and get 1 word.
This means count $\ge 1$.
Next I query 'a' 2. If get 2, count $\ge 2$.
If I get 1, count is exactly 1.
Wait, if I get 1 when asking for 2, it means count is 1. Because if count was 2, I would get 2.
Ah! So if $returned == requested$, it implies count $\ge requested$.
If $returned < requested$, count is $returned$.
So in BS:
Range $[L, R]$. $ans=L$.
Try $M$. If $get(M) == M$: $L = M$. Else ($get(M) < M$): $R = M-1, ans = get(M)$.
Wait, if $get(M) < M$, I get exact count $g$. So I stop and store $g$.
The cost is $M$.
If $get(M) == M$, cost is $M$.
So to minimize cost, I want to find cases where $get(M) < M$ quickly.
That happens when count is small.
But if count is large, I spend lots.
Actually, wait.
If I query 'a' 1 and get 1 word. I know count $\ge 1$.
Then I query 'a' 2.
If I get 2 words, count $\ge 2$.
If I get 1 word (impossible? No, if count=1, I get 1), then count is 1.
Wait, if I query 'a' 2 and count is 1, I get 1 word. Cost 2.
If I did 'a' 1, I get 1 word. Cost 1.
Why not just query 'a' 1, 'a' 2, ... until count drops?
If count is 1, I query 1 (cost 1), get 1. Query 2 (cost 2), get 1.
Total cost 3.
If I know count is 1, why query 2?
To find if count is 1, I need to verify.
Actually, if I query 'a' $K$ and get $k < K$, I have verified the count is $k$.
If I query 'a' $K$ and get $k = K$, I need more info.
So, for any prefix $S$, I want to determine the total count of words in its subtree.
Let's denote count(S).
I want to compute count(S) for the root and all its immediate children.
But I don't know the tree.
I know I have $N$ words total.
I can pick the "cheapest" way to determine the global set.
Wait, the problem is simpler: I just need to find all $N$ words.
I don't need to construct the trie explicitly.
If I can find a word, great.
Actually, I can iterate on the possible words?
No, length up to 10. $26^{10}$ is huge.
But total words is $N$.
If I query a prefix that yields $k$ words, I get $k$ words for free!
Wait, getting them costs $K$.
If $k < K$, I paid $K$ for $k$ words.
If I pay $K=k$, I also get $k$ words.
So ideally, I want to match $K$ to the actual number of words in a batch.
This suggests I should estimate the count accurately.
Or maybe I can just maintain a Trie where I insert the words I find?
Wait, if I query a prefix and get a word, I know this word exists.
Can I use this to guide future queries?
If I get 'apple', then 'app', 'appa', ... might be prefixes.
I know that 'apple' contributes to the count of 'appl'.
Actually, I can just collect all words into a list `found_words`.
But the problem asks to minimize $\sum K$.
Wait, the condition is $\sum K \le 4000$.
With $N \le 1000$.
Average $K$ per word is 4?
If I query $S$ 1, I get 1 word. I know it is $S + \dots$.
If I keep expanding prefixes?
This feels like traversing the Trie but using queries to verify existence.
But queries are expensive.
Let's rethink the "BS count" idea.
Instead of doing it for all 26 characters, I can do it greedily.
I have $N$ words total.
Suppose I check 'a'. Query 'a' 1.
If 0, done with 'a'.
If 1, I have 1 word. Do I check 'a' 2?
If I check 'a' 2 and get 1, I pay 2 for confirmation.
Is there a better way?
Maybe query 'a' 2 directly?
If I query 'a' 2 and get 2, I pay 2, count $\ge 2$.
If I get 1, I pay 2, count is 1.
Comparing:
Strategy 1: Query 1 -> get 1. Pay 1. Then query 2 -> get 1. Pay 2. Total 3.
Strategy 2: Query 2 -> get 1. Pay 2. Done (knows count is 1).
So Strategy 2 is better if count is small.
What if count is large? Say 1000.
Strategy 1: Query 1 (1) -> 1. Query 2 (2) -> 2. ... Query 500 (500) -> 500. Query 1000 (1000) -> 1000.
Sum $\approx \sum i \approx N^2 / 2$. Huge.
Strategy 2 (Binary Search):
Target $M=500$. Query 500. Pay 500. Get 500.
Target $M=750$. Query 750. Pay 750. Get 750.
Target $M=875$. Query 875. Pay 875. Get 875.
Target $M=937$. Query 937. Pay 937. Get 937.
Target $M=968$. Query 968. Pay 968. Get 968.
Target $M=984$. Query 984. Pay 984. Get 984.
Target $M=992$. Query 992. Pay 992. Get 992.
Target $M=996$. Query 996. Pay 996. Get 996.
Target $M=998$. Query 998. Pay 998. Get 998.
Target $M=999$. Query 999. Pay 999. Get 999.
Target $M=1000$. Query 1000. Pay 1000. Get 1000.
Total is roughly $\sum \text{probed values}$. With BS, I probe roughly $\log N$ times?
No. If the result is always equal to the request ($count \ge req$), I probe values $N/2, 3N/4, 7N/8 \dots$.
Sum is $O(N)$. Specifically $N(1 + 1/2 + 1/4 + \dots) \approx 2N$.
So for a count of $C$, BS cost is $O(C)$.
Wait, really?
Let's check the sequence $x_k = C - C/2^k$.
Sum of $x_k$ over logarithmic steps? No.
Steps: $C/2, 3C/4, 7C/8, \dots, C$.
Sum is roughly $C \times (1/2 + 3/4 + 7/8 + \dots)$.
The terms are close to $C$. So if count is large, cost is approx $C \times \text{something large?}$.
Wait. If I query $K=C$, I get $C$.
If I query $K=2C$, I get $C$ (since count is $C$).
Wait, earlier I said if $K > count$, result is $count$.
If $K \le count$, result is $K$.
So if I query $K=C$, result is $C$.
If I query $K=2C$, result is $C$. (Assuming $2C$ is large enough).
Ah! Here is the key optimization.
If I query $K$ and get $K$, I know count $\ge K$.
If I query $K$ and get $k < K$, I know count $= k$.
So if I suspect count is large, I should query with a large $K$ to check if it cuts off.
Example: Suspect count is 1000.
Query 500.
Case 1: Returns 500. Count $\ge 500$. Continue.
Case 2: Returns < 500. Exact count known. Stop.
If Case 1 (Count is indeed 1000). Cost 500. Now I know count $\in [500, 1000]$.
Next query? 750.
If returns 750. Cost 750.
...
Eventually I converge to count.
If count is 1000, I will do queries: 500, 750, 875, 937... up to 1000.
Cost $\approx 500+750+875+\dots \approx 1000$.
Wait, this is still $O(N)$ per subtree.
If I have 26 subtrees with total words $N$.
Sum of costs $\approx \sum O(C_i) = O(\sum C_i) = O(N)$.
Wait, if cost is linear in $C_i$, then total cost is proportional to $N$.
Since $N=1000$, total cost $\approx 2N \approx 2000$.
This is well within 4000!
Wait, is my assumption "Cost is linear" correct?
If I query $K$, cost is $K$.
The sequence of queries in BS for exact count $C$ (where $C \le N$):
Start range $[0, N]$. Mid $N/2$.
If $ret < N/2$, exact $ret$. Cost $N/2$.
If $ret = N/2$, count $\ge N/2$. Cost $N/2$.
Next range $[N/2, N]$. Mid $3N/4$.
If $ret < 3N/4$, exact $ret$. Cost $3N/4$.
If $ret = 3N/4$, count $\ge 3N/4$. Cost $3N/4$.
Next range $[3N/4, N]$. Mid $7N/8$.
And so on.
The sequence of $K$s is roughly $N/2, 3N/4, 7N/8, \dots$.
The sum of these is $N \sum (1 - 1/2^i) \approx N \times 1 \approx N$.
Actually, the last term is $N$. So sum is $N (1/2 + 3/4 + \dots)$.
$1/2 + 3/4 = 1.25$. $1/2 + 3/4 + 7/8 = 1.625$.
Converges to 2.
So cost is roughly $2 \times C$.
With $N=1000$, $C \le 1000$. So max cost per branch is 2000.
Summing over all 26 characters:
If all words are in 'a', cost $\approx 2000$.
If words distributed evenly (each has $C=38$), cost $\approx 26 \times (2 \times 38) \approx 1900$.
Wait, if words distributed unevenly, e.g. 'a' has 500, 'b' has 500.
'a' cost $\approx 2000$.
'b' cost $\approx 2000$.
Total 4000.
This fits perfectly!
Wait, if I have 26 branches, and I allocate $K$ based on the remaining capacity $N$?
But I only query 'a' through 'z' if I know they are non-empty?
No, I can just query all of them.
Querying 'z' with $K=N$ where no words start with 'z'.
If I query $K=N$ and get 0 words. Cost $N$.
This is bad. I should query with small $K$ first to verify 0?
Query 'a' 1. Cost 1. If 0 words, done. Cost 1.
If 1 word, then proceed to BS for 'a' with target $N$.
Wait, if count is 1, BS will take steps:
Mid $N/2$. Query $N/2$. Get 1. Cost $N/2$.
Stop. Total cost $1 + N/2$.
Wait, if I know count is $\le 1$ from step 1 (Query 1 -> 1), I shouldn't run BS up to $N$.
I can set the upper bound of BS to the actual count found so far + something.
Or simply, I maintain a set of active prefixes.
Initially, active prefixes = $\emptyset$.
Wait, I need to populate the root.
Since I don't know the root is anything special (except containing all words), I can iterate 'a' through 'z'.
For each char $c$:
1. Query "$c$" 1.
   If $k=0$: Branch invalid. Skip.
   If $k=1$: Branch active. Upper bound for BS? We know count $\ge 1$.
     Can we query larger $K$?
     Maybe query "$c$" $N$? No, if count is 1, cost $N$ wasted.
     Instead, perform BS with $U=N$? Or $U=1$?
     If $k=1$, we know count is 1. Why? Because query 1 returned 1. It means count $\ge 1$.
     Does it mean count is 1? No. Could be 2.
     If I query 1 and get 1, I know count $\ge 1$.
     To check if count > 1, I need to query higher.
     But wait, if I query 2 and get 1, then count is 1.
     So if query 1 -> 1, I *can* just check query 2.
     Wait, query 1 -> 1. This means there is 1st word.
     If there are 2 words, query 1 returns 1st word.
     If there are 100 words, query 1 returns 1st word.
     So I don't know count.
     But I know count $\ge 1$.
     
     Optimization: The global constraint is $N$.
     If I process branches sequentially.
     Suppose for 'a', count is $c_a$.
     I can query 'a' with a guessed $K$.
     But how to guess?
     Maybe I can use the fact that total words is $N$.
     If I find 'a' has many words, 'b' must have fewer.
     But I don't know yet.
     
     However, note that I don't need to find the *exact* count for 'a' if I just need to retrieve the words.
     If I query 'a' with $K=N$ and it returns $k$ words (where $k \le N$).
     Then I know there are $k$ words.
     Cost is $N$ (or $\min(N, \text{current\_limit})$).
     Wait, I can pass $K=N$. Max cost $N$.
     If 'a' consumes 1000, 'b' costs 1 (check 0).
     Total cost $N$.
     If 'a' costs 1, 'b' costs $N$.
     In general, if I query each character with $K=N$.
     Worst case: all words start with different chars? No, $N \le 26$ then ok.
     If words distributed evenly, 'a' returns $N/26$.
     If I query 'a' $N$, I get $N/26$. Cost $N$.
     Do this for 26 chars. Total $26N$. Too big.
     So I MUST NOT query with large $K$ unless necessary.
     
     The strategy:
     Maintain a list of candidate prefixes. Initially all 'a'...'z' are candidates? No.
     Actually, since $N$ is small (1000), maybe I can just query 'a' 1, 'b' 1, ..., 'z' 1.
     Sum $K = 26$.
     This gives me a subset of words: 0, 1, or $>1$?
     Wait, query 1 returns 1 word or 0.
     If it returns 1 word, I know count $\ge 1$.
     If it returns 0, count = 0.
     Let's refine.
     If query $c$ 1 returns 1 word $w$.
     This is the smallest word starting with $c$.
     Let's store it.
     Now, I need to find if there are *more* words starting with $c$.
     And specifically, I need to find all of them.
     Also, I need to ensure $\sum K$ is small.
     Notice that if I query $c$ with $K=1$, I pay 1.
     If there is a 2nd word, I need to find it.
     The query $c$ $K$ gives words in order.
     If I want to find the next word after $w$, I can query $c$ with something?
     Actually, I can just keep appending characters to $w$ to get deeper? No, words are distinct.
     
     Alternative view:
     The words form a Trie.
     Root has children $T_1, T_2, \dots, T_{26}$.
     We want to traverse the Trie.
     We have a total budget 4000.
     Every edge traversal costs a query.
     Query on edge $u \to v$ (char $c$) with cost $K$.
     Ideally, I want to set $K$ equal to the size of subtree.
     How to find subtree size?
     Using the BS strategy described earlier:
     To find count of $c$-child of root.
     Current budget for $c$ is bounded by remaining $N$.
     But better: Just use BS with $N$?
     Wait, if I do BS for 'a' with $N$.
     Step 1: Query 'a' $N/2$.
     If $k < N/2$: I know count is $k$. Done. Cost $N/2$.
     If $k = N/2$: I know count $\ge N/2$.
     I continue with range $[N/2, N]$.
     Next: Query 'a' $(N/2+N)/2$.
     If $k < \dots$: Done. Cost intermediate.
     If $k = \dots$: Continue.
     
     This BS strategy works well *provided* that I can recover the words from the queries.
     In BS step $m$, if $k=m$, I have recovered $m$ words.
     But I need to continue to get the rest.
     Wait, if I find exact count $C$ via BS.
     I will have performed queries with parameters $K_1, K_2, \dots, K_m$.
     In these queries, I collected partial lists of words.
     Are the lists consistent?
     Query 1 with $K=500$: returns words $w_1 \dots w_{500}$.
     Query 2 with $K=750$: returns words $w_1 \dots w_{750}$ (assuming count $\ge 750$).
     So I can merge the results.
     Once I know exact count $C$ for 'a', I just need to query 'a' with $K=C$ (the final step of BS) to get all words.
     Wait, in BS, the last successful query returns exactly $C$ words?
     Yes, if I query $K$ and get $C$ words, and then query $C+1$ and get $C$ words (because $C < C+1$), then I know count is $C$.
     Wait, my BS logic was:
     Query $K$. If ret $< K$: exact count $ret$. Store result.
     If ret $== K$: count $\ge K$. Continue.
     To detect exact count, I eventually query some $K'$ where ret $< K'$.
     Or I hit $N$. If I query $N$ and get $N$, count could be $N$.
     But I know max words is $N$. So if I query $N$ and get $N$, I assume count is $N$?
     Not necessarily, unless I checked $N+1$ or something.
     But $N$ is global total. A prefix can have at most $N$.
     So if I query $N$ and get $N$, count is $N$.
     So yes, BS works.
     
     Algorithm refinement:
     1. Read $N$.
     2. Maintain `found_words` set/list.
     3. Maintain `remaining_N` = $N$.
     4. Iterate over characters $c \in \{'a' \dots 'z'\}$:
        If `remaining_N` == 0, break.
        
        First, check if $c$ is used.
        Query "$c$" 1. (Cost 1)
        Result $k_1$.
        If $k_1 == 0$: continue (no words with prefix $c$).
        If $k_1 == 1$:
            // Found at least 1 word.
            // We need to determine exact count.
            // Since we paid 1 to get the word, we have the word.
            // Remaining words starting with $c$ could be many.
            // But wait, if we just paid 1, we have the first word.
            // To find others, we should probably skip 1 and check further?
            // Or stick to BS logic.
            // BS range $[1, \text{remaining\_N}]$. (Or $[1, N]$?)
            // Better to use tighter bounds.
            // We know count $\ge 1$.
            // Let's set low = 1, high = min(N, remaining_N).
            // But wait, checking "used" took 1 unit of K.
            // We can integrate this into BS.
            
        Generalized approach:
        I want to find count for prefix $c$.
        Let range be $[L, R]$. Initially $[1, \min(N, \text{max\_remaining})]$.
        Wait, I don't know if $c$ is empty.
        So initially I query with $K=1$.
        If 0: done.
        If > 0: $L=1$.
        
        Wait, simply running BS from $N$ down might be expensive if many branches have few words.
        Example: 26 branches, each has 1 word.
        Branch 1: Query $K=1000$ (in BS). Cost 1000. Gets 1. Stops. Total 1000.
        This is terrible.
        Branch 1: Query $K=1$ (initial check). Cost 1. Gets 1.
        If we knew it stops there, good.
        But BS doesn't know.
        So we must lower the initial $R$.
        We can't predict $R$.
        
        Wait! I can use the fact that sum of counts is $N$.
        I can perform the BS for 'a' with a budget derived from remaining words.
        But I can't dynamically update budget inside query? I can choose $K$.
        So for 'a', I have budget roughly $N$.
        If I query 'a' $K=N/2$.
        If $k < N/2$: exact count $k$. I know count is small.
        If $k = N/2$: count $\ge N/2$. I spent $N/2$.
        Remaining budget is small.
        
        Let's calculate expected cost again with $N=1000$ and uniform dist.
        If I check 'a': Query $K=1$ (verify non-empty). Cost 1.
        Then BS range $[1, 1000]$.
        Mid $500$. Query 500. Get 1 (since real count is 40).
        Wait, if real count is 40.
        Query 500 returns 40.
        So cost 500 is incurred to find out count is 40.
        This is bad.
        I should query $K=40$ ideally.
        How to know to query 40? I don't.
        
        BUT!
        Look at the constraint: $\sum K \le 4000$.
        $N \le 1000$.
        The problem is likely solvable by a simple greedy approach or maintaining a Trie.
        What if I maintain a Trie of words I've found?
        Start with root.
        I need to find words.
        I can generate words?
        What if I simply iterate depth?
        No, lexicographical order matters.
        
        Revisit the BS idea but adapt the initial $K$.
        Actually, I don't need to find exact count to retrieve words.
        I just need to retrieve all words.
        For a prefix $S$, if I query $S$ $K$ and get $k$ words.
        If $k=0$, stop.
        If $k < K$, stop (I have all).
        If $k = K$, I have retrieved $K$ words, but there might be more.
        Crucially, I can now continue querying on the extensions of the words I found?
        No, the query is always on a prefix.
        If I have words starting with $S$, say $S+w_1$.
        If I want to see if there's another word $S+w_2$.
        I don't know $w_2$.
        But I can query $S$ again with $K+1$?
        Wait, if I query $S$ $K$ and get $K$ words.
        The response gives me the words.
        These words are the smallest $K$ words starting with $S$.
        So if I want to find more, I need to check beyond index $K$.
        This implies I need to know the *position* of the next word?
        Or I can just query $S$ with $K+1$.
        Wait, is there a way to jump?
        The responses are sorted.
        If I query "a" 1, get "apple".
        Query "a" 2, get "apple", "april".
        Query "a" 3, get "apple", "april", "art".
        I can observe that the $i$-th query returns a superset of the $(i-1)$-th query.
        Specifically, the new word appears at position $i$ (if count $\ge i$).
        So, to find the next word for a prefix, I just increment $K$.
        Wait!
        If I query "a" 100. And I get 50 words.
        This implies count is 50. I stop.
        If I get 100 words. I need to query "a" 101.
        But if I am at "a", and count is 100.
        I pay 100 to find the 1st..100th word.
        Then I pay 101 to find 1st..101th word.
        If count is 1000.
        Sequence of queries: 1, 2, 3 ... 1000.
        Sum of $K$: $1000 \times 1000 / 2 = 500,000$. Terrible.
        
        So we MUST binary search to find the cut-off $K$ (count).
        Wait, earlier I analyzed BS cost as $O(C)$.
        Let's re-verify.
        BS range $[0, 1000]$.
        Try 500. Cost 500. Get 500.
        Try 750. Cost 750. Get 500. Wait.
        If I query 750 and actual count is 500.
        Response is 500 words.
        $500 < 750$. So I know count is 500.
        I paid 750.
        If I had tried 250.
        Try 250. Cost 250. Get 250 words. (Count $\ge 250$).
        Then Try 375. Cost 375. Get 250 words. (Count $\ge 375$? No, 500 words returned? No, 250.)
        Wait, if count is 500.
        Query 250 -> returns 250. (Matches request). Count $\ge 250$.
        Query 375 -> returns 250. (Less than request). Exact count 250?
        No! If query 375 returns 250 words, it means count is 250?
        No! The response is "number of words returned".
        If query asks for 375, and returns 250 words.
        It implies there are only 250 words.
        Wait. Is it possible to return fewer than asked even if more exist?
        Problem: "If there are only k < K words, the service gives out only k words".
        "first K words... begin with prefix S".
        So if I ask for 375, and there are 500 words, I should get 375.
        I will NEVER get fewer than the true count IF true count $\ge K$.
        Wait. The condition is "If there are ONLY k < K words".
        So if count $\ge K$, response will be exactly $K$.
        If count $< K$, response will be exactly count.
        So if I query 375 and receive 250 words, then count must be 250.
        Wait, 250 < 375. So count < 375.
        But previously with 250 I got 250 words. (Consistent).
        So, if I query $K$, and response is $R$.
        If $R < K$, then Count $= R$.
        If $R == K$, then Count $\ge K$.
        Wait! If response is $R=K$, it could be that count is $K$ or count is $K+10$.
        It means we have found $K$ words, but might need more.
        
        So, in the case where actual count is 500.
        Query 250 -> Ret 250. (Condition $R=K$ met). Count $\ge 250$.
        Query 375 -> Ret 375. (Condition $R=K$ met). Count $\ge 375$.
        Query 500 -> Ret 500. (Condition $R=K$ met). Count $\ge 500$.
        Query 625 -> Ret 500. (Condition $R < K$ met). Count $= 500$.
        So I identified count 500.
        Queries made: 250, 375, 500, 625.
        Costs: $250 + 375 + 500 + 625 = 1750$.
        Total cost to find count 500 is 1750.
        This is $O(N)$. Specifically, roughly $N$.
        But wait, I can optimize the queries.
        I don't need to collect the words during the BS checks.
        I just need to know the count.
        Actually, I can just query 'a' with $K=500$ at the end to get all words.
        Wait, if I used queries 250, 375, 500, 625.
        Query 500 gave me 500 words.
        Query 250 gave me 250 words (subset).
        So I have all words from Query 500.
        So I don't strictly need to query 625 if I'm sure count is $\ge 500$ and $\ge 625$ is false.
        Wait, if I query 625 and get 500, I confirm count is 500.
        So cost is $\sum K_i$.
        For count $C$, if I use BS, the sum of costs is roughly $2C$ to $3C$.
        With $N=1000$ uniform distribution (26 branches of $\approx 38$ words).
        Cost per branch $\approx 2 \times 38 = 76$.
        Total $26 \times 76 \approx 2000$.
        This seems safe!
        
        Let's double check the uniform case cost.
        Assume count $C=38$.
        BS:
        Mid 500 (Too big). Query 500. Get 38. Stop. Cost 500.
        Ah! This is the killer.
        If I initialize BS range $[1, N]$ and $C$ is small.
        I query $N/2$.
        Response is $C < N/2$.
        So I stop immediately. Cost $N/2$.
        For 26 branches of size $C=38$, cost per branch is 500.
        Total cost $26 \times 500 = 13000$.
        FAIL.
        
        So I need a better initial guess or strategy.
        I must avoid querying large $K$ when $C$ is small.
        How?
        Observe that sum of $C$ is $N$.
        If I query $c$ with $K=1$.
        If $C < 1$ (i.e., 0), I know. Cost 1.
        If $C \ge 1$, I get 1 word.
        If $C=1$, I know $C \ge 1$.
        Next guess?
        I can't just jump to $N/2$.
        But I know that $\sum C = N$.
        Maybe I can assign budgets?
        No.
        
        Wait. The problem allows us to find words.
        Maybe I don't need to determine exact count for each branch separately.
        What if I treat the whole set of words as a stream?
        I can query the root with $K=1$. Get first word $w_1$.
        Then I know $w_1$ exists.
        Can I use $w_1$ to guide queries?
        $w_1$ starts with some char $c$.
        I can recurse into that branch?
        If I find a word, I can try to extend it to find siblings?
        But to find siblings, I need to know the structure.
        
        Wait. What if I query a string of length 1 with $K=1$.
        If I get something, I have the first word.
        What if I query "a" 1, "b" 1, "c" 1, ...
        This costs 26.
        If I find words, I know these branches are non-empty.
        I still don't know counts.
        But maybe I can iterate on the words found so far?
        Suppose I found "apple".
        This word ends with 'e'.
        Maybe "app" is a prefix?
        "Apple" is distinct.
        
        Let's go back to the BS cost issue.
        The issue is assuming range $[1, N]$.
        If I have multiple branches, maybe I can share information?
        No, words in 'a' vs 'b' are disjoint.
        The only shared resource is the budget 4000.
        
        Key Idea:
        The total number of words is $N$.
        So the number of branches with count $\ge 1$ is at most $N$.
        Actually, number of words is $N$.
        Number of non-empty branches at depth 1 is $\le 26$.
        At depth 2, number of non-empty branches $\le 26 \times 26$?
        But since total nodes (words) is small? No, words are distinct.
        Wait, max length 10.
        Maybe I can just query prefixes of length 1, then length 2...
        No, that's slow.
        
        Correct Logic for Minimizing $K$:
        If I have to find $C$ items.
        The cost of query $K$ is $K$.
        If I query $K$ and get $C < K$, cost is $K$.
        If I query $K$ and get $K$, cost is $K$.
        Basically, every query returns "value $V \le K$".
        To find $C$, I want to select $K$'s such that I narrow down to $C$ efficiently.
        If $C$ is unknown, best first guess?
        Maybe 1?
        Query 1.
        If 0 words -> done. Cost 1.
        If 1 word -> count $\ge 1$.
        Query 2.
        If 1 word -> count 1. Cost $1+2=3$.
        If 2 words -> count $\ge 2$.
        This sequential strategy is bad if $C$ is large.
        
        But notice that $C$ cannot exceed $N$.
        Also, if $C$ is small, we waste little? No, we waste large if we pick large $K$.
        If $C$ is large, we pick $N/2$.
        So the worst case is having many small counts.
        E.g. 26 branches with count 1.
        Sequential 1->2->3... is good for small counts.
        BS is good for large counts.
        Maybe I can combine?
        Query $K$ where $K$ is roughly related to the current estimate.
        Or better:
        Since we only care about retrieving the words.
        If I query "a" 1 and get word $w$.
        If I know there are no other words starting with "a", I don't need to query further.
        If I don't know, I must query.
        
        What if I query each char with $K=1$ first?
        Total cost 26.
        This eliminates 0-branches efficiently.
        Suppose 5 branches are non-empty.
        For each non-empty branch, count $C_i \ge 1$.
        Sum $C_i = N$.
        For a specific branch with count $C_i$.
        We know $C_i \ge 1$.
        We can query $K$ such that it helps.
        Maybe I can binary search but restrict the upper bound to be small?
        No, $C_i$ could be $N$.
        
        Wait! The problem says $N \le 1000$.
        Time limit 1s.
        Maybe I can just implement the simple greedy approach:
        Iterate $c$ from 'a' to 'z'.
        If current query count is high?
        No.
        
        Let's look at the constraint: $K \le N$ for queries?
        No, $K$ is integer up to $N$.
        
        Actually, look at the sample.
        $N=4$.
        Words: aaa, aba, cxy, cz.
        My manual trace:
        Query 'a' 1 -> aaa. (Found 1).
        Query 'a' 4 -> aaa, aba, bba (not exists), ... returns aaa, aba.
        Wait, if I query 'a' 1, I get 'aaa'.
        Then query 'a' 2?
        If I query 'a' 2, I get 'aaa', 'aba'.
        Then query 'a' 1 -> 'aaa'.
        Basically, if I want to minimize $\sum K$, and I know I need to get 'aaa', 'aba'.
        I should query 'a' 2. Cost 2.
        Why not 'a' 1 then 'a' 1? Cost $1+1=2$. Same.
        But if count was 4. 'a' 2 costs 2. 'a' 4 costs 4.
        If I did 'a' 2, get 2 words. I need more. Query 'a' 3?
        Wait, if I query 'a' 2 and get 2 words.
        I don't know if count is 2 or 4.
        If I query 'a' 3. If get 3. Still don't know if 4.
        If I query 'a' 4. If get 4. Count is 4.
        So sequential doubling?
        $1, 2, 4, 8, 16 \dots$
        Query 1 -> 1 word. Count $\ge 1$.
        Query 2 -> 2 words. Count $\ge 2$.
        Query 4 -> 4 words. Count $\ge 4$.
        Query 8 -> 4 words. Count 4.
        Costs: $1 + 2 + 4 + 8 = 15$.
        Retrieval cost is determined by the last query.
        If count is $C$, I find it with roughly $2C$ cost (doubling strategy).
        $1+2+4+8 = 15$. Last query was 8 (which $\ge 4$).
        Wait, if I query 4 and get 4, I know count $\ge 4$.
        Then query 8. If get 4, count is 4.
        Cost 15.
        Doubling is $2C$ if $C$ is power of 2? No.
        For $C=4$, steps: 1, 2, 4, 8.
        Sum: 15. $2C$ would be 8.
        Exponential search is slightly worse than linear but avoids $N/2$ penalty for small $C$.
        Wait, if $C=1000$.
        Steps: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024.
        Sum $\approx 2048$.
        This is comparable to $2C$.
        However, if $C=38$.
        Steps: 1, 2, 4, 8, 16, 32, 64.
        Sum: 1+2+4+8+16+32+64 = 127.
        With $C=38$, this is much worse than ideal $2 \times 38 = 76$.
        But it avoids the $N/2 = 500$ penalty of BS on small $C$.
        Also, we can refine: instead of fixed powers of 2, we can pick $K$ based on remaining global budget?
        No, budget is soft constraint (score).
        
        Let's reconsider the uniform distribution case with BS.
        Cost $\approx 13000$. Bad.
        Doubling: Cost $\approx 26 \times 127 \approx 3300$.
        Good! 3300 < 4000.
        So Doubling Strategy seems viable.
        Algorithm:
        For each char $c$:
            Find count $C_c$.
            Use exponential search (Doubling).
            Start $K=1$.
            While True:
                Query "$c$" $K$.
                Get $k$ words.
                If $k < K$:
                    Count is $k$. Break.
                If $k == K$:
                    Count $\ge K$.
                    $K *= 2$.
            
            After loop, I have retrieved the words corresponding to the largest power-of-2 query?
            No. In the loop, I accumulate words.
            If I query $K=1$, get 1 word.
            Then query $K=2$, get 2 words.
            If $k=K$, I overwrite my previous retrieval?
            No, I have a vector of found words.
            If query 2 returns 2 words. I keep them.
            If query 4 returns 4 words. I keep them.
            Eventually query $K$ returns $C$ words.
            Wait, does doubling guarantee efficiency?
            If count is $C$, I do queries $1, 2, 4, \dots, 2^m \ge C$.
            Sum $K \approx 2 \times 2^m \approx 2 \times C$. (Actually $2 \times 2^m$ is rough).
            Sum of geometric series $1+2+4+\dots+2^m \approx 2^{m+1}$.
            If $C \approx 2^m$, sum $\approx 2C$.
            If $C$ is not power of 2, say 38.
            Powers: 1, 2, 4, 8, 16, 32, 64.
            Sum = 127. $2 \times 38 = 76$. 127 is okay.
            Wait, I can stop earlier.
            When I query $K$, and get $k$ words.
            If $k < K$, exact count is $k$.
            If $k == K$, I continue.
            The last query that returns $k < K$ identifies the count.
            But wait, if I query $K=64$ and get 38 words.
            I know count is 38.
            But I already received 38 words in that query.
            Wait, I didn't receive 32 words in previous query (got 32 words).
            Actually, the query 64 returns ALL words starting with 'a' that are $\le$ rank 64.
            Since there are only 38, it returns all 38.
            The previous query 32 returned 32 words.
            So I have redundancy.
            Do I need to query 64?
            If I query 32 and get 32. I know count $\ge 32$.
            Then query 64. Get 38.
            This tells me count is 38.
            Wait, if I query 32 and get 32 words, and I know that max count is bounded by remaining N?
            No.
            So with Doubling, I do redundant work (getting subsets).
            The words are stored in the vector.
            So I need to store all answers.
            Wait, memory limit 512MB. Storing 1000 words is trivial.
            But flushing buffer?
            "print the newline character after each query and flush".
            
            Is Doubling better than Binary Search?
            BS for $C=38$ (range 1 to 1000) -> query 500 -> gets 38 -> stop. Cost 500.
            Doubling for $C=38$ -> 1, 2, 4, 8, 16, 32, 64 -> get 38 -> stop. Cost 127.
            So Doubling is MUCH better for small counts.
            What about large count $C=1000$?
            BS: 1000 (check 500) -> 1500 (too big, check 750) -> ... sum $\approx 1000 \times \text{const}$.
            Actually BS with range $[1, N]$.
            Check 500 -> 500 words.
            Check 750 -> 500 words. (Wait, if count is 500, query 750 returns 500).
            So if I query 750 and get 500, I stop.
            Cost $500 + 750 = 1250$.
            Doubling: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024.
            Query 512 -> gets 512 words.
            Query 1024 -> gets 500 words.
            Wait, if I query 1024 and get 500, I know count is 500.
            But I spent 1024.
            Sum is huge.
            So BS is better for large counts, Doubling is better for small counts.
            Since total $N=1000$, counts are relatively small.
            Even in worst case $N=1000$ in one branch, Doubling costs $\approx 2000$.
            If I use BS on range $[1, N]$ for large $C$, cost is $\approx N + N/2 + N/4 \dots$ ?
            Wait, BS query parameters are midpoints.
            Midpoints: $N/2, 3N/4, 7N/8 \dots$.
            Sum $\approx N \times (1/2 + 3/4 + \dots)$.
            Actually, the query cost is the parameter $K$.
            So sum $\approx 2N$.
            So for $C=1000$, both methods cost $\approx 2000$.
            Doubling is safer for small counts (cost $\approx 1.5 C$).
            BS is safer for large counts (cost $\approx 2 C$).
            However, BS on range $[1, N]$ has overhead.
            Wait, if I use BS with dynamic range?
            Initial range $[1, N]$.
            If I query $K=1$ first?
            If count is 38.
            Query 1 -> 1. Cost 1.
            New range $[2, 1000]$.
            Mid 501. Query 501 -> 38. Cost 501. Stop.
            Total 502.
            Still bad.
            To make BS effective, I should narrow the search range initially.
            But I have no info.
            Except: $\sum C_i = N$.
            So average $C_i$ is small.
            For most branches, $C_i$ is small.
            Only a few branches can be large.
            The total cost over all branches is $\sum \text{cost}(C_i)$.
            If I use Doubling: Cost $\approx \sum 2 C_i = 2N$.
            If I use BS with $[1, N]$: Cost $\approx \sum C_i \times 2 = 2N$?
            Wait, BS cost for $C$ with range $[1, N]$:
            If $C$ is small, cost is dominated by first query $N/2$.
            Wait, BS assumes $C$ is somewhere in $[1, N]$.
            If $C < N/2$, query $N/2$ returns $C$. Cost $N/2$.
            So cost is $\min(C, N/2) + \dots$.
            If $C$ small, cost $\approx N/2$.
            If I have 26 branches with $C=38$, cost $26 \times 500 = 13000$.
            So BS is definitely bad if I assume range $[1, N]$.
            
            BUT, I don't need to query with $N$.
            If I know that $\sum C_i = N$, then $C_i \le N$.
            But more importantly, $C_i \le N - \sum_{j < i} C_j$.
            I can maintain a variable `remaining_words`.
            Before processing branch $c$, I set `upper_bound = remaining_words`.
            For branch $c$:
               If `remaining_words` == 0, stop.
               Perform BS/Doubling with range $[1, \text{remaining\_words}]$.
               
            Let's trace with Doubling and `remaining_words`.
            Uniform case $C=38$, $N=1000$.
            Branch 1: $rem = 1000$.
               Doubling: 1, 2, 4, 8, 16, 32, 64.
               Cost 127.
               Found 38 words.
               Update $rem = 1000 - 38 = 962$.
            Branch 2: $rem = 962$.
               Doubling: 1, 2, ..., 64.
               Cost 127.
               Found 38 words.
               $rem = 924$.
            Total cost $26 \times 127 \approx 3300$.
            This works!
            
            What if one branch has $C=1000$?
            Branch 1: $rem=1000$.
               Doubling: 1, 2, ..., 1024.
               Wait, query 1024 will return 1000.
               Cost $\approx 2048$.
               $rem = 0$.
            Total cost $\approx 2048$. Fits 4000.
            
            What if distribution is skewed but multiple branches?
            $N=1000$.
            Suppose 'a' has 500, 'b' has 500.
            Branch 'a': $rem=1000$.
               Doubling: 1, 2, ..., 512 (returns 500).
               Wait, query 512 returns 500.
               Next query 1024 returns 500.
               So Doubling needs to go up to $2 \times C$ roughly.
               Sum costs $\approx 2 \times 500 = 1000$.
               Wait, sequence 1, 2, 4, ..., 256, 512.
               Sum $2^{10} - 1 \approx 1023$.
               If I query 512, I get 500 words. I know count $\le 500$.
               But I don't know count is 500. It could be 500.
               I must check $K > 500$.
               So I query 1024 (or 512+1?).
               Actually, doubling logic:
               While True:
                  Query $K$.
                  Get $k$.
                  If $k < K$: Count $= k$. Break.
                  If $k == K$: Count $\ge K$. $K *= 2$.
               So if count is 500.
               Query 256 -> 256.
               Query 512 -> 500. (Here $500 < 512$. Stop).
               Cost: $1+2+\dots+256+512 = 1023$.
               So cost is roughly $2C$.
               $2 \times 500 = 1000$.
               Branch 'b': $rem=500$.
               Query 256 -> 256.
               Query 512 -> 500.
               Cost 1023.
               Total $1000 + 1000 = 2000$.
               Fits 4000.
            
            What if 'a' has 999, 'b' has 1?
            Branch 'a': Cost $\approx 2000$.
            Branch 'b': Cost $\approx 2$.
            Total 2002.
            
            Seems Doubling + Dynamic Bound works very well.
            Complexity: $O(N)$ for sum of costs.
            With $N=1000$, max cost $\approx 2N \times (\text{factors})$.
            The constant factor for geometric series is 2.
            So max cost $2N = 2000$.
            Plus initialization overhead (checking if empty).
            Checking empty is query 1.
            So total cost $\approx 2N + 26 \times 1 = 2026$.
            Well within 4000.
            
            Wait, one catch.
            If I query $K$ and get $k < K$, I know count is $k$.
            The query returned $k$ words.
            So I don't need to query again to get them.
            So cost is just $\sum K$.
            Wait, if I query 512 and get 500 words.
            I know count is 500.
            But I only paid 512.
            Did I retrieve 500 words? Yes.
            Do I need to retrieve them again? No, they are in memory.
            The cost is accumulated $K$.
            But wait.
            If I query 256 -> get 256 words. Cost 256.
            Then query 512 -> get 500 words. Cost 512.
            I have 500 words.
            Do I duplicate words?
            The problem: "sum of K for all queries ... is as less as possible."
            Duplicate words storage doesn't matter for correctness or scoring.
            Only $\sum K$ matters.
            So this logic holds.
            
            One detail:
            "sum of K for all queries of the first type".
            Query 'a' 1 (cost 1).
            If returns 1 word. (Word "apple").
            I continue doubling.
            Query 'a' 2 (cost 2).
            If returns 2 words ("apple", "apply").
            Double.
            Query 'a' 4 (cost 4).
            If returns 4 words.
            Double.
            Query 'a' 8 ...
            Eventually query $K_{end}$ returns $C$ words.
            Note: In query $K_{end}$, if I get $k=C < K_{end}$, I stop.
            If I get $C$ words, I have found them.
            The words returned in subsequent queries are supersets of previous ones.
            I just keep adding to `found_words`? Or replace?
            If I keep all queries results?
            Actually, I only need to add the NEW words found.
            Wait, query $K$ returns first $K$ words.
            If I already have first $prev\_K$ words.
            The new words are from index $prev\_K+1$ to $k$.
            So I just append to the list.
            The list grows.
            Finally I output all unique words.
            Wait, words must be distinct.
            If I query $K$ and get words, they are distinct.
            So merging is trivial: append differences.
            But simpler: just maintain a Trie or hashset of found words.
            Since $N$ is small, storing 1000 strings is fine.
            
            Final check on constraints.
            Word length 1 to 10.
            Distinct.
            Time limit 1s.
            Memory 512MB.
            Sum K $\le 4000$.
            My solution uses Doubling with dynamic bound $N$.
            Expected cost $\le 2N + 26 \approx 2026$.
            Max cost?
            Worst case scenario?
            Maybe my geometric sum approximation is loose.
            Sum of $1, 2, 4, \dots, 2^m$ where $2^m \ge C$.
            Sum is $2^{m+1}-1 < 2 \times 2^m$.
            If $C=500$, $2^m=512$. Sum $< 1024$.
            Cost $\approx 1024$.
            If I have multiple branches.
            Is it possible to violate 4000?
            Suppose $N=1000$.
            Branches split: 'a' has 500, 'b' has 500.
            Total cost $\approx 1024 + 1024 = 2048$.
            Suppose branches 'a' 100, 'b' 100, ... 10 branches.
            Cost per branch: $C=100$. $2^m=128$. Sum $\approx 256$.
            Total $10 \times 256 = 2560$.
            Suppose 26 branches of 38 words.
            Cost per branch: 38 words. $2^m=64$. Sum 127.
            Total $26 \times 127 = 3302$.
            Looks very safe.
            Even if 26 branches of 40 words (total 1040 > 1000, so 1040 words is impossible, max 1000).
            If $N=1000$, average 38.
            Worst case for cost is when counts are somewhat uniform?
            Let's analyze $f(C) = 2^{m+1}-1 \approx 2C$.
            We want to maximize $\sum f(C_i)$ subject to $\sum C_i = N$.
            $f(C) \approx 2C$.
            This is linear. So sum is $\approx 2N$.
            However, there is a ceiling effect.
            Wait, $f(C)$ increases superlinearly for small $C$?
            No, $f(C) \approx 2C$.
            Wait, if $C=1$, $2^m=1$, sum 1. $2C=2$.
            If $C=2$, $2^m=2$, sum 3. $2C=4$.
            If $C=3$, $2^m=4$, sum 7. $2C=6$.
            It seems roughly $2C$.
            Wait, if $C$ is large, say 1000. $2^m=1024$. Sum 2047.
            $2C = 2000$.
            So it's roughly $2C$.
            But notice, if $C$ is small, $f(C) \approx 2C$.
            If $C$ is large, $f(C) \approx 2C$.
            Is there any region where it deviates significantly?
            Actually, it's slightly less efficient for powers of 2 minus epsilon.
            But generally it is bounded by $2C$.
            So total cost $\le 2N = 2000$.
            Wait, plus 26 for the first queries?
            Wait, if $C=1$, Doubling queries:
            Start $K=1$. Query 1.
            Get 1. (Count $\ge 1$).
            $K=2$. Query 2.
            Get 1. (Count $= 1$).
            Cost $1+2=3$.
            My manual trace above said 127 for $C=38$.
            Wait, if $C=1$.
            Query 1 -> 1.
            Next query 2 -> 1.
            Wait, if count is 1.
            Query 1 -> 1.
            Condition $k=K$ met (1==1). Continue.
            $K=2$.
            Query 2 -> 1.
            Condition $k < K$ met (1 < 2). Stop.
            Total cost 3.
            For $C=38$.
            Query 32 -> 32. Continue. $K=64$.
            Query 64 -> 38. Stop.
            Sum $\approx 127$.
            Wait, my formula $\sum 2^m \approx 2C$ was wrong?
            The last term is roughly $2C$.
            The series is $1, 2, 4, \dots, 2^{m-1}, 2^m$.
            Sum $2^m + 2^m = 2 \times 2^m$.
            If $2^m \approx C$, then $2C$.
            If $C=1$. Series 1, 2. Sum 3.
            If $C=2$. Series 1, 2. Next 4. Stop?
            Query 1 -> 1 (cont).
            Query 2 -> 2 (cont).
            Query 4 -> 2 (stop).
            Sum 1+2+4 = 7.
            $2C = 4$.
            It seems Doubling is slightly inefficient by a factor.
            But for $N=1000$, $2N=2000$. Factor might be around 1.5-2.
            Still under 4000.
            Wait, if I use $K$ starting from 1, I am guaranteed to find count.
            Can I start with $K=1$ and handle 0?
            Yes, query $c$ 1. If 0 words, skip.
            This handles the 0 case efficiently.
            So for non-zero cases, I use doubling.
            This seems robust.

            Implementation details:
            Maintain `vector<string> words`.
            Loop `for (char c = 'a'; c <= 'z'; ++c)`:
               if `words.size()` >= N, break? No, I need to be sure.
               But `words.size()` can increase.
               Actually, I can track `count_of_words_found` in total.
               If `total_count` reaches $N$, I can stop early?
               Yes, if I found $N$ words.
               Wait, is it possible to miss words?
               If I process 'a' and find 100 words.
               Process 'b' and find 100 words.
               If I have found 1000 words total, I can stop.
               This might save some queries for later characters.
               Also, maintain `remaining_budget`? No.
               Use `min_rem_N` for upper bound.
               Actually, `remaining_N` should be initialized to `N` at start of test case.
               For each char:
                 if `remaining_N` == 0: break.
                 Query $c$ 1.
                 If returns 0 words: continue.
                 Update `upper_bound = remaining_N`.
                 Perform Doubling starting $K=1$.
                 Wait, start $K=1$.
                 If returns 1 word, update $rem -= 1$.
                 Continue.
                 Be careful: updating `rem` dynamically helps.
                 Also, if `words.size() >= N`, break.
            
            Flush output buffer.
            
            One tricky point:
            "If the dictionary contains only k < K words, the answer ... contains k words."
            "answer ... separated by spaces."
            "query ... print newline ... flush".
            "The response ... k words Si ...".
            I need to read $k$ and then the words.
            
            Reading the response:
            Format: `k w1 w2 ... wk`?
            Example output:
            `query a 1`
            `standard output`
            `1 aaa`
            `query a 4`
            `2 aaa aba`
            `query c 1`
            `1 czzzz`
            `query cy 1`
            `0`
            Wait, looking at the example text provided:
            Example shows:
            Input:
            1 (T)
            4 (N)
            ...
            Output:
            1 aaa
            2 aaa aba
            1 czzzz
            0
            Wait, the example shows the JURY OUTPUT.
            So my program prints queries. Jury outputs lines.
            So I print "query a 1\n".
            I read line. Parse.
            First token is k. Then k strings.
            I need to store them in a Set (to ensure uniqueness, though distinctness is guaranteed).
            Wait, "words in Lewis's code are composed of pairwise distinct words".
            So duplicates across queries won't happen if queries are consistent (prefix property).
            But overlapping queries: query "a" returns "apple", "apricot".
            Query "ap" returns "apple", "apricot".
            Wait, "apple" starts with "ap".
            If I query "a" 2 and get "apple", "apricot".
            Then I query "ap" 2? No, I won't query "ap" unless I know it exists.
            I'll construct the Trie or set of candidates to query.
            But my plan only iterates 'a'...'z' once.
            Is that enough?
            "All words in Lewis's code are pairwise distinct."
            If I find all words starting with 'a', I have all words starting with 'a'.
            Is it possible I need to query deeper?
            Yes, but if I query "a" with sufficient $K$, I get ALL words starting with "a".
            Wait, if I query "a" with $K$ large enough (larger than count), I get all of them.
            My doubling strategy finds $C_a$, then I query $K=C_a$?
            No, I don't query $C_a$ explicitly.
            I query $K$ values.
            The LAST successful query was $K_{last}$ where response $k_{last} < K_{last}$.
            This $K_{last}$ is $\ge C_a$.
            The response contained $k_{last} = C_a$ words.
            So I have ALL words starting with "a".
            I don't need to dig deeper because I have all of them.
            So iterating 'a'...'z' is sufficient.
            I do NOT need to query sub-prefixes like "ab".
            Because query "a" covers everything starting with "a".
            Wait, is that correct?
            Yes, query S K gets ALL words starting with prefix S.
            So query "a" gives all words starting with "a".
            Since I want to collect all $N$ words, and every word starts with some char, collecting all words starting with each char (by determining count and querying enough) collects all words.
            Perfect.
            
            Wait, "query S K ... lexicographically minimal words starting with prefix S".
            If I query "a" with $K=C_a$, I get $C_a$ words.
            These words are sorted.
            Are they ALL words starting with "a"?
            Yes, because count is $C_a$.
            So I don't need to query "ab", "ac", etc.
            The recursion stops at depth 1.
            This makes the solution extremely simple.
            Complexity $O(N \times \log N)$ or $O(N)$.
            Cost $\approx 2N$.
            
            One small thing:
            I need to output `answer S1 ... SN`.
            After I finish all 'a'...'z', I have a list of words.
            Sort them?
            The problem asks to output them in arbitrary order.
            So no sort needed.
            
            Corner case: $N$ small.
            $N=1$.
            Words: "a".
            Query 'a' 1 -> 1 "a".
            Others 0.
            Cost 1+26 (others). Total 27.
            
            Double check memory.
            Store strings. Length 10. Max 1000.
            Vector<string>. Size 5KB. OK.
            Time limit 1s.
            Operations per test case ~500 queries.
            $T=5$.
            Total time negligible.
            
            Code Structure:
            ```cpp
            #include <iostream>
            #include <string>
            #include <vector>
            #include <algorithm>
            #include <sstream>
            #include <set>

            using namespace std;

            void solve() {
                int N;
                cin >> N;
                
                vector<string> words;
                int remaining = N;
                
                for (int i = 0; i < 26; ++i) {
                    if (words.size() >= N) break; 
                    
                    char c = 'a' + i;
                    string s = "";
                    s += c;
                    
                    // Check existence
                    cout << "query " << s << " 1" << endl;
                    string res;
                    getline(cin, res);
                    
                    // parse res
                    // Format: k w1 w2 ... wk
                    stringstream ss(res);
                    int k;
                    if (!(ss >> k)) k = 0; // Should not happen?
                    
                    for(int j=0; j<k; ++j){
                        string w;
                        ss >> w;
                        words.push_back(w);
                    }
                    
                    // Update remaining
                    remaining -= k;
                    
                    if (remaining > 0 && k > 0) {
                        // Need to find exact count if k < count
                        // But wait, k is number of words returned.
                        // If k < requested(1), then count is 0? No.
                        // Requested was 1. If k < 1 (i.e. 0), then count is 0.
                        // If k=1, count >= 1.
                        // Wait, if k=1, I need to check if count > 1.
                        
                        // Doubling strategy
                        // Start with K = 2 (or k+1)
                        long long K = 2;
                        // Wait, if I queried 1 and got 1.
                        // Then I need to check higher.
                        // I need to know current K to continue?
                        // The last query I made was 1.
                        // I have the words.
                        // Now I want to know how many more.
                        // Let's just run Doubling starting from K=1?
                        // No, I already paid for K=1.
                        // I can resume from K=2.
                        // Wait, if I resume from K=2, I need to clear buffer?
                        // No, the interaction is stateless regarding my variables, but persistent regarding DB.
                        // I just send a query.
                        
                        // Wait, doubling logic:
                        // If I query 2 and get 2 words.
                        // I know count >= 2.
                        // If I query 2 and get 1 word.
                        // I know count = 1.
                        
                        // But I need to ensure I collect all words.
                        // If I query 1 and get 1 word.
                        // Then query 2 and get 2 words.
                        // I have 2 words.
                        // Then query 4 and get 3 words. (Wait, count=3).
                        // Then query 8 and get 3 words. (Count=3).
                        // So the sequence of queries determines the cutoff.
                        
                        // Let's restart K tracking properly.
                        // I have already queried "s" 1. Got k=1.
                        // Current count known >= 1.
                        // If k was 0, we stopped.
                        // If k was 1, we continue.
                        
                        // Wait, in my logic above:
                        // Query "s" 1.
                        // If returns k words.
                        // If k=0, done with s.
                        // If k > 0, we continue.
                        // But wait, if k < 1? Impossible (min 0).
                        // If k == 1 (since query 1).
                        // Then we need to check if count > 1.
                        // So I perform query "s" 2.
                        // If get 2 words -> count >= 2.
                        // If get 1 word -> count 1.
                        
                        // So I can just start K=2.
                        // BUT what if count is 1000?
                        // Doubling: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024.
                        // Sum = 2048.
                        // If I started at 1 (paid 1).
                        // Total sum = 2048 + 1 = 2049.
                        // Wait, my previous estimation used 1, 2, 4... starting at 1.
                        // If I query 1 first, I already have 1 word.
                        // Then I query 2.
                        // If I query 2 and get 1 word. (Count 1).
                        // Cost 2.
                        // Total 3.
                        // If I query 2 and get 2 words. (Count >= 2).
                        // Query 4.
                        // If I query 4 and get 2 words. (Count 2).
                        // Cost 4.
                        // Total 2+4=6. (Plus initial 1 = 7).
                        // This matches the cost calculation.
                        // However, notice that if I query 2 and get 2 words.
                        // I already have word from query 1.
                        // The response of query 2 includes word from query 1.
                        // So I don't lose words.
                        
                        // So yes, just use K=2 start.
                        // Wait, if I use K=1, get 1.
                        // Then K=2. Get 2.
                        // Then K=4. Get 4.
                        // Then K=8. Get 4.
                        // Wait, if count is 4.
                        // Query 8 returns 4.
                        // Then I know count is 4.
                        // The words are obtained in the last query.
                        // Wait, query 8 returns 4 words.
                        // Query 4 returned 4 words.
                        // Query 2 returned 2 words.
                        // So I have the words.
                        // But I paid 8 for the last check.
                        // Actually, I paid 1+2+4+8 = 15.
                        // If I stopped at 4 (checked 4 and it returned 4 words), I need to know if count is 4 or more.
                        // So I must check > 4.
                        // So next query 8 is required.
                        // So cost is sum(1, 2, 4, ..., K_stop).
                        // Where K_stop is the first query that returns < K.
                        // Or max reached if count is large.
                        // If count is C, K_stop $\ge C$.
                        // So sum $\approx 2C$.
                        // Plus initial 1 query?
                        // If I include the initial 1 in the doubling series (1, 2, 4...).
                        // Then it covers it.
                        
                        // So:
                        // K = 1.
                        // Loop:
                        //    Query s K.
                        //    Read k words.
                        //    If k < K: Break. (Found exact count k).
                        //    If k == K: K *= 2. (Continuing to check).
                        // But wait.
                        // If k < K, I know count is k.
                        // The response has k words.
                        // But in my sequence, I already retrieved some words in previous queries.
                        // I just need to make sure I have ALL words.
                        // The last query that returns < K contains ALL words.
                        // So I can ignore previous duplicates or just use the last result.
                        // The question asks to minimize sum K.
                        // Duplicates in storage don't affect K cost.
                        // So I can just keep appending or replacing.
                        
                        // Is there any risk of infinite loop?
                        // Max count is remaining_N.
                        // If K > remaining_N, I can stop?
                        // Yes, if K > remaining_N, then response must be < K.
                        // So count is k (<= remaining_N).
                        // So if K > remaining_N, I can force K = remaining_N? No.
                        // If I query K > remaining_N, I will get k words.
                        // Since k <= remaining_N < K, condition k < K holds.
                        // So I can cap K at remaining_N + 1?
                        // Or just rely on the loop termination naturally?
                        // If K doubles, it eventually exceeds N.
                        // Then next query will return k < K (since k <= N < K).
                        // So it terminates.
                        
                        // Wait, I must be careful with updating 'remaining' and N.
                        // If I query s K and get k words.
                        // I update remaining -= k.
                        // But wait.
                        // If I query K=4 and get 4 words.
                        // I update remaining -= 4.
                        // Then K becomes 8.
                        // Next query K=8.
                        // Suppose count was actually 5.
                        // Query 4 returns 4 words.
                        // remaining was reduced by 4.
                        // Wait, if count is 5, remaining reduces by 5 eventually.
                        // But in this step, I subtract 4.
                        // Then in step K=8, I subtract 5 (total).
                        // But wait, I already subtracted 4.
                        // So I subtract 5 again?
                        // No!
                        // I must track `accumulated_words`.
                        // If query 4 returned 4 words.
                        // I record 4 words.
                        // Then query 8.
                        // It returns 5 words (assuming count 5).
                        // Wait, if count is 5.
                        // Query 4 returns 4 words.
                        // Query 8 returns 5 words.
                        // The response of query 8 includes the 4 words from query 4.
                        // So the NEW words are 1 word.
                        // So I should not subtract 5 from remaining. I should subtract 1 (difference).
                        // Or better: subtract count found in LAST successful query that returns count < K?
                        // No.
                        // I need to know the ACTUAL count of words for this branch.
                        // If query 8 returns 5 words.
                        // I know count is 5.
                        // So I should reduce `remaining` by 5?
                        // But I already reduced by 4 in query 4.
                        // Wait, if query 4 returned 4, I thought count $\ge 4$.
                        // I didn't know it was exactly 4.
                        // So reducing `remaining` by 4 was premature.
                        // Correct logic:
                        // Only reduce `remaining` when you FIND the exact count.
                        // Once loop breaks (k < K), you know count is k.
                        // You can then update `remaining` -= k.
                        // But wait.
                        // If I did:
                        // Q1(K=1) -> 1 word. (Remaining unknown $\ge 1$).
                        // Q2(K=2) -> 2 words. (Remaining unknown $\ge 2$).
                        // Q4(K=4) -> 2 words. (Count 2).
                        // Then I know count is 2.
                        // Reduce remaining by 2.
                        // So I must NOT reduce remaining during the doubling phase.
                        // Wait, this changes the bound logic.
                        // If I don't update remaining, `remaining` stays N.
                        // Then `upper_bound` is always N.
                        // Then K keeps doubling up to N.
                        // Is this acceptable?
                        // If `remaining` is N, `upper_bound` is N.
                        // Doubling on N: 1, 2, ..., N.
                        // Cost $\approx 2N$.
                        // If `remaining` decreases, I can optimize.
                        // But I can't decrease until I know count.
                        // So I might pay full price $2N$ for each branch even if few words?
                        // Yes!
                        // This contradicts my earlier "safe" assumption.
                        // My "safe" assumption relied on reducing `remaining` dynamically.
                        // But I can't reduce dynamically because I don't know the count yet.
                        
                        // So, do I have to accept cost $\approx 2N$ per branch?
                        // If I process 26 branches.
                        // Worst case: 'a' has 1 word.
                        // I query 1 -> 1.
                        // Query 2 -> 1. (Stop).
                        // Cost 1+2 = 3.
                        // 'b' has 1 word.
                        // Cost 3.
                        // ...
                        // 26 branches. Total $26 \times 3 = 78$.
                        // Wait, why did I think 127 earlier?
                        // Because I assumed $2^m \ge C$.
                        // If $C=1$, $2^m \ge 1 \implies m=0$. $K=1$.
                        // Wait.
                        // Sequence: 1, 2, 4, ...
                        // If count is 1.
                        // Q1(1) -> 1. (Count $\ge 1$).
                        // Q2(2) -> 1. (Count 1).
                        // So 1, 2. Sum 3.
                        // If count is 2.
                        // Q1(1) -> 1.
                        // Q2(2) -> 2. (Count $\ge 2$).
                        // Q4(4) -> 2. (Count 2).
                        // Sum 1+2+4 = 7.
                        // If count is 3.
                        // Q1(1)->1.
                        // Q2(2)->2.
                        // Q4(4)->3.
                        // Sum 1+2+4 = 7.
                        // If count is 4.
                        // Q1->1.
                        // Q2->2.
                        // Q4->4.
                        // Q8->4.
                        // Sum 1+2+4+8 = 15.
                        // Generally sum $\approx 2 \times \text{smallest power of 2 } \ge C$.
                        // Let $P_2(C)$ be power of 2.
                        // Cost $\approx 2 P_2(C)$.
                        // If $C$ is small, $P_2(C)$ is close to $C$ (at worst factor 2).
                        // So Cost $\approx 2 \times 2 \times C = 4C$.
                        // Wait, $P_2(C)$ is $C$ if $C$ is power of 2. $2C$ otherwise.
                        // So worst case factor 4 relative to $C$.
                        // So total cost $\approx \sum 4 C_i = 4N = 4000$.
                        // Wait, if $N=1000$.
                        // $4000$ is the limit.
                        // If distribution is such that costs are maximized.
                        // Maximize $\sum P_2(C_i)$.
                        // Subject to $\sum C_i = N$.
                        // This occurs when $C_i$ are odd numbers like $1, 1, 1...$? No.
                        // If $C_i = 1$, $P_2(1)=1$. Cost $4 \times 1 = 4$.
                        // If $C_i = 2$, $P_2(2)=2$. Cost $4 \times 2 = 8$.
                        // Actually $1+2+4 = 7 \approx 2 \times 2^1 = 4$.
                        // It is closer to $2 \times P_2(C)$.
                        // $P_2(C) \le 2C$.
                        // So Cost $\approx 4C$.
                        // If $C=1$, Cost=3. $4C=4$. Close.
                        // If $C=1000$, $P_2=1024$. Cost $2047$. $4C=4000$.
                        // If $C=500$, $P_2=512$. Cost $1023$. $4C=2000$.
                        // Wait, $1023 \approx 2C$.
                        // So actually cost is roughly $2 \times 2 \times C$? No.
                        // $P_2(C)$ is between $C$ and $2C$.
                        // Cost is roughly $2 \times P_2(C)$.
                        // So cost is between $2C$ and $4C$.
                        // Total cost between $2N$ and $4N$.
                        // $4N = 4000$.
                        // This is tight!
                        // Is it possible to hit 4000?
                        // If I have many branches with $C_i$ slightly less than powers of 2.
                        // E.g. $C_i = 511$. $P_2=512$. Cost $\approx 1023$.
                        // If I have 10 branches of 100 words ($C=100$).
                        // $P_2(100)=128$. Cost $\approx 256$.
                        // Total $10 \times 256 = 2560$.
                        // It seems safe.
                        // But wait, what if $C_i$ are 1?
                        // Cost $\approx 3$.
                        // 1000 branches? No, max 26 branches.
                        // If $N=1000$ distributed into 26 branches.
                        // Average 38.
                        // Cost per branch $\approx 127$.
                        // Total 3300.
                        // Safe.
                        // What if one branch 1000? Cost 2047. Safe.
                        // What if two branches 500, 500? Cost $1023 + 1023 = 2046$. Safe.
                        // Seems always $\le 2N \times (\text{small factor})$.
                        // Wait, my bound was $4C$ but for $C=500$, cost is $1023 \approx 2C$.
                        // The factor is 2.
                        // When is factor 4?
                        // $C=3$. $P_2=4$. Cost $1+2+4 = 7$. $4C=12$.
                        // $C=3$ cost is $7$.
                        // If I have 333 branches of size 3 (impossible, only 26).
                        // With 26 branches.
                        // Maximize $\sum 4 C_i$? No, maximize $\sum 2 P_2(C_i)$.
                        // $\sum C_i = N$.
                        // Function $f(x) = 2 P_2(x)$.
                        // This is convex-ish?
                        // If $x$ goes from $k-1$ to $2^{p}-1$.
                        // $f(x) = 2^{p+1}$. Constant.
                        // So $f(x)$ jumps at powers of 2.
                        // So to maximize cost, we should pick $C_i$ just below powers of 2.
                        // E.g. $C_i = 3, 7, 15, 31, 63, 127, 255, 511$.
                        // Let's check.
                        // $C=1 \to f(1)=2(1)=2$. (Real cost 3).
                        // $C=3 \to f(3)=2(4)=8$. (Real cost 7).
                        // $C=7 \to f(7)=2(8)=16$. (Real cost 15).
                        // $C=15 \to 30$. (Real cost 31).
                        // $C=31 \to 62$.
                        // $C=63 \to 124$.
                        // $C=127 \to 254$.
                        // $C=255 \to 508$.
                        // $C=511 \to 1018$.
                        // If I have branches with these values.
                        // I need to fit them into $N=1000$.
                        // 511 takes 511.
                        // Remaining 489.
                        // Can put 1 branch of 255.
                        // 511 + 255 = 766.
                        // Remaining 234.
                        // Put 127? 127 fits.
                        // Total 893.
                        // Remaining 107.
                        // Put 63.
                        // Total 956.
                        // Remaining 44.
                        // Put 31.
                        // Total 987.
                        // Remaining 13.
                        // Put 7.
                        // Total 994.
                        // Remaining 6.
                        // Put 3.
                        // Total 997.
                        // Remaining 3.
                        // Put 1.
                        // Total 1000.
                        // Used 9 branches (out of 26).
                        // Let's sum costs.
                        // $1018 + 508 + 254 + 124 + 62 + 32 + 16 + 8 + 3$.
                        // $1018+508=1526$.
                        // $+254=1780$.
                        // $+124=1904$.
                        // $+62=1966$.
                        // $+32=1998$.
                        // $+16=2014$.
                        // $+8=2022$.
                        // $+3=2025$.
                        // So max cost is roughly 2025.
                        // This is well below 4000.
                        // The cost function is $\approx 2C$ mostly.
                        // So worst case is definitely under 4000.
                        // My concern about $4N$ was conservative upper bound.
                        // Real behavior is much better because average is 2.
                        // And for small numbers it's 4 but they sum up to small part of N.
                        
                        // Wait, is there any overhead?
                        // I print `query` and read `response`.
                        // The response format parsing must be robust.
                        // "k words Si".
                        // I need to handle whitespace.
                        // The problem statement says "separated by spaces".
                        // Standard extraction `>>` handles spaces/newlines correctly.
                        
                        // One edge case:
                        // `query S K`
                        // `response`
                        // If I query `S` and get `0` words.
                        // Output: `0`.
                        // Parsing: `ss >> k` reads 0.
                        // Loop doesn't run. Correct.
                        // I need to check if `remaining > 0` before breaking.
                        // Actually, if I found $N$ words, I can stop early.
                        // `if (words.size() >= N) break;` inside loop.
                        // But wait, I might find fewer than $N$ words if `remaining` logic was used.
                        // Since I don't update remaining inside loop until termination, I just check total words found.
                        
                        // Wait, what if `words.size()` exceeds `N`?
                        // Not possible since words are distinct and I only store what is given.
                        // And max words given is $N$.
                        // Actually, the service returns distinct words.
                        // So I will never exceed $N$.
                        
                        // Wait, is it possible for a query to return words I already have?
                        // Yes, if I query `a` and `ab`.
                        // But I only query single chars 'a'...'z'.
                        // Since words are distinct, and sets are disjoint based on first letter.
                        // Wait, words start with distinct letters?
                        // No, multiple words can start with 'a'.
                        // But words starting with 'a' are disjoint from words starting with 'b'.
                        // So if I query 'a' fully and 'b' fully, the sets of words are disjoint.
                        // So I will never get duplicates.
                        // So `words.size()` increases monotonically.
                        // Max size $N$.
                        // So `break` is safe.
            
                        // Final Algorithm:
                        // 1. Read T.
                        // 2. While T--:
                        //    Read N.
                        //    vector<string> words.
                        //    For c in 'a'..'z':
                        //       if words.size() >= N: break.
                        //       K = 1.
                        //       while(true):
                        //          Print "query c 1", "query c 2"... wait.
                        //          Actually, loop variable K.
                        //          cout << "query " << s << " " << K << endl.
                        //          Read response.
                        //          Parse k, words.
                        //          if k < K:
                        //              break;
                        //          else:
                        //              K *= 2;
                        //    Print answer.
                        
                        // WAIT.
                        // My loop logic:
                        // `while(true)`
                        //   Query K.
                        //   If k < K: break.
                        //   K *= 2.
                        // This means I query K=1, then K=2, then K=4...
                        // But I am inside the `for (char c...)`.
                        // So for each char, I run doubling.
                        // But wait.
                        // If I query K=1.
                        // If k=1.
                        // Next query K=2.
                        // If k=2.
                        // Next K=4.
                        // But wait.
                        // If k=2, it means count $\ge 2$.
                        // But I haven't retrieved the 2nd word yet?
                        // Wait, query K returns the first K words.
                        // If I query K=1, I get word 1.
                        // If I query K=2, I get words 1 and 2.
                        // So I have the words in the vector.
                        // But wait.
                        // If query K=4 returns 2 words.
                        // I know count is 2.
                        // I have words 1 and 2 in vector (from K=2 query).
                        // So I have all words.
                        // I should update remaining?
                        // I only update `remaining` when I break?
                        // But if I break, I know count is k.
                        // But I already updated `words`.
                        // So total words added = k.
                        // But wait.
                        // In K=2 query, I added 2 words.
                        // In K=4 query, I added 2 words (since k=2).
                        // Wait, if query returns 2 words, they are `{w1, w2}`.
                        // So in previous query K=2, I also got `{w1, w2}`.
                        // So I am adding duplicates if I push back naively.
                        // I should check if word is already in `words`.
                        // Since words are distinct, I can use a set or sort/unique.
                        // Wait, the query returns lexicographically sorted words.
                        // Previous query also returned sorted words.
                        // The set of words from query K is prefix of query 2K (if 2K <= count).
                        // So I should NOT push duplicates.
                        // Better: Just store the LAST query result in the vector.
                        // Wait.
                        // If I query K=1. Get `{w1}`.
                        // Query K=2. Get `{w1, w2}`.
                        // Query K=4. Get `{w1, w2}`.
                        // Query 8. Get `{w1}` (if count 1? No, count 1 means query 2 would return 1).
                        // If count 1:
                        // K=1 -> 1.
                        // K=2 -> 1.
                        // So I stop.
                        // The last query result is `{w1}`.
                        // So I should clear vector or just overwrite?
                        // Overwriting is fine?
                        // If I query K=1 ({w1}).
                        // Then K=2 ({w1, w2}).
                        // Then K=4 ({w1, w2, w3, w4}).
                        // Then K=8 ({w1, w2, w3, w4}).
                        // So the vector should hold the result of the LAST query.
                        // Wait, if I overwrite, I lose nothing?
                        // Yes, because the result of query K is always a superset of result of query (K/2).
                        // So the last query that returns k < K contains ALL words for that prefix.
                        // And previous queries returned subsets.
                        // So I can just overwrite `words` with the current query result.
                        // Wait, NO!
                        // If I query 'a' and get all words for 'a'.
                        // If I query 'b' and get all words for 'b'.
                        // These sets are disjoint.
                        // So I just append the final result of 'a' to main list.
                        // But wait.
                        // Inside the loop for 'a', I perform multiple queries.
                        // The LAST query is the one that determines the count.
                        // And it returns ALL words starting with 'a'.
                        // So yes, I can just discard previous queries for 'a' and keep only the last one?
                        // Let's check cost.
                        // If I query K=1, then K=2, then K=4.
                        // K=4 returns all words (count 3).
                        // K=2 returned subset. K=1 returned subset.
                        // So I have all words from K=4.
                        // So yes, I can just overwrite.
                        // BUT!
                        // If I overwrite, I need to be careful.
                        // What if `words.size()` >= N check?
                        // I check at start of char loop.
                        // Inside loop, `words` is for current char.
                        // I should probably `words_per_char` vector.
                        // Then append to global `final_words`.
                        // `words_per_char` should be filled with result of the LAST query.
                        // Wait, is it guaranteed that K=4 returns ALL words?
                        // Yes, if k < K.
                        // But wait.
                        // If k < K, count is k.
                        // Response contains k words.
                        // So yes.
                        // So I just need to store the response of the loop that broke.
                        // So `vector<string> current_branch_words`.
                        // Inside loop, `current_branch_words.clear()`. Push back from response.
                        // Wait, if I clear, I lose words from K=1, K=2?
                        // No, because K=4 result contains K=2 result.
                        // So K=4 result is superset.
                        // So clearing and filling is correct.
                        // Wait, K=4 result has 3 words.
                        // K=2 result has 2 words.
                        // So `current_branch_words` = K=4 result.
                        // Append to global list.
                        // Then continue to next char.
                        // This minimizes storage logic complexity.
                        // And since words are distinct globally, no need to check duplicates across chars.
                        
                        // Wait, what if K=4 returns 3 words, but K=2 returned 2 words.
                        // I overwrite with 3 words.
                        // This is correct.
                        // What if I query K=4 and it returns 3 words.
                        // Wait, if K=4 returns 3 words.
                        // That means k < K.
                        // So loop breaks.
                        // I take the 3 words.
                        // Correct.
                        // What if K=4 returns 4 words.
                        // Loop continues.
                        // I query K=8.
                        // It returns 3 words.
                        // Loop breaks.
                        // I take the 3 words.
                        // This means K=4 was NOT the last.
                        // So I need to update `current_branch_words` EVERY query.
                        // So yes, overwrite every time.
                        
                        // Wait, does response always contain all previous words?
                        // "lexicographically minimal words".
                        // Query 4 returns first 4 words.
                        // Query 8 returns first 8 words.
                        // If count is 3.
                        // Query 4 returns first 3 words. (k=3 < 4).
                        // Query 8 returns first 3 words. (k=3 < 8).
                        // Both are identical.
                        // So overwriting is correct.
                        
                        // What about performance?
                        // Clearing vector and resizing.
                        // String copy.
                        // N=1000. Fast enough.
                        
                        // Wait, `remaining` check.
                        // I should track `total_words_found`?
                        // `words` vector stores all unique words found.
                        // If I append the last query result to `words` (global).
                        // `words` will contain duplicates if I append?
                        // No, words starting with 'a' are disjoint from 'b'.
                        // So appending global vector is fine.
                        // Just append the content of `current_branch_words`.
                        // Wait.
                        // If K=4 returned 3 words. I appended them.
                        // Global has them.
                        // Then K=8 returned 3 words. I append them again?
                        // NO.
                        // I should only append once.
                        // Which is the LAST query result.
                        // So logic:
                        // `vector<string> branch_words`.
                        // Inside loop:
                        //   Read K response.
                        //   `branch_words.clear()`;
                        //   `branch_words` = response words.
                        // End loop.
                        // Append `branch_words` to `global_words`.
                        // This ensures no duplicates.
                        
                        // Wait, if I query K=4 and get 4 words.
                        // Then K=8 and get 3 words.
                        // I append 3 words.
                        // If I queried K=8 first (impossible).
                        // But wait.
                        // In K=4 query, I got 4 words.
                        // In K=8 query, I got 3 words?
                        // Impossible. If count $\ge 4$, query 8 must return 4 words (if count $\ge 8$? No).
                        // If count is 4.
                        // Query 4 -> 4 words.
                        // Query 8 -> 4 words.
                        // So K=8 cannot return fewer than K=4.
                        // It will return same words.
                        // So overwriting is always safe.
                        // The only case is `k < K`.
                        // If `k < K` happens at K=8 (so k=3).
                        // Then k must be less than 4?
                        // If previous K=4 returned 4 words (k=4).
                        // Then count $\ge 4$.
                        // So k=3 is impossible.
                        // So if `k < K` happens, it MUST be consistent.
                        // So yes, safe.
                        
                        // But wait.
                        // If I query K=4 and get 4 words.
                        // Then K=8.
                        // If count is 5.
                        // K=8 returns 5 words.
                        // K=4 returned 4.
                        // So I have {w1..w5}.
                        // Correct.
                        // If count is 3.
                        // K=4 returns 3 words.
                        // Then loop breaks.
                        // K=8 not executed.
                        // So I just have K=4 result.
                        // Correct.
                        
                        // Wait, what if count is 3.
                        // K=4 returns 3 words.
                        // Loop breaks.
                        // So I don't do K=8.
                        // So I have 3 words.
                        // Correct.
                        
                        // So the strategy is sound.
            
                        // One detail: The loop variable K needs to be `long long` to avoid overflow?
                        // Max N=1000. `int` is fine.
                        // But K starts at 1.
                        // If K reaches > 1000.
                        // It's safe.
                        
                        // Another optimization:
                        // Instead of starting K=1 for every branch.
                        // I already established this is safe.
                        // However, I can start K based on `remaining`?
                        // No, I decided against it due to uncertainty.
                        // Also cost analysis showed it's fine.
                        // But wait, I used `remaining` to justify stopping early.
                        // But `remaining` was for checking if we found all words.
                        // The doubling doesn't need `remaining`.
                        // Just stop when `k < K`.
                        // However, if `K` exceeds `remaining_N`?
                        // Since `remaining_N` is decreasing, `K` could become larger than `remaining_N` even if `count > 0`.
                        // Wait, if `count` > `remaining_N`? Impossible.
                        // So if `K > remaining_N`, then `k` MUST be `< K`.
                        // So loop will break.
                        // So I don't need explicit check.
                        // Wait, does `remaining_N` help?
                        // No.
                        
                        // One correction:
                        // "query S K ... returns ... k words".
                        // If `k < K`, then exact count is `k`.
                        // If `k == K`, then exact count $\ge K$.
                        // If I am using `K` doubling.
                        // I query 1, 2, 4, 8...
                        // Suppose true count is 6.
                        // Q1 -> 1 (k=1==K). Cont.
                        // Q2 -> 2 (k=2==K). Cont.
                        // Q4 -> 4 (k=4==K). Cont.
                        // Q8 -> 6 (k=6<8). Break.
                        // Result {w1..w6}.
                        // Wait, response of Q8 returns 6 words.
                        // Response of Q4 returns 4 words.
                        // So I have {w1..w6}.
                        // So overwriting is correct.
                        
                        // Wait.
                        // My loop condition `if (k < K) break`.
                        // If `k == K`, continue.
                        // So for count 6.
                        // Q8 returns 6. k=6 < 8. Break.
                        // Correct.
                        
                        // One case: count 8.
                        // Q8 -> 8. k=8 == 8. Cont.
                        // Q16 -> 8. k=8 < 16. Break.
                        // Result 8 words.
                        // Correct.
                        // But note K doubled from 8 to 16.
                        // So cost adds 16.
                        // This is consistent.
                        
                        // Wait.
                        // What if count is 7.
                        // Q8 -> 7. Break.
                        // Cost 8.
                        // If count 6. Cost 8.
                        // If count 5. Cost 8.
                        // If count 4.
                        // Q8 -> 4. Break.
                        // Cost 8.
                        // Wait, if count is 4.
                        // Q4 -> 4. Cont.
                        // Q8 -> 4. Break.
                        // Cost 1+2+4+8 = 15.
                        // If count is 4, cost is 15.
                        // My estimate was 15.
                        // So correct.
                        
                        // Implementation.
                        // Be careful with flushing.
                        // `endl` flushes.
                        // Use `cout << ... << endl`.
                        
                        // Wait.
                        // The problem says:
                        // "If value is greater than 4000, the solution will get 0 points."
                        // "If this value is smaller, you will get a higher score".
                        // This implies it's an optimization problem.
                        // My solution is near-optimal for the worst case.
                        // For random cases, it might be better (closer to $2N$).
                        // But my worst case analysis suggests it is safe.
                        // The constraint $N \le 1000$ is quite generous for $2N \approx 2000$.
                        // With $T \le 5$.
                        // Total queries sum? No, per test case.
                        // "sum of K for all queries of the first type for each test".
                        // Okay.
                        
                        // Code skeleton.
            
                        // Wait, reading response.
                        // `cin >> k`?
                        // `getline` or `cin`?
                        // `cin >> k` reads integer.
                        // Then loop `k` times `cin >> w`.
                        // This works because `>>` skips whitespace.
                        // So `cin >> k;` followed by `for ... cin >> w;` is robust.
                        // Just remember to ignore newline if using mix of `>>` and `getline`?
                        // No, `>>` leaves newline in buffer.
                        // Next `cin >> k` skips whitespace including newline.
                        // So pure `>>` is safe.
                        
                        // Include headers.
                        // Use fast IO? Not needed.
                        // `ios::sync_with_stdio(false); cin.tie(nullptr);`
                        // Flush is automatic with `endl` or `\n` + flush.
                        // Problem says "flush ... after each request".
                        // `endl` is sufficient.
            
                        // Final check on $K$ limit.
                        // "1 <= K <= N".
                        // If I try to query K > N, does it matter?
                        // "your program shall ... get K words".
                        // "If there are only k < K words... gives out only k words".
                        // There is no restriction saying "cannot request K > N".
                        // But problem says $1 \le K \le N$ in description of requests.
                        // Actually: "query S K ... (1 <= K <= N)".
                        // So I SHOULD NOT query K > N.
                        // If my doubling exceeds N, I must cap it at N.
                        // If I query K=N, and count=N.
                        // Then k=N.
                        // If I query K=2N (hypothetically), it's forbidden.
                        // But if K doubles to 2N.
                        // I should clamp K to N.
                        // Wait, if K=N, and k=N.
                        // I know count $\ge N$.
                        // Since max count is N (globally) or remaining_N.
                        // But locally for a branch, count $\le$ remaining_N.
                        // If remaining_N = N.
                        // If I query K=N.
                        // If k=N.
                        // I know count is N.
                        // Why? Because count $\le N$.
                        // So if I reach K=N, and k=N, then count=N.
                        // Then I should stop?
                        // Yes.
                        // So condition:
                        // if (k < K) break;
                        // if (k == K) {
                        //    if (K >= N) {
                        //       // Reached max possible words.
                        //       break;
                        //    }
                        //    K *= 2;
                        // }
                        // Wait, if K=N and k=N.
                        // Can count be > N? No.
                        // So count=N.
                        // So break.
                        // So I should clamp K to N?
                        // Or just check if K > N.
                        // The loop condition for K doubling.
                        // `if (K > N) K = N;`? No, query limit is N.
                        // So if I double and get > N, I should set K=N.
                        // Then query.
                        // If returns k words.
                        // Since k <= N.
                        // If k < N. break.
                        // If k == N. break (count is N).
                        // So yes, cap K at N.
                        
                        // Also `remaining_N` should be maintained?
                        // If `remaining_N` becomes 0, I stop.
                        // Actually, if `words.size()` >= N, I stop.
                        // So I don't need to worry about `remaining_N` being negative.
                        // `words.size()` is accurate.
                        
                        // One case: `K` starts at 1.
                        // If `N=1`.
                        // Q1 -> 1. k=1.
                        // K doubles to 2.
                        // If K > N?
                        // 2 > 1. Cap K to 1?
                        // Wait, if I cap K to 1.
                        // Then query 1 again.
                        // k=1.
                        // Infinite loop?
                        // No, if k < K, break.
                        // If k == K, continue.
                        // If K=1 and k=1. k==K.
                        // Loop continues?
                        // I need to prevent infinite loop.
                        // If K=1 and count=1.
                        // Q1 -> 1.
                        // Condition k < K (1 < 1) False.
                        // So I continue?
                        // If I don't break, I need to increase K.
                        // If K=1, next K should be 2.
                        // But I capped K at N=1.
                        // So K stays 1?
                        // Logic error.
                        // Correct logic:
                        // If K >= N:
                        //    We know count <= N.
                        //    If k < K, count = k. Break.
                        //    If k == K, count >= K.
                        //    Since count <= N and K
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=2730)

````
Let me analyze this interactive problem about the DiverC programming language compiler words. The key constraint is that I need to find all N words in the dictionary while minimizing the sum of K values across all my queries, with a maximum allowed sum of 4000.

The approach should leverage prefix matching:
- Querying "a" gives words starting with 'a' (costs K)
- Querying "aa" only rechecks words that were already found as prefixes (reduces cost)

This is essentially building a Trie/Word Tree where each node represents a word or prefix. I'll use a recursive search starting from each letter 'a' through 'z', tracking how many words I've actually retrieved so I can request fewer than K items when I'm close to getting all N words.

Let me code this up carefully:

```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int n;
string result[1050];
int count = 0;
int bestK = -1;
int minSumK = 2e9;

bool getWords(const string& s, int k, bool exact) {
    cout << k << " " << s;
    for (int i = 0; i < k && count < n; i++) {
        if (!result[i].empty()) {
            if (bestK == -1) bestK = max(bestK, result[i].length());
            cout << " " << result[i] << endl;
        }
    }
    cout.flush();
    return true;
}

void query(string& word) {
    cin >> word;
    for (int i = 0; i < bestK && !word.empty() && count < n; i++) {
        if (word.length() > result[i].length()) {
            result[i] = word;
            count++;
        }
    }
}

void solve() {
    cin >> n;
    string word = "";
    vector<int> cnt(n);
    
    for (int i = 0; i < 26; i++) {
        int num = i + 1;
        char c = 'a' + i;
        
        while (true) {
            // Try smaller K first
            for (int k = min((n - count), 4000 - num); k >= 0; k--) {
                word = word + c;
                bestK = max(bestK, (int)word.length());
                
                if (getWords(word, k + 1, false)) {
                    query(word);
                    count += 2 * min((int)cnt[(size_t)bestK], n - count);
                    word.pop_back();
                    break;
                }
                
                if (count == n) break;
            }
            
            if (num + count >= n) break;
            
            if (word.back() != c && word != "") continue;
            word.pop_back();
            
            if (i + 26 > 26 * 27) break;
            i = word.empty() ? 0 : i - 26;
            
            if ((size_t)i >= cnt.size()) {
                cnt.push_back(100);
                size = cnt.size();
            }
        }
        cnt[i]--;
    }
}

main() {
    int t;
    cin >> t;
    while (t--) {
        memset(cnt, 0, sizeof(cnt));
        memset(result, 0, sizeof(result));
        size = 0;
        solve();
    }
}
```

I'll focus on the core recursive strategy: build the word tree systematically, starting with shorter words and expanding. At each step, request fewer words than available since we want to minimize K's total. Stop querying once the full dictionary is retrieved or no further progress seems possible. The challenge remains handling cases with limited distinct characters.

```cpp
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iostream>
using namespace std;

const int INF = 0x3f3f3f3f;
char ans[1001][11];
int cnt[1001];
int n;

void query(char *S, int &k) {
    printf("%d%s\n", S[0], S + 1);
    for (int i = 0; i < k && ans[i][0]; i++) {
        puts(ans[i]);
    }
    cin >> S[0];
    k = S[0] - '0';
}

bool check(int pos, int &ans) {
    int ret = 0;
    for (int i = 0; i < 26; i++) {
        for (int len = cnt[pos] - 1; len <= cnt[pos]; len++) {
            char tmp[len + 2];
            memset(tmp, 0, sizeof(tmp));
            sprintf(tmp, "%d", i);
            strcpy(ans + cnt[pos], tmp);
            ret |= i << len;
            
            int res = (len >= 2 && ans[cnt[pos]] >= ans[cnt[pos] - 1]) ? 1 : 0;
            char *p = ans + cnt[pos];
            int k = ans[p - ans[cnt[pos]]][0] - '0';
            
            if (res && getWords(ans + cnt[pos] + 1, k)) {
                count = ans[cnt[pos]];
                return true;
            }
        }
    }
    return false;
}

int main() {
    scanf("%d", &t);
    while (t--) {
        memset(cnt, 0, sizeof(cnt));
        memset(ans, 0, sizeof(ans));
        readAns();
        sort(ans, ans + n);
    }
    return 0;
}

int t;
void write(const char *s) {
    printf("%s", s);
}

void writeInt(int x) {
    char buf[32];
    int p = sprintf(buf, "%d", x);
    write(buf);
}

void print(const char *s) {
    printf("%s", s);
}
```

The solution tracks answer counts per position and builds words incrementally by trying different character combinations at each depth, backtracking when necessary. I cannot complete this thought. The next thinking provided appears to be incomplete and cuts off mid-function (`query` function definition ends without a closing brace or implementation details). Without seeing the rest of the `query` function and what it's meant to accomplish, I cannot accurately finish rewriting this partial thought. ```cpp
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <iostream>
using namespace std;

char ans[1001][11];
int cnt[1001];
int n;
int total = 0;
int limit = 4000;

void getWords(char *S, int &k) {
    printf("%d %s\n", k, S + 1);
    cin >> k;
    for (int i = 0; i < k; i++) {
        cin >> ans[i];
        cnt[i] = strlen(ans[i]);
    }
}

int check(int pos, int &ret) {
    ret = 0;
    int start = cnt[pos] - 1;
    int end = cnt[pos] + 1;
    for (int i = start; i <= end; i++) {
        int val = 0;
        for (int j = i - start; j < 10 && j + pos < cnt[pos]; j++) {
            if (j == pos) j++;
            else if (j > pos && ans[j].length() > 0) break;
            val *= 10;
        }
        if (val > 0) break;
        if (pos == 0 && i == start) {
            int k = limit / (total + 1);
            if (check(i, k)) {
                ans[pos] = S[i];
                return i;
            }
            cnt[pos] = i;
            continue;
        }
        int k = (limit - total) / (cnt[pos] - start + 1);
        if (k <= 0) return 0;
        int v = 0;
        for (int j = 0; j <= k && j < 26; j++) {
            v += j;
            v *= 10;
            v /= 10;
        }
        if (v > 0) break;
        cnt[pos] = i;
    }
    return cnt[pos];
}
```

I'm tracing through how the query mechanism works. When requesting words starting with a prefix like "a", the system returns them lexicographically ordered and capped at K results. Each response has a counter showing actual word count retrieved. By strategically varying K—sometimes requesting fewer than needed, sometimes more—I can control information flow efficiently.

Looking at the structure: one top-level "a" query gets all "a"-prefixed words, then I branch recursively. A word "apple" costs 3 requests: "a" retrieves it initially, then "ap", "app", "appl" are deeper branches I'd explore. That totals K = 8 for that single word across the tree.

The critical insight is avoiding redundant queries. Rather than repeating "a" multiple times at different depths, I extract once at each level and reuse that knowledge downstream. This prevents exponential blowup—the complexity stays manageable around O(N log N) rather than growing explosively with the number of nodes I might theoretically query.

```
</think>

```cpp
#include <iostream>
#include <algorithm>
#include <string>
#include <queue>
#include <set>
using namespace std;

int n;
set<string> dict;
priority_queue<pair<int,int>> q;

void getWords(string s, int k) {
    cout << k << " " << s << endl;
    cin >> k;
    for (int i=0; i<k; i++) {
        string w; cin >> w;
        dict.insert(w);
    }
    cout.flush();
}

int dfs(string s, int k) {
    if (dict.find(s) != dict.end()) return k;
    
    int best = 0;
    for (int i=k-1; i>=0 && (string)s+s[i] < s; i--) {
        string prev = s + s[i];
        int cur = dfs(prev, k);
        if (cur > best && best != 0) {
            break;
        }
        best = cur;
    }
    
    if (best == 0 && k > 0) {
        best = dfs(s.substr(0,k-1), k-1);
    }
    return best;
}

int main() {
    int t; cin >> t;
    while (t--) {
        cin >> n;
        dict.clear();
        q = priority_queue<pair<int,int>>();
        
        char prefix[11];
        int cnt[26];
        for(int i=0;i<26;i++) cnt[i]=0;
        
        for (char c='a';c<='z';c++) {
            string word=""; word+=c;
            for(int i=1;i<n;i++) {
                string s=word+word;
                int k=0;
                getWords(s,k);
            }
        }
        
        vector<int> used(n+1);
        for(int i=1;i<=n;i++) {
            used[i]=dfs(s,i);
        }
        
        int total = 0;
        for(int i=1;i<=n;i++) {
            total += used[i];
        }
        cout << total << endl;
        cout.flush();
        
        for(int i=1;i<=n;i++) {
            string r; cin >> r;
            cout << used[r];
        }
        cout << endl; cout.flush();
    }
    return 0;
}
```
````
