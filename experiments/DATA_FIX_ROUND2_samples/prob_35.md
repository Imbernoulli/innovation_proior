# 题 35 — base均分 9.5 vs 直接SFT均分 0.0(掉点 9.5)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=2,得分 47.5,completion_tokens=28375)

````
The problem asks us to find the unique element in a hidden permutation $a$ of size $2n-1$, where every number from 1 to $n$ appears exactly twice except for one number that appears once. We can query if a value $x$ exists at index $i$ in the subset $S$. Wait, let me re-read carefully.

"You can ask queries in the following format, where S is a subset of {1,2,…,2n−1} and x is an integer in [1,n]:
? x |S| S1 S2 ... S|S|

The answer to this query is: does there exist i ∈ S such that a_i = x?"

So we query if $x$ is present in positions specified by $S$. The array $a$ contains numbers $1 \dots n$ where every number appears twice except one appearing once. So total length is $2n-1$.

Wait, the example explanation says:
"? 187 1 1" -> asking if there exists $i \in \{1\}$ such that $a_1 = 187$. This checks a single position.
Response: 0.
Then output "! 187". This means they somehow guessed 187 is the unique one. That doesn't make sense with just one query unless there's a trick or I misunderstood something huge.

Let me re-read the problem statement provided in the prompt.
"This is an interactive problem.
There is a hidden array a containing all the numbers from 1 to n, and all of them appear twice except one (which only appears once).
You can ask queries... ? x |S| S1 ...
The answer is: does there exist i in S such that a_i = x?"

Actually, the scoring says "If you solve it with 500 queries, you get 100 points. If you solve it with 5000 queries, you get 0 points."
But the constraints say Time limit 5 seconds, Memory 1024 MB.
Usually, if there are multiple test cases ($t \le 20$), and $n=300$, the total number of elements across test cases is not fixed per problem but usually sum of N over test cases is constrained. Here it seems each test case has its own $n=300$. With 20 test cases, if we use 5000 queries per test case, that's way too much. But wait, the scoring says "solve it with 500 queries". Ah, so the limit is actually 500 queries for full points, but up to 5000 allowed for 0 points (or partial). The standard Codeforces interaction usually allows checking logic within reasonable bounds. 5000 is quite high for $n=300$. $O(n)$ or $O(n \log n)$ should be fine. $O(n^2)$ might be pushing it (90000 ops, but queries count matters, complexity of operations inside loop matters less than query count).

Wait, looking closely at the example input/output explanation:
Input:
1
300
0
Output:
? 187 1 1
0
! 187
Explanation:
Query: does a[1] = 187? -> No.
We then output ! 187.
Fortunately, the answer is correct.

This looks like a completely made-up example or very specific. It suggests we might need to guess correctly? Or maybe the problem description in the prompt is slightly different from what is typically expected in competitive programming problems involving "finding a duplicate/unique" where we check existence.
Wait, finding the unique element usually involves XORing indices or values. But here we query existence.

Let's consider the properties.
Array size $2n-1$. Elements $1 \dots n$. Each appears 2 times, except one appears 1 time.
Total occurrences = $2(n-1) + 1 = 2n - 1$. Correct.

Query: `? x |S| S1 ...` returns 1 if $\exists i \in S$ such that $a[i] == x$, else 0.
We want to find $y$ such that $y$ appears exactly once in $a$. All other $z \neq y$ appear exactly twice.

Let's analyze the query type.
If we query $x$ against ALL indices $1 \dots 2n-1$:
- If $x$ is the unique number, it appears once.
- If $x$ is not the unique number, it appears twice.
The query asks "Does $x$ exist in the set $S$?".
So `? x 2n-1 1 2 ... 2n-1` will always return 1 because every number appears at least once. That doesn't help distinguish between count 1 and count 2.

However, notice the problem statement again. "Note that the interactor is not adaptive, which means that the hidden array does not depend on the queries you make." This is crucial.

Since $n=300$ and we have 5000 queries allowed per test case (with bonus for <500).
With non-adaptive interactor, we can try to determine positions of numbers or counts indirectly.

Actually, if I pick two indices $u$ and $v$.
Suppose I want to know if $a[u]$ equals some value. But the query is about a *value* $x$ appearing in a set of *indices*.
Basically, given indices $S$, does the array contain value $x$ at any position in $S$.

Let's try to group indices.
If I take $S = \{1, 2\}$ and query $x=a[1]$. Then result is 1.
But I don't know $a[1]$.
Wait, if I assume a value $v$, query if it exists in $\{1\}$.
`? v 1 1`.
If yes, then $a[1] == v$. Then we found a position for $v$.
If no, $a[1] \neq v$.
This allows us to probe specific values at specific positions.
But since we don't know the values, probing $1 \dots n$ for $a[1]$ takes $n$ queries.
Doing this for all positions takes $(2n-1)n \approx 600 \times 300 = 180,000$ queries. Way too many.
We need something smarter.

We know the multiset of values is known: $\{1, \dots, n\}$ with one appearing once, rest twice.
Total size $2n-1 = 599$.

Let's consider the structure.
Every number appears 1 or 2 times.
If we query $x$ on indices $\{u, v\}$.
If $x$ is the unique number, it appears once in array.
If it is located at $u$, query returns 1. If at $v$, query returns 1. If elsewhere, 0.
If $x$ is duplicated, say at $p$ and $q$. Query on $\{u, v\}$ returns 1 iff $\{u,v\} \cap \{p,q\} \neq \emptyset$.

Key Idea:
Consider the property of pairs. For every $x \neq unique\_val$, there are exactly two indices $i, j$ such that $a[i]=a[j]=x$.
For $unique\_val$, there is exactly one index $k$ such that $a[k]=unique\_val$.

What if we partition the indices into pairs?
Indices $1 \dots 2n-1$. Total odd number of indices.
One index is left alone.
Maybe we can pair indices $1..2n-2$. Index $2n-1$ is singleton.
Or arbitrary pairing.

Let's look at a strategy using subsets of indices.
If we query $x$ on set $S$.
Count of appearances of $x$ in $a$ restricted to $S$ is unknown directly.
But we can observe:
If $x$ is unique in array, it contributes 1 to the answer if $S$ contains its position, 0 otherwise.
If $x$ is double in array, it contributes 1 to the answer if $S$ contains at least one of its positions.

Actually, there is a known technique for this problem. It's from a contest (Codeforces Round #something).
Problem Name: "Hidden Array", maybe? Or similar.
The query asks if $x$ is in $S$.
If we select a random subset of indices? No.

Let's rethink. $n=300$. Max queries ~ 5000.
If we can identify the unique value efficiently.
Consider the following strategy:
Iterate $x$ from $1$ to $n$.
Check if $x$ appears more than once.
How?
We need to find if there are at least two distinct indices $i \neq j$ such that $a[i] = a[j] = x$.
Or just find if the set of indices mapping to $x$ has size 1 or 2.
We know the size is either 1 or 2 globally.
We can check if $x$ appears at index 1? `? x 1 1`.
If yes, we record pos1.
Is there another occurrence?
To check if $x$ appears elsewhere, we need to know other indices.
This seems to require knowing positions.

Alternative Approach:
Can we find the unique element by querying sets of indices without knowing values first?
Actually, the query requires a value $x$.
So we must guess values or narrow down.

Wait, $n=300$. $O(n \log n)$ is $300 \times 8 \approx 2400$. Fits well within 5000. Even $O(n^2)$ is $90000$ queries which is too many (limit 5000).
So we need roughly $O(n \log n)$ or linear queries.

Let's try this logic:
Suppose we assume a candidate set of indices $S$ has a certain property related to $x$.
Actually, consider splitting the array indices into two halves $L$ and $R$.
For a fixed $x$, if we query $S=L$, do we find $x$?
If $x$ is unique, it is in $L$ or $R$ or not $L$ or not $R$.
Wait, unique $x$ is at exactly one index.
Dupes $x$ are at exactly two indices.
Suppose we pick a random partition of indices.
Query $x$ on $L$. Result 1 means $x$ is in $L$.
Query $x$ on $R$. Result 1 means $x$ is in $R$.
Case $x$ is unique:
- Pos in $L$: Q(L)=1, Q(R)=0.
- Pos in $R$: Q(L)=0, Q(R)=1.
Case $x$ is dupe:
- Both pos in $L$: Q(L)=1, Q(R)=0.
- One pos in $L$, one in $R$: Q(L)=1, Q(R)=1.
- Both pos in $R$: Q(L)=0, Q(R)=1.

This suggests a binary search approach for the location of $x$ IF we knew $x$.
But we don't know which $x$ corresponds to the pattern.
However, the pattern (distribution of counts) is uniform-ish.
Wait, if $x$ is unique, $Q(L)=1 \iff Q(R)=0$.
If $x$ is duped, $Q(L)=1, Q(R)=0$ OR $Q(L)=0, Q(R)=1$ OR $Q(L)=1, Q(R)=1$.
Specifically, $Q(L) \oplus Q(R) = 1$ if unique? No.
Unique in $L \implies Q(L)=1, Q(R)=0 \implies Q(L)+Q(R)=1$.
Unique in $R \implies Q(L)=0, Q(R)=1 \implies Q(L)+Q(R)=1$.
Dups both in $L \implies Q(L)=1, Q(R)=0 \implies sum=1$.
Dups split $\implies Q(L)=1, Q(R)=1 \implies sum=2$.
Dups both in $R \implies Q(L)=0, Q(R)=1 \implies sum=1$.

So checking on half indices doesn't separate "Unique" vs "Duplicates" perfectly because duplicates also result in sum=1 often.
BUT, duplicates result in sum=2 sometimes (if one in L, one in R).
Unique NEVER results in sum=2 (since it has only 1 pos).
So if we find an $x$ such that $Q(L)=1$ and $Q(R)=1$, then $x$ MUST be a duplicate (appears twice).
If we find $x$ such that $Q(L)+Q(R)=1$, it could be Unique OR Duplicate (both in same half).

Wait, we can verify if $x$ is a duplicate by just checking if it exists twice.
If we can determine which numbers appear twice, the remaining one is the answer.
But determining presence for all $x$ takes too many queries?
We have 300 numbers. Checking each with a large query.
If we take $S_{half}$ as first $n-1$ indices.
For each $x \in 1\dots n$:
Query $x$ on $S_{half}$.
If 0, then $x$ is not in first half.
If 1, then $x$ is in first half.
If $x$ is unique and in first half, it is in first half.
If $x$ is duplicate and both in first half, it is in first half.
If $x$ is duplicate and split, it is in first half (1 occurrence).
So "in first half" tells us nothing definitive.

However, notice the constraint: Total length $2n-1$.
We can iterate through indices $i=1 \dots 2n-1$.
For a specific index $i$, what is $a[i]$?
We can find $a[i]$ by binary searching among $1 \dots n$.
To check if $a[i] == k$, query `? k 1 i`. (Indices subset size 2: just index $i$? Wait, subset size must be $\ge 1$. Yes.)
Actually `? k 1 i` checks if $a[i] == k$.
Finding $a[i]$ takes $\log n$ queries. Total $(2n-1)\log n \approx 600 \times 8 \approx 4800$.
This is extremely close to the 5000 limit.
Wait, $\log_2(300) \approx 8.2$.
Worst case 9 queries per element. $600 \times 9 = 5400$. Too risky.
Average might be okay but worst case TLE (queries).
Also we don't know if $a[i]$ is unique or duped. If we reconstruct full array, we find counts.
But we need to identify WHICH element is unique.
Actually, if we find the exact array values, we are done. Just scan frequency map.
Cost: Sum of log(N) per index.
Can we optimize?
We don't need to find $a[i]$ for every $i$. We just need to count frequencies.
But finding frequencies of $x$ requires knowing where they are.

Let's rethink the query `? x |S| S...`.
We want to maximize info per query.
The problem allows 5000 queries. $N=300$. $5000/300 \approx 16$.
So we can afford ~16 queries per number.
Binary search for position of $x$ takes $\approx 9$ queries.
If we iterate $x$ from $1$ to $n$.
Find first occurrence of $x$. Let it be $pos(x)$.
To find $pos(x)$ in range $[1, 2n-1]$, we can binary search.
Range $[L, R]$. Mid $M$. Check if $x$ in $[L, M]$.
Query `? x (M-L+1) L ... M`.
Result:
1: $x$ is in left half. Recurse Left.
0: $x$ is NOT in left half. Since $x$ exists somewhere, recurse Right.
This takes $\lceil \log_2(2n-1) \rceil \approx 9$ queries.
Once we find ONE index for $x$, say $idx$.
We have established $a[idx] = x$.
Now we need to check if $x$ appears again.
Search for second occurrence in the rest of the indices.
Binary search again?
Wait, if $x$ is the unique one, there is NO second occurrence.
If $x$ is duplicate, there IS a second occurrence.
If we find two indices, $x$ is duplicate.
If we finish search and find only one, $x$ is unique.
But how many searches?
For each $x \in 1 \dots n$.
First search: locate *one* instance. (Cost $\approx 9$)
Second search: locate *another* instance?
If we don't find another, it's unique.
BUT, if we do the first search exhaustively, we effectively find *all* instances if we just mark "found"? No, the query is boolean "does exist".
Wait, if we find *one* position, we can't easily know if there's another without more work.
Actually, if $x$ is unique, it appears once. If we find it, great.
If $x$ is duplicate, it appears twice. Finding one instance takes 9 queries.
After finding one instance, we know where it is. To find if there is another, we can query the rest of the array excluding the found one.
Indices: $1 \dots 2n-1$. Remove found index.
Binary search on the remaining indices?
This costs another $\approx 9$ queries.
Total $2 \times 9 = 18$ queries per number.
Total $300 \times 18 = 5400$. Still slightly over 5000.
Plus overhead.
However, note that if $x$ is unique, we find it in first pass (cost 9), and then we realize it shouldn't appear again. Can we verify it doesn't appear again faster?
Well, if we assume $x$ is unique, we just need to confirm it.
But if we assume $x$ is duplicate, we need to find the second one.
Can we optimize the "find one instance"?
We don't need to find the EXACT position of $x$ for every $x$.
Just the set of unique numbers.

Let's look at the property again.
Most numbers appear twice. One appears once.
Suppose we fix a random bitstring of indices.
Say $P$ is a random partition of indices.
Query all $x$ on $P$. Count how many return 1.
If $x$ is unique and in $P$, returns 1.
If $x$ is dup and in $P$, returns 1.
This gives probability. Not deterministic.

Back to the binary search idea.
We need to reduce queries.
We can maintain a set of candidates.
Initially all $1 \dots n$ are candidates.
Can we eliminate numbers?
If we query $x$ on $S$.
If response is 0, then $x$ is not in $S$. This eliminates occurrences.
If response is 1, $x$ is in $S$.
If we use disjoint partitions of the array indices: $I_1, I_2$. $|I_1|=|I_2|=n-1$, overlap small?
Wait, total indices $2n-1 = 599$.
Split into two sets of indices $A$ and $B$ such that $A \cap B = \emptyset$.
Ideally $|A| \approx n, |B| \approx n$.
Query all $x \in 1 \dots n$ on $A$.
Store results: $HasInA[x]$.
Query all $x \in 1 \dots n$ on $B$.
Store results: $HasInB[x]$.
Note: For unique $x$, it is in exactly one index, so in $A$ or $B$ (but not both).
For dup $x$, it is in two indices. Could be:
- Both in $A$: In A=1, In B=0.
- Both in $B$: In A=0, In B=1.
- One in A, one in B: In A=1, In B=1.
So unique $x$ implies $(HasInA[x], HasInB[x])$ is $(1,0)$ or $(0,1)$.
Dup $x$ implies $(1,0), (0,1)$ or $(1,1)$.
This doesn't uniquely identify $x$ based on counts in these two sets. Because dup with both in A acts same as unique in B.
However, we know there is exactly ONE unique $x$.
Let $U$ be the set of unique candidates.
For all dup $x$, count of ones is either 1 (split or both in A/B) or 2 (one in each).
Wait, this still isn't distinguishing.

Wait, we can use the fact that if we can't find $x$ in $A$, it means all its occurrences are in $B$.
If we find $x$ in $A$, it could be unique or dup.
But notice: For duplicates, they occur at TWO positions.
So for a specific duplicate $x$, $x$ cannot be missing from $A$ AND missing from $B$.
i.e., $HasInA[x]$ or $HasInB[x]$ must be true. (Since $x$ exists in $A \cup B$).
And for unique $x$, same holds.
This doesn't filter anything.

Let's reconsider the cost.
We have 5000 queries.
If we can check "Is $x$ unique?" with fewer than average 20 queries.
Or if we can find the unique number faster.

Consider the following strategy:
We want to find the number that appears 1 time.
All others appear 2 times.
This is equivalent to finding the number $x$ such that Occurrences($x$) mod 2 = 1?
No, unique is 1, duplicates are 2. Yes.
If we could determine the number of times $x$ appears in the whole array.
If we query a SUBSET of indices, we just get a boolean.
We can query the WHOLE array $S = \{1 \dots 2n-1\}$.
It will ALWAYS return 1 for every $x \in 1 \dots n$. So useless.

Wait! We don't need to know the positions.
We just need to distinguish "Exists once" vs "Exists twice".
Suppose we pick a random permutation of indices $\sigma$.
Query $x$ on $\{\sigma_1\}$.
Query $x$ on $\{\sigma_1, \sigma_2\}$.
This finds the first two positions.
Too slow.

Wait, is there a simpler logic?
$n=300$. $5000$ queries.
If we use the binary search method for each $x$.
Cost to find *positions*:
For each $x$:
1. Find first position $p_1$ via binary search. Cost $\approx \log(2n) \approx 9$.
2. If we haven't found a second position yet?
Wait, if $x$ is the unique one, after finding $p_1$, it won't appear again.
If $x$ is a duplicate, we must find $p_2$.
However, do we need to verify uniqueness?
Actually, if we process numbers one by one, removing them from consideration.
Wait, if we find the position of $x$, we "use" that index? No, the hidden array is static. Indices are fixed.
But we can track which indices are confirmed to hold a specific value.
If we find that $a[p] = x$. Then for future queries on $x$, we know $p$ is an index where $x$ is.
If we see another index $q$ where $a[q]=x$, then $x$ is duped.
This suggests:
Maintain `map<int, vector<int>> locations`.
For each $x \in 1 \dots n$:
  Find first location of $x$.
    Binary Search on $1 \dots 2n-1$. (Cost 9)
    Found $idx$. Add to locations[$x$].
  Now check if another location exists.
    Binary Search on the remaining indices?
    Wait, Binary Search requires an order (subset definition).
    Remaining indices is a set of size $2n-2$.
    We can define subset $S'$ explicitly as all $j \neq idx$.
    We query $S'$.
    Result 1: There is another copy.
      Then we need to find its index?
      If we know there is another copy, we can just find it using binary search on $S'$.
      Cost $\approx 9$.
      Then add to locations[$x$].
      Mark $x$ as DUPLICATE.
    Result 0: No other copy.
      Mark $x$ as UNIQUE (candidate).
      Note: Wait, this logic assumes if we didn't find a second copy in $S'$, then it's unique.
      BUT, if $x$ was duped and both copies were in $S'$, query would return 1.
      If $x$ was duped and the second copy is AT $idx$? Impossible, index is unique per value.
      So if result 0, it means NO other index has value $x$. Thus $x$ is unique.
      
    Wait, this works IF we know $x$ appeared at least once.
    What if $x$ is NOT in the array?
    Problem says "hidden array containing all the numbers from 1 to n". So all appear.
    
    Algorithm refinement:
    Initialize `candidates = {}`.
    Iterate $x$ from $1$ to $n$.
      Step 1: Find first occurrence.
        Binary Search on range $[1, 2n-1]$.
        Wait, we can optimize range? Initially full range.
        Once we find $p$, we know $x$ is at $p$.
        Next, check if $x$ appears anywhere else.
        The set of possible locations is $[1, 2n-1] \setminus \{p\}$.
        However, we need to binary search over a SET.
        We can construct the subset explicitly.
        Subset construction cost? No, printing list is part of query. Length up to 600.
        Printing $600$ integers is fine.
        
        So, Query: `? x (2n-2) s1 ... s_{2n-2}` where $s_i$ are all indices except $p$.
        If response 1:
          There exists another copy.
          We need to find it.
          Binary Search on the remaining indices.
          Cost $\approx 9$.
          Once found, $x$ is duplicate. Ignore.
        If response 0:
          $x$ does not appear elsewhere.
          $x$ is unique.
          Candidate found. Break early?
          
    Optimization:
    We can stop as soon as we find a unique candidate?
    "Your task is to find the number appearing exactly once".
    Once we identify it, we print it and exit.
    Wait, we need to be sure it is the unique one.
    Is it possible to have false positives?
    My logic:
    If I find $x$ at $p$.
    If I query rest of array for $x$ and get 0 -> $x$ appears nowhere else -> $x$ appears once.
    This is logically sound. Since every number appears at least once.
    So as soon as we find one unique candidate, we are done.
    We only continue searching for other $x$ if we haven't found the unique one yet.
    Wait, how do we ensure we find the unique one?
    The unique one is one of $1 \dots n$.
    Eventually we will query the unique number.
    Will it be detected?
    When we reach the unique number $U$:
    Binary search finds one occurrence at $p_U$.
    Then we query rest of array for $U$.
    Response 0.
    We deduce $U$ is unique.
    We stop.
    
    Worst case query count:
    Suppose the unique number is $n$.
    We process $1 \dots n-1$ (duplicates) then $n$ (unique).
    For each duplicate $x \in 1 \dots n-1$:
      Binary search for first pos: $\approx 9$ queries.
      Query for second pos existence: $\approx 1$ query (size $2n-2$).
      If exists, binary search for second pos: $\approx 9$ queries.
      Total per duplicate: $9 + 1 + 9 = 19$.
    Number of duplicates: $n-1 \approx 300$.
    $300 \times 19 = 5700$.
    This exceeds 5000.
    So we need optimization.

    Where can we cut down?
    Notice: For duplicate $x$, we spend 9 queries to find one instance, and another 9 to find second.
    But we know it appears TWICE.
    Why not find both simultaneously or better?
    Also, do we really need to find the index of the second instance for duplicates?
    If we find ONE instance of $x$, and then query for OTHER instances.
    If we detect it's a duplicate (query returns 1), we know it's duplicate. Do we strictly need the location?
    Maybe just knowing "Duplicate" is enough.
    Then we skip binary search for the second location.
    Saving 9 queries per duplicate.
    New cost per duplicate: $9 + 1 = 10$.
    Total $300 \times 10 = 3000$.
    Plus unique candidate processing (19 or 10): $\approx 10$.
    Total $\approx 3010$.
    This fits comfortably within 5000.
    
    Is it safe to skip finding the second location?
    Yes, because once we find *any* $x$ that turns out to be a duplicate (by seeing it elsewhere), we know it's NOT the unique number.
    The only thing we care about is identifying THE unique number.
    Wait. If we find that $x$ is a duplicate, we just continue to next $y$.
    Eventually we find $U$ which yields 0 when queried on the rest.
    
    Are there edge cases?
    What if the Binary Search fails to find an instance? Impossible, every $x$ exists.
    What if the first BS locates the second instance? Same.
    Wait, does finding an instance imply we know it's at least one? Yes.
    Does finding it elsewhere imply 2?
    If it was unique, it wouldn't be elsewhere.
    If it was duplicate, it would be elsewhere.
    So checking existence in $Rest$ distinguishes Unique vs Dup.
    If Exist(rest) == 0 => Unique. Stop.
    If Exist(rest) == 1 => Duplicate. Continue.
    
    Wait, for the duplicates, do we need to be careful about "first instance"?
    Binary Search returns *some* index in the range $[1, 2n-1]$.
    Once we find index $p$, we exclude it for the rest check.
    If there was another instance, it is in Rest.
    Correct.
    However, finding $p$ requires BS.
    Can we do BS faster?
    Range $2n-1$ size.
    Steps: $\lceil \log_2(599) \rceil = 10$. (Since $2^{10}=1024$).
    So max 10 queries. Average slightly less.
    So $10$ queries to find first.
    $1$ query to check existence in remainder.
    $300 \times (10+1) = 3300$ queries.
    Plus final unique check:
    For unique, BS finds it (10 queries).
    Check remainder: 1 query.
    Returns 0.
    Total $3310$.
    This is safe.
    Actually, the check for existence in remainder can be optimized?
    If we know it exists at $p$.
    The query needs to specify $2n-2$ indices.
    Constructing string of 599 integers is not expensive for printer.
    
    Wait, 5000 queries allowed.
    Time Limit 5 seconds.
    Printing/flushing is handled by cout.
    Input reading $t$. Loop.
    
    One detail:
    In C++, constructing the list $S$ of size $2n-2$ takes time.
    But $n=300 \implies 600$ integers. Fast.
    
    Wait, the BS needs to work on specific ranges of indices?
    BS on indices works fine.
    Subset $S_L$ constructed as $1 \dots mid$.
    Subset $S_R$ as $mid+1 \dots end$.
    This covers disjoint ranges, simplifying things.
    
    Wait, checking existence on $Rest$ requires specifying ALL other indices.
    Indices $1 \dots p-1, p+1 \dots 2n-1$.
    Yes.
    
    One potential issue:
    Is it guaranteed that $a$ contains $1 \dots n$? Yes.
    So finding ANY occurrence is guaranteed.
    
    Refinement:
    Can we optimize the BS?
    The BS takes $\lceil \log_2(Size) \rceil$.
    For size 599, it's 10.
    For size $<2$ (after removing found index), if we needed to find 2nd, it would be small.
    But we decided to skip finding 2nd.
    So cost is $BS(first) + CheckExistence$.
    
    Wait, is it possible to skip the CheckExistence query for dups?
    No, we need to know if it's a duplicate to discard it.
    If we don't query, we assume it's unique and stop?
    No, that would be wrong if current $x$ is duplicate.
    Wait, if $x$ is duplicate, we MUST verify it's not the unique number.
    So we need to know if $x$ is dup or unique.
    
    Wait, do we really need to process $x$ in order $1 \dots n$?
    We just need to find ONE unique.
    We can iterate until we find one.
    Worst case is unique is $n$.
    We process $1 \dots n-1$ as dups.
    Each takes 11 queries.
    $300 \times 11 = 3300$.
    This is safe.
    
    Wait, I said 10 queries for BS. $\lceil \log_2(600) \rceil = 10$.
    Correct.
    Check existence query is 1.
    Total 11.
    $300 \times 11 = 3300$.
    Plus final one 11.
    Total 3311.
    Safe margin.
    
    Wait, what if the interactor is strict about formatting?
    "print a line in the format described above. As a response ... -1 if malformed".
    Format: `? x |S| S1 ... S|S|`.
    Need to flush stdout.
    
    Another thought:
    Can we improve the query cost?
    If we simply query `? x 1 2 ... 2n-1` (all indices). Answer is always 1. Useless.
    We need subsets.
    Is there a way to avoid BS for finding first occurrence?
    Maybe check smaller sets?
    Like, split indices into groups of $K$.
    If I query $x$ on group $G_1$.
    If yes, $x$ is in $G_1$.
    If no, $x$ is in $G_2 \dots$.
    Then recurse on subgroup.
    This is exactly BS.
    
    Implementation details:
    Function `binary_search_first_val(target, range_start, range_end)`
    Returns index `idx` or `0` if not found (should not happen).
    Inside:
      `mid = start + (end - start) / 2`
      Query $x$ on $[start, mid]$.
      If 1: new range $[start, mid]$.
      Else: new range $[mid+1, end]$.
      Base case: range valid? Size $\ge 1$.
      At leaf (size 1), return index.
      Wait, if size 1, query is trivial: `? x 1 mid`. (subset has 1 element).
      If we know $x$ exists in the whole range, and we keep narrowing down, eventually range becomes size 1.
      Then we know it's there.
      Actually, query `? x 1 mid` checks if $x$ is in $[start, mid]$.
      If range size is 1, we check `? x 1 start`.
      Wait, if we narrow down to size 1, do we need to query?
      If previous step returned 1 for a larger range, we know $x$ is in sub-range.
      If we shrink until range size 1, say `[p, p]`, then we know $x$ is at $p$.
      Is this rigorous?
      Yes. Because $x$ exists in `[start, end]`. We verify existence in left/right.
      If we are at `[p, p]`, and we know $x$ is in `[p, p]`, then $x$ is at $p$.
      So we don't need a query for the last step if logic holds.
      However, the BS logic relies on "If query(mid) returns 1, then in Left".
      But we need to handle the base case correctly.
      Usually BS on existence returns True/False.
      Here existence is guaranteed in global range.
      We just need to find the spot.
      
      Algorithm for `find_pos(x, l, r)`:
      While `l != r`:
        `mid = l + (r - l) / 2`
        // Construct S = indices from l to mid
        Ask `? x (mid-l+1) l ... mid`
        if ans == 1: `r = mid`
        else: `l = mid + 1`
      Return `l`
      
      Complexity:
      Initially `l=1, r=2n-1`.
      While `l != r`:
        Queries: `log_2(2n-1)` approx 10.
        After loop, `l` is the index.
      This finds `l`.
      Wait, do we need to know $a[l]=x$? Yes.
      And we don't need to verify it's the index with value x?
      Since we assumed $x$ is in `[l, r]` initially.
      Wait, in recursion:
      If `query([l, mid])` is 0, then $x$ is NOT in `[l, mid]`.
      Since we know $x$ is in `[l, r]`, it MUST be in `[mid+1, r]`.
      So setting `l = mid + 1` is correct.
      So yes, this finds the index.
      
    After finding `p = find_pos(x, 1, 2n-1)`.
    Now we want to check if $x$ appears elsewhere.
    Set $S_{rest} = \{1 \dots 2n-1\} \setminus \{p\}$.
    Ask `? x (2n-2) s1 ... s_{2n-2}`.
    If 1: Dup.
    If 0: Unique. Found solution.
    
    Wait, what if `find_pos` doesn't work for unique numbers correctly?
    It will work. The unique number exists at some $k$.
    The BS logic will narrow down to $k$.
    Then we check rest.
    If $x$ is unique, it is at $k$.
    Rest has NO $x$.
    So query returns 0.
    Correct.
    
    Corner case:
    $n=300$. $2n-1 = 599$.
    Wait, `2n-1` indices.
    BS loop condition `l != r`.
    Example: `l=1, r=2`.
    `mid=1`.
    Query range `[1, 1]`.
    If 1: `r=1`. Loop ends. `l=1`. Correct.
    If 0: `l=2`. Loop ends. `l=2`. Correct.
    Works.
    
    Wait, maximum indices printed in query?
    The query asks for $|S|$ integers.
    $S$ can be up to 600 integers.
    The limit on output length isn't stated but typically 512KB or similar limits apply to IO.
    Standard CP problem limits usually allow large outputs per query if necessary.
    Time limit 5s is generous.
    Constructing 600 ints takes microseconds.
    
    Wait, what if $n$ is small?
    Constraints: $n=300$ per test case. $t \le 20$.
    Max queries 5000.
    Wait, is the 5000 limit cumulative or per test case?
    "You may ask up to 5000 queries in each test case."
    So per test case.
    If $t=20$, total queries could be 100,000.
    But 5 seconds is enough for 5000 queries * 20?
    Usually time limit applies to execution of YOUR code.
    Reading/Writing overhead for 100,000 queries might be significant?
    If $t=20$, we might run 20 times our logic.
    Wait, if my solution uses 3300 queries per test case, total is $66000$.
    With 5 seconds.
    Typical CP limits allow $\approx 10^7$ operations per second.
    Interacting with interactor involves I/O which is slower.
    But 60,000 lines of output is usually OK in 5s.
    Usually 5000 queries per test case suggests we can afford $O(n)$ per test case where constant factor is manageable.
    My solution is basically $O(n \log n)$ queries.
    Wait, I'm re-reading "Time limit per test: 5 seconds".
    Does "test" refer to the whole input file or each test case in the batch?
    Usually "per test" in Codeforces context refers to the total runtime of your program for the whole test file.
    If $T=20$, and $N=300$.
    If I submit code that uses 3000 queries, running 20 times is 60000 queries.
    If 5000 is the limit PER test case, maybe the judge checks each independently?
    Usually, if the problem says "up to 5000 queries in each test case", it means the checker resets the counter for each test case.
    However, the time limit is global (5 seconds for all).
    If I perform 60k I/O operations, it might take 2-3 seconds.
    Flushing stdout 60k times might be slow.
    Optimizing I/O is important.
    We should avoid flushing too often if not needed, but flushing after each query is mandatory.
    
    However, 60000 queries might be tight for 5 seconds if I/O is slow.
    Maybe there is an $O(n)$ solution or better query usage?
    100% score requires $\le 500$ queries.
    The prompt mentions: "If you solve it with at most 500 queries, you get 100 points. If you solve it with 5000 queries, you get 0 points."
    So I should aim for lower than 5000 queries.
    My plan uses $\approx 3300$ queries per test case.
    If $T=20$, that's bad.
    Wait, "Scoring: If you solve it with at most 500 queries..."
    Wait, the scoring rules usually apply per test case or overall?
    "Each test contains multiple test cases."
    Usually scoring is per test case or total over all test cases?
    Often it's total over all test cases.
    "If you solve the problem with at most 500 queries" might imply sum of queries over all test cases $\le 500$?
    Or per test case?
    Given $N=300$, solving in 500 queries total over all test cases (sum $N$ maybe bounded?)
    Wait, problem statement says: "The first line of each test case contains a single integer n (n = 300)".
    So every test case has $N=300$.
    So if there are 20 test cases, sum of N is 6000.
    If limit is 500 queries total, that's impossible for $O(\log N)$ approach (needs $> 300$).
    Wait, if it's per test case, then 3300 is too high (aiming for 500).
    The prompt text: "Scoring ... If you solve it with at most 500 queries, you get 100 points. ... up to 5000 queries ... get 0 points".
    If 3300 gets 0 points, I need to find a better solution.
    Is it possible to do it in $O(n)$ queries?
    Wait, $n=300$. We need to find the unique element.
    Can we do it in $\approx n$ queries?
    Yes, $n$ queries.
    How?
    If we iterate $x$ from 1 to $n$.
    Check if $x$ is unique?
    How to check if $x$ is unique quickly?
    We need to find if it appears once or twice.
    This means checking if there are two indices $i, j$ with $a[i]=a[j]=x$.
    If we can check existence efficiently.
    
    Wait, if we partition the array into $n-1$ pairs and one singleton?
    No, positions are arbitrary.
    
    Let's look at the constraints and scoring again.
    Usually in interactive problems, 5000 is the hard limit. 500 is bonus.
    If 3300 is accepted as a valid "solution" (getting partial score or full if total N small?), but for safety I should try to optimize.
    My solution uses $\approx 11$ queries per number.
    Total $300 \times 11 \approx 3300$.
    Is it possible to reduce the "find first pos" cost?
    For the first number processed (likely unique or duplicate), we waste queries.
    But wait, if we process numbers randomly or in parallel?
    Wait, we can't process in parallel for interactive.
    
    Actually, is there a strategy where we don't need to locate the index?
    The query `? x |S| S1...` tells us if $x$ is in $S$.
    Consider the query: `? x 2 1 2`. (Is $x$ at 1 or 2?).
    If 1, it is in $\{1,2\}$.
    If 0, it is outside.
    This confirms presence.
    
    Is there a way to find the unique number WITHOUT finding its position?
    Wait, "find the number appearing exactly once".
    We know the multiset.
    If we could count occurrences of each number, we'd know the unique one.
    Occurrence count of $x$ is either 1 or 2.
    If we can distinguish count=1 from count=2 with few queries.
    For a fixed $x$, how many queries to decide if it's unique?
    We know it exists. So we know count $\ge 1$.
    If we can prove count $\neq 2$.
    If we find it at index $p$.
    And prove it's NOT at any other index.
    This requires checking existence in complement set.
    My method does exactly that.
    
    Can we speed up the check "not at any other index"?
    If we pick $S_{random}$.
    Query `? x |S_random| S_...`
    Expected behavior:
    If $x$ is unique, prob(x in S) = $|S| / (2n-1)$.
    If $x$ is dup, prob(x in S) = $1 - (2n-1-|S|)/(2n-1)^2$? No.
    Prob(x at index $i \in S$ or index $j \in S$).
    Since it's a fixed distribution, this doesn't give probabilistic guarantee easily.
    Deterministic is preferred.
    
    Let's reconsider the cost 11 queries per $x$.
    Can we skip finding the position for dups?
    If $x$ is duplicate, it is at $p1, p2$.
    If we check if $x$ is in $S$ (complement of some set).
    Wait, if we query on ALL indices $1 \dots 2n-1$, we get 1.
    If we query on HALF indices.
    Split indices into $L, R$.
    Query $x$ on $L$.
    Query $x$ on $R$.
    Case Unique:
      Unique in $L$: $Q(L)=1, Q(R)=0$.
      Unique in $R$: $Q(L)=0, Q(R)=1$.
    Case Dup:
      Dup in $L$: $Q(L)=1, Q(R)=0$.
      Dup in $R$: $Q(L)=0, Q(R)=1$.
      Dup split: $Q(L)=1, Q(R)=1$.
    
    So if $Q(L)=0$ and $Q(R)=0$, impossible (exists).
    So $(1,0), (0,1), (1,1)$.
    If we get $(1,1)$, it's definitely Dup.
    If we get $(1,0)$ or $(0,1)$, it could be Unique or Dup (both in $L$ or both in $R$).
    This partitioning doesn't identify unique immediately.
    Unless we split recursively?
    Tree depth 10.
    At each node, we split the range of indices currently considered.
    But we have multiple numbers.
    We can't run tree for all numbers.
    
    Wait, we have 300 numbers.
    If we process $x=1 \dots n$.
    For a specific $x$, if we just want to distinguish Unique vs Dup.
    We can check: "Is $x$ at index $p$?"
    If we find ANY index $p$, then check if "Is $x$ at $Rest$?"
    This is what I did.
    The BS to find $p$ costs $\approx \log(2n-1)$.
    The check costs 1.
    Total $\approx 11$.
    Can we reduce the BS cost?
    BS requires creating $S$.
    If we just query `? x 1 2 ... 599`.
    It returns 1.
    We need to check if it is in a smaller subset.
    Maybe we don't need BS?
    What if we simply pick the index $x$? No, indices are random.
    
    Wait! There is a very clever solution with $O(N)$ queries (approx $N$).
    Strategy:
    For $i = 1 \dots 2n-1$.
    Maintain a set of potential candidates for the unique number.
    Actually, simpler:
    For each index $i$, we want to know $a[i]$.
    But we can't.
    
    Let's re-read carefully. "Hidden array a containing all numbers from 1 to n... appears twice except one".
    This is a variation of the "odd man out" problem but with existence queries.
    
    Wait, what if we just iterate $x$ from $1 \dots n$.
    Instead of BS to find *any* position, can we find the position of $x$ relative to something?
    Actually, the problem is symmetric for $x$.
    
    Wait, if I use the solution $11 \times 300 = 3300$, and if time limit allows it, it is acceptable for passing (as long as it's under 5000).
    However, the example shows a solution getting 100 points for $\le 500$.
    Getting 3300/5000 might yield low points.
    Let's try to optimize.
    
    Is it possible to check existence of $x$ in $Rest$ without BS?
    We need to remove the FOUND index $p$.
    What if we just query on a fixed index $1$?
    `? x 1 1`.
    If yes, $p=1$.
    If no, $p \neq 1$.
    If we assume $p \neq 1$, then we query on $1$.
    Wait, if $p=1$ and we miss it, we think $x$ is not at 1.
    If we don't query, we don't know where it is.
    But if we find $p$, we can remove it.
    Finding $p$ via BS is efficient.
    
    Wait, can we assume $p=1$?
    No.
    But what if we query $x$ on index 1? `? x 1 1`.
    If 1, then $a[1] = x$.
    If 0, then $a[1] \neq x$.
    Then $x$ is at some $k > 1$.
    We need to find $k$.
    We can binary search in range $[2, 2n-1]$.
    Cost is roughly same as BS on $[1, 2n-1]$ (log 599 vs log 598).
    The optimization comes from reducing the search space faster.
    
    Wait, what if we use a randomized subset?
    We can query $x$ on a random subset of indices of size $\approx n$.
    If $x$ is unique, it falls in with prob 0.5.
    If $x$ is dup, it falls in with prob 1 - (prob neither in S).
    If size $S=n$, prob neither is $\frac{1}{C}$.
    If unique falls in, good.
    If dup falls in, good.
    If dup doesn't fall in, we missed it.
    This is probabilistic.
    
    Is there a way to eliminate candidates?
    If we query $x$ on index 1.
    If yes, $a[1]=x$. Then $x$ is at 1.
    If no, $x$ is not at 1.
    If we check $x=1 \dots n$ on index 1.
    If any $x$ gives 1, then $a[1]$ is known.
    Only one $x$ (namely $a[1]$) will return 1.
    Others will return 0.
    So we identify $a[1]$. Let it be $val_1$.
    Then we know $a[1]$ appears twice (unless it is the unique one).
    We need to find where the second occurrence of $val_1$ is (if any).
    If we find it, $val_1$ is duplicate.
    Then we proceed to find $a[2]$.
    Wait, to find $a[2]$, we query $x \in \{1..n\} \setminus \{a[1]\}$ on index 2?
    We know $a[1]$. If we query $a[1]$ on index 2.
    If $a[2] = a[1]$, we find duplicate.
    If not, we check which $x$ matches index 2?
    To find $a[2]$, we can just query $x$ on $\{2\}$ for all $x \notin \{a[1]\}$.
    But that takes $O(n)$ queries to find $a[2]$.
    We have $O(n)$ total queries available?
    If we find $a[1]$ in 1 query (scanning 1..n? No, scanning 1..n takes $n$ queries).
    Then we know $a[1]$.
    Then we need to find $a[2]$.
    We can check if $a[2] == a[1]$. (1 query).
    If yes, find second occurrence of $a[1]$.
    Wait, if $a[2] == a[1]$, then we found BOTH occurrences?
    Not necessarily. $a[1]$ could be the unique one?
    If $a[1]$ is unique, then $a[1]$ is the unique number. Done.
    If $a[1]$ is duplicate, we need to find the other index.
    Wait, if we find $a[1]$ and $a[2]=a[1]$, then $a[1]$ is duplicate.
    Wait, if $a[1]$ is unique, it never repeats. So $a[2]$ cannot be $a[1]$.
    Wait, is it possible $a[1]$ is unique? Yes.
    Is it possible $a[1]$ is dup and $a[2]$ is NOT $a[1]$?
    If $a[1]$ is dup, it appears at two indices. $1$ and some $k$.
    If $k \neq 2$, then $a[2] \neq a[1]$.
    So finding $a[1]$ and $a[2]$ helps.
    Wait, iterating $a[1] \to a[2] \dots$
    Step 1: Scan $x \in 1 \dots n$ on index 1.
    Only one $x$ (the actual value) will return 1.
    This takes $n$ queries?
    Yes, for each $x$, query `? x 1 1`.
    If we do this for all $x$, it takes $n$ queries.
    We identify $v = a[1]$.
    Step 2: Scan $x \in 1 \dots n$ on index 2.
    Identify $u = a[2]$.
    Compare $u$ and $v$.
    If $u=v$, we have a collision.
    Wait, $u=v$ implies $a[1]$ and $a[2]$ are same value.
    This means $a[1]$ appears twice (at 1 and 2).
    So $v$ is a duplicate. We ignore it.
    If $u \neq v$, we have two distinct values identified at indices 1 and 2.
    Repeat.
    We continue this for $i=1 \dots 2n-1$.
    Wait, we only need to identify the values.
    Once we know the full array (values), we can compute frequency map.
    Scanning all $2n-1$ indices.
    For each index $i$, we find $a[i]$ by checking all $x \in 1 \dots n$ against index $i$.
    This takes $n$ queries per index.
    Total $(2n-1)n \approx 300 \times 600 = 180,000$.
    Too slow.
    
    Wait, we can optimize Step 1.
    Instead of checking $1 \dots n$, we can binary search for $a[1]$.
    To find value $V$ at index 1.
    We check `? V 1 1`.
    This takes $\log n$ queries.
    Cost to find $a[1]$: $\log n$.
    Cost to find $a[2]$: $\log n$.
    ...
    Total cost to read array: $(2n-1) \log n \approx 600 \times 8 = 4800$.
    This is dangerously close to 5000.
    However, we only need to find the unique number.
    Can we stop early?
    If we encounter a value that we've seen before, we mark it as seen.
    If we see a value for the first time, store it.
    We need to check if a value appears a third time (impossible) or second time.
    But we need to know the FULL set of values to distinguish "seen 1 time" vs "seen 2 times".
    Wait, if we find the unique number $U$.
    We find $U$ at index $k$.
    If we then find $U$ at index $m \neq k$, we know it's not unique (contradiction).
    If we finish processing all indices and never saw $U$ a second time.
    Wait, we don't process ALL indices.
    Just enough to find $U$'s second copy if it exists.
    
    Actually, if we use the $4800$ strategy (read array), we might find $U$.
    If we find $U$ and later don't find $U$ again, it's unique.
    If we find $U$ again, it's duplicate.
    So we can find the unique number using full array reconstruction if we stop as soon as we are sure?
    How sure are we?
    If we find $U$ at $k$.
    If we scan subsequent indices and don't find $U$, we conclude $U$ is unique.
    Wait, if we find $U$ at $k$, we have seen it 1 time.
    We need to be sure it won't appear later.
    If we process indices in order $1 \dots 2n-1$.
    If we find $U$ at $i$. We record it.
    For any $j > i$, if we find $U$, then $U$ is dup.
    BUT, if we process $U$ (which is unknown) at index $i$, we learn its value.
    But we don't know it's $U$ (the target unique number). It might be a common duplicate.
    Wait, if we learn $a[i]=V$.
    We check $V$'s status (seen count).
    If count of $V$ becomes 2. Then $V$ is duplicate.
    If count of $V$ stays 1 after all indices? Then $V$ is unique.
    BUT, we don't scan ALL indices.
    We stop if we find a value $V$ such that it *must* be the unique one.
    When can we conclude?
    We conclude $V$ is unique ONLY if we have verified it appears exactly once.
    This means checking all other indices for presence of $V$.
    Or checking that all other candidates are eliminated.
    
    Let's go back to the logic: Iterate $x \in 1 \dots n$. Check if $x$ is unique.
    Cost per $x$ was $\approx 11$.
    Total $3300$.
    The bottleneck is BS to find $p$.
    Wait, is there a way to do it in less than $\log n$?
    For checking existence in a subset, $\log n$ is optimal information theoretic bound if we can only query yes/no on subsets.
    However, we can optimize.
    We want to find if $x$ is unique.
    We query $x$ on $S_{small}$?
    If we partition indices into blocks.
    Query $x$ on block 1.
    Query $x$ on block 2.
    ...
    If $x$ is unique, it appears in exactly ONE block.
    If $x$ is duplicate, it appears in 1 block or 2 blocks (or 3? No, max 2).
    Wait, if we use blocks.
    Suppose we divide indices into $n-1$ pairs (blocks of size 2).
    Plus one singleton.
    Say $P_1, P_2, \dots, P_{n-1}$ and $S$.
    Query $x$ on $P_1 \cup P_2 \cup \dots \cup P_k$.
    This merges queries? No.
    We can query $x$ on union of blocks.
    Query on a SINGLE block $P_j = \{u, v\}$.
    Query `? x 2 u v`.
    If 1: $x$ is at $u$ or $v$.
    If 0: $x$ is elsewhere.
    This helps localize $x$.
    
    If $x$ is unique:
    It falls into exactly one block (pair or singleton) that contains its index.
    Query on that block returns 1.
    Query on other blocks returns 0.
    Wait, if we know $x$ is in block $B$.
    Then we can refine?
    Block size is 2.
    We can distinguish unique vs dup?
    If $x$ is dup:
    It occupies 2 positions.
    These positions might be in same block or different blocks.
    Case 1: Both in $B_j$. Query on $B_j$ returns 1.
    Case 2: One in $B_j$, one in $B_k$. Query on $B_j$ returns 1, on $B_k$ returns 1.
    If $x$ is unique:
    Occupies 1 position in $B_m$. Query on $B_m$ returns 1. Other blocks 0.
    
    So, if we find $x$ present in $B_j$ and $B_k$ ($j \neq k$), then $x$ is DUP.
    If we find $x$ present in only one block, it could be Unique OR Dup (both in $B_j$).
    So checking "count of blocks containing $x$" gives us:
    Count $\ge 2 \implies$ Dup.
    Count $= 1 \implies$ Uncertain (Unique or Dup contained in single block).
    
    We can iterate $j=1 \dots n-1$.
    Query $x$ on $B_j$.
    Keep track of how many blocks return 1.
    If at any point count $\ge 2$, $x$ is DUP.
    If we finish loop and count $= 1$, $x$ is U. (Uncertain? No, wait).
    If count $= 1$, we know $x$ is in $B_m$.
    We need to verify if it's Unique or Dup in $B_m$.
    $B_m$ has 2 indices (say $\{u, v\}$).
    If $x$ is in $B_m$, it is at $u$ or $v$.
    Is it possible $x$ is in $B_m$ and ALSO in $B_k$?
    If we found count $= 1$, we checked all other blocks and they returned 0.
    So $x$ is NOT in any other block.
    Thus $x$ is contained solely within $B_m$.
    Since $B_m$ has at most 2 positions, and $x$ is at one or both.
    If $x$ is at both, it's a Dup.
    If $x$ is at one, it's Unique?
    Wait, if $x$ is at both positions in $B_m$, then $x$ appears twice.
    But we said earlier "Unique number appears ONCE".
    "All numbers from 1 to n appear twice except one".
    So a Unique number appears at ONE index globally.
    A Duplicate number appears at TWO indices globally.
    So if $x$ is confined to $B_m$, and we know $B_m$ has size 2.
    If $x$ is at both $u, v \in B_m$, then count of appearances is 2.
    If $x$ is at exactly one $w \in B_m$, then count is 1.
    So we need to resolve this local ambiguity.
    Query `? x 1 u`.
    If 1, $x$ is at $u$.
    Then check `? x 1 v`.
    If 0, unique. If 1, dup.
    So cost:
    For each $x$:
      Iterate $j=1 \dots n-1$. (Size $n-1 \approx 300$).
      Query $x$ on $B_j$.
      If found in $B_j$ and $B_k$, return Dup.
      If found in $B_m$ and no others.
      Resolve locally in $B_m$:
        Query on $\{u\}$. Query on $\{v\}$.
        If count > 1 $\implies$ Dup.
        Else Unique.
    Total queries:
    Most dups will trigger "found in 2 blocks" early?
    Or maybe found in 1 block, then resolved.
    If we just iterate blocks.
    For a Dup:
      It could be in $B_1, B_1$ (both pos in $B_1$). Detected in $B_1$.
        If we detect $B_1$ returns 1. We know it's in $B_1$.
        But we don't know if it's 1 or 2 occurrences in $B_1$ yet.
        If we just check "count of blocks".
        Wait, for Dup in $B_1$ (both pos), it returns 1 for $B_1$. Count=1.
        So we don't distinguish Dup(in $B_1$) vs Unique(in $B_1$) just by block counts.
        We must resolve.
        Resolution cost: 2 queries (check $u, v$).
        Total per $x$: $1 (\text{check } u) + 1 (\text{check } v) + (n-1)$ (loop) = $n+1 \approx 301$.
        $300 \times 301 \approx 90000$. Too many.
        Wait, we don't need to check $u, v$ for ALL blocks?
        For Dup in $B_1$ (both in $B_1$), we need to verify it's Dup.
        But we can just rely on the fact that if it's in $B_1$ and nowhere else, it could be unique or dup.
        BUT, we can't afford $O(n)$ per $x$.
    
    Okay, let's revisit the BS solution ($3300$ queries).
    It seems robust and likely intended given $N=300$ and 5000 limit.
    Why?
    Because 3300 is within 5000.
    If 5000 queries is allowed for 0 points, and we have a solution with 3300, we get partial score (linear).
    Wait, prompt says "If you solve it with at most 500 queries, you get 100 points."
    If I get 3300, I might get near 0.
    But maybe I can optimize.
    Can I check "is $x$ unique" faster?
    What if I skip the BS?
    Instead of finding index $p$ of $x$.
    If I query `? x (size of S) ...`.
    I want to verify that $x$ appears exactly once.
    If I check if $x$ appears in $1..2n-2$ and NOT in $Rest$?
    Basically finding existence.
    Wait, I can use a randomized subset of size $N$ (indices).
    If I query $x$ on random subset $S$ of size $N \approx 300$.
    If $x$ is unique, it appears with prob $0.5$.
    If $x$ is dup, it appears with prob $\approx 0.7$ (roughly).
    If I repeat this for say 20 trials.
    If $x$ passes 20 trials as likely-dup, then it is dup?
    But we need certainty.
    
    Wait, I recall a similar problem where the limit was tighter or looser.
    But actually, with $N=300$ and $TLE=5s$, $3300$ queries might be accepted as the best solution if 500 is too hard.
    Let's implement the $O(N \log N)$ query solution.
    Is there any simple optimization to reduce constant factor?
    The BS for $p$ starts at $[1, 2n-1]$.
    The check for rest is 1 query.
    Actually, we can combine?
    No.
    But wait.
    If $x$ is unique.
    It is at $p$.
    If we query `? x 1 2 ... 2n-1` (all indices). Always 1.
    We need to find if it's unique.
    Actually, notice:
    Unique element has freq 1. Duplication has freq 2.
    This is the "Odd Man Out" problem.
    If we could sum $a[i]$? No.
    But we have existence queries.
    Existence queries are essentially parity?
    No, they are boolean.
    If we could find all positions of $x$, we sum their count.
    
    Wait! There is a property of $x$ if we partition indices into $n-1$ disjoint pairs.
    If we query $x$ on each pair.
    If $x$ is dup, it must be in at least one pair? No, if both pos in same pair.
    If $x$ is unique, it is in exactly one pair (and one pos).
    If $x$ is dup, it is in:
      1 pair (if both pos in it)? No, pair has size 2.
      1 pair (if both pos in it).
      2 pairs (if split).
    Wait, if we partition indices $1 \dots 2n-1$ into $n-1$ pairs and 1 singleton.
    Let pairs be $P_1, \dots, P_{n-1}$. Singleton $S$.
    Query $x$ on each $P_i$.
    If $x$ is Unique:
      It is at exactly one position.
      That position belongs to some $P_k$ or $S$.
      If in $P_k$, query on $P_k$ returns 1. Others 0.
      If in $S$, query on $S$ (singleton) returns 1.
    If $x$ is Dup:
      Positions $u, v$.
      Case 1: $u, v \in P_k$. Query $P_k$ returns 1.
      Case 2: $u \in P_k, v \in P_m$ ($k \neq m$). Query $P_k$ and $P_m$ return 1.
      Case 3: $u \in S, v \in P_k$. Query $S$ and $P_k$ return 1.
      
    So, if we count number of pairs/singletons returning 1.
    Unique: Count 1. (In some block).
    Dup (both in $P_k$): Count 1.
    Dup (split): Count 2.
    So Count $\ge 2$ implies Dup.
    Count 1 implies Unique OR Dup-contained-in-block.
    
    If we encounter "Count 1".
    We have a block $B$ (could be $P_k$ or $S$) such that $x \in B$.
    And we verified $x \notin B'$ for all $B' \neq B$.
    We need to distinguish Unique vs Dup-in-$B$.
    This requires finding occurrences in $B$.
    $B$ has size 2 (or 1 for S).
    If $B=P_k=\{u, v\}$.
    Query $x$ on $\{u\}$.
    If 1, found. Query $\{v\}$.
    If 1, Dup. If 0, Unique.
    So this resolves the ambiguity with 1 extra query (if $B=S$ trivial).
    Wait, checking $P_k$ requires 1 query.
    Wait, if we do this for all blocks, cost is $n-1$ queries.
    If we find one with count $\ge 2$, $x$ is Dup. STOP (for this $x$).
    If we finish loop with count 1. We found the block $B$.
    Verify $B$ has 1 or 2 occurrences.
    Cost to verify: 1 query (checking other element in pair).
    So total cost per $x$: $n$ queries.
    Total for $x=1 \dots n$: $300 \times 300 = 90,000$. Too slow.
    
    Wait, we don't need to check *all* blocks.
    For Dup, we only need to find it *once* having $\ge 2$ blocks.
    For Unique, we need to find it has 1 block.
    But we don't know which $x$ is Unique.
    We can iterate $x$ and check.
    Is there any way to speed up?
    
    Actually, we can just use the BS strategy (3300). It is much faster than 90000.
    The BS is $\log(600) \approx 10$ queries.
    Checking the rest is 1 query.
    So 11 queries per $x$.
    $300 \times 11 = 3300$.
    This is the most efficient deterministic approach I've found.
    Can I reduce it further?
    Notice that if $x$ is dup, it is at $p1, p2$.
    I search for $p1$. Then search for $p2$.
    Wait, if I find $p1$.
    I don't need to search for $p2$ if I can prove it's dup.
    But proving dup requires finding $p2$ (or verifying existence).
    Verifying existence in $Rest$ takes 1 query.
    So if I can verify existence fast.
    
    Wait, is it possible to query `? x (2n-2) s1 ...` efficiently?
    Yes, we generate indices.
    What if we avoid BS for the FIRST occurrence?
    We don't need to find index if we just need to know "Is there another?"
    Wait, to query "Is there another?", we need to exclude the first one.
    So we MUST find the first one.
    
    Is there a way to find index without $\log N$ queries?
    Not with subset queries.
    
    Wait, there's a possibility to optimize the loop.
    Instead of checking all $x \in 1 \dots n$.
    We can check $x$ randomly?
    If we check $x$ randomly, and $x$ is the unique one, we find it in $\approx 1/n$ steps.
    If we find it, we are done (verify it's unique).
    But if we check a dup, we waste time.
    If we check $k$ numbers randomly.
    Prob we hit Unique is $1/n$.
    Exp steps to find: $n$.
    So still proportional to $n$.
    But if we hit it, we save queries?
    No, we pay cost to verify.
    
    Actually, the BS solution is probably the intended "passable" solution.
    If we have to aim for 500, that implies something very smart.
    Like, checking if $x$ is unique using properties of XOR sum?
    No, existence query doesn't give sums.
    
    Wait, the scoring is tricky.
    "If you solve it with at most 500 queries, you get 100 points.
     If you solve it with 5000 queries, you get 0 points."
    "For values in between, the score decreases linearly."
    This means we want to minimize queries.
    $3300$ vs $500$. $3300$ gets low points.
    Can we solve it with $\approx 300$ queries?
    That means 1 query per number.
    Is there a query that returns "freq mod 2"?
    No.
    
    However, notice that $N$ is up to 300.
    Total elements $2N-1 = 599$.
    We need to find one number.
    Consider the XOR sum of all indices?
    Wait, if we knew the array, XORing all values gives unique.
    But we don't know array.
    Wait!
    The query format: `? x |S| S...`.
    What if we query `? x 1 2 ... (2n-1)` (whole array)?
    Answer is always 1.
    But what if we query `? x S` where S is a subset of indices?
    Answer is 1 iff $\exists i \in S, a_i=x$.
    
    Wait, let's revisit the 300 queries limit.
    Maybe we don't need to check every number.
    We just need to find ONE.
    But we don't know which.
    
    Let's check the constraints again.
    5000 queries limit.
    Scoring 500 queries for 100 points.
    This implies there is a solution around 300-500 queries.
    If my solution is 3300, I should try to beat it.
    
    Is there an algorithm like:
    Iterate $i=1 \dots 2n-1$.
    Assume $a[i]$ is the unique one.
    To verify this, we must show that no other index $j$ has $a[j] = a[i]$.
    If we check all $j$, cost $O(n)$.
    If we check efficiently...
    Wait, if we pick $i$, we need to check if $a[i]$ is unique.
    We can check if $a[i]$ appears elsewhere.
    We can binary search for *another* occurrence.
    Wait, if we find *one* occurrence $p$.
    Then we search for another.
    But we don't know $a[i]$.
    
    Wait, I realized something.
    If we pick an index $k$.
    We want to know if $a[k]$ is unique.
    This value is unique if no other index maps to it.
    If we iterate $k$ from $1 \dots 2n-1$.
    Pick $k=1$.
    Value $V = a[1]$.
    Check if $V$ appears at any $j \neq 1$.
    If we find $V$ at $j$, then $V$ is dup.
    Then $a[1]$ is NOT the unique one.
    So we proceed to $k=2$.
    But to do this, we need to find $V=a[1]$.
    We don't know $a[1]$.
    Wait, if we query `? x 1 1` for all $x$.
    We find $a[1]$ in $n$ queries.
    Then we check existence of $a[1]$ in rest.
    This is $n + 1$ queries to check if $a[1]$ is unique.
    If not, check $a[2]$.
    If $a[1]$ was dup, we wasted $n+1$ queries.
    If $a[1]$ is unique, we spent $n+1$ queries.
    Total $O(n^2)$. Bad.
    
    What if we use the fact that there is ONLY ONE unique element.
    Suppose we maintain a pool of possible unique numbers.
    Initially $\{1, \dots, n\}$.
    If we find a number $x$ that is a duplicate, we remove $x$ from pool.
    If we find $x$ is unique, we are done.
    How to check if $x$ is duplicate efficiently?
    We check if it appears at 2 places.
    Using my previous method: $11$ queries.
    Wait, $3300$ is derived from $300 \times 11$.
    Can we reduce the queries for checking duplicates?
    If $x$ is a duplicate.
    We find one pos $p$.
    We check rest.
    If we find another pos $q$.
    Do we need to find $q$?
    If we find $q$, we know $x$ is dup.
    If we find $q$, we spent $\approx 9$ queries (BS).
    Is there a way to detect duplication without finding the second position?
    Yes, the "check existence" query (on $Rest$) returns 1.
    That is sufficient to say "It is a duplicate".
    So we DO NOT need to find the second position.
    Cost is $9$ (BS) $+ 1$ (Check) $= 10$ queries.
    Wait, is BS cost 9?
    $2n-1 = 599$.
    $\lceil \log_2 599 \rceil = 10$.
    So 10 queries.
    Total per number: 11.
    This is what I calculated.
    Wait, can we optimize BS?
    For a fixed range size $M$, BS takes $\lceil \log_2 M \rceil$.
    If we just check existence, we don't need BS.
    But to exclude the found position, we need its location.
    So we need BS.
    
    Wait! Is it possible to perform BS on the VALUE domain instead of index domain?
    No, we need index domain to locate.
    
    Wait, is there any constraint I missed?
    "You can ask queries...".
    Maybe I can query sets of size 2?
    
    Let's check the solution limit. 5000 queries.
    The example output says: "? 187 1 1".
    Wait, if I query index 1 with value 187.
    It answers 0.
    Wait, if $a$ contains $1 \dots n$.
    $n=300$.
    Is it possible to assume $a[i]$ is small/large?
    No.
    
    Let's verify the 3300 queries calculation.
    $N=300$. Duplicates = 299. Unique = 1.
    Worst case: We process all 299 dups first.
    Cost per dup:
      BS for one position: $\lceil \log_2 599 \rceil = 10$.
      Check existence in rest: 1.
      If 1 (Dup) -> found. Done.
      If 0 (Unique) -> found. Stop.
    So max cost: $299 \times 11 = 3289$.
    Then for the last one (unique):
      BS: 10.
      Check rest: 1.
      If 0 -> Unique. Stop.
    Total $\approx 3300$.
    Plus parsing/flushing overhead.
    With 5000 limit, it fits.
    The "500 queries" scoring is aggressive.
    Maybe we can use the BS more efficiently.
    Actually, notice that $a[i]$ for $i$ is one of $1 \dots n$.
    Maybe we can check $x$ against a random subset of size 1?
    No.
    
    Wait, what if we just iterate $x$ from 1 to $n$.
    We perform the check.
    If we find it's dup, great.
    If we find it's unique, we output and terminate.
    This works.
    The risk is the BS taking 10 queries.
    Is it possible to perform BS faster?
    Maybe we can reuse info?
    No, queries are specific to $x$.
    
    Wait, I found a resource for this problem. It seems to be Codeforces Problem 1515F (maybe?).
    Wait, searching for "interactive problem hidden array n=300 5000 queries".
    It matches Codeforces Round #??? "Interactive Grid" or something?
    No, it matches **Codeforces Problem 1270F**? No.
    It matches **Codeforces 1530B**?
    Wait, the problem statement text: "This is an interactive problem. Hidden array a ... numbers 1 to n ... appears twice except one".
    This is Codeforces **Problem 1677F**? No.
    Actually, it is **Codeforces 1760F**?
    Wait, it's **Codeforces 1526E**?
    
    Found it. It's **Codeforces 1500**ish?
    Actually, it is **Codeforces 1612D**?
    Wait, no.
    Let's ignore external references and trust my analysis.
    3300 queries is definitely safe for 5000 limit.
    However, the problem might be solvable with $O(N)$ queries (e.g. $\approx 300$).
    How?
    If we treat indices as nodes in a graph?
    Wait, if we can group indices.
    Partition indices into pairs $(1, 2), (3, 4), \dots$.
    Query $x$ on $\{1, 2\}$.
    If 1, $x$ is at 1 or 2.
    If 0, $x$ is not at 1 or 2.
    If we iterate $x=1 \dots n$.
    For each $x$, we check all pairs.
    If we find it in $\ge 2$ pairs -> Dup.
    If we find it in exactly 1 pair (or single pair $S$) -> Uncertain.
    Wait, if $x$ is in $P_1$ and $P_2$ (split), count=2 -> Dup.
    If $x$ is in $P_1$ (both in $P_1$), count=1.
    If $x$ is Unique, it is in some pair (say $P_1$), count=1.
    So both Dup-contained-and-Unique look same!
    Wait, Dup contained in pair means it is at both 1 and 2.
    Unique means it is at 1 (or 2).
    They both occupy 1 pair.
    The difference is number of occurences IN that pair.
    If $x$ is at 1 and 2.
    We can distinguish by querying $x$ on $\{1\}$ and $\{2\}$.
    If $x$ on $\{1\}$ is 1 and $x$ on $\{2\}$ is 1 $\implies$ Dup.
    If $x$ on $\{1\}$ is 1 and $x$ on $\{2\}$ is 0 $\implies$ Unique (assuming not elsewhere).
    So if we find $x$ in exactly 1 pair.
    We need to check inside that pair.
    Cost to check inside: 1 query (if we check one element, if 0, then maybe 1? No, we don't know where).
    Wait, if we query $\{1\}$ and get 1. Then check $\{2\}$. If 0, then Unique. If 1, then Dup.
    Cost 2 queries to resolve.
    If $x$ is in $\ge 2$ pairs $\implies$ Dup.
    So algorithm:
    Divide indices into $n-1$ pairs and 1 singleton $S$.
    Query $x$ on all pairs.
    Count how many pairs return 1.
    If count $\ge 2$, $x$ is Dup.
    If count $= 1$, let the pair be $P_k$.
      Check $x$ on $P_k$.
      Wait, we already know $x$ is in $P_k$.
      We need to know if it's at 1st or 2nd index.
      If $x$ is at 1st index of $P_k$, query $\{1\}$.
      Wait, we can just query $x$ on $\{u\}$ where $u \in P_k$.
      If 1: check $\{v\}$. If 1 -> Dup. If 0 -> Unique?
        Wait, if $x$ is Unique, it is at $u$ and NOT $v$.
        So query $\{v\}$ returns 0.
        So if query $\{u\}$ is 1 and $\{v\}$ is 0 -> Unique.
        Wait, this assumes $x$ is NOT in other pairs. We checked that.
    So cost per $x$:
      Check pairs: $n-1$ queries.
      Check inside: 2 queries.
      Total $n+1 \approx 301$.
    Total cost for $n$ numbers: $n(n+1) \approx 90,000$. Too slow.
    
    Wait, we don't need to check all pairs for all numbers.
    Most numbers will return 0 for most pairs.
    Wait, we only need to check pairs sequentially.
    If we encounter count $\ge 2$, we break early.
    But Unique numbers will traverse all pairs.
    So worst case $O(n^2)$.
    
    Is there any way to check if $x$ is dup faster?
    What if we use the binary search approach but with a twist?
    We check for $x$ in $S_{left}$ and $S_{right}$ simultaneously?
    No.
    
    Let's stick to the $O(N \log N)$ approach. It is robust.
    The implementation details matter.
    Max queries 3311. Limit 5000.
    This is sufficient to get "non-zero" score.
    Wait, if I want to maximize score, I should try to get closer to 500.
    Is it possible to check if $x$ is unique using 1 query?
    If I construct a subset $S_x$ such that $|S_x|$ is small?
    No.
    
    Wait, there's a solution with $O(N)$ queries if we can identify the unique element directly.
    But we need to distinguish.
    The problem is equivalent to: "Given a multiset where items appear twice except one appearing once".
    We can access items via queries.
    If we query `? x (2n-1) 1 ... (2n-1)`.
    This doesn't give us count.
    
    Wait! There is a trick.
    We want to find $x$ such that count($x$) = 1.
    If we query $x$ on indices $S$.
    This doesn't give count.
    
    Let's assume the solution with 3300 queries is acceptable as "correct" implementation.
    The code should be clean.
    The main function:
    Read $t$. Loop.
    Read $n$.
    Loop $x$ from 1 to $n$.
    Binary search to find pos.
    Check rest.
    If unique, print and return.
    
    Wait, if I iterate $x=1..n$.
    The BS starts at 1.
    Range $[1, 2n-1]$.
    This works.
    One detail: `2n-1` is the number of indices.
    The query format requires `|S|` and then `S` elements.
    Generating `S` dynamically.
    
    I will implement this.
    I'll write helper functions to generate strings/lists.
    
    One edge case:
    If $n$ is small?
    Constraints $n=300$. Fixed.
    $t \le 20$.
    
    Code structure:
    ```cpp
    #include <iostream>
    #include <vector>
    #include <numeric>
    #include <sstream>
    
    using namespace std;
    
    void solve() {
        int n;
        cin >> n;
        int num_indices = 2 * n - 1;
        vector<int> unique_candidate;
        unique_candidate.reserve(n);
        
        for (int x = 1; x <= n; ++x) {
            // Find first occurrence
            // BS on range [1, num_indices]
            int l = 1, r = num_indices;
            int found_idx = -1;
            
            while (l < r) {
                int mid = l + (r - l) / 2;
                // Construct subset S = [l, mid]
                // Check size
                int size_S = mid - l + 1;
                cout << "? " << x << " " << size_S;
                for (int i = l; i <= mid; ++i) {
                    cout << " " << i;
                }
                cout << endl;
                cout.flush();
                
                int res;
                cin >> res;
                if (res == -1) return; // Error
                
                if (res == 1) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }
            // After loop l is the index
            found_idx = l;
            
            // Check if x appears in Rest
            // Rest is 1..num_indices excluding found_idx
            vector<int> rest;
            rest.reserve(num_indices - 1);
            for (int i = 1; i <= num_indices; ++i) {
                if (i != found_idx) rest.push_back(i);
            }
            
            cout << "? " << x << " " << rest.size();
            for (int idx : rest) cout << " " << idx;
            cout << endl;
            cout.flush();
            
            int res_rest;
            cin >> res_rest;
            if (res_rest == -1) return;
            
            if (res_rest == 0) {
                // Unique
                cout << "! " << x << endl;
                cout.flush();
                return;
            } else {
                // Dup, continue
            }
        }
    }
    
    int main() {
        ios_base::sync_with_stdio(false);
        cin.tie(NULL);
        int t;
        cin >> t;
        while(t--) {
            solve();
        }
        return 0;
    }
    ```
    Wait, I should verify the `ios_base::sync_with_stdio(false)` with interactive problem.
    Generally `cout` flushing is critical.
    `cin` buffering might be an issue.
    Usually it is safer to use `endl` or manual flush.
    `ios_base::sync_with_stdio(false)` disables sync. Mixing `cout` and `cin` is fine if flushed.
    However, `cin` waits.
    Usually in interactive, we use `std::cout` and `std::cin`.
    Disable sync is generally recommended for speed but can cause issues with interactive protocols if not flushed properly.
    I will keep it enabled or just use standard stream.
    Better to NOT disable sync for interactive problems to avoid buffering subtleties?
    Actually, disabling sync makes I/O faster but ensures buffering behavior follows C++ standards.
    With `cout << ... << endl;` it flushes anyway.
    With `cout.flush()` it's explicit.
    I'll use `cout << ... << endl;` which is safe.
    The loop reads `res`.
    `cin >> res` skips whitespace.
    
    Wait, `l` calculation.
    Range `[l, r]` indices.
    `mid` splits into `[l, mid]` and `[mid+1, r]`.
    If `res == 1` (in left), `r = mid`.
    If `res == 0` (not in left), `l = mid + 1`.
    Loop terminates when `l == r`.
    Then `l` is the index.
    This is correct.
    
    Double check the query syntax.
    `? x |S| S1 ... S|S|`
    My code prints `? x size_s elem1 ... elmsize`.
    Matches format.
    Spaces between tokens.
    `cout << " " << i`.
    Correct.
    
    Performance Check:
    300 * 11 = 3300 queries.
    Total output volume per query:
    `? x size ...`
    Indices are up to 600.
    String length per query $\approx 10 \times 600 \approx 6000$ chars?
    Actually, average index length 3 digits. Spaces 1 char.
    $600 \times 4 \approx 2400$ bytes.
    Total bytes $\approx 3300 \times 2400 \approx 8$ MB.
    Within time limit.
    Wait, constructing vector `rest` of size 600 every time.
    This adds to memory/time overhead.
    Optimization: Instead of storing `rest`, we can just print it.
    Printing in loop is better.
    Wait, `rest.size()` is needed.
    We can calculate `rest.size()` before loop.
    But we need to know `found_idx` to construct loop.
    Wait, if `x` is unique, we find `found_idx`.
    If `x` is dup, we find `found_idx`.
    Wait, is it possible `found_idx` changes?
    No, for a fixed `x`, BS converges to a specific index.
    But wait!
    If `x` appears twice at `p1` and `p2`.
    BS might find `p1` or `p2`.
    If BS finds `p1`. Rest excludes `p1`. `p2` is in Rest.
    Query on Rest returns 1. (Because `p2` is in Rest).
    If BS finds `p2`. Rest excludes `p2`. `p1` is in Rest.
    Query on Rest returns 1.
    So finding ANY index is fine.
    
    The logic holds.
    
    Wait, what if $2n-1$ indices.
    `n` is up to 300.
    `2n-1` is 599.
    Vector allocation 599 ints is tiny.
    Doing it 300 times is fine.
    
    One small thing: `ios_base::sync_with_stdio(false); cin.tie(NULL);`
    In interactive problems, mixing `scanf`/`printf` with streams is bad.
    I am using streams.
    Disabling sync is fine as long as we don't mix C-style IO.
    And it speeds up `cout`.
    
    Wait, what if the query returns -1?
    "exit immediately".
    Added.
    
    Is the BS guaranteed to converge?
    Yes, existence is guaranteed (every number exists).
    
    Final check on complexity.
    $T=20$.
    Per test case queries $\approx 3300$.
    Total queries $66000$.
    Wait, "Time limit per test: 5 seconds".
    This usually means total execution time.
    66000 queries with heavy output might TLE if output buffer is large.
    However, usually time limit is generous enough.
    If I get TLE, I can optimize by avoiding vector allocations.
    I can print `rest` elements without vector.
    
    Revised printing logic:
    ```cpp
    // After finding found_idx
    cout << "? " << x << " " << (num_indices - 1);
    for (int i = 1; i <= num_indices; ++i) {
        if (i != found_idx) cout << " " << i;
    }
    cout << endl;
    cout.flush();
    ```
    This avoids `vector<int> rest`.
    Saves allocation time.
    
    Wait, is there any ambiguity if `res` returns 1?
    If $x$ is dup and we found `found_idx`.
    If `found_idx` is one of the two positions.
    Then the other position is in `Rest`.
    So `Res` will be 1.
    So we classify as Dup.
    If `x` is unique and `found_idx` is its position.
    No other position in `Rest`.
    So `Res` will be 0.
    So we classify as Unique.
    
    Wait, what if $x$ appears 3 times? Impossible per problem.
    
    Wait, `mid` calculation `l + (r - l) / 2`.
    Integer arithmetic.
    If `l=1, r=2`. `mid=1`.
    Query `[1, 1]`.
    If 1: `r=1`. `l=1`. Loop ends.
    If 0: `l=2`. Loop ends.
    If BS finds 1 (correct).
    If BS finds 2 (impossible if exists in 1, but if 2 exists and 1 doesn't... wait).
    If `x` is at 2.
    Query `[1, 1]` returns 0.
    `l` becomes 2.
    Loop terminates. `found_idx = 2`.
    Correct.
    
    Wait, what if $n$ is small like 1?
    Problem says $n=300$.
    
    Is there any chance `mid` is invalid? No.
    
    The code seems solid.
    I will produce the solution.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

void solve() {
    int n;
    if (!(cin >> n)) return;
    
    int num_indices = 2 * n - 1;
    
    // We iterate through numbers 1 to n.
    // For each number, we try to determine if it is the unique number.
    // To do this, we find one occurrence of the number.
    // Then we check if it appears anywhere else in the array.
    // If it doesn't appear anywhere else, it's the unique number.
    
    for (int x = 1; x <= n; ++x) {
        // Binary Search to find the FIRST occurrence of x in range [1, num_indices]
        int l = 1, r = num_indices;
        int found_idx = -1;
        
        // We know x must appear at least once, so BS will succeed.
        while (l < r) {
            int mid = l + (r - l) / 2;
            // We check if x exists in indices [l, mid]
            int size_subset = mid - l + 1;
            
            cout << "? " << x << " " << size_subset;
            for (int i = l; i <= mid; ++i) {
                cout << " " << i;
            }
            cout << "\n";
            cout.flush();
            
            int res;
            cin >> res;
            if (res == -1) return; // Invalid query
            
            if (res == 1) {
                // x is in the left half [l, mid]
                r = mid;
            } else {
                // x is not in the left half, so it must be in [mid+1, r]
                l = mid + 1;
            }
        }
        found_idx = l;
        
        // Now we know x is at 'found_idx'.
        // Check if x exists anywhere else in the array (excluding found_idx).
        // We construct a subset of all indices except 'found_idx'.
        cout << "? " << x << " " << (num_indices - 1);
        for (int i = 1; i <= num_indices; ++i) {
            if (i != found_idx) {
                cout << " " << i;
            }
        }
        cout << "\n";
        cout.flush();
        
        int res_rest;
        cin >> res_rest;
        if (res_rest == -1) return; // Invalid query
        
        if (res_rest == 0) {
            // x does not appear anywhere else, so it is the unique number.
            cout << "! " << x << "\n";
            cout.flush();
            return; // Found it, exit for this test case
        } else {
            // x appears elsewhere, so it is a duplicate number.
            // Continue to next number.
        }
    }
}

int main() {
    // Optimize I/O operations, though minimal sync disable is needed for interactive
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int t;
    if (cin >> t) {
        while (t--) {
            solve();
        }
    }
    return 0;
}
```
Wait, one small correction.
If `res` is -1, I should return immediately.
My code handles it.
Wait, what if `found_idx` is not updated correctly?
Loop `while(l < r)`.
If `res == 1`, `r = mid`.
Eventually `l == r`.
Since `x` exists in `[1, num_indices]`, there is a valid index.
So `found_idx` will be valid.
One subtle issue: `cout << "? "` puts spaces.
Correct.

Wait, check time limit again.
5000 queries per test case.
My solution uses $\approx 3300$ queries for $n=300$.
Is it possible to exceed 5000 due to `cout`? No, query count is limited by interactor.
Printing overhead is separate.
I'll stick with the logic.
Double check includes. `<vector>`, `<iostream>`, `<algorithm>` used.
`return` inside `main` is fine.
The `return` inside `solve` is fine as it exits the function, loop continues for `t`.
Actually `cin >> n` is read inside `solve`.
If `solve` returns, next iteration reads next `n`.
Wait, `solve` is called `t` times.
Inside `solve`, if we find the unique number, we print `! x` and return.
Next test case starts.
This is correct.
Wait, what if loop finishes without finding unique number?
This should theoretically not happen given problem guarantees one unique number.
If my logic was flawed, loop finishes.
But problem guarantees one unique number.
My logic: Iterate all numbers. One must be unique.
If I encounter it, I return.
So I must return.
If I don't return (e.g. logic error), I will print nothing or crash.
But logic seems sound.
Wait, if I return from `solve`, I print `! x`.
Wait, I should NOT print `! x` in `solve` if `return` happens later?
Wait, `cout << "! " << x` is executed BEFORE return.
Correct.
Then `return;` jumps to end of function.
Next test case processes.
Seems correct.

Final check on the problem statement regarding "exit immediately".
"If the integer you read is -1, it means that the answer to the previous test case was wrong, and you should exit immediately."
My code checks `res == -1` and `return`.
If it's `res_rest == -1`, return.
This is consistent.

One small detail:
"The answer to this query is: does there exist i ∈ S such that a_i = x ?"
My interpretation is correct.

```cpp
/*
    Problem: Interactive Hidden Array
    Solution: 
    We iterate through each number x from 1 to n.
    For each x, we determine if it is the unique number.
    Since every number appears at least once, we first find ANY position of x using Binary Search.
    Binary Search takes ceil(log2(2n-1)) queries.
    Then, we query the existence of x in the set of all indices excluding the found position.
    If x does not exist in the rest, it means x appeared exactly once (since we found it once and no other occurrence exists).
    If x exists in the rest, it appeared at least twice (duplicate).
    The unique number will eventually be identified and we output it.
    Complexity: O(n log n) queries. For n=300, approx 3300 queries.
*/
```
I will include this in the output.
Wait, I should remove comments or keep them minimal?
The instruction says "wrap the code". No restriction on comments.
But I should make sure it compiles.
Headers included.

One tiny thing: `endl` forces flush. `\n` does not.
I used `\n` and `cout.flush()` explicitly.
Better performance.

Wait, inside BS loop, I use `mid` calculation.
`int mid = l + (r - l) / 2;`
If `l=r-1`, `mid=l`.
Left half `[l, l]` size 1.
Right half `[l+1, l]` empty? No, loop condition `l < r` prevents empty right half.
Correct.

The solution seems robust.

```cpp
#include <iostream>
#include <vector>

using namespace std;

void solve() {
    int n;
    if (!(cin >> n)) return;
    
    int num_indices = 2 * n - 1;
    
    // Try to identify the unique number
    for (int x = 1; x <= n; ++x) {
        // Step 1: Find one occurrence of x using Binary Search
        // We know x exists in the array, so we can narrow down to a single index.
        int l = 1, r = num_indices;
        int found_idx = -1;
        
        while (l < r) {
            int mid = l + (r - l) / 2;
            // Check subset S = [l, mid]
            int size_S = mid - l + 1;
            
            cout << "? " << x << " " << size_S;
            for (int i = l; i <= mid; ++i) {
                cout << " " << i;
            }
            cout << "\n";
            cout.flush();
            
            int res;
            cin >> res;
            if (res == -1) return; // Terminate on error
            
            if (res == 1) {
                // x is in [l, mid]
                r = mid;
            } else {
                // x is in [mid + 1, r]
                l = mid + 1;
            }
        }
        found_idx = l;
        
        // Step 2: Check if x appears anywhere else
        // We construct the subset of all indices EXCEPT found_idx
        cout << "? " << x << " " << (num_indices - 1);
        for (int i = 1; i <= num_indices; ++i) {
            if (i != found_idx) {
                cout << " " << i;
            }
        }
        cout << "\n";
        cout.flush();
        
        int res_rest;
        cin >> res_rest;
        if (res_rest == -1) return; // Terminate on error
        
        if (res_rest == 0) {
            // x does not appear elsewhere, so it is the unique number
            cout << "! " << x << "\n";
            cout.flush();
            return;
        }
        // If res_rest == 1, x is a duplicate, continue to next x
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int t;
    if (cin >> t) {
        while (t--) {
            solve();
        }
    }
    return 0;
}
```
This looks complete.

Wait, check `cout << "? " << x << " " << size_S;`.
If `size_S` is 0? Impossible as `l <= mid` always (size >= 1).
Indices are 1-based.
Looks good.
Memory usage is low. Time complexity is dominated by query count and I/O.

One check: "If you make an invalid query, you will get -1".
My code prints `? x size S`.
If `x` is out of range [1, n]? No, `x` iterates 1 to `n`.
If `size` out of range? No.
Indices are within `1..2n-1`.
So query is valid.
The solution is ready.

One minor optimization:
`ios_base::sync_with_stdio(false);`
In interactive problems, `sync_with_stdio(false)` is sometimes discouraged if `cin`/`cout` are mixed with unbuffered reads/writes, but here we are fully C++. It is generally safe and improves speed.
However, some judges might behave weirdly. But standard CP practice is fine.
Wait, `cout.flush()` is essential.

Double check the time limit again. 5 seconds.
With 3300 queries, it's fine.
Wait, `2n-1` is 599.
Max `num_indices` 599.
Loop `for (int x = 1; x <= n; ++x)` runs 300 times.
Inner loop runs $\approx 10$ times.
Outer print loop runs 300 times.
Total operations roughly $300 \times (10 + 300) \approx 90000$.
Negligible.
Complexity is fine.

```cpp
// Final verification of the code logic
// Loop x=1 to n.
// BS finds index.
// Query rest.
// If 0 -> Unique, print and return.
// If 1 -> Dup, loop continues.
// Since there is exactly 1 unique, we must eventually find it.
// We will not print anything else.
// Output ends with '! y'.
// This matches the required format.
```
Ready to output.
</think>

```cpp
#include <iostream>
#include <vector>

using namespace std;

/**
 * Solves a single test case for the interactive problem.
 * The goal is to find the unique number that appears exactly once in the array.
 * 
 * Strategy:
 * We iterate through numbers x from 1 to n. For each number x, we attempt to verify if it is the unique number.
 * We know that every number appears at least once.
 * 1. Find the first occurrence of x. Since we are querying for existence of x in a range, we can perform a binary search
 *    over the indices of the array [1, 2n-1] to pinpoint an index p where a[p] == x. This takes approximately log2(2n-1) queries.
 * 2. After finding one index 'found_idx' where x is present, we check if x appears anywhere else in the array.
 *    We construct a subset S consisting of all indices from 1 to 2n-1 EXCLUDING 'found_idx'.
 *    We query "? x |S| s1 ... s|S|".
 *    - If the response is 0: It implies x does not exist in any index other than 'found_idx'. Since we know x exists (at least once),
 *      this confirms x appears exactly once. We have found the unique number.
 *    - If the response is 1: It implies x exists in at least one index other than 'found_idx'. Since the problem states numbers appear twice except one,
 *      this confirms x appears twice (is a duplicate). We discard it and continue to the next number.
 * 
 * Complexity:
 * Time Complexity: O(n * log(n)) queries per test case.
 * With n=300, 2n-1 = 599. log2(599) ≈ 10 queries per number to find an index. 1 query to check rest. Total ≈ 11 queries per number.
 * For n=300, total queries ≈ 3300, which is well within the 5000 limit.
 */

void solve() {
    int n;
    if (!(cin >> n)) return;
    
    int num_indices = 2 * n - 1;
    
    // Iterate through each number from 1 to n to check if it is the unique one.
    for (int x = 1; x <= n; ++x) {
        // Step 1: Binary Search to find one occurrence of x.
        // We search in the range [1, num_indices].
        // Since x is guaranteed to exist in the array, the search is guaranteed to succeed.
        int l = 1, r = num_indices;
        int found_idx = -1;
        
        while (l < r) {
            int mid = l + (r - l) / 2;
            // Define subset S as indices [l, mid]
            int size_S = mid - l + 1;
            
            cout << "? " << x << " " << size_S;
            for (int i = l; i <= mid; ++i) {
                cout << " " << i;
            }
            cout << "\n";
            cout.flush();
            
            int res;
            cin >> res;
            if (res == -1) return; // Exit immediately on invalid query
            
            if (res == 1) {
                // x is present in the left half [l, mid]
                r = mid;
            } else {
                // x is not present in the left half, so it must be in the right half [mid + 1, r]
                l = mid + 1;
            }
        }
        found_idx = l;
        
        // Step 2: Check if x appears anywhere else in the array (excluding the found index).
        // We form a subset of all indices except 'found_idx'.
        cout << "? " << x << " " << (num_indices - 1);
        for (int i = 1; i <= num_indices; ++i) {
            if (i != found_idx) {
                cout << " " << i;
            }
        }
        cout << "\n";
        cout.flush();
        
        int res_rest;
        cin >> res_rest;
        if (res_rest == -1) return; // Exit immediately on invalid query
        
        if (res_rest == 0) {
            // x does not appear in the rest of the array, meaning it appears exactly once globally.
            // We found the answer.
            cout << "! " << x << "\n";
            cout.flush();
            return; // End this test case
        }
        // If res_rest == 1, x appears twice (it is a duplicate), so we continue to the next x.
    }
}

int main() {
    // Fast I/O is generally useful, ensuring flush for interactive problems
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int t;
    if (cin >> t) {
        while (t--) {
            solve();
        }
    }
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=623)

````
I need to find a number that appears exactly once in an array of length 2n-1, where all numbers from 1 to n appear twice except one.

Key observation: If I query the entire set {1, 2, ..., 2n-1} for some value x, and the answer is YES, then x is the unique element! This is because if x appeared twice, both positions would be in our queried set S, so we'd get "yes" only for the unique element (the duplicate wouldn't both be checked - wait, actually we need to think about this more carefully).

Actually let me reconsider. The hidden array has indices 1 to 2n-1. All numbers 1 to n appear, with one appearing once and the rest appearing twice. So if I query ? x (all positions), I get "yes" iff there exists position i such that a_i = x. 

If x is the unique element, it appears at exactly one position, so querying all positions gives YES.
If x is a duplicate element, it appears at exactly two positions, so querying all positions gives YES.

So this approach doesn't immediately distinguish them.

Let me think differently. The total number of pairs is 599. Each pair shares the same value. So we have n values total, one appears once, n-1 appear twice. Total positions = 2*(n-1) + 1 = 2n-1 ✓

What if I use the positions directly? Let me look at position parity or something similar...

Actually here's a key insight: For any two positions i and j, either they hold the same value (so a_i = a_j), or they hold different values.

Consider this: take any two distinct positions i < j. Query ? a_i {i, j} — i.e., ask if either position i or j contains the value a_i.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        if (cin.peek() == -1 || cin.eof()) {
            return 0;
        }
        
        vector<int> a(n);
        for (int i = 1; i <= n - 1; ++i) {
            a[i] = i;
            a[n - i] = i;
            int p = query(a, 2 * n - 1, i);
            cout << "? " << i << " " << n - i << endl;
        }
        
        int ans = n;
        for (int i = 1; i <= n; ++i) {
            ans ^= i;
        }
        cout << "! " << ans << endl;
    }
    return 0;
}
```
````
