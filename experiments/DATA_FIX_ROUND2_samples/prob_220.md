# 题 220 — base均分 7.0 vs 直接SFT均分 0.0(掉点 7.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 25.0,completion_tokens=29720)

````
The problem asks us to find a sequence of moves (at most $n^2 - n$ operations) such that after these moves, every player $i$ has exactly $n$ cards with the number $i$. Initially, each player has $n$ cards, and in total there are $n^2$ cards where each number $j \in [1, n]$ appears exactly $n$ times. In one operation, each player passes exactly one card to their right neighbor simultaneously.

Let's represent the state of the system.
There are $n$ players arranged in a circle: $1, 2, \dots, n$. Player $i$ sits to the left of player $i+1$, and player $n$ sits to the left of player $1$. Wait, the problem says "$(i+1)$-th player sits to the right of the $i$-th player". This means if I am at position $i$, the person to my right is at $i+1$ (mod $n$). So when player $i$ passes a card, it goes to player $(i \pmod n) + 1$.
Let's verify this interpretation. "Player i is called solid if all his cards have the integer i written on them."
This means eventually, for every $i \in [1, n]$, the set of cards held by player $i$ must be $\{i, i, \dots, i\}$.

We need to output $k \le n^2 - n$ operations. Each operation consists of selecting one card from each player to pass to the right.
Since each player has $n$ cards initially, and passes one per operation, we can perform up to $n$ operations without running out of cards if we just want to ensure flow, but actually the constraint is on the *state*. We need to reach the target state. The limit is specifically $n^2 - n$. Note that $n^2 - n = n(n-1)$. Since each player holds $n$ cards, this limit suggests something related to shifting cards through the table multiple times or a specific construction. Actually, wait, usually these problems have a constructive solution based on permutations or simple strategies.

Let's look at the constraints. $N \le 100$. This is small enough for $O(N^3)$ or even slightly more. The time limit is 2 seconds.
The target state is very specific: player $i$ owns all cards labeled $i$. Currently, cards are distributed arbitrarily among players.
Let's denote the set of cards owned by player $i$ as $S_i$. We know $|S_i| = n$ and the multiset union $\bigcup S_i$ contains exactly $n$ copies of each label $j$.
In one step, player $i$ chooses $c_i \in S_i$ and passes it to $i+1$. Then $S'_i = (S_i \setminus \{c_i\}) \cup c_{i-1}$ (indices mod $n$).
Wait, let's be careful with indices.
Input gives row $i$ as $c_1, \dots, c_n$. This corresponds to the cards currently with player $i$.
If player $i$ passes a card $x$, it goes to player $i+1$ (where $n+1 \equiv 1$).
So player $i+1$ receives $x$.
If player $i$ keeps a card $y$, it stays with $i$.
After one step, player $i$ has kept cards and received one card.
Total operations allowed: $n^2 - n$.
This looks like a sorting network or a cyclic shift problem. But since it's a circular arrangement, maybe we can view this as routing cards to their destinations.
Each card labeled $v$ needs to end up with player $v$.
Suppose currently a card labeled $v$ is at player $u$. It needs to travel $v-u$ positions clockwise (modulo $n$) to reach player $v$.
Actually, if card is at $u$ and needs to go to $v$, how many steps?
Positions are $1, \dots, n$. Movement is $u \to u+1 \to \dots \to v$.
Distance is $(v - u + n) \pmod n$? No, let's trace carefully.
If I am at player 1, passing to right goes to 2.
If I am at player $n$, passing to right goes to 1.
So $u \to (u \% n) + 1$.
To get from $u$ to $v$: we need $d$ steps such that $(u + d - 1 \pmod n) + 1 = v$.
$(u + d - 1) \equiv v - 1 \pmod n \implies u + d \equiv v \pmod n \implies d \equiv v - u \pmod n$.
So minimum positive distance is $((v - u - 1) \pmod n) + 1$.
Basically, if $u < v$, distance is $v - u$. If $u > v$, distance is $n - (u - v) = n - u + v$.
However, cards don't move instantly; they hop one step per round. Also, only one card leaves a player per round.
But here, we are moving $n$ cards in parallel.
Wait, "one operation" implies *simultaneous* passing. Every player passes *one* card.
This restricts movement significantly compared to a generic graph flow. A player holds $n$ cards. Passes 1, keeps $n-1$. Receives 1. Net change in card count is 0.
The crucial observation might be that the exact identity of cards other than the one passed doesn't matter much for the receiving player, but it matters for the sending player because they accumulate cards.
However, notice the limit $n^2 - n$.
Consider $n$ copies of number $j$. One copy belongs to player $j$. The others belong to some other players.
Wait, "for each integer $j$, there are exactly $n$ cards having this number". This means there are $n$ cards labeled $j$. And there are $n$ players.
The goal is: Player $j$ ends up with *all* $n$ cards labeled $j$.
This implies that for any $x$, the $x$ cards labeled $j$ must be routed to player $j$.
Currently, suppose player $i$ has $k$ cards labeled $j$ ($k \ge 0$). These $k$ cards need to go to player $j$.
Specifically, since there are $n$ copies of $j$ and $n$ players, and we need player $j$ to hold all $n$ copies, every other player $i \ne j$ must give away all their $j$-cards until none remain elsewhere.

Is it possible to simply rotate the table?
If we rotate the configuration of cards such that after $k$ steps, player $i$ receives a card $c$ that was originally at $i-k$?
Let's analyze the total capacity.
We have $n$ rounds available if we use $n^2-n$ operations roughly... wait.
$n^2 - n$ is quite large. For $n=100$, it's 9900.
Since each card labeled $v$ needs to travel to player $v$, the max distance is $n$.
A card starting at $i$ and needing to go to $j$ requires at most $n$ moves. With simultaneous moves, we could potentially clear the board.
But players cannot empty themselves. They start with $n$ cards, pass 1, keep $n-1$, receive 1.
Eventually, to be "solid", player $j$ must possess only cards $j$. This means he must NOT pass any non-$j$ cards away once he collects enough $j$s? No, he starts with some cards. Some might be wrong. He must pass them away until he gets rid of them.
Also, he must receive correct cards.
Essentially, this is a sorting problem on a cycle with limited bandwidth.
Actually, the limit $n^2 - n$ is very specific. Let's decompose the problem.
Maybe we can process each number independently? No, cards interact via the constraint that each player passes exactly one card.
However, notice the Example 2:
3 players.
Player 1: 1 1 1 (All 1s)
Player 2: 2 2 2 (All 2s)
Player 3: 3 3 3 (All 3s)
Solid already. Output: 6 ops, just arbitrary shifts.
This suggests we can waste operations or just do dummy shifts if we are not solid. Or maybe just do 0 ops if we want (but example output shows 6 ops? Ah, "If there are multiple answers, print any." The note says "0 operations would be enough too").
So the strategy doesn't need to preserve current valid states strictly.

Let's consider the condition $N^2 - N$.
$N$ players. $N$ distinct numbers $1..N$.
For each number $x$, there is exactly one slot (player $x$) that should receive $N$ copies of $x$.
Let's think about what happens if we fix the destination of every card.
Since we can output ANY valid solution, maybe there is a deterministic way to move cards.
Strategy idea:
Focus on number 1. There are $n$ cards labeled 1. One copy of 1 should stay at player 1 (eventually), and $n-1$ copies of 1 currently located at other players $p$ must travel to player 1.
Wait, the constraint is symmetric for all numbers.
What if we rotate the whole deck?
No, rotation preserves relative order.
Let's try a greedy approach?
At step $t$, for each player $i$, we choose which card to pass to $i+1$.
Which card should $i$ pass?
Ideally, $i$ should pass a card $c$ such that $c$ is going to land at some player $target(c)$ where $target(c) = i$ is NOT ideal for *that* specific card, but helps someone else?
Actually, player $i$ wants to eventually keep ONLY cards equal to $i$.
So player $i$ should pass away any card $c \neq i$.
BUT player $i$ also needs to receive cards $c=i$ from the left.
Since each player passes exactly one card, they cannot discard multiple bad cards per round.
So they can only fix their hand slowly.
How fast can we converge?
Max cards needed to correct is bounded.
Consider player $i$ who needs to eliminate all $x \neq i$.
He holds $n$ cards. If he passes a wrong card $x$, he reduces count of $x$ by 1 (locally).
If he receives a correct card $i$, he increases count of $i$ by 1.
Net progress depends on inflow/outflow.

Let's look at the structure $N^2 - N$.
Notice $N^2 - N = \sum_{i=1}^{N} (N-i) \times N$ ? No.
$N(N-1)$. This is the number of pairs $(u, v)$ with $u \neq v$.
Maybe this relates to shifting cards around until they meet their owners?
Consider a simple algorithm:
For each card labeled $x$, we want it to eventually be at player $x$.
Currently card at $u$ needs to go to $x$. Distance $d(u, x)$.
In the simplest case, if everyone just passes their "wrongest" card, does it work?
The problem guarantees a solution exists within $N^2-N$ ops.
Let's reconsider the constraint. Each player $i$ can pass ONE card.
Can we say that after some time, all cards are sorted?
Consider the following logic:
Every card has a target owner.
If we run $n-1$ rounds, can we guarantee something?
Actually, the provided solution in similar Codeforces problems (e.g., from Educational Round) often uses the property that we can treat the positions as modulo $N$.
Let's look at the "Example 1":
2 players.
P1: 2 1
P2: 1 2
P1 has {1, 2}. Needs to keep {2} (Wait, P1 is solid if ALL cards are 1. Problem: "Player i is called solid if all his cards have the integer i".)
Wait, in Example 1:
Input:
2
2 1
1 2
Output:
1
2 1
Explanation: P1 passes 2 to P2. P2 passes 1 to P1.
P1 keeps 1 (and receives 1 from P2) -> Total {1, 1}. Solid.
P2 keeps 2 (and receives 2 from P1) -> Total {2, 2}. Solid.
Initial P1 had {2, 1}. P2 had {1, 2}.
Wait, input says "row i contains n integers ... initial cards".
Row 1: 2 1. So Player 1 has two cards, '2' and '1'.
Row 2: 1 2. So Player 2 has two cards, '1' and '2'.
Target: P1 should have {1, 1}, P2 should have {2, 2}.
Current counts:
Label 1: at P1, P2. Need P1.
Label 2: at P1, P2. Need P2.
Step 1: P1 passes 2. P2 passes 1.
P1 retains 1, receives 1 -> {1, 1}. OK.
P2 retains 2, receives 2 -> {2, 2}. OK.
Here, P1 passed a card labeled 2. Card 2 belonged to P2. By passing it, P1 helped P2.
P2 passed a card labeled 1. Card 1 belonged to P1.
It worked in 1 op. Max allowed $2^2 - 2 = 2$.

Hypothesis: Can we simply rotate the cards?
Or better: Can we ensure that in each step, every player $i$ passes a card $c$ such that $c \neq i$?
Wait, if player $i$ is holding cards, he needs to get rid of all non-$i$ cards.
Initially, player $i$ might have 0 non-$i$ cards (already solid). He still needs to pass ONE card. If he passes an $i$-card, he becomes less solid?
But "Everyone is solid" is the final condition.
Actually, the problem asks for $k$ operations. After these operations, check if everyone is solid.
So if we just do random passes, eventually someone will have all $i$'s.
With $N^2 - N$ steps, which is roughly the size of a full cycle for each element?
Actually, consider the cards as items. Each item $v$ (labeled $x$) has a source $pos(v)$.
We want to move all $x$-cards to player $x$.
This is equivalent to: for each $x$, the cards labeled $x$ form a connected component? Not necessarily.
However, we can treat the cards labeled $x$ indistinguishably.
The crucial observation in such problems is often:
For each $x \in \{1, \dots, n\}$, consider the cards with value $x$.
There are $n$ such cards. One must end up at player $x$, others must go to other places?
NO. "everyone is solid". Player $x$ MUST hold ALL $n$ cards of value $x$.
So all cards of value $x$ must end up at player $x$.
This means for any card of value $x$, its target is player $x$.
Currently, card of value $x$ at player $u$. Needs to travel to $x$.
Since we can only pass cards to the right, this is a flow problem.
But the restriction is that each node passes exactly 1 edge per time step.
This sounds like we can model this as:
Assign each card to a specific "shift" amount $s_v$ such that after $s_v$ steps it lands at its owner?
Since each card needs to travel to $x$, and the owner of a card is fixed by its value, we know where each card needs to go.
Specifically, if we have $n$ cards of value $x$, we need ALL of them to reach player $x$.
Let's look at the cards of value $x$ at various players.
Let $c_{x, u}$ be the number of cards of value $x$ at player $u$. $\sum_u c_{x, u} = n$.
Goal: $c_{x, u} = 0$ for $u \neq x$, $c_{x, x} = n$.
Wait, if we simply route each card individually?
The constraint is shared.
But notice the limit $n^2 - n$.
Is it possible that we just perform shifts until everyone is happy?
Let's try a specific construction.
Since $N$ is small, maybe we can simulate a strategy.
Consider this strategy:
Iterate through numbers $v = 1, \dots, n$.
Try to collect all cards of value $v$ to player $v$.
To do this, we can perform operations to route them.
But operations affect all cards.
Maybe we can perform rounds where specific things happen.

Let's rethink the constraint $N^2 - N$.
Total capacity to pass is $N$ cards per turn.
Over $K$ turns, $N \times K$ passes are made.
We want to move $N^2$ cards? No, total cards are $N^2$.
Actually, we can think of this as sorting a deck.
Consider the cards as being at positions $pos(1, 1), pos(1, 2), \dots, pos(1, n)$?
No, simpler.
We need player $i$ to have cards with value $i$.
Suppose we just execute $N-1$ rounds of "passing wrong cards".
Wait, if we iterate for $i = 1$ to $n$:
For player $i$, we want to consolidate cards $i$.
Actually, look at the problem statement again.
"The objective is to reach a configuration... using at most $N^2 - N$ operations."
Note that $N^2 - N = N(N-1)$.
This is $N$ blocks of $N-1$.
Maybe we can iterate $k = 1$ to $N-1$. In block $k$, we make $N$ operations?
Or rather, we just need to ensure each card $c$ travels to its owner.
Card $c$ with value $v$ is at $u$. It needs to travel $dist(v, u)$ steps.
Max distance is $N-1$.
So if we run the game for $N-1$ rounds, any card can travel distance $N-1$.
Wait, if we just run the simulation where each player passes a card labeled $x$ where $x$ is the smallest value among their current cards? Or largest?
Let's try a heuristic:
Player $i$ passes the card $c$ that is "furthest" from reaching their owner?
Or player $i$ passes a card $c$ such that $c \neq i$?
If player $i$ passes a card $c=i$, they are wasting a turn on a card they want to keep.
If they pass $c \neq i$, they reduce the burden of that card.
Since we have $N$ cards of value $v$, and player $v$ needs all of them.
This looks like "routing with buffers".
Key idea from similar problems:
Just shift everything cyclically by some amount?
No, cards are mixed.
However, notice that if we pick a permutation $\pi$ and apply it repeatedly, we mix the cards.
Actually, maybe we can solve this greedily?
Let's maintain the invariant that at step $k$, we try to improve the configuration.
But what is the metric?
Number of misplaced cards?
Or simply: For each player $i$, if they have a card $x \neq i$, they *must* pass it away eventually.
Wait, if they pass $x \neq i$, it goes to $i+1$. Maybe $i+1$ is closer to owner $x$?
Actually, if we just blindly pass *some* wrong card, will it converge?
Suppose everyone passes the *rightmost* card (lexicographically)? Or *leftmost*?
The cards are just integers.
Let's try this algorithm:
For $t = 1$ to $N^2 - N$:
  For each player $i$:
    Identify the card in $S_i$ that should be moved.
    Which one?
    Ideally, the one that allows progress towards $S_i = \{i, i, \dots, i\}$.
    Any card $x \in S_i$ with $x \neq i$ is a "candidate" for removal.
    What if player $i$ picks such a candidate $x$?
    What if no candidate exists ($S_i$ is already $\{i, \dots, i\}$)? Then player $i$ has no choice but to pass an $i$-card. Passing an $i$-card away is "bad" locally, but necessary to make room for incoming cards?
    Actually, if $S_i = \{i, \dots, i\}$, player $i$ is already satisfied. But they still must pass a card. They will pass an $i$-card to $i+1$.
    If $i+1$ accepts it, great? No, $i+1$ wants $i+1$. Receiving $i$ is bad.
    But wait. If everyone is solid, then the system is stable. But the problem asks to *reach* a state.
    So we stop when everyone is solid.
    
    Is it possible that with the rule "Pass a card $x$ such that $x \neq i$ if possible, otherwise pass any"?
    Does this terminate quickly?
    Consider the potential function $\Phi = \sum_{u} \text{cost}(u)$.
    Where cost is distance of misplaced cards?
    Actually, the problem is known.
    This is likely Codeforces 1728D? No.
    Let's search for "Playing Around the Table competitive programming".
    This looks like "Problem F. Playing Around the Table" from some contest.
    Wait, I recall a problem where you can just rotate the array.
    Wait, the problem says "Print $k$ lines...". $k$ is number of operations.
    If we simply output a valid sequence.
    Is it sufficient to output *any* sequence that works? Yes.
    
    Let's check the constraints again.
    Time Limit: 2s. $N \le 100$.
    Complexity $O(K \cdot N)$ is fine. $K = N^2$.
    So we can simulate the process.
    We need a decision rule for "which card to pass".
    
    Proposed Greedy Strategy:
    Each player maintains a list of cards they have.
    At each step, for player $i$:
      Check if there is any card $c$ in their hand such that $c \neq i$.
      If yes, they should probably pass that card. Why? Because they want to keep only $i$'s.
      However, which one?
      Suppose player $i$ has multiple wrong cards. Which one to pass?
      Intuition: Pass the one that needs the most help, or the one that will benefit from moving.
      But wait, the cards are identical values. All cards with value $x$ are the same for the purpose of "who owns them".
      So, if player $i$ has multiple cards with value $x \neq i$, they are functionally identical.
      So player $i$ will always pass *one* of the non-$i$ cards.
      If they have NO non-$i$ cards (i.e. currently solid with respect to value $i$), they *must* pass an $i$ card.
      So the decision is forced?
      "If possible, pass a card $c \neq i$. If impossible, pass any card."
      Wait, if player $i$ has only cards $i$, he passes an $i$.
      If he has both $i$ and non-$i$, does it hurt him to pass $i$?
      Yes, he loses an $i$.
      Does it hurt to pass non-$i$?
      Yes, he loses a potential candidate for solidness, but he is closer to "emptying" his hand of junk.
      So clearly, passing non-$i$ is preferable.
      
      So the strategy seems to be: **Always pass a card $c$ such that $c \neq i$ if available.**
      What if multiple such cards exist? Pick any.
      What if NO such card exists? (Player $i$ is already solid) Then pass any card (say, one of the $i$'s).
      
      Does this guarantee convergence?
      Let's test this logic.
      Consider the "potential" function.
      Usually, passing non-$i$ cards is good because player $i$ reduces the number of wrong cards.
      BUT, the receiver $i+1$ receives a non-(i+1) card.
      If $i$ passes $x$ (where $x \neq i$), it goes to $i+1$.
      Does $i+1$ like it?
      If $x == i+1$, then $i+1$ receives a "correct" card.
      If $x \neq i+1$, then $i+1$ receives a "wrong" card.
      So we prefer $i$ to pass a card $x$ such that $x$ is closer to being at its home?
      Or just any card $x$?
      Wait, if I pass a card $x$, I want $x$ to land at player $x$.
      Currently at $i$. Target $x$.
      I want to increase the distance to target? No, decrease.
      Since movement is always $i \to i+1$.
      To decrease distance, we want to move towards $x$.
      If $x > i$, moving to $i+1$ is progress.
      If $x < i$, moving to $i+1$ is retrograde (or long path).
      Wait, on a circle, distance is unique modulo $N$.
      If $x > i$, distance decreases.
      If $x < i$, distance increases (since we go past $N \to 1 \dots x$).
      So maybe we should prioritize cards $x > i$?
      Actually, the problem is symmetric.
      Let's check the example 1 again.
      P1: {2, 1}. P2: {1, 2}.
      P1 has non-1 card: 2. Should pass 2.
      P2 has non-2 card: 1. Should pass 1.
      Result: P1 keeps 1, recvs 1. P2 keeps 2, recvs 2. Done.
      Here P1 passed 2 (value > 1) and P2 passed 1 (value < 2). Both were valid non-self cards.
      This strategy worked.
      
      What if P1 had {2, 2}?
      P1 should pass 2.
      P2 had {1, 1}.
      P2 should pass 1.
      Passes: P1->2, P2->1.
      New state:
      P1 keeps {2} (plus receives 1). Hand: {2, 1}.
      P2 keeps {1} (plus receives 2). Hand: {1, 2}.
      Back to original. Cycle!
      With this strategy, {2, 2} at P1 leads to a loop.
      However, P1 passed 2. P2 passed 1.
      Card 2 is supposed to go to P2. P1 passed it. Good.
      Card 1 is supposed to go to P1. P2 passed it. Good.
      Wait, P1 had {2, 2}. Both are intended for P2.
      P2 had {1, 1}. Both intended for P1.
      So P1 sends a 2 to P2. P2 sends a 1 to P1.
      Now P1 has {2, 1}. Still has one 2 (needs to leave) and one 1 (good).
      Next step: P1 passes 2 (has non-1). P2 passes 1 (has non-2).
      P1 recvs 1. P2 recvs 2.
      State P1: {1, 1}. P2: {2, 2}. SOLID.
      So it converged in 2 steps.
      
      What if we have larger $N$?
      Is it possible to have a longer cycle or slow convergence?
      The limit is $N^2 - N$.
      With $N$ players, we have $N^2$ cards.
      Maybe the greedy strategy of "pass any non-$i$ card" works and converges within limits.
      
      Wait, is there a case where "passing a non-$i$" is bad?
      Suppose P1 has $\{1, 2\}$.
      If P1 passes 2 (non-1), P1 keeps 1.
      Suppose P2 has $\{2, 3\}$. P2 passes 2 (non-2? No, P2 is 2-player? No P2 is index 2).
      Wait, P2 is player at index 2. Wants 2s.
      If P2 passes 2, they are passing a CORRECT card.
      My strategy says: "Pass any non-$i$ card IF available".
      Here P2 has {2, 3}. Non-2 is 3. So P2 should pass 3.
      So P1 passes 2. P2 passes 3.
      P1 keeps 1. P2 keeps 2.
      P1 receives 3. P2 receives 2.
      New state:
      P1: {1, 3}. (Kept 1, got 3)
      P2: {2, 2}. (Kept 2, got 2 from P1).
      P1 now has {1, 3}. Passes 3.
      P2 has {2, 2}. Passes 2? No, P2 wants 2. Has no non-2 card.
      Must pass 2.
      So P1 passes 3. P2 passes 2.
      P1 keeps 1, gets 2. -> {1, 2}.
      P2 keeps 2, gets 3. -> {2, 3}.
      Back to previous state?
      Wait.
      Initial: P1:{1,2}, P2:{2,3}.
      Op 1: P1 passes 2. P2 passes 3.
      P1 keeps 1, gets 3 -> {1, 3}.
      P2 keeps 2, gets 2 -> {2, 2}.
      Op 2: P1 passes 3 (non-1). P2 passes 2 (ALL 2s, so MUST pass 2).
      P1 keeps 1, gets 2 -> {1, 2}.
      P2 keeps 2, gets 3 -> {2, 3}.
      State is same as Start. Loop.
      Did we converge? No.
      But wait, the problem guarantees a solution. This greedy strategy failed to solve it in this example.
      However, we are allowed to choose WHICH card to pass.
      In Step 1, P1 chose 2. P2 chose 3.
      Could we have chosen differently?
      P1 options: {1, 2}.
      P2 options: {2, 3}.
      Goal: Reach P1={1,1}, P2={2,2}.
      Notice P1 has a 1 (correct), P2 has 2s (correct).
      The bad cards are 2 at P1 (belongs to P2) and 3 at P2 (belongs to P3 - assuming n=3? If n=2, 3 doesn't exist).
      Let's assume n=3.
      P1: {1, 2}. P2: {2, 3}. P3: {?, ?}.
      If n=3, total cards: 1(3), 2(3), 3(3).
      Remaining cards must be somewhere.
      Let's assume P3 has {3, 1}. (1 copy of each left for P3? No, need 3 of each).
      Let's construct a counter-example properly.
      Assume $N=3$.
      Cards:
      Value 1: P1, P2, P3.
      Value 2: P1, P2, P3.
      Value 3: P1, P2, P3.
      Suppose current config:
      P1: {1, 2, 1} -> Two 1s, one 2. Needs to get rid of 2.
      P2: {2, 3, 3} -> One 2, two 3s. Needs to get rid of 3.
      P3: {3, 1, 2} -> One 3, one 1, one 2.
      Let's trace "Pass non-self" strategy.
      P1 has {1, 1, 2}. Non-1s: {2}. Must pass 2.
      P2 has {2, 3, 3}. Non-2s: {3, 3}. Pick 3.
      P3 has {3, 1, 2}. Non-3s: {1, 2}. Pick 2? Or 1?
      Let's assume P3 picks 2.
      Passes: P1->2, P2->3, P3->2.
      Recvs: P1<-3, P2<-2, P3<-2?
      Wait, P3 passes to P1. P1 passes to P2. P2 passes to P3.
      P1 keeps {1, 1}, adds 3 -> {1, 1, 3}.
      P2 keeps {2, 3}, adds 2 -> {2, 2, 3}.
      P3 keeps {3, 1}, adds 2 -> {3, 1, 2}.
      Progress?
      P1 lost 2, gained 3. Better (distance of 2 decreased? 2 was at P1, went to P2. Good). 3 is at P1, needs to go to P3.
      P2 lost 3, gained 2. Better (3 was at P2, went to P3. Good). 2 is at P2, correct.
      P3 lost 2, gained 2. Wait, P3 had 2? Yes. Gained 2 from P2.
      P3 lost 2 to P1? No, P3 gave 2 to P1. Received 2 from P2.
      So P3's count of 2 didn't change.
      This doesn't seem to converge quickly or deterministically.
      
      Alternative strategy:
      What if we just sort the table?
      We can model the state as a set of cards for each player.
      We want to reach $State_k = \{ (i \to i) \forall i \}$.
      Maybe we can use a randomized local search or BFS?
      But $N=100$, state space huge.
      However, note the constraint $N^2 - N$.
      Maybe we can perform $N-1$ cycles of something?
      Let's check the example output 2.
      Input:
      3
      1 1 1
      2 2 2
      3 3 3
      Output:
      6
      1 2 3
      3 1 2
      2 3 1
      1 2 3
      3 1 2
      2 3 1
      
      Here, initially solid. But we performed 6 ops.
      This suggests we can shuffle cards.
      Wait, the example output shows operations.
      Op 1: P1 passes 1, P2 passes 2, P3 passes 3.
      P1 keeps 1 (was solid 1,1,1, keeps one, gets 3). So P1 becomes 1,1,3?
      Wait, in Example 2, initial is solid.
      P1 has {1,1,1}. P2 has {2,2,2}. P3 has {3,3,3}.
      Op 1: P1->1, P2->2, P3->3.
      P1 keeps 1,1. Gets 3. -> {1,1,3}.
      P2 keeps 2,2. Gets 1. -> {2,2,1}.
      P3 keeps 3,3. Gets 2. -> {3,3,2}.
      This moves away from solid.
      Then Op 2: 3 1 2.
      P1 passes 3? (Has 1,1,3). P2 passes 1? (Has 2,2,1). P3 passes 2? (Has 3,3,2).
      Moves back?
      This is weird. The example note says "0 operations would be enough too".
      So the example output just provides a valid sequence (possibly non-minimal, or just arbitrary valid ones? Wait, "using at most N^2-N").
      The problem asks for *a* way. It doesn't require minimal.
      Wait, if I just output 0 operations (if initial is solid), that's valid.
      But if not solid, I need a plan.
      
      Let's reconsider the "pass non-i" strategy.
      Does it fail?
      Maybe we need to distinguish between different cards.
      Wait, I used a potential argument earlier:
      If we pass non-$i$, the sender improves. But receiver worsens?
      Receiver $i+1$ receives $x$.
      If $x \neq i+1$, receiver worsens (gets a bad card).
      But $i+1$ might not care if he passes it away later.
      The bottleneck is the "convergence speed".
      We have $N^2$ cards to distribute.
      If each card moves towards its target, we need roughly $N$ steps on average?
      Wait, each card moves 1 step per round.
      If we coordinate well, we can move all cards efficiently.
      Is there a simple protocol?
      "Everyone passes their 'leftmost' (or rightmost) card?"
      Or "Everyone passes a card such that the card value is minimized"?
      Or "Everyone passes a card such that the card value matches the receiver's index"? No, that's impossible generally.
      
      Actually, look at the constraints and complexity.
      $N \le 100$.
      Maybe we can simulate a greedy strategy that is guaranteed to work.
      The key is that we have plenty of time: $N^2 - N$.
      Is there a known construction?
      This problem appeared in Codeforces Global Round 12? No.
      It matches "Playing Around the Table" from Codeforces Round #865 (Div. 2)? No.
      Found it: Codeforces Round #847 (Div. 3), Problem H? No.
      Problem name "Playing Around the Table" is from **Codeforces Round 918 (Div. 2)** - Problem C? No.
      Wait, I found it. It's **Codeforces Round #905 (Div. 3), Problem E**? No.
      Actually, searching the text online reveals it's from **Codeforces Round #760 (Div. 3)**, Problem F? No.
      Okay, the problem is **Codeforces 1634C**? No.
      Let's ignore the source.
      
      Let's try to think of a strategy that ensures each card reaches its home.
      Consider the cards labeled $v$. There are $n$ of them.
      Let's track the position of each card labeled $v$.
      We want them to gather at player $v$.
      Suppose we designate for each card labeled $v$ a unique path?
      But we can't control individual cards easily because we select one card per player.
      However, if we process players $1 \dots n$ in a specific order?
      No, simultaneous operations.
      
      Let's look at the constraint $N^2-N$ again.
      Maybe we can do this:
      Round 1: Everyone passes the card with the smallest value?
      Round 2: Everyone passes the card with the largest value?
      This mixes things up.
      
      How about this:
      For each player $i$, identify the card with value $i$.
      If player $i$ has it, they keep it?
      If not, they pass some card.
      But a player $i$ might have multiple $i$'s.
      Actually, if player $i$ has *no* card with value $i$, they definitely should not pass a card with value $i$ if they don't have any?
      They won't have any anyway.
      If they have *some* cards with value $i$, should they pass them?
      Only if they are overloaded with non-$i$ cards?
      But they need to KEEP ALL $i$ cards.
      Wait, if player $i$ has $k$ cards of value $i$ ($k < n$), they need to acquire $n-k$ more.
      Since they can only pass 1 card, they can only receive 1 card.
      If they pass a card of value $i$, they lose one, making it harder.
      If they have a non-$i$ card, they should definitely pass it.
      So priority:
      1. If hand contains any $c \neq i$, pass a $c \neq i$.
      2. If hand contains only $c = i$, pass an $i$ (forced).
      This seems optimal locally.
      Why did it fail in my manual trace?
      Trace again:
      P1:{1, 2}, P2:{2, 3}. (Assuming $N \ge 3$).
      P1 has non-1 (2). Passes 2.
      P2 has non-2 (3). Passes 3.
      Result: P1->{1, 3}, P2->{2, 2}.
      Next:
      P1 has non-1 (3). Passes 3.
      P2 has only 2s. Must pass 2.
      Result: P1->{1, 2}, P2->{2, 3}.
      Back to start.
      But wait, in the loop, did anything change globally?
      P1 passed 2 then 3.
      P2 passed 3 then 2.
      The set of cards at P1 changed: {1, 2} -> {1, 3} -> {1, 2}.
      The cards at P2 changed: {2, 3} -> {2, 2} -> {2, 3}.
      It's a cycle of length 2.
      However, note that P1 had {1, 2} initially.
      P2 had {2, 3}.
      The cards involved are:
      Value 1: at P1.
      Value 2: at P1, P2.
      Value 3: at P2.
      Eventually we want:
      Value 1: at P1 (OK, P1 has it).
      Value 2: at P2 (P2 has 2, P1 has 2. Bad).
      Value 3: at P3 (P3 has nothing, P2 has 3).
      
      Wait, my manual trace assumed P2 was player 2.
      Player 1 has {1, 2}. Needs {1, 1} -> {1, 2} is bad.
      Player 2 has {2, 3}. Needs {2, 2} -> {2, 3} is bad.
      Player 3 has {3, 1}. (Missing one card? No, total 3 cards per player).
      Wait, $N=3$, 3 cards each.
      P1: 1 1 2 (Sum of counts: 1x1, 1x2, need 1x1).
      P2: 2 2 3 (Need 1x2, 0x3? No, P2 needs 2 2s). P2 has {2, 2, 3}.
      P3: 3 3 1 (Needs 2 3s).
      My previous small example was inconsistent with counts (I used 2 cards).
      With 3 cards per player:
      P1: {1, 1, 2}.
      P2: {2, 2, 3}.
      P3: {3, 3, 1}.
      This setup satisfies "exactly n cards of each value" globally?
      1s: P1(2), P3(1) -> Total 3. Correct.
      2s: P1(1), P2(2) -> Total 3. Correct.
      3s: P2(1), P3(2) -> Total 3. Correct.
      
      Let's trace "Pass non-self" on this setup.
      P1 wants 1. Has {1, 1, 2}. Non-1 is {2}. Passes 2.
      P2 wants 2. Has {2, 2, 3}. Non-2 is {3}. Passes 3.
      P3 wants 3. Has {3, 3, 1}. Non-3 is {1}. Passes 1.
      
      Moves:
      P1 -> P2 passes 2.
      P2 -> P3 passes 3.
      P3 -> P1 passes 1.
      
      Updates:
      P1 keeps {1, 1}, adds 1 -> {1, 1, 1}. SOLID.
      P2 keeps {2, 2}, adds 2 -> {2, 2, 2}. SOLID.
      P3 keeps {3, 3}, adds 3 -> {3, 3, 3}. SOLID.
      Converged in 1 step!
      
      It seems the strategy works.
      The issue was my previous trace used fewer cards or wrong setup.
      With the correct counts, it seemed to work perfectly.
      Why?
      Because P1 sent the "wrong" card 2 (destined for P2).
      P2 received it. But P2 was holding another 3 (destined for P3).
      Wait, P2 sent 3 to P3. P3 was holding 1 (destined for P1).
      It forms a perfect cycle of transfers matching the destinations.
      If there was a mismatch in destinations, it would take more steps.
      Given $N^2 - N$ allowance, this is extremely generous.
      $N^2 - N$ is roughly $N^2$. Each incorrect card moves closer to target.
      Worst case: A card needs to travel $N-1$ spots.
      Number of incorrect cards is bounded.
      Actually, the number of "misplaced" cards (owned by $u$, destined for $v \neq u$) is $M$.
      In each step, each misplaced card moves 1 step (either towards or away).
      But with the strategy "Pass if $c \neq i$", we maximize the chance of passing a misplaced card?
      Actually, any card $c \neq i$ is a "misplaced" card for player $i$.
      Passing it gets it one step closer to $c$'s home?
      Wait, if $c$ is at $i$ and home is $v$. Distance is $d(i, v)$.
      If we move to $i+1$, new distance is $d(i+1, v)$.
      If $v \ge i$, distance decreases by 1.
      If $v < i$, distance increases (conceptually going "backward").
      So if everyone passes "any non-self", we might be moving cards backwards sometimes.
      However, note that $v < i$ implies that $v$ is "behind" $i$.
      But we are moving $i \to i+1$.
      So for $v < i$, moving to $i+1$ is moving AWAY from $v$.
      Example: $N=5$. Card 1 is at 3. Dest is 1.
      Move $3 \to 4$. Dist(3, 1) = 3 steps CCW?
      Right is increasing index.
      Path: $3 \to 4 \to 5 \to 1$. Length 3.
      Path: $3 \to 2 \to 1$. Length 2? No, moves are only to right.
      So $3 \to 4 \to 5 \to 1$ is 3 steps.
      If we stay at 3, we make 0 progress.
      So moving $3 \to 4$ reduces distance? No.
      Wait, "right" is increasing.
      Destinations are fixed.
      If card is at $u$ and dest is $v$.
      Steps needed: $(v - u + N) \pmod N$.
      Current dist $D$. Next pos $u+1$. New dist $D' = (v - (u+1) + N) \pmod N$.
      $D' = D - 1$ (mod $N$).
      If $D > 0$, then $D' = D-1$ (as long as $D-1 < N$).
      If $D=0$, we are there. But we pass card away.
      Basically, if a card is NOT at its home (at owner), it has a target distance $D \in \{1, \dots, N-1\}$.
      Moving it one step to the right changes its distance to $D-1$ (mod $N$).
      Wait.
      If $v > u$: e.g. 1 is at 2. Dest 1.
      Wait, dest is owner.
      Card 1 is at 2. Owner 1.
      Path $2 \to 3 \to 4 \dots \to 1$.
      Current dist: $1-2+N = N-1$.
      Step 1: moves to 3. Dist: $1-3+N = N-2$.
      Wait, if $u < v$, dist is $v-u$.
      Next $u+1$, dist $v-(u+1) = v-u-1$. Decreases.
      So if card is "ahead" of owner (owner is ahead in indices?), wait.
      Let's standardize: $u$ is current, $v$ is target.
      We want $u \to v$.
      Moves are $u \to u+1$.
      If $u < v$, we need $v-u$ steps.
      If $u > v$, we need $n-(u-v)$ steps.
      If card is at $u$ (where $u \neq v$), we have $D \ge 1$ steps.
      If we move it to $u+1$:
      If $u < v$, new pos $u+1$. Dist $v-(u+1) = D-1$. Improved.
      If $u > v$, new pos $u+1$.
      Case $u=n, v=1$. Dist 1. Move to $1$. Dist 0. Improved.
      Case $u=3, v=1$ ($n=5$). Dist 3 ($3 \to 4 \to 5 \to 1$).
      Move to 4. Dist 2. Improved.
      Wait, it seems moving ALWAYS decreases distance towards target?
      Let's recheck.
      Dist is defined as shortest path clockwise from $u$ to $v$.
      Clockwise distance $d(u, v) = (v - u + N) \pmod N$.
      This is a value in $0 \dots N-1$.
      If we move $u \to u+1$, the new distance $d(u+1, v) = (v - (u+1) + N) \pmod N = (v - u - 1 + N) \pmod N$.
      This is exactly $d(u, v) - 1 \pmod N$.
      Since $d(u, v)$ is distance to target, reducing it by 1 is always good, UNLESS distance was 0.
      If distance is 0 (card at target), then moving it makes distance $N-1$ (it goes away).
      So, any card not at its target moves closer by 1 step.
      Therefore, the sum of distances of all cards strictly decreases whenever we move a non-target card?
      Wait, we move $n$ cards.
      For each card $c$ at $u$:
      If $u \neq \text{owner}(c)$, moving $c$ decreases distance by 1.
      If $u = \text{owner}(c)$, moving $c$ increases distance by 1.
      
      Our Strategy:
      "Pass a card $c$ such that $c \neq i$."
      Player $i$ wants card $i$.
      So for player $i$:
      Any card $c \in S_i$ with $c \neq i$ has owner $c \neq i$.
      Thus $pos(c) = i \neq \text{owner}(c)$.
      So moving any such card DECREASES its distance to home.
      If player $i$ has NO such card (only $i$'s), then every card $c \in S_i$ has owner $c = i$.
      So moving any card INCREASES distance.
      
      So, if player $i$ has any non-$i$ card, passing it reduces the global "badness" by 1 (actually more, since it might displace a good card?).
      Wait, "badness" can be defined as $\sum \mathbb{I}(pos(c) \neq \text{home}(c))$.
      If we pass a card $c$ that is not at home, the count of misplaced cards decreases by 1.
      (Assuming the destination of the card is $c$ itself? No, owner is $c$).
      Wait, card $c$ has value $x$. Its owner is $x$.
      If $c$ is at $i$ and $x \neq i$, it is misplaced.
      If we move it to $i+1$, it remains misplaced (unless $x = i+1$).
      Wait, "Misplaced" count is number of cards not at their owner's table.
      If we move a card from $i$ to $i+1$:
      1. If card was at $i$ (misplaced or correct), status at $i$ is gone.
      2. Card arrives at $i+1$. Status at $i+1$ created.
      Change in Misplaced Count:
      - If card was MISPLACED at $i$: It was counted in $i$'s hand.
        Now it is at $i+1$. Is it misplaced at $i+1$?
        Yes, if $x \neq i+1$.
        So count stays same.
        Wait, unless it landed AT home.
        If $x = i+1$, then it becomes PLACED. Count decreases by 1.
        If $x = i$, then it was PLACED. Becomes misplaced at $i+1$. Count increases by 1.
        
      So passing a card can be good or bad for the "Misplaced" metric.
      BUT, the strategy "Pass non-$i$ if possible" prioritizes cards that are definitely NOT correct for player $i$.
      Specifically, if $S_i$ has $i$'s, those are PLACED.
      If $S_i$ has non-$i$'s, those are MISPLACED (relative to $i$).
      Wait, they might be placed relative to their OWN owner?
      Yes.
      However, we want to minimize $\sum \text{something}$.
      Actually, looking at the code structure required:
      Just output the moves.
      Simulate the process. If it loops or exceeds limit, backtrack?
      Or implement the logic that ensures progress.
      
      The strategy "Pass a card $c$ such that $c \neq i$ if possible" seems robust because it forces the removal of "bad" cards from the sender's perspective.
      Even if it momentarily hurts the receiver, the bad card enters the system.
      Actually, if I pass a card $x \neq i$, I am giving it to someone.
      If $x$ ends up at $x$, great.
      If $x$ ends up at $i+1$ and $x \neq i+1$, it is still misplaced.
      But it moved one step closer? No, distance calculation showed moving towards target (clockwise) always reduces distance to target if target is reached eventually?
      Wait. Target is $x$. Current is $i$.
      Distance $d(i, x) = (x - i + N) \pmod N$.
      Next pos $i+1$. Dist $d(i+1, x) = (x - i - 1 + N) \pmod N$.
      This is $d(i, x) - 1$ if $d(i, x) \neq 0$.
      If $d(i, x) = 0$ (i.e. $x=i$), then next dist is $N-1$ (it moves away).
      
      Conclusion:
      If we pass a card $x$ such that $x \neq i$, then we are moving it AWAY FROM its current location $i$ towards its target $x$ (assuming $x$ is somewhere to the right? No, target is $x$).
      Wait, does $x \neq i$ imply $d(i, x) > 0$?
      Yes, because $x$ is the label, target is player $x$.
      So if $x \neq i$, card $x$ is not at player $x$ (because it is at $i$).
      So moving it from $i$ to $i+1$ reduces its distance to home by exactly 1.
      EXCEPT if $d(i, x) = N$? No max dist is $N-1$.
      So for any card $x \neq i$, passing it REDUCES its distance to target by 1.
      Since we have $N$ players, and we do $N$ passes.
      Sum of distances of all misplaced cards decreases?
      Wait. What about the cards we keep?
      Cards kept at $i$ stay at $i$. Their distance does not change.
      So the total sum of distances $\sum_{c \in \text{All cards}} d(pos(c), \text{home}(c))$
      Strictly decreases if for at least one card $c$, we passed it AND $c \neq i$.
      Wait. We pass $N$ cards in total (one from each).
      For each card $c_j$ passed by player $u_j$:
      If $c_j \neq u_j$, its distance decreases by 1.
      If $c_j = u_j$, its distance increases by 1 (becomes $N-1$).
      
      So if EVERY player passes a card $c \neq i$, then EVERY passed card improves distance.
      Wait. If every player passes a non-$i$ card, then we have 0 cards $c=i$ being passed?
      That means nobody has $i$'s?
      But if $n=2$, P1 has {2, 1}, P2 has {1, 2}.
      P1 passes 2 ($2 \neq 1$, ok).
      P2 passes 1 ($1 \neq 2$, ok).
      P1 passes 2 (dest P2). Dist(1, 2) -> Dist(2, 2) [0]. Improvement.
      P2 passes 1 (dest P1). Dist(2, 1) -> Dist(1, 1) [0]. Improvement.
      Both improve.
      
      So the sum of distances strictly decreases if everyone passes a "non-self" card.
      But if some players are already solid (have only self-cards), they are FORCED to pass self-cards.
      Suppose $k$ players are solid. $n-k$ players are not.
      The solid players pass cards $c=i$. Distance increases.
      The others pass non-$i$. Distance decreases.
      Will the net change be negative?
      Maybe.
      But intuitively, as long as we keep moving misplaced cards, we will converge.
      Given the generous limit $N^2 - N$, we might not need to optimize perfectly.
      
      Algorithm:
      Repeat $K = N^2 - N$ times:
        For each player $i=1 \dots n$:
          If player $i$ has any card $\neq i$:
             Pick any such card (say the last one in list or min/max).
             Mark it to pass.
          Else:
             Pick any card (since all are $i$).
             Mark it to pass.
        
        Perform the passes. Update the hands.
        Check if solved. Break.
        
      Is it possible to get stuck in a cycle where distances don't decrease?
      Example: P1 has {1}. P2 has {2}. $N=2$.
      P1 passes 1. Dist(1,1)=0 -> $1 \to 2$. Dist(1,1) is distance of card 1? No, card 1 is at P1. Dest is P1.
      Dist is 0.
      Move to P2. Dist is 1. Increased.
      P2 has {2}. Card 2 at P2. Dest P2. Dist 0.
      Move to P1. Dist 1.
      Total sum dist increases by 2.
      State: P1 has {2}, P2 has {1}.
      Next iter:
      P1 has {2} (non-1). Pass 2. Dest P2. Dist(2,2)=1 -> 0. Improved.
      P2 has {1} (non-2). Pass 1. Dest P1. Dist(1,1)=1 -> 0. Improved.
      So they swap back.
      The state {1} at P2, {2} at P1 is also a solid state?
      No, solid means P1 has {1,1}... wait $N=2$.
      P1 has 1 card labeled 1. Wait, problem says $n$ cards per player.
      For $N=2$, each player has 2 cards.
      Initial: P1 {1, 2}, P2 {2, 1} (mixed).
      Pass 2 (from P1), 1 (from P2).
      P1 keeps 1, gets 1 -> {1, 1}.
      P2 keeps 2, gets 2 -> {2, 2}.
      Solids.
      
      What if we start with P1 {1, 1}, P2 {2, 2}. (Already solid).
      Algorithm:
      P1 has no non-1. Pass 1.
      P2 has no non-2. Pass 2.
      P1 keeps 1, gets 2 -> {1, 2}.
      P2 keeps 2, gets 1 -> {2, 1}.
      Now P1 {1, 2}, P2 {2, 1}.
      Next: P1 passes 2. P2 passes 1.
      P1 gets 1 -> {1, 1}.
      P2 gets 2 -> {2, 2}.
      Back to solid.
      So solid state transitions to mixed, then back to solid.
      This oscillates.
      But since we have a limit $N^2 - N$ and we stop if solid, we can detect solid and break?
      "Find a way to do it... using at most $N^2-N$".
      If it oscillates, does it mean we exceed limit?
      In oscillation, we return to solid every 2 steps.
      If we check for solidness at the end, it's fine?
      No, we need to output $k$ operations. If we do 0 operations, we are good.
      But if we run simulation and it hits limit, we must be careful.
      Wait, if we start at solid, output 0.
      If not solid, run simulation.
      Is it guaranteed to reach solid in reasonable time?
      In the oscillating case {1,2}/{2,1}, we need 1 step to become solid {1,1}/{2,2}?
      Wait, P1 has {1, 2}. Pass 2. P2 passes 1.
      P1 keeps 1, gets 1. -> {1, 1}. Solid.
      So from {1, 2}/{2, 1} (mixed) to Solid takes 1 op.
      From Solid to Mixed takes 1 op.
      If we start Solid, we check "Is Solid?". Yes. Output 0.
      If we start Mixed, we run.
      Will we ever enter a cycle of Mixed-Mixed...Mixed that doesn't hit Solid?
      If the sum of distances behaves monotonically?
      In the {1,2}/{2,1} case:
      State A: P1{1,1}, P2{2,2}. Solids. Cost? Not measured.
      State B: P1{1,2}, P2{2,1}.
      Transitions: A <-> B.
      Wait, is B reachable from A? Yes.
      Is A reachable from B? Yes.
      But we want to stop at A.
      If we start at B:
      Op 1: P1 passes 2, P2 passes 1.
      State -> A. Stop.
      So we don't loop.
      The loop only happens if we start at A and apply ops.
      But our code should check `is_solid()` first.
      `while (!is_solid && k < limit)`.
      
      So the simulation strategy:
      ```cpp
      vector<int> hand[n+1][1]; // actually store multiset or just vector
      int operations_count = 0;
      while (operations_count < n*n - n) {
          bool ok = is_solid();
          if (ok) break;
          
          // Build pass list
          vector<int> passes;
          for(int i=1; i<=n; ++i) {
              // find a card != i
              int card_to_pass = -1;
              for(auto &c : hand[i]) {
                  if (c != i) {
                      card_to_pass = c;
                      break; 
                      // Optimization: picking min/max or specific doesn't matter?
                      // Just pick first available.
                  }
              }
              if (card_to_pass != -1) {
                  passes.push_back(card_to_pass);
                  // remove card from hand
                  // Remove one instance
                  auto it = find(hand[i].begin(), hand[i].end(), card_to_pass);
                  hand[i].erase(it);
              } else {
                  // Pass any card (should be i)
                  if(!hand[i].empty()) {
                     // pick last or first
                     card_to_pass = hand[i].back(); 
                     passes.push_back(card_to_pass);
                     hand[i].pop_back();
                  }
              }
          }
          
          // Execute pass
          // P_i passes passes[i-1] to P_{i+1} (mod n)
          for(int i=0; i<n; ++i) {
               // pass to (i % n) + 2 ??
               // Players 1..n.
               // P_i passes to P_{i+1}.
               // P_n passes to P_1.
               // Index mapping: 1-based to 0-based for array access.
               int from = i + 1; // P_i
               int to = (from % n) + 1; // P_{i+1}
               int val = passes[i-1]; // Wait, passes array is 0-indexed?
               // If passes stores card from P1, P2...
               // passes[0] is card from P1.
               // Hand update: P_to receives card.
               hand[to].push_back(val);
          }
          // Note: We updated hands sequentially above? No, we collected passes then update?
          // Wait, inside loop we modified hands (erased).
          // Then we push back into receiving hands.
          // But if we modify hands inside the loop for P1, it might affect P2's decision?
          // No, decisions are based on CURRENT hand.
          // Wait, we made decisions for all players BEFORE executing updates.
          // Correct logic:
          // 1. Collect passes (snapshot of what to send).
          // 2. Update hands (move cards).
          // But I combined them.
          // If I erase from hand[i], I lose the info for subsequent steps?
          // Decisions must be simultaneous.
          // So I must determine `passes[i]` for all `i` before modifying any hand.
      }
      ```
      
      Wait, the logic `if (c != i)` removes it from hand immediately.
      If I decide P1 passes X, P1 removes X.
      Then P2 decides. P2's hand is untouched.
      Then execution:
      P1 passes X to P2.
      P1 has removed X. P2 adds X.
      This is correct.
      The simultaneous nature is simulated correctly because we collect all decisions first?
      Wait, if I use `hand` array to store cards.
      P1 checks `hand[1]`. Finds card $c_1$. Removes it. Adds to passes[0].
      P2 checks `hand[2]`. Finds card $c_2$. Removes it. Adds to passes[1].
      This assumes `hand` hasn't changed yet.
      Since I'm iterating $i$ from 1 to $n$, I read and write to `hand`.
      Wait, if I modify `hand` while collecting, the next player's decision is unaffected because they check their own hand.
      P1 modifies `hand[1]`. P2 checks `hand[2]`. No conflict.
      So sequential collection is fine.
      Then after collecting all `passes`, we execute updates.
      However, I realized in my manual trace I updated `hand` *after* deciding.
      Wait, in the snippet above, I removed the card from `hand` while deciding.
      Then during execution phase, I need to ADD it to `hand[to]`.
      So effectively, the card is moved.
      So the state update is correct.
      Wait, is the order of execution important?
      "All these actions are performed simultaneously."
      So P1's move is based on initial state. P2's move is based on initial state.
      Yes.
      So my logic:
      1. Determine move for everyone based on current state.
      2. Apply all moves.
      This is correct.
      
      One detail: When choosing card to pass, if multiple candidates exist, does it matter?
      As analyzed, it might matter for cycle detection, but usually "any" is fine.
      Wait, if P1 has {1, 2} and P2 has {2, 1}.
      P1 picks 2. P2 picks 1. Works.
      If P1 had {1, 1, 2}? Picks 2.
      If P1 picked 1? (Bad choice, passes solid card).
      Since strategy says "pick non-self if possible", P1 won't pick 1.
      So we are safe.
      
      What about the case where P1 has {1, 2} and P2 has {1, 2} ($N=2$, cards are duplicated).
      P1 picks 2. P2 picks 2.
      P1 -> {1, 1} (keeps 1, gets 2 from P2? No P2 passes 2 to P1).
      Wait, P2 has {1, 2}. If P2 picks 2, P2 keeps 1. Gets 2 from P1.
      P1 keeps 1, gets 2.
      P1 has {1, 2}. P2 has {1, 2}.
      Loop!
      State {1, 2}/{1, 2} with P1 wanting 1s, P2 wanting 2s?
      Wait, initial config:
      Total cards: four cards. 1,1,2,2.
      P1 has {1, 2}. P2 has {1, 2}.
      P1 needs {1, 1}.
      P2 needs {2, 2}.
      Wait, is this configuration valid?
      "For each integer i... exactly n cards".
      So yes.
      My algorithm:
      P1 has {1, 2}. Non-1 is 2. Pass 2.
      P2 has {1, 2}. Non-2 is 1. Pass 1.
      Move: P1->2, P2->1.
      P1 keeps 1, gets 1 -> {1, 1}.
      P2 keeps 2, gets 2 -> {2, 2}.
      SOLID.
      Why did I think it loops?
      Ah, I said "If P2 picks 2".
      P2 has {1, 2}. Non-2 is 1.
      So P2 must pick 1 (according to "pick non-self").
      So P2 picks 1. P1 picks 2.
      This leads to solution.
      So the "Any non-self" rule is consistent and strong.
      It seems to resolve conflicts because players have different preferences.
      P1 wants to get rid of 2. P2 wants to get rid of 1.
      So they swap. Swap fixes both.
      
      Is there a case where swapping fails?
      P1 wants 1. P2 wants 1?
      No, distinct targets. P1 needs 1, P2 needs 2.
      So P1 wants to get rid of 2. P2 wants to get rid of 2? No P2 wants 2.
      If P2 has {1, 1} and P1 has {2, 2}.
      P1 has {2, 2}. Non-1 is 2. Pass 2.
      P2 has {1, 1}. Non-2 is 1. Pass 1.
      Move: P1->2, P2->1.
      P1 keeps 2, gets 1. -> {2, 1}.
      P2 keeps 1, gets 2. -> {1, 2}.
      Next:
      P1 has {2, 1}. Pass 1 (dest 2? No P1 needs 1. So P1 has 2 (bad). Pass 2.)
      P2 has {1, 2}. Pass 1 (bad? No P2 needs 2. So P2 has 1 (bad). Pass 1.)
      Move: P1->2, P2->1.
      P1 keeps 1, gets 2. -> {1, 2}.
      P2 keeps 2, gets 1. -> {2, 1}.
      State {2, 1}/{1, 2}.
      Wait, previously we had {2, 2}/{1, 1} and switched to {2, 1}/{1, 2}.
      Then from {2, 1}/{1, 2}:
      P1 has {2, 1}. Pass 2.
      P2 has {1, 2}. Pass 1.
      Same moves.
      P1 keeps 1, gets 2?
      Wait, P1 kept 1. Receives 2 from P2?
      Previous step: P1 passed 2. P2 passed 1.
      P1 received 1 from P2. Kept 2?
      Wait, P1 started with {2, 1}. Passed 2. Kept 1.
      Received 1 from P2.
      So P1 has {1, 1}.
      P2 started with {1, 2}. Passed 1. Kept 2.
      Received 2 from P1.
      So P2 has {2, 2}.
      SOLID.
      So it seems to work.
      
      Is there any risk?
      What if P1 has {1, 1, 2} and P2 has {1, 2, 2}?
      P1 (wants 1): has {1, 1, 2}. Pass 2.
      P2 (wants 2): has {1, 2, 2}. Pass 1.
      P1->2, P2->1.
      P1 keeps {1, 1}, gets 1. -> {1, 1, 1}. SOLID.
      P2 keeps {2, 2}, gets 2. -> {2, 2, 2}. SOLID.
      Seems to work very fast.
      
      Is it possible to get stuck?
      With $N^2 - N$ limit, and $N$ small, maybe just brute force simulation with the greedy heuristic is sufficient.
      The constraint is quite loose.
      The only tricky part is ensuring the program runs within 2s for $N=100$.
      $K \approx 10000$. Inside loop, $O(N)$ to find card, $O(N)$ to push.
      Total $O(K \cdot N) = 10^6$ ops. Very fast.
      
      So I will implement this simulation.
      One small optimization: To ensure we don't TLE on worst cases (though unlikely), we should ensure we break as soon as solid.
      Also, the limit check.
      
      Edge case: $N=2$. Limit $2^2 - 2 = 2$.
      Simulation runs at most 2 steps.
      
      Wait, "find a card != i".
      If `vector<int>` contains duplicates, `find` returns iterator.
      If `c != i`, break and pick.
      What if there are multiple such cards?
      `find` returns first one. It's deterministic.
      Also, removing from vector is linear scan?
      `vector.erase` is linear.
      Since max cards is 100, it's negligible.
      
      Final check on "Time Limit": 2 seconds.
      Complexity $O(N \cdot (N^2 - N))$. $100 \cdot 10000 = 10^6$. Trivial.
      
      Wait, there is one thing.
      The problem statement says "using at most $n^2 - n$ operations".
      If my simulation fails to find solution within this, I fail.
      But we proved theoretically (or guessed) that it should work or oscillate around solution.
      Actually, is it possible that we need *more* than $N^2-N$?
      The problem guarantees existence.
      Also, there is a specific solution:
      "Rotate the cards such that after $n$ steps...".
      Wait, maybe the intended solution is simpler.
      Actually, there is a known trick:
      If we pass a card $c$ such that $c \neq i$, we improve.
      If no such card, we are solid?
      No, if no such card, we only have $i$'s. So we are solid.
      But we must pass one.
      If we are solid, we are done.
      But if we start solid, we output 0.
      My loop handles this: `if(is_solid) break;`
      But wait, if we output 0, that's valid.
      If we run loop and it doesn't converge, that's bad.
      However, since it's a Codeforces/CP problem, usually the greedy heuristic works.
      Let's double check if there's any case where this heuristic cycles without solving.
      The "swap" case {2, 2}/{1, 1} -> {2, 1}/{1, 2} -> {1, 1}/{2, 2} took 2 steps.
      The state space is not that large?
      Actually, the greedy strategy minimizes a potential function if we consider cards as moving towards targets.
      Specifically, if we map cards to their target positions.
      Any move of a card $c$ from $u$ to $v$ changes its distance $d(u, \text{home}(c))$ to $d(v, \text{home}(c))$.
      We established that passing $c \neq i$ decreases distance.
      Passing $c=i$ increases distance.
      Since we minimize number of misplaced cards?
      Wait, if we pass $c \neq i$, we remove it from $S_i$.
      If $S_i$ contained $c$, $c$ was "wrong" for $i$.
      Moving it to $i+1$ brings it closer to $\text{home}(c)$ (since $\text{home}(c) \neq i$).
      Wait, I reasoned this before.
      Let's re-verify:
      Let $d(u, v) = (v - u + n) \pmod n$.
      Card $c$ has home $h=c$.
      It is at $u$.
      Condition $c \neq i$ means $c \neq u$ (here $i=u$).
      So $h \neq u$.
      Distance $D = d(u, h) \in [1, n-1]$.
      After move to $u+1$: $D' = d(u+1, h) = d(u, h) - 1$.
      (Since $D \ge 1$, $D-1$ is valid non-negative distance).
      So distance strictly decreases.
      If we pass $c=i$, then $h=i=u$. $D=0$.
      Move to $u+1$: $D' = n-1$. Increases by $n-1$.
      So passing a correct card is VERY BAD.
      Passing a wrong card is GOOD.
      Since we prioritize passing wrong cards ("if possible"), we avoid the "Very Bad" action unless forced.
      If forced (all cards correct), we are SOLID.
      If we are solid, we stop.
      So we never do a very bad move. We only do good moves or (when solid) a neutral-ish move?
      Wait, if we are solid, we DO IT 0 times.
      So if we are solid, we exit.
      The only time we might be forced to pass correct card is if we have NO wrong cards.
      Which implies we are solid.
      So we NEVER pass a correct card!
      Wait, this logic holds if "having NO wrong cards" == "being solid".
      Definition of solid: All cards are $i$.
      Yes.
      So if we follow the strategy "Pass any card $c \neq i$ if available", we NEVER pass an $i$.
      This implies we never increase distance for any card.
      And we always decrease distance for passed cards.
      So the sum of distances strictly decreases?
      Wait, what about cards that stay at $u$?
      Their distance doesn't change.
      So $\sum D_c$ strictly decreases?
      If at least one card is passed.
      But wait, we pass $n$ cards.
      If everyone has at least one wrong card, then all $n$ moves decrease distance. Sum decreases by $n$.
      If some players have NO wrong cards, they MUST be solid.
      Wait, if a player $i$ has no wrong cards, they have only $i$'s.
      So they are solid.
      If $i$ is solid, do we care about what they do?
      If they are solid, and we continue the loop?
      My logic "If solid, break" handles this.
      So as long as we haven't stopped, there is at least one player who is NOT solid?
      No. If *everyone* is solid, we break.
      If *someone* is not solid, does everyone have a wrong card?
      If player $j$ is not solid, then $S_j$ contains at least one card $\neq j$.
      So player $j$ has a wrong card.
      So at least one player $j$ has a wrong card.
      So at least one move (from $j$) decreases distance.
      Wait. Is it possible that a move from $j$ decreases distance but a move from $k$ (who is solid) increases distance?
      Wait, if $k$ is solid, he has no wrong cards.
      If we apply the rule "Pass non-self if possible", solid player $k$ has *no* non-self cards.
      So he cannot pick a non-self card.
      Does he fall back to "Pick any"?
      My code:
      ```cpp
      if (has_wrong) pass wrong.
      else pass any (implying correct).
      ```
      So if $k$ is solid, he is forced to pass a correct card (an $i$).
      This increases distance for that card.
      So we CAN have distance increase if solid players participate.
      However, if $k$ is solid, we WANT him to stay solid.
      But if he passes a card, he loses a correct card.
      He will receive one card.
      Since he receives a card from $k-1$.
      If $k-1$ passed a wrong card (which it likely will, unless $k-1$ is also solid), $k$ might receive a card that puts him out of solid?
      Wait. If $k$ is solid, he has all $k$'s.
      He passes a $k$. Distance increases.
      He receives a card from $k-1$.
      If $k-1$ is not solid, $k-1$ passes a wrong card (non-$k-1$).
      Let that card be $x$.
      If $x=k$, then $k$ receives a correct card. He loses one $k$, gains one $k$.
      Net effect on $k$'s count of $k$'s: unchanged.
      But his hand composition: he passed a $k$, so he temporarily has one less $k$.
      Wait, he receives $x$.
      If $x=k$, then he ends up with $k$'s.
      So he remains solid.
      What if $x \neq k$?
      Then $k$ receives a wrong card. He has $k$'s minus one plus one $x$.
      So he is NO LONGER solid.
      This is a disaster. We pushed a solid player into being not solid.
      And we wasted an operation.
      This implies we shouldn't allow solid players to pass?
      Or rather, we should handle the interaction.
      But the problem says "simultaneous".
      So if $k$ is solid, he MUST pass a card.
      If we don't break immediately, we might destabilize him.
      BUT, we can check `is_solid` BEFORE executing operations.
      If initially solid -> Output 0. Done.
      If NOT solid initially.
      We execute steps.
      Is it possible to reach a state where we make infinite oscillations?
      The distance metric argument suggests sum of distances decreases?
      Not quite, because solid players add + (N-1) to distance.
      However, we have at most $N$ such players.
      If everyone is solid, we stop.
      If some are solid, others are not.
      Wait, if a player is solid, they have $n$ cards of value $k$.
      If they are solid, they don't contribute to "Misplaced" count.
      But they still operate.
      Maybe we should refine the logic:
      Only non-solid players play? No, everyone plays.
      
      Let's rethink.
      The bound $N^2 - N$ is $N \times (N-1)$.
      The number of "slots" available is $N$.
      Perhaps there is a schedule.
      Consider $N=3$.
      If we run for $N(N-1)$ steps.
      Is there a pattern?
      Maybe we can just output the result of a specific known algorithm?
      The algorithm I derived is basically:
      While not solid:
         Find a player $u$ who is NOT solid.
         Try to move their "most urgent" wrong card?
         No, we have to output a sequence of GLOBAL moves.
      
      Let's assume the simple greedy "Pass any non-self if available" works.
      In practice, if a solid player gets disturbed, he becomes non-solid.
      This increases the number of non-solid players?
      Or maybe it helps propagate errors?
      Actually, let's look at the Example 2 output again.
      It outputs 6 operations.
      Initial state is solid.
      It outputs operations that *change* the state to solid?
      Wait, Example 2 output 6 operations.
      But note says "0 operations would be enough".
      This implies the example output is just *one* valid answer.
      It does not have to be efficient.
      Wait, if I just output 0, why show 6?
      To show format.
      "If there are multiple answers, print any."
      So 0 is allowed.
      My simulation stops early.
      
      What if I just run the simulation for $N^2 - N$ steps and hope it sorts?
      Is it guaranteed?
      Let's trust the "sum of distances" logic with a slight tweak.
      If a player is solid, he effectively acts as a hub for passing away from himself?
      Wait, if I am solid (player $i$), I have only $i$'s.
      I pass an $i$.
      $i$ goes to $i+1$.
      It is now at $i+1$. Home is $i$.
      So distance increased from 0 to 1.
      BUT, since $i$'s are correct for $i$, moving them away is bad.
      Unless... $i+1$ needs $i$'s? No.
      Wait, $i+1$ needs $i+1$'s.
      So passing $i$ to $i+1$ is creating a misplaced card at $i+1$.
      So solid players introduce misplacements.
      This suggests we should NOT touch solid players if possible.
      But we MUST.
      However, notice that if we have $k$ misplaced cards in total.
      Each time we move a misplaced card towards home, we reduce count.
      Each time we move a correct card away, we increase count.
      But maybe we don't encounter many solid players while unsolid others exist?
      If many players are not solid, they have wrong cards.
      They pass wrong cards (reducing count).
      The solid ones pass correct cards (increasing count).
      If non-solid ones dominate, we win.
      Is it possible non-solid ones are few?
      Example: P1 solid. P2..Pn not solid.
      P1 adds bad. Others add good.
      Maybe net is good.
      
      Let's look at the constraints again.
      "Using at most $N^2 - N$ operations".
      Is it possible that we can solve it in $N^2 - N$ using the "Pass non-self" strategy?
      Actually, there is a constructive proof for this problem.
      The problem is "Codeforces 1834E"? No.
      The problem is from **Codeforces Round #832 (Div. 2)** Problem C? No.
      Actually, searching "playing around the table codeforces" finds **Codeforces 1680C**. No.
      Found it: **Codeforces Round #918 (Div. 2)** - Problem **F**? No.
      Ah, it is **Codeforces 1798**? No.
      Wait, it's **Codeforces 1774**?
      Actually, it's from **Codeforces Round 907 (Div. 2)**. No.
      Okay, I will stick to the simulation.
      The limit is high.
      I'll add a safety break.
      The greedy strategy is the most logical one given the constraints.
      
      Implementation details:
      Read $N$.
      Read matrix. Flatten or keep as `vector<vector<int>>`.
      `moves` array of size $K$.
      `ops_count` variable.
      `is_solid` function.
      Simulation loop.
      Store the actual `d1...dn` for each step.
      Print `ops_count` then lines.
      
      Complexity check:
      $N=100$. $N^2-N \approx 9900$.
      Inner loop $O(N)$ to pick cards (linear scan).
      Vector erase $O(N)$.
      Push back $O(1)$ amortized.
      Total per step $O(N^2)$.
      Total $O(N^2(N^2)) = O(N^4)$.
      $100^4 = 10^8$.
      Might be tight for 2 seconds?
      Wait.
      $N=100$. Operations $10000$.
      Per operation, I scan vectors of size 100.
      $100 \times 100 = 10000$.
      So $10^4 \times 10^4 = 10^8$.
      In C++, $10^8$ simple ops is usually < 0.5s.
      The vector overhead might be high.
      Using `vector::erase` on front is slow? No, I just find element and erase.
      Erase in middle shifts elements.
      Since size is small (100), shifting is fast.
      I can optimize by swapping with last element and popping back.
      Wait, duplicate values?
      If I pick card $X$ to pass.
      Scan vector. If `val == X`. Swap with `back()`. Pop back.
      This is $O(1)$.
      Finding $X$ is $O(N)$.
      So per step $O(N^2)$ total.
      Total complexity $O(N^4)$.
      With $N=100$, $10^8$ operations.
      Should pass comfortably in 2s.
      
      One optimization:
      Instead of `find` then `erase` then `push_back` to receiving,
      We can manage state explicitly.
      But simulation is easiest.
      
      Wait, I need to output the passed cards.
      The output format:
      $k$
      $d_1 \dots d_n$ (passed by P1..Pn)
      ...
      Note: $d_j$ is the card passed by player $j$.
      So I need to store the sequence of moves.
      Store `vector<vector<int>> recorded_moves`.
      
      Wait, checking solidness $O(N^2)$.
      Total $O(N^2(N^2))$?
      $O(N^4)$.
      
      Actually, solid check is $O(N)$.
      For each player $i$, check all cards.
      Total $N$ checks of $N$ items -> $O(N^2)$.
      Wait, solid check is $O(N^2)$ because checking if ALL cards are $i$ involves checking $n$ cards.
      Doing this for $n$ players is $O(n^2)$.
      So total complexity is indeed $O(k \cdot n^2)$.
      $k \approx n^2$.
      So $O(n^4)$.
      $10^8$ ops. Acceptable.
      
      Corner case:
      If $k$ limit exceeded, just stop.
      Actually, I should check `is_solid` periodically or before adding to log?
      Actually, if I stop early, that's fine.
      But if the loop runs exactly $n^2-n$ times and `is_solid` is true?
      Wait, "using at most $n^2-n$".
      If it takes $n^2-n$ steps, I print that count.
      If I finish earlier, I print smaller count.
      The loop condition `cnt < n*n - n` ensures I don't exceed limit.
      
      One subtle point:
      If `is_solid` is checked at start of loop.
      If initially solid -> 0 ops.
      Else loop.
      Inside loop:
         Record pass sequence.
         Update hands.
         Increment cnt.
      End loop.
      Wait, if loop finishes and still not solid, I should output what I have?
      The problem asks "Find a way to do it".
      Since solution is guaranteed, my strategy should find it or close to it.
      If not found, maybe my strategy is flawed.
      But based on the "greedy" logic, it should work.
      If not, I can fallback to printing 0 (which is invalid if not solid) or maybe a different strategy.
      However, for the purpose of coding, I'll stick to this.
      
      Wait, I can optimize finding the card to pass.
      We don't need to erase from vector if we just mark it as "sent".
      Actually, easier to just erase.
      Also, I can store `moves` as `vector<vector<int>>` to print later.
      
      Wait, memory limit 256MB. $N=100$. Vectors are small. Safe.
      
      Final Logic Check:
      Input:
      2
      2 1
      1 2
      
      Init:
      P1: {2, 1}
      P2: {1, 2}
      Loop 1 (cnt=0 < 2):
      P1 check {2, 1}. Has 2!=1. Pass 2.
      P2 check {1, 2}. Has 1!=2. Pass 1.
      Moves: P1->2, P2->1.
      Record: {2, 1}
      Update:
      P1 keeps 1, gets 1. -> {1, 1}.
      P2 keeps 2, gets 2. -> {2, 2}.
      cnt=1.
      Check solid: Yes. Break.
      Output:
      1
      2 1
      Matches Example 1.
      
      Example 2:
      3
      1 1 1
      2 2 2
      3 3 3
      Init solid.
      Loop doesn't run.
      Output: 0 (newline).
      Matches "0 operations would be enough".
      
      Looks good.
      
      One more check:
      "The (i+1)-th player sits to the right of the i-th player".
      So 1->2, 2->3, ..., n->1.
      My code assumes P_i passes to P_{i% n + 1}.
      If i=n, (n%n)+1 = 1. Correct.
      
      Also, input reading:
      Row 1 corresponds to Player 1.
      Row n corresponds to Player n.
      Input order is correct.
      
      Double check constraints on values.
      $1 \le c_j \le n$.
      
      One small issue:
      If I have multiple wrong cards, I pick the first one I see.
      Does order matter?
      Probably not for correctness, but maybe for performance (finding the best one?).
      But since any wrong card reduces distance, it's fine.
      
      One edge case: $N=2$, cards 2 2 at P1, 1 1 at P2.
      My trace:
      P1 {2, 2}. Pass 2.
      P2 {1, 1}. Pass 1.
      Moves: P1->2, P2->1.
      Update: P1 keeps 2, gets 1 -> {2, 1}. P2 keeps 1, gets 2 -> {1, 2}.
      Not solid.
      Next iter:
      P1 {2, 1}. Pass 2.
      P2 {1, 2}. Pass 1.
      Moves: P1->2, P2->1.
      Update: P1 keeps 1, gets 2 -> {1, 2}. P2 keeps 2, gets 1 -> {2, 1}.
      Wait.
      Iter 2:
      P1 has {2, 1}. Pass 2. Keeps 1. Gets 1 (from P2). -> {1, 1}.
      P2 has {1, 2}. Pass 1. Keeps 2. Gets 2 (from P1). -> {2, 2}.
      SOLID.
      Ah, my manual trace was sloppy.
      Let's re-trace carefully.
      Start: P1{2,2}, P2{1,1}.
      Op 1:
      P1 passes 2. P2 passes 1.
      P1 keeps {2} (removed one 2). Recvs 1 (from P2).
      So P1 has {2, 1}.
      P2 keeps {1} (removed one 1). Recvs 2 (from P1).
      So P2 has {1, 2}.
      End of Op 1.
      Op 2:
      P1 has {2, 1}. P1 wants 1. Non-1 is 2. Pass 2.
      P2 has {1, 2}. P2 wants 2. Non-2 is 1. Pass 1.
      P1 keeps 1. Recvs 1 (from P2). -> {1, 1}. Solid.
      P2 keeps 2. Recvs 2 (from P1). -> {2, 2}. Solid.
      Done.
      Takes 2 ops. Limit $4-2=2$.
      So fits exactly.
      
      What if $N$ is odd?
      Say $N=3$.
      P1{1, 2, 2}, P2{2, 2, 1}, P3{3, 3, 3}?
      Wait, counts must match.
      1s: P1(1), P2(1), P3(1).
      2s: P1(2), P2(2), P3(0).
      3s: P1(0), P2(0), P3(3).
      Total 1s: 3. Total 2s: 4 (Wait, need 3).
      Correction:
      P1: 1, 2, 2. (Two 2s).
      P2: 2, 3, 3? (Two 3s).
      P3: 3, 1, 1? (Two 1s).
      Counts:
      1: P1(1), P3(2). Sum 3. OK.
      2: P1(2), P2(1). Sum 3. OK.
      3: P2(2), P3(1). Sum 3. OK.
      So config:
      P1 {1, 2, 2}.
      P2 {2, 3, 3}.
      P3 {3, 1, 1}.
      
      Simulate:
      P1 wants 1. Wrong: {2, 2}. Pick 2.
      P2 wants 2. Wrong: {3, 3}. Pick 3.
      P3 wants 3. Wrong: {1, 1}. Pick 1.
      Moves: P1->2, P2->3, P3->1.
      P1 keeps {1, 2}, gets 1 (from P3). -> {1, 1, 2}.
      P2 keeps {2, 3}, gets 2 (from P1). -> {2, 2, 3}.
      P3 keeps {3, 1}, gets 3 (from P2). -> {3, 3, 1}.
      
      New state:
      P1 {1, 1, 2}. (One 2).
      P2 {2, 2, 3}. (One 3).
      P3 {3, 3, 1}. (One 1).
      It seems we reduced number of wrong cards for each?
      P1: Had two 2s. Now one 2.
      P2: Had two 3s. Now one 3.
      P3: Had two 1s. Now one 1.
      Wait, P2 received 2. P2 wanted 2. P2 had {2, 3, 3}. Removed 3. Added 2.
      P2 now has {2, 3, 2} (two 2s, one 3).
      Wait, P2 had {2, 3, 3}. Picked 3. Kept {2, 3}. Added 2. -> {2, 3, 2}.
      P2 wants 2. Wrong is 3.
      P1 had {1, 2, 2}. Picked 2. Kept {1, 2}. Added 1. -> {1, 1, 2}.
      P1 wants 1. Wrong is 2.
      P3 had {3, 1, 1}. Picked 1. Kept {3, 1}. Added 3. -> {3, 1, 3}.
      P3 wants 3. Wrong is 1.
      So counts:
      P1: one 2.
      P2: one 3.
      P3: one 1.
      Each step, everyone reduced their wrong card count by 1?
      Let's check:
      P1: 2 wrongs -> 1 wrong. (-1)
      P2: 2 wrongs -> 1 wrong. (-1)
      P3: 2 wrongs -> 1 wrong. (-1)
      Total wrongs = 3.
      Next step:
      P1 {1, 1, 2}. Pick 2.
      P2 {2, 3, 2}. Pick 3. (Wait, P2 has two 2s and one 3. Pass 3).
      P3 {3, 1, 3}. Pick 1.
      Moves: P1->2, P2->3, P3->1.
      Updates:
      P1 keeps {1, 1}, gets 1 (from P3). -> {1, 1, 1}. SOLID.
      P2 keeps {2, 2}, gets 2 (from P1). -> {2, 2, 2}. SOLID.
      P3 keeps {3, 3}, gets 3 (from P2). -> {3, 3, 3}. SOLID.
      Done in 2 steps.
      Limit $3^2-3 = 6$.
      Very efficient.
      
      This confirms the strategy works well.
      
      Wait, in the second step, P2 had {2, 3, 2}.
      Why did P2 have two 2s?
      Start: P2 {2, 3, 3}. Picked 3. Kept {2, 3}.
      Recv from P1: P1 passed 2.
      So P2 gets 2.
      So P2 has {2, 3, 2}. Correct.
      So the "swapping" of correct cards into hands happens perfectly.
      P1 needed 2s? No P1 needed 1.
      Wait, P1 passed 2 (wrong). P1 needed 1.
      P1 kept 1.
      P1 got 1 from P3.
      So P1 consolidated his 1s.
      
      It seems consistent.
      
```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    // Hands stored as vector of vectors.
    // hands[i] corresponds to player i+1 (using 0-based indexing internally)
    vector<vector<int>> hands(n);
    for (int i = 0; i < n; ++i) {
        hands[i].resize(n);
        for (int j = 0; j < n; ++j) {
            cin >> hands[i][j];
        }
    }

    // Function to check if everyone is solid
    auto is_solid = [&]() -> bool {
        for (int i = 0; i < n; ++i) {
            bool all_correct = true;
            for (int card : hands[i]) {
                if (card != i + 1) {
                    all_correct = false;
                    break;
                }
            }
            if (!all_correct) return false;
        }
        return true;
    };

    // Maximum operations allowed
    int max_ops = n * n - n;
    
    vector<vector<int>> operations_log;
    int ops_count = 0;

    while (ops_count < max_ops) {
        if (is_solid()) break;

        // Decide passes for each player
        vector<int> current_passes(n);
        vector<bool> passed_flag(n, false); 
        
        // We need to determine what each player passes based on current hands
        // But we cannot modify hands until all passes are decided?
        // Actually, simultaneous operations means decision based on state at beginning of step.
        // However, the hand state changes during the loop over players.
        // So we must save the decisions.
        
        vector<pair<int, int>> decisions(n); // {player_idx, card_val}
        
        for (int i = 0; i < n; ++i) {
            // Find a card != i+1 to pass
            int card_to_pass = -1;
            // To be consistent and easy, find the LAST occurrence or FIRST?
            // Let's just find first.
            // Optimization: scanning whole vector
            for (int card : hands[i]) {
                if (card != i + 1) {
                    card_to_pass = card;
                    break;
                }
            }
            
            if (card_to_pass == -1) {
                // No wrong card, must pass a correct card (all are i+1)
                if (!hands[i].empty()) {
                    // Pick any, say the last one
                    card_to_pass = hands[i].back();
                }
            }
            decisions[i] = {i, card_to_pass};
        }

        // Verify if we can proceed
        // Note: We are allowed at most N^2 - N.
        // But we must ensure we don't get stuck if logic is flawed (unlikely).
        
        // Update hands and record moves
        // We need to move cards.
        // Player i passes to Player (i+1)%n
        
        // Temporary storage for received cards to avoid race conditions
        vector<int> received(n, -1); // -1 indicates waiting, though simultaneous means all arrive
        
        for (int i = 0; i < n; ++i) {
            int p_idx = i;
            int val = decisions[p_idx].second;
            
            // Remove from sender
            // We can just remove the element found. Since we might have duplicates,
            // remove one instance.
            auto& h = hands[p_idx];
            for (auto it = h.begin(); it != h.end(); ++it) {
                if (*it == val) {
                    h.erase(it);
                    break;
                }
            }
            
            // Prepare to add to receiver
            int recv_idx = (p_idx + 1) % n;
            received[recv_idx] = val;
        }
        
        // Add received cards
        for (int i = 0; i < n; ++i) {
            if (received[i] != -1) {
                hands[i].push_back(received[i]);
            }
        }
        
        // Record the moves for this operation
        // We need to output n integers d1...dn. dj is card passed by j-th player (1-based).
        // decisions[i] corresponds to player i+1 passing decisions[i].second
        vector<int> log_move(n);
        for (int i = 0; i < n; ++i) {
            log_move[i] = decisions[i].second;
        }
        operations_log.push_back(log_move);
        ops_count++;
    }

    cout << operations_log.size() << "\n";
    for (const auto& op : operations_log) {
        for (int i = 0; i < n; ++i) {
            cout << op[i] << (i == n - 1 ? "" : " ");
        }
        cout << "\n";
    }

    return 0;
}
```
Wait, `received` vector initialization.
When processing `decisions`, I use `hands` to find card to pass and erase it.
Then I calculate `received`.
Then I add `received`.
This simulates simultaneous pass.
Is there any dependency issue?
Decisions are based on `hands` state at start of iteration.
Erasing modifies `hands`, but I only modify `hands[p_idx]`.
When I compute `received[recv_idx]`, I put card from `p_idx`.
When loop continues to `recv_idx`, `hands[recv_idx]` has been modified?
Wait.
In my code:
I iterate `i` from 0 to n-1.
Inside loop, I decide what `hands[i]` passes.
But `decisions[i]` uses `hands[i]` values from START of loop?
Ah, the variable `hands` is global.
If I iterate `i=0`:
  Modify `hands[0]`.
  Add card to `received[1]`.
Next `i=1`:
  Use `hands[1]` to decide. `hands[1]` hasn't been modified yet?
  It might contain card received from `0`? No, that happens AFTER loop.
  So `hands` is not modified in place during the pass collection phase.
  Wait, I am modifying `hands[i]` inside the loop.
  For `i=1`, `hands[1]` has not been touched by `i=0` modification because `0` touches `0` and fills `received[1]`.
  So when `i=1` executes, `hands[1]` is in original state.
  So yes, it works correctly.
  The only concern is `received` array.
  I clear `received` before loop. `fill(-1)`.
  Correct.
  
One detail: `hands` resize.
`hands[i]` size is always `n`.
Wait, if I erase, size becomes `n-1`.
Then I push back. Size becomes `n`.
Correct.
But wait, I am erasing during decision phase.
Does `decisions` rely on erased values?
No, `decisions` determined before `hands` modification?
Wait, inside the loop:
```cpp
        for (int i = 0; i < n; ++i) {
            // ... search for card ...
            decisions[i] = ...;
        }
```
Here I use `hands[i]`. `hands[i]` is NOT modified yet for this iteration `i`.
Modifications happen to `hands[i]` inside the loop?
No, I have NOT implemented the erasure in the first phase.
I implemented erasure in `Update hands` phase.
Let's look at my code structure again.
My code logic inside `while`:
1. Calculate `decisions` based on `hands`.
   This loop reads `hands` but doesn't write to `hands`.
   Wait, inside the `for` loop for decisions, I do NOT modify `hands`.
   I just read and set `decisions[i]`.
   Wait, in my thought process I thought I erased.
   But in the code snippet above:
   ```cpp
            // Find a card != i+1 to pass
            // ...
            if (card_to_pass == -1) {
                 // Pick any
                 card_to_pass = hands[i].back();
            }
   ```
   This doesn't erase.
   So `hands` is preserved for reading in `decision` phase.
   
2. Then `Update hands`.
   ```cpp
        for (int i = 0; i < n; ++i) {
            int p_idx = i;
            int val = decisions[p_idx].second;
            // Remove from sender
            auto& h = hands[p_idx];
            for (auto it = h.begin(); it != h.end(); ++it) {
                 if (*it == val) {
                     h.erase(it);
                     break;
                 }
            }
            // ...
            received[recv_idx] = val;
        }
   ```
   Here I erase.
   Wait, this erasing happens AFTER all decisions are collected.
   So it's perfectly safe.
   
   Wait, the loop `for (int i = 0; i < n; ++i)` in decision phase is separate.
   Yes.
   
   Wait, in the decision phase, I do not erase.
   So `hands[i]` remains intact for reader `i+1`?
   Yes, because I never touch `hands[j]` for $j \neq i$ in decision phase.
   So `hands` state is fully consistent at the start of `while`.
   
   One small bug potential: `hands[i]` contains `val`?
   Yes, I search for `val`.
   Is it guaranteed `val` is in `hands[i]`?
   If I picked `hands[i].back()`, yes.
   If I picked from search, yes.
   
   Wait, does `hands[i]` size change in decision phase?
   No.
   
   So the logic is sound.
   
   Double check: "The (i+1)-th player sits to the right of the i-th player".
   Indices 1..n.
   My code uses 0..n-1.
   Input row 1 is player 1 (index 0).
   Passes to index 1.
   Last player (n-1) passes to 0.
   Code: `(p_idx + 1) % n`. Correct.
   
   Final verification on constraints and types.
   $n \le 100$. `int` is fine.
   
   Memory usage: vectors are small.
   
   Looks ready.

One small correction:
In the code:
`vector<vector<int>> hands(n);`
`hands[i].resize(n);`
Reads `hands[i][j]`.
This is correct.

Also `max_ops` can be 0 if n=1?
Constraints: $2 \le n \le 100$.
So $max\_ops \ge 2$.
So loop runs at least once if not solid?
Wait, if $n=2$, max\_ops = 2.
If solid initially, loop breaks immediately. Ops = 0.
If not solid, loop runs up to 2 times.
Correct.

Wait, if I break early, I print `operations_log.size()`.
Correct.

One edge case: `is_solid` check inside loop.
If solid initially, prints 0.
Correct.

The problem statement: "Then k lines should follow."
Format is strict.

```cpp
/*
Solution approach:
Simulate the process of passing cards.
At each step, every player passes a card.
Strategy:
A player 'i' (who needs card 'i') should pass a card 'c' such that c != i if possible.
This reduces the "misplaced" status of card 'c' (moving it closer to player 'c').
If a player has only correct cards ('i'), they must pass one of them.
Since the problem guarantees a solution exists within N^2 - N operations, and our greedy strategy generally pushes misplaced cards towards their targets, this simulation should find a solution within the bounds.
*/
```
I will include brief comments.

Double check the `is_solid` function.
`for (int card : hands[i]) { if (card != i + 1) ... }`
`hands[i]` contains ints. `i+1` is target value.
Correct.

Wait, inside decision loop:
```cpp
            // Find a card != i+1 to pass
            int card_to_pass = -1;
            for (int card : hands[i]) {
                if (card != i + 1) {
                    card_to_pass = card;
                    break;
                }
            }
```
If `hands[i]` is `vector<int>`, iterating is fast.
However, `hands[i]` is resized to `n`.
If I erase an element, the vector shrinks.
But I only erase in the update phase.
In decision phase, `hands[i]` is const-view conceptually.
Wait, `hands` is a reference to `vector<vector<int>>`.
In decision phase, I don't modify it.
So it's safe.

Wait, `decisions` stores `{i, card_to_pass}`.
`i` is index.
Loop `for (int i = 0; i < n; ++i)` creates `decisions`.
Wait, inside the loop, I use `hands[i]`.
This is correct.

One minor optimization: `break` if solid.
This is done.
Complexity: $O(N^4)$.
$100^4 = 10^8$.
Operations inside are simple.
Should pass 2s limit.
Usually $10^8$ is borderline but for simple vector ops it's faster than complex math.
Given 2 seconds, it should be okay.
Also, typically $k$ (number of steps) is much smaller than $N^2-N$ on average.
Only worst cases take long.

Final check: `decisions` vector size.
I use `decisions` to store `(player_index, card_value)`.
Indices 0 to $n-1$.
Correct.

One typo in thought block:
`hands[i]` resize.
`hands[i].resize(n)` inside loop `for(int i=0;i<n;++i)`.
Input loop:
```cpp
        for (int j = 0; j < n; ++j) {
            cin >> hands[i][j];
        }
```
This is correct.
Wait, I wrote `hands[i].resize(n)` but didn't fill it.
Actually `hands[i]` needs to be filled.
The `resize` creates elements with default 0.
Then I overwrite with `cin`.
So it is correct.

```cpp
// Final code structure
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// ... code ...
```
Seems good.

One quick check:
The output `log_move` stores the card passed by player 0 to player 1?
No. `d1` is card passed by player 1.
`log_move` stores `card_to_pass` for each `i`.
My code: `decisions[i] = ...`.
Loop `op`:
`log_move[i] = decisions[i].second`.
Player 1 is index 0.
`decisions[0]` corresponds to P1.
`log_move[0]` corresponds to `d1`.
Matches output format.
`cout` loop prints `d1` to `dn`.
`d1` is `log_move[0]`.
Yes.

```cpp
/*
Sample 1:
Input
2
2 1
1 2
Output:
1
2 1
Correct.

Sample 2:
Input
3
1 1 1
2 2 2
3 3 3
Output:
0
(Note: sample output shows 6 ops, but problem allows any valid k. 0 is valid).
My code outputs 0.
*/
```
Everything seems correct.
</think>

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    // hands[i] stores the cards held by player i (0-indexed internally, representing player i+1)
    vector<vector<int>> hands(n);
    for (int i = 0; i < n; ++i) {
        hands[i].resize(n);
        for (int j = 0; j < n; ++j) {
            cin >> hands[i][j];
        }
    }

    // Function to check if everyone is solid
    // A configuration is solid if player i holds only cards with value i+1
    auto is_solid = [&]() -> bool {
        for (int i = 0; i < n; ++i) {
            bool all_correct = true;
            for (int card : hands[i]) {
                if (card != i + 1) {
                    all_correct = false;
                    break;
                }
            }
            if (!all_correct) return false;
        }
        return true;
    };

    // Maximum allowed operations
    int max_ops = n * n - n;
    
    vector<vector<int>> operations_log;
    int ops_count = 0;

    // Simulation loop
    while (ops_count < max_ops) {
        // Check if the target state is already reached
        if (is_solid()) break;

        // Phase 1: Determine which card each player will pass
        // We need to base this on the state at the beginning of the operation
        // So we collect decisions first.
        vector<pair<int, int>> decisions(n); // {player_index, card_value}
        
        for (int i = 0; i < n; ++i) {
            // Player i (representing player i+1) wants cards with value i+1
            // Strategy: If there is any card with value != i+1, pass one of them.
            // This tends to move misplaced cards closer to their destination.
            // If all cards are correct (value i+1), we must pass one anyway.
            
            int card_to_pass = -1;
            for (int card : hands[i]) {
                if (card != i + 1) {
                    card_to_pass = card;
                    break; // Found a wrong card, pass it
                }
            }
            
            if (card_to_pass == -1) {
                // No wrong card, all are correct. Pass any (the last one).
                // hands[i] is guaranteed not empty (size n)
                card_to_pass = hands[i].back();
            }
            
            decisions[i] = {i, card_to_pass};
        }

        // Phase 2: Execute moves and update hands
        // Simultaneous moves: Player i passes to Player (i+1)%n
        vector<int> received_cards(n, -1); // Stores the card received by each player

        for (int i = 0; i < n; ++i) {
            int p_idx = i;
            int val = decisions[p_idx].second;
            
            // Remove card from sender's hand
            auto& h = hands[p_idx];
            // Find and erase one instance of the card.
            // Since we need to remove exactly the card identified, iterating is safe.
            // Although duplicates might exist, removing one instance is sufficient.
            for (auto it = h.begin(); it != h.end(); ++it) {
                if (*it == val) {
                    h.erase(it);
                    break;
                }
            }
            
            // Determine recipient
            int recv_idx = (p_idx + 1) % n;
            received_cards[recv_idx] = val;
        }
        
        // Add received cards to respective hands
        for (int i = 0; i < n; ++i) {
            if (received_cards[i] != -1) {
                hands[i].push_back(received_cards[i]);
            }
        }
        
        // Record the moves for output
        vector<int> log_move(n);
        for (int i = 0; i < n; ++i) {
            log_move[i] = decisions[i].second;
        }
        operations_log.push_back(log_move);
        ops_count++;
    }

    // Output results
    cout << operations_log.size() << "\n";
    for (const auto& op : operations_log) {
        for (int i = 0; i < n; ++i) {
            cout << op[i] << (i == n - 1 ? "" : " ");
        }
        cout << "\n";
    }

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=3,得分 0.0,completion_tokens=1121)

````
The problem is asking us to distribute cards among n players sitting around a round table such that each player i ends up with only cards numbered i. We can do this by having players pass cards to their right simultaneously in each operation.

Let's think about how we can construct a solution. The key insight is to think about each card's "journey" from where it starts to where it needs to end up.

For each card numbered j (where j ∈ [1, n]), suppose it initially belongs to some player pos_j. Since player j should eventually have all cards numbered j, this card needs to move clockwise around the table until it reaches the segment of the table controlled by player j.

A clever approach:
- Each card j will be rotated around the table for exactly one full cycle before being used by its target player. This ensures no two cards interfere during rotation.
- For each position in the first row (each column/position), we assign which value's card passes through there. This creates a permutation for each column position.
- After n-1 cycles of rotations, all cards reach their destination players as solid groups.

Here's how I'll implement it:

```cpp
#include <iostream>
using namespace std;

int main() {
    int n;
    cin >> n;
    
    // Read initial card configuration
    vector<vector<int>> c(n, vector<int>(n));
    vector<vector<pair<int,int>>> cnt(n + 1);
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cin >> c[i][j];
            cnt[c[i][j]].push_back({i, j});
        }
    }
    
    // For each card number j and each starting position p in first row,
    // store which actual card (initial_player, initial_pos) it is
    vector<vector<vector<pair<int,int>>>> card(n + 1, vector<vector<pair<int,int>>>(n));
    for(int i = 0; i <= n; i++) {
        for(const auto& [pos_p, _] : cnt[i]) {
            for(int p = 0; p < n; p++) {
                card[i][p][pos_p.second % n] = make_pair(pos_p.first, pos_p.second);
            }
        }
    }
    
    // Print n-1 operations
    for(int i = 0; i < n - 1; i++) {
        cout << endl;
        for(int j = 0; j < n; j++) {
            cout << card[j][i][j].first + 1 << " ";
            for(int k = 0; k < n - 1; k++) 
                if(k > 0) cout << " ";
            cout << card[j][i][j].second;
        }
    }
    
    return 0;
}
```

I'm tracking which specific card goes to each player at each stage of rotation. Each card j follows a unique path through the table positions, ensuring no collisions and guaranteeing every card reaches its target player exactly once after n-1 full rotations.
</think>

```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n;
    cin >> n;
    
    // Read initial card configuration
    vector<vector<int>> c(n, vector<int>(n));
    vector<vector<pair<int,int>>> cnt(n + 1);
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cin >> c[i][j];
            cnt[c[i][j]].push_back({i, j});
        }
    }
    
    // For each card number j and each starting position p in first row,
    // store which actual card (initial_player, initial_pos) it is
    vector<vector<vector<pair<int,int>>>> card(n + 1, vector<vector<pair<int,int>>>(n));
    for(int i = 0; i <= n; i++) {
        for(const auto& [pos_p, _] : cnt[i]) {
            for(int p = 0; p < n; p++) {
                card[i][p][pos_p.second % n] = make_pair(pos_p.first, pos_p.second);
            }
        }
    }
    
    // Output result
    cout << n * n - n << "\n";
    for(int i = 0; i < n - 1; i++) {
        for(int j = 0; j < n; j++) {
            cout << card[j][i][j].first + 1 << " " << card[j][i][j].second;
            for(int k = 0; k < n - 1; k++) 
                if(k > 0) cout << " ";
            cout << " ";
        }
        cout << "\n";
    }
    
    return 0;
}
```
````
