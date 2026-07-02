# 题 142 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The problem asks us to rearrange $n \times m$ balls from $n$ poles into a configuration where each of the $n+1$ poles contains balls of only one color. Initially, there are $n$ poles (let's call them source poles), each having $m$ balls stacked vertically. The $(n+1)$-th pole is empty. We need to move balls such that eventually every ball ends up on a pole sharing its color, meaning we partition the balls by color onto distinct poles. Since there are $n$ colors and $n$ initial poles with $m$ balls of various colors, and finally $n+1$ available slots (poles), but we want each final pole to be monochromatic. Wait, let me re-read the goal carefully.

"You must use at most 2,000,000 operations."
"minimize the number of operations needed."
"The final distribution of colors to poles does not matter, as long as each pole contains balls of at most one color."

Initially, pole $i$ ($1 \le i \le n$) has $m$ balls. Let's say the sequence of colors in pole $i$ is $c_{i,1}, c_{i,2}, \dots, c_{i,m}$, where $c_{i,j}$ is the color of the $j$-th ball from the bottom. Pole $n+1$ is empty.
There are $n$ different colors total. Each color appears exactly $m$ times.
We have $n$ poles initially. There are $n$ colors. So if we end up with a configuration where each pole is monochromatic, since we have $n+1$ poles available, we can fit all $m$ balls of one color on one pole.
Wait, if we put all $m$ balls of a specific color onto one pole, and do this for all $n$ colors, we would need $n$ poles. We have $n+1$ poles available. However, initially we have $n$ poles occupied. One is empty at start.
Actually, if we simply collect all balls of color $K$ on pole $P_K$, then each pole $P_K$ will have $m$ balls. The remaining pole(s) could be empty. But since we have $n$ colors and $n$ non-empty poles initially, it seems natural to map each color to one of the poles.
Since we have $n+1$ poles, and $n$ colors, maybe we don't even need to fill all poles? Or maybe some pole needs to take more than one color? No, constraint says "at most one color". So we must distribute the $n$ sets of $m$ balls onto distinct poles.
Wait, initially we have $n$ piles. Total balls $nm$. Each pile size $m$.
If we consolidate all balls of color $C_1$ to pole $1$, all balls of color $C_2$ to pole $2$, ..., all balls of color $C_n$ to pole $n$. Then each pole $i$ ends up containing balls of color $C_i$ (which is the set of balls originally belonging to color $C_i$). Note that initially a pole might contain mixed colors.
So the target state is: Pole $k$ contains a stack of $m$ balls, all of the same color $C_k$.
Total operations limit is $2,000,000$. Given $N \le 50, M \le 400$. Total balls $50 \times 400 = 20,000$. Operations limit is huge compared to total balls, so we just need an efficient strategy. Actually, $N^2$ or something similar is likely fine.
Specifically, the problem guarantees a valid solution exists. This is related to sorting or permutation networks.
Wait, if we treat each position in the pole as a slot, say $1..m$.
Consider the top ball of every non-empty pole. Initially there are $n$ balls on top of poles $1..n$.
We can move these balls around.
Constraint: "Pole y must have at most m-1 balls". This is crucial. A pole cannot hold more than $m-1$ balls temporarily?
Wait, the input describes the balls stacked vertically. Bottom to top.
Usually in such problems (Tower of Hanoi variant or sorting stacks), constraints on capacity apply. Here, no pole can exceed $m$ balls total? Or specifically during the process, a pole cannot have $m$ balls to receive another? Yes, "Pole y must have at most m-1 balls". So you cannot put a ball on top of a full stack of $m$ balls.
However, in the final configuration, each pole holding a color will have $m$ balls. How do we achieve that if we can't place on a full stack?
The last ball of color $C$ must go to a pole that already has $m-1$ balls of color $C$.
This implies we should build stacks one color at a time.
Suppose we decide that Pole 1 will eventually hold Color 1, Pole 2 will hold Color 2, etc. Or some permutation.
Let's denote the set of positions for each color. Since there are $m$ balls of color $c$, they need to be placed somewhere.
Since a pole can accept moves only if it has $< m$ balls, and we want to reach $m$ balls of a specific color, we should probably leave room for the final stacking.
But wait, if we are moving a ball *off* a pole, that reduces the count.
Maybe the strategy involves identifying the correct topmost ball and moving it to a destination.
But balls below a moved ball might block access.
This looks like a variation of "sorting on parallel processors" or "rearranging arrays".
Actually, this specific problem ("Ball Game") is from Codeforces (Problem 1667B? No, checking context... It resembles a standard competitive programming problem structure).
Let's look at the sample.
Sample Input:
2 3 (n=2, m=3)
1 1 2
2 1 2
Poles:
P1 (bottom -> top): 1, 1, 2
P2 (bottom -> top): 2, 1, 2
P3: Empty.
Goal: Monochromatic poles.
Colors present: 1, 2. Count(1)=3, Count(2)=3.
Initial counts per pole: P1 has two 1s, one 2. P2 has two 2s, one 1.
Output uses 6 ops.
1 3 (move top of P1 [2] to P3. P1: 1, 1; P3: 2)
2 3 (move top of P2 [2] to P3. P2: 2, 1; P3: 2, 2)
2 3 (move top of P2 [1] to P3. P2: 2; P3: 2, 2, 1 -- Invalid? Wait.)
Constraint: P3 must have at most m-1 = 2 balls when receiving?
In the sample output trace:
Init:
P1: [1, 1, 2]
P2: [2, 1, 2]
P3: []
Op 1: 1 -> 3. Move '2' from P1.
P1: [1, 1]. Top is '1'. P3: ['2']. Size 1 (<= 2). OK.
Op 2: 2 -> 3. Move '2' from P2.
P2: [2, 1]. Top is '1'. P3: ['2', '2']. Size 2 (<= 2). OK.
Op 3: 2 -> 3. Move '1' from P2.
P2: [2]. Top is nothing? Wait. P2 has 1 ball. Top is 2.
Let's re-read the sample logic. The sample output moves are:
1 3
2 3
2 3
3 1
3 2
3 2

Trace again:
State: P1=[1,1,2], P2=[2,1,2], P3=[]
1. 1->3 (moves 2 from P1). State: P1=[1,1], P2=[2,1,2], P3=[2].
2. 2->3 (moves 2 from P2). State: P1=[1,1], P2=[2,1], P3=[2,2].
3. 2->3 (moves 1 from P2). Wait. P2 was [2,1] (top is 1). Moves 1. P2=[2]. P3=[2,2,1].
   P3 now has 3 balls. Limit is $m-1 = 2$. This seems to violate the rule "Pole y must have at most m-1 balls" if interpreted strictly as "before adding a ball".
   Wait, the rule is "Pole y must have at most m-1 balls". This usually means before the operation, count(y) <= m-1. After operation count becomes <= m.
   In step 3: Before op, P3 has 2 balls. So 2 <= 2 holds. After op, P3 has 3 balls. This violates the condition for subsequent steps or for reaching size m?
   Wait, $m=3$. Constraint "at most m-1 balls" means max size allowed *when receiving* is $m-1$. If you have $m-1$ balls, you can add one to make it $m$. You cannot add another.
   So P3 reaching size 3 (which is $m$) is allowed ONLY if the previous count was $m-1$. If P3 receives a 3rd ball, it goes from 2 to 3. This is fine. If it tries to receive a 4th, it fails.
   So in Sample Op 3, P3 went from 2 to 3. This is valid.
   But wait, later in sample output:
   4. 3 -> 1 (Move top of P3). Top of P3 is '1' (from bottom to top: 2, 2, 1). Moves '1'.
      State: P3=[2,2]. P1=[1,1]. Becomes P1=[1,1,1].
   5. 3 -> 2 (Move top of P3). Top is '2'. Moves '2'.
      State: P3=[2]. P2=[2]. Becomes P2=[2,2].
   6. 3 -> 2 (Move top of P3). Top is '2'. Moves '2'.
      State: P3=[]. P2=[2,2,2].

   Final State:
   P1: [1,1,1] (Color 1)
   P2: [2,2,2] (Color 2)
   P3: [] (Empty)
   All good.

Algorithm Idea:
This problem is equivalent to sorting columns or moving elements such that columns become uniform.
Since $N$ is small (50) and $M$ is small (400), $O(N^2 M)$ or $O(N^3 M)$ might be acceptable. $50^3 \times 400 \approx 5,000,000$, slightly high but maybe acceptable within 4s.
Actually, total operations limit is $2,000,000$.
With $N, M$ given, total balls $NM \approx 20,000$.
The constraint "each pole must have at most $m-1$ balls to receive" suggests we need to keep space.
Essentially, we have $N$ active poles.
The problem is finding a matching between colors and poles.
Since initially poles $1..N$ are full (size $m$), and pole $N+1$ is empty.
We must move balls from $1..N$ to $N+1$ or swap between them?
Actually, in the sample, we used P3 (empty initially) as a buffer.
Wait, the sample solution moved things to P3 to sort them, then moved them back to P1 and P2.
This looks like merging/sorting.
However, note that we have $N$ colors and $N$ final destinations (since each destination takes a color).
Since initially we have $N$ full poles, and we need to distribute into $N$ poles (some permutation), effectively we just need to permute the colors within the poles.
Wait, if we just permute the balls within the poles, we keep them on poles $1..N$. But the constraint says "Pole y must have at most m-1 balls". If we stay on $1..N$, they all have $m$ balls. We cannot move *to* any of them. So we MUST use P3 (or any empty pole, or pole with fewer balls) as a temporary storage.
Wait, actually, we can move balls off a pole if it has > 1 ball? No, balls are stacked. We can only move top.
To rearrange a pile of size $m$ to be sorted/uniform, we often need a buffer.
Since $N+1$ poles exist, and only $N+1$ slots, we have plenty of buffers.
Wait, actually, if we have $N+1$ poles and need to fill them up, we can think of the problem as follows:
Each color class corresponds to a set of balls. We want to group them together.
Because $N, M$ are small enough, maybe a constructive algorithm works.
Key observation from similar problems (e.g. Codeforces 1569D, or similar):
We can iterate through positions $k$ from $1$ to $m$. For the $k$-th position (counting from bottom or top?), we try to align the colors.
Actually, a common strategy for this "Ball Game" is to focus on one position at a time?
No, the order matters.
Let's consider the balls. Each color $c$ appears $m$ times.
We want to select a permutation $\sigma$ of colors $\{1..n\}$ mapping to poles $\{1..n+1\}$. Wait, pole $N+1$ is initially empty. But we can also use it.
Wait, do we have to use $N+1$ poles? No, we just need a valid config.
Since we start with $N$ piles of $M$ balls, total volume $NM$. If we end up with $N$ filled piles, that uses $N$ of the $N+1$ poles. One will remain empty.
Or we can distribute differently. But $M$ balls of same color must be contiguous in a stack? No, "all balls of the same color are on the same pole". They don't have to be at the bottom or top, just on the pole. But since they are a single stack on a pole, they form a contiguous vertical segment? Yes, because we cannot insert a ball in the middle. Balls on a pole are always in a stack (contiguous from bottom to some height).
Thus, if pole $y$ has balls of color $c$, all $m$ balls of color $c$ must be in pole $y$.
This means each pole ends up either empty or full (with one color).
Since there are $n$ colors and each requires $m$ balls, and each pole can hold at most $m$ balls (implied by the constraint "receive only if <m"), actually a pole can hold at most $m$ balls.
Wait, can a pole hold more than $m$ balls? "Pole y must have at most m-1 balls" (constraint to move to y). This prevents y from exceeding $m$ balls.
Can a pole ever have more than $m$ balls? If it starts with $m$ and receives 0, yes. If it receives 1, it was $<m$. It never exceeds $m$.
Since total balls $n \times m$, and we have $n+1$ poles. If we pack efficiently, we need $n$ poles to hold $m$ balls each. The $(n+1)$-th pole stays empty.
So the problem is: Rearrange the stacks so that each pole $i$ becomes monochromatic with color $C_i$.
Note: Since initially poles $1..n$ have $m$ balls, they are full. We cannot push anything INTO them directly without removing something first.
We have pole $n+1$ which is empty. This acts as the main buffer.

Strategy:
We can simulate the process.
We need to construct the target configuration. Which pole gets which color?
Let's fix the assignment: Pole $i$ will store Color $i$. (Wait, colors are given $1..n$).
But we have $n$ poles $1..n$ and one empty $n+1$. We can just assign Color $c$ to Pole $c$.
Then Pole $1$ should end up with all balls of Color 1.
Currently Pole 1 has a mix. Pole 2 has a mix.
We need to move wrong-colored balls away.
This sounds like sorting.
We can iterate through each layer $k$ from $1$ to $m$ (from bottom to top?). Or top to bottom?
If we fix the bottom-most layer of the final stack (balls at positions $m$ down to 1?), we might find a conflict.
Actually, a very specific greedy strategy works for this type of problem.
Since $N$ is small ($N \le 50$), and $M \le 400$, maybe we can check all possible permutations of colors to poles? No, $N!$ is too big. But maybe we can determine them greedily?
Actually, we can observe that for a specific color, say Color $X$, we know how many instances are currently on Pole $Y$.
Wait, simpler approach:
Look at the topmost ball on each of the current active poles. There are at least 3 poles available (initially $N \ge 2$).
Actually, notice that we have $N$ initial piles. $N+1$ slots.
Since we want to minimize operations, maybe we can process balls layer by layer?
For the balls at position $k$ (where $1$ is top, $m$ is bottom, or vice versa), let's say we want to identify which balls should end up where.
Consider the "target" column. If Pole $P$ is assigned Color $C$, then the $k$-th ball from the bottom of Pole $P$ (in the final config) must be Color $C$.
If currently at position $k$ of Pole $Q$ (where $Q$ currently holds some ball of Color $C'$), and $C' \neq C$, we might need to move it.
However, balls are stacked. We can only access the top.
This suggests we should work from the Top (smallest index of depth) downwards, or rather, clear the top layers first.
Suppose we decide the final permutation of colors to poles. Let's say Pole $p$ gets Color $col[p]$.
Then for every height $h$ (from $1$ to $m$), Pole $p$ must have Color $col[p]$ at height $h$.
Currently, Pole $q$ has some color at height $h$.
If Pole $q$ is NOT destined to be $col[q]$, then the ball at height $h$ is misplaced.
It must be moved to its correct pole eventually.
Crucially, we can only move the top ball.
So, let's iterate through the "layers" of the towers from top to bottom (or bottom to top? Moving top implies top-down processing).
Wait, if I fix the assignment of poles to colors, say Pole $i$ becomes Color $A_i$.
Now, for every level $h=1..m$:
Every Pole $i$ must have a ball of Color $A_i$.
If Pole $i$ currently has a ball of Color $X \neq A_i$ at level $h$, this ball must be removed and moved to the pole destined for Color $X$.
Let's see. We can perform moves.
We need to pick a pivot or strategy.
There is a known result for this problem. Since $M$ is small, maybe we can simply cycle?
Actually, the constraint "pole y must have at most m-1 balls" means we can't put a ball on a full tower. This forces us to move balls out of full towers before putting new ones in?
Wait, if Pole $1$ has $m$ balls, it is full. It can't receive. It MUST give up balls.
But initially ALL $N$ poles are full.
So we MUST move balls off the full poles to the empty pole ($N+1$).
Once we move enough balls off a full pole, it becomes empty enough to receive.
Since we want to reach the state where each pole is monochromatic, we effectively need to sort the balls.
Let's look at the constraints on operations: $2,000,000$. $N=50, M=400 \implies NM = 20,000$.
Even if we move each ball $100$ times, it's 2 million. So overhead is allowed.

Let's try a simple heuristic:
Iterate through the balls that *belong* to Color 1. Find their locations.
Move them to Pole 1.
Move balls of other colors out of the way.
Since we have to maintain the constraint on the buffer, it's tricky.
Let's reconsider the "layer" idea.
Since we have $N+1$ poles, and $N$ colors.
If we designate that Pole $i$ will hold Color $i$ (for $i=1..n$).
And Pole $n+1$ will be empty.
We need to gather all balls of Color 1 into Pole 1, etc.
The balls of Color 1 are currently distributed across poles $1..n$.
Suppose Color 1 appears $k$ times on Pole $P$.
We can just move those balls to Pole 1?
But they might be buried under other colors. To extract them, we need to remove the ones above them.
Those ones above must be temporarily stored in Pole $N+1$ (buffer).
Since Pole $N+1$ has size $m-1$ capacity (limit $m-1$ balls? No, "Pole y must have at most m-1 balls" applies to receiving. But what is the capacity of Pole $N+1$? It's implicitly bounded by the rule. If you can't put on a full pile, max size is $m$.
Actually, the problem statement says "Pole x must have at least one ball", "Pole y must have at most m-1 balls".
This implies no pole can ever exceed $m$ balls. If it reaches $m$, you can't add to it anymore.
Initially poles $1..n$ have $m$ balls. Pole $n+1$ has 0.
So we can put balls into $n+1$ until it reaches $m$. But wait, can we put on a pile with $m$ balls? NO. Max balls for destination is $m-1$. So after moving a ball, pile becomes size $m$. Next ball cannot go there.
So max capacity of any pole is $m$ balls. And once a pole has $m$ balls, it becomes immutable (cannot receive).
Therefore, in the final configuration, every occupied pole has exactly $m$ balls (full).
Since we have $n$ colors, and each occupies a pole, we will end up with $n$ full poles and one empty pole.
To facilitate movement, we should keep at least one pole that isn't full (or partially full) to act as a buffer?
Actually, initially we have $n$ full and $1$ empty.
When we start moving balls to the empty one ($N+1$), it fills up. Once it has $m$ balls, it becomes frozen.
But wait, if it becomes full, it can no longer be used as a buffer.
Does this mean we can only perform operations such that we don't need a buffer?
But we have to unbury balls.
Actually, notice that we have $N \ge 2$.
Suppose we pick two poles $U$ and $V$ (from $1..N$). Both are full.
We can move balls from $U$ to $V$?
No, $V$ has $m$ balls (Full). Cannot receive.
So we MUST move balls from FULL poles to the EMPTY pole ($N+1$).
Let's call $N+1$ the buffer.
While we move balls to buffer, we are essentially collecting "junk" (wrongly colored balls) or useful balls.
Wait, if we just dump everything into the buffer, we lose order.
Is it better to think about the final destination?
Let's assume we fixed the target mapping: Pole $i$ gets Color $c_i$.
Let's just focus on the "Top" row of the configuration?
Wait, if we have a full pile, the top ball is easily accessible.
The bottom ball is hidden.
Since we can only remove the top ball, we are forced to clear the tops to get to the bottom? No, we want to move the specific balls to the destination.
Actually, there is a specific property.
Look at the colors of the balls at height $k$ (from bottom).
In the final configuration, all poles $1..n$ (except one) will have balls.
Actually, if we fix the target coloring (Pole $i$ gets Color $c_i$), then at any height $h$, the pole destined for Color $C$ MUST hold a ball of Color $C$.
Let's check if we can enforce this layer by layer.
Suppose we process heights from $m$ down to $1$ (Bottom to Top).
At height $m$ (bottom), we want to establish who sits at the bottom.
The balls currently sitting at the bottom of piles $1..n$ are $b_{1,m}, b_{2,m}, \dots, b_{n,m}$.
These are the balls at the very bottom of the stacks.
Notice that we CANNOT move these balls yet unless we remove all balls above them (which means clearing the whole stack). That's too much work.
Wait, if we process from TOP to BOTTOM (height $1$ to $m$)?
The balls at height 1 (top of stack) are easy to access.
Let's see. At any point, the set of available balls is the collection of tops of all non-empty poles.
Since we want to form monochromatic stacks, maybe we can just ensure that the top ball of the final configuration matches the target.
Wait, if Pole $i$ is destined for Color $C$, and currently has Color $D$ on top. If $D=C$, good. If $D \neq C$, we must move $D$ somewhere else. Where?
To a pole that wants Color $D$.
Let's say we maintain the invariant that for every color, the balls destined for it are being collected.
But this sounds complex to coordinate.
Let's look at the constraints and limits again. $N \le 50, M \le 400$.
We have plenty of operations.
Let's try a randomized approach or a greedy simulation.
Wait, there is a known solution using a "cycle" or "rotation" logic if we were just moving items. But here items are distinct.
However, the problem is simpler: We just want to group them.
What if we perform the following:
We identify which pole will host which color.
For simplicity, let's say Pole $i$ hosts Color $i$. (Wait, colors are numbers, not indices. Colors are $1..n$).
We can decide Pole $k$ will host Color $k$ if we map the initial arrangement.
Actually, any permutation works. Let's stick to: Pole $i$ gets Color $i$.
So we need to collect all Color 1 balls to Pole 1, etc.
Since we have $N+1$ poles and initially $N$ full, we can use Pole $N+1$ as temporary storage.
But Pole $N+1$ can only hold $m-1$ balls (wait, the rule is "destination must have $<m$" so max size is $m$).
So $N+1$ can accept $m$ balls.
Is there a strategy where we just move balls cyclically?
Let's check the Sample Output again.
Sample Logic:
P1 has {1,1,2}. P2 has {2,1,2}.
Target: P1 -> {1,1,1}, P2 -> {2,2,2}.
Wait, P3 is empty.
Steps moved '2' from P1 to P3. Now P3 has {2}. P1 has {1,1}.
Then moved '2' from P2 to P3. P3 has {2,2}. P2 has {2,1}.
Then moved '1' from P2 to P3. P3 has {2,2,1}. P2 has {2}.
Then moved '1' from P3 to P1. P1 has {1,1,1}. P3 has {2,2}.
Then moved '2' from P3 to P2. P2 has {2,2}. P3 has {2}.
Then moved '2' from P3 to P2. P2 has {2,2,2}. P3 empty.
Result: P1={1,1,1}, P2={2,2,2}.
Observation:
They cleared the top ball of P1 (the '2') because P1 needed '1' at the bottom?
Wait, P1 ended up with 1s. It started with 1,1,2 (bottom->top).
So P1 needs 1 at pos 1, 1 at pos 2, 1 at pos 3.
Original P1: 1, 1, 2.
It had '2' on top. This '2' did not belong on P1 (assuming P1 stores Color 1).
So it moved '2' out.
Similarly P2 started 2, 1, 2. It needed 2s. It had '1' on top. Moved '1' out.
Where did they go? To P3.
P3 accumulated '2's and '1's.
Eventually, P3 dumped '1's to P1 and '2's to P2.
This looks like merging.
But why did P3 dump '1' first?
P3 had {2, 2, 1} (bottom->top). Top is '1'.
P1 received '1'. P2 received '2'.
It seems the algorithm picked a destination based on the color.
Basically:
Identify the target color for each ball.
Wait, determining target color for each ball is trivial if we fix the mapping of poles to colors.
Let's say Pole $i$ is assigned Color $C_i$.
We want to move balls to match this.
The sample logic effectively processed the top layers and fixed them.
However, there is a catch. When moving balls, we might disturb balls we haven't processed yet.
But wait, the top ball is unique. If we move the top ball, we expose the next.
So we can process the "top" layers of the stacks.
Let's refine this:
We want to form $N$ monochromatic stacks of height $m$.
Let's choose a permutation $\pi$ of $\{1..N\}$ (colors) to map to poles $\{1..N\}$ (indices). Wait, we have $N$ poles available. $N+1$ total.
Let's reserve Pole $N+1$ as the "dump" pole initially.
We need to move balls from $1..N$ to $N+1$ or among themselves.
Let's try a strategy based on sorting.
Actually, this problem is solved by iterating over the number of balls $k$ from $m$ down to $1$? Or $1$ to $m$?
Consider the sample again. P1 has '2' at top. P2 has '1' at top.
If we assign P1 -> Color 1, P2 -> Color 2.
P1 top '2' is bad. Move it. Where?
If we move it to P2? P2 has Color 2. '2' belongs to P2.
If we move '2' from P1 to P2, P2 adds a '2'.
Then P2 has {2,1,2,2} (size 4? No, P2 had 3 balls. Full. Cannot receive).
So we MUST move '2' from P1 to a pole that is NOT full.
Only P3 is empty.
So we MUST dump to P3.
This confirms we need to use P3.
General Strategy:
We have a set of poles. Some are full, some might not be.
Wait, initially all $1..n$ are full.
So to perform ANY move to a pole $x \in \{1..n\}$, that pole must have $<m$ balls.
So initially, we can't move TO any pole in $1..n$.
We MUST move FROM $1..n$ TO $n+1$.
We can do this until $n+1$ is full or we stop.
Actually, once $n+1$ has $m$ balls, it also becomes full.
But we have $n+1$ poles. If all become full, we are stuck?
We can never start if all are full?
Wait, the operation is "move top from x to y". Source x loses 1 ball, Dest y gains 1.
Capacity constraint: $y$ must have $< m$ before receiving.
Source $x$ must have $\ge 1$.
After move, $x$ decreases, $y$ increases.
Net change in sum of sizes is 0.
If we reach a state where all piles are size $m$, no move is possible.
But we need to solve the puzzle. So we must end in a state where some piles are full and others might not?
Wait, target configuration: $N$ piles full, 1 empty. Or $N$ piles full (one of $1..n+1$).
Actually, if $N+1$ piles are full, we have $> nm$ balls? No, total balls $nm$. So at most $N$ piles can be full (since $N$ colors * $m$). Wait, $N$ piles full is $N \times m = nm$.
So exactly $N$ poles will be full, and 1 empty (if all poles used) or distributed.
Since $nm$ balls, and each full pole holds $m$, exactly $n$ poles will be full.
If we end up with $n$ poles full, we must have performed operations.
Can we get stuck?
If we have a configuration where all non-empty poles are full, we can't move.
If we have at least one non-full pole (size $< m$) AND a non-empty pole (size $\ge 1$), we can move.
The initial state has $N$ full poles, 1 empty.
We move to empty pole. It becomes size $k$. If we stop there, we can move between non-empty?
No, we need to swap balls to sort them.
If we have a partial pile, we can use it as a buffer.
Since we start with 1 empty pole ($n+1$), we can move balls there.
As soon as $n+1$ is full, we must stop filling it or empty it.
Wait, if we empty $n+1$, we are back to start.
The trick is: we can use the "partially full" nature to reorder.
Actually, a very simple constructive algorithm exists.
Since $N$ and $M$ are small, we can simulate a cycle.
But we need to determine WHICH balls go WHERE.
Wait, we can just define the target positions explicitly.
For example, we want Pole $1$ to have Color $C_1$, Pole $2$ to have $C_2$, etc.
The simplest case is $C_i = \text{color}(i)$? No, colors are just labels.
Let's fix the target: Pole $i$ (for $i=1..n$) gets color $i$.
Actually, this assumes we have $n$ distinct colors $1..n$. Yes, problem says "n different colors".
But the input lines describe the colors on each pole.
So Pole 1 has some mix of colors.
We can arbitrarily assign: Pole 1 will contain balls of Color $X_1$, Pole 2 balls of Color $X_2$, ...
Is it easier if we assign Pole $i$ to contain balls of Color $i$ (from the set of available colors)?
Yes. Wait, the input defines the colors. We just need to match.
Since there are $N$ colors in total and $N$ poles initially, we can map each color to a pole.
Wait, there are $N$ poles initially. But there are $N$ colors.
So each color will occupy one pole in the final configuration.
Which pole? It doesn't matter.
Let's just map Color $k$ to Pole $k$ (if Pole $k$ ends up holding Color $k$).
But wait, the colors are arbitrary.
Actually, we can just look at the colors present.
Let's create an array `target_pole[1..N]` where `target_pole[c]` stores which pole should hold Color `c`.
We can simply set `target_pole[c] = c` ? No, colors are values $1..n$. Poles are $1..n$.
We can set `target_pole[color] = color`?
Wait, if we have Color 1, 2, 3.
Poles 1, 2, 3, 4. (n=3).
Can we map Color 1 to Pole 1, Color 2 to Pole 2, Color 3 to Pole 3?
Yes.
But wait, Pole $N+1$ is the empty one.
If we map $C \to P_C$, then $P_N$ holds $C_N$, $P_{N+1}$ empty.
This leaves $P_1..P_N$ to be filled.
So we decide: Target Configuration is: Pole $i$ contains balls of Color $i$ for $i=1..n$. (Assuming we label colors $1..n$ to match poles).
Wait, if input has colors like {1, 2}, but we have poles 1..2.
It works.
So we assign Pole $i$ to store Color $i$.
Then we want to gather all instances of Color $i$ to Pole $i$.
Current State: Pile $u$ contains some balls.
We need to move Color $i$ balls to Pole $i$.
If a Color $i$ ball is on Pile $u$ ($u \neq i$), it needs to move.
Constraints:
Pile $u$ has $m$ balls. Pile $i$ has $m$ balls.
We can't move to $i$. We can't move to $u$.
We must use $N+1$.
So basically, we need to execute a sorting pass.
Since $N, M$ are small, we can afford to be slightly inefficient.
Here is a robust strategy:
We can view the poles as a ring or simply list $1..n+1$.
Wait, the sample output shows operations moving to $n+1$ then back.
This is basically "Bubble Sort" or "Exchange Sort"?
But we can't exchange directly.
Here's a simplified strategy often used in such problems:
Iterate through colors. For Color 1, gather all occurrences to Pole 1.
To gather Color 1 to Pole 1:
We look at all piles. Identify topmost balls.
If top of pile $p$ is Color 1:
Check if we can move it to Pole 1.
Pole 1 is full (usually). So no.
So we must move it elsewhere temporarily?
Wait, if Pole 1 is full, we can't put anything there.
So we need to move balls OUT of Pole 1 first to make space.
If we move a non-Color 1 ball OUT of Pole 1, it helps clear space for Color 1 balls?
Actually, if we want to build a stack of Color 1, we need to put balls IN.
Since we can't put on full stack, we must reduce the stack.
Since we want to increase the stack of Color 1 (from 0 to m), and decrease others, we should aim to move balls.
But we start with all full.
So initially, we MUST move balls FROM $1..n$ TO $n+1$.
Why? Because $1..n$ are full, can't receive.
$n+1$ is empty, can receive.
We can continue filling $n+1$ until it is full?
If we fill $n+1$, it becomes immutable.
So maybe we shouldn't fill $n+1$ completely.
Wait, if we don't fill $n+1$, we can't do much?
Actually, we can use $n+1$ as a source.
If $n+1$ has balls, we can move from $n+1$ to anywhere.
So:
Step 1: Load all "useful" balls into some buffer?
No, we need to preserve order/stack property.
Let's look at the "sample" logic again.
It effectively processed layers.
Actually, the strategy "Process each color one by one" is viable.
Let's iterate $c = 1 \dots n$.
For current color $c$:
We want to collect all balls of color $c$ to Pole $c$.
Wait, we can't move balls ONTO Pole $c$ if it's full.
So we must move balls OFF Pole $c$ too?
Actually, this suggests we should just treat the problem as "Sorting a deck of cards with limited slots".
But there is a simpler approach.
Observe that we have $N$ colors.
If we perform a global rotation/cycle, we might sort them?
No, colors are arbitrary.
Let's consider the solution from the editorial or similar problems (Codeforces Round #something, Problem D/B etc).
This problem looks like "Ball Game" from a contest.
Found similar problem: "Codeforces 1568F"? No.
"Codeforces 1694C"? No.
"Codeforces 1678D"?
Wait, search query "Ball Game n+1 poles".
Ah, this looks like "Codeforces 1667E"? No.
Okay, let's rely on logic.
We have $n$ colors, each appearing $m$ times.
We want to move all balls of color $k$ to Pole $k$. (Assume we fix mapping $Color \to Pole$).
Actually, we don't need to fix mapping beforehand. But we can just say Pole $i$ should have Color $C_i$.
Since $n \le 50$, we can just try to satisfy the requirement for all colors.
A naive algorithm:
Repeat until all piles are sorted:
  Find a pile $u$ and pile $v$ such that moving top of $u$ to $v$ makes progress.
Progress means: The top of $u$ belongs to $v$.
Condition: Top of $u$ has color $X$. We assign Pole $v$ to Color $X$.
Then we move $X$ from $u$ to $v$.
We need to ensure $v$ is not full.
If $v$ is full, we cannot move to $v$.
If ALL available destination piles $v$ (that accept Color X) are full, we have a problem?
Wait, we have $n+1$ piles. $n$ colors.
If we assign each color to a unique pile. Say Color $X$ to Pole $k_X$.
If Pole $k_X$ is full, we can't put Color $X$ there yet.
But Color $X$ needs to be there eventually.
So we must move other balls away to free up space?
But we can't move into a full pile. We must move FROM full pile.
So if we have a full pile $u$, we can move its top ball to some pile $w$ ($w \neq u$).
If $w$ is empty or partial, it accepts.
If $w$ is full, it rejects.
If all piles are full, we are stuck.
But initially $n$ piles are full, 1 empty.
So we can always move from full to empty.
Once empty is full, we might be stuck IF no other piles are partial.
But if we have a pile $v$ that wants Color $X$, and we have Color $X$ on top of $u$.
If we move $u \to v$ (Color X to X-pole), we are good.
If $v$ is full, we can't do it.
But $v$ is full $\implies$ it has $m$ balls. If it is supposed to have Color $X$, then it is ALREADY sorted (assuming all balls on it are X).
Wait, if $v$ is supposed to hold Color $X$ and is full, then it MUST have $m$ balls of Color $X$.
If it currently has a ball of Color $Y$ on top, and $Y=X$, great.
If $Y \neq X$, then it contains a bad ball.
But we assumed it is sorted. Contradiction.
So if $v$ is full, it must be perfectly sorted to be a valid final state.
If it is sorted, we should NOT touch it.
But if it is not sorted (has wrong ball on top), it cannot be full (because we established it has wrong color).
Wait, a pile can have $m$ balls but be mixed.
If $v$ is full ($m$ balls) but wrong color on top, can we put a correct color on top?
NO. Capacity $m$ prevents adding a ball.
So if we have a "full" pile $v$ that needs to be emptied to sort it, we MUST move balls FROM $v$ TO somewhere else.
This implies we need a spare pile.
If all $n+1$ piles are full (impossible as max balls $nm$, so $n$ piles full leaves 1 empty? No, $nm$ balls fit in $n$ piles of size $m$).
Wait, we have $nm$ balls. We have $n+1$ piles.
Max capacity of one pile is $m$.
Total capacity $m(n+1)$.
We have $nm$ balls.
So at most $n$ piles can be full. At least one pile must be empty.
Wait, if $n$ piles are full, we have $nm$ balls. The $(n+1)$-th is empty.
So we NEVER have a situation where all $n+1$ piles are full.
There will ALWAYS be at least one pile that is NOT full.
Wait, if we move a ball from a full pile to a non-full pile, the count of full piles decreases (if dest becomes full) or stays same.
Wait, if we move to an empty pile, full count decreases (source becomes non-full, dest becomes size 1).
Actually, if we have $k$ full piles and $n+1-k$ non-full.
We want to move balls to fill them up correctly.
The issue is: If a full pile is incorrect, we must empty it.
If all piles that are "destined" for certain colors are full but incorrect, we are in trouble?
Example: Color 1 destined for Pole 1. Pole 1 has balls {2, 2, 2} (size 3, full).
We must move these out. Where? To Pole 2 (destined for Color 2) or Pole 3 (destined for Color 3) or Pole 4 (empty).
But Pole 2 might be full of {2, 2, 2}.
If Pole 2 is full of correct balls, we shouldn't mess it up.
But we can use Pole 3 or Pole 4 as buffer.
If Pole 3 and 4 are full or empty?
Basically, we have plenty of room.
Algorithm:
Maintain the target assignment: `target_pole[i]` = which pole should hold ball of color `i`.
Initially we can set `target_pole[i] = i` for all colors `i` present in the input. (Since colors are 1..n, we can map Color $c$ to Pole $c$? No, input colors are just IDs).
Let's assume colors are numbered $1 \dots n$. We will assign Pole $1 \dots n$ to hold colors $1 \dots n$ respectively. (Wait, the input gives us the balls. The colors are identifiers).
Yes, we can arbitrarily decide: "Pole $i$ will eventually contain all balls of color $i$".
Wait, Pole indices are $1 \dots n+1$.
Pole $n+1$ is initially empty. We can use it as the destination for Color $n+1$? No, only $n$ colors.
So we will have $n$ occupied poles and 1 empty.
Which poles get which colors? It doesn't matter.
Let's just say we want Pole $i$ (for $i=1..n$) to hold Color $i$.
But wait, we don't know which color will end up in Pole $i$. We can choose.
However, maybe it's easier to just iterate through the balls in the system.
Here is a generic solver for this type of problem:
We can repeatedly find a pair $(u, v)$ such that:
1. Pole $u$ is not empty.
2. Pole $v$ is not full (can receive).
3. The top ball of $u$ is of color $C$.
4. We want to move ball $C$ to some pole $P$.
If we chose to move it to $P$, then $v$ should be $P$.
So if $v$ is assigned to color $C$, move $u \to v$.
If $v$ is not assigned to color $C$, moving $C$ to $v$ puts it in a "bad" spot.
Unless $v$ is a temporary storage.
But we want to MINIMIZE operations.
Moving to a temp storage is 1 op. Moving to final is 1 op.
Ideally we want to move to final.
So: Look for any full/non-full poles.
Find a ball at top of $u$ that belongs to a "correct" pole $P$ (where $P$ is intended for Color $C$).
Try to move it to $P$.
If $P$ is full, can't move.
If $P$ is not full, move!
If $P$ is full, we can't move it to $P$ directly.
This implies we need to sort.
But wait, if $P$ is full, it means it already has $m$ balls. If it was assigned to Color $C$, and it's full, does it have Color $C$ balls? Not necessarily.
If $P$ is full and assigned to Color $C$, but has wrong ball on top, then $P$ is definitely not sorted.
But since it's full, we can't put the right ball there.
We MUST remove the wrong ball from $P$.
Move it to some buffer $W$.
Then try to put the correct ball there.
This suggests a bubble-sort like pass.
But doing this naively might take too many ops.
Given the limit 2,000,000, and total balls $\approx 20,000$, we can do roughly 100 ops per ball.
We can implement a "Cycle Sort" or similar.
Actually, the optimal solution (or close to it) can be found by simulating the process greedily.
However, there is a very simple algorithm that works:
Since $N \ge 2$, we can pick two poles.
Let's say Pole 1 and Pole 2.
Move top of Pole 1 to Pole $N+1$.
Move top of Pole 2 to Pole 1.
...
This shuffles.
But let's look at the sample code structure.
Input reading is straightforward.
We need to output the operations.
Let's write a function `solve()`.
We can just run a simulation loop that finds valid moves.
Heuristic:
Always try to move a ball of Color $C$ to a pole assigned to Color $C$.
If the target pole is full, move it to a temporary pole? Or move a ball OUT of target to make room?
Actually, if we have a ball of Color $C$ at top of $u$, and Target($C$) = $v$.
If $v$ is not full, move $u \to v$.
If $v$ is full, this means $v$ is "blocked".
If $v$ is blocked, we must empty $v$.
But wait, if $v$ is full, it has $m$ balls. If it has some Color $C'$ balls, maybe we should just move those balls out?
Wait, if $v$ is assigned to Color $C$, and it's full, we MUST be careful.
Actually, if $v$ is assigned to Color $C$, and we find a ball of Color $C$ on top of $u$, it means we have a potential to sort $v$.
But $v$ is full. This implies $v$ has reached its capacity. If it contains balls of Color $C$ (correct), then it is sorted (if all are $C$). If it contains mixed balls, it is not sorted.
But we cannot add more balls.
So if $v$ is full, we must have already put $m$ balls into it.
If they are all Color $C$, we are done with $v$.
If not, then the process is messed up.
So, if $v$ is assigned to $C$ and full, we should check if it's good.
Wait, if we assign dynamically.
Better approach:
We have $N+1$ poles.
We can simply rotate the balls.
Wait, let's look at the complexity.
$N, M \le 50, 400$.
Operations: $2 \cdot 10^6$.
Maybe we can implement a "Layer-by-Layer" construction.
We process height $k=1$ to $m$ (from top).
Actually, top is easiest.
Let's track which pole is supposed to hold which ball at height $k$.
Wait, this is complex because we don't know the final order of colors on a pole.
Wait, "final distribution of colors to poles does not matter".
But "all balls of the same color are on the same pole".
This implies each pole contains balls of ONE color.
So if Pole 1 contains Color 1, it has 3 balls of Color 1 (in sample).
The order of balls of the same color in the stack doesn't matter (they are identical).
So Pole $P$ must contain only balls of Color $C$.
This simplifies things greatly.
We just need to gather all Color $X$ into one pile.
Let's just iterate $X = 1 \dots n$.
For each color $X$:
  Find where balls of Color $X$ are.
  Move them all to a designated pole (say Pole $P_X$).
Wait, if we move them one by one, we might disturb balls of other colors.
Also, we must respect capacity.
But we have empty space (initially $P_{N+1}$).
Also we can designate a pole to be the "home" for Color $X$.
Let's just say we map Color $X \to$ Pole $X$. (Since colors are $1..n$, and poles $1..n$ exist. Pole $N+1$ is spare).
Wait, can we just use Pole $1$ for Color 1, Pole $2$ for Color 2?
Yes.
So Target: Pole $i$ should contain all balls of Color $i$.
Wait, what if Color $i$ is missing on Pole $i$?
Pole $i$ initially has balls of mixed colors.
So we must bring balls from other poles.
Since we have limited moves, we should try to avoid moving a ball twice.
Is there a strategy that ensures minimal moves?
Wait, "minimize the number of operations".
But "no more than 2,000,000".
Since $N, M$ are small, minimizing is hard if we don't know the lower bound.
But maybe any greedy solution that sorts is fast enough?
Actually, if we just move balls to the correct pile whenever possible, it should converge.
However, we have to be careful about the "Full Pile" constraint.
If Destination is full, we can't put.
If Source is full, we can't take from it? No, we can take from it as long as it has $\ge 1$.
The constraint is only on DESTINATION.
So, if we want to move Color $X$ ball to Pole $P_X$.
If $P_X$ is full, we are stuck (unless $P_X$ is already sorted).
Wait, if $P_X$ is full and we want to move Color $X$ to it, it implies we want to increase size of $P_X$? No, if it's full, size is $m$.
So if $P_X$ is full, we CANNOT add any more balls.
This means $P_X$ must already contain $m$ balls.
Since we want $P_X$ to contain ONLY Color $X$, all $m$ balls currently on $P_X$ must be Color $X$.
If any ball on $P_X$ is NOT Color $X$, then $P_X$ is "bad".
But we can't remove balls from $P_X$ if we want to add balls?
Actually, we CAN remove balls. Removing ball decreases size. Adding ball increases size.
So we can take a "bad" ball out of $P_X$, put it in a buffer, and then later replace it with a "good" ball?
Yes.
So if $P_X$ is full and contains a bad ball (wrong color on top):
1. Move the bad ball to a temporary pile (buffer).
2. Eventually put correct ball into $P_X$.
Wait, if $P_X$ becomes smaller, we can receive balls.
So the strategy is:
Maintain "Target Piles".
Assign Color $c$ to Pole $c$ (or just distinct permutation).
We want to sort all poles $1..n$ into monochromatic.
Loop:
  Scan all piles. Find the top ball of some pile $u$.
  Let its color be $C$.
  Check where $C$ belongs. Suppose it belongs to Pole $T_C$.
  Case 1: $u == T_C$.
    The ball is in its home.
    We should probably LEAVE IT THERE if possible? Or maybe move it if needed?
    Since it's the top, it blocks access to balls underneath.
    If the ball underneath is also $C$, good.
    If it's not, we need to remove this top ball. But it's already home.
    So if $u$ is destined for $C$, and top is $C$, we are happy with this position *temporarily*.
    If $u$ is not full, we can leave it. If full, we can't put more.
    Ideally, we want to keep balls of Color $C$ on $T_C$.
  Case 2: $u != T_C$.
    Ball $C$ is at $u$. It needs to go to $T_C$.
    If $T_C$ is NOT full:
      Move $u \to T_C$.
      (We just move it closer to destination).
    If $T_C$ IS full:
      We need to make space at $T_C$.
      This implies $T_C$ has a wrong ball blocking? Or just full of correct balls?
      If $T_C$ is full, it must have $m$ balls.
      If it has $m$ balls of Color $C$, we are DONE with $C$ at $T_C$.
        Wait, if $T_C$ is full of $C$, then $T_C$ is sorted.
        But we are looking at $u$ which has Color $C$.
        If $T_C$ is full and sorted, and $u$ has Color $C$ (and $u \neq T_C$),
        then we have duplicate Color $C$ balls on different poles. Impossible.
        Total balls of $C$ is $m$. If $T_C$ has $m$ balls of $C$, then $u$ cannot have a ball of $C$ (since total count is $m$).
        So if $T_C$ is full and sorted, we can't have any other $C$.
      Thus, if $T_C$ is full and NOT sorted (contains some non-$C$ ball), it's "blocking".
      If $T_C$ is full and sorted, then we have a contradiction with existence of $C$ at $u$.
      So $T_C$ cannot be full and sorted while there is another $C$ somewhere else.
      Therefore, if $T_C$ is full, it MUST contain at least one non-$C$ ball.
      So we must move a ball OUT of $T_C$.
      Which one? We can move the top one (it blocks others).
      If top of $T_C$ is wrong color $D$, move it to somewhere.
      Wait, moving top of $T_C$ creates space.
      Then we can move our ball $C$ to $T_C$.
      But we moved a ball out (of color $D$). Now $T_C$ has a vacancy.
      Does $D$ go home?
      Maybe $D$ belongs to $T_D$. If we move $D$ to $T_D$ immediately?
      But $T_D$ might be full.
      This looks like we just need to find a cycle.
      
      Algorithm refined:
      Assign each Color $c$ to a pole $P_c$. (Map $1..n$ to $1..n$).
      While not all poles sorted (i.e. for every pole $p$, it has size $m$ and consists of color $c$ where $P_c = p$):
        Find a move.
        Iterate over all poles $u=1..n+1$.
        If $u$ is empty, continue.
        Let $top = pile[u].front()$ (Top ball color).
        Let $target = P[top]$.
        If $u == target$:
          Ball is home. Good.
          But wait, if $u$ is full, and ball is correct, we ignore it (maybe?).
          But if $u$ is NOT full, maybe we need to put more balls of $top$ here.
          Actually, if $u==target$ and $top$ is correct, this ball is part of the solution.
          However, it might be covering a ball that needs to be moved out.
          Wait, if $u$ has top $top$ (correct), and underneath is wrong, we need to move the top $top$ away to access the wrong one.
          So we shouldn't leave it if it covers a bad ball.
          But we can detect that only if we look inside.
          If we don't know what's underneath, we might blindly move top.
          If we move top away, it's still "home". We can put it back later?
          Maybe we should move it to a buffer to sort the rest, then put it back.
        Else ($u \neq target$):
          Ball needs to move to $target$.
          Check if $target$ is full.
          If not full: Move $u \to target$.
          If full:
            We need to empty $target$.
            Move top of $target$ to somewhere.
            Where? To $u$ (the current source)? Or a buffer?
            Actually, if we move top of $target$ to $u$, we might merge stacks?
            Wait, if $u$ is full?
            If $u$ is full, we can't receive.
            
      Wait, $N \ge 2$. We have at least 2 poles.
      This suggests we can swap balls between 2 piles.
      Also we have Pole $N+1$.
      
      Key Insight:
      If we have $u \neq target$, and $target$ is full.
      Move top of $target$ to $N+1$ (Buffer).
      Then move ball from $u$ to $target$.
      Then move ball from $N+1$ to $target$ (maybe? No, $N+1$ is buffer).
      Wait, if we move ball from $target$ to $N+1$, $target$ becomes less full.
      But $N+1$ accumulates balls.
      We need to clean $N+1$ eventually.
      
      Is there a simpler loop?
      Since $N, M$ are small, we can simulate the "Bubble Sort" of balls.
      Specifically:
      We want to sort the balls.
      Let's treat the state as $N$ lists.
      We can try to fix the top element of each pole.
      Actually, there is a much simpler algorithm provided in similar contexts:
      Just repeat:
      Pick any pole $u$ with a ball.
      Pick a pole $v$ with space.
      If top of $u$ belongs to $v$ (based on pre-assigned mapping), move $u \to v$.
      What if $v$ is full?
      Then we skip $v$.
      We keep trying until we find a valid move.
      If no valid move improves the state (i.e., we can't put ball in its home), we must move it to a buffer.
      This is essentially a greedy placement.
      
      Let's formalize the Greedy Placement.
      Pre-assign colors to poles:
      Since there are $n$ colors and $n$ initial poles, and 1 empty.
      We can assign Color $c$ to Pole $c$ (for $c=1..n$).
      Pole $n+1$ is buffer.
      Condition for "Valid Move":
      Source $u$ has $\ge 1$ ball.
      Dest $v$ has $< m$ balls.
      Ball $b$ on top of $u$ has color $c$.
      Target for $c$ is $Target(c)$.
      If $v == Target(c)$, move! (Greedy).
      If we encounter a full $Target(c)$, we can't move to $Target(c)$.
      If we can't move to $Target(c)$, can we move to buffer $N+1$?
      If we move to buffer, we are storing it.
      Later, when buffer is empty or full, we can move it to $Target(c)$?
      Actually, we should prioritize moving to $Target(c)$ if possible.
      If $Target(c)$ is full, we must move to Buffer?
      Or maybe we should just move any ball to any empty/full-suitable pole?
      No, random moves might not sort.
      
      Wait, what if we just perform a "Rotate" of all balls?
      Suppose we have $n$ colors.
      We can shift the content of poles?
      Consider the sequence of colors in the input.
      Let's just use a simulation loop with priority:
      1. Try to move ball of Color $c$ to Pole $Target(c)$.
         Only valid if $Target(c)$ is not full.
      2. If not possible, move ball of Color $c$ to Pole $N+1$.
         Only valid if $N+1$ is not full.
         (Actually, if $Target(c)$ is full, maybe $N+1$ is also full? No, max capacity $N+1$. Usually there's space).
         But we need to manage the buffer.
      
      Wait, if $Target(c)$ is full, and we have a ball $c$ at $u$, it means $u$ is not empty.
      If $Target(c)$ is full, then it contains $m$ balls.
      Since we want $Target(c)$ to have Color $c$, and it has $m$ balls.
      As argued before, if it is full and contains Color $c$ somewhere, it must be mixed (since total count of $c$ is $m$, if all on $Target(c)$, no $c$ at $u$).
      So if $Target(c)$ is full, it contains non-$c$ balls.
      If we have a $c$ at $u$, we MUST move it to $Target(c)$.
      But $Target(c)$ is full.
      So we MUST empty $Target(c)$ first.
      We can't empty $Target(c)$ unless we move its top ball somewhere.
      If we move top ball (color $d$) from $Target(c)$ to buffer.
      Then move $c$ to $Target(c)$.
      Then put $d$ into $Target(d)$.
      This works nicely. It's like resolving conflicts.
      
      So the logic is:
      Loop:
        Find any pole $u$ with a top ball of Color $c$.
        Let $t = Target(c)$.
        If $u == t$:
           Do nothing (or handle if it covers wrong balls? Harder).
           Maybe we should assume if $u==t$ and top is correct, it's stable.
        Else ($u \neq t$):
           If $t$ is not full:
             Move $u \to t$.
           Else ($t$ is full):
             We need to make space.
             The only way to make space in $t$ is to move a ball FROM $t$.
             But $t$ is full.
             Let's pick the top ball of $t$. Let it be $d$.
             Move $t \to N+1$ (buffer).
             Now $t$ is not full.
             Move $u \to t$.
             Now we placed $c$.
             We moved $d$ to buffer.
             Repeat logic for $d$.
      
      Wait, this looks like a recursive chain.
      $u \to t$ (needs space), $t \to Buf$ (space created), $Buf \to t$?
      Wait, if we move $t \to Buf$, we lost the ball $d$ from $t$.
      It is now in Buf.
      We should probably move $d$ to $Target(d)$.
      So if $Target(d)$ is full, recurse?
      This could cause long chains or loops if not managed.
      With $N, M$ small, recursion depth is not an issue?
      Maybe. But we need to be efficient.
      Actually, we can just collect balls in a buffer?
      No, buffer has capacity limit.
      But wait, we can use multiple buffers. $N+1$ is one. Others are full initially.
      Wait, if we start with all full, we MUST move to $N+1$.
      Once $N+1$ gets balls, it can send them back.
      Actually, we can treat $N+1$ as the ONLY buffer.
      We can keep moving balls to $N+1$ to sort them?
      
      Wait, there is a very standard solution for this problem:
      "Since we have $N$ piles of $M$ balls and $N$ colors, we can treat this as sorting $M$ rows?"
      Actually, looking at the scoring and constraints, it is likely an interactive-like or constructive problem.
      Let's look at the sample solution again.
      Moves: 1->3, 2->3, 2->3, 3->1, 3->2, 3->2.
      Notice the pattern:
      Collect balls to 3.
      Distribute from 3.
      This suggests a "Hub and Spoke" model.
      Pole $N+1$ acts as the hub.
      Phase 1: Move all balls from Poles $1..N$ to Pole $N+1$.
      If we do this blindly, Pole $N+1$ becomes full ($m$ balls).
      We cannot move MORE.
      So we can't move all balls.
      Wait, in sample: $m=3$.
      P1: 2 balls moved to P3.
      P2: 2 balls moved to P3.
      Total 4 balls in P3? No.
      Moves:
      1 3 (1st ball of P1? No, top ball).
      P1 top was 2. Moved to P3. P3 has {2}.
      2 3. P2 top was 2. Moved to P3. P3 has {2, 2}.
      2 3. P2 top was 1. Moved to P3. P3 has {2, 2, 1}.
      Wait, P2 top was 1? Initial P2: 2, 1, 2 (bottom->top).
      Top is 2. Move 2. P2 becomes {2, 1}. Top is 1.
      Next op 2 3 moves 1.
      So yes, we moved 2 balls from P1 and 2 balls from P2.
      Total 4 balls.
      $m=3$.
      P3 capacity for receiving is $m-1$ (so max size $m$).
      So P3 has 3 balls. It is full.
      We stop moving TO P3.
      Then we move FROM P3.
      3 1 (moves 1). P1 becomes {1, 1, 1}.
      3 2 (moves 2). P2 becomes {2, 2}.
      3 2 (moves 2). P2 becomes {2, 2, 2}.
      
      This looks like:
      1. Identify which color goes where?
      In sample, P1 gets {1,1,1}, P2 gets {2,2,2}.
      The balls of Color 1 and 2 are separated.
      But we moved them to P3 indiscriminately?
      We just collected them.
      Why did we pick P3? Because it was empty.
      Why didn't we move P1 to P2 or P2 to P1?
      Because they were full.
      So Step 1: Use P3 to collect excess balls?
      But which balls?
      It seems we can just move top balls to P3 until P3 is full.
      Then use P3 to distribute.
      
      Is it guaranteed that collecting top balls works?
      If we move balls from $1..N$ to $N+1$, we are essentially peeling off layers.
      Since we peel from top, we access higher layers first.
      Eventually we clear the stack.
      
      Generalized Algorithm:
      1. Move all balls from Poles $1..N$ to Pole $N+1$?
         Wait, Pole $N+1$ fills up after $m$ balls.
         So we move balls from $1..N$ to $N+1$ until $N+1$ is full.
         BUT, if $N+1$ fills up, we might miss some balls that we needed to move earlier?
         Wait, if $N+1$ fills up, it means we collected $m$ balls.
         We can't put more.
         So we have a bottleneck.
         
      Let's rethink.
      Since we want to separate colors, and we have $N$ colors.
      Maybe we can just process colors one by one.
      But "Process Color 1" might require moving balls of other colors.
      
      Wait, there's a constraint "You must use at most 2,000,000 operations."
      With $N, M$, total balls $20,000$.
      Even if we shuffle everything to $N+1$ and back, that's $O(NM)$ ops.
      $NM = 20,000$.
      Shuffling each ball 10 times is 200,000.
      So efficiency is not critical as long as complexity is polynomial.
      $O(N \cdot M \cdot \text{something})$.
      
      Proposed Algorithm:
      We have $N+1$ poles.
      We can arbitrarily map colors to poles.
      Actually, we don't need to map beforehand.
      We just want to group them.
      
      Let's try to simulate a simple "Sort" strategy:
      Since we have plenty of ops, we can just move the top ball of any pile to the "best" destination.
      But which is best?
      The best destination is the one matching the ball's color.
      Let's maintain the `cnt[col]` for each color? No.
      Let's maintain a queue of balls waiting to be placed? No.
      
      Actually, here is a very robust strategy:
      Cycle all balls.
      For every color $c \in 1..n$:
        Move all balls of color $c$ to a dedicated pole (say Pole $c$).
        Wait, if we do this sequentially.
        For Color 1:
          Collect all balls of Color 1.
          Where are they? Distributed across $1..n+1$.
          Move them to Pole 1.
          To do this efficiently:
          Look at top of all piles.
          If top is Color 1, move to Pole 1.
          But if Pole 1 is full (initially), we can't.
          So we must move balls FROM Pole 1.
          If we move a non-1 ball FROM Pole 1, we might uncover a 1?
          If we move a 1 FROM Pole 1, we have nowhere to put it (if 1 is full).
          But Pole 1 *is* the target for 1.
          So if Pole 1 has a 1, and we want to put 1s there.
          If it's full, it's full of 1s? No, maybe mixed.
          If it has a 1 on top, we should KEEP it there.
          Wait, if it has a 1 on top, but underneath is a 2.
          We can't access 2.
          But we don't need to access 2 for Color 1.
          We just need to place Color 1s there.
          Wait, if we have a 2 on top of 1, we can't see the 1.
          But we know 2 is Color 2. It belongs to Pole 2.
          So we should move 2 to Pole 2.
          This clears the stack!
          So, for each ball on top:
            If color is $c$:
              Move to Pole $c$.
          If Pole $c$ is full:
            Then we must move a ball FROM Pole $c$ to somewhere.
            Which one?
            Maybe move the ball from Pole $c$ to a buffer?
            Or cycle: $c \to Buf \to c$?
            Wait, if Pole $c$ is full and has a ball $c$ on top.
            If we want to move it to Pole $c$, it is ALREADY there.
            But if Pole $c$ has a ball $d \neq c$ on top.
            We move $d$ to Pole $d$.
            If Pole $d$ is full, move its ball $e$ to Pole $e$.
            This creates a chain.
            This chain must terminate.
            Since there are finite states.
            
            Wait, if we just follow the chain, we might cycle forever?
            Example: Pole 1 has top 2. Pole 2 has top 1.
            1->2, 2->1. Cycle.
            We need a break. Pole 3 (Buffer).
            Move 2 to 3. Then move 1 to 2.
            Then move 2 to 1.
            This is a standard "unstable sort" logic.
            With Pole $N+1$ acting as sink/buffer.
            
      Let's implement this logic:
      Loop until no poles have wrong balls?
      Or just run for enough iterations.
      Priority:
      1. Try to move a ball $c$ from $u$ to Pole $P_c$ (where $P_c$ is assigned pole for color $c$).
      2. If $P_c$ is full, then we can't move there.
         Instead, we move the blocking ball (top of $P_c$) to $N+1$.
         Or just move the ball $c$ to $N+1$?
         If we move $c$ to $N+1$, we delay it.
      
      Wait, the "Chain" logic is good.
      If we want to move $c$ to $P_c$, and $P_c$ is full.
      Then $P_c$ contains a ball that needs to be moved out (otherwise it would be sorted, but we have another $c$ somewhere).
      If $P_c$ has ball $d$ on top. Move $d$ to $N+1$.
      Then $P_c$ is empty-ish, put $c$ there.
      But $d$ is in $N+1$.
      We need to move $d$ to $P_d$.
      But maybe $P_d$ is also full?
      Then move $e$ from $P_d$ to $N+1$.
      Recurse.
      
      Since $N$ is small, this recursion/iteration won't be deep or long?
      But we must ensure termination.
      If we just cycle $1 \to 2 \to 1$, we need to break.
      Breaking via $N+1$ is key.
      Since $N+1$ has capacity, but we shouldn't fill it completely.
      Actually, since we have $N$ colors, and $N+1$ poles.
      We can use $N+1$ to absorb one cycle.
      
      Let's simplify.
      Just pick the top ball of ANY non-empty pole.
      Move it to the pole corresponding to its color.
      If destination is full, move the top ball of destination to the *same* source?
      No, that's swapping.
      Better: move top ball of destination to $N+1$.
      Then move the first ball to destination.
      Wait, this increases ops.
      But is it safe?
      Yes, if we move $c$ to $P_c$, and $P_c$ was blocking with $d$.
      Now $P_c$ has $c$. $d$ is at $N+1$.
      We process $d$ next. Move to $P_d$.
      If $P_d$ is full, move its blocker to $N+1$.
      But $N+1$ might fill up.
      However, total operations limit is large.
      We can just run a greedy simulation.
      
      Simulation details:
      Data structures:
      `vector<int> poles[n+1];` (using 0-based or 1-based. Let's use 1-based `poles[1..n+1]`)
      Function `move(u, v)`: pop from `poles[u]`, push to `poles[v]`.
      Record op.
      
      Main loop:
      Iterate $u$ from $1$ to $n+1$.
        If `poles[u]` empty, continue.
        Get color $c = poles[u].back()$.
        Determine target pole $T$ for color $c$.
        (How to determine $T$? We can pre-calculate counts. Since we know final state is monochromatic, and we have $n$ colors, we can just map color $c$ to Pole $c$ (renaming colors if needed? No, colors are $1..n$. Poles $1..n$. Map Color $c$ to Pole $c$.)
        Wait, if input has colors $1..n$ and we have poles $1..n$.
        We can just say Color $c$ goes to Pole $c$.
        Exception: Pole $n+1$ is empty. Can we use it?
        If we map Color $c \to$ Pole $c$, then Pole $n+1$ remains empty.
        Is this optimal? Maybe.
        Does it matter? "Final distribution... does not matter".
        So assigning Color $c$ to Pole $c$ is valid.
        
        If $u == T$:
           Do nothing?
           Maybe move if $u$ has other balls?
           Actually if $u$ has correct ball, it's "sorted" locally.
           We only care if it blocks a bad ball?
           We don't know what's underneath.
           But maybe we shouldn't touch sorted parts?
           Let's assume if $u=T$ and top is $c$, it's stable.
        Else ($u \neq T$):
           If `poles[T].size() < m`:
               Move $u \to T$.
           Else (`poles[T].size() == m`):
               // T is full. Must make space.
               // Move top of T to somewhere.
               // Where? To N+1?
               // Wait, if we move top of T (say color $d$) to N+1.
               // Then move $u \to T$.
               // But what about $d$?
               // $d$ should go to Pole $d$.
               // If Pole $d$ is full, we have a problem?
               // We can handle this recursively?
               // But to avoid infinite recursion, maybe use a specific buffer strategy.
               
      Wait, we can improve this.
      If $T$ is full, we can't move to $T$.
      If we have $N+1$ poles, and $T$ is one of them.
      We can just move the blocker to $N+1$ (temporary).
      But $N+1$ might fill up.
      If $N+1$ fills up, it's full. Then we have a deadlock (no destination).
      But wait, if all $N+1$ are full, we are stuck?
      No, if $N+1$ is full, it means we have collected $m$ balls there.
      But we have $nm$ balls total.
      If all $N$ poles are full, we are stuck initially.
      Wait, initially poles $1..N$ are full.
      We MUST move to $N+1$.
      If we fill $N+1$ to $m$, it becomes full.
      Then we can't move to ANY pole?
      Wait, if all poles are full ($N+1$ poles full), we have $m(N+1)$ balls.
      But we only have $nm$ balls.
      So it is IMPOSSIBLE for ALL $N+1$ poles to be full.
      Because $nm < m(n+1)$.
      So at least one pole has $< m$ balls.
      Wait, we start with $N$ full, 1 empty.
      So we can move to $N+1$.
      But once $N+1$ becomes full, and we emptied a pole $1$, we have $N-1$ full, 1 full ($N+1$), 1 empty ($1$).
      Wait, moving from full to empty:
      Start: Full($1..n$), Empty($n+1$).
      Move $1 \to n+1$: Full($2..n$), Partial($n+1$), Empty($1$).
      So we always have an empty/partial pole.
      Wait, if we move from Full to Partial:
      Source becomes Partial. Dest becomes larger.
      So we always have space.
      But the logic `If poles[T].size() < m` allows moving.
      The only restriction is `poles[T].size() < m`.
      Wait, what if $T$ is full?
      Then we can't move to $T$.
      But if $T$ is full, and it has a wrong ball on top.
      We need to remove it.
      Remove from $T$ to where?
      To a pole that accepts it? Or just any empty slot.
      If we move to an empty slot, we are making progress.
      But if we use $N+1$ as buffer.
      And if $N+1$ is full... wait, as proven, we can never fill all $N+1$ poles.
      So there is ALWAYS a pole with space.
      So we can always move the blocker from $T$ to *any* pole with space.
      Which one? The one that is currently empty/partial and not $T$.
      But we must pick a valid destination.
      Let's pick the one with smallest height? Or just any.
      Or better, we can use the "Next" pole in the assignment.
      
      Let's refine:
      Assign Color $c$ to Pole $P_c$.
      (Pole indices $1..n+1$).
      Wait, colors are $1..n$. Poles $1..n+1$.
      Let's map Color $c$ to Pole $c$ (since poles $1..n$ exist).
      Pole $n+1$ is unused (empty).
      
      Loop:
        Scan all piles $u=1..n+1$.
        If `poles[u]` empty, skip.
        Let $c$ be top color.
        If $u == P_c$:
           If `poles[u]` is full:
              // It's good, but we might need to remove top to access below?
              // Or just skip?
              // If we skip, we leave it alone.
              // But if it has mixed balls, we must eventually sort it.
              // But we can't access below.
              // However, if we just leave it, we assume it's fine.
              // If it's sorted, fine. If mixed, we fail.
              // So we must not skip if it's mixed.
              // But detecting mix requires knowledge of content.
              // Maybe we should always process.
              // If $u$ is full, and we can't remove top (since we want to keep it?)
              // Actually, if $u$ is full, we can't add to it.
              // If it has correct top, we might want to move it out to make room? No.
              // If it has correct top, and we want to put *more* correct balls, impossible (full).
              // So if $u$ is full and top is correct, we hope it stays.
              // But if underlying is wrong...
              // Maybe we should assume the input can be solved.
              // Let's just try to move to target.
           Skip. (Optimization: don't touch if sorted).
           
        Else ($u \neq P_c$):
           Target $T = P_c$.
           If `poles[T]` is not full:
              Move $u \to T$.
           Else ($T$ is full):
              // Need space.
              // Find a pole with space.
              // Candidates: Any pole with size $< m$.
              // Let's find $S$ (a pole with size $< m$).
              // Move $T \to S$. (Remove blocker).
              // Move $u \to T$. (Put correct ball).
              // Now we have moved $blocker$ to $S$.
              // Continue loop.
              
      Wait, this creates a dependency.
      If we move $blocker$ to $S$, $S$ becomes fuller.
      Is $S$ a safe bet?
      Maybe we should prioritize moving $blocker$ to $N+1$?
      Or if $N+1$ is full?
      Wait, we proved not all can be full.
      So there exists some $S$ with space.
      Just pick $S$.
      Wait, if we cycle moves $T \to S \to U \to T$, we waste ops.
      We want to direct flow.
      Better strategy:
      Find a valid move that improves the score.
      Score could be "number of sorted poles"? Or "number of balls in correct position".
      Since we want to minimize ops, we shouldn't undo work.
      But finding the optimal is hard.
      Greedy with a "Buffer" approach seems sufficient.
      
      Let's implement:
      Map Color $c$ to Pole $c$ (for $c=1..n$).
      Loop `while changed`:
         Changed = false.
         For $u$ in $1..n+1$:
            If `poles[u]` empty continue.
            $c = poles[u].back()$.
            Target $T$.
            If $T$ is full:
               // Blocker at $T$.
               // Find a pole $S$ with space.
               // Move $T \to S$.
               // Mark $T$ not full?
               // Update counts.
               // $u \to T$ might be valid now.
               // We do this move $T \to S$.
               // Now loop continues.
               // Note: $T \to S$ is a move.
               // Should we record it? Yes.
               // Is it beneficial? It exposes $S$ to receive $blocker$.
               // And frees $T$.
               // If we pick $S$ wisely (e.g. empty pole), good.
            Else:
               // Space available.
               // Move $u \to T$.
               // Increment correct balls count.
               
      Wait, the `while changed` with nested loop might be slow?
      $2,000,000$ ops limit.
      Inside loop, we might scan $N+1$.
      If we do many passes, it's $O(K \cdot N)$.
      With $K$ large, maybe fine.
      However, we can optimize:
      Instead of scanning all, just find a specific violation.
      Actually, there's a risk of getting stuck in loops if not careful.
      E.g. $u$ has $c$ needs $T$. $T$ full, has $d$.
      Move $T \to S$. $T$ free. Move $u \to T$.
      Now $S$ has $d$. $d$ needs $P_d$.
      If $P_d$ is full...
      If we keep picking random $S$, we might just cycle.
      Better $S$ choice:
      If we can find a pole $P_d$ with space, move $d$ there directly?
      No, we don't have control over $d$'s position relative to $T$ yet. $d$ is at $T$'s top.
      So we MUST move $d$ first.
      So we move $d$ to *somewhere*.
      Ideally to $P_d$ if it has space.
      But $P_d$ is full?
      If $P_d$ is full, then we have a chain.
      If we have a chain of blockers, we can traverse it.
      Chain: $T \to d$ needs $P_d$ (full, blocker $e$) $\to P_d \to e$ needs $P_e$ (full, blocker $f$)...
      This chain must end in a pole that is NOT full.
      Since not all are full, the chain terminates at some $S$.
      So:
      1. Identify the ball $c$ at $u$ needing $T$.
      2. Check if $T$ is full.
         If not, $u \to T$. Done.
         If yes, $T$ is full. We need to resolve $T$.
         Look at top of $T$, say $d$.
         Check if $P_d$ is full.
         If $P_d$ not full, move $d \to P_d$. Then move $c \to T$?
         Wait, we can't do both in one atomic step.
         So we do $T \to P_d$ (move blocker).
         Then $u \to T$.
         Then process $d$ (now at $P_d$). If $P_d$ becomes full? No, it has space.
         Wait, if $P_d$ wasn't full, it accepts $d$.
         Does $P_d$ become full? Depends on remaining balls.
         If it becomes full, fine.
      3. If $P_d$ IS full.
         Look at top of $P_d$, say $e$.
         Check if $P_e$ is full.
         If $P_e$ is full, move $e \to P_e$ (recursive step).
         Essentially, we need to drain the chain until we hit an empty slot.
         Since we know at least one slot has space (globally), and we only move balls along pointers.
         Is it true that we can drain?
         Yes.
         So algorithm:
         Maintain state.
         While exists $u$ with top $c$ such that $u \neq P_c$:
            If $P_c$ is full:
                // Drain $P_c$.
                Identify chain starting at $P_c$ top.
                Trace chain until we find a pole $Z$ that can receive the current ball.
                Wait, tracing:
                Current ball is $d$ (top of $P_c$). Needs $P_d$.
                If $P_d$ has space -> Move $d \to P_d$. Break chain.
                If $P_d$ full -> $d$ blocks $P_d$.
                New ball $e$ (top of $P_d$). Needs $P_e$.
                If $P_e$ has space -> Move $e \to P_e$.
                If $P_e$ full -> ...
                Basically, we can just find the first node in the chain (starting from $P_c$ top) that has space at its target.
                Wait, this is circular.
                The targets are fixed ($P_x$).
                If $P_c$ is full, we move top ($d$) to... where?
                If we move $d$ to $P_d$, and $P_d$ is full, we don't move $d$ to $P_d$.
                We need to move $d$ to a location where it can sit safely?
                Or we can just rotate?
                Actually, if $P_c$ is full, it means we can't put $c$ there.
                Wait, if $P_c$ is full and has $d$ on top.
                If $d$ belongs to $P_c$ (i.e. $P_c$ is assigned to color $d$?), then we have consistency.
                But we assigned Color $x \to P_x$.
                So if $P_c$ is full, and top is $d$.
                If $d=c$, then $P_c$ is full of $c$. Sorted.
                But we are considering a ball $c$ at $u$ ($u \neq P_c$).
                If $P_c$ is full of $c$, then all $m$ balls of $c$ are there.
                So $u$ cannot have $c$.
                Contradiction.
                So if $P_c$ is full, it CANNOT be full of $c$.
                Thus $P_c$ contains some ball $d \neq c$ on top (or elsewhere).
                If $P_c$ is full, it has some "wrong" ball blocking it from accepting $c$.
                So we MUST remove $d$.
                Move $d$ to $P_d$?
                If $P_d$ is full, it also has a wrong ball $e$.
                Move $e$ to $P_e$?
                Since we have finite capacity and finite balls, this path must eventually lead to a pole that has space.
                But wait, "having space" means size $< m$.
                Is it guaranteed that there is a pole with space?
                Yes, globally.
                But we need to connect to it.
                If we just follow the "Needs" graph: $T \to P_{top(T)} \to P_{top(P_{top(T)})} \dots$
                This path follows the needs.
                Will it reach a sink?
                Not necessarily. It could cycle.
                $P_1 \to P_2 \to P_1$.
                $P_1$ full (top 2), $P_2$ full (top 1).
                If we try to put 1 in $P_1$. $P_1$ full (top 2). Move 2 to $P_2$.
                $P_2$ full (top 1). Move 1 to $P_1$.
                Cycle! $P_1 \leftrightarrow P_2$.
                We need to break cycles.
                To break cycles, we MUST use $N+1$.
                So if we hit a cycle, we divert to $N+1$.
                But how do we know if we hit a cycle?
                We can just check if the target is full.
                If we move $d$ from $P_c$ to $N+1$.
                Then we can put $c$ into $P_c$.
                But wait, if we move $d$ to $N+1$, $d$ is now in $N+1$.
                We should eventually move $d$ to $P_d$.
                If we move $d$ to $N+1$, we just pushed it back to queue?
                But $N+1$ is a sink.
                
      Okay, here's the plan:
      We will implement a simulation.
      Since $N, M$ are small, we can afford to move balls around.
      We assign Color $c$ to Pole $c$ for $c=1..n$.
      Pole $n+1$ is "Global Buffer".
      We run a loop:
      While true:
         Found_move = false.
         For each pole $u$ (1 to $n+1$):
             If `poles[u]` empty, continue.
             $c = poles[u].back()$.
             Target $T = c$. (Since Color $c$ maps to Pole $c$).
             
             If $u == T$:
                 continue. (Ball in correct place).
                 
             // Need to move $c$ to $T$.
             // Check if $T$ is available.
             // Available means: `poles[T]` has size $< m$.
             if `poles[T].size() < m`:
                 Move $u \to T$.
                 Found_move = true.
                 Record op.
                 // Reset loop (restart scan) to pick up new tops?
                 // Or continue?
                 // Continuation is faster.
                 // If we move, $T$ gets bigger. $u$ gets smaller.
                 // Top of $T$ is now $c$ (we just put it).
                 // So next iteration sees $T$ with $c$ on top.
                 // Next iter: $u=T$, $c$ matches $T$. Good.
                 // So we made progress.
             else:
                 // T is full. We must move the top ball of T.
                 // Who is top of T? Let it be $d$.
                 // We want to move $d$ somewhere.
                 // Where? Ideally to $P_d$.
                 // Or to $N+1$.
                 // Let's try to move $d$ to $P_d$.
                 // If $P_d$ is full, we move $P_d$'s top...
                 // This leads to chain resolution.
                 
                 // Actually, to simplify logic, let's just find ANY valid move that is not identity.
                 // A valid move is $src \to dest$ where $dest$ has space.
                 // We want to prefer moves that put balls into their Home.
                 // If we can't do that, any move to Home-less is okay.
                 
                 // Let's refine the priority.
                 // Iterate all pairs $(u, v)$ such that $u$ not empty, $v$ has space.
                 // Ball $b = poles[u].back()$. Target for $b$ is $b$ (Pole $b$).
                 // If $v == b$ (target matches destination), it's Priority 1.
                 // If $v != b$, it's Priority 0.
                 // Perform Highest Priority Move.
                 // If no Priority 1 moves possible, perform Priority 0 move.
                 
                 // Does Priority 0 help?
                 // It moves a ball out of way.
                 // But we might move balls randomly.
                 // However, since we have limited space, maybe we can't move to Target.
                 // We move to a buffer.
                 // Then later, Target frees up.
                 
                 // Implementation detail:
                 // Keep iterating. If we finish a pass with no moves, we are stuck?
                 // Wait, we proved we are never fully stuck.
                 // But we might need to break cycles.
                 // Random moves break cycles.
                 
                 // Let's implement:
                 // 1. Find move $u \to v$ where $poles[v].size() < m$.
                 // 2. Among candidates, prioritize $v$ where $poles[u].back() == v$.
                 // 3. If ties, pick any.
                 
                 // Wait, what if no such move exists?
                 // Means for every non-empty $u$, and every $v$ with space:
                 //   $poles[u].back() \neq v$.
                 // This means we can't put any ball into its home.
                 // But we MUST move.
                 // So pick any valid move.
                 
                 // Is this sufficient?
                 // Let's trace cycle breaking.
                 // $P_1$ full of $2$. $P_2$ full of $1$.
                 // Space in $P_1$ (assume size 0 for simplicity, actually $N=2, m=2$).
                 // Assume $P_3$ empty.
                 // Pairs $(u,v)$ with space:
                 // $1 \to 3$. Ball 2. Target 2. $v=3 \neq 2$. Low prio.
                 // $2 \to 3$. Ball 1. Target 1. $v=3 \neq 1$. Low prio.
                 // $1 \to 2$ (no space). $2 \to 1$ (no space).
                 // So we do $1 \to 3$. $P_1$ empty. $P_3$ has 2.
                 // Now $2 \to 1$. Ball 1. Target 1. $v=1$. High prio!
                 // Move $2 \to 1$. $P_2$ empty. $P_1$ has 1.
                 // Now $P_1$ has 1. $P_2$ has 2.
                 // $P_3$ has 2.
                 // $P_2$ (has 2) can move to 2? No, $P_2$ is empty.
                 // Wait, ball 1 is at $P_1$. Target 1. $P_1$ is its target.
                 // Ball 2 is at $P_2$ (wait, $P_2$ empty? No $P_2$ empty in my trace).
                 // Let's re-trace.
                 // Init: P1:{2}, P2:{1}, P3:{} (capacity 2).
                 // Move $1 \to 3$ (Ball 2).
                 // State: P1:{}, P2:{1}, P3:{2}.
                 // Valid moves: $2 \to 1$ (Ball 1 to P1). $1 == Target$. Move.
                 // State: P2:{}, P3:{2}, P1:{1}.
                 // Now P1 has 1. Correct.
                 // P3 has 2. Target 2 is P2. P2 empty.
                 // Move $3 \to 2$ (Ball 2).
                 // State: P1:{1}, P2:{2}, P3:{}.
                 // Done.
                 
                 // It works!
                 // Does it minimize operations?
                 // Seems greedy and effective.
                 
                 // Complexity?
                 // In each step we make at least one ball closer to its home?
                 // Not necessarily. Moving $1 \to 3$ moved 2 away from home (2!=3). Distance 1?
                 // Maybe distance metric is number of swaps.
                 // But since we prioritize correct homes, it converges.
                 // Worst case $O(NM)$.
                 // Constraints allow this.
                 
                 // One Edge Case:
                 // We have $N+1$ poles. $n$ colors.
                 // Poles $1..n$ assigned to colors $1..n$.
                 // But wait, $P_{n+1}$ is just a buffer.
                 // It doesn't have a color assigned.
                 // My logic assumes `Target(ball) = ball`.
                 // This means we assign Color $c$ to Pole $c$.
                 // Is this valid for all inputs?
                 // Input has colors $1..n$.
                 // We have poles $1..n$ available.
                 // Yes.
                 
                 // Wait, what if Pole $c$ is initially empty? (Not possible, 1..n full).
                 // What if we want to move a ball from $P_{n+1}$ to $P_c$?
                 // Logic handles it.
                 
                 // Another thing: `Find highest priority move`.
                 // We scan all $u, v$.
                 // Time per op: $(N+1)^2$.
                 // Total ops $\approx NM$.
                 // Total time $NM(N+1)^2 \approx 20000 \times 2500 \approx 5 \times 10^7$.
                 // Fits in 4 seconds (usually $10^8$ ops/sec).
                 
                 // Wait, optimization:
                 // We can scan only $u$ that have balls.
                 // And find a $v$ that is empty or partial.
                 // We can maintain list of available $v$'s?
                 // Or just iterate $u$, then iterate $v$.
                 // Optimization is good but likely not needed.
                 
                 // One detail:
                 // "minimize the number of operations".
                 // My greedy might not be optimal but should be within limit.
                 // But can we do better?
                 // Sample output had 6 ops for $2 \times 3$. $NM = 6$.
                 // My logic produced 6 ops?
                 // Let's trace sample with greedy.
                 // P1:{1,1,2}, P2:{2,1,2}, P3:{}.
                 // Target: 1->1, 2->2, 3->Empty.
                 // Ops:
                 // P1 top 2. Target 2 (P2).
                 // P2 top 2. Target 2 (P2).
                 // P2 top 2 matches target (P2)!
                 // But wait, P2 is Full (3). Cannot receive.
                 // P3 is Empty.
                 // Scanning pairs:
                 // u=2 (top 2). v=3 (space). Prio 0.
                 // u=1 (top 2). v=3 (space). Prio 0.
                 // Any Prio 1?
                 // u=2 top 2 -> Target 2. P2 is full. No.
                 // u=1 top 2 -> Target 2. P2 full. No.
                 // u=2 top 2 -> v=P2 full. No.
                 // u=2 top 2 -> v=P1? P1 full.
                 // u=1 top 2 -> v=P2 full.
                 // u=1 top 2 -> v=P3. Move 1->3.
                 // (Moves 2 from P1).
                 // State: P1:{1,1}, P2:{2,1,2}, P3:{2}.
                 // Next iter:
                 // u=2 top 2 -> Target P2. P2 Full.
                 // u=2 top 2 -> v=P3. P3 size 1. Move 2->3.
                 // State: P1:{1,1}, P2:{2,1}, P3:{2,2}.
                 // Next iter:
                 // u=2 top 1 -> Target P1. P1 size 2 (<3). Prio 1!
                 // Move 2->1.
                 // State: P1:{1,1,1}, P2:{2}, P3:{2,2}.
                 // Next iter:
                 // u=2 top 2 -> Target P2. P2 size 1. Prio 1.
                 // Move 2->2.
                 // State: P1:{1,1,1}, P2:{2,2}, P3:{2,2}.
                 // Next iter:
                 // u=2 top 2 -> Target P2. P2 Full.
                 // u=3 top 2 -> Target P2. P2 Full.
                 // u=3 top 2 -> Target P2. P2 Full.
                 // Wait, P3 has {2,2}. Top is 2.
                 // Target 2 is P2. P2 is Full.
                 // P3 -> P2 blocked.
                 // P3 -> P1? P1 Full.
                 // P3 -> P2? Blocked.
                 // Are there any other moves?
                 // u=2 top 2. Target 2.
                 // u=3 top 2. Target 2.
                 // P2 is Full.
                 // Any space?
                 // P1 Full. P2 Full. P3 Full.
                 // Wait, P3 has {2,2}?
                 // In sample logic, P3 became {2,2,1}.
                 // My manual trace got P3:{2,2}. Why?
                 // Ah, in sample P1 took the 1 from P2.
                 // P2 had {2,1,2}. Move 2. Then move 1.
                 // My trace: P2 top 2 -> P3.
                 // Wait, P2 top 2 -> P3 is Prio 0.
                 // Did I miss anything?
                 // Ah, in Sample, they moved 1 from P2 to P3.
                 // Why?
                 // P2 had {2,1,2}. Top 2.
                 // Move 2 -> P3.
                 // P2 top 1. Move 1 -> P3.
                 // My logic:
                 // Iter 2: u=2 top 2. Move 2->3.
                 // State P1:{1,1}, P2:{2,1}, P3:{2,2}.
                 // Iter 3: u=2 top 1. Target P1. P1 size 2. Prio 1.
                 // Move 2->1.
                 // P1:{1,1,1}.
                 // State P2:{2}. P3:{2,2}.
                 // Iter 4: u=2 top 2. Target P2. P2 size 1. Prio 1.
                 // Move 2->2.
                 // State P1:{1,1,1}, P2:{2,2}, P3:{2,2}.
                 // Iter 5:
                 // P2 Full. P1 Full. P3 has {2,2}.
                 // u=3 top 2. Target P2.
                 // P2 Full.
                 // u=3 top 2. Target P2. P2 Full.
                 // No valid move to Home.
                 // Need Prio 0.
                 // u=3 top 2. v=P1 Full.
                 // u=3 top 2. v=P2 Full.
                 // u=3 top 2. v=P2 Full.
                 // No space anywhere?
                 // Wait, in Sample output, they moved 2 from P3 to P1? No.
                 // They moved 1 from P3 to P1.
                 // My P3 had {2,2}. No 1.
                 // Where did 1 go?
                 // In my trace, P2 had {2,1,2}. I moved 2 to P3.
                 // Then I saw P2 top 1. I moved 1 to P1.
                 // P1 received 1. Correct.
                 // So P2 became {2}.
                 // P3 had {2,2}.
                 // Wait, did I miss moving the 1?
                 // In Sample output, they moved 1 from P2 to P3.
                 // P3 had {2,2}.
                 // So P3 became {2,2,1}.
                 // Then they moved 1 to P1.
                 // My trace did: P2->{2,1}->(move 2 to 3)->{2,1}->{1 to 1} -> {2}.
                 // So I moved 1 directly to P1.
                 // Why did sample move 1 to P3 first?
                 // Maybe because P1 was full?
                 // In my trace, P1 had {1,1}. Not full.
                 // Ah, in sample P1 had {1,1,2}.
                 // I moved 2 from P1. P1 became {1,1}.
                 // So P1 had space.
                 // Wait, Sample input:
                 // 1 1 2
                 // 2 1 2
                 // P1: 1,1,2.
                 // P2: 2,1,2.
                 // Sample ops:
                 // 1 3 (move 2).
                 // 2 3 (move 2).
                 // 2 3 (move 1).
                 // 3 1 (move 1).
                 // 3 2 (move 2).
                 // 3 2 (move 2).
                 // My trace logic differs only in that I moved 1 to P1 immediately.
                 // The sample buffered it in P3.
                 // My logic moved 1 to P1 directly (Prio 1).
                 // Is P1 allowed to receive?
                 // P1 size 2 < 3. Yes.
                 // Why did sample buffer?
                 // Maybe to save ops? Or maybe sample output is just one valid solution.
                 // Wait, my trace ended up stuck (All full).
                 // Wait, why stuck?
                 // P1 Full ({1,1,1}). P2 Full ({2,2}). P3 Full ({2,2}).
                 // Wait, P3 has {2,2}. Size 2.
                 // Capacity 3. Not full.
                 // Why did I say P3 Full?
                 // Ah, I stopped there.
                 // Iter 5: P3 has {2,2}. Top 2.
                 // Target 2 (P2). P2 Full.
                 // So P3 cannot move to P2.
                 // Can move to P1? P1 Full.
                 // Can move to P3? No self.
                 // Can move to P2 (via cycle)?
                 // Wait, I said P3 has {2,2}.
                 // I missed the fact that P2 had {2}.
                 // Wait, P2 became {2,2} in my trace.
                 // How?
                 // Iter 4: Move 2 from P2 to P2?
                 // Wait, P2 had {2}. Target P2.
                 // Move 2 to P2.
                 // Wait, "move ball from u to v".
                 // If $u=v$, invalid.
                 // So I couldn't move 2 to P2 if $u$ was P2.
                 // Wait, my trace said "u=2 top 2. Move 2->2".
                 // That's invalid.
                 // P2 had {2}. So I can't move top of P2 to P2.
                 // I must move it OUT of P2.
                 // So P2 has 2 (ball). P2 is target for 2.
                 // But I can't put 2 into P2 (it's already there).
                 // Wait, if P2 is {2}, and P2 is full.
                 // Then I can't add more 2s.
                 // P3 has {2,2}.
                 // P2 is full.
                 // P1 is full.
                 // P3 has 2. Target P2.
                 // Move 3 -> 2 (Blocked).
                 // I need space.
                 // P3 has 2 balls. P1, P2 full.
                 // I can move P3 to P3? No.
                 // I need to move a ball OUT of a full pile.
                 // P1 Full, P2 Full.
                 // Move from P1?
                 // P1 top is 1. Target 1.
                 // Move 1 -> P3?
                 // If I move 1 -> P3. P3 becomes {2,2,1}.
                 // Then P1 has {1,1}.
                 // Then P2 has {2,2}.
                 // Then move P2 top 2 to P1? No.
                 // This seems messy.
                 
                 // The issue with my greedy is it prioritizes Home moves.
                 // But sometimes it forces a full pile to accept a ball, but it shouldn't.
                 // Wait, P1 full ({1,1,1}) - Sorted.
                 // P2 full ({2,2}) - Partial.
                 // Wait, in my trace P2 became {2,2}.
                 // P3 has {2,2}.
                 // We need to empty P3 to sort P2?
                 // But P3 has 2s. P2 wants 2s.
                 // We should merge P3 into P2.
                 // P2 has {2,2}. Capacity 3.
                 // P3 has {2,2}.
                 // We can move 2 from P3 to P2.
                 // But P2 has {2,2} -> wait, my trace P2 had {2,2} and P3 had {2,2}?
                 // Let's re-check trace.
                 // Init: P1:{1,1,2}, P2:{2,1,2}, P3:{}.
                 // 1. P1->3 (2). P1:{1,1}. P3:{2}.
                 // 2. P2->3 (2). P2:{2,1}. P3:{2,2}.
                 // 3. P2->1 (1). P2:{2}. P1:{1,1,1}.
                 // 4. P2->2 (??). P2 has {2}. Move 2 to P2?
                 //   No, move P2->2 is invalid.
                 //   Wait, P2 has {2}. It needs to be full?
                 //   It is NOT full.
                 //   We need to put 2 into P2.
                 //   Where is 2? In P3.
                 //   P3 has {2,2}.
                 //   Move P3->2 (2).
                 //   P3:{2}. P2:{2,2}.
                 // 5. P3->2 (2).
                 //   P3:{}. P2:{2,2,2}.
                 //   Total 5 ops.
                 //   State: P1:{1,1,1}, P2:{2,2,2}, P3:{}.
                 //   Perfect.
                 
                 // So my greedy logic failed at step 4 because I tried to move self-to-self.
                 // Correction:
                 // When checking $u \to v$, ensure $u \neq v$.
                 // And ensure $u$ is not empty (guaranteed by loop).
                 // My manual trace error: I thought I could move 2 from P2 to P2.
                 // So if I implement correctly (check $u \neq v$), it should work.
                 
                 // But what if P2 is Full ({2,2,2})?
                 // Then we are done.
                 // So the loop will continue until done.
                 // The only risk is if it enters a dead state where no $u \to v$ is valid.
                 // Dead state: All poles $u$ with balls have $v$ full for ALL valid $v$.
                 // And we established there is at least one slot with space.
                 // So if a pole $u$ has balls, and ALL valid destinations are full.
                 // This implies that for all $v$ with space, $v$ is full?
                 // Contradiction.
                 // Wait.
                 // Condition: "valid destinations are full".
                 // If $S = \{v | \text{size}(v) < m\}$ is not empty.
                 // We can pick ANY $v \in S$.
                 // My greedy picks one.
                 // So it's fine.
                 
                 // Wait, does my greedy loop check all $u, v$ pairs?
                 // Yes.
                 // And pick best one.
                 // So it's robust.
                 
                 // Wait, what if I pick a suboptimal move?
                 // It might not be monotonic.
                 // But with 2M ops, it's fine.
                 // The logic `Prio 1` vs `Prio 0` ensures we sort as much as possible.
                 
                 // One optimization:
                 // `Target` is defined as $c$ (Color $c$) $\to$ Pole $c$.
                 // This assumes Pole $c$ is the destination.
                 // Is it always safe?
                 // We have $n$ colors. Poles $1..n$ and $n+1$.
                 // If we use $P_1..P_n$, we leave $P_{n+1}$ empty.
                 // This is consistent.
                 
                 // One minor issue:
                 // If $N >$ number of colors?
                 // Problem says "$n$ colors". "n poles".
                 // And "n*m balls".
                 // So colors $1..n$ appear.
                 // So we map color $c \to$ Pole $c$.
                 // Poles $1..n$ correspond to colors $1..n$.
                 // Pole $n+1$ is buffer.
                 // This works perfectly.
                 
                 // Wait, what if color $c$ is not $1..n$?
                 // Problem says "colors are numbered from 1 to n".
                 // So mapping is 1-to-1.
                 
                 // Final Algorithm:
                 // Read N, M.
                 // Read poles $1..n$. Pole $n+1$ empty.
                 // Assign Target(Color $c$) = Pole $c$.
                 // Loop:
                 //    found = false.
                 //    For each $u$ from $1..n+1$:
                 //       If `poles[u]` empty continue.
                 //       If `poles[u].size() == m` and Top matches $u$ (if we treat $u$ as home):
                 //           Wait, we don't check if $u$ is home, we check if Top matches Target.
                 //           Actually, checking $u$'s target is useful.
                 //           If $poles[u]$ contains correct balls...
                 //           Wait, I'll stick to finding the best move.
                 //    Iterate all pairs $(u, v)$ with $u \neq v$, `poles[u]` not empty, `poles[v].size() < m`.
                 //       Candidate Move $(u, v)$ with ball $c = poles[u].back()$.
                 //       Score = (Target($c$) == v) ? 1 : 0.
                 //       Pick pair with max Score. If tie, any.
                 //       Execute move.
                 //       Store op.
                 //       If no moves possible (set empty):
                 //          Wait, we proved this state is unreachable unless solved?
                 //          Or maybe stuck in cycle?
                 //          If we get stuck, it means no valid moves?
                 //          But we know there is space.
                 //          Wait, if no moves possible, it means for every $u$ (non-empty), every $v$ (space) is invalid?
                 //          No, valid move is defined by capacity.
                 //          So valid moves always exist as long as there is a space.
                 //          So loop terminates only when solved (all sorted).
                 //          Wait, "Solved" means each pole is monochromatic?
                 //          Or just "Sorted" (all correct balls in Home).
                 //          If all correct balls in Home, and no incorrect balls outside, then solved.
                 //          But we might have incorrect balls in Home?
                 //          No, if $P_u$ has $u$ on top and is full, it's sorted?
                 //          No, might have mixed below.
                 //          But if we can't move anything, we are stuck.
                 //          Stuck $\implies$ No moves.
                 //          Since we prioritize Home moves, if we are not stuck, we make progress.
                 //          What if we get stuck but not solved?
                 //          Example: P1 has {2}. P2 has {1}. P3 empty.
                 //          Target: 1->P1, 2->P2.
                 //          P1 top 2. Target P2.
                 //          P2 top 1. Target P1.
                 //          Valid moves: P1->P3 (2 to 3). P2->P3 (1 to 3).
                 //          Scores: 0 vs 0.
                 //          Pick P1->P3. P1 empty. P3 has {2}.
                 //          Next: P2->P1 (1 to 1). Score 1.
                 //          Move 2->1. P2 empty. P1 has {1}.
                 //          Next: P3->P2 (2 to 2). Score 1.
                 //          Move 2->2. P3 empty. P2 has {2}.
                 //          Solved.
                 //          
                 //          So the strategy works.
                 
                 // Implementation:
                 // Use `std::vector<int>` for each pole.
                 // Be careful with 1-based indexing.
                 
                 // Max ops check:
                 // $N=50, M=400$.
                 // $NM = 20,000$.
                 // Operations roughly proportional to $NM \times N$.
                 // $20,000 \times 50 = 1,000,000$.
                 // Within 2M.
                 
                 // Complexity of inner loop:
                 // Outer loop runs $K$ times.
                 // Inner loop $O(N^2)$.
                 // $K \approx 2 \cdot 10^6$ worst case?
                 // $2 \cdot 10^6 \times 2500 \approx 5 \cdot 10^9$. Too slow!
                 // Wait, $K$ is the number of *ops*, not iterations.
                 // If we make 1 op per outer loop iteration, $K$ ops means $K \cdot N^2$.
                 // So we need $O(1)$ selection per op.
                 
                 // Optimization:
                 // We don't need to scan all pairs.
                 // Just iterate all $u$.
                 // Find best $v$.
                 // For a fixed $u$, we can check all $v$.
                 // We want to maximize Score.
                 // Score depends on $Target(ball\_u) == v$.
                 // So we want $v = Target(ball\_u)$ if possible (and has space).
                 // If $Target(ball\_u)$ has space, pick it! (Priority 1).
                 // If not, pick any $v$ with space. (Priority 0).
                 // If multiple $v$ with space, pick any.
                 
                 // Wait, if multiple $u$ have balls, we have a choice.
                 // Which $u$ should we pick?
                 // If $u_1$ has ball $c_1$ targeting $T_1$ (available).
                 // $u_2$ has ball $c_2$ targeting $T_2$ (available).
                 // Both valid.
                 // Pick any.
                 
                 // So optimized logic:
                 // 1. Find any $u$ such that `poles[u]` not empty AND `Target(top)` is not full.
                 //    If found, Move $u \to Target(top)$.
                 // 2. Else (no immediate home placement possible):
                 //    Find any $u$ (not empty) and any $v$ (has space).
                 //    Move $u \to v$.
                 
                 // Does this ensure progress?
                 // Step 1 handles Priority 1 moves.
                 // Step 2 handles fallback.
                 // Is Step 2 safe?
                 // If Step 1 is not possible, it means for all $u$, $Target(top)$ is full.
                 // This means we can't put any ball in its Home.
                 // We must move balls to Buffer.
                 // By moving to Buffer (or any available space), we enable future placements.
                 // Since we prioritize Step 1, we sort whenever possible.
                 // This minimizes moves significantly.
                 
                 // Is it correct to pick *any* $v$ for Step 2?
                 // Maybe pick $v=n+1$ (global buffer) to concentrate junk?
                 // But we have space in other piles too.
                 // Does it matter?
                 // If we move to $v$ which has space, we reduce $u$'s burden.
                 // Maybe picking $v$ close to $u$ or $Target$?
                 // Since ops are cheap, just pick $n+1$ if space, or otherwise first available.
                 // Or just iterate $u$ and find $v$.
                 
                 // Wait, Step 1 says "If found".
                 // What if we have multiple such $u$?
                 // Pick any.
                 // Wait, if $u_1$ has ball $c_1$ with space at $T_1$.
                 // $u_2$ has ball $c_2$ with space at $T_2$.
                 // If we move $u_1 \to T_1$, $T_1$ fills up.
                 // Maybe $T_1$ was needed by $u_2$?
                 // If $T_1$ was needed by $u_2$, then $c_2$ must be Color of $T_1$.
                 // If we fill $T_1$ with $c_1$, then $c_2$ has nowhere to go.
                 // But $c_1$ and $c_2$ are distinct colors?
                 // Yes. $T_1$ is for $c_1$. $T_2$ is for $c_2$.
                 // So no collision on target ownership.
                 // So picking any is safe.
                 
                 // Wait, collision on space?
                 // If we fill $T_1$ (space consumed), we might make $u_2$ (which wanted to go to $T_2$) unaffected, unless $T_1$ and $T_2$ share space?
                 // Each pole is independent. Filling $T_1$ consumes space in $T_1$.
                 // Does it affect $u_2$?
                 // If $u_2$ wanted to go to $T_2$, and $T_2$ has space.
                 // It doesn't care about $T_1$.
                 // Unless $T_1$ and $T_2$ are the same pole?
                 // Impossible since targets are distinct.
                 // So filling $T_1$ is local.
                 // So Step 1 is safe.
                 
                 // One catch: "Target" is determined by Color $c$.
                 // $Target(c) = c$.
                 // Wait, what if color $c$ is not $1..n$?
                 // Input: colors $1..n$.
                 // Poles $1..n+1$.
                 // So $Target(c) \in \{1..n\}$.
                 
                 // Logic refined:
                 // Loop:
                 //   bool step1_possible = false;
                 //   int u_step1 = -1;
                 //   For $u=1..n+1$:
                 //     if `poles[u]` empty continue.
                 //     $c = poles[u].back()$.
                 //     $T = c$.
                 //     if `poles[T].size() < m`:
                 //       u_step1 = u; break;
                 
                 //   if u_step1 != -1:
                 //     Move `u_step1` -> `Target(poels[u_step1].back())`.
                 //     Record op.
                 //     Continue (reset scan for fresh state, or continue loop? 
                 //     Resetting is safer to avoid ordering bias, but continuation works too.
                 //     Since ops are fast, reset is safer).
                 //   else:
                 //     // Step 2: Find any move.
                 //     bool step2_done = false;
                 //     For $u=1..n+1$:
                 //       if `poles[u]` empty continue.
                 //       For $v=1..n+1$:
                 //         if $u == v$ continue.
                 //         if `poles[v].size() < m`:
                 //           Move $u \to v$.
                 //           Record op.
                 //           step2_done = true;
                 //           Break both loops.
                 
                 //   if !step1_possible && !step2_done:
                 //     return (Solved).
                 
                 // This logic is $O(N)$ per op.
                 // Total ops limit 2M.
                 // Total time $O(\text{Ops} \cdot N) \approx 2 \cdot 10^6 \cdot 50 \approx 10^8$.
                 // Acceptable.
                 
                 // Wait, we need to verify "Solved" condition.
                 // Actually, we can just run for max 2,000,000 ops.
                 // The problem guarantees valid solution.
                 // So we will find it.
                 
                 // Wait, one corner case:
                 // If `u_step1` loop finds a move, we break immediately.
                 // Is it better to process multiple?
                 // No, one is enough.
                 
                 // Code structure:
                 // Includes.
                 // Vectors for poles.
                 // Loop.
                 // Output k lines.
                 // Wait, k is number of ops.
                 
                 // We need to store operations.
                 // Vector of pairs `ops`.
                 
                 // Memory limit 512MB. Safe.
                 
                 // One subtle bug:
                 // If we move $u \to T$, $T$ becomes closer to full.
                 // Next loop, $T$ might be full.
                 // Correct.
                 
                 // Re-checking the logic:
                 // "If $Target(c)$ is full" -> handled by Step 2.
                 // Is Step 2 always finding a move?
                 // Yes, because there exists at least one pole with space (unless all full, which means solved? No, if all full, sum is $(N+1)m$. But we have $Nm$ balls. So at least one empty).
                 // So Step 2 will always find a pair.
                 // Infinite loop?
                 // If we oscillate $u \leftrightarrow v$?
                 // With Step 1 prioritizing correctness, it pushes towards solution.
                 // If Step 1 fails, we shuffle.
                 // Shuffling should eventually clear a target.
                 
                 // Double check "Target" assignment.
                 // Input colors $1..n$. Poles $1..n$.
                 // Mapping Color $c$ -> Pole $c$ is valid.
                 // Wait, what if Pole $c$ is full initially?
                 // Then we can't put ball $c$ there.
                 // But we are looking for a move.
                 // If we find a ball $c$ in $u$, and $P_c$ is full.
                 // Step 1 ignores it.
                 // Step 2 moves it to Buffer.
                 // This is consistent.
                 
                 // Is it possible that Step 2 picks a move that hurts?
                 // E.g. moves a correct ball $c$ away from $P_c$?
                 // If we move $c$ to $v$.
                 // If $v \neq P_c$, then $c$ is displaced.
                 // But Step 1 checks if we can put $c$ into $P_c$.
                 // If $P_c$ is full, we can't.
                 // So we *must* displace.
                 // Displacing to any safe $v$ is necessary.
                 // The greedy choice of *any* safe $v$ is fine.
                 
                 // Final check on constraints.
                 // $n, m \le 50, 400$.
                 // Ops limit $2 \cdot 10^6$.
                 
                 // Looks solid.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> poles(n + 2); // 1-based, poles 1..n, pole n+1 is buffer
    // We read n lines for poles 1 to n.
    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j < m; ++j) {
            int color;
            cin >> color;
            poles[i].push_back(color);
        }
    }
    // Pole n+1 is empty by default

    vector<pair<int, int>> ops;
    
    // Predefined mapping: Color c -> Pole c
    // Since colors are 1..n, and we have poles 1..n available.
    // We will try to put balls of color c into pole c.
    
    auto get_target = [&](int color) {
        // Returns the pole index where color c should reside.
        // If color is c, target is pole c.
        return color;
    };

    // Simulation loop
    // We limit iterations to prevent infinite loops (though logic guarantees convergence)
    // Total operations allowed is 2,000,000.
    // Since we process balls, and total balls is small, this is safe.
    
    while (true) {
        int u_step1 = -1;
        int u_step2 = -1;
        int v_step2 = -1;

        // Pass 1: Try to move a ball to its target pole if target has space
        for (int u = 1; u <= n + 1; ++u) {
            if (poles[u].empty()) continue;
            int color = poles[u].back();
            int target_pole = color; // Color c -> Pole c
            
            // Check if target pole is not full
            if (target_pole <= n + 1 && poles[target_pole].size() < m) {
                // Check bounds just in case (though color is 1..n)
                // Note: colors are 1..n. Poles 1..n+1.

````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=1442)

````
I think I need to read the problem again carefully and understand the core constraint and goal. The key insight seems to be that we're dealing with m balls of each of n colors, initially distributed across n+1 poles, with n having exactly m balls on their own colored pole.

The operation constraint is critical: we can move a top ball from pole x to pole y only if pole y has at most m-1 balls. This means we can never fill any pole completely - every non-target pole must stay strictly below capacity m.

Let me reconsider what "m" represents in this context. Reading back carefully: there are n different colors, each appearing m times total. So the total number of balls is actually n*m, not m per color. Each of the n initial poles (1 through n) holds m balls, and they're arranged such that all balls on pole i have the same color. Pole n+1 starts empty.

So if I map this correctly: for n=2, m=3, there are 6 total balls (3 red + 3 blue). Pole 1 has 3 red balls, pole 2 has 3 blue balls, pole 3 is empty. The constraints say:
- Pole x needs at least one ball to source from (always true since no pole is ever allowed to go completely empty)
- Pole y needs at most m-1 balls before receiving a move (so it stays at or below m-1 after each drop)

This means every pole except the final destination can hold at most m-1 balls safely. Since a pole holds at most m-1, and each color appears m times, no single pole can ever receive all m copies of one color. To gather an entire color together, a pole needs more room than the constraint allows... unless I'm reading wrong about which color is where.

Actually, wait - let me reconsider what the input actually specifies. The sample shows `2 3` meaning n=2 colors and m=3 balls per color, then pole 1 has `[1 1 2]` and pole 2 has `[2 1 2]`. So pole 1's stack is bottom-to-top: color 1, color 1, color 2. That's not monochromatic! I was wrong about the initial distribution being sorted by color.

Re-reading the problem confirms this: "each color appears exactly m times" and the sample clearly shows mixed colors within each pole. So both starting poles contain interleaved colors, and I need to sort each color's m instances onto separate poles. Given the output shows 6 operations transforming pole 1 from [1,1,2] to empty, pole 2 from [2,1,2] to [2,1,1,2], and pole 3 to [2,2,1,1,2], the strategy isn't preserving order but rather consolidating colors across poles. Looking at the actual output: moves happen from pole 1 to 3 (the two color-1s), then repeatedly from pole 2 to 3 (one color-2, then color-1, then color-2 again), leaving pole 3 with five balls and pole 2 with three remaining balls.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 505;
int a[N][N]; // a[color][pole] -> stack height
vector<int> s[N]; // s[pole] -> balls on this pole
int cnt[N]; // count of each color currently used

// Find a free pole (capacity < m)
int find_free() {
    for (int j = 1; j <= n + 1; j++) {
        if ((int)s[j].size() < m) return j;
    }
    return 0;
}

// Move top ball from pole x to pole y
void move(int x, int y) {
    int col = s[x].back();
    s[x].pop_back();
    s[y].push_back(col);
    cnt[col]++;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m, n;
    cin >> n >> m;

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < m; j++) {
            int c;
            cin >> c;
            s[i].push_back(c);
            cnt[c]++;
        }
    }

    vector<pair<int, int>> ops;

    // Phase 1: Sort each color's balls into a single pole using a temp pole.
    for (int col = 1; col <= n; col++) {
        int free_p = find_free();
        for (int i = 1; i <= n; i++) {
            while (!s[i].empty() && s[i].back() == col) {
                move(i, free_p);
            }
        }
    }

    // Phase 2: Fill all available poles with m balls of one color until last color is left
    vector<int> free_poles;
    for (int i = 1; i <= n + 1; i++) {
        if ((int)s[i].size() < m) free_poles.push_back(i);
    }

    while ((int)s[free_poles.back()].size() < m) {
        int p = free_poles.back();
        free_poles.pop_back();

        for (int col = 1; col <= n; col++) {
            int src = 0;
            for (int i = 1; i <= n + 1; i++) {
                if (!s[i].empty() && s[i].back() == col) {
                    src = i;
                    break;
                }
            }
            if (src) move(src, p);
            else break;
        }
    }

    cout << ops.size() + 4 * cnt[n] + 3 * (cnt[n] - 1) << "\n";
    for (auto &p : ops) cout << p.first << " " << p.second << "\n";
    for (int k = 0; k < 4 * cnt[n]; k++) cout << n + 1 << " " << n + 1 - k << "\n";
    for (int k = 0; k < 3 * (cnt[n] - 1); k++) cout << n + 1 << " " << n + 1 - k << "\n";

    return 0;
}
```
````
