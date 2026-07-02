# 题 117 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The solution to this problem requires a strategic approach to determine the equations of lines that are hidden from us. The problem is asking us to identify N lines, each defined by their slope and y-intercept (a_i, b_i), through at most Q_max queries.

1.  **Understanding the Query:**
    When we provide a point $(x, y)$, the interactor returns $S = \sum_{i=1}^N \text{dist}((x, y), L_i)$.
    The distance from a point $(x_0, y_0)$ to a line $ax + by + c = 0$ is $\frac{|ax_0 + by_0 + c|}{\sqrt{a^2 + b^2}}$.
    Here, the lines are given as $y = a_i x + b_i$, which can be rewritten as $-a_i x + y - b_i = 0$.
    So the normalized form has coefficients $A = -a_i$, $B = 1$, $C = -b_i$. The denominator is $\sqrt{a_i^2 + 1}$.
    Thus, the distance for line $i$ is $\frac{|-a_i x + y - b_i|}{\sqrt{a_i^2 + 1}}$.

    Wait, usually in competitive programming problems involving "hidden lines" and query limits where the number of unknown parameters is high ($2N$ for $2N$ coordinates) but $N$ is small ($N \le 100$), there's often a pattern or an algebraic trick. However, here $Q_{max} = 10^4$ and $N \le 100$. This suggests we might need around $2N$ to $3N$ queries just to solve for the parameters, and potentially more if we rely on iterative refinement or error reduction. But $Q_{max}$ is large enough for this specific constraint.

2.  **Strategy Selection:**
    Since we don't know the lines, we cannot simply use linear regression because the distance formula involves absolute values, making it non-linear and non-differentiable in a simple way. The function $f(x, y) = \sum \frac{|-a_i x + y - b_i|}{\sqrt{a_i^2 + 1}}$ looks like the sum of distances from a point to lines. This function is convex. The minimum value occurs when the gradients align in a specific way. However, the function value itself at arbitrary points gives us data points.

    Let's look at the structure. If we pick two lines $L_1$ and $L_2$, their distance contribution to the query depends on the relative position of the query point to each line. If the query point is on the line, distance is 0.

    Actually, since $N$ is up to 100 and $a_i, b_i$ are integers in $[-10^4, 10^4]$, maybe we can leverage the fact that the coefficients are relatively small integers. But the domain is huge.

    A common strategy for this type of problem ("reconstruct objects via queries") involves:
    *   Finding intersections of lines.
    *   Using collinearity properties.
    *   Or simpler: Just solving the system. But with absolute values, it's hard.

    However, note that we are asked to output `! a_1 ... a_n b_1 ... b_n`. The order doesn't matter.

    Key observation: The term for a line $i$ is $\frac{|-a_i x + y - b_i|}{\sqrt{a_i^2 + 1}}$.
    If we choose points such that we minimize the "denominator" impact or simplify the numerator?
    Usually, determining slopes $a_i$ is harder than intercepts $b_i$ once slopes are known.

    Let's consider finding one line first.
    Suppose we find a line $y = ax + b$. Then we reduce to $N-1$ lines.
    How to find one line?
    If we query points far away from the lines? No, the territory is bounded $[-10^{12}, 10^{12}]$.

    Let's look at the constraints again. $N \le 100$. We have roughly $2N = 200$ unknown parameters. $Q_{max} = 10000$. We have plenty of queries.
    Even brute force over slopes isn't feasible (domain too large), but the slopes are small integers $[-10^4, 10^4]$. That's still $20,001$ possibilities per line. $20000^{100}$ is impossible. But we know they are distinct (no parallel lines).
    Wait, actually, the slopes are integers. There are only about 20,000 possible distinct integer slopes.
    With no two lines being parallel, all $a_i$ must be distinct.
    This means $a_i \in [-10000, 10000]$ and all unique.
    This is a massive constraint! We can probably iterate or guess slopes using some method.
    Or maybe not, iterating $O(N)$ lines is better, but how do we group them?

    Actually, let's step back. Is there a simpler mathematical property?
    If we fix a line $L$, the distance function is piecewise linear (convex).
    The sum of convex functions is convex.
    The gradient is not defined at the line itself, but subgradient exists.
    
    However, a much simpler observation is available:
    If we pick a very large $X$ and $Y$, say $X=0, Y=K$.
    Distance $\approx |y - a_i X - b_i| / \sqrt{...}$
    If $X$ is very large, $a_i X$ dominates $y$ and $b_i$.
    Specifically, if $X > 10^4 \cdot 10^4$, then $a_i X$ could be huge.
    Let's consider specific coordinate choices.
    
    Consider querying on the axis. $(x, 0)$ or $(0, y)$.
    Or even better:
    Notice that $b_i = y - a_i x$ for any point $(x, y)$ on the line.
    If we had a point on the line, we get one equation.
    If we have two points on the line, we have two equations -> solved.
    
    Can we find one point on the line?
    If we find *any* point $(x, y)$ where distance to line $i$ is 0, great. But we don't know which line is which.
    
    Let's try a randomized approach or a binary search?
    The slopes are integers.
    What if we just output random guesses? The score drops with queries.
    
    Let's assume there is a constructive way.
    Actually, looking at similar Codeforces/ICPC interactive problems:
    Often, we fix one parameter to guess or narrow down.
    However, $N$ is up to 100. Solving 100 lines individually seems required.
    
    Let's reconsider the equation: $D_i(x, y) = \frac{|-a_i x + y - b_i|}{\sqrt{a_i^2 + 1}}$.
    Sum $S = \sum D_i$.
    
    Idea:
    Pick a point $P_1(0, 0)$. Measure $S_1 = \sum \frac{|-b_i|}{\sqrt{a_i^2+1}}$.
    Pick a point $P_2(X, 0)$ where $X$ is large. $S_2 = \sum \frac{|-a_i X - b_i|}{\sqrt{a_i^2+1}}$.
    Pick a point $P_3(0, Y)$ where $Y$ is large.
    
    With these, we have non-linear equations.
    
    Alternative Strategy:
    Guess $a_i$ directly.
    Since $a_i$ are integers in $[-10^4, 10^4]$, and all distinct.
    There are at most 100 values chosen from this range.
    If we can somehow verify if a candidate slope belongs to the set of lines?
    Or rather, we need to find the specific pair $(a_i, b_i)$.
    
    Actually, maybe we don't need to distinguish lines yet. We just need to list the pairs.
    
    Wait, the problem says "No two lines are parallel". This implies all $a_i$ are distinct.
    Given $N \le 100$, and $a_i$ are integers.
    Is it possible to isolate individual lines?
    
    Let's analyze the magnitude of terms.
    If $|x|$ and $|y|$ are small (e.g. within the range of coefficients), terms like $|-a_i x + y - b_i|$ vary slowly with $x, y$.
    If we make $x$ extremely large, the term behaves like $|a_i|/\sqrt{a_i^2+1} \cdot |x| + \dots \approx 1 \cdot |x|$.
    Wait, $\lim_{X \to \infty} |-a_i X + Y - b_i| / \sqrt{a_i^2+1} = \frac{|a_i|}{\sqrt{a_i^2+1}} X \approx X$.
    For different slopes, the coefficient of $X$ is $\frac{|a_i|}{\sqrt{a_i^2+1}}$. This is strictly increasing with $|a_i|$.
    It seems if we pick huge $X$, the total sum is dominated by terms with largest $|a_i|$.
    Specifically, let $k(X) = \sum \frac{a_i X + b_i'}{\sqrt{a_i^2+1}}$ (assuming signs match).
    Actually, since the region is limited, we can go up to $10^{12}$.
    With $X$ large, say $10^{12}$. The term is roughly $X \cdot \frac{|a_i|}{\sqrt{a_i^2+1}}$.
    Since $a_i \in [-10^4, 10^4]$, the ratio is close to 1.
    Specifically, if $a_i = 10^4$, ratio $\approx 1$.
    If $a_i = 0$, ratio $= 0$.
    So the contribution varies significantly based on $|a_i|$.
    
    This suggests that with large $x$, we can perhaps recover information about the set of slopes $\{a_i\}$.
    Specifically, if we query at $(X, Y)$ with $X \gg Y$, the dominant part of the sum comes from lines where $a_i \neq 0$.
    Wait, if $a_i$ is positive or negative, $|-a_i X| = |a_i| X$.
    So the sum behaves like $X \sum \frac{|a_i|}{\sqrt{a_i^2+1}}$.
    Let $W = \sum \frac{|a_i|}{\sqrt{a_i^2+1}}$. Then $S \approx W \cdot X$.
    This gives us $\sum f(a_i)$.
    By varying $Y$, we might separate them further.
    But $W$ is a sum of smooth functions of integers.
    
    Maybe we can use the fact that slopes are integers.
    Let's try to determine all $a_i$ first.
    With $N$ up to 100, and $a_i$ distinct integers, maybe we can find the multiset $\{a_i\}$.
    But recovering the specific values from sums like $\sum g(a_i)$ is generally hard without knowing $g$ is invertible or something.
    However, if we query many points on axes, we get a system of equations? No.
    
    Let's rethink.
    If we can recover *one* line completely, say $y = a_1 x + b_1$.
    Then we remove it from the sum. How?
    We subtract its contribution.
    But we don't know its contribution initially.
    Once we guess it, we calculate theoretical distances for all previous queries and subtract.
    So, the strategy is:
    1. Find one line.
    2. Subtract its contribution to the stored answers.
    3. Repeat $N$ times.
    
    Problem reduces to: Find ONE line among $N$.
    Since $N$ is small, maybe brute force one line is too expensive?
    Slope guessing? There are 20000 possibilities. $10^5$ operations for one check is fast enough.
    But we have $Q$ limit. We can only ask queries.
    
    Let's refine the "Find one line" idea.
    Is it possible to find the line with minimal $b_i$ or something? No.
    
    Let's go back to the geometry.
    Intersection of lines.
    If we pick two points $(x_1, y_1)$ and $(x_2, y_2)$ far apart.
    The difference in distances?
    $d((x,y), L) = \frac{|-ax + y - b|}{\sqrt{a^2+1}}$.
    If $(-ax+y-b)$ has constant sign across the query region for a specific line, then $d$ is linear.
    If it flips sign, it's V-shaped.
    The sum of V-shapes is also V-shaped-like (piecewise linear convex).
    The vertices of these V-shapes lie on the lines $y = a_i x + b_i$.
    If we take the derivative w.r.t. $x$ along $y=c$, we get step functions.
    The points where the slope changes are exactly the intersection of the horizontal line $y=c$ with the hidden lines.
    So, if we move along $y=0$, the value $f(x) = \sum d((x,0), L_i)$ is a piecewise linear function. Its corners occur at $x$ where $0 = a_i x + b_i \implies x = -b_i/a_i$.
    These $x$ values are the x-coordinates of the intersections of the horizontal line $y=0$ with the hidden lines.
    Similarly, moving along $y=Y$, the corners occur where $Y = a_i x + b_i \implies x = (Y-b_i)/a_i$.
    Moving along $x=0$, the corners occur where $y = b_i$.
    
    Since $a_i, b_i$ are integers between $-10^4, 10^4$, and $x, y$ query points can be integers.
    Wait, if we pick $y=0$, the "breakpoints" are at $x = -b_i/a_i$.
    These are rational numbers. The query allows real coordinates? "x and y do not have to be integers".
    However, the return value is a real number (implied, though usually double precision is fine).
    If we sample $f(x)$ along $y=0$, we can detect the breakpoints.
    How many breakpoints? At most $N$.
    How do we find the exact locations of these breakpoints?
    We can perform binary search on the derivative or the function itself?
    The function is convex along the path $y=0$.
    $f(x)$ is convex. The minimum is somewhere, then it increases.
    The points where the "slope" changes correspond to crossing the lines.
    But wait, the distance function is $|ax+b|/C$. The derivative jumps by $2/C$ (scaled by sign change).
    So the sum of distances is a sum of such functions.
    It is a piecewise linear convex function.
    We can find the minimum by ternary search? Yes, since it's convex.
    But finding the exact locations of the kinks (which give us $-b_i/a_i$) is harder than just minimizing.
    Because the kinks might be very close or sparse?
    Also, we have $N$ up to 100.
    If we can find all kinks for one specific $y$ (say $y=0$), we get 100 potential x-coordinates.
    But these x-coordinates depend on $a_i$ too ($ -b_i/a_i $). We don't know $a_i$ or $b_i$. We only know $a_i \cdot (-b_i/a_i) = -b_i$ is integer? No.
    
    Let's check the constraints again.
    $a_i, b_i$ are integers.
    If we find the breakpoint $x_k$ corresponding to line $k$ on $y=0$.
    Then the line passes through $(x_k, 0)$. Equation: $y - 0 = a_k (x - x_k) \implies y = a_k x - a_k x_k$.
    So $b_k = -a_k x_k$.
    This implies $b_k$ and $a_k$ have the same relation as determined by the intercept.
    But this assumes we found $x_k$.
    
    Wait, if we find ALL kinks on $y=0$, we identify 100 lines intersecting $y=0$.
    Each intersection point $(x_i, 0)$ means line $i$ passes through it.
    But this doesn't uniquely identify the line (we need $a_i$).
    We know it's one of the hidden lines.
    But we don't know the index.
    However, if we do this for another $y$ (say $y=1$), the kinks will shift.
    For line $i$:
    On $y=0$, $x_{int}^{(0)} = -b_i/a_i$.
    On $y=1$, $x_{int}^{(1)} = (1-b_i)/a_i = x_{int}^{(0)} + 1/a_i$.
    The difference is $1/a_i$.
    So if we have two sets of x-coordinates for kinks on $y=0$ and $y=1$, say sets $X^{(0)}$ and $X^{(1)}$.
    For every $x \in X^{(0)}$, there corresponds exactly one $x' \in X^{(1)}$ such that $x' = x + 1/a_i$.
    Wait, $1/a_i$ depends on the specific line.
    If we have $N$ lines, we have $N$ kinks on $y=0$.
    On $y=1$, we have $N$ kinks.
    By matching the "correspondence", we can deduce $1/a_i$.
    Matching might be ambiguous if multiple lines have same $1/a_i$? No, no two lines parallel $\implies a_i$ distinct $\implies 1/a_i$ distinct.
    So we can identify the set of values $\{1/a_i\}$.
    Actually, we can identify the slopes $a_i$ by calculating differences $x' - x$.
    But we don't know the pairing.
    However, if $a_i$ are integers, then $1/a_i$ are rationals.
    Wait, finding the exact rational differences.
    
    Algorithm Refined:
    1.  Query points on line $y=0$ (or any horizontal line). Sample enough points to reconstruct the piecewise linear function $f(y=0, x)$.
        Wait, querying $10^4$ points to fully resolve 100 kinks? Yes, easy.
        Actually, since it's convex, the function decreases then increases.
        The kinks occur where lines cross $y=0$.
        The distance function is convex, so it's minimized at the projection of some intersection?
        Actually, the function $g(x) = \sum |a_i x + b_i|/c_i$ is minimized near the geometric median?
        But we just need to locate the kinks.
        Since $a_i, b_i$ are integers, the intersections $x = -b_i/a_i$ are rational.
        If we query at intervals of small $\epsilon$, we can approximate the kinks?
        Or simpler:
        Binary search for kinks.
        We know the minimum is at some $x_{min}$. The slope of the function is $\sum \pm 1/c_i \cdot a_i$.
        Wait, the derivative of $|-ax+b|$ is $sgn(\dots) \cdot (-a)/\sqrt{a^2+1}$.
        So the slope is constant between lines.
        The slope changes by adding/subtracting $a_i/\sqrt{a_i^2+1}$.
        As long as we can track the cumulative slope, we can find the kinks.
        Start at left-most kink (where $x$ is very negative). All terms have argument $<0$ (since $-a_i x + b_i > 0$ for $x \to -\infty$ assuming $a_i > 0$? Wait. $-a_i x + b_i$. If $x \to -\infty$, sign depends on $-a_i$.
        Case 1: $a_i > 0$. $-a_i x \to +\infty$. Argument positive. Dist $= (-a_i x + b_i)/c_i$. Deriv w.r.t $x$: $-a_i/c_i$.
        Case 2: $a_i < 0$. $-a_i x \to -\infty$. Argument negative. Dist $= (-(-a_i x + b_i))/c_i = (a_i x - b_i)/c_i$. Deriv w.r.t $x$: $a_i/c_i$ (which is negative, since $a_i < 0$).
        Wait, if $a_i < 0$, deriv is $a_i/c_i < 0$.
        In both cases, for $x \ll 0$ (very negative), the derivative w.r.t $x$ is $-|a_i|/c_i$.
        Since $a_i$ are distinct, we can compute the expected derivative accumulation.
        Actually, just observe: for $x \to -\infty$, the sum $S(x)$ is linear with slope $\sum \frac{-|a_i|}{\sqrt{a_i^2+1}}$.
        We can just measure $S(x)$ at two large negative $x$. We get the initial slope.
        Then we scan $x$ increasing. The slope should increase by steps.
        At each step, we detect where the slope changes.
        This change corresponds to crossing a line.
        We record the $x$ location of the transition. Let this be $x_k$.
        We continue until $x \to +\infty$, where slope becomes $\sum \frac{|a_i|}{\sqrt{a_i^2+1}}$.
        During this sweep, we collect $N$ critical $x$ coordinates where the slope changes.
        These are the intercepts of lines with $y=0$.
        Let these coordinates be $p_1, p_2, \dots, p_N$.
        These satisfy $p_k = -b_{\pi(k)} / a_{\pi(k)}$ for some permutation.
        
    2.  To distinguish the lines (get $a_i, b_i$):
        Repeat the process for $y=1$.
        Collect $q_1, q_2, \dots, q_N$.
        These satisfy $q_j = (1 - b_{\sigma(j)}) / a_{\sigma(j)} = p_j + 1/a_{\sigma(j)}$ (assuming correct matching).
        Basically $q_k - p_k = 1/a_k$.
        We need to match $p$'s and $q$'s.
        Note that $1/a_i$ can be calculated if we know the correspondence.
        Without correspondence, we have a bipartite matching problem.
        But wait, the set of values $\{1/a_i\}$ are unique (distinct integers $\implies$ distinct reciprocals).
        So $q_m - p_l = 1/a$ would imply $a = 1/(q_m - p_l)$.
        Since $a$ must be integer, $q_m - p_l$ must be of form $1/k$.
        So we can just check all pairs $(p_i, q_j)$ and find those where difference is close to a reciprocal of an integer.
        Or even simpler: calculate $q_j - p_i$. If it is $1/A$, then $A$ is a candidate slope.
        Since there are only $N^2$ pairs and $N$ lines, we can just collect valid integer slopes $A$ from the differences.
        Actually, for each $p_i$, there is a unique $j$ such that $p_i$ and $q_j$ belong to the same line.
        We don't need to know indices. We just need to assign $a_i$ to each intercept.
        Wait, we need to pair them to recover $b_i = -a_i p_i$.
        So yes, for each $p_i$, find $a_i$ such that $p_i$ and some $q$ differ by $1/a_i$.
        Specifically, for a fixed $p_i$, we test $k \in [1, 10000]$ and $k \in [-10000, -1]$ and check if $|q_m - p_i - 1/k| < \epsilon$ for some $q_m$.
        Or just: since $a_i$ are integers, we can iterate all possible $a \in [-10000, 10000]$.
        Calculate potential intercept $x$ such that $x$ was in our set $P$ and $q = x + 1/a$ is in $Q$.
        If we find such a pair, we tentatively assign $a$ to that line.
        Since we know all $a_i$ are distinct, this assignment should be unique.
        
    3.  After finding all $a_i$ and mapping them to intersection points on $y=0$ (giving $-b_i/a_i$), we can find $b_i = -a_i \times (\text{intercept}_i)$.
    
    Wait, floating point precision issues?
    Queries return doubles? The problem statement says "returns the sum of distances". Standard float/double arithmetic.
    $x, y$ can be real.
    However, we need to handle the case where $p_i$ and $q_j$ are very close due to precision?
    Constraints: $a_i, b_i$ up to $10^4$.
    Intercepts $p_i = -b_i/a_i$. Magnitude $\approx 10^4/1$. Max $\approx 10^4$.
    Difference $1/a_i$ is at least $1/10^4 = 10^{-4}$.
    If $a_i$ are small integers, difference is large. If large, difference is small.
    Double precision has ~15 digits. $10^{-4}$ is well within precision.
    The sum of distances is computed with double.
    Error margin: $\approx 10^{-10}$ maybe?
    With query count $Q_{max} = 10000$, we can afford sampling dense enough to find kinks.
    However, binary search is safer to pinpoint kinks.
    For each interval between sorted known kinks, the function is linear.
    We can estimate the slope by comparing values at endpoints.
    Actually, since the function is convex, we can just ternary search for minima? No, minima doesn't give kinks.
    We want to find ALL kinks.
    Since we know there are exactly $N$ kinks in $x \in [\min(-b/a), \max(-b/a)] \approx [-10^4, 10^4]$.
    We can just sample with step size $\delta$.
    But we don't know exact locations.
    Actually, since $a_i, b_i$ are integers, the kinks are at rational coordinates.
    The density of kinks might be low if they are separated.
    However, in worst case they are clustered?
    Max range is $10^4$. Min gap between $p_i$'s and $p_j$'s?
    They are distinct rationals with denominators up to $10^4$.
    Gap can be small, e.g., $|1/10000 - 1/10001| \approx 10^{-8}$.
    Double precision is good enough to distinguish them?
    With $10^{-8}$ resolution, standard double works.
    Sampling strategy:
    Sweep $x$ from $-10^5$ to $10^5$ (covering all possible intercepts).
    Check derivative?
    Better: The function $S(x)$ is convex.
    If we query $x$ at regular intervals, say $\Delta x$.
    If $S(x+\Delta x) - S(x)$ changes significantly compared to neighbors, we are near a kink.
    Actually, just collecting $S(x)$ at many points is enough.
    Wait, we need to know WHICH point corresponds to which kink.
    Since the kinks are isolated, if we sort the query points $x_1 < x_2 < \dots$, and see the discrete second derivative or first derivative jump.
    Or simply: We can binary search for each interval where derivative changes.
    But we need the full sequence.
    
    Wait, there is a catch. The kinks appear on $y=0$.
    The "derivative" of $S(x)$ jumps.
    Jump height is related to $a_i/\sqrt{a_i^2+1}$.
    Since $a_i$ integers, jumps are distinct?
    $f'(a) = -|a|/\sqrt{a^2+1}$.
    Values for $a=1$ vs $a=2$.
    $|-1|/\sqrt{2} \approx 0.707$. $|-2|/\sqrt{5} \approx 0.894$.
    Yes, distinct.
    
    Plan:
    1.  Run a sweep on $y=0$. Identify all $N$ critical x-values (let's call them $P_1, \dots, P_N$).
        To do this robustly:
        Query $x$ at $x_0 = -10^6, x_1 = -10^6 + \delta, \dots$.
        Detect breaks by monitoring the value of the derivative approximation $(S(x) - S(x-\delta))/\delta$.
        Actually, simpler: Just sample at a rate sufficient to capture transitions.
        Since max coordinate range is $10^4$, and min separation is roughly $10^{-8}$ (for large denominators), sampling frequency of $10^{-5}$ might miss things?
        Wait, $1/10000 - 1/10001 \approx 10^{-8}$.
        If we just query $10^5$ times, step size $\approx 200$ (range $10^5$).
        That's not enough.
        Wait, $a_i \in [-10000, 10000]$. $b_i \in [-10000, 10000]$.
        Range of intercepts $[-b/a]$ is roughly $[-10000, 10000]$.
        If we sample $2000$ points, step size is 50. Too coarse.
        We need finer sampling.
        But wait, the constraints say $a_i, b_i$ are *unknown integers*.
        The question is, is the gap always $10^{-8}$?
        If $a_i$ are large, $1/a_i$ is small.
        If $a_i = 10000$, $1/a_i = 0.0001$.
        If $a_i$ are distinct, we can have $10000$ and $-9999$? No.
        $10000$ is integer. $a_i$ distinct.
        So gaps can be quite small.
        However, we can just query *randomly*? No, we need to find exact spots.
        
        Wait, if the gap is that small, maybe we can use the query result magnitude itself?
        Or just use more queries. We have 10,000 queries.
        We can use 2000 for $y=0$ and 2000 for $y=1$.
        Step size $2000 / 20000 = 0.1$ over range $10^4$.
        $0.1$ is much larger than $10^{-8}$.
        This implies we CANNOT resolve the kinks by simple uniform sampling if kinks are dense.
        BUT, notice that $x$ coordinates of intersections are rational numbers $-b/a$.
        Are they distributed randomly?
        Actually, $N$ is small.
        Wait, I am worried about the worst-case distribution of lines forcing dense kinks.
        However, we have $Q_{max} = 10000$.
        If we use binary search to find the kinks?
        We know the function is convex and piecewise linear.
        We know the global minimum is at some $x_{min}$.
        Between kinks, the slope is constant.
        We can detect kinks by checking where the slope changes.
        Algorithm:
        1. Find approximate minimum $x_{min}$ using ternary search (or similar). Range $[-10^5, 10^5]$.
        2. From $x_{min}$, move left. The slope starts at some value.
           Wait, going left from min, slope is negative. It stays constant until next kink.
           Wait, the slope of $S(x)$ along $y=0$ is $\sum \text{sign} \cdot (-a_i/c_i)$.
           Actually, $S'(x) = \sum \frac{-a_i}{\sqrt{a_i^2+1}} \cdot \text{sgn}(-a_i x + b_i)$.
           This derivative is a step function. It changes value exactly at $x$ where $-a_i x + b_i = 0$.
           At $x = -b_i/a_i$.
           So yes, finding points where derivative changes is equivalent to finding kinks.
           
           We can perform this by scanning outwards from the minimum $x_{min}$.
           Start at $x_{min}$.
           Compute approximate slope $m = (S(x_{min}) - S(x_{prev})) / \epsilon$.
           Actually, just take finite difference.
           Walk left until slope matches expected behavior?
           Wait, slope is monotonic (increasing) with $x$ because $S$ is convex.
           The slope changes by increments of magnitude $|a_i|/c_i$ (actually signed sum changes).
           Wait, at a kink for line $i$, the term $|-a_i x + b_i|/\sqrt{}$ goes from decreasing to increasing (wrt $x$)?
           Term is $|a_i x - B|$.
           If $x$ crosses $B/a_i$, the term switches from $-(a_i x - B)/c$ to $(a_i x - B)/c$ (assuming $a_i>0$).
           Derivative goes from $-a_i/c$ to $+a_i/c$. Change is $2a_i/c$.
           So the derivative jumps by $2|a_i|/\sqrt{a_i^2+1}$ (magnitude).
           Since all $a_i$ distinct, these magnitudes might be distinct?
           Not necessarily, e.g. $2/ \sqrt{5} \approx 0.89$, maybe another combo equals it?
           $2/\sqrt{5}$ vs something else.
           But we just detect a jump in derivative.
           
           To locate a jump:
           Current $x$ position. Known slope $s$.
           Move a bit $\delta$. Check new slope.
           If change detected, update $x$.
           How to move optimally?
           Since the jump magnitude can be anything, but $a_i$ are integers.
           Wait, this sounds complicated to implement perfectly with floats.
           
           IS THERE ANOTHER WAY?
           Let's revisit the input range.
           Coordinates $[-10^{12}, 10^{12}]$.
           $a_i, b_i \in [-10^4, 10^4]$.
           The kinks are within $[-10^4, 10^4]$.
           Wait, why did I worry about $10^{-8}$?
           Ah, because we want to FIND THE EXACT LOCATIONS.
           If we query at $x=0$, $x=10^{-5}$, etc?
           If the kinks are spaced by $10^{-8}$, uniform sampling misses them.
           However, we know that the kinks are rational numbers $-b/a$.
           The values of $x$ are of form $P/Q$.
           Maybe we can exploit the integer nature?
           But we don't know $a, b$.
           
           Let's use the property that we have 10000 queries.
           We can just perform binary search to find *each* kink.
           We know the function is convex.
           We need to find all roots of $-a_i x + b_i = 0$.
           Wait, finding roots of unknown lines?
           Wait, the function $S(x)$ on $y=0$ is $S(x) = \sum \frac{|-a_i x + b_i|}{\sqrt{a_i^2+1}}$.
           This function has minimum at some $x^*$.
           If we perturb $x$ slightly from $x^*$, does it help?
           No.
           However, notice that if we take TWO different horizontal lines $y=0$ and $y=1$.
           We get two sequences of x-intercepts $X_1, \dots, X_N$ and $Y_1, \dots, Y_N$.
           For the same line, the x-coordinate shifts by $1/a$.
           If we can determine $a$, we know the shift.
           
           Wait, what if we assume we know $a_i$?
           If we knew all $a_i$, then for each line $i$, we would have intersection points $x_{i, y} = (y - b_i)/a_i$.
           If we know $a_i$, we can write $b_i = y - a_i x_{i, y}$.
           So if we find intersection points for two $y$'s, say $y_1$ and $y_2$, for the SAME line $i$:
           $x_{i, 1}$ satisfies $x_{i, 1} = (y_1 - b_i)/a_i$.
           $x_{i, 2}$ satisfies $x_{i, 2} = (y_2 - b_i)/a_i$.
           Eliminating $b_i$: $a_i (x_{i, 2} - x_{i, 1}) = y_2 - y_1$.
           So $a_i = \frac{y_2 - y_1}{x_{i, 2} - x_{i, 1}}$.
           If we can correctly match $x_{i, 1}$ with $x_{i, 2}$, we recover $a_i$.
           Then $b_i = y_1 - a_i x_{i, 1}$.
           
           So the core difficulty remains: MATCHING the intercepts.
           And RECOVERING the set of intercepts (finding the locations).
           
           Recovering locations of kinks on $y=0$.
           We can find the locations of kinks using binary search if we can evaluate the derivative?
           We don't have direct access to derivative.
           But we can approximate it.
           If we sample $S(x)$ at sufficiently dense intervals, we can spot local minima/maxima of the derivative (which are jumps).
           Actually, since $S(x)$ is piecewise linear, the slope is constant.
           So if we query $x_1, x_2, x_3$, the differences $S(x_2)-S(x_1)$ and $S(x_3)-S(x_2)$ should be proportional to $(x_2-x_1)$ and $(x_3-x_2)$ times constant slope.
           If there is a kink in $(x_1, x_2)$, then the apparent slope will change abruptly.
           To guarantee finding the kink, we need $\delta$ smaller than the minimum distance between kinks.
           Minimum distance between two rationals with denominator $\le 10000$ and numerators $\le 10000$?
           The difference is $|\frac{n_1}{d_1} - \frac{n_2}{d_2}| = \frac{|n_1 d_2 - n_2 d_1|}{d_1 d_2}$.
           Numerator is integer $\ge 1$ (if distinct). Denominator $\le 10^8$.
           So min difference $\ge 10^{-8}$.
           This is problematic for uniform sampling with $Q=10000$.
           But we can use an adaptive strategy!
           
           We know the function $S(x)$ is convex.
           We can sample widely, say every 100 units.
           If the "local linearity" assumption fails (i.e. slope varies too much between adjacent points), we suspect a kink is between them.
           Then we zoom in and repeat (divide segment in half, etc).
           Like finding roots or local extrema, but finding regions where slope changes.
           Since there are at most 100 segments, we can refine them.
           Wait, if we zoom in, we need to maintain state.
           Actually, since we just need to find ALL $x$-coordinates where slope changes.
           We can start with a large interval covering all possible intercepts, say $[-10^4 - \epsilon, 10^4 + \epsilon]$.
           Split into two halves, pick midpoint $m$. Query $S(m)$?
           Not enough info to decide if a kink exists in left or right or split.
           However, we know the function is convex.
           We can find the MINIMUM $x^*$ where derivative is zero (if any) or minimum value.
           From $x^*$, we can walk out.
           Going Left from $x^*$: slope starts positive (or zero) and increases to negative? No, convex means derivative goes from negative to positive.
           So to the left of minimum, derivative is negative. As we decrease $x$, derivative becomes more negative?
           Wait. $S''(x) \ge 0$. $S'(x)$ is increasing.
           Left of minimum, $S'(x) < 0$. Right of minimum, $S'(x) > 0$.
           At $x \to -\infty$, $S'(x) \approx \sum -|a_i|/c_i$.
           At $x \to +\infty$, $S'(x) \approx \sum |a_i|/c_i$.
           So $S'(x)$ is increasing from some negative value to some positive value.
           The points where $S'(x)$ jumps correspond to the kinks.
           Since $S'(x)$ is non-decreasing (step function), we can find the points of discontinuity.
           We can binary search for the transition points.
           How to binary search?
           We need to know the current slope level.
           If we query at $x_a, x_b, x_c$, we can check if slope is consistent.
           But maybe easier:
           Just run the "zoom in" procedure on the whole range?
           Since we have $Q=10000$, we can definitely afford recursive subdivision.
           Algorithm:
           1. Define range $R = [-20000, 20000]$.
           2. Function `find_kinks(left_bound, right_bound)`? No.
           The kinks are isolated.
           We can perform a sweep line.
           Pick a starting point, say minimum of domain?
           Actually, we can just query $S(x)$ at points spaced exponentially or adaptively?
           
           Actually, a simpler logic:
           We have $N$ lines.
           We can assume they are ordered by intercept? No.
           But we know $a_i$ are integers.
           What if we query points specifically to test for integer slopes?
           No, $x, y$ are continuous.
           
           Let's go with the "adaptive search for derivative changes".
           We need to find all points where the derivative of $S(x)$ (along $y=0$) jumps.
           We know the function $f(x) = S(x, 0)$.
           $f(x)$ is convex.
           We can find the minimum $x_{min}$ where $f'(x) \approx 0$. (Use ternary search).
           Then, expand from $x_{min}$.
           Actually, better:
           Use a stack of intervals. Initially $\{[L, R]\}$.
           While stack not empty:
             Pop interval $[u, v]$.
             Midpoint $m = (u+v)/2$.
             Evaluate slopes in left half?
             Wait, checking if slope changes requires evaluating at least 3 points.
             Let's use a fixed number of queries to probe.
             Since $N$ is small, we can just find all $N$ kinks.
             
           Strategy:
           1. Find minimum $x_{opt}$ on $y=0$ using Ternary Search. (Cost ~100 queries).
           2. Iterate from $x_{opt}$ outwards to left and right?
              Actually, since we know there are $N$ kinks and function is convex, we can try to find kinks by checking the "curvature" or slope variations.
              Since we have $Q_{max} = 10000$, we can perform a "binary search for each kink"?
              But we don't know where they are.
              However, the union of all kinks spans the range where $f'$ changes.
              If we pick a range large enough to contain all kinks.
              Say $[-10^4, 10^4]$.
              Divide this range into $M$ bins?
              If we have budget for 100 queries, we can't sample too densely.
              
              Wait! The problem allows us to send "non-integers".
              But also, if we just want to recover the lines, maybe we don't need the kinks?
              No, finding the lines seems tied to finding kinks.
              
              Let's step back. Is there a simpler way?
              What if we pick 2 points on $y=0$ with very different $x$?
              Or rather, use the fact that $N$ is up to 100.
              Maybe just outputting the answer immediately? No, we need to know.
              
              Wait, I recall a similar problem from a contest.
              It was Codeforces Round #800 Div 2? No.
              Problem "Lines" on CF.
              It turned out that one can simply query $x=0, y=0$ and then $x=0, y=1$, etc?
              Or use specific points.
              Wait, if we query $x=0, y=y_k$.
              $S_k = \sum \frac{|y_k - b_i|}{\sqrt{a_i^2+1}}$.
              If we pick $y_k$ very large, say $Y$.
              Terms with $|b_i|$ negligible compared to $Y$.
              $S \approx \sum \frac{Y - b_i \text{sgn}(Y)}{\sqrt{a_i^2+1}}$?
              Wait, $|Y - b_i| \approx Y - b_i$ if $Y$ is large enough and $b_i$ is small?
              Actually, distance to line $ax+b=0$ at $(0, Y)$.
              Line eq: $a_i x - y + b_i = 0$? No $y = a_i x + b_i \implies a_i x - y + b_i = 0$.
              Distance: $|a_i(0) - Y + b_i| / \sqrt{a_i^2+1} = |b_i - Y| / \sqrt{a_i^2+1}$.
              If $Y$ is large positive ($> 10^4$), then $b_i - Y < 0$.
              So distance is $(Y - b_i) / \sqrt{a_i^2+1}$.
              Sum $S_Y = \sum \frac{Y - b_i}{\sqrt{a_i^2+1}} = Y \sum \frac{1}{\sqrt{a_i^2+1}} - \sum \frac{b_i}{\sqrt{a_i^2+1}}$.
              This gives us two unknown sums: $\sum \frac{1}{\sqrt{a_i^2+1}}$ and $\sum \frac{b_i}{\sqrt{a_i^2+1}}$.
              We need to solve for $N$ lines. This single query gives one scalar equation.
              We need $N$ equations. So we need $N$ queries of type $Y$ (with $x=0$).
              However, we need to decouple $a_i$ from $b_i$.
              
              Notice that the coefficients $1/\sqrt{a_i^2+1}$ depend on $a_i$.
              If we know the set $\{a_i\}$, we can treat $1/\sqrt{a_i^2+1}$ as known constants.
              Then the second term $\sum \frac{b_i}{\sqrt{a_i^2+1}}$ is a linear combination of $b_i$'s.
              But we don't know which $b_i$ corresponds to which $a_i$.
              However, if we know $\{a_i\}$, we can permute $b_i$'s to fit the values?
              Actually, if we do this for $Y=1$, $Y=2$, ..., $Y=N$?
              Wait, if we choose $Y$ large enough such that $Y > b_i$, the expression holds.
              But wait, for $x=0$, we assumed the term is $|Y - b_i|$.
              Since $b_i$ is around $10^4$, choosing $Y$ very large ensures $Y - b_i > 0$.
              Then $|Y - b_i| = Y - b_i$.
              Then we have linear system:
              $S_Y = Y A - B$ where $A = \sum \frac{1}{\sqrt{a_i^2+1}}$ and $B = \sum \frac{b_i}{\sqrt{a_i^2+1}}$.
              Wait, this gives the SAME linear system regardless of $Y$?
              $S_Y = Y (\sum \frac{1}{\dots}) - (\sum \frac{b_i}{\dots})$.
              So $S_{Y1} = Y_1 A - B$, $S_{Y2} = Y_2 A - B$.
              This is just 1 equation. Two variables.
              It doesn't help identifying individual lines unless we know the values.
              
              We need more variation.
              If we use $x \neq 0$.
              For $x=X$ (large), $y=Y$ (large).
              The term is $|-a_i X + Y - b_i| / \sqrt{a_i^2+1}$.
              If we make $X$ large enough, the $-a_i X$ term dominates.
              So $|-a_i X + Y - b_i| \approx |a_i X|$.
              More precisely, if we choose $X$ such that $X$ is very large, say $10^{10}$.
              Then $-a_i X + Y - b_i$.
              Since $a_i \ge -10^4$ and $X=10^{10}$, term is $\approx -a_i X$.
              Sign of $-a_i X$ depends on sign of $a_i$.
              If $a_i > 0$, term negative. $|...| = a_i X - Y + b_i$.
              If $a_i < 0$, term positive. $|...| = -a_i X + Y - b_i = |a_i| X + Y - b_i$.
              So the sum splits into two parts based on sign of $a_i$.
              Let $Pos = \{i : a_i > 0\}$, $Neg = \{i : a_i < 0\}$.
              Total sum $S \approx \sum_{i \in Pos} \frac{a_i X - Y + b_i}{\sqrt{a_i^2+1}} + \sum_{i \in Neg} \frac{|a_i| X + Y - b_i}{\sqrt{a_i^2+1}}$.
              Group by terms involving $X, Y, b_i$.
              Coefficients for $X$: $\sum_{Pos} \frac{a_i}{\sqrt{a_i^2+1}} + \sum_{Neg} \frac{|a_i|}{\sqrt{a_i^2+1}}$.
              Since $a_i$ appears as $|a_i|$ here, this doesn't tell us sign of $a_i$ easily?
              Wait, $a_i$ in the numerator is the actual $a_i$.
              For $i \in Pos$, coeff is $a_i/\sqrt{}$. For $i \in Neg$, coeff is $|a_i|/\sqrt{} = -a_i/\sqrt{}$.
              So effectively coeff is $\frac{|a_i|}{\sqrt{a_i^2+1}}$. This is independent of sign of $a_i$.
              So large $X$ loses sign information of $a_i$.
              
              Okay, what if we use the fact that $a_i$ are integers?
              We can iterate over all possible $a_i$? No.
              
              Back to the KINK approach.
              It is the most promising because finding exact intersection points gives us strong constraints.
              We need to refine the search for kinks.
              We established min separation is $\approx 10^{-8}$.
              But actually, we only need to recover lines.
              Maybe we don't need perfect kink locations, just accurate enough to infer $a, b$.
              If we find $x_k$ and $y_k$ (on $y=0$ and $y=1$), the difference $y_k - x_k = 1/a_i$.
              If we have error $\epsilon$ in $x_k, y_k$, the error in $1/a_i$ is $\epsilon$.
              Then error in $a_i$ estimation? $a_i \approx 1/(val)$.
              If error in val is small, we can identify $a_i$.
              Given $Q_{max}=10000$, we can surely afford to spend time refining.
              Adaptive algorithm:
              1. Identify range of interest for $x$ intercepts on $y=0$. Range $[-10^4, 10^4]$?
                 Wait, $b_i$ up to $10^4$, $a_i$ can be 1. Intercept $\approx -10^4$.
                 If $a_i$ is small, say 0? No, "No two lines parallel" and "integer". 0 is allowed.
                 If $a_i = 0$, intercept is $-b_i/0$ (undefined).
                 Wait, if $a_i=0$, line is $y=b_i$.
                 Intersection with $y=0$ is NONE (or whole line if $b_i=0$).
                 Wait, if $a_i=0$, distance to $(x, 0)$ is $|0 - 0 - b_i|/1 = |b_i|$. Constant w.r.t $x$.
                 So lines with $a_i=0$ do NOT produce kinks on $y=0$. They add constant offset to the function.
                 Lines with $a_i \neq 0$ produce kinks.
                 So we can recover all lines with $a_i \neq 0$ using the kink method.
                 Then handle $a_i = 0$ separately.
                 
                 If $a_i \neq 0$, intercepts are rational.
                 We can search for kinks.
                 Algorithm:
                 Scan $x$ from $-10000$ to $10000$.
                 Use binary search to find each kink.
                 Since we know there are at most 100 kinks.
                 How to trigger binary search?
                 We can use the property of convexity.
                 But wait, we need to detect a jump in derivative.
                 Since we can't query derivative, we query function values.
                 If $S(x)$ curve is "flat-ish" or linear, slope is stable.
                 If it bends, kink is nearby.
                 Actually, simpler:
                 Just use ternary search to find the GLOBAL MINIMUM of $S(x)$.
                 Let this be $x_{min}$.
                 This is NOT a kink necessarily.
                 But it helps us anchor.
                 
                 Actually, let's use the property that we have 10000 queries.
                 We can query $S(x)$ at $x_1 = -10000$, $x_2 = -9999$, ..., $x_{20000} = 10000$.
                 That's 20000 queries. We have 10000 limit.
                 But we can be smarter.
                 We only need to find the locations of the $N$ kinks.
                 And $N \le 100$.
                 So we can query roughly $2N$ points? No, we need precision.
                 But maybe we can find them recursively.
                 
                 Recursive Function `FindKinks(min_x, max_x, k_count)`:
                 Base case: if $k\_count == 0$, done? No, we don't know count yet?
                 Wait, $N$ is known.
                 So we expect to find $N$ kinks.
                 But actually, $N$ includes the ones with $a_i \neq 0$.
                 Let's assume all $a_i \neq 0$ for now.
                 We want to partition the range into $N$ sub-segments?
                 No, we want to find the boundaries.
                 Since we have $Q_{max} = 10000$, we can just allocate $\approx 100$ queries per kink detection?
                 Wait, we can use a "density" approach.
                 If we sample a range $[L, R]$ with $k$ samples.
                 We can check if the "total curvature" inside is consistent with 0, 1, or >1 kinks.
                 But calculating curvature from samples is noisy.
                 
                 Let's pivot to a strategy that doesn't require finding ALL kinks first.
                 What if we just guess the set of slopes?
                 There are few combinations? No.
                 
                 Let's look at the problem statement again.
                 $a_i, b_i$ integers.
                 Maybe we can guess $a_i$ by looking at the derivatives?
                 
                 Actually, there's a much simpler trick.
                 If we pick a point $(x, y)$ far away, say $x=10^{12}, y=0$.
                 Then distance to line $i$ is approx $|-a_i x|/c_i$.
                 $S \approx \sum \frac{|a_i| x}{\sqrt{a_i^2+1}} = x \sum \frac{|a_i|}{\sqrt{a_i^2+1}}$.
                 This doesn't help distinguishing lines.
                 
                 Wait, I'm overcomplicating.
                 The intended solution likely involves exploiting the discrete nature of $a_i, b_i$.
                 With $Q_{max}=10000$, and $N \le 100$.
                 We can afford to solve for each line independently IF we can isolate them.
                 
                 Let's reconsider the "Intersecting lines" logic.
                 We found that finding $p_i$ (intercept of $y=0$) and $q_i$ (intercept of $y=1$) allows finding $a_i$.
                 $q_i - p_i = 1/a_i$.
                 If we can identify the pairs $(p_i, q_i)$, we win.
                 The only issue is finding $p_i$'s accurately.
                 Let's focus ONLY on finding $N$ points $p_1, \dots, p_N$ on $y=0$.
                 We know they are in $[-10^4, 10^4]$ (roughly).
                 We know they are rational numbers.
                 And we have $N$ of them.
                 Wait, if we pick $x$ values to be integers.
                 Is it possible that $p_i$ is never an integer?
                 Yes.
                 However, notice that we can determine $p_i$ by solving for root of line.
                 
                 Here is a viable plan:
                 1. Query a lot of points on $y=0$ to build a piecewise linear model.
                    But doing this efficiently:
                    We can use a priority queue to explore the most likely regions?
                    Or simply:
                    Since we need high precision, let's do this:
                    Query at $x = -10000, -9999, ..., 10000$?? No, too many.
                    But wait, if we sample sparsely, say every 50 units.
                    We get points where function is convex.
                    If the derivative jumps, the average slope changes.
                    We can detect if there are any kinks in an interval by measuring the deviation from linearity.
                    E.g. check $S(mid)$ vs interpolated from $S(L), S(R)$.
                    If $S(mid)$ deviates from linear interpolation significantly, there is a kink (or multiple).
                    Wait, distance function is convex, so $S(mid) \le (S(L)+S(R))/2$.
                    Actually strict convexity means strict inequality if kink inside.
                    So, we can use "Divide and Conquer" to find the locations of kinks.
                    Start with range $[-20000, 20000]$.
                    Midpoint $m$. Check convexity condition?
                    Wait, the condition $S(mid) < (S(L)+S(R))/2$ doesn't prove kink. It proves curvature.
                    For a piecewise linear convex function, strict inequality holds in EVERY sub-interval containing at least one kink?
                    No, if it's linear (no kink), equality holds.
                    So, if $S(m) \approx (S(L)+S(R))/2$, then NO kinks in $(L, R)$.
                    Wait, if there are NO kinks, the function is a straight line on $[L, R]$.
                    If there IS a kink, it bends, so $S(m)$ will be strictly below the chord.
                    So we can test an interval.
                    If strictly convex, then there is at least 1 kink.
                    But how many?
                    If we keep splitting, eventually we isolate single kinks.
                    
                    Algorithm for finding all kinks:
                    1. Initialize a list of intervals $I = \{[-20000, 20000]\}$.
                    2. We know total kinks = $N_{active}$.
                       We don't know $N_{active}$ (lines with $a_i \neq 0$).
                       But we know $N$ lines total.
                       We can just run this until no intervals remain.
                       But we need to be careful about floating point errors and query cost.
                       Since we have $10^4$ queries, and we can perform binary search-like operations.
                       Number of intervals can grow up to $N$.
                       If we split each interval, we might exceed query limit if we are not smart.
                       
                       Actually, we can just collect the intervals one by one?
                       No, they can be anywhere.
                       However, since we can stop an interval if we confirm it contains 0 kinks.
                       How to confirm 0 kinks?
                       Check linearity.
                       Query $L, m, R$. If $S(m)$ matches linear interp, likely 0 kinks.
                       Wait, floating point precision might mask it.
                       If $S(m)$ differs by $\epsilon$, we assume 1 kink? Or multiple?
                       If we are deep in recursion, the interval is small.
                       
                       Correctness of linearity check:
                       If the interval $[L, R]$ contains NO kinks, then $S(x)$ is linear on $[L, R]$.
                       So $S(m)$ must equal $(S(L)(R-m) + S(R)(m-L))/(R-L)$? No, just $(S(L)+S(R))/2$ if $m$ is mid.
                       Wait, if $L, m, R$ are equally spaced, linear implies $S(m) = (S(L)+S(R))/2$.
                       If there is a kink, $S(m) < (S(L)+S(R))/2$.
                       The deviation scales with square of interval width?
                       For piecewise linear function with a corner at $k \in (L, R)$.
                       The drop is roughly proportional to area of triangle formed by kink.
                       If interval is small, deviation is small.
                       So we can check if deviation is significant.
                       Since we have 15 bits of mantissa (double), we can detect deviations as long as function value isn't tiny and drop is resolvable.
                       Function values are up to $\approx 10^4 \times 100 \approx 10^6$.
                       Precision $\approx 10^{-10}$.
                       If kink width is $10^{-4}$, deviation $\approx (10^{-4})^2 \times \text{slope\_change}$.
                       Slope change $\approx 1$. Deviation $\approx 10^{-8}$.
                       This is resolvable.
                       
                       So:
                       We can perform a divide-and-conquer search for all kinks.
                       Function `Search(L, R, target_kinks)`.
                       Wait, we don't know how many kinks in $[L, R]$.
                       Actually, we just need to find ALL of them.
                       We can start with one big interval.
                       Check mid $m$.
                       If linear ($|S(m) - (S(L)+S(R))/2| < \theta$), then NO kinks in $(L, R)$?
                       Wait, this assumes at most 1 kink? No.
                       If there are 0 kinks, it is linear.
                       If there are $\ge 1$ kinks, is it guaranteed to be non-linear? Yes, piecewise linear with a bend is non-linear.
                       So if it IS linear, we discard.
                       If NOT linear (strictly concave-ish towards origin), then there is at least one kink.
                       Does "not linear" mean exactly one kink?
                       It means "at least one".
                       If we split $[L, R]$ into $[L, m]$ and $[m, R]$.
                       We can recurse.
                       Since max depth is bounded?
                       If we have 100 lines, we can have 100 kinks.
                       Total leaves in recursion tree $\approx 200$?
                       Number of queries needed $\approx N \times (\text{depth})$.
                       If we just split intelligently?
                       Actually, simpler:
                       Since $N$ is small, maybe we can just find $N$ points.
                       
                       Revised Algorithm for Kinks:
                       Queue of segments $Q$. Initially $\{[-20000, 20000]\}$.
                       Loop while $Q$ not empty:
                         Pop $[l, r]$.
                         Mid $m = (l+r)/2$.
                         Query $l, m, r$.
                         Calculate deviation.
                         If deviation is "small enough":
                           Mark $[l, r]$ as flat (contains no kinks). Continue.
                         Else:
                           Push $[l, m], [m, r]$ to queue.
                   
                       But wait, this might create $2^N$ intervals?
                       Yes, if we just recurse blindly.
                       But we know total kinks is $N$.
                       So if a segment contains multiple kinks, we split.
                       Eventually each leaf contains 0 or 1 kink?
                       Wait, if a leaf contains 0 kinks, the deviation will be 0.
                       So we won't push children if it's linear.
                       But wait, floating point error might make it look non-linear even if linear.
                       Also, if there are MULTIPLE kinks in a segment, deviation is still non-zero.
                       But if we just want to find the kinks, we might end up with a leaf that has multiple kinks.
                       How to ensure leaves have 0 or 1 kinks?
                       We can add a constraint: if segment length is small enough (below threshold), stop.
                       But we want to find exact location.
                       
                       Actually, since we have 10000 queries, we can probably just query enough to find them.
                       The problem is efficiency.
                       However, we can optimize:
                       We want to find locations of $N$ points.
                       We can try to sample uniformly first?
                       If we sample 2000 points in range of 40000. Gap is 20.
                       If kinks are separated by > 20, we hit them.
                       If kinks are closer, we need to zoom in.
                       But we only need to zoom in where necessary.
                       Most kinks are likely separated? No guarantee.
                       But wait, with $a_i$ distinct integers, and $b_i$ integers.
                       Separation can be arbitrarily small.
                       BUT, maybe we can assume the problem constraints allow standard search?
                       Let's check constraints: $a_i \in [-10^4, 10^4]$.
                       Worst case: $10000$ distinct values? No $N \le 100$.
                       Wait, the integers are small range?
                       If we have $a_i = \{1, 2, \dots, 100\}$.
                       Then intercepts $-b_i/a_i$.
                       If $b_i$ are tuned, they can be very close.
                       
                       Okay, let's look at the query budget again. $10^4$.
                       We can afford $O(N \log(\text{Range}/\epsilon))$ queries.
                       We need to find $\approx 100$ kinks.
                       If we perform binary search for EACH kink?
                       We don't know positions.
                       
                       Let's combine strategies.
                       1. Use "Linear Check" to find "dense" regions.
                          Actually, if we have 100 kinks, and we just split everything, we end up with a quadtree of sorts.
                          The total number of nodes visited to isolate $N$ features in 2D (interval splitting) is roughly $O(N)$.
                          Why? Because each split adds 1 node. We stop when interval is "flat" or size is small.
                          If we stop when size < $\delta$, we need $\delta$ such that we can resolve kinks.
                          We want to isolate kinks to $\approx 10^{-9}$ accuracy?
                          Actually, just need accuracy $\approx 10^{-5}$ to distinguish $1/a$.
                          Range $4 \cdot 10^4$. Accuracy $10^{-5}$ means factor $10^9$.
                          Splitting needs $\approx 30$ levels.
                          Total queries might be high.
                          
                       But wait!
                       If we query $x=0$, $x=\text{huge}$, we get $A$ and $B$ sums.
                       We already explored this. It doesn't help identifying lines.
                       
                       Let's reconsider the "Guess $a_i$" approach.
                       $a_i$ are integers.
                       Can we iterate over all possible sets of $a_i$? No.
                       Can we determine $a_i$ individually?
                       
                       What if we use the query "distance to line"?
                       $D(x, y) = |Ax+By+C|/\sqrt{A^2+B^2}$.
                       If we query $(x, y)$ such that $y$ is integer multiple of $x$?
                       Or $y=0$.
                       If we query $(x, y) = (0, b)$ for all $b \in [-10000, 10000]$?
                       If we pick $y$ equal to some $b_i$, the distance for that line is 0?
                       No, distance to line $y = a_i x + b_i$.
                       At $(0, b)$, dist $= |b - b_i| / \sqrt{a_i^2+1}$.
                       If we pick $b = b_i$, dist $= 0$.
                       But we don't know $b_i$.
                       If we query $y = Y$ such that it lies ON one of the lines at some $x$?
                       This is getting circular.
                       
                       Let's go back to the code implementation requirements.
                       We just need to submit code.
                       The constraints are tight but $Q$ is generous.
                       
                       Implementation plan:
                       We will attempt to find the intercepts of lines with $y=0$ using a recursive search.
                       We assume there are $N_{active} \le N$ active lines (those with $a_i \neq 0$).
                       Lines with $a_i = 0$ contribute a constant distance and don't affect $S(x)$ shape (linear part).
                       We can detect them later.
                       
                       Search Logic:
                       Function `solve(l, r)`:
                         Returns list of intercepts in $[l, r]$.
                         Base case: if interval width is small, or too many kinks detected?
                         Actually, just check if function is linear.
                         To check linearity, sample 3 points $l, m, r$.
                         If non-linear, recurse.
                         
                       Refined Check:
                       $S_L = \text{query}(l, 0)$, $S_R = \text{query}(r, 0)$, $S_M = \text{query}(m, 0)$.
                       Linear Expectation $E = (S_L(S_R - S_M) + S_M(S_L - S_R))/...$ No.
                       Just $E = (S_L + S_R)/2$ for midpoint.
                       Error $\epsilon = |S_M - E|$.
                       Threshold for "non-linear": e.g. $10^{-7}$.
                       If $\epsilon > \text{thresh}$, then kink(s) exist.
                       Recurse.
                       If $\epsilon < \text{thresh}$, mark as flat.
                       
                       Wait, if there are multiple kinks, error accumulates?
                       Yes, but we just want to detect presence.
                       However, this might lead to $O(2^N)$ complexity in worst case if we split everything unnecessarily?
                       No, if it's not linear, we split.
                       Eventually intervals become small or flat.
                       Since we need to find exact kink locations, we can stop splitting when we have isolated the kink sufficiently.
                       How to isolate?
                       If we split $[l, r]$ and find both halves are linear, then the original was linear.
                       If one half is non-linear, we keep it.
                       Basically, we prune linear branches.
                       We only visit nodes that contain at least 1 kink.
                       There are at most $N$ such nodes (leaves containing kinks).
                       Wait, internal nodes containing multiple kinks will split until kinks are separated.
                       If kinks are extremely close, we split deeply.
                       Depth depends on gap size.
                       If gap is $10^{-8}$, depth is $\approx 30$.
                       Total queries $\approx N \times 30 + \text{overhead}$.
                       $100 \times 30 = 3000$ queries.
                       Plus overhead for checking linearity.
                       Seems safe within $10^4$.
                       
                       Corner case: Multiple kinks VERY close together (distance $< 10^{-8}$).
                       Given integer constraints, maybe not?
                       $p_1 = -b_1/a_1$, $p_2 = -b_2/a_2$.
                       $|p_1 - p_2| = |a_2 b_1 - a_1 b_2| / (a_1 a_2)$.
                       Numerator $\ge 1$. Denominator $\le 10^8$.
                       Min distance $\approx 10^{-8}$.
                       This is the bottleneck.
                       Our binary search precision needs to handle this.
                       But with $10^{-12}$ tolerance (double), it's risky.
                       However, we don't need $10^{-8}$ precision for $x$-location to identify the line if we have another constraint.
                       If we have two lines $L_1, L_2$ with very close x-intercepts.
                       On $y=0$, they cross near same point.
                       On $y=1$, they cross at $x' \approx x \pm 1/a$.
                       If $a_1, a_2$ are different (they must be distinct integers), then $1/a_1 \neq 1/a_2$.
                       So the shifts $1/a_1$ and $1/a_2$ are different.
                       If we can find the set of $x$-intercepts for $y=0$ and $y=1$.
                       Even if $x_1, x_2$ are close in $y=0$ list.
                       We need to match them correctly.
                       If $x_1 \approx x_2$, then for any $x \in \{x_1, x_2\}$, we look at $x' \in \{y\text{-intercepts}\}$.
                       $x'_i = x_i + 1/a_i$.
                       If $x_1, x_2$ are very close, $x'_1, x'_2$ are separated by $|1/a_1 - 1/a_2| \approx |(a_2-a_1)/(a_1 a_2)|$.
                       Since $a$ distinct integers, $|a_2-a_1| \ge 1$.
                       If $a \approx 10^4$, diff $\approx 10^{-8}$.
                       So the separation in $y=1$ is same order as $y=0$.
                       So resolving precision is the main issue.
                       
                       Wait, if $a_i$ are large, $1/a_i$ is small.
                       If $a_i$ are small, $1/a_i$ is large.
                       But $a_i$ distinct.
                       If $a_i \in \{1, \dots, 100\}$, min diff is $\approx 1/10000$. Large.
                       If $a_i \in \{9000, \dots, 10000\}$, min diff $\approx 1/10^8$. Small.
                       But if $a_i$ are large, $b_i$ also bounded?
                       Yes $b_i \in [-10^4, 10^4]$.
                       Wait, if $a_i$ is large, intercept $-b_i/a_i$ is small (close to 0).
                       So most lines cluster near 0?
                       Range of intercepts is indeed small when $a$ is large.
                       Max intercept spread is when $a=1$, $b=-10^4 \implies 10^4$.
                       But when $a=10000$, spread is $10^4/10000 = 1$.
                       So kinks cluster near 0 for large $a$.
                       Clustering is worse when $a$ is large.
                       However, if $a$ is large, $1/a$ is small, making matching $x$ to $x'$ hard?
                       Actually, if $x, x'$ are both close to 0, and difference is $1/a$.
                       If we can find $1/a$ accurately, we are fine.
                       $1/a$ for $a=10000$ is $10^{-4}$. Easy.
                       $1/a$ for $a=1$ is 1. Easy.
                       The hardest is matching $x$ and $x'$ when $a$ are large AND kinks overlap?
                       Wait, $1/a$ is determined by difference.
                       If we find difference $\delta$, we check if $\delta$ is reciprocal of integer.
                       If we have high precision, we are good.
                       But if $a$ is large, we need precise location of kinks.
                       
                       Wait, if we use the "Linearity Test", we can detect if a region contains ANY kink.
                       If we have a region with multiple kinks that are indistinguishable due to precision, the linearity test will fail.
                       We will split.
                       Eventually we stop when interval is small enough.
                       If kinks are effectively merged in the query resolution, we treat them as one.
                       But wait, lines are distinct.
                       If we merge them, we might think $a$ corresponds to a composite.
                       BUT, we have 10000 queries.
                       If we reach depth 60 (resolution $10^{-4}$?), maybe enough?
                       Range $4 \cdot 10^4$. $2^{-60}$ is extremely small.
                       Actually, we just need to differentiate $1/a$.
                       Smallest diff in $1/a$ is $10^{-8}$.
                       To detect this difference in $x$-coordinates, we need resolution $\approx 10^{-8}$.
                       Interval width $40000$. Ratio $40000 / 10^{-8} = 4 \cdot 10^{12}$.
                       Bits $\approx \log_2(4 \cdot 10^{12}) \approx 42$.
                       Depth 42 is safe.
                       Total queries $\approx 100 \times 42 \approx 4200$.
                       This fits!
                       
                       So we just implement the recursive subdivision until width is small or flat?
                       No, until we have identified the points.
                       Wait, if we have a region with 0 kinks, we stop.
                       If we have 1 kink, we split until... width is small?
                       Or until the value of derivative is clear?
                       Actually, we can simply collect points.
                       We need to output $a_i, b_i$.
                       Once we have a "cluster" of kinks (even if merged), we can't solve for $a_i$ uniquely.
                       But we need $N$ distinct lines.
                       If we can't separate them, we fail.
                       However, problem guarantees distinct $a_i$.
                       If we can't separate kinks, it means we couldn't find them.
                       But given $Q=10000$, we should be able to.
                       Wait, if $a_i=10000$ and $a_j=10001$.
                       Then $1/a_i \approx 10^{-4}$, $1/a_j \approx 9.999 \cdot 10^{-5}$.
                       Diff $\approx 10^{-9}$.
                       We need resolution $10^{-9}$.
                       Bits $\approx 45$.
                       $100 \times 45 = 4500$. Safe.
                       
                       Wait, what about $b_i$?
                       $x_{intercept} = -b/a$.
                       If $b$ changes, $x$ shifts.
                       If $a=10000$, and $b$ varies.
                       $x_1 = -b_1/10000$, $x_2 = -b_2/10000$.
                       If $b_1, b_2$ are distinct integers. Diff $\ge 10^{-4}$.
                       If $a_1 \neq a_2$, diff in $x$ can be smaller?
                       Example $x_1 = -1/10000 = -0.0001$.
                       $x_2 = -1/9999 \approx -0.00010001$.
                       Diff $\approx 10^{-8}$.
                       This is the tightest case.
                       So depth 45-50 is required.
                       With $100$ lines, total nodes visited?
                       If each kink is isolated early?
                       Actually, worst case is all kinks are in small region.
                       If they are in small region, we visit fewer regions?
                       No, we visit small regions for each kink.
                       So roughly $N \times \text{Depth}$.
                       $100 \times 50 = 5000$ queries.
                       We have $10000$.
                       We have margin.
                       
                       So the strategy:
                       1. Query $y=0$ points to find intercepts $P_1 \dots P_N$.
                          Use recursive search `FindIntercepts(L, R, List)`.
                          Actually, we can maintain a global list of found intercepts.
                          Function `Split(L, R, k_count_found_in_region?)`
                          Wait, simpler:
                          Just use a priority queue of segments?
                          Or just stack.
                          Stack elements $(L, R)$.
                          Loop:
                            Pop $(L, R)$.
                            Query $L, M, R$.
                            If almost linear (error < eps), skip.
                            If non-linear, Push $(L, M), (M, R)$.
                          BUT, we need to avoid infinite loops or exploring flat regions with noise.
                          Set a MAX_DEPTH or MIN_WIDTH.
                          MIN_WIDTH $10^{-10}$.
                          If width < min_width, we assume we found the point (it's $L$ or $R$?).
                          Wait, if non-linear at base, we still didn't find the kink location exactly.
                          Just return $L$? Or average?
                          Actually, if width is small, $S(M)$ will likely show the "bump".
                          Wait, if interval is small, $S$ is basically determined by 1 line.
                          Wait, the bump logic:
                          If interval has a kink, $S(M)$ is strictly less than linear interp.
                          If we go down to leaf, we might have multiple kinks if they are super close.
                          But we reasoned they won't be THAT close given integer constraints and range.
                          So we assume each leaf has at most 1 kink.
                          When we stop, we assume the kink is near the region.
                          Actually, we can find the "local minimum" of the function segment to pinpoint the kink more accurately.
                          Since distance sum is convex, in an interval with a kink (and no other kinks), the function is V-shaped.
                          The minimum of V-shape is AT the kink.
                          So, if we isolate an interval containing exactly 1 kink, the global minimum of $S(x)$ restricted to this interval IS the kink!
                          So we don't need to pinpoint exact rational value.
                          We just find the x where the function is minimized (locally).
                          We can do this with ternary search inside the leaf.
                          Or simply, if we stop splitting because it's linear?
                          If it's linear, no kink.
                          If it's non-linear, it has kink.
                          We can just minimize $S(x)$ in $[L, R]$ using ternary search.
                          This gives us precise $x_k$.
                          
                          Revised Logic:
                          1. Initialize list of segments $\{[-10^5, 10^5]\}$.
                          2. While queue not empty:
                             Pop $[l, r]$.
                             If $r - l < \text{threshold}$:
                               Find minimum of $S(x)$ in $[l, r]$ (using ternary search).
                               Store this $x$ as a candidate intercept.
                               Continue.
                             Mid $m = (l+r)/2$.
                             Query $l, m, r$.
                             Check convexity deviation: $\Delta = S(m) - (S(l) + S(r))/2$.
                             Wait, for convex function $f$, $f(m) \le (f(l)+f(r))/2$.
                             Deviation is $(S(l) + S(r))/2 - S(m)$.
                             If deviation is negligible (say $< 10^{-9}$), then linear, no kink.
                             Else:
                               Push $[l, m], [m, r]$.
                            
                          Wait, if we have multiple kinks in $[l, r]$, deviation will be non-negligible.
                          We split.
                          Eventually we have intervals with single kink.
                          Inside a single-kink interval, the function looks like $V$ (or $\Lambda$?).
                          No, distance is $|ax+b|$. Sum of distances is convex.
                          Along $y=0$, $d_i(x) = |-a_i x + b_i|/c_i$.
                          If $a_i \neq 0$, this is V-shaped.
                          The sum is convex.
                          So a single kink means the function decreases then increases.
                          So it has a unique minimum in the interval.
                          So finding the minimum of $S(x)$ in that interval gives us the intercept $x_k$.
                          Perfect.
                          
                          Wait, what about multiple kinks?
                          The sum is still convex.
                          So we keep splitting until we isolate single kinks.
                          Does the function have multiple minima?
                          No, convex function has unique minimum.
                          But it can have inflection points (kinks).
                          Wait, if there are 2 kinks in an interval.
                          $L --- k1 --- k2 --- R$.
                          Slope increases at $k1$ and $k2$.
                          If the slope starts negative at $L$ and ends positive at $R$, the minimum is between $k1$ and $k2$.
                          So ternary search will converge to minimum.
                          But we want to find BOTH $k1$ and $k2$.
                          The minimum location doesn't tell us $k1, k2$ individually.
                          So we MUST split until the interval contains at most 1 kink?
                          Actually, if interval contains $>1$ kink, we need to detect it.
                          If we just check convexity deviation, it's hard to say "is it >1 kink?".
                          But we know total number of kinks is $N_{active}$.
                          Wait, can we just trust the "splitting" to work?
                          If we split $[L, R]$ where there are 2 kinks.
                          Both halves might contain kinks.
                          We continue splitting until each child has 1 kink (or 0).
                          If a child has 2 kinks, it splits again.
                          Eventually, we end up with $N$ intervals, each with 1 kink (plus some empty intervals).
                          This requires finding the kinks.
                          The problem is distinguishing 0 kinks from 1 kink?
                          Deviation check handles 0 kinks (linear).
                          What about 1 kink? Deviation non-zero.
                          So if non-linear, we assume it has kinks.
                          Could it have 3 kinks? Yes.
                          But 3 kinks implies 2 "turns".
                          Actually, convex function cannot have multiple turns. It is piecewise linear with increasing slope.
                          So slope goes $v_1 < v_2 < \dots < v_N$.
                          So if we have 2 kinks, we have 3 linear segments.
                          Checking middle point $m$ vs line $LR$.
                          If we are inside $LR$ but near a kink, it detects it.
                          If we are in middle of 3-segment chain (between $k1, k2$), it might be linear locally?
                          Wait, if we pick $m$ randomly, probability to land between $k1, k2$ (small gap) is low.
                          But we might pick it in the middle of a flat region.
                          Then we split.
                          One child will contain $k1$, other $k2$.
                          Eventually we split $k1$ region and $k2$ region apart.
                          So yes, recursive splitting works to isolate regions with kinks.
                          
                          The only risk is:
                          1. Floating point inaccuracies causing false positives/negatives.
                          2. Infinite recursion if noise persists.
                          
                          Solution:
                          Limit recursion depth (e.g., 60) or min width ($10^{-9}$).
                          Also, if we are unsure (noise), maybe just use wider range.
                          Given $Q_{max}=10000$, and we have roughly 50 queries per line, we should be OK.
                          But we need to find ALL kinks.
                          If we have 100 lines, we need to find 100 points.
                          If we blindly split everywhere, we might blow up queries.
                          Optimization:
                          Only split if deviation is significant.
                          And maybe cap the number of leaves?
                          Wait, we don't know how many kinks there are.
                          But we know $N$.
                          If we split too much, we waste queries.
                          Can we prioritize?
                          We can check "number of kinks" heuristically.
                          Or just stick to the strategy and hope noise is low.
                          Since the inputs are integers, the function values are deterministic and exact (except for distance calc).
                          The calculation of distance involves `sqrt`, which introduces irrationals.
                          But `abs` preserves sign.
                          
                          Wait, the sum is convex.
                          We want to find the breakpoints.
                          
                          Actually, there is a specialized technique for finding kinks in sums of absolute values.
                          But we can't afford complex algos.
                          Let's stick to Divide & Conquer with convexity check.
                          We just need to make sure we terminate.
                          
                          Final Plan for Phase 1:
                          Query $y=0$ intercepts.
                          Initial range $[-20000, 20000]$.
                          Stack of ranges.
                          While stack not empty:
                            Pop $[l, r]$.
                            If $r - l < \epsilon$:
                              Add $l$ to intercepts.
                              (Actually we need to check min to be sure? No, if width small, $l \approx$ kink)
                              Break.
                            Mid $m$.
                            Query $L, m, R$.
                            Check linearity.
                            If linear: discard.
                            Else: push $L, m$ and $m, R$.
                          
                          We need to store collected intercepts to process $y=1$.
                          
                          Wait, we need to collect ALL intercepts.
                          If we discard "linear", we save queries.
                          If we encounter a region with 2 kinks, we split, eventually finding them.
                          Is it possible to have "false linear" regions?
                          If noise makes a linear region look curved?
                          Noise is small. We use loose epsilon.
                          Epsilon $\approx 10^{-9}$ is safe.
                          With double precision, machine epsilon is $10^{-16}$.
                          Errors accumulate.
                          But values are up to $10^6$.
                          Relative error $10^{-15} \times 10^6 = 10^{-9}$.
                          So $10^{-9}$ is borderline.
                          Maybe $10^{-7}$ is better.
                          
                          Phase 2:
                          Now we have a list $P$ of $x$-intercepts for $y=0$.
                          We need $y=1$ intercepts.
                          Do the same for $y=1$. Get list $Q$.
                          Both lists should have size $N$.
                          Then match $p \in P$ with $q \in Q$ such that $q - p \approx 1/a$.
                          Since $a$ are distinct integers, we can just sort $P$ and $Q$?
                          Wait, the order of lines is unknown.
                          But we know $a$ determines the shift.
                          Shift $s = q - p$.
                          If we can identify $s$'s, we can find $a = 1/s$.
                          However, we don't know which $p$ goes with which $q$.
                          But we can try to find ALL $a$'s.
                          We can just find the set of differences.
                          Wait, matching is a bipartite matching problem.
                          But since we know $a_i$ are distinct integers.
                          If we find ALL possible pairs $(p, q)$ such that $q-p = 1/a$ for integer $a$, we can identify lines.
                          Algorithm:
                          Initialize empty set of valid slopes $A$.
                          Valid candidates $a \in [-10000, 10000]$.
                          Iterate through all $p \in P$.
                          For each $a \in \text{Candidates}$:
                            Ideally, if line exists with slope $a$, then there exists $q \in Q$ such that $q \approx p + 1/a$.
                            We check this.
                            Since we don't know which $p$ corresponds to which $a$, we just collect evidence.
                            Wait, we need to reconstruct the list of lines.
                            We need to associate $p$ with $a$.
                            If we find a valid $q$ for a pair $(p, q)$ consistent with $a$, we claim line with slope $a$ intersects $y=0$ at $p$.
                            But we might have multiple choices.
                            However, $a$ must be unique per line.
                            So if we find $p$ matches $q$ for $a$, and no other $p'$ matches $q$ for $a'$, etc.
                            Actually, simpler:
                            For each $p \in P$, we try to find a compatible $a$.
                            Wait, we don't know which $a$.
                            But we know there are $N$ distinct $a$'s.
                            We can just compute the set of ALL potential differences $q - p$.
                            Filter those that are very close to $1/a$.
                            If we have $N$ valid matches, we are done.
                            Since $N \le 100$ and candidates $\approx 10000$.
                            Maybe iterate $a$.
                            For a specific $a$, check if there is a shift $1/a$.
                            Look for $q - p \approx 1/a$ in the set of all pairs.
                            Since $a$ distinct, each $a$ appears at most once as a valid shift.
                            So we can iterate all $a \in [-10000, 10000]$.
                            Check if any pair $(p_i, q_j)$ has $q_j - p_i \approx 1/a$.
                            If yes, record $a$ and the associated $p_i$.
                            Since we need to recover ALL $N$ lines, we should find $N$ such $a$'s.
                            
                            Handling duplicates?
                            Some $a$ might not be present (if $a_i$ outside range? No, constrained).
                            So we loop $a$. If match found, assign line.
                            
                            Then once we have $a_i$ and $p_i$ (x-intercept at $y=0$), we calculate $b_i = -a_i p_i$.
                            Wait, $p_i = -b_i/a_i$. So $b_i = -a_i p_i$.
                            
                            One edge case: Lines with $a_i = 0$.
                            Intercept at $y=0$ is undefined (parallel to $x$-axis).
                            These lines do not contribute to the bumps in $S(x)$ on $y=0$.
                            They contribute to the background constant (actually constant slope? No, slope is const).
                            Wait, $d((x,0), y=b) = |b|$. Constant.
                            So $S(x)$ includes $\sum_{a=0} |b|$. This is added to the whole sum.
                            So for $y=0$, lines with $a=0$ do not create kinks.
                            They DO NOT shift the x-intercepts of others.
                            So the algorithm finds all $a \neq 0$.
                            We need to handle $a=0$ separately.
                            How?
                            We know total $N$.
                            Let $N_{found} = |Found|$.
                            $N_{zero} = N - N_{found}$.
                            We know the values of $b_0 \in [-10000, 10000]$ for these zero-slope lines.
                            But we need to distinguish them?
                            Actually, if multiple lines have $a=0$, they are parallel.
                            Problem says "No two lines are parallel".
                            So AT MOST ONE line has $a=0$.
                            So either 0 or 1 line has $a=0$.
                            If we found $N$ lines with $a \neq 0$, done.
                            If $N-1$, then 1 line has $a=0$.
                            How to find $b$ for $a=0$?
                            We have query values.
                            We can use any $S(x, 0)$.
                            Wait, $S(x, 0)$ doesn't separate $a=0$ line?
                            Actually, $S(x, 0) = \sum_{a \neq 0} \text{Var}(x) + |b_{zero}|$.
                            The variable part comes from $a \neq 0$ lines.
                            If we can calculate the theoretical sum for $a \neq 0$ lines, we can subtract from observed $S(x, 0)$ to find $|b_{zero}|$.
                            Wait, we need to know which query point yields min distance or something.
                            At the optimal $x$ (where $\sum_{a \neq 0}$ is minimized), the gradient is 0.
                            Value is $S_{min} = \sum_{a \neq 0} 0 + \dots$? No, $\sum |a x + b|$.
                            Minimized value is sum of distances to projections.
                            The distance for $a \neq 0$ is 0 at $x_k$? No, at $x_k$, distance to line $k$ is 0.
                            At $x=x_k$, sum of other distances is $\ge 0$.
                            So at some $x$, the sum of $a \neq 0$ terms is 0? No, only for the specific line passing through $(x, 0)$.
                            But we can't make all $a \neq 0$ terms 0 simultaneously.
                            Wait, $S(x)$ for $a \neq 0$ is minimized at some $x_{opt}$.
                            At $x_{opt}$, value is $S_{part}$.
                            If we have $S_{observed}$ at $x_{opt}$.
                            Then $|b_{zero}| = S_{observed} - S_{part}$.
                            Wait, we need $x_{opt}$ to be such that $a \neq 0$ lines are handled.
                            Actually, just pick any $x$, say $x=0$.
                            $S(0, 0) = \sum_{a \neq 0} \frac{|-b_i|}{\sqrt{a_i^2+1}} + |b_{zero}|$.
                            We can calculate $\sum_{a \neq 0} \dots$ using our recovered parameters.
                            So we can find $|b_{zero}|$.
                            Since $b_{zero} \in [-10000, 10000]$, we can just pick sign?
                            $b_{zero} = \pm |b_{zero}|$.
                            Is there a way to determine sign?
                            Query $S(0, 1)$?
                            For $a \neq 0$ lines, $d(0, 1) \neq 0$ generally.
                            For $a=0$ line ($y=b_{zero}$), dist is $|1 - b_{zero}|$.
                            This changes with $b$.
                            If we solve for $b_{zero}$ from 2 points, we are good.
                            
                            So final algorithm:
                            1. Query $y=0$ intercepts to get set $P$ of size $K$.
                            2. Query $y=1$ intercepts to get set $Q$ of size $K$.
                            3. Match $P$ and $Q$ to find all non-zero slopes $a_i$.
                               For each $a \in [-10000, 10000]$, check if there exists pair $(p, q)$ s.t. $q - p \approx 1/a$.
                               Record $a$ and associated $p$.
                            4. If $K < N$, missing $N-K$ lines with $a=0$. (Should be $\le 1$).
                            5. Calculate contributions of recovered lines to $S(x, 0)$ and $S(0, 0)$.
                               Actually, just use one query point, e.g., $(0, 0)$, and the recovered lines.
                               $S_{recov} = \sum \text{dist}((0,0), \text{line}_i)$.
                               Observed $S_{tot} = S(0, 0)$.
                               Diff $= |b_{zero}|$.
                               Wait, is it $|b_{zero}|$?
                               Dist from $(0,0)$ to $y=b$ is $|b|$. Correct.
                               So $|b_{zero}| = S_{tot} - S_{recov}$.
                               If $S_{tot} < S_{recov}$, error or logic flaw? Should hold.
                            6. Determine sign of $b_{zero}$?
                               Query $S(0, 0)$ again? Same.
                               Need a point to resolve sign.
                               $y=1$?
                               $S(0, 1)$ observed.
                               Calc $S_{recov}$ for $(0, 1)$.
                               Diff $= |1 - b_{zero}|$.
                               Solve $|1-b| = D$. Two solutions for $b$.
                               Check against $|b| = D_1$?
                               $|1-b|^2 = D^2$ and $b^2 = D_1^2$.
                               Solve quadratic. One solution valid.
                               
                            Wait, we need to query $(0, 0)$ and $(0, 1)$ for $S$.
                            We already queried some points during interception finding.
                            The interception finding queries were generic $y=0$ points.
                            $(0,0)$ is one of them (or close).
                            We need to explicitly query $(0, 0)$ and $(0, 1)$?
                            Or just use stored queries?
                            Stored queries used for linearity checks.
                            We can just issue explicit queries for final step.
                            
                            Wait, finding $K$ might be $< N$.
                            If $K < N$, we found $K$ lines with $a \neq 0$.
                            The remaining $N-K$ lines have $a=0$.
                            Wait, "No two lines parallel".
                            So only one $a=0$ line possible.
                            Thus $N-K$ must be 0 or 1.
                            So at most 1 zero-slope line.
                            
                            Complexity:
                            Interception finding: $Q \approx 5000$.
                            Matching: $10000 \times 100$. Fast.
                            Final query: $O(1)$.
                            Total $Q$ well within $10000$.
                            
                            Implementation details:
                            Use `double`.
                            Be careful with query count.
                            
                            One optimization:
                            Don't do full recursion on everything if not needed.
                            But recursion logic is sound.
                            We need to be careful with query limit.
                            Let's bound recursion depth or count.
                            Also use a global counter for queries.
                            
                            One detail:
                            Query format: "? x y".
                            Must print newline.
                            Interactor prints answer on next line.
                            Read string or double?
                            Input: "The interactor returns the sum..."
                            It's a double.
                            Format: "answer".
                            
                            Code Structure:
                            ```cpp
                            #include <iostream>
                            #include <vector>
                            #include <cmath>
                            #include <iomanip>
                            #include <algorithm>
                            #include <set>
                            
                            using namespace std;
                            
                            int N;
                            double Q_used = 0;
                            const double EPS = 1e-9; // Tolerance
                            const double THRESH = 1e-9; // Convexity deviation
                    
                            bool query(double x, double y) {
                                cout << "? " << fixed << setprecision(15) << x << " " << y << endl;
                                // Wait, reading response.
                                // We need to read from cin.
                                // But this is a function.
                                // Just put query logic inline or class.
                                return true; 
                            }
                            
                            int main() {
                                // cin >> N;
                                // ... logic ...
                            }
                            ```
                            Wait, interaction loop.
                            
                            One more thing:
                            We need to identify lines with $a \neq 0$.
                            If $N_{found} == N$, we are good.
                            If $N_{found} < N$, handle $a=0$.
                            
                            Let's double check the $a=0$ case logic.
                            We find $b_{zero}$ from magnitude.
                            We solve $|b| = D_1$, $|1-b| = D_2$.
                            Squared: $b^2 = D_1^2$, $b^2 - 2b + 1 = D_2^2$.
                            Substitute $b^2$: $D_1^2 - 2b + 1 = D_2^2 \implies 2b = D_1^2 + 1 - D_2^2$.
                            $b = (D_1^2 + 1 - D_2^2)/2$.
                            Unique solution.
                            
                            Wait, $a=0$ case:
                            We need to ensure $a=0$ lines are handled correctly.
                            The interception search for $y=0$ on lines with $a=0$ produces no kinks (or infinite, handled by ignoring).
                            So $K$ will be number of $a \neq 0$ lines.
                            
                            Another detail:
                            The search for intercepts needs to cover the full range.
                            Intercept range $[-10^4, 10^4]$.
                            But if $b=10000, a=-10000$, intercept is $1$.
                            If $b=-10000, a=1$, intercept $10000$.
                            Range is definitely $[-10000, 10000]$.
                            Wait, if $a$ is close to 0?
                            Constraint: $a_i$ integers. So min non-zero $|a|=1$.
                            Then $|x_{int}| = |b/a| \le 10000/1 = 10000$.
                            So range $[-10000, 10000]$ is safe.
                            Expand slightly to $[-10005, 10005]$.
                            
                            Wait, "No two lines are parallel".
                            This means all $a_i$ are distinct.
                            So if we find all $a_i$, we are unique.
                            
                            Wait, what about the sign of $a_i$?
                            $a_i$ can be positive or negative.
                            My slope iteration $[-10000, 10000]$ covers both.
                            
                            Final check on matching $P$ and $Q$.
                            We find all pairs $(p, q)$ satisfying $q - p \approx 1/a$.
                            If there are multiple $a$'s fitting, it means ambiguity.
                            But since $a$ distinct, we just find the set of valid $a$'s.
                            Actually, each $p$ must correspond to exactly one $a$.
                            Each $q$ must correspond to exactly one $a$.
                            If we iterate $a \in Candidates$, check if $p \in P$ and $q = p + 1/a \in Q$.
                            We can use `std::vector<double> P` and `std::vector<double> Q`.
                            Sort both vectors.
                            For a fixed $a$, check if difference between $p$ and some $q$ is close to $1/a$.
                            Since $P$ and $Q$ are sorted, we can use pointers/binary search?
                            Or just iterate all $P$ and binary search in $Q$ for target $p + 1/a$.
                            This is efficient.
                            If found, we found a match.
                            Since $N \le 100$, this is fast.
                            We need to mark used elements?
                            If we find match for $p_i$ and $q_j$ with slope $a$, we assign $a$ to $p_i$.
                            We should ensure each $a$ is used only once.
                            
                            Wait, $P$ and $Q$ are unordered sets?
                            The order in vector doesn't matter for values, but for matching we might need to be careful.
                            Actually, if we just collect ALL valid $(a, p)$ tuples.
                            There should be exactly $N$ valid tuples (if $a=0$ is handled).
                            Wait, we only care about $a \neq 0$.
                            So if $a=0$, we don't find any tuple.
                            Then we handle $a=0$ separately.
                            
                            Wait, what if two lines have slopes $a$ and $a'$ such that $1/a = 1/a'$? Impossible for distinct integers.
                            So $1/a$ is unique identifier.
                            
                            Wait, we have a potential issue:
                            We have $P$ and $Q$.
                            We don't know the index mapping.
                            But we can just iterate all $a$.
                            For each $a$, check if there exists $p \in P$ such that $p + 1/a \in Q$.
                            If so, we extract $p$.
                            Wait, what if multiple $p$'s match?
                            If $p_1$ matches $a$ and $p_2$ matches $a$, then we have ambiguity.
                            But this implies two lines have same slope $a$, which violates "No two lines parallel".
                            So each $a$ matches at most one $p$.
                            And each $p$ matches at most one $a$.
                            So we can greedily match.
                            
                            Algorithm refined:
                            1. Find intercepts $P$ (size $K$) on $y=0$.
                            2. Find intercepts $Q$ (size $K$) on $y=1$.
                            3. Initialize map `lines`. Key: `p` (index or value).
                            4. Loop $a$ from $-10000$ to $10000$ (skip 0):
                               Target shift $shift = 1.0/a$.
                               For each $p$ in $P$:
                                  Needed $q = p + shift$.
                                  Check if $q$ exists in $Q$ (within epsilon).
                                  If yes:
                                     Assign line with intercept $p$ and slope $a$.
                                     Remove $p$ from $P$ and $q$ from $Q$ (to prevent reuse).
                            5. If lines found $K < N$, then $N-K$ lines have slope 0.
                               Check if $N-K == 1$.
                               If so, solve for $b$.
                               Else (should be 0), done.
                            6. Output.
                            
                            Wait, we need to be careful with "check if exists in Q".
                            Since we remove items, we need a multiset or boolean flag array if we mapped $P, Q$ to indices.
                            Since $Q$ values are doubles, floating point comparison is tricky.
                            Better to just mark used $a$.
                            If we find match, mark $p$ used and $a$ used.
                            
                            Wait, is it possible to have false matches?
                            $q - p \approx 1/a$.
                            E.g. $p=1.0, q=1.001, a=1000$. $1/a=0.001$.
                            Is it possible for unrelated lines to produce this?
                            With $N=100$, and dense values, yes.
                            But we need to distinguish true matches.
                            However, if a line with slope $a$ exists, it GUARANTEES $q - p = 1/a$.
                            If we find a match for $a$, it is extremely likely to be the line.
                            Could we find a match for wrong $a$?
                            $q - p = \text{something else}$.
                            Unlikely to accidentally equal $1/a'$ for a different $a'$ unless constructed adversarially.
                            Given constraints, it's okay.
                            
                            Wait, $q$ and $p$ are from $P$ and $Q$.
                            If we match $p$ to $q$ for $a$, we consume $p$ and $q$.
                            Is it possible that $p$ matched $a$ (true) and also $p'$ matched $a$ (false)?
                            If $p$ and $p'$ are distinct intercepts, they correspond to distinct lines.
                            If both have slope $a$, impossible.
                            So we are safe.
                            
                            What about matching $p$ to $q$ for WRONG $a$ (false positive)?
                            e.g. $q - p \approx 1/a_{wrong}$.
                            If we accept it, we lose the real line.
                            But if we exhaust all $a$'s, and we find a match for $a_{wrong}$?
                            We need to ensure we don't make false assignments.
                            However, since $N$ is small, maybe we just pick the best match?
                            Actually, if we assume the input is valid, valid matches should be distinct.
                            Maybe sort $P$ and $Q$ and try to match?
                            But sorting doesn't help with $1/a$.
                            However, we know there are exactly $K$ valid $a$'s.
                            If we iterate all $a$, we can check consistency.
                            Let's just use a heuristic:
                            Count how many valid matches we have for each $a$.
                            Keep only those with high confidence?
                            Or just take the first match and remove?
                            Wait, if there is a false match, it consumes a resource.
                            But if we assume valid construction, the match should be "strong".
                            Let's assume standard CP conditions where inputs aren't adversarial on floating point.
                            
                            One small optimization:
                            Instead of iterating $p$ for every $a$, iterate $a$ then find $p$.
                            Or iterate $p$, try to find best $a$?
                            No, iterating $a$ is better because we know the domain of $a$.
                            Domain size $\approx 20000$.
                            For each $a$, we check $K$ $p$'s. Total ops $20000 \times 100$. Very fast.
                            
                            Let's write the code.
                            
                            One detail:
                            The recursion depth.
                            If we hit max depth, we stop.
                            If we find nothing, we just use $P$.
                            
                            Wait, memory limit 256MB. Plenty.
                            
                            One last thing:
                            We need to handle the $a=0$ case properly.
                            If $a=0$ line exists, its $b$ is integer.
                            The magnitude $|b|$ is derived from query diffs.
                            $b = \pm |b|$.
                            We can determine sign by checking $S(0, 0)$ vs $S(0, 1)$.
                            Wait, $S(0, 0) = \sum |b_{i}| / \dots + |b_{zero}|$.
                            $S(0, 1) = \sum \dots + |1 - b_{zero}|$.
                            This gives equation for $|b_{zero}|$.
                            Wait, previously I said it gives sign.
                            Yes.
                            $b_{zero}$ can be $X$ or $-X$.
                            $|X| = D_1$. $|1-X| = D_2$.
                            Solve for $X$.
                            $|X| = D_1 \implies X = \pm D_1$.
                            Check which one satisfies $|1-X| = D_2$.
                            If $D_1 = D_2$, then $|X| = |1-X| \implies X = 0.5$ (but $X$ integer).
                            So usually unique solution?
                            Or symmetric.
                            If $D_1 = 3, D_2 = 4$.
                            $|X| = 3 \implies X = \pm 3$.
                            $|1-X| = |1 - (-3)| = 4$. Matches.
                            $|1-3| = 2 \neq 4$.
                            So sign is resolved.
                            Exception: if both satisfy?
                            $|1-X| = |1+X| \implies X=0$. (Impossible if $D_1 \neq 1$).
                            If $X=0$, $b=0$. $|0|=D_1 \implies D_1=0$.
                            If $D_1 \neq 0$, then unique.
                            Wait, if $X=0.5$? $b$ is integer.
                            So uniqueness guaranteed for integer $b$ (except possibly if $D_1, D_2$ ambiguous? No).
                            
                            So logic holds.

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <map>

using namespace std;

// Global variables to store problem state
int N;
double Q_used = 0;
const int Q_MAX = 10000;
const double EPS = 1e-10;
const double CONVEXITY_THRESHOLD = 1e-8; 

struct Point {
    double x;
};

// Stores found lines: slope and x-intercept at y=0
struct Line {
    int id;
    double a;
    double b_intercept; // -b/a
};

bool query_lines(const vector<Point>& pts) {
    // Query format ? x y
    // We will just query the specific points needed
    // But here we just output? No, we need to interact.
    return true;
}

void solve() {
    cin >> N;
    
    // Arrays to store intercepts
    vector<double> p_x(N_active);
    vector<double> q_x(N_active);
    // We don't know N_active yet, use vector and resize
    
    // Helper lambda to find intercepts on y=Y
    auto find_intercepts = [&](double Y) {
        // Range [-10005, 10005]
        double L = -10005.0;
        double R = 10005.0;
        vector<pair<double, double>> stack;
        stack.push_back({L, R});
        
        vector<double> found_intercepts;
        
        // We limit iterations to avoid TLE/TLE
        // We will perform DFS/BFS on stack
        
        while (!stack.empty() && Q_used < Q_MAX) {
            pair<double, double> curr = stack.back();
            stack.pop_back();
            double l = curr.first;
            double r = curr.second;
            
            if (r - l < 1e-7) { // Leaf check for isolation?
                // If interval is small, check if it contains kink?
                // Or if we reached min width, we just take center or l
                // But we need to verify it has a kink.
                // With convexity check, if linear, ignore.
                // Let's assume if r-l small, we might need to verify.
                // But to save queries, we just take L as approximation?
                // Better: check if it's linear before accepting.
                // Actually, if it's linear, it shouldn't be pushed if we checked linearity properly.
                // We rely on linearity check earlier.
                // If we are here, we must have split it.
                // So it contains at least one kink.
                // Take midpoint or l?
                // Let's refine: we can run ternary search here.
                // But that costs 20+ queries.
                // With 1e-7 precision, ternary search is not worth it if we just take midpoint?
                // No, if we split enough, the "bump" will be sharp.
                // The minimum of the V shape IS the kink.
                // So we should run a mini ternary search.
                
                // Let's do ternary search
                double low = l;
                double high = r;
                for(int i=0; i<30; ++i) {
                    double m1 = low + (high - low)/3;
                    double m2 = high - (high - low)/3;
                    cout << "? " << m1 << " " << Y << endl;
                    cin >> double_s;
                    Q_used++;
                    // ... need actual query handling
                }
                continue; 
            }
            
            // Optimization: If we have processed a segment and determined it has no kink, skip.
            // We track this?
            // Actually, just check linearity.
            
            double mid = (l + r) / 2.0;
            // Query three points
            // Need to query l, mid, r.
            // But we need to read results.
            // We need a wrapper.
        }
        return found_intercepts;
    };
    
    // ...
}
```

Wait, I need to implement the query handler carefully.
Using `cin >> val`.
Also manage `Q_used` carefully.
And implement the "FindIntercepts" function correctly.
Since the logic requires nested interactions, putting it in `main` or helper function is good.
The stack-based search avoids recursion depth issues and is iterative.

Let's refine the search.
For a range $[l, r]$:
If $r - l < \epsilon$, we assume we have found it. We can just append $l$ to list?
Wait, if the interval contains a kink, and we stop, we don't know where.
But if we stopped because $r-l$ is small, the kink is somewhere in there.
However, we can use the fact that $S(x)$ is minimized at the kink (locally).
So we should do ternary search if we detect a non-linear region but it's small?
Actually, if we detect non-linearity, we know there is a kink.
If we stop splitting when $r-l < \delta$, we haven't localized it to a point.
But wait, if we find that the function is NOT linear, then there is a kink.
If we check linearity at mid.
If linear, we stop.
If not linear, we split.
Eventually we get intervals with non-zero width.
If we restrict width to e.g. $10^{-6}$, we might be stuck.
Is it necessary to find exact location?
We need $b_i$ accurate.
$b_i = -a_i x_{int}$.
If $a_i \approx 10000$, error in $b_i$ requires error in $x$ of $10^{-10}$.
Wait, $a_i$ is integer, $b_i$ integer.
So exact $x$ matters to distinguish lines.
But if $x$ is close to $b/a$, maybe we can just round to nearest rational?
No, we don't know $b, a$.
We rely on $x$ difference matching $1/a$.
This implies we need $x$ accurate enough to measure $1/a$.
If $x$ is off by $\delta$, measured $1/a$ is off by $\delta$.
If $1/a$ is small (large $a$), we need small $\delta$.
If $a=10000$, $1/a=10^{-4}$.
We need resolution $10^{-5}$ or better.
Interval width $10^{-5}$ is achievable with 30 divisions.
So stopping at small width is fine, provided we picked the correct $x$.
Wait, if interval contains 1 kink, the function is V-shaped.
The vertex is the kink.
So running Ternary Search on the interval finds the kink.
This is better than just taking $l$ or midpoint.
Ternary Search takes $\approx 20$ queries per kink.
With $N=100$, total $\approx 2000$.
Plus checking linearity for parents.
This is acceptable.

Corrected Plan:
1. Initial stack `[-10005, 10005]`.
2. Loop:
   Pop $[l, r]$.
   If $r - l < \text{TOLERANCE\_MIN}$ (e.g. $1e-5$):
      Perform Ternary Search in $[l, r]$ to find min.
      Add result to `P`.
      Stop processing this node.
   Else:
      Query $S(l), S(mid), S(r)$.
      Check if linear.
      Condition: $|S(mid) - (S(l)+S(r))/2| < \text{EPS\_LINEAR}$.
      If linear: Continue (ignore).
      Else: Push $[l, mid]$ and $[mid, r]$ to stack.

Note: We need to be careful with "Linear Check".
If the function is convex, $S(mid) \le (S(l)+S(r))/2$.
If strictly less, there is a kink (or curvature).
Since it's sum of lines, strictly less means kink exists.
Is it possible for curvature without kink? No.
So strict inequality implies kink.
Is it possible for linear function to fail due to precision?
Yes. So we use a threshold.
Threshold should be higher than noise.
Given $S$ up to $10^6$, noise $\approx 10^{-10}$ relative $\implies 10^{-4}$ absolute.
Wait, double precision is $10^{-15}$.
With sums of 100 terms, error grows by $\sqrt{100} \approx 10$ or $100$.
So error $\approx 10^{-13}$.
Absolute error for value $10^6$ is $10^{-7}$.
Wait, intermediate calculations.
Distance calculation involves sqrt.
Error $\approx 10^{-15} \times \text{value}$.
Summing 100 terms: worst case $100 \times 10^{-15} \times 10^6 \approx 10^{-7}$.
So linear check threshold should be around $10^{-6}$ to be safe against noise.
Or check "curvature" more strictly.
Maybe $10^{-7}$?
Actually, the kinks are created by changing slope by amounts like $|a|/c \approx 1$.
So jump is large.
The deviation $S(mid) - E$ scales with distance from kink squared times slope change.
If we are very close to kink, deviation is small.
If we are not close, deviation is large.
Wait, we just need to detect *presence* of kink.
If we are at $l, r$ enclosing a kink near center $m$, deviation is large.
If kink is at $l$ or $r$, deviation is small?
If kink at $l$, $S$ is linear on $(l, r)$. Then deviation is 0 (ideal).
If kink at $l$, we should not split.
But we initialized range.
If kink is exactly at boundary, we can't detect it inside.
But we assume kinks are strictly inside range?
Intercepts $-b/a$ can be exactly boundaries.
But boundaries are integers $\approx -10000$.
Intercepts are rational.
It is unlikely to be exactly boundary of $[-10005, 10005]$.
Wait, if $b/a = 10005$, then yes.
To avoid this, make range slightly larger or shift.
Let's shift to $[-10010, 10010]$.
Or just query a bit outside?
Wait, we can't query outside territory? Territory is $[-10^{12}, 10^{12}]$.
So we can query anywhere.
Just extend range to $[-20000, 20000]$.

Revised Search Logic:
Stack of intervals.
Process:
Pop $[l, r]$.
Check linear?
Need to query $S(l), S(r), S(mid)$.
If linear, stop.
Else, recurse.
Stop recursion if $r - l < \epsilon$.
Then run TS to find min.

Query Count Control:
We have 10000.
Max splits per kink $\approx \log_2(20000/\epsilon)$.
If $\epsilon = 10^{-5}$, log factor $\approx 30$.
Queries per kink $\approx 1 \text{ (initial check)} + 2 \times \text{depth}$?
No, each node in binary tree costs 3 queries.
Leaves (kinks) cost 2 (TS) queries.
Total nodes $\approx 2 \times N$.
Total queries $\approx 3 \times (2N \times 30)$?
Too much.
Actually, we don't visit all nodes.
We prune linear nodes.
If a node has NO kinks, we pay 3 queries and discard.
If it HAS kinks, we split and visit children.
How many nodes without kinks?
At most 1 root.
Wait, if we split $[L, R]$ with kinks in $[L, M]$ and no kinks in $[M, R]$.
We check $[L, R]$ (cost 3).
Split. Process $[L, M]$ (has kink).
Process $[M, R]$ (no kink). Cost 3. Discard.
So for each kink, we might traverse siblings.
Total queries $\approx O(N \times \text{depth})$.
$100 \times 30 = 3000$.
Safe.

Final check:
What if $k$-th kink is in $[L, R]$, but $[L, R]$ looks linear?
Noise makes it look linear.
Then we discard a kink.
This is bad.
We must ensure sensitivity.
Since slope jump is $\approx 1$ or more.
Min interval width for deviation to be $> 10^{-6}$.
Triangle area $\approx \text{slope\_jump} \times \text{height} \times \text{width} / 2$.
Height $\approx \text{width} / 2$ (if mid is peak).
So deviation $\approx \text{jump} \times \text{width}^2$.
If jump $\approx 1$, deviation $\approx \text{width}^2$.
We need deviation $> 10^{-7}$ (noise limit).
$\text{width}^2 > 10^{-7} \implies \text{width} > 10^{-3.5} \approx 3 \cdot 10^{-4}$.
So if interval is $> 10^{-4}$, we should detect kink.
This is much larger than precision requirement.
Wait, if we have narrow intervals $10^{-4}$, we detect kink.
So we won't split infinitely small intervals unless kink is truly there?
Wait, if interval is $10^{-4}$, deviation is $10^{-8}$ (smaller than noise $10^{-7}$).
Wait, my noise estimation $10^{-7}$ might be optimistic/pessimistic.
If noise is high, we need to widen the threshold.
But we also need to find kink location.
If we stop splitting at width $10^{-3}$, and run TS.
TS takes 2 queries to refine to $1/3$ width.
If we run 20 iterations of TS, width becomes $10^{-3} \cdot 10^{-6} \approx 10^{-9}$.
Cost 40 queries.
This balances well.

One detail: Ternary Search needs query points.
We need to query $m1, m2$.
We need to be careful not to query duplicate points (though it's cheap).
TS loop runs for fixed steps.

Okay, ready to code.

```cpp
/*
Problem: Lines
Solution:
1. Detect kinks (x-intercepts on y=0) using convexity.
2. Detect kinks (x-intercepts on y=1) similarly.
3. Match kinks to find slopes a.
4. Infer b.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

double N_val; // Input N
int Q_limit = 10000;
int queries = 0;
double Q_max = 10000.0;

double query_point(double x, double y) {
    if (queries >= Q_limit) {
        // Fatal, but for safety
        exit(0);
    }
    cout << "? " << fixed << setprecision(10) << x << " " << y << endl;
    queries++;
    double ans;
    if (!(cin >> ans)) {
        exit(0);
    }
    return ans;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    // We need to store intercepts for y=0
    vector<double> p_x;
    // We need to store intercepts for y=1
    vector<double> q_x;

    // Range for search
    // Intercepts can be up to 10000/1 = 10000. 
    // Let's use [-10000, 10000]. 
    // Add buffer.
    double L_MIN = -10005.0;
    double R_MAX = 10005.0;
    
    // Function to find intercepts
    auto get_intercepts = [&](double Y) {
        vector<pair<double, double>> stack;
        stack.push_back({L_MIN, R_MAX});
        
        vector<double> results;

        while (!stack.empty()) {
            if (queries >= Q_limit) break;
            auto cur = stack.back();
            stack.pop_back();
            double l = cur.first;
            double r = cur.second;
            
            if (r - l < 1e-4) { 
                // Interval is small. Assume it contains a kink if we reached here (or just search).
                // Actually, if we stop splitting, we need to check if it's a kink or just empty space.
                // But we only push if non-linear.
                // However, if we start with large range, and no kink exists, we might not split.
                // Wait, if linear, we don't push children.
                // So if we are at a leaf (due to size), it means we stopped because it was SMALL.
                // Does it contain a kink?
                // If it didn't contain a kink, we should have pruned it when it was larger?
                // Not necessarily, noise might hide it.
                // But if we found it non-linear at size > 1e-4, we split.
                // If it became non-linear again at small size, we would have split.
                // Wait, if r-l < 1e-4, deviation < (1e-4)^2 = 1e-8.
                // If noise ~ 1e-7, we can't detect.
                // So if r-l is small, we can't reliably check linearity.
                // We can assume it IS a kink if we reached here?
                // No, empty space might shrink by luck? No.
                // But we prune linear regions.
                // So if we are at a region of size < 1e-4, it MUST be a region where linearity was ambiguous.
                // However, we want to collect points.
                // So we can run Ternary Search to find local min.
                // Wait, if it's empty space, TS will return min of convex function.
                // For empty space (linear), min is at boundary?
                // Actually, sum of dists is convex. Min is somewhere.
                // We can just return the midpoint?
                // Let's stick to: if r-l < 1e-4, run TS to find min.
                // But wait, if it's not a kink (empty space), min might be far?
                // No, the function is globally convex.
                // If we have isolated linear segment, the min is at the global minimum of that segment.
                // But we want kinks.
                // Actually, the safest bet:
                // If r - l < 1e-4, perform TS to find min. 
                // If deviation is high, it's a kink. If low, maybe noise?
                // Let's just assume it IS a kink location. 
                // We need to collect ALL kinks.
                // So we should check linearity BEFORE pushing children.
                // If linear, discard.
                // If non-linear, push children.
                // If we hit limit, we take min?
                // But we might be in linear region.
                // So: if linear, discard.
                // Else if small, take l (or mid).
                // Let's refine:
                
                // To check linearity, we need 3 points.
                // If we are at depth limit, we can't check.
                // But with r-l < 1e-4, we can't trust
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=566)

````
I need to think about how to solve this problem. We have N lines of the form y = a_i * x + b_i, and we can query points to get the sum of distances from those points to all N lines.

The key insight I should look for is in the properties of distance sums and how they relate to the line parameters. For a single point (x, y), the perpendicular distance to a line y = a_i * x + b_i is |a_i*x - y + b_i| / sqrt(a_i^2 + 1). So if we know a few carefully chosen query points, the sum of these absolute values gives us equations involving a_i and b_i that we can analyze.

Since Q_max is much larger than Q_min and we only have about 10-40 queries to find 100 lines, I should expect each query reveals information about many lines at once, and the structure should let me recover most lines with very few queries rather than trying to find each one individually.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;
void set_line(int i) { cout << "? " << i/50*2e12+100<<i%50*400<<endl; fflush(stdout); }
int main() { int n; cin >> n; vector<int>a(n),b(n);for(int j=0;j<n;j++)set_line(j);for(int k=0;k<n;k++){if((k%17)==3){long long A=a[k],B=b[k];double X=k*(k-1)/2,BETA=(double)(k-1)*(k)/80,kA=A/BETA;if(isnan(kA))continue;vector<pair<int,int>>L;for(long long i=-k;i<=k;i+=2)for(long long j=i+2;j<=k;j++)j!=k?{if((j-i)%8==4&&((A*j+B*i-A*i-B*j)/8)%2==0)L.push_back({A*A*(i-j)*j,i*B+j*(A-A)}):,A=j,j=i}:;long long g=__gcd(abs(A),abs(B)),A/=g,B/=g;if(abs(A)<5&&abs(B)>5)swap(A,B);for(auto p:L)a[k]=abs(A/p.first)+2*p.second*b[k]/p.second,b[k]=-(b[k]%(B+p.second))-p.second;}}}for(int j=0;j<n;j++){cout<<"!";for(int i=0;i<n;i++)cout<<" "<<a[i]<<" "<<b[i]<<endl;}return 0;
}
```
````
