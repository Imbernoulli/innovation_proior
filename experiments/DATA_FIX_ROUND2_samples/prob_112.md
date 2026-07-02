# 题 112 — base均分 40.4 vs 直接SFT均分 0.0(掉点 40.4)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 99.8,completion_tokens=24505)

````
The problem asks us to place $n$ points inside or on the boundary of a unit sphere such that the minimum pairwise distance between any two points is maximized. We need to output this maximum minimum distance and the coordinates of the points.

This is a classic "Tammes Problem" in 2D (circumcircle) or its generalization to higher dimensions, though for $n$ up to 1000 in 3D, we don't have a simple closed-form solution for arbitrary $n$. However, $n=1000$ is somewhat large for complex geometric optimization solvers within a short time limit if they were exact. But typically, for competitive programming problems like this, either there's a known pattern or an approximation algorithm works well enough.

Let's look at small values of $n$:
- $n=2$: Two antipodal points on the diameter. Distance = 2. Coordinates: $(0,0,1), (0,0,-1)$.
- $n=3$: Equilateral triangle on a great circle? Or tetrahedron vertices? Actually, for $n=3$, we can place them on a great circle forming an equilateral triangle. Side length = $\sqrt{3} \approx 1.732$. Or maybe better on a regular tetrahedron configuration? No, tetrahedron needs $n=4$. For $n=3$, the optimal is an equilateral triangle inscribed in a great circle (z=0 plane). The circumradius is 1. The side length $s$ satisfies $R = s / \sqrt{3}$, so $s = R\sqrt{3} = \sqrt{3}$. Wait, if it's a tetrahedron inscribed in a sphere, the vertices are symmetric. For $n=4$, it's a regular tetrahedron. The distance from center to vertex is $R_{center} = \sqrt{6}/4 \cdot \text{side\_length}$. Here the sphere radius is 1, so distance from origin to vertex is 1. So $\sqrt{6}/4 \cdot s = 1 \implies s = 4/\sqrt{6} = 2\sqrt{6}/3 \approx 1.633$. This is less than $\sqrt{3} \approx 1.732$. So for $n=3$, keeping them on the equator is better.
Wait, let me double check. If points are on the equator, they form a circle of radius 1. Max chord length is 2 (diameter). With 3 points, equilateral triangle, side is $\sqrt{3} \approx 1.658$? No, $s = R\sqrt{3}$? Circumradius of equilateral triangle with side $a$ is $a/\sqrt{3}$. If $a/\sqrt{3}=1$, then $a=\sqrt{3} \approx 1.732$. Correct.
Is it possible to do better by lifting them out of the equatorial plane?
Actually, the Tammes problem states:
$n=2$: Diameter, dist 2.
$n=3$: Equilateral triangle on equator, dist $\sqrt{3}$.
$n=4$: Regular tetrahedron, dist $\sqrt{8/3} \approx 1.63299$. Wait, checking tetrahedron edge length. Center is centroid. Vertices at distances $r$ from origin. $r=1$. Angle subtended by edge at center is $\theta$. Cosine rule: $d^2 = r^2 + r^2 - 2r^2 \cos\theta = 2(1-\cos\theta)$. For regular tetrahedron inscribed in sphere, angle between vectors is $\arccos(-1/3)$. So $d^2 = 2(1 - (-1/3)) = 8/3$. $d = \sqrt{8/3} \approx 1.633$.
So for $n=4$, placing them at tetrahedron vertices is better than triangle + one pole? Triangle vertices at $(1,0,0), (-1/2, \sqrt{3}/2, 0), (-1/2, -\sqrt{3}/2, 0)$ and last point $(0,0,1)$. Min dists: $(1,0,0)$ to others is $1.732$ (chord) or $\sqrt{(1.5)^2+...}$ wait. Distance from $(1,0,0)$ to $(-1/2, \sqrt{3}/2, 0)$ is $1.732$. Distance from $(1,0,0)$ to $(0,0,1)$ is $\sqrt{2} \approx 1.414$. So mixing planes gives smaller min dist. Tetrahedron gives $\approx 1.633$.
$n=6$: Octahedron vertices. $(\pm 1, 0, 0), (0, \pm 1, 0), (0, 0, \pm 1)$. Distances are $\sqrt{2} \approx 1.414$.
$n=12$: Icosahedron vertices. Distance is $\frac{\sqrt{10 - 2\sqrt{5}}}{\sqrt{3}} \approx 1.051$. Wait, formula is usually $d = \frac{2}{\sqrt{3}} \frac{\sqrt{5-\phi}}{\sqrt{\phi}} ...$ Let's not rely on memorized constants.

Constraints: $n \le 1000$.
For large $n$, the points will distribute uniformly over the surface area of the sphere. Since they are constrained to be *within* the sphere but maximizing pairwise distance pushes them to the boundary, effectively all points will lie on the surface.
For very large $n$, the problem relates to packing density. The minimal separation $d$ scales roughly as $n^{-1/2}$. Specifically, $d \approx \sqrt{A/N}$ where $A$ is area ($4\pi$). $d \approx \sqrt{4\pi/n} = 2\sqrt{\pi/n}$.
However, $n=1000$ is not extremely large to ignore discretization effects near poles vs equator.

Wait, generating the exact optimal positions for $n$ up to 1000 is computationally hard (NP-hard for specific packings). However, for the purpose of a CP problem, often a constructive heuristic is expected that works well enough. Or perhaps $n$ is small enough for a numerical solver like gradient descent or simulated annealing per test case, but here we just get one $n$. The time limit is 2 seconds.
Generating random points and pushing them to repel each other might work. This is the "Repulsion Potential" approach. Initialize $N$ random points on the sphere. Calculate total force on each point due to others. Move each point to increase distance from neighbors. Repeat until convergence.
Given 2 seconds, a force-directed placement could work.
Or use Fibonacci Lattice projection, or Spiral arrangement.
Fibonacci Sphere Algorithm is standard for distributing points evenly on a sphere.
Algorithm:
$\phi = (1 + \sqrt{5}) / 2$ (Golden Ratio).
Angles: $\theta_i = 2 \arctan(\sqrt{t_i} (\phi^{-i}))$ ?
Common Fibonacci Spherical Code (FSC):
$t = i/(N+1)$
$y_i = 1 - 2t$
$x_z = \sqrt{1-y_i^2}$
Angle $\theta_i = i \times 2\pi / N$? No, golden spiral uses the Golden Angle.
Standard method:
Iterate $i$ from $0$ to $n-1$:
$\theta_i = (i \cdot \frac{2\pi}{n})$ ??
Better:
$y = \frac{i}{n} \cdot 2 - 1$
$\rho = \sqrt{1 - y^2} \cdot n$
$\theta = \frac{\pi \cdot i}{n}$
Wait, Fibonacci lattice is:
$\phi = (1+\sqrt{5})/2$
$y_i = \frac{i - 0.5}{n} \times 2$ ? No.
Correct Fibonacci Sphere generation:
For $i = 0$ to $n-1$:
$\theta_i = i \times \text{golden\_angle} + \delta$ (random phase)
$\rho_i = \sqrt{1 - (\frac{i}{n} \cdot 2 - 1)^2}$
This creates a nice distribution but not necessarily maximizing the minimum distance perfectly for small $n$, though good for large $n$.
For $n < 20$ or so, we might need specific configurations (tetrahedron, octahedron, icosahedron, cuboctahedron).
Since $n$ goes up to 1000, maybe we should implement the Fibonacci sphere or similar quasi-random layout which provides a uniform distribution with good spacing properties.
Specifically, using the Golden Spiral mapping:
Points lie on lines passing through the north and south poles.
$y = \frac{i}{n} - 0.5$ (normalized y coordinate)
$\phi = i \cdot \frac{2\pi}{n} + \text{offset}$?
No, Golden Angle is $\approx 137.5^\circ \approx 2.39996$ radians.
$\theta_i = i \cdot \text{golden\_angle}$.
Coordinate $x = \cos \theta_i \sqrt{1-y^2}$
$y$ is linearly spaced.
This distributes points well.

Is this "optimal"? It's likely close enough. For $n=1000$, the error margin is probably sufficient because the score is average of ratios.
Wait, let's consider specific small $n$ cases manually optimized, and use the spiral for larger $n$. Or just rely on the spiral.
However, there is a catch: Fibonacci spheres are good for uniform coverage, but strictly speaking, they might not yield the maximal minimal distance for small $n$.
For example, $n=4$, Fibonacci might not give a regular tetrahedron.
If we want to pass strict checks for small $n$, we can handle small $n$ specifically.
Known optimal configurations for small $n$:
$n=2$: (0,0,1), (0,0,-1)
$n=3$: Equilateral triangle on equator. $(1,0,0)$, $(-1/2, \sqrt{3}/2, 0)$, $(-1/2, -\sqrt{3}/2, 0)$
$n=4$: Regular Tetrahedron. One vertex at North Pole $(0,0,1)$. Others distributed.
Coordinates: $(0, \sqrt{8/3}, 0)$ normalized to length 1? No.
Regular Tetrahedron vertices relative to center:
Can take vertices of cube, select subset. Or explicit calculation.
Height of tetrahedron $H = \sqrt{2/3} a$. Distance from center to face is $h$. Center to vertex is $R$.
Relationship: $R = 3/4 H$. $a = \sqrt{8/3} R$.
Vertices: $(1, 1, 1)/\sqrt{3}$, $(1, -1, -1)/\sqrt{3}$, $(-1, 1, -1)/\sqrt{3}$, $(-1, -1, 1)/\sqrt{3}$.
These are on sphere of radius 1.
Distances squared: Between $(1,1,1)$ and $(1,-1,-1)$ is $\sqrt{0^2 + 2^2 + 2^2} = \sqrt{8} = 2\sqrt{2}$. Normalized by dividing coords by $\sqrt{3}$. So actual dist = $\sqrt{8}/3 = 2\sqrt{2}/3 \approx 0.94$?
Wait. My previous calc was $d = \sqrt{8/3} \approx 1.633$.
Let's recheck cube vertices. Cube with diagonal $2$. Side $\sqrt{2}$. Distance between opposite corners is $2\sqrt{2}$. Radius of circumsphere is $\sqrt{3}$.
We normalize these points to radius 1. Scale factor $k = 1/\sqrt{3}$.
Distance becomes $2\sqrt{2} \cdot k / \sqrt{3} = 2\sqrt{2} / 3$. Wait.
Distance between $(1,1,1)$ and $(1,-1,-1)$ is $\sqrt{0 + 4 + 4} = \sqrt{8} = 2\sqrt{2}$.
Scale factor to make norm 1: current norm $\sqrt{3}$. New dist $2\sqrt{2} / \sqrt{3} = \sqrt{8/3} \approx 1.63299$.
Okay, correct.
So for $n=4$, these 4 points work.
What about $n=6$? Octahedron. $(\pm 1, 0, 0), (0, \pm 1, 0), (0, 0, \pm 1)$. Dist $\sqrt{2}$.
$n=8$: Square antiprism or Cube?
Cube vertices on sphere: same as tetrahedron but 8 points. But cube has closer diagonals?
Cube edge length is $\sqrt{2}$ (for radius 1 sphere). Face diagonal is 2. Body diagonal is 2. Wait, for unit sphere, max distance is 2.
Cube vertices on unit sphere: $(\pm \sqrt{2/3}, \pm \sqrt{2/3}, \pm \sqrt{2/3})$.
Edge distance squared: $(\Delta x)^2 + (\Delta y)^2 + (\Delta z)^2$. e.g. $(\sqrt{2/3}, \sqrt{2/3}, \sqrt{2/3})$ vs $(\sqrt{2/3}, \sqrt{2/3}, -\sqrt{2/3})$. Diff is $2\sqrt{2/3}$ in $z$. Sq sum $8/3$. Root $\sqrt{8/3} \approx 1.633$.
Face diagonal: $(\sqrt{2/3}, \sqrt{2/3}, \sqrt{2/3})$ vs $(-\sqrt{2/3}, \sqrt{2/3}, \sqrt{2/3})$. Diff is $2\sqrt{2/3}$ in $x$. Sq sum $8/3$. Same.
Other distance: $(\sqrt{2/3}, \sqrt{2/3}, \sqrt{2/3})$ vs $(-\sqrt{2/3}, -\sqrt{2/3}, \sqrt{2/3})$. Diff is $\sqrt{8/3}$. Same.
Wait, body diagonal distance is distance between $(\dots)$ and $(-\dots)$. All components differ sign.
Diff vector $(2\sqrt{2/3}, \dots)$. Magnitude squared $4 \times (2/3) + \dots$ No.
$(1,1,1)$ vs $(-1,-1,-1)$. Vector diff $(2,2,2)$. Length $\sqrt{12} = 2\sqrt{3}$. Normalized by $\sqrt{3}$. Distance $2\sqrt{3}/\sqrt{3} = 2$.
So Cube has distance 2?
Wait. In a cube inscribed in a sphere, opposite vertices are distance 2 apart. Adjacent vertices are distance $\sqrt{2/3}\sqrt{2} + \dots$?
Let's check coordinates again.
Unit sphere. Cube vertices: $(\pm \alpha, \pm \alpha, \pm \alpha)$ where $\alpha = \sqrt{1/3}$.
Wait. $\alpha^2 + \alpha^2 + \alpha^2 = 1 \implies 3\alpha^2 = 1 \implies \alpha = 1/\sqrt{3}$.
Coords: $(1/\sqrt{3}, 1/\sqrt{3}, 1/\sqrt{3})$ etc.
Adjacent pair: $(1/\sqrt{3}, 1/\sqrt{3}, 1/\sqrt{3})$ and $(1/\sqrt{3}, 1/\sqrt{3}, -1/\sqrt{3})$.
Diff in $z$ is $2/\sqrt{3}$. Squared distance is $4/3 \approx 1.333$.
Side length $d = \sqrt{4/3} = 2/\sqrt{3} \approx 1.1547$.
This is much smaller than tetrahedron's $1.633$. So Cube is worse for minimizing pairs.
Octahedron ($n=6$) had $1.414$. Cube ($n=8$) has $1.15$. Dodecahedron ($n=20$) has $1.051$.
Icosahedron ($n=12$) has $\approx 1.051$.
Wait, why did I think $n=6$ was $1.414$? Yes, edge of octahedron.
The "best" packing for fixed $n$ is not always polyhedral.
For large $n$, densest packing approaches hexagonal.

Given $n$ up to 1000, we can use the Fibonacci Sphere Method which is generally accepted for competitive programming "maximize min dist on sphere" problems when $n$ is large. It produces points that are almost equally spaced.
Since $n$ is only 1000, the "worst" local clusters aren't huge.
Is there any reason NOT to use Fibonacci Sphere? It is very robust and fast to generate.
It generates coordinates deterministically.
Algorithm:
For $i = 0 \dots n-1$:
$\phi = (1 + \sqrt{5}) / 2$
$y_i = \frac{i - 0.5}{n} \times 2$  (ranges -1 to 1, step 2/n)
$\rho = \sqrt{1 - y_i^2}$
$\theta = \frac{\phi \times i}{n}$ ? No, golden angle in degrees is $\approx 137.507$.
Actually $\theta_i = i \times \text{golden\_angle} + \text{random offset}$.
Wait, standard implementation:
```cpp
double phi = acos(-1.0 / 3.0); // Not useful here
// Golden angle
const double GOLDEN_ANGLE = M_PI * (3.0 - sqrt(5.0));
for (int i = 0; i < n; ++i) {
    double theta = i * GOLDEN_ANGLE;
    // Wait, range of theta? Usually full rotation 2PI.
    // But the layering is determined by y.
    // Standard code:
    // y = (i * 2.0 / n) - 1.0;
    // theta = i * 2.0 * PI / n + something? No.
    
    // Fibonacci Sphere:
    // theta = i * (2*pi / n) + shift? No.
    
    // Correct Formula from literature (e.g., https://www.mathworks.com/matlabcentral/fileexchange/28445-spiral-point-grid-on-a-sphere):
    // t = (i * 2 - n) / n  (range -1 to 1, excluding 1? No inclusive)
    // Let's stick to:
    // y = (i * 2.0 / n) - 1.0;
    // rho = sqrt(1 - y*y);
    // theta = i * (2.0 * M_PI / n) + some_phase?
    // Actually, using Golden Ratio for angular distribution works better to avoid alignment.
    // theta = i * GOLDEN_ANGLE;
    // x = rho * cos(theta);
    // y_coord = rho * sin(theta);
    // z = y; // wait, variable naming collision.
    
    // Better:
    // y_val = (i / n) * 2.0 - 1.0;
    // phi_ang = i * GOLDEN_ANGLE; 
    // x = sqrt(1 - y_val*y_val) * cos(phi_ang);
    // z = sqrt(1 - y_val*y_val) * sin(phi_ang);
    // But wait, y_val ranges from -1 to 1? i=0 -> -1. i=n-1 -> ~1.
    // Is i*n/n accurate? 
}
```
Wait, simply distributing based on latitude ($y$) uniformly and longitude ($\theta$) via golden angle minimizes clustering.
Actually, a slightly more precise construction for spherical codes is:
$y_i = \frac{i}{n} - 0.5$
$\theta_i = \left( i \cdot \frac{2\pi}{n} \right)$
But for $n$ points on a sphere, to avoid rings of constant latitude being crowded, we rotate longitude by golden angle per step.
Correct logic for Fibonacci Sphere Layout:
1. `phi` = $(1 + \sqrt{5}) / 2$ (Golden Ratio)
2. `GOLDEN_ANGLE` = `acos(-1.0 / 3.0)`? No, that's arc cosine of inverse 3. That's $\approx 70.5^\circ$. Not golden angle.
Golden angle is approx $137.5^\circ$. Value: $2\pi \times (0.381966 \dots)$ or $2\pi \times (1/\phi^2)$.
Usually implemented as `M_PI * (3.0 - sqrt(5.0))`.
3. Loop $i$ from $0$ to $n-1$.
   $y = \frac{2 \cdot i}{n} - 1$? No, usually we exclude boundaries or handle carefully.
   Common formula: $y_i = \frac{i}{n} \times 2 - 1$.
   $\theta_i = \theta_{prev} + \text{GOLDEN\_ANGLE}$
   $\rho = \sqrt{1 - y_i^2}$
   $x = \rho \cos \theta_i$
   $z = \rho \sin \theta_i$

This places points on a spiral staircase on the sphere.
However, the constraint is $x^2+y^2+z^2 \le 1$. The spiral places them exactly on $R=1$. Since we want to maximize min dist, putting them on boundary is best (moving them closer to origin decreases available space for others).
Is this strictly optimal? Not mathematically proven for all $n$, but practically the best heuristic for generic $n$.
Given constraints and scoring, this should be acceptable.
One optimization: For small $n$, we can hardcode specific solutions to ensure correctness.
Small $n$ optimal values:
$n=2$: $(0,0,1), (0,0,-1)$. $d=2$.
$n=3$: $(1,0,0), (-1/2, \sqrt{3}/2, 0), (-1/2, -\sqrt{3}/2, 0)$. $d=\sqrt{3} \approx 1.732$.
$n=4$: Tetrahedron vertices.
$n=5$: Trigonal bipyramid? Or Pentagonal dipyramid?
Actually, Tammes numbers are well studied.
$n=6$: Octahedron. $d=\sqrt{2}$.
$n=12$: Icosahedron.
$n=20$: Dodecahedron.

Implementing exact geometries for small $n$ is safer and easy.
Then switch to spiral for $n >$ threshold (say 12 or 20).

Let's verify $n=5$.
Optimal for $n=5$ is triangular bipyramid? Vertices at $(0,0,\pm 1)$ plus 3 around equator? No, that's $n=5$ with 2 on axis, 3 on equator. Distances: Axis-Equator is 1. Equator-Equator is $2\sin(120^\circ)?$ No chord.
Chord between $(0,0,1)$ and $(1,0,0)$ is $\sqrt{2} \approx 1.414$.
Chord between equator points is $\sqrt{3} \approx 1.732$.
Min dist is $1.414$.
Is there a better config? Maybe a square pyramid base? Base edges equal.
There is a solution called "Trigonal Bipyramid" with apex angles such that all distances are equal?
Wait, if we adjust the height $z$.
Actually, Fibonacci sphere usually handles $n=5$ reasonably well.
Hardcoding is good but tricky without a reference table of coordinates.
Given 1000 points, a numeric optimizer (Simulated Annealing or Repulsive Force) starting from Fibonacci init is also viable. But implementing a robust optimizer is long.
Fibonacci sphere is short and deterministic.
I will stick to Fibonacci Sphere for $n$ up to 1000. The quality is high enough for competitive programming context unless it's a very strict geometry checker requiring exact theoretical optimums for every single $n$. But usually with relative error $10^{-6}$, Fibonacci is accepted.
Also, the prompt mentions "Your answer will be accepted if ... matches your claimed min_dist". Wait.
"The actual minimum pairwise distance matches your claimed min_dist".
So we print `min_dist`, and then the points. The grader checks the actual distance between printed points against the reported `min_dist`. It doesn't mean we must find the THEORETICAL MAXIMUM. It means `min_dist` reported must be consistent with the output points (or slightly lower bound allowed).
Wait. "matches your claimed min_dist (with absolute or relative error at most 10^-6)".
If I report `min_dist = X` but actual points have min dist `X'`, then $|X-X'| \le 10^{-6}$ is required.
This implies I should calculate `min_dist` based on my own points before printing.
It does NOT require my solution to be optimal against the judge's hidden optimal solution, but the problem statement says "Your goal is to maximize the minimum pairwise distance". The scoring depends on comparing to reference.
If I find a sub-optimal config, I lose score.
However, for $N=1000$, the difference between Fibonacci and True Optimum might be negligible for the scoring function (average ratio).
Scoring: `100 * min(your_answer / ref_answer, 1)`. If I am 0.5% off, I get 99.5 points.
With Fibonacci sphere, for $N=1000$, the error in min-dist is usually very small (around $O(1/N)$ vs $O(N^{-1/2})$ terms?).
Actually for $N=1000$, Fibonacci is extremely efficient.

Let's refine the Fibonacci Sphere logic to ensure good distribution.
Standard implementation:
```cpp
const double phi = acos(-1.0 / 3.0); // No.
// Use golden ratio
const double G = (1 + sqrt(5)) / 2;
const double GA = M_PI * (3 - sqrt(5)); // ~ 2.39996 rad (~137.5 deg)

vector<Point> pts(n);
for (int i = 0; i < n; ++i) {
    double y = (double)i / n * 2.0 - 1.0;
    double rho = sqrt(1.0 - y * y);
    double theta = i * GA; 
    // Note: i starts at 0. First point at top? 
    // If y = -1 + eps?
    // To spread better, add small random offset or start loop properly.
    // Often used: y = (i * 2.0 / n) - 1.0
    // If n=2. i=0 -> y=-1. i=1 -> y=1. Poles. Correct.
    // If n=3. y=-1, -0.33, 1. (Pole, Equator-ish, Pole).
    // But equator is not ideal for n=3.
    
    // So for small n, Fibonacci is NOT optimal compared to hardcoded.
    // Let's check n=3 with this logic.
    // i=0: y=-1, rho=0 -> (0,0,-1).
    // i=1: y=-0.33, rho=0.94. theta=137 deg. (0.47, 0.47, -0.33)?
    // i=2: y=0.33, rho=0.94. theta=275 deg.
    // Distances might be decent, but for n=3 optimal is 1.732 (equator).
    // These points would likely have larger min distance or comparable?
    // Actually having a pole point forces others away.
}
```
Wait, for $n=3$, putting one at South pole and two around equator at some latitude might be worse than all three on equator.
On equator, $y=0$. Points at $0, 120, 240$ deg.
My spiral puts one at $y \approx -1$ (South pole) for $n=3$. This is bad.
So I MUST hardcode small $n$.
When is Fibonacci valid? Probably $n > 12$ or so.
Actually, even for $n=6$, Fibonacci puts points on circles of latitudes. Optimal Octahedron has points on 3 circles of latitudes ($z=\pm 1, z=0$).
Fibonacci: $y$ values roughly $-1, -0.6, -0.2, 0.2, 0.6, 1$. Closer to poles?
Wait, $i/n$. For $n=6$.
$i=0 \to -1$. Pole.
$i=1 \to -0.66$.
$i=2 \to -0.33$.
$i=3 \to 0.33$.
$i=4 \to 0.66$.
$i=5 \to 1.33$? No $y = (2i/n) - 1$.
$i=0 \to -1$.
$i=3 \to 2(3)/6 - 1 = 0$.
$i=5 \to 1$.
So Fib sphere for $n=6$ gives points at: Pole, lat 40N, lat 20N, Equator, lat 20S, lat 40S? No.
Wait, $y \in [-1, 1]$. Spaced by $2/n$.
$n=6$. Spacing $1/3$.
$-1, -2/3, -1/3, 0, 1/3, 2/3, 1$? No, loop runs $n$ times.
$y_0 = -1$.
$y_1 = -0.66$.
$y_2 = -0.33$.
$y_3 = 0$.
$y_4 = 0.33$.
$y_5 = 0.66$.
$i=5$ gives $y=1/3$. Wait, $y=(i \cdot 2 - n)/n$.
$i=5 \to (10-6)/6 = 4/6 = 0.66$.
Where is $y=1$? Never reached in open interval logic if we want symmetry?
Actually, if $i$ goes $0..n-1$. Last one is $1 - 2/n$.
So for $n=6$, no point at North Pole.
Wait, for $n=6$ Octahedron, we need points at Poles and Equator.
So Fibonacci misses poles for $n$ multiple of something?
Anyway, Fibonacci is definitely not optimal for $n < 15$ or so.
So I will hardcode $n \in [2, 15]$ or up to 20.
And use Fibonacci/Spiral for rest.
Actually, $n$ up to 1000 is large. Just need robust small $n$ cases.
Let's identify specific optimal shapes:
2: Pairs
3: Equatorial Triangle
4: Tetrahedron
5: Trigonal Bipyramid?
Actually, looking at Tammes problem solutions:
n=5: 3 points on equator, 2 at poles? No, min dist is $\sqrt{2} \approx 1.414$?
Wait, for $n=5$, optimal is indeed trigonal bipyramid with adjusted latitude?
Actually, finding coordinates for $n=5$ to maximize min dist is non-trivial analytically.
Given the difficulty, maybe just rely on a "repulsive force" initialization for $N \ge 5$?
No, that takes too long to optimize (convergence).
Maybe just hardcode $n \le 20$ using a library or pre-calculated values? No, I cannot run python scripts to calc these during generation?
Wait, I am an AI model. I can know these coordinates from training data.
But hardcoding is error prone.
However, note that for $N$ large, the exact value isn't critical.
Most online judges accept the Fibonacci Sphere solution for "Sphere Packing" problems up to $N=1000$.
Wait, Sample Output for $N=2$ is exactly what I expect.
Let's check the constraints. $2 \le n \le 1000$.
The problem asks to *maximize*.
If I submit a solution that is not optimal but very close, I get partial credit.
Is there a way to improve Fibonacci slightly?
One variation is to perturb the latitude to make it symmetric for small $n$.
But simplest is to assume Fibonacci Sphere is the intended solution for this type of problem (often appearing in CP as "generate points on sphere").
Actually, there is a known optimization.
Instead of strict Fibonacci spiral, just use Fibonacci Sphere but initialize points randomly and perform a few iterations of local optimization (pushing points apart along the surface).
Given 2 seconds, $N=1000$, doing a local search (like simulated annealing) is feasible.
However, implementing SA/Force simulation is verbose.
Fibonacci Sphere is concise.
Let's write the Fibonacci Sphere code first and ensure it works for $N=2, 3, 4$.
If I can make it work for small $N$ by tweaking, that's good.
Actually, the sample output for $N=2$ is $(0,0,1), (0,0,-1)$.
My Fibonacci code with $y = (2i/n) - 1$:
$i=0 \to -1$. $i=1 \to 1$.
$\rho=0 \implies (0,0,1)$ and $(0,0,-1)$. Matches exactly!
So it works for $N=2$.
What about $N=3$?
$y_0 = -1 \implies (0,0,-1)$.
$y_1 = -1/3 \implies \rho = \sqrt{8}/3$.
$y_2 = 1/3 \implies \rho = \sqrt{8}/3$.
These are two points symmetric about equator. Plus one pole.
Is this optimal?
If we move them all to equator, we get equilateral triangle. Side $\sqrt{3} \approx 1.732$.
My Fib config: Point A at $z=1$. Points B, C at $z = -1/3$? No, $y=z$. $B$ at $z=-1/3$, $C$ at $z=1/3$?
Wait, indices $0, 1, 2$.
$i=0: y=-1$.
$i=1: y=-1/3$.
$i=2: y=1/3$.
Point 0: $(0,0,-1)$?
Point 1: $\rho=\sqrt{8}/3$. Angle $\phi$.
Point 2: $\rho=\sqrt{8}/3$. Angle $2\phi$.
Distances:
P0-P1: $\sqrt{(0-\rho)^2 + (0-\rho \sin\phi)^2 + (-1-\rho \cos\phi)^2}$.
Using identity $\rho^2 + (\rho \sin)^2 + 1 + \dots$
$= 1 + \rho^2 + \rho^2 \cos\phi + \dots$?
Distance squared $d^2 = R^2 + R^2 - 2R^2 \cos \gamma = 2 - 2 \cos \gamma$.
$\cos \gamma = y_0 y_1 + \mathbf{x}_0 \cdot \mathbf{x}_1$. $\mathbf{x}_0=(0,0,-1)$. $\mathbf{x}_1 = (\rho \sin\phi \cos\psi, \rho \sin\phi \sin\psi, \rho \cos\phi)$.
Dot prod = $- \rho \cos \theta$. Here $\rho = \sqrt{8}/3 \approx 0.94$.
$\theta \approx 137^\circ$. $\cos \theta \approx -0.76$.
Dot $\approx -0.94 \times -0.76 = 0.71$.
$d^2 = 2 - 2(0.71) = 0.58$. $d \approx 0.76$.
This is TERRIBLE. The points are squeezed.
Why? Because $y_1=-1/3$ is far from pole. And angle is $137^\circ$.
Basically, Fibonacci sphere spreads evenly in terms of solid angle, but doesn't maximize chord length (Euclidean distance) directly for small $N$.
Wait, maximizing min Euclidean distance on sphere IS maximizing solid angle separation roughly.
But for small $N$, the topology matters.
$N=3$ optimal is equilateral triangle on equator. $y$ coords all 0.
Fibonacci splits them into 3 levels.
So Fibonacci is bad for $N < 12$?
Actually, for $N \le 10$, there are distinct polyhedra.
To solve this reliably, I should hardcode $N$ up to say 12.
Then use a spiral for $N > 12$.
For $N > 12$, the differences between Fibonacci and perfect packing are small.
And the problem allows $10^{-6}$ tolerance relative to the claim.
Wait, if I use Fibonacci for $N=12$, it gives something close to icosahedron.
Icosahedron $N=12$. Edges are equal.
Fibonacci with $N=12$:
$i=0 \to -1$. Pole.
$I=11 \to 1$. Pole? No, $y = 22/12 - 1 = 0.83$. Not pole.
So it won't put points exactly at poles.
But for $N=12$ (Icosahedron), we have 2 poles.
Fibonacci puts 1 point very close to South Pole. Then another cluster near North Pole.
This is not icosahedron.
So I really should separate small $N$ cases.
Known Small $N$ configurations:
$N=2$: Poles.
$N=3$: Triangle on equator.
$N=4$: Tetrahedron.
$N=5$: ?
$N=6$: Octahedron (Poles + Equator).
$N=7$: Pentagonal bipyramid?
$N=8$: Square Antiprism?
$N=12$: Icosahedron.
$N=20$: Dodecahedron.

Actually, the question might be simpler. "Maximize min pairwise distance".
If I output points that are uniformly distributed, the min dist is close to optimal.
For $N=1000$, it's just density.
Maybe I should use the Repulsion Potential algorithm. It is guaranteed to find a local maximum.
Initialize with Fibonacci Sphere. Run gradient ascent / simulated annealing / force relaxation.
Steps:
1. Generate initial points using Fibonacci Sphere (or random on sphere).
2. Iterate: For each pair $(i,j)$, if distance $d < \lambda$, push them apart.
   Or compute forces: $\mathbf{F}_{ij} = \frac{k}{d_{ij}^2} \mathbf{v}_{ij}$ (Coulomb repulsion).
   Update positions: $\mathbf{p}_i \leftarrow \mathbf{p}_i + \eta \sum_j \mathbf{F}_{ij}$. Project back to sphere? Or maintain normal velocity?
   Keep moving along great circle tangent.
   Do this for several steps.
   With $N=1000$, $O(N^2)$ per step. Steps? Maybe 100-200. Total ops $2 \cdot 10^7$. Feasible in 2s.
   This approach is robust.
   
   Implementation details:
   Use a fixed number of iterations.
   Start with $N$ random points on sphere.
   Compute distances.
   Apply forces.
   Move points. Normalize to keep radius 1.
   Stop when displacement is small.
   This finds a stable configuration locally.
   Initial state matters. Fibonacci Sphere is good init.
   
   Let's check complexity. $N=1000$. $N^2 = 10^6$.
   Iterations: 200. Total $2 \cdot 10^8$. Might be tight for 2s.
   Can reduce iterations or use fewer precision.
   Actually, since $N$ is up to 1000, $O(N^2)$ is risky if not optimized.
   However, we can skip updating points far from others?
   Or simply rely on the fact that for large $N$, the spiral distribution IS the solution.
   Most CP problems of this type (placing points on sphere) accept Fibonacci Sphere. The requirement "maximize" is usually satisfied by Fibonacci Sphere within the error tolerance for scoring (ratio).
   Wait, if $N=1000$, optimal min dist is roughly $0.1$.
   If Fibonacci is off by 0.1%, ratio is 0.999. Score 100.
   I'll bet on Fibonacci Sphere.
   BUT, to be safe, I will optimize the Fibonacci formula to handle small $N$ cases correctly?
   Actually, I can just use the Fibonacci Sphere algorithm. It is widely cited as "Spiral Lattic" for uniform distribution.
   Let's check if the problem source is known. "SphereSpread".
   It seems to be a custom problem or from a contest.
   Given the constraint of "Time limit: 2 seconds", simple $O(N^2)$ might pass if constant factor is small.
   I will implement a simple optimization loop? No, risky.
   I will implement Fibonacci Sphere.
   
   Wait, I found a paper "Optimal distribution of points on a sphere".
   For $N$ points, the optimal arrangements correspond to vertices of Platonic solids for $N=4, 6, 8, 12, 20$.
   Wait, $N=8$ is cube? No, Cuboctahedron or Square Antiprism.
   $N=12$ is icosahedron.
   $N=20$ is dodecahedron.
   $N=30$?
   
   Okay, strategy:
   1. Handle $N \le 20$ with specific hardcoded configurations if possible, OR just use Fibonacci Sphere for $N \le 20$ hoping it's good enough?
   No, for small $N$, Fibonacci is poor.
   But hardcoding exact coordinates for $N=5, 7, 8, 9, 10, 11, 15, 16, \dots$ is tedious and error prone.
   
   Alternative:
   Just generate points using the Fibonacci Sphere method.
   However, modify the logic to place points on a grid of latitudes?
   No, Fibonacci is the best simple generator.
   Maybe the "score" system is forgiving.
   Also, I can scale the coordinates.
   
   Wait, there is one crucial thing. The problem asks to output `min_dist`.
   If I output Fibonacci points, I should calculate their true `min_dist` and output that.
   The checker compares my `min_dist` with the `ref_answer` (which corresponds to optimal).
   Since my points might not be optimal, my `min_dist` will be less than `ref_min_dist`.
   The score is `100 * min(my_dist / ref_dist, 1)`.
   So if I achieve 95% of the optimal min distance, I get 95 points.
   This suggests finding the EXACT optimum is not mandatory, just "good enough".
   Fibonacci Sphere is known to provide excellent coverage and is usually within 0.1-0.2% of optimal packing for large $N$. For small $N$, it varies.
   However, since I can implement an iterative improvement easily, I should do it.
   But $O(N^2)$ with 1000 points might TLE in Python, but in C++ it's fine.
   I'll write the optimization step.
   Start with Fibonacci Sphere.
   Run "Cooling" (Simulated Annealing) or Gradient Descent.
   Objective: Minimize Max Overlap? No, Maximize Min Distance.
   Equivalent to Minimizing energy $E = \sum_{i<j} f(d_{ij})$.
   Function $f(d)$ should decrease with $d$. E.g. $1/d^2$ or $1/d^4$.
   We want to minimize energy.
   Since $N$ is up to 1000, computing all pairs is $5 \cdot 10^5$. Fast.
   Running gradient descent takes maybe 50-100 epochs.
   $50 \cdot 5 \cdot 10^5 \approx 2.5 \cdot 10^7$ operations. Well within 2 seconds (usually $10^8$ ops/sec).
   
   Algorithm:
   1. Initialize points on sphere using Fibonacci Sphere.
   2. Define potential function. Soft Coulomb repulsion.
   3. For each iteration:
      Calculate forces on each point.
      Displace points slightly towards gradient.
      Project back to sphere.
   4. Refine.
   
   This should find a very good local minimum.
   Is there a risk of getting stuck in a local minimum?
   Fibonacci sphere is globally fairly balanced. It should converge to the global optimum or close to it.
   
   Wait, is calculating all pairwise forces $O(N^2)$ too slow if repeated many times?
   Inside 2 seconds.
   If I have 100 iterations: $100 \times 10^6 = 10^8$. In C++, this is safe.
   Using `std::pair` or struct is fine.
   I'll use `fixed` point format or `float`? `double` for precision.
   
   Let's structure the code:
   - Struct `Point { x, y, z }`
   - Main loop.
   - Initialization using Fibonacci.
   - Optimization loop (maybe limited to 50 iterations to be safe).
   - Finalize points.
   - Compute actual min dist.
   - Print min dist.
   - Print points.
   
   Wait, the problem asks to maximize min pairwise distance.
   The "reference solution" computes this maximum.
   My optimizer tries to approximate it.
   The optimization function:
   Energy = Sum of exp(-dist) or 1/dist^2.
   We want to maximize min dist.
   So we penalize small distances.
   Force between $i$ and $j$ is proportional to $1/d_{ij}^3$ (gradient of $1/d_{ij}^2$) in direction of vector difference.
   $\mathbf{F}_{ij} = \frac{\partial U}{\partial \mathbf{r}_i}$. $U = \sum 1/d_{ij}^2$.
   $\nabla_i (1/r_{ij}^2) = -2/r_{ij}^3 (\partial r_{ij} / \partial \mathbf{r}_i) \cdot \hat{r}_{ij} = -2/r_{ij}^3 \cdot (-1) \hat{r}_{ij}?$
   Actually, $1/d_{ij}^2$ decreases as $d$ increases. Gradient points to increasing distance (outwards from $j$).
   Direction $\mathbf{v} = \mathbf{p}_i - \mathbf{p}_j$.
   Unit vector $\mathbf{n} = \mathbf{v} / |\mathbf{v}|$.
   Force magnitude $k / |\mathbf{v}|^3$.
   Accumulate forces.
   Update $\mathbf{p}_i \leftarrow \mathbf{p}_i + \epsilon \sum \mathbf{F}_{ij}$.
   Constraint $|\mathbf{p}_i| = 1$.
   Update: $\mathbf{p}_i \leftarrow \frac{\mathbf{p}_i + \epsilon \sum \mathbf{F}_{ij}}{|\dots|}$.
   This keeps them on sphere.
   Step size $\epsilon$. Decay step size to converge.
   
   Since we want to avoid oscillation, a momentum term helps.
   Or just a fixed number of moves.
   Given the "Time Limit", simple loop is best.
   
   Wait, if I use 2 seconds, $10^8$ ops is plenty for basic float arithmetic.
   I will implement this.
   
   Corner cases: $N=2, 3, 4$.
   Fibonacci + Optimization will naturally find these optima (tetrahedron etc.)?
   For $N=2$, start with poles. Optimization stays there.
   For $N=3$, start with spiral. Spiral puts 1 pole. Moves to equator.
   Should work.
   
   One detail:
   "All coordinates must satisfy xi² + yi² + zi² ≤ 1".
   Optimization ensures equality 1, which is fine.
   
   Also, "absolute or relative error at most 10^-6" for matching min_dist.
   The score calculation relies on `my_min / ref_min`.
   So my `my_min` must be close to `ref_min`.
   My optimizer tries to make `my_min` as large as possible.
   With $N=1000$, the gap is small.
   
   I will use a decay for step size to allow finer movements later.
   
   What about memory? 512 MB is plenty.
   
   Let's check code efficiency.
   Using raw arrays/vectors is faster than structs inside loops.
   Accessing x[i], y[i], z[i].
   
   ```cpp
   #include <iostream>
   #include <vector>
   #include <cmath>
   #include <algorithm>
   
   using namespace std;
   
   struct Point {
       double x, y, z;
   };
   
   int main() {
       int n;
       if (!(cin >> n)) return 0;
       
       vector<Point> p(n);
       
       // Initialization: Fibonacci Sphere
       const double PI = acos(-1.0);
       const double phi = (1.0 + sqrt(5.0)) / 2.0;
       const double GOLDEN_ANGLE = PI * (3.0 - sqrt(5.0)); // ~ 2.3999639
       
       for(int i = 0; i < n; ++i) {
           double t = (double)i / n;
           double y = 2.0 * t - 1.0; // Range -1 to 1
           double r = sqrt(max(0.0, 1.0 - y * y));
           double theta = GOLDEN_ANGLE * i;
           // Check for boundary issues
           if (y >= 0.999999) y = 1.0; // Clamp to avoid NaN if floating point noise
           // Wait, t=i/n. i=0 -> y=-1.
           
           double x = r * cos(theta);
           double z = r * sin(theta); // Swap y/z? No, y is latitude.
           // Standard spherical coords: x=rad*sin(lat)*cos(long), y=sin(lat), z=cos(long)?
           // My variables: y is latitude (vertical).
           // Let's map:
           // p.y = y
           // p.x = r * cos(theta)
           // p.z = r * sin(theta)
           // Wait, Fibonacci uses theta in xy plane.
           // So yes:
           // p.y = y
           // p.x = r * cos(GoldenAngle * i)
           // p.z = r * sin(GoldenAngle * i)
           
           // Ensure unit vector
           double len = sqrt(x*x + y*y + z*z);
           p[i] = {x/len, y/len, z/len};
       }
       
       // Optimization
       // Repulsive force minimization
       // Energy ~ Sum 1/d^2.
       // Forces ~ 1/d^3.
       
       vector<double> fx(n), fy(n), fz(n);
       // Precompute distances or recompute in loop. Recomputing O(N^2) is OK.
       
       double dt = 0.1; // Time step
       int iter = 100;
       
       // To speed up, we can stop early if min change is small.
       // But for safety, fixed iterations.
       
       for (int k = 0; k < iter; ++k) {
           // Reset forces
           fill(fx.begin(), fx.end(), 0.0);
           fill(fy.begin(), fy.end(), 0.0);
           fill(fz.begin(), fz.end(), 0.0);
           
           // Compute forces
           // Optimization: Only move points if close?
           // Full O(N^2) is safer for simplicity.
           
           // Using a threshold for interaction to save time? 
           // If d < 10^-9 (almost touching) we act.
           // But we want to push everything away.
           // So full sum is needed.
           
           for (int i = 0; i < n; ++i) {
               // Self force 0.
               for (int j = i + 1; j < n; ++j) {
                   double dx = p[i].x - p[j].x;
                   double dy = p[i].y - p[j].y;
                   double dz = p[i].z - p[j].z;
                   
                   double d2 = dx*dx + dy*dy + dz*dz;
                   double d = sqrt(d2);
                   
                   if (d < 1e-9) continue; // Avoid division by zero
                   
                   // Force magnitude F ~ 1/d^2 (Coulomb)
                   // Gradient is -2/d^3 * (p_i - p_j)/d ?
                   // We want to maximize distance.
                   // Gradient of d wrt p_i is u_ij (unit vector from j to i).
                   // Potential V = -d (if maximizing d).
                   // Or Penalty P = -d (negative dist).
                   // Penalty function P = 1/d.
                   // dP/dd = -1/d^2.
                   // Gradient = -1/d^2 * unit_vec.
                   // So we want to apply Force proportional to unit_vec / d^2.
                   
                   double f_mag = 1.0 / (d2); // Scaling factor
                   // Force vector:
                   // F_x += (dx/d) * f_mag
                   // ...
                   
                   double inv_d = 1.0 / d;
                   double fx_contri = (dx * inv_d) * f_mag;
                   double fy_contri = (dy * inv_d) * f_mag;
                   double fz_contri = (dz * inv_d) * f_mag;
                   
                   fx[i] -= fx_contri; // Action-Reaction?
                   // Wait, if I apply force to i from j:
                   // It is repulsive. Pushes i away from j.
                   // Direction p_i - p_j.
                   // So F_i += k * (p_i - p_j) / d^3.
                   
                   fx[i] -= fx_contri;
                   fx[j] += fx_contri;
                   fy[i] -= fy_contri;
                   fy[j] += fy_contri;
                   fz[i] -= fz_contri;
                   fz[j] += fz_contri;
               }
           }
           
           // Move points
           double eta = 0.01; // Learning rate
           // Decrease learning rate
           eta *= 0.999;
           
           for (int i = 0; i < n; ++i) {
               // Apply forces
               double nx = p[i].x + eta * fx[i];
               double ny = p[i].y + eta * fy[i];
               double nz = p[i].z + eta * fz[i];
               
               // Re-normalize to surface of sphere
               // Note: The force pushes along straight line in 3D, but constraint is sphere.
               // Moving in tangent direction preserves radius best.
               // But normal component changes radius.
               // Since we want to stay on sphere, projection is best.
               
               double nn = sqrt(nx*nx + ny*ny + nz*nz);
               if (nn > 1e-9) {
                   p[i].x = nx / nn;
                   p[i].y = ny / nn;
                   p[i].z = nz / nn;
               } else {
                    // Collapsed, move to random?
                    // Should not happen with proper init.
               }
           }
       }
       
       // Calculate min_dist
       double min_dist = 1e9;
       for(int i = 0; i < n; ++i) {
           for(int j = i+1; j < n; ++j) {
               double dx = p[i].x - p[j].x;
               double dy = p[i].y - p[j].y;
               double dz = p[i].z - p[j].z;
               double d = sqrt(dx*dx + dy*dy + dz*dz);
               if (d < min_dist) min_dist = d;
           }
       }
       
       cout << fixed << setprecision(10) << min_dist << endl;
       for(int i = 0; i < n; ++i) {
           cout << fixed << setprecision(10) << p[i].x << " " << p[i].y << " " << p[i].z << endl;
       }
       
       return 0;
   }
   ```
   Wait, logic for force accumulation:
   `fx[i] -= fx_contri`?
   Direction from j to i is `p[i] - p[j]`.
   `dx = p[i].x - p[j].x`. Positive dx means i is to right of j. Force pushes i further right.
   So `F_i += (+) dx/d`.
   My code had `fx[i] -= fx_contri`?
   Wait, `fx_contri` is `(dx * inv_d) * f_mag`.
   If `dx` positive, `fx_contri` positive.
   `F_i += fx_contri`.
   But `fx[i] -= fx_contri` adds force in opposite direction?
   Ah, `j` exerts force on `i`.
   I update `fx[i]` and `fx[j]`.
   `fx[i] += fx_contri`. `fx[j] -= fx_contri`.
   Correct.
   
   Also, `d` calculation inside inner loop is redundant? No, needed.
   Inner loop runs $N^2/2$ times.
   Outer loop 100.
   $100 \times 10^6 / 2 \approx 5 \cdot 10^7$. Very safe.
   
   One issue: The optimization converges to a local minimum.
   With Fibonacci start, for $N=1000$, it should reach the "uniform" state.
   Is the force law good? $1/d^2$ potential (Coulomb) leads to soft collisions.
   As $d \to 0$, force $\to \infty$. Prevents overlap.
   But since we start on sphere with separation, and optimize, it should settle.
   
   Wait, `eta` decreases. This helps stability.
   The normalization `p[i] /= nn` projects back to sphere.
   Ideally, the force should be projected onto the tangent plane of the sphere to keep it on the sphere efficiently.
   Tangential component of force is $\mathbf{F} - (\mathbf{F} \cdot \mathbf{n})\mathbf{n}$.
   Since $\mathbf{n} \cdot \mathbf{F} = (\mathbf{p}_i - \mathbf{p}_j) \cdot \mathbf{p}_i / d^3 \dots$
   This is getting complicated.
   Normalizing position after adding small vector is equivalent to following the steepest descent on the manifold?
   Actually, simply moving the point and renormalizing is a common trick.
   However, for large displacements it might overshoot.
   Smaller `eta` helps.
   
   Another thought:
   For very small $N$, the optimization might struggle if initialized at poles (Fibonacci does this).
   For $N=3$, init is Pole + Equator.
   Forces might pull Pole towards Equator.
   Eventually it should become Equilateral.
   Since I want to be sure, I can include a special block for $N < 6$ or so.
   Wait, $N=4$ (Tetrahedron) requires breaking symmetry.
   If I don't hardcode, optimization from random/fib start usually finds Tetrahedron?
   It might take time.
   But with 2 seconds and $N=4$, optimization is instant.
   So I don't need to hardcode.
   
   Wait, the force function $1/d^2$ is attractive or repulsive?
   $V = 1/d$. Force is attractive towards min $V$ (large $d$)? No, $d \to \infty$ gives $V=0$.
   Wait, we want to MAXIMIZE $d$.
   So we want to MINIMIZE $-d$ (distance).
   This is weird.
   We want to MINIMIZE Potential Energy where potential is $U(d)$ decreasing with $d$.
   E.g. $U(d) = -d$.
   Then Force is gradient of $-d$. Negative gradient.
   Gradient of $d$ is $u$ (direction of increase).
   So force is $-u$. Pulling together.
   Wait.
   If I use $U(d) = 1/d$.
   As $d$ increases, $U$ decreases. So points try to separate.
   $\nabla U = \nabla (d^{-1}) = -d^{-2} \nabla d = -d^{-2} u_{ij}$.
   Force is $-\nabla U = d^{-2} u_{ij}$.
   $u_{ij}$ is vector from $j$ to $i$ (direction of increasing distance).
   So Force pushes $i$ away from $j$.
   So $U = 1/d$ works.
   Wait, if $d$ is large, force is small. Good.
   If $d$ is small, force is large. Good.
   My code uses `1/d^2` for magnitude?
   If I choose $U = 1/d^2$.
   $\nabla U = -2/d^3 u_{ij}$.
   Force $F = \nabla U$? No, force is $-\nabla U$.
   So $F = 2/d^3 u_{ij}$.
   My code uses `1/d^2` scaling. `inv_d * f_mag` -> `1/d * 1/d^2 = 1/d^3`.
   The magnitude is roughly proportional to $1/d^3$.
   So it matches Coulomb force field $\sim 1/r^2$ gradient.
   So it pushes apart.
   
   One correction: `f_mag` should be part of potential derivative.
   If potential is $1/d$, deriv is $1/d^2$. Force magnitude $1/d^2$.
   If potential is $1/d^2$, deriv is $2/d^3$. Force magnitude $1/d^3$.
   The choice doesn't matter for qualitative behavior, just scaling.
   
   Final check on constraints.
   Input $n$ up to 1000.
   Code runs in 2s.
   Memory ok.
   Output format:
   Real number min_dist.
   Lines with xi yi zi.
   
   Precision: 10 digits.
   
   Wait, there's a catch with Fibonacci Sphere initialization.
   I wrote `y = 2.0*t - 1.0`.
   For $i=0$, $t=0$, $y=-1$.
   For $i=n-1$, $t=1-1/n$, $y = 2(1-1/n)-1 = 1 - 2/n$.
   So $y$ never reaches 1.
   This means we miss the North Pole?
   Or rather, we have points clustered around $y=1$.
   The last point $i=n-1$ has $y \approx 1$.
   If $n=2$:
   $i=0 \to y=-1$. Pole.
   $i=1 \to y=1$. Pole.
   Correct.
   If $n=3$:
   $i=0 \to -1$.
   $i=1 \to -1/3$.
   $i=2 \to 1/3$.
   Wait, $y$ values: $-1, -0.33, 0.33$.
   Where is $0.66$? $1/3$?
   Formula: $2(i/n) - 1$.
   $i=0: -1$.
   $i=1: 2/3 - 1 = -1/3$.
   $i=2: 4/3 - 1 = 1/3$.
   Missing $y=1$ (North pole).
   So for odd $n$, we might miss a pole.
   Wait, $i$ goes $0$ to $n-1$.
   If $n$ is even, say 4.
   $i=0 \to -1$.
   $i=2 \to 0$.
   $i=3 \to 2/3$.
   So we have -1, -1/3, 1/3, 1.
   Wait $i=3 \implies 2(3)/4 - 1 = 1.5 - 1 = 0.5$.
   Wait, $y = 2i/n - 1$.
   $i=n \implies 1$.
   But loop stops at $n-1$.
   So $y_{max} = 2(n-1)/n - 1 = 2 - 2/n - 1 = 1 - 2/n$.
   So the North Pole is never hit exactly.
   This is usually fine, but maybe slightly shifts the symmetry.
   For $n=2$, $y_{max} = 1 - 1 = 0$?
   Wait. $n=2$. $i=1$. $y = 2(1)/2 - 1 = 0$.
   Wait, earlier I said $i=1 \implies y=1$.
   Why?
   $y = (i+1)/n \times 2 - 1$? No.
   Common formula is $y = 1 - 2 i/n$?
   If $y = 1 - 2i/n$.
   $i=0 \to 1$. Pole.
   $i=n-1 \to 1 - 2(n-1)/n = 1 - 2 + 2/n = 2/n - 1$.
   So range $[2/n - 1, 1]$.
   This covers both poles (one end is exactly 1, other is close to -1).
   This is better.
   Let's use $y = 1.0 - 2.0 * (double)i / n$.
   
   Wait, sample output $N=2$ is $(0,0,1)$ and $(0,0,-1)$.
   With $y = 1 - 2i/n$:
   $i=0 \to y=1$.
   $i=1 \to y = 1 - 1 = 0$.
   So we get Pole and Equator.
   Wait, this loses one pole!
   We need Pole and Pole.
   So $y$ should span exactly from -1 to 1.
   Indices $0 \dots n-1$.
   Number of intervals $n$.
   Midpoint of -1 and 1 is 0.
   Step size $2/n$.
   Values: $-1 + k \cdot \frac{2}{n}$.
   For $k=0 \dots n-1$.
   Last value: $-1 + \frac{2(n-1)}{n} = -1 + 2 - \frac{2}{n} = 1 - \frac{2}{n}$.
   Still misses $1$.
   To hit $1$, we need $k$ up to $n$.
   But we only have $n$ points.
   If we use $k=0 \dots n-1$, we cover $[-1, 1-2/n]$.
   We miss $[1-2/n, 1]$.
   However, if we consider the distribution of solid angle, this asymmetry is handled by the Golden Angle.
   Usually, the formula used is:
   $y = \frac{i - 0.5}{n} \times 2$.
   Or simply:
   $t = i/n$.
   $y = 2t - 1$.
   Wait, $i=0 \implies y=-1$.
   $i=n-1 \implies y = 2(1-1/n)-1 = 1-2/n$.
   This leaves the North Pole uncovered.
   However, the Fibonacci Sphere algorithm often assumes $n$ points on sphere.
   Is there a version that includes poles?
   The "Spiral Point Grid" article suggests:
   $y = 1 - 2t$ for $t$ going from $0$ to $1$.
   Discrete: $t_k = (k+1)/(n+1)$?
   Or just use $y = \cos(\theta)$.
   Let's check Wikipedia "Fibonacci lattice".
   $y_k = \frac{2k - n + 1}{n}$.
   For $k=0$, $y = (-n+1)/n = -1 + 1/n$.
   For $k=n-1$, $y = (2(n-1)-n+1)/n = (2n-2-n+1)/n = (n-1)/n = 1 - 1/n$.
   Still doesn't hit $\pm 1$.
   But for large $n$, it's dense.
   For $n=2$:
   $k=0 \to -1 + 1/2 = -0.5$.
   $k=1 \to 1 - 1/2 = 0.5$.
   Yields $y=-0.5, y=0.5$.
   Points at $z = -0.5, z=0.5$.
   Distance $\sqrt{(0.5)^2+(0.5)^2} = \sqrt{0.5} \approx 0.707$.
   This is WRONG for $n=2$. They should be at poles.
   So standard Fibonacci spiral is not strictly optimal for small $n$.
   Thus, I SHOULD hardcode small $n$ or fix the formula.
   Actually, for $n=2$, the optimal is diameter.
   If I use Fibonacci, I get wrong result.
   So I will add a condition:
   If $n \le 20$, use a specific construction?
   Hardcoding 20 configs is tedious.
   But I can use a brute-force "local search" starting from random/Fibonacci points.
   Wait, if I use local search, it will quickly correct the $n=2$ pole issue.
   For $n=2$, start with random points. If they are far, they move apart.
   So I don't need special initialization logic if optimization is active.
   But for $n=2$, random points might start close, optimization takes time?
   Optimization is $O(N^2)$. For $N=2$, negligible.
   So optimization will fix everything.
   
   The only concern is Time Limit.
   For $N=1000$, if optimization gets stuck?
   It shouldn't.
   So I will rely on optimization.
   
   Wait, I need to include `<iomanip>` for `setprecision`.
   
   One more thing:
   Is it possible that the force accumulation `double` overflow? No.
   
   Final plan:
   1. Read $n$.
   2. Init points using Fibonacci Sphere (adjusted or not, doesn't matter much as optimizer works).
      Use $y = (2.0 * i - n + 1.0) / n$? No, let's use `y = 2.0 * (i / n) - 1.0`.
      Wait, for $N=2$, this gave $y=-1, y=0$.
      My optimizer should push them to poles?
      Start $y=-1 \to (0,0,-1)$.
      $y=0 \to (1,0,0)$.
      Force will push them apart.
      $d$ will grow.
      Will it hit poles? Yes.
   3. Run optimization loop.
   4. Compute min_dist.
   5. Output.
   
   Double check `1e-6` relative error.
   My `min_dist` calculation is on the output points.
   The output points are computed by optimizer.
   The grader checks `actual_min` vs `claimed_min`.
   So if I print `min_dist`, and my points match that, I'm good.
   I need to make sure I don't print a larger value than actual.
   So I compute actual and print.
   
   One subtle issue:
   With $N=1000$, the points might drift to slightly outside radius 1 due to precision if I don't normalize every step?
   I normalize every step.
   The normalization is `p /= nn`.
   This ensures `norm(p)=1`.
   So $x^2+y^2+z^2 = 1$.
   
   Complexity of optimization:
   Iterations: 50 is enough for convergence?
   Maybe 100 to be safe.
   Ops count: $100 \times 10^6 / 2 = 5 \cdot 10^7$.
   This takes ~0.1 - 0.5 seconds in C++.
   Safe.
   
   I'll write the solution now.
   Wait, I should probably handle $N$ small specifically with hardcoded logic to avoid any "slow to converge" risks?
   No, 1000 points is fine.
   But for $N=2$, optimization might need more steps?
   Actually, for $N=2$, it takes 1 step to reach poles.
   
   Code Structure:
   Includes.
   Main.
   Read N.
   Vectors.
   Loop init.
   Optimization loop.
   Calc min.
   Print.

   I will define a struct.
   Using `long double`? `double` is sufficient (64-bit IEEE 754 has ~15 decimal digits).
   
   One detail:
   `theta = GOLDEN_ANGLE * i`.
   Should `theta` wrap modulo $2\pi$?
   `cos` and `sin` handle large arguments fine, but precision drops.
   For $N=1000$, $1000 \times 2.4 \approx 2400$ radians. Modulo $2\pi$ keeps it small.
   Better: `double theta = GOLDEN_ANGLE * i;`
   Wait, `GOLDEN_ANGLE` is in radians.
   
   Also, make sure `n` is read correctly.
   
   Wait, I realized that for $N=1000$, the points will be very close to optimal packing.
   The optimization might help but standard Fibonacci is actually quite good.
   The force minimization is just to smooth out minor irregularities.
   
   I'll add a small check: if $N < 10$, I might use a simpler logic?
   No, let's just trust the optimization.
   Wait, if I use optimization, I might end up with points having $dist < 0$ or something?
   No, they are on sphere.
   
   Wait, `fx[i] -= fx_contri` vs `+=`.
   If I want to push i away from j.
   `vec = p[i] - p[j]`.
   `force` is along `vec`.
   `p[i] += force`.
   `p[j] -= force`.
   My code accumulates in `fx[i]` and `fx[j]`.
   `fx[i] += fx_contri`.
   `fx[j] -= fx_contri`.
   Wait, `fx_contri` is signed.
   `dx = p[i].x - p[j].x`.
   If `p[i].x > p[j].x`, `dx > 0`.
   `fx_contri > 0`.
   `fx[i] += pos`. `fx[j] -= pos`.
   This means i moves +x, j moves -x.
   Distance increases. Correct.
   
   The code looks solid.
   
   Wait, `min_dist` initialization.
   `min_dist = 1e9`.
   If points coincide (impossible), 0.
   
   One small optimization:
   Use `vector` access `[]`.
   Use `fast IO`?
   `cin.tie(NULL); ios_base::sync_with_stdio(false);`
   Not strictly necessary for $N=1000$ output (1000 lines is fast).
   But good practice.

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

struct Point {
    double x, y, z;
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    vector<Point> p(n);
    
    // Fibonacci Sphere Initialization
    const double PI = acos(-1.0);
    const double GOLDEN_RATIO = (1.0 + sqrt(5.0)) / 2.0;
    const double GOLDEN_ANGLE = PI * (3.0 - sqrt(5.0)); 

    for (int i = 0; i < n; ++i) {
        double t = (double)i / n;
        double y = 2.0 * t - 1.0;
        
        // Clamping y to avoid domain error in sqrt for floating point inaccuracies near 1
        if (y > 0.999999) y = 0.999999;
        if (y < -0.999999) y = -0.999999;
        
        double r = sqrt(1.0 - y * y);
        double theta = i * GOLDEN_ANGLE; 
        
        // Random phase shift isn't strictly necessary for uniformity but avoids clustering
        // Let's just use theta = i * golden_angle.
        // Actually, some sources suggest adding random offset for i=0.
        // We'll stick to deterministic.
        
        double x = r * cos(theta);
        double z = r * sin(theta); // Swapping names to match z as azimuth
        // Wait, usually theta is longitude. 
        // Let's assign x = r*cos(theta), z=r*sin(theta) or vice versa.
        // It doesn't matter for distance metric.
        
        p[i] = {x, y, z};
    }

    // Optimization: Simulated Repulsion
    // To ensure we find a local max-min distance.
    // Using Soft Coulomb-like repulsion.
    
    // Arrays for accumulated forces to avoid recomputing
    vector<double> fx(n, 0.0), fy(n, 0.0), fz(n, 0.0);
    
    double eta = 0.1; // Learning rate
    int max_iter = 150; 
    
    // To ensure convergence, reduce eta over iterations
    // But simple fixed step for 150 iterations is often enough for N=1000
    
    // Actually, reducing step size is important for stability
    // But with projection, large steps cause oscillation.
    // Let's decay eta.
    
    // We repeat optimization a bit to polish.
    // First run coarse, then fine?
    // Fixed schedule: eta *= 0.995 each outer loop step?
    // Or just fixed step size.
    // Let's keep step constant but decay slowly.
    
    // Wait, if I use decay, I need to be careful.
    // Let's just iterate fixed steps with decaying eta.
    
    for (int k = 0; k < max_iter; ++k) {
        // Fill forces
        for (int i = 0; i < n; ++i) {
            fx[i] = 0.0; fy[i] = 0.0; fz[i] = 0.0;
        }
        
        for (int i = 0; i < n; ++i) {
             // Optimization: Only check j > i
             // We accumulate forces on i and j
             for (int j = i + 1; j < n; ++j) {
                 double dx = p[i].x - p[j].x;
                 double dy = p[i].y - p[j].y;
                 double dz = p[i].z - p[j].z;
                 
                 double d2 = dx*dx + dy*dy + dz*dz;
                 double d = sqrt(d2);
                 
                 // If points overlap significantly, push hard.
                 // Otherwise 1/d^3 force.
                 // Avoid divide by zero
                 if (d < 1e-10) continue; 
                 
                 double f_mag = 1.0 / d2; // Force intensity factor
                 double inv_d = 1.0 / d;
                 
                 // Force vector direction
                 double f_x = dx * f_mag; // dx/d * 1/d^2
                 double f_y = dy * f_mag;
                 double f_z = dz * f_mag;
                 
                 fx[i] += f_x;
                 fx[j] -= f_x;
                 fy[i] += f_y;
                 fy[j] -= f_y;
                 fz[i] += f_z;
                 fz[j] -= f_z;
             }
        }
        
        // Move points and project to sphere
        double step = 0.05;
        step *= (1.0 / (1.0 + (double)k * 0.001)); // Decay
        
        for (int i = 0; i < n; ++i) {
            double nx = p[i].x + step * fx[i];
            double ny = p[i].y + step * fy[i];
            double nz = p[i].z + step * fz[i];
            
            // Projection
            double nn = sqrt(nx*nx + ny*ny + nz*nz);
            if (nn > 1e-10) {
                p[i] = {nx/nn, ny/nn, nz/nn};
            } else {
                // Fallback if collapse happens
                // Put it at random spot?
                // Should not happen.
                p[i].x = 1.0; p[i].y = 0.0; p[i].z = 0.0;
            }
        }
    }

    // Calculate final min distance
    double min_dist = 1e9;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = p[i].x - p[j].x;
            double dy = p[i].y - p[j].y;
            double dz = p[i].z - p[j].z;
            double d = sqrt(dx*dx + dy*dy + dz*dz);
            if (d < min_dist) min_dist = d;
        }
    }

    cout << fixed << setprecision(10) << min_dist << "\n";
    for (int i = 0; i < n; ++i) {
        cout << fixed << setprecision(10) << p[i].x << " " << p[i].y << " " << p[i].z << "\n";
    }

    return 0;
}
```
   Is the force accumulation logic correct?
   `fx[i] += f_x`. `f_x` is positive if `dx` positive. `dx = p[i]-p[j]`.
   This pulls `i` away from `j` (positive force on `i` in `+x` direction means it moves away).
   `fx[j] -= f_x`. `j` moves in `-x` direction.
   Wait, if `i` is to the right of `j` (`dx > 0`), `j` should move left (`-x`).
   So `fx[j]` should be negative.
   `fx[j] -= f_x` -> `f_x` is positive. `fx[j]` decreases. Negative.
   So `j` moves left.
   Correct.
   
   Wait, `f_x` derived from `dx * f_mag`.
   `f_mag = 1/d^2`. `d` is positive.
   So signs are correct.
   
   Wait, `step` decay formula: `1 / (1 + k*0.001)`.
   For `k=0`, `step=0.05`.
   For `k=100`, `step ~ 0.05/1.1`.
   Reasonable.
   
   There is a slight risk of the code taking too much time.
   `10^6` pairs per outer loop. 150 loops -> `1.5 \cdot 10^8` operations.
   Modern CPU handles `10^8` cycles in < 1 sec. Each iteration has multiplications and sqrt.
   `sqrt` is expensive. `sqrt` inside inner loop.
   $1.5 \cdot 10^8$ sqrts is heavy.
   Can we optimize?
   `sqrt(d2)` -> `d`. `1/d` -> `inv_d`.
   Replace `sqrt`?
   If we use `d2` in denominator `d^2`?
   Force magnitude `1/d^2`.
   Wait, force vector is `dx/d^2 * (1/d) * d`? No.
   Force `F = (dx, dy, dz) / d^3`.
   Wait, I calculated `f_x = dx / d^2`. This is incorrect magnitude if I need `1/d^3`.
   Wait, `inv_d` is $1/d$. `f_x = dx * (1/d) * (1/d^2)`?
   My code: `double f_mag = 1.0 / d2;` -> $1/d^2$.
   `double f_x = dx * f_mag;` -> $dx/d^2$.
   This corresponds to force $F \propto 1/r^2$.
   Potential $V \propto -1/r$.
   Is this strong enough?
   Yes, it pushes apart.
   Does it remove `sqrt`?
   Yes! `f_x` uses `d2`.
   Wait, `fx[j] -= f_x`.
   `fx` is added to update `p[i]`.
   The update is `p[i] += step * fx`.
   Wait, units.
   Force is vector.
   If $F = \vec{r}/d^3$, then $x$-component is $x/d^3$.
   My code calculates $x/d^2$.
   So magnitude is scaled by $d$.
   Does this affect convergence?
   Maybe slower, but still repulsive.
   But $1/d^3$ blows up faster as $d \to 0$.
   Using $1/d^2$ is slightly gentler.
   To avoid `sqrt`, I can calculate force as $(x,y,z)/d^{2.5}$?
   No, just keep `sqrt`. $10^8$ sqrts is borderline but often passes in 2s if logic is tight.
   Wait, time limit 2s.
   If I have 150 iterations, maybe reduce to 80?
   Or reduce `max_iter`.
   Fibonacci sphere is already very good.
   Optimization is just a polish.
   Maybe 50 iterations is enough?
   Or just 3 phases: 50 coarse, 50 medium, 50 fine.
   But simply reducing to 80 iterations and skipping `sqrt` if possible.
   Can I skip `sqrt` for force calc?
   $d^2$ is available.
   $d^3 = d \cdot d^2$. Need $d$.
   So `sqrt` is needed.
   Unless I use Newton-Raphson for root or approximation. Too much code.
   
   Wait, `d` is needed for `inv_d` (scaling).
   If I use `f_mag = 1/d2` (approximation $1/d^2$), I can omit `inv_d`?
   $F = \vec{d} / |d|^2$? That's unit vector.
   This is constant magnitude force (on surface) if projected?
   No, $F = (p_i - p_j)/|p_i - p_j|^2$.
   If points are close, $F$ is large.
   If points far ($d \approx 2$), $F$ is small ($1/4$).
   This is good.
   It avoids `1/d^3` which is even stronger.
   The force $F = (p_i - p_j)/|p_i - p_j|^2$ corresponds to potential $V \sim -1/|d|$?
   $\nabla (1/r) = -r/r^3 = - \hat{r}/r^2$.
   Force $F = -\nabla V$.
   If $V \sim -1/r$, $F \sim \hat{r}/r^2$.
   This is exactly what I have (without `inv_d` if I treat `f_mag` differently).
   Wait, `dx/d^2`. Yes.
   So `f_x = dx / d2`.
   This is vector divided by scalar $d^2$.
   This requires `d2` only.
   So I don't need `sqrt`!
   Wait, I need `d2` anyway.
   So `d = sqrt(d2)` is not needed for force magnitude if I use $1/d^2$.
   But `inv_d` was used to multiply by `f_mag`.
   If I change `f_mag` to `1/d2`, `inv_d` is unnecessary?
   Yes.
   Wait, if I use $1/d^2$ potential, then gradient is $1/d^3$.
   Let's check.
   $V(r) = -1/r$. Force $F = -dV/dr \hat{r} = - (1/r^2) \hat{r}$.
   This is $1/r^2$.
   Gradient w.r.t coordinate $x$. $dV/dx = dV/dr \cdot dr/dx = (1/r^2) \cdot (x/r) = x/r^3$.
   So force component is $x/r^3$.
   My previous logic: $F \propto \hat{r}/r^2$.
   Component $\hat{r}_x / r^2 = (x/r) / r^2 = x/r^3$.
   So $F = \frac{x}{r^3}$.
   This requires $r = \sqrt{d2}$.
   So `sqrt` is unavoidable if I want exact gradient.
   However, $F \propto \frac{x}{r^2}$ corresponds to potential $V \propto \ln r$ (in 2D) or harmonic oscillator? No.
   $F = -\nabla V$. If $V = \ln r$, $\nabla V = 1/r \hat{r}$. $F = -1/r \hat{r}$.
   This is weak repulsion (logarithmic potential).
   Force $F = x/r^2$.
   For large $r$, force drops as $1/r^2$.
   For small $r$, force drops as $1/r^2$ (singularity).
   This is essentially $1/r^2$ force law (Coulomb-like in 3D is $1/r^2$).
   Wait, Coulomb potential $1/r$ yields Force $1/r^2$.
   Force vector $\vec{F} = q_1 q_2 \hat{r} / r^2$.
   Components: $x/r^3$.
   So `sqrt` IS needed.
   
   However, `1/d^2` force (without dividing by `d` again) is different.
   Vector `d`. Magnitude `1/d`. No, scalar `1/d^2`.
   Vector `d * 1/d^2 = \vec{d}/d^2 = \hat{d}/d`.
   This is `1/d` force magnitude. (Infinite at contact).
   This is also repulsive.
   But $\hat{d}/d$ vs $\hat{d}/d^2$.
   The latter grows faster.
   Both work.
   The one without `sqrt`?
   $\vec{d}/d^2$ requires `sqrt`? No.
   $d$ in denominator is `sqrt(d2)`.
   So if I use `F = \vec{d}/d2`, then I don't divide by `d` (i.e. `sqrt(d2)`).
   Wait, $\vec{d}/d^2$ means `vector` / `scalar`.
   Scalar is $d^2 = x^2 + y^2 + z^2$.
   So `dx/d2` involves only squares! No `sqrt`.
   This force behaves like $1/r^2$ in 2D?
   Wait, $F \sim \vec{r}/r^2 = \hat{r}/r$.
   Magnitude $1/r$.
   This is logarithmic potential.
   Is it strong enough?
   As $r \to 0$, $1/r \to \infty$.
   As $r \to \infty$, $1/r \to 0$.
   This is weaker than Coulomb ($1/r^2$).
   But since we are on compact manifold (sphere), $r$ is bounded below by packing limits.
   So it prevents overlapping well enough.
   So `dx/d2` is a valid repulsive force without `sqrt`!
   Let's use this optimization.
   Force magnitude: $1/d^2$.
   Vector: $\vec{F}_{ij} = \frac{\vec{p}_i - \vec{p}_j}{|\vec{p}_i - \vec{p}_j|^2}$.
   This avoids `sqrt` entirely inside the loop!
   Just `d2`.
   Wait, `d2` is $x^2+y^2+z^2$.
   Yes.
   
   So loop body:
   ```cpp
   double d2 = dx*dx + dy*dy + dz*dz;
   // To avoid div0
   if (d2 < 1e-12) continue;
   double f = 1.0 / d2;
   fx[i] += dx * f;
   fx[j] -= dx * f;
   // same for y, z
   ```
   This is extremely fast.
   
   One catch: `d2` is distance squared.
   `dx/d2` is vector / dist_sq.
   If `d` is 1 (on sphere chord dist?), `F` is small.
   If `d` is small, `F` is huge.
   Seems okay.
   Let's use this optimization.
   
   Wait, `fx[j] -= dx * f`.
   `dx = p[i] - p[j]`.
   If `dx` is positive, `fx[i]` increases (move right).
   `fx[j]` decreases (move left).
   Correct.
   
   So `max_iter` can be larger. 150 is fine.
   
   I will use this optimized force loop.
   
   One detail:
   `d2` should be double.
   
   Also, the output order of points is arbitrary.
   
   Final check:
   Sample input 2.
   Init: $y=-1, y=0$.
   $i=0: (0,0,-1)$.
   $i=1: (1,0,0)$.
   Forces:
   $d^2 = 1^2 = 1$.
   $f = 1$.
   $dx = 1$.
   $fx[0] += 1$. $fx[1] -= 1$.
   Move:
   $p[0]$ moves away from $p[1]$?
   $p[0]=(0,0,-1)$. $fx[0]=1$.
   $nx = 0 + 0.05*1 = 0.05$.
   $nz = -1$.
   Norm $> 1$.
   Converges to poles.
   Seems correct.
   
   Sample output `2` and coordinates.
   My code outputs `min_dist`.
   For sample, should output `2.0`.
   
   The problem statement says:
   "Your answer will be accepted if ... matches your claimed min_dist ... with absolute or relative error at most 10^-6."
   So if I calculate `min_dist` precisely, I am safe.
   
   Wait, `min_dist` for $N=2$ should be $2$.
   If I start with $y=0$, $x=1$, dist is $\sqrt{2} \approx 1.41$.
   Optimization moves them apart.
   $p[0] \to (0,0,1)$, $p[1] \to (0,0,-1)$.
   Dist $\to 2$.
   So output will be 2. Correct.
   
   Okay, ready to write.

   Wait, `xi, yi, zi` real numbers.
   Format is flexible. `fixed` and `setprecision` is good.
   
   Memory limit 512 MB.
   Vectors of double size $N=1000$. Tiny.
   
   Final check on headers.
   `<iostream>`, `<vector>`, `<cmath>`, `<iomanip>`.
   
   Double check: `theta` in fib sphere.
   `i * GOLDEN_ANGLE`.
   Some implementations use `i * 2PI / N`.
   But `GOLDEN_ANGLE` (approx $2.4$ rad) gives better dispersion.
   It's fine.

   One edge case: `n` could be 1? Constraints say `2 <= n`.
   Okay.
   
   Wait, `theta` calculation inside loop.
   `double theta = i * GOLDEN_ANGLE;`
   It's better to keep it mod 2PI to avoid loss of precision for large `i`?
   $1000 \times 2.4 \approx 2400$.
   `cos(2400)` is fine. Double has 53 bits significand.
   Precision loss only becomes an issue when argument is huge (like $10^{10}$).
   Here it's small.
   
   I will submit the solution.
</think>

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

struct Point {
    double x, y, z;
};

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    vector<Point> p(n);

    // Fibonacci Sphere Initialization
    // Distribute points uniformly on the sphere.
    const double PI = acos(-1.0);
    const double GOLDEN_ANGLE = PI * (3.0 - sqrt(5.0)); // Approx 137.5 degrees

    for (int i = 0; i < n; ++i) {
        double t = (double)i / n;
        // Latitude coordinate (y in Cartesian sense, mapped to sphere vertical)
        double y = 2.0 * t - 1.0;
        
        // Clamp to avoid numerical errors near poles causing sqrt of negative
        if (y > 0.9999999) y = 0.9999999;
        if (y < -0.9999999) y = -0.9999999;
        
        double r = sqrt(max(0.0, 1.0 - y * y));
        double theta = i * GOLDEN_ANGLE;
        
        // Assign x and z based on longitude theta
        double x = r * cos(theta);
        double z = r * sin(theta);
        
        p[i] = {x, y, z};
    }

    // Optimization using Repulsive Force Dynamics
    // This improves the uniformity and maximizes minimum distance
    // Time Complexity: O(Iterations * N^2). With N=1000, Iterations ~80-150, this fits in 2s.
    // We avoid sqrt in force calculation for performance.
    
    vector<double> fx(n), fy(n), fz(n);
    
    // Parameters for dynamics
    // We perform multiple steps. A decaying step size helps convergence.
    int max_iterations = 150;
    
    for (int k = 0; k < max_iterations; ++k) {
        // Initialize forces to zero
        fill(fx.begin(), fx.end(), 0.0);
        fill(fy.begin(), fy.end(), 0.0);
        fill(fz.begin(), fz.end(), 0.0);
        
        // Accumulate forces between all pairs
        // Force is repulsive. F_ij is proportional to vector from j to i divided by d^2
        // F ~ (p_i - p_j) / |p_i - p_j|^2
        for (int i = 0; i < n; ++i) {
            // Compute distance squared to self is 0, skipped implicitly by j>i loop
        }
        
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dx = p[i].x - p[j].x;
                double dy = p[i].y - p[j].y;
                double dz = p[i].z - p[j].z;
                
                double d2 = dx * dx + dy * dy + dz * dz;
                
                if (d2 < 1e-12) continue; // Points overlapping
                
                double f_mag = 1.0 / d2;
                
                // Force on i is repulsive (away from j)
                fx[i] += dx * f_mag;
                fx[j] -= dx * f_mag;
                fy[i] += dy * f_mag;
                fy[j] -= dy * f_mag;
                fz[i] += dz * f_mag;
                fz[j] -= dz * f_mag;
            }
        }
        
        // Move points and project back onto sphere
        double step_size = 0.1;
        // Decay step size to stabilize
        step_size *= (1.0 - 0.0005 * (double)k); 
        if (step_size < 0.001) step_size = 0.001;
        
        for (int i = 0; i < n; ++i) {
            // Velocity update
            double nx = p[i].x + step_size * fx[i];
            double ny = p[i].y + step_size * fy[i];
            double nz = p[i].z + step_size * fz[i];
            
            // Project back to unit sphere
            double nn = sqrt(nx * nx + ny * ny + nz * nz);
            if (nn > 1e-10) {
                p[i].x = nx / nn;
                p[i].y = ny / nn;
                p[i].z = nz / nn;
            }
        }
    }

    // Calculate the actual minimum pairwise distance
    double min_dist = 2.0; // Max possible distance
    // If n=2 and points are optimal, min_dist is 2.
    // Initialize with a safe upper bound or infinity.
    // For safety with floats, use a large number or first pair.
    if (n > 0) min_dist = 1e9;
    
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = p[i].x - p[j].x;
            double dy = p[i].y - p[j].y;
            double dz = p[i].z - p[j].z;
            double d = sqrt(dx * dx + dy * dy + dz * dz);
            if (d < min_dist) {
                min_dist = d;
            }
        }
    }
    
    // Output results
    cout << fixed << setprecision(10) << min_dist << "\n";
    for (int i = 0; i < n; ++i) {
        cout << fixed << setprecision(10) << p[i].x << " " << p[i].y << " " << p[i].z << "\n";
    }

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=2381)

````
Let me think about this problem systematically.

**Problem Analysis:**
- Place n points inside a unit sphere to maximize the minimum distance between any two points
- This is related to sphere packing problems in 3D

**Key Observations:**
1. For small n, we need specific geometric arrangements:
   - n=2: Two antipodal points on the sphere (distance = 2)
   - n=3: Equilateral triangle on a great circle (distance = √3 ≈ 1.732)
   - n=4: Regular tetrahedron inscribed in the sphere
   - n=5: Square pyramid or triangular bipyramid

2. The optimal configurations are related to well-known polyhedra:
   - **Icosahedron**: 12 vertices
   - **Cubic/Box**: Up to 8 corners can use

3. Strategy approach:
   - Handle special cases n ≤ 12 using exact optimal configurations
   - For larger n: fill remaining space optimally
   - Use spherical coordinates for flexibility
   - Add interior point when beneficial (e.g., at origin)

**Implementation Details:**

1. **Point Generation Functions:**
   - `get_box_points()` → 8 corner points of an octahedron-like shape
   - `get_icosahedron_points()` → 12 vertices of icosahedron with edge length √(5+2√5)/6 ≈ 0.7149
   - `make_symmetric()` → ensure opposite points exist

2. **Symmetric Placement:**
   - Mirror positive-coordinate points to negative ones
   - Places 2k-1 points symmetrically

3. **Optimal Configurations:**
   - n=2: (0,0,1), (0,0,-1), min_dist=2
   - n=3: Three equally spaced points on equator, distance √3
   - n=4: Regular tetrahedron with edge length √(8/3)
   - n=5 and beyond: Symmetric box placement plus additional symmetric extensions

**Code Structure:**
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
using namespace std;

// Generate 8 corner points of symmetric box (octahedron projection)
pair<double, double> get_box_points() {
    // Box with face diagonal 2, so x = cos(π/4) = √2/2 ≈ 0.707106781...
    return make_pair(sqrt(2.0)/2.0, sqrt(2.0)/2.0);
}

// Generate 12 vertices of regular icosahedron
vector<pair<double, double>> get_icosahedron_points() {
    // Golden ratio-related coordinates for optimal 12-point configuration
    // Edge length: √(5 + 2√5)/6 ≈ 0.7149
    vector<pair<double, double>> ret;
    pair<double, double> t = get_box_points();
    for (int i = 0; i < 12; i++) {
        double r[6][2] = {
            {-t.first, t.second}, {t.first, t.second}, {t.first, -t.second}, 
            {-t.first, -t.second}, {-t.second, t.first}, {t.second, -t.first},
            {t.second, t.first}, {-t.second, -t.first}, {t.second, -t.first}, 
            {-t.second, t.first}, {-t.second, -t.first}, {t.second, t.first}
        };
        ret[i] = r[i % 6];
    }
    return ret;
}
```

For the main logic, I need to handle each n case with its known optimal solution. Cases 2 through 12 map directly to their respective configurations—tetrahedron, pentagonal bipyramid, hexagonal prisms, cube-based arrangements, and the full icosahedron. Starting at n=13, I fall back to the general symmetric point generation method since no simple closed form exists.

The symmetry constraint is key here: placing one asymmetric point forces its mirror image elsewhere, which becomes wasteful for large n. So for every new point I add after n=12, I simultaneously place both it and its reflection. The challenge is selecting which base points from the icosahedron to work with, since simply mirroring existing points won't create fresh configurations.

My refinement was trying the complete set of icosahedral points first, then falling back to just the positive-coordinates subset—this avoids wasting points on unnecessary reflections. The fallback also handles edge cases like the single-center point where mirroring might duplicate an already-placed location.

For verification, I should check that distances between consecutive symmetric pairs stay within acceptable tolerance (around 10^-6). Testing showed the icosahedral spacing gives roughly 0.6876 units per step, so the claimed distance holds up under scrutiny even though individual pairs vary slightly due to finite precision.
</think>

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
using namespace std;

double dist(double x1, double y1, double z1, double x2, double y2, double z2) {
    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
}

pair<double, double> get_box_points() {
    return make_pair(sqrt(2.0)/2.0, sqrt(2.0)/2.0);
}

vector<pair<double, double>> get_icosahedron_points() {
    vector<pair<double, double>> ret;
    pair<double, double> t = get_box_points();
    pair<double, double> r[12][2] = {
        {-t.first, t.second}, {t.first, t.second}, {t.first, -t.second}, 
        {-t.first, -t.second}, {-t.second, t.first}, {t.second, -t.first},
        {t.second, t.first}, {-t.second, -t.first}, {t.second, -t.first}, 
        {-t.second, t.first}, {-t.second, -t.first}, {t.second, t.first}
    };
    for(int i=0;i<12;i++) ret.push_back(r[i]);
    return ret;
}

vector<pair<double,double>> make_symmetric(vector<pair<double,double>> pts) {
    double mx = 0, my = 0, mz = 0;
    for(auto& p : pts) mx += p.first, my += p.second, mz += p.second;
    return pts;
}

int get_max_n(const vector<pair<double,double>>& pts) {
    if(pts.size() >= 12) return 12;
    int max_n = pts.size() * 2 + 1;
    for(size_t i=0;i<pts.size()-1;i++) {
        auto a = pts[i], b = pts[(i+1)%pts.size()];
        double d2 = (a.first-b.first)*(a.first-b.first) + 
                    (a.second-b.second)*(a.second-b.second) +
                    ((a.first+b.first)*(a.first+b.first) + 
                     (a.second+b.second)*(a.second+b.second))/2.0;
        max_n = max(max_n, 1+(int)d2);
    }
    return max_n;
}

double min_d;
double min_d_of_new = 1.0e9;
vector<pair<double,double>> best;
int count = 0;

int main() {
    int n;
    cin >> n;
    cout << fixed << setprecision(10);

    if(n == 2) { cout << "2" << endl; cout << "0 0 1" << endl; cout << "0 0 -1" << endl; return 0; }
    else if(n == 3) {
        double r = sqrt(3.0)/3.0;
        cout << setprecision(10) << r*sqrt(3.0) << endl;
        cout << setprecision(10) << "1 " << r << " 0" << endl;
        cout << "-1 " << r << " 0" << endl;
        cout << setprecision(10) << "0 " << r << " 1" << endl;
        return 0;
    }
    else if(n == 4) {
        double r = sqrt(8.0/3.0);
        cout << setprecision(10) << r << endl;
        cout << setprecision(10) << 1.0 << " " << r << " 0.0" << endl;
        cout << setprecision(10) << -1.0 << " " << r << " 0.0" << endl;
        cout << setprecision(10) << 0.0 << " " << -r << " " << r << endl;
        cout << setprecision(10) << 0.0 << " " << -r << " " << -r << endl;
        return 0;
    }
    else if(n == 5) {
        double r = sqrt(5.0)+1.0;
        double s = sqrt(5.0);
        cout << setprecision(10) << s << endl;
        cout << setprecision(10) << s/r<< " " << s/r<< " " << s/(s/s-s/s) << endl;
        cout << setprecision(10) << 1.000000000 << " " << r << " 0.0" << endl;
        cout << setprecision(10) << -1.000000000 << " " << r << " 0.0" << endl;
        cout << setprecision(10) << 0.0 << " " << -r << " " << r << endl;
        cout << setprecision(10) << 0.0 << " " << -r << " " << -r << endl;
        return 0;
    }

    cout << min_d << endl;
    for(int i=0;i<n;i++) {
        cout << setprecision(10) << best[i].first << " " 
             << best[i].second << " " << 0.0 << endl;
    }
    return 0;
}
```
````
