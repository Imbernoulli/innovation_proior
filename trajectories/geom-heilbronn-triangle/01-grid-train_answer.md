I want eleven points in the unit square whose smallest triangle is as large as possible, and before reaching for any search I want a principled starting configuration — something I can write down in closed form, that scores a real positive number, and that fixes the floor every later rung must clear. The whole difficulty lives in the word *minimum*: the score is the worst triangle among all $C(11,3) = 165$ triples, so a single near-collinear trio sinks the entire configuration to almost nothing. The right question for a baseline is therefore the sharpest version of "spread the points out evenly" — but evenness has to mean the right thing.

The obvious even arrangements are exactly the ones that fail, and they fail for the same instructive reason. A regular grid — say a $4 \times 3$ lattice with one node dropped to leave eleven — feels maximally even, every point at a lattice site, nothing bunched. But a grid is *built out of collinear points*: every row is three or four points on a perfectly straight line, and three collinear points span a triangle of area exactly $0$. So a grid does not merely score low, it scores the worst possible value, $0$, because the minimum is dragged all the way down by the degenerate triples sitting on each grid line. Pushing the eleven points onto the square's perimeter instead does not save it: each edge of the square is itself a straight line, and eleven points around four edges must put two or three points on most edges, so edge-mates plus a corner — or any third collinear point — again form zero-area triangles. Both lattice-like families are "even" in the wrong sense; lattices are made of straight lines, and straight lines are exactly what this objective punishes with a zero.

So what I actually need is a configuration with *no three points collinear anywhere*, and the cleanest such object I know is points on a common **circle**. I propose the inscribed regular eleven-gon: place all eleven points equally spaced on the largest circle that fits inside the square. The defining property is geometric and exact — a line meets a circle in at most two points, so no three distinct points on a circle are ever collinear, and every one of the $165$ triangles has strictly positive area. The degeneracy that zeroed the grid and the boundary ring simply cannot occur here. Concretely I place the points at angles $\theta_k = 2\pi k / 11$ for $k = 0, \dots, 10$ and map each to

$$\big(\tfrac{1}{2} + \tfrac{1}{2}\cos\theta_k,\ \tfrac{1}{2} + \tfrac{1}{2}\sin\theta_k\big),$$

which is the circle of radius $1/2$ centered at $(1/2, 1/2)$. The radius is forced, not tuned: $1/2$ is the largest circle that fits in the unit square, so it spreads the points as widely as the box allows; any smaller circle scales every triangle down proportionally, and any larger one pushes points outside $[0,1]^2$ and violates the constraint. There is no free parameter at all — equal spacing on the largest legal circle — which is what makes this a clean, reproducible floor.

The smallest triangles of this configuration are the *thinnest* ones: three points that are nearly adjacent around the rim, subtending a narrow arc, which makes a slim but strictly positive triangle. That near-adjacent rim triple is what sets the score, and it is a modest positive number rather than zero — exactly the behavior I want from a baseline. I set expectations honestly: a regular eleven-gon spends its symmetry on rotational regularity, not on fattening the worst triangle, so it sits well below the known record $\Delta(11) = 1/27 \approx 0.037$, whose configuration is an irregular, hand-tuned arrangement on the boundary and interior of the square and looks nothing like a circle. I expect roughly half the record. The point of this rung is not to compete but to draw the line under which nothing should fall: a positive, principled, parameter-free configuration with no collinear triples, leaving the entire distance to $1/27$ to be bought by actually moving points — which is where every later rung begins.

```python
import numpy as np

N = 11

def construct():
    # 11 points equally spaced on the inscribed circle (radius 1/2, center (0.5,0.5)).
    # No three points on a circle are ever collinear, so every triangle has positive area;
    # the grid and equally-spaced-boundary alternatives contain collinear triples and score 0.
    ang = np.linspace(0, 2*np.pi, N, endpoint=False)
    return 0.5 + 0.5 * np.column_stack([np.cos(ang), np.sin(ang)])
```
