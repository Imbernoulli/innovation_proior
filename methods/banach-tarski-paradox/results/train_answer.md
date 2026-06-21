The question is whether a solid ball in three-dimensional Euclidean space can be cut into finitely many pieces, moved by rigid motions, and reassembled into two balls congruent to the original. Ordinary volume intuition says this is impossible, but that intuition relies on the pieces being Lebesgue measurable. Once the pieces are allowed to be arbitrary subsets, the measure-theoretic guard disappears and the algebraic structure of the rotation group can be exploited.

Existing ideas fall short in complementary ways. Measurable equidecomposition correctly preserves volume, yet it only rules out measurable pieces. Planar measure intuition blocks paradoxical decompositions in two dimensions, but the three-dimensional rotation group is far more complicated. Hausdorff's sphere result points to unusual behavior on the surface of a sphere, yet it leaves the countable exceptional set and the passage to the full ball unresolved. The free-group paradox gives the right algebraic engine, but by itself it is only a statement about words, not about geometric points. Orbit decomposition almost connects the two, yet it requires selecting representatives from each orbit, which is exactly where the axiom of choice enters. What is needed is a construction that combines all of these ingredients and handles the geometric details.

The method is the Banach-Tarski paradox. It begins by embedding a rank-two free group F = <a,b> into SO(3), the rotation group of the sphere. Such a free subgroup exists; one standard construction uses rotations about two perpendicular axes through an angle whose cosine is 3/5. Every nonidentity rotation fixes exactly two points on the sphere, so the union D of all fixed points of nonidentity elements of F is countable. Removing D from the sphere leaves a set on which F acts freely.

The next step is where ordinary constructive geometry ends. The orbits of F partition the punctured sphere S^2 \ D, and by the axiom of choice we select a set M containing exactly one representative from each orbit. Every point of S^2 \ D then has a unique representation as g m with g in F and m in M. For any subset Q of F, define Q M = { q m : q in Q, m in M }. Because the representation is unique, this operation preserves disjoint unions and transports the algebraic paradox of F to the sphere. The free group can be split into two disjoint blocks P_a and P_b, each of which is equidecomposable with the whole group under left multiplication by a and b respectively. Transporting these blocks through M gives two disjoint subsets of S^2 \ D, each rotation-equidecomposable with S^2 \ D.

The countable exceptional set D is absorbed by a countable shift. Choose a rotation rho around an axis that misses D so that the sets D, rho D, rho^2 D, ... are pairwise disjoint; this is possible because only countably many angles cause collisions among countably many points. The union of this countable chain is congruent to its image under rho, which removes D. Therefore S^2 is rotation-equidecomposable with S^2 \ D. Putting S_1 = P_a M union D and S_2 = P_b M partitions the whole sphere into two subsets, each equidecomposable with the entire sphere.

To pass from the sphere to the ball, extend radially. For E subset S^2, define the cone C(E) = { r x : 0 < r <= 1, x in E }. Rotations commute with radial scaling, so the sphere equidecomposition lifts to an equidecomposition of the punctured ball B^3 \ {0}. The missing center is absorbed by the same countable-shift trick along a ray, so the full ball is equidecomposable with the punctured ball. Thus the original ball can be partitioned into finitely many arbitrary subsets that, after rigid motions, form two balls congruent to the original.

The construction forces nonmeasurability. If a finitely additive volume defined on all subsets of the ball were invariant under rigid motions and agreed with ordinary volume, then each of the two equidecomposable copies would have the original volume V, while their disjoint union is the original ball. Finite additivity would give V = 2V, which is impossible. Hence the pieces cannot all be measurable, and the paradox shows that volume cannot be extended to all subsets while preserving invariance and finite additivity.

```python
"""
Symbolic illustration of the Banach-Tarski paradox.
The full construction is non-constructive; this script demonstrates the
algebraic core and checks the geometric lifting on sample points.
"""
import numpy as np

# ---------- free group on two generators ----------
INVERSE = {'a': 'A', 'A': 'a', 'b': 'B', 'B': 'b'}

def reduce_word(w):
    out = []
    for c in w:
        if out and out[-1] == INVERSE[c]:
            out.pop()
        else:
            out.append(c)
    return ''.join(out)

def mul(u, v):
    return reduce_word(u + v)

def generate_words(max_len):
    """All reduced words of length <= max_len over a, A, b, B."""
    words = ['']
    frontier = ['']
    for _ in range(max_len):
        nxt = []
        for w in frontier:
            for c in 'aAbB':
                if not (w and w[-1] == INVERSE[c]):
                    nxt.append(w + c)
        frontier = nxt
        words.extend(frontier)
    return words

# ---------- algebraic paradox of F_2 ----------
MAX_LEN = 4
words = generate_words(MAX_LEN)
A_pos = {w for w in words if w.startswith('a')}
A_neg = {w for w in words if w.startswith('A')}
B_pos = {w for w in words if w.startswith('b')}
B_neg = {w for w in words if w.startswith('B')}

# Inside a smaller word ball the shift identities hold exactly.
sub = {w for w in words if len(w) <= MAX_LEN - 1}
a_shift = {mul('a', w) for w in A_neg} & sub
b_shift = {mul('b', w) for w in B_neg} & sub
print("A+ ∪ a A- covers truncated F2:", (A_pos & sub) | a_shift == sub)
print("B+ ∪ b B- covers truncated F2:", (B_pos & sub) | b_shift == sub)
print("A+ disjoint from a A-:", (A_pos & sub).isdisjoint(a_shift))
print("B+ disjoint from b B-:", (B_pos & sub).isdisjoint(b_shift))

# ---------- explicit free subgroup of SO(3) ----------
def rotation_matrix(axis, theta):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

c, s = 3.0 / 5.0, 4.0 / 5.0
A = rotation_matrix([1, 0, 0], np.arctan2(s, c))
B = rotation_matrix([0, 1, 0], np.arctan2(s, c))

def apply_word(w, x):
    v = x.copy()
    for c in reversed(w):
        if c == 'a':
            v = A @ v
        elif c == 'A':
            v = A.T @ v
        elif c == 'b':
            v = B @ v
        elif c == 'B':
            v = B.T @ v
    return v

# Distinct reduced words move a generic point to distinct places.
np.random.seed(0)
x = np.random.randn(3)
x /= np.linalg.norm(x)
seen = {}
free = True
for w in words:
    y = apply_word(w, x)
    key = tuple(np.round(y, 8))
    if key in seen:
        free = False
        print("Collision:", w, seen[key])
        break
    seen[key] = w
print("Freeness on generic point:", free)

# ---------- orbit representatives and transport ----------
def orbit_representatives(points, words, eps=1e-6):
    """Greedy choice set M for a finite sample; stands in for the axiom of choice."""
    def key(v):
        return tuple(np.round(v / eps).astype(int))
    index = {key(points[i]): i for i in range(len(points))}
    used = set()
    reps = []
    for i in range(len(points)):
        if i in used:
            continue
        reps.append(i)
        for w in words:
            y = apply_word(w, points[i])
            k = key(y)
            if k in index:
                used.add(index[k])
    return reps

pts = np.random.randn(100, 3)
pts /= np.linalg.norm(pts, axis=1, keepdims=True)
reps = orbit_representatives(pts, words)

# Transport the algebraic pieces P_a = A+ ∪ A- and P_b = B+ ∪ B- to the sphere.
P_a = A_pos | A_neg
P_b = B_pos | B_neg
P_a_pts = np.array([apply_word(w, pts[r]) for r in reps for w in P_a])
P_b_pts = np.array([apply_word(w, pts[r]) for r in reps for w in P_b])
print("Transported piece sizes:", len(P_a_pts), len(P_b_pts))

# ---------- countable exceptional set absorption ----------
def absorbing_rotation(countable_set, axis):
    """Find a rotation rho so that rho^k(D) are disjoint on the sample."""
    for theta in np.linspace(0.01, np.pi, 200):
        rho = rotation_matrix(axis, theta)
        images = []
        for p in countable_set:
            v = p.copy()
            for _ in range(5):
                images.append(tuple(np.round(v, 6)))
                v = rho @ v
        if len(images) == len(set(images)):
            return rho, theta
    return None, None

D = np.array([apply_word(w, pts[0]) for w in words[:8]])
rho, theta = absorbing_rotation(D, [0, 0, 1])
print("Absorbing rotation angle:", theta)
```
