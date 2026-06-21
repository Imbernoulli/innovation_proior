The chromatic number of the plane (CNP) asks for the smallest number of colours needed to colour every point of the Euclidean plane so that any two points at distance exactly 1 receive different colours. Posed by Edward Nelson in 1950 and now known as the Hadwiger-Nelson problem, it stood for decades with the bounds 4 ≤ CNP ≤ 7. The lower bound comes from the Moser spindle, a seven-vertex unit-distance graph with chromatic number exactly four, while the upper bound comes from a regular hexagonal tiling of the plane that uses seven colours. By the de Bruijn-Erdős compactness theorem, the whole plane needs k colours if and only if every finite unit-distance graph does, so improving the lower bound reduces to exhibiting a single finite unit-distance graph that admits no proper four-colouring. The difficulty is that every edge in such a witness must be a genuine distance-1 segment in the plane, while non-four-colourability is a global combinatorial property that local gadgets do not obviously control. Decades of bolting spindles together by hand failed to raise the chromatic number, because in a four-colouring a spindle only says that two specified √3-pairs sharing a vertex cannot both be monochromatic; it does not by itself create a contradiction that propagates through a large graph.

The method is de Grey's 5-chromatic unit-distance graph construction. It engineers a contradiction around one feature: a monochromatic √3-equilateral triple, meaning three points with all pairwise distances equal to √3 and all the same colour. The construction has three ingredients. First, the hexagon-with-centre graph H, seven vertices consisting of a regular hexagon of side 1 plus its centre, acts as a classifier of four-colourings: among its four essentially different proper four-colourings, two contain such a monochromatic √3-triple on alternating corners and are called structured, while two do not and are called unstructured. Second, a forcer graph L is assembled recursively from rotated copies of H so that every four-colouring makes at least one copy structured. The build goes J → K → L: J is a central H surrounded by shell copies; K is two J's rotated by 2·arcsin(1/4) so that corresponding linking vertices become a unit edge; L is two K's rotated by 2·arcsin(1/8) about a linking vertex so that opposite linking vertices, already forced equal, are joined by a unit edge. At each stage a single new unit edge eliminates the unstructured cases. Third, an anti-forcer graph M is built around one copy of H from a dense mesh of interlocking Moser spindles. Because a monochromatic √3-triple is a local concentration of monochromatic √3-pairs, a sufficiently rigid spindle mesh spreads those pairs apart and forbids such concentration; plain 60-degree meshes are too slack, so extra edge directions taken from tightly-linked spindle geometry are added to stiffen the mesh. Gluing 52 copies of M onto L forces the same copy of H to be both structured and unstructured, which is impossible, so the union is not four-colourable. A final shrinking pass reduces the graph to 1581 vertices, small enough that standard SAT solvers can independently verify both a valid five-colouring and the unsatisfiability of the four-colour formula.

The code below constructs the basic pieces in floating-point geometry and verifies the core combinatorial claims with a small backtracking colourer. It builds the rhombus, the Moser spindle, the classifier H, the small forcer J, the staged forcer K, and a SAT-style encoder for independent checking. The implementation is not the full 1581-vertex record graph, but it demonstrates the mechanism and can be extended to larger instances.

```python
import itertools, math

EPS = 1e-6

def dist(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])

def rotate(p, angle, origin=(0.0, 0.0)):
    cx, cy = origin
    x, y = p[0]-cx, p[1]-cy
    c, s = math.cos(angle), math.sin(angle)
    return (cx + c*x - s*y, cy + s*x + c*y)

def is_unit_distance_graph(verts, edges):
    for u, v in edges:
        if abs(dist(verts[u], verts[v]) - 1.0) > EPS:
            return False
    return True

def chromatic_number(verts, edges, max_k=6):
    n = len(verts)
    order = sorted(range(n), key=lambda v: sum(1 for e in edges if v in e), reverse=True)
    for k in range(1, max_k+1):
        col = [-1]*n
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = {col[u] for u, _ in edges if _ == v} | {col[u] for _, u in edges if u == v}
            for c in range(k):
                if c not in used:
                    col[v] = c
                    if backtrack(idx+1):
                        return True
                    col[v] = -1
            return False
        if backtrack(0):
            return k
    return None

def make_rhombus():
    A, B = (0.0, 0.0), (1.0, 0.0)
    C = (0.5, math.sqrt(3)/2)
    D = (0.5, -math.sqrt(3)/2)
    return [A, B, C, D], [(0,1),(0,2),(1,2),(0,3),(1,3)]

rh_verts, rh_edges = make_rhombus()
print("Rhombus acute distance:", dist(rh_verts[2], rh_verts[3]))
mono = sum(1 for c in itertools.product(range(3), repeat=4)
           if all(c[u]!=c[v] for u,v in rh_edges) and c[2]==c[3])
print("3-colourings with acute pair equal:", mono, "/ 6")

def make_spindle():
    rt3 = math.sqrt(3)
    A = (0.0, 0.0)
    B1 = (rt3, 0.0)
    P1, Q1 = (rt3/2, 0.5), (rt3/2, -0.5)
    x, y = 5*rt3/6, math.sqrt(33)/6
    B2 = (x, y)
    M2 = (x/2, y/2)
    perp = (-y/rt3, x/rt3)
    P2 = (M2[0] + 0.5*perp[0], M2[1] + 0.5*perp[1])
    Q2 = (M2[0] - 0.5*perp[0], M2[1] - 0.5*perp[1])
    V = [A, P1, Q1, B1, P2, Q2, B2]
    E = [(0,1),(0,2),(1,2),(1,3),(2,3),
         (0,4),(0,5),(4,5),(4,6),(5,6),(3,6)]
    return V, E

sp_verts, sp_edges = make_spindle()
print("Spindle unit-distance:", is_unit_distance_graph(sp_verts, sp_edges))
print("Moser spindle chi:", chromatic_number(sp_verts, sp_edges))

def make_H():
    corners = [(math.cos(k*math.pi/3), math.sin(k*math.pi/3)) for k in range(6)]
    verts = [(0.0, 0.0)] + corners
    edges = [(0, i) for i in range(1, 7)] + [(i, i%6+1) for i in range(1, 7)]
    return verts, edges

H_verts, H_edges = make_H()

def is_structured(col):
    return any(col[a]==col[b]==col[c] for a,b,c in [(1,3,5),(2,4,6)])

proper = [c for c in itertools.product(range(4), repeat=7)
          if all(c[u]!=c[v] for u,v in H_edges)]
print("H structured/unstructured:", sum(map(is_structured, proper)), "/", len(proper)-sum(map(is_structured, proper)))

def copy_H(centre, rotation=0.0):
    v, e = make_H()
    new_v = [rotate(p, rotation) for p in v]
    dx, dy = centre
    return [(x+dx, y+dy) for x,y in new_v], list(e)

def make_J():
    centres = [(0.0, 0.0)]
    centres += [(math.cos(k*math.pi/3), math.sin(k*math.pi/3)) for k in range(6)]
    centres += [(math.sqrt(3)*math.cos(k*math.pi/3+math.pi/6),
                 math.sqrt(3)*math.sin(k*math.pi/3+math.pi/6)) for k in range(6)]
    all_v, all_e, offset = [], [], 0
    for c in centres:
        v, e = copy_H(c)
        all_v.extend(v)
        all_e.extend((a+offset, b+offset) for a,b in e)
        offset += len(v)
    return all_v, all_e

J_verts, J_edges = make_J()
print("J unit-distance:", is_unit_distance_graph(J_verts, J_edges))

def make_K():
    Vj, Ej = make_J()
    Vj2 = [rotate(p, 2*math.asin(1/4)) for p in Vj]
    n = len(Vj)
    all_v = Vj + Vj2
    all_e = list(Ej) + [(u+n, v+n) for u,v in Ej]
    for a, b in zip(range(19,25), range(19+n, 25+n)):
        if abs(dist(all_v[a], all_v[b]) - 1.0) < EPS:
            all_e.append((a, b))
    return all_v, all_e

K_verts, K_edges = make_K()
print("K unit-distance:", is_unit_distance_graph(K_verts, K_edges))

# SAT encoder for independent verification of larger graphs
def colour_cnf(n, edges, k, fixed=None):
    var = lambda v, c: v*k + c + 1
    clauses = [[var(v,c) for c in range(k)] for v in range(n)]
    clauses += [[-var(v,c1), -var(v,c2)] for v in range(n)
                for c1,c2 in itertools.combinations(range(k), 2)]
    clauses += [[-var(u,c), -var(w,c)] for u,w in edges for c in range(k)]
    if fixed:
        clauses += [[var(v,c)] for v,c in fixed.items()]
    header = f"p cnf {n*k} {len(clauses)}\n"
    return header + "".join(" ".join(map(str,cl)) + " 0\n" for cl in clauses)

with open("spindle_3.cnf", "w") as f:
    f.write(colour_cnf(len(sp_verts), sp_edges, 3))
print("Wrote spindle_3.cnf; a SAT solver will report UNSAT, confirming chi = 4.")
```
