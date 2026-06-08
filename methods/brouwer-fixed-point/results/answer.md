# Brouwer's fixed-point theorem (via Sperner's lemma)

## The problem it solves

Does every continuous self-map of a "filled, no-hole" compact body have a point that doesn't move? A fixed point is an existence claim about an unknown point of an infinite set, with no formula to solve and only continuity to work with. The theorem certifies the equilibrium exists, under the weakest hypothesis (continuity), and ties the guarantee to the *shape* of the domain.

## Statement

**Brouwer's fixed-point theorem.** Let `K ⊂ ℝⁿ` be nonempty, compact, and convex (e.g. the closed ball `Dⁿ`, the simplex `Δⁿ`, a cube). Every continuous `f : K → K` has a point `x*` with `f(x*) = x*`.

Both compactness and convexity (no hole) are essential: `f(x)=x²` on the open `(0,1)` has no fixed point (not compact); rotation of an annulus has none (not convex / has a hole); the antipodal map of a sphere has none (the sphere is not convex/filled).

## The key idea

Negate the conclusion and reach a topological obstruction. If `f : Dⁿ → Dⁿ` had **no** fixed point, then `x ≠ f(x)` everywhere. Put `d(x)=x−f(x)` and shoot the ray from `f(x)` through `x` to the boundary:

`r(x)=x+t(x)(x−f(x))`, with `t(x)≥0` the least exit time such that `r(x)∈Sⁿ⁻¹`.

The exit time is the continuous nonnegative root

`t(x)=(-x·d(x)+sqrt((x·d(x))²+|d(x)|²(1−|x|²)))/|d(x)|²`.

For `x∈Sⁿ⁻¹`, this gives `t(x)=0` (and for `t>0`, `|x+t(x−f(x))|>1`), so `r|_{Sⁿ⁻¹}=id`: this is a continuous **retraction** of the ball onto its boundary. No such retraction exists — for `n≥2`, the boundary sphere carries a nonzero invariant (`H_{n−1}(Sⁿ⁻¹) ≅ ℤ`) that the contractible ball kills (`H_{n−1}(Dⁿ) = 0`), so `r_*∘i_* = id` would factor `id_ℤ` through `0`; for `n=1`, the same obstruction is reduced `H_0` or connectedness. Sperner's lemma gives the elementary mod-2 version of this obstruction, valid for all `n`.

## Sperner's lemma (the combinatorial core)

**Statement.** Triangulate the `n`-simplex `Δⁿ` and color its vertices with `n+1` colors so that (i) the `n+1` corners get distinct colors, and (ii) a vertex on a face uses only that face's corner-colors (a *Sperner labeling*). Then the number of **rainbow** cells (cells whose `n+1` vertices show all `n+1` colors) is **odd** — in particular `≥ 1`.

**Proof (induction on `n`).**
- `n = 0`: a single colored point — one rainbow cell.
- Inductive step. Count facets (`(n−1)`-faces of cells) colored exactly with the palette `{1,…,n}`. A rainbow cell owns exactly one such facet: drop its unique `0` vertex. A cell using exactly the colors `{1,…,n}` and no `0` has exactly one repeated color and owns exactly two such facets: drop either copy of the repeated color. Every other cell owns none, because after deleting one vertex the remaining facet either still has a `0` or still misses some color in `{1,…,n}`. Cell-by-cell this gives `R + 2C` facets, where `R` = #rainbow cells and `C` = #cells using exactly `{1,…,n}`. Counting the same facets by location — each interior facet shared by two cells, each boundary facet by one — gives `D_O + 2D_I`. Hence
  `R + 2C = D_O + 2D_I  ⇒  R ≡ D_O (mod 2).`
  A `{1,…,n}`-colored facet can lie on the boundary only on the face of `Δⁿ` spanned by the corners `1,…,n`, which is itself a Sperner-labeled `(n−1)`-simplex; its rainbow `(n−1)`-cells are exactly those boundary facets. By induction `D_O` is odd, so `R` is odd. ∎

## Sperner ⇒ Brouwer

Work on `Δⁿ = {x ∈ ℝⁿ⁺¹ : x_i ≥ 0, Σ x_i = 1}`; let `f : Δⁿ → Δⁿ` be continuous and suppose `f(x) ≠ x` for all `x`.

- **Labeling.** Since `Σ x_i = Σ f(x)_i = 1` and `f(x) ≠ x`, some coordinate has `f(x)_i < x_i`. Color `x` by such an `i`. This is a legal Sperner labeling: a corner `e_l` has only coordinate `l` nonzero, so only `l` can decrease (corners get distinct colors); on a face the zero coordinates can't be the decreasing one, so colors stay within the face.
- **Limit.** Take triangulations with mesh `→ 0`. Each has a rainbow cell (Sperner). Choose one vertex from each selected cell; compactness gives a convergent subsequence with limit `x*`, and the mesh bound forces every vertex of those cells to converge to the same `x*`. For every color `i` the cell has a vertex `v_i` with `f(v_i)_i < (v_i)_i`; by continuity, in the limit `f(x*)_i ≤ x*_i` for **all** `i`. Since `Σ f(x*)_i = Σ x*_i = 1`, the inequality is equality in every coordinate: `f(x*) = x*` — contradiction. Hence a fixed point exists. ∎

**Extension.** The fixed-point property is preserved by homeomorphism: if `h : Δᵈ → K` is a homeomorphism and `f : K → K` is continuous, then `h⁻¹∘f∘h` has a fixed point `z`, so `h(z)` is fixed by `f`. For a nonempty compact convex `K ⊂ ℝⁿ`, let `d` be the dimension of its affine hull. If `d=0`, `K` is a point. If `d>0`, choose `p` in the relative interior of `K`; for each unit direction `u`, the radial function `ρ(u)=max{t≥0:p+tu∈K}` is finite, positive, and continuous, and `ru ↦ p+rρ(u)u` maps the closed `d`-ball homeomorphically onto `K`. Since the closed `d`-ball is homeomorphic to `Δᵈ`, the theorem holds for all nonempty compact convex sets. ∎

## Witness code

A small checker that exhibits the core: encode `f` as a Sperner labeling, find forced rainbow cells at each mesh, and track centroid approximants whose convergent subsequences limit to fixed points.

```python
import numpy as np

def label_vertex(x, f, tol=0.0):
    """Sperner color of x (barycentric coords): an index i with f(x)_i < x_i.
    Exists whenever f(x) != x; legal because a zero coordinate can't decrease,
    so corners get distinct colors and face-vertices stay within the face."""
    fx = f(x)
    decreasing = np.flatnonzero(fx < x - tol)
    if len(decreasing) == 0:
        raise ValueError("label undefined at a fixed point or numerical tie")
    return int(decreasing[0])

def fully_labeled_cells(vertices, cells, f):
    """Return rainbow cells in a Sperner-labeled triangulation.
    Sperner's lemma guarantees an odd, hence nonzero, number."""
    labels = [label_vertex(v, f) for v in vertices]
    n_colors = vertices.shape[1]                    # = n+1 for Delta^n
    rainbow = []
    for cell in cells:                              # cell: indices of n+1 verts
        if len({labels[i] for i in cell}) == n_colors:
            rainbow.append(cell)
    return rainbow

def fixed_point_approximants(f, simplex_vertices, refine, n_refinements):
    """Refine (mesh -> 0); extract a rainbow cell each scale.
    Any convergent subsequence of returned centroids has limit x* with
    f(x*)=x*: coordinatewise <= plus equal sums forces equality."""
    vertices, cells = simplex_vertices, [tuple(range(len(simplex_vertices)))]
    centroids = []
    for _ in range(n_refinements):
        vertices, cells = refine(vertices, cells)
        rainbow = fully_labeled_cells(vertices, cells, f)
        if not rainbow:
            raise AssertionError("Sperner violated")  # cannot happen
        cell = rainbow[0]
        centroids.append(np.mean([vertices[i] for i in cell], axis=0))
    return centroids
```
