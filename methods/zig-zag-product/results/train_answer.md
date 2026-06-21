We know from random-graph arguments that constant-degree expanders are abundant: a random bounded-degree regular graph mixes well and has its largest nontrivial normalized eigenvalue $\lambda$ bounded safely below $1$ with high probability. But knowing such graphs exist is not the same as being able to name one, walk it locally, and prove its expansion without sampling through an enormous random space. What we actually need is a deterministic neighbor rule, a degree that stays a fixed constant as the vertex count grows without bound, and a proof that $\lambda \le c < 1$ at every size. The existence proof is global and nonconstructive; it gives no recursive, locally addressable, step-by-step-verifiable construction.

The two obvious operations each solve half of this and break the other half. Graph powering takes a two-step walk, which squares the nontrivial eigenvalues — if $G$ has eigenvalue bound $\lambda$, then $G^2$ has roughly $\lambda^2$, a genuine improvement in the spectral gap — but it also squares the degree, so iterating it produces strong expansion at the cost of a graph that is no longer sparse. Powering is therefore an expansion amplifier that creates a degree problem. The opposite move, replacing each high-degree vertex by a small cloud of local edges, lowers the degree but is not enough on its own: the large graph's outgoing edge is selected by an edge label, and if a walker enters a cloud carrying a biased label distribution, the next global step inherits that bias even when the big graph itself is a good expander. Dense or complete graphs mix instantly but have degree growing with $N$, so they cannot serve as a constant-degree primitive. And algebraic Cayley/Ramanujan constructions give excellent explicit families but rely on heavy machinery and expose no simple recursive mechanism that turns one fixed small expander into arbitrarily large ones. The missing ingredient is to make the edge label itself expand.

I propose the zig-zag product. Rather than displaying a single large random-looking graph all at once, it decomposes expansion into two composable sources: a large graph that supplies the global skeleton and a small graph that supplies local mixing inside each vertex-cloud. Concretely, take a $D$-regular graph $G$ on $N$ vertices and a $d$-regular graph $H$ whose number of vertices equals exactly $D$, the degree of $G$. The product $G \,\textcircled{z}\, H$ replaces each vertex of $G$ by a cloud that is a copy of $H$, so a vertex of the product is a pair $(v,a)$ where $v$ is a vertex of $G$ and $a \in [D]$ is simultaneously a position inside the cloud and an edge label for $G$. The resulting graph has $N\cdot D$ vertices and degree only $d^2$. A single step does three moves in sequence: a small step in $H$ to randomize the label, one large step in $G$ using that randomized label, and a second small step in $H$ at the destination cloud. The two local moves are not decoration. The first zig prepares the global step by spreading the label distribution before the global move; the second zag dissipates the structure the global step leaves behind, repairing the local distribution after arrival.

What makes it work is that expansion splits cleanly into a local job and a global job. The large graph is responsible for transporting mass between far-apart regions; the small graph is responsible for ensuring that, inside each cloud, the mass is not concentrated on a few labels. If a distribution's entropy lives in which cloud it occupies, $G$ helps; if its entropy is missing inside the cloud, $H$ helps. The product succeeds because these two corrections interfere constructively rather than compete, so the composed graph inherits roughly the size of $G$, the low degree controlled by $H$, and a combined expansion guarantee from both. This is why low degree is not merely a parameter constraint: a low-degree graph has only a few choices per step and so cannot hide behind dense connectivity, and zig-zag keeps that number of choices small by outsourcing the *quality* of each limited choice to the well-understood small graph. The small graph does not make the network large; it makes every limited choice act as if it were better randomized.

The reason this resolves the explicit-construction problem is that it converts the one-shot existence phenomenon into a stable, recursive amplifier. I do not need to discover a fresh large random-looking graph at every size. I hardwire a single fixed constant-size expander $H$, found once, and then iterate a deterministic recipe: square the current graph to amplify global expansion — which temporarily raises the degree — and then take the zig-zag product with $H$, whose vertex count is chosen to match that temporary degree, to bring the degree back down to a constant while losing only a controlled amount of expansion. The recurrence is therefore

$$ G_{t+1} \;=\; G_t^2 \,\textcircled{z}\, H, $$

where squaring is the expansion amplifier, the zig-zag is the degree regulator, and $H$ is the reusable local-randomness engine. Each round multiplies the vertex count by a constant factor, returns the degree to the same constant, and keeps $\lambda$ bounded away from $1$ because the spectral loss in the zig-zag step is compensated by the gain from the preceding squaring. The construction is explicit because a neighbor query is just a constant number of rotation-map calls, and the proof is recursive because the zig-zag product theorem composes the expansion guarantees of $G$ and $H$ into one for the product — no search through an exponentially large candidate space is ever required.

The cleanest way to expose the constructive content is through rotation maps. The rotation map of a $D$-regular graph sends a (vertex, edge-label) pair $(v,a)$ to the pair $(w,b)$, where $w$ is the endpoint reached along the $a$-th edge and $b$ is the label of that same edge viewed from $w$. The zig-zag step is then literally a composition of three rotation-map calls — $\mathrm{rot}_H$, then $\mathrm{rot}_G$, then $\mathrm{rot}_H$ — threading the back-labels through so the product itself is a valid rotation map with edge-label pairs $(i,j)$:

```python
def rotation_map_G(v, a):
    """Return (w, b): G-edge (v,a) lands at w with back-label b."""
    # Example: explicit D-regular rule on Z_N (D must match |H|)
    w = (v + (2 * a + 1)) % G_N
    b = a
    return w, b

def rotation_map_H(a, i):
    """Return (a_prime, i_back): H-edge (a,i) lands at a_prime with back-label i_back."""
    # Example: explicit d-regular rule on Z_D
    a_prime = (a + i) % H_D
    i_back = (-i) % H_D
    return a_prime, i_back

def zig_zag_rotation_map(rot_g, rot_h, vertex, edge_label_pair):
    v, a = vertex
    i, j = edge_label_pair

    a_prime, i_back = rot_h(a, i)   # zig inside source cloud
    w, b_prime = rot_g(v, a_prime)  # global step in G
    b, j_back = rot_h(b_prime, j)   # zag inside destination cloud

    return (w, b), (j_back, i_back)
```
