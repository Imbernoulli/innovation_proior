The task is to take $N$ points living in $\mathbb{R}^P$ and drop them onto a flat 2D canvas so a person can read off the structure of the data, and the structure has two parts that a 2D canvas cannot hold simultaneously. *Local* structure is each point's neighborhood — who its neighbors are and roughly in what order — and preserving it makes clusters compact and same-class points cohere. *Global* structure is how the neighborhoods sit relative to one another — whether cluster A is between B and C, whether the data is one long curve or a hierarchy — and preserving it makes the overall layout legible. The brutal fact underneath is capacity: a 2D embedding provably cannot preserve every pairwise relationship of a high-dimensional cloud, so a method must *choose* what to keep and be honest that the rest is sacrificed. The goal is to preserve both kinds of structure at once, robustly — not depending on a lucky initialization or fragile tuning — from a sparse, chosen set of point relationships, with no labels at fit time, scalable to large $N$.

The methods of the day each get one kind of structure right and the other wrong. PCA and MDS are linear: faithful coarse layout but they flatten any curved manifold. t-SNE and UMAP preserve neighborhoods beautifully and make crisp clusters, but their global layout is unreliable — clusters scattered with no meaningful between-ness, sometimes false clusters that are not really there. TriMap does better globally but, as I will show, borrows its global structure from its PCA initialization rather than producing it. The deeper problem is that the literature offers a pile of empirical choices — "we use this loss, this graph, this initialization, and it works" — which gives me nothing to improve on. I want principles. Every one of these methods fits the same graph-based frame: from the high-dimensional data construct a weighted graph whose *components* are edges $(i,j)$ or triplets $(i,j,k)$, then minimize over the low-dimensional positions $Y$ an objective $\sum_{\text{components}} \mathrm{Weight}^X \cdot \mathrm{Loss}^Y$, where the weight is fixed from the original data and the loss depends on embedded positions. Under gradient descent the loss becomes *forces*: attraction pulling chosen pairs together, repulsion pushing others apart. The methods differ only in which components they pick and what force they put on each — two knobs I can reason about separately.

The losses themselves look nothing alike — a KL divergence, a fuzzy cross-entropy, a triplet ratio — so I compare the thing that actually drives the optimization, the forces, on the plane of the two low-dimensional distances $d_{ij}$ (a neighbor of $i$) and $d_{ik}$ (a far point). On this plane the good methods share a force pattern that I can write as six principles. Monotonicity: as a neighbor recedes the loss should rise ($\partial\mathrm{Loss}/\partial d_{ij}\ge 0$, attraction), as a far point recedes the loss should fall ($\partial\mathrm{Loss}/\partial d_{ik}\le 0$, repulsion). Asymmetry: once a far point is comfortably far, stop spending force on it and point the gradient left toward shrinking $d_{ij}$; dually, when a far point has crept too close, push it away hard. Magnitudes: small force on an already-close neighbor (no point making it closer), large force on a too-close intruder, and — the deepest one — the attractive force should be *unimodal* in $d_{ij}$, strongest at moderate distance and *vanishing for neighbors that are hopelessly far*. That give-up tail is the limited-capacity principle: if a neighbor has ended up very far in the embedding it usually means I genuinely cannot preserve it, and yanking it with a huge force only distorts everyone else to save one neighbor I cannot save anyway. The bad alternatives get exactly this wrong — one has force *increasing* with $d_{ij}$, fighting hardest to save the neighbors it cannot save; another has near-zero gradient where intruders are too close, so it never separates them and everything crowds.

Checking six geometric conditions one triplet at a time is hopeless, so I look for a structural shortcut, and I find one: restrict to *separable* losses,
$$\mathrm{Loss} = \sum_{ij}\mathrm{Loss}_{\mathrm{attr}}(d_{ij}) + \sum_{ik}\mathrm{Loss}_{\mathrm{rep}}(d_{ik}),$$
with $f(d_{ij}) = \partial\mathrm{Loss}_{\mathrm{attr}}/\partial d_{ij}$ the attractive strength and $g(d_{ik}) = -\partial\mathrm{Loss}_{\mathrm{rep}}/\partial d_{ik}$ the repulsive strength. For a single triplet the gradient components are simply $f(d_{ij})$ and $-g(d_{ik})$ — completely decoupled. The claim is that it suffices for $f$ and $g$ to be nonnegative, unimodal, and to vanish at both ends, $\lim_{d\to 0} = \lim_{d\to\infty} = 0$. Every principle then drops out: $f,g\ge 0$ give monotonicity; $\lim_{d_{ik}\to\infty} g = 0$ makes the repulsion-to-attraction ratio vanish for far points (gradient turns left), and the dual limit $\lim_{d_{ij}\to\infty} f = 0$ turns it up when an intruder is too close; vanishing $f$ near zero gives the small force on close neighbors; unimodality gives an interior repulsive hump for too-close points; and $f$ unimodal with $f\to 0$ at infinity gives the moderate-distance peak and the give-up tail. So I never needed triplets to couple the two distances — a separable *edge* loss obeys all six principles, which is a relief because triplets are awkward and expensive, and it suggests the triplet machinery does less than it looks. (UMAP's own force functions, $f \propto d_{ij}^{2b-1}/(1+a\,d_{ij}^{2b})$ and $g \propto d_{ik}/((\varepsilon+d_{ik}^2)(1+a\,d_{ik}^{2b}))$, satisfy this sufficient condition for $b>0.5$, which is reassuring: the condition describes what already works.)

But there is a hole big enough to drive the whole problem through: everything above is about *local* structure. The plane only ever talks about neighbors and far points. A thought experiment makes the gap precise. Take a 0-1 triplet loss that is zero whenever each chosen near point is closer than its paired far point, and confine the triplet set to triplets that contain a near neighbor as $j$. This loss can be driven exactly to zero — yet "each near point closer than its paired far point" says *nothing* about how the far points are arranged relative to one another. I can fold a smooth curve into a tangled mess and still hit zero loss. Zero loss, perfect by its own measure, global structure destroyed. The reason is that these losses only put force between neighbors (attract) and too-close far points (repulse); for a point that is genuinely far in low dimensions the force is zero, and crucially it is zero whether the point is moderately far or extremely far. The relative distances among far points — which *are* the global structure — never enter the objective. Computing t-SNE's force confirms it rather than asserting it. Its gradient on $i$ splits into $F_{\mathrm{attract}} = 4 p_{ij} d_{ij}(1+d_{ij}^2)^{-1}$, where $p_{ij} \propto \exp(-\|x_i-x_j\|^2/2\sigma^2)$ in the *original* space is minuscule for non-neighbors (modern implementations zero it beyond $3\cdot$Perplexity neighbors), and $F_{\mathrm{repulse}} = 4 q_{ij} d_{ij}(1+d_{ij}^2)^{-1}$. Substituting $q_{ij} = a_{ij}/(a_{ij}+B_{ij})$ with $a_{ij} = (1+d_{ij}^2)^{-1}$ and $B_{ij}$ the normalizer, the repulsion is bounded by a $\Theta(1/d_{ij})$ envelope and, for fixed $B_{ij}$, decays as $\Theta(1/d_{ij}^3)$, with derivative $\sim -12/(B_{ij} d_{ij}^4)\to 0$ flattening out. So both the force and its slope are near zero for any far pair, *independent of their true high-dimensional distance* — derived, not assumed. UMAP has the same narrow working zone. And TriMap, the globally better one, does not escape this: about 90% of its triplets contain a nearest neighbor as $j$, so almost all its attraction is local, and its repulsion is near-flat once the far point is past close. Its global structure comes from its PCA initialization — remove the random triplets and almost nothing changes, remove the PCA init and the global layout collapses — and its tempered-log weights come out nearly all equal, so the weighting does little too. TriMap does not solve the problem; it borrows the answer from its initializer, which is exactly the fragility I want to kill.

I propose PaCMAP — Pairwise Controlled Manifold Approximation Projection. The missing ingredient is now precise: a third kind of pair, neither nearest neighbor nor random-far but at *moderate* distance, which I attract, so that the objective itself contains information about the mid-range arrangement and can build a global layout that does not depend on a lucky start. Call them mid-near pairs. Neighbors handle local structure, a repulsion handles crowding, and the mid-near pairs handle the global skeleton — three pair types, all plain edges, no triplets, because I proved separable edge losses already obey every local principle. Picking the mid-near pairs cheaply is a sampling trick: for each point $i$, draw six points uniformly and take the *second closest*. With six uniform draws the closest tends to be a genuine near point and the second-closest sits in the lower-middle of the distance distribution — moderate, not nearest, not random-far — and it is a nearly free order statistic that needs no ranking. Neighbors are the $n_{NB}$ nearest by a *scaled* distance $d^2_{\text{select}}(i,j) = \|x_i-x_j\|^2/(\sigma_i\sigma_j)$, where $\sigma_i$ is the average distance from $i$ to its 4th-through-6th Euclidean neighbors — a local scale estimate that skips the degenerate-nearest and stays in the genuinely local band, so that "neighbor" means the same thing in dense and sparse regions; this scaling is used only for *selection*, never in the loss, which acts on embedded positions. To reuse a fast k-NN routine I over-fetch the $\min(n_{NB}+50, N-1)$ Euclidean neighbors and re-rank that shortlist by the scaled distance. Far pairs are the cheapest: random non-neighbors, the repulsion set.

The loss is three separable rational terms built on the transformed distance $\tilde d_{ab} = \|y_a-y_b\|^2 + 1$ borrowed from t-SNE's Student-t kernel — the $+1$ keeps it bounded below and its slow growth near zero gives the small gradient for tiny $d_{ij}$ that the close-neighbor principle wants. For neighbors I want a unimodal attraction that peaks at moderate distance and saturates for large distance, which a saturating rational delivers; the same shape over a *much wider* range serves the mid-near pairs; and a decaying term gives repulsion strong only when an intruder is too close:
$$\mathrm{Loss} = w_{NB}\!\!\sum_{\text{neighbors}}\!\!\frac{\tilde d_{ij}}{10+\tilde d_{ij}} \;+\; w_{MN}\!\!\sum_{\text{mid-near}}\!\!\frac{\tilde d_{ik}}{10000+\tilde d_{ik}} \;+\; w_{FP}\!\!\sum_{\text{further}}\!\!\frac{1}{1+\tilde d_{il}}.$$
The neighbor term approaches 1 as $\tilde d$ grows — it saturates, so the loss stops rising and the force dies for hopelessly far neighbors (the give-up tail), while near $\tilde d\approx 1$ it is small and slowly varying. The constant 10 sets the width of the attractive working zone. The mid-near term has the same form with denominator 10000, which keeps it in its slowly rising, essentially linear regime out to much larger $\tilde d$ before saturating, so a mid-near pair feels a gentle, persistent pull across a broad distance range — exactly the "organize the global skeleton" behavior, separate from and wider than the narrow neighbor zone. The further term $1/(1+\tilde d)$ is large when $\tilde d$ is small (intruder close, big penalty) and decays to nothing as the far point recedes. The constants 10, 10000, 1 are working-zone tuners, not sacred; what is load-bearing is that the loss is separable, the forces obey the principles, and the mid-near attraction is a separate, wider-ranging term. Differentiating each — using $d/d\tilde d\,[\tilde d/(c+\tilde d)] = c/(c+\tilde d)^2$, $d/d\tilde d\,[1/(1+\tilde d)] = -1/(1+\tilde d)^2$, and the chain rule $\partial\tilde d/\partial y_i = 2(y_i-y_j)$ — gives the forces on $y_i$:
$$\text{neighbor: } +w_{NB}\frac{20}{(10+\tilde d)^2}(y_i-y_j),\quad \text{mid-near: } +w_{MN}\frac{20000}{(10000+\tilde d)^2}(y_i-y_j),\quad \text{further: } -w_{FP}\frac{2}{(1+\tilde d)^2}(y_i-y_j).$$
The attractive sign points the gradient on $y_i$ *away* from $j$, so descent steps toward $j$; the minus on the further term points the gradient toward $j$, so descent pushes away. Multiplying each scalar coefficient by the vector $\|y_i-y_j\|$, the actual attractive force is zero at distance 0, peaks at moderate distance, and decays like $1/d^3$ — small for already-close and small again for hopelessly far neighbors — while the repulsive force is zero at exact coincidence, peaks in a close band, then dies, so a too-close intruder is shoved and an already-far point ignored. Exactly the shapes the principles demand.

Because the forces have narrow working zones, two facts follow. The danger: if early optimization flings a pair of true neighbors apart, their attraction has already saturated and they can never come back — a false split, a phantom cluster. The opportunity: once global structure is in place, the long-range forces are gentle and will not tear it up during local refinement. So I build global structure first with a strong long-range pull, then freeze it in and refine locally — a coarse-to-fine schedule in the spirit of t-SNE's early exaggeration and simulated annealing, except I exaggerate the *mid-near* (global) pull, not the neighbor pull. The schedule on $(w_{NB}, w_{MN}, w_{FP})$ has three phases. Phase one (100 iterations) builds the global skeleton: $w_{NB}=2$, $w_{FP}=1$, and $w_{MN}$ annealed from 1000 toward 3 by $w_{MN}(\mathrm{itr}) = (1-\mathrm{itr}/100)\cdot 1000 + (\mathrm{itr}/100)\cdot 3$ — starting overwhelming so mid-near pairs organize the moderate-distance arrangement, then easing so the dominant pull does not later fight local refinement. Phase two (100 iterations) stabilizes: $w_{NB}=3$, $w_{MN}=3$, $w_{FP}=1$, tightening neighborhoods while a small nonzero mid-near weight keeps the skeleton from drifting. Phase three (250 iterations) refines local structure: $w_{MN}=0$, $w_{NB}=1$, $w_{FP}=1$, so attraction and repulsion on the local scale dominate, pulling neighbors tight and letting repulsion carve clean cluster boundaries. The modest phase-one $w_{NB}$ keeps neighbors from being forced to specific spots before the structure is coarse, avoiding the saturating-fling failure.

Initialization is PCA scaled down by 0.01, and the scaling matters: forces have a working zone in absolute distance, so a too-wide start lands every force in its dead zone and freezes the optimizer at step one, while a tight start keeps the forces live. PCA is just a head start here, not a crutch as it was for TriMap — the mid-near phase is the mechanism that *creates* global structure, so a random initialization of $10^{-4}\cdot N(0,I)$ should also organize; if it could not, the graph itself would not be carrying enough global information. The optimizer is Adam with $\mathrm{lr}=1.0$, $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-7}$, run for 450 iterations: the three force terms span very different magnitudes ($w_{MN}$ starts at 1000) and change across phases, so per-coordinate adaptive steps keep the effective step sane where a single global rate would be wrong for some term, and the forces are bounded rationals so a large base rate is safe. Defaults follow from the structure: $n_{NB}=10$, mid-near count $n_{MN}=\mathrm{round}(0.5\cdot n_{NB})=5$, further count $n_{FP}=\mathrm{round}(2\cdot n_{NB})=20$ — repulsion needs more samples than attraction to suppress crowding everywhere, and far pairs are cheapest. If $P>100$ I PCA-preprocess to 100 dimensions first, for speed and a cleaner k-NN search in a denoised space.

```python
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors


def _draw_excluding(n, size, rng, banned):
    banned = set(int(x) for x in banned)
    out = []
    while len(out) < size:
        k = int(rng.integers(n))
        if k not in banned:
            out.append(k)
            banned.add(k)
    return np.asarray(out, dtype=np.int64)


def _select_components(X, n_neighbors, n_MN, n_FP, rng):
    n = X.shape[0]
    # neighbor pairs: nearest by scaled distance ||x_i-x_j||^2 / (sig_i sig_j)
    n_extra = min(n_neighbors + 50, n - 1)
    dist, nbrs = NearestNeighbors(n_neighbors=n_extra + 1).fit(X).kneighbors(X)
    dist, nbrs = dist[:, 1:], nbrs[:, 1:]                       # drop self
    sig = np.maximum(dist[:, 3:6].mean(axis=1), 1e-10)          # avg dist to 4th-6th NN
    scaled = (dist ** 2) / (sig[:, None] * sig[nbrs])
    order = np.argsort(scaled, axis=1)[:, :n_neighbors]
    rows = np.repeat(np.arange(n), n_neighbors)
    nb = np.stack([rows, np.take_along_axis(nbrs, order, axis=1).ravel()], 1)

    # mid-near pairs: 2nd-closest of 6 uniform samples
    mn = np.empty((n * n_MN, 2), dtype=np.int64); t = 0
    for i in range(n):
        picked_for_i = []
        for _ in range(n_MN):
            s = _draw_excluding(n, 6, rng, [i] + picked_for_i)
            d = ((X[s] - X[i]) ** 2).sum(1)
            picked = int(s[np.argsort(d)[1]])
            picked_for_i.append(picked)
            mn[t] = (i, picked); t += 1
    mn = mn[:t]

    # further pairs: random non-neighbors
    nbr_of = nb[:, 1].reshape(n, n_neighbors)
    fp = np.empty((n * n_FP, 2), dtype=np.int64); t = 0
    for i in range(n):
        banned = set(nbr_of[i].tolist()) | {i}
        for _ in range(n_FP):
            k = int(_draw_excluding(n, 1, rng, banned)[0])
            banned.add(k)
            fp[t] = (i, k); t += 1
    return nb, mn, fp[:t]


def _grad(Y, nb, mn, fp, w_nb, w_mn, w_fp):
    grad = np.zeros_like(Y)
    for pairs, c, w in ((nb, 10.0, w_nb), (mn, 10000.0, w_mn)):   # attraction
        i, j = pairs[:, 0], pairs[:, 1]
        diff = Y[i] - Y[j]
        dt = (diff ** 2).sum(1) + 1.0                            # d~ = ||y_i-y_j||^2 + 1
        upd = (w * (2.0 * c) / (c + dt) ** 2)[:, None] * diff
        np.add.at(grad, i, upd); np.add.at(grad, j, -upd)
    i, j = fp[:, 0], fp[:, 1]                                    # repulsion
    diff = Y[i] - Y[j]
    dt = (diff ** 2).sum(1) + 1.0
    upd = (w_fp * 2.0 / (1.0 + dt) ** 2)[:, None] * diff
    np.add.at(grad, i, -upd); np.add.at(grad, j, upd)
    return grad


def _weights(itr, p1, p2):
    if itr < p1:                                                # phase 1: build global
        return 2.0, (1 - itr / p1) * 1000.0 + (itr / p1) * 3.0, 1.0
    if itr < p1 + p2:                                           # phase 2: stabilize
        return 3.0, 3.0, 1.0
    return 1.0, 0.0, 1.0                                        # phase 3: refine local


def pacmap_fit_transform(X, n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0,
                         num_iters=(100, 100, 250), lr=1.0, random_state=None):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] > 100:                                        # denoise / speed kNN
        X = X - X.mean(0)
        X = TruncatedSVD(n_components=100, random_state=random_state).fit_transform(X)
        pca_solution = True
        pca_init = None
    else:
        X = X - X.min()
        X = X / max(X.max(), 1e-12)
        X = X - X.mean(0)
        pca_solution = False
        pca_init = PCA(n_components=n_components, random_state=random_state).fit(X)
    n_MN = int(round(n_neighbors * MN_ratio))
    n_FP = int(round(n_neighbors * FP_ratio))
    nb, mn, fp = _select_components(X.astype(np.float32), n_neighbors, n_MN, n_FP, rng)

    Y = (0.01 * X[:, :n_components] if pca_solution             # PCA init, scaled small
         else 0.01 * pca_init.transform(X))
    Y = np.ascontiguousarray(Y, dtype=np.float64)

    beta1, beta2, eps = 0.9, 0.999, 1e-7                        # Adam
    m = np.zeros_like(Y); v = np.zeros_like(Y)
    p1, p2, _ = num_iters
    for itr in range(sum(num_iters)):
        w_nb, w_mn, w_fp = _weights(itr, p1, p2)
        g = _grad(Y, nb, mn, fp, w_nb, w_mn, w_fp)
        lr_t = lr * np.sqrt(1 - beta2 ** (itr + 1)) / (1 - beta1 ** (itr + 1))
        m += (1 - beta1) * (g - m)
        v += (1 - beta2) * (g ** 2 - v)
        Y -= lr_t * m / (np.sqrt(v) + eps)
    return Y
```
