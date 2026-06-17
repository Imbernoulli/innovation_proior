Let me start from what actually goes wrong when I try to predict links. I have a partially observed network, and for a pair of nodes `(x, y)` with no edge between them I want a number saying how likely the edge is real-but-missing or about-to-form. The whole toolbox I reach for first is heuristic scores: count common neighbors, or sum Adamic-Adar, or run Katz, or rooted PageRank. And on some networks these are wonderful — Common Neighbors on a social graph is hard to beat, costs nothing, needs no training. But I keep getting burned, and I want to be precise about why. Each of these scores is a fixed function of the structure that bakes in *one* hypothesis about how links come to exist. Common Neighbors says: the more friends two people share, the more likely they're friends. That is a real mechanism in social networks. But in a protein-protein interaction network the data say the opposite — two proteins that share many interaction partners are *less* likely to interact directly. So the very same score that wins on one network is worse than a coin flip on another, and I have no way, short of trying them all and peeking at a validation set, to know which assumption matches the network in front of me. On a power grid or a router topology, essentially all of the classic scores sit near chance, because none of their assumptions describe how those networks wire up. The brittleness isn't a tuning problem; it's structural. I'm choosing a hypothesis by hand when I should be *learning* it.

So the goal sharpens: instead of picking a heuristic, learn the heuristic — learn, from the network itself, what structural pattern around a pair of nodes predicts that they link. The natural shape for that is supervised: take a pair `(x, y)`, look at the structure around it, and train a function from "the structure around the pair" to "link / no-link". And there's a precedent I can build on. There's a method that extracts, for each candidate pair, a local *enclosing subgraph* — grow outward from `x` and `y`, add their 1-hop neighbors, then 2-hop, and so on until you've collected more than `K` vertices — then runs a hashing-based Weisfeiler-Lehman relabeling to put the vertices into a consistent order, truncates the subgraph down to exactly `K` vertices by dropping the last ones in that order, and feeds the resulting fixed `K×K` adjacency matrix into a fully-connected network that classifies it. That already does the thing I want: it learns a network-specific predictor rather than assuming one, and it beats the hand-crafted scores. So why not stop here?

Because of the truncation. A fully-connected network needs a fixed-size input — a `K×K` matrix, always the same `K` — and that forces me to chop every enclosing subgraph down to `K` vertices. A pair with a rich, dense neighborhood gets most of it thrown away; the model literally cannot read each pair's full `h`-hop neighborhood, only a `K`-vertex truncation of it. That's losing exactly the structure I'm trying to learn from. And because the input is an adjacency matrix and nothing else, there's no slot to also feed in learned node embeddings or node attributes — the latent and explicit features that I know, from prior work, help when combined with topology. Two real limitations, both traceable to the same root: a fully-connected network demanding a fixed-size tensor. Wall. I want to keep the "learn from the enclosing subgraph" idea but drop the fixed-size straitjacket.

But before I redesign the learner, there's a question gnawing at me that I have to settle first, because it decides whether the whole local-subgraph program even makes sense. The strongest heuristics — Katz, rooted PageRank, SimRank — are the *high-order* ones, and "high-order" means, by definition, they read the entire network. Katz sums over walks of *every* length between `x` and `y`. So if I want to learn something as good as Katz, naïvely it seems I need the enclosing subgraph to *be* the whole network — take `h` so large that the `h`-hop neighborhood swallows the graph. For any real network that's hopeless: time and memory blow up. So either local subgraphs are fundamentally too weak to learn high-order structure, in which case this approach is capped at second-order heuristics, or there's something I'm missing. Do I actually need a huge `h` to capture a high-order score? I need to *prove* one way or the other before I commit.

Let me get the trivial part out of the way first, because it frames everything. Define the `h`-hop enclosing subgraph `G^h_{x,y}` as the subgraph of `G` induced by the node set `{ i : d(i,x) ≤ h or d(i,y) ≤ h }` — every node within `h` hops of either center, and the edges among them. Now, what's the order of a heuristic? It's the maximum hop of neighborhood its definition reads. So if a heuristic is order `h`, its value depends only on nodes within `h` hops of `x` or `y` — which is exactly the node set of `G^h_{x,y}`. That gives me, for free: any `h`-order heuristic for `(x,y)` is computable *exactly* from `G^h_{x,y}`. A 2-hop enclosing subgraph contains everything needed for any first- or second-order score. Good — so local subgraphs trivially cover the low-order heuristics. The entire fight is over the high-order ones.

Let me look hard at what a high-order heuristic actually *is*, structurally, rather than at its "reads the whole graph" reputation. Katz is `Σ_{l=1}^∞ β^l |walks^{(l)}(x,y)|` with `0 < β < 1`. Stare at the weights: a walk of length `l` is multiplied by `β^l`, and `β` is small — in practice people set it to something like `5e-4`. So a length-1 contribution is weighted `5e-4`, a length-2 one `2.5e-7`, a length-3 one `1.25e-10`. The tail of long walks is multiplied into oblivion. The "sum over all walk lengths to infinity" is real, but the *contribution* of the far tail is geometrically tiny. The same shape shows up everywhere I look. Rooted PageRank, through the inverse-P-distance identity, is `(1-α) Σ_{w:x⇝y} P[w] α^{len(w)}` — again a sum over walks, again each walk damped by `α^{len}`, `α<1`. SimRank, through its walk expansion, is `Σ_w P[w] γ^{len(w)}` over simultaneous walks that meet, damped by `γ^{len}`, `γ<1`. Every high-order heuristic I respect has the same skeleton: a sum over a length index `l`, with each term carrying a factor that *decays geometrically* in `l`.

That's the clue. Let me abstract it. Call `H(x,y)` a `γ`-decaying heuristic if it has the form

  `H(x,y) = η · Σ_{l=1}^∞ γ^l f(x,y,l)`,

with a decay factor `γ ∈ (0,1)`, a positive constant (or bounded-in-`γ`) prefactor `η`, and a nonnegative `f(x,y,l)`. Katz, PageRank, SimRank all fit this — I'll verify the fits carefully in a moment. The question I actually care about becomes: if `H` is `γ`-decaying, how well can I approximate it using only the *local* structure `G^h_{x,y}`?

Here's the move. I can't compute the infinite sum from a local subgraph, but I don't have to — I only need to compute the first chunk of terms, the ones whose `f(x,y,l)` is determined by the local structure, and let the rest go. Suppose `f(x,y,l)` is computable from `G^h_{x,y}` for all `l` up to some cutoff `g(h)` — where `g(h)` grows with `h` — and approximate `H` by the truncated sum

  `H̃(x,y) := η · Σ_{l=1}^{g(h)} γ^l f(x,y,l)`.

The error is whatever I dropped, the tail from `g(h)+1` to infinity:

  `|H(x,y) − H̃(x,y)| = η · Σ_{l=g(h)+1}^∞ γ^l f(x,y,l)`.

Now I need a handle on `f` in the tail. Suppose `f(x,y,l) ≤ λ^l` for some `λ < 1/γ`. Then each tail term `γ^l f(x,y,l) ≤ (γλ)^l`, and `γλ < 1`, so the tail is a convergent geometric series:

  `|H − H̃| ≤ η · Σ_{l=g(h)+1}^∞ (γλ)^l = η · (γλ)^{g(h)+1} / (1 − γλ)`.

Let me make the dependence on `h` explicit. Take `g(h) = a·h + b` linear in `h` with integers `a > 0, b`. Then the bound is `η (γλ)^{ah+b+1} / (1 − γλ)`, which is `η/(1-γλ) · (γλ)^{b+1} · ((γλ)^a)^h` — a constant times `((γλ)^a)^h`, and since `γλ < 1` the base `(γλ)^a < 1`, so the error **decays at least exponentially in `h`**. There it is in black and white. I do *not* need `h` to reach across the whole network; the approximation error to a high-order heuristic shrinks geometrically with the radius I read, so even a small `h` gets me very close. A small `γλ` and a large `a` make it decay faster. The "effective order" of these famous high-order heuristics is, it turns out, low. The local-subgraph program isn't capped at second order after all — it can chase the high-order scores down to exponentially small error from a small neighborhood.

I've been waving at "`f` is computable from `G^h_{x,y}` for `l` up to `g(h)`" — I need to actually nail what `g(h)` is, because the whole exponent rides on it. The `f` for Katz is the count of length-`l` walks between `x` and `y`. When is *every* length-`l` walk between `x` and `y` fully contained in `G^h_{x,y}`? Take any walk `w = ⟨x, v_1, …, v_{l-1}, y⟩` of length `l`, and pick any intermediate node `v_i`. Suppose, for contradiction, that `v_i` is far from *both* centers: `d(v_i, x) ≥ h+1` and `d(v_i, y) ≥ h+1`. The walk splits at `v_i` into a sub-walk from `x` to `v_i` and one from `v_i` to `y`, and the length of a walk between two nodes is at least their shortest-path distance, so

  `l = |⟨x,…,v_i⟩| + |⟨v_i,…,y⟩| ≥ d(v_i,x) + d(v_i,y) ≥ (h+1) + (h+1) = 2h+2`.

So if `l ≤ 2h+1`, no intermediate node can be far from both centers — every `v_i` has `d(v_i,x) ≤ h` or `d(v_i,y) ≤ h`, which by definition puts it in `G^h_{x,y}`. Hence every walk of length up to `2h+1` between `x` and `y` lives entirely in the enclosing subgraph. That pins down `g(h) = 2h+1`, i.e. `a = 2, b = 1`. Concretely, from a 1-hop enclosing subgraph I can already count walks up to length 3; from 2-hop, up to length 5. The local structure reaches further along walks than its hop radius would suggest, because a walk only has to stay near *one* of the two centers, not both.

Now let me actually check the three big heuristics are `γ`-decaying *and* satisfy the two properties — the `λ`-bound (property 1) and the local-computability `g(h)` (property 2) — because the theorem is only useful if its hypotheses hold for the cases I care about.

Katz first. `Katz(x,y) = Σ_{l=1}^∞ β^l |walks^{(l)}(x,y)|`. This is already in `γ`-decaying form with `η = 1`, `γ = β`, and `f(x,y,l) = |walks^{(l)}(x,y)|`. Property 2 is exactly the lemma I just proved: walk counts of length `l ≤ 2h+1` are readable from `G^h_{x,y}`, so `g(h) = 2h+1`. Property 1 needs a bound `f(x,y,l) ≤ λ^l`. The number of length-`l` walks between two fixed nodes is `[A^l]_{x,y}`, so I need to bound entries of powers of the adjacency matrix. Claim: `[A^l]_{i,j} ≤ d^l` for every `i,j`, where `d` is the maximum node degree. Induction on `l`. Base case `l=1`: `A_{i,j}` is `0` or `1`, certainly `≤ d`. Inductive step: assume `[A^l]_{i,k} ≤ d^l` for all `i,k`; then

  `[A^{l+1}]_{i,j} = Σ_{k} [A^l]_{i,k} A_{k,j} ≤ d^l Σ_k A_{k,j} ≤ d^l · d = d^{l+1}`,

where `Σ_k A_{k,j}` is just the degree of `j`, at most `d`. So `f(x,y,l) = [A^l]_{x,y} ≤ d^l`, giving `λ = d`. Property 1 wants `λ < 1/γ`, i.e. `d < 1/β`. With `β ≈ 5e-4`, `1/β = 2000`, so as long as the max degree is under a couple thousand — true for essentially every network I'd run this on — Katz is `γ`-decaying with all properties satisfied. So Katz is approximable from a small enclosing subgraph with exponentially small error. The famous global heuristic was secretly local all along.

rooted PageRank. The raw definition `π_x = αPπ_x + (1-α)e_x` doesn't look like a walk sum, which is why I reach for the inverse-P-distance identity: `[π_x]_y = (1-α) Σ_{w:x⇝y} P[w] α^{len(w)}`. Group the walks by length: `[π_x]_y = (1-α) Σ_{l=1}^∞ ( Σ_{w:x⇝y, len(w)=l} P[w] ) α^l`. Define `f(x,y,l) := Σ_{w:x⇝y, len(w)=l} P[w]`, and it's `γ`-decaying with `η = 1-α` and `γ = α`. The property-1 bound is even cleaner here. `f(x,y,l)` is the total probability of all length-`l` walks from `x` to `y` — i.e. the probability that the random walker, started at `x`, sits at `y` after exactly `l` steps. Summed over all destinations `z`, those probabilities are a distribution: `Σ_z f(x,z,l) = 1`. So `f(x,y,l) ≤ 1 < 1/α` (since `α < 1`), and property 1 holds with `λ = 1`. Property 2 is the same walk lemma: a probability of length-`l` walks `x⇝y` is computable from `G^h_{x,y}` for `l ≤ 2h+1`. So rooted PageRank is `γ`-decaying with both properties — local-approximable, exponentially.

SimRank. From its walk expansion `s(x,y) = Σ_w P[w] γ^{len(w)}` over simultaneous walks `(x,y) ⊸ (z,z)` — one walk from `x`, one from `y`, first meeting at any common node `z`. Group by length: `s(x,y) = Σ_{l=1}^∞ ( Σ_{w:(x,y)⊸(z,z), len=l} P[w] ) γ^l`, so `f(x,y,l) := Σ_{w, len=l} P[w]` and it's `γ`-decaying with `η=1`, decay `γ`. Property 1: `f(x,y,l)` is again a sum of walk probabilities, bounded by `1 < 1/γ`, so `λ = 1`. Property 2 holds for `l ≤ h` here (the two walks each need to stay within the radius). Either way: SimRank is `γ`-decaying, both properties hold, exponentially local-approximable.

Three for three, and it stops feeling like a coincidence. Why should *every* good high-order heuristic have this geometric-decay skeleton? Because that's what makes them good. A heuristic that put non-vanishing weight on arbitrarily distant structure would be dominated by parts of the network that have essentially nothing to do with whether `x` and `y` link — remote regions are intuitively irrelevant to a local link. The successful heuristics are exactly the ones that down-weight far structure exponentially, and that exponential down-weighting is *precisely* the property that makes them computable from a local subgraph. The thing that makes a global heuristic *work* is the same thing that makes it *local*. So the conclusion I needed is firm: local enclosing subgraphs contain enough information to compute or tightly approximate first-order, second-order, and the standard high-order heuristics — which means a model with enough learning capacity, fed those subgraphs, can in principle match or beat any of them, and isn't restricted to reproducing predefined heuristics at all; it can learn *general* structural features, including ones no existing heuristic names. And I get to use a small `h`. The local-subgraph program is justified, and now I can go redesign the learner without the fear that I've capped myself below the global heuristics.

Back to the learner. I want to keep enclosing subgraphs but drop the fixed-size truncation, so I need a model that swallows graphs of *arbitrary* size and a per-node feature matrix, and outputs a graph-level label. That's exactly what a graph-classification GNN does: graph-convolution layers that mix each node with its neighbors, then an aggregation layer that pools nodes into one graph vector for classification. Crucially a GNN accepts variable node counts, so no truncation — I feed the *whole* enclosing subgraph. And it takes a continuous node-information matrix `X`, which gives me the slot WLNM never had: I can put learned embeddings and node attributes into `X` alongside the structure. So: transform link prediction into *subgraph classification*. For each sampled positive and negative link, extract its enclosing subgraph, attach a node-information matrix, and train a graph-level GNN to classify "this subgraph has a link in the center" vs not.

But the instant I write that down I hit a problem that the fully-connected predecessor didn't have, precisely because a GNN treats nodes more symmetrically. A standard GNN convolution updates each node from its neighbors with shared weights, and the aggregation pools all nodes — typically just sums their features — into one graph vector. That symmetry is a feature for ordinary graph classification (a molecule's label shouldn't depend on how you number its atoms), but it's a disaster here. My enclosing subgraph is *not* an undifferentiated blob: it has a center, the pair `(x,y)` whose link I'm asking about, and everything else is context arranged around that center. If the GNN treats all nodes the same and just sums them, it has no way to know *which two nodes are the target pair*. It can't locate the link it's supposed to score. Two enclosing subgraphs with completely different target pairs but the same overall shape would look identical to it. That throws away the single most important piece of information — where the question is being asked. So I need to break the symmetry: I need to *mark* the nodes so the GNN can tell the center pair from the periphery, and more generally tell apart nodes that sit in structurally different positions relative to `(x,y)`.

So I'll add a structural *label* to every node in the enclosing subgraph — an integer that encodes its role relative to the two centers — and one-hot it into the node-information matrix `X`. Now: what should the labeling actually be? Let me reason from what it has to accomplish rather than guess a scheme. First, the two target nodes `x` and `y` must get a single distinctive label, different from everyone else, so the GNN can always find the pair whose link it's scoring. Second, two nodes should get the *same* label exactly when they play the same structural role with respect to the link — and the natural measure of a node `i`'s role in an enclosing subgraph is its position relative to *both* centers, captured by the pair of distances `(d(i,x), d(i,y))`. I'll call that the node's *double radius*. Nodes on the same "orbit" — same distance to `x` and same distance to `y` — are structurally interchangeable with respect to the link, so they should share a label; nodes on different orbits should differ. That's the whole specification: target pair → one special label; otherwise, label `i` and `j` the same iff `(d(i,x), d(i,y)) = (d(j,x), d(j,y))`.

Let me build a concrete labeling that satisfies this. Give `x` and `y` the label `1`. Then go outward in shells of increasing radius. The node closest in to both, double radius `(1,1)`, gets `2`. Then `(1,2)` and `(2,1)` — same role by symmetry of the two centers, so the same label — get `3`. `(1,3)` and `(3,1)` get `4`. `(2,2)` gets `5`. `(1,4)`/`(4,1)` get `6`. `(2,3)`/`(3,2)` get `7`. And so on, peeling shells. I want a clean ordering rule so this is well-defined for any double radius, not an ad-hoc list. Looking at the sequence, the labels grow with how far out the node is, and I'm measuring "far out" by two things in order: first the *sum* of the two distances, `d(i,x) + d(i,y)` — bigger sum, farther shell, bigger label; and within a fixed sum, by the *product* `d(i,x)·d(i,y)` — bigger product, bigger label. So the rule is: if `d(i,x)+d(i,y) ≠ d(j,x)+d(j,y)`, then the smaller sum gets the smaller label; if the sums tie, the smaller product gets the smaller label. (Smaller sum = smaller arithmetic-mean distance to the pair; smaller product, for a fixed sum, = smaller geometric-mean distance — so the labels rank nodes by how centrally they sit between `x` and `y`.) That's a total order on double radii, so it gives a consistent integer to every orbit. Call it Double-Radius Node Labeling.

Now I'd really like a closed form, because computing labels by sorting orbits per subgraph is clunky, and a closed form means I can hash any `(d(i,x), d(i,y))` straight to its label. Let `d_x = d(i,x)`, `d_y = d(i,y)`, and `d = d_x + d_y` the shell index. Within shell `d`, the orbits are `(1, d-1), (2, d-2), …` and their symmetric partners, and the labeling counts them by closeness to the diagonal. Let me figure out how many labels are used up by all the *strictly smaller* shells, then add the offset within the current shell. Count nodes (orbits, counting `(a,b)` and `(b,a)` once) with sum `< d`: working it out shell by shell, the cumulative count through shell `d-1` is `(d/2)·((d/2) + (d%2) − 1)` using integer division — let me sanity-check that against the explicit list rather than trust it. Within the current shell, the offset is `min(d_x, d_y)`, since orbits in a shell are ordered by their smaller coordinate. Putting it together and adding `1` for the base label of `x,y`:

  `f_l(i) = 1 + min(d_x, d_y) + (d/2)·[ (d/2) + (d%2) − 1 ]`,

with `/` integer division and `%` the remainder. Let me verify it reproduces my hand-built list, term by term, because if a single case is off the GNN gets a wrong role map. `(1,1)`: `d=2`, `min=1`, `(d/2)=1`, `(d%2)=0` → `1 + 1 + 1·(1+0−1) = 1+1+0 = 2` ✓. `(1,2)`: `d=3`, `min=1`, `d/2=1`, `d%2=1` → `1 + 1 + 1·(1+1−1) = 1+1+1 = 3` ✓. `(1,3)`: `d=4`, `min=1`, `d/2=2`, `d%2=0` → `1 + 1 + 2·(2+0−1) = 1+1+2 = 4` ✓. `(2,2)`: `d=4`, `min=2`, `d/2=2` → `1 + 2 + 2·(2−1) = 1+2+2 = 5` ✓. `(1,4)`: `d=5`, `min=1`, `d/2=2`, `d%2=1` → `1 + 1 + 2·(2+1−1) = 1+1+4 = 6` ✓. `(2,3)`: `d=5`, `min=2`, `d/2=2`, `d%2=1` → `1 + 2 + 2·(2+1−1) = 1+2+4 = 7` ✓. Six for six, including both the tie case `(2,2)` and the odd-sum cases. The formula is a perfect hash from double radius to label — closed-form, no per-subgraph sorting.

A subtlety in computing the distances themselves, and it matters. When I measure `d(i, x)`, I should temporarily *remove* `y` from the subgraph (and remove `x` when measuring `d(i, y)`). Why: if `y` is left in, `d(i,x)` is at most `d(i,y) + d(x,y)` — a path from `i` to `x` could shortcut through `y` — so the presence of the *other* center contaminates the radius I'm trying to read. I want `i`'s pure distance to `x`, the true radius, so I knock out `y` first. And two more edge cases. If a node can't reach `x` at all once `y` is removed (or can't reach `y` once `x` is removed), `d(i,x)=∞` or `d(i,y)=∞`; I give such nodes a null label `0` — they're disconnected from a center, off the orbit grid entirely. Finally, the labels go into `X` as *one-hot* vectors. That throws away the magnitude ordering I built into the integers — but I don't need the magnitudes here. The earlier method used WL labels to define a fine vertex *ordering* (so it wanted labels as discriminating as possible, to break ties); I'm using labels only to mark structural *roles* for the GNN to consume as features, so coarse one-hot role indicators are exactly right, and the GNN learns whatever it needs from them.

Now the second source of information I want in `X`: latent node features, learned embeddings, on top of the structural labels. Concatenate each node's embedding to its label vector. But generating those embeddings turns out to be a trap I have to think through. The obvious thing is to run a network-embedding method on the observed graph `G` and read off node vectors. The problem: my positive training links are a subset of the observed edges, `E_p ⊆ E`. So if I embed on `G`, the embeddings will encode the existence of exactly those training edges. Then when the GNN sees a positive training subgraph, its node embeddings already *say* "these two are connected" — the GNN can fit that signal trivially, classify every training positive by it, get great training accuracy, and learn nothing that generalizes, because at test time the held-out edges were never in `G` and carry no such embedded signal. I've watched a model do exactly this: latch onto the leaked existence info and overfit. So embedding on `G` poisons the latent features. The fix has to make the positive and negative training links *indistinguishable* from the embeddings' point of view. Here's the trick: temporarily inject the negative training links `E_n` into the edge set and compute the embeddings on `G' = (V, E ∪ E_n)`. Now both positive *and* negative training links are present as edges when the embeddings are formed, so both get the same kind of "these are connected" signal baked in — the embeddings no longer separate positives from negatives, and the GNN is forced to learn from genuine structure instead of the leak. Negative injection. And the same edge-leak logic applies one level up, to the subgraph structure itself: in a *positive* training link's enclosing subgraph, the edge `(x,y)` is literally present, and that edge *is* the label — it won't be there for any test pair, so if I leave it in, the GNN just reads the answer off the center edge. So I must delete the edge between the two target nodes from every positive subgraph before feeding it in. (Explicit node attributes, when a dataset has them, concatenate into `X` the same way as a third block — but most of these networks have none, so the structural label plus the embedding is the usual `X`.)

So the framework is taking shape: for each sampled positive and negative link, extract its `h`-hop enclosing subgraph (dropping the center edge on positives), build the node-information matrix `X` from one-hot DRNL labels (plus embeddings/attributes when available), and train a graph-level GNN to classify subgraphs. I still need the GNN itself. I want a graph classifier that (a) takes arbitrary-size graphs, (b) extracts multi-hop substructure features per node, and (c) pools to a fixed-size graph vector in a way that *keeps* information rather than washing it out. For the clean graph-classification derivation, the natural convolution is a localized, propagation-style layer: linearly transform the node features, propagate to neighbors (including self), normalize, nonlinearity —

  `Z = f( D̃^{-1} Ã X W )`,   `Ã = A + I`,   `D̃` the diagonal degree of `Ã`,

so that row `i` becomes `f( (1/(|Γ(i)|+1)) [ X_i W + Σ_{j∈Γ(i)} X_j W ] )` — each node's new state is its own and its neighbors' transformed features, averaged. This is the same first-order, one-hop-per-layer propagation family as the renormalized spectral convolution (which uses the symmetric `D̃^{-1/2} Ã D̃^{-1/2}`), and it can be read as a differentiable, trainable relaxation of Weisfeiler-Lehman color refinement — exactly the kind of multi-scale structural feature extractor I want. When I instantiate the code in PyG, I can use `GCNConv` inside the same DGCNN shell and keep its optional `edge_weight` path. Stack several such layers and *concatenate* each layer's output, `[Z^1, Z^2, …, Z^h]`, so each node ends with a descriptor spanning 1-hop through `h`-hop substructure.

The pooling is where ordinary GNNs throw away too much — summing all node states into one vector is permutation-invariant but loses the individual-node and topology information I worked to extract. Instead, *sort* the nodes into a consistent order using their final convolutional states — the reference `global_sort_pool` path sorts by the last channel descending — then truncate or zero-pad the sorted sequence to a fixed length `k` and run an ordinary 1-D CNN over it. Sorting by structural features is permutation-invariant in the sense that matters — isomorphic subgraphs sort to the same sequence, so they get the same representation — but it preserves each node's identity and their relative arrangement, and it lets a 1-D CNN read the nodes in a meaningful order the way an image CNN reads pixels. Concretely: four convolution layers with `32, 32, 32, 1` channels concatenated, then the sort-pooling to `k`, then two 1-D conv layers, a dense layer, and the link/no-link output. Choose `k` from the training subgraph-size distribution, using the reference convention of the `k`-quantile value when `k <= 1` and at least `10` nodes. Train with cross-entropy for some tens of epochs and keep the model with the best validation loss.

One more thing the theory hands me: how big should `h` be? The exponential-decay result says the marginal information from each extra hop shrinks geometrically, so I should not pay for a large radius by default. I'll choose `h` only from `{1, 2}`: `h = 3` can make a subgraph explode if it happens to swallow a hub node, and the tail bound already argues for small effective order. A cheap selector: if the second-order score Adamic-Adar beats the first-order Common Neighbors on a validation split, the network rewards two-hop structure, so use `h = 2`; otherwise `h = 1`. And a clean sanity check on the whole design: take `h = 0`. Then the enclosing subgraph is just the two isolated target nodes, the propagation matrix is the identity, the convolutions reduce to plain neural nets on the two nodes' embeddings, and pooling their embeddings recovers exactly a latent-feature link predictor (node2vec-style with a Hadamard/sum readout, or matrix factorization with an inner-product readout). So the latent-feature methods are the `h=0` special case of this framework, and turning `h` up from there is precisely adding the enclosing-subgraph structure on top of them. That tells me the design is a strict generalization, not a fourth competitor.

So let me write the whole thing as the code I'd actually run, filling the single empty slot — the link predictor. The center piece is the DRNL labeling exactly as I derived it (distance to one center with the other removed; the perfect-hash formula; null label for unreachable; one-hot), the enclosing-subgraph extraction (BFS out `h` hops, drop the center edge), and the sorting-pooling graph classifier. The indexing and sort-pool path need to be concrete, because off-by-one distance labels or a different pooling shape would change the model.

```python
import math
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path

import torch
import torch.nn.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, Embedding, ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_sort_pool


def neighbors(fringe, A):
    return set(A[list(fringe)].indices)


def drnl_node_labeling(adj, src, dst):
    # Double-Radius Node Labeling: integer role label per node in an enclosing
    # subgraph, from its distances to the two target nodes (src, dst).
    src, dst = (dst, src) if src > dst else (src, dst)

    # distance to src measured with dst removed, and vice versa, so the radius
    # to one center is not shortcut through the other center.
    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]
    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst                         # d = d_x + d_y
    dist_over_2, dist_mod_2 = dist // 2, dist % 2       # (d/2), (d%2)

    # f_l = 1 + min(d_x, d_y) + (d/2)*[(d/2) + (d%2) - 1]   -- the perfect hash
    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.                                         # target pair -> label 1
    z[dst] = 1.
    z[torch.isnan(z)] = 0.                              # unreachable -> null label 0
    return z.to(torch.long)


def k_hop_subgraph(src, dst, num_hops, A, node_features=None, y=1):
    # Enclosing subgraph: BFS out num_hops hops from {src, dst}.
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for dist in range(1, num_hops + 1):
        fringe = neighbors(fringe, A) - visited        # next hop's new nodes
        visited |= fringe
        if len(fringe) == 0:
            break
        nodes += list(fringe)
        dists += [dist] * len(fringe)
    subgraph = A[nodes, :][:, nodes]
    subgraph[0, 1] = 0                                  # drop the target edge (it is the label)
    subgraph[1, 0] = 0
    if node_features is not None:
        node_features = node_features[nodes]
    return nodes, subgraph, dists, node_features, y


def construct_pyg_graph(node_ids, adj, dists, node_features, y):
    # Pack one enclosing subgraph as a PyG Data object; z = DRNL labels.
    u, v, r = ssp.find(adj)
    edge_index = torch.stack([torch.LongTensor(u), torch.LongTensor(v)], 0)
    edge_weight = torch.LongTensor(r).to(torch.float)
    z = drnl_node_labeling(adj, 0, 1)                  # targets are at positions 0, 1
    return Data(node_features, edge_index, edge_weight=edge_weight, y=torch.tensor([y]), z=z,
                node_id=torch.LongTensor(node_ids), num_nodes=adj.shape[0])


class LinkPredictor(torch.nn.Module):
    """SEAL: classify each candidate edge's enclosing subgraph. The graph-level
    GNN is DGCNN -- propagation convolutions + a sorting pool + a 1-D CNN head."""
    def __init__(self, hidden_channels, num_layers, max_z, k=0.6,
                 train_dataset=None, dynamic_train=False,
                 use_feature=False, node_embedding=None):
        super().__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding

        if k <= 1:                                     # k as a percentile of subgraph sizes
            if train_dataset is None:
                k = 30
            elif dynamic_train:
                num_nodes = sorted([g.num_nodes for g in train_dataset[:1000]])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
            else:
                num_nodes = sorted([g.num_nodes for g in train_dataset])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = int(k)

        self.z_embedding = Embedding(max_z, hidden_channels)   # one-hot DRNL label -> dense
        initial_channels = hidden_channels
        if use_feature:
            initial_channels += train_dataset.num_features      # explicit attributes
        if node_embedding is not None:
            initial_channels += node_embedding.embedding_dim    # latent (negative-injected) embeddings

        # propagation convolutions; concatenate every layer's output per node
        self.convs = ModuleList()
        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))          # last channel drives the sort

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1     # concat width per node
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def encode(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        # node-information matrix X = [DRNL label || explicit attrs || latent embedding]
        h = self.z_embedding(z)
        if h.ndim == 3:
            h = h.sum(dim=1)
        if self.use_feature and x is not None:
            h = torch.cat([h, x.to(torch.float)], 1)
        if self.node_embedding is not None and node_id is not None:
            h = torch.cat([h, self.node_embedding(node_id)], 1)
        xs = [h]
        for conv in self.convs:                        # multi-hop substructure features
            xs.append(torch.tanh(conv(xs[-1], edge_index, edge_weight)))
        return torch.cat(xs[1:], dim=-1)               # per-node concat of all layers

    def decode(self, node_states, batch):
        # sort nodes by structural features, keep k, run a 1-D CNN -> link score
        x = global_sort_pool(node_states, batch, self.k)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x)                            # logit: subgraph has a center link?

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        node_states = self.encode(z, edge_index, batch, x=x,
                                  edge_weight=edge_weight, node_id=node_id)
        return self.decode(node_states, batch)
```

Let me trace the whole causal chain back. I started stuck because every predefined heuristic hard-codes one assumption about link formation, so each is great on the matching networks and useless on the rest, and I can't pick the right one in advance — I should *learn* the structural signal from the network. The existing learn-from-subgraphs method already did that but was crippled by a fully-connected network needing fixed-size input, which forced truncating the neighborhood and left no slot for node features. Before fixing the learner I had to know whether local subgraphs are even enough to learn the strong *high-order* heuristics, which by definition read the whole network — so I abstracted Katz, PageRank, SimRank into a single `γ`-decaying form `η Σ γ^l f(x,y,l)` and proved that truncating the sum to the terms a small `h`-hop subgraph can compute gives error decaying *at least exponentially* in `h`, with the walk-lemma pinning the cutoff at `g(h)=2h+1`; checking the three heuristics fit confirmed the local-subgraph program reaches the global heuristics with a small radius, and isn't capped below them. That freed me to replace the fixed-size learner with a graph-level GNN that takes arbitrary-size subgraphs and a continuous node-information matrix. But a symmetric GNN couldn't tell which two nodes are the target pair, so I derived a node labeling from first principles — distinctive label for the pair, same label for nodes sharing a double radius `(d(i,x), d(i,y))` — ordered by sum-then-product of the radii, and found its perfect-hash closed form, measuring each radius with the other center removed and one-hotting the result. To stop the latent embeddings from leaking train-link existence I inject the negative links into the graph before embedding, and I delete the center edge from every positive subgraph for the same reason. The GNN is one-hop graph convolutions whose per-layer outputs concatenate into multi-hop node descriptors, pooled not by summing but by sorting on structural features and running a 1-D CNN, so the readout keeps node and topology information; `h` stays in `{1,2}` because the exponential decay argues for small radii and larger subgraphs can explode around hubs, and `h=0` recovers the latent-feature methods exactly — so the whole construction is a generalization of them with enclosing-subgraph structure layered on.
