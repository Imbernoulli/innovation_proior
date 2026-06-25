OK, let me think about this from scratch. I have this whole zoo of graph networks now — GCN, GraphSAGE, the gated graph nets, the message-passing nets, attention-based ones — and they all win on benchmarks, and they were all designed the same way: someone picks an aggregator, picks a way to fold the neighbors into the center node, picks a pooling, and tunes until the test accuracy looks good. And it works. But I genuinely cannot answer the most basic question about any of them: how *powerful* is this thing? If I hand a graph network two graphs that are structurally different, will it give them different embeddings, or are there pairs of graphs it is simply *unable* to tell apart no matter how I train it, no matter how wide or deep? Is there a ceiling? And if there is, which of these design knobs — mean vs max vs sum, one linear layer vs a deeper net — actually pushes against the ceiling, and which are just decoration?

Let me first write down what every one of these models actually is, stripped to its skeleton, because I suspect the differences I care about are smaller than they look. Each node v starts with a feature h_v^{(0)} = X_v. Then for k rounds:

    a_v^{(k)} = AGGREGATE^{(k)}( { h_u^{(k-1)} : u ∈ N(v) } )
    h_v^{(k)} = COMBINE^{(k)}( h_v^{(k-1)}, a_v^{(k)} )

and for a whole-graph vector I pool: h_G = READOUT({ h_v^{(K)} : v ∈ G }). GCN is this with AGGREGATE = mean over N(v)∪{v}, COMBINE folded in, and the transform a single W followed by ReLU. GraphSAGE's pooling variant is AGGREGATE = element-wise max of ReLU(W·h_u), COMBINE = concatenate the center with the aggregate and apply a linear map. So really the only things that vary are: what function squashes the bag of neighbors, and what transform sits around it. Everything else is shared.

So my question reduces to: what does a node's vector after k rounds actually *represent*, and when do two nodes (or two graphs) get forced to the same vector? Let me think about what h_v^{(k)} captures. Round 1, v sees its immediate neighbors' features. Round 2, each of those neighbors has already absorbed *its* neighbors, so v now sees two hops out. After k rounds, h_v^{(k)} is a summary of the rooted subtree of height k hanging off v — the tree you get by unrolling the neighborhood, neighbors-of-neighbors, and so on. Two nodes should be allowed to collide — get the same embedding — only if these rooted subtrees are genuinely identical, same shape, same features at corresponding positions. A maximally powerful version of this family would *never* collapse two different rooted subtrees.

Now here's the thing that nags at me. This "relabel a node by looking at itself plus the bag of its neighbors, iterate" — I have seen this exact loop before, and it is not in deep learning. It's the Weisfeiler-Lehman test for graph isomorphism. WL does color refinement: every node has a color, and each round it recolors itself by hashing the pair (my color, the multiset of my neighbors' colors):

    l_v^{(k)} = HASH( l_v^{(k-1)}, {{ l_u^{(k-1)} : u ∈ N(v) }} )

and two graphs are declared non-isomorphic the instant their multisets of colors disagree at some round. WL is a beast — it correctly separates almost every pair of graphs you'll meet, failing only on nasty symmetric things like regular graphs. And structurally it *is* a graph network: aggregate the neighbors, update. The only difference is WL's HASH is injective — different (color, neighbor-multiset) inputs always get a brand-new distinct color, so WL never throws away a distinction once it has found one. A graph network, by contrast, runs everything through continuous, lossy functions — a mean, a max, a linear layer — that can squash two different inputs to the same output.

That comparison feels like the right ruler. Exact graph isomorphism is the dream target, but deciding isomorphism has no known polynomial algorithm — chasing it directly is hopeless. WL is the strong, efficient, *almost*-complete stand-in, and it has the exact same shape as my networks. So let me measure power against WL: can a graph network separate the graphs WL can separate? Can it do *more*? Let me try to bound it.

Suppose WL can *not* tell G1 and G2 apart. That means at every iteration i from 0 to k, the two graphs have the identical multiset of WL labels — and more than that, the identical multiset of *neighborhoods*, i.e. the same collection of pairs (l_v^{(i)}, {{l_u^{(i)} : u∈N(v)}}). Why "more than that"? Because if at some iteration i the neighborhood-multisets had differed, WL's injective hash would have produced different label-multisets at iteration i+1, and WL would have separated the graphs — contradiction. So as long as WL stays stuck, the neighborhood-multisets march in lockstep across both graphs.

Now I want to show the graph network is *also* stuck — that A(G1) = A(G2). Let me set up an induction *within a single graph* first: claim that whenever two nodes share a WL label, l_v^{(i)} = l_u^{(i)}, they also share the GNN feature, h_v^{(i)} = h_u^{(i)}. Base case i=0: both WL and the GNN start from the same input features, so equal labels mean equal features. Inductive step: suppose it holds at iteration j, and take u,v with l_v^{(j+1)} = l_u^{(j+1)}. Because WL's hash is injective, equal new labels force equal inputs to the hash:

    ( l_v^{(j)}, {{ l_w^{(j)} : w∈N(v) }} ) = ( l_u^{(j)}, {{ l_w^{(j)} : w∈N(u) }} ).

By the inductive hypothesis, equal WL labels mean equal GNN features for every node mentioned, so the GNN's own (center, neighbor-multiset) inputs match too:

    ( h_v^{(j)}, {{ h_w^{(j)} : w∈N(v) }} ) = ( h_u^{(j)}, {{ h_w^{(j)} : w∈N(u) }} ).

But the GNN applies the *same* AGGREGATE and COMBINE everywhere, and a function fed equal inputs returns equal outputs, so h_v^{(j+1)} = h_u^{(j+1)}. Induction closes. So within each graph there is a well-defined map φ with h_v^{(i)} = φ(l_v^{(i)}). Now lift it to the two graphs: since G1 and G2 have the identical multiset of WL neighborhoods, applying φ to everything gives them the identical multiset of GNN neighborhood-features, hence the identical multiset {h_v^{(i+1)}}, all the way to {h_v^{(k)}}. The readout is permutation-invariant over that multiset, so it returns the same value: A(G1) = A(G2). Contrapositive: if the network ever separates two graphs, WL separates them too — so no neighborhood-aggregation GNN distinguishes a pair that 1-WL leaves merged, regardless of width, depth, or training. That gives me an upper bound, and it doesn't depend on the aggregator at all; it only used that the layer is a fixed function applied uniformly.

So power lives strictly below WL. The next question is whether anything in this family actually *attains* it, or whether the popular models sit far underneath. Look back at where the upper-bound proof had slack: I never needed the GNN's functions to be injective, so two distinct WL-neighborhoods could be sent to the same feature and the bound would still hold — the inequality is one-directional precisely because lossy functions are allowed. WL's hash, by contrast, is injective. So the natural thing to test is whether *restoring* injectivity — making AGGREGATE/COMBINE and the readout injective on multisets — is enough to turn the inequality into an equality, or whether something subtler is needed. Let me try to push the proof through under that assumption and watch for the place it breaks.

Assume the layer has the form h_v^{(k)} = φ( h_v^{(k-1)}, f({{h_u^{(k-1)} : u∈N(v)}}) ) with f (operating on the neighbor multiset) and φ both injective, and the graph readout injective. Take G1, G2 that WL separates at iteration K. Because the readout is injective, it's enough to show the network produces different *multisets* of node features at iteration K. I'll show by induction that there's an injective φ_k with h_v^{(k)} = φ_k(l_v^{(k)}) — i.e. the GNN features are a faithful (injective) recoding of WL's labels. Base k=0: identical inputs, so φ_0 = identity, injective. Step: assume h_v^{(k-1)} = φ_{k-1}(l_v^{(k-1)}) with φ_{k-1} injective. Substitute into the layer:

    h_v^{(k)} = φ( φ_{k-1}(l_v^{(k-1)}), f({{ φ_{k-1}(l_u^{(k-1)}) : u∈N(v) }}) ).

Every piece here — φ_{k-1}, f, φ — is injective, and a composition of injective functions is injective, so the right side is an injective function of ( l_v^{(k-1)}, {{l_u^{(k-1)} : u∈N(v)}} ); call it ψ of that pair, ψ injective. But WL's own update is l_v^{(k)} = g( l_v^{(k-1)}, {{l_u^{(k-1)}}} ) with g the injective hash, so I can write that pair as g^{-1}(l_v^{(k)}) (g is injective on its image, so g^{-1} is well-defined there). Then h_v^{(k)} = ψ(g^{-1}(l_v^{(k)})), and φ_k := ψ∘g^{-1} is injective. Induction closes. At iteration K, WL says the label-multisets {l_v^{(K)}} differ between G1 and G2; pushing through the injective φ_k, the feature-multisets {h_v^{(K)}} = {φ_K(l_v^{(K)})} also differ; the injective readout then separates the graphs. The proof did go through, and the only properties it used were the three injectivities — so injectivity on multisets, at each of the three stages, is what closes the gap between the GNN and WL. Nothing about width or smoothness entered; the requirement is purely "don't collapse distinct multisets." That tells me what to engineer, but it's an abstract requirement — I now have to find concrete, neural-realizable functions that actually satisfy it, and check that the popular ones don't.

Wait — I should pin down whether "injective" is even the right notion at deeper layers. I'm assuming node features live in a countable universe so that I can talk about injections the discrete way. Input features are countable, fine. But after a layer, features are some function of multisets of the previous features — does countability survive, or could the hidden features blow up into an uncountable continuum where "injective" stops being the right tool? Let me check it propagates. I want: a function defined on bounded-size multisets of a countable set has a countable range. ℕ×ℕ is countable — concretely φ(m,n) = 2^{m-1}(2n-1) is a bijection ℕ×ℕ → ℕ because every positive integer factors uniquely as a power of two times an odd number — and by induction any finite Cartesian product of countable sets is countable. Now add a dummy symbol e to X; X' = X∪{e} is still countable. Since X is countable there's an injection Z: X→ℕ; sort each multiset's elements by Z into x_1,…,x_n, and since sizes are bounded by some k, pad with e to fixed length: h(X) = (x_1,…,x_n,e,…,e) ∈ X'^k. Different multisets give different sorted-padded tuples, so h is injective, and X'^k is countable. Therefore the set of bounded multisets is countable. A layer's possible outputs are just the image of that countable set; equivalently, for each output choose one input multiset that produced it, which injects the range into a countable domain. Induction over layers: countability propagates. Good — injectivity is exactly the right lens at every depth.

Now the engineering question: I need an aggregator over multisets that is injective, ideally *universal* (able to express any multiset function once composed with a learned transform), and I want it to be something a neural net can realize. There's a known result for *sets*: any permutation-invariant set function decomposes as ρ(Σ_{x∈S} f(x)) — sum the per-element features, then transform. That's the Deep Sets template. But my neighbors form a *multiset*, not a set — two neighbors can be identical, and that multiplicity is structural information I must not lose. Does sum-decomposition survive the jump from sets to multisets? Let me try to build f explicitly.

X is countable, so fix an injection Z: X→ℕ. Bound the multiset size: |X| < N for some integer N. Try f(x) = N^{-Z(x)}. Then Σ_{x∈X} f(x) = Σ_x N^{-Z(x)}, and because each element of X contributes N^{-Z(x)} once per occurrence, the claim is that the total is a number whose base-N representation has, in the Z(x)-th place, the *count* (multiplicity) of x — and since multiplicities are all below N, there's no carrying between places, so distinct multisets give distinct sums. I want to actually see this work rather than wave at it, so let me take N=4 (multisets of size < 4), elements a,b,c with Z(a)=1, Z(b)=2, Z(c)=3, and two genuinely different multisets: X1 = {a,a,b} and X2 = {a,b,b,c}. Then

    Σ_{X1} f = 2·4^{-1} + 1·4^{-2}          = 0.5 + 0.0625            = 0.5625,
    Σ_{X2} f = 1·4^{-1} + 2·4^{-2} + 1·4^{-3} = 0.25 + 0.125 + 0.015625 = 0.390625,

different, as wanted. And the digits really do read back the counts: scale Σ_{X1} f by 4^{3} to clear the fractions, 0.5625·64 = 36 = (2,1,0) in base 4 (2·16 + 1·4 + 0), and 36's base-4 digits at places 1,2,3 are exactly the multiplicities (a:2, b:1, c:0). Doing the same to X2: 0.390625·64 = 25 = (1,2,1) in base 4, i.e. a:1, b:2, c:1 — the true counts again. (Sizes stay below 4 precisely so no place overflows into a carry; that's why the bound N matters.) So on this concrete pair the sum recovers the full multiplicity profile, and the general argument is the same digit-by-digit reasoning. The sum is injective on bounded multisets. And it's universal: for any multiset function g I want, define φ on the image of the sum by φ(Σ_x f(x)) := g(X); this is well-defined precisely because the sum is injective (each achievable sum-value comes from a unique X), and it's permutation-invariant by construction. So g(X) = φ(Σ_x f(x)). Sum decomposition works for multisets — *with* multiplicity preserved. That's the multiset extension of Deep Sets, and it gives me the injective AGGREGATE the theorem demands.

So the choice of aggregator is not free; sum is special. Let me make sure I understand *why* the popular alternatives — mean and max, the ones GCN and GraphSAGE use — are not injective, because that's the practical payoff. They're permutation-invariant, so they're valid multiset functions, but I claim they lose information that sum keeps. Let me find graphs they confuse.

Start with the most embarrassing case. Take any graph where every node has the *same* feature a. Then for any f, f(a) is one fixed vector everywhere. Mean of f(a) over any neighborhood is f(a); max of f(a) over any neighborhood is f(a). By induction every node, in every graph, stays at the same representation forever — mean and max see *no structure at all* on an unlabeled graph. So two unlabeled graphs that are different (say a node with two neighbors vs three) are indistinguishable to mean/max. Sum, though: 2·f(a) for the two-neighbor node and 3·f(a) for the three-neighbor node — different. Sum at least counts. (Aside: if I fed node *degree* as the feature instead of a constant, mean could in principle reconstruct sum, since it'd know the count; but max still couldn't.)

Now a labeled case to separate mean from max. Color the neighbors and give them concrete one-dimensional features so I can just compute: h_g = 1, h_r = 10. Suppose node v has neighbor multiset {green, red} and node v' has {green, red, red}. Max over the neighbors gives max(1,10) = 10 in both cases — the second red doesn't change the max — so v and v' collapse under max. Mean gives ½(1+10) = 5.5 vs ⅓(1+10+10) = 7, which differ, so mean survives this one; sum gives 11 vs 21, also distinct. So far max < {mean, sum}. Now push harder: v has neighbors {green, red}, v' has {green, green, red, red}. Mean: ½(1+10) = 5.5 vs ¼(1+1+10+10) = 22/4 = 5.5 — *equal*, so here mean collapses too; max is max=10 in both, also equal. Sum: 11 vs 22, still distinct. So this single pair already orders them — max merges {g,r} with {g,r,r}, mean merges {g,r} with {g,g,r,r}, and sum keeps both apart — giving the ranking sum ⊐ mean ⊐ max in discriminative power over multisets. (I picked h_g=1, h_r=10 only so the arithmetic is transparent; the inflated-copy collapse for mean is structural, not an artifact of these numbers, as the next paragraph shows.)

I can characterize exactly what mean and max throw away. For mean, take X_1 = (S,m) and X_2 = (S, k·m): same distinct elements, every multiplicity scaled by the same integer k. Then |X_2| = k|X_1| and Σ_{X_2} f = k·Σ_{X_1} f, so the mean (1/|X_2|)Σ_{X_2} f = (1/(k|X_1|))·k·Σ_{X_1} f = (1/|X_1|)Σ_{X_1} f — *identical*. So mean cannot distinguish a multiset from any "inflated" copy of itself; it captures only the **distribution** (the proportions), not the exact counts.

For a suitable f, that is exactly mean's boundary. Let f(x)=N^{-2Z(x)} and write B=N^2. Suppose X and Y have sizes M,L<N and count profiles m_i,n_i under Z. If their means match, then

    Σ_i (m_i/M) B^{-i} = Σ_i (n_i/L) B^{-i},

so after multiplying by ML,

    Σ_i (L m_i − M n_i) B^{-i} = 0.

Only finitely many coefficients are nonzero, and every coefficient a_i = Lm_i − Mn_i has |a_i| < B. Multiply by the largest power B^r that clears the denominators. Reducing the resulting integer equality modulo B forces a_r to be divisible by B; since |a_r| < B, a_r=0. Strip that last term and repeat backward. Every a_i is zero, so Lm_i=Mn_i for every i, i.e. m_i/M=n_i/L. Equal means imply equal distributions, and equal distributions obviously imply equal means. Good: mean can be made injective on distributions, but it still cannot recover absolute multiplicities. For max, the story is even starker: max_{x∈X} f(x) only depends on which distinct elements are present — pick f one-hot, f_i(x)=1 iff i=Z(x), then max over a multiset returns the indicator of its *underlying set*. So max captures neither counts nor proportions, only the **support**. That actually explains the folklore: max-pooling is great when you want the robust "skeleton" of a structure (it's why point-cloud nets like max), and mean is great when the *distribution* of neighbor features is what matters for the label (which is common in node classification with rich features) — but neither can serve as a *maximally powerful* graph aggregator, because both are provably non-injective on multisets. Sum is the one that keeps everything.

So the AGGREGATE should be a sum, and I should learn f as a neural net — universal approximation lets an MLP stand in for the abstract f, and another for φ. But hold on: a lot of existing models, including GCN, don't use an MLP for that transform — they use a *single* linear layer followed by a ReLU, σ∘W. Is that enough to realize the injective f I need? Let me test whether one bias-free layer, summed over a neighborhood, can already separate multisets. Try the smallest counterexample with positive numbers: X_1 = {1,1,1,1,1} and X_2 = {2,3}. They're different multisets but Σx is 5 in both. Now take any linear W with no bias. For positive inputs x, at any output coordinate Wx has a fixed sign across all x (the sign of that row of W). At coordinates where the row is negative, ReLU zeros out everything, contributing equally (zero) to both sums. At coordinates where the row is positive, ReLU is the identity there and is *linear*, so

    Σ_{x∈X} ReLU(Wx) = ReLU(W Σ_{x∈X} x)

on those coordinates (and trivially on the zeroed ones). Since Σ_{X_1} x = Σ_{X_2} x = 5, the two sums are identical for *every* W. Let me confirm I haven't fooled myself with a special W by trying an arbitrary one: take a 4-output map with rows (W = [1.69, 1.03, −0.87, −0.45], one weight each since inputs are scalar). On X_1 = {1,1,1,1,1}, each ReLU(Wx) for x=1 is (1.69, 1.03, 0, 0) and summing five copies gives (8.44, 5.16, 0, 0). On X_2 = {2,3}: ReLU(W·2) = (3.38, 2.06, 0, 0), ReLU(W·3) = (5.06, 3.10, 0, 0), summing gives (8.44, 5.16, 0, 0) — identical. And ReLU(W·5) = (8.44, 5.16, 0, 0) matches, exactly as the identity Σ ReLU(Wx) = ReLU(W·Σx) predicts. The reason is ReLU's positive-homogeneity plus the missing bias: with all-positive inputs the layer can't break the linearity, so the network degenerates into "sum the inputs, then one linear map" — and summing first destroys the distinction. (A bias and a big enough output dimension could rescue *this particular* example by bending some inputs across the ReLU kink, but a single linear+ReLU is still a generalized linear model, not a universal approximator of multiset functions, so it can't realize an arbitrary injective f. I'd expect that to show up as underfitting — embeddings that don't capture structural similarity well enough for a simple downstream classifier — though that's a claim I'd want to check against training-set accuracy on the benchmarks, not something the algebra alone settles.) So the transform has to be a real MLP — at least two layers — not σ∘W. That's a representational requirement, not a tuning preference.

Now the last subtlety, and it's a real one. My injectivity theorem assumed the layer had the form φ(h_v^{(k-1)}, f(neighbor-multiset)) — the center node kept *separate* from the aggregate. But the cleanest thing to build with a sum would be to just throw the center into the bag and sum over {self} ∪ neighbors. Does that lose anything? Consider the chain a–b–b and the chain b–a–b. Look at the middle node. In the first, the middle node is b with neighbors {a,b}; in the second, the middle node is a with neighbors {b,b}. WL keeps these as distinct rooted things: (b, {a,b}) vs (a, {b,b}). Let me check what a flat self-plus-neighbors sum does with the same f as before, Z(a)=1, Z(b)=2, N=4, so f(a)=¼, f(b)=1/16. Flat-merging the center into the bag: the first node's bag is {b; a,b} = {a,b,b} with sum ¼ + 1/16 + 1/16 = 0.375; the second node's bag is {a; b,b} = {a,b,b} with sum ¼ + 1/16 + 1/16 = 0.375 — *identical*. So the flat sum really does collapse them; throwing the center in with the neighbors threw away which element was the root, exactly as WL would have kept track of. I need to keep the center distinguishable. The minimal fix that preserves the sum structure: weight the center differently. Try h(c, X) = (1+ε)·f(c) + Σ_{x∈X} f(x) — sum the neighbors, add a *scaled* copy of the center. Before proving anything, let me just see whether a scaled center actually breaks the tie on this pair. Take ε = √2 − 1, so (1+ε) = √2. First node (center b, neighbors {a,b}): √2·(1/16) + (¼ + 1/16) = 0.0884 + 0.3125 = 0.4009. Second node (center a, neighbors {b,b}): √2·(¼) + (1/16 + 1/16) = 0.3536 + 0.125 = 0.4786. Different — the scaled center separates the two roots that the flat sum merged. So the device works on the example; now I should pin down for which ε it works in general, over arbitrary pairs (c, X).

Use the same f(x) = N^{-Z(x)} as before and suppose, for contradiction, that two distinct pairs collide: h(c,X) = h(c',X'). Two cases. Case 1, c'=c but X'≠X: then (1+ε)f(c) cancels and Σ_X f = Σ_{X'} f, contradicting the injectivity of the plain sum on multisets. Case 2, c'≠c: rearrange the collision to isolate ε,

    ε·( f(c) − f(c') ) = ( f(c') + Σ_{X'} f ) − ( f(c) + Σ_X f ).

The right side is a finite sum and difference of values N^{-Z(·)}, all rational, so it's rational. On the left, f(c) − f(c') = N^{-Z(c)} − N^{-Z(c')} is a *nonzero* rational (nonzero because Z is injective and c≠c'). So if I choose ε **irrational**, the left side is irrational and the right side rational — impossible. Contradiction. So for any irrational ε (in fact infinitely many ε), h(c,X) = (1+ε)f(c) + Σ_x f(x) is injective over (center, multiset) pairs, and by the same well-definedness argument any function over such pairs decomposes as φ((1+ε)f(c) + Σ f(x)). The (1+ε) term is exactly what tags the center as the root so it never gets confused with its neighborhood. I can either *learn* ε by gradient descent, or fix ε=0 and make the update a plain sum over {self}∪neighbors, i.e. add a self-loop. The learned-ε version is the one tied directly to the injectivity argument; the fixed-0 version is the simpler practical variant and a useful ablation of how much that center tag matters.

Putting it together, the layer is: sum the neighbors, add (1+ε) times the center, run the result through an MLP.

    h_v^{(k)} = MLP^{(k)}( (1+ε^{(k)})·h_v^{(k-1)} + Σ_{u∈N(v)} h_u^{(k-1)} ).

One MLP per layer is enough even though my theory used two abstract functions f and φ: the next layer's f^{(k+1)} can be folded into this layer's φ^{(k)} since an MLP represents a composition of functions, so I let one MLP model f^{(k+1)}∘φ^{(k)}. And in the very first layer, if the inputs are one-hot, the sum of one-hots is already injective (it's a count vector), so I don't even need an MLP before that first summation.

Now the graph-level readout. Theorem says I need it injective over the multiset of node features, and I just proved sum is the injective multiset aggregator — so sum-pool the node features. But there's a wrinkle about *which* layer's features to pool. As k grows, node representations get more global (bigger subtrees) and more discriminative — I want enough iterations for power. Yet the deepest features are also the most specialized; sometimes the shallower, more local features generalize better to unseen graphs. Rather than gamble on one depth, use *all* of them: sum-pool the node features at each layer separately, then concatenate across layers,

    h_G = CONCAT( READOUT({ h_v^{(k)} : v∈G }) : k = 0,1,…,K ),

with READOUT a sum. This is the jumping-knowledge idea — let the final representation reach back to every depth. Each per-layer sum-readout is injective on its multiset, and concatenation never merges what its components keep apart (if two graphs differ in any single layer's pooled vector, they differ in the concatenation), so the all-depth readout is at least as discriminating as the best single layer — including the deepest one I'd otherwise have to commit to. And because a node's height-k feature is a learned embedding of a height-k rooted subtree, summing those features over the graph is a learnable analogue of counting subtrees — structurally the same histogram the WL subtree kernel forms by hand, except the subtrees are embedded into a continuous space so *similar* subtrees land near each other, which one-hot WL labels can't do. I'll hold the "generalizes the WL subtree kernel" claim as a structural correspondence rather than a theorem — the kernel's exact feature is a hard count and mine is a learned sum of embeddings, so the precise sense in which one contains the other is something I'd want to state carefully — but the injectivity of each per-layer sum is what I actually need here, and that I've checked.

Let me check the constructed net against the three hypotheses the equivalence proof actually consumed, one at a time, because the theorem only bites if all three hold. Hypothesis one, injective neighbor aggregation: I'm summing f-encoded neighbor features, and I verified above that the f(x)=N^{-Z(x)} sum recovers exact counts on a concrete pair — so the aggregate is injective on bounded neighbor multisets. Hypothesis two, injective combine over (center, neighbor-aggregate): I'm using (1+ε)·f(center) + Σ_neighbors with ε irrational, and I both proved and watched it separate the a–b–b / b–a–b roots that the flat sum merged — injective. Hypothesis three, injective readout over the node-feature multiset: each per-layer sum-pool is the same multiset sum, injective by the same digit argument. All three hold, so the equivalence applies and this net matches WL's discriminative power, which the upper bound says is the most any message-passing GNN can reach. Conversely, every degraded variant fails a *specific* hypothesis with a *specific* witness I computed: mean instead of sum fails aggregation injectivity (it merged {g,r} with {g,g,r,r}, both 5.5); max fails it harder (it merged {g,r} with {g,r,r}, both 10); one linear+ReLU fails the universal-f requirement (it merged {1,1,1,1,1} with {2,3}, both (8.44,5.16,0,0)); flat-merging the center fails combine injectivity (both roots 0.375). So the four design choices aren't interchangeable knobs — each one is the thing that repairs a hypothesis I watched a simpler choice break.

Now let me write it as real code. The transform is a small MLP (Linear → BatchNorm → ReLU stack, with a plain Linear if I degrade it to one layer for the ablation). The graph net stacks these layers; sum and average pooling become sparse matrix multiplies over a block-diagonal batch adjacency, max pooling uses padded neighbor lists, and the center weighting is the (1+ε) reweight, with ε either a learned parameter or fixed by putting self-loops into the pooled neighborhood. The expressive setting uses sum for both neighbor aggregation and graph pooling; the average and max branches stay in the class because I need them as controlled ablations. The readout pools every layer and pushes each pooled vector through its own linear head; summing those scores is the implementation form of concatenating all depths and then applying one linear classifier.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """The transform that plays the role of f / phi.
    num_layers == 1 is the linear ablation; num_layers >= 2 is the MLP case.
    """
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_layers = num_layers
        self.linear_or_not = True

        if num_layers < 1:
            raise ValueError("number of layers should be positive")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        h = x
        for layer in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
        return self.linears[self.num_layers - 1](h)


class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, center_weighting,
                 graph_pooling_type, neighbor_pooling_type, device):
        super().__init__()
        self.num_layers = num_layers
        self.final_dropout = final_dropout
        self.center_weighting = center_weighting
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.device = device

        # one learned eps per message-passing layer when the center is kept separate
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        self.mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layers - 1):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.mlps.append(MLP(num_mlp_layers, in_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linears_prediction = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.linears_prediction.append(nn.Linear(in_dim, output_dim))

    def preprocess_neighbors_for_maxpool(self, batch_graph):
        max_deg = max(graph.max_neighbor for graph in batch_graph)
        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            for j, neighbors in enumerate(graph.neighbors):
                pad = [n + start_idx[i] for n in neighbors]
                pad.extend([-1] * (max_deg - len(pad)))
                if not self.center_weighting:
                    # eps fixed to 0: pool the center together with its neighbors.
                    pad.append(j + start_idx[i])
                padded_neighbor_list.append(pad)

        return torch.LongTensor(padded_neighbor_list).to(self.device)

    def preprocess_neighbors_for_matrix_pool(self, batch_graph):
        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])

        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        if not self.center_weighting:
            num_node = start_idx[-1]
            self_loop = torch.arange(num_node, dtype=torch.long)
            self_loop_edge = torch.stack([self_loop, self_loop])
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, torch.ones(num_node)], 0)

        Adj_block = torch.sparse.FloatTensor(
            Adj_block_idx, Adj_block_elem,
            torch.Size([start_idx[-1], start_idx[-1]]))
        return Adj_block.to(self.device)

    def preprocess_graph_pool(self, batch_graph):
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx, elem = [], []
        for i, graph in enumerate(batch_graph):
            if self.graph_pooling_type == "average":
                elem.extend([1.0 / len(graph.g)] * len(graph.g))
            else:
                elem.extend([1.0] * len(graph.g))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1])])

        idx = torch.LongTensor(idx).transpose(0, 1)
        elem = torch.FloatTensor(elem)
        graph_pool = torch.sparse.FloatTensor(
            idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        # Padded -1 indices select this dummy row, which cannot affect the max.
        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape(1, -1).to(self.device)])
        return torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]

    def next_layer_with_center_weighting(self, h, layer,
                                         padded_neighbor_list=None, Adj_block=None):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(
                    Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        pooled = pooled + (1 + self.eps[layer]) * h
        pooled_rep = self.mlps[layer](pooled)
        return F.relu(self.batch_norms[layer](pooled_rep))

    def next_layer(self, h, layer, padded_neighbor_list=None, Adj_block=None):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                degree = torch.spmm(
                    Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled / degree

        pooled_rep = self.mlps[layer](pooled)
        return F.relu(self.batch_norms[layer](pooled_rep))

    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.preprocess_graph_pool(batch_graph)

        padded_neighbor_list, Adj_block = None, None
        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.preprocess_neighbors_for_maxpool(batch_graph)
        else:
            Adj_block = self.preprocess_neighbors_for_matrix_pool(batch_graph)

        hidden_rep = [X_concat]
        h = X_concat
        for layer in range(self.num_layers - 1):
            if self.center_weighting:
                h = self.next_layer_with_center_weighting(
                    h, layer, padded_neighbor_list, Adj_block)
            else:
                h = self.next_layer(h, layer, padded_neighbor_list, Adj_block)
            hidden_rep.append(h)

        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            pooled_h = torch.spmm(graph_pool, h)
            score_over_layer += F.dropout(
                self.linears_prediction[layer](pooled_h),
                self.final_dropout, training=self.training)
        return score_over_layer
```

The causal chain, end to end: I start by asking for the ceiling on what a message-passing GNN can distinguish, recognize that the loop is structurally the WL test, and get the upper bound that no such GNN beats WL. The proof leaks power exactly where lossy functions collapse distinct inputs, so reaching WL requires aggregation, combine, and readout to be injective on multisets. Sum is the one of {sum, mean, max} that keeps the full multiset; mean keeps only the distribution, max only the support, and each failure has an explicit confusing pair. A sum of learned features is universal over bounded multisets; a single linear+ReLU degenerates to a plain sum and underfits, so the transform has to be a real MLP. Merging the center with its neighbors loses the root, so I tag the center with (1+ε), which is injective for irrational ε and becomes a practical learned parameter or a fixed-zero self-loop variant. Pooling each layer and combining all depths gives the graph readout. Sum aggregation, MLP transform, center weighting, and multi-layer readout are all doing one job: preserve every distinction that WL preserves, while letting the embeddings become learned continuous features instead of hard hashes.
