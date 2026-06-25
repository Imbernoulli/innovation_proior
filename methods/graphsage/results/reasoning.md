OK, let me think this through from scratch. The thing that keeps biting me in practice is this: I have a graph that never stops changing. New Reddit posts every minute, new papers every year, and sometimes a whole new graph shows up — a protein-protein interaction network for an organism I've never trained on. And what I want, in production, is the embedding of a node *right now*, the instant it appears, fed straight into whatever classifier sits downstream. No waiting.

But every node-embedding method I have on the shelf assumes the opposite. Take the random-walk methods — DeepWalk, node2vec, LINE. The recipe is: roll out a bunch of truncated random walks, treat each walk as a "sentence" of node IDs, and train so that nodes co-occurring on a walk have similar dot products, with negative sampling for the ones that don't. Concretely the thing being optimized has the shape α Σ_{i,j∈A} f(zᵢᵀz_j) + β Σ_{i,j∈B} g(zᵢᵀz_j), positives and negatives. It works, it's a strong baseline. But stare at *what the parameters are*. The parameters are the embeddings. There's a table Z ∈ R^{|V|×d}, one free row per node, and training is literally fitting those rows. So when a new node arrives, it has no row. There is nothing to look up. The only way to get its embedding is to add a row and run more SGD until that row settles — and on the inductive task that's painfully slow, a hundred times slower at test time or worse, because "test time" now contains an optimization loop. That's the wall: the method is transductive by *construction*, not by accident. The embedding is a parameter, and parameters don't generalize to inputs you didn't have at train time.

And there's a second, sneakier problem with that whole family, which I want to pin down because it might kill the "just retrain on the bigger graph" escape hatch. The only error signal in α Σ f(zᵢᵀz_j) + β Σ g(zᵢᵀz_j) comes through dot products zᵢᵀz_j. So suppose I take any orthogonal matrix Q and rotate every embedding, z ↦ Qz. Then zᵢᵀz_j ↦ (Qzᵢ)ᵀ(Qz_j) = zᵢᵀQᵀQz_j = zᵢᵀz_j, unchanged, since QᵀQ = I. Let me not just assert this — let me actually try to break it. Take four random embeddings in R⁶ and a random orthogonal Q (QR of a Gaussian matrix), and compare the Gram matrices ZZᵀ and (ZQᵀ)(ZQᵀ)ᵀ entry by entry. The largest absolute difference comes out around 9e-16 — machine epsilon, i.e. exactly zero up to floating point — while QᵀQ deviates from I by the same order. So the objective genuinely cannot see Q; this isn't a hand-wave. Levy and Goldberg's reading makes it starker still: this family is implicitly factorizing some random-walk-statistics matrix M, learning Z with ZZᵀ ≈ M, and ZQᵀQZᵀ = ZZᵀ for any orthogonal Q. So the embedding space is free to sit at an arbitrary rotation. Train on graph A, train separately on graph B, and B's space is rotated arbitrarily relative to A's — a classifier fit on A's coordinates reads B's coordinates as noise. Even on one graph, re-running SGD as it grows lets the whole space spin relative to where it was when I trained the classifier. So cross-graph transfer isn't just slow here; it's ill-posed. (I'll note in passing the one partial escape, Planetoid-I — it's inductive, but it only uses the graph as a *regularizer* during training and never looks at a node's neighborhood at inference, so it doesn't actually exploit graph structure to embed a new node. Not the mechanism I want.) And on top of all this, these methods throw away node features entirely, which is absurd when my nodes come with text, with gene sets, with profile information.

So what would *have* to be true for inductivity? The new node must get its embedding by a *computation on things I can observe about it* — its features, its neighbors — using parameters I already trained and froze. The parameters can't be the embeddings. The parameters have to be a *function*, and the embedding has to be the *output* of that function applied to (features, neighborhood). If I have that, a never-seen node, even in a never-seen graph, just runs the function — no row to add, no SGD to wait on. That reframing — optimize a function that generates embeddings, not a table of embeddings — is at least necessary; whether it's sufficient, and what the function should be, is what the rest of this has to work out.

Now, is there anything on the shelf that already learns a function of features with shared parameters? Yes: the spectral graph convolution line, the GCN rule. It's H^{(l+1)} = σ( D̂^{-1/2} Â D̂^{-1/2} H^{(l)} W^{(l)} ), where Â = A + I adds self-loops and D̂ is Â's degree matrix; it falls out of taking a spectral filter on the graph Laplacian, truncating to first order, and renormalizing. The crucial thing for me is *what's being learned*: the W^{(l)} are weight matrices shared across all nodes, acting on features. That's a function, not a lookup table. So GCN already took the conceptual step I need — it just didn't take it where I need it.

Because look at how that rule is actually evaluated. D̂^{-1/2} Â D̂^{-1/2} H^{(l)} is one big matrix multiply against the *entire* normalized adjacency of the whole graph. Every node participates in every forward pass; the full graph Laplacian has to be in memory; training fits all node representations simultaneously, on one fixed graph, semi-supervised. There's no minibatch over nodes, no notion of "here's a single new node, give me its vector from its own neighborhood." And the dense whole-graph operation is O(|V|)-ish, which doesn't survive a graph with hundreds of thousands of nodes, let alone a brand-new graph. So GCN is transductive in *practice* even though its parameters are inductive in *spirit*. The fix has to be: keep the shared-weights-on-features idea, throw away the whole-graph matrix multiply.

That symmetric-normalized multiply, when I unfold it for a single node, is just: each node takes a (degree-)weighted average of its neighbors' representations, mixes in its own, and pushes the result through W and a nonlinearity. There's nothing global about the *meaning* of it — only the *implementation* (the big Â matmul) is global. So write it node-locally. For a node v, aggregate the previous-layer representations of v's immediate neighbors into one vector, combine with v's own previous-layer vector, transform. Then stack these. One layer reaches the 1-hop neighborhood; two layers reach 2 hops (because a neighbor's layer-1 vector already summarizes *its* neighbors); K layers reach K hops. So "depth" is literally "search radius," same as GCN's depth, but now each node only ever needs its own K-hop neighborhood — never the whole graph. The forward pass becomes:

start with h⁰_v = x_v, the raw features. Then for k = 1 … K, for each node v:
  h^k_{N(v)} = AGGREGATE_k({ h^{k-1}_u : u ∈ N(v) }),
  h^k_v = σ( W^k · CONCAT( h^{k-1}_v , h^k_{N(v)} ) ),
and finally normalize. Output z_v = h^K_v. The intuition is exactly the iterative one: at each round nodes pull in their neighbors, and as it iterates they pull from further and further out. (This shape isn't new in spirit — it's the same loop the Weisfeiler-Lehman vertex-refinement test runs, and the old graph-neural-network and structure2vec works do message passing too; but those aimed at labeling whole graphs for isomorphism or classification, not at producing a per-node embedding that generalizes to unseen nodes. I'm bending the loop to my purpose.)

Two choices in that forward pass I do not want to make on autopilot. First, the CONCAT. The lazy thing would be to fold v's own vector into the same averaging pot as its neighbors and run one W over the lump — that's basically what GCN does with its self-loop Â = A + I. But think about what happens over K layers if I do that: v's own signal gets averaged in with a growing crowd of neighbors-of-neighbors and washed out; by layer K, v's identity is diluted. If instead I keep h^{k-1}_v *separate* and concatenate it with the aggregated neighborhood before W, then v always has a clean, undiluted channel for its own representation carried forward. It's a skip connection between search depths — the node's own information shortcuts past the aggregation at every layer. I expect this to matter, and the natural ablation is exactly the fold-it-in version; I'd bet the concat wins, and meaningfully. Second, the per-layer normalization. After computing h^k_v I divide by its ℓ2 norm. Why: across K layers, repeated linear maps plus aggregation can let the magnitudes drift and blow up or collapse; normalizing to the unit sphere keeps everything comparable layer to layer. And it dovetails with the loss I'm about to build, which compares nodes by dot product — on the unit sphere a dot product is a cosine, so the loss is comparing *directions*, which is what I actually mean by "similar."

Now AGGREGATE itself. This is the real design surface, and there's a hard constraint that rules out the obvious. A node's neighbors are a *set* — there is no first neighbor, no canonical order. Whatever I do to them must give the same answer under any permutation; otherwise the embedding of a node depends on the arbitrary order I happened to enumerate its adjacency list, which is nonsense, and worse, an unseen node enumerated in a new order would get a different vector than the same node enumerated another way. So AGGREGATE must be symmetric (permutation-invariant), and I'd also like it trainable and expressive. Those three pull against each other, so let me try candidates.

The cheapest symmetric thing: the mean. h^k_{N(v)} = elementwise mean of { h^{k-1}_u }. Symmetric, trivially. And notice — if I take the mean *including* v itself and run a single shared W over it (i.e. drop the concat, fold self in), I get h^k_v = σ( W · MEAN({h^{k-1}_v} ∪ {h^{k-1}_u : u ∈ N(v)}) ), which looks an awful lot like the inductive face of the GCN convolution: a node mixes its own vector with its neighbors', linear-maps, nonlinearity. I want to check how literally that "is GCN" — because if the mean aggregator *exactly* reproduces GCN I should be careful about claiming novelty, and if it doesn't I should know where they diverge. So take a concrete small graph: a triangle on {0,1,2} with a pendant node 3 hanging off 0, one-hot features, W = I, σ dropped. For node 0, fold-self-in mean averages over {0,1,2,3}, giving mixing weights (¼, ¼, ¼, ¼) — uniform. Now the actual GCN row: Â = A+I, D̂ its degrees, and row 0 of D̂^{-1/2} Â D̂^{-1/2} comes out (0.25, 0.289, 0.289, 0.354). Those are *not* a scalar multiple of the uniform vector — the ratios are 1, 1.155, 1.155, 1.414. So my "up to a normalization constant" intuition is wrong as stated: GCN weights each edge by deg_i^{-1/2} deg_j^{-1/2}, putting *more* weight on low-degree neighbors (here the pendant 3), whereas the mean aggregator weights everyone uniformly. They coincide in *form* — degree-aware self+neighbor averaging then shared W — not in the exact coefficients. So the honest statement is: the mean aggregator is the same *shape* of operation as GCN, a sibling rather than a strict generalization; GCN's particular symmetric normalization is one more design choice I'm choosing not to copy. Good — that also tells me a GCN-style baseline is worth keeping as its own aggregator variant, not folded away. But either way the mean is a fixed, linear pooling — no parameters of its own, low capacity. It can't, for instance, pick out "is there *any* neighbor with property X"; it always blurs.

Can I do better while staying symmetric? I want something trainable that can highlight distinctive features of the neighbor set rather than averaging them away. Idea: push *each* neighbor's vector through its own shared little network first, then combine with a symmetric pool. If the combine is an elementwise max, the whole thing is still permutation-invariant (max doesn't care about order), but now each output dimension can act like "does any neighbor strongly trigger this learned feature?" — a max is a soft existential over the set, where the mean was a blur. So:

  AGGREGATE^pool = max({ σ( W_pool · h^{k-1}_u + b ) : u ∈ N(v) }),

max elementwise. This is lifted straight from how people learn over unordered point sets — per-point MLP, then symmetric pool — and it's both symmetric and trainable and high-capacity. (Max versus mean for the pool? I'd try both; I expect them close, and if so I keep max for the existential semantics.) Of the three this is the one I'd bet on, and the existential semantics are the reason — the max can isolate a single distinctive neighbor that the mean would drown.

One more candidate, for capacity's sake: an LSTM. LSTMs are very expressive sequence models. The problem is glaring — an LSTM is the opposite of symmetric; it reads its inputs in order and the order matters. That violates my hard constraint. But I can paper over it: feed the LSTM a *random permutation* of the neighbors each time, so over training it sees many orderings and is pushed to be roughly order-agnostic. It's a hack, not a guarantee, and it'll be the slowest of the three. Worth keeping as a high-capacity option to see if the extra expressiveness pays off despite the broken symmetry.

So I have three aggregators — mean (cheap, ≈ inductive GCN as the no-concat special case), pool (symmetric, trainable, my favorite), LSTM (expressive but order-hacked, slow) — all dropping into the same AGGREGATE slot.

Now the scaling problem I waved at earlier comes back to bite, and it's about N(v). If N(v) means "all of v's neighbors," then my per-batch cost is at the mercy of the degree distribution. Real graphs are heavy-tailed: there are hub nodes with enormous degree, and a Reddit post-to-post graph has average degree in the hundreds. One hub in my batch, or one hub two hops out, and the support set explodes — worst case I'm dragging in O(|V|) nodes to process a single batch, and memory per batch is unpredictable. That defeats the entire point of going node-local. So redefine N(v): instead of the full neighbor set, draw a *fixed-size uniform sample* of neighbors, of size S_k, and draw a *fresh* sample at each depth k. (Fresh per depth so the samples don't correlate across layers.) If a node has fewer neighbors than S_k, sample with replacement to keep the tensor shape fixed. Now the per-batch footprint is pinned: with K layers and sample sizes S_1 … S_K, the work is O(∏_{i=1}^K S_i), a constant I choose, no matter how hub-ridden the graph is. I pay variance — I'm subsampling neighborhoods — but I buy predictable, bounded batches, which is the only way this runs on a 200K-node graph.

How big should K and the S_k be? K is search radius. K=1 only sees immediate neighbors and can miss the information that arrives when a neighbor has first summarized *its* neighbors. Each extra hop, though, multiplies the support size by S_k — cost compounds geometrically. So I want the *smallest* K that captures enough structure. K=2 is the first serious choice: it lets the target see a sampled two-hop ego network while keeping the support size to roughly S_2 immediate neighbors and S_1·S_2 two-hop neighbors from the target's point of view. Modest samples such as S_1=25 at the first algorithm layer and S_2=10 at the second already make the total support small enough to fit in ordinary minibatches.

There's a subtlety in *how* I actually run this in minibatches that I have to get right, because the recursion as written is top-down (compute layer 1, then layer 2…) but the dependency is bottom-up. I want representations for a batch B of target nodes. To compute their layer-K vectors I need their sampled neighbors' layer-(K−1) vectors, which need *their* sampled neighbors' layer-(K−2) vectors, and so on. So I first go *backward*: start with B^K = B, and for k = K down to 1, expand B^{k−1} = B^k ∪ (sampled neighbors of every node in B^k). That collects exactly the nodes I'll need — B^K ⊆ B^{K−1} ⊆ … ⊆ B^0 — and nothing else. Then I go *forward*: set h⁰ = features on B^0, and for k = 1 … K aggregate and transform over B^k. At each forward step the inputs I need (layer-(k−1) of a node and of its sampled neighbors) were computed in the previous step, so it all closes up, and I never touch a node outside the batch's K-hop sampled support. One counterintuitive bookkeeping note: because the sampling is indexed from the target's side, "sample S_1 at iteration 1 and S_2 at iteration 2" means a target node ends up with S_2 immediate neighbors and S_1·S_2 two-hop neighbors. Fine, as long as I keep the indices straight.

Now the loss — and this is where the inductive reframing has to prove it didn't break anything. I want the old, good property: nearby nodes get similar embeddings, distant nodes get distinct ones, no labels required so the embeddings are task-agnostic and reusable. The random-walk methods got this with a SkipGram negative-sampling objective. But their objective was over a *parameter table* z_u; I have no table — my z_u is *generated* by the functions from features and neighborhood. So I just take the same objective shape and feed it my generated z's instead of looked-up ones:

  J(z_u) = − log σ(z_uᵀ z_v) − Q · E_{v_n ∼ P_n(v)} [ log σ(− z_uᵀ z_{v_n}) ],

where v is a node that co-occurs with u on a fixed-length random walk (the "positive," a nearby node), σ is the sigmoid, P_n is a negative-sampling distribution over nodes, and Q is the number of negatives. Read it: the first term, −log σ(z_uᵀz_v), is small when z_uᵀz_v is large and positive — it *pulls* co-occurring nodes' embeddings together. The second term, −Q·E log σ(−z_uᵀz_{v_n}), is small when z_uᵀz_{v_n} is large and *negative* — it *pushes* random non-neighbors apart. The only — but decisive — difference from DeepWalk is that z_u here is the output of the aggregator functions, so the gradient flows back into W^k, W_pool, the LSTM weights — the *functions* — not into a per-node vector. The structure that pulls nearby-similar/far-distinct is now baked into a generator that an unseen node can simply run. For P_n I'll use the standard degree^{3/4} smoothing from the word-embedding world, with on the order of Q=20 negatives — the recipe that's been tuned to death already, no reason to reinvent it. And if I happen to care only about one downstream task, I can swap J for a plain supervised cross-entropy on the labels; the generator is the same either way.

The inductive mechanics are in place, but something still nags at me: this thing is fundamentally *feature*-driven — it aggregates feature vectors. Can it actually learn anything about graph *structure*, or is it secretly just smoothing features around? Let me push on this. Set K = |V|, all weight matrices to the identity, the aggregator to a hashing function, and no nonlinearity. Then my recursion is exactly: each node's new label is a hash of the multiset of its neighbors' labels, iterated. That *is* the Weisfeiler-Lehman vertex-refinement isomorphism test, "naive vertex refinement" — the classic test that, by iterating neighbor-aggregation-then-hash, assigns distinct labels to structurally distinct nodes for a broad class of graphs (it can fail on some, but it's valid widely). So my architecture is, on the nose, a continuous and trainable relaxation of WL: replace the hash with learned neural aggregators. That's not a proof of anything yet, but it tells me the *shape* of my computation is exactly the shape of a known structure-detector, which makes me think a precise structural guarantee should be reachable.

Let me try to make it precise — can the method actually compute a genuine structural quantity? Pick a concrete, nontrivial one: the clustering coefficient c_v, the fraction of possible edges among v's neighbors that actually exist, c_v = 2·(#edges among v's neighbors) / (d_v(d_v−1)). It's a real structural measure, the building block of higher-order motifs, and it has nothing to do with features per se. Claim I want to establish: if every node has a sufficiently distinct feature vector, then there's a parameter setting of my pool-based architecture that approximates every node's clustering coefficient to arbitrary precision. Note the flavor — this is an *identifiability / existence* statement (a parameter setting exists), not a claim about whether SGD finds it efficiently; that's fine, it tells me the representational power is there. Assume there's a constant C > 0 with ‖x_v − x_{v'}‖₂ > C for all pairs of nodes — all features pairwise separated.

Two facts give me the leverage. A max over per-element MLPs can approximate any Hausdorff-continuous symmetric function on a set to arbitrary ε (that's the point-set result), and a single-hidden-layer MLP can approximate any continuous function to arbitrary ε (Hornik).

First I need a "bump" function that fires on chosen anchor points and is silent far from them. Lemma: for a fixed C > 0 and any finite set of nodes D, there's a continuous g : U → R with g(x) > ε whenever x equals some anchor x_v (v ∈ D) and g(x) ≤ −ε whenever x is farther than C from *every* anchor. I should not reuse the final tolerance as the internal margin, because with the common 3|D| coefficient the singleton case lands exactly on that margin. So I pick η with ε < η < 0.5 and build the separator with η. Let d_v = ‖x − x_v‖₂, and set g(x) = Σ_{v∈D} g_v(x) with

  g_v(x) = 3|D|η / (b d_v² + 1) − 2η,   where b = (3|D| − 1)/C² > 0.

Check the properties. At d_v = 0, g_v = (3|D| − 2)η. As d_v → ∞, g_v → −2η and never drops below that limit. At d_v = C: b·C² + 1 = (3|D|−1) + 1 = 3|D|, so g_v(C) = η − 2η = −η; and since g_v decreases in d_v, g_v ≤ −η for all d_v ≥ C. Now the sum. If x is far from *all* anchors, every term is at most −η, so g(x) ≤ −|D|η ≤ −η < −ε. If x hits some anchor exactly, split the sum:

  g(x) = g_v(x) + Σ_{v'≠v} g_{v'}(x) ≥ (3|D| − 2)η − 2(|D|−1)η = |D|η ≥ η > ε.

For |D|>1 the non-anchor terms are actually greater than their −2η limits at finite distances; for |D|=1 there are no non-anchor terms and g(x)=η, still above ε. And g is continuous, being a finite sum of continuous functions. That settles the bump function I needed with a real margin on both sides.

Second, by Hornik, this g can be approximated to arbitrary precision by a standard single-hidden-layer MLP with a non-constant monotone activation (ReLU is fine). I choose the approximation error δ < η − ε, so f_θ stays positive on anchors and negative outside the C-balls. This is just the universal approximation theorem, but the margin bookkeeping matters.

Third — the structural payoff. I want to assign every node a locally unique one-hot indicator vector, robustly, using the same kind of learned functions. How many dimensions do I need? I only need two nodes to be distinguishable if they can co-occur in some node's 2-hop neighborhood, because those are the nodes whose indicators may collide in the rows I use to count edges. If two nodes both lie inside N²(u), their graph distance is at most four through u. So I take the fourth graph power G⁴: put an edge between any two distinct nodes whose distance in G is at most four, and let χ(G⁴) be its chromatic number. A proper coloring of G⁴ guarantees that no two nodes co-occurring in any 2-hop neighborhood share a color. Equivalently, χ(G⁴) dimensions suffice to give every such pair distinct one-hot vectors. Each color is a subset D ⊆ V that can all map to the *same* indicator without conflict. Now use the bump separator: with features pairwise more than C apart, for each color subset D there is an MLP whose output is positive on D and negative outside D. One more ReLU turns that into a nonnegative color detector. Build χ(G⁴) such detectors, one per color, and after normalization the one active coordinate becomes an indicator. So with at least two hidden layers in the first local function, I get h¹_v as one-hot indicators in dimension χ(G⁴), distinct for every pair of nodes co-occurring in a 2-hop neighborhood. That's the part that needs the pool-style universal approximation; the mean can't manufacture these selectors.

Now assemble the clustering coefficient. Use the pool aggregator at all depths, four iterations. From Lemma 3, after depth k=1 every node in v's 2-hop neighborhood has a unique one-hot indicator h¹_v. Set the weight matrices at the next depths to identity and aggregators to *sum* the unnormalized neighbor representations (the normalization constant I can always recover by having the aggregator prepend a unit value and inverting it at a later layer — so ignore normalization in the bookkeeping). At depth k=2: each node concatenates its own indicator with the sum of its neighbors' indicators, which is exactly its row of the adjacency matrix, so h²_v = h¹_v ⊕ A_v, where A_v is v's adjacency row (over the relevant subgraph) and ⊕ is concatenation. At depth k=3: sum the neighbors' h² and concatenate,

  h³_v = h¹_v ⊕ A_v ⊕ ( Σ_{u ∈ N(v)} h¹_u ⊕ A_u ).

Slice h³_v with m = χ(G⁴) the indicator dimension: a ≡ h³_v[0:m] is v's own one-hot indicator; b ≡ h³_v[m:2m] is v's adjacency row A_v; and c ≡ the block holding Σ_{u∈N(v)} A_u, the sum of v's neighbors' adjacency rows. Since b has ones exactly at v's neighbors and zero at v itself, bᵀc is

  Σ_{i∈N(v)} Σ_{u∈N(v)} A_{u i}.

For an undirected simple graph, each edge among v's neighbors appears twice in that double sum, once in each direction, and v's own incident edges do not appear because b_v = 0. So if t_v is the number of edges induced by N(v), then bᵀc = 2t_v and the clustering coefficient is

  bᵀc / (d_v(d_v − 1)) = 2t_v / (d_v(d_v − 1)) = c_v.

This identity is the load-bearing step, so I should not trust the index-chasing on faith — let me actually evaluate bᵀc on a small graph and confront it with a direct edge count. Take five nodes with edges {0-1, 0-2, 0-3, 1-2, 1-4, 3-4}. For v=0, N(0)={1,2,3}; b=A_0 has ones at 1,2,3; c = A_1+A_2+A_3. Computing bᵀc by hand-machine gives 2, and the only edge inside N(0) is 1-2, so t_0=1 and 2t_0=2 — they agree, and c_0 = 2/(3·2) = ⅓. Running the same over every node: v=1 has N={0,2,4} with the single edge 0-2, bᵀc=2=2·1, c_1=⅓; v=2 has N={0,1}, edge 0-1 present, bᵀc=2=2·1, c_2=1 (a closed triangle, as it should be); v=3 has N={0,4}, no edge between them, bᵀc=0=2·0, c_3=0; v=4 has N={1,3}, no edge, bᵀc=0, c_4=0. Every case matches, including the degree-2 triangle (c=1) and the open wedges (c=0). And I can read off the self-exclusion directly: b_v = A_{vv} = 0 because the graph is simple, so v's own incident edges never enter the count — which is exactly why no subtraction by d_v belongs here. Subtracting d_v would be correcting for closed-neighborhood self-incidence that b does not include. This is a continuous (Lipschitz on nodes with d_v ≥ 2, with isolated and degree-one nodes handled by the usual convention c_v = 0) function of h³_v, so by Hornik one more layer — a single-hidden-layer MLP, equivalently a fourth iteration that ignores neighbors — approximates it to within ε on the relevant domain. So after K=4 iterations (or K=3 plus a final MLP), |z_v − c_v| < ε for all v. Tracing the construction end to end, each link held up under its own check — the bump margins, the small-graph bᵀc count, the index slicing — so I'm now fairly confident in the conclusion I was only conjecturing earlier: the feature-based aggregator really can compute a genuine structural quantity to arbitrary precision, and the place it leans hardest is the pool aggregator's symmetric universal approximation in the indicator step.

And it almost-surely holds for free in the generic case: if I draw features from any distribution absolutely continuous w.r.t. Lebesgue measure, the probability that two nodes coincide is zero, so the pairwise-distinctness assumption holds almost surely — the corollary. The structure-learning ability doesn't even require hand-crafted features. (It does lean on the pool aggregator's universal symmetric power throughout, which is again why I'd expect the GCN/mean variants to be less robust when features are weak or replaced by noise.) The whole construction is conservative — fewer iterations or weaker assumptions might suffice, and the worst-case dimension χ(G⁴) can be as large as O(|V|), but that's no worse than a featureless GCN's O(|V|) parameters, so it's not an unreasonable price for an existence result.

The code now follows the derivation directly: a sampling-aware aggregator, an encoder that does the concat-then-W-then-ReLU layer, two stacked encoders for K=2, an unsupervised graph loss, and a supervised head.

Before I trust it, the one property the whole effort exists for is inductivity, and that's cheap to test directly — so let me build a four-node toy graph (a triangle {0,1,2} plus a pendant 3), stack two mean encoders, run a forward pass, and then *freeze every parameter*. Node 2, present at "train" time, comes out of enc2 as a unit vector. Now I introduce a node id 99 that was never in the graph or the feature table when the weights were set: I give it its own feature row and a single edge to node 2, and call enc2([99]). It returns a perfectly good unit-norm embedding — a different direction than node 2's, shaped by 99's own features and its one neighbor — with no parameter row for 99 anywhere and no optimization step taken on its behalf. That's the transductive wall from the very first paragraph actually gone: the random-walk table had nothing to look up for a new node, and here the answer is a single feed-forward pass over frozen functions. Seeing it run on an unseen id is the check that the function-not-table reframing did what it was supposed to.

```python
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def _node_list(nodes):
    return [int(n) for n in nodes]


def _sample(neigh, fallback, num_sample):
    neigh = list(neigh)
    if not neigh:
        neigh = [fallback]
    if num_sample is None:
        return neigh
    if len(neigh) >= num_sample:
        return random.sample(neigh, num_sample)
    return random.choices(neigh, k=num_sample)


# Aggregate an unordered neighbor set, with fixed-size uniform sampling so the
# per-batch cost is bounded regardless of hub-node degrees.
class MeanAggregator(nn.Module):
    def __init__(self, features, gcn=False):
        super().__init__()
        self.features = features   # node ids -> previous-layer reprs h^{k-1}
        self.gcn = gcn             # fold self into the mean (=> inductive GCN) vs. keep separate
        self.out_dim = None

    def forward(self, nodes, to_neighs, num_sample=10):
        nodes = _node_list(nodes)
        samp = [_sample(neigh, nodes[i], num_sample) for i, neigh in enumerate(to_neighs)]
        if self.gcn:                                   # GCN special case: self in the pot
            samp = [s + [nodes[i]] for i, s in enumerate(samp)]
        uniq = sorted(set().union(*samp))
        idx = {n: i for i, n in enumerate(uniq)}
        mask = torch.zeros(len(samp), len(uniq))
        for row, neigh in enumerate(samp):
            for n in neigh:
                mask[row, idx[n]] += 1.0
        mask = mask / mask.sum(1, keepdim=True).clamp_min(1.0)
        embed = self.features(torch.as_tensor(uniq, dtype=torch.long))
        return mask.to(embed.device).mm(embed)

# Pool aggregator: per-neighbor MLP then elementwise max — symmetric, trainable,
# and the high-capacity / theory-backed choice.
class PoolAggregator(nn.Module):
    def __init__(self, features, in_dim, hidden=None):
        super().__init__()
        self.features = features
        self.out_dim = in_dim if hidden is None else hidden
        self.mlp = nn.Linear(in_dim, self.out_dim)

    def forward(self, nodes, to_neighs, num_sample=10):
        nodes = _node_list(nodes)
        out = []
        for i, neigh in enumerate(to_neighs):
            samp = _sample(neigh, nodes[i], num_sample)
            h = self.features(torch.as_tensor(samp, dtype=torch.long))
            out.append(F.relu(self.mlp(h)).max(0).values)
        return torch.stack(out, 0)

# One depth: aggregate sampled neighbors, CONCAT with self (skip connection across
# depths), transform by shared W, ReLU, then ell2-normalize each node column.
class Encoder(nn.Module):
    def __init__(self, features, feat_dim, embed_dim, adj_lists, aggregator,
                 num_sample=10, gcn=False):
        super().__init__()
        self.features, self.adj_lists = features, adj_lists
        self.aggregator, self.num_sample, self.gcn = aggregator, num_sample, gcn
        self.embed_dim = embed_dim
        neigh_dim = feat_dim if aggregator.out_dim is None else aggregator.out_dim
        in_dim = neigh_dim if gcn else feat_dim + neigh_dim
        self.weight = nn.Parameter(torch.empty(embed_dim, in_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        nodes = _node_list(nodes)
        neigh = self.aggregator(nodes, [self.adj_lists[n] for n in nodes],
                                self.num_sample)        # h^k_{N(v)}
        if not self.gcn:                                # h^k_v = sigma(W . concat(h^{k-1}_v, h^k_N))
            self_feats = self.features(torch.as_tensor(nodes, dtype=torch.long))
            combined = torch.cat([self_feats, neigh], dim=1)
        else:                                           # GCN variant: no concat
            combined = neigh
        h = F.relu(self.weight.to(combined.device).mm(combined.t()))
        return F.normalize(h, p=2, dim=0, eps=1e-12)

# Unsupervised graph-based loss on GENERATED embeddings in row form [batch, dim].
# If an encoder returns [dim, batch], pass enc(nodes).t(). Negatives may be
# [batch, Q, dim] or shared [Q, dim].
def unsupervised_loss(z_u, z_v, z_neg, Q=None):
    pos = F.logsigmoid((z_u * z_v).sum(1))
    if z_neg.dim() == 2:
        neg_scores = z_u.mm(z_neg.t())
    else:
        neg_scores = (z_u.unsqueeze(1) * z_neg).sum(2)
    q = neg_scores.size(1) if Q is None else Q
    neg = F.logsigmoid(-neg_scores).mean(1)
    return -(pos + q * neg).mean()

class SupervisedGraphSAGE(nn.Module):
    def __init__(self, num_classes, enc):
        super().__init__()
        self.enc = enc
        self.weight = nn.Parameter(torch.empty(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        return self.weight.to(embeds.device).mm(embeds).t()

    def loss(self, nodes, labels):
        return F.cross_entropy(self.forward(nodes), labels.squeeze())

# K=2: stack two encoders. enc2 aggregates over enc1(n).t(), so enc1's
# [dim, batch] output is transposed back to [batch, dim] as a feature table.
def build(features, feat_dim, adj_lists, num_classes, gcn=False):
    agg1 = MeanAggregator(features, gcn=gcn)
    enc1 = Encoder(features, feat_dim, 256, adj_lists, agg1, num_sample=10, gcn=gcn)
    enc1_rows = lambda n: enc1(n).t()
    agg2 = MeanAggregator(enc1_rows, gcn=gcn)
    enc2 = Encoder(enc1_rows, enc1.embed_dim, 256, adj_lists, agg2,
                   num_sample=25, gcn=gcn)
    return SupervisedGraphSAGE(num_classes, enc2)
```

The causal chain, start to finish: I needed embeddings for nodes (and graphs) unseen at train time, but every method on the shelf optimizes a per-node table and is transductive by construction — and its dot-product objective is even rotationally ambiguous across graphs. GCN had already moved to shared weights on features, but trapped that good idea inside a whole-graph Laplacian multiply. So I rewrote the convolution as a local recursion — each node aggregates a *fixed-size uniform sample* of its neighbors and concatenates with its own carried-forward representation, ℓ2-normalized, stacked K=2 deep — which turns the learned object into a small set of *aggregator functions* an unseen node can simply run. I made the aggregator permutation-invariant (mean as the cheap option — the same *shape* as the GCN convolution though, as I checked on a small graph, not its exact normalization; a per-neighbor-MLP-then-max pool as the expressive, theory-backed choice; an order-randomized LSTM as the high-capacity option), trained it either with a supervised loss or with a SkipGram negative-sampling loss applied to the *generated* embeddings, and worked out — via a coloring argument and two universal-approximation theorems, with the bᵀc identity checked node-by-node on a small graph — that the pool variant can recover genuine structural quantities like the clustering coefficient to arbitrary precision, which is why feature-based aggregation can still learn about graph structure rather than merely smoothing features.
