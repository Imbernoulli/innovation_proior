Let me start from what actually goes wrong when I try to generate graphs. I have a training set of graphs, each with categorical node attributes and categorical edge attributes — and "no edge" is just one more edge category — and I want to sample fresh graphs that match the statistics of the data: degree distribution, clustering, the counts of small substructures. Two facts about graphs sit underneath everything I'll do, and I have to respect both or I lose before I start. A graph has no canonical node order: the same graph is one of up to n! adjacency matrices, so if I treat the adjacency as a fixed array and learn over it, I'll waste all my capacity memorizing orderings, and the likelihood — which is properly the sum over all n! permutations — is intractable. And a graph is sparse and discrete: O(n) edges out of O(n²) slots, attributes that are classes rather than numbers, and the thing I actually care about is the *structure* — which edges exist, how cycles and connected components are arranged. So whatever I build has to be permutation equivariant with a permutation-invariant training signal, and it has to keep the objects discrete and structured.

What's the strongest generative recipe available to borrow? Diffusion. On images and audio it's beaten everything. The shape of it: a fixed forward process that slowly corrupts a clean datapoint into noise through a Markov chain, q(z¹,…,z^T | x) = q(z¹|x) ∏ q(z^t|z^{t-1}), and a learned network that inverts one step at a time; to sample I draw from a simple prior and denoise repeatedly. And there's a subtlety I want to keep, because it's the whole reason diffusion trains well: you do *not* train the network to predict z^{t-1}. That's a terrible target — it's high-variance, it depends on the particular noise you happened to sample on the way down. Instead, when the posterior q(z^{t-1} | z^t, x) is tractable, you use the *clean* x as the regression target and reconstruct the reverse step analytically from it. That removes a huge amount of label noise. DDPM does this by predicting the clean x₀ (or equivalently the noise) and rebuilding the reverse mean μ̃_t(x_t, x₀) in closed form.

So what do I need for any diffusion model to be efficient? Three properties, and let me hold them in front of me because they'll dictate every design choice. First, q(z^t | x) must have a closed form, so I can train every timestep in parallel without unrolling the chain — sample a random t, jump straight to z^t, learn. Second, q(z^{t-1} | z^t, x) must have a closed form, so I can use x as the target and recover the reverse step from the prediction. Third, the limit q_∞ = lim q(z^T|x) must not depend on x, so I can use it as the sampling prior. For Gaussian noise all three hold cleanly: q(x_t|x₀) = N(√ᾱ_t x₀, (1−ᾱ_t)I), the posterior is Gaussian, the limit is N(0,I).

The obvious move, then, is the one already on the table: embed the graph in a continuous space — one-hot the node types, take the adjacency as a real matrix — add Gaussian noise to everything, and learn the score. That's exactly what the score-based graph models do. So let me actually run the forward process in my head and watch what happens to a graph. At t=0 I have a clean sparse graph: an adjacency that's almost all zeros with a few ones, attributes one-hot. I add a little Gaussian noise. A bit more. By the middle of the chain every entry of the adjacency is some real number around the noise scale — the matrix is *dense*. And now stop and ask: what is the degree of node i in this object? What's the number of connected components? How many triangles? None of these are defined. There are no edges, just a fog of continuous values. The sparsity that *was* the data is gone, destroyed by the very first property I wanted — a tractable forward marginal — because Gaussian noise spreads mass everywhere. The denoiser is staring at a blurry dense tensor and being asked to recover a sparse graph from it, and there's no intermediate object on which "this is a graph with this structure" even makes sense. That's the wall. Worse, it has a downstream cost I'll come back to: I'd love to *help* the denoiser by feeding it structural descriptors — cycle counts, spectral features — but you cannot compute a cycle count on a fog. Continuous diffusion forecloses that whole avenue. So Gaussian noise is the wrong noise for graphs. The problem isn't diffusion; it's that I continuized a thing that was discrete.

So don't continuize it. Keep the graph discrete the entire way down. Add noise that *edits* the graph — flips an edge in or out, changes a node's or edge's category — but leaves it, at every step, an honest discrete graph. If I can do that, then z^t is always sparse and structured, every descriptor stays defined, and denoising becomes: given a corrupted graph, say what each node and edge should really be. Now, can discrete noise even satisfy my three properties? This is where the discrete-diffusion machinery for text and images comes in. Take a single categorical variable with K classes, one-hot it as a row vector x. The forward step isn't "add noise," it's "multiply by a transition matrix": [Q^t]_{ij} = q(z^t = j | z^{t-1} = i), the probability of category i jumping to category j, so q(z^t | z^{t-1}) = z^{t-1} Q^t as a row-vector–matrix product. Because the chain is Markov, the t-step marginal is just the product of the matrices: q(z^t | x) = x Q̄^t with Q̄^t = Q¹ Q² … Q^t. As long as I can precompute or write Q̄^t in closed form, property 1 holds — I jump straight to z^t. Property 2, the posterior, comes from Bayes' rule, and let me actually derive it rather than wave at it, because the exact form is what I'll sample from later.

I want q(z^{t-1} | z^t, x). Bayes: q(z^{t-1} | z^t, x) ∝ q(z^t | z^{t-1}, x) q(z^{t-1} | x). The forward process is Markov, so conditioning on x adds nothing to the first factor: q(z^t | z^{t-1}, x) = q(z^t | z^{t-1}). Now I need that as a function of the candidate previous class. From the definition of the transition matrix, the likelihood that candidate z^{t-1}=k lands at the observed class z^t is the k-th entry of z^t (Q^t)′, the row vector z^t times the transpose of Q^t. The second factor is just the marginal, q(z^{t-1} | x) = x Q̄^{t-1}. The remaining normalizer doesn't depend on z^{t-1}. Multiplying the two pointwise:

  q(z^{t-1} | z^t, x) ∝ z^t (Q^t)′ ⊙ x Q̄^{t-1},

with ⊙ the elementwise product. (The exact normalizer is x Q̄^t (z^t)′, the total mass, which is a scalar.) So the posterior over the previous category is, for each candidate class, the product of "how easily z^t came from this class in one step" and "how likely this class was after t−1 steps from x." Closed form — property 2 holds. And property 3: pick the transition so the limit forgets x. The simplest, most-studied choice is the uniform transition, Q^t = α^t I + (1−α^t) 11′/K — with probability α^t stay put, with probability 1−α^t jump to a uniformly random class. It's doubly stochastic with positive entries, so its stationary distribution is uniform over the K classes, independent of x. All three properties, in an honestly discrete setting.

Now I lift this from one variable to a graph. The first temptation is to diffuse on the *graph state* — the whole graph as one categorical object. Dead on arrival: the number of graph states is astronomical, and the transition matrix would be that size squared. I can't build it. But I don't have to. Look at how image diffusion handles a million pixels: it applies the noise *independently per pixel*. Do the same here — diffuse independently on each node and each edge. The state space I build transition matrices over is then not "graphs," it's just node types X (cardinality a) and edge types E (cardinality b), two tiny matrices Q_X (a×a) and Q_E (b×b). Adding noise to form G^t = (X^t, E^t) is sampling each node's type from X^{t-1} Q_X^t and each edge's type from E^{t-1} Q_E^t, and the t-step jump is q(G^t | G) = (X Q̄_X^t, E Q̄_E^t). For undirected graphs I only noise the upper triangle of E and mirror it, so symmetry is preserved by construction. Each node, each edge, carries its own little categorical diffusion, all sharing the same two transition matrices and schedule.

The denoiser, φ_θ, takes the noisy graph G^t and outputs, for every node and every edge, a predicted distribution over its *clean* class — p̂^X for nodes, p̂^E for edges. And here the discreteness pays off in the most satisfying way. Because I parameterize the network to predict the clean graph (the x₀-style target, following the diffusion lesson about label noise), and because the clean target of each node and edge is a *class label*, the entire generative problem has dissolved into a pile of independent classification tasks: "what type is this node really?", "what type is this edge really?" No graph matching, no decoding a continuous adjacency, no alignment — just cross-entropy on each node and each edge. The loss is

  l(p̂^G, G) = Σ_i CE(x_i, p̂^X_i) + λ Σ_{ij} CE(e_{ij}, p̂^E_{ij}),

with λ a positive scalar trading off node vs. edge accuracy. Contrast that with a VAE, which has to solve a hard distribution-learning problem and often a graph match; diffusion turns it into supervised classification, because the forward process already told us, for each element, exactly what the clean answer was.

But wait — I should check the two properties I swore to protect, equivariance and invariance, because they're easy to break and fatal if broken. The loss first. If I permute the graph by π, the predicted and target tensors permute together, and because I use the *same* per-node function and the *same* per-edge function everywhere, the sum just reindexes:

  l(π·Ĝ, π·G) = Σ_i CE(π·X̂_i, x_{π^{-1}(i)}) + λ Σ_{ij} CE(π·Ê_{ij}, e_{π^{-1}(i),π^{-1}(j)}) = Σ_i CE(X̂_i, x_i) + λ Σ_{ij} CE(Ê_{ij}, e_{ij}) = l(Ĝ, G).

Permutation invariant — and notice this is *exactly* because the loss decomposes as a sum of identical per-node and per-edge terms. If I'd used a loss that mixed nodes in an order-dependent way, I'd have lost it. So I require the architecture to be permutation equivariant and the loss to be this decomposable per-element sum, and together they mean a gradient step doesn't change if I relabel the training graph — no permutation augmentation needed. There's a deeper reason matching is unnecessary that's worth saying: the diffusion process *keeps track of which node is which* at every step — it's like a physical process on distinguishable points — so the correspondence between noisy and clean is never lost and never needs to be recovered by search.

There's one more thing equivariance buys but doesn't finish: tractable likelihood. In general the likelihood of a graph is the sum over all n! permutations, intractable. The escape is to make the *generated* distribution exchangeable — every permutation of a generated graph equally likely — because then I can evaluate likelihood on a single representative. When is the output exchangeable? There's a clean result for diffusion: if the limit/prior distribution is invariant to the group action and the reverse transitions are equivariant, then the generated distribution is invariant. My limit is a product of i.i.d. distributions over nodes and over edges — permutation invariant. My denoiser is equivariant. And the map from the network's clean-graph prediction to the reverse transition, p_θ(G^{t-1}|G^t) = Σ_G q(G^{t-1}, G | G^t) p̂_θ(G), is equivariant to joint permutations. So the generated distribution is exchangeable, and likelihood is tractable on one representative.

Now sampling, which is where the analytic posterior I derived earlier earns its keep. After training, given a noisy G^t, the network hands me p̂^G — a distribution over what the *clean* graph is, per element. But to step down the chain I need p_θ(G^{t-1} | G^t), the distribution over the slightly-less-noisy graph. I get it by marginalizing the analytic posterior over the network's clean-graph belief. For a single node:

  p_θ(x_i^{t-1} | G^t) = Σ_{x ∈ X} q(x_i^{t-1} | x_i = x, x_i^t) · p̂^X_i(x),

i.e. for each possible clean class x, take the exact posterior q(x_i^{t-1} | clean = x, noisy = x_i^t) — which I can compute from the transition matrices via the Bayes formula above — and weight it by the network's probability that the clean class was x; sum over x. Same for each edge with p̂^E. I model the joint reverse step as a product over all nodes and edges, p_θ(G^{t-1}|G^t) = ∏_i p_θ(x_i^{t-1}|G^t) ∏_{ij} p_θ(e_{ij}^{t-1}|G^t), and sample a discrete G^{t-1} from this product of categoricals — which becomes the input at the next step. So the loop is: start from a graph drawn from the limit prior, and for t = T down to 1, run φ_θ to predict the clean graph, fold that prediction through the analytic posterior to get p_θ(G^{t-1}|G^t), sample, repeat — landing on G^0. The same posterior equations, by the way, give the per-step KL terms of an evidence lower bound, so I can report a likelihood for model comparison even though I train on the simpler cross-entropy.

Let me pin down the schedule. I follow the cosine schedule that works well for diffusion: ᾱ^t = cos²(½π (t/T + s)/(1+s)), normalized so ᾱ^0 = 1, with a small s ≈ 0.008. It corrupts gently near both ends of the chain. With α^t the per-step survival probability and ᾱ^t = ∏ α^τ the cumulative one, the transition matrices for the uniform model are Q^t = α^t I + (1−α^t) 11′/K stepwise and, by the same identity, Q̄^t = ᾱ^t I + (1−ᾱ^t) 11′/K cumulatively, so I can jump straight to any t.

This is a complete model, and I could stop. But let me run the forward process once more in my head with the uniform noise and watch it, because something bothers me. My graphs are *sparse* — the marginal over edge types is wildly lopsided, with "no edge" dominating real edge types. The uniform limit, though, is a graph where every pair of nodes is, with probability 1/b, each edge type — a dense object, nothing like my data. So during the forward process the graph drifts from sparse-and-real toward dense-and-uniform, and the reverse process would have to spend many steps just making the graph sparse again before it can arrange real structure. That's a hole I dug myself. If the noise's limit distribution were *closer to the data*, the noisy graphs along the chain would look more like real graphs throughout, and the denoiser would have an easier job. The prior should resemble the data.

So which prior? It can't be arbitrary — I just proved I need the limit to be permutation invariant for exchangeability, which leaves me with a product form that uses one distribution u for every node and one distribution v for every edge, ∏_i u × ∏_{ij} v. This family cannot keep correlations; it can only keep one-site category frequencies. I have to be careful about the word "projection" here. If I ask for the exact Euclidean projection of an arbitrary full graph distribution onto the nonlinear product family, the first-order equations contain the correlations of the full tensor, and the one-site marginals are not generally a solution. That is too strong. The object this prior can actually represent is the collection of one-site marginals, so the honest L2 question is: after I apply the marginal map and throw away everything the prior cannot encode, which shared u and v are closest to all node and edge slots?

Start with the node side. Let p^X_i be the data marginal distribution of the type at node position i after whatever arbitrary ordering the tensors use. The prior uses the same u at every node, so the squared representable-statistic error is

  J_X(u) = Σ_i ‖u − p^X_i‖².

This is an honest Euclidean projection onto the linear set where all node slots share one categorical vector. Expanding gives

  J_X(u) = n‖u‖² − 2⟨u, Σ_i p^X_i⟩ + const
         = n‖u − (1/n)Σ_i p^X_i‖² + const,

so the minimizer is u* = (1/n)Σ_i p^X_i, the empirical node-type marginal. The edge term is the same:

  J_E(v) = Σ_{ij} ‖v − p^E_{ij}‖²
         = n²‖v − (1/n²)Σ_{ij} p^E_{ij}‖² + const,

so v* = (1/n²)Σ_{ij} p^E_{ij}, the empirical edge-type marginal. That is the optimal-prior theorem I can use without smuggling in false correlation claims: once exchangeability and tractability restrict the limit to an invariant independent prior, the L2 projection of the representable one-site statistics is the product of the data marginals. It is exactly the thing intuition wanted: the prior should know that "no edge" is common and rare bond or atom classes are rare, without pretending to know graph structure before denoising starts.

So I want transition matrices whose cumulative limit row converges to the marginal m, not to uniform. The fix is to swap the uniform target for the marginal target inside the transition. Define

  Q_X^t = α^t I + β^t 1_a m_X′,   Q_E^t = α^t I + β^t 1_b m_E′,

with β^t = 1 − α^t: with probability α^t stay put, with probability β^t jump to a class drawn from the *marginal* m (every row of 1 m′ is m). I have to check the cumulative product still has a closed form, or property 1 breaks. The key algebraic fact is that 1 m′ is idempotent: (1 m′)(1 m′) = 1 (m′1) m′ = 1·1·m′ = 1 m′, since m′1 = Σ m_i = 1. So two consecutive steps multiply as

  (α_s I + β_s 1m′)(α_t I + β_t 1m′)
    = α_s α_t I + (α_s β_t + β_s α_t + β_s β_t)1m′
    = α_s α_t I + (1 − α_s α_t)1m′.

Induction gives Q̄^t = ᾱ^t I + β̄^t 1 m′ with ᾱ^t = ∏_{τ=1}^t α^τ and β̄^t = 1 − ᾱ^t. As t → ∞, ᾱ^t → 0 and every row of Q̄^t → m: the limit is the marginal, exactly as the projection argument ordered, and still closed-form. In code this is the same shape as uniform, `Qt = (1−beta_t) I + beta_t (1 m′)`, with the row vector being the marginal instead of uniform 1/K. At the level of the forward marginals, the noised graphs now keep the right proportion of edges to non-edges all the way down, so the denoiser is no longer wasting its early reverse steps merely re-sparsifying a dense uniform sample. The marginal transition is forced by combining "the prior must be invariant" with "the invariant independent prior can at least preserve the data's one-site marginals."

Now I get to cash in the thing continuous diffusion couldn't do, and this is the second place keeping the graph discrete pays off. My denoiser is a graph network, and graph networks have a known ceiling: message-passing nets and graph transformers are bounded by the 1-Weisfeiler-Leman test, and concretely they *cannot count cycles* or detect simple substructures on their own. That's alarming for a generator — if the network can't even perceive a triangle, how will it reproduce the clustering and orbit statistics I'm scored on? I could reach for a strictly more powerful (and far more expensive) architecture. Cheaper and more direct: hand the network the features it can't compute itself, as extra inputs, computed *from the noisy graph at each step*. Cycle counts, spectral features of the Laplacian — descriptors known to capture exactly the structural properties the bare network misses. And the reason I *can* do this here, when GDSS can't, is precisely that my noisy graph is a real sparse discrete graph at every t, so "how many 3-cycles pass through node i" is a well-defined, computable quantity. On a Gaussian fog it's meaningless. So at each diffusion step I compute structural and spectral features z = f(G^t, t) and concatenate them to the network input. For cycles I use the closed-form trace/Frobenius formulas (counting traversals on a GPU would be hopeless, all the more since these are recomputed every step) — node-level counts of k-cycles up to k=5 and graph-level counts up to k=6, e.g. X₃ = diag(A³)/2 for triangles per node and y₃ = X₃′1/3 for the triangle total, with the higher-order ones built from products and traces of powers of A. For spectral features I take the Laplacian: the multiplicity of eigenvalue 0 gives the number of connected components, the first few nonzero eigenvalues are global shape, and the corresponding eigenvectors give node-level structure (an O(n³) eigendecomposition, fine for the graph sizes I care about). On molecules I also add current valency and molecular weight. These features should be an enhancement rather than a crutch, but they directly attack the expressivity gap, and the discrete noise model is what makes them available.

The denoising network itself I build as a graph transformer, because attention is a natural fit for edge prediction — every pair of nodes already has an attention score, which is exactly the object I want to turn into an edge type. I extend the standard graph-transformer layer to maintain three things at once: node features X, edge features E, and a small graph-level feature vector y (which also carries the timestep, normalized to [0,1], as a global feature). A layer does multi-head self-attention on the nodes, but the edge features modulate the attention rather than being passive: I let E feed into the unnormalized attention scores through a feature-wise linear modulation, score ← score·(E_mul + 1) + E_add, so an edge can both scale and shift the attention between its endpoints. The graph feature y likewise modulates X and E by FiLM, FiLM(M₁,M₂) = M₁W₁ + (M₁W₂)⊙M₂ + M₂. After attention I update the edge representations from the attention scores (edges are outputs, so they must be refreshed every layer), and update y by pooling node and edge features — a PNA-style pooling that concatenates max, min, mean and std, so the global feature sees several order-invariant summaries. Residual connections and layer normalization throughout, an FFN per stream. Cost is Θ(n²) per layer, unavoidable since I predict something for every edge. Every block — attention, FiLM, the PNA pooling (invariant), layer norm — is permutation equivariant or invariant, so the whole network is equivariant, which is what the exchangeability argument needed.

So I have the full method. Forward: per-node, per-edge categorical diffusion with marginal transitions on a cosine schedule, jumping straight to G^t via Q̄^t. Training: sample a t, sample the noisy G^t, compute structural/spectral features, run the graph transformer to predict the clean node and edge classes, and take a cross-entropy step — a permutation-invariant classification loss. Sampling: draw node count and a graph from the marginal limit, then iterate the reverse step, each time predicting the clean graph and folding it through the analytic posterior to sample G^{t-1}.

One more capability falls out almost for free, worth deriving because it shows the framework is flexible: conditioning generation on a graph-level property y_G without retraining the unconditional model. Train a separate regressor g_η(G^t) = ŷ to read the property off a noisy graph. The conditional reverse step factorizes as q̇(G^{t-1} | G^t, y_G) ∝ q(G^{t-1}|G^t) q̇(y_G | G^{t-1}) — the unconditional reverse times a likelihood of the property under the next state. The trouble: q̇(y_G | G^{t-1}) doesn't factorize over nodes and edges, so I can't evaluate it for every candidate G^{t-1}. Fix it with a first-order expansion. Treat G as a continuous tensor so ∇_G makes sense, and Taylor-expand the log-likelihood around G^t: log q̇(y_G | G^{t-1}) ≈ log q̇(y_G | G^t) + ⟨∇_G log q̇(y_G | G^t), G^{t-1} − G^t⟩, and the linear term *does* split into a sum over nodes and edges, ≈ c(G^t) + Σ_i ⟨∇_{x_i} log q̇, x_i^{t-1}⟩ + Σ_{ij} ⟨∇_{e_{ij}} log q̇, e_{ij}^{t-1}⟩. Assume the property is Gaussian around the regressor, q̇(y | G^t) = N(g(G^t), σ_y I), so ∇ log q̇ ∝ −∇ ‖ŷ − y_G‖². Then the guidance distribution is p_η(ŷ | G^{t-1}) ∝ exp(−λ ⟨∇_{G^t} ‖ŷ − y_G‖², G^{t-1}⟩), and I multiply it into the unconditional reverse step at sampling time — a discrete analogue of classifier guidance, pushing each step toward graphs with the target property, λ tuning how hard. The same masking trick lets me condition on a fixed subgraph (extend a given motif) by overwriting the kept nodes/edges at each reverse step.

Now let me write the code I'd actually run, filling the one empty slot in the harness — the generative mechanism. I'll keep it self-contained: the two transition matrices (marginal), the cosine schedule, the graph-transformer denoiser, the cross-entropy training step, and the reverse-sampling loop that marginalizes the analytic posterior over the network's clean-graph prediction. I'll represent node and edge *types* as one-hot categories (for a plain structural dataset, node type is trivial and edge type is "no edge" vs "edge," b = 2).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


def cosine_alpha_bar(T, s=0.008):
    # cumulative survival ᾱ^t = cos²(½π (t/T + s)/(1+s)), normalized so ᾱ^0 = 1
    steps = torch.arange(T + 1, dtype=torch.float64)
    abar = torch.cos(0.5 * math.pi * (steps / T + s) / (1 + s)) ** 2
    abar = abar / abar[0]
    return torch.clamp(abar, 1e-6, 1.0)           # [T+1], abar[0]=1; floor keeps Q rows valid


class GraphTransformerLayer(nn.Module):
    """Node, edge and graph-level streams with edge-aware FiLM attention."""
    def __init__(self, dim, n_heads=8, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or 4 * dim
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.e_mul = nn.Linear(dim, dim); self.e_add = nn.Linear(dim, dim)
        self.y_x_mul = nn.Linear(dim, dim); self.y_x_add = nn.Linear(dim, dim)
        self.y_e_mul = nn.Linear(dim, dim); self.y_e_add = nn.Linear(dim, dim)
        self.x_out = nn.Linear(dim, dim); self.e_out = nn.Linear(dim, dim)
        self.y_out = nn.Sequential(nn.Linear(9 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.normX1 = nn.LayerNorm(dim); self.normX2 = nn.LayerNorm(dim)
        self.normE1 = nn.LayerNorm(dim); self.normE2 = nn.LayerNorm(dim)
        self.normY1 = nn.LayerNorm(dim); self.normY2 = nn.LayerNorm(dim)
        self.ffX = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.ffE = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.ffY = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))

    def forward(self, X, E, y, node_mask):
        B, N, C = X.shape
        x_mask = node_mask.unsqueeze(-1).float()
        e_mask = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2)).unsqueeze(-1).float()
        q = self.q(X).view(B, N, self.n_heads, self.head_dim)
        k = self.k(X).view(B, N, self.n_heads, self.head_dim)
        v = self.v(X).view(B, N, self.n_heads, self.head_dim)
        scores = (q[:, :, None] * k[:, None, :]) / math.sqrt(self.head_dim)
        e_mul = self.e_mul(E).view(B, N, N, self.n_heads, self.head_dim)
        e_add = self.e_add(E).view(B, N, N, self.n_heads, self.head_dim)
        scores = scores * (e_mul + 1) + e_add
        logits = scores.sum(-1).masked_fill(e_mask.squeeze(-1).unsqueeze(-1) == 0, -1e9)
        newE = scores.flatten(start_dim=3)
        newE = self.y_e_add(y).view(B, 1, 1, C) + (self.y_e_mul(y).view(B, 1, 1, C) + 1) * newE
        E = self.normE1(E + self.e_out(newE) * e_mask)
        E = self.normE2(E + self.ffE(E))
        attn = F.softmax(logits, dim=2)
        out = torch.einsum('bijh,bjhd->bihd', attn, v).reshape(B, N, C)
        out = self.y_x_add(y).view(B, 1, C) + (self.y_x_mul(y).view(B, 1, C) + 1) * out
        X = self.normX1(X + self.x_out(out) * x_mask)
        X = self.normX2(X + self.ffX(X))
        x_count = x_mask.sum(1).clamp_min(1)
        e_count = e_mask.sum((1, 2)).clamp_min(1)
        x_mean = (X * x_mask).sum(1) / x_count
        e_mean = (E * e_mask).sum((1, 2)) / e_count
        x_min = X.masked_fill(x_mask == 0, float("inf")).min(1).values
        x_max = X.masked_fill(x_mask == 0, -float("inf")).max(1).values
        e_min = E.masked_fill(e_mask == 0, float("inf")).amin((1, 2))
        e_max = E.masked_fill(e_mask == 0, -float("inf")).amax((1, 2))
        x_std = torch.sqrt((((X - x_mean[:, None]) * x_mask) ** 2).sum(1) / x_count)
        e_std = torch.sqrt((((E - e_mean[:, None, None]) * e_mask) ** 2).sum((1, 2)) / e_count)
        x_pool = torch.cat([x_mean, x_min, x_max, x_std], dim=-1)
        e_pool = torch.cat([e_mean, e_min, e_max, e_std], dim=-1)
        y = self.normY1(y + self.y_out(torch.cat([y, x_pool, e_pool], dim=-1)))
        y = self.normY2(y + self.ffY(y))
        return X * x_mask, E * e_mask, y


class Denoiser(nn.Module):
    def __init__(self, a, b, extra_x=0, extra_e=0, extra_y=0, dim=256, n_layers=5, n_heads=8):
        super().__init__()
        self.a, self.b = a, b
        self.node_in = nn.Sequential(nn.Linear(a + extra_x, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.edge_in = nn.Sequential(nn.Linear(b + extra_e, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.y_in = nn.Sequential(nn.Linear(1 + extra_y, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.layers = nn.ModuleList([GraphTransformerLayer(dim, n_heads) for _ in range(n_layers)])
        self.node_out = nn.Linear(dim, a); self.edge_out = nn.Linear(dim, b)

    def forward(self, Xt, Et, t_frac, node_mask, extraX=None, extraE=None, extraY=None):
        B, N, _ = Xt.shape
        if extraX is not None:
            Xt = torch.cat([Xt, extraX], dim=-1)
        if extraE is not None:
            Et = torch.cat([Et, extraE], dim=-1)
        y_in = t_frac.view(B, 1).float()
        if extraY is not None:
            y_in = torch.cat([y_in, extraY], dim=-1)
        X = self.node_in(Xt); E = self.edge_in(Et); y = self.y_in(y_in)
        for layer in self.layers:
            X, E, y = layer(X, E, y, node_mask)
        edge_logits = self.edge_out(E)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        return self.node_out(X), edge_logits


class GraphGenerator(nn.Module):
    def __init__(self, max_nodes, n_node_types=1, n_edge_types=2,
                 extra_x=0, extra_e=0, extra_y=0, dim=256, n_layers=5, n_heads=8,
                 T=500, lr=2e-4, lambda_e=5.0, feature_fn=None, **kwargs):
        super().__init__()
        self.max_nodes, self.a, self.b = max_nodes, n_node_types, n_edge_types
        self.T, self.lambda_e = T, lambda_e
        self.denoiser = Denoiser(self.a, self.b, extra_x, extra_e, extra_y, dim, n_layers, n_heads)
        self.feature_fn = feature_fn
        self.register_buffer('abar', cosine_alpha_bar(T))
        self.register_buffer('mX', torch.ones(self.a) / self.a)
        self.register_buffer('mE', torch.ones(self.b) / self.b)
        self.optimizer = optim.AdamW(self.denoiser.parameters(), lr=lr, weight_decay=1e-12, amsgrad=True)
        self._counts = None

    def set_marginals(self, mX, mE):
        self.mX.copy_(mX); self.mE.copy_(mE)

    def _extra_features(self, Xt, Et, t_frac, node_mask):
        if self.feature_fn is None:
            return None, None, None
        return self.feature_fn(Xt, Et, t_frac, node_mask)

    def _Qbar(self, abar_t, m):
        K = m.numel(); I = torch.eye(K, device=m.device)
        one_m = torch.ones(K, 1, device=m.device) @ m.view(1, K)
        return abar_t * I + (1 - abar_t) * one_m

    def _Qt(self, abar_t, abar_s, m):
        K = m.numel(); I = torch.eye(K, device=m.device)
        one_m = torch.ones(K, 1, device=m.device) @ m.view(1, K)
        alpha_t = (abar_t / abar_s).clamp(0, 1)
        return alpha_t * I + (1 - alpha_t) * one_m

    def _apply_noise(self, Xoh, Eoh, node_mask, t_idx):
        B = Xoh.shape[0]
        Xt = torch.empty_like(Xoh); Et = torch.empty_like(Eoh)
        N = Xoh.shape[1]
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        edge_mask = edge_mask & ~torch.eye(N, device=Xoh.device, dtype=torch.bool).unsqueeze(0)
        for bb in range(B):
            at = self.abar[t_idx[bb]]
            pX = Xoh[bb] @ self._Qbar(at, self.mX)
            pE = Eoh[bb] @ self._Qbar(at, self.mE)
            Xt[bb] = F.one_hot(torch.multinomial(pX, 1).squeeze(-1), self.a).float()
            e = torch.multinomial(pE.reshape(-1, self.b), 1).reshape(pE.shape[:-1])
            e = torch.triu(e, 1); e = e + e.transpose(0, 1)
            Et[bb] = F.one_hot(e, self.b).float()
        Xt[~node_mask] = F.one_hot(torch.zeros((), dtype=torch.long, device=Xoh.device), self.a).float()
        Et[~edge_mask] = F.one_hot(torch.zeros((), dtype=torch.long, device=Eoh.device), self.b).float()
        return Xt, Et

    def train_step(self, Xoh, Eoh, y=None, node_mask=None, extraX=None, extraE=None, extraY=None):
        self.train(); self.optimizer.zero_grad()
        B, N, _ = Xoh.shape; device = Xoh.device
        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        edge_mask = edge_mask & ~torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        t_idx = torch.randint(1, self.T + 1, (B,), device=device)
        Xt, Et = self._apply_noise(Xoh, Eoh, node_mask, t_idx)
        t_frac = t_idx.float() / self.T
        autoX, autoE, autoY = self._extra_features(Xt, Et, t_frac, node_mask)
        extraX = extraX if extraX is not None else autoX
        extraE = extraE if extraE is not None else autoE
        extraY = extraY if extraY is not None else autoY
        if y is not None and y.numel() > 0:
            extraY = y if extraY is None else torch.cat([y, extraY], dim=-1)
        node_logits, edge_logits = self.denoiser(Xt, Et, t_frac, node_mask, extraX, extraE, extraY)
        node_ce = F.cross_entropy(node_logits.reshape(-1, self.a), Xoh.argmax(-1).reshape(-1), reduction='none')
        edge_ce = F.cross_entropy(edge_logits.reshape(-1, self.b), Eoh.argmax(-1).reshape(-1), reduction='none')
        node_loss = (node_ce * node_mask.reshape(-1).float()).sum() / node_mask.sum().clamp_min(1)
        edge_loss = (edge_ce * edge_mask.reshape(-1).float()).sum() / edge_mask.sum().clamp_min(1)
        loss = node_loss + self.lambda_e * edge_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": loss.item(), "edge_loss": edge_loss.item()}

    def _posterior_over_x0(self, zt, Qt, Qsb, Qtb):
        left = (zt @ Qt.transpose(-1, -2)).unsqueeze(-2)
        num = left * Qsb
        denom = (zt @ Qtb.transpose(-1, -2)).unsqueeze(-1)
        return num / denom.clamp_min(1e-6)

    @torch.no_grad()
    def sample(self, n_samples, device):
        self.eval(); N = self.max_nodes
        if self._counts is not None:
            nc = torch.multinomial(self._counts.to(device), n_samples, replacement=True).clamp_min(2)
        else:
            nc = torch.full((n_samples,), N, device=device)
        node_mask = torch.arange(N, device=device).unsqueeze(0) < nc.unsqueeze(1)
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        edge_mask = edge_mask & ~torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        e0 = torch.multinomial(self.mE.to(device), n_samples * N * N, replacement=True).view(n_samples, N, N)
        e0 = torch.triu(e0, 1); e0 = e0 + e0.transpose(1, 2)
        Et = F.one_hot(e0, self.b).float()
        Xt = F.one_hot(torch.multinomial(self.mX.to(device), n_samples * N, replacement=True).view(n_samples, N), self.a).float()
        Xt[~node_mask] = F.one_hot(torch.zeros((), dtype=torch.long, device=device), self.a).float()
        Et[~edge_mask] = F.one_hot(torch.zeros((), dtype=torch.long, device=device), self.b).float()
        for t in range(self.T - 1, -1, -1):
            abar_t, abar_s = self.abar[t + 1], self.abar[t]
            t_frac = torch.full((n_samples,), (t + 1) / self.T, device=device)
            extraX, extraE, extraY = self._extra_features(Xt, Et, t_frac, node_mask)
            node_logits, edge_logits = self.denoiser(Xt, Et, t_frac, node_mask, extraX, extraE, extraY)
            pX0 = F.softmax(node_logits, -1); pE0 = F.softmax(edge_logits, -1)
            postX = self._posterior_over_x0(Xt, self._Qt(abar_t, abar_s, self.mX),
                                            self._Qbar(abar_s, self.mX), self._Qbar(abar_t, self.mX))
            postE = self._posterior_over_x0(Et, self._Qt(abar_t, abar_s, self.mE),
                                            self._Qbar(abar_s, self.mE), self._Qbar(abar_t, self.mE))
            probX = (pX0.unsqueeze(-1) * postX).sum(-2).clamp_min(1e-6)
            probE = (pE0.unsqueeze(-1) * postE).sum(-2).clamp_min(1e-6)
            probX = probX / probX.sum(-1, keepdim=True)
            probE = probE / probE.sum(-1, keepdim=True)
            Xt = F.one_hot(torch.multinomial(probX.reshape(-1, self.a), 1).view(n_samples, N), self.a).float()
            e = torch.multinomial(probE.reshape(-1, self.b), 1).view(n_samples, N, N)
            e = torch.triu(e, 1); e = e + e.transpose(1, 2)
            Et = F.one_hot(e, self.b).float()
            Xt[~node_mask] = F.one_hot(torch.zeros((), dtype=torch.long, device=device), self.a).float()
            Et[~edge_mask] = F.one_hot(torch.zeros((), dtype=torch.long, device=device), self.b).float()
        return Xt, Et, nc.long()
```

Let me trace the causal chain back. I wanted order-free, structure-preserving one-shot graph generation, and diffusion was the strongest paradigm but every existing port continuized the graph — Gaussian noise destroys the sparsity and the structural notions that *are* the data, and forecloses feeding the denoiser structural features. So I kept the graph discrete the whole way: per-node and per-edge categorical diffusion with tiny transition matrices, which satisfies the three diffusion properties (closed-form marginal via Q̄^t, closed-form Bayes posterior, x-independent limit) while staying an honest graph at every step. Predicting the clean graph turned generation into permutation-invariant per-element classification — cross-entropy, no graph matching — and the equivariant network plus invariant product limit gave exchangeability and tractable likelihood. The uniform limit is still a poor endpoint for sparse graph data, because it forgets the empirical node and edge frequencies; projecting the one-site node and edge statistics that an invariant independent prior can actually represent gives the data marginals, which I realize with marginal transitions Q^t = α^t I + β^t 1 m′, idempotent enough to keep Q̄^t closed-form. The discreteness then lets me feed the network the cycle and spectral features it provably can't compute itself, attacking the expressivity ceiling. Sampling iterates the reverse step by folding the network's clean-graph prediction through the analytic posterior; classifier-style guidance via a first-order expansion adds conditioning for free. The whole thing trains as classification and samples by repeated denoising of a discrete graph.
