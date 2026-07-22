Let me start from what actually hurts when I try to learn a distribution over graphs and sample new ones. A graph `G` has no canonical labelling of its nodes — if I permute the node labels I get the same graph — so when I write it as an adjacency matrix `A^pi` I've secretly chosen an ordering `pi`, and the same `G` corresponds to as many as `N!` different matrices. The honest likelihood is therefore `p(G) = sum_pi p(A^pi)`, a sum over factorially many orderings, and that single fact poisons everything: I can't compute the objective exactly, I can't define a clean reconstruction loss without first deciding which node of my sample corresponds to which node of the target, and any sequential generator I build will depend on an ordering I had to invent. On top of that I want three things at once and the existing models each give me only some of them: I want edge generation that respects how correlated real edges are, I want to handle the permutation problem without enumerating `N!` anything, and I want it to scale to graphs with hundreds or thousands of nodes in feasible time. Nobody has all three. So the game is to find a single generation process that is expressive about edges, principled about orderings, and fast.

Let me look hard at what's on the table and where each one breaks, because the gaps are going to tell me what to build. The cleanest-feeling option is the one-shot latent-variable route, GraphVAE: encode a graph to a latent `z`, decode a whole probabilistic adjacency matrix `Ã in [0,1]^{k x k}` on a fixed `k` nodes, edges independent Bernoulli given `z`. It's parallel and simple, but it has two diseases. First, to even compute the reconstruction term `p(G | G̃)` I have to align the decoded graph's nodes to the target's nodes — a graph-matching problem, because neither side has a canonical order — and approximate matching costs on the order of `k^4` with `k^2` parameters, which pins the whole thing to `k` of a few dozen nodes. Second, the edges are independent given `z`, and edge independence is exactly what's documented to wreck sample quality: in a real graph whether `(i,j)` is an edge depends heavily on the rest of the structure — shared neighbors, community membership, lattice regularity — and a factorized decoder can't represent that. So the one-shot route is fast-ish but doesn't scale and isn't expressive. Two strikes.

The opposite end is fully auto-regressive: build the graph as a sequence of local decisions, each conditioned on everything so far. That's the right instinct for expressiveness, because the factorization `p(G) = prod_t p(chunk_t | chunk_{<t})` is exact and lets every new edge depend on the whole partial graph — no independence assumption forced. Two instances exist. The GNN-in-the-loop model (Li et al. 2018, DeepGMG) runs a full multi-round message-passing network before *each* decision — add a node? add an edge? to which existing node? — refreshing every node's state with a GRU update over aggregated messages, then reading out a graph vector. Quality is strong because every decision sees the real topology. But running full propagation per individual node-or-edge decision costs, by the standard counting, something like `m * n^2 * diam(G)` for `m` edges and `n` nodes — it's hopeless past a few tens of nodes, and indeed it just won't finish on graphs of hundreds of nodes. The other instance, GraphRNN (You et al. 2018), drops the GNN: it represents the graph under an ordering as a sequence of adjacency vectors `S_i^pi in {0,1}^{i-1}` (node `i`'s edges back to earlier nodes) and models `p(S^pi) = prod_i p(S_i^pi | S_{<i}^pi)` with a graph-level RNN holding the state and an edge-level RNN emitting each `S_i^pi` entry-by-entry to capture intra-row edge correlations. By bounding everything to breadth-first-search orderings — many permutations collapse to one BFS ordering, and BFS bounds the adjacency-vector width to the frontier size `M` — it gets to `O(M*n)` and scales `~50x` past the earlier deep models. That's the most scalable general model so far, but stare at its gaps. The full model still makes `O(N^2)` generation decisions, one entry or column at a time. With no GNN in the loop, the next decision can't depend *directly* on the existing graph's topology; it has to be filtered through the RNN hidden state. And the sequential ordering creates a long-term bottleneck: two nodes that are one hop apart in the graph can be dozens of steps apart in the generation sequence, so the dependency between them must survive a long recurrent path — exactly where RNNs are weakest. Plus it commits to a single BFS ordering family described as efficient-but-arguably-suboptimal.

So I can see the shape of the corner I'm in. The GNN-in-the-loop gives me topology-aware decisions but is `O(N^2 * something)` slow because it pays full propagation per atomic decision and carries states forward. The RNN gives me scale but loses the direct topological conditioning and suffers the long-term bottleneck, while still spending `O(N^2)` steps. Three knobs are tangled together — number of sequential steps, whether decisions see the graph directly, and whether edges within a step are correlated — and every prior method trades one for another. Let me try to pull them apart.

The most painful axis is the number of sequential steps, because steps are the thing that can't be parallelized — `T` steps have to happen in order. GraphRNN spends one step per matrix entry (or column), so `O(N^2)` (or `O(N)` columns but with the entry-level RNN inside). What if I generate a *whole row* of the lower-triangular adjacency at once — all of node `i`'s back-edges in one shot — conditioned on the already-generated graph? Then I have `O(N)` steps. And why stop at one row: generate a *block* of `B` rows per step, and the step count drops to `T = ceil(N/B)`. Since I only need the lower triangle (undirected: `A^pi = L^pi + L^pi^T`), I'm generating `L^pi` block-row by block-row,

  p(L^pi) = prod_{t=1}^{T} p(L_{b_t}^pi | L_{b_1}^pi, ..., L_{b_{t-1}}^pi),

where `b_t = {B(t-1)+1, ..., Bt}` indexes the rows in block `t`. This is exact — it's just the chain rule grouped into blocks — and `B` is now an explicit speed knob: bigger `B` means fewer sequential steps means faster, at the cost of having to model a bigger chunk per step. That's the kind of clean trade-off I want, the graph-generation analogue of a convolutional stride. So the spine I'll try is an `O(N)`-step block-wise autoregressive process over the lower triangle.

Now the conditioning. I refuse to give up direct topology-awareness — that was the whole point of the GNN-in-the-loop — but I can't afford to carry GNN states forward across steps the way DeepGMG does, because that's what made it slow and what couples the steps and blocks parallel training. I don't *have* to carry state forward. At training time I have the ground-truth graph, so for each block-step the conditioning context — the already-generated subgraph — is a *known prefix* of the data. That's exactly the PixelCNN situation: with the prefix known, every conditional `p(L_{b_t}^pi | L_{<b_t}^pi)` can be evaluated independently and in parallel across `t`, no recurrence to unroll, even though *sampling* still goes step by step. So I'll run a fresh GNN on the relevant subgraph at each step, with no hidden state passed between steps. Each step pays its own propagation, but the steps don't serialize during training, and crucially each decision now sees the real topology directly. The long-term bottleneck dissolves too: two graph-adjacent nodes are one hop apart in the GNN regardless of how far apart they sit in the row ordering, so the dependency I care about no longer has to crawl through a long sequence.

Let me make the per-step computation concrete. At step `t` I have the subgraph on the previous `B(t-1)` nodes with their already-generated edges, and I'm about to add `B` new nodes. I need each new node to decide which existing nodes (and which other new nodes) to connect to. So I build an *augmented graph*: the existing subgraph, plus the `B` new nodes, plus *candidate* edges connecting each new node to every existing node and to each other — these augmented edges are the slots whose presence I'm trying to predict. Then I run message passing on this augmented graph so that a new node's representation absorbs the structure it's attaching to.

What goes into the GNN? I need an initial node representation. For an existing node I have its row of the adjacency, `L_{b_i}^pi`, which I'll just linearly embed, `h^0 = W L_{b_i}^pi + b`, mostly to shrink the dimension (rows can be as wide as `N`, and for large graphs I want a compact `H`-dim state). For the new block, there's nothing generated yet, so `h^0 = 0` for all `B` new nodes. That zero-initialization makes me uneasy, so let me trace what it does before going further. The `B` new nodes start with identical zero representations; message passing is permutation-equivariant; and if two new nodes also see the same neighbor structure, every propagation round will update them identically. Let me actually run one round on the smallest case — two new nodes `u, v` both at `h^0 = 0`, each attached to one existing node `w` — with a toy linear message/update, to see whether they really stay equal:

```
hw = [0.3, -0.7];  hu = hv = [0, 0]
update(h, neighbors) = tanh( h + sum_n W (h - h_n) )      # W a fixed 2x2
hu^1 = tanh( 0 + W(0 - hw) ) = [-0.244,  0.855]
hv^1 = tanh( 0 + W(0 - hw) ) = [-0.244,  0.855]
```

`hu^1 == hv^1` exactly. So after any number of rounds the two new nodes have the same final representation, and the edge logit they produce to any node `w` — `MLP(h_u^R - h_w)` versus `MLP(h_v^R - h_w)` — is literally the same number, forcing `theta_uw == theta_vw`. That's wrong: the two new nodes should be allowed to connect differently. The network *cannot* tell them apart. So I have to break this symmetry by hand, and separately I want the network to tell existing nodes from new nodes since they play different roles. Both needs are met by tagging each node's representation with a small marker `x_i`: a `B`-dimensional binary mask, all-zeros for an existing node and a one-of-`B` encoding of position-in-the-block for a new node. Re-running the same toy round with distinct markers appended (`x_u = [1,0]`, `x_v = [0,1]`) and a message that reads `h_i - h_j` concatenated with `x_i - x_j`, the two updates come out *unequal* — `allclose(h_u^1, h_v^1)` is now `False`. So that one marker does double duty: it distinguishes existing-vs-new, and it desymmetrizes the new nodes so they're no longer forced into identical edge distributions.

Now the message-passing equations. A message along edge `(i,j)` should be a function of the *pair* of endpoint representations. The natural pairwise feature is the difference `h_i - h_j` — it's the relative representation, it's antisymmetric in a way that respects edge directionality of the message, and it matches the gated message-passing style that worked in the graph-net line. So

  m_{ij}^r = f(h_i^r - h_j^r),     with f a small MLP (two layers, ReLU).

But not all candidate edges are equally informative — a new node should listen more to some neighbors than others, and especially the augmented (candidate) edges are guesses, not real structure. So I'll put a learned attention weight on each message before aggregating. I want the attention to also be able to behave differently for existing-vs-new and new-vs-new edges, which is exactly what the marker `x_i` is for. So I compute the attention from the *marked* representations,

  h~_i^r = [h_i^r, x_i],
  a_{ij}^r = sigmoid( g(h~_i^r - h~_j^r) ),     g a small MLP (two layers, ReLU),

a per-edge gate in `(0,1)` that scales the message it carries (I can let it gate each message channel rather than be a single scalar, which gives the attention a bit more room without changing the picture). Why sigmoid and not a softmax over neighbors? Because I'm not forcing the incoming attention to be a probability distribution that sums to one — each edge independently decides how much of its message to pass, which is the right semantics for "is this candidate edge worth attending to." Then the node state updates by aggregating the attended messages and folding them in through a GRU,

  h_i^{r+1} = GRU( h_i^r, sum_{j in N(i)} a_{ij}^r m_{ij}^r ).

The GRU is doing real work over `R` rounds: gated updates keep multi-round propagation stable instead of letting the state blow up or wash out, which is the lesson from the gated-graph-net line. After `R` rounds I have final node representations `h_i^R` that encode where each new node sits in the structure.

From those I have to emit the edge probabilities for the block. The simplest output is one Bernoulli per candidate edge, `theta_{i,j} = sigmoid(MLP(h_i^R - h_j^R))`, edges independent given the existing graph. And here I hit the wall I already saw with GraphVAE and GraphRNN-S: edge independence *within the block* compromises quality. With a block of `B` rows there are many candidate edges generated in one shot, and conditionally-independent Bernoullis can't represent, say, "either both of these edges fire or neither does." GraphRNN's fix was an edge-level RNN that emits the entries one at a time so each conditions on the previous — but that reintroduces exactly the per-entry sequentiality I just spent my whole step count to escape. I want intra-block edge *correlation* without an intra-block sequential model. Wall.

Let me think about what cheaply correlates a set of independent variables. If I make all the edges depend on a shared *latent*, then conditioned on the latent they can be independent, but marginally — integrating the latent out — they should be correlated. The cheapest latent is a discrete mixture index: a mixture of Bernoulli products. Introduce `K` components; component `k` has its own per-edge probabilities `theta_{k,i,j}` and a weight `alpha_k`,

  p(L_{b_t}^pi | ...) = sum_{k=1}^{K} alpha_k prod_{i in b_t} prod_{1<=j<=i} theta_{k,i,j}.

Here `theta_{k,i,j}` is shorthand for the Bernoulli probability assigned by component `k` to the realized binary edge slot; when the observed label is `y_{i,j}`, the factor is `theta_raw_{k,i,j}^{y_{i,j}} (1 - theta_raw_{k,i,j})^{1-y_{i,j}}`. Within a component the edges are fully factorial (so I keep parallelism — all `K` components' edge probabilities can be computed at once), and the hope is that the mixture over `k` couples them. But "hope" isn't enough — the whole point of this head is that the marginal is correlated, and I just claimed independent-Bernoulli fails precisely there, so I should check that the mixture actually buys me the correlation rather than assume it. Take the smallest non-trivial instance: two edge slots `e1, e2`, `K = 2`, weights `alpha = (0.5, 0.5)`, component 1 with per-edge probabilities `(0.9, 0.9)` and component 2 with `(0.1, 0.1)` — i.e. one mode where both edges tend to fire and one where neither does. Compute the marginals and the joint by hand:

```
P(e1=1) = 0.5*0.9 + 0.5*0.1 = 0.5
P(e2=1) = 0.5*0.9 + 0.5*0.1 = 0.5
P(e1=1, e2=1) = 0.5*(0.9*0.9) + 0.5*(0.1*0.1) = 0.5*0.81 + 0.5*0.01 = 0.41
P(e1=1) P(e2=1) = 0.5 * 0.5 = 0.25
Cov(e1, e2) = 0.41 - 0.25 = 0.16   (positive, large)
```

So the joint probability that both edges fire is `0.41`, far above the `0.25` it would be under independence — a covariance of `0.16`. The mixture genuinely correlates the edges even though each component is factorial. And the degenerate case checks out the other way: with `K = 1` a single component `(0.9, 0.9)` gives `P(e1=1, e2=1) = 0.81 = 0.9 * 0.9 = P(e1=1) P(e2=1)`, covariance exactly `0` — independent, as it must be. So the mixture strictly generalizes the independent-Bernoulli head and recovers it at `K = 1`, while for `K > 1` it puts real correlation on the table at a cost of only a factor of `K` in the output head, fully parallel, no sequential edge model. That's the property I needed and now I've seen it hold on numbers rather than just argued for it. I'll compute the per-edge per-component probabilities from the same pairwise difference, `theta_raw_{k,i,j} = sigmoid(MLP_theta(h_i^R - h_j^R))` with a `K`-dimensional output, and the mixture weights from a block-level summary — sum the per-edge logits over the block and softmax,

  alpha_1, ..., alpha_K = softmax( sum_{i in b_t, 1<=j<=i} MLP_alpha(h_i^R - h_j^R) ).

The sum-then-softmax gives one mixture weight vector *per block*, so the block picks a "mode" jointly and then the edges are drawn from that mode's factorial distribution — which is the right granularity, because the correlation I just measured is among edges *in the same block-step*.

Now I need the training loss, and I want it numerically sane, because a product of many Bernoulli probabilities underflows and a sum-of-products inside a log is a logsumexp waiting to happen. For one block-step (call it a subgraph) under component `k`, the log-probability of the ground-truth edges is `sum_e log p_k(label_e)`, where `p_k(label_e)` is `theta_{k,e}` if the edge is present and `1 - theta_{k,e}` if absent — that's just the negative binary cross-entropy `-BCE(logit_{k,e}, label_e)` summed over the block's candidate edges `e`. So per component I accumulate the per-edge BCE into a per-subgraph total, negate it, add `log alpha_k`, and combine the components with a logsumexp:

  log p(block) = logsumexp_k ( log alpha_k + sum_e log p_k(edge_e) )
              = logsumexp_k ( log alpha_k - sum_e BCE(logit_{k,e}, label_e) ).

That is exactly `log sum_k alpha_k prod_e theta_{k,e}` computed stably, with `theta_{k,e}` read as the Bernoulli probability assigned by component `k` to the observed label of edge slot `e`. The implementation averages the per-edge `alpha` logits over the block's edge slots before `log_softmax`, a normalization that keeps the same block-level mixture-weight semantics while making the scale less sensitive to block size. For a fixed ordering, the graph log-probability is the sum over block log-probs; the training code also divides the accumulated log-probability by the number of edge slots for its optimized loss, while keeping the unnormalized log-probability for reporting the negative log-probability. If I evaluate several canonical orderings for the same graph, the graph objective is the logsumexp over those per-ordering log-probabilities. This is the mixture-of-Bernoulli likelihood, trained with a logsumexp over components and another logsumexp over orderings, and it's all parallel: every block-step's conditional is evaluated on its known prefix at once.

I've handled steps, conditioning, and within-step edge correlation. The permutation problem is still sitting there: my factorization is over `L^pi` for *some* ordering `pi`, and the true likelihood marginalizes `pi` over `N!` orderings. The popular shortcut is to pick one canonical ordering (BFS, say) and just model `p(A^pi)` for that — but that's a bet on a single ordering being good for every graph, and different graphs plausibly want different orderings. I'd like to do better than one ordering without paying `N!`.

I can restrict the orderings to a small *family* of canonical orderings `Q = {pi_1, ..., pi_M}`, each one a deterministic function of the graph (so they're cheap to compute), chosen so no two give the same adjacency matrix. Then instead of the intractable `log sum_pi p(G,pi)` I'd optimize

  log sum_{pi in Q~} p(G, pi) = log sum_{pi in Q} p(A^pi),

where `Q~` is the set of all orderings mapping (under the canonicalization) to a member of `Q`. Is this actually a lower bound on `log p(G)`, and is it actually better than the single-ordering objective everyone defaults to? The argument is that `Q~` is a strict subset of the full set of `N!` orderings and every term `p(G,pi)` is nonnegative, so dropping the missing terms can only shrink the sum, and `log` is monotone — and that summing several nonnegative terms is at least as big as keeping only one. Let me put numbers on it to make sure the inequalities point the way I think. Suppose for some graph three canonical orderings give `p(A^pi) = 0.02, 0.05, 0.01`, and imagine the *rest* of the `N!` orderings contribute another `0.5` of mass:

```
single ordering :  log(0.02)                = -3.912
sum over Q      :  log(0.02+0.05+0.01)      = -2.526
full marginal   :  log(0.08 + 0.5)          = -0.545
```

`-3.912 <= -2.526 <= -0.545`. So the family objective sits strictly above the single-ordering objective and strictly below the true `log p(G)` — it is a valid lower bound, and a tighter one than the single-ordering default. And enlarging `Q` only adds more nonnegative terms, pushing the middle number up toward the truth, at compute proportional to `|Q|` — so `|Q|` is a knob trading bound-tightness (which should track model quality) against cost.

I'd like to understand *why* this particular objective is the right thing and not just "sum a few orderings and hope," so let me read it as variational inference. The true objective is `log p(G) = log sum_pi p(G,pi)`; introduce a variational posterior `q(pi|G)` over orderings and write the evidence lower bound,

  log p(G) >= E_{q(pi|G)}[ log p(G,pi) ] + H(q(pi|G)),

the standard ELBO, valid for any `q` (gap is the KL from `q` to the true posterior over orderings). When I restrict `pi` to my `M` canonical orderings, `q(pi|G)` is just a categorical over `M` items. Now — what's the *best* `q`? Maximize the ELBO over the categorical `q` subject to `sum_pi q(pi) = 1`. Lagrangian:

  J = sum_pi q(pi) [ log p(G,pi) - log q(pi) ] + lambda ( sum_pi q(pi) - 1 ).

Differentiate with respect to a particular `q(pi)`:

  dJ/dq(pi) = log p(G,pi) - log q(pi) - 1 + lambda = 0
            => log q(pi) = log p(G,pi) + lambda - 1
            => q(pi) proportional to p(G,pi).

Normalize over the family and the multiplier drops out:

  q*(pi|G) = p(G,pi) / ( sum_{pi' in Q~} p(G,pi') ).

Now substitute `q*` back into the ELBO and see what it reduces to. The bracket is `log p(G,pi) - log q*(pi) = log p(G,pi) - log p(G,pi) + log sum_{pi'} p(G,pi') = log sum_{pi'} p(G,pi')`, which is constant in `pi`, so

  E_{q*}[log p(G,pi)] + H(q*) = sum_pi q*(pi) * log sum_{pi'} p(G,pi') = log sum_{pi' in Q~} p(G,pi').

That is exactly the objective I wrote down above. So optimizing `log sum_{pi in Q} p(A^pi)` is *implicitly* choosing the optimal soft combination of orderings per graph — the model isn't forced onto one ordering, it lets each graph weight the orderings by how well they explain it, and `q*` is read off the model itself rather than learned as a separate network. The two derivations land on the same expression — the numeric lower-bound check and the ELBO-collapse — so the "sum a few orderings" objective is not a heuristic; it *is* the optimal-posterior ELBO under the restricted family.

I should pick the family. I want orderings driven only by graph structure (since I'm after a universal model, not a molecule-specific one), each cheap to compute. The obvious ones: the default ordering the data came in; degree-descending; the BFS-tree and the DFS-tree rooted at the largest-degree node (BFS being the GraphRNN choice). Let me add one with a different structural bias — a `k`-core ordering — because cohesive-group structure is exactly the kind of global property the local orderings miss. The `k`-core of a graph is the maximal subgraph in which every node has degree at least `k`; cores are nested (the `i`-core sits inside the `j`-core when `i > j`), and the whole core decomposition can be computed in linear time in the number of edges. So I partition the nodes by their largest core number, order the cores from largest to smallest, and within a core order by degree descending — a core-descending ordering that surfaces the dense cohesive center first. That gives me a generic, structurally-diverse `Q` from which to pick subsets.

Two more pieces and I think I'm done. First, the speed-quality knob deserves a second lever. The block size `B` cuts steps but degrades quality as it grows, because a big block has to predict many edges in one shot. But notice I can decouple training from sampling. Train with block `B` and a *stride* of 1 — i.e. learn to predict the next `B` rows from *every* sub-prefix under the ordering, so the model sees all the partial graphs. Then at *test* time I can choose any stride `S` with `1 <= S <= B`: generate a `B`-block but keep only the first `S` rows, then advance by `S`. With `S = B` neighboring blocks don't overlap and it's fastest, `T = ceil(N/B)`. With `S < B` neighboring blocks overlap by `B - S` rows, so the dependency among rows in a block gets modeled across more than one step, recovering quality, with `T = floor((N-B)/S) + 1` steps — and no retraining, since the same trained conditional is just evaluated at a different stride. That's a free quality dial at inference, inspired by the stride in convolutions.

Second, sampling itself. At generation I don't have the ground-truth prefix, so I genuinely go step by step. Start the adjacency empty (lower-triangular). At each step, pad in the new block, add the candidate (augmented) edges as fully-connected guesses, run the GNN on this subgraph, sample a mixture component `k` from `softmax` of the (block-averaged) `alpha` logits, then draw each new edge as a Bernoulli with probability `sigmoid(logit_{k,i,j})`. Append the sampled rows, advance by the stride, repeat until the graph is full, then symmetrize `A = L + L^T`. And I model the number of nodes separately — at sampling time I draw the graph size from the empirical distribution of training-graph sizes — so I know when to stop.

Let me write this as the real model, filling the per-step-conditional slot of the harness with: the dimension-reducing node embedding, the GNN with attentive GRU message passing, and the mixture-of-Bernoulli output and its stable log-likelihood loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

EPS = float(np.finfo(np.float32).eps)


class GNN(nn.Module):
    """Attentive GRU message passing on the augmented graph (one generation step).

    message m_ij = f(h_i - h_j); attention a_ij = sigmoid(g([h_i,x_i]-[h_j,x_j])).
    The code carries x_i/x_j as a per-edge feature; state update h_i <- GRU(h_i, sum_j a_ij m_ij).
    No state is carried across generation steps -> steps train in parallel.
    """

    def __init__(self, msg_dim, node_dim, edge_feat_dim,
                 num_prop=1, num_layer=7, att_hidden_dim=128):
        super().__init__()
        self.num_prop = num_prop
        self.num_layer = num_layer
        self.edge_feat_dim = edge_feat_dim
        # GRU node-state update per layer (gated -> stable over rounds)
        self.update_func = nn.ModuleList([
            nn.GRUCell(input_size=msg_dim, hidden_size=node_dim)
            for _ in range(num_layer)])
        # state_diff is h_i - h_j; edge_feat carries the x_i/x_j marker used by the implementation
        self.msg_func = nn.ModuleList([
            nn.Sequential(nn.Linear(node_dim + edge_feat_dim, msg_dim),
                          nn.ReLU(), nn.Linear(msg_dim, msg_dim))
            for _ in range(num_layer)])
        # attention g sees [h_i, x_i] and [h_j, x_j] through state_diff plus edge_feat
        self.att_head = nn.ModuleList([
            nn.Sequential(nn.Linear(node_dim + edge_feat_dim, att_hidden_dim),
                          nn.ReLU(), nn.Linear(att_hidden_dim, msg_dim), nn.Sigmoid())
            for _ in range(num_layer)])

    def _prop(self, state, edge, edge_feat, layer):
        # state_diff is h_i - h_j; edge_feat carries the x_i/x_j marker used by the implementation
        state_diff = state[edge[:, 0], :] - state[edge[:, 1], :]
        edge_input = torch.cat([state_diff, edge_feat], dim=1) if self.edge_feat_dim > 0 else state_diff
        msg = self.msg_func[layer](edge_input)
        # a_ij = sigmoid(g(.)); attended message
        msg = msg * self.att_head[layer](edge_input)
        # aggregate by sum into the receiving node
        agg = torch.zeros(state.shape[0], msg.shape[1], device=state.device)
        agg = agg.scatter_add(0, edge[:, [1]].expand(-1, msg.shape[1]), msg)
        # h_i <- GRU(h_i, agg)
        return self.update_func[layer](agg, state)

    def forward(self, node_feat, edge, edge_feat):
        state = node_feat
        for ii in range(self.num_layer):
            if ii > 0:
                state = F.relu(state)
            for _ in range(self.num_prop):
                state = self._prop(state, edge, edge_feat, ii)
        return state


class GraphGenerator(nn.Module):
    """Block-wise auto-regressive graph generator with a GNN-attention per-step
    conditional and a mixture-of-Bernoulli output over the block's edges."""

    def __init__(self, max_nodes, hidden_dim=128, embedding_dim=128,
                 num_GNN_layers=7, num_GNN_prop=1, num_mix_component=20,
                 num_canonical_order=1, block_size=1, sample_stride=1,
                 att_edge_dim=64, edge_weight=1.0, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.block_size = block_size            # B
        self.sample_stride = sample_stride      # S (<= B)
        self.num_mix = num_mix_component        # K
        self.num_canonical_order = num_canonical_order
        self.att_edge_dim = att_edge_dim

        # h^0 = W L_b + b : linear dim-reduce of the adjacency row(s) (eq for node rep)
        self.decoder_input = nn.Linear(max_nodes, embedding_dim)

        # one-step GNN; edge feature carries the existing/new marker x_i (2*att_edge_dim one-hot)
        self.decoder = GNN(msg_dim=hidden_dim, node_dim=hidden_dim,
                           edge_feat_dim=2 * att_edge_dim,
                           num_prop=num_GNN_prop, num_layer=num_GNN_layers)

        # per-edge per-component Bernoulli logits theta_{k,i,j} from (h_i^R - h_j^R)
        self.output_theta = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_mix_component))
        # mixture-weight logits alpha_k from (h_i^R - h_j^R), summed over the block
        self.output_alpha = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_mix_component))

        pos_weight = torch.ones([1]) * edge_weight
        self.adj_loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.num_nodes_pmf = None               # empirical p(#nodes), set from data

    # ---- one generation step: existing subgraph + B new nodes + candidate edges ----
    def _augmented_graph(self, A_prefix, jj, K, device):
        """Build augmented graph on jj existing + K new nodes: existing edges plus
        fully-connected candidate (augmented) edges to/among the new block."""
        adj = F.pad(A_prefix[:jj, :jj], (0, K, 0, K), 'constant', value=1.0)  # candidates = 1
        adj = torch.tril(adj, diagonal=-1)
        adj = adj + adj.transpose(0, 1)
        edges = adj.to_sparse().coalesce().indices().t()                      # M x 2
        # x_i marker: 0 for existing nodes, 1..K one-of-K for the new block
        att_idx = torch.cat([torch.zeros(jj).long(),
                             torch.arange(1, K + 1)]).to(device).view(-1, 1)
        # carry the marker as a one-hot edge feature: src-half | dst-half
        ef = torch.zeros(edges.shape[0], 2 * self.att_edge_dim, device=device)
        ef = ef.scatter(1, att_idx[edges[:, 0]], 1.0)
        ef = ef.scatter(1, att_idx[edges[:, 1]] + self.att_edge_dim, 1.0)
        return edges, ef

    def _step_logits(self, node_state, edges, edge_feat, idx_row, idx_col):
        h = self.decoder(node_state, edges, edge_feat)         # GNN: h^R
        diff = h[idx_row, :] - h[idx_col, :]                   # h_i^R - h_j^R per candidate edge
        log_theta = self.output_theta(diff)                    # E x K
        log_alpha = self.output_alpha(diff)                    # E x K
        return log_theta, log_alpha

    def training_loss(self, adj, node_counts):
        """Maximum-likelihood on mixture-Bernoulli block conditionals.

        adj has shape [batch, num_canonical_order, N, N]. For each ordering,
        teacher-force every sub-prefix; for each graph, logsumexp over the
        canonical-ordering log-probabilities.
        """
        self.train()
        device = adj.device
        B, C, N, _ = adj.shape
        K = self.block_size
        if self.num_nodes_pmf is None:
            self._fit_num_nodes_pmf(node_counts, N)

        total_log_prob = []
        L = torch.tril(adj, diagonal=-1)                       # [B, C, N, N]
        for b in range(B):
            n = int(node_counts[b].item())
            order_logps = []
            for c in range(C):
                step_logps = []
                for jj in range(0, n - K + 1):                 # stride-1 teacher forcing
                    edges, ef = self._augmented_graph(L[b, c], jj, K, device)
                    # node features: dim-reduced rows for existing nodes, zeros for the new block
                    node_state = torch.zeros(jj + K, self.embedding_dim, device=device)
                    if jj > 0:
                        node_state[:jj] = self.decoder_input(L[b, c, :jj, :N])
                    # predict the full block-row rectangle, then symmetrize/tril downstream
                    ir, ic = np.meshgrid(np.arange(jj, jj + K), np.arange(jj + K))
                    idx_row = torch.from_numpy(ir.reshape(-1)).long().to(device)
                    idx_col = torch.from_numpy(ic.reshape(-1)).long().to(device)
                    label = adj[b, c, idx_row, idx_col]
                    log_theta, log_alpha = self._step_logits(node_state, edges, ef, idx_row, idx_col)
                    step_logps.append(
                        self._mixture_bernoulli_logp(label, log_theta, log_alpha))
                if step_logps:
                    order_logps.append(torch.stack(step_logps).sum())
            if order_logps:
                total_log_prob.append(torch.logsumexp(torch.stack(order_logps), dim=0))
        log_prob = torch.stack(total_log_prob).mean()
        return -log_prob

    def _mixture_bernoulli_logp(self, label, log_theta, log_alpha):
        # per-component negative BCE summed over the block's edges = log p_k(block)
        bce = torch.stack([self.adj_loss_func(log_theta[:, k], label)
                           for k in range(log_theta.shape[1])], dim=1)
        comp_logp = -bce.sum(dim=0)                            # K : sum_e log p_k(edge_e)
        # alpha from block-averaged logits -> log_softmax (never form raw alpha)
        log_alpha = F.log_softmax(log_alpha.mean(dim=0), dim=-1)   # K
        # log sum_k alpha_k prod_e p_k(label_e)  (stable)
        return torch.logsumexp(comp_logp + log_alpha, dim=0)

    def _fit_num_nodes_pmf(self, node_counts, N):
        pmf = torch.zeros(N + 1, device=node_counts.device)
        for c in node_counts.long():
            pmf[c] += 1.0
        self.num_nodes_pmf = pmf / pmf.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, n_samples, device):
        """Row-wise autoregressive draw; component k ~ softmax(alpha), edges ~ Bernoulli(theta_k)."""
        self.eval()
        K = self.block_size
        S = self.sample_stride
        N = self.max_nodes
        A = torch.zeros(n_samples, N, N, device=device)
        for ii in range(0, N - K + 1, S):
            jj = ii + K
            A[:, ii:, :] = 0.0
            A = torch.tril(A, diagonal=-1)
            for b in range(n_samples):
                edges, ef = self._augmented_graph(A[b], ii, K, device)
                node_state = torch.zeros(ii + K, self.embedding_dim, device=device)
                if ii > 0:
                    node_state[:ii] = self.decoder_input(A[b, :ii, :N])
                ir, ic = np.meshgrid(np.arange(ii, jj), np.arange(jj))
                idx_row = torch.from_numpy(ir.reshape(-1)).long().to(device)
                idx_col = torch.from_numpy(ic.reshape(-1)).long().to(device)
                keep = idx_row > idx_col
                idx_row, idx_col = idx_row[keep], idx_col[keep]
                if idx_row.numel() == 0:
                    continue
                log_theta, log_alpha = self._step_logits(node_state, edges, ef, idx_row, idx_col)
                k = torch.multinomial(F.softmax(log_alpha.mean(dim=0), -1), 1).item()
                p = torch.sigmoid(log_theta[:, k])
                A[b, idx_row, idx_col] = torch.bernoulli(p)
        A = torch.tril(A, diagonal=-1)
        A = A + A.transpose(1, 2)                              # symmetrize: A = L + L^T
        # number of nodes: draw from empirical pmf, else read off connectivity
        if self.num_nodes_pmf is not None:
            node_counts = torch.multinomial(self.num_nodes_pmf, n_samples, replacement=True)
            node_counts = node_counts.clamp(min=2).to(device)
        else:
            node_counts = (A.sum(dim=-1) > 0).long().sum(dim=-1).clamp(min=2)
        return A, node_counts
```
