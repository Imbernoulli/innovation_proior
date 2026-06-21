The problem is to build a neural network whose input is a *set* — an unordered collection $\{x_1,\dots,x_n\}$ of variable size — when the target is a property of the whole set. Two requirements fall straight out of the definition. The model must be permutation invariant, $f(\{x_1,\dots,x_n\}) = f(\{x_{\pi(1)},\dots,x_{\pi(n)}\})$ for every permutation $\pi$, because the two argument multisets are literally the same object; and it must swallow any element count $n$. A plain feed-forward net fails both — fixed-size input, and it reads order. An RNN handles variable length but is order-sensitive: feeding it $\{x_1,x_2\}$ versus $\{x_2,x_1\}$ returns different answers, which is simply wrong for a set. The known way to satisfy both at once is to bake symmetry into the architecture by encoding each element independently with some $\varphi$, combining the encodings with a symmetric reduction, and post-processing with $\rho$,
$$\mathrm{net}(\{x_1,\dots,x_n\}) = \rho\big(\mathrm{pool}(\{\varphi(x_1),\dots,\varphi(x_n)\})\big),$$
where the symmetry of $\mathrm{pool}$ (sum, mean, max) makes the whole map permutation invariant regardless of $\varphi$ and $\rho$. This is not just a hack: with $\mathrm{pool}=\mathrm{sum}$ and continuous $\rho,\varphi$, the family $\rho(\mathrm{sum}(\varphi(\cdot)))$ — more precisely $\mathrm{rFF}(\mathrm{sum}(\mathrm{rFF}(\cdot)))$ — is a universal approximator of permutation-invariant set functions (Zaheer et al. 2017, Deep Sets). So why is this not the end of the story? Because "$\varphi$ acts on each element independently" is doing a great deal of damage. Every element is embedded in isolation, and only *afterward* does any combination happen, through a fixed symmetric reduction; all information about how the elements relate to one another is squeezed out before pooling. The sharp counterexample is amortized clustering: a network that maps a point set directly to its cluster centers must assign each point to a cluster *and* respect explaining-away, so the clusters do not fight over the same points — which is exactly why clustering is normally done by iterative refinement like EM, each step looking at all points jointly. A pooling net can only learn to *quantize* the space, and that quantization is baked into $\varphi$'s weights, so it cannot depend on the contents of the particular input set: two different point sets receive the same fixed partition. The failure mode is under-fitting on any task whose answer depends on the current set's internal geometry. What I actually need is for the elements to interact *during* encoding — pairwise and higher-order — and for the pooling to be learnable and content-dependent rather than a fixed mean or max.

I propose the Set Transformer, an attention-based set model built entirely from a single reusable attention block. The reason attention is the right primitive is its symmetry: with queries $Q\in\mathbb{R}^{n\times d_q}$, keys $K\in\mathbb{R}^{n_v\times d_q}$, and values $V\in\mathbb{R}^{n_v\times d_v}$, the operation $\mathrm{Att}(Q,K,V;\omega)=\omega(QK^\top)V$ scores each query against each key, with $\omega(\cdot)=\mathrm{softmax}(\cdot/\sqrt{d})$ a scaled softmax whose $1/\sqrt{d}$ keeps the dot products from blowing up and saturating the softmax as $d$ grows, then returns for each query a weighted average of the values. If I permute the key/value set — reorder the rows of $K$ and $V$ together — the columns of $QK^\top$ permute and the rows of $V$ permute the same way, and a softmax-weighted sum over those rows is invariant to their order; permute the *queries* and the output rows permute identically. So attention is permutation invariant in its keys/values and permutation equivariant in its queries, exactly the pair of symmetries a set model needs. I use the multi-head version: project $Q,K,V$ into $h$ subspaces with learnable $W_j^Q,W_j^K,W_j^V$, run a head in each, concatenate, and mix with $W^O$, because one shared similarity over the full $d$ dimensions is a narrow channel while $h$ heads attend in $h$ different learned subspaces and capture several relationships at once; I set $d_q=d_v=d$ and $d_q^M=d_v^M=d/h$ so the concatenation returns to width $d$.

I wrap this into a reusable Multihead Attention Block, the MAB, on two sets $X$ (queries) and $Y$ (keys/values):
$$H = \mathrm{LayerNorm}\big(X + \mathrm{Multihead}(X,Y,Y;\omega)\big),\qquad \mathrm{MAB}(X,Y) = \mathrm{LayerNorm}\big(H + \mathrm{rFF}(H)\big),$$
where $\mathrm{rFF}$ is a row-wise feed-forward layer applied identically and independently to each row, so it preserves equivariance. I take this from the Transformer encoder block precisely because that block already combines attention with the residual, normalization, and feed-forward structure that makes attention trainable in depth, but I strip two things. First the positional encoding: a set has no positions, and adding position information would make the block order-sensitive and destroy permutation invariance — this is the whole reason a Transformer cannot be used off the shelf. Second dropout, which I simply do not want here. Because the keys/values $Y$ enter only through the order-invariant attention sum and everything touching the query rows is row-wise, $\mathrm{MAB}(X,Y)$ is equivariant in $X$ and invariant to permutations of $Y$. To encode a set while letting its elements interact, I feed the set to itself, defining the Set Attention Block $\mathrm{SAB}(X):=\mathrm{MAB}(X,X)$: now every element queries every other element, and each output representation has absorbed information about all the others — pairwise interactions. Stack two SABs and the second sees representations that already encode pairwise structure, so it captures higher-order interactions. With $Q=K=V=X$ this might look as if it collapses to a plain residual block, but it does not, because the per-head projections $W_j^Q,W_j^K,W_j^V$ let each head compare *projected* views of the elements rather than identity-mixing; and SAB is permutation equivariant, since permuting the input rows permutes query, key, and value rows together, the attention matrix is conjugated by that same permutation, and every residual, layer norm, and row-wise feed-forward step preserves it.

The remaining problem with SAB is cost: it forms the full $n\times n$ attention matrix, $O(n^2)$ in time and memory, which is prohibitive for a point cloud of thousands of points or a large clustering dataset. The structural fact I exploit is that a big set usually has low-rank interaction structure — its $n$ elements can be summarized through far fewer than $n$ representatives — which is exactly the inducing-point idea from sparse Gaussian processes (Snelson & Ghahramani 2005) and the Nyström approximation (Fowlkes 2004): route everything through $m\ll n$ inducing points and pay $O(nm)$ instead of the full Gram matrix. I transplant this into attention by introducing $m$ trainable vectors $I\in\mathbb{R}^{m\times d}$, parameters of the block learned with everything else, and routing through them with two MABs to build the Induced Set Attention Block:
$$H = \mathrm{MAB}(I,X)\in\mathbb{R}^{m\times d},\qquad \mathrm{ISAB}_m(X) = \mathrm{MAB}(X,H)\in\mathbb{R}^{n\times d}.$$
First the inducing points $I$ are the queries and $X$ the key/value set, so $H$ is $m$ summary vectors of the whole set computed in $O(nm)$; then the set reads back, $X$ as queries and $H$ as keys/values, producing $n$ outputs again in $O(nm)$, linear in $n$ for fixed $m$. This is structurally a low-rank bottleneck — project the set onto $m$ summaries, then reconstruct an $n$-element output — but the goal is to extract good features, not to reconstruct, so the inducing points are free to learn whatever global structure best explains the set; in amortized clustering they can spread as landmarks on the plane and elements get compared indirectly through their proximity to those shared landmarks. ISAB stays permutation equivariant in $X$: in $H=\mathrm{MAB}(I,X)$ the queries are the fixed inducing points and $X$ is the key/value set, so by MAB's key/value invariance permuting $X$ leaves $H$ unchanged — $H$ is invariant in $X$; then in $\mathrm{MAB}(X,H)$ the set is the query and $H$ is now fixed, so by MAB's query equivariance the output rows permute exactly as $X$ does. So ISAB is equivariant like SAB, but linear-time.

For the decoder I replace fixed pooling, which weights every element equally (mean) or keeps just one by a fixed rule (max), with content-dependent weighting. The max-of-a-set target makes the case: the answer is recoverable from a single element, the largest, so the right aggregation would *find* and attend to it — impossible for mean, and max only works because that task happens to match its hard rule. So I introduce $k$ trainable seed vectors $S\in\mathbb{R}^{k\times d}$ and let them attend over the encoded set, Pooling by Multihead Attention:
$$\mathrm{PMA}_k(Z) = \mathrm{MAB}(S,\mathrm{rFF}(Z)).$$
The seeds are the queries and the feed-forward-refined set the keys/values, so the output is $k$ vectors, each a learned, content-dependent weighted readout, permutation invariant in $Z$ because $Z$ enters only as keys/values. Usually one seed suffices, but some problems need several correlated outputs — clustering wants $k$ cluster centers, and what each center should be depends on where the others are — so I use $k$ seeds and then run a SAB over them, $H=\mathrm{SAB}(\mathrm{PMA}_k(Z))$, whose self-attention among the $k$ pooled vectors lets the model reason about the clusters jointly rather than emit $k$ independent guesses. The full model is then an equivariant encoder, $\mathrm{SAB}(\mathrm{SAB}(X))$ for small $n$ or $\mathrm{ISAB}_m(\mathrm{ISAB}_m(X))$ for large $n$, followed by an invariant decoder $\mathrm{Decoder}(Z)=\mathrm{rFF}(\mathrm{SAB}(\mathrm{PMA}_k(Z)))$; equivariant-then-invariant composes to permutation invariant by construction, for any $n$.

I should confirm I have not lost expressive power. The mean is a special case of softmax attention: with a single zero query $s=0$ and $X$ as keys/values, $\mathrm{Att}(s,X,X;\mathrm{softmax})=\mathrm{softmax}(sX^\top/\sqrt d)X=\mathrm{softmax}(0)X=\tfrac1n\sum_i x_i$, because softmax of an all-zeros logit vector is uniform. Power means $M_p(z)=(\tfrac1n\sum_i z_i^p)^{1/p}$ follow by setting the seed to zero, using a front row-wise feed-forward map to realize $z\mapsto z^p$ and a back one to realize $u\mapsto u^{1/p}$ coordinatewise, one-dimensional heads whose value projection selects a coordinate, and forcing every query–key logit equal by zeroing the query projection and bias so the construction reduces to the zero-query mean above — note that merely projecting query and key onto the same coordinate would yield data-dependent logits, not a mean, so the zero-logit construction is what is needed. Exact sum pooling needs the broader activation family $\omega(t)=1+f(t)$ with $f(0)=0$ (identity, ReLU, centered sigmoid): a zero seed makes every logit $0$, so $\omega(0)=1$, every value gets weight one, and the output is $\sum_i z_i$. Finally, suppressing the attention contribution in every SAB and ISAB — $W^O=0$ in the full definition, or zeroing the value/attention path in the simplified block — collapses the encoder to an instance-wise row-wise map $Z=\mathrm{rFF}(X)$, after which the decoder forms $\mathrm{rFF}(\mathrm{sum}(Z))$ by the non-softmax sum construction, so the model realizes $\mathrm{rFF}(\mathrm{sum}(\mathrm{rFF}(X)))$, which is universal (Zaheer et al. 2017). Attention is therefore not *needed* for universality — the wider attention family contains Deep Sets — but it is the mechanism that makes interaction-heavy tasks learnable in practice, where the universal-in-theory pooling baseline under-fits. The practical module below uses the robust softmax version, dividing logits by $\sqrt{\dim_V}$, with $\mathrm{rFF}$ realized as a single linear-plus-ReLU residual, and the leading $\mathrm{rFF}(Z)$ inside PMA dropped because the preceding block already ends in a row-wise feed-forward layer.

```python
import torch, torch.nn as nn, torch.nn.functional as F, math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    def __init__(self, dim_in, num_outputs, dim_out,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super().__init__()
        self.enc = nn.Sequential(
            ISAB(dim_in, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_out))
    def forward(self, X):
        return self.dec(self.enc(X))
```
