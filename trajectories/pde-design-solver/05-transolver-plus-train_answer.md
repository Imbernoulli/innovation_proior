Transolver did exactly what I bet whole-domain physics attention would do, on every metric the graph rungs left on the table. rho_d climbed to 0.9896 — finally past *both* the U-Net's 0.981 and GraphSAGE's 0.978 — confirming that the residual after the graph trilogy really was global correlation, and that a single Physics-Attention layer relating the front-region state to the wake state directly is what orders designs correctly. The drag-magnitude error fell to 0.0136 (from GraphSAGE's 0.0193), and the field errors beat GraphSAGE across both published benchmarks: Car pressure 0.0809 and velocity 0.0218, AirfRANS pressure 0.0335 and velocity 0.0156. The AirCraft probe I flagged diverged upward to 0.684/0.411 — the overfitting-a-small-custom-set risk I named — but the published benchmarks are a clean sweep. The only lever I left open was the slicing itself: *can the assignment be made sharper and the slices more distinguishable than a single fixed-temperature softmax allows?*

The weakness is concrete. Transolver's slice assignment is a softmax over the $M$ slice axis with a *single learnable temperature per head* — one scalar scaling the logits identically for every point in that head. Two things about that are suboptimal once I stare at the geometry. First, the right sharpness is not uniform across the domain. A point squarely in the middle of a clean regime — deep in the wake, flat on the high-pressure nose — should commit hard to one slice; a point on a *boundary* between regimes (the separation line where flow detaches, a shock front, the edge of the front region) is genuinely ambiguous and should be allowed to split across slices. A single per-head temperature cannot express "be decisive here, be soft there"; it forces one sharpness on points whose ideal sharpness differs by where they sit in the flow. Second, even with a tuned temperature, a plain softmax over only $M=32$ logits tends to leave the slices *overlapping* — many points spread non-trivial mass across several slices, so the tokens are not as distinguishable as they could be, and attention among muddy tokens is attention among muddy states. The mass-normalized token of a slice that shares half its points with three neighbors is a blend, not a clean physical state. The two weaknesses are coupled: a global sharpness knob and a soft assignment that lets slices bleed into one another, yielding tokens less *eidetic* — less sharply distinguishable — than the physics warrants.

I propose **Transolver++**, whose contribution is making the slice assignment eidetic with two coupled changes. The first is an **adaptive per-point temperature**. Instead of one learned scalar per head, I predict a temperature *for each point* from that point's own feature — a small MLP mapping the per-head point feature ($\mathrm{dim\_head} \to \mathrm{slice\_num} \to 1$) to a positive scalar, added to a learnable per-head bias and clamped to a small positive floor so it never reaches zero. Now a point in a clean regime can learn a low temperature (sharp, decisive commitment) while a boundary point learns a high temperature (soft, hedged across the regimes it straddles), and the model chooses this *locally* from the feature, not globally. This is the natural generalization of Transolver's single temperature — Transolver is the special case where the predicted temperature is constant across points. The second change makes the assignment **more decisive** without collapsing the gradient: replace the plain softmax over the slice logits with a **Gumbel-softmax**, adding Gumbel noise to the logits before the softmax. The Gumbel-softmax is the continuous relaxation of sampling a hard one-hot assignment; with the adaptive temperature controlling its sharpness, it pushes each point's assignment toward a near-categorical choice (a point mostly *belongs* to one slice, the way a hard argmax would assign it) while staying differentiable, so the slice projection still gets a clean gradient. The noise also regularizes against the lazy near-uniform assignment a plain softmax can settle into. Together these make the slices eidetic: sharply distinguishable physical states, each token a clean representative rather than a blend, because points commit decisively and the per-point temperature lets the commitment vary with the local ambiguity. Attention among $M$ clean tokens is attention among genuinely distinct states — the same $O(M^2)$ cost, a sharper operator.

There is a second, independent gain hiding in this redesign, and it directly serves the parameter budget. Transolver's Physics-Attention uses a *two-stream* point projection: one linear map produces the feature that *decides* the slice (`x_mid`), a separate map produces the *content* averaged into the token (`fx_mid`). That decoupling doubles the input-projection parameters, but once the assignment is eidetic the content and assignment can share a single stream without much loss — the sharply-committed assignment already concentrates the right content into each token. So I collapse to a **single stream**: one projection `x_mid` serves both as the slice-decision feature *and* as the content averaged into the token, removing an entire `Linear(dim, inner_dim)` per attention layer. This matters *here specifically* because of the budget: the harness rejects any model over $1.05\times$ Transolver-256, and I want to run the finale at the *same* `n_hidden=256, slice_num=32` for a fair comparison, so I cannot afford to *add* parameters. The adaptive-temperature MLP is small, and dropping the second input stream more than pays for it, so the finale lands *under* the Transolver parameter count at equal width — comfortably inside the budget while being a strictly richer slicing operator. That is the 30–70% footprint reduction the method is known for, and on this edit surface it is what makes the upgrade admissible at all.

Landing this on the task's edit surface has one structural subtlety. The task ships `Physics_Attention_Irregular_Mesh` as a *read-only* module in `layers.Physics_Attention`, so I cannot edit it; but the edit surface lets me rewrite the entire `Custom.py` body, so I define the eidetic attention class **inline in Custom.py** — the same scaffold move Transolver used for its block, one level deeper. The block and `Model` wrapper stay byte-for-byte the canonical pre-norm Transolver skeleton (`fx = Attn(LN(fx)) + fx; fx = mlp(LN(fx)) + fx`, last block carrying the read-out, `preprocess` lifting `fun_dim + space_dim → n_hidden`, the learned `placeholder` bias, `geo` ignored), because the contribution is entirely in the attention sublayer. Inside the sublayer I strip two things from the reference that do not apply here: the `torch.distributed.nn.all_reduce` calls (those sum slice statistics across GPUs in the million-scale multi-GPU setting; here it is one mesh on one device, batch one, so there is nothing to all-reduce) and the gradient-checkpointing wrapper (a memory optimization, not part of the algorithm; the frozen loop handles training). What remains is exactly the algorithm: single-stream `x_mid` projection, a `proj_temperature` MLP plus a learnable bias clamped to a small positive floor, Gumbel-softmax slice weights with that adaptive temperature, mass-normalized tokens, scaled-dot-product attention among the $M$ tokens, deslice through the same weights, output projection. The slice projection keeps the orthogonal initialization (decorrelated slice directions at init), inherited from Transolver and equally motivated here.

The bar this finale must clear, falsifiable against Transolver's measured numbers since it carries no feedback of its own: sharper, more distinguishable slices should most help where physical regimes have sharp boundaries a soft single-temperature softmax blurs. On the published benchmarks I expect it to **at least match, and aim to beat, Transolver's rho_d 0.9896 and c_d 0.0136**, and to **reduce the field errors below Car 0.0809/0.0218 and AirfRANS 0.0335/0.0156** — particularly the velocity fields, where sharp separation and wake boundaries are exactly the ambiguous regions an adaptive per-point temperature is built to handle. rho_d is already near-saturated at 0.99, so the clearer win to look for is the field relative-L2, where there is more room. The strict requirement is that it not *regress* on the published benchmarks while staying *under* the Transolver-256 parameter budget — and the single-stream collapse is what guarantees the headroom. If the eidetic slicing genuinely produces cleaner tokens, the field errors should fall and the parameter count should drop at the same time. AirCraft I treat as task-internal and not a verdict either way, given Transolver already diverged there. With $M=32$ at width 256, bias init 0.5, clamp floor 0.01, and Gumbel `tau=1` noise, `CONFIG_OVERRIDES = {'n_hidden': 256, 'slice_num': 32}` keeps the comparison fair.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from einops import rearrange
from layers.Basic import MLP


def gumbel_softmax(logits, tau=1.0):
    # continuous relaxation of a hard slice assignment; differentiable, near-categorical
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    y = (logits + gumbel_noise) / tau
    return F.softmax(y, dim=-1)


class Physics_Attention_Irregular_Mesh_Eidetic(nn.Module):
    """Eidetic Physics-Attention: single-stream slicing with a per-point adaptive temperature
    and a Gumbel-softmax assignment, so slices are sharply distinguishable physical states.
    Cost O(N*M*C + M^2*C). all_reduce / checkpointing from the multi-GPU reference are dropped."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # learnable per-head bias added to the predicted per-point temperature
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        # per-point temperature predictor: dim_head -> slice_num -> 1 (positive via GELU)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )

        # SINGLE stream: x_mid both decides the slice AND supplies the token content
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)  # decorrelate slice directions at init

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, C = x.shape

        # (1) single-stream per-head point features
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()                       # B H N dim_head

        # (2) adaptive per-point temperature, floored positive
        temperature = self.proj_temperature(x_mid) + self.bias      # B H N 1
        temperature = torch.clamp(temperature, min=0.01)

        # (3) eidetic slice assignment: Gumbel-softmax over the M slice axis
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)  # B H N M
        slice_norm = slice_weights.sum(2)                           # B H M  (slice mass)
        slice_token = torch.einsum("bhnc,bhng->bhgc", x_mid, slice_weights)        # weighted sum
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None]
                                     .repeat(1, 1, 1, self.dim_head))               # -> weighted mean

        # (4) attention among the M physics-aware tokens
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(q, k, v)   # B H M dim_head

        # (5) deslice back to points through the SAME slice weights
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Transolver_block(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout=0., act='gelu', mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh_Eidetic(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.__name__ = 'Custom'
        self.args = args
        self.preprocess = MLP(args.fun_dim + args.space_dim, args.n_hidden * 2, args.n_hidden,
                              n_layers=0, res=False, act=args.act)
        self.blocks = nn.ModuleList([Transolver_block(num_heads=args.n_heads, hidden_dim=args.n_hidden,
                                                      dropout=args.dropout,
                                                      act=args.act,
                                                      mlp_ratio=args.mlp_ratio,
                                                      out_dim=args.out_dim,
                                                      slice_num=args.slice_num,
                                                      last_layer=(_ == args.n_layers - 1))
                                     for _ in range(args.n_layers)])
        self.placeholder = nn.Parameter((1 / (args.n_hidden)) * torch.rand(args.n_hidden, dtype=torch.float))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx, T=None, geo=None):
        # geo (mesh graph) ignored: Physics-Attention needs no graph
        if fx is not None:
            fx = torch.cat((x, fx), dim=-1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return fx


CONFIG_OVERRIDES = {'n_hidden': 256, 'slice_num': 32}
```
