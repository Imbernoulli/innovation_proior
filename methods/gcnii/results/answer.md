# GCNII: deep graph convolution via initial residual and identity mapping

**Problem.** Graph convolutions over-smooth: stacking $K$ renormalized layers realizes the fixed
operator $\tilde{\mathbf P}^{K}$, which contracts every signal to the degree-only stationary vector
$\boldsymbol\pi=\frac{\langle\tilde{\mathbf D}^{1/2}\mathbf 1,\mathbf x\rangle}{2m+n}
\tilde{\mathbf D}^{1/2}\mathbf 1$. For the lazy walk used to analyze a residual GCN,
$$\left|\mathbf p_i^{(K)}(j)-\frac{d_j+1}{2m+n}\right|
\le \sqrt{\frac{d_j+1}{d_i+1}}\left(1-\frac{\lambda_{\tilde G}^2}{2}\right)^K,$$
so a residual to the *previous* layer only slows convergence to the same fixed point.

**Key idea.** Two changes turn the vanilla layer into a genuinely deep one:
$$\mathbf H^{(\ell+1)}=\sigma\Big(\big((1-\alpha_\ell)\tilde{\mathbf P}\mathbf H^{(\ell)}+\alpha_\ell\mathbf H^{(0)}\big)\big((1-\beta_\ell)\mathbf I+\beta_\ell\mathbf W^{(\ell)}\big)\Big),\quad \tilde{\mathbf P}=\tilde{\mathbf D}^{-1/2}\tilde{\mathbf A}\tilde{\mathbf D}^{-1/2}.$$
- **Initial residual** to $\mathbf H^{(0)}$ (a learned map of the features), *not* the previous layer:
  every layer re-injects an $\alpha$-fraction of the input, so the deep limit is the input-carrying
  PageRank diffusion $\alpha(\mathbf I-(1-\alpha)\tilde{\mathbf P})^{-1}\mathbf H^{(0)}$ instead of the
  degree vector. $\alpha\approx0.1$.
- **Identity mapping** $(1-\beta_\ell)\mathbf I+\beta_\ell\mathbf W^{(\ell)}$ with
  $\beta_\ell=\log(\frac{\lambda}{\ell}+1)\approx\lambda/\ell$ decaying with depth: keeps the effective
  weight near identity, so the max singular value $s\approx1$ and the $s^{K}$ subspace collapse is
  defused; strong $L_2$ on $\mathbf W$ is safe because the linear-residual optimum has small norm and a
  unique global minimum. $\lambda\approx0.5$.

**Why it works.** Initial residual stops the *propagation* from smoothing; identity mapping stops the
*weights* from collapsing rank / overfitting the scarce labels. In the weak scalar recurrence
$\mathbf H^{(\ell+1)}=\gamma_\ell(\tilde{\mathbf P}\mathbf H^{(\ell)}+\mathbf x)$,
$$\mathbf H^{(K)}=\left(\sum_{\ell=0}^{K-1}
\left(\prod_{r=K-\ell-1}^{K-1}\gamma_r\right)(\mathbf I-\tilde{\mathbf L})^\ell\right)\mathbf x.$$
Matching those products to the binomial re-expansion
$\sum_{k=\ell}^{K-1}\theta_k(-1)^\ell{k\choose\ell}$ of
$\sum_k\theta_k\tilde{\mathbf L}^k$ gives arbitrary polynomial coefficients, where vanilla GCN's
coefficients are fixed. The same two terms also match one ISTA step for
$\frac12\|\mathbf B\mathbf x-\mathbf y\|_2^2+\lambda\|\mathbf x\|_1$:
$P_{\mu\lambda}((\mathbf I+\mu\mathbf W)\mathbf x^t+\mu\mathbf B^\top\mathbf y)$.

**Architecture / hyperparameters.** Fixed hidden width $H$: one input FC ($d\to H$) forms
$\mathbf H^{(0)}$, then $L$ conv layers ($H\to H$, weights $H\times H$ so cost is linear in depth), then
an output FC ($H\to$ classes). Dropout before each conv and the output FC, ReLU activation; Adam, lr
$0.01$, strong weight decay on conv weights, mild on the FCs. Semi-supervised citation settings: Cora
$L{=}64,\alpha{=}0.1,\lambda{=}0.5,H{=}64$; CiteSeer $L{=}32,\alpha{=}0.1,\lambda{=}0.6,H{=}256$; PubMed
$L{=}16,\alpha{=}0.1,\lambda{=}0.4,H{=}256$.

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """One sparse-adjacency layer with theta = log(lambda / layer + 1)."""

    def __init__(self, channels):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(channels, channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, h0, lamda, alpha, layer):
        theta = math.log(lamda / layer + 1.0)
        hi = torch.spmm(adj, x)
        support = (1 - alpha) * hi + alpha * h0
        return theta * torch.mm(support, self.weight) + (1 - theta) * support


class GCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout, alpha=0.1, lamda=0.5):
        super().__init__()
        self.convs = nn.ModuleList(GraphConvolution(hidden_channels) for _ in range(num_layers))
        self.fcs = nn.ModuleList([
            nn.Linear(in_channels, hidden_channels),
            nn.Linear(hidden_channels, out_channels),
        ])
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        h0 = layer_inner
        for i, conv in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                conv(layer_inner, adj, h0, self.lamda, self.alpha, i + 1)
            )
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)
```
