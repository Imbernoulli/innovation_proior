We want a high-capacity network to learn a brand-new task from a handful of labeled examples and to do it in a few gradient steps, after meta-training on many related tasks drawn from a distribution $p(T)$. The frame I work in is the dominant one and I am happy to keep it: meta-learn a shared initialization $\theta$, adapt to each new task by a few gradient steps on its support set, and train $\theta$ so that the *query* loss after adaptation is small across tasks, differentiating the post-adaptation loss back through the inner step into $\theta$. It scales, it is architecture-agnostic, it works. The trouble is concentrated in the one place I get to design and the place the dominant method leaves crudest: the inner update is plain SGD, $\theta' = \theta - \alpha \nabla L$, with one global scalar rate and the step direction locked to the raw gradient. The only thing meta-learned is *where to start*; *how* to move is a fixed rule. Recent results say the choice of inner-loop optimizer matters to final performance, which nags, because I am pouring all the meta-learning into the initialization while leaving the optimizer at its dumbest setting.

The classical cure for a bad gradient direction is preconditioning. Plain gradient descent zig-zags exactly when the loss surface is ill-conditioned — a long narrow valley where the gradient points across the valley and barely along it. Newton's method steps $\theta - \alpha H^{-1}\nabla L$ to rescale into the local quadratic geometry; natural gradient steps $\theta - \alpha F^{-1}\nabla L$ using the Fisher information matrix; K-FAC approximates the Fisher as a Kronecker product to make that affordable. All of them replace $\nabla L$ with $M\nabla L$ for a curvature matrix $M$. But two things go wrong if I drop one of these into a few-shot inner loop. First, they *compute* $H$ or $F$ from the current training loss and data, and in few-shot I have only $K$ examples; curvature estimated from five points is curvature of the support set, and stepping aggressively along it means fitting five points as fast as possible, which is precisely how I overfit — the few-shot pathology is that fitting the support set is the wrong objective, and curvature computed to fit it faster makes things worse. Second, a full $M$ is $\dim(\theta)\times\dim(\theta)$ — the square of a hundred-thousand-dimensional vector for a conv net — impossible to store or meta-learn. The cheap learned alternative already in the air is the diagonal: a per-parameter learning-rate vector, the Meta-SGD step $\theta' = \theta - \alpha\circ\nabla L$ with $\alpha$ learned. That rescales each coordinate independently but cannot express that the gradient on one input channel should be combined with the gradient on another, or that filter positions within a kernel covary. Real loss curvature is not diagonal; the off-diagonal blocks are exactly the coordinate dependencies the diagonal throws away.

I propose Meta-Curvature (MC): an inner-loop adaptation rule that keeps MAML's two-loop structure but replaces the raw inner gradient with a learned, Kronecker-factored preconditioning of it, meta-trained jointly with the initialization. Two ideas make it work, one conceptual and one structural. The conceptual move resolves "where does $M$ come from." I have something the classical methods do not — a whole distribution of tasks — and the same hypothesis that justifies a shared initialization justifies a shared preconditioner: there exist curvatures broadly applicable across a task family, fixed by the architecture, the loss, and the task distribution rather than by any one task's five points. So I do not compute $M$ from the support loss at all; I *learn* it across $p(T)$ by the same meta-objective that learns $\theta$, shaping $M$ so the post-adaptation *query* loss generalizes, not so the support fits faster. $M$ becomes a meta-parameter, trained for held-out performance. That is exactly why this beats dropping a Hessian or K-FAC into the inner loop: those estimate curvature from the support statistics and overfit; mine is meta-learned for generalization and never inverted.

The structural move resolves "how do I afford it." A layer's parameters are not a flat vector — they are a tensor. A conv weight is $W\in\mathbb{R}^{C_{out}\times C_{in}\times d}$ with $d = h\cdot w$ the filter size ($d=1$ for a linear layer), and its gradient $G$ has the same shape. A full $M$ flattens this and mixes all $C_{out}C_{in}d$ coordinates with all others, but the *natural* dependencies run along the three modes separately: among the $d$ elements within a filter, among the $C_{in}$ input channels, and among the $C_{out}$ output channels. So instead of one giant matrix I factor the transform into three small matrices, one per mode — $M_o\in\mathbb{R}^{C_{out}\times C_{out}}$, $M_i\in\mathbb{R}^{C_{in}\times C_{in}}$, $M_f\in\mathbb{R}^{d\times d}$ — and apply each along its own axis of the gradient tensor with the $n$-mode product $X\times_n M$ (unfold along mode $n$, multiply by $M$, fold back):

$$\mathrm{MC}(G) = G \times_3 M_f \times_2 M_i \times_1 M_o.$$

Each product does something legible: $\times_3 M_f$ linearly recombines the $d$ elements inside each filter (spatial structure of a kernel), $\times_2 M_i$ recombines the gradient across input channels, and $\times_1 M_o$ recombines across output channels so each filter's gradient becomes a linear combination of all filters' gradients in the layer. Composed, the three examine the dependencies of all the layer's gradient coordinates, but parameterized by $d^2 + C_{in}^2 + C_{out}^2$ numbers instead of $(C_{out}C_{in}d)^2$. That is the affordability win. This is genuinely a transform over all coordinates and not three diagonal rescalings dressed up: products on distinct modes commute, $G\times_3 M_f\times_2 M_i\times_1 M_o = G\times_1 M_o\times_2 M_i\times_3 M_f$, so they apply "all together," and if one vectorizes, $\mathrm{vec}(\mathrm{MC}(G)) = M_{mc}\,\mathrm{vec}(G)$ with $M_{mc} = M_o\otimes M_i\otimes M_f$ for this $[C_{out},C_{in},d]$ convention — a Kronecker product of the three small matrices that is never materialized. This is the same structure K-FAC uses to approximate the Fisher ($A\otimes G$), which reassures me the inductive bias is sound; the difference is decisive: K-FAC computes its factors from training-loss statistics and inverts them, while MC learns its factors for generalization and never inverts, and by splitting the input mode and filter mode into separate $M_i$ and $M_f$ it avoids K-FAC's large $C_{in}d\times C_{in}d$ block — cheaper, and (smaller factors having less capacity to memorize) less prone to overfitting the meta-training set.

The inner update is then, per layer,

$$\theta' = \theta - \alpha\cdot\mathrm{MC}(\nabla L_{\text{train}}(\theta)) = \theta - \alpha\cdot(\nabla L \times_3 M_f \times_2 M_i \times_1 M_o),$$

with $\alpha$ the fixed inner learning rate. The meta-parameters are the initialization $\theta$ *and* all the $\{M_o, M_i, M_f\}$ across layers, meta-trained together. Two decisions the construction forces. First, initialization: at the start I have no reason to prefer any curvature and I want to begin from behavior I trust, namely MAML. If every $M$ is the identity then $\mathrm{MC}(G) = G$ exactly, the inner step reduces to $\theta - \alpha\nabla L$, and I am running MAML; so I initialize all three factors per layer to identity and let meta-training deform them away as it discovers which coordinate recombinations help the query loss generalize. MC is thus a strict generalization of MAML at identity. (I am more careful about Meta-SGD: a diagonal full preconditioner is the Meta-SGD direction, while diagonal MC factors give a *separable* diagonal across modes, so MC is best described as the structured non-diagonal version of that idea, not as a container for an arbitrary Meta-SGD vector.) Second, the outer optimization: the whole point of folding $M$ into the existing frame is that the meta-gradient of the transformed inner step is computed by the same autodiff machinery MAML already uses — I build $\mathrm{MC}(\nabla L)$ as a differentiable node, subtract $\alpha\cdot\mathrm{MC}(\nabla L)$ keeping the subtraction in the graph, forward the query set, and backprop the query loss into both $\theta$ and the $M$ matrices. No matrix inversion, and no higher-order machinery beyond the Hessian-vector product MAML already pays. I use Adam for the outer loop.

It is worth seeing why MC should *generalize* rather than merely fit faster, because that is the claim that separates it from naive second-order methods. Take the meta-gradient of a single flattened transform $M$, with per-task inner update $\theta^\tau = \theta - \alpha M\nabla L_{\text{train}}(\theta)$:

$$\nabla_M L_{\text{val}}^\tau(\theta^\tau) = -\alpha\,\nabla_\theta L_{\text{val}}^\tau(\theta^\tau)\,\nabla_\theta L_{\text{train}}^\tau(\theta)^\top,$$

an outer product that resembles the empirical Fisher $F = \mathbb{E}[\nabla_\theta\log p(y|x)\,\nabla_\theta\log p(y|x)^\top]$ but differs in three ways, each a feature. It uses *separate* train and validation sets, so $M$ is shaped to help held-out points rather than the support it adapted on (a within-task Fisher never sees the validation set). The validation gradient is evaluated at the *post-adaptation* parameters $\theta^\tau$, so it knows where adaptation lands. And the Fisher is positive semidefinite by construction whereas $M$ is not forced to be — I let it be indefinite, trading the descent guarantee for the freedom to transform the gradient in whatever way most helps generalization. Meta-curvature is, in short, a Fisher-shaped object trained on held-out loss after adaptation instead of estimated from training loss — exactly the generalization-versus-fitting distinction the few-shot setting demands.

In implementation each parameter gets its own curvature factors. For a convolutional weight in PyTorch layout $[C_{out}, C_{in}, k_h, k_w]$ I fold the spatial dimensions into $d = k_h k_w$, create identity-initialized $M_o$, $M_i$, $M_f$, reshape the gradient to $[C_{out}, C_{in}, d]$, apply the three mode products, and reshape back. For a 2D linear weight $d = 1$ makes the filter factor a redundant scalar, so I keep only the input and output matrices, following the released code. For 1D bias and normalization parameters an elementwise scale initialized to ones is enough; a dense matrix there would invent cross-coordinate coupling I did not justify. The inner update stays the differentiable $p \leftarrow p - \alpha\,\mathrm{MC}(g)$, and the inner optimizer returns only the curvature factors as meta-parameters while the outer loop separately optimizes the shared initialization.

```python
from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor, nn

INNER_LR = 0.5


class MetaCurvature(nn.Module):
    """Per-parameter meta-curvature transform.

    Conv weights use output/input/filter mode factors; 2D weights follow the
    released code and use only input/output factors; 1D parameters use an
    elementwise scale.
    """

    def __init__(self, param: Tensor):
        super().__init__()
        kw = {"device": param.device, "dtype": param.dtype}  # factors on param's device
        if param.dim() > 2:                        # conv [Co,Ci,kh,kw]
            c_out, c_in = param.shape[0], param.shape[1]
            d = int(param[0, 0].numel())
            self.c_out, self.c_in, self.d = c_out, c_in, d
            self.M_o = nn.Parameter(torch.eye(c_out, **kw))
            self.M_i = nn.Parameter(torch.eye(c_in, **kw))
            self.M_f = nn.Parameter(torch.eye(d, **kw))
            self.kind = "tensor"
        elif param.dim() == 2:                     # linear [Co,Ci]
            c_out, c_in = param.shape
            self.c_out, self.c_in = c_out, c_in
            self.M_o = nn.Parameter(torch.eye(c_out, **kw))
            self.M_i = nn.Parameter(torch.eye(c_in, **kw))
            self.kind = "matrix"
        else:                                      # released code uses elementwise 1D scaling
            self.scale = nn.Parameter(torch.ones_like(param))
            self.kind = "vector"

    def transform(self, g: Tensor) -> Tensor:
        if self.kind == "vector":
            return self.scale * g
        if self.kind == "matrix":
            G = torch.einsum("oi,ji->oj", g, self.M_i)        # input mode
            return torch.einsum("oi,jo->ji", G, self.M_o)     # output mode
        G = g.reshape(self.c_out, self.c_in, self.d)
        G = torch.einsum("oid,fd->oif", G, self.M_f)         # x3 M_f  (filter mode)
        G = torch.einsum("oid,ji->ojd", G, self.M_i)         # x2 M_i  (input mode)
        G = torch.einsum("oid,jo->jid", G, self.M_o)         # x1 M_o  (output mode)
        return G.reshape(g.shape)


class InnerLoopOptimizer(nn.Module):
    """MAML inner loop with a learned Kronecker-factored gradient preconditioner.
    Identity init gives MAML; dense mode factors give the MC transform."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        super().__init__()
        self.inner_lr = inner_lr
        self.mc = nn.ModuleList([MetaCurvature(p) for p in model.parameters()])

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), retain_graph=True, create_graph=True
            )
            updates = [-self.inner_lr * mc.transform(g)
                       for mc, g in zip(self.mc, grads)]
            model = self._apply_update(model, updates)
        return model

    def _apply_update(self, model: nn.Module, updates: List[Tensor]) -> nn.Module:
        i = 0
        for p in model.parameters():
            p.update = updates[i]
            i += 1

        def reroute(module: nn.Module) -> nn.Module:
            for name in module._parameters:
                p = module._parameters[name]
                if p is not None and getattr(p, "update", None) is not None:
                    module._parameters[name] = p + p.update
                    p.update = None
            for name in module._modules:
                module._modules[name] = reroute(module._modules[name])
            return module

        return reroute(model)

    def meta_parameters(self) -> List[Tensor]:
        return list(self.mc.parameters())
```
