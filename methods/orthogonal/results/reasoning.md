Let me start with the part that keeps showing up every time I make a network deeper. A signal is multiplied by a weight matrix at each layer. The backward error is multiplied by transposed weight matrices on the way back. If the typical scale per layer is a little below one, depth turns that into almost zero; if it is a little above one, depth turns it into an explosion. That explains the slogan, but it does not tell me the learning timescale. I want the actual dynamics: what determines how fast a deep network learns, and what initial condition keeps that speed from degrading with depth?

The nonlinear network is too hard to solve first, so I strip away the nonlinearity and keep the depth. A stack of linear layers is still only a linear input-output map, so it will not be expressive, but the learning dynamics in the weights are still nonlinear. The loss depends on a product of matrices, and the gradient for one matrix contains the product of all the others. That is exactly the coupling that depth introduces, now in a form I can write down.

For the three-layer case, with one hidden layer, the map is $y = W^{32} W^{21} x$. Batch gradient descent on squared error gives

$$\Delta W^{21} = \lambda \sum_\mu W^{32T}(y^\mu x^{\mu T} - W^{32}W^{21}x^\mu x^{\mu T}),$$

$$\Delta W^{32} = \lambda \sum_\mu (y^\mu x^{\mu T} - W^{32}W^{21}x^\mu x^{\mu T})W^{21T}.$$

With a small learning rate I take continuous time and write $\tau = 1/\lambda$. The data enter through the input correlation $\Sigma^{11}=\sum_\mu x^\mu x^{\mu T}$ and the input-output correlation $\Sigma^{31}=\sum_\mu y^\mu x^{\mu T}$:

$$\tau {dW^{21}\over dt}=W^{32T}(\Sigma^{31}-W^{32}W^{21}\Sigma^{11}), \qquad
\tau {dW^{32}\over dt}=(\Sigma^{31}-W^{32}W^{21}\Sigma^{11})W^{21T}.$$

The equations are cubic in the weights. The network is linear in $x$, but learning is not linear in the weights. To isolate the input-output structure, assume whitened inputs, $\Sigma^{11}=I$. Then the only data-dependent object left is $\Sigma^{31}$, so I take its SVD,

$$\Sigma^{31}=USV^T=\sum_\alpha s_\alpha u_\alpha v_\alpha^T.$$

Here $v_\alpha$ and $u_\alpha$ are the input and output modes the task asks me to connect, and $s_\alpha$ is the strength of that connection, so this basis is at least a natural place to look for decoupling. Let me try the change of variables $W^{21}=\overline W^{21}V^T$ and $W^{32}=U\overline W^{32}$ and see whether the data factors actually drop out. Substituting into the first equation, $W^{32T}\Sigma^{31}=\overline W^{32T}U^T USV^T=\overline W^{32T}SV^T$, and $W^{32}W^{21}\Sigma^{11}=U\overline W^{32}\overline W^{21}V^T$ (using $\Sigma^{11}=I$), so the whole right side carries a left factor of $\overline W^{32T}$ and a right factor of $V^T$ once I also rewrite the time-derivative side as $\frac{d}{dt}(\overline W^{21}V^T)$. The $U$ and $V$ factors cancel on both sides, leaving

$$\tau {d\over dt}\overline W^{21}=\overline W^{32T}(S-\overline W^{32}\overline W^{21}), \qquad
\tau {d\over dt}\overline W^{32}=(S-\overline W^{32}\overline W^{21})\overline W^{21T}.$$

Now the structure is visible. Let $a^\alpha$ be the column of $\overline W^{21}$ for input mode $\alpha$, and let $b^\alpha$ be the row of $\overline W^{32}$ for output mode $\alpha$. The dynamics contain a cooperative term, which tries to make $a^\alpha \cdot b^\alpha$ equal $s_\alpha$, and competitive cross-terms, which penalize the wrong pairings $a^\alpha \cdot b^\beta$ for $\alpha \ne \beta$:

$$\tau {d a^\alpha\over dt}=(s_\alpha-a^\alpha\cdot b^\alpha)b^\alpha-\sum_{\gamma\ne\alpha} b^\gamma(a^\alpha\cdot b^\gamma),$$

$$\tau {d b^\alpha\over dt}=(s_\alpha-a^\alpha\cdot b^\alpha)a^\alpha-\sum_{\gamma\ne\alpha} a^\gamma(b^\alpha\cdot a^\gamma).$$

Before solving the time course I want to know the fixed-point landscape, because that decides whether slowness is a trapping problem or a timing problem. Linear networks have saddles, but no bad local minima. The stable solution keeps the strongest available input-output modes and realizes the best low-rank approximation of $\Sigma^{31}$. So if training is slow, the question is not which local minimum traps me; the question is how long the dynamics linger near saddle-like low-strength configurations before the useful modes grow.

I can make the modes independent by starting on a decoupled manifold: for each active mode, $a^\alpha$ and $b^\alpha$ point along the same hidden-unit direction $r^\alpha$, and the $r^\alpha$ directions are mutually orthonormal. Then every cross-mode dot product is zero, and the competitive terms stay zero. One mode reduces to two scalars, $a$ and $b$, with mode strength target $s$:

$$\tau {da\over dt}=b(s-ab), \qquad \tau {db\over dt}=a(s-ab).$$

The product $ab$ wants to approach $s$. There is also a conserved quantity. Check $a^2-b^2$:

$$ {d\over dt}(a^2-b^2)=2a\dot a-2b\dot b={2\over \tau}(ab(s-ab)-ab(s-ab))=0.$$

So each trajectory follows a hyperbola $a^2-b^2=\hbox{constant}$ toward the fixed manifold $ab=s$. The origin is a fixed point too, but unstable; small random weights can sit near it, which is exactly the plateau.

I do not fully trust the algebra yet, so I integrate the two scalar equations directly with $s=1$, $\tau=1$, starting from an asymmetric small point $a=0.05$, $b=0.02$ (Euler, $dt=10^{-4}$). The invariant $a^2-b^2$ starts at $0.00210$ and reads $0.00210$ at the end, drifting only in the sixth decimal, and $ab$ climbs to $0.99999\ldots\approx s$. So the conserved hyperbola and the $ab\to s$ attractor are both real, not an artifact of how I grouped terms.

The symmetric case $a=b$ is the cleanest timescale calculation, and it is the right approximation when the two sides start with comparable small norms. Let $u=ab=a^2$. Then

$$\tau {du\over dt}=\tau(\dot a b+a\dot b)=b^2(s-u)+a^2(s-u)=2u(s-u).$$

This is a logistic equation. Separating variables gives

$$t=\tau\int_{u_0}^{u_f}{du\over 2u(s-u)}
={\tau\over 2s}\log {u_f(s-u_0)\over u_0(s-u_f)}.$$

If $u_0=\epsilon$ and I stop at $u_f=s-\epsilon$, then $t\approx (\tau/s)\log(s/\epsilon)$, so the dominant timescale is $O(\tau/s)$ with only a logarithmic cutoff dependence. Stronger input-output modes learn faster. Solving for the whole trajectory gives

$$u(t)={s e^{2st/\tau}\over e^{2st/\tau}-1+s/u_0}.$$

Let me put numbers on the timescale to be sure the logarithm is the right cutoff dependence. With $s=1$, $\tau=1$, $\epsilon=10^{-4}$, the formula predicts $t=\log(10^4)=9.21$. Integrating $\tau\dot u=2u(s-u)$ from $u_0=10^{-4}$, $u$ first crosses $s-\epsilon$ at $t=9.2103$ — the same value. And the closed form for $u(t)$ above matches the numerical $u$ at $t=20$ to six figures. The shape this produces is a long flat stretch near zero followed by a fast rise, since $u$ sits near the unstable origin for most of the elapsed time and the positive feedback in the product only takes over once $u$ is no longer tiny. That accounts for the plateau-then-jump behavior I started from, with no extra mechanism added.

Now add depth. Let the network have $N_l$ layers, hence $N_l-1$ weight matrices. On the analogous decoupled manifold, one mode has scalar strengths $a_1,\ldots,a_{N_l-1}$ and composite strength

$$u=\prod_{i=1}^{N_l-1} a_i.$$

The scalar layer dynamics give $\tau \dot a_i=(s-u)u/a_i$. Therefore

$$\tau \dot u
= \tau\sum_i \left(\prod_{k\ne i}a_k\right)\dot a_i
= (s-u)u^2\sum_i a_i^{-2}.$$

On the symmetric submanifold, all $a_i=a=u^{1/(N_l-1)}$, so

$$\tau {du\over dt}=(N_l-1)u^{2-2/(N_l-1)}(s-u).$$

For $N_l=3$ this returns $2u(s-u)$. For very large depth the exponent approaches $2$ and the prefactor approaches $N_l$, so the continuous-time equation becomes $\tau\dot u=N_l u^2(s-u)$. At fixed $\tau$ that even seems to get faster with depth. That cannot be the whole story because discrete gradient descent must keep a stable finite step size, and deeper products sharpen the curvature.

So I compute the curvature on the same symmetric manifold. Use the squared scalar loss with the continuous-time scale included,

$$E_\tau(a_1,\ldots,a_{N_l-1})={1\over 2\tau}\left(s-\prod_k a_k\right)^2.$$

For $i\ne j$,

$$g={\partial^2 E_\tau\over \partial a_i\partial a_j}
={1\over\tau}\left(\prod_{k\ne j}a_k\right)\left(\prod_{k\ne i}a_k\right)
-{1\over\tau}\left(s-\prod_k a_k\right)\prod_{k\ne i,j}a_k,$$

and on the diagonal,

$$h={\partial^2 E_\tau\over \partial a_i^2}
={1\over\tau}\left(\prod_{k\ne i}a_k\right)^2.$$

Set every $a_i=a$. Then

$$g={2\over\tau}a^{2N_l-4}-{1\over\tau}s a^{N_l-3}, \qquad
h={1\over\tau}a^{2N_l-4}.$$

The Hessian is a constant-diagonal, constant-off-diagonal matrix. The all-ones direction, which is the direction the symmetric dynamics moves along, has eigenvalue

$$\lambda_1=h+(N_l-2)g
=(2N_l-3){1\over\tau}a^{2N_l-4}-(N_l-2){1\over\tau}s a^{N_l-3}.$$

The maximum over the learning interval occurs at the optimum $a_{\rm opt}=s^{1/(N_l-1)}$. At that point $u=\prod_k a_k=a_{\rm opt}^{N_l-1}=s$, so the $(s-u)$ piece of $g$ vanishes and only the first term survives; substituting makes both powers carry the same factor $a^{2N_l-4}=s^{(2N_l-4)/(N_l-1)}$, and I get

$$\lambda_1(a_{\rm opt})=(N_l-1){1\over\tau}s^{(2N_l-4)/(N_l-1)}.$$

I check this against the explicit Hessian entries before trusting it. With $s=2$, $\tau=1$: for $N_l=3$, plugging $a_{\rm opt}=\sqrt2$ into $h$ and $g$ and forming $h+(N_l-2)g$ gives $4.000000$, and the closed form gives $(N_l-1)s^{(2N_l-4)/(N_l-1)}=2\cdot 2^1=4$. For $N_l=5$ both give $11.3137$, and for $N_l=11$ both give $34.822$. The formula and the entrywise Hessian agree, so the eigenvalue is right. For large $N_l$, this scales like $(N_l-1)s^2/\tau$, so a stable first-order learning rate scales as

$$\alpha_{\rm opt}\sim O\left({1\over N_l s^2}\right).$$

After folding in that depth-dependent rate, the infinite-depth delay over the three-layer case is finite:

$$t_\infty-t_3={cs\over u_0}-{cs\over u_f}\approx {cs\over \epsilon}$$

when $u_0=\epsilon$ and $u_f=s-\epsilon$. The important condition is hiding in that expression: the starting composite strength $u_0$ must be $O(1)$. If each layer starts with scalar strength $a_0<1$, then $u_0=a_0^{N_l-1}$ vanishes exponentially with depth, and the finite-delay story collapses. The exponent makes this brutal: at $N_l=51$, $a_0=0.9$ already gives $u_0=0.9^{50}\approx 5\times10^{-3}$, and any value below one keeps shrinking, while $a_0=1.1$ gives $1.1^{50}\approx 117$, an explosion. Only $a_0=1$ holds at $u_0=1$ for every depth. The initialization problem is now sharper: I need a data-independent way to make the end-to-end product preserve mode strength through arbitrarily many layers, and the per-layer multiplicative scale it sees has to be exactly one, not approximately one.

Greedy pretraining shows one way this can happen, but it uses data. In a linear autoencoder, pretraining is PCA: the input-output correlation for the pretraining target is the input correlation itself, $\Sigma^{11}=Q\Lambda Q^T$. Pretraining pushes the layer product toward a map diagonal in that PCA basis, with diagonal strengths near one. During supervised fine-tuning, this is useful when the task's right singular vectors $V$ match the input principal directions $Q$: then the network already sits on the decoupled manifold with composite strengths near one, which by the $u_0$ analysis above is exactly the condition for the depth delay to stay finite. That is a plausible account of the observed speedup, though I have not measured it here. Either way it does not give me the rule I want, because it needs an unsupervised phase and access to the data.

The obvious data-free candidate is scaled Gaussian initialization. For an `N x N` layer with entries drawn from $N(0,1/N)$,

$$E[W^T W]_{ij}=\sum_k E[W_{ki}W_{kj}]=\delta_{ij},$$

so $E[v^T W^T W v]=v^T v$. This is exactly the average norm-preservation argument. My worry is that this is only a statement about the mean, and the per-layer factor I actually need to control is the whole singular spectrum of the product, not its average. A single scaled Gaussian matrix has a spread of singular values, with squared singular values following the Marchenko-Pastur law, and I do not know offhand whether that spread averages out or compounds across depth.

So I measure it. I build products of depth $1,5,20,50$ from independent $64\times64$ matrices, one set scaled Gaussian with entries $N(0,1/N)$, one set orthogonal, and look at the singular values of the product:

```text
depth   Gaussian s_min   s_max    cond        Orth s_min  s_max
   1     9.8e-03         2.0e0    2.0e2            1.0000  1.0000
   5     3.9e-07         2.9e0    7.4e6            1.0000  1.0000
  20     5.1e-18         4.5e0    8.8e17           1.0000  1.0000
  50     1.8e-18         6.4e0    3.6e18           1.0000  1.0000
```

This is decisive, and worse than I expected. The Gaussian product's mean squared singular value does stay near one for a while (about $0.74$ at depth 20), so the average norm-preservation claim is not wrong. But the condition number runs away to $10^{18}$ by depth 20: the smallest singular value collapses to machine zero while the largest grows past one. The product preserves a typical vector's norm on average precisely because a few directions blow up while almost all directions are crushed. The orthogonal product, by contrast, sits at exactly one across the whole spectrum at every depth. So average preservation really is too weak, and the gap is not a constant-factor nuisance — it is exponential in depth.

The eigenvalue picture says why this is not a harmless fluctuation. The eigenvalues of Gaussian products concentrate near the origin with depth, while the singular values spread, as the measured $s_{\max}/s_{\min}$ already shows. A large mismatch between eigenvalues and singular values is the signature of a non-normal matrix, with highly non-orthogonal eigenvectors. In backpropagation, the transpose of that product has the same singular values, so error components in the small-singular-value subspace are crushed by factors like the measured $10^{-18}$ before they reach early layers. This is vanishing gradient behavior even though the average norm calculation looked fine.

So the condition I actually need is a full-spectrum condition: the product of layer Jacobians should act like a near-isometry, with as many singular values as possible in a narrow band around an $O(1)$ constant. Norm preservation in expectation is only the first moment shadow of this. The whole spectrum has to stay controlled.

The orthogonal column above already hints at what does control it. To get every singular value exactly one I want a matrix whose columns or rows form an isometric basis: if $Q^TQ=I$ in the square case, every direction has its norm preserved exactly. The question for depth is whether that property survives multiplication. Algebraically it should, since $(Q_2Q_1)^T(Q_2Q_1)=Q_1^TQ_2^TQ_2Q_1=Q_1^TQ_1=I$, and the measured orthogonal column held at $1.0000$ out to depth 50. To make sure I'm not fooling myself with a short product, I form a depth-30 product of $8\times8$ orthogonal factors and compute $\|P^TP-I\|_F$: it is $6\times10^{-15}$, i.e. zero to rounding. So initializing each layer as an isometric matrix keeps the end-to-end product isometric before training begins, which is the data-free way to hold $u_0$ at $O(1)$ regardless of depth.

The rectangular case is not a problem; it just limits what exact isometry can mean. A matrix with fewer rows than columns can have orthonormal rows, and a matrix with more rows than columns can have orthonormal columns. In either case, all nonzero singular values are one, which is the best available condition for a rectangular layer. For a convolution kernel, I flatten the trailing dimensions so the tensor is viewed as `out_channels x (in_channels * kH * kW)`.

Now I need the random draw to be the right one. Drawing a Gaussian matrix and taking QR gives an orthonormal factor, but the raw QR output has a sign ambiguity: if $Z=QR$, then $(Q\Lambda)(\Lambda^{-1}R)$ is also a valid factorization for any diagonal sign matrix $\Lambda$. A numerical QR routine fixes those signs by convention, which biases the distribution. The clean way is to force the diagonal of $R$ positive. In real arithmetic, set

$$d=\operatorname{sign}(\operatorname{diag}(R)), \qquad Q \leftarrow Q\,d.$$

In code this is `d = torch.diag(r, 0)`, `ph = d.sign()`, `q *= ph`. If `rows < cols`, transpose before QR and transpose back after, so the returned matrix is semi-isometric in the requested orientation. Finally multiply by a gain.

The nonlinear case tells me what that gain is trying to do. Take

$$x_i^{l+1}=\sum_j g W_{ij}^{(l+1,l)}\phi(x_j^l),$$

with the linear part isometric. Track the population variance

$$q^l={1\over N}\sum_i (x_i^l)^2.$$

Because the matrix preserves sums of squares,

$$q^{l+1}=g^2 {1\over N}\sum_j \phi(x_j^l)^2.$$

Approximating the layer activities as Gaussian with variance $q^l$ gives the one-dimensional recursion

$$q^{l+1}=g^2\int Dz\,\phi(\sqrt{q^l}z)^2.$$

For $\tanh$, near zero, $\tanh(\sqrt q z)\approx \sqrt q z$, so the recursion has slope $g^2$ at the origin. The zero fixed point is stable for $g<1$ and loses stability at $g_c=1$. That is the edge of chaos: below it activity dies; above it activity propagates with nonlinear saturation; at the boundary the linear gain balances the damping. For ReLU, the Gaussian second moment should be $\int Dz\,\max(0,\sqrt q z)^2=q/2$, since ReLU keeps half the Gaussian mass and that half carries half the variance by symmetry. I sanity-check the factor with a Monte Carlo at $q=1.7$: $\frac1M\sum\max(0,\sqrt{1.7}\,z)^2=0.850$ against $q/2=0.85$. So the unit-variance condition is $g^2/2=1$, hence $g=\sqrt 2$. A linear layer uses gain `1`.

There is one practical distinction I should not blur. The mathematical tanh edge in this analysis is $g_c=1$. PyTorch's `calculate_gain("tanh")` returns `5/3` for its own variance heuristic. The initializer I need to write is for ReLU-style convolutional networks, so the relevant library gain is `calculate_gain("relu") = sqrt(2)`. I should call the tested PyTorch primitive rather than reimplement QR inside the model hook, because that primitive already does the flattening, transpose for `rows < cols`, QR, Haar sign correction, copy, and gain multiplication.

So the initialization hook becomes exactly this:

```python
def initialize_weights(model, config):
    """Orthogonal initialization.

    Conv2d & Linear: orthogonal matrix (gain=sqrt(2) for ReLU).
    BatchNorm2d: weight=1, bias=0.
    """
    gain = nn.init.calculate_gain('relu')
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=gain)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```

Reading the chain back the way I built it: solving the linear dynamics turned learning into mode growth, with $\tau \dot u=2u(s-u)$ for a three-layer mode and a timescale of $O(\tau/s)$ that I confirmed numerically to four figures. In deeper nets the mode strength is a product across layers, and even after the stable learning rate shrinks like $1/(N_l s^2)$ — the eigenvalue formula I checked against the explicit Hessian — depth adds only a finite delay provided the initial composite strength is $O(1)$, which by $u_0=a_0^{N_l-1}$ forces a per-layer scale of exactly one. Scaled Gaussian weights satisfy the average-norm condition but, as the depth-50 product showed with a condition number of $10^{18}$, their products become non-normal and spectrally anisotropic, so gradients vanish in most directions. Isometric layer matrices keep all nonzero singular values at one and preserve that under multiplication (the depth-30 product stayed isometric to $10^{-15}$), and a nonlinearity-dependent gain — $\sqrt2$ for ReLU, from the second moment I checked — places the nonlinear network near the variance-propagating regime. The implementation calls the library's Haar-corrected semi-isometric initializer on convolutional and linear weights, leaves the bias-free convolution layers alone, zeroes the linear bias, and keeps batch-normalization affine parameters at the identity.
