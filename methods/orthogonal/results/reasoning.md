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

This basis should be the right coordinate system because $v_\alpha$ and $u_\alpha$ are the input and output modes the task asks me to connect, and $s_\alpha$ is the strength of that connection. Write $W^{21}=\overline W^{21}V^T$ and $W^{32}=U\overline W^{32}$. The $U$ and $V$ factors cancel, leaving

$$\tau {d\over dt}\overline W^{21}=\overline W^{32T}(S-\overline W^{32}\overline W^{21}), \qquad
\tau {d\over dt}\overline W^{32}=(S-\overline W^{32}\overline W^{21})\overline W^{21T}.$$

Now the structure is visible. Let $a^\alpha$ be the column of $\overline W^{21}$ for input mode $\alpha$, and let $b^\alpha$ be the row of $\overline W^{32}$ for output mode $\alpha$. The dynamics contain a cooperative term, which tries to make $a^\alpha \cdot b^\alpha$ equal $s_\alpha$, and competitive cross-terms, which penalize the wrong pairings $a^\alpha \cdot b^\beta$ for $\alpha \ne \beta$:

$$\tau {d a^\alpha\over dt}=(s_\alpha-a^\alpha\cdot b^\alpha)b^\alpha-\sum_{\gamma\ne\alpha} b^\gamma(a^\alpha\cdot b^\gamma),$$

$$\tau {d b^\alpha\over dt}=(s_\alpha-a^\alpha\cdot b^\alpha)a^\alpha-\sum_{\gamma\ne\alpha} a^\gamma(b^\alpha\cdot a^\gamma).$$

The fixed-point picture matters before I solve the time course. Linear networks have saddles, but no bad local minima. The stable solution keeps the strongest available input-output modes and realizes the best low-rank approximation of $\Sigma^{31}$. So if training is slow, the question is not which local minimum traps me; the question is how long the dynamics linger near saddle-like low-strength configurations before the useful modes grow.

I can make the modes independent by starting on a decoupled manifold: for each active mode, $a^\alpha$ and $b^\alpha$ point along the same hidden-unit direction $r^\alpha$, and the $r^\alpha$ directions are mutually orthonormal. Then every cross-mode dot product is zero, and the competitive terms stay zero. One mode reduces to two scalars, $a$ and $b$, with mode strength target $s$:

$$\tau {da\over dt}=b(s-ab), \qquad \tau {db\over dt}=a(s-ab).$$

The product $ab$ wants to approach $s$. There is also a conserved quantity. Check $a^2-b^2$:

$$ {d\over dt}(a^2-b^2)=2a\dot a-2b\dot b={2\over \tau}(ab(s-ab)-ab(s-ab))=0.$$

So each trajectory follows a hyperbola $a^2-b^2=\hbox{constant}$ toward the fixed manifold $ab=s$. The origin is a fixed point too, but unstable; small random weights can sit near it, which is exactly the plateau.

The symmetric case $a=b$ is the cleanest timescale calculation, and it is the right approximation when the two sides start with comparable small norms. Let $u=ab=a^2$. Then

$$\tau {du\over dt}=\tau(\dot a b+a\dot b)=b^2(s-u)+a^2(s-u)=2u(s-u).$$

This is a logistic equation. Separating variables gives

$$t=\tau\int_{u_0}^{u_f}{du\over 2u(s-u)}
={\tau\over 2s}\log {u_f(s-u_0)\over u_0(s-u_f)}.$$

If $u_0=\epsilon$ and I stop at $u_f=s-\epsilon$, then $t\approx (\tau/s)\log(s/\epsilon)$, so the dominant timescale is $O(\tau/s)$ with only a logarithmic cutoff dependence. Stronger input-output modes learn faster. Solving for the whole trajectory gives

$$u(t)={s e^{2st/\tau}\over e^{2st/\tau}-1+s/u_0}.$$

That sigmoid explains the plateau and sudden transition without inventing any extra mechanism: the mode starts near the unstable origin, then the positive feedback in the product takes over.

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

The maximum over the learning interval occurs at the optimum $a_{\rm opt}=s^{1/(N_l-1)}$. Substituting that value makes both powers carry the same factor, and I get

$$\lambda_1(a_{\rm opt})=(N_l-1){1\over\tau}s^{(2N_l-4)/(N_l-1)}.$$

For large $N_l$, this scales like $(N_l-1)s^2/\tau$, so a stable first-order learning rate scales as

$$\alpha_{\rm opt}\sim O\left({1\over N_l s^2}\right).$$

After folding in that depth-dependent rate, the infinite-depth delay over the three-layer case is finite:

$$t_\infty-t_3={cs\over u_0}-{cs\over u_f}\approx {cs\over \epsilon}$$

when $u_0=\epsilon$ and $u_f=s-\epsilon$. The important condition is hiding in that expression: the starting composite strength $u_0$ must be $O(1)$. If each layer starts with scalar strength $a_0<1$, then $u_0=a_0^{N_l-1}$ vanishes exponentially with depth, and the finite-delay story collapses. If $a_0>1$, the product explodes. The initialization problem is now sharper: I need a data-independent way to make the end-to-end product preserve mode strength through arbitrarily many layers.

Greedy pretraining shows one way this can happen, but it uses data. In a linear autoencoder, pretraining is PCA: the input-output correlation for the pretraining target is the input correlation itself, $\Sigma^{11}=Q\Lambda Q^T$. Pretraining pushes the layer product toward a map diagonal in that PCA basis, with diagonal strengths near one. During supervised fine-tuning, this is useful when the task's right singular vectors $V$ match the input principal directions $Q$. Then pretraining has put the network close to the decoupled, high-strength manifold. That explains the speedup, but it does not give me the rule I want, because it needs an unsupervised phase and access to the data.

The obvious data-free candidate is scaled Gaussian initialization. For an `N x N` layer with entries drawn from $N(0,1/N)$,

$$E[W^T W]_{ij}=\sum_k E[W_{ki}W_{kj}]=\delta_{ij},$$

so $E[v^T W^T W v]=v^T v$. This is exactly the average norm-preservation argument. But average preservation is too weak. A single scaled Gaussian matrix has a spread of singular values, with squared singular values following the Marchenko-Pastur law. When I multiply many independent such matrices, the end-to-end product can still preserve the norm of a typical random vector on average while becoming extremely anisotropic: most singular values shrink toward zero, and a small tail becomes very large.

The eigenvalue picture says why this is not a harmless fluctuation. The eigenvalues of Gaussian products concentrate near the origin with depth, while the singular values spread. A large mismatch between eigenvalues and singular values is the signature of a non-normal matrix, with highly non-orthogonal eigenvectors. In backpropagation, the transpose of that product has the same singular values, so error components in the small-singular-value subspace are crushed before they reach early layers. This is vanishing gradient behavior even though the average norm calculation looked fine.

So the condition I actually need is a full-spectrum condition: the product of layer Jacobians should act like a near-isometry, with as many singular values as possible in a narrow band around an $O(1)$ constant. Norm preservation in expectation is only the first moment shadow of this. The whole spectrum has to stay controlled.

What matrix has every singular value exactly equal to one? A matrix whose columns or rows form an isometric basis. If $Q^TQ=I$ in the square case, every direction has its norm preserved exactly. Products keep the same property, because $(Q_2Q_1)^T(Q_2Q_1)=Q_1^TQ_2^TQ_2Q_1=I$. That is the data-free way to keep $u_0$ from decaying with depth: initialize each layer as an isometric matrix, so the end-to-end product is isometric before training begins.

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

For $\tanh$, near zero, $\tanh(\sqrt q z)\approx \sqrt q z$, so the recursion has slope $g^2$ at the origin. The zero fixed point is stable for $g<1$ and loses stability at $g_c=1$. That is the edge of chaos: below it activity dies; above it activity propagates with nonlinear saturation; at the boundary the linear gain balances the damping. For ReLU, the Gaussian second moment is $\int Dz\,\max(0,\sqrt q z)^2=q/2$, so the unit-variance condition is $g^2/2=1$, hence $g=\sqrt 2$. A linear layer uses gain `1`.

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

The causal chain is now tight. Solving the linear dynamics turns learning into mode growth, with $\tau \dot u=2u(s-u)$ for a three-layer mode and timescale $O(\tau/s)$. In deeper nets, the mode strength is a product across layers, and even after the stable learning rate shrinks like $1/(N_l s^2)$, depth adds only a finite delay if the initial composite strength is $O(1)$. Scaled Gaussian weights preserve average norm but their products become non-normal and spectrally anisotropic, so gradients vanish in most directions. Isometric layer matrices keep all nonzero singular values at one, products preserve that property, and a nonlinearity-dependent gain places the nonlinear network near the variance-propagating regime. The implementation calls the library's Haar-corrected semi-isometric initializer on convolutional and linear weights, leaves the bias-free convolution layers alone, zeroes the linear bias, and keeps batch-normalization affine parameters at the identity.
