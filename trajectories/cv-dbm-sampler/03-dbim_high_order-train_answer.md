DBIM did exactly what I expected on the two easy workloads and stalled exactly where I feared on the third. Edges→Handbags fell from DDBM's 11.139 to **5.180** and ImageNet inpainting from 10.556 to **6.070** — roughly halving FID at one-tenth the calls — and the DIODE wall-clock cratered from 2149 seconds to 271. But DIODE came in at **15.002**, essentially tied with DDBM's 15.811: the deterministic-leaning first-order update bought almost nothing there. The diagnosis is not about stochasticity or the model; on the hard translation, *first-order Euler is leaving accuracy on the table at five steps*. Each DBIM step holds the predictor's $\hat x_0$ constant over the step, and on DIODE — where $\hat x_0$ swings hard between large steps — that flat approximation accumulates real error. The fix must keep everything that worked — same marginal-preserving family, same booting noise, five calls — and only make each step *more accurate*, ideally for free.

The doorway is to see what continuous object the deterministic DBIM step is discretizing. Set $\eta = 0$, write $t_{n+1} = t$, $t_n = t - \Delta t$, and stare at $x_{t-\Delta t} = a_{t-\Delta t} x_T + b_{t-\Delta t} \hat x_0 + (c_{t-\Delta t}/c_t)(x_t - a_t x_T - b_t \hat x_0)$. The troublesome factor is that $c_t$ ratio, so I divide through by $c_{t-\Delta t}$ and regroup:
$$\frac{x_{t-\Delta t}}{c_{t-\Delta t}} = \frac{x_t}{c_t} + \Big(\frac{a_{t-\Delta t}}{c_{t-\Delta t}} - \frac{a_t}{c_t}\Big) x_T + \Big(\frac{b_{t-\Delta t}}{c_{t-\Delta t}} - \frac{b_t}{c_t}\Big)\hat x_0.$$
This is a finite difference of $x_t/c_t$ equal to finite differences of $a_t/c_t$ and $b_t/c_t$ weighting the two endpoints — the Euler discretization of $d(x_t/c_t) = x_T\,d(a_t/c_t) + \hat x_\theta\,d(b_t/c_t)$. So the natural state variable is $x_t/c_t$, not $x_t$, and the noise scale $c_t$ that was blowing up the linear part has been divided out, leaving a clean low-curvature ODE — the bridge analogue of how DDIM became an Euler step on $x/\sqrt{\alpha}$. Expanding this back into $dx_t$ via the product rule and comparing against the bridge PF-ODE (score replaced by the data predictor) matches all three coefficients term for term: $c'/c = f + g^2/\sigma^2 - g^2/2c^2$, $a' - a c'/c = g^2 a/2c^2$, $b' - b c'/c = -g^2 b/2c^2$. The deterministic DBIM step is not an approximation to the PF-ODE; it is an exact reparameterization into coordinates where the linear part is trivial.

I propose the **high-order exponential-integrator DBIM solver**, which exploits that cleanness. The ODE is *semi-linear* — a linear-in-$x$ part plus a nonlinear network part — and the linear part is now isolated, so I never let a generic integrator chew on it: cancel it analytically with variation-of-constants and only approximate the integral of the smooth network output. Writing $dx_t = [A x_t + B_T x_T + B_\theta \hat x_\theta]\,dt$ with $A = c'/c$, the integrating factor is $e^{\int A} = c_t/c_s$; the $x_T$ integral closes in elementary form because $x_T$ is constant. The only genuinely hard integral carries the network output, and I change variable to the bridge log-SNR $\lambda_t = \log(b_t/c_t)$ — since $b_t/c_t = \sqrt{\mathrm{SNR}_t - \mathrm{SNR}_T}$, this is the half-log of the excess signal-to-noise over the endpoint, playing the role $\log(\alpha/\sigma)$ plays for ordinary diffusion. In this variable the exact solution from $t$ to $s < t$ is
$$x_s = \frac{c_s}{c_t} x_t + \Big(a_s - \frac{c_s}{c_t} a_t\Big) x_T + c_s \int_{\lambda_t}^{\lambda_s} e^{\lambda}\, \hat x_\theta\big(x_{t_\lambda}, t_\lambda, x_T\big)\, d\lambda.$$
The linear and endpoint parts are exact; all discretization error lives in that one exponentially-weighted integral — exactly the error DBIM's first order was paying on DIODE.

To raise the order I Taylor-expand $\hat x_\theta$ as a function of $\lambda$ about the current node and integrate each Taylor term against $e^\lambda$ exactly. With $h$ the step in $\lambda$ ($\lambda$ increases as time decreases, so $h > 0$), repeated integration by parts gives the scalar coefficients $\int e^\lambda d\lambda = e^{\lambda_s}(1 - e^{-h})$, $\int (\lambda - \lambda_t) e^\lambda d\lambda = e^{\lambda_s}(h - 1 + e^{-h})$, $\int \tfrac12(\lambda - \lambda_t)^2 e^\lambda d\lambda = e^{\lambda_s}(h^2/2 - h + 1 - e^{-h})$ — the $\varphi$-functions of exponential integrators, handed to me by the integrals rather than chosen. Keeping one bracket is first order and reduces exactly to the DBIM Euler step; two terms is second order, three is third.

The decision that controls cost under five calls is how to estimate the $\lambda$-derivatives of the network output. The single-step road inserts an extra intermediate timestep and finite-differences there — but that is an extra denoiser call per step, so a $k$-th-order single-step method costs $k$ calls per step and five calls buy only $\lfloor 5/k \rfloor$ steps; I cannot afford to halve my already-tiny step count. The multistep road, Adams-Bashforth style, finite-differences against the predictor outputs I *already computed* at previous timesteps — sitting in a buffer, free. Multistep costs exactly one new call per step, so five calls buy five steps and each step's $h$ is smaller, shrinking the dropped $O(h^{k+1})$ error too. Under a tight budget multistep is the only sane choice. With one previous time $u$, the first derivative is the backward difference $\hat x_t^{(1)} \approx (\hat x_t - \hat x_u)/h_1$, $h_1 = \lambda_t - \lambda_u$; for third order I fit the unique quadratic through the three most recent outputs and read off its first and second derivatives at the current node, with unequal-spacing divided-difference weights that keep the estimate consistent on the non-uniform schedule I am forced into.

The non-uniformity and the edge cases fall out of the same booting logic DBIM already needed. The first step is still singular ($c_{t_{\max}} = 0$), so before the loop I take the same stochastic booting sample — predict at $t_{\max}$, draw the booting noise, seed the first interior state with $a x_T + b \hat x_0 + c\cdot\text{noise}$, the one shot of stochasticity and the same latent. The first ordinary loop transition has no trustworthy $\lambda$-history yet — only the boot prediction at the pinned endpoint where $\lambda(t_{\max})$ is singular — so it must drop to first order. And at the last step I drop to first order too ("lower-order final"): as $h$ shrinks near the sharp endpoint the derivative estimates get noisy and a clean low-order finish avoids amplifying that noise into the final image. So the schedule is: boot sample, order-1 first transition, order-$k$ multistep in the middle, order-1 finish, with the buffer of past outputs filling between. This is DBIM's lower-order-final logic lifted to the high-order solver, spending the *same* five calls — one booting prediction plus four transitions — so any gain is pure accuracy.

Two grounding details where the harness's edit differs from the clean derivation: the canonical high-order solver *raises* on an unsupported order, but this editable `sample_dbim` silently *coerces* `order` $\notin \{2,3\}$ to `order = 2` — a defensive default so a bad caller still runs at second order — and it prints a `"Step order N"` line on rank 0 each iteration, pure diagnostics with no numeric effect. The default is `order = 2`, `lower_order_final = True`, and the signature drops `eta` entirely: this variant is the deterministic $\eta = 0$ solver, with no stochasticity dial — the only noise is the boot. Everything else — the booting prediction, the mask re-blend each call, the six-value return with the booting noise in the sixth slot — is identical to DBIM.

```python
@torch.no_grad()
def sample_dbim(
    denoiser,
    diffusion,
    x,
    ts,
    mask=None,
    order=2,
    lower_order_final=True,
    seed=None,
    **kwargs,
):
    if order not in [2, 3]:
        order=2
    x_T = x
    path = []
    pred_x0 = []

    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    u = diffusion.t_max
    if u == 1.0:
        u -= 5e-5
    u = [u for _ in range(order - 1)]
    xu_hat = [x0_hat.detach().clone() for _ in range(order - 1)]

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        # First Order Update, t < s
        if (lower_order_final and i + 1 == len(ts) - 1) or (i == 0):
            if dist.get_rank() == 0:
                print("Step order 1")
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]

            tmp_var = c_t / c_s
            coeff_xs = tmp_var
            coeff_x0_hat = b_t - tmp_var * b_s
            coeff_xT = a_t - tmp_var * a_s

            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            x_old = x
            x = coeff_xs * x_old + coeff_x0_hat * x0_hat + coeff_xT * x_T

        # Second Order Update, t < s < u
        elif order == 2 or i == 1:
            if dist.get_rank() == 0:
                print("Step order 2")
            a_u, b_u, c_u = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u, lambda_s, lambda_t = (
                torch.log(b_u / c_u),
                torch.log(b_s / c_s),
                torch.log(b_t / c_t),
            )

            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            h = lambda_t - lambda_s
            h2 = lambda_s - lambda_u
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat + (torch.exp(-h) + h - 1) * (x0_hat - xu_hat[-1]) / h2
            )
            x_old = x
            x = x_old * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        elif order == 3:
            if dist.get_rank() == 0:
                print("Step order 3")
            a_u1, b_u1, c_u1 = [
                append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-1] * ones)
            ]
            a_u2, b_u2, c_u2 = [
                append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-2] * ones)
            ]
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u2, lambda_u1, lambda_s, lambda_t = (
                torch.log(b_u2 / c_u2),
                torch.log(b_u1 / c_u1),
                torch.log(b_s / c_s),
                torch.log(b_t / c_t),
            )
            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1

            h = lambda_t - lambda_s
            h1 = lambda_s - lambda_u1
            h2 = lambda_u1 - lambda_u2
            dx0_hat = ((x0_hat - xu_hat[-1]) * (2 * h1 + h2) / h1 - (xu_hat[-1] - xu_hat[-2]) * h1 / h2) / (h1 + h2)
            d2x0_hat = 2 * ((x0_hat - xu_hat[-1]) / h1 - (xu_hat[-1] - xu_hat[-2]) / h2) / (h1 + h2)
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat
                + (torch.exp(-h) + h - 1) * dx0_hat
                + (h**2 / 2 - h + 1 - torch.exp(-h)) * d2x0_hat
            )
            x_old = x
            x = x_old * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        u.append(s)
        u.pop(0)
        xu_hat.append(x0_hat)
        xu_hat.pop(0)

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())

    return x, path, nfe, pred_x0, ts, first_noise
```
