The step-decay schedule came back and the consumed $\epsilon$ told the story: on MNIST and Fashion-MNIST two of three seeds reported $6.398$ rather than $3.0$, and the mean spend was $5.27$ on the grayscale tasks and $4.13$ on CIFAR-10 — roughly double the budget everyone else sat at, and *still* its accuracy (MNIST 95.62, FMNIST 78.93, CIFAR-10 54.02) trailed. That is the worst of both worlds, and it confirms the risk I had flagged: the equivalent-uniform multiplier reported through `get_effective_sigma` did not round-trip through the fixed harness accountant. The $\sqrt{\text{steps}/\sum_t 1/\sigma_t^2}$ I handed back was computed for the tCDP/RDP notion of "equivalent," but the harness charges a different, fixed RDP bound, and on the late low-noise plateaus the real spend ran ahead of the reported scalar — precisely on the seeds where the staircase cut noise. The accuracy column is therefore not a like-for-like number. The lesson is sharp: laundering a non-uniform schedule through an accountant I cannot edit introduced a place where claimed and spent budget could come apart, and they did. Before I try to be clever again, I need to stand on the one mechanism whose budget I can certify *exactly* through this harness — the mechanism the accountant was written for.

I propose **standard DP-SGD**: fixed per-sample $L_2$ clipping with constant Gaussian noise, the canonical mechanism of Abadi et al. (2016), built here from the privacy requirement up so I know exactly why its accounting is the one the harness honors. The released object is a model whose weights an adversary can read, and the guarantee is worst-case: $M$ is $(\epsilon,\delta)$-private if for any two datasets differing in one record and any outcome set $S$, $\Pr[M(d)\in S] \le e^\epsilon \Pr[M(d')\in S] + \delta$. Privatizing the *endpoint* of training is dead on arrival here — the Gaussian mechanism would need the $L_2$-sensitivity of "the final weights of SGD on a non-convex loss," and one changed example amplified through thousands of steps can move the endpoint arbitrarily, so the only honest noise level is hopeless. So I privatize the *process* instead, controlling the data's influence at the one object it touches each step: the gradient.

The obstacle is that the per-example gradient norm is unbounded — an outlier produces an enormous gradient, so the *summed* gradient has unbounded sensitivity and noise cannot be calibrated. I have to make the sensitivity finite by force, by capping each example's contribution. The cleanest cap in $L_2$ — the norm the Gaussian mechanism wants — is to rescale each per-sample gradient,
$$\bar g = \frac{g}{\max(1,\ \|g\|/C)},$$
which leaves $g$ untouched when $\|g\| \le C$ and otherwise scales its norm down to exactly $C$ while keeping its direction. This is the `clip_factor = min(1, C/||g||)` the mechanism computes, and it is computed *flat* — one norm over the whole per-example gradient vector, not per layer — because the sensitivity is of the whole vector. The load-bearing detail is the *placement*: ordinary deep learning clips the *averaged* batch gradient after summing, which bounds the whole-batch norm but says nothing about any individual's contribution — a giant per-example gradient can hide inside a batch whose average is small. Privacy must bound *each person's* influence on the released sum, so the clip is applied *per example, before averaging*. With that, the sensitivity is exact: under the add/remove convention $d = d' \cup \{\text{one example}\}$, the two summed-clipped aggregates differ by a single clipped vector of norm at most $C$, so the $L_2$-sensitivity of the sum is $C$ and of the mean over $B$ examples is $C/B$.

Now the noise. To privatize a sum of sensitivity $C$, add $N(0,\sigma^2 C^2 I)$; dividing by the batch size to get the average gives noise of standard deviation $\sigma\,C/B$ per coordinate — exactly `noise_multiplier * max_grad_norm / batch_size`. Note how cleanly the two factor: $C$ is the sensitivity set by the gradient geometry, and $\sigma$ is a unitless multiplier that, together with $C$, sets the privacy. The descent step is then ordinary SGD on this noised average, which is what the frozen optimizer does. Per step: take the batch, compute per-example gradients, clip each in $L_2$ to $C$, average, add $N(0,(\sigma C/B)^2)$, and step.

The accounting is the heart of why *this* rung is trustworthy. Running $T$ subsampled-Gaussian steps, advanced composition would give $\epsilon \approx \epsilon_0\sqrt{2T\ln(1/\delta')}$ — the $\sqrt T$ is what makes iterative private training feasible — but it composes only each step's *tail*, blind to the known Gaussian shape, and overpays a $\sqrt{\log(T/\delta)}$ factor, pushing the reported $\epsilon$ toward ten where the structure allows far less. The fix is the moments/RDP accountant: track $\alpha(\lambda)$, the log-MGF of the per-step privacy loss; it composes *linearly* across the $T$ steps; convert back to $(\epsilon,\delta)$ by a Markov tail bound minimized over $\lambda$; and bound the single-step subsampled-Gaussian moment by $q^2\lambda(\lambda+1)/((1-q)\sigma^2)$, whose first-order term vanishes — which is why the cost is $O(q^2)$, not $O(q)$, and why subsampling at rate $q = B/n$ makes the budget cheap. Summing over $T$ and optimizing $\lambda$ yields $\sigma \ge c\,q\sqrt{T\log(1/\delta)}/\epsilon$, single-digit $\epsilon$ on a real net. This is exactly the shape of the harness's fixed `compute_epsilon(steps, sigma, q, delta)`: an RDP bound that assumes a single uniform $\sigma$ across all steps, which `calibrate_noise_to_epsilon` inverts by binary search.

And there is the reason this rung's budget is exact while the schedule's drifted. The harness accountant *is* the uniform-$\sigma$ moments accountant. The standard mechanism reports that very $\sigma$ unchanged through `get_effective_sigma`, so the spent and reported budgets are the same object — no translation, no place for them to come apart. The schedule had to translate a non-uniform run into an "equivalent" scalar and the translation leaked; the standard mechanism never translates anything, so it sits at $3.0016$/$2.9986$ by construction. That is the whole reason I retreat to it as the reference rung: every later mechanism must beat *this* at this exact budget, not above it. I expect consumed $\epsilon \approx 3.0$ on all nine runs with no $6.398$ anywhere, grayscale accuracy at least matching the overspending schedule (proving its extra spend bought nothing), and a CIFAR-10 number in the mid-50s that becomes the honest floor where later, correctly-accounted adaptive ideas can look for real headroom.

```python
class DPMechanism:
    """Standard DP-SGD (Abadi et al., 2016).

    Fixed per-sample gradient clipping + constant Gaussian noise.
    """

    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.n_params = n_params
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

    def clip_and_noise(self, per_sample_grads, step, epoch):
        batch_size = per_sample_grads[0].shape[0]

        # Compute per-sample gradient norms (flat norm across all parameters)
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norms = flat.norm(2, dim=1)  # [B]

        # Clip per-sample gradients
        clip_factor = (self.max_grad_norm / norms.clamp(min=1e-8)).clamp(max=1.0)  # [B]

        noised_grads = []
        for g in per_sample_grads:
            shape = [batch_size] + [1] * (g.dim() - 1)
            clipped = g * clip_factor.reshape(shape)

            # Average over batch
            avg = clipped.mean(dim=0)

            # Add calibrated Gaussian noise
            noise = torch.randn_like(avg) * (
                self.noise_multiplier * self.max_grad_norm / batch_size
            )
            noised_grads.append(avg + noise)

        return noised_grads

    def get_effective_sigma(self, step, epoch):
        return self.noise_multiplier
```
