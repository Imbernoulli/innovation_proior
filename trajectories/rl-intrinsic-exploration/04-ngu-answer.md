# Step 4 answer (finale) — NGU (Never Give Up)

**Method.** Novelty on two timescales, combined multiplicatively: a *within-episode* kNN
pseudo-count that resets every episode (so the agent never gives up re-exploring), modulated by RND's
slow *lifelong* novelty (so genuinely-mastered regions stop paying extra).
$$i_t = r^{\text{episodic}}_t \cdot \min\!\big\{\max\{\alpha_t,1\},\,L\big\},\qquad L=5.$$

**Bonus module.**
- *Controllable-state embedding $f$* — conv encoder trained by **inverse dynamics** (predict $a_t$
  from $f(x_t),f(x_{t+1})$), so $f$ ignores uncontrollable variation. (Reused from step 2.)
- *Episodic memory* — per-environment slot memory of $f(x)$, **emptied each episode**. The episodic
  bonus is a count bonus $1/(\sqrt{\sum_{N_k}K}+c)$ with the count approximated by a kernel sum over
  the $k$ nearest neighbors; inverse kernel $K=\epsilon/(d^2/d_m^2+\epsilon)$ with running $k$-th-NN
  distance scale $d_m^2$; cluster floor $\xi$ (near-duplicate frames count as one state); saturation
  cap $s_m$ (already-saturated this episode → zero bonus).
- *Lifelong modulator* — RND distillation error normalized to $\alpha_t=1+(\mathrm{err}-\mu_e)/\sigma_e$,
  clipped into $[1,L]$. (Reused from step 3.)
- *Combine* — multiply; floor at 1 means the lifelong factor never kills the episodic drive (never
  give up), cap at $L$ means a spike can't explode it; $i_t\to r^{\text{episodic}}_t$ as everything
  is mastered.
- *Agent separation (UVFA)* — because this bonus does **not** vanish, a family $Q(x,a,\beta_i)$ over
  $r^{\beta_i}=e+\beta_i i$ with $\beta_0=0$ (clean exploit head, greedily retrievable) and
  $\beta_{N-1}=\beta$ (max explore), each with its own discount, keeps dense-reward games unharmed.

**What it tests.** Whether adding the missing *within-episode* timescale turns RND's lucky-seed
Private Eye jackpot into a *reliable* result — rescuing the dead seed and lifting the negative `auc`
— without giving up Tutankham/Frostbite.

Full derivation (two-timescale argument, kNN pseudo-count + kernel + cluster/cap, multiplicative
floor, UVFA family and discount schedule): `methods/ngu/`.

```python
class IntrinsicBonusModule(nn.Module):
    @torch.no_grad()
    def compute_bonus(self, obs, next_obs, actions):
        emb = self.encoder(last_frame(next_obs).float())               # controllable state f(x_t)
        r_epi = torch.tensor([self.memory.episodic_reward(i, emb[i])    # within-episode kNN count
                              for i in range(emb.shape[0])],
                             device=self.device, dtype=torch.float32)
        n = self._normalize_obs(next_obs)
        err = (self.predictor(n) - self.target(n)).pow(2).sum(1)        # RND lifelong error
        self.err_rms.update(err.cpu().numpy())
        alpha = (1.0 + (err - float(self.err_rms.mean)) /
                 float(np.sqrt(self.err_rms.var + 1e-8))).clamp(1.0, L)
        return (r_epi * alpha).detach()                                 # episodic * clip(alpha,1,L)

    def loss(self, batch_obs, batch_next_obs, batch_actions):
        f_t = self.encoder(last_frame(batch_obs).float())
        f_tp1 = self.encoder(last_frame(batch_next_obs).float())
        inv = F.cross_entropy(self.inverse_model(torch.cat([f_t, f_tp1], 1)), batch_actions.long())
        n = self._normalize_obs(batch_next_obs)
        rnd = F.mse_loss(self.predictor(n), self.target(n).detach(), reduction="none").mean(-1)
        mask = (torch.rand(len(rnd), device=self.device) < self.args.update_proportion).float()
        return inv + (rnd * mask).sum() / torch.clamp(mask.sum(), min=1.0)

def mix_advantages(ext_advantages, int_advantages, args):
    return args.ext_coef * ext_advantages + args.int_coef * int_advantages
```
