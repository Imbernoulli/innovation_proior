The Hilbert rep did what I asked and stopped exactly where I predicted. Replacing raw goal axes with axes where Euclidean distance is reachability lifted success clearly above the identity floor on antmaze-large and cube-single, while on pointmaze-large it barely moved — consistent with raw position already being a serviceable code there. But the residual weakness I flagged showed up precisely where I said it would: the symmetric metric leaves the most on the table on the manipulation environment and trails on antmaze, tying only on pointmaze. That is structural, not a tuning artifact, and naming the structure drives this step. The Hilbert value is $V(s,g) = -\lVert \phi(s) - \phi(g) \rVert$, and a Euclidean norm is *rigidly symmetric*: $\lVert \phi(s) - \phi(g) \rVert = \lVert \phi(g) - \phi(s) \rVert$. So the code is forced to assume reaching $g$ from $s$ costs the same as reaching $s$ from $g$. In reversible navigation that is roughly true, which is why pointmaze and antmaze were not punished too hard; in manipulation it is false — pushing a cube off a ledge is easy, lifting it back is hard, many transitions are one-way. There is a second, subtler limitation that sharpens the fix: even setting asymmetry aside, a shared-embedding metric $\lVert \phi(s) - \phi(g) \rVert$ is *not a universal approximator* of two-argument functions; not every pairwise function can be written as a Euclidean distance between codes from one shared map. The reaching value $d^*(s,g)$ is in general neither symmetric nor metric-shaped, so to carry the full reachability relation I need an aggregator that is directed — different maps for state and goal — and expressive enough to approximate an arbitrary continuous two-argument value.

I build the right object up from the reaching problem rather than guessing an architecture. With the OGBench shifted indicator reward $r(s,g) = \mathbf{1}(s = g) - 1$ and an absorbing goal, the optimal value is a monotone transform of the optimal reaching time, $V^*(s, g) = -(1 - \gamma^{d^*(s,g)})/(1 - \gamma)$; either way the value *is* the reaching structure in log-discount coordinates. Now ask what a goal really is to a controller — not just "far from here," but the thing every state has some reaching relation to. So the ideal representation of $g$ is the whole function of incoming relations,
$$\phi^\vee(g) = \big(s \mapsto d^*(s, g)\big),$$
identifying a goal by *all* its temporal distances from the state space — the familiar slogan that an object is determined by its relations to every other object (the Riesz / kernel-section / Yoneda viewpoint). The slogan only earns its keep if the code satisfies the control requirements: sufficiency (it keeps everything an optimal greedy step needs) and noise invariance (it drops exogenous detail). Sufficiency: if I am handed only $f = \phi^\vee(g)$ so that for any successor $s'$ I can query $f(s') = d^*(s', g)$, then the one-step greedy action needs only successor values, $\arg\max_a \mathbb{E}_{s'}[\gamma^{f(s')}] = \arg\max_a \mathbb{E}_{s'}[V^*(s', g)] = \arg\max_a Q^*(s, a, g)$ — the functional throws away nothing the optimal policy needs and never touches the raw goal. Noise invariance: in a block-structured observation model where each observation maps to a latent and the reward is latent, two goal observations with the same latent induce identical latent rewards on every trajectory, so their optimal values agree at every state and $\phi^\vee(g_1) = \phi^\vee(g_2)$ — exogenous observation noise that does not change the latent task disappears. That is the floor's failure (joint angles, scene texture riding into the value) finally addressed at the level of the *target*, not patched after the fact.

I cannot store an arbitrary function per goal and I do not know $d^*$, but the sufficiency argument tells me exactly how to finitize: the functional is always *paired with a state and evaluated*, $f(s') = d^*(s', g)$. So model the two-argument value surface through two embeddings, $V(s, g) \approx f(\psi(s), \phi(g))$, and export the goal-side vector as the finite code. The only open choice is the aggregator $f$. The Hilbert rung chose the metric $f = -\lVert \psi - \phi \rVert$ with $\psi = \phi$, which I just diagnosed as too rigid: symmetric and non-universal. The simplest aggregator that fixes both problems is the **Dual bilinear (inner-product) representation**:
$$V(s, g) = \psi(s)^\top \phi(g) / \sqrt{d}.$$
Two things fall out immediately. It is *directed*: $\psi$ and $\phi$ are different maps, so $\psi(s)^\top \phi(g)$ need not equal $\psi(g)^\top \phi(s)$, and the code can represent the asymmetric reaching costs the metric could not — directly attacking the manipulation gap. And it is *universal*: with learned feature maps of sufficient width, sums of separable products approximate any continuous two-variable function on a compact domain, so the bilinear value can represent reaching-value surfaces the metric provably cannot. The $/\sqrt{d}$ scaling is load-bearing: without it the dot product is a sum of $d$ terms whose scale grows with the representation width, so changing `rep_dim` would silently change the initial value and gradient magnitude; dividing by $\sqrt{d}$ keeps the score order-one across widths — the same square-root scaling attention uses, and what makes training stable at $\texttt{rep\_dim} = 256$.

The offline learning keeps the *exact same recipe* as the Hilbert rung — only the value parameterization changes from $-\lVert \psi - \phi \rVert$ to $\psi^\top \phi / \sqrt{d}$. The reasoning is identical: the optimal Bellman backup's $\max$ over actions is unsafe offline because it queries out-of-distribution actions, so I use expectile regression for an in-sample max, and the harness contract is the same — the fixed GCIVL value loss does not backprop into the embeddings, so `compute_rep_loss` runs a self-contained loop that shapes $\phi$/$\psi$ using the private twin `rep_critic` and its EMA `target_rep_critic` maintained by the loop. The two-part loss carries over verbatim in structure. The **representation value** is now the bilinear score $v = (\phi(s) \cdot \psi(g)).\text{sum}(-1) / \sqrt{d}$ — here the harness names the state branch `phi` and the goal branch `psi`, with the value computing $\phi(\text{obs}) \cdot \psi(\text{goal}) / \sqrt{\texttt{rep\_dim}}$ summed over the last axis — fit by expectile regression toward the target critic: $\mathrm{adv} = q_t - v$, $q_t = \min(q_{1,t}, q_{2,t})$ from `target_rep_critic`, loss $\lvert \kappa - \mathbf{1}(\mathrm{adv} < 0)\rvert\,\mathrm{adv}^2$ with $\kappa = 0.7$. The **critic** is the ordinary TD fit, $\mathrm{td} = r + \gamma\,\mathrm{mask}\cdot v(\text{next}\_s, g)$ under stop-gradient, both online heads regressed to it with squared error. The total `rep_loss` is the sum, and `encode_goal` returns the goal branch $\psi(g)$ averaged over the ensemble. The interlock is the same as before — the critic learns a bootstrapped reaching value from data, the expectile loss drags the bilinear value toward an in-support max of that estimate, and because the value is now $\psi^\top \phi / \sqrt{d}$, that drag injects the *directed* reaching structure into the two embeddings. The discount story is unchanged ($\gamma$ for TD stability, converging to a discounted approximation), and I no longer need the square-root floor the metric required — there is no $\sqrt{\lVert\cdot\rVert}$ singularity in an inner product, so the bilinear value is numerically smoother at init, one fewer fragile constant than the Hilbert rung.

One design point I defend explicitly, because it is what makes this a *representation* method and not just a different value head: I keep the bilinear representation value strictly separate from the downstream control value, exactly as the harness forces. The bilinear structure is what makes the goal code relational and compact, but a constrained inner-product head can be *too* constrained to extract a good policy from — in antmaze the goal code can mostly care about the x-y target, while the controller still needs joint angles and velocities to choose actions. So the bilinear value learns $\phi(g)$, and a separate monolithic GCIVL value/actor consumes it to do control. Two value functions are not redundant: one shapes the representation, the other extracts control. That is precisely why the edit surface gives the module its own private `rep_critic` rather than reusing the agent's value, and why this rung does not touch anything downstream.

The falsifiable prediction against the Hilbert numbers follows from the two things the bilinear change buys — direction and universality — both paying off most where the symmetric metric hurt most. On **cube-single**, where I attributed the Hilbert shortfall to irreversible, asymmetric reaching a symmetric norm cannot encode, the directed inner product should be the largest improvement on the ladder — the rung's strongest claim, and if cube does *not* improve over Hilbert, the asymmetry diagnosis is wrong. On **antmaze-large**, where reaching is mostly reversible but the value surface is still richer than a shared-metric embedding can represent, universality should give a clear but smaller gain. On **pointmaze-large**, where Hilbert already essentially matched the floor and reaching is benign, I expect bilinear and Hilbert close, within noise. So: bilinear $>$ Hilbert most on cube, clearly on antmaze, roughly tied on pointmaze, with the bilinear average above the Hilbert average — the whole improvement traceable to having dropped the symmetric-metric assumption for a directed, universal aggregator.

```python
# EDITABLE region of dual-goal-representations/custom_train.py — step 3: bilinear (inner-product) dual rep
class GoalRepresentation(nn.Module):
    """Bilinear dual goal representation.

    Separate state (phi) and goal (psi) embeddings; V(s,g) = phi(s)^T psi(g) / sqrt(d).
    Goal code = psi(g) averaged across the ensemble. Trained by IQL expectile
    regression with a separate MLP critic.
    """

    obs_dim: int
    rep_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True
    rep_expectile: float = 0.7
    discount: float = 0.99

    def setup(self):
        mlp_module = ensemblize(MLP, 2)
        self.phi = mlp_module((*self.hidden_dims, self.rep_dim),
                              activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.rep_dim),
                              activate_final=False, layer_norm=self.layer_norm)
        critic_module = ensemblize(MLP, 2)
        self.rep_critic = critic_module((*self.hidden_dims, 1),
                                        activate_final=False, layer_norm=self.layer_norm)
        self.target_rep_critic = critic_module((*self.hidden_dims, 1),
                                               activate_final=False, layer_norm=self.layer_norm)

    def _bilinear_value(self, observations, goals):
        phi_s = self.phi(observations)
        psi_g = self.psi(goals)
        v = (phi_s * psi_g / jnp.sqrt(self.rep_dim)).sum(axis=-1)
        return v

    def encode_goal(self, goals):
        psi_g = self.psi(goals)            # (2, batch, rep_dim)
        return psi_g.mean(axis=0)          # (batch, rep_dim)

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        critic_input_obs = jnp.concatenate([observations, goals], axis=-1)
        if actions is not None:
            critic_input_obs = jnp.concatenate([critic_input_obs, actions], axis=-1)
        q1_t, q2_t = self.target_rep_critic(critic_input_obs)
        q1_t = q1_t.squeeze(-1)
        q2_t = q2_t.squeeze(-1)
        q_t = jax.lax.stop_gradient(jnp.minimum(q1_t, q2_t))

        v = self._bilinear_value(observations, goals)
        v_mean = v.mean(axis=0)

        adv = q_t - v_mean
        weight = jnp.where(adv >= 0, self.rep_expectile, (1 - self.rep_expectile))
        value_loss = (weight * (adv ** 2)).mean()

        next_v = self._bilinear_value(next_observations, goals)
        next_v_mean = next_v.mean(axis=0)
        td_target = jax.lax.stop_gradient(rewards + self.discount * masks * next_v_mean)

        critic_input_obs_cur = jnp.concatenate([observations, goals], axis=-1)
        if actions is not None:
            critic_input_obs_cur = jnp.concatenate([critic_input_obs_cur, actions], axis=-1)
        q1, q2 = self.rep_critic(critic_input_obs_cur)
        q1 = q1.squeeze(-1)
        q2 = q2.squeeze(-1)
        critic_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()

        total_loss = value_loss + critic_loss
        info = {
            'rep_value_loss': value_loss,
            'rep_critic_loss': critic_loss,
            'rep_v_mean': v_mean.mean(),
        }
        return total_loss, info

    def __call__(self, goals, observations=None, next_observations=None,
                 rewards=None, masks=None, actions=None, mode='encode'):
        if mode == 'rep_loss':
            return self.compute_rep_loss(
                observations, goals, next_observations, rewards, masks, actions)
        return self.encode_goal(goals)
```
