The identity floor told me exactly what I expected: raw goals are roughly competitive only where the raw coordinate is already close to the quantity that matters — the point's position in pointmaze-large — and they leave the most on the table precisely where the reachability geometry diverges most from raw-coordinate proximity, with cube-single weakest and antmaze-large in between, its goal-relevant x-y target buried under joint angles and velocities the policy must ignore. That is not a learning-capacity failure; the GCIVL agent is competent and is being handed the wrong axes. So the fix is to stop feeding raw goals and learn a map $\phi(g)$ whose *geometry already encodes reachability*, slot it into the same `encode_goal` hook so the value and actor see $\phi(g)$ everywhere they used to see $g$, and turn the floor's empty `compute_rep_loss` into a real auxiliary objective that trains $\phi$. Everything downstream stays fixed.

The hinge is deciding what "close in representation space" should mean, and the only notion that matters for goal reaching is temporal: how many steps an optimal policy needs to get from one state to another. Write $d^*(s, g)$ for that optimal temporal distance. I want a $\phi$ such that ordinary Euclidean distance between $\phi(s)$ and $\phi(g)$ *is* that temporal distance — but I have no $d^*$ labels, only trajectories. A classical identity rescues this. Set up the goal-reaching reward as $r(s,g) = -1$ unless $s = g$, terminating at the goal; then the return is the negative count of steps to reach $g$, so $V^*(s, g) = -d^*(s, g)$. "Distance to goal" and "goal-conditioned value function" are the same object — and value functions I *can* learn offline. So rather than regress $d^*$ directly, I learn a goal-conditioned value and *force its functional form to be a distance between two embeddings*. This is the **Dual Hilbert (symmetric metric) representation**:
$$V(s, g) = -\lVert \phi(s) - \phi(g) \rVert, \qquad \phi: \text{obs} \to \mathbb{R}^D \ (\ell_2, \text{ a Hilbert space}).$$
Train $V$ to satisfy the goal-conditioned Bellman equations and, at convergence, $\lVert \phi(s) - \phi(g) \rVert \approx -V(s, g) \approx d^*(s, g)$ — the isometry I wanted, with the value-learning machinery injecting $d^*$ into $\phi$ and no distance label ever needed.

The parameterization buys structure before any training. It is symmetric in its two arguments, so a single shared encoder $\phi$ serves both states and goals: I encode a state and a goal with the same map and take the distance between the codes, which is natural because a distance takes two points from one space. It gives $V(s,s) = 0$ for free (being at the goal costs nothing) and $V \le 0$ always (temporal distance is nonnegative). The goal code handed downstream is simply $\phi(g)$, and this is what answers the floor's failure: where the identity dumped joint angles and texture straight into the value, $\phi$ is trained so only the reachability-relevant content survives into the distance.

I have to confront the symmetry I just praised, because temporal distance is *not* symmetric — climbing a cliff costs more than descending it, a one-way passage is one step one way and infinity the other, so $d^*(s,g) \ne d^*(g,s)$ in general; $d^*$ is a quasimetric. But I am not going to drive the policy with this possibly-wrong parameterized $V$. I want $\phi$ only as a *representation*, a coordinate system handed to the downstream GCIVL agent, which keeps its own separate value and actor. So the question is not "is the symmetric metric exactly $d^*$" but "is the best symmetric approximation of $d^*$ a good enough coordinate system." There is even a deeper obstruction: not every metric, even a symmetric one, embeds isometrically into a Euclidean space. So I reframe the target as the best *approximate* symmetric Hilbert embedding of the MDP's temporal structure — honest, and exactly what the value objective optimizes. This matters for the environments at hand: antmaze and pointmaze navigation are largely reversible, so a symmetric metric should fit them well; cube manipulation has irreversible structure where the symmetry assumption starts to bite. I insist on $\ell_2$ specifically because it is induced by an inner product, so the space is a Hilbert space carrying directions, projections, and midpoints — $\ell_1$ and $\ell_\infty$ give metric spaces but no consistent inner product. Since fixing the space to $\mathbb{R}^D$ with $\ell_2$ costs nothing in the objective, I bank the stronger geometry: a direction $\phi(g) - \phi(s)$ becomes an actionable control signal for later.

The real work is training $V(s,g) = -\lVert \phi(s) - \phi(g) \rVert$ toward $V^* = -d^*$ from a fixed offline dataset of suboptimal trajectories. The optimal Bellman backup is a $\max$ over reachable next states, and offline I cannot take that $\max$ without evaluating transitions absent from the data — querying out-of-distribution actions is the classic route to catastrophic overestimation. The tool for an in-sample max is expectile regression, the move behind IQL: the expectile loss $L_\kappa(u) = \lvert \kappa - \mathbf{1}(u < 0) \rvert\, u^2$ with $\kappa > 1/2$ weights positive residuals more than negative, so fitting a scalar to a distribution of targets pulls the fit toward the top of the support. Regress $V(s,g)$ toward TD targets over the dataset's own transitions with an upper expectile and $V$ is pulled toward the best achievable backup among transitions that actually occur — an optimal-style max that never invents a transition. The behavior policy can be terrible on average; as long as the dataset occasionally contains the good next step out of $s$, the upper expectile latches onto it.

This task's harness shapes how I realize that. It does not give me a single distance-parameterized value head to read $d^*$ off directly: the fixed agent owns its own downstream IVL value, and that loss is computed so it does *not* backprop into $\phi$. Only the module's `rep_loss` trains the representation. So inside `compute_rep_loss` I run a self-contained value-learning loop whose only purpose is to shape $\phi$, using the private twin MLP critic `rep_critic` and its EMA target copy `target_rep_critic` that the loop maintains. The loss is two interlocking parts. The **representation value** is the Hilbert distance itself, $v = -\sqrt{\max(\lVert \phi_s - \phi_g \rVert^2, \varepsilon)}$, fit by expectile regression toward the target critic: the advantage is $\mathrm{adv} = q_t - v$ with $q_t = \min(q_{1,t}, q_{2,t})$ from the frozen `target_rep_critic`, and the loss is $\lvert \kappa - \mathbf{1}(\mathrm{adv} < 0)\rvert\,\mathrm{adv}^2$ with $\kappa = \texttt{rep\_expectile}$. This is the in-sample-max pull on the distance value, keyed by the target critic so it is stable under bootstrapping. The **critic** is an ordinary TD fit: the target is the Hilbert value at the next state, $\mathrm{td} = r + \gamma\,\mathrm{mask}\cdot v(\text{next}\_s, g)$ under stop-gradient, and both online critic heads regress to it with squared error. The total `rep_loss` is the sum; `encode_goal` returns the mean of the shared $\phi$ ensemble. The two pieces interlock — the critic learns a bootstrapped reaching value from data, the expectile loss drags the *distance* parameterization up toward an in-support max of that estimate, and because the distance is $-\lVert \phi_s - \phi_g \rVert$, that drag is exactly what injects temporal structure into $\phi$. Keeping the representation value strictly separate from the downstream control value is the whole reason the harness exposes a private `rep_critic`: the distance head shapes the code, the GCIVL value extracts the policy.

Two load-bearing details. The discount: $d^*$ is undiscounted (a raw step count), yet I carry $\gamma$ into the backup, because the undiscounted goal value grows with horizon and is numerically nasty to fit with bootstrapped TD; $\gamma$ keeps the backup contractive and the magnitudes bounded, at the cost that I converge to a *discounted* approximation. Combined with the symmetry approximation and the embeddability obstruction, the honest description of the fixed point is "the best discounted symmetric Hilbert approximation of the MDP's temporal distances." The square-root floor is not optional: $\lVert \phi_s - \phi_g \rVert = \sqrt{\sum (\phi_s - \phi_g)^2}$ has derivative $1/(2\sqrt{\cdot})$, which blows up as the squared distance goes to zero — at init, or for $s$ near $g$ — so I compute $\sqrt{\max(\text{squared\_dist}, 10^{-6})}$. That bounds the gradient near zero and removes the NaN while being negligible where the distance is healthy; a small constant, but the difference between training and diverging on the first batch. I use the harness $\kappa = 0.7$, an upper expectile aggressive enough to chase the good in-support backup without becoming brittle and chasing the single luckiest transition.

The falsifiable claim against the identity floor follows from the mechanism — replace raw goal axes with axes where distance is reachability — and should help most where the floor hurt most: on antmaze-large the Hilbert code should collapse the nuisance pose and lift success clearly above identity; on cube-single it should help too but leave the most on the table, because manipulation has irreversible transitions the symmetric $\lVert\cdot\rVert$ cannot represent; on pointmaze-large, where raw position was already a decent code, I expect it roughly level with the floor. That residual weakness — relative to a *directed*, non-symmetric aggregator — should show up precisely on the manipulation environment where reachability is most asymmetric, and that gap is what the next step attacks by dropping the symmetric metric for an inner-product value.

```python
# EDITABLE region of dual-goal-representations/custom_train.py — step 2: Hilbert (symmetric) dual rep
class GoalRepresentation(nn.Module):
    """Hilbert (symmetric) dual goal representation.

    Shared phi for states and goals; V(s,g) = -||phi(s) - phi(g)||.
    Goal code = phi(g) averaged across the ensemble. Trained by IQL
    expectile regression with a separate MLP critic.
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
        critic_module = ensemblize(MLP, 2)
        self.rep_critic = critic_module((*self.hidden_dims, 1),
                                        activate_final=False, layer_norm=self.layer_norm)
        self.target_rep_critic = critic_module((*self.hidden_dims, 1),
                                               activate_final=False, layer_norm=self.layer_norm)

    def _hilbert_value(self, observations, goals):
        phi_s = self.phi(observations)
        phi_g = self.phi(goals)
        squared_dist = jnp.square(phi_s - phi_g).sum(axis=-1)
        v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))
        return v

    def encode_goal(self, goals):
        phi_g = self.phi(goals)            # (2, batch, rep_dim)
        return phi_g.mean(axis=0)          # (batch, rep_dim)

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        critic_input_obs = jnp.concatenate([observations, goals], axis=-1)
        if actions is not None:
            critic_input_obs = jnp.concatenate([critic_input_obs, actions], axis=-1)
        q1_t, q2_t = self.target_rep_critic(critic_input_obs)
        q1_t = q1_t.squeeze(-1)
        q2_t = q2_t.squeeze(-1)
        q_t = jax.lax.stop_gradient(jnp.minimum(q1_t, q2_t))

        v = self._hilbert_value(observations, goals)
        v_mean = v.mean(axis=0)

        adv = q_t - v_mean
        weight = jnp.where(adv >= 0, self.rep_expectile, (1 - self.rep_expectile))
        value_loss = (weight * (adv ** 2)).mean()

        next_v = self._hilbert_value(next_observations, goals)
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
