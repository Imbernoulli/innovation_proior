I start from the fact that my usual continuous-control agent already has a workable algorithm. SAC,
DDPG, and TD3-style updates can produce competent agents with small MLPs, so if scaling fails when I
make those MLPs bigger, I should not immediately add a new loss or a new replay trick. The first
question is whether the network class is the wrong object. In supervised learning, overparameterized
models do not generalize merely because they are large; they generalize because their architecture
and optimizer make simple functions easy to find. A bare RL MLP has much less of that bias. It sees
non-stationary data, learns from a bootstrapped target, and then the policy exploits whatever value
surface the critic currently represents. Extra capacity in that setting is dangerous unless the
architecture makes a simple solution the default.

The input is the first place where a plain MLP wastes capacity. A state vector is not a normalized
image tensor. One coordinate can be a joint angle, another a velocity, another a contact signal, and
their distribution changes as the policy visits new states. If I let the network see raw values, a
large critic can attach importance to scale rather than meaning. I want every coordinate to arrive
approximately centered and scaled, but I cannot use fixed dataset statistics because the data stream
is created online. So I maintain running per-coordinate statistics. For a single new observation
o_t, with delta_t = o_t - mu_{t-1}, the update is
mu_t = mu_{t-1} + delta_t / t and
sigma_t^2 = ((t - 1) / t) * (sigma_{t-1}^2 + delta_t^2 / t). Then I feed
(o_t - mu_t) / sqrt(sigma_t^2 + epsilon). In code I can implement the same idea by the standard
parallel running-mean/variance update over batches. The important point is the axis: the statistics
are per observation coordinate across samples, not per sample across features. That is why this is
not RMSNorm.

Now I need the body of the network to make an identity-like map easy. A deep plain MLP has to push
information through every nonlinearity, so even a nearly linear value function is represented by a
chain of transformations that must cooperate. A residual block changes the default. If the block is
x + F(x), then F can be small and the block behaves like the identity. The nonlinear part becomes an
additive correction rather than the only path through the layer. That is exactly the kind of
simplicity bias I need: the large network can express a complicated correction, but it is not forced
to use one.

I still have to decide where normalization belongs. If I normalize after the residual addition at
every block, I interfere with the identity path. If I normalize before the nonlinear branch, the
branch sees controlled features while the skip carries the residual stream directly. So each block
should compute x_{l+1} = x_l + MLP(LayerNorm(x_l)). The MLP is the standard inverted bottleneck:
Linear(d_h, 4 d_h), ReLU, Linear(4 d_h, d_h). The factor four matters as a constant because it is
where most of the scalable parameters live, and it matches the implementation. The linears in this
branch use He-normal initialization, the ReLU-compatible fan-in scaling, with zero biases in the
PyTorch equivalent and zero biases by default in the Flax reference.

At first this seems to be enough normalization, but the stream after L residual additions is still a
sum of the embedded input plus all learned corrections. The head should not have to chase the scale
of that sum as I increase depth or width. So after the last residual block I apply one more
LayerNorm before the policy or value head. This is the post-layer normalization in the full network:
not a replacement for the pre-LN inside the branch, but a final scale control before prediction.
The resulting encoder is therefore running-stat observation normalization outside the network,
linear embedding with orthogonal gain 1, L residual blocks of pre-LN feedforward corrections, and a
final LayerNorm before the head.

I need to be careful about where the observation normalization lives in code. The formal description
treats it as an observation-normalization layer, but the reference implementation wraps the agent.
The wrapper updates running statistics when actions are sampled during training, then normalizes observations
both for acting and for replay batches. The actor receives normalized observations. The critic
receives normalized observations concatenated with the raw action. If I put another RSNorm inside
the actor and critic modules, I would double-normalize in that implementation. So the faithful
software boundary is a wrapper for the observation stream plus a network encoder for the residual
body.

The policy head also matters. I might be tempted to present only a deterministic tanh actor because
the architecture can be used with DDPG, but the main off-policy reference is SAC. There the actor is
a squashed Gaussian: the encoder feeds one linear head for the mean and one for log standard
deviation, the raw log standard deviation is squashed by tanh into [-10, 2], and the sampled normal
action is passed through tanh. The critic is simpler: concatenate observation and action, encode,
then use a linear scalar Q head. When clipped double Q is enabled, there are two independently
parameterized critics and the SAC target or actor loss uses the minimum. In the released config,
that clipped double critic is enabled for episodic HumanoidBench settings and disabled otherwise.

The constants now line up. For SAC with this architecture, the actor uses one residual block at
hidden width 128. The critic uses two residual blocks at hidden width 512. The optimizer is AdamW
with learning rate 1e-4 and weight decay 1e-2 for actor and critic. The target critic momentum is
tau = 5e-3. The SAC temperature starts at 1e-2, has learning rate 1e-4, and the code sets the
entropy target coefficient to -0.5 times the action dimension, so I keep the sign explicit rather
than only tracking the entropy magnitude. Replay ratio is 2. DDPG uses the same actor and critic
architecture constants, the same learning rate and weight decay, and Gaussian exploration noise
with standard deviation 0.1.

I also want a way to say why the design should work without turning the performance plots into the
reasoning. The diagnostic measures functional simplicity at initialization by sampling random network
parameters, evaluating the random function on a grid, decomposing it in frequency space, and using a
frequency-weighted average of Fourier coefficients as complexity c(f). The simplicity score is
approximately E_theta[1 / c(f_theta)]. That score is not a training objective and it is not a proof
that every larger model improves. It is a diagnostic for the architectural bias. The useful claim is
more modest and stronger: when the same RL algorithm is run with this normalized residual encoder,
scaling the critic becomes beneficial over the tested range where the plain MLP degrades, and the
component ablations show that removing any of the three ingredients hurts performance.

There is one subtle correction to make in my own story. I should not say the residual block alone
keeps feature magnitudes bounded. The residual block preserves an identity path and computes its
nonlinear correction from normalized features. The final LayerNorm is what standardizes the stream
before prediction. Running-stat normalization is a separate input-side control. If I collapse these
roles together, I lose the actual design: input statistics handle drifting coordinates, pre-LN
residual feedforward blocks make nonlinear corrections optional, and post-LN controls the head
input. The method is the combination, not any one part in isolation.

So the final construction is simple. Normalize observations with running per-coordinate statistics.
Embed to the hidden width with an orthogonal linear map. Repeat a residual block whose branch is
LayerNorm, Linear to 4x width, ReLU, Linear back down, and add the skip. Apply one final LayerNorm.
Attach the existing actor or critic head and leave the RL update intact. The critic can be made
wider and deeper than the actor because it carries the harder bootstrapped regression problem. The
architecture earns its name by making the easy, simple function path cheap while leaving enough
capacity for the corrections that the control task actually needs.
