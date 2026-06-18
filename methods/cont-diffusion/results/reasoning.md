Let me start from what actually goes wrong, because the fast version of this head already works and I need to be honest about why it is not enough. I have a parallel-decoding trunk: it ingests vision, language, and proprioception, and at the action slots it hands me a block of final-layer hidden states, one bundle per future timestep in the chunk. The cheapest thing I can do, and the thing I would reach for first, is run each bundle through a small MLP, read off a 7-D continuous action, and train it to match the logged demonstration with a mean absolute error. One trunk pass, a negligible head, full continuous precision, fast convergence. So what is wrong with it? Stare at the objective. Minimizing E|a - a_pred| over the data picks, for each observation, the conditional median of the logged actions; minimizing E(a - a_pred)^2 would pick the conditional mean. Both are a single number per observation. That is fine if, given what the robot sees, there is essentially one right action. But demonstrations are not always like that. At a fork an expert sometimes goes left and sometimes right; sometimes regrasps now, sometimes later; the logged actions at nearly identical observations can form a multimodal distribution. A point estimator handed a bimodal target has to choose one compromise action, and the compromise between "swerve left" and "swerve right" may be "drive straight into the obstacle." This is not a tuning problem I can anneal away. The head outputs one vector; the world wants a distribution. So the real task is not just "predict the action"; it is "represent p(actions | observation) richly enough to be multimodal," and then sample one action from it.

So I need a head that models a whole conditional distribution over the action chunk, not a point. I could put a mixture density network on top and predict Gaussian-mixture weights, means, and variances, but then I have to pick the number of components up front, keep the components alive, and hope a fixed small mixture is enough for the geometry of valid action chunks. I could go energy-based and learn E(o, a), which is expressive and handles multimodality, but then sampling means inner-loop optimization or MCMC, and training needs negative samples. I want something that still trains with a regression-flavored loss, with no adversary, partition function, or negative sampler, while producing samples from a multimodal continuous distribution.

The construction that fits is to deliberately corrupt a clean action chunk x_0 with Gaussian noise, then learn to reverse that corruption. The forward step is

  q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) x_{t-1}, beta_t I).

After enough small steps the data approaches pure noise. If I learn the reverse direction, I can start from x_T ~ N(0,I) and denoise down to a clean action chunk. Because generation starts from fresh noise, different seeds can flow toward different valid action modes.

I do not want to iterate the noising chain during training, so I need the closed form. Write alpha_t = 1 - beta_t and bar_alpha_t = prod_{s=1}^t alpha_s. Each step scales the previous state by sqrt(alpha_t) and injects variance beta_t. For two steps, the independent noise variances become (1 - alpha_t) and alpha_t(1 - alpha_{t-1}), so their sum is

  (1 - alpha_t) + alpha_t(1 - alpha_{t-1}) = 1 - alpha_t alpha_{t-1}.

Inducting gives the useful marginal

  x_t = sqrt(bar_alpha_t) x_0 + sqrt(1 - bar_alpha_t) eps,    eps ~ N(0, I).

That lets me sample any noise level in one shot. Now I need to decide what the reverse model predicts. The true posterior q(x_{t-1} | x_t, x_0) is tractable because the process is linear-Gaussian, and its mean is a combination of x_t and x_0. But x_0 is unavailable at sampling time. The forward marginal gives

  x_0 = (x_t - sqrt(1 - bar_alpha_t) eps) / sqrt(bar_alpha_t),

so the only missing quantity is the noise eps that was added. Predicting eps with eps_theta(x_t, t) gives the DDPM reverse mean

  mu_theta(x_t,t) = (1/sqrt(alpha_t)) (x_t - ((1 - alpha_t)/sqrt(1 - bar_alpha_t)) eps_theta(x_t,t)).

The same reparameterization turns the simplified practical training objective into plain noise-prediction MSE:

  L = E || eps - eps_theta(sqrt(bar_alpha_t) x_0 + sqrt(1 - bar_alpha_t) eps, t) ||^2.

This is the loss shape I want: sample a clean chunk, sample noise, sample a timestep, form x_t in one shot, predict the noise, and take MSE. The important ablation is that directly predicting the posterior mean is competitive only with the weighted variational objective, while epsilon prediction works much better with the simplified unweighted objective. So I use epsilon prediction.

Sampling can start from the ancestral DDPM reverse step,

  x_{t-1} = (1/sqrt(alpha_t)) (x_t - ((1 - alpha_t)/sqrt(1 - bar_alpha_t)) eps_theta(x_t,t)) + sigma_t z.

In its plain form this costs one model pass per training timestep. For a 7B trunk, that is a serious latency cost. DDIM gives the handle I need: the training objective only fixes the marginals q(x_t | x_0), not the exact Markovian joint, so I can use a non-Markovian process with the same objective and choose a shorter sampling trajectory. First I estimate the clean chunk,

  xhat_0 = (x_t - sqrt(1 - bar_alpha_t) eps_theta(x_t,t)) / sqrt(bar_alpha_t),

then, for the next lower selected timestep s, I jump by

  x_s = sqrt(bar_alpha_s) xhat_0 + sqrt(1 - bar_alpha_s - sigma_t^2) eps_theta(x_t,t) + sigma_t z.

With sigma_t = 0 the step is deterministic after the starting noise is fixed, and I can run the same trained denoiser on a subsequence of timesteps. In this scaffold I still use the fixed 50-step setting, but keeping the DDIM form makes the latency-quality tradeoff explicit.

The noise schedule also matters. A linear beta ramp can push bar_alpha_t close to zero too early, leaving late steps with little signal to denoise. The cosine schedule defines bar_alpha_t directly:

  bar_alpha_t = f(t)/f(0),   f(t) = cos^2(((t/T + s)/(1 + s)) * pi/2).

This changes slowly near the endpoints and more steadily in the middle. Improved DDPM motivates it that way, Diffusion Policy reports the square-cosine schedule as best for its control tasks, and the code path uses Diffusers' squaredcos_cap_v2 schedule. So I choose that schedule.

Now I need to attach the denoiser to this VLA scaffold. A from-scratch diffusion policy can keep a separate noise-prediction network and condition it with FiLM or cross-attention. Here the observation is already fused inside the 7B trunk, and the only handle I have is the action-slot hidden states. The trunk itself should be the conditioning network. Therefore I need to feed the current noisy action chunk and diffusion timestep into the trunk, let bidirectional attention mix them with vision, language, and proprioception, and read the noise prediction from the resulting action-slot hidden states.

The noisy chunk has shape (B, K, 7), and the action layout already uses one token slot per action dimension. I therefore flatten the chunk to K*7 scalars, add a singleton feature dimension, and project each scalar from R^1 to the LLM hidden size. This is why the noisy-action projector is per scalar rather than per 7-D action: it preserves the one-slot-per-action-dimension layout already expected by the trunk. The timestep becomes one sinusoidal embedding token so the trunk knows the current noise level.

One training forward is now straightforward. I sample t and eps, form x_t, project the K*7 noisy scalars into action-token embeddings, encode t, and call the trunk with both. At train time I extract the action slots with the current-action and next-actions masks; at eval time I take the contiguous K*7 span after the prompt. Then I reshape (B, K*7, d) into (B, K, 7d), because each future timestep has seven hidden states that together should predict that timestep's 7-D noise. A small pre-LN MLP-residual model maps 7d to 7, and the loss is MSE against the actual eps.

At inference I draw a normalized action chunk from N(0,I), set the DDIM timesteps, and repeat: encode the current timestep, project the current noisy chunk, run the trunk, extract action hidden states, predict the noise, and call the scheduler step. The cost is one trunk pass per selected diffusion step. That is the price I pay to represent a distribution rather than a single action vector.

The final design is therefore fixed: per-scalar noisy-action projector, sinusoidal timestep token, DDIM scheduler with squared-cosine schedule, MLP-residual epsilon predictor over grouped action-slot hidden states, one-shot forward noising during training, MSE on epsilon, and reverse DDIM rollout for prediction. The answer should keep the code in the scaffold slot, while this reasoning stays focused on why each piece is forced by the objective and by the VLA action-token layout.
