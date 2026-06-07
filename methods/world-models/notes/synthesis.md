# World Models — synthesis (arXiv 1803.10122, verified — Ha & Schmidhuber 2018)

## Pain point / goal
RL agents benefit from a good representation of past+present and a predictive model of the future. Large expressive RNNs would make great predictive models, but model-free RL is bottlenecked by the CREDIT ASSIGNMENT problem — hard to train millions of weights of a large net with sparse reward signal. So small nets are used in practice, sacrificing capacity. Goal: efficiently train a LARGE network for RL by splitting the agent into a large WORLD MODEL (trained unsupervised, by backprop) + a SMALL CONTROLLER (trained by black-box search). Small controller -> credit assignment lives in a tiny search space; capacity lives in the world model. Distills concepts from a line of RNN-world-model+controller papers 1990-2015 (Schmidhuber).

## Three components (V, M, C)
- **V (Vision) = Convolutional VAE.** Compress each 64x64x3 frame into latent z. Car Racing N_z=32, Doom N_z=64. Encoder 4 conv (stride 2, relu) -> mu, sigma (size N_z); z ~ N(mu, sigma I) reparameterized. Decoder fc(1024) -> 4 deconv (stride 2, relu; output sigmoid to [0,1]). Loss = L2 reconstruction + KL. Gaussian prior limits info capacity AND makes world model robust to unrealistic z from M. Trained 1 epoch on random-policy data.
- **M (Memory) = MDN-RNN.** LSTM + Mixture Density Network output. Models P(z_{t+1} | a_t, z_t, h_t) as a mixture of (diagonal/factored) Gaussians, NOT a deterministic prediction — because environments are stochastic. Car Racing LSTM 256 units; Doom 512 units. 5 Gaussian mixtures, no correlation rho (factored). For Doom, M also predicts done d_t (binary). Temperature tau scales sampling uncertainty (from SketchRNN). Trained 20 epochs. Teacher forcing: store mu,sigma per frame, re-sample z~N(mu,sigma) each batch to avoid overfitting a specific z.
  - MDN parameterization (grounded in ctallec code): linear head outputs (2*L+1)*G (+2 for r,d): mus (G,L), sigmas=exp(.) (G,L), pi=log_softmax over G. 
  - GMM NLL loss: -log sum_k pi_k N(z_{t+1} | mu_k, sigma_k), computed stably with log-sum-exp (subtract max log prob). log N summed over feature dim, + logpi, logsumexp over mixtures.
- **C (Controller) = single linear layer.** a_t = W_c [z_t ; h_t] + b_c. Deliberately tiny: Car Racing 867 params, Doom 1088. tanh to bound actions. Trained SEPARATELY from V,M.
  - Car Racing h_t = LSTM output h. Doom: both cell c and output h.
  - Optimized by CMA-ES (good up to a few thousand params). Population 64, each agent 16 rollouts with different seeds, fitness = average cumulative reward over the 16. ES advantages: only needs final cumulative reward (no per-step credit), trivially parallel, robust.

## Integration / rollout (verified pseudocode)
obs = env.reset(); h = rnn.initial_state()
while not done: z = vae.encode(obs); a = controller.action([z,h]); obs,reward,done = env.step(a); h = rnn.forward([a,z,h])
C sees [z_t ; h_t]: z = current observation compressed; h = M's hidden state which encodes the distribution of the FUTURE. So C gets present + predicted-future. Car Racing experiment: V-only controller wobbles (632); V+M (z and h) stable (906) — h gives predictive context, so agent acts reflexively without explicit planning.

## Dreaming (VizDoom: learning inside the dream)
Once M predicts P(z_{t+1}, d_{t+1} | a_t, z_t, h_t), it IS a full RL environment: wrap M as a gym.Env. Train C ENTIRELY in latent space inside M's hallucination — V is NOT needed during the dream (no pixel decoding), only to encode the initial collection data and (optionally) to visualize. Transfer policy back to real env. Take Cover: survive (reward = timesteps alive), solved if avg > 750 over 100 rollouts. Trained in dream (score ~900), transferred to real env -> ~1100.

## Cheating the world model + temperature (the key wall)
A controller trained inside M can find ADVERSARIAL policies that exploit M's imperfections — e.g. move so monsters never shoot fireballs; visits OOD states where M is wrong. Worse: C has access to ALL of M's hidden states (the "game engine internals"), so it can directly manipulate them. This is WHY many prior works learn dynamics models but DON'T fully replace the real env (deterministic models are easily exploited).
FIX: use the MDN-RNN's STOCHASTICITY. A mixture-of-Gaussians models a distribution of outcomes; even a deterministic env gets approximated as stochastic. Crank the TEMPERATURE tau to control randomness/exploitability tradeoff: higher tau -> more uncertain dream, harder to exploit. tau=0.1 -> mode collapse (monsters never shoot -> perfect dream score but fails in reality). tau=1.15 best transfer. Agents that do well at high tau transfer better. The discrete modes of the mixture matter for discrete random events (monster shoots or not) — a single Gaussian (like V) can't represent that; the MDN can.

## Iterative training procedure (for harder tasks)
1. Init M,C random. 2. Rollout real env N times, save a_t, x_t. 3. Train M to model P(x_{t+1}, r_{t+1}, a_{t+1}, d_{t+1} | x_t, a_t, h_t); train C inside M. 4. Repeat if not done. One iteration sufficed for the simple tasks. Curiosity: flip sign of M's loss in real env to reward exploring where M predicts poorly (compression-improvement reward, Schmidhuber). M can absorb motor skills, freeing C for higher-level skills.

## Lineage / baselines
- Schmidhuber 1990 "Making the World Differentiable" + 1990-91 papers (s05a/b/c): RNN world model + controller, step-by-step planning, DETERMINISTIC M (exploitable). World Models is closer to these but uses evolution for C.
- "Learning to Think" (Schmidhuber 2015): unifying RNN-M + RL-C framework; C can use M's subroutines or ignore a flawed M. World Models uses its terminology; iterative procedure from here. But World Models still uses M's predictions to teach C (older C-M style) + evolution.
- PILCO (Deisenroth & Rasmussen 2011): GP dynamics, sample many trajectories to train controller. Bayesian uncertainty helps exploitability. But GPs don't scale to high-dim pixels.
- Deep PILCO / Bayesian NN dynamics: low-dim states.
- VAE (Kingma & Welling 2014). MDN (Bishop 1994). MDN-RNN: Graves 2013 (handwriting), SketchRNN (Ha 2017). CMA-ES (Hansen). ES for RL (Salimans/OpenAI 2017, stablees).
- Prior latent-space control: embed-to-control, deep spatial autoencoders, from-pixels-to-torques — autoencoder bottleneck features to control pendulum. World Models adds the temporal/predictive M.

## Design-decision -> why
- VAE not plain AE: Gaussian prior bounds info capacity per frame + robustness to unrealistic z.
- MDN (mixture) not single Gaussian for M: env is stochastic + discrete random events (fireball or not) need multiple modes; also gives temperature knob for exploitability control. (V can be single Gaussian since it's just per-frame.)
- Linear C: credit assignment search space tiny -> ES feasible; capacity in V,M.
- CMA-ES not gradient/RL for C: only needs final reward (no backprop through env/dream), parallelizable, robust; works up to few thousand params.
- Feed h_t (not just z_t) to C: h encodes the future distribution -> reflexive control without explicit planning rollouts.
- Temperature tau: control realism vs exploitability; prevents controller from cheating M.
- Train V,M,C separately (not end-to-end): more practical, each <1hr on a GPU.

## Code grounding (ctallec/world-models PyTorch, verified)
- vae.py: Encoder conv(3->32->64->128->256, k4 s2) -> fc_mu, fc_logsigma; reparam z=mu+sigma*eps; Decoder fc(1024)->deconv(1024->128->64->32->3, k5/k5/k6/k6 s2), sigmoid out.
- mdrnn.py: gmm_linear Linear(hiddens, (2*latents+1)*gaussians + 2). LSTM(latents+actions, hiddens). split mus/sigmas=exp/pi=log_softmax; rs = [...,-2]; ds = [...,-1]. gmm_loss = -log sum_k pi_k prod_f N(z_f|mu, sigma) via logsumexp.
- controller.py: single Linear(latents+hiddens, actions).
VAE loss = BCE/MSE recon + KLD. MDRNN loss = gmm_loss(next_z) + MSE(reward) + BCE(terminal).
