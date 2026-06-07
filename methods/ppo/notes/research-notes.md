# PPO research notes (Phase 1)

## Primary paper (read in full, 12 pp incl. appendices A,B)
Schulman, Wolski, Dhariwal, Radford, Klimov. "Proximal Policy Optimization Algorithms", arXiv 1707.06347 (v2, Aug 2017).
Key equations verified directly against the PDF:
- (1) PG estimator ĝ = Ê[∇log π_θ(a|s) Â]; (2) L^PG = Ê[log π_θ Â]
- (3)-(4) TRPO: max Ê[ r_t Â ] s.t. Ê[KL[π_old, π_θ]] ≤ δ ; r_t = π_θ/π_old
- (5) TRPO penalty form: max Ê[ r_t Â − β KL ]
- (6) L^CPI = Ê[ r_t Â ]  (CPI = conservative policy iteration, Kakade & Langford 2002)
- (7) L^CLIP = Ê[ min( r_t Â, clip(r_t, 1−ε, 1+ε) Â ) ], ε=0.2
- (8) L^KLPEN adaptive: max Ê[ r_t Â − β KL ]; d=Ê[KL]; if d<d_targ/1.5 β/=2, if d>d_targ*1.5 β*=2
- (9) L^{CLIP+VF+S} = Ê[ L^CLIP − c1 L^VF + c2 S[π_θ] ]; L^VF = (V_θ−V_targ)^2
- (10) finite-horizon Â (A3C style); (11)-(12) truncated GAE: Â_t = Σ (γλ)^l δ_{t+l}, δ_t = r_t+γV(s_{t+1})−V(s_t)
- Algorithm 1: N actors × T steps, build surrogate on NT samples, K epochs minibatch SGD/Adam.
- Hyperparams (Table 3 MuJoCo): T=2048, Adam 3e-4, K=10 epochs, minibatch 64, γ=0.99, λ=0.95.
- Table 1 ablation: clip ε=0.2 best (0.82); no clip/penalty -0.39; adaptive/fixed KL all ~0.6-0.74.

## Load-bearing ancestors
- Vanilla policy gradient / REINFORCE (Williams 1992): score-function estimator, high variance, one update per sample, destructive large steps if reused.
- TRPO (Schulman 2015b, Sch+15b): surrogate + hard KL trust region; solved via conjugate gradient on Fisher-vector products + line search. Pain points: second-order, complex; needs Fisher-vector machinery; incompatible with param sharing / dropout / noise; one big constrained step, no cheap minibatch reuse.
- Kakade & Langford 2002 (KL02): conservative policy iteration, surrogate lower bound on policy performance -> source of "CPI" name and the surrogate-bound idea.
- A3C/A2C (Mnih 2016, Mni+16): parallel actor-critic, n-step returns Â_t = -V(s_t)+r_t+...+γ^{T-t}V(s_T), shared policy/value net, entropy bonus for exploration. PPO adopts the parallel-actor + entropy + fixed-T rollout structure.
- GAE (Schulman 2015a, Sch+15a): exponentially-weighted TD residuals; λ trades bias/variance; λ=0 -> 1-step TD, λ=1 -> Monte-Carlo. PPO uses truncated GAE for Â.
- Importance sampling: ratio r_t = π_θ/π_old lets off-policy reuse of old data, but ratio variance blows up as policies diverge -> motivates trust region / clipping.
- DQN (Mnih 2015): contemporary contender; works on Atari but not continuous control, poorly understood.

## Field state at the time (2017)
Three contenders: deep Q-learning (DQN), vanilla PG, and trust-region/natural PG (TRPO). Each flawed: DQN fails on continuous control & is brittle; vanilla PG has poor data efficiency/robustness; TRPO is complicated and not architecture-flexible. Open need: a scalable, data-efficient, robust, simple first-order method.

## Canonical implementation fetched
- CleanRL ppo_continuous_action.py (353 lines) and ppo.py in ppo/code/. The continuous version mirrors PPO's MuJoCo setup: 2x64 tanh MLP, separate actor/critic, Gaussian policy with state-independent logstd, GAE loop, max(pg_loss1,pg_loss2) clipped surrogate, clipped value loss, entropy bonus, Adam, minibatch K-epoch updates, LR anneal, adv normalization, grad clip. This grounds the Phase 2 code.

## Third-party explainers consulted
- OpenAI Spinning Up PPO: clean statement of L(s,a,θ_k,θ)=min(ratio·A, g(ε,A)); positive/negative advantage ceiling intuition; clip as regularizer / pessimistic lower bound; early-stop on KL.
- Spinning Up TRPO: conjugate-gradient + Fisher-vector-product framing and its cost.
