# Sources retrieved and read this run (three-source bottom line)

## (1) Primary sources — read in full
- **IPPO primary:** de Witt, Gupta, Makoviichuk, Makoviychuk, Torr, Sun, Whiteson — "Is Independent
  Learning All You Need in the StarCraft Multi-Agent Challenge?" arXiv:2011.09533 (2020).
  arXiv LaTeX source extracted to `src/ippo/` (all sections read: abstract, intro, related work,
  background/preliminary, method §4 with the IPPO loss / GAE advantage / value clipping, discussion,
  conclusion; including the commented-out policy-improvement-lemma and coordinate-ascent appendix
  reasoning, which supplied the omitted derivation backbone for reasoning.md).
- **Co-definitive primary (the one the task cites):** Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu —
  "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games." arXiv:2103.01955,
  NeurIPS Datasets & Benchmarks 2022. Source in `src/mappo/`; read prelim, MAPPO/IPPO definition,
  related work, and §5.2 "Input Representation to Value Function" (the IND/EP/CL/AS/FP critic-input
  design space — the crux of the centralized-vs-decentralized critic question).

## (2) Background / load-bearing ancestors — read for core idea + the limitation that motivated IPPO
- **PPO:** Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347 (2017).
  PDF `refs/ppo-1707.06347.pdf`, text `refs/ppo.txt`. Clipped surrogate eq (7), combined objective
  eq (9), truncated GAE eq (11-12) read and verified verbatim.
- **TRPO** (KL trust region) and **GAE** (Schulman et al. 2016) via the PPO paper's background +
  the IPPO/MAPPO prelims.
- **COMA / Central-V / MADDPG / VDN / QMIX** (centralized critic + factorization ancestors and their
  gaps) via the IPPO related work and the Amato survey below.

## (3) Third-party explainer — read for the design rationale (the why)
- **Amato, "An (Initial/) Introduction to Centralized Training for Decentralized Execution in
  Cooperative Multi-Agent Reinforcement Learning,"** arXiv:2409.03052.
  PDF `refs/ctde-intro-2409.03052.pdf`, text `refs/ctde-intro.txt`. Read §4.1-4.7: decentralized vs
  centralized critics, IACC/IA2CC, MAPPO eq (18-19), **IPPO eq (20-21)** (`V_i(h_i)`, local advantage,
  local return target, parameter sharing → one shared actor+critic), §4.6 state-based critics are
  biased under partial observability with the Dec-Tiger example and the unbiasedness identity
  `Q^π(h,a) = E_{s|h}[Q^π(h,s,a)]`, and §4.7 the explicit **bias-variance tradeoff**: centralized
  critics raise actor-update variance (must marginalize peers out) and scale poorly; decentralized
  critics already remove peer info; the Peshkin et al. (2000) equivalence of joint and summed
  decentralized policy gradients; agent-id for specialization under parameter sharing.

## Canonical implementation (the artifact the trace lands on)
- EPyMARL `ACCritic` (`src/modules/critics/ac.py`) = IPPO critic; selected by `critic_type: "ac_critic"`
  in `ippo.yaml`. `CentralVCritic` (`centralV.py`) = MAPPO; `PPOLearner` (`ppo_learner.py`) = the fixed
  learner (clipped surrogate, n-step returns q_nstep=5, target critic, Adam, common-reward broadcast).
  The task's `ippo_critic.edit.py` reproduces ACCritic exactly. Read directly from the repo at
  `/srv/home/bohanlyu/MLS-Bench/vendor/external_packages/epymarl/`.

## Self-account
No author self-account / award lecture / blog for IPPO exists in SELF_ACCOUNT_SOURCES.md, and a
budgeted search located none. Flagged. Trace reconstructed from primary + ancestors + the Amato
explainer; the omitted-reasoning backbone came from the IPPO paper's own commented-out
policy-improvement-lemma and coordinate-ascent sections plus Amato §4.7.

## Repair pass (svfix W3_primary_plus_ancestors, 2026-08-19) — quality-gate re-check

TRIAGE (class=A) flagged the decisive step "reject centralised state critic V(s) for local
critic V(z^a); V(s) biased under partial observability, V(h,s) unbiased" and pointed at the
already-on-disk Amato survey (refs/ctde-intro.txt, §4.6, lines 1055-1090) as unused material to
cite. A first fix attempt quoted Amato §4.6 into reasoning.md at that step. An independent
verifier rejected it: the quote is real (Amato 2409.03052 §4.6 does state the Dec-Tiger
state-critic-bias result qualitatively and cites Lyu et al. 2022/2023 for it) but decorative —
reasoning.md paras 7-9 already derive the same conclusion through the trace's OWN worked
Dec-Tiger arithmetic (sensor accuracy 0.85, listen cost -1, payoffs +10/-100 — the textbook
Dec-Tiger parameters from Nair et al. 2003 / Oliehoek & Amato 2016, not from ctde-intro.txt,
which states the example qualitatively with no numbers), and para 10 derives V(h,s)'s
unbiasedness directly via the tower rule rather than citing Amato's Q^π(h,a)=E_{s|h}[Q^π(h,s,a)]
identity. Deleting the Amato sentence left the step's justification fully intact, so the
citation did not carry weight; grafting it back would repeat the same defect.

Re-checked against the quality gate directly (SONNET_FIX_PROMPT.md §Quality gate):
- (a) genuinely derived on the page: yes. The "obvious move" is centralising the critic on
  ground-truth state s (§CTDE degree of freedom, para 1-4); it fails for a checkable reason
  computed on the page: belief after k=0 vs k=3 consistent "R" observations under
  LR=0.85/0.15≈5.67 gives b_0=0.5, b_3≈0.9945, so V(h0)=10(0.5)-100(0.5)=-45 and
  V(h3)=10(0.9945)-100(0.0055)≈+9.40, forcing V(s)=½(-45)+½(9.40)=-17.8 — 27.2 below the true
  confident-history value, with the same collapse making an information-gathering action
  (b_1≈0.85→b_2≈0.97, ΔV≈+13.2) register as ΔV(s)=0. Independently re-verified all of this
  arithmetic here; it is correct to the stated precision (b_3=5.6667^3/(5.6667^3+1)=0.99453,
  V(h3)=9.395, V(s)=-17.8, b_1=5.6667/6.6667=0.850, b_2=5.6667^2/(5.6667^2+1)=0.9698,
  V(b_2)=6.677). The resolution (recover unbiasedness via V(h,s), tower rule) follows
  mechanically from that computation, not asserted.
- (b) the obstacle is the trace's own honest computation using a standard textbook problem
  (Dec-Tiger, Nair et al. 2003) — this is exactly the gate's explicit "trace's own honest
  computation" branch, independent of whether ctde-intro.txt is cited.
- The subsequent moves (V(h,s) unbiased but potentially high-variance -> reject bare
  non-stationarity objection via PPO clip's actual mechanism -> coordinate-ascent telescoping
  proof -> Peshkin et al. 2000 gradient-equivalence) are already grounded with real citations
  (Tan 1993, Claus & Boutilier 1998, Peshkin et al. 2000) present in reasoning.md paras 12/19,
  none of which were touched.

Verdict: sound_as_is. No source grafted onto reasoning.md/answer.md/train_answer.md for this
step. No factual error found (arithmetic re-verified above). Amato §4.6 remains useful general
background (documented under (3) above) but is not load-bearing for the Dec-Tiger derivation
specifically, and should not be cited inline at that passage.
