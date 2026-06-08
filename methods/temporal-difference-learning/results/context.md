# Context: Learning to predict from experience with an incompletely known system

## Research question

The problem is *learning to predict*: using past experience with a partially known dynamical system to forecast its future behavior — whether a chess position leads to a win, whether a cloud formation brings rain, how far the market will move, whether a word in a waveform is about to be recognized. These predictions arrive not as isolated questions but along a *temporal sequence*: a walk evolves state by state, a game unfolds move by move, a year of economic data accumulates day by day, and at each step partial evidence about the eventual outcome is revealed. The eventual outcome is known only at the end (a multi-step prediction problem), but informative changes happen all along the way.

A solution has to do two things well. It must be *cheap and incremental* — usable on a stream of input as it arrives, without storing the whole sequence and dumping all the computation at the end. And it must make *efficient use of experience* — converge fast and predict accurately from a finite, often small, amount of data. The tension is that the obvious learning signal, the gap between a prediction and the true outcome, is only available at the very end of a sequence and is a single noisy draw. The question is whether the *sequential structure itself* — the relation between one prediction and the next — can be turned into a learning signal that is both cheaper to compute and statistically better.

## Background

The dominant learning paradigm is **supervised learning**: associate pairs of items, recall the second when shown the first. Any prediction problem can be forced into this mold by pairing the data on which a prediction is made with the actual outcome — to predict Saturday's weather, pair Monday's measurements with Saturday's observed weather, pair Tuesday's with Saturday's, and so on. This pairwise framing is easy to analyze and is the workhorse of pattern classification, system identification, and associative memory, but it discards the sequential relation between the days. Predictions are taken to be functions P(x, w) of an observation vector x and a weight vector w; in the simplest, linear case P = wᵀx.

The field's understanding of *credit assignment* splits two ways. **Structural** credit assignment asks which part of a system to change to affect an output — the problem backpropagation (Rumelhart, Hinton & Williams, 1985) solves by propagating error derivatives backward through a differentiable network. **Temporal** credit assignment (Minsky's 1961 "Steps Toward Artificial Intelligence" named the credit-assignment problem) asks which earlier decision in a sequence deserves credit for a later success. The two are orthogonal and combinable.

Three threads sit behind any attempt at sequential prediction. From **optimal control**, Bellman (1957) characterized an optimal return (value) function by a functional equation — the Bellman equation — and the family of methods that solve it iteratively became dynamic programming. These methods are *iterative and incremental* in the sense that they reach the answer by successive approximation, each new estimate built from the current estimates of neighboring states; but they require a full model of the system's transition probabilities and sweep the entire state space, and they suffer the curse of dimensionality. From **animal learning psychology**, the notion of a *secondary reinforcer* — a stimulus that, having been paired with food or pain, takes on reinforcing power of its own — captures the idea that a *prediction* of reward can itself act as reward, letting credit pass back along a chain. From **engineering learning rules**, the Widrow-Hoff delta rule gave a robust incremental way to drive weights by an error signal.

A diagnostic fact about supervised prediction sets up the whole problem. Consider a two-person game with a state long known to be "bad" (loses 90% of the time). A novel state is encountered, the play passes through the bad state, and the game happens to end in a *win*. A supervised method pairs the novel state with the win and concludes the novel state is good. But the novel state led to a position known to usually lose; what happened afterward was luck. The intuitively right conclusion — the novel state is bad — comes from comparing it to the *next* prediction, not to the noisy final outcome. The final outcome is corrupted by random factors that occur *after* the state being evaluated; a subsequent prediction can be a less noisy performance standard. That statistical fact about sequential prediction is the crux the field has not exploited.

## Baselines

**Widrow-Hoff / LMS / delta rule (Widrow & Hoff, 1960).** The prototypical supervised update. For a linear predictor P_t = wᵀx_t with outcome z,
  Δw_t = α (z − wᵀx_t) x_t,
where α is a learning rate. The scalar error z − wᵀx_t is the gap between prediction and what it should have been; multiplying by x_t points each weight in the direction that reduces the error. Robust, effective, and the best-understood learning rule of its time (Widrow & Stearns, 1985); under repeated presentation of a finite training set it converges to the weights that minimize RMS error *on that training set*. Gap: every increment depends on the outcome z, so on a multi-step problem all observations and gradients must be remembered until the sequence ends and then updated in a burst — O(M) memory and peak compute for a length-M sequence. And it minimizes training-set error, which is not the same as predicting future experience well; it ignores the sequence's structure entirely.

**Rescorla-Wagner model of Pavlovian conditioning (Rescorla & Wagner, 1972).** Learning occurs when events violate expectations: the change in a conditioned stimulus's associative strength is driven by the discrepancy between the actual reinforcement λ on the trial and the composite prediction V̄ = Σ_i V_i X_i summed over present stimuli,
  ΔV_i = β (λ − V̄) · α_i X_i,
with β, α_i positive constants and X_i an indicator that stimulus i is present. This is structurally the Widrow-Hoff rule in psychological clothing (λ ↔ z, V̄ ↔ prediction), and it accounts for blocking and overshadowing through competition for a limited amount of "surprise" λ − V̄. Gap: it is a *trial-level* model — it collapses an entire trial into one λ and one update and is blind to the *timing of stimuli within a trial*, which empirically matters a great deal (the CS–US interstimulus interval strongly shapes learning). It also mispredicts second-order conditioning: when a stimulus B is paired with an already-trained A (no US present, so λ = 0), the model can only predict B's association decreasing or staying flat, yet animals form a positive B association. A real-time model is needed, one in which the effective reinforcement varies *within* the trial.

**Samuel's checker player (Samuel, 1959).** The earliest use of a successive-prediction idea. The evaluation of a board position is treated as a prediction of how the game will turn out from there; for each pair of successive positions, the difference between their evaluations ("delta") is used to adjust the earlier position's evaluation toward the later one (the later one computed by a minimax-backed-up lookahead search). Gap: there is *no terminal grounding* — no position has an a priori correct evaluation and the last position's value is never tied to the actual game outcome, so the consistency constraint "each evaluation should match its successor's" is satisfied by useless constant functions. Samuel patched this with a non-modifiable piece-advantage term and, when self-play made the program worse, by zeroing the largest weight. The update was ad hoc and never analyzed for convergence.

**Dynamic programming (Bellman, 1957).** The value function V obeys V(s) = E[r + γ V(s′)], and DP solves this by iterating an estimate of V using the current estimates of successor states — it *bootstraps*, updating estimates from other estimates. Gap: it requires the transition model and sweeps the whole state space; it is a planning method given a model, not a method for learning from raw experience.

**Witten's adaptive controller (Witten, 1977).** Contains the earliest published rule of the form that updates a discounted-cost prediction by the discrepancy (c_{t+1} + γ P_{t+1}) − P_t, embedded inside an adaptive controller for a Markov environment. Gap: it lived inside a controller, was never isolated or named, and its sketched convergence argument was incomplete (the stated theorem appears not to hold).

**Adaptive Heuristic Critic / bucket brigade (Barto, Sutton & Anderson, 1983; Sutton, 1984; Holland, 1986).** Successive-prediction updates used as components of larger trial-and-error control systems — a critic that learns to predict reward, or production-rule strengths passed back along chains. Gap: studied only inside complex systems, entangled with control and action selection, never analyzed on their own as prediction methods, never proved to converge to correct predictions.

## Evaluation settings

The natural yardstick is a simple, fully analyzable dynamical system whose true predictions can be computed in closed form. A **bounded random walk** serves: states A–B–C–D–E–F–G, every walk starts in the center D, each step moves left or right with probability ½, and the walk terminates on entering edge state A (outcome z = 0) or G (outcome z = 1). The learner sees an observation-outcome sequence (e.g. x_D, x_C, x_D, x_E, x_F, 1) with each nonterminal state represented by a distinct unit basis vector, so the prediction for a state is just one component of w. The estimand is the probability of right-side termination from each state; the ideal values (1/6, 2/6, 3/6, 4/6, 5/6 for B–F) are computable from the Markov chain, so a learning rule's predictions can be scored against ground truth.

The metric is root-mean-squared error between learned and ideal predictions, averaged over many independently generated training sets (100 sets of 10 sequences each) for statistical stability. Two protocols are relevant: *repeated presentation*, where a fixed training set is shown over and over until the weights stop changing (probing the asymptotic fixed point of a rule), and *single presentation* with after-each-sequence updates across a range of learning rates α (probing speed of learning from limited data). The broader, harder domains the methods ultimately target — game-playing evaluation, pole balancing, speech and process monitoring — are out of scope for clean measurement; the random walk is chosen precisely because it isolates the one feature conjectured to matter: that the data come from a dynamical system with an evolving, partially observed state.

## Code framework

The scaffold is a harness for multi-step prediction over observation-outcome sequences. A linear predictor and the conventional wait-for-the-outcome supervised update already exist; the slot to be filled is the rule that turns the *sequence of predictions* into incremental weight changes.

```python
import numpy as np

def predict(w, x):
    # linear prediction P(x, w) = w^T x ; gradient wrt w is x
    return float(w @ x)

def grad_predict(w, x):
    return x  # linear case: dP/dw = x

def run_sequence(w, observations):
    # observations: list of feature vectors x_1..x_m
    # produce the sequence of predictions P_1..P_m
    return [predict(w, x) for x in observations]

class SupervisedPredictor:
    """Prototypical pairwise (Widrow-Hoff) update: each increment uses the
    final outcome z, so it is computed only after the sequence ends."""
    def update(self, w, observations, outcome, alpha):
        dw = np.zeros_like(w)
        preds = [predict(w, x) for x in observations]
        for x, P in zip(observations, preds):
            dw += alpha * (outcome - P) * grad_predict(w, x)   # depends on z
        return w + dw   # must wait for outcome; O(m) state held until the end

class SequencePredictor:
    """Placeholder for a streaming rule: drive weight changes from the relation
    between temporally successive predictions, computed as the sequence is
    experienced rather than after the outcome is known."""
    def __init__(self):
        self.running_summary = None   # per-sequence state to be defined

    def step(self, w, x_t, x_next, reward, alpha):
        # TODO: increment w from successive predictions, incrementally
        pass

    def end_of_sequence(self):
        # TODO: reset whatever per-sequence state the rule carries
        pass
```
