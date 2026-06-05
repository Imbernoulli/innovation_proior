OK, let me lay out the contradiction plainly, because the whole thing starts as a clash of two beliefs that both have evidence behind them. Classical statistics says: as I make a model more complex, bias drops but variance rises, and past some sweet spot the variance term takes over, so test error is U-shaped in complexity — bigger is better up to a point and worse after. The prescription that falls out is "don't overparameterize" and "stop training before you overfit." But modern deep learning lives in flagrant violation of this. Networks with millions of parameters — enough to memorize randomly-labeled data outright — generalize beautifully, and the reliable experience is that bigger is *better*, and training all the way to zero training error often *helps*. Both of these are real. So either the classical U-curve is wrong, or the modern experience is a special case, or — and this is what I want to chase — they're two pieces of one curve and I've been looking at them separately.

The reconciling observation is already on the table: if I keep increasing model complexity past the point where the model can just barely fit the training data — the *interpolation threshold*, where training error hits zero — test error, which had been climbing (the classical U going up), turns around and *descends again*. So the real shape isn't a U; it's a U followed by a second descent. Down, up to a peak right at interpolation, then down again into the over-parameterized regime. That single picture holds both wisdoms: the left half is the classical bias-variance U (under-parameterized, complexity eventually hurts), the right half is the modern regime (over-parameterized, more complexity helps). The peak sits exactly where the model transitions from "can't quite fit the data" to "can fit it with room to spare."

But this picture, as it stands, is organized entirely around the *number of parameters*. And that nags at me, because the number of parameters is not the only thing I change that affects how a model fits. Consider what I actually do in practice. I make the network wider — more parameters, sure. But I also train it *longer* — same parameters, yet a network trained for 4000 epochs fits the data in a way a network trained for 10 epochs cannot. I turn data augmentation on or off. I add or remove weight decay. None of training-time, augmentation, or regularization is a "parameter," yet every one of them changes how much of the data the procedure can fit, and therefore — if the interpolation-threshold story is right — should move the peak. So parameter count is the wrong axis. It's *one* knob among several that all push the same underlying thing. I need to find that underlying thing: a single scalar that all of these knobs feed into.

What is the thing they have in common? Width lets the model fit more. Training longer lets it fit more. Less regularization lets it fit more. Turning off augmentation lets it fit more (fewer effective constraints). In every case "more complexity" means "the procedure can drive training error to zero on a larger training set." That's the invariant. So let me *define* complexity that way. Take a *training procedure* T to be the whole pipeline — architecture, optimizer, number of steps, augmentation, regularization, everything — anything that maps a labeled training set S to a trained classifier T(S). Then define the effective complexity of T as the largest number of samples it can still fit:

  EMC_{D,ε}(T) := max { n : E_{S ~ D^n}[ Error_S(T(S)) ] ≤ ε },

the maximum n such that, on average over training sets of size n drawn from the distribution, the procedure achieves training error at most ε (some small threshold; I'll use ε ≈ 0.1 to stand in for "essentially zero train error"). Call it Effective Model Complexity. The crucial property is that EMC is a property of the *procedure*, not just the architecture: a wider net has higher EMC, but so does the *same* net trained for *more steps*, and so does the *same* net with augmentation *off*. Training time, in particular, increases EMC — the longer I train, the more samples I can drive to zero error. This is the single axis I was missing; all those disparate knobs are just different ways of moving EMC.

Let me check this definition against what it has to do — locate the peak — and against the obvious alternative complexity measures, because if a classical measure already worked I wouldn't need a new one. The peak sits at the interpolation threshold, where the model just barely fits the data. Two facts about *where* that peak lands will be the test. First, adding label noise *moves* the peak (noisy labels are harder to fit, so you need a bigger model / more training to interpolate — EMC must reach a larger effective n). So the right complexity measure has to depend on the *labels* of the distribution, not just the inputs. Rademacher complexity and VC dimension are defined by a model family's ability to fit *arbitrary / random* labels — they're blind to the actual label distribution, so they can't explain why real label noise relocates the peak. EMC, by depending on Error_S over the *true* (noised) distribution, does. Second, epoch-wise effects and augmentation effects move the peak too — and those are properties of the *training procedure*, not the architecture alone. Rademacher and VC depend only on the model family and data, never on how you trained; they're structurally incapable of capturing a peak that shifts when you only change the number of epochs. EMC depends on T, so it can. So EMC differs from the classical measures in exactly the two ways needed to be the right axis: it sees the true labels, and it sees the training procedure.

Now I need to state the hypothesis in terms of EMC, and notice it has to be a statement about *where EMC sits relative to n*, the number of training samples I actually have. The interpolation threshold is "model can just fit n samples," i.e. EMC ≈ n. If EMC is sufficiently *smaller* than n, the procedure cannot fit the data I have, so increasing EMC should decrease test error: that is the classical under-parameterized regime, where more complexity helps because I am still struggling to fit. If EMC is sufficiently *larger* than n, the procedure fits the data with room to spare, so increasing EMC should also decrease test error: that is the modern over-parameterized regime, where bigger or longer is better. The only unstable place is EMC ≈ n. There the procedure can just barely fit the data, so a perturbation that increases EMC might decrease or might increase test error. That is the danger zone, the peak.

So test error peaks around EMC ≈ n, and there's a critical interval of EMC values around n inside which more complexity can hurt and outside which it helps. I won't pretend I can formally pin "sufficiently smaller/larger" or the width of that interval — those depend on D and T in ways I don't fully understand — but the qualitative claim is sharp: the sign of "does more complexity help?" flips depending on which side of EMC ≈ n you're on.

If this is the right axis, I should be able to cross EMC ≈ n in several different ways and see the same critical behavior each time.

Start by holding the training procedure's *length* fixed (train to completion, a large fixed number of steps) and varying the model *size*. As width grows, EMC grows, sweeping from EMC ≪ n through EMC ≈ n to EMC ≫ n. So I predict test error vs. model size: classical U on the small side, a peak right at the size where the model first interpolates the training set, then a second descent for larger models. That's model-wise double descent, and it implies — bluntly — that a bigger model should be worse when it moves the system into the critical interval. And since label noise, data augmentation, and more samples all raise the interpolation threshold (they make the data harder to fit, so EMC ≈ n happens at a larger model), every one of those should *shift the peak toward larger models*. That's a strong, falsifiable prediction the parameter-count framing alone wouldn't have generated.

The sharper test is the one the parameter-count view structurally cannot produce: hold the model *fixed* (a single large architecture) and vary *training time*. Since training longer increases EMC, a sufficiently large model *transitions from under- to over-parameterized over the course of a single training run*: early on EMC ≪ n (hasn't fit the data yet), at some epoch EMC ≈ n (just interpolating), after enough training EMC ≫ n. So along the time axis I predict the same shape: test error decreases at first (learning), then *increases* around the epoch where the model reaches ≈0 training error (passing through the critical region), then *decreases again* with continued training. Epoch-wise double descent. The startling consequence: *training longer can correct overfitting* — if you stop right when test error starts rising (the classical early-stopping advice) you stop in the middle of the peak and miss the second descent. For a medium model that only *barely* reaches zero training error, EMC ends up ≈ n and never climbs past, so you get just the classical U and early stopping is genuinely best. For a small model that never interpolates, EMC stays ≪ n and test error falls monotonically with training time. Three regimes, all corollaries of one hypothesis.

The sample-count direction is stranger because it moves the *other* side of the EMC-vs-n comparison. Hold the model and procedure fixed and vary n. Increasing n with EMC fixed pushes the procedure from over-parameterized (EMC ≫ n) toward critically and then under-parameterized (EMC ≪ n). Now think about what more data does to the model-size curve. It does two opposing things. It shrinks test error overall (more data, better estimates — the universally-agreed effect). But it *also shifts the interpolation peak to the right*, because fitting more samples requires a larger model, so the peak — the place where the curve is *high* — moves to model sizes that previously sat in the good over-parameterized valley. Near the critical regime these two effects fight: the downward push from more data and the upward push from the peak sliding onto you can cancel, so I should expect a band of model sizes where much more data does not help. And if the rightward shift wins locally, then for a fixed model *more data can hurt* — directly contradicting the one thing both classical and modern camps agreed on. Sample-wise non-monotonicity. The mechanism is exactly the same EMC ≈ n crossing, just traversed by moving n instead of EMC.

I want a mechanism for *why* the peak is high, not just where it is, and the cleanest place to get it is the linear / random-feature setting where I can reason exactly. At the interpolation threshold, the number of effective parameters equals the number of samples, so there is essentially a *single* solution that fits the training data — one interpolating model, no slack. A unique fit with no freedom is maximally sensitive: it's forced to bend itself to every training point, including the noisy or mis-specified ones, and "just barely fitting" means accommodating a slightly-wrong label distorts the whole solution's global structure, blowing up test error. Now go over-parameterized: there are *many* models that interpolate the training set, a whole solution space, and gradient descent from zero picks out the minimum-norm one — which can *absorb* the noise on the training points (it has the spare capacity to fit them) while keeping a well-behaved, low-norm structure that still matches the distribution. So the peak is a sensitivity spike at the unique-solution point, and the second descent is the optimizer having room to memorize noise harmlessly. This is provable for linear least squares and random features, and it appears even with *no* label noise whenever the model family mis-specifies the true distribution — which tells me label noise isn't fundamental. Label noise is just a controllable proxy for "harder distribution" / more model mis-specification: even *pseudorandom* noise that a Bayes-optimal classifier could invert would produce the identical double descent. That's why I add label noise in experiments — to *amplify* the peak into clear view — while believing the phenomenon is really about mis-specification.

The random-feature model makes the whole story concrete enough to anchor on: a two-layer network with a fixed random first layer of width d and a trained second layer (MSE loss, gradient flow → minimum-norm). Here EMC is *exactly* the width d. So on a grid of test error over (number of samples n, width d), the hypothesis predicts the high ridge runs precisely along n = d — and crossing the grid horizontally gives model-wise double descent, crossing it vertically gives sample-wise double descent. Same peak, same n ≈ EMC law, in a setting simple enough to compute. It also shows double descent is in no way special to deep nets; it's a property of the over-/under-parameterized transition itself.

One more consequence to nail down, about early stopping, because it both follows from the hypothesis and resolves a practical confusion. These phenomena should often *vanish* under optimal early stopping. That's not a contradiction — it's a prediction. If early stopping halts training before the model reaches ≈0 training error, then by definition EMC never reaches n; you stay on the under-parameterized side and never enter the critical region, so there's no peak to see. Optimal early stopping, in EMC terms, is a way of keeping EMC below n. So the classical "early stopping helps" advice is right *precisely in the critical regime* and not a universal law; over-parameterized training-to-completion is exactly the regime where pushing past the peak pays off.

Now I can write the operational harness. There's no new architecture or loss here — the contribution is the EMC definition and the hypothesis, and the way to make it real is the measurement harness: define EMC operationally, sweep one knob at a time while recording train and test error, and read the peak off at the interpolation threshold. The load-bearing pieces are (1) the label-noising step, drawn once so the noisy training set is fixed across epochs; (2) measuring EMC as the largest sample count reaching ≤ε training error; and (3) the three sweeps (over width, over epochs, over n) that each cross EMC ≈ n.

```python
import numpy as np

EPS = 0.1  # "approximately zero" train-error threshold

def add_label_noise(labels, p, num_classes, rng):
    # each label kept w.p. (1-p), else replaced by a uniform incorrect label.
    # drawn ONCE: the noisy training set is fixed across all epochs (not re-sampled).
    flip = rng.random(len(labels)) < p
    noisy = labels.copy()
    replacement = rng.integers(0, num_classes - 1, size=int(flip.sum()))
    original = labels[flip]
    noisy[flip] = replacement + (replacement >= original)
    return noisy

def make_model(width, fixed):
    # fixed.make_model is the existing architecture factory: ResNet18 [k,2k,4k,8k],
    # 5-layer CNN [k,2k,4k,8k]+FC, Transformer d_model with d_ff=4*d_model, or RFF width d.
    return fixed.make_model(width)

def train(model, train_data, test_data, optimizer, num_steps, fixed, record_every=None):
    # fixed.train uses the existing loss and optimizer protocol:
    # cross-entropy for vision, label-smoothed CE for Transformers, MSE for RFF.
    return fixed.train(model, train_data, test_data, optimizer, num_steps, record_every=record_every)

def effective_complexity(procedure, distribution, sample_grid, trials, epsilon=EPS):
    # EMC = the largest n on which the procedure reaches <= EPS average TRAIN error.
    emc = 0
    for n in sorted(sample_grid):
        train_err = np.mean([procedure(distribution.sample(n)).train_error()
                             for _ in range(trials)])
        if train_err <= epsilon:
            emc = n        # still interpolates n samples -> EMC at least n
    return emc

# (1) MODEL-WISE: fix #steps large, vary width. Peak where EMC(width) ≈ n (the interpolation threshold).
def model_size_sweep(widths, fixed):
    train_x, y, test_x, test_y = fixed.train_x, fixed.y, fixed.test_x, fixed.test_y
    p, num_classes, rng = fixed.p, fixed.num_classes, fixed.rng
    noisy_y = add_label_noise(y, p, num_classes, rng)        # label noise amplifies the peak
    curve = []
    for w in widths:
        m = make_model(w, fixed)
        tr, te, _ = train(m, (train_x, noisy_y), (test_x, test_y), fixed.make_optimizer(m),
                          fixed.num_steps, fixed)
        curve.append((w, tr, te))                            # te: down, peak at interpolation, down
    return curve

# (2) EPOCH-WISE: fix a large model, record test error over training time.
def training_time_sweep(width, step_budget, fixed):
    train_x, y, test_x, test_y = fixed.train_x, fixed.y, fixed.test_x, fixed.test_y
    p, num_classes, rng = fixed.p, fixed.num_classes, fixed.rng
    noisy_y = add_label_noise(y, p, num_classes, rng)
    m = make_model(width, fixed)                             # large enough to go under->over-param
    _, _, history = train(m, (train_x, noisy_y), (test_x, test_y), fixed.make_optimizer(m),
                          step_budget, fixed, record_every=1)
    return history                                           # te: down, up near interpolation, down again

# (3) SAMPLE-WISE: fix model+procedure, vary n. Crosses EMC ≈ n from the other side.
def sample_count_sweep(width, sample_sizes, fixed):
    curve = []
    for n in sample_sizes:
        x, y = fixed.dataset.subset(n)
        noisy_y = add_label_noise(y, fixed.p, fixed.num_classes, fixed.rng)
        m = make_model(width, fixed)
        tr, te, _ = train(m, (x, noisy_y), (fixed.test_x, fixed.test_y),
                          fixed.make_optimizer(m), fixed.num_steps, fixed)
        curve.append((n, te))                                # near critical n: more data may not help / may hurt
    return curve
```

The causal chain, end to end: classical theory says test error is U-shaped in complexity (bigger eventually worse) while modern practice says bigger and longer are better — and both are right on their own side of the *interpolation threshold*, the point where the model just fits the data, past which a *second descent* sets in. Organizing this around parameter count is too narrow, because training time, augmentation, and regularization also change how much a procedure can fit; the invariant they all move is the largest sample set the procedure can drive to zero training error, which I define as Effective Model Complexity. EMC depends on the true labels and the training procedure — exactly the two things Rademacher/VC ignore and exactly what's needed to locate a peak that moves with label noise and with epochs. The generalized hypothesis is that test error peaks at EMC ≈ n and that increasing complexity helps on both sides but can hurt in the critical interval; crossing EMC ≈ n three different ways — by width, by epochs, by sample count — yields model-wise double descent (bigger can be worse), epoch-wise double descent (training longer can undo overfitting), and sample-wise non-monotonicity (more data can hurt). The peak is the unique-interpolating-solution's noise sensitivity, provable in the random-feature model where EMC = width and the high ridge runs along n = d, with label noise serving as an amplifying proxy for model mis-specification — and early stopping makes the phenomena vanish precisely because it keeps EMC below n.
