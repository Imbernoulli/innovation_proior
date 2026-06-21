The goal is to remove the human from the loop of designing convolutional image classifiers: to automatically discover an architecture that classifies as well as, or better than, the hand-tuned designs that experts have spent years on — AlexNet, Inception, ResNet, DenseNet. The methods that exist split into two camps with complementary defects. Reinforcement-learning search holds the accuracy records — an LSTM controller emits an architecture token by token and is nudged by policy gradient toward the architectures that scored well — but it is heavy: I am training a whole second neural network by backpropagation, carrying its own learning-rate schedule, a greediness knob, batching, and a replay buffer on top of all the hyperparameters of the classifiers I am actually searching for; two implementations of the same RL algorithm have been shown to give different answers; and the controller update couples the workers together, which fights the one thing I most need here, clean asynchronous parallelism, because every candidate has to be trained from scratch before I even know if it is any good and that training is the entire cost of the enterprise. Evolution is the opposite — keep a population of trained models, copy the good ones, mutate them, throw away the bad ones, no controller and nothing tying the workers together — and it is wonderfully simple, but the field's standing belief is that evolved image classifiers have never actually beaten hand-designs on realistic images; people who tried it got close on CIFAR-10 and stalled just short. So the real question is not whether to use evolution but what the one thing wrong with plain evolution is that keeps it just short, and whether it can be fixed without piling on the complexity I went to evolution to escape.

The natural starting point is the steady-state tournament loop. I keep $P$ trained models. Each cycle I sample $S$ of them at random, take the highest-accuracy one as the parent, mutate it into a child, train and evaluate the child, and add it to the population; to hold the population at size $P$, the textbook move — and what large-scale evolution did just before this — is to remove the *worst* model of the sample. That loop is clean, asynchronous (one model in, one out, no generational barrier where fast machines idle waiting on the slow ones), and the tournament gives a tunable amount of greed. The greed is worth making precise, because $S$ is the only real knob. Writing the birth-life-death equation $m_{t+1} = m_t + (\text{births}) - (\text{deaths})$ for the count of the single best class and asking how long until it takes over a population of size $n$, the takeover time for an $s$-ary tournament with $s>1$ comes out as
$$t^* \approx \frac{1}{\ln s}\,\big[\ln n + \ln(\ln n)\big],$$
so the dependence on $s$ runs through $1/\ln s$: doubling $s$ roughly halves the takeover time, the best model floods the population twice as fast. $S$ is exactly the dial on selective pressure — $S=1$ is no selection at all, pure random search, and as $S$ climbs toward $P$ the search gets greedier. So I have a simple, parallel loop with one interpretable knob, and it does not beat hand-designs. The reason it does not is that `train_and_eval` is *noisy*: random initialization, random data ordering, stochastic regularization mean that re-training the exact same architecture gives a different validation accuracy, with a spread of about a percent. The accuracy I am selecting on is the architecture's true quality plus a noise term, and that exposes the trap built into kill-the-worst. Suppose a mediocre architecture is trained on a lucky day and posts an accuracy a full noise-width above where it deserves. Under kill-the-worst it is *high* accuracy, so it never loses a tournament, never gets evicted; it just sits there, and because it sits there at a lucky score it keeps being chosen as the parent, spawns child after child, and the whole population collapses onto a fluctuation that was never actually good. Premature convergence, caused by the very rule meant to help: rewarding high observed accuracy with permanent residency is rewarding noise with permanent residency, and the longer a model can live by looking good, the more the search overfits to the luck in its single evaluation. The within-frame patches do not fix the root cause. Lowering $S$ only slows the takeover — the lucky model still never dies, it just takes longer to flood, and I lose the early-search quality I care about. Re-evaluating models to average out noise costs a full training run each time, roughly halving how many distinct architectures I can afford. Capping how many children a model may produce adds a meta-parameter and a per-model counter, the complexity I am trying to avoid. Each patch either misses the real disease — a good-looking model lives forever — or buys the fix with cost or knobs I will not pay.

So I cut the disease at its root, and I call the result regularized evolution, also aging evolution. The disease is the coupling between observed score and lifespan: score high, live long, reproduce a lot. I sever it completely — a model's lifespan has *nothing to do with its accuracy*. Instead of removing the worst model each cycle, I remove the *oldest*: the newest child goes in on the right, the most senior model comes off the left, no matter how good either one is. The single line `population.popleft()` is the entire contribution; everything else is the plainest tournament loop. Every model now has the same fixed maximum lifespan, on the order of $P$ cycles, and then it is gone, so the best model cannot homestead. My first reaction is that this should be worse — deliberately throwing away good models on a schedule sounds like throwing away the point of selection — but the key is to trace what happens to a good *architecture* rather than a good *model*, because that distinction is suddenly load-bearing. A model is one trained instance: an architecture plus the luck of its one training run, and under aging that instance is doomed, it ages out soon. The only way a good architecture survives in the population is to be re-created: a model carrying it must win a tournament and produce a child that also carries (most of) that architecture, and that child is a fresh model with its own independent noise draw. For the architecture to persist across generations it has to keep winning tournaments as it is re-instantiated, which means it has to keep posting good accuracy on *independent* re-trainings. A model that was lucky once cannot ride that single score to immortality, because age evicts it; to leave descendants it must be re-trained, and on re-training the luck is gone — a one-time fluctuation re-trains back down to its true mediocre level, its child loses the next tournament, and the lineage dies. The only architectures that stay are the ones that score well *repeatedly*, on independent runs — the ones whose high accuracy is a property of the architecture and not of one lucky evaluation. So kill-the-oldest does not throw away signal; it throws away the noise-riders and keeps the architectures robust to training noise. That is the whole mechanism: under kill-the-worst the way to persist is to have a high *observed* score, which a lucky model has, so the population accumulates whatever was lucky; under kill-the-oldest the way to persist is to be *re-discoverable*, so the population accumulates whatever is genuinely good. I am injecting prior knowledge — I want architectures that re-train well, and I constrain the survivors to exactly those — and constraining the solution set to suppress fitting to noise is regularization in the plain mathematical sense, which is why the name is *regularized* evolution.

What makes this the version to keep is that it costs *no* new meta-parameter. Age is never stored: it is just position in a FIFO queue, append-right for newest and pop-left for oldest, so the same two knobs $P$ and $S$ remain, with $S$ controlling selective pressure through the takeover time above. This is unlike the older aging idea, ALPS, which attaches age to *genes* and splits the population into protected age-layers at the cost of two extra meta-parameters, the age-gap and the number of layers; here age attaches to *individuals* and is used for one thing only, identifying who is oldest. The rest of the loop stays bare on purpose, since every piece I do not add is a piece I do not have to tune. No recombination, because there is no meaningful way to splice two cell graphs and single mutations already reach the whole space. No fitness sharing or explicit diversity machinery, because the age-based turnover already supplies the diversity. Tournament selection rather than scanning for the global best each cycle, because the tournament is $O(1)$ on a small sample and reads the population locally — perfect for asynchronous workers each grabbing $S$ random entries with replacement and leaving the population untouched — whereas a global-best scan forces a global read and is maximally greedy. And the loop is steady-state, one-in-one-out, so workers never idle on a generational barrier. The exploration is the minimal pair of mutations that reaches every point of the cell space, where an architecture is a normal cell and a reduction cell, each built from five pairwise combinations that take two existing hidden states, apply an op to each, and add the results into a new hidden state, with unused states concatenated into the cell output. A *hidden-state mutation* picks a cell, one of its five combinations, one of that combination's two inputs, and rewires it to another hidden state in the cell, with the single constraint of no loops so the net stays feed-forward. An *op mutation* makes the same selection but swaps the op for a different one from the fixed menu (none/identity, $3\times3$/$5\times5$/$7\times7$ separable conv, $3\times3$ average and max pool, $3\times3$ dilated separable conv, the $1\times7$-then-$7\times1$ conv). An *identity mutation* changes nothing, at a small fixed untuned probability around $0.05$, which mostly gives a parent an occasional free re-draw of the training noise. One mutation per cycle, chosen at random: mutation supplies exploration, parent selection supplies exploitation, and the two are completely decoupled. To convince myself the mechanism is really about noise and not about convnets, I strip the architecture search away and keep only the noise: a model is a $D$-bit string, its accuracy is the fraction of bits matching a fixed target plus a small Gaussian of standard deviation around $0.01$ (tuned to the neural-net noise), a single optimum and no landscape so the only difficulty is the corrupted readout. The prediction is that aging and kill-the-worst tie at low $D$, where the problem is easy and noise barely matters, and aging pulls ahead as $D$ grows and the signal-per-bit shrinks relative to the fixed noise — a gap that widens with the noise is exactly the fingerprint of a noise-navigation mechanism. At scale the knobs used are $P = 100$ and $S = 25$, lightly tuned over five settings.

```python
import collections
import random

DIM = 100            # genome length (stands in for the architecture)
NOISE_STDEV = 0.01   # matches observed neural-net training noise


class Model:
    """An evaluated individual. 'Age' is NOT stored — it is the model's position
    in the FIFO population queue (leftmost = oldest)."""
    def __init__(self):
        self.arch = None       # the genome (here an int bit-string of length DIM)
        self.accuracy = None   # noisy validation accuracy (the 'fitness')

    def __str__(self):
        return '{0:b}'.format(self.arch)


def _sum_bits(arch):
    total = 0
    for _ in range(DIM):
        total += arch & 1
        arch >>= 1
    return total


def train_and_eval(arch):
    """Stand-in for building, training, and evaluating an architecture.
    NOISY: re-evaluating the same arch gives a different number."""
    accuracy = float(_sum_bits(arch)) / float(DIM)
    accuracy += random.gauss(mu=0.0, sigma=NOISE_STDEV)
    return min(1.0, max(0.0, accuracy))           # clip to [0, 1]


def random_architecture():
    """Sample a valid architecture uniformly over the search space."""
    return random.randint(0, 2 ** DIM - 1)


def mutate_arch(parent_arch):
    """One small random change. In the cell space this is a hidden-state rewire
    or an op relabel; in this toy space it flips one random bit."""
    position = random.randint(0, DIM - 1)
    return parent_arch ^ (1 << position)


def regularized_evolution(cycles, population_size, sample_size):
    """Aging / regularized evolution.

    Args:
        cycles:          total number of models to evaluate (the budget, C).
        population_size: P, models kept alive at once.
        sample_size:     S, tournament size = the selective-pressure knob.
    Returns:
        the highest-accuracy Model ever evaluated.
    """
    population = collections.deque()
    history = []

    # Initialize the population with random, evaluated models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture()
        model.accuracy = train_and_eval(model.arch)
        population.append(model)
        history.append(model)

    # Steady-state evolution: each cycle produces one model and removes one.
    while len(history) < cycles:
        # Tournament selection: S random members; population is left untouched.
        sample = [random.choice(list(population)) for _ in range(sample_size)]
        parent = max(sample, key=lambda m: m.accuracy)      # exploitation

        child = Model()
        child.arch = mutate_arch(parent.arch)               # exploration
        child.accuracy = train_and_eval(child.arch)         # fresh noise draw
        population.append(child)                            # newest -> right
        history.append(child)

        population.popleft()                               # AGING: evict OLDEST

    return max(history, key=lambda m: m.accuracy)


if __name__ == '__main__':
    best = regularized_evolution(cycles=1000, population_size=100, sample_size=10)
    print('best accuracy:', best.accuracy)
```
