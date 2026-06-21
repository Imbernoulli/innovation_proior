I would call this method the Kelly criterion. It is a rule for deciding what fraction of current wealth to risk on a wager or investment when the payoff is reinvested and the same decision will be repeated many times. The starting point is not a taste for caution or a philosophical claim about the value of money. It is the observation that repeated reinvestment turns capital into a product of per-round multipliers, and the natural way to study a product over many rounds is to take its logarithm. Once the logarithm is taken, the product becomes a sum, the law of large numbers applies, and almost every long trajectory reveals the same exponential growth rate. The Kelly criterion chooses the action that maximizes that rate.

The communication channel analogy makes the logic especially clear. Shannon defines the rate of a noisy channel as the mutual information between the sent symbol and the received symbol, and he gives that rate operational meaning through coding: below capacity, messages can be reconstructed with arbitrarily small error. But suppose the receiver is not trying to decode a message at all. Instead, the receiver observes a noisy signal about an event, then must act before the event becomes public, and the action is rewarded according to market odds. In that setting there is no external loss table to justify the value of the signal. The Kelly criterion supplies the value by treating capital allocation as a decoder whose score is long-run growth. When the odds are fair, meaning each outcome pays the inverse of its prior probability, the optimal growth-maximizing allocation is exactly the posterior distribution of the outcome given the signal. The maximum growth rate then equals the channel rate, the mutual information between outcome and observation. Information becomes spendable wealth.

This is different from maximizing expected terminal wealth. A policy that maximizes expected value can recommend betting everything whenever the wager has a positive edge, because rare favorable paths with enormous multipliers dominate the average. But the average is not what a single trajectory experiences. Almost all paths are driven by the typical per-round behavior, and a single total loss multiplies capital by zero and ends the game. The Kelly criterion rejects that policy because a zero multiplier is catastrophic in a product, not merely a finite additive setback. It optimizes the geometric mean of multipliers rather than the arithmetic mean, which is why the logarithm appears.

For the binary case that is easiest to teach and simulate, let the probability of winning be p, the probability of losing be q = 1 - p, and the net odds be b to 1, so that a winning stake returns the original stake plus b times the stake. If I bet a fraction f of current wealth, the two possible multipliers for that round are 1 + b f and 1 - f. The expected log growth rate is therefore g(f) = p log(1 + b f) + q log(1 - f). To find the optimal fraction I differentiate: g'(f) = p b / (1 + b f) - q / (1 - f). Setting this to zero and solving gives b p (1 - f) = q (1 + b f), which simplifies to b f = b p - q. The interior solution is f* = (b p - q) / b, equivalently p - q / b. The second derivative is always negative when both outcomes can occur, so this stationary point is the unique maximizer whenever it lies between zero and one. If the edge b p - q is non-positive, the best action is to bet nothing. With no shorting and no forced betting, the practical formula is f* = max(0, (b p - q) / b). For the special case of even-money odds where b = 1, the formula reduces to f* = 2p - 1, so a fifty-five percent chance of winning justifies risking ten percent of wealth.

The criterion generalizes cleanly to a horse-race with multiple outcomes. Let s index the outcomes, r index the received signals, a(s|r) be the fraction of wealth allocated to outcome s after seeing r, and alpha_s be the gross payoff odds for outcome s including the returned stake. The per-round capital multiplier when outcome s occurs is alpha_s a(s|r), and the long-run growth rate is the expectation over both signals and outcomes of the logarithm of that multiplier. Under fair odds alpha_s = 1 / p(s), the growth-maximizing allocation matches the posterior, a*(s|r) = p(s|r). Any deviation from the posterior incurs a mismatch penalty measured by the relative entropy between the true posterior and the chosen allocation. This turns portfolio choice into an information-matching problem.

It is important to keep the Kelly criterion separate from expected-utility theory. Von Neumann and Morgenstern showed that a few axioms imply the existence of a utility function, and Bernoulli long ago proposed a logarithmic utility of wealth. The Kelly criterion happens to involve a logarithm, but its justification is not that the decision maker feels logarithmic satisfaction from wealth. The logarithm is there because wealth is multiplicative across rounds. If the stakes were fixed outside dollars that could not be reinvested, maximizing expected payoff would be the right thing to do. The Kelly criterion is therefore the right rule for a repeated reinvested setting, not for an isolated gamble.

The following Python script illustrates the binary Kelly rule and checks some of the claims above. It defines the Kelly fraction, simulates a long sequence of reinvested wagers, compares the realized growth rate with the theoretical one, and also shows what happens under an all-in expected-value policy. The simulation is deliberately simple so the correspondence between the formula and the wealth trajectory is easy to verify.

```python
import random
import math

def kelly_fraction(p, b):
    """Return the no-short, no-forced-bet Kelly fraction for a binary wager."""
    q = 1.0 - p
    return max(0.0, (b * p - q) / b)

def simulate_wealth(p, b, f, n_rounds, seed=0):
    """Simulate reinvested wealth starting at 1 using fraction f each round."""
    rng = random.Random(seed)
    wealth = 1.0
    for _ in range(n_rounds):
        if rng.random() < p:
            wealth *= 1.0 + b * f
        else:
            wealth *= 1.0 - f
    return wealth

def theoretical_growth_rate(p, b, f):
    """Expected log growth rate g(f) = p log(1+bf) + q log(1-f)."""
    q = 1.0 - p
    return p * math.log(1.0 + b * f) + q * math.log(1.0 - f)

if __name__ == "__main__":
    p = 0.55          # probability of winning
    b = 1.0           # even-money net odds
    n = 100_000       # number of rounds

    f_star = kelly_fraction(p, b)
    print(f"Kelly fraction f* = {f_star:.6f}")

    final_wealth = simulate_wealth(p, b, f_star, n)
    realized_rate = math.log(final_wealth) / n
    predicted_rate = theoretical_growth_rate(p, b, f_star)
    print(f"Realized log growth rate: {realized_rate:.6f}")
    print(f"Theoretical log growth rate: {predicted_rate:.6f}")

    # Compare with an all-in policy, which maximizes expected wealth but usually goes broke.
    f_all_in = 1.0
    final_all_in = simulate_wealth(p, b, f_all_in, n)
    print(f"All-in final wealth: {final_all_in:.6e}")

    # Verify that the Kelly fraction is the maximizer by scanning nearby fractions.
    best_f, best_g = max(
        ((f, theoretical_growth_rate(p, b, f)) for f in [i / 200.0 for i in range(101)]),
        key=lambda x: x[1],
    )
    print(f"Best scanned fraction: {best_f:.4f} with growth rate {best_g:.6f}")
```

In summary, the Kelly criterion is the growth-optimal allocation rule for repeated reinvested decisions. It converts the problem of acting on partial information into the maximization of an expected logarithmic growth rate, derives the posterior-matching allocation in the multi-outcome fair-odds case, and gives the clean closed-form fraction p - q / b for binary wagers. It is not a risk-aversion axiom and it is not the same as maximizing expected wealth; it is the rule that makes the long-run wealth process grow as fast as the available information allows.
