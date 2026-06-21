# Thompson Sampling

Maintain a posterior over the unknown reward parameter of each action. At each decision, choose each action with probability equal to its posterior probability of being optimal.

For Bernoulli rewards, start each action `k` with the uniform prior `Beta(1,1)`. After `r_k` successes and `s_k` failures,

```text
p_k | data ~ Beta(r_k + 1, s_k + 1).
```

The decision rule is

```text
Pr(play action k | data) = Pr(p_k = max_j p_j | data).
```

Equivalently, draw one parameter sample from each posterior and play the action with the largest sampled parameter:

```python
theta_k ~ Beta(r_k + 1, s_k + 1)  for each k
play argmax_k theta_k
observe reward
update only the chosen action's success/failure count
```

For two treatments, Thompson's exact small-sample probability is

```text
Pr(p2 > p1)
  = sum_{a=0}^{r2} C(r1+r2-a, r1) C(s1+s2+1+a, s1)
    / C(n1+n2+2, n1+1),

where n_i = r_i + s_i.
```

Then `Pr(p1 > p2) = 1 - Pr(p2 > p1)`.

The distinctive point is not just the beta posterior. It is the decision principle: posterior uncertainty about which treatment is best becomes the randomization law for the next treatment. If `P = Pr(p1 > p2)` and `Q = 1-P`, assigning treatment 1 with probability `P` gives expected inferior-treatment allocation `2PQ`, at most one half and strictly below one half whenever the evidence favors either treatment. Exploration is therefore targeted at actions that could still be best and fades as their posterior probability of optimality fades.
