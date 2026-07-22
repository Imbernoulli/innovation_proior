I need the treated group's post-period outcome in the world where the treatment never arrives, but that object is missing. I can observe the treated group after treatment and I can observe an untreated group over the same dates, so I ask what each simple comparison buys me and what it leaves behind.

If I compare the two groups only after treatment, I mix the treatment effect with whatever fixed difference already separates the groups. That fixed difference can be social, geographic, industrial, epidemiological, or institutional; its exact source does not matter for the algebra. A later gap cannot by itself tell me whether the policy caused the gap or merely revealed an older level difference.

If I compare the treated group to itself before and after treatment, I remove that fixed level difference because the group is serving as its own baseline. But I now mix the treatment effect with time movement. Wages, disease, demand, measurement, or macro conditions may move between the two dates even if the policy does nothing. A before-after change is therefore not the causal effect unless the untreated time movement is zero, and that is rarely the right assumption.

So each single contrast carries one nuisance I cannot get rid of: the cross-group contrast carries a level, the before-after contrast carries a trend. The thing I want to try is to use the comparison group not as a substitute level but as a substitute movement. Let me write down the four group-by-period means and actually push the subtraction through, because I want to see which nuisances cancel and which survive rather than just hope they do. Call the four cell means by period and group:

```text
m(0,0) = E(Y | G = 0, Post = 0)
m(0,1) = E(Y | G = 0, Post = 1)
m(1,0) = E(Y | G = 1, Post = 0)
m(1,1) = E(Y | G = 1, Post = 1).
```

The treated group's own change from pre to post is `m(1,1) - m(1,0)`. The comparison group's change over the same dates is `m(0,1) - m(0,0)`. The candidate estimator is the difference of those two differences:

```text
tau = [m(1,1) - m(1,0)] - [m(0,1) - m(0,0)].
```

To check what this actually contains, I impose a transparent model of the means and see what falls out. Suppose each cell mean is a fixed group level plus a common time shift plus, for the treated post cell only, the treatment effect: `m(g,t) = a + g*gamma + t*lambda + g*t*tau`. Then the four cells are `a`, `a + lambda`, `a + gamma`, `a + gamma + lambda + tau`. Substituting:

```text
treated change      = (a + gamma + lambda + tau) - (a + gamma) = lambda + tau
comparison change   = (a + lambda) - a                          = lambda
difference          = (lambda + tau) - lambda                   = tau.
```

The `gamma` never appears in either change — it was differenced away the moment each group was compared to itself, which is the level cancellation I wanted. The `lambda` appears in both changes identically and dies in the outer subtraction. What is left is exactly `tau`, and only `tau`. That is the computation, not an assertion: under a common additive time shift the double difference returns the treatment effect with both nuisances gone.

But that derivation assumed the time shift is the same `lambda` for both groups, and that is the assumption I should be nervous about, not the level difference. Let me stress it with a concrete numeric case where the assumption fails, to see how badly the estimator misbehaves. Suppose the true treatment effect is `tau = 0`, so the policy does nothing, but the treated group was on a steeper underlying path. Untreated, let the comparison group rise from 20 to 22 (a trend of +2) and let the treated group's *untreated* path rise from 30 to 34 (a trend of +4). Because `tau = 0`, the treated group's observed post value is also 34. Then:

```text
treated change    = 34 - 30 = 4
comparison change  = 22 - 20 = 2
tau_hat            = 4 - 2 = 2.
```

The estimator reports an effect of +2 when the truth is 0. So the design does not need equal levels — the treated group sat at 30 versus 20 and the level gap caused no problem at all — but it does need equal untreated trends; a +4-vs-+2 trend mismatch passed straight through into a spurious +2. And I can see the failure is not bounded: if the treated underlying trend had been +10, the same arithmetic would have reported +8. A trend mismatch large enough can even flip the sign of a real effect. That tells me precisely which belief the method rests on.

Writing that out as a general decomposition rather than one example: the observed treated change is the untreated treated trend plus the true effect, and the observed comparison change is the untreated comparison trend. So

```text
tau_hat = [ (treated untreated trend) + tau ] - (comparison untreated trend)
        = tau + [ (treated untreated trend) - (comparison untreated trend) ].
```
