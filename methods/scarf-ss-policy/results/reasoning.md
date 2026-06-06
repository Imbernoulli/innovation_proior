OK, let me set this up honestly from the pain. I run a warehouse for one product. At the start of each period I look at how much I have on hand, then I decide how much to reorder. The shipment shows up right away — fast supplier, no lead time to worry about yet. Then random demand $w$ hits. Whatever's left over costs me $h$ per unit to hold; whatever I'm short costs me $p$ per unit in penalty, and I backlog it (carry it as negative inventory into next period). And the order itself: $c$ per unit, plus — here's the thing that's bugging me — a flat charge $K$ every time I place an order at all, no matter how big or small. A truck has to roll, paperwork has to be filed, a line has to be changed over. Ordering one widget and ordering five hundred both eat that same $K$.

I want the rule that minimizes my total expected cost over a horizon of, say, $N$ periods. Not the cost for one instance — the *policy*: given my inventory $x$ at the start of period $k$, what's the order? And I don't want a giant lookup table that just says "at $x=37$ in period 4, order 12." I want to know if the optimal policy has a *shape* I can write down with a couple of numbers and explain to the floor.

Let me get the single period straight first, because everything is built on it. One shot: I order up to level $S$, demand $w$ arrives, I pay $h(S-w)^+ + p(w-S)^+$. Expected cost $\bar L(S)=E[h(S-w)^+ + p(w-S)^+]$. That's an expectation of a convex function of $S$ (each of $(S-w)^+$ and $(w-S)^+$ is convex in $S$), so $\bar L$ is convex. Where's the minimum? Differentiate: $\frac{d}{dS}E[h(S-w)^+ + p(w-S)^+] = h\,P(w<S) - p\,P(w>S) = h F(S) - p(1-F(S))$. Set to zero: $(h+p)F(S)=p$, so $F(S^*)=\frac{p}{p+h}$. The critical fractile. Stock more when shortages bite harder ($p$ big) or holding is cheap ($h$ small). Clean. This is the newsvendor, and it's the target the whole multi-period thing has to reduce to. But it has no fixed cost and no tomorrow.

Now chain the periods. The tool is the functional equation. Let $J_k^*(x)$ be the optimal expected cost from the start of period $k$ onward, given I walk in with inventory $x$. Decision: order $u\ge0$. The natural variable isn't $u$ but the post-order level $y=x+u$, with the constraint $y\ge x$ (I can't un-order). Immediate cost: the ordering charge plus the expected holding/shortage. Then I transition to $x_{k+1}=y-w$ and pay the optimal cost-to-go from there. So

$$J_k^*(x)=\min_{y\ge x}\Big\{K\,\delta(y-x)+c(y-x)+\bar L(y)+\gamma\,E_w[J_{k+1}^*(y-w)]\Big\},$$

where $\delta(z)=1$ if $z>0$ and $0$ otherwise, $\bar L(y)=E_w[L(y-w)]$ with $L(x)=p\,x^- + h\,x^+$, and $\gamma\in(0,1]$ discounts. Terminal: $J_N^*(x)=v_N(x)$, some convex nonnegative cost on what's left. The $-cx$ that should come from $c(y-x)$ — let me pull the $c$ apart. Group the $y$-dependent stuff: define

$$G_k(y)=cy+\bar L(y)+\gamma\,E_w[J_{k+1}^*(y-w)].$$

Then the term in braces is $K\delta(y-x)+G_k(y)-cx$, and

$$J_k^*(x)=\min_{y\ge x}\big\{K\,\delta(y-x)+G_k(y)\big\}-cx.$$

Good. The whole problem now lives in: minimize $G_k$ over $y\ge x$, with a $K$ toll the instant you move off $y=x$. Everything is about the shape of $G_k$.

Let me do the easy world first, $K=0$, and feel exactly why it's easy — because that tells me what I'll lose. Suppose $J_{k+1}^*$ is convex. Then $\bar L(y)=E_w[L(y-w)]$ is convex (expectation of convex), $E_w[J_{k+1}^*(y-w)]$ is convex (same reason), $cy$ is linear, so $G_k$ is convex. Is it coercive — does it blow up as $|y|\to\pm\infty$? As $y\to+\infty$ I'm holding a mountain, $\bar L\approx h\cdot y$, and $cy$ adds to that, and $c+h>0$, so up it goes. As $y\to-\infty$ I'm deep in backlog, $\bar L\approx p\cdot(-y)$, while $cy\to-\infty$ at rate $c$; net slope $\approx -p+c$... wait, I need $-p+c<0$, i.e. $p>c$, for it to go up. That's a real assumption and it makes sense: if the per-unit penalty for being short were *cheaper* than buying the unit, I'd just never order and eat backlog forever. Assume $p>c$. So $G_k$ convex and coercive. It has a single unconstrained minimizer $S_k$. Now minimize over $y\ge x$: if $x<S_k$, the constraint isn't binding at the optimum, go to $S_k$; if $x\ge S_k$, I'm already past the bottom and $G_k$ is nondecreasing to the right of $S_k$, so the best feasible $y$ is $y=x$ — don't order. That's a base-stock policy: one number $S_k$, order up to it iff below. And plug back: for $x\ge S_k$, $J_k^*(x)=G_k(x)-cx$; for $x<S_k$, $J_k^*(x)=G_k(S_k)-cx$, a constant minus $cx$. Stitched together, $J_k^*(x)=G_k(\max(x,S_k))-cx$, which is convex (a convex function with its left tail flattened to the chord-from-below at $S_k$, minus a linear term). Convex in, convex out. The induction closes, base-stock optimal at every stage. Beautiful, and entirely powered by one fact: *convexity is preserved by the operator.*

Now turn $K$ back on and watch it break. The trouble is the $K\delta(y-x)$ — the toll. For a given $x$, I'm comparing "stay at $x$, pay $G_k(x)$" against "jump to the best $y>x$, pay $K+G_k(y)$, best of which is $K+G_k(S_k)$ if $x<S_k$." So

$$J_k^*(x)+cx=\min\big\{\,G_k(x),\ K+G_k(S_k)\,\big\}\quad(\text{for }x\le S_k).$$

Picture it. Way out left (very low inventory) $G_k(x)$ is huge, way above $K+G_k(S_k)$, so the min picks the flat line $K+G_k(S_k)$ — I order. As $x$ climbs toward $S_k$, $G_k(x)$ falls; at some crossover level — call it $s_k$ — $G_k(s_k)=K+G_k(S_k)$, and for $x$ just above that, $G_k(x)<K+G_k(S_k)$, so I stop ordering and just sit at $x$. So $J_k^*(x)+cx$ traces the flat line $K+G_k(S_k)$ up to $s_k$, then *switches* to following $G_k(x)$. At $s_k$ the two pieces meet continuously — fine — but look at the *derivative*. To the left, flat (slope 0). To the right, it's $G_k'(x)$, and just to the right of $s_k$, where is $G_k$ heading? Down, toward its minimum at $S_k>s_k$. So $G_k'$ is *negative* there. The function $J_k^*+cx$ goes flat, then kinks and heads *downward* into the well at $S_k$. A convex function can't do that — can't go flat and then turn downward. So $J_k^*$ is **not convex**.

And it can be worse than a kink. If $G_k$ itself had been built with a jump from a previous stage, $J_k^*$ can have an outright downward discontinuity. Either way: the convexity-preservation engine that made $K=0$ trivial is dead at the very first backward step. I can't even start the induction.

Let me make sure I'm not missing an escape hatch. Is the optimal policy maybe still base-stock, just with the proof harder? No — and I can see why physically. The $K$ toll is exactly what makes me *not* want to top up to $S_k$ every period; that would mean paying $K$ constantly to nudge inventory by tiny amounts. The economical thing is to *let inventory drift down* without ordering, and only when it's low enough that the future savings clear the toll, place *one* big order. So I expect a *band*: a reorder point $s_k$ (don't order while $x\ge s_k$) and an order-up-to level $S_k$ (when you do order, go all the way to $S_k$). Two numbers, not one — that gap $S_k-s_k$ is the batch. That's the $(s,S)$ rule, and the picture above already produced exactly those two numbers: $s_k$ is the crossover, $S_k$ is the minimizer. So I *believe* the shape. What I'm missing is a property of $G_k$ — weaker than convexity, since convexity is false — that (a) forces this $(s,S)$ shape on the minimization, and (b) is *preserved by the operator* so the induction can run.

So what property does $G_k$ actually have? Let me stare at the failure. Convexity says: the chord between any two points lies above the graph. That failed because of the downward jump / the flat-then-down behavior — the graph can poke *above* the chord by a bit. How much? The whole structure of the problem says the offending gap is bounded by $K$: the jump in $J_k^*$ is at most $K$ (it's the toll), and the flat top sits exactly $K$ above the minimum. So maybe the right relaxation is: the graph must lie below the chord *raised by $K$ at its right end*.

Let me write that down and test it. Convexity is: for $y\le y'$, $0\le\theta\le1$,
$$f(\theta y+(1-\theta)y')\le \theta f(y)+(1-\theta)f(y').$$
Relax the right endpoint upward by $K$:
$$f(\theta y+(1-\theta)y')\ \le\ \theta f(y)+(1-\theta)\big(K+f(y')\big).$$
Call $f$ **$K$-convex** if this holds for all $y\le y'$ and all $\theta\in[0,1]$. With $K=0$ it's ordinary convexity. Geometrically: on $[y,y']$, $f$ must lie below the line segment from $(y,f(y))$ to $(y',f(y')+K)$ — the *right* end of the chord is lifted by $K$, the left end is not. So a $K$-convex function can wobble above ordinary convexity, can have downward jumps (bounded, as I'll see), can even have several local minima — but in a controlled way. This is exactly the slack I need: my $J_k^*+cx$ that went flat at $K+G_k(S_k)$ and then dipped to $G_k(S_k)$ — the dip below the flat is exactly $K$, so it should still sit under any $K$-lifted chord.

Let me sanity-check the "bounded downward jump" claim, because it's the intuitive content. Take $y<y'$ and let $y'\downarrow y$. The chord's left value is $f(y)$, right value $f(y')+K$; the point $\theta y+(1-\theta)y'$ at $\theta\to1$ approaches $y$ from the right. The inequality forces $\limsup_{z\downarrow y}f(z)\le f(y)+K$ in the relevant configuration — a jump at $y$ can drop the value but the graph just to the right can't sit more than... hmm, let me be careful and instead get a cleaner equivalent form that I can actually compute with.

Here's a more useful restatement. I want to compare three collinear points: a left point $y-b$, a middle point $y$, and a right point $y+a$, with $b>0$, $a\ge0$. The middle $y$ is the convex combination $\theta(y-b)+(1-\theta)(y+a)$ with... let me solve: $y=\theta(y-b)+(1-\theta)(y+a)$ gives $0=-\theta b+(1-\theta)a$, so $\theta=\frac{a}{a+b}$, $1-\theta=\frac{b}{a+b}$. Plug into the $K$-convexity inequality with left $=y-b$, right $=y+a$:
$$f(y)\le \frac{a}{a+b}f(y-b)+\frac{b}{a+b}\big(K+f(y+a)\big).$$
Multiply by $\frac{a+b}{b}$:
$$\frac{a+b}{b}f(y)\le \frac{a}{b}f(y-b)+K+f(y+a).$$
Rearrange, moving the $f(y)$ terms together: $\frac{a+b}{b}f(y)-\frac{a}{b}f(y-b)=f(y)+\frac{a}{b}\big(f(y)-f(y-b)\big)$. So
$$K+f(y+a)\ \ge\ f(y)+\frac{a}{b}\big[f(y)-f(y-b)\big],\qquad \forall\,y\in\mathbb R,\ a\ge0,\ b>0.$$
This is the working form of $K$-convexity — Scarf's inequality. Read it: $\frac{f(y)-f(y-b)}{b}$ is a backward slope at $y$; the right side extrapolates that slope forward by $a$; the inequality says $f(y+a)$ can't fall below that linear extrapolation by more than $K$. Ordinary convexity ($K=0$) is "the function lies above its own backward tangents"; $K$-convexity allows a $K$ of give. And if $f$ is differentiable, send $b\to0$ so the backward slope becomes $f'(y)$, rename $y\to x$, $y+a\to y$ (so $a=y-x\ge0$):
$$K+f(y)\ \ge\ f(x)+f'(x)(y-x),\qquad x\le y.$$
"The graph lies above its tangents, minus $K$." Nice and checkable.

Now I have to verify the two things I demanded. First: is $K$-convexity preserved by the operator $f\mapsto cy+\bar L+\gamma E_w[f(y-w)]$ and then $\min_{y\ge x}$-with-toll? Second: does $K$-convex + continuous + coercive *force* the $(s,S)$ shape on the minimization? Let me do preservation first, since if it fails none of this matters.

Building blocks of $G_k$. The term $cy$ is linear, hence $0$-convex. The term $\bar L(y)=E_w[L(y-w)]$ is convex, hence $0$-convex. The term $\gamma E_w[J_{k+1}^*(y-w)]$ — I'm assuming $J_{k+1}^*$ is $K$-convex (induction hypothesis); I need two closure facts.

Closure fact 1: positive combinations. If $f_1$ is $K_1$-convex and $f_2$ is $K_2$-convex and $\alpha,\beta>0$, is $\alpha f_1+\beta f_2$ $(\alpha K_1+\beta K_2)$-convex? Just add the defining inequalities. For each $i$, $f_i(\theta y+(1-\theta)y')\le \theta f_i(y)+(1-\theta)(K_i+f_i(y'))$. Multiply the first by $\alpha$, the second by $\beta$, add:
$$(\alpha f_1+\beta f_2)(\theta y+(1-\theta)y')\le \theta(\alpha f_1+\beta f_2)(y)+(1-\theta)\big[(\alpha K_1+\beta K_2)+(\alpha f_1+\beta f_2)(y')\big].$$
Yes. So combinations of $K$-convex things are $K$-convex with the $K$'s combining linearly. In particular $cy$ ($0$-convex) plus $\bar L$ ($0$-convex) plus $\gamma\times$(a $K$-convex thing) will be $\gamma K$-convex — I'll come back to the discount bookkeeping.

Closure fact 2: expectation over the demand shift. If $f$ is $K$-convex, is $g(y)=E_w[f(y-w)]$ also $K$-convex? Fix $w$ at one value: $y\mapsto f(y-w)$ is just a horizontal shift of $f$. Does shifting preserve $K$-convexity? Plug $y-w,\,y'-w$ into the definition for $f$: $f(\theta(y-w)+(1-\theta)(y'-w))\le \theta f(y-w)+(1-\theta)(K+f(y'-w))$, and the argument on the left is $(\theta y+(1-\theta)y')-w$. So $y\mapsto f(y-w)$ is $K$-convex with the *same* $K$ — translation doesn't touch it. Now average over $w$: the $K$-convexity inequality is linear in $f$, so taking $E_w$ of both sides (assuming $E_w|f(y-w)|<\infty$ so everything is finite) preserves it. Hence $g$ is $K$-convex. Good. (Same idea proved $\bar L$ convex above: $L$ is $0$-convex, shift, average.)

So with $J_{k+1}^*$ $K$-convex: $\gamma E_w[J_{k+1}^*(y-w)]$ is $\gamma K$-convex by facts 2 then 1; adding $cy$ and $\bar L$ (both $0$-convex) keeps it $\gamma K$-convex. For the toll to be exactly $K$ I want $G_k$ to be at least $K$-convex; since $\gamma\le1$, $\gamma K\le K$, and a $\gamma K$-convex function is automatically $K$-convex (lifting the right end of the chord by *more* only makes the inequality easier). So $G_k$ is $K$-convex. It's coercive (the $cy+\bar L$ tails dominate, same $c+h>0$, $p>c$ argument as before; the $K$-convex future term is bounded below appropriately since costs are nonnegative). And continuous, given $w$ bounded so $\bar L$ and the expectation of a continuous-enough $J_{k+1}^*$ are continuous. So: **$J_{k+1}^*$ $K$-convex $\Rightarrow$ $G_k$ $K$-convex, coercive, continuous.** Preservation through the "build $G_k$" half holds. Now I need the minimization to (a) be the $(s,S)$ rule and (b) hand back a $K$-convex $J_k^*$.

Let me extract the $(s,S)$ structure from "$f$ continuous, coercive, $K$-convex" abstractly, with $f=G_k$. Coercive + continuous $\Rightarrow$ a global minimizer exists; call it $S$ (if several, the structure will pin which). Define $s$ as the smallest $z\le S$ with $f(z)=f(S)+K$. Does such a $z$ exist and is it $\le S$? At $z=S$, $f(S)<f(S)+K$ (since $K>0$). As $z\to-\infty$, coercivity sends $f(z)\to+\infty>f(S)+K$. By continuity there's a crossing; take the smallest one, that's $s\le S$. I claim $(s,S)$ is the policy: order up to $S$ iff $x<s$.

I need four properties, and let me prove them, because the policy correctness rides on them.

(1) $S$ minimizes $f$ — by construction.

(2) $f(s)=f(S)+K$, and $f(y)>f(s)$ for all $y<s$. The equality is the definition of $s$. For the strict inequality: suppose some $y<s$ had $f(y)\le f(s)=f(S)+K$. There's a small lemma I keep needing, so let me prove it cleanly. *Lemma:* if $f$ is $K$-convex, $u<v$, and $f(u)=K+f(v)$, then $f(z)\le K+f(v)$ for all $z\in[u,v]$. Proof: write $z=\theta u+(1-\theta)v$; $K$-convexity gives $f(z)\le \theta f(u)+(1-\theta)(K+f(v))=\theta(K+f(v))+(1-\theta)(K+f(v))=K+f(v)$, using $f(u)=K+f(v)$. Done. The consequence I want: $f$ can cross the level $K+f(\cdot)$ "from above" at most in a controlled way; concretely, on $(-\infty,S)$ the equation $f(z)=f(S)+K$ can't have a solution with $f$ dipping *below* the level between it and $S$. Now back to the claim: if $y<s$ had $f(y)\le f(S)+K=f(s)$, then since $f(y)\ge f(S)$ (S is the min) and $f$ is continuous and large at $-\infty$, there'd be a point $y'\le y<s$ with $f(y')=f(S)+K$ as well — but the lemma applied with $u=y'$, $v=S$ would force $f\le f(S)+K$ on all of $[y',S]$, in particular at every level down to $s$, contradicting that $s$ is the *smallest* such crossing point (we'd have found a smaller one $y'<s$). So no such $y$ exists: $f(y)>f(s)$ for $y<s$. 

(3) $f$ is decreasing on $(-\infty,s)$. Take $y<y'\le s$. Suppose $f(y)<f(y')$ for contradiction (i.e. $f$ went up somewhere left of $s$). Apply the Scarf inequality with the backward slope at $y'$ being negative-or-using points... cleaner: use $K$-convexity on the triple $y< y' \le s< S$. By (2), every point left of $s$ has $f>f(s)=f(S)+K$. Now the $K$-convexity chord from $y$ to $S$ (lift right end by $K$): for $z=y'\in[y,S]$, $f(y')\le \theta f(y)+(1-\theta)(f(S)+K)$ where $\theta=\frac{S-y'}{S-y}\in(0,1)$. Since $f(S)+K=f(s)<f(y')$ (by (2), as $y'<s$ would give $f(y')>f(s)$... careful, $y'\le s$; take $y'<s$ strictly first), we get $f(y')\le \theta f(y)+(1-\theta)f(y') $ wait let me substitute $f(S)+K=f(s)$ and use $f(s)<f(y')$: $f(y')\le \theta f(y)+(1-\theta)f(s)<\theta f(y)+(1-\theta)f(y')$. So $f(y')-(1-\theta)f(y')<\theta f(y)$, i.e. $\theta f(y')<\theta f(y)$, i.e. $f(y')<f(y)$. That contradicts the assumption $f(y)<f(y')$. Hence $f(y)\ge f(y')$ for $y<y'<s$ — $f$ is nonincreasing (decreasing in the weak sense) on $(-\infty,s)$. So as $x$ falls below $s$, the cost-of-not-ordering $G_k(x)$ only rises — never a reason to wait.

(4) $f(y)\le f(y')+K$ for all $s\le y\le y'$. This is the "no profitable re-order once you're in the band" fact. Take $s\le y\le y'$. By $K$-convexity write $y=\theta s+(1-\theta)y'$? No, $y$ may exceed... if $y\in[s,y']$ then $y=\theta s+(1-\theta)y'$ for some $\theta\in[0,1]$. Hmm, I actually want to bound $f(y)$ by $f(y')+K$ directly. Use the Scarf inequality form $K+f(y'+a)\ge f(y')+\frac{a}{b}[f(y')-f(y'-b)]$... let me instead argue: for $y\ge s$, $f(y)\le f(S)+K$. Why: if $s\le y\le S$, apply the Lemma with $u=s,v=S$ ($f(s)=K+f(S)$): $f(y)\le K+f(S)$ on $[s,S]$. If $y\ge S$, then I use $K$-convexity on triple $s\le S\le y$: from the inequality, $K+f(y)\ge f(S)+\frac{y-S}{S-s}[f(S)-f(s)]=f(S)+\frac{y-S}{S-s}(-K)$, not directly what I want. Let me just take the cleaner target actually needed for the policy and verify it below rather than property (4) in full generality; (4) for the range $s\le y\le S$ is the Lemma, which is what the ordering decision uses.

Now translate to the *decision*. Starting at $x$, I choose between not ordering (cost $G_k(x)$) and ordering up to the best $y>x$ (cost $K+\min_{y>x}G_k(y)$). Two cases.

Case $x<s$: I want to show ordering up to $S$ beats not ordering, i.e. $K+G_k(S)<G_k(x)$. By (2), $G_k(x)>G_k(s)=K+G_k(S)$ for $x<s$. Exactly that. So order, and order to the global minimizer $S$ (nothing $>x$ is cheaper than $S$). 

Case $x\ge s$: I want to show not ordering beats any order. Any order goes to some $y>x\ge s$, costing $K+G_k(y)\ge K+G_k(S)$. Not ordering costs $G_k(x)$. So I need $G_k(x)\le K+G_k(y)$ for all $y>x$. Since $x\ge s$ and $y>x\ge s$, and using (4)/the Lemma on $[s,S]$ plus the right-tail behavior: for $x\ge s$, $G_k(x)\le G_k(y)+K$ holds — concretely, if $x,y\in[s,S]$ both lie under the level $K+G_k(S)$, and to the right of $S$, $K$-convexity gives $G_k(y)+K\ge G_k(x)$ because the chord from $x$ (lifted... ). Let me just nail the right tail: for $S\le x\le y$, $K$-convexity (Scarf form, backward slope at $x$ from the left, which is $\ge0$ near/after the min) gives $K+G_k(y)\ge G_k(x)+\frac{y-x}{b}[G_k(x)-G_k(x-b)]\ge G_k(x)$ once the bracketed backward slope is $\ge0$, which it is to the right of the minimizer. And for $s\le x\le S\le y$: $G_k(x)\le K+G_k(S)\le K+G_k(y)$ (first by the Lemma, second since $S$ is the min). And for $s\le x\le y\le S$: $G_k(x)\le K+G_k(S)\le K+G_k(y)$? Need $G_k(y)\le G_k(S)$... no, $G_k(y)\ge G_k(S)$, so $K+G_k(y)\ge K+G_k(S)\ge G_k(x)$. Good in all subcases. So for $x\ge s$, no order beats every order: don't order. 

That's the $(s,S)$ policy, proven from "continuous coercive $K$-convex": don't order while $x\ge s_k$; when $x<s_k$, order up to $S_k$. And the two numbers are read straight off $G_k$: $S_k=\arg\min G_k$, and $s_k=$ the smallest $z\le S_k$ with $G_k(z)=G_k(S_k)+K$. When $K=0$, the equation $G_k(z)=G_k(S_k)$ has $z=S_k$ as its solution near the min, so $s_k=S_k$ and the band collapses — base-stock falls out as the special case. The whole structure degrades gracefully to the $K=0$ world. That's the tell I had the right generalization.

Last piece: close the induction by showing the *output* $J_k^*$ is again $K$-convex (and nonnegative, continuous), so the next step back can reuse everything. From the policy I just derived, plug the optimal action back in. Define $\tilde G_k(x):=J_k^*(x)+cx$ (strip the linear part; $K$-convexity of $J_k^*$ is equivalent to $K$-convexity of $\tilde G_k$ since adding the $0$-convex $cx$ doesn't change the class). For $x<s_k$ I order to $S_k$: $J_k^*(x)=K+G_k(S_k)-cx$, so $\tilde G_k(x)=K+G_k(S_k)$, a constant. For $x\ge s_k$ I don't order: $J_k^*(x)=G_k(x)-cx$, so $\tilde G_k(x)=G_k(x)$. So

$$\tilde G_k(x)=\begin{cases} K+G_k(S_k)=G_k(s_k), & x\le s_k,\\[2pt] G_k(x), & x\ge s_k.\end{cases}$$

Continuous at $s_k$ since $G_k(s_k)=K+G_k(S_k)$ by definition of $s_k$. Nonnegative since it's a cost-to-go of nonnegative costs. Now $K$-convexity of $\tilde G_k$, straight from Definition: take $y\le y'$, check $\tilde G_k(z)\le$ the $K$-lifted chord from $(y,\tilde G_k(y))$ to $(y',\tilde G_k(y'))$ for $z\in[y,y']$. Three cases on where $s_k$ sits.

Both right, $s_k\le y\le y'$: here $\tilde G_k\equiv G_k$ and $G_k$ is $K$-convex, done.

Both left, $y\le y'\le s_k$: $\tilde G_k$ is constant $=G_k(s_k)$, and a constant is $0$-convex hence $K$-convex, done.

The interesting one, $y<s_k<y'$. On $[y,y']$ I must keep $\tilde G_k(z)$ below the segment $\ell(z)$ from $(y,\,\tilde G_k(y))=(y,\,G_k(s_k))$ to $(y',\,\tilde G_k(y')+K)=(y',\,G_k(y')+K)$. First, is this segment increasing? Its left height is $G_k(s_k)=K+G_k(S_k)$, its right height is $K+G_k(y')$, and $G_k(y')\ge G_k(S_k)$ since $S_k$ is the global min — so right $\ge$ left, the segment is nondecreasing. Now split $[y,y']$ at $s_k$. For $z\in[y,s_k]$: $\tilde G_k(z)=G_k(s_k)$ (the flat part), and $\ell$ is nondecreasing starting at $\ell(y)=G_k(s_k)$, so $\ell(z)\ge G_k(s_k)=\tilde G_k(z)$. Good. For $z\in[s_k,y']$: $\tilde G_k(z)=G_k(z)$. By $K$-convexity of $G_k$ on $[s_k,y']$, $G_k(z)$ lies below the $K$-lifted chord from $(s_k,G_k(s_k))$ to $(y',G_k(y')+K)$ — call it $\ell'(z)$. And $\ell'$ vs $\ell$: both end at the same right point $(y',G_k(y')+K)$; at the left, $\ell$ passes through $(y,G_k(s_k))$ and $\ell'$ through $(s_k,G_k(s_k))$ — same height $G_k(s_k)$ but $\ell'$'s left anchor is at $s_k\ge y$, i.e. *inside*. Two lines sharing the right endpoint, with $\ell'$'s left anchor at the same height but further right on an increasing line: $\ell'$ is the steeper one over $[s_k,y']$, and since they meet at the right and $\ell$ is at height $G_k(s_k)$ already at $y\le s_k$ (so $\ell\le G_k(s_k)$... no — $\ell$ is increasing, $\ell(s_k)\ge \ell(y)=G_k(s_k)=\ell'(s_k)$). So $\ell(s_k)\ge \ell'(s_k)$ and $\ell(y')=\ell'(y')$; both linear, hence $\ell(z)\ge \ell'(z)$ on $[s_k,y']$. Therefore $\tilde G_k(z)=G_k(z)\le \ell'(z)\le \ell(z)$. Below the chord, as required. All three cases check, so $\tilde G_k$ — and thus $J_k^*$ — is $K$-convex. Nonnegative, continuous, $K$-convex: exactly the induction hypothesis for stage $k-1$.

Base case: $J_N^*=v_N$, convex (= $0$-convex) and nonnegative. Induction runs from $N$ down to $1$. So at *every* stage $G_k$ is $K$-convex/coercive/continuous, an $(s_k,S_k)$ policy is optimal, and the value function stays $K$-convex. The fixed cost broke ordinary convexity; $K$-convexity is precisely the dented version that survives the min-plus-expectation operator and still pins the policy to two numbers. That's the whole thing.

Now land it on code. A discretized backward DP on the inventory grid. The one efficiency move worth making: in $\theta_t(x)=\min_{y\ge x}\{K\delta(y-x)+c(y-x)+\bar L(y)+\gamma E[\theta_{t+1}(y-w)]\}$, the part $H_t(y):=\bar L(y)+\gamma E[\theta_{t+1}(y-w)]$ doesn't depend on $x$, so precompute $H_t(y)$ once per $(t,y)$ and reuse it across all $x$ in the inner min — that's where the cost is (the expectation over demand). Then for each $x$, scan $y\ge x$. Read off $S_t$ as the order-up-to level at the smallest $x$ in range (everyone low enough orders to the same $S_t$), and $s_t$ as the largest $x$ still ordering up to $S_t$ — equivalently the crossover where not-ordering takes over. When $K=0$ these coincide.

```python
import numpy as np
from scipy.stats import norm

def finite_horizon_dp(num_periods, holding_cost, stockout_cost,
                      terminal_holding_cost, terminal_stockout_cost,
                      purchase_cost, fixed_cost,
                      demand_mean, demand_sd,
                      discount_factor=1.0, initial_inventory_level=0.0,
                      d_spread=4, s_spread=5):
    """Backward DP for the finite-horizon (s,S) inventory problem.

    theta_t(x) = min_{y>=x} { K*1[y>x] + c*(y-x) + Lbar(y) + gamma*E[theta_{t+1}(y-D)] }.
    K-convexity of theta_t makes the minimizer an (s,S) rule; we read s_t, S_t off it.
    """
    T = num_periods
    def lst(v):  # broadcast a scalar to a 1-indexed per-period list
        return [None] + [v]*T if np.isscalar(v) else v
    h, p, c, K = lst(holding_cost), lst(stockout_cost), lst(purchase_cost), lst(fixed_cost)
    mu, sd, g = lst(demand_mean), lst(demand_sd), lst(discount_factor)

    # Demand support: truncate at mu +/- d_spread*sigma, clipped at 0.
    d_min = int(max(0, round(min(mu[1:]) - d_spread*max(sd[1:]))))
    d_max = int(round(max(mu[1:]) + d_spread*max(sd[1:])))
    d_range = np.arange(d_min, d_max+1)

    # Inventory grid: around the newsvendor s and s + EOQ-with-backorders for S.
    nv = [None] + [mu[t] for t in range(1, T+1)]                 # crude s estimate
    Q  = [None] + [np.sqrt(2*K[t]*mu[t]/max(h[t], 1e-9)) for t in range(1, T+1)]  # batch scale
    x_min = int(round(min(nv[1:]) - max(mu[1:]) - max(sd[1:])*(s_spread+d_spread)))
    x_max = int(round(max(nv[1:]) + max(Q[1:]) + max(sd[1:])*s_spread))
    x_range = np.arange(x_min, x_max+1)
    nx = len(x_range)

    reorder_points = [0]*(T+1)        # s_t
    order_up_to_levels = [0]*(T+1)    # S_t
    theta = np.zeros((T+2, nx))       # theta[t, x-x_min] = cost-to-go

    # Terminal cost theta_{T+1}(x) = h_T x^+ + p_T x^-.
    theta[T+1, :] = terminal_holding_cost*np.maximum(x_range, 0) \
                  + terminal_stockout_cost*np.maximum(-x_range, 0)

    for t in range(T, 0, -1):
        # Demand pmf on the truncated support (normal, integer-binned).
        prob = norm.cdf(d_range+0.5, mu[t], sd[t]) - norm.cdf(d_range-0.5, mu[t], sd[t])

        # Precompute H_t(y) = Lbar(y) + gamma * E[theta_{t+1}(y - D)] for every y on the grid.
        H = np.zeros(nx)
        for j, y in enumerate(x_range):
            hold  = np.dot(prob, np.maximum(y - d_range, 0))
            short = np.dot(prob, np.maximum(d_range - y, 0))
            Lbar = h[t]*hold + p[t]*short                       # newsvendor expected cost
            d_eff = np.clip(d_range, y - x_max, y - x_min)        # keep y-d on the grid
            future = g[t]*np.dot(prob, theta[t+1, (y - d_eff) - x_min])
            H[j] = Lbar + future

        # For each starting x, minimize over post-order y >= x with the fixed-cost toll.
        oul = np.zeros(nx)
        for i, x in enumerate(x_range):
            best_cost, best_y = np.inf, x
            for j in range(i, nx):                               # y ranges over x..x_max
                y = x_range[j]
                order = (K[t] + c[t]*(y - x)) if y > x else 0.0   # K only when y>x
                cost = order + H[j]
                if cost < best_cost:
                    best_cost, best_y = cost, y
            theta[t, i] = best_cost
            oul[i] = best_y

        # Read (s_t, S_t) off the action profile: S_t = OUL at the lowest x;
        # s_t = largest x that still orders up to S_t (the crossover).
        S_t = oul[0]
        order_up_to_levels[t] = S_t
        s_idx = 0
        while s_idx+1 < nx and oul[s_idx+1] == S_t:
            s_idx += 1
        reorder_points[t] = x_range[s_idx]                       # == S_t when K=0

    total_cost = theta[1, int(round(initial_inventory_level)) - x_min]
    return reorder_points, order_up_to_levels, total_cost
```

So the causal chain, top to bottom: the fixed charge $K$ makes topping-up-every-period wasteful, so I expect batching — wait, then order one lump — which is a two-number $(s,S)$ band, not a one-number base stock. The Bellman recursion reduces each stage to minimizing $G_k(y)$ over $y\ge x$ with a $K$ toll on moving. Without $K$, $G_k$ is convex and the operator preserves convexity, giving base stock. With $K$ the cost-to-go grows a bounded downward dent and convexity dies. The dent is exactly bounded by $K$, which is the chord-lifted-by-$K$ relaxation: $K$-convexity. It's closed under positive combination and under the demand-shift expectation, so the operator preserves it; continuous-coercive-$K$-convex forces the minimization to be order-up-to-$S$-iff-below-$s$; and plugging that policy back yields a $K$-convex value function again, so backward induction runs the whole horizon. The two numbers drop straight out of $G_k$ — $S_k$ its minimizer, $s_k$ the smallest level where $G_k$ exceeds the minimum by $K$ — and collapse to one when $K=0$.
