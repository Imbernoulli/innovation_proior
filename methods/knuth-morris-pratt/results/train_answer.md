We want to find every position at which a nonempty pattern $W$ of length $m$ occurs as a contiguous substring of a long text $S$ of length $n$ — the operation a text editor performs constantly. The obvious method aligns the pattern at every text start, compares forward until a mismatch, and on failure resets to the first pattern character and advances the start by one. This costs $O(nm)$ in the worst case — the input $a^k b$ inside $a^N b$ matches a long run of $a$'s at every one of the $N$ alignments before failing on the final $b$, redoing nearly the whole pattern each time. Worse than the asymptotics, in practice, is that to re-try at the next start the scanner must *back the text pointer up* and re-read characters it has already passed; if the text streams in from a file or an editor buffer, backing up means holding those characters around and the buffering bookkeeping becomes genuinely painful. The structural waste is that a partial match is informative — after matching $abc$ and then hitting $x \ne$ the fourth pattern character, we know the last four text characters were exactly $a\,b\,c\,x$ — yet the naive scan throws that knowledge away, backs up, and rediscovers it. What we actually want is one forward pass over the text that *never moves the text pointer backward* and still finds every match; that single property removes both the $O(nm)$ re-reading and the buffering pain at once, and makes a real-time character-at-a-time reader possible. Thompson's regular-expression scanner reads without backup but pays an $O(mn)$ factor and does no precomputation exploiting the pattern's internal repetitions, which is exactly the cost we want to remove for the common case of a fixed literal pattern. The intuition that a partial match should tell us how far we may slide is felt but, by hand, there is no systematic rule converting "matched $j-1$ characters then failed" into "slide exactly this far, never looking back."

The route to that rule comes from an unexpected place: Cook's theorem (S. A. Cook, 1972) that any language recognized by a two-way deterministic pushdown automaton — a finite control, a stack, and a single read-only head that may move both directions — can be recognized on a random-access machine in $O(n)$ time, no matter how slow the pushdown machine is, even though such a machine may run for an exponential number of steps. The proof is *constructive*; it builds the linear procedure out of the given machine. Tracing that construction reveals its engine. The pushdown machine's full state includes its entire stack, and there are exponentially many possible stacks — the source of the exponential time — but what matters next is only the **surface configuration**, the triple of state, head position, and top-of-stack symbol, since buried symbols cannot affect the machine until they are exposed by popping. There are only $O(n)$ surface configurations. For each one $c$ at stack height $h$, the machine eventually first drops below $h$; the configuration at that moment is the **terminator** of $c$, and it depends only on $c$, not on what is buried beneath. Terminators compose — to find where $c$ pops below its height after a push, find the terminator of the post-push configuration and continue — so the whole forward simulation is a recursive computation of terminators that is **memoized** in a table indexed by surface configuration. Because there are $O(n)$ configurations and each terminator is computed once and looked up forever after, the total work is $O(n)$. The exponential blow-up was the machine re-deriving the same sub-computations as it shuffled its stack; sharing them through the table is precisely what kills the repetition.

I propose the Knuth–Morris–Pratt algorithm, which distills that memoization down to the string-matching automaton. Stripping away the pushdown scaffolding, the only useful residue of a failed comparison is a compact surface fact: the matched-prefix length $j$ — "I had matched $j-1$ pattern characters, then the next disagreed with the text." Everything needed to recover is a function of $j$ and the pattern alone; it does not depend on the unread text. So Cook's surface-configuration table collapses from "indexed by surface configuration" to "indexed by matched-prefix length $j$," a single small array $\mathrm{next}[\,j\,]$ telling us, after a mismatch at $\mathrm{pattern}[j]$, which pattern position to compare against $\mathrm{text}[k]$ next. The matcher keeps a text pointer $k$ and a pattern pointer $j$, the pattern aligned so $\mathrm{pattern}[1]$ sits at text position $k-j+1$. When $\mathrm{text}[k]=\mathrm{pattern}[j]$ it advances both; otherwise it sets $j := \mathrm{next}[\,j\,]$ and repeats until either the characters match or $j$ falls to $0$, meaning no prefix survives and the pattern slides entirely past while the text advances:
$$
\textbf{while } j>0 \text{ and } \mathrm{text}[k]\neq\mathrm{pattern}[j]\textbf{ do } j := \mathrm{next}[\,j\,]; \qquad k := k+1,\; j := j+1.
$$
The defining property falls straight out of this loop: $k$ is only ever incremented; recovery on mismatch happens *entirely* by shrinking $j$ through the table, so the text pointer never moves backward.

What must $\mathrm{next}[\,j\,]$ be for this to be correct? When about to slide, we know the last text characters up to and including $\mathrm{text}[k]=x$ are $\mathrm{pattern}[1]\dots\mathrm{pattern}[j-1]\,x$ with $x\neq\mathrm{pattern}[j]$. We want the smallest safe slide — the *largest* surviving prefix — i.e. the largest $i<j$ such that $\mathrm{pattern}[1..i-1]$ is both a prefix and a suffix of $\mathrm{pattern}[1..j-1]$, a **border**. Let $f[j]$ be the length-plus-one position of the longest proper border of $\mathrm{pattern}[1..j-1]$, with $f[1]=0$ by convention; a border of length $i-1$ means a shift of $j-i$, and taking the largest $i$ minimizes the slide and so never skips a possible match. There is one sharpening worth keeping. If the longest border resumes the comparison at $\mathrm{pattern}[i]$ but $\mathrm{pattern}[i]=\mathrm{pattern}[j]$ — the very character that just failed against $x$ — then comparing $x$ against $\mathrm{pattern}[i]$ is guaranteed to fail again, since $x\neq\mathrm{pattern}[j]=\mathrm{pattern}[i]$. So I demand $\mathrm{pattern}[i]\neq\mathrm{pattern}[j]$ in the definition, giving the refined table: $\mathrm{next}[1]=0$, and for $j>1$,
$$
\mathrm{next}[\,j\,]=\begin{cases}f[j] & \text{if } \mathrm{pattern}[j]\neq\mathrm{pattern}[f[j]],\\[2pt]\mathrm{next}[\,f[j]\,] & \text{if } \mathrm{pattern}[j]=\mathrm{pattern}[f[j]].\end{cases}
$$
The second line is right because when $\mathrm{pattern}[j]=\mathrm{pattern}[f[j]]$, resuming at $f[j]$ is the wasted comparison, so we take whatever a mismatch at $f[j]$ would have taken, namely $\mathrm{next}[f[j]]$, already computed. Equivalently $\mathrm{next}[\,j\,]$ is the largest $i<j$ with $\mathrm{pattern}[1..i-1]=\mathrm{pattern}[j-i+1..j-1]$ and $\mathrm{pattern}[i]\neq\mathrm{pattern}[j]$, or $0$.

The elegant turn is that computing $f$ is *the same matching problem applied to the pattern against itself*: $f[j]$ asks for the longest prefix of the pattern that matches a suffix of $\mathrm{pattern}[1..j-1]$ — two copies of the pattern sliding against each other, the borders being the overlaps. So preprocessing reuses the matcher with the text replaced by the pattern, carrying a candidate $t$ for the current border length, sliding $t := \mathrm{next}[t]$ on disagreement and incrementing it on agreement, folding the refinement in as it goes so $f$ never needs to be stored separately. This is $O(m)$: $t$ increases by exactly $1$ a total of $m-1$ times and stays $\ge 0$, while $t := \mathrm{next}[t]$ strictly decreases it, so the sliding can fire at most $m-1$ times overall. The identical amortized argument bounds matching — the text pointer advances $\le n$ times, $j$ is increased by $1$ only when $k$ advances, and each strict decrease $j := \mathrm{next}[\,j\,]$ is charged to a prior increase, so it fires $\le n$ times total. Matching is $O(n)$, preprocessing $O(m)$, the total $O(m+n)$, the text pointer never backs up, and only the pattern and its $O(m)$ table live in memory while the text streams past — alphabet-independent throughout. Correctness rests on the invariant, with $p=k-j$, that $\mathrm{text}[p+i]=\mathrm{pattern}[i]$ for $1\le i<j$ and no full match begins left of $p$: on a match $k$ and $j$ advance with $p$ fixed, extending the matched prefix; on a slide the new start $p'=k-\mathrm{next}[\,j\,]$ lies strictly right of $p$, and every start skipped between them is ruled out either because its overlap is not a border of $\mathrm{pattern}[1..j-1]$ (so it disagrees with an already-matched text character) or because it is a border whose next pattern character equals $\mathrm{pattern}[j]$ (so it disagrees with $\mathrm{text}[k]$ since $\mathrm{text}[k]\neq\mathrm{pattern}[j]$), so no match is ever skipped.

The refinement matters beyond mere cleanliness for the *per-character delay*, the work between two consecutive single-character text inputs — the figure of merit for a real-time reader. With the plain $f$, the pattern $a^m$ against $a^{m-1}b\dots$ makes one end mismatch cascade $j$ through $m, m-1, \dots, 1$, redundantly re-comparing the same text character $m-1$ times before $k$ advances; the refined $\mathrm{next}$ short-circuits the entire chain to one step because $\mathrm{pattern}[i]=\mathrm{pattern}[j]$ all along it. The worst case for the refined version is sharp: consecutive slides correspond to nested borders, i.e. to a prefix carrying several periods at once, which the Fine–Wilf phenomenon constrains, and the string packing the most nested distinct periods into the least length is the **Fibonacci string** $b_1=b,\ b_2=a,\ b_n=b_{n-1}b_{n-2}$ ($ab, aba, abaab, abaababa, \dots$, with $|b_n|=F_n$). Its near-commutative self-overlap makes the $\mathrm{next}$ values form a long descending chain through successive Fibonacci numbers, and since $F_k\approx\varphi^k/\sqrt5$ with $\varphi=(1+\sqrt5)/2$ the golden ratio, the number of consecutive $j := \mathrm{next}[\,j\,]$ steps while one text character is scanned is $\Theta(\log_\varphi m)$ — at most $1+\log_\varphi m$. So between two successive inputs only $O(\log m)$ work elapses and a real-time reader is fine, a bound the unrefined $f$ does not achieve.

For the executable form I switch to $0$-based indexing and fold the "slide fully past, advance the text" case into a sentinel: the table holds $-1$ at the front, so a mismatch driving the pattern pointer below $0$ signals "advance the text pointer and reset" with no special-case branch, while the extra final slot $T[\,\mathrm{len}(W)\,]$ holds the restart border length after a complete match so overlapping matches are still found.

```python
def preprocess(W):
    # Failure / "next" table from the pattern alone (O(m)).
    # For k < len(W), T[k] resumes after a mismatch at W[k] without
    # moving the text pointer. T[len(W)] restarts after a full match.
    # T[0] = -1 is the sentinel for "no prefix survives."
    T = [0] * (len(W) + 1)
    T[0] = -1
    pos, cnd = 1, 0          # cnd = current border length (the f[j] role)
    while pos < len(W):
        if W[pos] == W[cnd]:
            T[pos] = T[cnd]   # refinement: resuming at cnd would re-mismatch
        else:
            T[pos] = cnd
            while cnd >= 0 and W[pos] != W[cnd]:
                cnd = T[cnd]  # slide the pattern against itself
        pos += 1
        cnd += 1
    T[pos] = cnd
    return T


def search(S, W):
    T = preprocess(W)
    matches = []
    k = 0   # text pointer — only ever advances
    j = 0   # pattern pointer
    while k < len(S):
        if W[j] == S[k]:
            k += 1
            j += 1
            if j == len(W):
                matches.append(k - j)   # match starts at k - j
                j = T[j]
        else:
            j = T[j]                    # slide pattern, text pointer fixed
            if j < 0:
                k += 1
                j += 1
    return matches
```
