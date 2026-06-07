Right, let me get this sub-problem clear before I write any code. The translation program reads a Russian sentence and has to look each word up in the dictionary. The dictionary is this enormous thing on magnetic tape, in alphabetical order, A to Z, and it takes minutes just to wind the tape from one end to the other. Tape is hopeless for jumping around — if I look up a word near Z and the next word starts with B, I have to rewind, and I've thrown away minutes. So the only sane way to look up a whole sentence is to read the tape once, straight through, and catch every word as I pass it. But that only works if the words of the sentence are already in alphabetical order when I start the pass. So the thing I actually have to program, sitting here in fast memory, is: take the handful of words in this sentence and put them in order, quickly. Everything else about the translation hangs off this little in-memory sort. Fine. I know Mercury Autocode well enough; I'll just think up a sort and write it.

The obvious one comes to me first. Go through the list, and whenever two neighbours are in the wrong order, swap them; keep passing over the list until no swaps happen. It plainly works — every pass drags the biggest remaining word up to its end, like a bubble rising. But let me think about how much work that is before I commit to it, because it feels wasteful and I want to know why. The trouble is that a swap only ever moves a word past its immediate neighbour, one position. So if a word belongs right at the far end but starts near the front, it can only crawl one slot per pass, and it needs something like N passes to get there — and each pass walks the whole list of N words. That's on the order of N times N comparisons and swaps. For a short sentence that's tolerable, but it offends me: most of those comparisons are between two words that are already roughly in the right order relative to each other, and comparing them teaches me nothing. I'm spending nearly all my effort confirming things I already half-knew. The one-slot-at-a-time movement is the disease, and the adjacent-swap structure is the cause. I reject it. There ought to be something that moves a word a long way in one go.

So what would let a word jump? The crawl happens because every comparison is local — neighbour against neighbour. What if a comparison were against something *global* instead, something that tells me not "are these two in order" but "which broad region of the final array does this word belong in"? Suppose I pick one word out of the list — any one — and call its key the dividing value; let me call it the bound. Now I make one comparison per word, every other word against this one bound. A word whose key is below the bound belongs somewhere in the lower part of the final order; a word above the bound belongs in the upper part. That single comparison has placed the word on the correct *side* — not its exact slot, but the right half — and it didn't matter how far that word had to travel to get to its side. The movement is no longer one-slot-at-a-time; a word can leap across the whole array to the correct side in one shot.

Let me see what I've got after I've done that to every word. I've rearranged the list so that there is a dividing line, every word below it has a key at or below the bound, and every word above it has a key at or above the bound. I have not necessarily put the bound-word into a single final slot yet; the useful fact is weaker and exactly enough. The two sides never have to interact again. Nothing below the line will ever need to pass above it, or vice versa, because every cross-line comparison is already settled by the bound. So I can sort the lower group entirely on its own, and sort the upper group entirely on its own, and when both are sorted internally, the whole list is sorted. The hard problem of sorting N words has dissolved into: one linear sweep to partition, plus two completely independent smaller sorting problems. That independence is the whole prize — it's exactly what bubble sort never has, where every part of the array can still affect every other part right up to the end.

And the two smaller problems are the *same kind of problem*. So I don't need a different idea for them — I do the same thing again. Pick a bound inside the left group, partition it into its own lower and upper pieces; same for the right group; and keep going. Each time, the pieces get smaller. When a piece is down to one word, or none, there's nothing to do — a single word is trivially in order — and that piece is finished. The recursion bottoms out there.

Let me convince myself this is genuinely faster and not just bubble sort in disguise. One partition of a group of size n costs about n comparisons — one per word against the bound. If the bound roughly halves the group each time, then I partition the whole array (cost ~N), then two halves (cost ~N total across them), then four quarters (~N total), and so on. The sizes halve each level, so there are about log2(N) levels, and each level costs about N across all its pieces. That's about N log2(N) work, instead of N squared. For a list of any real size that's a colossal difference, and it's close to the least any sorting-by-comparison could possibly cost — to tell apart the N! possible orderings with yes/no comparisons you need at least about log2(N!), which is about N log2(N) comparisons, and I'm in that neighbourhood. The leap I wanted — a word crossing the array in one move — is precisely what buys the drop from N² to N log N: a single comparison against the bound does the work that a whole sweep of adjacent swaps was doing.

Now I have to actually perform the partition in the store, in place, because I haven't got room to copy the list somewhere else — the whole point is to fill the store with words to be sorted. I don't want to build the lower and upper groups in fresh arrays. Can I rearrange the words *within* their own stretch of store so the line falls out? Picture two pointers, one at the bottom of the stretch, one at the top. The lower pointer walks upward. As long as the word it's looking at has a key at or below the bound, it's on the correct side — leave it, step up. It only stops when it finds a word with key *above* the bound: that word is on the wrong side, it belongs up top. Now start the upper pointer walking down from the top. As long as its word has a key at or above the bound, that's fine, step down; it stops when it finds a word with key *below* the bound — also on the wrong side, belongs down low. So now I'm holding two words, each sitting in the other's territory. Exchange them. One swap, and both are now on their correct sides. Step both pointers inward and carry on — lower walks up again, upper walks down again — exchanging each offending pair as they're found. I keep going until the two pointers meet and cross. When they cross, every word below the crossing point is at or below the bound and every word above it is at or above the bound: the dividing line is exactly where they crossed, and the partition is done. One sweep, in place, only exchanges — no second array. That's exactly what I needed.

Let me stress-test the partition on the nasty cases before I trust it, because a sort that loops forever is worse than a slow one. What if the bound I picked happens to be the very largest key in the group? Then every word is at or below it, the lower pointer never finds a word above the bound, and it runs off the top of the group. Or the bound is the smallest, and the upper pointer runs off the bottom. Or every key is equal. In those cases the "dividing line" wants to sit at the very edge — and then one of my two resulting pieces is the *entire* group again, unchanged. If I recurse on a piece that's the same size as what I started with, I never shrink, and I loop forever. That's a real wall and I have to close it.

The fix has to guarantee that every partition makes genuine progress: no recursive call is allowed to receive the same whole segment back again. The word I took the bound from is the natural anchor, and the stopping-on-strict-inequality rules help me keep track of it. The lower scan passes over words whose keys are less than or equal to the bound and stops only at a key greater than the bound. The upper scan passes over words whose keys are greater than or equal to the bound and stops only at a key less than the bound. So the bound-word itself, having key exactly equal to the bound, is not one of the two offending words exchanged during the inward scans; it stays where it was until the crossing is known.

Now suppose the pointers have crossed. If the bound-word lies in the lower resulting segment, then within that lower segment it is as large as any key needs to be; I can exchange it with the highest-addressed word of that lower segment and then shorten the lower segment by one. If the bound-word lies in the upper resulting segment, it is correspondingly as small as the upper side needs; I exchange it with the lowest-addressed word of the upper segment and shorten the upper segment by one. If the crossing has already left it between the two recursive ranges, there is nothing to do. After this adjustment, neither returned range can be the original segment; in the equality edge cases the ranges that remain are already singletons. The degenerate "partition the same whole group again" case is gone, and the recursion is forced to bottom out.

Which word should I take the bound from? Any word in the group has a key that's in range, so any choice is correct. But the *speed* depends on it: if the bound keeps landing near the largest or smallest key, the split is lopsided — one piece nearly the whole group — and the level count creeps back toward N instead of log N, dragging me toward N² in the worst case. A bound near the middle of the values gives the balanced halving I costed out above. I can't know the median without work, but I don't need to: if I pick the bound-word at random from the group, then on random data the split is balanced on average, and even on data that arrives already sorted — which would murder a fixed "always take the first word" choice — the randomness keeps me honest. So: choose the bound-word at random. (Later I can sharpen this by taking the median of a small sample, if the extra fuss pays for itself, but random is enough to get the average behaviour right.)

So I can write the partition. I sit down and code it in Mercury Autocode — pick a word for the bound, run the two pointers in, exchange the offending pairs, and make the little final adjustment that keeps the next ranges from repeating the same segment. That part comes out cleanly; it's just a loop with two indices and an exchange. And then I hit the second wall, and this one I cannot get around.

The algorithm doesn't end when one partition finishes. After I partition the whole array I have two pieces still to sort, and after I partition one of *those* I have two more, and so on. At any moment there's a whole collection of pieces that are partitioned-but-not-yet-sorted, waiting their turn, and I have to remember all of them — the bottom and top address of each — and keep coming back to them. There's a natural order to it: when I split a piece I should dive into one of its halves immediately, and the other half waits; and when I finish something, I should return to the *most recently* set-aside piece, the way you'd resume the innermost interrupted task before the outer ones. It's a last-in, first-out list of pending pieces. In a language that let a routine call itself, this would administer itself — diving into a sub-piece would just be the routine invoking itself on that sub-piece, and the machinery underneath would remember where to come back to. But Mercury Autocode has nothing of the sort. A routine can't call itself; there's no stack handed to me; if I want this list of pending pieces I have to build it by hand in a fixed block of store and manage every index of it myself — push a piece's two addresses on when I postpone it, pop them off when I come back, and never lose track. I try to write that bookkeeping and it turns into a mess. Getting the partition right was easy; administering the open-ended list of postponed segments by hand defeats me. I can't finish the program. I set it aside. The idea is plainly good — I'm convinced of the partition and the recursive structure and the cost — but I cannot express it on this machine.

It sits unfinished for a while. Then, going through the report on this new language ALGOL, I come across recursive procedures — a procedure that is allowed to call itself. And there it is: the exact thing I couldn't build by hand. If a sorting procedure can invoke *itself* on a sub-piece, then the list of postponed pieces I was struggling to administer is kept for me, automatically, by the language — the chain of suspended self-calls *is* that last-in-first-out list, each call remembering its own piece and where to resume. The whole problem that defeated me in Mercury Autocode — the management of the nest of pending segments — simply vanishes; it's done behind the scenes. Now I can write the thing the way I always saw it: to sort a piece, if it has more than one word, partition it and then sort the lower piece and sort the upper piece — each "sort" a call to the same procedure on a smaller piece, the recursion bottoming out when a piece has one word or none. It comes out in a few lines.

```python
import random

def key(item):
    return item                        # the sort key (for words, the spelling)

def exchange(a, p, q):
    a[p], a[q] = a[q], a[p]            # the only data-movement primitive

def partition(a, lo, hi):
    f = random.randint(lo, hi)         # choose an actual item, so the bound is in range
    bound = key(a[f])
    i = lo                             # lower pointer, walks up
    j = hi                             # upper pointer, walks down

    while True:
        while i < hi and key(a[i]) <= bound:
            i += 1                     # stop at the first key greater than the bound
        while j > lo and key(a[j]) >= bound:
            j -= 1                     # stop at the first key less than the bound

        if i < j:
            exchange(a, i, j)          # two words are in each other's territory
            i += 1
            j -= 1
            continue

        # The line is between j and i.  If the bound-word is still inside
        # one recursive side, move it to that side's edge and exclude it.
        if i < f:                      # bound-word lies in the upper segment
            exchange(a, i, f)
            i += 1
        elif f < j:                    # bound-word lies in the lower segment
            exchange(a, f, j)
            j -= 1
        return i, j                    # recurse on a[lo..j] and a[i..hi]

def sort(a, lo=0, hi=None):
    if hi is None:
        hi = len(a) - 1
    if lo < hi:                        # a piece of 0 or 1 words is already sorted
        i, j = partition(a, lo, hi)
        sort(a, lo, j)                 # lower piece, sorted independently
        sort(a, i, hi)                 # upper piece, sorted independently
    return a
```

That's the version recursion lets me write so elegantly that the hard part is invisible. But I should not forget what was actually hard, because not every machine I'll meet has recursion, and I want to know exactly what the recursion is doing for me. It is keeping that last-in-first-out list of postponed pieces. So let me also write it the way I'd have had to on Mercury Autocode — managing the list myself — both to prove I now understand the thing that beat me, and because it's the honest core of the method. I'll call the list a nest: a block of store where I push a piece's bounds when I postpone it and pop the most recently pushed piece when I'm ready for it.

There's one more thing I want from the by-hand version that the recursive one hides. If I always postpone one piece and dive into the other, how big can the nest get? In the worst case — every split lopsided — the chain of postponed pieces could be as long as the number of words, and I'd have to reserve that much store for the nest, which is exactly the kind of waste I was trying to avoid. But I get to choose *which* piece to postpone. If I always dive into the *smaller* piece and postpone the *larger*, then each time I push something onto the nest, the piece I keep working on is at most half of what I had — so the depth of postponed pieces can't exceed about log2(N). That caps the nest at a size I can know and reserve in advance. So:

```python
def sort_with_nest(a):
    lo, hi = 0, len(a) - 1
    nest = []                          # pushdown list of postponed pieces

    def segment_size(segment):
        first, last = segment
        return max(0, last - first + 1)

    while True:
        while lo < hi:
            i, j = partition(a, lo, hi)
            left = (lo, j)
            right = (i, hi)
            left_size = segment_size(left)
            right_size = segment_size(right)

            # Continue on the smaller piece and postpone the larger one;
            # then the current piece at least halves before another push.
            if left_size < right_size:
                if right_size > 1:
                    nest.append(right)
                lo, hi = left
            else:
                if left_size > 1:
                    nest.append(left)
                lo, hi = right

        if not nest:
            break
        lo, hi = nest.pop()            # resume the most recently postponed piece
    return a
```

The causal chain, end to end: I need the words of a sentence in alphabetical order so I can look them all up in one pass of a tape dictionary; the obvious neighbour-swapping sort is N^2 because a misplaced word can only crawl one slot at a time; so instead I compare every word against a single chosen bound, which sends each word to its correct side of a dividing line in one pass; the two sides are then independent and sorted the same way recursively, giving N log N when the splits are balanced; I make the bound random when the data cannot be trusted to supply random-looking bounds; I perform the partition in place with two pointers scanning inward, exchanging the pair that each pointer finds on the wrong side, and stopping when the pointers cross; I keep the recursion from repeating the same segment by using the bound-word itself to exclude at least one position, or by reaching only singleton pieces in the equality edge cases; the partition is easy to program but the list of postponed pieces is not, and on a machine without recursion it defeats me until ALGOL's recursive procedures keep that last-in-first-out nest for me automatically — and the nest, made explicit, is bounded to log2(N) by always postponing the larger piece and continuing into the smaller.
