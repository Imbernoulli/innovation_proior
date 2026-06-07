# Bradley Efron — self-account of discovering the bootstrap (captured quotes)

Captured this run from accessible interviews/retrospectives. The canonical retrospective
"Bradley Efron: A Conversation with Good Friends" (Holmes, Morris & Tibshirani,
*Statistical Science* 18(2):268–281, 2003) and "Second Thoughts on the Bootstrap"
(*Statistical Science* 18(2):135–140, 2003) are on Project Euclid but were Incapsula-blocked
to direct download and would not render via WebFetch; ResearchGate copy returned HTTP 403.
The two interviews below ARE accessible and carry the same genesis narrative.

## Source 1 — Statistics Views interview
"Empirical Bayes has been the most riveting topic for me. It still seems like magic sometimes":
An interview with Bradley Efron. statisticsviews.com.
URL: https://www.statisticsviews.com/article/empirical-bayes-has-been-the-most-riveting-topic-for-me-it-still-seems-like-magic-sometimes-an-interview-with-bradley-efron/

Verbatim / near-verbatim:
- "The jackknife was a hot and mysterious topic when I was a graduate student. My advisor,
  Rupert Miller, wrote a paper called 'A trustworthy jackknife'... The bootstrap started out
  as an attempt to put the jackknife on familiar statistical grounds, the first paper being
  titled 'Bootstrap methods: another look at the jackknife'."
- The jackknife was "a hot and mysterious topic" during his graduate studies; Miller's
  "A trustworthy jackknife" showed "examples of when the jackknife worked or failed."
- "The breakthrough came during a 1972–1973 visit to Imperial College. After Miller delivered
  a lecture on the jackknife, Professor David Cox suggested to Efron that 'this was a promising
  area.'"
- "I've always wanted to write a paper that is pure statistics and not math in statistics
  disguise. Hard to do, but the first bootstrap paper came close."
- "My advice to young colleagues is to always hang around brilliant colleagues."

## Source 2 — Caltech Heritage Project oral history (interviewer David Zierler)
URL: https://heritageproject.caltech.edu/interviews/bradley-efron
(also https://heritageproject.caltech.edu/interviews-updates/bradley-efron)

Verbatim:
- "The only thing I've ever done that's really had outside resonance of any real size is the
  bootstrap."
- (on Rupert Miller's jackknife problem) "The bootstrap came right out of that. That was a
  lucky break, to be told what was an interesting problem."
- "First of all, the problem's important. It's, how do you assign accuracy to a statistical
  estimate? The older theories require mathematical analysis, making models, and stuff like
  that. Bootstrap's almost automatic. In that sense, it's like the maximum likelihood. You can
  make one program that does it all the time. It's not hard to do."
- "When you say that the percentage of Democrats is 54% plus or minus 3%, you get some feeling
  what that plus or minus 3% means."
- "It was originally a computationally intensive statistical method. I wrote a paper with Persi
  Diaconis for Scientific American on computationally intensive methods, and it was the first of
  those."
- (paraphrase of how it works, in his words) "If you get some data, and you have a parameter
  you're interested in, like the mean of some quantity. You can estimate it from the data, but
  how accurately are you estimating it?"

## Source 3 — Dave Giles blog "Bradley Efron and the Bootstrap" (Econometrics Beat, 2019)
URL: https://davegiles.blogspot.com/2019/01/bradley-efron-and-bootstrap.html
- Quotes from the 2003 "Conversation": the 1979 Annals paper was initially given a hard time
  because it "didn't have any theorems in it." (Efron himself, on the Annals reception.)

## Primary-source corroboration of naming (Efron 1979, Acknowledgments)
From the 1979 Annals paper itself, p.25:
- "I also wish to thank the many friends who suggested names more colorful than Bootstrap,
  including Swiss Army Knife, Meat Axe, Swan-Dive, Jack-Rabbit, and my personal favorite, the
  Shotgun, which, to paraphrase Tukey, 'can blow the head off any problem if the statistician
  can stand the resulting mess.'"
- Title/abstract frame: "This article attempts to explain the jackknife in terms of a more
  primitive method, named the 'bootstrap' for reasons which will become obvious."
- Acknowledgments credit "Professors Rupert Miller and David Hinkley for numerous discussions."

## How the self-account shapes the trace
- Backbone = Efron's own framing: the jackknife is "hot and mysterious"; Miller's "trustworthy
  jackknife" shows it sometimes works, sometimes fails; the goal is to "put the jackknife on
  familiar statistical grounds" — i.e. find the more primitive thing the jackknife is an
  approximation OF. That is exactly the paper's logic (jackknife = linear/delta approximation
  to the bootstrap).
- The leap (resample with replacement from the empirical distribution, recompute the statistic
  many times) is presented by Efron as the natural plug-in once you ask "what is the jackknife
  estimating?" and once computers make recomputation cheap ("computationally intensive...
  almost automatic... one program that does it all the time").
