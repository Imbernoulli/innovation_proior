# Self-Account Candidates — Cross-Field Catalog

A web-grounded catalog of **methods / discoveries that have a strong AUTHOR SELF-ACCOUNT** — the
inventor's own first-person record of how they found it (memoir, retrospective essay, award lecture,
notebook/archive, oral-history interview, or real-time blog/diary). These are the highest-value
grounding targets for discovery-trace generation.

**This is a catalog only — no traces here.** New entries that do NOT duplicate
`SELF_ACCOUNT_SOURCES.md` (Dijkstra EWD, Karatsuba, Feynman path integral, Villani, Poincaré,
Hadamard, Hamming *You & Your Research*, Knuth, Heilbronn/Quanta, Cooley–Tukey FFT) or the
`TARGETS_BACKLOG.md` ★ Priority add-ons (quicksort/Hoare, dijkstra/EWD, hamming-codes, KMP/Knuth,
simplex/Dantzig, renormalization-group/Wilson, dancing-links, tex-line-breaking, nash-embedding,
fft-cooley-tukey).

URL-verification convention: Nobel lecture HTML pages and author-site HTML were content-verified via
WebFetch where possible. Nobel-lecture **PDFs** at `nobelprize.org/uploads/...` resolve as real PDFs
but are image/binary so WebFetch can't OCR them; each is corroborated by its matching
`nobelprize.org/.../lecture/` HTML page (also listed). Scanned Turing-lecture PDFs likewise resolve
but are image-based. "Verified resolves" = the URL returns the document; "content-verified" = a
first-person passage was actually read back.

Legend for accessibility: **FREE** = free full text; **FREE (archive)** = free via Internet Archive
borrow/scan; **PAYWALL** = paywalled with no free full text found.

---

## Computer science / algorithms

| slug | title | inventor(s) | self-account (type + URL) | discovery hook | accessibility |
| --- | --- | --- | --- | --- | --- |
| viterbi-algorithm | Viterbi algorithm (optimal decoding of convolutional codes) | Andrew J. Viterbi | Retrospective essay "A Personal History of the Viterbi Algorithm" (IEEE Signal Processing Mag. 2006) — academia.edu copy: https://www.academia.edu/29441312/A_personal_history_of_the_viterbi_algorithm ; USC 50th-anniversary recounting: https://magazine.viterbi.usc.edu/spring-2017/intro/the-viterbi-algorithm-at-50/ | He invented it to *simplify teaching* convolutional codes in his info-theory class; "obsessed for several months… the right argument came together in my head while I was watching my little children" — no pen handy, had to wait to get home and write it up. | FREE (academia.edu copy + USC essay); IEEE original PAYWALL |
| public-key-cryptography | Public-key cryptography / Diffie–Hellman key exchange | Whitfield Diffie, Martin Hellman | Diffie's own historical memoir "The First Ten Years of Public-Key Cryptography" (Proc. IEEE 1988), free mirror: https://www.cs.virginia.edu/~evans/greatworks/diffie.pdf ; Hellman CBI oral history (linked from Turing bib): https://amturing.acm.org/award_winners/hellman_4055781.cfm | Diffie's years-long obsession searching for a way to communicate securely without a shared secret; Hellman's "breakthrough early one morning in May 1976." | FREE (UVa mirror PDF) |
| belief-propagation | Belief propagation / Bayesian networks | Judea Pearl | Memoir-style tech report "A Personal Journey into Bayesian Networks" (UCLA R-476, 2018): https://ftp.cs.ucla.edu/pub/stat_ser/r476.pdf ; Turing page: https://amturing.acm.org/award_winners/pearl_2658896.cfm | First-person account of the road from rule-based AI dissatisfaction to graphical models and the distributed message-passing ("Reverend Bayes") inference algorithm. | FREE (UCLA ftp PDF, verified resolves; 642KB) |
| lamport-distributed | Lamport clocks / Bakery algorithm / Paxos (distributed systems) | Leslie Lamport | "My Writings" — per-paper first-person annotations on the origin of each result: https://lamport.azurewebsites.net/pubs/pubs.html ; Turing oral history: https://amturing.acm.org/pdf/LamportTuringTranscript%20.pdf | Content-verified annotation: "I feel that I did not invent the bakery algorithm, I discovered it." "Time, Clocks" began when a replicated-DB report he was sent "wasn't quite right because it permitted… causality" to be violated. | FREE (HTML, content-verified) |
| np-completeness | NP-completeness / Reducibility among combinatorial problems | Richard M. Karp | 1985 Turing Award lecture "Combinatorics, Complexity, and Randomness" — free scanned PDF: https://www.cs.umd.edu/~gasarch/COURSES/452/S21/notes/KarpTuringAward.pdf | His personal narrative of theoretical CS forming around Cook's theorem and the 21 NP-complete reductions; why TSP looks "inherently intractable." | FREE (scanned PDF, verified resolves) |
| dfs-graph-algorithms | Depth-first-search-based graph algorithms / data structures | Robert Tarjan (with John Hopcroft) | Turing oral history (ACM): https://amturing.acm.org/pdf/TarjanTuringTranscript.pdf ; HP Labs inventor interview "The art of the algorithm": https://www.hpl.hp.com/news/2004/oct_dec/tarjan.html | "Hopcroft came in one day with a slick biconnectivity algorithm based on DFS"; they formalized it and the worst-case-analysis program followed. | FREE (ACM PDF + HP HTML) |
| gan | Generative adversarial networks (GAN) | Ian Goodfellow | His own oft-retold origin (AI Podcast ep.25 "an argument in a bar"): https://soundcloud.com/theaipodcast/what-are-generative-adversarial-networks-ian-goodfellow-explains ; narrated in MIT Tech Review "The GANfather": https://www.technologyreview.com/2018/02/21/145289/the-ganfather-the-man-whos-given-machines-the-gift-of-imagination/ | Argument at Les 3 Brasseurs in Montréal → he realized a *discriminator* could train a *generator* in one game → coded it that night, "worked the first time." | FREE (podcast + article) |
| simulated-annealing | Optimization by simulated annealing | Scott Kirkpatrick, C.D. Gelatt, M.P. Vecchi | IBM Research first-person retrospective page (Kirkpatrick): https://researcher.watson.ibm.com/researcher/view_page.php?id=6983 ; primary paper free PDF: https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/TemperAnneal/KirkpatrickAnnealScience1983.pdf | The statistical-mechanics-as-optimization analogy: treat a hard combinatorial cost like an energy and slowly "cool" it. | FREE (paper PDF); retrospective page FREE |
| algorithmic-lll | The Algorithmic Lovász Local Lemma (constructive LLL) | Robin A. Moser (and Gábor Tardos) | STOC 2009 talk recorded by Fortnow/GASARCH (https://blog.computationalcomplexity.org/2009/06/kolmogorov-complexity-proof-of-lov.html) + Tao's canonical write-up of the talk's framing "Moser's entropy compression argument" (https://terrytao.wordpress.com/2009/08/05/mosers-entropy-compression-argument/) | Moser found the *slick* incompressibility proof **while preparing his STOC talk** ("one of the best STOC talks ever"); the talk's information-theoretic / Kolmogorov-complexity argument is NOT in the written paper, which carries the equivalent combinatorial composite-witness counting. The omitted reasoning: why the one-line "random assignment, resample any violated event, repeat" beats the prior freeze-and-brute-force algorithms stuck at 2^{k/c}. | FREE (both blogs); both saved in `methods/algorithmic-lll/refs/` |

## Information theory / EE / signal processing

| slug | title | inventor(s) | self-account (type + URL) | discovery hook | accessibility |
| --- | --- | --- | --- | --- | --- |
| shannon-information-theory | A Mathematical Theory of Communication | Claude E. Shannon | IEEE/ETHW oral history (interviewer Robert Price, focuses on the 1940s genesis & Wiener relationship): https://ethw.org/Oral-History:Claude_E._Shannon | Shannon says his wartime cryptography and communication insights "were so close together you couldn't separate them"; the 1945 classified report shows the theory essentially worked out before 1948. | FREE (ETHW transcript) |
| kalman-filter | Kalman filter (recursive linear estimation) | Rudolf E. Kálmán | IEEE/ETHW biographical + interview material: https://ethw.org/Rudolf_E._Kalman ; "Life and Works" memoir compilation: https://www.sciencedirect.com/science/article/pii/S2405896317300903 | The state-space/recursive reformulation of Wiener filtering — Kálmán's framing of estimation as a system-theory problem, recounted in his own talks/interviews. | FREE (ETHW); ScienceDirect partly PAYWALL |
| daubechies-wavelets | Orthonormal compactly-supported wavelets | Ingrid Daubechies (with Mallat, Meyer, Morlet antecedents) | *Ten Lectures on Wavelets* front-matter/preface (her own narration of the field's synthesis): https://archive.org/details/tenlecturesonwav0000daub ; SIAM front matter: https://epubs.siam.org/doi/pdf/10.1137/1.9781611970104.fm | Wavelets as a *synthesis* of engineering subband coding, physics coherent states/RG, and Calderón–Zygmund math; her construction of compactly supported orthonormal bases. | FREE (archive borrow); SIAM front-matter FREE |

## Pure mathematics

| slug | title | inventor(s) | self-account (type + URL) | discovery hook | accessibility |
| --- | --- | --- | --- | --- | --- |
| fermat-last-theorem | Proof of Fermat's Last Theorem (modularity) | Andrew Wiles | NOVA/PBS first-person interview "Solving Fermat": https://www.pbs.org/wgbh/nova/article/andrew-wiles-fermat/ | "Sitting at my desk thinking about the last stage of the proof" the morning the *fix* came — after the 1993 gap nearly killed it; the famous "suddenly, totally unexpectedly" aha. | FREE (PBS) |
| fractal-geometry | Fractal geometry | Benoit B. Mandelbrot | Memoir *The Fractalist: Memoir of a Scientific Maverick* (2012): https://archive.org/details/fractalistmemoir0000mand_e9n5 | "I started looking in the trash cans of science"; a scrap from a wastebasket read on the Paris métro → his 1965 psycholinguistics paper → noise on phone lines → fractals. | FREE (archive borrow); book PAYWALL retail |
| penrose-singularity | Trapped surfaces & the singularity theorem (and Penrose tiling) | Roger Penrose | Nobel lecture 2020 (HTML + PDF): https://www.nobelprize.org/prizes/physics/2020/penrose/lecture/ ; biographical: https://www.nobelprize.org/prizes/physics/2020/penrose/biographical/ | The much-told account of the idea of a "trapped surface" arriving mid-conversation while crossing a London street with Ivor Robinson — a thought that surfaced only after the talk resumed. | FREE (Nobel HTML; lecture PDF verified resolves) |

## Physics

| slug | title | inventor(s) | self-account (type + URL) | discovery hook | accessibility |
| --- | --- | --- | --- | --- | --- |
| cmb-discovery | Discovery of the cosmic microwave background | Arno Penzias, Robert W. Wilson | Penzias Nobel lecture "The Origin of Elements" (PDF): https://www.nobelprize.org/uploads/2018/06/penzias-lecture.pdf ; Wilson lecture "The CMB Radiation" (PDF): https://www.nobelprize.org/uploads/2018/06/wilson-lecture-1.pdf ; HTML: https://www.nobelprize.org/prizes/physics/1978/penzias/lecture/ | The famous "excess antenna temperature" they could not eliminate — pigeons, droppings, then the dawning realization it was the universe. | FREE (Nobel) |
| lorenz-chaos | Deterministic chaos / sensitive dependence ("butterfly effect") | Edward N. Lorenz | His own AAAS 1972 talk text "Does the flap of a butterfly's wings in Brazil set off a tornado in Texas?" (widely reprinted); MIT Tech Review recounting "When the Butterfly Effect Took Flight": https://www.technologyreview.com/2011/02/22/196987/when-the-butterfly-effect-took-flight/ ; "Chaos at Fifty" history w/ his account: https://arxiv.org/pdf/1306.5777 | Re-entering a rounded value (.506 for .506127) after a coffee → the run diverged wildly → sensitive dependence on initial conditions. | FREE (arXiv + Tech Review); talk text FREE in many reprints |
| thooft-renormalization | Renormalization of non-abelian gauge theories | Gerardus 't Hooft | Nobel lecture (HTML): https://www.nobelprize.org/prizes/physics/1999/thooft/lecture/ ; first-person recollections "The Glorious Days of Physics" (arXiv hep-th/9812203): https://arxiv.org/pdf/hep-th/9812203 | His own "recollections of the turbulent days preceding the establishment of the Standard Model" — how the renormalizability proof came together in his thesis. | FREE (Nobel HTML + arXiv) |
| feynman-qed-note | (cross-ref) Path integral / QED | Richard Feynman | Already in SELF_ACCOUNT_SOURCES (Nobel lecture). Listed only to avoid re-adding. | — | FREE |

## Chemistry

| slug | title | inventor(s) | self-account (type + URL) | discovery hook | accessibility |
| --- | --- | --- | --- | --- | --- |
| pcr | Polymerase chain reaction (PCR) | Kary B. Mullis | Nobel lecture "The Polymerase Chain Reaction" (1993): https://www.nobelprize.org/prizes/chemistry/1993/mullis/lecture/ | Content-verified: the late-night Highway 128 drive — "I stopped the car at mile marker 46.7… In the glove compartment I found some paper and a pen"; the exponential-doubling idea hitting him at the wheel. | FREE (Nobel HTML, content-verified) |
| directed-evolution | Directed evolution of enzymes | Frances H. Arnold | Nobel lecture (2018): https://www.nobelprize.org/prizes/chemistry/2018/arnold/lecture/ ; biographical: https://www.nobelprize.org/prizes/chemistry/2018/arnold/biographical/ | Giving up on rational design and instead "letting evolution do the work" — iterated mutation + selection on proteins; her account of the conceptual surrender. | FREE (Nobel) |

## Biology / medicine

| slug | title | inventor(s) | self-account (type + URL) | discovery hook | accessibility |
| --- | --- | --- | --- | --- | --- |
| h-pylori-ulcers | H. pylori causes peptic ulcers (self-experiment) | Barry J. Marshall (with J. Robin Warren) | Nobel lecture (PDF): https://www.nobelprize.org/uploads/2018/06/marshall-lecture.pdf ; HTML: https://www.nobelprize.org/prizes/medicine/2005/marshall/lecture/ ; Nobel interview: https://www.nobelprize.org/prizes/medicine/2005/marshall/interview/ | No animal model + two years of skeptics → he drank a culture of H. pylori himself (told neither ethics board nor wife) and gave himself gastritis. | FREE (Nobel) |
| transposons | Transposable elements ("jumping genes") | Barbara McClintock | Nobel lecture "The Significance of Responses of the Genome to Challenge" (PDF): https://www.nobelprize.org/uploads/2018/06/mcclintock-lecture.pdf ; HTML: https://www.nobelprize.org/prizes/medicine/1983/mcclintock/lecture/ | Her own account of reading the maize genome's "responses to challenge" — control systems she saw decades before the field accepted mobile elements. | FREE (Nobel) |
| double-helix | Structure of DNA (the discovery, as lived) | James D. Watson (with Crick) | Memoir *The Double Helix: A Personal Account of the Discovery of the Structure of DNA* (1968): https://archive.org/details/doublehelixpers00wats_0 | The canonical (and contested) day-by-day first-person narrative — model-building, Franklin's Photo 51, the base-pairing click. | FREE (archive borrow) |

## Statistics

| slug | title | inventor(s) | self-account (type + URL) | discovery hook | accessibility |
| --- | --- | --- | --- | --- | --- |
| bootstrap | The bootstrap (resampling inference) | Bradley Efron | Retrospective "Second Thoughts on the Bootstrap" (Statistical Science 2003), free: https://projecteuclid.org/euclid.ss/1063994968 ; encyclopedia narrative of his development: https://www.encyclopedia.com/science/encyclopedias-almanacs-transcripts-and-maps/efrons-development-bootstrap | Pushing the jackknife to its logical extreme once computers made simulation cheap — "thinking the unthinkable" amount of recomputation. | FREE (Project Euclid) |
| exploratory-data-analysis | Exploratory data analysis (boxplot, stem-and-leaf, robust ideas) | John W. Tukey | His program book *Exploratory Data Analysis* (1977) reframing data work in his own voice: https://archive.org/details/exploratorydataa0000tuke ; Wikipedia bio overview: https://en.wikipedia.org/wiki/John_Tukey | The deliberate shift from confirmatory statistics to "look at the data first" — his own manifesto for exploration. | FREE (archive borrow) |

## Economics / finance

| slug | title | inventor(s) | self-account (type + URL) | discovery hook | accessibility |
| --- | --- | --- | --- | --- | --- |
| portfolio-theory | Modern portfolio theory (mean–variance) | Harry M. Markowitz | Nobel lecture "Foundations of Portfolio Theory" (PDF): https://www.nobelprize.org/uploads/2018/06/markowitz-lecture.pdf ; HTML: https://www.nobelprize.org/prizes/economic-sciences/1990/markowitz/lecture/ | His own telling of the library afternoon reading Williams' *Theory of Investment Value*, realizing diversification = variance, and that risk/return is a *trade-off frontier*. | FREE (Nobel) |
| black-scholes-merton | Option-pricing formula | Myron Scholes, Robert C. Merton (with Fischer Black) | Scholes Nobel lecture: https://www.nobelprize.org/prizes/economic-sciences/1997/scholes/lecture/ ; Merton Nobel lecture & bio: https://www.nobelprize.org/prizes/economic-sciences/1997/merton/biographical/ | Their own accounts of the no-arbitrage / dynamic-hedging insight that lets you price an option by replicating it. | FREE (Nobel) |

---

## Notes on a few that are unexpectedly rich

- **Lamport, "My Writings"** — rare: an author who annotates *every* paper with its backstory and the
  mistake/insight that produced it ("I discovered, not invented, the bakery algorithm"). Effectively a
  self-account index for a whole career; content-verified.
- **Pearl, "A Personal Journey into Bayesian Networks"** — a full memoir-as-tech-report, freely on the
  UCLA ftp server; narrates the path away from rule-based AI to message passing.
- **Mullis Nobel lecture** — unusually literary for a Nobel lecture; the Highway-128 drive is narrated
  beat by beat (mile marker and all).
- **'t Hooft "The Glorious Days of Physics"** (arXiv) — an explicit first-person memoir of the
  Standard-Model years, beyond the formal Nobel lecture.
- **Lorenz's 1972 AAAS butterfly talk** — the author himself coining the framing the whole field now
  uses; widely reprinted free.
</content>
</invoke>
