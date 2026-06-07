# Cooley self-account — sources fetched and genesis facts

## Sources obtained (in refs/)
- **`cooley_oral_history.txt`** (+ `.html`) — IEEE/ETHW Oral-History: James W. Cooley,
  https://ethw.org/Oral-History:James_W._Cooley . This is the PRIMARY first-person self-account
  used as the backbone. Full transcript extracted. Cooley narrates the genesis in his own words.
- **`rockmore_fft_family.txt`** (+ `.pdf`) — Daniel N. Rockmore, "The FFT — an algorithm the whole
  family can use" (1999), https://www.cs.dartmouth.edu/~rockmore/cse-fft.pdf . Secondary, but
  independently corroborates the Garwin/Tukey-doodling/Kennedy-SAC/nuclear-test-detection genesis
  and the He-3 spin "cover problem." Used only as corroboration, not quoted in-frame.

## Source NOT obtained as full text
- James W. Cooley, **"How the FFT Gained Acceptance"** (ACM Conf. History of Sci. & Numeric
  Computation 1987, repr. IEEE SP Magazine Jan 1992; DOI 10.1109/79.109204 / 10.1145/41579.41589).
  Every accessible copy (IEEE Xplore, ACM DL, academia.edu, DeepDyve, semanticscholar) sits behind a
  paywall/login wall; the IEEE `iel4/...pdf` URL redirects to an auth page, the Wayback Machine has
  no snapshot, and academia.edu serves only a login stub. The genesis content of that paper, however,
  is **the same genesis Cooley narrates first-person in the ETHW oral history** (Garwin → Kennedy SAC
  meeting → Tukey doodling → nuclear-test detection by seismometers around the USSR → He-3 spin cover
  problem → N log N → IBM/7094), so the backbone is fully grounded on a primary self-account.

## Genesis (Cooley's own words, ETHW oral history — the real discovery path)
1. **Garwin brings the problem.** "Dick Garwin was here, and I had known him before... he came here
   with some ideas from John Tukey. They were both on President Kennedy's Scientific Advisory
   Council. Dick Garwin said John Tukey was doodling while at one of these meetings, and he asked him
   what he was doing, and he said... he was working on an idea for doing Fourier transform much faster
   than people do it now." Tukey's idea: write N = A·B, decompose the length-N transform into small
   transforms of size A and B; continue the factorization to get N=2^k in N·log N operations.
2. **The real driver (Garwin's, hidden from Cooley at first).** "...it was very, very important to
   Dick Garwin, because at the time they were discussing ways of limiting atomic bomb testing, and
   Dick Garwin had the idea of setting seismometers all around the Soviet Union, since the Soviet
   Union wouldn't allow on site testing. So he thought he could do it from outside, but to do it, you
   would have to process all the seismic signals... by Fourier transforms. But the computing power at
   the time was not enough... Now, this was his incentive, but he didn't discuss that, he came to me
   and he made up another problem."
3. **The cover problem.** "...he described another problem, looking for periodicities in the spin
   orientations of Helium-3 in a crystal. I didn't think it was very important, and I kind of put it
   on the back burner when I was doing something else, but he came and kept prodding. Finally they got
   a program done to do a three-dimensional Fourier transform, where N was the power of two in each
   dimension." (Year: "I first came here, I guess that was 1963.")
4. **Cooley's prior Tukey contact** (why the idea wasn't a surprise): he had already programmed
   spectral analysis of wind velocities for Tukey (Blackman–Tukey method — ironically designed to
   AVOID large Fourier transforms by doing the correlation first, then short transforms). Knew Tukey
   from the IAS computer project and from square dancing.
5. **Cooley turns the idea into the published algorithm.** "I just wrote a paper... simply describing
   the basics of the method and of course you could have N, any number of factors and then show you
   can get the smallest number of operations of N, as a power of 2 or 4, but any set of factors will
   do, which John pointed out right from the beginning."
6. **Why it got published at all** (patent angle): a patent lawyer ("Thomas") said it had patent
   possibilities; but since Tukey wasn't an IBMer and algorithms couldn't then be patented, IBM put
   it in the public domain to protect its right to use it; a footnote credited Ray Miller and
   Winograd with a device implementation to make the public-domain dedication stick.
7. **Precursors Cooley learned of only AFTER publishing:** Phil Rudnick wrote to Cooley that he'd
   gotten the idea from Lanczos's paper (Danielson–Lanczos, J. Franklin Inst. 1942); Cooley met
   Lanczos at Yale, who said he hadn't publicized it because "at the time, N was not very big." Tukey
   pointed Cooley to Frank Yates's factorial-experiment indexing scheme (no twiddle factor between
   iterations). Gauss (1805) was found even later by historians (Heideman–Johnson–Burrus).

## How this changes the trace
- Old motivation was abstract ("can I beat N^2?"). New motivation is seated on the REAL genesis:
  Garwin walks in with a pressing computational wall — detecting Soviet nuclear tests from the
  spectra of seismometer time series ringed around the USSR, an off-shore monitoring scheme to make a
  test ban verifiable without on-site inspection — plus a 3-D He-3 spin-periodicity transform; the
  computers of 1963 can't process the volume at N^2; and Tukey's doodled factorization idea (N=A·B →
  small transforms) is the lever Garwin hands me. The math derivation (parity split → twiddle →
  butterfly → bit reversal → general composite N) stays intact and correct; only the framing/aha is
  re-seated. The general-composite-N form is Tukey's "any set of factors" framing; radix-2 even/odd
  is the special case I program first (3-D, power of two per dimension).
