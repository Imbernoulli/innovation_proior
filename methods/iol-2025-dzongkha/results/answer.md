# Answer

## The deciphered system

Dzongkha writes numbers two ways. **System A** is **base 20** (with a four-hundreds place, 20² = 400); **System B** is **base 10**. The same digit appears in several stem-shapes depending on its role, and one syllable (`ɲiɕu`) means 20 in System B but 400 in System A.

### Units 1–9 (shared)

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| ci | ɲi | sum | ʑi | ŋa | ɖu | dyn | ge | gu |

### Teens (a ten-front + the unit)

10 = cutãm; 11–19: cuci, cuɲi, cusum, cyʑi, ceŋa, cuɖu, cupdyn, copge, cygu.
(The ten-front is cu- / cy- / ce- / cup- / cop- depending on the unit.)

### System A — base 20, big-endian, places joined by `da`

- **Twenties** are marked by `ke`: `ke X da Y` = **20·X + Y**, where X is the digit (or teen) naming the twenties and Y is the remainder under twenty. A bare place drops the `da Y` tail: `ke ʑi` = 20·4 = 80; `ke ceŋa` = 20·15 = 300 (ceŋa = 15).
- **Four-hundreds** are marked by `ɲiɕu`: `ɲiɕu X da ke β da α` = **400·X + 20·β + α**.
- **Fractional overcounting.** `pɟe` = **1/2** and `ko` = **3/4** mark how far into the *current* place you have climbed; the named digit is the place *in progress*, not the count completed:
  - `ke pɟe-da X` = 20·(X−1) + 10 = **20X − 10**
  - `ke ko-da X`  = 20·(X−1) + 15 = **20X − 5**
  - `ɲiɕu pɟe-da X` = 400·(X−1) + 200 = **400X − 200**
  - `ɲiɕu ko-da X`  = 400·(X−1) + 300 = **400X − 100**

### System B — base 10

- **Round tens** = digit-stem + `cu`: sumcu = 30, ʑipcu = 40, ŋapcu = 50, ɖukcu = 60, dyncu = 70, gepcu = 80. **Twenty is special**: `ɲiɕu` = 20 (with `-ɕu`, not `-cu`).
- **Ten + units** = tens-stem + unit: tsaɲi = 22, tsaŋa = 25, soʑi = 34, ʑedyn = 47, ŋaŋa = 55, dønɖu = 76, ɟagu = 89. (Tens-stems: tsa-=20s, so-=30s, ʑe-=40s, ŋa-=50s, døn-=70s, ɟa-=80s.)
- **Hundreds** = digit-stem + `ɟa`: sumɟa = 300, ŋapɟa = 500, ɖukɟa = 600, dynɟa = 700, ɲiɟa = 200. `cutãm` = standalone 10, so ɲiɟa cutãm = 210. The 80-plus-unit stem is `ɟa-`, so ɟasum = 83 and ɟaɖu = 86.

The verified digit table (one row per System-B role):

| role | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| unit (k) | ci | ɲi | sum | ʑi | ŋa | ɖu | dyn | ge | gu |
| 10+k front | cu- | cu- | cu- | cy- | ce- | cu- | cup- | cop- | cy- |
| k×10 stem | cutãm | ɲi- | sum- | ʑip- | ŋap- | ɖuk- | dyn- | gep- | (gup-) |
| k×10(+…) stem | — | tsa- | so- | ʑe- | ŋa- | (re-) | døn- | ɟa- | (go-) |

## (a) The blanks

- **[X] = 60 = ɖukcu**
- **[Y] = 62 = ke sum da ɲi**  (20·3 + 2)
- **[Z] = 885 = ɲiɕu ɲi da ke ʑi da ŋa**  (400·2 + 20·4 + 5)

## (b) The equalities in digits

| | digits | filled equality |
|---|---|---|
| (1) | 13 + 70 = 83 | cusum + ke pɟe-da ʑi = ɟasum |
| (2) | 800 = 20 × 40 | ɲiɕu ɲi = ɲiɕu × ʑipcu |
| (3) | 469 = 50 × 9 + 19 | ɲiɕu ci da ke sum da gu = (ŋapcu × gu) + cygu |
| (4) | 600 + 110 = 500 + 210 | ɲiɕu pɟe-da ɲi + ke pɟe-da ɖu = ŋapɟa + ɲiɟa cutãm |
| (5) | 2 × 3/4 + 1/2 = 2 | (ɲi × ko) + pɟe = ɲi |
| (6) | 1100 × 1/2 + 50 = 600 | (ɲiɕu ko-da sum × pɟe) + ke pɟe-da sum = ɖukɟa |
| (7) | 736 = 84 × 4 + 400 | ɲiɕu ci da ke cuɖu da cuɖu = (ɟaʑi × ʑi) + ʑipɟa |
| (8) | 2 × 609 = 60 × 20 + 18 | ɲi × ɲiɕu ci da ke cutãm da gu = (ɖukcu × ɲiɕu) + copge |
| (9) | 62 + 24 = 86 | ke sum da ɲi + ke ci da ʑi = ɟaɖu |
| (10) | 885 + 115 = 700 + 300 | ɲiɕu ɲi da ke ʑi da ŋa + ke ko-da ɖu = dynɟa + sumɟa |

## (c) 75 and 570 in both systems

- **75** = System A **ke ko-da ʑi** (20·4 − 5)  =  System B **dønŋa** (70 + 5)
- **570** = System A **ɲiɕu ci da ke pɟe-da gu** (400 + (20·9 − 10) = 400 + 170)  =  System B **ŋapɟa dyncu** (500 + 70)
