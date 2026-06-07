# Kalman self-account excerpts (sources)

## 1. The train insight (Grewal & Andrews, *Kalman Filtering: Theory and Practice Using MATLAB*, 2e, Wiley 2001, §1.2 "Discovery of the Kalman Filter"; in refs/grewal-andrews.txt lines ~1366-1390)

"In late November of 1958, not long after coming to RIAS, Kalman was returning by train to Baltimore from a visit to Princeton. At around 11 PM, the train was halted for about an hour just outside Baltimore. It was late, he was tired, and he had a headache. While he was trapped there on the train for that hour, an idea occurred to him: Why not apply the notion of state variables to the Wiener filtering problem? He was too tired to think much more about it that evening, but it marked the beginning of a great exercise to do just that. He read through Loève's book on probability theory and equated expectation with projection. That proved to be pivotal in the derivation of the Kalman filter. With the additional assumption of finite dimensionality, he was able to derive the Wiener filter as what we now call the Kalman filter."

## 2. Kalman's own quotes — UCLA talk "System Theory: Past and Present," April 17, 1991 (quoted in Grewal-Andrews §1.2)

- On choosing a mechanical-engineering journal (ASME J. Basic Eng.) over an EE venue: "When you fear stepping on hallowed ground with entrenched interests, it is best to go sideways."
- The continuous-time (Kalman-Bucy) paper was once rejected because a referee said one step in the proof "cannot possibly be true." (It was true.)

## 3. Kalman's early-research framing (Grewal-Andrews §1.2)

Reading Ragazzini's 1952 sampled-data paper, "the idea occurred to him that there is no fundamental difference between continuous and discrete linear systems. The two must be equivalent in some sense." This started his algebra/systems-theory and state-space program.

## 4. Bucy's contribution (Grewal-Andrews §1.2)

Richard S. Bucy, also at RIAS, "suggested to Kalman that the Wiener-Hopf equation is equivalent to the matrix Riccati equation - if one assumes a finite-dimensional state-space model."

## 5. Schmidt / NASA Ames first-hand account (L. McGee & S. Schmidt, "Discovery of the Kalman Filter as a Practical Tool for Aerospace and Industry," NASA TM-86847, 1985; refs/schmidt-1981-discovery.txt)

- Fall 1960, Kalman (unaware of the Apollo work) called and arranged to visit Schmidt at NASA Ames; presented his 1960 paper to the Dynamics Analysis Branch. "Because the staff had been thinking of filter theory as a way of handling the problem, the presentation hit a responsive chord. In particular, the sequential solution features of Dr. Kalman's formulation were of interest because they could certainly relieve some of the computational problems we were facing with the IBM 704."
- Why Wiener filtering failed for Apollo midcourse navigation: nonlinearity + an "irregular series of discrete measurements"; "We could not find an approach that would permit applications of the Weiner filter theory without making approximations that would either severely restrict the observation system or destroy the inherent accuracy."
- Schmidt's predict/update decomposition: "The solution to the problem was obtained by decomposing the original formulation into a discrete-time update portion and a discrete-time optical measurement update portion which provided a much more natural and intuitively appealing way of expressing Dr. Kalman's algorithm. Looking back, the decomposition seems almost trivial; at the time, however, it was a major and critical step forward."
- The EKF (relinearize about the current estimate): "We reasoned that 'on the average,' the estimated state would be closer to the actual, or true, state than to the reference, or nominal, state and thus the linearity of the approximation would be retained better." Confirmed by an accidental-input incident.

NOTE: The ETHW "Oral-History:Rudolf_Kalman" page exists but is EMPTY (no transcript). The ETHW biography page (refs/kalman_ethw_bio.html) is third-party biographical, not first-person.
