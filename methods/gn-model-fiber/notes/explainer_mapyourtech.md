# Third-party explainer capture (mapyourtech.com — GN model for optical transmission)
G_NLI(f) = (16/27) γ² ∫∫ G(f1)G(f2)G(f1+f2−f) |ρ(f1,f2,f,Ls)|² df1 df2
16/27 = dual-pol polarization averaging.
Integrated: P_NLI ≈ (16/27) γ² P_ch³ L_eff² (...) · N  → cubic in P_ch, ~linear in N (incoherent, high dispersion).
GOSNR⁻¹ = OSNR_ASE⁻¹ + OSNR_NLI⁻¹;  OSNR_NLI = P_ch / P_NLI.
P_opt = [P_ASE / (2 η)]^(1/3);  at optimum P_ASE = 2 P_NLI; MaxGSNR = P/(1.5 P_ASE) (3-dB rule).
Multi-span: P_NLI,total = N·P_NLI,span (incoherent, high dispersion) vs N² coherent.
