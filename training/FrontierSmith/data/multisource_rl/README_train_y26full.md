NOTE (2026-08-18): train_y26full.parquet CONTENT WAS REPLACED before job
12584423 started. It now carries the BARE time sentence "It is now year 2026."
as the system prompt (user decision: the convention is time-only; the persona
and delivery-clause sentences were legacy additions from the SFT remediation
and should not propagate). The filename is kept only because the queued rlv13
chain's env points at this exact path and resubmitting would forfeit queue age.
Canonical copy: train_time2026.parquet (identical content).
