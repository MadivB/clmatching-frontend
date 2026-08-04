# v0.5.6 release — FINAL performance (10 flow files, 126 events, COMPLETE)

Chain: v0.5.5 clustering + v0.4.1 matching, canonical flags; runs 2026-07-16 (jobs 56283204 + 56290441); 0 event errors.

| metric | value |
|---|---|
| NS total-eff (TPCs <=1.4 GeV, energy-weighted, 100% coverage) | **95.62%** |
| all-TPC final efficiency | 94.76% |
| coverage | 100.0% |
| events | 126 (all events of files 0000001-0000010) |
| mean time/event (2 workers/GPU) | 290 s (~145 s/GPU incl. release-dump writing) |

Per-file NS total-eff (n events):

- 0000001: 95.74%  (13 ev)
- 0000002: 95.73%  (13 ev)
- 0000003: 95.87%  (13 ev)
- 0000004: 95.84%  (12 ev)
- 0000005: 95.82%  (12 ev)
- 0000006: 95.49%  (13 ev)
- 0000007: 95.05%  (13 ev)
- 0000008: 95.77%  (12 ev)
- 0000009: 95.56%  (13 ev)
- 0000010: 95.28%  (12 ev)

Offline clustering baselines (file 0000001, recluster.py --skip-rerun):
energy-weighted purity 0.9880, completeness 0.3887 — the numbers to beat.
