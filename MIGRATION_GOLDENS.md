# Migration goldens — integrate/upstream-base

Recorded 2026-07-21, RTX 5090, WebGPU, idle GPU (nvidia-smi bracketed),
`bench_batch_sweep3 <scene> <size> 1024 5 20`, in-worktree target dir.
Branch tip at recording: post khal-unified repoint.

## M1 findings (supersede the original bug note)

1. The chain@1024 anomaly was upstream's SERIAL dynamics tier auto-engaging at
   `total_mb >= 1024` (`multibody_set.rs pack_lanes()`). The serial tier's
   dynamics numerically diverge from the lane tier: ~1e-4 relative after ONE
   step on a contact-free chain (build-stable, same-binary A/B), ~3% after 25
   steps — the different trajectory is also why the sanity pair count read 0.
   M1 makes the tier explicit instead of scale-dependent (physics an RL policy
   sees must not depend on env count): default = lane-parallel;
   `NEXUS_SERIAL_MB=1` forces serial; `NEXUS_SERIAL_MB=auto` restores
   upstream's crossover heuristic. Cost at 1024 envs (coriolis): ~10%
   (546→617 µs). Report the divergence upstream.
2. Long-horizon chain checksums are NOT bit-stable across rebuilds of
   identical source (~0.7% after 25 steps; bimodal-deterministic — GPU
   pipeline-cache/FMA-contraction variance amplified by chain dynamics).
   Step-1 checksums ARE bit-stable across builds. VERIFICATION POLICY:
   bit-exact comparisons only within a single binary; across builds use
   per-env relative tolerance 1e-2 on 25-step checksums, or step-1 checksums
   for bit-exactness. boxes is insensitive (bit-stable across builds).

Golden 25-step chain checksums below are from the M0 binary; the M1 binary
reproduces 52.054231@1 exactly and ~51.69/env at ≥256 (within the documented
cross-build tolerance).

## boxes 3 (checksum / max|pos| / max pairs/env / avg µs/step)

| batches | checksum | max pos | pairs/env | µs/step |
|---|---|---|---|---|
| 1 | 19.211716 | 3.259 | 27 | 928 |
| 4 | 76.846864 | 3.259 | 27 | 909 |
| 16 | 307.387455 | 3.259 | 27 | 939 |
| 64 | 1229.549819 | 3.259 | 27 | 933 |
| 256 | 4918.199278 | 3.259 | 27 | 784 |
| 1024 | 19610.245018 | 3.261 | 27 | 916 |

## chain 6 (implicit default)

| batches | checksum | max pos | pairs/env | µs/step |
|---|---|---|---|---|
| 1 | 52.054231 | 17.475 | 1 | 1019 |
| 4 | 208.216926 | 17.475 | 1 | 1035 |
| 16 | 832.867702 | 17.475 | 1 | 968 |
| 64 | 3331.470810 | 17.475 | 1 | 1058 |
| 256 | 13325.883240 | 17.475 | 1 | 1093 |
| 1024 | 52930.732422 | 17.474 | 1 | 1017 | (M1 lane-tier default)

## chain 6 + NEXUS_EXPLICIT_CORIOLIS=1

| batches | checksum | max pos | pairs/env | µs/step |
|---|---|---|---|---|
| 16 | 838.076553 | 17.482 | 1 | 554 |
| 64 | 3352.306213 | 17.482 | 1 | 665 |
| 256 | 13409.224854 | 17.482 | 1 | 564 |
| 1024 | 53164.047607 | 17.467 | 1 | 617 | (M1 lane-tier default)

Non-finite count is 0 everywhere.
