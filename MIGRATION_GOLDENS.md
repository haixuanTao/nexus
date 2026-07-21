# Migration goldens — integrate/upstream-base

Recorded 2026-07-21, RTX 5090, WebGPU, idle GPU (nvidia-smi bracketed),
`bench_batch_sweep3 <scene> <size> 1024 5 20`, in-worktree target dir.
Branch tip at recording: post khal-unified repoint.

KNOWN BUG at recording time (M1 target): chain@1024 reports `max pairs/env 0`
and different physics (max|pos| 17.600 vs 17.475) — real pair loss above 256
batches, not just a metric artifact. 1024-batch chain goldens below are
therefore PRE-FIX values and expected to change in M1; all ≤256 values must
stay bit-identical through M1.

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
| 1024 | 54619.140747 | 17.600 | 0 (BUG) | 1090 |

## chain 6 + NEXUS_EXPLICIT_CORIOLIS=1

| batches | checksum | max pos | pairs/env | µs/step |
|---|---|---|---|---|
| 16 | 838.076553 | 17.482 | 1 | 554 |
| 64 | 3352.306213 | 17.482 | 1 | 665 |
| 256 | 13409.224854 | 17.482 | 1 | 564 |
| 1024 | 54876.013916 | 17.600 | 0 (BUG) | 619 |

Non-finite count is 0 everywhere.
