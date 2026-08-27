<!--
SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# Safe-margin tier-selective x3

This is one frozen, Dev-blind experiment based on
`3fbdfe84f7a7ccee247d3cc0536d11ada6a21ec9`. It does not change any
production/default router file: normal safe-margin remains x1.

The candidate composes the two frozen sibling contracts without tuning:
Fast uses group-atomic, equal-cardinality matched-spend x3 swaps from the
complete x1 selection; Balanced is byte-for-byte the x1 decision sequence;
Premium uses the full x3 efficiency-bucket allocation with every x1 gain,
cost, budget, group, tail, and concentration guard intact. Runtime selection
uses only existing safe-margin predictions and prompt-derived signatures.
Malformed or invariant-breaking evidence returns the affected tier to x1.

## Terminal result

Train matched the frozen pre-diagnostic exactly: baseline
`0.672982954545`, candidate `0.673295454545`, and weighted delta
`+0.000312500000`. Fast quality improved by
`+0.0005681818181818181818181818182`; CruxEval and GSM8K were the two
positive families, all 9/9 families were nonnegative, and the worst delta was
zero. Fast made 9 matched swaps while retaining 404 x1 promotions and reducing
predicted incremental spend from `1.001563053921` to `0.976627896549`.
Balanced was exact x1. Premium changed 29 decisions and passed the full x3
guard and budget contract. All Train gates passed.

Public Dev was loaded and evaluated once. The candidate scored
`0.672727272727`, below both the x1 baseline `0.673181818182` and the
strict `> 0.673182` gate. Every tier budget and structural invariant passed,
but the score boundary failed. Safety therefore remained closed with zero
resamples. The candidate is rejected and safe-margin x1 remains the default.

Before the terminal run, an incorrect nonexistent Dev path was supplied after
a successful Train gate. That path lookup loaded no Dev bytes and performed no
Dev evaluation; the corrected terminal run used the materialized Dev data from
the authorized efficiency-bucket sibling. The canonical report counts the
successful terminal run (Train 1 load/evaluation, Dev 1 load/evaluation,
safety 0).

## Reproduction

Run with Python 3.11 and materialized public inputs:

```console
PYTHONPATH=src:baselines:tools python3.11 \
  tools/run_safe_margin_tier_selective_x3.py
PYTHONPATH=src:baselines:tools python3.11 \
  tools/run_safe_margin_tier_selective_x3.py --reemit-existing
```

The experiment exits `1` for the expected Dev rejection. Re-emission reads
no Train or Dev data and must preserve
[`report.v1.json`](report.v1.json) byte-for-byte.
