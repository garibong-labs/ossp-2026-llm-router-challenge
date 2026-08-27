<!--
SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# Safe-margin Fast matched-spend x3

This is one frozen, Dev-blind experiment based on commit
`3fbdfe84f7a7ccee247d3cc0536d11ada6a21ec9`. It does not change the normal
safe-margin router or its one-bucket-per-octave default.

## Frozen hypothesis and method

The hypothesis was that three efficiency buckets per octave could refine Fast
`ax31-light` to `ax31` ranking enough to improve realized quality without
changing the x1 promotion count or increasing its predicted incremental spend.
The runner first freezes the complete x1 selection. Balanced, Premium, and the
Premium `ax31` to `axk1-think` stage are returned exactly as x1. In Fast it
visits eligible unselected x3 groups from best rank to worst, then x1-selected
groups of the same cardinality from worst rank to best. It accepts only a
strict x3-rank improvement whose incoming predicted increment is no greater
than the outgoing increment. The full content signature breaks ties. Missing
or uncertain exact matches retain x1.

No other bucket count, threshold, feature, model, learned head, allowlist,
family rule, or Dev-derived special case may be evaluated. There is no tuning
after any result.

## Reproduction

Materialize Train using the repository-pinned public-source procedure, then
run from the repository root with Python 3.11:

```console
PYTHONPATH=src:baselines:tools python3.11 \
  tools/run_safe_margin_fast_matched_spend_x3.py
PYTHONPATH=src:baselines:tools python3.11 \
  tools/run_safe_margin_fast_matched_spend_x3.py --reemit-existing
PYTHONPATH=src:baselines:tools python3.11 \
  tools/run_safe_margin_fast_matched_spend_x3.py --reemit-existing
```

The experiment command exits `1` for this expected rejected result. The two
re-emissions do not open Train or Dev and must preserve the report bytes.

## Result and decision

Train baseline weighted score was `0.672982954545`; the candidate score was
`0.673210227273`, a delta of `+0.000227272728`. Fast quality improved by
`+0.0005681818181818181818181818182`. Balanced and Premium decisions matched
x1, Fast retained exactly 404 promotions, repeated output was deterministic,
and predicted Fast incremental spend fell from `1.001563053921` to
`0.976627896549` (predicted ratio `1.139948618882` to `1.136464424031`).

All nine family deltas were nonnegative and the worst was zero, but only
CruxEval improved strictly, so the positive-family count was `1/9`, below the
frozen `4/9` Train requirement. The Train gate therefore failed. Public Dev
was never materialized, loaded, hashed, inspected, or scored; safety was not
eligible and was not run. The candidate is rejected and the production/default
x1 safe-margin path remains unchanged.

The canonical terminal report is [`report.v1.json`](report.v1.json), with
SHA-256 `1fc401ad79d342b334ea4e9d1dac6bc2f7d2e59dc5ff8aaa27b41a3ddb18b9db`.
