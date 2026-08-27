<!--
SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
SPDX-License-Identifier: Apache-2.0
-->

# Safe-margin Fast double-consensus matched spend

This directory records one frozen, fail-closed experiment based on commit
`3fbdfe84f7a7ccee247d3cc0536d11ada6a21ec9`. It does not change the
production safe-margin router or its one-bucket-per-octave default.

## Frozen candidate

The candidate starts with the exact x1 Fast decisions. It reconstructs only
the existing Fast `ax31-light` to `ax31` eligible groups, using the x3
efficiency buckets from the matched-spend experiment. A deterministic,
equal-cardinality group swap is allowed only when all of these conditions
hold:

- the incoming group has a strictly higher x3 efficiency bucket;
- its mean frozen residual-consensus prediction is strictly higher;
- its predicted incremental load is no greater than the outgoing group's;
- neither group has participated in another swap.

The residual head and expanded structural feature contract come from PR #4's
Train-frozen residual-consensus artifact. Runtime residual features are
computed only from bounded prompt or message role/content. Episode identity,
split, benchmark family, outcomes, row position, Train, and Dev are not inputs
to selection. Missing, malformed, non-finite, or tied signals retain x1.
Stable content-derived signatures only resolve ordering among otherwise
eligible pairs.

Balanced and Premium are exact x1, Fast keeps the exact x1 model counts,
`axk1-think` is never introduced in Fast, and the candidate predicted Fast
increment cannot exceed x1. No alternate threshold, bucket count, head,
model, allowlist, family rule, or exception was evaluated.

The protocol SHA-256 is
`f7552ed172d2907766f3fb5fb4c3dd3bea24e1f9826d1bcd0b0bf69d9dca930d`.
The local frozen artifact SHA-256 is
`cf33622f9c6d31b21a44fe65530c562b4d8519e797029b8ddd69508ea2e3b964`;
it is JSON-value-identical to the PR #4 artifact, whose original byte hash is
`6eb4094e2e5eef0644ecfaf9b2c0d05d7d0732ff150da4249eae3bcd8ea1024b`.

## Terminal result

The complete Train gate failed:

- weighted score: `0.672982954545` x1 to `0.672755681818` candidate,
  delta `-0.000227272727`;
- Fast quality delta: `-0.0005681818181818181818181818182`;
- positive families: `0/9`; nonnegative families: `8/9`;
- worst family delta: deepmind-mathematics
  `-0.0013201320132013201320132013201320132013201320132013201320132013201320132013201320`;
- all other family deltas: exactly zero;
- Balanced/Premium exact x1: pass;
- Fast model counts exact x1: pass, with 404 `ax31` promotions;
- predicted Fast incremental spend: `1.001563053921` x1 to
  `0.982641573720` candidate, pass;
- deterministic repeated output: pass;
- matched group swaps: 4; planner fallback: false.

The Train input and outcomes SHA-256 values were respectively
`029a0fb1f70432a05b837a1291d86d42278bb202d808a6a12911b0dae8628ac4`
and `97a5a787086b3e1d9fa9c7945518543540e527ea248df4a4760de581b612a4ba`.
Train loads/evaluations were `1/1`. Because Train failed, Public Dev
loads/evaluations are `0/0`, and safety evaluations/resamples are `0/0`.

The candidate is rejected. The submission default remains production x1.
The canonical terminal report is [`report.v1.json`](report.v1.json), SHA-256
`35f90bc72872f38da37edf30497fc00121689cec4f5011435eff1f91bde6b8e7`.

## Reproduction

Materialize only Train using the repository-pinned public-source procedure,
then run from the repository root with Python 3.11:

```console
PYTHONPATH=src:baselines:tools python3.11 \
  tools/run_safe_margin_fast_double_consensus.py
PYTHONPATH=src:baselines:tools python3.11 \
  tools/run_safe_margin_fast_double_consensus.py --reemit-existing
PYTHONPATH=src:baselines:tools python3.11 \
  tools/run_safe_margin_fast_double_consensus.py --reemit-existing
```

The evaluation command exits `1` for the expected Train rejection. Report
re-emission opens neither Train nor Dev and must preserve the report bytes.
