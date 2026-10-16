# LexForm

Research implementation beside the existing speech and FIR pipeline. It does not change statutory mapping, cognizability or filing rules. The collapsible LexForm workspace contains **Inspect facts** and **Statutory forms**; experiment administration stays in the CLI.

## Using the application

Start `./start.sh`, record or upload speech, verify the English working text, then open LexForm. Choose **Use English working text**, select IPC or BNS, and **Evaluate elements**. Expand a ranked provision for its four group scores and prototype memberships. **View provision** opens its program and source text. Scores are compatibility scores, not probabilities of guilt. Retained/abstained follows development-set calibration and is not an officer's legal decision. Nothing here automatically applies a section to the FIR.

The shipped frozen-encoder checkpoint contains learned heads and statutory features, not case facts or the base encoder weights. On a new machine fetch that public base once with `python -m research.lexform setup --assets encoder`; application inference then runs offline. The backend preloads the checkpoint on CPU and retains it in memory alongside the speech models.

## Setup

Use the existing Linux checkout and virtual environment. No WSL paths are assumed.

```sh
uv sync --extra full --extra research
uv run --extra research python -m research.lexform audit
uv run --extra research python -m unittest research.lexform.tests.test_lexform
uv run --extra research python -m research.lexform smoke
```

`datasets`, `bitsandbytes` and `pyarrow` are not required. All experiment model loads use local files only. Fetch encoder weights separately before training. Qwen decomposition uses a local quantized GGUF through the optional `decomposition` extra; its model is loaded once per decomposition job and each successful section is cached. It never runs in a training loop.

Place the original fixed `train.jsonl`, `dev.jsonl`, `test.jsonl`, `secs.jsonl` under `data/ilsi`, or set `LEXFORM_DATA_DIR`. Experiments never download missing data. The required split sizes are 42,835 / 10,200 / 13,039; text overlap, labels and hashes are validated. Set `LEXFORM_OUTPUT_DIR` to change the default ignored `artifacts/lexform` directory.

## Phase commands

Run from the repository root. Each invocation creates an immutable run directory, logs the command, dependency versions and configuration, and writes a JSON outcome, including blocked and failed attempts. Full experiments require a passing P0 result on identical input hashes. P2 and P3 also require a completed P1 diagnosis.

Those gates apply to the canonical research protocol. `presentation.json` is a separately identified supplemental-data profile: frozen InLegalBERT features, a trainable shared bank/form inducer/fact projection, and initialization from the 62 element references. It does **not** claim the original full-fine-tuning/decomposition experiment is complete. It can train and calibrate independently of a pending historical-baseline comparison. Full source counts are verified; empty records and text duplicates in evaluation splits are excluded and logged. The imported anonymized SIGR-Law supplement is not byte-identical to raw ILSI, so its scores must not be presented as reproduction of the historical 0.700 result.

```sh
python -m research.lexform setup --assets dataset --dataset-source sigr-supplement
python -m research.lexform setup --assets encoder
python -m research.lexform features --config research/lexform/presentation.json
python -m research.lexform train --config research/lexform/presentation.json
python -m research.lexform calibrate --checkpoint artifacts/lexform/runs/TRAIN_RUN_ID
python -m research.lexform bundle --checkpoint artifacts/lexform/runs/TRAIN_RUN_ID
```

Frozen features are cached in ignored SQLite files keyed by encoder revision, pooling and context recipe. Bundling copies only explicitly selected model artifacts and measured results, never the corpus or feature cache. Qwen decomposition is an optional offline source-analysis job for this profile; its exact-quote checked clauses are distinct from the learned prototype programs. Unsupported source forms are logged, never invented to satisfy the template.

```sh
uv run --extra research python -m research.lexform baseline
uv run --extra research python -m research.lexform diagnose --config research/lexform/config.json
uv run --extra research --extra decomposition python -m research.lexform decompose --gguf /absolute/path/model.Q4_K_M.gguf
uv run --extra research python -m research.lexform train --config research/lexform/config.json
uv run --extra research python -m research.lexform ablate --config research/lexform/config.json
uv run --extra research python -m research.lexform calibrate --checkpoint artifacts/lexform/runs/TRAIN_RUN_ID
uv run --extra research python -m research.lexform transfer --checkpoint artifacts/lexform/runs/TRAIN_RUN_ID
uv run --extra research python -m research.lexform report
```

`--device cpu|cuda` and `--run-id NAME` are optional. Change the encoder path, loss, pooling, context, norm or model variant in a separate JSON config; every value is persisted. The default supplied config uses 256 predicates, four groups, 512-token windows, at most eight chunks, batch size two and accumulation eight. GPU fit depends on document lengths, encoder and other resident models; stop the speech service before GPU training if it occupies the budget. The model uses encoder gradient checkpointing and CUDA bfloat16 autocast. No benchmark or memory claim is inferred from the synthetic smoke run.

## Mathematical choices

The form inducer attends to the shared bank and keeps the top four memberships per group, normalized relative to the largest membership. Its full attention distribution receives an entropy sparsity penalty. The forward program is AND across groups and OR within groups. The Gödel option uses normalized weighted log-sum-exp OR and normalized soft-min AND with separately learned temperatures. These approach max/min as temperature approaches zero. Product and Łukasiewicz use their exact fuzzy operators. All outputs remain within [0, 1]. An absent handwritten group is neutral for conjunction.

Element grounding pulls prototypes toward reference descriptions; cached induced clauses initialize additional bank entries. Prototype labels shown in the UI describe initialization and are not guaranteed to describe their final learned meaning. Section and complaint encoders also contain learned information; the bank is not literally the only location of legal knowledge.

Training losses include section BCE (or the selected imbalance ablation), grounding, membership entropy, and permutation-invariant form alignment. Anchors are high-confidence unflagged BNS rows only. Alignment is disabled for fixed handwritten and per-section-bank variants because those cannot induce comparable unseen-code forms.

## Evaluation limits

- P0 compares the executed result to the supplied historical 0.700 micro-F1 ± 0.01. It does not mark the gate passed because a command exists. The dev split selects thresholds; the test split is used only for final reporting.
- P1 uses independent one-factor changes around BCE / CLS / truncation, rather than attributing a collapse to one cause without measurements.
- P2 validates exact source quotations and rejects forms that cannot fit the template. It reports lexical token-F1 coverage of the 62 references; semantic/legal quality still needs manual annotation. The four-group depth-two approximation does not fully capture all statutory exceptions or cross-references.
- P3's handwritten ablation covers only 29 IPC labels. Its metrics are explicitly scoped and cannot be presented as improvements on the full 100-label baseline. Full shared-bank, conjunction, per-section and flat heads use the same encoder and splits.
- P4 calibrates score and weakest-group floors only on development data. Risk is mean one-minus-document-F1 among documents with any retained label. No accepted documents means undefined risk, not zero risk. Matched-coverage comparisons report the nearest observed TF-IDF coverage and its residual.
- P5 groups IPC anchor rows by BNS target before holding out targets, preventing many-to-one target leakage. Anchor-trained models are **mapping-supervised transfer**, not strict zero-shot learning. For the unanchored experiment, set `anchor_weight` to zero and `anchor_fraction` to zero. Report it separately.
- Correspondence provides root-section proxy labels, not independently annotated BNS facts or subsection-level ground truth. Full-358 quality cannot be established from the old 100-label corpus. Only completely mapped cases enter proxy F1. Only cases with exclusively non-BNS source labels enter the unsupported-case abstention metric; mixed cases are not assumed to require total abstention. A missing eligible subset yields a null metric, never a fabricated score.
- The source corpus is court facts, not spoken complaints. Tamil–English transfer remains a separate unmeasured extension.

## Integration

The research API lives at `/v1/research/lexform/{status,runs,statutes,inspect}`. Inspection requires a completed P3 checkpoint and P4 calibration, loads it locally on CPU, and retains the last loaded model. Predictions remain in the research panel and never populate the FIR. API input is not written into experiment files. Start both applications with the existing `./start.sh`.

The provided brief's historical results and novelty assertions remain hypotheses until reproduced or verified. Reports include negative results and unfinished phases. No automatic commits or pushes are performed by experiment commands.
