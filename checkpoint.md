# Development checkpoint — 16 October 2026

## Working implementation

- Existing local speech recording/upload, playback, retained audio, direct English translation, officer review and IF-1 drafting are preserved.
- LexForm is implemented beside the production legal pipeline: a 256-predicate shared bank, statute-text form inducer, fact activations, four AND-of-OR groups, three fuzzy-logic families, element grounding, filtered mapping-anchor alignment and weakest-group abstention.
- The frontend now contains only **Inspect facts** and **Statutory forms**. Overview, phase indicators and experiment tables have been removed.
- Inspection can use the verified English working text or typed facts. IPC and BNS run through the learned architecture; predictions never automatically populate the FIR.
- Frozen encoder features, lightweight learned-head checkpoints, statutory feature precomputation, portable bundling and resident model loading are implemented. Inference is offline after initial public-model setup.
- The canonical experiment commands, decomposition cache, independent diagnosis, five model variants, calibration, transfer proxy evaluation and result logging remain available in the CLI.

## Actual local inputs and training

- Downloaded the public anonymized SIGR-Law ILSI supplement, not the original raw Zenodo files. Source counts: 42,835 / 10,200 / 13,039; 100 IPC section texts. Provenance and hashes are retained locally.
- Usable splits after explicit empty-record and evaluation-overlap exclusions: **42,750 train / 10,037 dev / 12,794 test**. No raw case text or feature cache is committed.
- Downloaded and verified the pinned public InLegalBERT encoder. Document features for 65,146 unique texts are prepared locally.
- `research/lexform/presentation.json` trains the real bank, form inducer and projection layers on the usable full corpus, with the base encoder frozen and the bank initialized from the existing 62 element references. It is not a rule-based or synthetic demo. This recipe differs from the original full-fine-tuning/decomposition experiment and must be identified as such.
- The official quantized Qwen2.5-7B model and CUDA-capable llama-cpp-python are available locally. Full 458-section teacher decomposition has not been executed. Learned prototype programs and source-quoted teacher decompositions are distinct artifacts.
- Supplemental TF-IDF baseline completed: **0.5043 micro-F1 / 0.3901 macro-F1** on 12,794 usable test cases (run `20261016T021922-5ee080f6`). These are measured supplemental-data scores, not the historical raw-ILSI benchmark.
- `lexform-grounded-v1` completed 12 real training epochs: **0.3599 micro-F1 / 0.1284 macro-F1**, 58/100 labels used at the development-selected score threshold 0.525. This first frozen-encoder model does not beat TF-IDF. Its learned weights, programs and statutory vectors are bundled in the repository; the public base encoder remains a separate initial download.
- Calibration completed using development data. **No operating point satisfied target risk 0.25 with at least 30 retained cases.** The conservative gate therefore abstains on every candidate (threshold above 1); ranked compatibility scores and group inspection still work. Do not loosen the gate merely to make the presentation look successful, or describe ranked candidates as validated recommendations.

## Verification

- 18 automated tests pass, including operator semantics/gradients, mapping safeguards, split exclusions, frozen-feature caching and actual checkpoint round-trip on an explicitly synthetic test fixture.
- Frontend production build passes. Vite reports the existing large-JavaScript-chunk advisory; it is not a build failure.
- Lockfile consistency and Python compilation checks pass.
- Live application health passes with speech, direct audio translation, text translation and the existing classifier ready. Both IPC and BNS inspection endpoints return eight real scored candidates with four groups each (about 50–60 ms for the short smoke input after startup).
- Browser verification confirms only the two functional LexForm tabs remain, and **Evaluate elements** returns the real model response. The catalog contains 456 learned programs in bundled v1; the subsection-collision explanation below accounts for the two missing distinct IPC forms.
- Bundled-only inference was tested with an empty alternate experiment-output directory: the shipped model loads and produces eight four-group BNS candidates without the local training artifacts. The corrected parser produces 100 unique IDs for future training.

## Still to do

1. **Retrain after the subsection-normalization fix included in this commit.** Final catalog verification caught the initial run collapsing IPC 294(b) into 294 and 376(2) into 376. Both completed runs above used that earlier parser: 100 output columns represented only 98 unique IPC roots. Their measured scores are diagnostic artifacts, not valid results for the planned distinct-100-label benchmark. The source parser now preserves subsection suffixes and regression tests cover both collisions. The bundled v1 deliberately retains its original metadata/weights; it is not relabeled as a corrected model. Its catalog shows 98 unique IPC roots + 358 BNS provisions. Run a fresh named training job and baseline before drawing research conclusions.
2. Improve and validate model accuracy; a functional trained prototype is not evidence of legal reliability or superiority to TF-IDF. A short phone-theft smoke input returned irrelevant top rankings in v1; this is an actual observed weakness, not a validated demo case.
3. Execute and manually review all 458 source decompositions; run the controlled encoder diagnosis and all model/operator ablations.
4. Complete the full held-out IPC→BNS proxy evaluation and risk–coverage analysis. Mapping supervision is not independent BNS ground truth or strict unsupervised zero-shot transfer.
5. Validate short spoken-complaint inputs and noisy Tamil–English speech. Court-fact performance does not establish police-booth performance.
6. Establish independently reviewed explanations, legal review of forms, and deployment-grade access controls/retention policy. Prototype descriptions currently identify initialization, not verified learned legal semantics.
7. Reproduce historical numbers only on the original raw fixed splits. Supplemental anonymized data cannot substantiate a claim of reproducing the historical 0.700/0.580 scores.

## Usage

Run `./start.sh`. Record/upload audio, verify the English text, open **LexForm → Inspect facts**, choose **Use English working text**, select **IPC** or **BNS**, then **Evaluate elements**. Expand a result to see group scores; **View provision** opens its program and statute text. Retained/abstained is a calibrated research output, not a filing decision.

On a fresh machine, install the existing full dependencies and fetch the public base encoder once with `python -m research.lexform setup --assets encoder`. The research commands and caveats are in `research/lexform/USAGE.md`. Audio, case data, feature caches and multi-gigabyte base/teacher weights remain excluded from Git.
