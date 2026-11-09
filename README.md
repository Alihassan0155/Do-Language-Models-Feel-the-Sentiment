# Do Language Models Feel the Sentiment?

Code-only research implementation for extracting emotion-related directions,
validating representations, studying interventions, and evaluating supervised
sentiment correction with frozen language models.

**CODE ONLY:** datasets, experimental results, predictions, sample IDs, split
manifests, derived emotion directions, trained classifiers, model weights,
activations, bootstrap draws, logs, and figures are not distributed. The source
generates such material only when you explicitly run it in your own workspace.
This source distribution contains no historical Git metadata. Hosting status
must be verified separately before making an availability claim.

## Installation

Use Python 3.12 and a separate environment. Install a PyTorch CUDA wheel suitable
for your hardware first, then install `requirements.txt`. Dependency bounds are
API constraints, not a verified historical environment lock. No experiment or
model inference was run to validate this prepared release.

```powershell
python -m venv "$env:LOCALAPPDATA\EmotionVectorsEnv"
& "$env:LOCALAPPDATA\EmotionVectorsEnv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies include PyTorch, Transformers, datasets, NNsight, NumPy, pandas,
SciPy, scikit-learn, joblib, PyArrow, psutil, huggingface_hub, and matplotlib.
Install umap-learn separately for optional UMAP plots. The full Amazon runner
uses Windows `msvcrt` locking; it requires Windows unless that locking adapter
is ported and reviewed. CUDA is required by the extraction/intervention paths.
Do not use CPU fallback to claim equivalent GPU numerical results.

## Source and generated-data separation

Root Python files implement the early stages. `pipeline/` contains Python-only
extractions of later-stage implementations. Source paths and data paths are
separate: `release_paths.py` resolves source imports from this release while
generated material belongs to an explicitly selected workspace.

Always launch experiments with:

```powershell
$work = Join-Path $env:LOCALAPPDATA 'EmotionVectorsRuns\new-study'
python run_pipeline.py --workspace $work SCRIPT.py -- SCRIPT_ARGUMENTS
```

Replace uppercase placeholders before execution. Do not point the workspace at
the original research repository or place generated outputs in the source tree.
The launcher does not create a workspace until the selected script needs it.
Some legacy preparation scripts do work at startup, so inspect their source
before execution; invoking arbitrary legacy scripts with `--help` is not a safe
universal dry run. Review source statically without importing experiment modules.

## Dataset and checkpoint acquisition

Obtain authorized copies from the original providers:

- GoEmotions: `google-research-datasets/go_emotions`, simplified configuration.
- SST-2: `stanfordnlp/sst2`.
- TweetEval sentiment: `cardiffnlp/tweet_eval`, sentiment configuration.
- EmpatheticDialogues: the original ParlAI distribution referenced by
  `prepare_empathetic_stage_c.py`.
- Amazon Polarity: `mteb/amazon_polarity`; target preparation uses its test split.

Review dataset terms and privacy obligations before acquisition. No dataset text
or samples are included. Downloaded and processed data must remain private.
Locally generated provenance hashes and IDs are never release assets.

Models are Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct,
Qwen/Qwen2.5-3B-Instruct, and the separate google-bert/bert-large-uncased control.
Acquire and record explicit provider revisions. For example:

```powershell
python run_pipeline.py --workspace $work load_assets.py -- --datasets
python run_pipeline.py --workspace $work load_assets.py -- --model 0.5b --revision YOUR_REVIEWED_REVISION
```

Repeat acquisition for all required models. The helper records resolved provider
checkpoint commits in a private workspace revision map. Model/tokenizer loads
that previously depended on cache defaults now require this map; explicit later
protocol revisions remain pinned. A conflicting model revision requires a fresh
workspace. This adaptation changes provenance enforcement, not model arithmetic,
but new revisions must not be called historically equivalent.

## Execution order and interfaces

All commands below are for a future authorized run, not release verification.
Keep the same workspace and dataset revisions throughout a run. Legacy exact
count assertions deliberately reject incompatible input versions.

### 1. Sentiment provenance and emotion extraction

The historical Stage A preparation chain is:
`evaluate_all_templates.py` -> `evaluate_sst2_train_12k.py` ->
`merge_sst2_results.py` -> `bert_stage_a.py` -> `qwen3b_stage_a.py` ->
`report_four_models.py`. Use the scripts' checkpoint options to keep the expected
local layouts. These stages generate local predictions and bind sample
provenance used by Stage D/E. They are not shipped predictions.

The emotion path is `select_goemotions_b1.py` -> `prepare_goemotions_b2.py` ->
`build_stage_b_vectors.py` (Qwen) / `build_stage_b_bert_vectors.py` (BERT) ->
`build_stage_b_final_vectors.py`. These CLI scripts expose model, batch-size,
and output options where supported. Preserve raw text-token pooling and the
strict single-label selection policy. Denoising removes layer-specific neutral
PCs explaining at least half the neutral variance, rather than a fixed rank.

```powershell
python run_pipeline.py --workspace $work select_goemotions_b1.py
python run_pipeline.py --workspace $work prepare_goemotions_b2.py
python run_pipeline.py --workspace $work build_stage_b_vectors.py -- --help
python run_pipeline.py --workspace $work build_stage_b_final_vectors.py -- --help
```

### 2. Validation, selection, geometry, and probes

Run `run_stage_c_goemotions.py`, `prepare_empathetic_stage_c.py`, and
`run_stage_c_empathetic.py`, in that order, for the required models. The final
script saves all eligible three-layer-window selection scores and the maximizing
center. Both emotion-validation datasets inform selection. Endpoint layers are
not eligible centers; EmpatheticDialogues selected-layer performance is not an
untouched external test.

Run `stage_d.py` after generating Stage A provenance. Run `run_stage_e.py`
with `--phase prepare`, `extract`, `fit`, `evaluate`, and `report`, or explicitly
choose `--phase all`. Then run `stage_f.py` and, optionally,
`stage_f_context_diagnostics.py`. `analyze_stage_b_geometry.py` provides the
direction-geometry analysis. Stage E uses training-only scaling/PCA and a
validation-selected logistic-regression regularization grid; it does not refit
on validation data. Do not use `--reset` on material you intend to retain.

### 3. Steering and ablation

Run `stage_g.py` for pilot steering, then `stage_g_confirmation.py` for its
development control and confirmation. The latter exposes `--phase` and `--model`;
respect the positive-control gate. Run `stage_h.py --phase all` for full subspace
ablation after the required confirmation sample preparation. Full removal is
uncentered by default; a training-centered sensitivity option is explicit.

Run `stage_h_distortion_matched.py --phase all` for TweetEval matched ablation.
For SST-2, use `pipeline/e2_sst2_distortion_matched_20261018/run_e2.py` with
`--phase prepare`, then `smoke`, then `run`. It builds a new training-derived
holdout and excludes previously used rows/texts in the current workspace.
It is not the same population as the original validation-proxy full ablation.

Matched removal retains the original per-example rounded perturbation-energy
calculation, capped coefficient, bisection, and tolerances. Historical TweetEval
instructions allow neutral. For its binary readout, use
`pipeline/binary_reanalysis_20261019/reanalyze.py`. The separate
`pipeline/asd_ste117_binary_prompt/run117.py` (`--phase prepare`, `smoke`,
`full`, `matched`, or `all`, as defined by its parser) changes the prompt as
well and must not be substituted silently for the historical analysis.

### 4. Source correction

After Stage E extraction, create local margins with the new orchestration:

```powershell
python run_pipeline.py --workspace $work prepare_correction_inputs.py -- --execute --batch-size 8
python run_pipeline.py --workspace $work pipeline/stage_i_sentiment_correction/stage_i_a.py
python run_pipeline.py --workspace $work pipeline/stage_i_emotion_guided_correction/stage_i_b.py -- --phase all
python run_pipeline.py --workspace $work pipeline/stage_i_sst2/stage_i_sst2.py -- --phase all
python run_pipeline.py --workspace $work pipeline/stage_i_tweeteval_matched_controls/stage_i_tweeteval_controls.py -- --phase all
```

The orchestration computes new local binary margins; it does not recover private
scores. It uses FP16 SDPA and float32 log-softmax/subtraction. Historical comparison
tables were removed from the first margin-correction reporter. Fitting, validation
selection, train-only scaling, rank controls, and paired intervals are retained.
Exact source-count checks remain and may require matching provider revisions.

### 5. Frozen Amazon transfer

```powershell
python run_pipeline.py --workspace $work prepare_transfer_config.py -- --execute
python run_pipeline.py --workspace $work pipeline/stage_k_amazon_transfer/stage2_dataset/prepare_amazon_stage2.py
python run_pipeline.py --workspace $work pipeline/stage_k_amazon_transfer/stage3a_pilot/run_stage3a.py
python run_pipeline.py --workspace $work pipeline/stage_k_amazon_transfer/stage3a_pilot/verify_stage3_resume.py
python run_pipeline.py --workspace $work pipeline/stage_k_amazon_transfer/stage3b_full_inference/run_stage3b_full.py
python run_pipeline.py --workspace $work pipeline/stage_k_amazon_transfer/stage4_evaluation/evaluate_stage_k4.py
```

The new configuration builder inventories only locally generated source heads
and identities. Target labels are stripped before inference. Predictions are
frozen before the evaluation label join. No target fitting/selection is allowed.
The same frozen target sample is used by every source/model configuration.
Pickle/joblib artifacts must come only from your trusted local run.

### 6. Statistics

The source-stage reporters implement their paired class-stratified intervals.
For final matched example/control-seed resampling, run
`pipeline/matched_ablation_bootstrap_50000_20261104/analyze.py` only after both
matched score databases exist. It retains 50,000 draws, Holm step-down p-values,
nominal percentile intervals, and Bonferroni basic simultaneous intervals.
Bonferroni intervals are not Holm-adjusted intervals. Historical interval parity
checks are omitted: comparison draws are recomputed from your local predictions.

## Seeds and numerical constraints

Default sampling/training seed is 42; confirmation sampling uses 43. Pilot and
probe random directions use 0-9; confirmation uses 1000-1049; rank-matched/full
ablation controls use 2000-2049. E2 sampling uses 20261018. Matched bootstrap uses
example RNG 45 and paired-seed RNG 46. Full-ablation example bootstrap uses 44;
source correction and Amazon use 47. EmpatheticDialogues selection uses the
string seed defined in its preparation source. These are algorithm settings,
not published evaluation sample identifiers.

Extraction is generally FP16 with float32 pooling; projections/fitting use
float64. Confirmation/full ablation for Qwen 1.5B uses FP32. Selected layers are
read from your local Stage C selection rather than published historical values.
Inference batch changes, dependency changes, kernels, checkpoint revisions,
dataset revisions, and different prior-workspace exclusions can alter outputs.
Do not claim bitwise or sample-identical historical reproduction from this
code-only release.

## Release adaptations and limits

No models, data downloads, evaluations, training runs, or statistical simulations
were executed to prepare this directory. Validation is static only. There are
no published experimental values or private artifact copies in the distribution.

### Preserved implementation

The release extracts original Python algorithms for raw emotion contrasts,
neutral covariance/eigendecomposition, matched-seven AUROC layer selection,
training-only probes, Gaussian QR controls, post-block token interventions,
rounded-energy bisection, source correction, frozen transfer, and bootstrap/Holm
statistics. Python AST formatting was normalized during extraction; source bytes
are therefore different even where expressions and computations are unchanged.

### Changes that can affect reproduction

- Workspace and source paths are separated. Use the launcher; do not rely on
  historical absolute paths or run-directory-relative imports.
- Dataset digests and target revisions are generated/read locally, not embedded.
  Provider revisions must be reviewed before running. Loads without a source
  revision now use the private locally recorded acquisition map; missing entries
  fail rather than fall back to a moving provider default.
- Later correction/binary runners read the locally selected layer rather than
  assuming historical selected-layer values. Different validation inputs can
  therefore select different layers.
- New margin-input preparation uses explicit FP16 SDPA. It preserves candidate
  IDs, readout position, float32 subtraction, and prompt definition, but cannot
  establish equivalence to every original attention kernel or batching pattern.
- The asset helper now supplies missing baseline loader exports and defers all
  acquisition to explicit calls. Its baseline loader uses left padding and
  model eval mode. This repairs an incomplete original helper rather than proving
  exact historical baseline equivalence.
- Margin correction no longer requires unpublished historical comparison
  tables. Those comparisons are omitted; fitting and current-run metrics remain.
- The final matched bootstrap no longer verifies old saved intervals. It
  recomputes its smaller-draw comparison using the same current-run predictions.
- The binary-prompt runner uses the shipped binary engine rather than fragile
  text replacement of a differently formatted source file.
- Amazon setup generates its source-head inventory from local runs. Manually
  authored historical planning reports are no longer preflight prerequisites.
  Numerical/model/data checks remain. Resume verification explicitly records its
  completion so the full runner can check it.
- Requirements are API bounds, not a complete historical lock. Different BLAS,
  CUDA, PyTorch, Transformers, sklearn, or NumPy versions can change behavior.

### Prerequisites and assumptions

The extracted entrypoints retain exact dataset-count/shape assertions and
historical directory naming as runtime interface contracts. These are not
distributed data. Fresh runs must generate the expected upstream protocols,
scores, bases, and heads in order; they are intentionally absent initially.

The source-only adaptation has not been integration-tested. Static correctness
does not prove runtime compatibility, successful GPU execution, numerical parity,
or availability of a particular provider revision. Check the first local run
stage by stage before making reproducibility claims.

Historical source exposure, review-level independence, and checkpoint identity
for original extraction cannot be established by this release. Validation data
informed selection, and original prompt/readout/population variants must remain
distinct. Matching perturbation energy does not equate all directional damage
or establish uniquely emotional causal necessity.

## Hardware and rights

Use a CUDA GPU with sufficient memory for model weights, padded inputs, logits,
and intermediate activations. Smaller batches reduce memory demand but may
change numerical rounding. Keep generous disk space for private caches. No
universal GPU-capacity or runtime guarantee is made.

Dataset and model terms apply independently of this source. No raw or derived
research content is redistributed. A source license has not been selected;
public access alone does not grant a software license. Ownership, third-party
attribution, dataset permissions, and model-license review remain the owner's
responsibility before publication.
