# RetailAction offline training experiment

This is a separate **three-label clip-presence baseline** (`take`, `put`, `touch`), not MERL
fine-tuning, a temporal-localization benchmark, or a shoplifting detector. Both camera views
form one example. The existing MERL model, five-class head, configuration, and application
runtime remain unchanged. No new model is enabled by this experiment.

## Verified contracts and deliberate limits

- Original publisher archives are checked against pinned SHA-256 values. Training refuses to
  run until `TRANSFER_STATUS.json` confirms all three Drive archives are verified.
- Pose data is supplied by the dataset, not by Shop's CPU extractor. A future integration must
  independently validate pose generation, subject association, feature compatibility and cost.
- Actual training metadata contains 24 named joints, null pose observations, missing timestamp
  arrays and some sequences longer than the dataset card's stated 32. Features use a separate
  `retailaction_sparse_pose_clip_v1` contract with 100 channels and bounded padding to 128
  observations. No truncation, five-Hz interpolation or invented physical timestamps occurs.
- Each joint has two coordinates in the supplied order, confidence and validity. Off-image,
  missing and nonfinite coordinates are masked. Physical dt/elapsed values each have masks.
  The proposed 60-second clipping scale is frozen in the schema; it is not an optimality claim.
- Only presence of annotated action classes within a clip is supervised. Action interval
  interpretation is deliberately **not** used as frame-level ground truth. Missing annotation
  containers are quarantined; explicit empty annotation lists mean no labeled action.
- Invalid metadata and examples with no usable pose are explicitly reported. The proposed
  quality gate rejects a split with over 1% unusable clips or over 10% removal within any class. Review exclusions, especially for
  rare classes, before accepting any result; the threshold is an engineering starting point.
- Official split membership is retained. This follows the publisher's shopper split; it does
  not establish a held-out-store test. Training code never reads final-test labels.
- Reported AP is per paired clip, including tied scores and class support counts. It is not
  comparable with MERL per-step or temporal-segment mAP. Sigmoid scores are uncalibrated.

## Colab T4 workflow

Use `notebooks/Shop_RetailAction_T4.ipynb`. It preserves Colab's CUDA-enabled PyTorch rather
than installing `requirements-cpu.txt` or the entire application. The inference environment
in Pop!_OS remains CPU-only. No automatic paid upgrade or other accelerator fallback exists.

1. Wait for the dataset upload gate to be complete.
2. Select **Runtime → Change runtime type → T4 GPU** and connect your Drive.
3. Verify the code bundle and local copies of training/validation archives.
4. Run the 32-training-clip learning pilot. Finite gradients and decreasing tiny-subset loss
   must pass. This is a learning check, not a model-quality or accuracy claim.
5. Run the full experiment, select the best checkpoint using validation only, and review
   class counts, coverage exclusions, class AP, learning curves and constant-score baselines.
6. Freeze model and schema before `ml/retailaction/evaluate.py` reads final-test labels.
   A Drive registry reserves the test for one frozen model and blocks reuse for model selection.
7. Benchmark on the actual target CPU and intended store footage before any runtime change.

The notebook runs the full experiment after the upload/schema/quality and learning-pilot gates
pass. `RUN_FULL_TRAINING=False` can be used for a pilot-only session. No paid runtime is purchased.

The trainer refuses CPU or non-T4 execution. `last.pt` contains model, optimizer, AMP scaler,
epoch, the exact committed best weights and history; `best.pt` is selected on validation. Save both to the separate RetailAction
Drive directory every epoch. If `last.pt` exists, rerun with `--resume` and the same output directory after a
Colab interruption. If neither `last.pt` nor `best.pt` exists, rerun without `--resume`.
If only `best.pt` exists, restore the matching last checkpoint or start a new run ID. A code/schema/config mismatch rejects resume. Source archives are retained
on Drive; local Colab copies are ephemeral. The epoch budget may be extended on resume; patience and other saved settings must match.
A completion marker binds final evaluation to a finished validation run. Use only one Colab
runtime per Drive run directory; local advisory locks prevent concurrent writers within that
runtime and recover after process exit. Exact bitwise reproducibility is not promised.

## Local checks (no real-data model training)

```bash
.venv/bin/python -m pytest tests/test_retailaction.py tests/test_tcn.py tests/test_colab_workflow.py -q
.venv/bin/python ml/retailaction/train.py --help
```

A real-data preprocessing pilot can run locally without CUDA; actual model training cannot.
A new checkpoint has a separate `shop_retailaction_clip_tcn_v1` format and cannot be silently
loaded as the existing MERL checkpoint.

## Commercial status

Development/research experiment only; release approval remains false. Preserve Standard AI's
attribution and deidentification obligations and review its additional commercial terms.
Never reinterpret these labels as criminal intent, payment status or confirmed theft.

Sources:
- https://huggingface.co/datasets/standard-cognition/RetailAction/blob/main/README.md
- https://huggingface.co/datasets/standard-cognition/RetailAction/blob/main/LICENSE
- https://research.google.com/colaboratory/faq.html

Final-test reservation assumes exactly one active evaluator across **all** Colab runtimes
sharing the registry, not merely one process per run ID. Local locks do not serialize separate
machines. The registry rejects subsequent selection reuse in this supported single-evaluator
workflow. The evaluator checks the completed-run marker, exact checkpoint snapshot, implementation
fingerprint and weights before reserving/reading test labels.

The final-test registry path is derived from the shared `--dataset-root` and official test
archive SHA-256; there is no per-experiment registry override. Use the same selected dataset
root for every validation experiment. Existing final outputs are checked before returning success.
