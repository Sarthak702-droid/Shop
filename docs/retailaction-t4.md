# RetailAction training on Colab T4

The experiment is implemented under `ml/retailaction/` and launched with
`notebooks/Shop_RetailAction_T4.ipynb`. It is an offline three-label **clip-presence** experiment,
not a replacement for MERL's five-class model or a Shop runtime integration.

## Required sequence

1. Finish training-archive upload to Drive; all three archive sizes and SHA-256 values must match
   the publisher. The notebook fails before training if this gate is incomplete.
2. Check sample media/metadata pairing and the full metadata quality report. Actual source poses
   have missing observations and timestamps; some sequences exceed 32 observations. Versioned
   masks preserve these facts. Inconsistent records are quarantined and exclusions are reported.
3. Run the small T4 learning pilot. No local CPU model training is substituted.
4. Train on train, select on validation, and keep separate RetailAction checkpoints on Drive.
5. Finish validation comparisons, then freeze one model for the separate final-test command.
6. Validate compatible feature extraction, target Pop!_OS CPU speed and intended-store utility
   before enabling any integration. The current runtime remains unchanged and disabled for this model.

## Colab operation

Open the notebook in Colab, select a T4 GPU runtime and mount your own Drive. The default root
is `/content/drive/MyDrive/ShopliftingAI-Data`; adjust it only if this folder is mounted elsewhere.
Place the matching `shop_retailaction_t4_source.zip` in the root folder's `notebooks/` directory.
The notebook verifies its embedded bundle hash before importing any code.

Do not install `requirements-cpu.txt` into Colab. Use its existing CUDA-enabled PyTorch. Colab's
T4 availability and session duration are not guaranteed; no paid upgrade is automatic.

Use the same `RUN_ID` after interruption. The notebook resumes matching last checkpoints and
rejects a changed code/schema/config. New validation experiments use a new run ID. The global
final-test registry prevents using test scores to choose another checkpoint. Final-test execution
is disabled while validation experiments are being explored. Keep `MODE='train'` for the pilot
and full training. After settling validation choices, use `MODE='evaluate'` with the same
`RUN_ID` in a fresh T4 runtime and rerun the notebook. This mode stages only the test archive,
skips training, and frees notebook-owned local development caches. Final reports and predictions
are retained under the shared Drive root. Restart the runtime when switching source bundles.

## Evidence and current limitations

Actual local checks are recorded in `var/retailaction/preprocessing_pilot.json` and the test run.
A passing preprocessing pilot only establishes finite feature construction and the source scan;
it is not GPU training, validation accuracy, an end-to-end video benchmark, or release approval.
The notebook's GPU execution is explicitly unverified until a real Colab run is recorded.

Both views remain one sample. Features come from the publisher's pose data, with a new
100-channel schema. No trained RetailAction checkpoint is supplied. No inference adapter is
added to Shop. See `ml/retailaction/README.md` for exact feature, quality and evaluation contracts.

Sources:
- [Original RetailAction card](https://huggingface.co/datasets/standard-cognition/RetailAction/blob/main/README.md)
- [Original license](https://huggingface.co/datasets/standard-cognition/RetailAction/blob/main/LICENSE)
- [Colab resource limits](https://research.google.com/colaboratory/faq.html)

Use only one active final-test evaluator across all Colab runtimes sharing the dataset registry.
The file registry protects sequential model selection; it is not a distributed lock across
machines. Completion records, code fingerprints and immutable checkpoint snapshots are checked
before final-test labels are read. A resumed epoch budget can be extended; patience and other
saved training settings cannot be changed within the same run.
