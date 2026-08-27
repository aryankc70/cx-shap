# CX-SHAP: Concordance-Guided Cross-Domain Explainable AI for Clinical Decision Support

A multi-task clinical risk prediction system (MMNet) paired with CX-SHAP, an
explainability framework that decomposes attributions into shared vs.
task-specific components, measures cross-method (SHAP/LIME) concordance as a
trust signal, and validates attribution directions against sourced clinical
guidelines.

This is an independent rebuild of an M.S. thesis project, reconstructed from
scratch (original code was lost) with an expanded, real-dataset-backed
validation suite.

## What's here

- **MMNet**: a monotonicity-constrained multi-task neural network predicting
  postpartum hemorrhage (PPH), neonatal sepsis, and hypoxic-ischemic
  encephalopathy (HIE) jointly.
- **O-DIL**: three-phase training procedure (representation learning → focal
  loss + oversampling → weighted BCE + threshold tuning) for extreme class
  imbalance (HIE prevalence ~1.5%).
- **CX-SHAP**: four-component explainability framework —
  attribution decomposition, cross-method concordance scoring, temporal
  smoothing, and clinical guideline alignment.
- **4 datasets**: a custom synthetic maternal-neonatal generator, plus 3 real
  clinical datasets (UCI Maternal Health Risk, UCI Cardiotocography,
  PhysioNet 2019 Sepsis Challenge) for end-to-end real-data validation.

## Key findings

- O-DIL improves F1 across all three tasks over a strong BCE baseline
  (largest gain on the rarest class, HIE), rather than rescuing a collapsed
  model — the correlation strength in this dataset avoids the F1=0 failure
  mode reported in the original thesis.
- Cross-method concordance (SHAP vs. LIME) does **not** rise with stronger
  feature-outcome correlation - it stayed in a similar range to the
  original thesis's much weaker-signal dataset, suggesting concordance
  reflects something about decision-boundary complexity independent of
  both model accuracy and data learnability.
- Guideline alignment reveals that weight-clamping monotonicity constraints
  (Runje & Shankaranarayana, 2023) don't propagate end-to-end through
  unconstrained downstream layers - a feature can be monotonic at the input
  layer and still have an inconsistent effect on the final prediction.
- Real-data validation shows an honest synthetic-to-real generalization gap
  (AUC 0.995-0.998 synthetic vs. 0.84-0.98 real), measured directly rather
  than only discussed narratively.

## Project structure
src/
data/ # synthetic generator, real dataset loaders, preprocessing
models/ # MMNet architecture
training/ # baseline BCE, O-DIL, MLflow tracking
explainability/ # all 4 CX-SHAP components + full pipeline
evaluation/ # ablation study, real-data validation, visualization
tests/ # unit tests (12, all passing, run via CI)
results/ # metrics JSON, figures, trained model weights


## Reproducing this work

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Data
python -m src.data.synthetic_generator
python -m src.data.real_data_loader
python -m src.data.physionet_sepsis_loader   # requires downloading .psv files first, see docstring
python -m src.data.preprocessing

# Model
python -m src.training.baseline_train
python -m src.training.odil_train
python -m src.evaluation.ablation
python -m src.evaluation.real_data_validation

# Explainability
python -m src.explainability.attribution_decomposition
python -m src.explainability.concordance
python -m src.explainability.temporal_smoothing
python -m src.explainability.guideline_alignment
python -m src.explainability.cx_shap_pipeline

# Figures + tests
python -m src.evaluation.visualize_results
python -m pytest tests/ -v
```

## Guideline knowledge base

The clinical guideline knowledge base used in Component 4 (`src/explainability/guideline_alignment.py`)
is a deliberately small, well-sourced subset - not exhaustive coverage.
Sources: SIRS criteria (1991 ACCP/SCCM Consensus), Sepsis-3/qSOFA (Singer et
al. 2016), ACOG Practice Bulletin on PPH (2017), FIGO 2015 CTG consensus
guidelines, NICE NG229 (2022). See the module docstring for full citations
and documented exclusions (e.g., blood pressure's genuine bidirectionality
for PPH risk).

## Hardware benchmark: the GPU paradox is batch-size-dependent

The original thesis reported GPU training as ~19x faster than CPU at scale,
with GPU single-sample inference ~8x *slower* (the "GPU paradox"). This
rebuild reran the benchmark independently (Mac CPU vs. Colab T4 GPU) across
5 data sizes (1K-1M) and 3 batch modes, and found a more complete picture:

**Training time is conditional on batch size:**

| Batch mode | CPU (1M rows) | GPU (1M rows) | Winner |
|---|---|---|---|
| batch=64 | 32.96s | 214.11s | CPU, 6.5x |
| batch=2048 | 2.90s | 7.21s | CPU, 2.5x |
| full-batch | 2.83s | 0.29s | **GPU, 9.8x** |

GPU only wins training once batch size and data volume are both large
enough to amortize fixed kernel-launch/transfer overhead - it still loses
at n=1,000 even at full-batch.

**Inference latency never crosses over, in any configuration**: GPU stays
~0.43-0.65ms, CPU stays ~0.03ms, regardless of what batch mode the model
was trained with. This is the unconditional part of the paradox.

The practical implication is more nuanced than "train on GPU, infer on
CPU": for a model this small (3,587 params, 18 features), GPU only helps
training under specific batch/data-volume conditions, and never helps
single-sample clinical inference.

Original thesis benchmark cited for comparison, not reproduced 1:1 (that
version used 3 hardware platforms including a Jetson Orin Nano edge device,
and did not report batch size):

> KC, A. (2026). *CX-SHAP: Concordance-Guided Cross-Domain Explainable AI
> for Clinical Decision Support* [Unpublished master's thesis]. Utica
> University.

## Cross-domain validation

CX-SHAP was applied to finance, manufacturing, and environment domains
using the exact same O-DIL training and CX-SHAP explanation pipeline as
healthcare - zero code changes, only dataset/feature/task configuration
differs.

| Domain | Best AUC | Mean concordance (rho) | Mean guideline alignment |
|---|---|---|---|
| Healthcare | 0.998 | 0.408 | 0.322 |
| Finance | 1.000 | 0.032 | 0.496 |
| Manufacturing | 1.000 | 0.518 | 0.544 |
| Environment | 0.998 | 0.193 | 0.465 |

Finance is the standout: a functionally perfect model (AUC~1.0) paired
with concordance near zero - SHAP and LIME are statistically uncorrelated
despite the model being highly accurate. Concordance also varies *within*
a domain, not just across domains (manufacturing's heat_failure task
scores 0.78/high-trust while tool_wear_failure in the same domain scores
0.30/low-trust) - a finding beyond what the original thesis reported.

Guideline directions for finance/manufacturing/environment are domain
heuristics, not clinically-sourced like healthcare's Component 4 knowledge
base - see `src/explainability/domain_heuristics_rationale.py` for the
reasoning and caveats behind each one.

## Status

Research prototype, not clinically validated. Not intended to inform real
patient care decisions.