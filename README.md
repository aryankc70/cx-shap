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

## Status

Research prototype, not clinically validated. Not intended to inform real
patient care decisions.