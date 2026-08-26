"""
CX-SHAP Component 4: Guideline alignment. Dataset-agnostic.

Healthcare uses a small, clinically-sourced knowledge base (see
HEALTHCARE_GUIDELINE_KB below - unchanged from Step 12, same citations).
Cross-domain (finance/manufacturing/environment) uses DOMAIN HEURISTICS
from cross_domain_generators.py - explicitly NOT clinically-validated,
applied uniformly across a domain's tasks since we don't have per-task
guideline sources for those domains. This distinction must be preserved
in any write-up: healthcare's alignment score means something different
(guideline-validated) than the cross-domain ones (heuristic-only).
"""

import json
import numpy as np

HEALTHCARE_GUIDELINE_KB = {
    "pph": {
        "heart_rate": {"direction": +1, "source": "Escobar et al. 2022 (FIGO shock index)"},
        "parity": {"direction": +1, "source": "ACOG Practice Bulletin, PPH (2017)"},
        "systolic_bp": {"direction": None, "source": "context-dependent - excluded"},
        "diastolic_bp": {"direction": None, "source": "context-dependent - excluded"},
        "uterine_contractions": {"direction": None, "source": "construct mismatch - excluded"},
    },
    "sepsis": {
        "body_temp_f": {"direction": "abnormal", "source": "SIRS criteria (1991 ACCP/SCCM)"},
        "heart_rate": {"direction": +1, "source": "SIRS criteria (1991 ACCP/SCCM)"},
        "systolic_bp": {"direction": -1, "source": "qSOFA / Sepsis-3 (Singer et al. 2016)"},
    },
    "hie": {
        "fetal_heart_rate": {"direction": "out_of_range_110_160", "source": "NICE NG229 (2022)"},
        "prolonged_decelerations": {"direction": +1, "source": "FIGO 2015 consensus guidelines"},
        "abnormal_short_term_variability": {"direction": "reduced_only", "source": "FIGO 2015; NICE NG229 (2022)"},
    },
}


def build_domain_kb(guideline_dirs_heuristic: dict, task_names: list) -> dict:
    """Applies the same flat heuristic direction dict to every task in a
    cross-domain dataset (documented simplification - see module docstring)."""
    kb = {}
    for task in task_names:
        kb[task] = {
            feat: {"direction": direction, "source": "domain heuristic, not clinically validated"}
            for feat, direction in guideline_dirs_heuristic.items()
        }
    return kb


def check_direction_match(feature_name, direction_spec, shap_value, feature_raw_value=None):
    if direction_spec is None:
        return None
    if direction_spec == +1:
        return shap_value > 0
    if direction_spec == -1:
        return shap_value < 0
    return None  # "abnormal"/"reduced_only"/"out_of_range_*" - not simply checkable via sign alone


def compute_guideline_alignment(phi_per_task, guideline_kb, feature_names):
    """guideline_kb: dict of task_name -> {feature_name: {"direction":..., "source":...}}."""
    results = {}

    for task, phi in phi_per_task.items():
        kb = guideline_kb.get(task, {})
        checkable_features = [f for f, spec in kb.items() if spec["direction"] in (+1, -1) and f in feature_names]

        if not checkable_features:
            results[task] = {"alignment": None, "note": "no simply-checkable features in KB for this task", "n_features_checked": 0}
            continue

        per_sample_scores = []
        for i in range(phi.shape[0]):
            matches = []
            for feat in checkable_features:
                idx = feature_names.index(feat)
                direction = kb[feat]["direction"]
                match = check_direction_match(feat, direction, phi[i, idx])
                if match is not None:
                    matches.append(match)
            if matches:
                per_sample_scores.append(np.mean(matches))

        alignment = float(np.mean(per_sample_scores)) if per_sample_scores else None
        results[task] = {
            "alignment": round(alignment, 4) if alignment is not None else None,
            "n_features_checked": len(checkable_features),
            "features_checked": checkable_features,
            "excluded_features": [f for f, spec in kb.items() if spec["direction"] not in (+1, -1)],
        }

    return results


if __name__ == "__main__":
    from src.explainability.attribution_decomposition import compute_decomposition
    from src.data.synthetic_generator import ALL_FEATURES

    phi_per_task, phi_shared, phi_task_specific, summary = compute_decomposition(
        dataset_name="synthetic", task_names=["pph", "sepsis", "hie"], feature_names=ALL_FEATURES
    )

    results = compute_guideline_alignment(phi_per_task, HEALTHCARE_GUIDELINE_KB, ALL_FEATURES)
    print("\nGuideline alignment results (synthetic, regression check):")
    for task, r in results.items():
        print(f"  {task}: alignment={r['alignment']}  checked={r.get('features_checked')}  excluded={r.get('excluded_features')}")

    with open("results/guideline_alignment_results_synthetic.json", "w") as f:
        json.dump(results, f, indent=2)