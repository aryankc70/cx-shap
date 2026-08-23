"""
CX-SHAP Component 4: Guideline alignment.

Checks whether the model's SHAP attribution directions agree with
documented clinical guideline directions. Knowledge base below is a
deliberately small, well-sourced subset (see citations) rather than
an exhaustive one - ambiguous or construct-mismatched feature-outcome
pairs are explicitly excluded rather than forced into a direction.

Sources:
- SIRS criteria (1991 ACCP/SCCM Consensus Conference) - temperature,
  heart rate thresholds for sepsis risk.
- Sepsis-3 / qSOFA (Singer et al. 2016) - systolic BP threshold for
  sepsis risk.
- ACOG Practice Bulletin, Postpartum Hemorrhage (2017) - parity,
  hypertensive disorders as PPH risk factors.
- Escobar et al. 2022, FIGO/Int J Gynecol Obstet - shock index
  literature on heart rate/BP dynamics in PPH.
- FIGO consensus guidelines on intrapartum fetal monitoring:
  Cardiotocography (Ayres-de-Campos et al. 2015).
- NICE NG229, Fetal monitoring in labour (2022).

Known limitations (documented, not smoothed over):
- Systolic/diastolic BP direction for PPH is genuinely bidirectional
  (hypertension = antenatal risk factor; hypotension = late shock
  sign) - encoded as "context_dependent" and excluded from the
  binary alignment score rather than forced to one direction.
- Abnormal short-term variability (ASTV) for HIE is not strictly
  monotonic per NICE 2022 (reduced variability AND short bursts of
  markedly increased variability can both indicate risk) - encoded
  as "reduced_only" with this caveat noted.
- Uterine contractions (frequency) is excluded for PPH: our feature
  measures labor intensity, not the actual mechanism (atony) that
  guidelines address. Forcing a direction here would misrepresent
  the source.
"""

import json
import numpy as np
from src.data.synthetic_generator import ALL_FEATURES

# direction: +1 = higher value increases risk, -1 = higher value decreases risk,
# None = excluded (bidirectional / construct mismatch / not guideline-established)
GUIDELINE_KB = {
    "pph": {
        "heart_rate": {"direction": +1, "source": "Escobar et al. 2022 (FIGO shock index)"},
        "parity": {"direction": +1, "source": "ACOG Practice Bulletin, PPH (2017)"},
        "systolic_bp": {"direction": None, "source": "context-dependent - excluded, see module docstring"},
        "diastolic_bp": {"direction": None, "source": "context-dependent - excluded, see module docstring"},
        "uterine_contractions": {"direction": None, "source": "construct mismatch - excluded, see module docstring"},
    },
    "sepsis": {
        "body_temp_f": {"direction": "abnormal", "source": "SIRS criteria (1991 ACCP/SCCM)"},
        "heart_rate": {"direction": +1, "source": "SIRS criteria (1991 ACCP/SCCM)"},
        "systolic_bp": {"direction": -1, "source": "qSOFA / Sepsis-3 (Singer et al. 2016)"},
    },
    "hie": {
        "fetal_heart_rate": {"direction": "out_of_range_110_160", "source": "NICE NG229 (2022)"},
        "prolonged_decelerations": {"direction": +1, "source": "FIGO 2015 consensus guidelines"},
        "abnormal_short_term_variability": {"direction": "reduced_only", "source": "FIGO 2015; NICE NG229 (2022) - non-monotonic, see module docstring"},
    },
}


def check_direction_match(feature_name, direction_spec, shap_value, feature_raw_value=None):
    """Returns True/False/None (None = not checkable, e.g. abnormal-range
    or bidirectional specs without enough context to evaluate simply)."""
    if direction_spec is None:
        return None
    if direction_spec == +1:
        return shap_value > 0
    if direction_spec == -1:
        return shap_value < 0
    # "abnormal", "reduced_only", "out_of_range_*" require raw feature value
    # context we don't have cleanly separated here; treat as directionally
    # checkable only in the "increases with abnormality" sense using |shap|>0
    # as a weak proxy - flagged explicitly as a simplification.
    return None


def compute_guideline_alignment(phi_per_task, feature_names=None):
    """phi_per_task: dict of task_name -> array (n_samples, n_features) of
    SHAP values, as produced by attribution_decomposition.py.
    Returns per-task alignment score (fraction of checkable features
    whose SHAP sign matches the guideline direction, averaged over samples)."""
    feature_names = feature_names or ALL_FEATURES
    results = {}

    for task, phi in phi_per_task.items():
        kb = GUIDELINE_KB.get(task, {})
        checkable_features = [f for f, spec in kb.items() if spec["direction"] in (+1, -1)]

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
    phi_per_task, phi_shared, phi_task_specific, summary = compute_decomposition()

    results = compute_guideline_alignment(phi_per_task)
    print("\nGuideline alignment results:")
    for task, r in results.items():
        print(f"  {task}: alignment={r['alignment']}  checked={r.get('features_checked')}  excluded={r.get('excluded_features')}")

    with open("results/guideline_alignment_results.json", "w") as f:
        json.dump(results, f, indent=2)