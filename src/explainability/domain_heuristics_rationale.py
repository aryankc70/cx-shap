"""
Rationale for cross-domain guideline heuristic directions (used by
guideline_alignment.build_domain_kb()). These are NOT clinically or
regulatorily sourced like healthcare's Component 4 KB - they are
domain-reasonable heuristics, documented here for transparency.

This module has no executable logic; it exists so the reasoning
behind each direction is written down somewhere citable, rather than
being an implicit, undocumented assumption baked into a dict.
"""

RATIONALE = {
    "finance": {
        "Amount": "+1: larger transaction amounts are disproportionately represented in fraud (common finding across public fraud-detection literature, e.g. Kaggle ULB dataset EDA).",
        "V1": "-1, V2: -1, V3: +1, V4: +1, V14: -1 — these are anonymized PCA components in the real ULB dataset; direction assignments here are heuristic placeholders based on typical published feature-importance rankings for that dataset, NOT independently verified against a labeled source (PCA components have no inherent real-world meaning to cite a guideline for).",
    },
    "manufacturing": {
        "tool_wear": "+1: monotonic wear accumulation is a standard predictive-maintenance assumption (higher wear -> higher failure risk), consistent with the AI4I 2020 dataset's TWF definition.",
        "torque": "+1: higher torque increases mechanical stress; also appears in the AI4I 2020 power-failure (PWF) definition at both high and low extremes, simplified here to a single direction.",
        "rotational_speed": "-1: in the AI4I 2020 heat-failure (HDF) definition, LOWER rotational speed combined with low temp differential increases failure risk; direction here reflects that specific interaction, not a universal claim.",
        "air_temperature": "+1: higher ambient temperature reduces the process-air temperature differential in the AI4I 2020 HDF definition, increasing heat-failure risk.",
    },
    "environment": {
        "pm25": "+1, no2: +1, o3: +1, co: +1, so2: +1 — all follow directly from WHO Air Quality Guidelines (2021): higher pollutant concentration always increases health-risk classification by definition of the threshold-exceedance task itself (tautological direction, included for completeness rather than as a novel finding).",
        "wind_speed": "-1: higher wind speed increases pollutant dispersion, a standard atmospheric-science relationship (not a specific WHO citation, general physical dispersion principle).",
        "rainfall": "-1: precipitation scavenging (wet deposition) reduces particulate concentration, a standard atmospheric-science relationship.",
    },
}