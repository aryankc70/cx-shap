from src.explainability.concordance import trust_level


def test_trust_level_thresholds():
    assert trust_level(0.85) == "high"
    assert trust_level(0.70) == "high"
    assert trust_level(0.55) == "moderate"
    assert trust_level(0.40) == "moderate"
    assert trust_level(0.20) == "low"
    assert trust_level(0.0) == "low"
    assert trust_level(-0.3) == "unreliable"