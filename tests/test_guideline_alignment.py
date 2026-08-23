from src.explainability.guideline_alignment import check_direction_match


def test_direction_match_positive():
    assert check_direction_match("f", +1, 0.5) is True
    assert check_direction_match("f", +1, -0.5) is False


def test_direction_match_negative():
    assert check_direction_match("f", -1, -0.5) is True
    assert check_direction_match("f", -1, 0.5) is False


def test_direction_none_returns_none():
    assert check_direction_match("f", None, 0.5) is None


def test_direction_special_string_returns_none():
    # "abnormal" / "reduced_only" / "out_of_range_*" specs aren't simply
    # checkable with sign alone - documented simplification.
    assert check_direction_match("f", "abnormal", 0.5) is None