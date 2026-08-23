from src.data.synthetic_generator import generate_dataset, GeneratorConfig


def test_prevalence_matches_target_within_tolerance():
    config = GeneratorConfig(n_samples=2000, seed=1)
    df, meta = generate_dataset(config)

    assert abs(meta["prevalence"]["pph"] - config.pph_prevalence) < 0.02
    assert abs(meta["prevalence"]["sepsis"] - config.sepsis_prevalence) < 0.02
    assert abs(meta["prevalence"]["hie"] - config.hie_prevalence) < 0.01


def test_reproducible_with_same_seed():
    config = GeneratorConfig(n_samples=500, seed=99)
    df1, _ = generate_dataset(config)
    df2, _ = generate_dataset(config)
    assert df1.equals(df2)