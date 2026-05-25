from hyppo.ontology.quality_gates_qa import (
    check_training_data,
    check_coverage,
    check_stability,
    classify_stability,
    apply_realization_coefficient,
)


def test_qa01_insufficient():
    assert check_training_data(100, min_points=180) == "insufficient"


def test_qa01_sufficient():
    assert check_training_data(200, min_points=180) == "sufficient"


def test_qa02_low_coverage():
    assert check_coverage(0.50, min_coverage=0.70) == "low_coverage"


def test_qa02_ok_coverage():
    assert check_coverage(0.80, min_coverage=0.70) == "ok"


def test_qa03_stable():
    assert check_stability(delta_oil_pct=0.01, threshold=0.02) == "stable"


def test_qa03_unstable():
    assert check_stability(delta_oil_pct=0.05, threshold=0.02) == "unstable"


def test_qa04_classes():
    assert classify_stability(0.70) == "stable"
    assert classify_stability(0.50) == "medium"
    assert classify_stability(0.20) == "unstable"


def test_qa05_realization():
    assert apply_realization_coefficient(100.0, 0.60) == 60.0
