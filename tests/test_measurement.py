import numpy as np
import pytest

from worldline_qmc import measurement


def test_measurement_accumulator_basic_stats() -> None:
    acc = measurement.MeasurementAccumulator()
    samples = [np.exp(1j * angle) for angle in (0.0, np.pi / 2, np.pi)]
    for sample in samples:
        acc.push(sample)

    averages = acc.averages()
    assert pytest.approx(averages["re"], rel=1e-9, abs=1e-9) == 0.0
    assert pytest.approx(averages["im"], rel=1e-9, abs=1e-9) == 1.0 / 3.0
    assert pytest.approx(averages["abs"], rel=1e-9, abs=1e-9) == 1.0

    variances = acc.variances()
    assert variances["re"] > 0.0
    assert variances["im"] > 0.0
    assert variances["abs"] == pytest.approx(0.0, abs=1e-12)


def test_measurement_accumulator_rejects_non_finite() -> None:
    acc = measurement.MeasurementAccumulator()
    with pytest.raises(ValueError):
        acc.push(complex(float("nan"), 0.0))


def test_measurement_accumulator_empty() -> None:
    acc = measurement.MeasurementAccumulator()
    assert acc.averages() == {"re": 0.0, "im": 0.0, "abs": 0.0}
    assert acc.variances() == {"re": 0.0, "im": 0.0, "abs": 0.0}
