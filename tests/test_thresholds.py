"""教学容限判据的单元测试（对应 quantum_circuit_studio/thresholds.py）。"""

import pytest

from quantum_circuit_studio.thresholds import (
    compose_bitflip,
    flip_pattern_probability,
    ghz_tvd,
    support_mass,
    teaching_threshold,
)


def test_compose_bitflip_matches_single_and_double_application():
    assert compose_bitflip(0.1, 0) == 0.0
    assert compose_bitflip(0.1, 1) == pytest.approx(0.1)
    assert compose_bitflip(0.1, 2) == pytest.approx(0.18)


def test_flip_pattern_distribution_is_normalised():
    for num_wires in (2, 3, 4, 5):
        total = sum(
            flip_pattern_probability(num_wires, 0.1, pattern)
            for pattern in range(1 << num_wires)
        )
        assert total == pytest.approx(1.0, abs=1e-12)


def test_ghz_tvd_matches_known_values():
    assert ghz_tvd(2, 0.0) == pytest.approx(0.0)
    assert ghz_tvd(2, 0.1) == pytest.approx(0.18, abs=1e-12)
    assert ghz_tvd(4, 0.02) == pytest.approx(0.11376577, abs=1e-6)
    assert ghz_tvd(8, 0.1) == pytest.approx(0.76595, abs=1e-5)


def test_tvd_is_monotonic_in_noise():
    for num_wires in (2, 4, 8):
        values = [ghz_tvd(num_wires, p) for p in (0.0, 0.05, 0.1, 0.2, 0.3)]
        assert values == sorted(values)


def test_teaching_threshold_shrinks_with_wire_count():
    thresholds = [teaching_threshold(n, 0.1) for n in range(2, 9)]
    assert thresholds == sorted(thresholds, reverse=True)
    assert thresholds[0] == pytest.approx(0.0528, abs=5e-4)
    assert thresholds[-1] == pytest.approx(0.0076, abs=2e-4)


def test_single_wire_has_no_visible_noise():
    assert ghz_tvd(1, 0.3) == pytest.approx(0.0)
    assert teaching_threshold(1) == 0.5
    assert support_mass(1, 0.3) == pytest.approx(1.0)
