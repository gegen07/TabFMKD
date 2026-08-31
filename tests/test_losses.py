import numpy as np
import pytest

from tabfm_kd.losses import (
    adaptive_temperatures,
    confidence_weights,
    mix_hard_soft,
    mix_regression_targets,
    one_hot,
    soften_probabilities,
)


def test_soften_probabilities_flattens_as_temperature_grows():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    sharp = soften_probabilities(probs, 0.5)
    soft = soften_probabilities(probs, 5.0)
    assert sharp[0, 0] > probs[0, 0]
    assert soft[0, 0] < probs[0, 0]
    assert np.allclose(soft.sum(axis=1), 1.0)


def test_adaptive_temperatures_raise_on_high_entropy():
    probs = np.array(
        [
            [0.95, 0.05],
            [0.50, 0.50],
        ]
    )
    temps = adaptive_temperatures(probs, base_temperature=3.0)
    assert temps[1] > temps[0]
    assert np.all(temps > 0)


def test_confidence_weights_prefer_peaked_predictions():
    probs = np.array([[0.99, 0.01], [0.55, 0.45]])
    weights = confidence_weights(probs)
    assert weights[0] > weights[1]


def test_mix_hard_soft_respects_alpha():
    soft = np.array([[0.7, 0.3]])
    hard = one_hot(np.array([0]), 2)
    mixed = mix_hard_soft(soft, hard, alpha=0.5)
    assert np.allclose(mixed, [[0.85, 0.15]])


def test_mix_regression_targets():
    mixed = mix_regression_targets(np.array([10.0, 0.0]), np.array([0.0, 10.0]), 0.25)
    assert np.allclose(mixed, [2.5, 7.5])


def test_invalid_temperature_raises():
    with pytest.raises(ValueError):
        soften_probabilities(np.array([[0.5, 0.5]]), 0.0)
