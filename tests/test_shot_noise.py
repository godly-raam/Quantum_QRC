import numpy as np

from modules.fast_sim import zz_pairs
from modules.readout import customer_pairs
from scripts.benchmark_fair import generate_instance
from scripts.shot_noise_ablation import (
    LargeESN,
    RandomFourierFeatures,
    add_shot_noise,
    rows_from_vector,
)


def test_add_shot_noise_matches_binomial_variance():
    value = 0.3
    shots = 1_000
    samples = 20_000
    rng = np.random.default_rng(7)
    noisy = add_shot_noise(np.full(samples, value), shots, rng)
    expected = (1.0 - value**2) / shots
    assert abs(float(np.var(noisy)) - expected) < 0.2 * expected
    assert np.all(np.abs(noisy) <= 1.0)


def test_add_shot_noise_is_nearly_exact_with_many_shots():
    values = np.array([-0.8, -0.1, 0.0, 0.4, 0.9])
    noisy = add_shot_noise(values, 1_000_000, np.random.default_rng(3))
    assert np.allclose(noisy, values, atol=5e-3)


def test_large_esn_and_rff_have_same_feature_count():
    size = 8
    expected = 3 * size + len(zz_pairs(size))
    instance = generate_instance(123, size)
    esn = LargeESN(size, seed=1)
    rff = RandomFourierFeatures(size, seed=2, gamma=1.0)
    rff.fit([instance])

    esn_vector = esn.feature_vector(instance.distances)
    rff_vector = rff.feature_vector(instance.distances)
    assert esn_vector.shape == (expected,)
    assert rff_vector.shape == (expected,)
    assert rows_from_vector(esn_vector, size).shape[0] == len(customer_pairs(size))
