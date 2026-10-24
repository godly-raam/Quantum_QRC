import numpy as np

from scripts.null_control import gaussian_like, make_null_features, shuffle_columns


def test_shuffled_features_preserve_column_marginals():
    rng = np.random.default_rng(11)
    train = rng.normal(size=(17, 9))
    test = rng.normal(size=(23, 9))

    shuffled_train, shuffled_test = shuffle_columns(train, test, seed=3)

    assert shuffled_train.shape == train.shape
    assert shuffled_test.shape == test.shape
    assert np.allclose(
        np.sort(shuffled_train, axis=0), np.sort(train, axis=0)
    )
    assert np.allclose(
        np.sort(shuffled_test, axis=0), np.sort(test, axis=0)
    )
    assert np.allclose(shuffled_train.mean(axis=0), train.mean(axis=0))
    assert np.allclose(shuffled_train.std(axis=0), train.std(axis=0))


def test_seeded_null_generation_is_deterministic():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(19, 6))
    test = rng.normal(size=(13, 6))

    first = make_null_features(train, test, seed=123)
    second = make_null_features(train, test, seed=123)
    different = make_null_features(train, test, seed=124)

    for name in ("shuffled", "gaussian"):
        assert np.array_equal(first[name][0], second[name][0])
        assert np.array_equal(first[name][1], second[name][1])
        assert not np.array_equal(first[name][0], different[name][0])

    expected_train, expected_test = gaussian_like(train, test, seed=123)
    assert np.array_equal(first["gaussian"][0], expected_train)
    assert np.array_equal(first["gaussian"][1], expected_test)
