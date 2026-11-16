# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("name", ["min", "max"])
@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        pytest.param(
            np.longdouble,
            marks=pytest.mark.skipif(
                np.dtype(np.longdouble).itemsize == 8,
                reason="no distinct extended floating-point representation",
            ),
        ),
    ],
)
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("initial", [None, 0, np.nan])
def test_segmented_real_extrema_propagate_nan(name, dtype, axis, initial):
    values = np.array(
        [[np.nan, 2, np.nan, 4], [1, 3, np.nan, np.nan], [5, 6, np.nan, 8]],
        dtype=dtype,
    )
    expected = getattr(np, name)(values, axis=axis, initial=initial)
    actual = getattr(ak, name)(ak.Array(values), axis=axis, initial=initial)
    assert ak.to_numpy(actual).dtype == expected.dtype
    np.testing.assert_array_equal(ak.to_numpy(actual), expected)


@pytest.mark.parametrize("name", ["min", "max"])
@pytest.mark.parametrize(
    "dtype",
    [
        np.complex64,
        np.complex128,
        pytest.param(
            np.clongdouble,
            marks=pytest.mark.skipif(
                np.dtype(np.clongdouble).itemsize == 16,
                reason="no distinct extended complex representation",
            ),
        ),
    ],
)
def test_segmented_complex_extrema_propagate_first_nan(name, dtype):
    values = np.array(
        [[1 + 2j, complex(3, np.nan), complex(np.nan, 4)], [3 + 2j, 3 + 1j, 2j]],
        dtype=dtype,
    )
    expected = getattr(np, name)(values, axis=-1)
    actual = ak.to_numpy(getattr(ak, name)(ak.Array(values), axis=-1))
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)


@pytest.mark.parametrize("name", ["min", "max"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_segmented_complex_extrema_propagate_nan_across_batches(name, dtype):
    clean = [complex(np.inf, -np.inf), complex(-np.inf, np.inf), 1 + 2j] * 4
    values = np.tile(np.array(clean, dtype=dtype), (len(clean) + 1, 1))
    expected = getattr(np, name)(values, axis=-1)
    for i in range(len(clean)):
        first_nan = complex(i, np.nan) if i % 2 else complex(np.nan, i)
        values[i, i] = first_nan
        if i + 1 < len(clean):
            values[i, i + 1] = complex(np.nan, np.nan)
        expected[i] = first_nan

    actual = ak.to_numpy(getattr(ak, name)(ak.Array(values), axis=-1))
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)


@pytest.mark.parametrize("name", ["nanmin", "nanmax"])
def test_nan_extrema_still_skip_nan_with_empty_and_missing_lists(name):
    array = ak.Array([[np.nan, None, 5.0, 2.0], [], [np.nan, None], [3.0]])
    actual = getattr(ak, name)(array, axis=-1)
    assert actual.to_list() == [getattr(np, name)([np.nan, 5.0, 2.0]), None, None, 3.0]


@pytest.mark.parametrize("name", ["min", "max"])
def test_segmented_extrema_propagate_nan_in_ragged_and_missing_lists(name):
    array = ak.Array([[None, 5.0, np.nan, 2.0], [], [None], [3.0]])
    actual = getattr(ak, name)(array, axis=-1).to_list()
    assert np.isnan(actual[0])
    assert actual[1:] == [None, None, 3.0]
