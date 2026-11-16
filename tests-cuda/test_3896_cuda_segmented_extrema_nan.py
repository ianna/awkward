# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("name", ["min", "max"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.complex128])
def test_segmented_extrema_propagate_nan(name, dtype):
    values = np.array([[1, np.nan, 4], [np.nan, 2, 9], [3, 1, 4]], dtype=dtype)
    array = ak.to_backend(ak.Array(values), "cuda")
    expected = getattr(np, name)(values, axis=-1)
    actual = ak.to_numpy(getattr(ak, name)(array, axis=-1))
    np.testing.assert_array_equal(actual, expected)
