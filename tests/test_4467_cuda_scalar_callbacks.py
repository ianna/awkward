# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import ast
import functools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import awkward as ak


@pytest.fixture(scope="module")
def callbacks():
    source = Path(ak.__file__).parent / "_connect/cuda/_compute.py"
    names = {
        "awkward_reduce_centered_sumofsquares",
        "_make_widening_cast",
        "_make_square_to_float64",
        "_make_power_to_float64",
        "_make_complex_power",
        "_make_complex_square",
        "_add_complex_values",
        "_centered_complex_magnitude",
        "infer_complex_dtype",
        "make_segment_views",
        "_complex_transform_reduce",
        "awkward_reduce_sumofpowers_complex",
        "awkward_reduce_sumofsquares_complex",
        "awkward_reduce_centered_sumofsquares_complex",
    }
    tree = ast.parse(source.read_text())
    tree.body = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]

    class Transform:
        def __init__(self, values, op):
            self.values, self.op = values, op

        def __getitem__(self, i):
            return self.op(self.values[i])

    class Zip:
        def __init__(self, a, b):
            self.a, self.b = a, b

        def __getitem__(self, i):
            return SimpleNamespace(field_0=self.a[i], field_1=self.b[i])

    def reduce(**kwargs):
        for row, (start, stop) in enumerate(
            zip(kwargs["start_offsets_in"], kwargs["end_offsets_in"], strict=True)
        ):
            value = kwargs["h_init"].item()
            for i in range(start, stop):
                item = kwargs["d_in"][i]
                op = kwargs["op"]
                value = value + item if op == "sum" else op(value, item)
            kwargs["d_out"][row] = value

    namespace = {
        "__name__": "awkward._connect.cuda._compute",
        "np": np,
        "cp": np,
        "functools": functools,
        "TransformIterator": Transform,
        "ZipIterator": Zip,
        "segmented_reduce": reduce,
        "OpKind": SimpleNamespace(PLUS="sum"),
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.float32])
def test_real_callback_casts(callbacks, dtype):
    value = dtype(1)
    assert callbacks._make_widening_cast(dtype, np.float64)(value) == 1
    assert callbacks._make_square_to_float64(dtype)(value) == 1
    for power in (-2, 0, 3, 8):
        assert callbacks._make_power_to_float64(dtype, power)(value) == 1


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("power", [-3, -1, 0, 1, 2, 7])
def test_complex_callback_arithmetic(callbacks, dtype, power):
    function = callbacks._make_complex_power(dtype, power)
    assert function is callbacks._make_complex_power(dtype, power)
    for value in (dtype(2 + 3j), dtype(3 + 2j), dtype(0)):
        with np.errstate(all="ignore"):
            np.testing.assert_allclose(
                function(value), value**power, rtol=2e-5, equal_nan=True
            )
        np.testing.assert_allclose(
            callbacks._make_complex_square(dtype)(value), value * value
        )


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_reduction_orchestration(callbacks, dtype):
    real = np.float32 if dtype == np.complex64 else np.float64
    data = np.array([99, 1 + 2j, 3 - 1j, 2 + 4j, 88], dtype=dtype)
    offsets = np.array([1, 3, 3, 4], dtype=np.int64)
    output = np.zeros(3, dtype=dtype)
    callbacks.awkward_reduce_sumofsquares_complex(
        output.view(real), data.view(real), offsets, 3
    )
    np.testing.assert_allclose(output, [sum(data[1:3] ** 2), 0, data[3] ** 2])
    callbacks.awkward_reduce_sumofpowers_complex(
        output.view(real), data.view(real), offsets, 3, 3
    )
    np.testing.assert_allclose(output, [sum(data[1:3] ** 3), 0, data[3] ** 3])
    means = np.array([2 + 0.5j, 0, 2 + 4j], dtype=dtype)
    result = np.full(3, -99, dtype=real)
    callbacks.awkward_reduce_centered_sumofsquares_complex(
        result, data.view(real), offsets, 3, means.view(real)
    )
    np.testing.assert_allclose(result, [np.sum(abs(data[1:3] - means[0]) ** 2), 0, 0])
    callbacks.awkward_reduce_centered_sumofsquares_complex(
        result, data.view(real), offsets[:1], 0, means.view(real)
    )


@pytest.mark.parametrize("primitive,format", [("float128", "g"), ("complex256", "Zg")])
def test_extended_awkward1_form_pickle(primitive, format):
    try:
        dtype = ak.types.numpytype.primitive_to_dtype(primitive)
    except TypeError:
        form = ak.forms.NumpyForm.__new__(ak.forms.NumpyForm)
        with pytest.raises(TypeError, match="unrecognized primitive"):
            form.__setstate__(
                (False, None, None, (), 16 if primitive == "float128" else 32, format)
            )
        return
    form = ak.forms.NumpyForm.__new__(ak.forms.NumpyForm)
    form.__setstate__((False, None, None, (), dtype.itemsize, format))
    assert form.primitive == primitive


@pytest.mark.parametrize("primitive", ["float128", "complex256"])
def test_numba_rejects_extended_primitive_explicitly(primitive):
    pytest.importorskip("numba")
    ak.numba.register_and_check()
    from awkward._connect.numba.layout import numba_type_of_primitive

    with pytest.raises(TypeError, match="Numba does not support"):
        numba_type_of_primitive(primitive)


def test_real_centered_callback_offsets_and_empty(callbacks):
    data = np.array([99, 1, 3, 88], dtype=np.int32)
    offsets = np.array([1, 3, 3], dtype=np.int64)
    result = np.full(2, -99.0)
    callbacks.awkward_reduce_centered_sumofsquares(
        result, data, offsets, 2, np.array([2.0, 0.0])
    )
    np.testing.assert_array_equal(result, [2, 0])
    callbacks.awkward_reduce_centered_sumofsquares(
        result, data, offsets[:1], 0, np.array([])
    )
    np.testing.assert_array_equal(result, [2, 0])


def test_numba_half_read_guard():
    numba = pytest.importorskip("numba")
    ak.numba.register_and_check()
    from awkward._connect.numba.layout import NumpyArrayType

    layout = NumpyArrayType(numba.types.Array(numba.types.float16, 1, "C"), {})
    with pytest.raises(TypeError, match="Numba cannot read float16"):
        layout.getitem_at(None)
