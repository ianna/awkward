# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import awkward as ak
from awkward._connect.cuda._index_guard import PendingValidation, guard_index
from awkward._nplikes.array_like import maybe_materialize


def test_metadata_and_aliases():
    calls = []
    validation = PendingValidation(lambda: calls.append("checked"))
    original = ak.index.Index64(np.array([2, 0, 1]))
    guarded = guard_index(original, validation)
    alias = guarded[1:]
    assert guarded.length == 3
    assert guarded.dtype == np.dtype(np.int64)
    assert calls == []
    np.testing.assert_array_equal(maybe_materialize(alias.data)[0], [0, 1])
    np.testing.assert_array_equal(maybe_materialize(guarded.data)[0], [2, 0, 1])
    assert calls == ["checked"]


@pytest.mark.parametrize(
    "consume",
    [
        lambda index: index[0],
        lambda index: maybe_materialize(index.data)[0],
        lambda index: index.__array_interface__,
        lambda index: copy.copy(index)[0],
        lambda index: index[1:][0],
        lambda index: index.to64()[0],
        lambda index: index.__dlpack__(),
    ],
)
def test_failed_validation_never_exposes_partial_output(consume):
    output = np.array([2, 9999999, 9999999])
    calls = []

    def validate():
        calls.append("checked")
        raise ValueError("index out of range at row 1; attempted 99")

    guarded = guard_index(ak.index.Index64(output), PendingValidation(validate))
    for _ in range(2):
        with pytest.raises(ValueError, match="row 1; attempted 99"):
            consume(guarded)
    assert calls == ["checked"]
    np.testing.assert_array_equal(output, [2, 9999999, 9999999])


def test_concurrent_consumers_validate_once():
    calls = []
    state = PendingValidation(lambda: calls.append("checked"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: state.check(), range(32)))
    assert calls == ["checked"]


def test_recursive_consumption_fails_closed():
    def recurse():
        state.check()

    state = PendingValidation(recurse)
    with pytest.raises(RuntimeError, match="recursive"):
        state.check()
    with pytest.raises(RuntimeError, match="recursive"):
        state.check()


def test_interrupted_validation_can_retry():
    calls = []

    def validate():
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt

    state = PendingValidation(validate)
    with pytest.raises(KeyboardInterrupt):
        state.check()
    state.check()
    state.check()
    assert calls == [1, 1]
