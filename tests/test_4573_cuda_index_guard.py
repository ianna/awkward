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


@pytest.mark.parametrize("fail", [False, True])
def test_simplify_index_scope_and_guard(fail):
    from awkward._connect.cuda import CompletionRecord
    from awkward._connect.cuda._index_guard import _defer_simplify, simplify_index

    record = CompletionRecord()
    record._resolve(ValueError("index out of range") if fail else None)

    class Backend:
        def __getitem__(self, key):
            def kernel(*args):
                assert _defer_simplify.get()
                return record

            return kernel

    source = ak.index.Index64(np.array([0, 1]))
    result = simplify_index(Backend(), source, source, source)
    assert not _defer_simplify.get()
    assert result.length == 2
    if fail:
        with pytest.raises(ValueError, match="index out of range"):
            result[0]
    else:
        assert result[1] == 1


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int32, np.uint32, np.int64])
@pytest.mark.parametrize("fail", [False, True])
def test_guarded_raw_conversion_and_aliases(dtype, fail, monkeypatch):
    import awkward._connect.cuda._index_guard as module
    from awkward._nplikes.numpy import Numpy

    calls = []
    transfers = []
    actual = module.to_nplike

    def validate():
        calls.append("validated")
        if fail:
            raise ValueError("index out of range")

    def convert(array, nplike, *, from_nplike):
        # Reproduce the cross-backend entry point: conversion must receive an
        # actual validated array, never an unmaterialized VirtualNDArray.
        assert isinstance(array, np.ndarray)
        transfers.append("transferred")
        return actual(array, nplike, from_nplike=from_nplike)

    monkeypatch.setattr(module, "to_nplike", convert)
    source = ak.index.Index(np.array([2, 0, 1], dtype=dtype))
    guarded = guard_index(source, PendingValidation(validate))
    aliases = [guarded, guarded[1:], copy.copy(guarded)]
    assert calls == []
    for alias in aliases:
        if fail:
            with pytest.raises(ValueError, match="index out of range"):
                alias.to_nplike(Numpy.instance())
        else:
            np.testing.assert_array_equal(
                alias.raw(Numpy.instance()), [0, 1] if alias.length == 2 else [2, 0, 1]
            )
    assert calls == ["validated"]
    assert len(transfers) == (0 if fail else 3)


def test_guarded_to64_copies_and_typetracer_stays_lazy():
    from awkward._nplikes.typetracer import TypeTracer

    calls = []
    original = np.array([2, 0, 1], dtype=np.int32)
    guarded = guard_index(
        ak.index.Index32(original), PendingValidation(lambda: calls.append(1))
    )
    traced = guarded.to_nplike(TypeTracer.instance())
    assert traced.length == 3
    assert calls == []
    converted = guarded.to64()
    assert calls == [1]
    assert converted.dtype == np.dtype(np.int64)
    assert not np.shares_memory(converted.data, original)


def test_ordinary_virtual_conversion_still_requires_materialization():
    from types import SimpleNamespace

    from awkward._nplikes import to_nplike
    from awkward._nplikes.numpy import Numpy
    from awkward._nplikes.virtual import VirtualNDArray

    nplike = Numpy.instance()
    array = VirtualNDArray(nplike, (3,), np.dtype(np.int64), lambda: np.arange(3))
    target = SimpleNamespace(known_data=True, supports_virtual_arrays=True)
    with pytest.raises(TypeError, match="not yet materialized"):
        to_nplike(array, target, from_nplike=nplike)
    assert not array.is_materialized
