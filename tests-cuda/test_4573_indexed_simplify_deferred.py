# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak
import awkward._connect.cuda as cuda
from awkward._nplikes.virtual import VirtualNDArray

cp = pytest.importorskip("cupy")
pytest.importorskip("cuda.compute")


@pytest.mark.parametrize("option", [False, True])
@pytest.mark.parametrize(
    "length,error_row", [(32, None), (129, None), (129, 0), (129, 64), (129, 128)]
)
@pytest.mark.parametrize("external_drain", [False, True])
def test_public_guarded_simplification(
    option, length, error_row, external_drain, monkeypatch
):
    from awkward._connect.cuda import _indexedarray

    original = _indexedarray.simplify_status
    reads = []

    class Status:
        def __init__(self, array):
            self.array = array
            self.device = array.device

        def get(self, **kwargs):
            reads.append(1)
            return self.array.get(**kwargs)

    def prepare(*args, **kwargs):
        return Status(original(*args, **kwargs))

    monkeypatch.setattr(_indexedarray, "simplify_status", prepare)
    cls = ak.contents.IndexedOptionArray if option else ak.contents.IndexedArray
    outer = np.arange(length, dtype=np.int64) % 3
    if error_row is not None:
        outer[error_row] = 99
    with cp.cuda.Stream(non_blocking=True) as stream:
        inner = cls(
            ak.index.Index64(cp.asarray([2, 0, 1])),
            ak.contents.NumpyArray(cp.asarray([10, 20, 30])),
        )
        result = cls.simplified(ak.index.Index64(cp.asarray(outer)), inner)
        assert isinstance(result.index.data, VirtualNDArray)
        assert result.length == length
        assert reads == []  # No per-kernel status transfer during construction.
        if external_drain:
            if error_row is None:
                cuda.synchronize_cuda(stream)
            else:
                with pytest.raises(ValueError, match="index out of range") as caught:
                    cuda.synchronize_cuda(stream)
                assert (caught.value.id, caught.value.attempt) == (error_row, 99)
        for _ in range(2):
            if error_row is None:
                assert ak.to_list(result) == np.array([30, 10, 20])[outer].tolist()
            else:
                # These consumers must not read untouched index-tail entries.
                with pytest.raises(ValueError, match="index out of range") as caught:
                    ak.to_list(result)
                assert (caught.value.id, caught.value.attempt) == (error_row, 99)
                with pytest.raises(ValueError, match="index out of range"):
                    result._carry(
                        ak.index.Index64(cp.asarray([length - 1])), allow_lazy=False
                    )
        assert reads == [1]
        cuda.synchronize_cuda(stream)


def test_chained_field_selection_guards_previous_output():
    with cp.cuda.Stream(non_blocking=True) as stream:
        inner = ak.contents.IndexedArray(
            ak.index.Index64(cp.asarray([1, 0])),
            ak.contents.NumpyArray(cp.asarray([10, 20])),
        )
        record = ak.contents.RecordArray([inner], ["x"])
        source = ak.Array(
            ak.contents.IndexedArray(ak.index.Index64(cp.asarray([0, 99])), record)
        )
        result = source["x"]
        with pytest.raises(ValueError, match="index out of range"):
            ak.to_backend(result, "cpu")
        cuda.synchronize_cuda(stream)


@pytest.mark.parametrize("option", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_guarded_backend_transfer_and_aliases(option, invalid):
    cls = ak.contents.IndexedOptionArray if option else ak.contents.IndexedArray
    with cp.cuda.Stream(non_blocking=True) as stream:
        inner = cls(
            ak.index.Index64(cp.asarray([1, 0])),
            ak.contents.NumpyArray(cp.asarray([10, 20])),
        )
        guarded = cls.simplified(
            ak.index.Index64(cp.asarray([0, 99 if invalid else 1])), inner
        )
        assert isinstance(guarded.index.data, VirtualNDArray)
        for layout in (guarded, guarded[1:]):
            if invalid:
                with pytest.raises(ValueError, match="index out of range"):
                    ak.to_backend(layout, "cpu", highlevel=False)
            else:
                assert ak.to_backend(layout, "cpu", highlevel=False).to_list() == (
                    [20, 10] if layout.length == 2 else [10]
                )
        cuda.synchronize_cuda(stream)


def test_empty_drain_still_fences_raw_cupy_work():
    with cp.cuda.Stream(non_blocking=True) as stream:
        cuda.synchronize_cuda(
            stream
        )  # clear any earlier batch on a reused stream pointer
        output = cp.empty(1024, dtype=cp.int64)
        output.fill(123)
        done = cp.cuda.Event()
        done.record(stream)
        cuda.synchronize_cuda(stream)
        assert done.query()
        assert cp.asnumpy(output).tolist() == [123] * 1024


def test_invalid_guard_precedes_content_transfer(monkeypatch):
    transfers = []
    original = ak.contents.NumpyArray._to_backend
    with cp.cuda.Stream(non_blocking=True) as stream:
        content = ak.contents.NumpyArray(cp.asarray([10, 20]))
        inner = ak.contents.IndexedArray(ak.index.Index64(cp.asarray([1, 0])), content)
        result = ak.contents.IndexedArray.simplified(
            ak.index.Index64(cp.asarray([0, 99])), inner
        )

        def transfer(self, backend):
            if self is content:
                transfers.append(1)
            return original(self, backend)

        monkeypatch.setattr(ak.contents.NumpyArray, "_to_backend", transfer)
        with pytest.raises(ValueError, match="index out of range"):
            ak.to_backend(result, "cpu")
        assert transfers == []
        cuda.synchronize_cuda(stream)
