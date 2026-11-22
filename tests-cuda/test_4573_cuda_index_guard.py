# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak
from awkward._connect.cuda._index_guard import PendingValidation, guard_index
from awkward._nplikes.array_like import maybe_materialize

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("export", [False, True])
def test_nonblocking_stream_guard(fail, export):
    stream = cp.cuda.Stream(non_blocking=True)
    calls = []
    with stream:
        output = cp.asarray([2, 9999999 if fail else 0], dtype=cp.int64)
        status = cp.asarray([1 if fail else 2, 99], dtype=cp.int64)

    def validate():
        calls.append(1)
        stream.synchronize()
        row, attempt = status.get(stream=stream).tolist()
        if row < 2:
            raise ValueError(f"index out of range at row {row}; attempted {attempt}")

    guarded = guard_index(ak.index.Index64(output), PendingValidation(validate))
    assert guarded.length == 2
    assert calls == []

    def consume():
        if export:
            return cp.asarray(guarded.data)
        return maybe_materialize(guarded[1:].data)[0]

    for _ in range(2):
        if fail:
            with pytest.raises(ValueError, match="row 1; attempted 99"):
                consume()
        else:
            with cp.cuda.Stream(non_blocking=True) as consumer:
                actual = consume()
                consumer.synchronize()
                assert cp.asnumpy(actual).tolist() == ([2, 0] if export else [0])
    assert calls == [1]
    assert cp.asnumpy(output).tolist() == [2, 9999999 if fail else 0]
