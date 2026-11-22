# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from types import SimpleNamespace

import pytest

import awkward._connect.cuda as cuda


class Status:
    def __init__(self, value):
        self.value = value
        self.device = SimpleNamespace(id=0)
        self.reads = 0

    def get(self, *, stream=None):
        self.reads += 1
        return self

    def tolist(self):
        return self.value


class Stream:
    def __init__(self, ptr):
        self.ptr = ptr

    def synchronize(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def state(monkeypatch):
    stream = Stream(17)
    monkeypatch.setattr(cuda, "cuda_streamptr_to_contexts", {})
    monkeypatch.setattr(
        cuda,
        "cupy",
        SimpleNamespace(
            array=Status,
            cuda=SimpleNamespace(
                get_current_stream=lambda: stream,
                runtime=SimpleNamespace(getDevice=lambda: 0),
            ),
        ),
    )
    return stream


def decode(value):
    row, attempt = value
    return None if row == -1 else ValueError(f"index out of range at {row}: {attempt}")


def enqueue(stream, value, context=None):
    invocation = cuda.Invocation("test", context)
    status = Status(value)
    token = cuda._active_invocation.set((invocation, stream.ptr, 0))
    try:
        cuda.defer_status(status, decode, stream=stream, keepalive=(object(),))
    finally:
        cuda._active_invocation.reset(token)
    return invocation, status


def test_deferred_read_and_successful_drain(state):
    invocation, status = enqueue(state, [-1, 0])
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    assert status.reads == 0
    cuda.synchronize_cuda(state)
    assert status.reads == 1
    assert cuda.cuda_streamptr_to_contexts[state.ptr][1] == []


@pytest.mark.parametrize("legacy_first", [False, True])
def test_earliest_invocation_wins(state, monkeypatch, legacy_first):
    invocation, status = enqueue(state, [3, 99])
    legacy = cuda.Invocation("legacy", None)
    monkeypatch.setitem(cuda.kernel_errors, "legacy", ["legacy error"])
    contexts = [legacy, invocation] if legacy_first else [invocation, legacy]
    word = 0 if legacy_first else 1 << cuda.ERROR_BITS
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(word), contexts)
    with pytest.raises(
        ValueError, match="legacy error" if legacy_first else "at 3: 99"
    ):
        cuda.synchronize_cuda(state)
    assert status.reads == (0 if legacy_first else 1)
    assert cuda.cuda_streamptr_to_contexts[state.ptr][1] == []


def test_context_and_decoder_failure_cleanup(state):
    class Context:
        def decorate_exception(self, cls, error):
            return cls(f"context: {error}")

    invocation, _ = enqueue(state, [2, -9], Context())
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    with pytest.raises(ValueError, match="context: index out of range at 2: -9"):
        cuda.synchronize_cuda(state)
    invocation = cuda.Invocation("test", None)
    invocation.deferred_statuses.append((Status([]), decode, ()))
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    with pytest.raises(ValueError):
        cuda.synchronize_cuda(state)
    assert cuda.cuda_streamptr_to_contexts[state.ptr][1] == []


def test_registration_requires_owner(state):
    with pytest.raises(RuntimeError, match="active"):
        cuda.defer_status(Status([-1, 0]), decode, stream=state)
    invocation = cuda.Invocation("test", None)
    token = cuda._active_invocation.set((invocation, 999, 0))
    try:
        with pytest.raises(ValueError, match="stream/device"):
            cuda.defer_status(Status([-1, 0]), decode, stream=state)
    finally:
        cuda._active_invocation.reset(token)
    assert invocation.deferred_statuses == []


@pytest.mark.parametrize("raises", [False, True])
def test_dispatch_scope_restored(state, monkeypatch, raises):
    from awkward._kernels import BaseKernel, CudaComputeKernel

    monkeypatch.setattr(
        cuda.cupy.cuda, "Device", lambda: SimpleNamespace(id=0), raising=False
    )
    status = Status([-1, 0])

    def impl():
        cuda.defer_status(status, decode, stream=state)
        if raises:
            raise RuntimeError("launch failed")

    kernel = object.__new__(CudaComputeKernel)
    BaseKernel.__init__(kernel, impl, ("test",))
    if raises:
        with pytest.raises(RuntimeError, match="launch failed"):
            kernel()
    else:
        kernel()
    assert cuda._active_invocation.get() is None
    assert status.reads == 0
    cuda.synchronize_cuda(state)
    assert status.reads == 1


def test_stream_isolation_and_scratch_lifetime(state):
    import weakref

    class Scratch:
        pass

    other = Stream(18)
    first, _ = enqueue(state, [-1, 0])
    second, _ = enqueue(other, [4, 10])
    scratch = Scratch()
    reference = weakref.ref(scratch)
    status, decoder, _ = first.deferred_statuses[0]
    first.deferred_statuses[0] = (status, decoder, (scratch,))
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [first])
    cuda.cuda_streamptr_to_contexts[other.ptr] = (Status(cuda.NO_ERROR), [second])
    del scratch, first
    assert reference() is not None
    cuda.synchronize_cuda(state)
    assert reference() is None
    assert len(cuda.cuda_streamptr_to_contexts[other.ptr][1]) == 1
    with pytest.raises(ValueError, match="at 4: 10"):
        cuda.synchronize_cuda(other)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("external_drain", [False, True])
def test_guard_retains_completion_after_queue_reset(state, fail, external_drain):
    import numpy as np

    import awkward as ak
    from awkward._connect.cuda._index_guard import guard_index

    invocation, status = enqueue(state, [1, 99] if fail else [-1, 0])
    record = invocation.completion
    guarded = guard_index(ak.index.Index64(np.array([2, 9999999])), record)
    alias = guarded[1:]
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    if external_drain:
        if fail:
            with pytest.raises(ValueError, match="at 1: 99"):
                cuda.synchronize_cuda(state)
        else:
            cuda.synchronize_cuda(state)
    for _ in range(2):
        if fail:
            with pytest.raises(ValueError, match="at 1: 99"):
                alias[0]
        else:
            assert alias[0] == 9999999  # no error, buffer is exposed only after check
    assert status.reads == 1
    assert invocation.deferred_statuses == []
    assert record._stream is None
    assert cuda.cuda_streamptr_to_contexts[state.ptr][1] == []
    # A second successful drain cannot erase the previous failure.
    cuda.synchronize_cuda(state)
    if fail:
        with pytest.raises(ValueError, match="at 1: 99"):
            record.check()


def test_later_completion_fails_closed_without_reading_its_status(state):
    first, _ = enqueue(state, [0, 55])
    later, status = enqueue(state, [-1, 0])
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [first, later])
    with pytest.raises(ValueError, match="at 0: 55"):
        cuda.synchronize_cuda(state)
    with pytest.raises(ValueError, match="at 0: 55"):
        later.completion.check()
    assert status.reads == 0
    assert later.deferred_statuses == []


def test_prefix_success_survives_later_failure(state):
    first, _ = enqueue(state, [-1, 0])
    later, _ = enqueue(state, [2, 55])
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [first, later])
    with pytest.raises(ValueError):
        cuda.synchronize_cuda(state)
    first.completion.check()
    with pytest.raises(ValueError, match="at 2: 55"):
        later.completion.check()


def test_completion_failure_releases_scratch_and_traceback(state):
    import weakref

    invocation, status = enqueue(state, [0, 55])
    reference = weakref.ref(status)
    record = invocation.completion
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    del status
    with pytest.raises(ValueError):
        cuda.synchronize_cuda(state)
    assert reference() is None
    for _ in range(2):
        with pytest.raises(ValueError):
            record.check()
        assert record._error.__traceback__ is None


def test_completion_wrong_device_is_retryable(state, monkeypatch):
    invocation, status = enqueue(state, [-1, 0])
    record = invocation.completion
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    monkeypatch.setattr(cuda.cupy.cuda.runtime, "getDevice", lambda: 1)
    with pytest.raises(RuntimeError, match="producer device"):
        record.check()
    assert status.reads == 0
    monkeypatch.setattr(cuda.cupy.cuda.runtime, "getDevice", lambda: 0)
    record.check()
    assert status.reads == 1


def test_decoding_failure_is_persistent(state):
    invocation, status = enqueue(state, [])
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    with pytest.raises(ValueError, match="unpack"):
        cuda.synchronize_cuda(state)
    with pytest.raises(ValueError, match="unpack"):
        invocation.completion.check()
    assert status.reads == 1


def test_completion_record_return_and_terminal_registration(state):
    invocation = cuda.Invocation("test", None)
    token = cuda._active_invocation.set((invocation, state.ptr, 0))
    try:
        record = cuda.defer_status(Status([-1, 0]), decode, stream=state)
        assert record is invocation.completion
        cuda.cuda_streamptr_to_contexts[state.ptr] = (
            Status(cuda.NO_ERROR),
            [invocation],
        )
        record.check()
        with pytest.raises(RuntimeError, match="after invocation completion"):
            cuda.defer_status(Status([-1, 0]), decode, stream=state)
    finally:
        cuda._active_invocation.reset(token)


def test_concurrent_completion_checks_read_status_once(state):
    from concurrent.futures import ThreadPoolExecutor

    invocation, status = enqueue(state, [-1, 0])
    record = invocation.completion
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: record.check(), range(32)))
    assert status.reads == 1


def test_status_transfer_failure_is_persistent(state):
    invocation, status = enqueue(state, [-1, 0])

    def broken_transfer(**kwargs):
        raise RuntimeError("status transfer failed")

    status.get = broken_transfer
    record = invocation.completion
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    with pytest.raises(RuntimeError, match="status transfer failed"):
        cuda.synchronize_cuda(state)
    with pytest.raises(RuntimeError, match="status transfer failed"):
        record.check()
    assert invocation.deferred_statuses == []


def test_empty_queue_fences_without_status_transfer_or_allocation(state, monkeypatch):
    fences = []
    state.synchronize = lambda: fences.append(1)
    status = Status(cuda.NO_ERROR)
    cuda.cuda_streamptr_to_contexts[state.ptr] = (status, [])

    def no_allocation(*args):
        raise AssertionError("empty drain allocated an error buffer")

    monkeypatch.setattr(cuda.cupy, "array", no_allocation)
    cuda.synchronize_cuda(state)
    assert fences == [1]
    assert status.reads == 0
    cuda.cuda_streamptr_to_contexts.clear()
    cuda.synchronize_cuda(state)
    assert fences == [1, 1]


@pytest.mark.parametrize("external_drain", [False, True])
def test_consumer_error_notes_are_metadata_only_and_not_shared(state, external_drain):
    import awkward as ak

    class ProducerContext:
        def decorate_exception(self, cls, error):
            error.add_note("producer")
            return error

    invocation, _ = enqueue(state, [1, 99], ProducerContext())
    cuda.cuda_streamptr_to_contexts[state.ptr] = (Status(cuda.NO_ERROR), [invocation])
    if external_drain:
        with pytest.raises(ValueError):
            cuda.synchronize_cuda(state)

    class NeverFormat:
        def __repr__(self):
            raise AssertionError("consumer formatted array values")

    for _ in range(2):
        with pytest.raises(ValueError, match="at 1: 99") as caught:
            with ak._errors.OperationErrorContext("consumer", [NeverFormat()], {}):
                invocation.completion.check()
        assert caught.value.__notes__[0] == "producer"
        assert "<NeverFormat>" in caught.value.__notes__[1]
        assert len(caught.value.__notes__) == 2
    assert invocation.completion._error.__notes__ == ["producer"]
