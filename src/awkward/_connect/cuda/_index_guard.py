# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Experimental index-consumption barrier; not enabled in production dispatch."""

from contextvars import ContextVar
from threading import RLock

from awkward._nplikes import to_nplike
from awkward._nplikes.array_like import maybe_materialize
from awkward._nplikes.virtual import VirtualNDArray
from awkward.index import Index8, Index32, Index64, IndexU8, IndexU32


class PendingValidation:
    """Validate once before exposing data, retaining failure for every alias.

    The callback must fence the producer stream and raise the canonical error.
    It must not consume this guarded index. Capture invocation-owned status and
    scratch in the callback, never a mutable global stream queue alone: another
    drain must not erase a failure that this index still needs to report.
    """

    def __init__(self, validate):
        self._validate = validate
        self._error = None
        self._ready = False
        self._checking = False
        self._lock = RLock()

    def check(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            if self._error is not None:
                raise self._error
            if self._checking:
                raise RuntimeError("recursive consumption of an unvalidated index")
            self._checking = True
            try:
                self._validate()
            except Exception as error:
                self._error = error
                raise
            else:
                self._ready = True
            finally:
                self._checking = False
                if self._ready or self._error is not None:
                    self._validate = None


class _GuardedIndexMixin:
    def _validate_before_transfer(self, nplike):
        if nplike.known_data:
            # Reject before copying any content, even when conversion normally
            # visits content before its index. Typetracer remains metadata-only.
            maybe_materialize(self.data)

    def raw(self, nplike):
        # This buffer already exists: only validation is pending. Ordinary
        # user-created virtual arrays still require explicit materialization.
        if nplike.known_data:
            (data,) = maybe_materialize(self.data)
            return to_nplike(data, nplike, from_nplike=self.nplike)
        # Typetracer conversion must remain metadata-only.
        return super().raw(nplike)

    def to64(self):
        # Preserve the normal dtype-conversion copy semantics. Once checked,
        # the copied index needs no pending-validation wrapper.
        (data,) = maybe_materialize(self.data)
        return Index64(self.nplike.astype(data, dtype=Index64._expected_dtype))


class _GuardedIndex8(_GuardedIndexMixin, Index8):
    pass


class _GuardedIndexU8(_GuardedIndexMixin, IndexU8):
    pass


class _GuardedIndex32(_GuardedIndexMixin, Index32):
    pass


class _GuardedIndexU32(_GuardedIndexMixin, IndexU32):
    pass


class _GuardedIndex64(_GuardedIndexMixin, Index64):
    pass


_guarded_index_classes = {
    cls._expected_dtype: cls
    for cls in (
        _GuardedIndex8,
        _GuardedIndexU8,
        _GuardedIndex32,
        _GuardedIndexU32,
        _GuardedIndex64,
    )
}


def guard_index(index, validation):
    """Wrap an already allocated index without reading or changing its data.

    Reuse VirtualNDArray's materialization boundary so slices, copies, direct
    indexing and array-interface/DLPack exports cannot expose unvalidated data.
    Metadata (dtype/length) remains available without a wait. Existing aliases
    of the original buffer are NOT guarded; only publish the returned index.
    """
    # CompletionRecord implements check() directly; PendingValidation remains
    # useful for standalone validators and microbenchmarks.
    data = index.data

    def consume():
        validation.check()
        return data

    buffer = VirtualNDArray(index.nplike, (index.length,), index.dtype, consume)
    return _guarded_index_classes[index.dtype](
        buffer, metadata=index._metadata, nplike=index.nplike
    )


_defer_simplify = ContextVar("awkward_defer_indexed_simplify", default=False)


def simplify_index(backend, result, outer, inner):
    """Opt into deferred simplify only where the output can be guarded.

    Direct kernel callers retain their synchronous KernelError contract. CPU,
    JAX and typetracer callers do not enter this CUDA-only helper.
    """
    token = _defer_simplify.set(True)
    try:
        error = backend[
            "awkward_IndexedArray_simplify",
            result.dtype.type,
            outer.dtype.type,
            inner.dtype.type,
        ](result.data, outer.data, outer.length, inner.data, inner.length)
    finally:
        _defer_simplify.reset(token)
    from awkward._connect.cuda import CompletionRecord

    if isinstance(error, CompletionRecord):
        return guard_index(result, error)
    backend.maybe_kernel_error(error)
    return result
