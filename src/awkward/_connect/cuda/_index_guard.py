# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Experimental index-consumption barrier; not enabled in production dispatch."""

from threading import RLock

from awkward._nplikes.virtual import VirtualNDArray


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
    return type(index)(buffer, metadata=index._metadata, nplike=index.nplike)
