# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE


import contextvars
import copy
import glob
import os
from threading import RLock

import numpy

try:
    import cupy

    error_message = None

except ModuleNotFoundError:
    cupy = None
    error_message = """to use {0}, you must install cupy:

    pip install cupy

or

    conda install -c conda-forge cupy
"""

from awkward._connect.cuda.reducers import get_cuda_compute_reducer  # noqa: F401

cuda_streamptr_to_contexts = {}
kernel_errors = {}
kernel = None

ERROR_BITS = 8
COMPACT_ERROR_CODE = (1 << ERROR_BITS) - 1
NO_ERROR = numpy.iinfo(numpy.uint64).max

# When the pending-invocation list for a stream reaches this length, the
# kernel dispatchers in awkward._kernels call synchronize_cuda() to check for
# errors and drain it. This bounds the memory held by Invocation/ErrorContext
# objects in workloads that never synchronize explicitly, at the cost of one
# stream synchronization per MAX_PENDING_INVOCATIONS kernel launches.
MAX_PENDING_INVOCATIONS = 8192


dtype_to_ctype = {
    numpy.bool_: "bool",
    numpy.int8: "int8_t",
    numpy.uint8: "uint8_t",
    numpy.int16: "int16_t",
    numpy.uint16: "uint16_t",
    numpy.int32: "int32_t",
    numpy.uint32: "uint32_t",
    numpy.int64: "int64_t",
    numpy.uint64: "uint64_t",
    numpy.float32: "float",
    numpy.float64: "double",
}


def fetch_specialization(keys):
    specialized_name = keys[0].replace("'", "") + "<"

    keys = keys[1:]
    for key in keys[:-1]:
        if dtype_to_ctype.get(key) is None:
            dtype_string = repr(key)
            specialized_name = specialized_name + dtype_string.split("'")[1] + "_t, "
        else:
            specialized_name = specialized_name + dtype_to_ctype[key] + ", "

    if dtype_to_ctype.get(keys[-1]) is None:
        dtype_string = repr(keys[-1])
        specialized_name = specialized_name + dtype_string.split("'")[1] + "_t>"
    else:
        specialized_name = specialized_name + dtype_to_ctype[keys[-1]] + ">"

    return specialized_name


def fetch_template_specializations(kernel_dict, cuda_src):
    # Only signatures whose name matches a __global__ template in the CUDA
    # sources get specialized. This skips kernels dispatched to cuda.compute
    # and launchers built from several sub-kernels (e.g. '_a', '_b'), while
    # including those sub-kernels' own signatures.
    import re

    kernel_templates = set(re.findall(r"__global__\s+void\s+(\w+)\s*\(", cuda_src))

    return [
        fetch_specialization(list(keys))
        for keys in kernel_dict
        if keys[0] in kernel_templates
    ]


def populate_kernel_errors(kernel_name, cu_file):
    import re

    pattern_errstring = "// message:"

    result_errstring = [_.start() for _ in re.finditer(pattern_errstring, cu_file)]

    err_strings = []
    for index in result_errstring:
        error = cu_file[index:]
        error = error[error.find('"') : error.find("\n")]
        error = error.replace('"', "")
        err_strings.append(error)

    kernel_errors[kernel_name] = err_strings


_active_invocation = contextvars.ContextVar("awkward_cuda_invocation", default=None)
_drain_lock = RLock()


def _copy_error(error):
    """Copy a cached outcome without sharing mutable notes or traceback owners."""
    result = copy.copy(error).with_traceback(None)
    result.__context__ = None
    result.__cause__ = None
    if hasattr(error, "__notes__"):
        result.__notes__ = list(error.__notes__)
    return result


class CompletionRecord:
    """Persistent, host-side outcome of one invocation, independent of its queue.

    A guarded buffer may retain this record after the stream batch is drained.
    Failed batches cancel later invocations conservatively: their partial output
    must never become readable just because their own status was not inspected.
    No device buffers or operation arguments are retained after completion.
    """

    def __init__(self):
        self._resolved = False
        self._error = None
        self._stream = None
        self._device_id = None

    def check(self):
        if not self._resolved:
            with _drain_lock:
                if not self._resolved:
                    if self._stream is None:
                        raise RuntimeError("completion record has no registered stream")
                    if import_cupy().cuda.runtime.getDevice() != self._device_id:
                        raise RuntimeError(
                            "completion must be drained on its producer device"
                        )
                    try:
                        synchronize_cuda(self._stream)
                    except Exception:
                        # Unwinding through a different consumer operation must
                        # not pretty-print an invalid, CUDA-backed result.
                        from awkward._errors import ErrorContext

                        ErrorContext.cuda_primary()
                        raise
                    if not self._resolved:
                        raise RuntimeError(
                            "completion was not resolved by its stream drain"
                        )
        if self._error is not None:
            # Never raise the stored exception itself: its traceback would pin
            # completed batches and their scratch through the persistent record.
            from awkward._errors import ErrorContext

            ErrorContext.cuda_primary()
            raise _copy_error(self._error)

    def _resolve(self, error):
        if not self._resolved:
            self._error = error
            self._stream = None
            self._resolved = True


def compact_error_state():
    """Return this invocation's stream-owned error word and runtime ordinal.

    Publishers must use the recorded stream and one writer per launch. The
    compact code reserves the low bits of this invocation's existing error word;
    detailed status remains private and is transferred only on failure.
    """
    active = _active_invocation.get()
    if active is None:
        raise RuntimeError("compact error state requires an active invocation")
    invocation, _stream_ptr, device = active
    word = invocation.error_word
    if word is None or word.device.id != device:
        raise RuntimeError("compact error word must belong to the producer device")
    return word, invocation.error_index


def defer_status(status, decode, *, stream, keepalive=(), compact=False):
    """Retain a device status until the owning compute invocation is drained.

    ``decode(host_status)`` returns None on success or an exception containing
    the canonical message, first failing position and attempted index. No host
    read occurs here. The producer must enqueue on ``stream`` and guard unsafe
    downstream reads/writes on device; registration alone does not do that.
    Scratch must be private to the invocation and remain unmodified until drain.
    Returns a persistent CompletionRecord suitable for guard_index validation.
    """
    active = _active_invocation.get()
    if active is None:
        raise RuntimeError("defer_status requires an active CUDA compute invocation")
    invocation, stream_ptr, device_id = active
    if stream.ptr != stream_ptr or status.device.id != device_id:
        raise ValueError("deferred status must belong to the invocation stream/device")
    if invocation.completion._resolved:
        raise RuntimeError("cannot register status after invocation completion")
    if compact:
        compact_error_state()
    if invocation.deferred_statuses and invocation.compact_status != compact:
        raise ValueError("cannot mix compact and ordinary statuses in one invocation")
    invocation.compact_status = compact
    invocation.completion._device_id = device_id
    invocation.completion._stream = stream
    invocation.deferred_statuses.append((status, decode, keepalive))
    return invocation.completion


class Invocation:
    def __init__(self, name, error_context):
        self.completion = CompletionRecord()
        self.deferred_statuses = []
        self.compact_status = False
        self.error_word = None
        self.error_index = None
        self._name = name
        self._error_context = error_context

    @property
    def name(self):
        return self._name

    @property
    def error_context(self):
        return self._error_context


def import_cupy(name="Awkward Arrays with CUDA"):
    if cupy is None:
        raise ModuleNotFoundError(error_message.format(name))

    return cupy


def initialize_cuda_kernels(cupy):
    if cupy is not None:
        global kernel

        if kernel is None:
            import awkward._connect.cuda._kernel_signatures

            cuda_src = f"#define ERROR_BITS {ERROR_BITS}\n#define NO_ERROR {NO_ERROR}"

            cuda_kernels_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "cuda_kernels"
            )

            with open(
                os.path.join(cuda_kernels_path, "cuda_common.cu"),
                encoding="utf-8",
            ) as header_file:
                cuda_src = cuda_src + "\n" + header_file.read()
            for filename in glob.glob(os.path.join(cuda_kernels_path, "awkward_*.cu")):
                with open(filename, encoding="utf-8") as cu_file:
                    cu_code = cu_file.read()
                    populate_kernel_errors(
                        filename[filename.find("awkward_") : filename.find(".cu")],
                        cu_code,
                    )
                    cuda_src = cuda_src + "\n" + cu_code

            # Pass an empty Raw Module to fetch all template specializations
            template_specializations = fetch_template_specializations(
                awkward._connect.cuda._kernel_signatures.by_signature(None), cuda_src
            )
            cuda_kernel_templates = cupy.RawModule(
                code=cuda_src,
                options=(
                    "--std=c++11",
                    "--diag-suppress=186",
                ),
                name_expressions=template_specializations,
            )
            kernel = awkward._connect.cuda._kernel_signatures.by_signature(
                cuda_kernel_templates
            )

        return kernel
    else:
        raise ModuleNotFoundError(error_message.format("Awkward Arrays with CUDA"))


def synchronize_cuda(stream=None):
    with _drain_lock:
        _synchronize_cuda(stream)


def _synchronize_cuda(stream):
    cupy = import_cupy("Awkward Arrays with CUDA")
    if stream is None:
        stream = cupy.cuda.get_current_stream()

    stream.synchronize()
    batch = cuda_streamptr_to_contexts.get(stream.ptr)
    if batch is None or not batch[1]:
        # Still fence raw CuPy work, but an empty invocation queue has no
        # status to transfer or error word to reallocate/reset.
        return
    contexts = batch[1]
    # Detach only after the fence; scratch remains owned until results resolve.
    with stream:
        cuda_streamptr_to_contexts[stream.ptr] = (cupy.array(NO_ERROR), [])
    first_error = None
    try:
        invocation_index = batch[0].get(stream=stream).tolist()
        legacy_index = (
            None
            if invocation_index == NO_ERROR
            else int(invocation_index) >> ERROR_BITS
        )
        if legacy_index is not None and not 0 <= legacy_index < len(contexts):
            raise RuntimeError("CUDA error indicator refers outside the pending batch")
        for index, invocation in enumerate(contexts):
            if first_error is None:
                error = None
                try:
                    # Compact publishers set the shared word in their existing
                    # write operation. A successful batch needs no payload reads.
                    if not invocation.compact_status or index == legacy_index:
                        for status, decode, _keepalive in invocation.deferred_statuses:
                            error = decode(status.get(stream=stream).tolist())
                            if error is not None:
                                break
                    if error is None and index == legacy_index:
                        code = int(invocation_index) & ((1 << ERROR_BITS) - 1)
                        if invocation.compact_status and code == COMPACT_ERROR_CODE:
                            error = RuntimeError(
                                "compact CUDA error has no failing payload"
                            )
                        else:
                            error = ValueError(
                                f"{kernel_errors[invocation.name][code]} in compiled CUDA code ({invocation.name})"
                            )
                    if error is not None and invocation.error_context is not None:
                        error = invocation.error_context.decorate_exception(
                            type(error), error
                        )
                except Exception as decoding_error:
                    error = decoding_error
                if error is not None:
                    # Store a traceback-free copy; decoder tracebacks can own
                    # buffers even though they never appear in the message.
                    first_error = _copy_error(error)
            invocation.completion._resolve(first_error)
    except Exception as transfer_error:
        first_error = _copy_error(transfer_error)
    finally:
        for invocation in contexts:
            if not invocation.completion._resolved:
                invocation.completion._resolve(
                    first_error or RuntimeError("CUDA status draining interrupted")
                )
            if invocation.completion._resolved:
                invocation.deferred_statuses.clear()
    if first_error is not None:
        raise _copy_error(first_error)


def lazy(array):
    """Wrap a CUDA-backed array for lazy, fused execution.

    Internal entry point (``awkward._connect.cuda.lazy``); not exposed as a
    public ``ak.*`` name while the lazy/fusion work is a PoC.

    Args:
        array (ak.Array): A CUDA-backed array.

    Returns a :class:`awkward._connect.lazy._lazy_impl.LazyAwkwardArray`.

    Raises:
        TypeError: If ``array`` is not on the CUDA backend.
    """
    if array.layout.backend.name != "cuda":
        raise TypeError("cuda.lazy is only available for arrays with the CUDA backend")

    from awkward._connect.lazy._lazy_impl import lazy as _lazy

    return _lazy(array)


def to_cccl_iterator(array, *, dtype=None):
    """Build a ``cuda.compute`` iterator over an Awkward Array (zero-copy).

    This is the public, supported entry point for the recursive
    form-to-iterator construction that the fusion codegen and the eager CCCL
    helpers use internally. External ``cuda.compute`` code should call this
    instead of hand-extracting buffers from ``ak.to_buffers`` and assembling
    iterators by hand.

    The mapping follows the layout tree:

    - ``NumpyArray`` -> a CuPy buffer;
    - ``RecordArray`` -> ``ZipIterator`` over the field iterators;
    - ``IndexedArray`` -> ``PermutationIterator`` over (content, index);
    - ``ListOffsetArray`` -> the flattened content iterator (offsets returned
      separately in the metadata).

    Args:
        array (ak.Array, ak.Record, or ak.contents.Content): Source array. A
            CUDA-backed array is consumed zero-copy; an array on another backend
            is moved to the device first.
        dtype: If given, cast the leaf ``NumpyArray`` buffers to this dtype
            (e.g. ``numpy.float32``) while building the iterator.

    Returns an ``(iterator, metadata)`` tuple. ``iterator`` is a
    ``cuda.compute`` iterator (or CuPy buffer for a bare ``NumpyArray``);
    ``metadata`` is a dict ``{"form", "buffers", "offsets", "length", "count"}``
    where ``offsets`` is the list offsets as an ``int64`` CuPy array (or
    ``None`` for a non-list array), ``length`` is the top-level length, and
    ``count`` is the total number of items across all sublists.

    Raises:
        ModuleNotFoundError: If ``cupy`` is not installed.

    Internal entry point (``awkward._connect.cuda.to_cccl_iterator``); not
    exposed as a public ``ak.*`` name while the lazy/fusion work is a PoC.

    Examples:
        >>> import awkward as ak
        >>> from awkward._connect import cuda
        >>> arr = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0]], backend="cuda")
        >>> it, meta = cuda.to_cccl_iterator(arr)
        >>> meta["count"], len(meta["offsets"])
        (5, 3)
    """
    if cupy is None:
        raise ModuleNotFoundError(
            error_message.format("awkward._connect.cuda.to_cccl_iterator")
        )

    from awkward._connect.cuda.helpers import awkward_to_cccl_iterator

    return awkward_to_cccl_iterator(array, dtype=dtype)


__all__ = ("lazy", "to_cccl_iterator")
