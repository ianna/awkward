# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Separate diagnostic passes; never interpreted as benchmark latencies."""

import contextlib
import cProfile
import functools
import inspect
import json
import pstats
import sys
import time
from collections import defaultdict


def profile_key(filename, function):
    for prefix in ("/src/awkward/", "/site-packages/"):
        if prefix in filename:
            filename = prefix.strip("/") + "/" + filename.split(prefix, 1)[1]
            break
    return f"{filename}:{function}"


class Trace:
    def __init__(self, targets, cupy):
        self.targets = targets
        self.cupy = cupy
        self.active = False
        self.events = defaultdict(lambda: {"calls": 0, "host_ns": 0, "bytes": 0})
        self.pending_c = []
        self.restores = []

    def add(self, label, start, size=0):
        event = self.events[label]
        event["calls"] += 1
        event["host_ns"] += time.perf_counter_ns() - start
        event["bytes"] += size

    def wrap(self, label, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if not self.active:
                return function(*args, **kwargs)
            start = time.perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                self.add(label, start)

        return wrapped

    def c_event(self, frame, event, function):
        if not self.active or event not in ("c_call", "c_return", "c_exception"):
            return
        name = getattr(function, "__name__", "")
        owner = getattr(function, "__self__", None)
        label, size = None, 0
        if name == "get" and isinstance(owner, self.cupy.ndarray):
            label, size = "cupy.ndarray.get", owner.nbytes
        elif name == "synchronize" and type(owner).__module__.startswith("cupy"):
            label = "cupy.synchronize"
        if label is not None:
            if event == "c_call":
                self.pending_c.append((label, time.perf_counter_ns(), size))
            elif self.pending_c and self.pending_c[-1][0] == label:
                entry = self.pending_c.pop()
                self.add(*entry)

    def __enter__(self):
        self.previous = sys.getprofile()
        for owner, name, label in self.targets:
            if owner is None or not hasattr(owner, name):
                continue
            original = inspect.getattr_static(owner, name)
            self.restores.append((owner, name, original))
            if isinstance(original, classmethod):
                replacement = classmethod(self.wrap(label, original.__func__))
            else:
                replacement = self.wrap(label, getattr(owner, name))
            setattr(owner, name, replacement)
        sys.setprofile(self.c_event)
        return self

    def __exit__(self, *args):
        self.active = False
        sys.setprofile(self.previous)
        for owner, name, original in reversed(self.restores):
            setattr(owner, name, original)

    @contextlib.contextmanager
    def capture(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False


def diagnose(sample, check, *, samples, directory, key, cupy, cleanup):
    import awkward as ak
    import awkward._connect.cuda as cuda
    from awkward._nplikes.virtual import VirtualNDArray

    try:
        from awkward._connect.cuda import _index_guard as guard
    except ImportError:
        guard = None
    targets = [
        (cuda, "synchronize_cuda", "drain.public"),
        (cuda, "_synchronize_cuda", "drain.internal"),
        (getattr(cuda, "CompletionRecord", None), "check", "completion.check"),
        (getattr(cuda, "CompletionRecord", None), "_resolve", "completion.resolve"),
        (getattr(cuda, "CompletionRecord", None), "__init__", "completion.construct"),
        (cuda.Invocation, "__init__", "invocation.construct"),
        (guard, "guard_index", "guard.construct"),
        (getattr(guard, "_GuardedIndexMixin", None), "raw", "guard.raw"),
        (guard, "simplify_index", "guard.simplify_index"),
        (VirtualNDArray, "materialize", "virtual.materialize"),
        (ak._errors.ErrorContext, "cuda_primary", "context.capture"),
        (ak._errors.ErrorContext, "decorate_exception", "exception.decorate"),
        (ak._errors, "_cuda_argument", "context.argument"),
        (ak._backends.backend.Backend, "format_kernel_error", "exception.format"),
        (cupy, "empty", "allocation.cupy_empty"),
    ]
    with Trace(targets, cupy) as trace:
        for _ in range(samples):
            with trace.capture():
                result, errors, _, _ = sample()
            check(result, errors)  # no parity transfers or formatting in trace
            cleanup()  # parity may enqueue kernels; detach their queue outside capture
    profile = cProfile.Profile()
    for _ in range(samples):
        profile.enable()
        try:
            result, errors, _, _ = sample()
        finally:
            profile.disable()
        check(result, errors)
        cleanup()
    path = directory / (key.replace(":", "-") + ".pstats")
    profile.dump_stats(path)
    functions = {}
    for (filename, _line, name), (
        _primitive,
        calls,
        self_time,
        total,
        _callers,
    ) in pstats.Stats(profile).stats.items():
        label = profile_key(filename, name)
        values = functions.setdefault(
            label, {"calls_per_op": 0, "self_us_per_op": 0, "cumulative_us_per_op": 0}
        )
        values["calls_per_op"] += calls / samples
        values["self_us_per_op"] += self_time * 1e6 / samples
        values["cumulative_us_per_op"] += total * 1e6 / samples
    events = {
        name: {
            "calls_per_op": values["calls"] / samples,
            "inclusive_host_us_per_op": values["host_ns"] / samples / 1000,
            "bytes_per_op": values["bytes"] / samples,
        }
        for name, values in trace.events.items()
    }
    return {"trace": events, "profile": functions, "pstats": str(path), "parity": True}


def write_comparison(runs, output):
    # Diagnostic use is one fresh worker per side; all raw worker data retained.
    a, b = runs["baseline"][0], runs["candidate"][0]
    lines = [
        "# IndexedArray deferred-validation diagnostics",
        "",
        f"Baseline: `{a['sha']}`; candidate: `{b['sha']}`.",
        "",
        "All diagnostic samples passed CPU output/error parity. Trace and cProfile are separate warmed passes. They perturb execution: these are not benchmark latencies or pure GPU times. Trace host durations are inclusive and nested; do not sum them. Profile tables use exclusive self time. C-call get counts/bytes describe observed CuPy hooks, not a complete CUDA API trace. Missing hooks mean unobserved/unavailable, not zero device work. Parity work and its error queue are drained outside capture. Raw pstats retain caller relationships.",
    ]
    for key in a["cases"]:
        left, right = a["cases"][key]["diagnostics"], b["cases"][key]["diagnostics"]
        lines += [
            "",
            f"## {key}",
            "",
            "| Trace boundary | Calls/op baseline → candidate | Inclusive host us/op baseline → candidate | Observed get bytes/op baseline → candidate |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name in sorted(set(left["trace"]) | set(right["trace"])):

            def value(side, field, name=name):
                item = side["trace"].get(name)
                return "—" if item is None else f"{item[field]:.3f}"

            lines.append(
                f"| {name} | {value(left, 'calls_per_op')} → {value(right, 'calls_per_op')} | {value(left, 'inclusive_host_us_per_op')} → {value(right, 'inclusive_host_us_per_op')} | {value(left, 'bytes_per_op')} → {value(right, 'bytes_per_op')} |"
            )
        names = set(left["profile"]) | set(right["profile"])

        def self_time(side, name):
            return side["profile"].get(name, {}).get("self_us_per_op", 0)

        names = sorted(
            names,
            key=lambda name: self_time(right, name) - self_time(left, name),
            reverse=True,
        )[:20]
        lines += [
            "",
            "| Largest exclusive host increases | Baseline us/op | Candidate us/op | Delta us/op |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name in names:
            old, new = self_time(left, name), self_time(right, name)
            lines.append(f"| {name} | {old:.3f} | {new:.3f} | {new - old:+.3f} |")
    (output / "diagnostics.md").write_text("\n".join(lines) + "\n")
    (output / "diagnostics.json").write_text(json.dumps(runs, indent=2) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")
