# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Guard-only host overhead. Run CPU locally or CUDA on an allocated GPU node."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import awkward as ak
from awkward._connect.cuda._index_guard import PendingValidation, guard_index
from awkward._nplikes.array_like import maybe_materialize
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.numpy import Numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")
    if args.backend == "cuda":
        xp = Cupy.instance()._module
    else:
        xp = Numpy.instance()._module
    source = ak.index.Index64(xp.arange(32, dtype=xp.int64))
    ready = PendingValidation(lambda: None)
    ready.check()
    guarded = guard_index(source, ready)
    maybe_materialize(guarded.data)
    if args.backend == "cuda":
        xp.cuda.get_current_stream().synchronize()

    def plain():
        return ak.index.Index64(source.data, nplike=source.nplike)

    def wrapped():
        return guard_index(source, PendingValidation(lambda: None))

    factories = {
        "construct": (lambda: plain, lambda: wrapped),
        "metadata": (lambda: lambda: source.length, lambda: lambda: guarded.length),
        "validated_consume": (
            lambda: lambda: maybe_materialize(source.data)[0],
            lambda: lambda: maybe_materialize(guarded.data)[0],
        ),
        "first_consume_noop_validation": (
            lambda: iter([source] * args.iterations),
            lambda: iter([wrapped() for _ in range(args.iterations)]),
        ),
    }
    if args.backend == "cuda":
        status = xp.asarray([32, 0], dtype=xp.int64)
        producer = xp.cuda.get_current_stream()

        def transfer():
            row, _ = status.get(stream=producer).tolist()
            if row < 32:
                raise ValueError("invalid index")

        transfer()  # Warm transfer machinery outside timing.
        factories["first_consume_status_transfer"] = (
            lambda: iter([source] * args.iterations),
            lambda: iter(
                [
                    guard_index(source, PendingValidation(transfer))
                    for _ in range(args.iterations)
                ]
            ),
        )
    result = {}
    for name, factories_pair in factories.items():
        samples = [[], []]
        for repeat in range(args.repeats):
            for side in (0, 1) if repeat % 2 == 0 else (1, 0):
                operation = factories_pair[side]()
                start = time.perf_counter_ns()
                if name.startswith("first_consume"):
                    if name == "first_consume_status_transfer" and side == 0:
                        for _ in range(args.iterations):
                            transfer()
                            maybe_materialize(next(operation).data)
                    else:
                        for _ in range(args.iterations):
                            maybe_materialize(next(operation).data)
                else:
                    for _ in range(args.iterations):
                        operation()
                samples[side].append(
                    (time.perf_counter_ns() - start) / args.iterations / 1000
                )
        a, b = (statistics.median(values) for values in samples)
        result[name] = {
            "plain_us": a,
            "guard_us": b,
            "delta_us": b - a,
            "samples_us": samples,
        }
    report = {
        "backend": args.backend,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "note": "Host wrapper overhead only; no simplify kernel, public analysis or GPU speedup measurement. CUDA status-transfer row includes a completed 16-byte get on both sides; other rows have no validation transfer. First-consume preparation is excluded; construction measured separately. Alternating paired batches.",
        "results": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    lines = [report["note"]]
    lines.append("| Boundary | Plain host us | Guard host us | Delta us |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, values in result.items():
        lines.append(
            f"| {name} | {values['plain_us']:.3f} | {values['guard_us']:.3f} | {values['delta_us']:+.3f} |"
        )

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
