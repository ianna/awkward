# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Alternating fresh-process public-operation comparison; run on a GPU node."""

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def worker(args):
    import awkward as ak
    import awkward._connect.cuda as cuda
    from awkward._nplikes.cupy import Cupy
    from awkward._nplikes.numpy import Numpy

    cp = Cupy.instance()._module
    np = Numpy.instance()._module

    source_path = Path(ak.__file__).resolve()
    assert source_path.is_relative_to(args.source.resolve()), str(source_path)
    reports = {}
    stream = cp.cuda.Stream(non_blocking=True)

    def finish():
        if stream.ptr in cuda.cuda_streamptr_to_contexts:
            cuda.synchronize_cuda(stream)
        else:
            stream.synchronize()

    def operation(source, consume):
        result = source["x"]
        return ak.to_backend(result, "cpu") if consume else result

    def sample(source, consume):
        errors = []
        result = None
        start = time.perf_counter_ns()
        try:
            result = operation(source, consume)
        except Exception as error:
            errors.append(("call", error))
        host = time.perf_counter_ns()
        try:
            finish()
        except Exception as error:
            errors.append(("finish", error))
        end = time.perf_counter_ns()
        return result, errors, (host - start) / 1000, (end - start) / 1000

    for rows, width in ((32, 8), (4096, 8), (32, 4096)):
        for outer_option in (False, True):
            for inner_option in (False, True):
                for invalid in (False, True):
                    index = np.arange(rows, dtype=np.int64) % width
                    inner_index = np.arange(width - 1, -1, -1, dtype=np.int64)
                    if outer_option:
                        index[::5] = -1
                    if inner_option:
                        inner_index[::3] = -1
                    if invalid:
                        index[rows // 2] = width
                    outer_cls = (
                        ak.contents.IndexedOptionArray
                        if outer_option
                        else ak.contents.IndexedArray
                    )
                    inner_cls = (
                        ak.contents.IndexedOptionArray
                        if inner_option
                        else ak.contents.IndexedArray
                    )
                    inner = inner_cls(
                        ak.index.Index64(inner_index),
                        ak.contents.NumpyArray(np.arange(width, dtype=np.int64)),
                    )
                    record = ak.contents.RecordArray([inner], ["x"])
                    cpu = ak.Array(outer_cls(ak.index.Index64(index), record))
                    with stream:
                        gpu = ak.to_backend(cpu, "cuda")
                        finish()
                    for consume in (False, True):
                        key = f"{'option' if outer_option else 'indexed'}_{'option' if inner_option else 'indexed'}:{rows}:{width}:{'bounds' if invalid else 'valid'}:{'to_cpu' if consume else 'field'}"
                        expected = None
                        expected_error = None
                        try:
                            expected = ak.to_list(operation(cpu, consume))
                        except Exception as error:
                            expected_error = type(error)
                            assert "index out of range" in str(error)

                        def check(
                            result,
                            errors,
                            expected_error=expected_error,
                            expected=expected,
                            key=key,
                        ):
                            if expected_error is not None:
                                assert errors, f"{key}: missing error"
                                assert all(
                                    isinstance(e, expected_error)
                                    and "index out of range" in str(e)
                                    for _, e in errors
                                ), (key, errors)
                            else:
                                assert not errors, (key, errors)
                                assert ak.to_list(result) == expected, key

                        with stream:
                            for _ in range(args.warmup):
                                result, errors, _, _ = sample(gpu, consume)
                                check(result, errors)
                            hosts, totals = [], []
                            stages = {"call": 0, "finish": 0}
                            for _ in range(args.samples):
                                result, errors, host, total = sample(gpu, consume)
                                check(result, errors)  # outside both timing boundaries
                                hosts.append(host)
                                totals.append(total)
                                for stage, _ in errors:
                                    stages[stage] += 1
                        reports[key] = {
                            "host_us": statistics.median(hosts),
                            "completed_us": statistics.median(totals),
                            "host_samples_us": hosts,
                            "completed_samples_us": totals,
                            "error_stages": stages,
                            "parity": True,
                        }
    versions = {"python": sys.version, "cupy": cp.__version__, "numpy": np.__version__}
    for distribution in (
        "awkward-cpp",
        "cccl-python",
        "cuda-cccl",
        "numba-cuda",
        "numba-cuda-mlir",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    args.output.write_text(
        json.dumps(
            {
                "source": str(source_path),
                "sha": git_sha(args.source),
                "environment": versions,
                "cases": reports,
            },
            indent=2,
        )
        + "\n"
    )


def git_sha(source):
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.pairs < 1 or args.warmup < 1:
        parser.error("samples, pairs and warmup must be positive")
    if args.worker:
        worker(args)
        return
    if args.baseline is None or args.candidate is None:
        parser.error("baseline and candidate checkouts are required")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {
        "baseline": args.baseline.resolve(),
        "candidate": args.candidate.resolve(),
    }
    expected_shas = {name: git_sha(source) for name, source in sources.items()}
    runs = {name: [] for name in sources}
    for pair in range(args.pairs):
        for name in (
            ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
        ):
            source = sources[name]
            output = (args.output / f"{pair + 1}-{name}.json").resolve()
            env = dict(os.environ, PYTHONPATH=str(source / "src"))
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--source",
                    str(source),
                    "--samples",
                    str(args.samples),
                    "--warmup",
                    str(args.warmup),
                    "--output",
                    str(output),
                ],
                env=env,
                check=True,
            )
            run = json.loads(output.read_text())
            assert run["sha"] == expected_shas[name], (
                "checkout changed during comparison"
            )
            runs[name].append(run)
    lines = [
        "# Public IndexedArray deferred-validation comparison",
        "",
        f"Baseline: `{git_sha(sources['baseline'])}`; candidate: `{git_sha(sources['candidate'])}`.",
        "",
        "All samples passed CPU output/error parity. Three alternating fresh-process pairs by default. Inputs, JIT warmup and comparisons are outside timing. Completed calls include result construction, producer-stream completion and error draining. `to_cpu` additionally includes consuming/transporting the result. No profiling is enabled. Values are medians of per-process medians; timing ratios are candidate / baseline.",
        "",
        "| Case | Baseline completed us | Candidate completed us | Ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    summary = {}
    for key in runs["baseline"][0]["cases"]:
        a, b = [
            statistics.median(run["cases"][key]["completed_us"] for run in runs[name])
            for name in ("baseline", "candidate")
        ]
        summary[key] = {
            "baseline_completed_us": a,
            "candidate_completed_us": b,
            "ratio": b / a,
        }
        lines.append(f"| {key} | {a:.3f} | {b:.3f} | {b / a:.3f} |")
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
