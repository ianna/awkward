# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""GPU parity, warmed timing, allocation accounting and Nsight comparison."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from workloads import SHAPES, SUITES, workloads


def clone(values):
    return [x.copy() if isinstance(x, np.ndarray) else x for x in values]


def key_for(name, values):
    return (name, *(x.dtype.type for x in values if isinstance(x, np.ndarray)))


def canonical(checkout, suite):
    specs = yaml.safe_load((checkout / "kernel-specification.yml").read_text())[
        "kernels"
    ]
    data = {
        x["name"]: x["tests"]
        for x in json.loads((checkout / "kernel-test-data.json").read_text())[
            "unit-tests"
        ]
    }
    for spec in specs:
        if spec["name"] not in SUITES[suite]:
            continue
        for specialization in spec["specializations"]:
            for number, case in enumerate(data.get(spec["name"], [])):
                args = []
                representable = True
                for argument in specialization["args"]:
                    name, typename = argument["name"], argument["type"]
                    value = case["inputs"].get(name)
                    if "List[" in typename:
                        dtype = np.dtype(
                            typename.split("List[")[1].split("]")[0].removesuffix("_t")
                        )
                        source = value if value is not None else []
                        limits = np.iinfo(dtype)
                        if any(x < limits.min or x > limits.max for x in source):
                            representable = False
                            break
                        if argument["dir"] == "out":
                            capacity = max(
                                len(source), len(case["outputs"].get(name, []))
                            )
                            value = np.full(capacity + 4, limits.max - 2, dtype=dtype)
                            value[: len(source)] = source
                        else:
                            value = np.array(source, dtype=dtype)
                    args.append(value)
                if representable:
                    yield f"{specialization['name']}:{number}", spec["name"], args


def nsight_metrics(path):
    """Extract actual API calls, not Python wrapper counts, from an nsys export."""
    result = {
        "kernel_busy_ns": 0,
        "dtoh_bytes": 0,
        "driver_memory": {
            "events": [],
            "allocated_bytes": 0,
            "peak_live_in_range_bytes": 0,
            "retained_in_range_bytes": 0,
            "complete": True,
        },
        "api_calls": {},
        "memcpy_kinds": {},
        "synchronization_calls": 0,
        "blocking_memcpy_calls": 0,
        "synchronization_api_ns": 0,
        "query_calls": 0,
        "allocation_free_api_calls": 0,
    }
    with sqlite3.connect(path) as db:
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        bounds = None
        if "NVTX_EVENTS" in tables:
            ranges = list(
                db.execute(
                    "SELECT n.start, n.end FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE COALESCE(n.text,s.value)='measure' AND n.end IS NOT NULL"
                )
            )
            if len(ranges) != 1:
                raise ValueError("Expected exactly one completed measure NVTX range")
            bounds = ranges[0]
        result["api_scope"] = "measure" if bounds else "legacy-capture"
        found = False
        for table in ["CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"]:
            if table not in tables:
                continue
            found = True
            where = "WHERE a.start >= ? AND a.start <= ?" if bounds else ""
            query = f"SELECT s.value, COUNT(*), SUM(a.end-a.start) FROM {table} a JOIN StringIds s ON a.nameId=s.id {where} GROUP BY s.value"
            for name, count, duration in db.execute(query, bounds or ()):
                result["api_calls"][f"{table}:{name}"] = {
                    "count": count,
                    "total_ns": duration,
                }
                if "Synchronize" in name:
                    result["synchronization_calls"] += count
                    result["synchronization_api_ns"] += duration
                if "Query" in name:
                    result["query_calls"] += count
                if any(fragment in name for fragment in ("Malloc", "MemAlloc", "Free")):
                    result["allocation_free_api_calls"] += count
                if "Memcpy" in name and "Async" not in name:
                    result["blocking_memcpy_calls"] += count
        if not found:
            raise ValueError(
                "Nsight export contains no CUDA API table; synchronization is unmeasured"
            )
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            raise ValueError("Nsight export contains no GPU kernel activity")
        result["kernel_busy_ns"] = (
            db.execute(
                "SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()[0]
            or 0
        )
        if result["kernel_busy_ns"] <= 0:
            raise ValueError("Nsight export contains no measured GPU kernel activity")
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
            for kind, count, size in db.execute(
                "SELECT copyKind, COUNT(*), SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"
            ):
                result["memcpy_kinds"][str(kind)] = {"count": count, "bytes": size}
                if kind in (2, 4, 12):
                    result["dtoh_bytes"] += size
        memory = result["driver_memory"]
        if "CUDA_GPU_MEMORY_USAGE_EVENTS" in tables:
            live = {}
            query = "SELECT m.start, m.address, m.bytes, m.memKind, COALESCE(e.name,'unknown') FROM CUDA_GPU_MEMORY_USAGE_EVENTS m LEFT JOIN ENUM_CUDA_DEV_MEM_EVENT_OPER e ON m.memoryOperationType=e.id ORDER BY m.start"
            if bounds:
                query = query.replace(
                    " ORDER BY m.start",
                    " WHERE m.start >= ? AND m.start <= ? ORDER BY m.start",
                )
            for timestamp, address, size, kind, operation in db.execute(
                query, bounds or ()
            ):
                memory["events"].append(
                    {
                        "timestamp_ns": timestamp,
                        "address": address,
                        "bytes": size,
                        "kind": kind,
                        "operation": operation,
                    }
                )
                if kind not in (2, 3, 4):
                    continue
                op = operation.lower()
                if "free" in op or "dealloc" in op or "release" in op:
                    live.pop(address, None)
                elif "alloc" in op:
                    live[address] = size
                    memory["allocated_bytes"] += size
                else:
                    memory["complete"] = False
                memory["peak_live_in_range_bytes"] = max(
                    memory["peak_live_in_range_bytes"], sum(live.values())
                )
            memory["retained_in_range_bytes"] = sum(live.values())
        elif any(
            "Malloc" in name or "MemAlloc" in name for name in result["api_calls"]
        ):
            # Do not misreport unavailable foreign allocation sizes as zero.
            memory["complete"] = False
    return result


def allocation_hook(cp):
    class Hook(cp.cuda.MemoryHook):
        name = "awkward-migration-allocations"

        def __init__(self):
            self.requests = 0
            self.requested_bytes = 0
            self.upstream_bytes = 0
            self.live = {}
            self.peak_live_bytes = 0

        def malloc_preprocess(self, **kw):
            self.requests += 1
            self.requested_bytes += kw["size"]

        def alloc_preprocess(self, **kw):
            self.upstream_bytes += kw["mem_size"]

        def malloc_postprocess(self, **kw):
            if kw["pmem_id"]:
                self.live[kw["pmem_id"]] = kw["mem_size"]
                self.peak_live_bytes = max(
                    self.peak_live_bytes, sum(self.live.values())
                )

        def free_postprocess(self, **kw):
            self.live.pop(kw["pmem_id"], None)

    return Hook()


def environment(cp, checkout):
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"].lower().replace("_", "-")
        not in ("awkward", "awkward-cpp")
    }
    import awkward_cpp.cpu_kernels as native

    telemetry = None
    try:
        snapshot = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,temperature.gpu,clocks.sm,clocks.mem,power.draw",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        telemetry = snapshot.stdout if snapshot.returncode == 0 else snapshot.stderr
    except FileNotFoundError:
        pass
    return {
        "git_sha": subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.version,
        "gpu_telemetry": telemetry,
        "execution_environment": {
            name: os.environ.get(name)
            for name in [
                "CUDA_VISIBLE_DEVICES",
                "CUDA_LAUNCH_BLOCKING",
                "CUDA_MODULE_LOADING",
                "CUPY_ACCELERATORS",
                "CUPY_CUDA_ARRAY_INTERFACE_SYNC",
                "NUMBA_CUDA_USE_NVIDIA_BINDING",
            ]
        },
        "native_library_sha256": hashlib.sha256(
            Path(native.lib._name).read_bytes()
        ).hexdigest(),
        "packages": versions,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "device": {
            k: ((v.decode() if k == "name" else v.hex()) if isinstance(v, bytes) else v)
            for k, v in properties.items()
            if k
            in [
                "name",
                "uuid",
                "pciBusID",
                "pciDeviceID",
                "major",
                "minor",
                "multiProcessorCount",
                "totalGlobalMem",
            ]
        },
    }


def capture_call(nvtx, call, finish):
    # The outer capture must include completion of asynchronously launched work.
    # Only APIs inside the inner range count toward the measured operation.
    nvtx.RangePush("capture")
    try:
        nvtx.RangePush("measure")
        try:
            call()
        finally:
            nvtx.RangePop()
            finish()
    finally:
        nvtx.RangePop()


def worker(args):
    import cupy as cp

    import awkward as ak
    import awkward._connect.cuda as cuda
    from awkward._backends.cupy import CupyBackend
    from awkward._backends.numpy import NumpyBackend

    checkout = Path(args.checkout).resolve()
    if checkout not in Path(ak.__file__).resolve().parents:
        raise ValueError(
            "Imported awkward does not belong to --checkout; install the checkout first"
        )
    # Explicitly isolate a nondefault stream to exercise stream forwarding.
    stream = cp.cuda.Stream(non_blocking=True)
    gpu_backend, cpu_backend = CupyBackend.instance(), NumpyBackend.instance()

    def setup(name, values):
        original = clone(values)
        host = clone(values)
        key = key_for(name, host)
        expected_error = cpu_backend[key](*host)
        device = [cp.asarray(x) if isinstance(x, np.ndarray) else x for x in original]
        kernel = gpu_backend[key]

        def reset():
            for target, source in zip(device, original, strict=True):
                if isinstance(source, np.ndarray):
                    target.set(source, stream=stream)

        def call():
            return kernel(*device)

        return original, host, expected_error, device, reset, call

    def parity(name, values):
        _original, expected, cpu_error, device, _reset, call = setup(name, values)
        record = None
        raised = None
        try:
            record = call()
            cuda.synchronize_cuda(stream)
        except Exception as error:
            raised = str(error)
        expected_message = cpu_error.str.decode() if cpu_error.str else None
        actual_message = (
            record.str.decode() if record is not None and record.str else raised
        )
        if expected_message:
            if not actual_message or expected_message not in actual_message:
                raise AssertionError(
                    f"CPU error {expected_message!r}; CUDA error {actual_message!r}"
                )
            if record is not None:
                assert (record.id, record.attempt) == (cpu_error.id, cpu_error.attempt)
        elif actual_message:
            raise AssertionError(actual_message)
        # Compare every buffer, including output tails and unchanged inputs.
        for i, value in enumerate(expected):
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(
                    device[i].get(stream=stream), value, err_msg=f"{name} argument {i}"
                )
        return {
            "status": "passed",
            "error_metadata": "record"
            if record is not None
            else "exception-message-only"
            if expected_message
            else "none",
        }

    contracts = Path(args.contracts or checkout).resolve()
    contract_hash = hashlib.sha256(
        (contracts / "kernel-specification.yml").read_bytes()
        + (contracts / "kernel-test-data.json").read_bytes()
    ).hexdigest()
    Path(args.output).resolve().parent.mkdir(parents=True, exist_ok=True)
    results = {
        "environment": environment(cp, checkout),
        "suite": args.suite,
        "contract_hash": contract_hash,
        "settings": {
            "capture_version": 2,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "parity": [],
        "measurements": [],
    }

    def checkpoint():
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")

    if not args.trace_case:
        checkpoint()
    with stream:
        if not args.trace_case:
            for label, name, values in canonical(contracts, args.suite):
                try:
                    record = parity(name, values)
                except Exception as error:
                    record = {"status": "failed", "error": repr(error)}
                results["parity"].append({"case": label, **record})
                checkpoint()
        for rows, width in SHAPES:
            for name, values in workloads(args.suite, rows, width):
                label = f"{name}:{rows}:{width}"
                if args.trace_case and args.trace_case != label:
                    continue
                _original, _host, error, _device, reset, call = setup(name, values)
                assert not error.str
                # Skip fast-but-wrong implementations, while recording the failure.
                try:
                    parity(name, values)
                except Exception as error:
                    results["measurements"].append(
                        {"case": label, "status": "parity-failed", "error": repr(error)}
                    )
                    continue
                for _ in range(args.warmup):
                    reset()
                    call()
                    cuda.synchronize_cuda(stream)
                if args.trace_case:
                    reset()
                    stream.synchronize()
                    capture_call(cp.cuda.nvtx, call, stream.synchronize)
                    cuda.synchronize_cuda(stream)
                    return
                samples = []
                for _ in range(args.repeats):
                    host_ns = total_ns = device_ms = 0
                    for _ in range(args.iterations):
                        # In/out scratch (notably tmpstarts) must be reset before
                        # EVERY call, outside both wall and CUDA-event intervals.
                        reset()
                        stream.synchronize()
                        start, end = cp.cuda.Event(), cp.cuda.Event()
                        start.record(stream)
                        before = time.perf_counter_ns()
                        call()
                        host_end = time.perf_counter_ns()
                        end.record(stream)
                        end.synchronize()
                        cuda.synchronize_cuda(stream)
                        total_ns += time.perf_counter_ns() - before
                        host_ns += host_end - before
                        device_ms += cp.cuda.get_elapsed_time(start, end)
                    samples.append(
                        {
                            "host_us": host_ns / args.iterations / 1000,
                            "total_us": total_ns / args.iterations / 1000,
                            "stream_us": device_ms * 1000 / args.iterations,
                        }
                    )
                reset()
                stream.synchronize()
                hook = allocation_hook(cp)
                with hook:
                    call()
                stream.synchronize()
                entry = {
                    "case": label,
                    "status": "passed",
                    "samples": samples,
                    "median": {
                        key: statistics.median(x[key] for x in samples)
                        for key in samples[0]
                    },
                    "cupy_pool": {
                        "allocation_requests": hook.requests,
                        "requested_bytes": hook.requested_bytes,
                        "upstream_bytes": hook.upstream_bytes,
                        "peak_live_temporary_bytes": hook.peak_live_bytes,
                        "retained_bytes": sum(hook.live.values()),
                    },
                }
                entry["cuda_trace"] = None
                results["measurements"].append(entry)
                checkpoint()
                if args.nsys:
                    trace = Path(args.output).resolve().parent / label.replace(":", "-")
                    command = [
                        "nsys",
                        "profile",
                        "--trace=cuda,nvtx",
                        "--cuda-memory-usage=true",
                        "--sample=none",
                        "--capture-range=nvtx",
                        "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
                        "--nvtx-capture=capture",
                        "--capture-range-end=stop",
                        "--force-overwrite=true",
                        f"--output={trace}",
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "run",
                        "--checkout",
                        str(checkout),
                        "--suite",
                        args.suite,
                        "--contracts",
                        str(contracts),
                        "--trace-case",
                        label,
                        "--output",
                        str(trace) + ".json",
                        "--warmup",
                        str(args.warmup),
                    ]
                    subprocess.run(command, check=True)
                    subprocess.run(
                        [
                            "nsys",
                            "export",
                            "--type=sqlite",
                            "--force-overwrite=true",
                            f"--output={trace}.sqlite",
                            f"{trace}.nsys-rep",
                        ],
                        check=True,
                    )
                    entry["cuda_trace"] = nsight_metrics(str(trace) + ".sqlite")
                else:
                    entry["cuda_trace"] = None
                checkpoint()
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


def compare(args):
    baseline, candidate = [
        json.loads(Path(x).read_text()) for x in [args.baseline, args.candidate]
    ]
    for field in [
        "device",
        "packages",
        "python",
        "cuda_runtime",
        "cuda_driver",
        "execution_environment",
    ]:
        if baseline["environment"][field] != candidate["environment"][field]:
            raise ValueError(f"Incomparable environments: {field}")
    if baseline["suite"] != candidate["suite"]:
        raise ValueError("Mismatched suites")
    for field in ["contract_hash", "settings"]:
        if baseline[field] != candidate[field]:
            raise ValueError(f"Mismatched {field}")
    if not candidate["parity"] or {x["case"] for x in baseline["parity"]} != {
        x["case"] for x in candidate["parity"]
    }:
        raise ValueError("Missing or mismatched parity inventory")
    old = {x["case"]: x for x in baseline["measurements"]}
    new = {x["case"]: x for x in candidate["measurements"]}
    if old.keys() != new.keys():
        raise ValueError("Mismatched workload inventory")
    report = {
        "baseline_sha": baseline["environment"]["git_sha"],
        "candidate_sha": candidate["environment"]["git_sha"],
        "candidate_parity_passed": all(
            x["status"] == "passed" for x in candidate["parity"]
        ),
        "cases": [],
    }
    for case, after in new.items():
        before = old[case]
        entry = {"case": case, "baseline": before, "candidate": after}
        if before["status"] == after["status"] == "passed":
            entry["candidate_over_baseline"] = {
                key: after["median"][key] / before["median"][key]
                for key in before["median"]
            }
        entry["regressions"] = [
            key
            for key, ratio in entry.get("candidate_over_baseline", {}).items()
            if ratio > args.max_regression
        ]
        report["cases"].append(entry)
    report["complete_measurements"] = bool(report["cases"]) and all(
        x["baseline"]["status"] == x["candidate"]["status"] == "passed"
        and x["baseline"].get("cuda_trace") is not None
        and x["candidate"].get("cuda_trace") is not None
        and x["baseline"]["cuda_trace"]["driver_memory"]["complete"]
        and x["candidate"]["cuda_trace"]["driver_memory"]["complete"]
        for x in report["cases"]
    )
    report["no_timing_regressions"] = not any(x["regressions"] for x in report["cases"])
    report["review_gate_passed"] = (
        report["candidate_parity_passed"]
        and report["complete_measurements"]
        and report["no_timing_regressions"]
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"# CUDA migration comparison: {candidate['suite']}",
        "",
        f"Baseline: `{report['baseline_sha']}`; candidate: `{report['candidate_sha']}`.",
        "",
        f"Candidate canonical parity: {report['candidate_parity_passed']}. Review gate: {report['review_gate_passed']}.",
        "",
        "Values show baseline → candidate. Timings are unprofiled medians; trace counts and memory come from separate passes.",
        "",
        "| Workload | Host µs | Stream µs | Sync API calls | DToH bytes | Pool temporary peak bytes | Driver allocation peak bytes | Timing regressions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report["cases"]:
        before, after = entry["baseline"], entry["candidate"]
        if before["status"] != "passed" or after["status"] != "passed":
            lines.append(
                f"| {entry['case']} | {before['status']} → {after['status']} | | | | | | |"
            )
            continue

        def timing(key, before=before, after=after):
            return f"{before['median'][key]:.3f} → {after['median'][key]:.3f}"

        def traced(key, before=before, after=after):
            return " → ".join(
                str(x["cuda_trace"][key]) if x.get("cuda_trace") else "unmeasured"
                for x in (before, after)
            )

        pool = " → ".join(
            str(x.get("cupy_pool", {}).get("peak_live_temporary_bytes", "unmeasured"))
            for x in (before, after)
        )
        driver = " → ".join(
            str(
                x["cuda_trace"]["driver_memory"].get(
                    "peak_live_in_range_bytes", "unmeasured"
                )
            )
            if x.get("cuda_trace")
            else "unmeasured"
            for x in (before, after)
        )
        lines.append(
            f"| {entry['case']} | {timing('host_us')} | {timing('stream_us')} | {traced('synchronization_calls')} | {traced('dtoh_bytes')} | {pool} | {driver} | {', '.join(entry['regressions']) or 'none'} |"
        )
    Path(args.output).with_suffix(".md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--checkout", required=True)
    run.add_argument(
        "--contracts",
        help="Shared checkout supplying canonical spec/cases for both runs",
    )
    run.add_argument("--suite", choices=SUITES, required=True)
    run.add_argument("--iterations", type=int, default=20)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--nsys", action="store_true")
    run.add_argument("--trace-case")
    run.add_argument("--output", required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--baseline", required=True)
    comparison.add_argument("--candidate", required=True)
    comparison.add_argument("--output", required=True)
    comparison.add_argument("--max-regression", type=float, default=1.10)
    comparison.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        if min(args.iterations, args.repeats, args.warmup) <= 0:
            parser.error("iteration, repeat and warmup counts must be positive")
        worker(args)
    else:
        report = compare(args)
        if args.require_pass and not report["review_gate_passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
