# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from awkward._backends.numpy import NumpyBackend

pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "benchmarks/cuda_migrations"


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


workloads = load("workloads")
measure = load("measure")


@pytest.mark.parametrize("suite", workloads.SUITES)
@pytest.mark.parametrize("rows,width", [(1, 1), (4, 3), (32, 8)])
def test_workloads_native(suite, rows, width):
    names = []
    for name, values in workloads.workloads(suite, rows, width):
        names.append(name)
        original = measure.clone(values)
        result = NumpyBackend.instance()[measure.key_for(name, values)](*values)
        assert not result.str
        first = measure.clone(values)
        # Reconstruct/reset every argument, including in/out tmpstarts. The
        # second invocation must perform the same work and give identical data.
        values = measure.clone(original)
        result = NumpyBackend.instance()[measure.key_for(name, values)](*values)
        assert not result.str
        for before, after in zip(first, values, strict=True):
            np.testing.assert_array_equal(before, after)
    assert names == workloads.SUITES[suite]


def test_canonical_inventory():
    for suite, required in workloads.SUITES.items():
        cases = list(measure.canonical(ROOT, suite))
        assert {name for _, name, _ in cases} == set(required)
        assert len({label for label, _, _ in cases}) == len(cases)


def test_nsight(tmp_path):
    path = tmp_path / "trace.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES (1,'cudaStreamSynchronize'),(2,'cudaMemcpyAsync'),(3,'cudaMemcpy'),(4,'cudaMalloc');
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(nameId INTEGER, start INTEGER, end INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1,10,20),(1,30,35),(2,40,41),(3,42,46),(4,47,50);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10,20),(30,50);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(copyKind INTEGER, bytes INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (2,24),(2,16),(1,128);
            CREATE TABLE ENUM_CUDA_DEV_MEM_EVENT_OPER(id INTEGER, name TEXT);
            INSERT INTO ENUM_CUDA_DEV_MEM_EVENT_OPER VALUES (1,'ALLOCATION'),(2,'DEALLOCATION');
            CREATE TABLE CUDA_GPU_MEMORY_USAGE_EVENTS(start INTEGER,address INTEGER,bytes INTEGER,memKind INTEGER,memoryOperationType INTEGER);
            INSERT INTO CUDA_GPU_MEMORY_USAGE_EVENTS VALUES (1,100,256,2,1),(2,200,512,2,1),(3,100,256,2,2);
        """)
    result = measure.nsight_metrics(path)
    assert result["synchronization_calls"] == 2
    assert result["blocking_memcpy_calls"] == 1
    assert result["dtoh_bytes"] == 40
    assert result["kernel_busy_ns"] == 30
    assert result["driver_memory"]["peak_live_in_range_bytes"] == 768
    assert result["driver_memory"]["retained_in_range_bytes"] == 512
    assert result["driver_memory"]["complete"]


def test_missing_trace_is_not_zero(tmp_path):
    path = tmp_path / "empty.sqlite"
    with sqlite3.connect(path):
        pass
    with pytest.raises(ValueError, match="unmeasured"):
        measure.nsight_metrics(path)


def report():
    return {
        "environment": dict.fromkeys(
            [
                "device",
                "packages",
                "python",
                "cuda_runtime",
                "cuda_driver",
                "git_sha",
                "execution_environment",
            ],
            "same",
        ),
        "suite": "indexed",
        "settings": {"iterations": 20},
        "contract_hash": "123",
        "parity": [{"case": "edge", "status": "passed"}],
        "measurements": [
            {
                "case": "kernel:32:8",
                "status": "passed",
                "median": {"host_us": 10.0, "stream_us": 5.0, "total_us": 15.0},
                "cuda_trace": {
                    "synchronization_calls": 0,
                    "dtoh_bytes": 0,
                    "driver_memory": {"complete": True},
                },
            }
        ],
    }


def compare(tmp_path, before, after):
    baseline, candidate, output = [
        tmp_path / x for x in ["baseline.json", "candidate.json", "comparison.json"]
    ]
    baseline.write_text(json.dumps(before))
    candidate.write_text(json.dumps(after))
    return measure.compare(
        SimpleNamespace(
            baseline=baseline, candidate=candidate, output=output, max_regression=1.1
        )
    )


def test_comparison(tmp_path):
    assert compare(tmp_path, report(), report())["review_gate_passed"]


@pytest.mark.parametrize(
    "failure", ["slower", "parity", "trace", "allocation", "wrong-output"]
)
def test_comparison_rejects(tmp_path, failure):
    after = report()
    if failure == "slower":
        after["measurements"][0]["median"]["host_us"] = 20
    elif failure == "parity":
        after["parity"][0]["status"] = "failed"
    elif failure == "trace":
        after["measurements"][0]["cuda_trace"] = None
    elif failure == "allocation":
        after["measurements"][0]["cuda_trace"]["driver_memory"]["complete"] = False
    else:
        after["measurements"][0]["status"] = "parity-failed"
    assert not compare(tmp_path, report(), after)["review_gate_passed"]


@pytest.mark.parametrize(
    "field", ["device", "packages", "cuda_driver", "cuda_runtime", "python"]
)
def test_incomparable_environment(tmp_path, field):
    after = report()
    after["environment"][field] = "different"
    with pytest.raises(ValueError, match="Incomparable"):
        compare(tmp_path, report(), after)


def test_allocation_hook():
    hook = measure.allocation_hook(
        SimpleNamespace(cuda=SimpleNamespace(MemoryHook=object))
    )
    hook.malloc_preprocess(size=17)
    hook.malloc_postprocess(pmem_id=1, mem_size=512)
    hook.malloc_preprocess(size=1000)
    hook.alloc_preprocess(mem_size=1024)
    hook.malloc_postprocess(pmem_id=2, mem_size=1024)
    hook.free_postprocess(pmem_id=1)
    hook.free_postprocess(pmem_id=999)  # allocation made before accounting
    assert hook.requests == 2
    assert hook.requested_bytes == 1017
    assert hook.upstream_bytes == 1024
    assert hook.peak_live_bytes == 1536
    assert sum(hook.live.values()) == 1024


def test_worker_resets_inout_scratch(tmp_path, monkeypatch):
    import awkward._connect.cuda as cuda
    from awkward._backends.cupy import CupyBackend

    class DeviceArray(np.ndarray):
        def get(self, stream=None):
            return np.asarray(self).copy()

        def set(self, source, stream=None):
            self[:] = source

    class Stream:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def synchronize(self):
            pass

    class Event:
        def record(self, stream):
            pass

        def synchronize(self):
            pass

    class Hook(Stream):
        pass

    calls = []

    class Backend:
        def __getitem__(self, key):
            def call(*values):
                # The repeated performance cases have regular nonzero counts.
                # Check the original starts before permitting native writes;
                # a missed reset would otherwise write beyond the output.
                if values[-1] >= 32:
                    np.testing.assert_array_equal(
                        values[2], np.arange(values[-1]) * int(values[4][0])
                    )
                calls.append(key[0])
                return NumpyBackend.instance()[key](*values)

            return call

    cp = SimpleNamespace(
        asarray=lambda x: np.asarray(x).copy().view(DeviceArray),
        cuda=SimpleNamespace(
            Stream=Stream,
            Event=Event,
            MemoryHook=Hook,
            get_elapsed_time=lambda a, b: 1.0,
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(CupyBackend, "instance", Backend)
    monkeypatch.setattr(cuda, "synchronize_cuda", lambda stream: None)
    monkeypatch.setattr(
        measure, "environment", lambda cp, checkout: report()["environment"]
    )
    output = tmp_path / "worker.json"
    measure.worker(
        SimpleNamespace(
            checkout=ROOT,
            contracts=ROOT,
            output=output,
            suite="unionarray",
            trace_case=None,
            iterations=2,
            repeats=2,
            warmup=1,
            nsys=False,
        )
    )
    data = json.loads(output.read_text())
    assert all(x["status"] == "passed" for x in data["parity"])
    assert len(data["measurements"]) == 3
    assert len(calls) == len(data["parity"]) + 3 * 7
    assert all(x["cuda_trace"] is None for x in data["measurements"])
