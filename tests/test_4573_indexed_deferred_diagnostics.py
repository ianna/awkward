# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location(
    "indexed_deferred_profiles",
    Path(__file__).resolve().parents[1] / "studies/indexed_deferred/profiles.py",
)
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)


def test_separate_diagnostic_passes_exclude_parity(tmp_path):
    cupy = SimpleNamespace(empty=np.empty, ndarray=np.ndarray)
    original = cupy.empty
    checks, cleanups = [], []

    def sample():
        return cupy.empty(2), [], 0, 0

    def check(result, errors):
        assert len(result) == 2
        assert not errors
        cupy.empty(10)  # must not inflate traced operation allocations
        checks.append(1)

    result = profiles.diagnose(
        sample,
        check,
        samples=3,
        directory=tmp_path,
        key="valid:field",
        cupy=cupy,
        cleanup=lambda: cleanups.append(1),
    )
    assert cupy.empty is original
    assert len(checks) == len(cleanups) == 6
    assert result["trace"]["allocation.cupy_empty"]["calls_per_op"] == 1
    assert Path(result["pstats"]).is_file()
    assert result["profile"]
    runs = {
        name: [{"sha": name, "cases": {"valid:field": {"diagnostics": result}}}]
        for name in ("baseline", "candidate")
    }
    profiles.write_comparison(runs, tmp_path)
    report = (tmp_path / "diagnostics.md").read_text()
    assert "not benchmark latencies" in report
    assert "valid:field" in report


def test_trace_restores_functions_and_profile_hook_on_error():
    import sys

    target = SimpleNamespace(operation=lambda: None)
    original = target.operation
    hook = sys.getprofile()
    cupy = SimpleNamespace(ndarray=np.ndarray)
    try:
        with profiles.Trace([(target, "operation", "test")], cupy) as trace:
            with trace.capture():
                target.operation()
                raise ValueError("diagnostic failure")
    except ValueError:
        pass
    assert target.operation is original
    assert sys.getprofile() is hook
    assert trace.events["test"]["calls"] == 1
