# Guarded public IndexedArray simplification

Experimental integration based on PR #4484 at
`1c44307ec0fee1e6bf2fc06cef5ae49f4d093918`, including the null-count stream import fix, plus the deferred-status and guard
prototype. AI-assisted implementation/tests: OpenAI Codex.

Only the public IndexedArray/IndexedOptionArray `simplified` constructors opt in.
They register status, retain input/output/workspace, and wrap the output index
before publication. Guarded index subclasses validate before backend transfer;
ordinary user-created virtual arrays keep their explicit-materialization rule.
Direct calls to the kernel keep their synchronous KernelError
contract, including canonical first-error position/attempt and untouched tail.
CPU/JAX/typetracer execution remains on the original constructor path. No native
implementation, canonical signature, dependency or package version changes.

The guard fences validation at first consumption, including raw CUDA export and
CuPy indexing via the existing virtual-array materialization machinery. A drain
can resolve it earlier; persistent invocation records preserve a failure after
that drain. Failed records block subsequent use; a fresh, valid operation can
still proceed on the stream. This remains an experimental virtual-index wrapper,
not a device-only pipeline guard or a guarantee of end-to-end speedup.

## Della setup (activated awkward-cuda-bench environment)

Fetch on a node with GitHub access, before submitting work to a GPU node:

```bash
cd /home/iosborne/Projects/CUDA_BENCH

git -C awkward-matrix fetch https://github.com/ianna/awkward.git \
  ianna/cuda-indexed-simplify-deferred

git -C awkward-matrix worktree add --detach \
  ../awkward-indexed-deferred FETCH_HEAD

git -C awkward-matrix worktree add --detach \
  ../awkward-indexed-baseline 1c44307ec0fee1e6bf2fc06cef5ae49f4d093918

# Generate canonical signatures in both fresh worktrees (requires PyYAML).
python awkward-indexed-baseline/dev/generate-kernel-signatures.py
python awkward-indexed-deferred/dev/generate-kernel-signatures.py

# Prepare editable metadata/version files for both pure-Python checkouts.
python -m pip install --no-deps --no-build-isolation -e awkward-indexed-baseline
python -m pip install --no-deps --no-build-isolation -e awkward-indexed-deferred
```

## GPU validation and unprofiled comparison

Run on an allocated GPU node. The native awkward-cpp 59 already in the benchmark
environment is shared by both variants; no native rebuild is needed.

```bash
cd /home/iosborne/Projects/CUDA_BENCH/awkward-indexed-deferred

python -m pytest -q --tb=short \
  tests/test_4484_indexedarray_simplify_compute.py \
  tests/test_4484_cuda_deferred_status.py \
  tests/test_4573_cuda_index_guard.py \
  tests-cuda/test_4484_indexedarray_simplify_compute.py \
  tests-cuda/test_4573_indexed_simplify_deferred.py \
  tests-cuda/test_4573_cuda_index_guard.py \
  tests-cuda/test_4484_cuda_deferred_status.py

python studies/indexed_deferred/compare.py \
  --baseline ../awkward-indexed-baseline \
  --candidate . \
  --pairs 3 --samples 100 --warmup 3 \
  --output results/indexed-deferred

cat results/indexed-deferred/comparison.md
```

Use `&&` between validation and comparison when submitting a combined batch job.
The driver runs three fresh-process pairs in B/C, C/B, B/C order, setting each
worker's PYTHONPATH explicitly and verifying its loaded source and SHA. CPU
references, inputs, warmup and every-sample parity checks are outside timing.

Four indexed/option combinations, valid and bounds-error inputs, three shapes
(32:8, 4096:8, 32:4096), and two operations give 48 public cases. `field` measures
selection, construction and producer-stream completion/error draining; `to_cpu`
also includes consuming and transporting the result. Host-call and completed-call
samples plus error stages are retained per worker. Results use medians of worker
medians; ratios are candidate/baseline. No profiler or CUDA-event timing is used.
A parity failure aborts the comparison rather than producing a passing report.

GPU execution and timing results for this integration are pending. The earlier
Della guard-only results do not establish a speedup for these public operations.

## Separate warmed regression diagnostics

The user-reported three-pair Della comparison of baseline `1c44307ec` against
candidate `fe4d5ff3c` passed CPU output/error parity for all 48 cases, but every
completed-call median regressed:

| Operation | Candidate / baseline range |
| --- | ---: |
| Valid field | 1.077–1.122 |
| Valid field plus CPU transfer | 1.303–1.362 |
| Bounds-error field | 1.137–1.217 |
| Bounds-error field plus CPU transfer | 1.974–2.258 |

Keep this integration experimental. These results do not establish a benefit for
removing the per-kernel status read. No algorithm changes were made to add the
following diagnostic mode.

After updating the candidate checkout, run this on a GPU node in the same env:

```bash
python studies/indexed_deferred/compare.py \
  --baseline ../awkward-indexed-baseline \
  --candidate . \
  --diagnostics --pairs 1 --samples 100 --warmup 5 \
  --output results/indexed-deferred-diagnostics

cat results/indexed-deferred-diagnostics/diagnostics.md
```

This selects `indexed_indexed:4096:8` valid field, valid CPU transfer, and bounds
CPU transfer. Each worker warms the case, runs a wrapper/C-call trace, then runs
cProfile separately. It captures drain/completion calls, guard construction,
virtual materialization, context capture/formatting, pooled allocation calls,
and observed CuPy array `get` counts/bytes and synchronization calls. Nested trace
host durations overlap; never sum them. The report uses exclusive cProfile self
time for increases. Built-in hooks that are not observed are marked unavailable,
not proof that no transfer/synchronization occurred. Nsight is needed for a
complete CUDA API/device trace.

CPU parity checks and their queued GPU work are completed/drained outside each
profiled boundary. Raw worker JSON and per-case `.pstats` files retain full call
statistics and caller relationships. For example, inspect callers with:

```bash
python - <<'PY'
import pstats
pstats.Stats(
    "results/indexed-deferred-diagnostics/1-candidate/"
    "indexed_indexed-4096-8-valid-to_cpu.pstats"
).sort_stats("cumulative").print_callers("synchronize_cuda")
PY
```

These passes perturb execution. Use the earlier unprofiled comparison for latency,
not the trace/cProfile numbers. No production optimization is inferred from the
diagnostic instrumentation itself.

## Changes after the first diagnostic pass

The user-reported diagnostic run (`4815b7640`) retained parity. Valid `to_cpu`
showed two drains instead of one, five observed array gets instead of four, and
an extra 8-byte transfer. Bounds `to_cpu` showed four gets / 96 bytes instead of
two / 24 bytes, plus consumer pretty-printing and a second exception decoration.
These instrumented host durations are not benchmark latencies.

The next candidate adds three targeted changes:

- An empty invocation queue still fences raw CuPy work but does not read or
  allocate another error word. No queued work is skipped.
- Completion-record errors capture a metadata-only consumer context before
  unwinding. Producer and consumer notes are both preserved. Exception copies
  own their notes lists so repeated consumption cannot mutate the cached error.
- Guarded indices validate before content transfer to a known-data backend.
  Failure rejects before copying otherwise-unused content; typetracer conversion
  stays metadata-only and ordinary indices keep the original transfer order.

These changes leave simplify's validation/reduction/gather operations unchanged.
Guard construction/materialization still has overhead. Re-run separate diagnostics
and the alternating unprofiled comparison before claiming any improvement.
