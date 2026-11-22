# Guarded public IndexedArray simplification

Experimental integration based on PR #4484 at
`efbe2a47b3642ee5850f972ff4932df2f07b7f0a`, plus the deferred-status and guard
prototype. AI-assisted implementation/tests: OpenAI Codex.

Only the public IndexedArray/IndexedOptionArray `simplified` constructors opt in.
They register status, retain input/output/workspace, and wrap the output index
before publication. Direct calls to the kernel keep their synchronous KernelError
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
  ../awkward-indexed-baseline efbe2a47b3642ee5850f972ff4932df2f07b7f0a

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
