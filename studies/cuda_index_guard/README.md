# Pending index validation: isolated prototype

This branch stacks on deferred-status plumbing (#4573, `de517dbb5`). It does not
change production kernel dispatch, IndexedArray simplify, kernel specifications,
CPU kernels or package versions. AI-assisted implementation and measurements:
OpenAI Codex.

`PendingValidation` runs a caller-supplied validation callback once and caches its
success or exception. `guard_index` wraps an existing index with VirtualNDArray,
using its existing materialization boundary for reads and buffer exports. Shape
and dtype queries require no validation. Derived slices share the validation
state; failed validation never exposes the partially written raw output.

Only publish the guarded index. Previously exported raw arrays and pointers are
outside this prototype's protection. The callback must retain invocation-owned
status/scratch, fence the producer stream, preserve the canonical error/context,
and retain its own result independently of the stream's pending queue. Simply
calling `synchronize_cuda` from the callback is insufficient if another drain
has already consumed the failure. Integration with #4573 needs persistent
per-invocation completion records before enabling this in production.

A successful first consumer still waits for validation. This prototype moves the
wait to consumption; it does not eliminate that wait or guard subsequent GPU
kernels entirely on device. It also makes the returned index internally virtual,
which needs a broader compatibility audit before adoption.

## Local host measurements

Apple Silicon, Python 3.12. Three fresh processes, each with nine alternating
paired batches of 20,000 iterations. Plain and guarded indices use identical
32-element int64 buffers. First-consumption setup is excluded; construction is
measured separately. Raw samples are in `results/host-*.json`.

Median of process medians, microseconds per boundary:

| Boundary | Plain | Guard | Added host time |
| --- | ---: | ---: | ---: |
| Construction | 0.532 | 1.753 | 1.219 |
| Metadata length query | 0.051 | 0.046 | within measurement noise |
| Already validated consumption | 0.253 | 0.269 | 0.016 |
| First consumption, no-op validator | 0.254 | 1.790 | 1.539 |

These are host microbenchmarks, not measurements against a complete main build,
not a simplify implementation comparison and not public-analysis timings. No GPU
or CuPy is available locally. CUDA tests and GPU measurements remain pending.

## Della

Use an activated CUDA environment, an allocated GPU node and this checkout.

```bash
python -m pip install --no-build-isolation -e .
python -m pytest -q --tb=short \
  tests/test_4573_cuda_index_guard.py \
  tests-cuda/test_4573_cuda_index_guard.py

python studies/cuda_index_guard/benchmark.py \
  --backend cuda --iterations 2000 --repeats 9 \
  --output results/index-guard-cuda.json
```

The CUDA benchmark measures host wrapper overhead using real CuPy-backed indices.
Its separate status-transfer row includes the same completed 16-byte transfer on
both sides. That row quantifies the extra wrapper cost around validation, without
including a simplify kernel or claiming an end-to-end benefit. GPU tests exercise
nonblocking producer/consumer streams and reject invalid tail data before CUDA
array-interface export or slice consumption. Run unprofiled; use a separate pass
for Nsight tracing.
