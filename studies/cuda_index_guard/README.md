# Pending index validation: isolated prototype

This branch stacks on deferred-status plumbing (#4573, `de517dbb5`). It does not
enable production guard adoption in IndexedArray simplify or change kernel specifications,
CPU kernels or package versions. AI-assisted implementation and measurements:
OpenAI Codex.

`PendingValidation` runs a caller-supplied validation callback once and caches its
success or exception. `guard_index` wraps an existing index with VirtualNDArray,
using its existing materialization boundary for reads and buffer exports. Shape
and dtype queries require no validation. Derived slices share the validation
state; failed validation never exposes the partially written raw output.

Only publish the guarded index after producer dispatch returns. Previously exported
raw arrays and pointers are outside this prototype's protection.

`defer_status` now returns the invocation's persistent `CompletionRecord`. Pass
that record directly to `guard_index(index, completion)`. On first consumption it
drains the producer stream if necessary. An external drain resolves the same
record, so success/failure remains available after the queue is reset. No callback
that merely re-reads the current global stream queue is needed.

The first failing invocation retains its decorated exception. Later invocations
in that batch are conservatively marked failed with the same error, even if their
own status was not read; earlier successful invocations retain success. Repeated
checks raise fresh exception copies so the stored error does not pin drain
tracebacks or GPU scratch. Resolved records release their stream reference;
completed batches release status/scratch. Pending checks on the wrong CUDA device
fail without resolving the record, allowing retry on its producer device.

This extends the prototype's drain behavior; it has not been folded into #4573 or
adopted by the simplify migration. GPU validation of these new completion-record
paths is still required.

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
or CuPy is available locally. The original prototype subsequently passed 15 tests
on Della; the completion-record additions below still need GPU validation.

## Della

Use an activated CUDA environment, an allocated GPU node and this checkout.

```bash
python -m pip install --no-build-isolation -e .
python -m pytest -q --tb=short \
  tests/test_4484_cuda_deferred_status.py \
  tests/test_4573_cuda_index_guard.py \
  tests-cuda/test_4484_cuda_deferred_status.py \
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

## Original prototype Della results (before completion records)

User-reported GPU run: all 15 tests passed in 6.61 s. Nine alternating batches,
2,000 iterations each; results below are host microseconds per boundary.

| Boundary | Plain | Guard | Added host time |
| --- | ---: | ---: | ---: |
| Construction | 1.271 | 3.799 | 2.529 |
| Metadata | 0.138 | 0.092 | noise |
| Validated consumption | 0.555 | 0.593 | 0.038 |
| First consumption, no-op validation | 0.555 | 4.469 | 3.915 |
| First consumption, 16-byte status transfer | 16.272 | 21.314 | 5.042 |

These numbers do not validate or measure the new persistent completion records.
