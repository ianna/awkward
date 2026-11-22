# Compact stream error indicator prototype

Branch: `ianna/cuda-indexed-compact-status`, based on guarded candidate `ccb9e69ef`.
The guarded and synchronous baseline branches remain unchanged. AI-assisted code,
tests and benchmark setup: OpenAI Codex.

## What changes

The existing uint64 stream error word is shared by legacy and compact publishers.
Simplify publishes an invocation ordinal and reserved compact code in its existing
small operation or row-zero write operation. Only one work item writes per launch;
launches using this word are ordered on the same stream. A later code cannot
replace an earlier error. Different streams retain independent words. No extra
publication launch, scalar parameter upload or canonical kernel signature change
is introduced. Direct synchronous callers keep their KernelError contract.

The invocation ordinal is runtime CountingIterator state, not a Python scalar
captured by the callback. Prepared-operation caches distinguish the publishing
variant but do not key on ordinal. Host tests exercise changing ordinals without
additional factory preparation; real GPU compilation/reuse tests remain pending.

A successful compact batch copies only the shared word. It resolves completion
records without reading their detailed statuses. Failure fetches the earliest
reported invocation's payload to preserve its row/attempt/message, then cancels
later records conservatively. Existing noncompact status registrations remain
supported; mixed compact/noncompact statuses within one invocation are rejected.
A malformed indicator or contradictory payload fails closed.

Status, workspace and inputs stay private and alive until completion. The guard,
metadata-only exception handling, partial-write behavior and transfer-before-read
barrier from the earlier prototype remain. Native CPU code, kernel specifications,
package versions and dependencies are unchanged. Guard overhead remains: this is
not a demonstrated performance improvement yet.

## Benchmark boundary correction

Parity conversion can enqueue its own kernels and leave invocation entries after
a sample. The runner now explicitly completes/drains that parity work outside
both timing boundaries before starting the next sample, for BOTH variants. This
prevents parity-side work and ordinals from leaking into the measured operation.
Repeat baseline and candidate together with this runner. Do not compare absolute
medians directly with the earlier runner's results as if methodology were identical.
The diagnostic pass uses the same clean per-operation queue.

Expected diagnostic signature, to be verified on Della: valid field selection
uses one observed 8-byte get, rather than baseline's 16-byte payload plus 8-byte
word. Valid `to_cpu` adds its normal content/index transfers. Failure still needs
the detailed payload; it is not expected to save that read.

## Della setup

In the activated `awkward-cuda-bench` environment, fetch on a node with GitHub access:

```bash
cd /home/iosborne/Projects/CUDA_BENCH

git -C awkward-matrix fetch https://github.com/ianna/awkward.git \
  ianna/cuda-indexed-compact-status

git -C awkward-matrix worktree add --detach \
  ../awkward-indexed-compact FETCH_HEAD

cd awkward-indexed-compact
python dev/generate-kernel-signatures.py
python -m pip install --no-deps --no-build-isolation -e .
```

Keep the existing `../awkward-indexed-baseline` at #4484 `1c44307ec`. No native
rebuild is required. On an allocated GPU node, validate before measuring:

```bash
python -m pytest -x -q --tb=short \
  tests-cuda/test_4484_indexedarray_simplify_compute.py \
  tests-cuda/test_4573_indexed_simplify_deferred.py \
  tests-cuda/test_4573_cuda_index_guard.py \
  tests-cuda/test_4484_cuda_deferred_status.py &&
python studies/indexed_deferred/compare.py \
  --baseline ../awkward-indexed-baseline --candidate . \
  --pairs 3 --samples 100 --warmup 3 \
  --output results/indexed-compact
```

Run diagnostics separately:

```bash
python studies/indexed_deferred/compare.py \
  --baseline ../awkward-indexed-baseline --candidate . \
  --diagnostics --pairs 1 --samples 100 --warmup 5 \
  --output results/indexed-compact-diagnostics
```

The new GPU cases queue several operations without draining, vary ordinals after
warmup while forbidding new factories, check earliest-error propagation, and
check stream isolation. Existing GPU cases verify zero successful payload reads,
nonblocking streams, direct-kernel parity, partial writes and failure persistence.
GPU execution and end-to-end latency results for this compact variant are pending.
