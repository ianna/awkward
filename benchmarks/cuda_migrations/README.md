# CUDA migration validation (#4463)

This harness compares the twelve active migration candidates on the same GPU,
Python environment, dependency versions, CUDA runtime and driver. It does not
supply benchmark results: real GPU execution is required. The two obsolete
sorting-range kernels are excluded (removal PR #4488).

| Suite | Candidate branch in ianna/awkward | PR |
| --- | --- | --- |
| indexed | ianna/cuda-compute-indexed-simplify | #4484 |
| listarray | ianna/cuda-compute-listarray | #4481 |
| listoffsetarray | ianna/cuda-compute-listoffsetarray | #4483 |
| regulararray | ianna/cuda-compute-regulararray | #4485 |
| unionarray | ianna/cuda-compute-unionarray | #4486 |

## Run on the GPU runner

The manually dispatched **CUDA migration comparison** workflow takes a suite,
candidate repository/ref, and baseline ref. Run each suite against its branch
above. The default baseline is `7c403344b574980bbb020f98dc2a7b1bd5f80324`.
The workflow installs each checkout's native library before running it, so CPU
parity uses matching source and native kernels. Dependencies stay unchanged
between runs; comparisons reject environment mismatches. It uploads JSON,
Nsight `.nsys-rep` and SQLite exports even if a comparison fails.

The self-hosted runner needs Nsight Systems (`nsys`) installed and permission to
trace CUDA. Use an otherwise idle GPU. Workflow concurrency serializes its own
runs, but does not exclude unrelated GPU jobs. The workflow must be available
on the upstream default branch before normal manual dispatch is possible.

For local execution, keep this harness at a fixed absolute location outside
both checkouts. Install the existing GPU development dependencies and PyYAML.
Activate the **same environment** for both runs:

```bash
# Set these to absolute checkout paths.
HARNESS=/path/to/validation/benchmarks/cuda_migrations/measure.py
BASELINE=/path/to/baseline
CANDIDATE=/path/to/candidate

python -m pip install --no-deps --force-reinstall "$BASELINE/awkward-cpp"
python -m pip install --no-deps -e "$BASELINE"
python "$HARNESS" run --checkout "$BASELINE" --contracts "$CANDIDATE" \
  --suite listoffsetarray --nsys --output results/baseline/report.json

python -m pip install --no-deps --force-reinstall "$CANDIDATE/awkward-cpp"
python -m pip install --no-deps -e "$CANDIDATE"
python "$HARNESS" run --checkout "$CANDIDATE" --contracts "$CANDIDATE" \
  --suite listoffsetarray --nsys --output results/candidate/report.json

python "$HARNESS" compare --baseline results/baseline/report.json \
  --candidate results/candidate/report.json --output results/comparison.json \
  --require-pass
```

Use the candidate's contracts for **both** runs so they exercise the same edge
cases and dtype inventory. SHA-256 fingerprints reject differing spec/case data.
The workflow also runs the candidate PR's focused GPU tests, which cover public
operations and additional malformed/sliced inputs beyond the canonical cases.

## Measurements and interpretation

* **Parity:** canonical explicit cases across every supported specialization,
  comparing every buffer against the compiled CPU implementation, including
  unchanged inputs, output tails, in/out scratch, and partial writes on error.
  Returned errors compare message, id and attempt; legacy asynchronous exceptions
  expose only the message, and the report labels that limitation. Failed parity
  prevents performance comparisons for that workload. Canonical tests may expose
  known baseline defects; their failures remain visible in the baseline JSON.
* **Host latency:** wall time inside one warmed kernel call, including any
  synchronization that the implementation performs. Resetting inputs, external
  fences, compilation, validation, tracing, and allocation hooks are excluded.
* **Stream elapsed time:** CUDA events surrounding the call on a nondefault
  stream. This interval includes GPU idle gaps while Python submits work; it is
  not the sum of kernel execution times. Nsight separately reports the sum of
  GPU kernel durations (`kernel_busy_ns`) in an instrumented pass.
* **Total latency:** wall time through completion and the explicit asynchronous
  error check. The driver retains every repeat and reports medians, rather than
  hiding variability behind a single measurement.
* **Synchronization and transfers:** a fresh warmed process captures a single
  call inside the `measure` NVTX range. An outer `capture` range stays open
  until the stream finishes so asynchronous GPU activity is not truncated.
  API and allocation timestamps are restricted to `measure`, excluding that
  completion fence. Nsight reports runtime/driver API counts
  and durations, explicit synchronize calls, blocking memcpy API calls, and
  actual DToH transfer bytes (including array/UVM transfers). Harness reset,
  validation copies and final fences are outside capture. Runtime and driver
  layers may describe the same operation; counts are not unique barriers.
  Query polling and implicit synchronization in allocation/free APIs require
  inspecting the retained full API list/trace. A missing trace is unmeasured,
  never zero. CUDA event timings and API durations are recorded in separate
  passes and must not be added together.
* **Temporary allocations:** a separate warmed CuPy MemoryHook pass records
  requested counts/bytes (including pool hits), driver-backed pool growth,
  peak live newly requested pool blocks, and retained blocks. Inputs/outputs
  were allocated before this pass. Nsight memory tracing separately records
  driver allocation events, allocated bytes, peak live allocations originating
  inside capture, and their retained bytes, including allocations outside CuPy.
  Pool reuse can give zero driver allocations while still requesting temporary
  buffers. Do not add the pool and driver numbers: their storage can overlap.
  Neither peak is the process's total memory high-water mark. Unknown memory
  schemas/operations make allocation accounting incomplete rather than zero.

Workloads include 32 short rows, 4096 short rows, and 32 long rows. Warmup defaults
to 3 calls; timing defaults to 5 repeats of 20 independently reset calls.
Mutating scratch is reset before **every** measurement (especially UnionArray
`tmpstarts`), avoiding progressive offsets and out-of-bounds benchmark writes.
Nonlocal reduction timing uses prefix occupancy so the baseline's known
count-versus-last-position defect does not yield a misleading fast comparison;
canonical/focused parity tests independently exercise holes.

The comparison gate requires candidate parity, all workload timings/traces,
complete allocation accounting, and no timing median over the configurable
`--max-regression` ratio (default 1.10). This threshold flags investigation; it
is not a statistical significance test or automatic merge approval. Review
synchronization/space changes and repeat with more samples on an idle GPU. Keep
the raw implementation if representative workloads regress. Do not close #4463
merely because the tooling or implementation PRs exist.

## References

The implementation follows the documented [CuPy memory-hook callbacks](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryHook.html),
[Nsight capture and memory-tracing options](https://docs.nvidia.com/nsight-systems/UserGuide/),
and [Nsight SQLite schema](https://docs.nvidia.com/nsight-systems/AnalysisGuide/).
It avoids deprecated `cupyx.allow_synchronize`, which can miss synchronization
outside CuPy.
