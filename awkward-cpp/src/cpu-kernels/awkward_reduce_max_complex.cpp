// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cpu-kernels/awkward_reduce_max_complex.cpp", line)

#include "awkward/kernels.h"
#include <cmath>

template <typename OUT, typename IN>
ERROR awkward_reduce_max_complex(
  OUT* __restrict__ toptr,
  const IN* __restrict__ fromptr,
  const int64_t* __restrict__ offsets,
  int64_t outlength,
  OUT identity) {
  for (int64_t bin = 0; bin < outlength; bin++) {
    const int64_t start = offsets[bin];
    const int64_t stop = offsets[bin + 1];
    OUT best_re = identity;
    OUT best_im = 0;
    if (start < stop) {
      best_re = fromptr[start * 2];
      best_im = fromptr[start * 2 + 1];
      if (!std::isunordered(best_re, best_im)) {
        auto update = [&](IN x, IN y) {
          if (x > best_re || (x == best_re && y > best_im)) {
            best_re = x;
            best_im = y;
          }
        };
        int64_t i = start + 1;
        // Share one NaN branch across four values, keeping comparison order.
        for (; stop - i >= 4; i += 4) {
          const IN re0 = fromptr[2 * i], im0 = fromptr[2 * i + 1];
          const IN re1 = fromptr[2 * i + 2], im1 = fromptr[2 * i + 3];
          const IN re2 = fromptr[2 * i + 4], im2 = fromptr[2 * i + 5];
          const IN re3 = fromptr[2 * i + 6], im3 = fromptr[2 * i + 7];
          if (std::isunordered(re0, im0) | std::isunordered(re1, im1) |
              std::isunordered(re2, im2) | std::isunordered(re3, im3)) {
            break;
          }
          update(re0, im0);
          update(re1, im1);
          update(re2, im2);
          update(re3, im3);
        }
        // Replay a batch containing NaNs in order to retain the first one.
        for (; i < stop; i++) {
          const IN x = fromptr[i * 2];
          const IN y = fromptr[i * 2 + 1];
          if (std::isunordered(x, y)) {
            best_re = x;
            best_im = y;
            break;
          }
          update(x, y);
        }
      }
    }
    toptr[bin * 2] = best_re;
    toptr[bin * 2 + 1] = best_im;
  }
  return success();
}

#define WRAPPER(FUNC, OUT, IN) \
  ERROR FUNC(OUT* toptr, const IN* fromptr, const int64_t* offsets, int64_t outlength, OUT identity) { \
    return awkward_reduce_max_complex<OUT, IN>(toptr, fromptr, offsets, outlength, identity); \
  }

WRAPPER(awkward_reduce_max_complex64_complex64_64, float, float)
WRAPPER(awkward_reduce_max_complex128_complex128_64, double, double)
