# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Deterministic, resettable workloads for the eleven #4463 candidates."""

import numpy as np

SUITES = {
    "indexed": ["awkward_IndexedArray_simplify"],
    "listarray": [
        "awkward_ListArray_getitem_jagged_expand",
        "awkward_ListArray_getitem_next_array_advanced",
        "awkward_ListArray_getitem_next_range_spreadadvanced",
    ],
    "listoffsetarray": [
        "awkward_ListOffsetArray_drop_none_indexes",
        "awkward_ListOffsetArray_flatten_offsets",
        "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64",
        "awkward_ListOffsetArray_toRegularArray",
    ],
    "regulararray": [
        "awkward_RegularArray_getitem_next_array_regularize",
        "awkward_RegularArray_reduce_local_nextparents_64",
    ],
    "unionarray": ["awkward_UnionArray_nestedfill_tags_index"],
}
SHAPES = [(32, 8), (4096, 8), (32, 4096)]


def workloads(suite, rows, width):
    offsets = np.arange(rows + 1, dtype=np.int64) * width
    starts = offsets[:-1].copy()
    indexes = np.arange(rows * width, dtype=np.int64)

    def empty(n):
        return np.full(n + 4, 99, dtype=np.int64)

    counts = np.arange(rows) % (width + 1)
    distincts = (
        np.where(np.arange(width)[None, :] < counts[:, None], 0, -1)
        .reshape(-1)
        .astype(np.int64)
    )
    inventory = {
        "awkward_IndexedArray_simplify": [
            empty(rows),
            np.arange(rows, dtype=np.int64),
            rows,
            np.arange(rows, dtype=np.int64)[::-1].copy(),
            rows,
        ],
        "awkward_ListArray_getitem_jagged_expand": [
            empty(rows * width),
            empty(rows * width),
            np.arange(width + 1, dtype=np.int64),
            empty(rows * width),
            starts,
            offsets[1:].copy(),
            width,
            rows,
        ],
        "awkward_ListArray_getitem_next_array_advanced": [
            empty(rows),
            empty(rows),
            starts,
            offsets[1:].copy(),
            np.array([0, -1], dtype=np.int64),
            np.arange(rows, dtype=np.int64) % 2,
            rows,
            rows * width,
        ],
        "awkward_ListArray_getitem_next_range_spreadadvanced": [
            empty(rows * width),
            np.arange(rows, dtype=np.int64),
            offsets,
            rows,
        ],
        "awkward_ListOffsetArray_flatten_offsets": [
            empty(rows + 1),
            np.arange(rows + 1, dtype=np.int64),
            rows + 1,
            offsets,
        ],
        "awkward_ListOffsetArray_toRegularArray": [empty(1), offsets, rows + 1],
        "awkward_ListOffsetArray_drop_none_indexes": [
            empty(rows + 1),
            np.where(indexes % 3 == 0, -1, indexes),
            offsets,
            rows + 1,
            indexes.size,
        ],
        "awkward_ListOffsetArray_reduce_nonlocal_outstartsstops_64": [
            empty(rows),
            empty(rows),
            distincts,
            indexes.size,
            rows,
        ],
        "awkward_RegularArray_getitem_next_array_regularize": [
            empty(rows),
            -(np.arange(rows, dtype=np.int64) % width) - 1,
            rows,
            width,
        ],
        "awkward_RegularArray_reduce_local_nextparents_64": [
            empty(rows * width),
            width,
            rows,
        ],
        "awkward_UnionArray_nestedfill_tags_index": [
            empty(rows * width),
            empty(rows * width),
            starts.copy(),
            3,
            np.full(rows, width, dtype=np.int64),
            rows,
        ],
    }
    return [(name, inventory[name]) for name in SUITES[suite]]
