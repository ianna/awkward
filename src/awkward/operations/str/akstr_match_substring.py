# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("match_substring",)


import awkward as ak
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout


@high_level_function(module="ak.str")
def match_substring(
    array, pattern, *, ignore_case=False, highlevel=True, behavior=None
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        pattern (str or bytes): Substring pattern to look for inside `array`.
        ignore_case (bool): If True, perform a case-insensitive match;
            otherwise, the match is case-sensitive.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    For each string in the array, determine whether it contains the given
    literal `pattern`.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.match_substring](https://arrow.apache.org/docs/python/generated/pyarrow.compute.match_substring.html).

    See also: #ak.str.match_substring_regex.
    """
    # Dispatch
    yield (array,)

    # Implementation
    return _impl(array, pattern, ignore_case, highlevel, behavior)


def _impl(array, pattern, ignore_case, highlevel, behavior):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("ak.str.match_substring")
    layout = ak.to_layout(array, allow_record=False, allow_other=True)
    behavior = behavior_of(array, behavior=behavior)
    apply = ak.operations.str._get_ufunc_action(
        pc.match_substring,
        pc.match_substring,
        bytestring_to_string=False,
        ignore_case=ignore_case,
        pattern=pattern,
    )
    out = ak._do.recursively_apply(layout, apply, behavior=behavior)
    return wrap_layout(out, highlevel=highlevel, behavior=behavior)