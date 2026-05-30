"""
pipeline/utils.py

Low-level helpers shared across the pipeline.

Functions
---------
copy_helper         Safe copy for tensors, arrays, and Python containers.
serialize           Convert a tensor / ndarray to a hashable, JSON-safe value.
deserialize         Reconstruct a tensor from the output of ``serialize``.
stringify_mapping   Convert a high-level → low-level node mapping to a
                    JSON-serialisable dict of strings.
"""

import copy
from typing import Any, Dict

import numpy as np
import torch

from intervention.location import Location


# ---------------------------------------------------------------------------
# copy_helper
# ---------------------------------------------------------------------------

def copy_helper(x: Any) -> Any:
    """
    Return a safe independent copy of ``x``.

    * ``torch.Tensor``  → detached clone (no autograd history)
    * ``list``, ``tuple``, ``str``, ``dict``, ``np.ndarray`` → deep copy
    * Everything else (``int``, ``float``, ``bool``, …)  → returned as-is
      because Python primitives are already immutable.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (list, tuple, str, dict, np.ndarray)):
        return copy.deepcopy(x)
    return x  # primitives are immutable — safe to return directly


# ---------------------------------------------------------------------------
# serialize / deserialize
# ---------------------------------------------------------------------------

def _lists_to_nested_tuple(x: Any) -> Any:
    """Recursively convert every list inside ``x`` to a tuple."""
    if isinstance(x, list):
        return tuple(_lists_to_nested_tuple(item) for item in x)
    return x


def serialize(x: Any) -> Any:
    """
    Convert ``x`` to a hashable, JSON-safe Python value.

    Used to turn intermediate tensor activations into cache keys.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        * Tensors of any shape are converted to nested tuples of Python
          scalars via ``tolist()``, so the result is hashable.
        * NumPy arrays are converted to raw bytes via ``tobytes()``.
          **Note**: the bytes form is only suitable for use as a hash key;
          use the Tensor path if you need to reconstruct the value later.

    Returns
    -------
    tuple | int | float | bytes
        A hashable representation of ``x``.

    Raises
    ------
    TypeError
        If ``x`` is not a supported type.
    """
    if isinstance(x, torch.Tensor):
        # tolist() handles 0-D (scalar) through N-D tensors uniformly.
        # The result may be a plain Python scalar (0-D) or nested lists (N-D).
        return _lists_to_nested_tuple(x.tolist())

    if isinstance(x, np.ndarray):
        # tobytes() replaces the deprecated tostring().
        return x.tobytes()

    raise TypeError(
        f"serialize() does not support type {type(x).__name__!r}. "
        "Pass a torch.Tensor or np.ndarray."
    )


def deserialize(x: Any) -> torch.Tensor:
    """
    Reconstruct a ``torch.Tensor`` from the output of :func:`serialize`.

    Only works for the Tensor path of ``serialize`` (nested tuples / scalars).
    NumPy byte strings cannot be round-tripped without knowing the original
    dtype and shape.
    """
    return torch.tensor(x)


# ---------------------------------------------------------------------------
# stringify_mapping
# ---------------------------------------------------------------------------

def stringify_mapping(mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    Convert a high-level → low-level node mapping to a JSON-serialisable dict.

    Parameters
    ----------
    mapping : dict
        ``{high_node_name: {low_node_name: location, ...}, ...}``
        where ``location`` is any value accepted by
        :meth:`~pipeline.location.Location.loc_to_str`.

    Returns
    -------
    dict
        ``{high_node_name: {low_node_name: location_string, ...}, ...}``

    Example
    -------
    >>> stringify_mapping({"h1": {"l1": slice(0, 64)}})
    {'h1': {'l1': '0:64'}}
    """
    return {
        high_node: {
            low_node: Location.loc_to_str(loc)
            for low_node, loc in low_dict.items()
        }
        for high_node, low_dict in mapping.items()
    }


# ---------------------------------------------------------------------------
# Tests — run with:  python pipeline/utils.py
# ---------------------------------------------------------------------------
def _run_tests():

    # -----------------------------------------------------------------------
    # copy_helper
    # -----------------------------------------------------------------------

    # Tensor: returns a detached clone, not the same object
    t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    t_copy = copy_helper(t)
    assert t_copy is not t
    assert torch.equal(t_copy, t)
    assert not t_copy.requires_grad, "clone should not carry grad history"

    # Mutating the copy must not affect the original
    t_copy[0] = 99.0
    assert t[0].item() == 1.0

    # List: deep copy
    lst = [[1, 2], [3, 4]]
    lst_copy = copy_helper(lst)
    assert lst_copy == lst and lst_copy is not lst
    lst_copy[0][0] = 99
    assert lst[0][0] == 1

    # Primitives: returned as-is (immutable, no need to copy)
    assert copy_helper(42) == 42
    assert copy_helper(3.14) == 3.14
    assert copy_helper(True) is True

    print("  copy_helper ✓")

    # -----------------------------------------------------------------------
    # serialize
    # -----------------------------------------------------------------------

    # 0-D scalar tensor
    s = serialize(torch.tensor(5.0))
    assert s == 5.0

    # 1-D tensor → tuple of floats
    s = serialize(torch.tensor([1.0, 2.0, 3.0]))
    assert s == (1.0, 2.0, 3.0)

    # 2-D tensor → tuple of tuples
    s = serialize(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert s == ((1.0, 2.0), (3.0, 4.0))

    # 3-D tensor
    t3 = torch.zeros(2, 2, 2)
    s = serialize(t3)
    assert isinstance(s, tuple) and isinstance(s[0], tuple) and isinstance(s[0][0], tuple)

    # 4-D tensor (previously required a separate elif in the original)
    t4 = torch.zeros(2, 2, 2, 2)
    s = serialize(t4)
    assert isinstance(s[0][0][0], tuple)

    # 5-D tensor (would have raised NotImplementedError in the original!)
    t5 = torch.zeros(2, 2, 2, 2, 2)
    s = serialize(t5)
    assert isinstance(s, tuple)

    # numpy array → bytes
    arr = np.array([1.0, 2.0, 3.0])
    s = serialize(arr)
    assert isinstance(s, bytes)

    # Unsupported type raises TypeError
    try:
        serialize("hello")
        raise AssertionError("Should have raised TypeError")
    except TypeError:
        pass

    print("  serialize ✓")

    # -----------------------------------------------------------------------
    # deserialize (round-trip with serialize for tensors)
    # -----------------------------------------------------------------------

    for original in [
        torch.tensor(5.0),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    ]:
        reconstructed = deserialize(serialize(original))
        assert torch.equal(original, reconstructed), (
            f"Round-trip failed for shape {original.shape}"
        )

    print("  deserialize ✓")

    # -----------------------------------------------------------------------
    # stringify_mapping
    # -----------------------------------------------------------------------

    mapping = {
        "high_node_1": {"low_node_a": slice(0, 64)},
        "high_node_2": {"low_node_b": 5, "low_node_c": (0, slice(1, 3))},
    }
    result = stringify_mapping(mapping)

    assert result["high_node_1"]["low_node_a"] == "0:64"
    assert result["high_node_2"]["low_node_b"] == "5"
    assert result["high_node_2"]["low_node_c"] == "[0, 1:3]"

    print("  stringify_mapping ✓")

    print("\n✓  All utils tests passed")


if __name__ == "__main__":
    _run_tests()
