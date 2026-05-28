"""
pipeline/location.py

Utilities for specifying index/slice locations within tensors or arrays.

A Location describes *where* to apply an intervention on a node's output.
Use the module-level singleton ``LOC`` to capture Python index expressions:

    from pipeline.location import LOC

    LOC[:64]       →  slice(None, 64)
    LOC[0]         →  0
    LOC[1:3, 0]    →  (slice(1, 3), 0)

Static helpers on :class:`Location` convert between index objects and strings,
which is useful when persisting locations to a database or CSV file.
"""
from __future__ import annotations

from typing import Optional, Union

# Any valid Python index expression.
IndexLike = Union[int, bool, slice, tuple, list, type(...)]


class Location:
    """
    Captures Python index expressions and provides string serialisation helpers.

    Prefer the module-level singleton :data:`LOC` over constructing new
    instances::

        from pipeline.location import LOC
        loc = LOC[:5]      # → slice(None, 5)
        loc = LOC[0, 2:4]  # → (0, slice(2, 4))

    All other functionality is accessed through static methods.
    """

    # ------------------------------------------------------------------
    # Index capture
    # ------------------------------------------------------------------

    def __getitem__(self, item: IndexLike) -> IndexLike:
        """Return the Python index expression unchanged (captures slice/int syntax)."""
        return item

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def process(x) -> Optional[IndexLike]:
        """
        Normalise an arbitrary value to a canonical Python index object.

        * ``int``, ``bool``, ``list``, ``tuple``, ``slice``, ``Ellipsis``
          are returned unchanged.
        * ``str`` is parsed via :meth:`parse_str`.
        * ``None`` returns ``None``.

        Raises
        ------
        TypeError
            If ``x`` is not one of the supported types.
        """
        if x is None:
            return None
        if isinstance(x, (int, bool, list, tuple, slice)) or x is Ellipsis:
            return x
        if isinstance(x, str):
            return Location.parse_str(x)
        raise TypeError(f"Unsupported location type: {type(x).__name__!r}")

    # ------------------------------------------------------------------
    # String parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_str(s: str) -> IndexLike:
        """
        Parse a location string into a Python index object.

        Supports comma-separated multi-dimensional specs:

        ============  ====================
        Input string  Result
        ============  ====================
        ``"3"``       ``3``
        ``"1:4"``     ``slice(1, 4)``
        ``":5"``      ``slice(None, 5)``
        ``"1:"``      ``slice(1, None)``
        ``"..."``     ``Ellipsis``
        ``"0, 1:3"``  ``(0, slice(1, 3))``
        ============  ====================
        """
        s = s.strip()
        if "," in s:
            return tuple(Location._parse_dim(part) for part in s.split(","))
        return Location._parse_dim(s)

    @staticmethod
    def _parse_dim(s: str) -> Union[int, slice, type(...), bool]:
        """Parse a single-dimension specifier string."""
        s = s.strip("[] ")
        if s == "...":
            return Ellipsis
        if s == "True":
            return True
        if s == "False":
            return False
        if ":" in s:
            return Location._str_to_slice(s)
        return int(s)

    @staticmethod
    def _str_to_slice(s: str) -> slice:
        """Parse ``"start:stop"`` or ``"start:stop:step"`` into a ``slice``."""
        def _int_or_none(p: str) -> Optional[int]:
            p = p.strip()
            return int(p) if p else None

        return slice(*map(_int_or_none, s.split(":")))

    # ------------------------------------------------------------------
    # String serialisation  (inverse of parse_str)
    # ------------------------------------------------------------------

    @staticmethod
    def loc_to_str(loc: IndexLike) -> str:
        """
        Serialise a location object to a human-readable string.

        Inverse of :meth:`parse_str` for all supported index types.

        Examples
        --------
        >>> Location.loc_to_str(3)
        '3'
        >>> Location.loc_to_str(slice(1, 4))
        '1:4'
        >>> Location.loc_to_str((0, slice(1, 3)))
        '[0, 1:3]'
        """
        if isinstance(loc, (tuple, list)):
            return "[" + ", ".join(Location._dim_to_str(d) for d in loc) + "]"
        return Location._dim_to_str(loc)

    @staticmethod
    def _dim_to_str(d) -> str:
        if d is Ellipsis:
            return "..."
        if isinstance(d, slice):
            return Location._slice_to_str(d)
        return str(d)

    @staticmethod
    def _slice_to_str(s: slice) -> str:
        start = "" if s.start is None else str(s.start)
        stop = "" if s.stop is None else str(s.stop)
        if s.step is None:
            return f"{start}:{stop}"
        return f"{start}:{stop}:{s.step}"


# ---------------------------------------------------------------------------
# Module-level singleton — import and use this everywhere.
# ---------------------------------------------------------------------------
LOC = Location()


# ---------------------------------------------------------------------------
# Tests — run with:  python pipeline/location.py
# ---------------------------------------------------------------------------
def _run_tests():
    # --- Index capture via LOC singleton ---
    assert LOC[3] == 3
    assert LOC[:5] == slice(None, 5)
    assert LOC[1:3] == slice(1, 3)
    assert LOC[1:3:2] == slice(1, 3, 2)
    assert LOC[...] is Ellipsis
    assert LOC[0, 1:3] == (0, slice(1, 3))

    # --- process: pass-through types ---
    assert Location.process(3) == 3
    assert Location.process(slice(1, 4)) == slice(1, 4)
    assert Location.process([0, 1]) == [0, 1]
    assert Location.process(None) is None
    assert Location.process(...) is Ellipsis

    # --- process: string delegates to parse_str ---
    assert Location.process("1:4") == slice(1, 4)

    # --- parse_str: integers ---
    assert Location.parse_str("3") == 3
    assert Location.parse_str("-1") == -1

    # --- parse_str: slices ---
    assert Location.parse_str("1:4") == slice(1, 4)
    assert Location.parse_str(":5") == slice(None, 5)
    assert Location.parse_str("1:") == slice(1, None)
    assert Location.parse_str(":") == slice(None, None)
    assert Location.parse_str("1:4:2") == slice(1, 4, 2)

    # --- parse_str: special values ---
    assert Location.parse_str("...") is Ellipsis
    assert Location.parse_str("True") is True
    assert Location.parse_str("False") is False

    # --- parse_str: multi-dim tuple ---
    assert Location.parse_str("0, 1:3") == (0, slice(1, 3))
    assert Location.parse_str("..., :5") == (Ellipsis, slice(None, 5))

    # --- Round-trip: loc → str → loc ---
    cases = [
        3,
        slice(1, 4),
        slice(None, 5),
        slice(1, None),
        slice(None, None),
        slice(1, 4, 2),
        Ellipsis,
        (0, slice(1, 3)),
        (Ellipsis, slice(None, 5)),
    ]
    for loc in cases:
        s = Location.loc_to_str(loc)
        back = Location.parse_str(s)
        assert back == loc, f"Round-trip failed: {loc!r} → {s!r} → {back!r}"

    print("✓  All Location tests passed")


if __name__ == "__main__":
    _run_tests()
