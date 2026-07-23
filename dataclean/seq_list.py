"""
Select ion data files by iso-electronic sequence and optional filename range.
"""

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union


ION_TYPE_PREFIX = {
    "h-like": "01",
    "he-like": "02",
    "li-like": "03",
    "be-like": "04",
    "ne-like": "10",
}


def get_prefix_for_ion_type(ion_type: str) -> str:
    """Return the filename prefix for a supported iso-electronic sequence."""
    key = str(ion_type).strip().lower()
    if key not in ION_TYPE_PREFIX:
        raise ValueError(
            f"Unknown ion_type={ion_type!r}. "
            f"Valid values are: {', '.join(ION_TYPE_PREFIX)}"
        )
    return ION_TYPE_PREFIX[key]


def _normalize_suffixes(
    suffixes: Optional[Union[str, Sequence[str]]],
) -> Optional[tuple[str, ...]]:
    """Normalize one or more filename suffixes for case-insensitive matching."""
    if suffixes is None:
        return None
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    out = []
    for s in suffixes:
        s = str(s).strip().lower()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        out.append(s)
    return tuple(out) if out else None


def get_seq_list(
    folder_path: Union[str, Path],
    prefix: str,
    suffixes: Optional[Union[str, Sequence[str]]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[str]:
    """
    Return sorted filenames without extensions from a selected directory.

    Parameters
    ----------
    folder_path:
        Directory to scan.
    prefix:
        Filename prefix, such as ``01``, ``02``, or ``03``.
    suffixes:
        Optional suffix filter, such as ``.h5`` or ``[.h5, .hdf5]``.
        All suffixes are accepted when this argument is ``None``.
    start, end:
        Optional inclusive filename range, such as ``030604`` to ``033028``.
    """
    folder = Path(folder_path)
    suffix_filter = _normalize_suffixes(suffixes)
    start_s = None if start is None else str(start)
    end_s = None if end is None else str(end)

    if not folder.exists():
        raise FileNotFoundError(f"Cannot find folder: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    seq_list: List[str] = []
    for item in folder.iterdir():
        if not item.is_file():
            continue

        name = item.stem
        ext = item.suffix.lower()

        if not name.startswith(str(prefix)):
            continue
        if suffix_filter is not None and ext not in suffix_filter:
            continue
        if start_s is not None and name < start_s:
            continue
        if end_s is not None and name > end_s:
            continue

        seq_list.append(name)

    seq_list.sort()
    return seq_list


def get_seq_list_by_ion_type(
    folder_path: Union[str, Path],
    ion_type: str,
    suffixes: Optional[Union[str, Sequence[str]]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[str]:
    """Return ion filenames using an iso-electronic sequence label."""
    return get_seq_list(
        folder_path=folder_path,
        prefix=get_prefix_for_ion_type(ion_type),
        suffixes=suffixes,
        start=start,
        end=end,
    )


def main():
    """Print an example ion list for a selected directory and sequence."""
    folder_path = "./atomic_data"
    ion_type = "he-like"  # h-like / he-like / li-like / be-like / ne-like

    seq_list = get_seq_list_by_ion_type(folder_path, ion_type)
    print(seq_list)
    print("Total:", len(seq_list))


if __name__ == "__main__":
    main()
