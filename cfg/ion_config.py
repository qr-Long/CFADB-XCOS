#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ion_config.py

Centralize the ion-related configuration used by the cfg processing scripts.

Both cfg_train_data.py and cfg_ts.py read the same ion table from this module.
When adding new iso-electronic sequences, this is the primary place to update.
"""

from __future__ import annotations

from typing import Dict, Mapping, Set, Tuple


ION_CONFIG: Dict[str, Dict[str, object]] = {
    "h-like": {
        "prefix": "01",
        "pair_file": "h_like_pairs.tsv",
        "model_dir_name": "h-like",
        "electron_count": 1,
        "seed_electron_counts": (1,),
        "short_electron_counts": (),
        "shorten_prefixes": (),
        "aliases": ("h", "h-like", "h_like", "01"),
    },
    "he-like": {
        "prefix": "02",
        "pair_file": "he_like_pairs.tsv",
        "model_dir_name": "he-like",
        "electron_count": 2,
        "seed_electron_counts": (2,),
        "short_electron_counts": (),
        "shorten_prefixes": (),
        "aliases": ("he", "he-like", "he_like", "02"),
    },
    "li-like": {
        "prefix": "03",
        "pair_file": "li_like_pairs.tsv",
        "model_dir_name": "li-like",
        "electron_count": 3,
        # Keep compatibility with the previous logic: allow the full 3-electron form as well as the shortened 1-electron form after 1s2.
        # Preserve the 2-electron shorthand to avoid filtering older 2s.2p-style notation from legacy data.
        "seed_electron_counts": (1, 2, 3),
        "short_electron_counts": (1, 2),
        "shorten_prefixes": ("1s2",),
        "aliases": ("li", "li-like", "li_like", "03"),
    },
    "be-like": {
        "prefix": "04",
        "pair_file": "be_like_pairs.tsv",
        "model_dir_name": "be-like",
        "electron_count": 4,
        # Be-like: allow the full 4-electron form or the shortened 2-valence-electron form after 1s2.
        "seed_electron_counts": (2, 4),
        "short_electron_counts": (2,),
        "shorten_prefixes": ("1s2",),
        "aliases": ("be", "be-like", "be_like", "04"),
    },
    "ne-like": {
        "prefix": "10",
        "pair_file": "ne_like_pairs.tsv",
        "model_dir_name": "ne-like",
        "electron_count": 10,
        # Ne-like: allow the full 10-electron form, the 8-electron form after dropping 1s2, and the 6-electron form after dropping 2s2 as well.
        # This covers forms such as 1s2.2s2.2p5.3s / 2s2.2p5.3s / 2p5.3s.
        "seed_electron_counts": (6, 8, 10),
        "short_electron_counts": (6, 8),
        "shorten_prefixes": ("1s2.2s2", "1s2", "2s2"),
        "aliases": ("ne", "ne-like", "ne_like", "10"),
    },
}


def _build_alias_map(config: Mapping[str, Mapping[str, object]]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for canonical, info in config.items():
        aliases = set(info.get("aliases", ()))
        aliases.add(canonical)
        aliases.add(str(info.get("prefix", "")))
        for alias in aliases:
            key = str(alias).strip().lower().replace("_", "-")
            if key:
                alias_map[key] = canonical
    return alias_map


ION_ALIASES: Dict[str, str] = _build_alias_map(ION_CONFIG)


def normalize_ion_type(ion_type: str) -> str:
    """
    Normalize ion notation.

    Examples:
        h / h_like / h-like / 01       -> h-like
        be / be_like / be-like / 04    -> be-like
        ne / ne_like / ne-like / 10    -> ne-like
    """
    key = str(ion_type).strip().lower().replace("_", "-")
    if key not in ION_ALIASES:
        valid = ", ".join(sorted(ION_CONFIG.keys()))
        raise ValueError(f"Unknown ion_type={ion_type!r}. Please use one of: {valid}.")
    return ION_ALIASES[key]


def get_ion_info(ion_type: str) -> Dict[str, object]:
    """Return a copy of the configuration for an ion so callers cannot mutate the global table accidentally."""
    canonical = normalize_ion_type(ion_type)
    return dict(ION_CONFIG[canonical])


def get_ion_prefix(ion_type: str) -> str:
    return str(get_ion_info(ion_type)["prefix"])


def get_pair_file(ion_type: str) -> str:
    return str(get_ion_info(ion_type)["pair_file"])


def get_model_dir_name(ion_type: str) -> str:
    return str(get_ion_info(ion_type)["model_dir_name"])


def get_electron_count(ion_type: str) -> int:
    return int(get_ion_info(ion_type)["electron_count"])


def get_seed_electron_counts(ion_type: str) -> Set[int]:
    info = get_ion_info(ion_type)
    return {int(x) for x in info.get("seed_electron_counts", (info["electron_count"],))}


def get_short_electron_counts(ion_type: str) -> Set[int]:
    info = get_ion_info(ion_type)
    return {int(x) for x in info.get("short_electron_counts", ())}


def get_shorten_prefixes(ion_type: str) -> Tuple[str, ...]:
    """
    Return the complete closed-core prefixes that may be omitted.

    Prefixes are sorted by token count from longest to shortest so that 1s2.2s2 is matched before 1s2.
    """
    info = get_ion_info(ion_type)
    prefixes = [
        str(x).strip().lower().strip(".")
        for x in info.get("shorten_prefixes", ())
        if str(x).strip()
    ]
    prefixes = sorted(set(prefixes), key=lambda x: (-len(x.split(".")), x))
    return tuple(prefixes)
