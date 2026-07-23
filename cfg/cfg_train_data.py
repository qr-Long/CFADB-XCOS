#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cfg_train_data.py

Extract cfg strings from real HDF5 level tables (lev_*) as seeds and generate equivalent or near-equivalent training pairs.

Supported sequences: h-like / he-like / li-like / be-like / ne-like.
The ion-specific prefix, pair_file, model_dir_name, electron_count, and shortening rules are centralized in ion_config.py.

This version emphasizes:
  1. Nested parent-term handling for Be-like and Ne-like configurations
  2. Stepwise removal of multiple parent terms for medium pairs
  3. Multi-level closed-core shortening for Ne-like configurations
  4. Conservative treatment of .limit, keeping only identity and safe format-equivalent cases
  5. Restricting soft pairs to the same structural category
"""

from __future__ import annotations

import itertools
import os
import random
import re
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import h5py
import pandas as pd

try:
    from .ion_config import (
        ION_CONFIG,
        get_ion_info,
        get_seed_electron_counts,
        get_short_electron_counts,
        get_shorten_prefixes,
        normalize_ion_type,
    )
except ImportError:  # Allow direct execution with python cfg_train_data.py
    from ion_config import (  # type: ignore
        ION_CONFIG,
        get_ion_info,
        get_seed_electron_counts,
        get_short_electron_counts,
        get_shorten_prefixes,
        normalize_ion_type,
    )

# Optional: if the project provides datacollect.h5_tools.ds_to_df, it will be used first.
try:
    from datacollect import h5_tools as _h5_tools  # type: ignore
except Exception:
    _h5_tools = None


# -----------------------------
# Basic regular expressions
# -----------------------------

# Orbital token examples: 1s, 1s2, 2p, 2p6, 3d10, 4f
ORB_RE = re.compile(r"^(\d+)([spdfghijklmno])(\d*)$", re.IGNORECASE)

# Term symbols such as (3P), (1S), (1D)
TERM_RE = re.compile(r"\(([^()]*)\)")

L_ORDER = "spdfghijklmno"


# -----------------------------
# String cleaning and parent-term normalization
# -----------------------------

def _clean_dots(s: str) -> str:
    """Remove whitespace, merge repeated dots, and strip leading/trailing dots."""
    s = str(s)
    s = s.replace("..", ".")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\.+", ".", s)
    s = s.strip(".")
    return s


def _decode_cfg_value(v) -> str:
    """Safely convert cfg values read from HDF5/pandas into ordinary strings."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    return str(v)


def normalize_cfg(cfg: str) -> str:
    """Normalize to lowercase and clean dot separators."""
    s = _decode_cfg_value(cfg).strip().lower()
    if not s or s == "nan":
        return ""
    return _clean_dots(s)


def _find_matching_paren(s: str, start: int) -> int:
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _is_term_content(inner: str) -> bool:
    """
    Decide whether the contents inside parentheses look like a parent term / term,
    rather than a nested cfg group.

    For example, (4P), (1D), and (3P) should keep the parentheses, while
    (2s.2p2.(4P)) is a nested cfg group and should be expanded.
    """
    inner = str(inner).strip().lower()
    if not inner:
        return False
    if "." in inner or "(" in inner or ")" in inner:
        return False
    return bool(re.fullmatch(r"\d*[a-z]+\d*", inner))


def flatten_group_parens(cfg: str) -> str:
    """
    Expand a cfg segment enclosed by parentheses while preserving term brackets.

    Examples:
        1s.(2s.2p2.(4P)) -> 1s.2s.2p2.(4p)
        2s2.2p4.(1D).3d2.(3P) -> Preserve the term brackets as-is
    """
    s = normalize_cfg(cfg)
    if not s:
        return ""

    def rec(text: str) -> str:
        out: List[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch != "(":
                out.append(ch)
                i += 1
                continue

            j = _find_matching_paren(text, i)
            if j < 0:
                out.append(ch)
                i += 1
                continue

            inner = rec(text[i + 1:j])
            if _is_term_content(inner):
                out.append(f"({inner})")
            else:
                out.append(inner)
            i = j + 1

        return "".join(out)

    return _clean_dots(rec(s))


def _attach_term_style(cfg: str) -> str:
    """Attach the term bracket directly to the preceding orbital: 2p4.(1D) -> 2p4(1D)."""
    s = flatten_group_parens(cfg)
    s = s.replace(".(", "(")
    return _clean_dots(s)


def _dot_term_style(cfg: str) -> str:
    """Keep the dot before the term bracket: 2p4(1D) -> 2p4.(1D)."""
    s = flatten_group_parens(cfg)
    s = _attach_term_style(s)
    s = re.sub(r"(\d+[spdfghijklmno]\d*)\(", r"\1.(", s)
    return _clean_dots(s)


def hard_paren_variants(cfg: str) -> Set[str]:
    """
    Generate format-equivalent variants that preserve term information.

    Includes:
      - The original normalized form
      - The expanded nested cfg-group form
      - Both dotted and undotted term-bracket styles

    Examples:
      1s.(2s.2p2.(4P))
      1s.2s.2p2.(4p)
      1s.2s.2p2(4p)
    """
    s = normalize_cfg(cfg)
    if not s:
        return set()

    flat = flatten_group_parens(s)
    out = {s, flat, _attach_term_style(flat), _dot_term_style(flat)}
    return {_clean_dots(x) for x in out if x}


def term_drop_variants(cfg: str) -> Set[str]:
    """
    Generate medium variants by gradually removing parent terms.

    Examples:
      2s2.2p4(1D).3d2(3P)
        -> 2s2.2p4.3d2(3P)
        -> 2s2.2p4(1D).3d2
        -> 2s2.2p4.3d2

    Each resulting form is also provided in both term-dot styles.
    """
    base = _attach_term_style(cfg)
    if not base:
        return set()

    matches = list(TERM_RE.finditer(base))
    if not matches:
        return set()

    out: Set[str] = set()
    n = len(matches)

    # Term counts are usually small; if there are many, avoid an exponential 2^n blow-up and only generate single-removal and full-removal variants.
    masks: Iterable[Tuple[int, ...]]
    if n <= 8:
        masks = itertools.chain.from_iterable(itertools.combinations(range(n), r) for r in range(1, n + 1))
    else:
        masks = itertools.chain(((i,) for i in range(n)), (tuple(range(n)),))

    for remove_idx_tuple in masks:
        remove_idx = set(remove_idx_tuple)
        parts: List[str] = []
        last = 0
        for i, m in enumerate(matches):
            parts.append(base[last:m.start()])
            if i not in remove_idx:
                parts.append(m.group(0))
            last = m.end()
        parts.append(base[last:])
        v = _clean_dots("".join(parts))
        if not v:
            continue
        out.add(v)
        out.add(_attach_term_style(v))
        out.add(_dot_term_style(v))

    hard = hard_paren_variants(cfg)
    return {x for x in out if x and x not in hard}


def drop_term(cfg: str) -> str:
    """
    Remove all term symbols / parent terms and expand nested cfg groups first.

    Examples:
        2p(3P)                   -> 2p
        2p.(3P)                  -> 2p
        1s.(2s.2p2.(4P))         -> 1s.2s.2p2
        2s2.2p4.(1D).3d2.(3P)   -> 2s2.2p4.3d2
    """
    s = _attach_term_style(cfg)
    s = TERM_RE.sub("", s)
    return _clean_dots(s)


# -----------------------------
# Orbital parsing and filtering of clearly unphysical cases
# -----------------------------

def _orbital_parts(token: str) -> Optional[Tuple[int, str, int]]:
    """
    Return (n, l, occ), or None on failure.
    If occ is empty, treat it as 1.
    """
    token = normalize_cfg(token)
    m = ORB_RE.fullmatch(token)
    if not m:
        return None

    n = int(m.group(1))
    l = m.group(2).lower()
    occ_str = m.group(3)
    occ = int(occ_str) if occ_str else 1
    return n, l, occ


def max_occ_for_l(l: str) -> int:
    """Return the maximum occupancy for a subshell: s=2, p=6, d=10, f=14, ..."""
    l = str(l).lower()
    if l not in L_ORDER:
        return 0
    l_index = L_ORDER.index(l)
    return 2 * (2 * l_index + 1)


def is_valid_orbital_token(token: str, max_occ: Optional[int] = None) -> bool:
    """
    Check whether a single orbital token is broadly reasonable.

    This mainly avoids obviously invalid orbitals such as 1p, 1d, 2d, and 2f.
    It also supports real closed-shell notation such as 2p6 and 3d10.
    """
    parts = _orbital_parts(token)
    if parts is None:
        return False

    n, l, occ = parts
    if n < 1 or l not in L_ORDER:
        return False

    l_index = L_ORDER.index(l)
    if l_index >= n:
        return False

    occ_limit = max_occ if max_occ is not None else max_occ_for_l(l)
    return 1 <= occ <= occ_limit


def contains_limit_cfg(cfg: str) -> bool:
    """Check whether a cfg contains series limit / threshold markers such as .limit."""
    s = drop_term(cfg)
    if not s:
        return False
    return any(tok == "limit" or tok.endswith("limit") for tok in s.split("."))


def split_orbitals_no_term(cfg: str) -> List[str]:
    """Remove terms and split the cfg into orbital tokens using dots as separators."""
    s = drop_term(cfg)
    if not s:
        return []
    return [x for x in s.split(".") if x]


def electron_count_from_tokens(tokens: Sequence[str]) -> int:
    """Estimate the electron count from orbital token occupancies; return -1 if parsing fails."""
    total = 0
    for tok in tokens:
        parts = _orbital_parts(tok)
        if parts is None:
            return -1
        total += parts[2]
    return total


def _valid_orbital_tokens_from_cfg(cfg: str, *, allow_limit: bool = False) -> List[str]:
    toks = split_orbitals_no_term(cfg)
    if allow_limit:
        toks = [t for t in toks if t != "limit" and not t.endswith("limit")]
    return toks


# -----------------------------
# Seed filtering
# -----------------------------

def is_valid_seed(cfg: str, ion_type: str) -> bool:
    """
    Check whether a cfg is suitable as a seed for the current ion.

    For ordinary cfg values, the electron count must be present in ion_config.py's seed_electron_counts.
    For .limit cfg values, a conservative rule is used: at least one valid orbital token before limit is required, and later only identity / safe formatting-equivalent transformations are considered.
    """
    ion_type = normalize_ion_type(ion_type)
    s = drop_term(cfg)
    if not s:
        return False

    if contains_limit_cfg(s):
        toks = _valid_orbital_tokens_from_cfg(s, allow_limit=True)
        return bool(toks) and all(is_valid_orbital_token(t) for t in toks)

    toks = split_orbitals_no_term(s)
    if not toks:
        return False
    if not all(is_valid_orbital_token(t) for t in toks):
        return False

    e_count = electron_count_from_tokens(toks)
    if e_count < 0:
        return False
    return e_count in get_seed_electron_counts(ion_type)


# Compatibility shim: these functions remain available if other code calls them directly.
def is_hlike_seed(cfg: str) -> bool:
    return is_valid_seed(cfg, "h-like")


def is_helike_seed(cfg: str) -> bool:
    return is_valid_seed(cfg, "he-like")


def is_lilike_seed(cfg: str) -> bool:
    return is_valid_seed(cfg, "li-like")


def is_belike_seed(cfg: str) -> bool:
    return is_valid_seed(cfg, "be-like")


def is_nelike_seed(cfg: str) -> bool:
    return is_valid_seed(cfg, "ne-like")


# -----------------------------
# Shortening rules
# -----------------------------

def _strip_core_prefix_once(cfg_no_term: str, core_prefix: str) -> str:
    """Remove a complete core prefix only once."""
    s = _clean_dots(cfg_no_term)
    core = _clean_dots(core_prefix)
    if not core or s == core:
        return ""
    prefix = core + "."
    if s.startswith(prefix):
        return _clean_dots(s[len(prefix):])
    return ""


def _valid_short_cfg(rest: str, ion_type: str) -> bool:
    if not rest or contains_limit_cfg(rest):
        return False
    toks = split_orbitals_no_term(rest)
    if not toks:
        return False
    if not all(is_valid_orbital_token(t) for t in toks):
        return False
    return electron_count_from_tokens(toks) in get_short_electron_counts(ion_type)


def shorten_cfg(cfg: str, ion_type: str) -> Set[str]:
    """
    Generate closed-core shortening pairs from ion_config.py's shorten_prefixes.

    Supports multi-level shortening:
      Ne-like: 1s2.2s2.2p5.3s -> 2s2.2p5.3s -> 2p5.3s

    Does not handle .limit and does not omit open-shell levels.
    """
    ion_type = normalize_ion_type(ion_type)
    start = drop_term(cfg)
    out: Set[str] = set()
    if not start or contains_limit_cfg(start):
        return out

    prefixes = get_shorten_prefixes(ion_type)
    if not prefixes or not get_short_electron_counts(ion_type):
        return out

    queue = [start]
    seen = {start}
    while queue:
        cur = queue.pop(0)
        for core in prefixes:
            rest = _strip_core_prefix_once(cur, core)
            if not rest or rest in seen:
                continue
            seen.add(rest)
            if _valid_short_cfg(rest, ion_type):
                out.add(rest)
                queue.append(rest)
    return out


# -----------------------------
# Soft rules
# -----------------------------

def last_orbital(cfg: str) -> str:
    s = drop_term(cfg)
    parts = s.split(".")
    return parts[-1] if parts else s


def l_soft_variants(orb: str) -> List[str]:
    """
    Generate candidates for the last orbital with l +/- 1 and filter out obviously invalid orbitals.
    """
    parts = _orbital_parts(orb)
    if parts is None:
        return []

    n, l, occ = parts
    idx = L_ORDER.index(l)
    out: List[str] = []

    for new_idx in [idx - 1, idx + 1]:
        if 0 <= new_idx < len(L_ORDER):
            new_l = L_ORDER[new_idx]
            occ_str = "" if occ == 1 else str(occ)
            candidate = f"{n}{new_l}{occ_str}"
            if is_valid_orbital_token(candidate):
                out.append(candidate)
    return out


def cfg_structure_category(cfg: str, ion_type: str) -> Tuple[object, ...]:
    """
    Define the structural category for soft pairs.

    Soft variants are only allowed within the same category so that shortening families are not incorrectly pulled together by soft rules.
    For example, 2s2.2p5.3s and 2p5.3s can only be linked by shortening, not by soft.
    """
    ion_type = normalize_ion_type(ion_type)
    s = drop_term(cfg)
    if not s:
        return ("empty",)
    if contains_limit_cfg(s):
        return ("limit", s)

    toks = split_orbitals_no_term(s)
    if not toks:
        return ("empty",)

    prefix_without_last = ".".join(toks[:-1])
    e_before_last = electron_count_from_tokens(toks[:-1]) if len(toks) > 1 else 0

    if toks[0] == "1s":
        cls = "1s-hole"
    else:
        cls = "normal"
        for core in get_shorten_prefixes(ion_type):
            core_toks = core.split(".")
            if toks[:len(core_toks)] == core_toks:
                cls = f"closed-core:{core}"
                break

    return (cls, len(toks), prefix_without_last, e_before_last)


def same_structure_category(a: str, b: str, ion_type: str) -> bool:
    return cfg_structure_category(a, ion_type) == cfg_structure_category(b, ion_type)


def _build_soft_from_base(base: str, ion_type: str, real_cfg_pool: Optional[Set[str]] = None) -> Set[str]:
    """
    Generate soft variants for a base cfg by changing only the last orbital's l value.

    Additional constraints:
      1. Does not handle .limit
      2. soft_cfg must satisfy the current ion_type
      3. If real_cfg_pool is provided, soft_cfg must already exist there or be produced by safe rules
      4. base / soft_cfg must belong to the same structural category
    """
    ion_type = normalize_ion_type(ion_type)
    base = drop_term(base)
    if not base or contains_limit_cfg(base):
        return set()

    orb = last_orbital(base)
    out: Set[str] = set()

    for v in l_soft_variants(orb):
        if "." in base:
            parts = base.split(".")
            soft_cfg = _clean_dots(".".join(parts[:-1] + [v]))
        else:
            soft_cfg = _clean_dots(v)

        if not soft_cfg or soft_cfg == base:
            continue
        if not is_valid_seed(soft_cfg, ion_type):
            continue
        if not same_structure_category(base, soft_cfg, ion_type):
            continue
        if real_cfg_pool is not None and soft_cfg not in real_cfg_pool:
            continue
        out.add(soft_cfg)

    return out


# -----------------------------
# HDF5 reading
# -----------------------------

def _df_from_h5(path: str, key: str):
    """Read an HDF5 dataset into a DataFrame. Prefer the project's datacollect.h5_tools.ds_to_df helper."""
    if _h5_tools is not None:
        try:
            return _h5_tools.ds_to_df(path, key)  # type: ignore
        except Exception:
            pass

    try:
        with h5py.File(path, "r") as f:
            arr = f[key][()]
        return pd.DataFrame(arr)
    except Exception:
        return None


def read_cfgs_from_h5(path: str) -> List[str]:
    """Read cfg values from all lev_* tables in a single h5 file."""
    cfgs: List[str] = []

    with h5py.File(path, "r") as f:
        keys = list(f.keys())

    for key in keys:
        if not key.startswith("lev_"):
            continue

        df = _df_from_h5(path, key)
        if df is None:
            continue

        col = None
        for c in ["cfg", "CFG", "configuration", "config"]:
            if c in df.columns:
                col = c
                break

        if col is None:
            continue

        cfgs.extend([_decode_cfg_value(x) for x in df[col].tolist()])

    return cfgs


def find_h5_files(atomic_dir: str, prefix: str) -> List[str]:
    """Find h5/hdf5 files with the given prefix in a directory."""
    h5_files: List[str] = []
    if not os.path.isdir(atomic_dir):
        raise NotADirectoryError(f"atomic_dir does not exist: {atomic_dir}")

    for fn in os.listdir(atomic_dir):
        fn_low = fn.lower()
        if fn_low.startswith(prefix) and fn_low.endswith((".h5", ".hdf5")):
            h5_files.append(os.path.join(atomic_dir, fn))

    h5_files.sort()
    return h5_files


# -----------------------------
# Pair-construction helpers
# -----------------------------

def safe_cfg_family(cfg: str, ion_type: str) -> Set[str]:
    """Build safe variants of a real cfg that can be added to the real_cfg_pool."""
    family: Set[str] = set()
    for h in hard_paren_variants(cfg):
        if is_valid_seed(h, ion_type):
            family.add(h)
    for m in term_drop_variants(cfg):
        if is_valid_seed(m, ion_type):
            family.add(m)
    dt = drop_term(cfg)
    if dt and is_valid_seed(dt, ion_type):
        family.add(dt)
    for base in list(family) + [cfg, dt]:
        for s in shorten_cfg(base, ion_type):
            if s and is_valid_seed(s, ion_type):
                family.add(s)
    return family


# -----------------------------
# Main workflow
# -----------------------------

def main():
    # Optional: h-like / he-like / li-like / be-like / ne-like
    ion_type = "ne-like"

    # Directory containing h5 files whose names start with 01/02/03/04/10
    atomic_dir = "./clean_data"

    # Output directory for training pairs
    train_data_dir = "./map_data/cfg_ts/train_data"

    # Root directory for subsequent model outputs
    model_base_dir = "./map_data/cfg_ts"

    seed = 0
    max_seed_cfg = 80000
    max_pairs = 300000

    # Keep this True to avoid inventing complex Be-like / Ne-like cfg strings that do not actually exist.
    soft_must_exist_in_real_pool = True
    create_model_dir = True

    # ============================================================
    # Usually no need to change anything below
    # ============================================================

    ion_type = normalize_ion_type(ion_type)
    ion_info = get_ion_info(ion_type)

    prefix = str(ion_info["prefix"])
    out_path = os.path.join(train_data_dir, str(ion_info["pair_file"]))
    model_dir = os.path.join(model_base_dir, str(ion_info["model_dir_name"]))

    rng = random.Random(seed)

    h5_files = find_h5_files(atomic_dir, prefix)
    if not h5_files:
        raise FileNotFoundError(
            f"No {prefix}*.h5 or {prefix}*.hdf5 files found under {atomic_dir}"
        )

    print(f"ion_type: {ion_type}")
    print(f"h5 prefix: {prefix}")
    print(f"h5 files: {len(h5_files)}")
    print(f"train data output: {out_path}")
    print(f"model dir: {model_dir}")

    seeds: List[str] = []
    for fp in h5_files:
        try:
            seeds.extend(read_cfgs_from_h5(fp))
        except Exception as e:
            print(f"skip file: {fp}  reason: {e}")
            continue
        if len(seeds) >= max_seed_cfg:
            break

    seeds_norm: List[str] = []
    for s in seeds:
        s = normalize_cfg(s)
        if not s:
            continue
        if is_valid_seed(s, ion_type):
            seeds_norm.append(s)

    seeds_uniq = sorted(set(seeds_norm))
    rng.shuffle(seeds_uniq)

    # Build a real cfg pool to filter soft pairs more safely.
    real_cfg_pool: Set[str] = set(seeds_uniq)
    for cfg in seeds_uniq:
        real_cfg_pool.update(safe_cfg_family(cfg, ion_type))

    pairs: List[Tuple[str, str, float, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    def add_pair(a, b, w, rel):
        a = normalize_cfg(a)
        b = normalize_cfg(b)
        if not a or not b:
            return

        key = (a, b, rel)
        if key not in seen:
            seen.add(key)
            pairs.append((a, b, w, rel))

        key2 = (b, a, rel)
        if key2 not in seen:
            seen.add(key2)
            pairs.append((b, a, w, rel))

    for cfg in seeds_uniq:
        # hard: formatting-equivalent + identity.
        hard_vars = hard_paren_variants(cfg)
        for v in hard_vars:
            if v != cfg and is_valid_seed(v, ion_type):
                add_pair(cfg, v, 1.0, "hard")
        add_pair(cfg, cfg, 1.0, "hard")

        # .limit is kept conservatively: only identity / safe formatting-equivalent variants are allowed; it does not participate in medium / shortening / soft.
        if contains_limit_cfg(cfg):
            if len(pairs) >= max_pairs:
                break
            continue

        # medium: gradually remove parent terms, including removing all of them.
        medium_vars = term_drop_variants(cfg)
        cfg_noterm = drop_term(cfg)
        if cfg_noterm:
            medium_vars.add(cfg_noterm)

        for mv in sorted(medium_vars):
            if mv != cfg and is_valid_seed(mv, ion_type):
                add_pair(cfg, mv, 0.8, "medium")
                # After removing some terms, the resulting dot/parenthesis formatting remains hard-equivalent.
                for hv in hard_paren_variants(mv):
                    if hv != mv and is_valid_seed(hv, ion_type):
                        add_pair(mv, hv, 1.0, "hard")

        # shortening: try closed-core shortening for the original cfg, the term-stripped cfg, and progressively term-stripped variants.
        shortening_bases = {cfg, cfg_noterm} | medium_vars
        for base in sorted(x for x in shortening_bases if x):
            if contains_limit_cfg(base):
                continue
            for s in shorten_cfg(base, ion_type):
                if s and is_valid_seed(s, ion_type):
                    add_pair(base, s, 0.95, "shortening")

        # soft: only vary the last orbital's l by +/- 1 within the same structural category, and by default must already be present in the real_cfg_pool.
        soft_bases: List[str] = []
        for base in shortening_bases | set(shorten_cfg(cfg, ion_type)):
            base = drop_term(base)
            if not base or contains_limit_cfg(base):
                continue
            if base not in soft_bases and is_valid_seed(base, ion_type):
                soft_bases.append(base)

        pool = real_cfg_pool if soft_must_exist_in_real_pool else None
        for base in soft_bases:
            for soft_cfg in _build_soft_from_base(base, ion_type, real_cfg_pool=pool):
                add_pair(base, soft_cfg, 0.1, "soft")

        if len(pairs) >= max_pairs:
            break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("anchor\tpositive\tweight\trelation\n")
        for a, b, w, rel in pairs[:max_pairs]:
            f.write(f"{a}\t{b}\t{w:.3f}\t{rel}\n")

    if create_model_dir:
        os.makedirs(model_dir, exist_ok=True)

    print(f"seed cfg unique: {len(seeds_uniq)}")
    print(f"real cfg pool: {len(real_cfg_pool)}")
    print(f"pairs: {min(len(pairs), max_pairs)}")
    print(f"wrote train data: {out_path}")
    print(f"model dir ready: {model_dir}")


if __name__ == "__main__":
    main()
