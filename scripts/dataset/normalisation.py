from __future__ import annotations

import pandas as pd
from typet5 import PythonType
from typet5.type_env import AccuracyMetric


def to_limited(anno: str | None) -> str | None:
    if pd.isna(anno):
        return None

    pt = PythonType.from_str(anno)
    converter = AccuracyMetric(
        common_type_names=set(),
        normalize_types=True,
        relaxed_equality=False,
        match_base_only=False,
        ignore_namespace=False,
        ast_depth_limit=2
    )

    return str(converter.process_type(pt))


def to_adjusted(anno: str | None) -> str | None:
    if pd.isna(anno):
        return None

    pt = PythonType.from_str(anno)
    converter = AccuracyMetric(
        common_type_names=set(),
        normalize_types=True,
        relaxed_equality=True,
        match_base_only=False,
        ignore_namespace=True,
    )

    return str(converter.process_type(pt))


def to_base(anno: str | None) -> str | None:
    if pd.isna(anno):
        return None

    pt = PythonType.from_str(anno)
    converter = AccuracyMetric(
        common_type_names=set(),
        normalize_types=True,
        relaxed_equality=True,
        match_base_only=True,
        ignore_namespace=True,
    )

    return str(converter.process_type(pt))