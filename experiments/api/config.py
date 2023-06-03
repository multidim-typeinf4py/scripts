from __future__ import annotations

import dataclasses

from experiments.api import filters, mappers


@dataclasses.dataclass
class ExperimentConfig:
    mapping: mappers.MappingConfig
    filtering: filters.FilteringConfig
