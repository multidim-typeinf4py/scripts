from ._base import Inference

from .mypy import MyPy
from .pyreinfer import PyreInfer
from .pyrequery import PyreQuery

from .typewriter import TypeWriterTop10, TypeWriterTop1, TypeWriterTop5
from .typilus import TypilusTop10, TypilusTop1, TypilusTop5
from .t4py import Type4PyTop10, Type4PyTop1, Type4PyTop5
from .tt5 import TypeT5Top1

from .hit4py import HiType4PyTop10
from .hitypilus import HiTypilusTop10
from .hitypewriter import HiTypeWriterTop10

SUPPORTED_TOOLS: dict[str, type[Inference]] = {
    # Static inference tools
    MyPy.__name__.lower(): MyPy,
    PyreInfer.__name__.lower(): PyreInfer,
    PyreQuery.__name__.lower(): PyreQuery,

    # ML Models @ Top 1
    Type4PyTop1.__name__.lower(): Type4PyTop1,
    TypilusTop1.__name__.lower(): TypilusTop1,
    TypeWriterTop1.__name__.lower(): TypeWriterTop1,

    # ML Models @ Top 5
    Type4PyTop5.__name__.lower(): Type4PyTop5,
    TypilusTop5.__name__.lower(): TypilusTop5,
    TypeWriterTop5.__name__.lower(): TypeWriterTop5,

    # ML Models @ Top 10
    Type4PyTop10.__name__.lower(): Type4PyTop10,
    TypilusTop10.__name__.lower(): TypilusTop10,
    TypeWriterTop10.__name__.lower(): TypeWriterTop10,

    # TypeT5
    TypeT5Top1.__name__.lower(): TypeT5Top1,
    #TypeT5Top5.__name__.lower(): Type4PyTop5,
    #TypeT5Top10.__name__.lower(): TypeT5Top10,

    # Hybrid HiTyper integrations
    HiType4PyTop10.__name__.lower(): HiType4PyTop10,
    HiTypilusTop10.__name__.lower(): HiTypilusTop10,
    HiTypeWriterTop10.__name__.lower(): HiTypeWriterTop10,
}


def factory(value: str) -> type[Inference]:
    return SUPPORTED_TOOLS[value.lower()]


__all__ = [
    "SUPPORTED_TOOLS",
    "factory",
]
