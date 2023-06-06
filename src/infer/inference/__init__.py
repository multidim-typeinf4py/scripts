from ._base import Inference

from .mypy import MyPy
from .pyreinfer import PyreInfer
from .pyrequery import PyreQuery

from .typewriter import TypeWriterTop10
from .typilus import TypilusTop10
from .t4py import Type4PyTop10
from .tt5 import TypeT5Top10

from .hit4py import HiType4PyTop10
from .hitypilus import HiTypilusTop10
from .hitypewriter import HiTypewriterTop10

SUPPORTED_TOOLS: dict[str, type[Inference]] = {
    # Static inference tools
    MyPy.__name__.lower(): MyPy,
    PyreInfer.__name__.lower(): PyreInfer,
    PyreQuery.__name__.lower(): PyreQuery,

    # ML Models
    Type4PyTop10.__name__.lower(): Type4PyTop10,
    TypilusTop10.__name__.lower(): TypilusTop10,
    TypeWriterTop10.__name__.lower(): TypeWriterTop10,

    # Hybrid TypeT5
    TypeT5Top10.__name__.lower(): TypeT5Top10,

    # Hybrid HiTyper integrations
    HiType4PyTop10.__name__.lower(): HiType4PyTop10,
    HiTypilusTop10.__name__.lower(): HiTypilusTop10,
    HiTypewriterTop10.__name__.lower(): HiTypewriterTop10,
}


def factory(value: str) -> type[Inference]:
    return SUPPORTED_TOOLS[value.lower()]


__all__ = [
    "Inference",
    "MyPy",
    "PyreInfer",
    "PyreQuery",
    "TypeWriterTop10",
    "Type4PyTop10",
    "HiType4PyTop10",
    "SUPPORTED_TOOLS",
    "factory",
]
