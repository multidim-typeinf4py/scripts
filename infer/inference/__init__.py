from ._base import Inference

from .mypy import MyPy
from .pyreinfer import PyreInfer
from .pyrequery import PyreQuery

from .hity import HiTyperType4PyTop10
from .typewriter import TypeWriterTop10
from .t4py import Type4PyTop10
from .tt5 import TypeT5Top10

SUPPORTED_TOOLS: dict[str, type[Inference]] = {
    MyPy.__name__.lower(): MyPy,
    PyreInfer.__name__.lower(): PyreInfer,
    PyreQuery.__name__.lower(): PyreQuery,
    TypeWriterTop10.__name__.lower(): TypeWriterTop10,
    Type4PyTop10.__name__.lower(): Type4PyTop10,
    HiTyperType4PyTop10.__name__.lower(): HiTyperType4PyTop10,
    TypeT5Top10.__name__.lower(): TypeT5Top10,
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
    "HiTyperType4PyTop10",
    "SUPPORTED_TOOLS",
    "factory",
]
