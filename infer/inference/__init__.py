import pathlib
import typing

from ._base import Inference

from .mypy import MyPy
from .pyreinfer import PyreInfer
from .pyrequery import PyreQuery

from .hity import HiTyper
from .typewriter import TypeWriter
from .t4py import Type4PyN1


def _fix_ml_parameters(
    tool: type[Inference], **kwargs
) -> typing.Callable[[pathlib.Path, pathlib.Path, pathlib.Path], Inference]:
    assert not any(fix in kwargs for fix in ("mutable", "readonly", "cache"))
    return lambda mutable, readonly, cache: tool(
        mutable=mutable, readonly=readonly, cache=cache, **kwargs
    )


SUPPORTED_TOOLS: dict[str, type[Inference]] = {
    MyPy.__name__.lower(): MyPy,
    PyreInfer.__name__.lower(): PyreInfer,
    PyreQuery.__name__.lower(): PyreQuery,
    HiTyper.__name__.lower(): HiTyper,
    TypeWriter.__name__.lower(): TypeWriter,
    Type4PyN1.__name__.lower(): _fix_ml_parameters(
        Type4PyN1, model_path=pathlib.Path.cwd() / "models" / "type4py"
    ),
}


def factory(value: str) -> type[Inference]:
    return SUPPORTED_TOOLS[value.lower()]


__all__ = [
    "Inference",
    "MyPy",
    "PyreInfer",
    "PyreQuery",
    "TypeWriter",
    "Type4PyN1",
    "HiTyper",
    "SUPPORTED_TOOLS",
    "factory",
]
