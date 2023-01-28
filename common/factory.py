from infer.inference import Inference, MyPy, PyreInfer, PyreQuery, TypeWriter, Type4Py, HiTyper
from icr.resolution import ConflictResolution, SubtypeVoting, Delegation

_INFERENCE_FACTORY: dict[str, type[Inference]] = {
    MyPy.__name__.lower(): MyPy,
    PyreInfer.__name__.lower(): PyreInfer,
    PyreQuery.__name__.lower(): PyreQuery,
    HiTyper.__name__.lower(): HiTyper,
    TypeWriter.__name__.lower(): TypeWriter,
    Type4Py.__name__.lower(): Type4Py,
}


def _inference_factory(value: str) -> type[Inference]:
    return _INFERENCE_FACTORY[value.lower()]


_ENGINE_FACTORY: dict[str, type[ConflictResolution]] = {
    SubtypeVoting.__name__.lower(): SubtypeVoting,
    Delegation.__name__.lower(): Delegation,
}


def _engine_factory(value: str) -> type[ConflictResolution]:
    return _ENGINE_FACTORY[value.lower()]
