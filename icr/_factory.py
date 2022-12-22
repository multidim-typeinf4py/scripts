from .inference import Inference, MyPy, Pyre, TypeWriter, Type4Py, HiTyper
from .resolution import ConflictResolution, SubtypeVoting, DecisionTheory, Delegation

_INFERENCE_FACTORY: dict[str, type[Inference]] = {
    MyPy.__name__.lower(): MyPy,
    Pyre.__name__.lower(): Pyre,
    HiTyper.__name__.lower(): HiTyper,
    TypeWriter.__name__.lower(): TypeWriter,
    Type4Py.__name__.lower(): Type4Py,
}


def _inference_factory(value: str | None) -> type[Inference]:
    if value is None:
        return []

    return _INFERENCE_FACTORY[value.lower()]


_ENGINE_FACTORY: dict[str, type[ConflictResolution]] = {
    SubtypeVoting.__name__.lower(): SubtypeVoting,
    DecisionTheory.__name__.lower(): DecisionTheory,
    Delegation.__name__.lower(): Delegation,
}


def _engine_factory(value: str | None) -> type[ConflictResolution]:
    if value is None:
        return []

    return _ENGINE_FACTORY[value.lower()]
