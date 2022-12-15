from .inference import Inference, MyPy, Pyre, TypeWriter
from .resolution import ConflictResolution, Argumentation, DecisionTheory, Delegation

_INFERENCE_FACTORY: dict[str, type[Inference]] = {
    MyPy.__name__.lower(): MyPy,
    Pyre.__name__.lower(): Pyre,
    TypeWriter.__name__.lower(): TypeWriter,
}


def _inference_factory(value: str | None) -> Inference:
    if value is None:
        return []

    return _INFERENCE_FACTORY[value.lower()]


_ENGINE_FACTORY: dict[str, type[ConflictResolution]] = {
    Argumentation.__name__.lower(): Argumentation,
    DecisionTheory.__name__.lower(): DecisionTheory,
    Delegation.__name__.lower(): Delegation,
}


def _engine_factory(value: str | None) -> ConflictResolution:
    if value is None:
        return []

    return _ENGINE_FACTORY[value.lower()]
