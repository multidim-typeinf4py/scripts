import pathlib
from pprint import pprint
import itertools

import click

from .resolution import ConflictResolution, Argumentation, DecisionTheory, Delegation

from .inference import Inference, MyPy, Pyre, TypeWriter, Type4Py, HiTyper
from . import _factory


@click.command(
    name="icr",
    help="Intelligent Conflict Resolution on multiple inference methodologies",
)
@click.option(
    "-s",
    "--static",
    type=click.Choice(choices=[MyPy.__name__.lower(), Pyre.__name__.lower()], case_sensitive=False),
    callback=lambda ctx, _, value: [_factory._inference_factory(v) for v in value] if value else [],
    required=False,
    multiple=True,
)
@click.option(
    "-p",
    "--prob",
    type=click.Choice(
        choices=[HiTyper.__name__.lower(), TypeWriter.__name__.lower(), Type4Py.__name__.lower()],
        case_sensitive=False,
    ),
    callback=lambda ctx, _, value: [_factory._inference_factory(v) for v in value] if value else [],
    required=False,
    multiple=True,
)
@click.option(
    "-e",
    "--engine",
    type=click.Choice(
        choices=[
            Argumentation.__name__.lower(),
            DecisionTheory.__name__.lower(),
            Delegation.__name__.lower(),
        ],
        case_sensitive=False,
    ),
    callback=lambda ctx, _, value: _factory._engine_factory(value),
    required=True,
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(exists=False, path_type=pathlib.Path),
    required=False,
)
def entrypoint(
    static: list[type[Inference]],
    prob: list[type[Inference]],
    engine: type[ConflictResolution],
    input: pathlib.Path,
    output: pathlib.Path | None,
) -> None:
    statics = list(map(lambda ctor: ctor(input), static))
    probabilistics = list(map(lambda ctor: ctor(input), prob))

    infs = lambda: itertools.chain(statics, probabilistics)

    for inference in infs():
        inference.infer()

    inferences = {inference.method: inference.inferred for inference in infs()}

    for inferrer in infs():
        print(inferrer.inferred)

    # eng = engine()
    # inference = eng.forward(inferences)


if __name__ == "__main__":
    entrypoint()
