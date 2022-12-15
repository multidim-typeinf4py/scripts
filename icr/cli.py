import pathlib
import itertools

import click

from .resolution import ConflictResolution, Argumentation, DecisionTheory, Delegation

from .inference import Inference, MyPy, Pyre


@click.command(
    name="icr",
    help="Intelligent Conflict Resolution on multiple inference methodologies",
)
@click.option(
    "-s",
    "--static",
    type=click.Choice(choices=[MyPy.__name__.lower(), Pyre.__name__.lower()]),
    callback=lambda ctx, _, value: {
        MyPy.__name__.lower(): MyPy,
        Pyre.__name__.lower(): Pyre,
    }[value.lower()],
    multiple=True,
)
@click.option(
    "-e",
    "--engine",
    type=click.Choice(
        choices=[Argumentation.__name__, DecisionTheory.__name__, Delegation.__name__],
        case_sensitive=False,
    ),
    callback=lambda ctx, _, value: {
        Argumentation.__name__.lower(): Argumentation,
        DecisionTheory.__name__.lower(): DecisionTheory,
        Delegation.__name__.lower(): Delegation,
    }[value.lower()],
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
    type=click.Path(exists=False, dir_okay=True, path_type=pathlib.Path),
    required=False,
)
def entrypoint(
    static_ts: list[type[Inference]],
    engine_t: type[ConflictResolution],
    input: pathlib.Path,
    output: pathlib.Path | None,
) -> None:
    statics = list(map(lambda ctor: ctor(input), static_ts))
    for inference in itertools.chain(statics):
        inference.infer()

    inferences = {inference.method: inference.inferred for inference in itertools.chain(statics)}


if __name__ == "__main__":
    entrypoint()
