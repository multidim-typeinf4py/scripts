import functools

import click
import pathlib
import tqdm

import libcst
from libcst import codemod, matchers as m

from scripts.common import output, extending
from scripts.common.schemas import InferredSchema, TypeCollectionSchema, ExtendedTypeCollectionSchema

from scripts.infer.inference import PyreQuery
from scripts.infer.structure import DatasetFolderStructure

from scripts.infer.annotators.normalisation import Normalisation, Normaliser

from scripts.symbols.collector import build_type_collection
from scripts import utils

import pandas as pd
from pandera import typing as pt





@click.command(name="dataset", help="Consume dataset into inference agnostic DataFrame")
@click.option(
    "-d",
    "--dataset",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
)
@click.option(
    "-o",
    "--outpath",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path),
    required=True,
    help="Folder for type collection dataframe to be written into",
)
@click.option(
    "-r",
    "--overwrite",
    required=False,
    default=False,
    help="Overwrite existing results in output folder",
    is_flag=True,
)
@click.option(
    "-e",
    "--extended",
    required=False,
    default=False,
    help="Create ExtendedDatasetSchema",
    is_flag=True,
)
def cli_entrypoint(dataset: pathlib.Path, outpath: pathlib.Path, overwrite: bool, extended: bool) -> None:
    structure = DatasetFolderStructure(dataset_root=dataset)
    print(structure)

    normalisation_strategy = Normalisation(
        normalise_union_ts=True,
        remove_if_all_any=True,
        lowercase_aliases=True,
    )

    test_set = structure.test_set()
    for project, subset in (pbar := tqdm.tqdm(test_set.items())):
        pbar.set_description(desc=f"Collecting from {project}")

        dataset_io = output.DatasetIO(artifact_root=outpath, dataset=structure, repository=project)
        extended_dataset_io = output.ExtendedDatasetIO(artifact_root=outpath, dataset=structure, repository=project)
        if not overwrite and not extended and dataset_io.full_location().exists():
            print(
                f"Skipping {project}; dataset already exists, no extension requested, and overwrite flag was not provided!"
            )
            continue

        elif extended and dataset_io.full_location().exists() and not extended_dataset_io.full_location().exists():
            print(f"Loading {project}; base dataset already exists, extended dataset does NOT exist and was requested, loading from disk...")
            collection = dataset_io.read()

        else:
            with utils.scratchpad(project) as sc:
                # Normalise codebase annotations
                res = codemod.parallel_exec_transform_with_prettyprint(
                    transform=Normaliser(codemod.CodemodContext(), normalisation_strategy),
                    jobs=utils.worker_count(),
                    repo_root=sc,
                    files=[str(sc / f) for f in subset]
                )
                print(utils.format_parallel_exec_result("Normalising codebase", res))

                collection = build_type_collection(root=sc, allow_stubs=False, subset=subset).df
                dataset_io.write(collection)

        if not extended:
            continue

        if not overwrite and extended and extended_dataset_io.full_location().exists():
            print(f"Skipping {project}; extended dataset already exists and no extension was requested!")

        print("Building parametric representation")
        collection[ExtendedTypeCollectionSchema.parametric_anno] = collection[ExtendedTypeCollectionSchema.anno].progress_apply(
            lambda anno: extending.make_parametric(anno)
        )

        print("Simple or complex?")
        collection[ExtendedTypeCollectionSchema.simple_or_complex] = collection[ExtendedTypeCollectionSchema.anno].progress_apply(
            lambda anno: extending.is_simple_or_complex(anno)
        )

        collection = collection.pipe(pt.DataFrame[ExtendedTypeCollectionSchema])
        extended_dataset_io.write(collection)






def is_simple_or_complex(annotation: str | None) -> str | None:
    if pd.isna(annotation) or annotation == "":
        return None
    class ComplexityCounter(m.MatcherDecoratableVisitor):
        def __init__(self):
            super().__init__()
            self.counter = 0

        @m.call_if_not_inside(m.Attribute())
        def visit_Name(self, node: libcst.Name) -> None:
            self.counter += 1

        def visit_Attribute(self, node: libcst.Attribute) -> None:
            self.counter += 1

    visitor = ComplexityCounter()
    libcst.parse_expression(annotation).visit(visitor)

    return "simple" if visitor.counter <= 1 else "complex"




if __name__ == "__main__":
    tqdm.pandas()
    cli_entrypoint()
