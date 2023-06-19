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

from scripts.infer.annotators.tool_annotator import Normalisation

from scripts.symbols.collector import build_type_collection
from scripts import utils

import pandas as pd
from pandera import typing as pt


class Normaliser(codemod.Codemod):
    def __init__(self, context: codemod.CodemodContext, strategy: Normalisation) -> None:
        super().__init__(context=context)
        self.strategy = strategy

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        transformers = self.strategy.transformers(context=self.context)
        return functools.reduce(
            lambda mod, trans: trans.transform_module(mod),
            transformers,
            tree
        )


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
        unnest_union_t=True,
        lowercase_aliases=True,
        union_or_to_union_t=True,
        typing_text_to_str=True,
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
                utils.format_parallel_exec_result("Normalising codebase", res)

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


        #with utils.scratchpad(project) as sc:
        #    pbar.set_description(desc="Discovering Type Aliases with Pyre-Query")
        #    tool = PyreQuery()
        #    inferred = tool.infer(mutable=sc, readonly=project, subset=set)
        #    pyrequery_annotations = inferred.drop(
        #        columns=[InferredSchema.method, InferredSchema.topn]
        #    ).pipe(pt.DataFrame[TypeCollectionSchema])

        #    side_by_side = pd.merge(
        #        left=collection,
        #        right=pyrequery_annotations,
        #        how="left",
        #        on=[
        #            TypeCollectionSchema.file,
        #            TypeCollectionSchema.category,
        #            TypeCollectionSchema.qname,
        #            TypeCollectionSchema.qname_ssa,
        #        ],
        #        suffixes=("_ground_truth", "_pyrequery_augmented")
        #    )

        dataset_io.write(collection)






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
