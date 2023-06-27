import pathlib

import click
import tqdm
from libcst import codemod, metadata
from pandera import typing as pt

from scripts.common.output import ExtendedDatasetIO, ContextIO
from scripts.common.schemas import (
    TypeCollectionSchema,
    ExtendedTypeCollectionSchema,
)


@click.command(
    name="context", help="Create vectors to classify contexts of annotatables"
)
@click.option(
    "-d",
    "--dataset",
    type=click.Path(
        exists=True, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    required=True,
    help="Dataset to iterate over (can also be a singular project!)",
)
@click.option(
    "-o",
    "--outpath",
    type=click.Path(
        exists=False, file_okay=False, dir_okay=True, path_type=pathlib.Path
    ),
    required=True,
    help="Base folder for inference results to be written into",
)
@click.option(
    "-w",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite an existing output folder from previous run",
)
def cli_entrypoint(
    dataset: pathlib.Path,
    outpath: pathlib.Path,
    overwrite: bool,
) -> None:
    from scripts.context import RelevantFeatures
    from scripts.context.visitors import generate_context_vectors
    from scripts.infer.insertion import TypeAnnotationApplierTransformer
    from scripts.infer.structure import DatasetFolderStructure
    from scripts.tt5exp.remover import TT5AllAnnotRemover
    from scripts.utils import format_parallel_exec_result, scratchpad, worker_count

    structure = DatasetFolderStructure(dataset)
    test_set = structure.test_set()

    annotation_columns = [
        ExtendedTypeCollectionSchema.raw_anno,
        ExtendedTypeCollectionSchema.depth_limited_anno,
        ExtendedTypeCollectionSchema.adjusted_anno,
        ExtendedTypeCollectionSchema.base_anno,
    ]

    for project, subset in (pbar := tqdm.tqdm(test_set.items())):
        pbar.set_description(desc=f"{project}")

        # Load ExtendedTypeCollectionSchema with all forms of
        # labels as processed by TypeT5's normalisation
        extended_ground_truths = ExtendedDatasetIO(
            artifact_root=outpath,
            dataset=structure,
            repository=project,
        ).read()

        for target in annotation_columns:
            output_io = ContextIO(
                artifact_root=outpath,
                dataset=structure,
                repository=project,
                annotation_form=target,
            )

            if not overwrite and output_io.full_location().exists():
                print(
                    f"Skipping computing context on '{target}' on '{project}'; "
                    f"artifact exists and --overwrite was not specified"
                )
                continue

            # Remove all type annotations
            with scratchpad(project) as sc:
                files = codemod.gather_files([str(sc)])

                print(
                    f"Apply annotations from '{target}' as a scratchpad for context gathering!"
                )
                transformed_ground_truths = (
                    extended_ground_truths.rename(
                        columns={target: TypeCollectionSchema.anno}
                    )
                    .drop(columns=set(annotation_columns).difference({target}))
                    .pipe(pt.DataFrame[TypeCollectionSchema])
                )

                result = codemod.parallel_exec_transform_with_prettyprint(
                    transform=TT5AllAnnotRemover(codemod.CodemodContext()),
                    files=files,
                    jobs=worker_count(),
                    repo_root=str(sc),
                )
                print(
                    format_parallel_exec_result(
                        action="Annotation Removal", result=result
                    )
                )

                result = codemod.parallel_exec_transform_with_prettyprint(
                    transform=TypeAnnotationApplierTransformer(
                        context=codemod.CodemodContext(
                            metadata_manager=metadata.FullRepoManager(
                                repo_root_dir=sc,
                                paths=files,
                                providers={metadata.FullyQualifiedNameProvider},
                            )
                        ),
                        annotations=transformed_ground_truths,
                    ),
                    files=files,
                    jobs=worker_count(),
                    repo_root=str(sc),
                )
                print(
                    format_parallel_exec_result(
                        action="Annotation Application", result=result
                    )
                )

                context_vectors = generate_context_vectors(
                    features=RelevantFeatures.default(),
                    project=sc,
                    subset=subset,
                )

                print(f"Writing context vectors to {output_io.full_location()}")
                output_io.write(context_vectors)


if __name__ == "__main__":
    cli_entrypoint()
