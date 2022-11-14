import pathlib
import sys

import click
import libcst as cst

from libcst.codemod import _cli as cstcli
import libcst.codemod as codemod
import libcst.metadata as metadata
from libcst.codemod.visitors._apply_type_annotations import (
    TypeCollector as LibCSTTypeCollector,
)
from libcst.codemod.visitors._gather_imports import (
    GatherImportsVisitor,
)

from .common import storage


# TODO: technically not accurate as this is a visitor, not a transformer
# TODO: but there does not seem to be a nicer way to execute this visitor in parallel
class TypeCollectorVistor(codemod.ContextAwareTransformer):
    collection: storage.TypeCollection

    def __init__(self, context: codemod.CodemodContext) -> None:
        super().__init__(context)

    def transform_module_impl(self, tree: cst.Module) -> cst.Module:
        metadataed = metadata.MetadataWrapper(tree)

        imports_visitor = GatherImportsVisitor(context=self.context)
        metadataed.visit(imports_visitor)

        type_collector = LibCSTTypeCollector(
            existing_imports=imports_visitor.module_imports,
            module_imports=imports_visitor.symbol_mapping,
            context=self.context,
        )
        metadataed.visit(type_collector)

        annotations = type_collector.annotations
        return tree


@click.command(name="diff", short_help="test based unified diff of the provided files")
@click.option(
    "-i",
    "--input",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    required=True,
    help="Repository to gather",
)
def cli(root: pathlib.Path):
    result = codemod.parallel_exec_transform_with_prettyprint(
        transform=TypeCollectorVistor(context=codemod.CodemodContext()),
        files=cstcli.gather_files(root),
        jobs=1,
        blacklist_patterns=["__init__.py"],
        repo_root=str(root),
    )

    print(
        f"Finished codemodding {result.successes + result.skips + result.failures} files!",
        file=sys.stderr,
    )
    print(f" - Transformed {result.successes} files successfully.", file=sys.stderr)
    print(f" - Skipped {result.skips} files.", file=sys.stderr)
    print(f" - Failed to codemod {result.failures} files.", file=sys.stderr)
    print(f" - {result.warnings} warnings were generated.", file=sys.stderr)


if __name__ == "__main__":
    cli()
