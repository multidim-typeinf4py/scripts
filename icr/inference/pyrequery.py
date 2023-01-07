import weakref

import pathlib
import pandera.typing as pt

import libcst as cst
from libcst import metadata
from libcst import matchers as m
from libcst import helpers
from libcst.codemod import _cli as cstcli

from pyre_check.client.commands import start, stop
from pyre_check.client import configuration, command_arguments

import pandas as pd

from common.schemas import TypeCollectionCategory, TypeCollectionSchema, TypeCollectionSchemaColumns

from ._base import PerFileInference


class PyreQuery(PerFileInference):
    method = "pyre-query"

    def __init__(self, project: pathlib.Path) -> None:
        super().__init__(project)

        cmd_args = command_arguments.CommandArguments(
            dot_pyre_directory=project / ".pyre", source_directories=[str(project)]
        )
        config = configuration.create_configuration(
            arguments=cmd_args,
            base_directory=project,
        )

        start.run(
            configuration=config,
            start_arguments=command_arguments.StartArguments.create(
                command_argument=cmd_args, no_watchman=True
            ),
        )
        weakref.finalize(self, lambda: stop.run(config))

        self.repo_manager = metadata.FullRepoManager(
            repo_root_dir=str(self.project),
            paths=cstcli.gather_files([str(self.project)]),
            providers=[metadata.TypeInferenceProvider],
        )

    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[TypeCollectionSchema]:
        module = self.repo_manager.get_metadata_wrapper_for_path(str(self.project / relative))

        visitor = _PyreQuery2Annotations()
        module.visit(visitor)

        return (
            pd.DataFrame(visitor.annotations, columns=["category", "qname", "anno"])
            .assign(file=str(relative))
            .pipe(pt.DataFrame[TypeCollectionSchema])
        )


class _PyreQuery2Annotations(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        metadata.TypeInferenceProvider,
        metadata.ScopeProvider,
        metadata.QualifiedNameProvider,
    )

    def __init__(self) -> None:
        super().__init__()
        self.annotations: list[tuple[TypeCollectionCategory, str, str]] = []

    def visit_AssignTarget(self, node: cst.AssignTarget) -> bool | None:
        self._handle_assgn_tgt(node.target)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool | None:
        self._handle_assgn_tgt(node.target)

    def _handle_assgn_tgt(self, target: cst.BaseAssignTargetExpression) -> None:
        if not m.matches(target, m.Name() | m.Attribute(m.Name("self"), m.Name())):
            return None

        if (inferred := self._infer_type(target)) is None:
            return None

        match self.get_metadata(metadata.ScopeProvider, target):
            case metadata.ClassScope():
                self.annotations.append((TypeCollectionCategory.CLASS_ATTR, *inferred))

            case _:
                self.annotations.append((TypeCollectionCategory.VARIABLE, *inferred))

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        if (inferred := self._infer_type(node.name)) is None:
            return None
        self.annotations.append((TypeCollectionCategory.CALLABLE_RETURN, *inferred))

    def visit_Param(self, node: cst.Param) -> bool | None:
        if (inferred := self._infer_type(node.name)) is None:
            return None

        self.annotations.append((TypeCollectionCategory.CALLABLE_PARAMETER, *inferred))

    def _infer_type(self, node: cst.CSTNode) -> tuple[str, str] | None:
        if (anno := self.get_metadata(metadata.TypeInferenceProvider, node, None)) is None:
            return None
        if not (qname := self.get_metadata(metadata.QualifiedNameProvider, node)):
            return None

        full_qname = next(iter(qname)).name.replace(".<locals>.", ".")
        return full_qname, anno
