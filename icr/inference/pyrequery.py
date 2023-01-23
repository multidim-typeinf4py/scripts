import os
import pathlib
import re
import shutil
import weakref

import pandas._libs.missing as missing
import pandera.typing as pt

import libcst as cst
from libcst import metadata
from libcst import matchers as m
from libcst import helpers
from libcst.codemod import _cli as cstcli

from pyre_check.client.commands import start, stop
from pyre_check.client import configuration, command_arguments

import pandas as pd

from common import _helper
from common.schemas import InferredSchema, TypeCollectionCategory, TypeCollectionSchema

from ._base import PerFileInference


class PyreQuery(PerFileInference):
    method = "pyre-query"

    def __init__(self, project: pathlib.Path) -> None:
        super().__init__(project)

        cmd_args = command_arguments.CommandArguments(source_directories=[str(project)])
        self.config = config = configuration.create_configuration(
            arguments=cmd_args,
            base_directory=project,
        )

        start.run(
            configuration=config,
            start_arguments=command_arguments.StartArguments.create(
                command_argument=cmd_args, no_watchman=True
            ),
        )

        paths = cstcli.gather_files([str(self.project)])
        relpaths = [os.path.relpath(path, str(project)) for path in paths]
        self.repo_manager = metadata.FullRepoManager(
            repo_root_dir=str(self.project),
            paths=relpaths,
            providers=[metadata.TypeInferenceProvider],
        )

    def infer(self) -> None:
        super().infer()
        stop.run(self.config)

        if (dotdir := self.project / ".pyre").is_dir():
            shutil.rmtree(str(dotdir))

    def _infer_file(self, relative: pathlib.Path) -> pt.DataFrame[InferredSchema]:
        fullpath = str(self.project / relative)
        module = self.repo_manager.get_metadata_wrapper_for_path(str(relative))

        modpkg = helpers.calculate_module_and_package(
            repo_root=str(self.project), filename=fullpath
        )
        visitor = _PyreQuery2Annotations(modpkg)
        module.visit(visitor)

        df = pd.DataFrame(
            visitor.annotations,
            columns=[
                TypeCollectionSchema.category,
                TypeCollectionSchema.qname,
                TypeCollectionSchema.anno,
            ],
        ).assign(file=str(relative))
        df = _helper.generate_qname_ssas_for_file(df)

        return df.assign(method=self.method, topn=0).pipe(pt.DataFrame[InferredSchema])


class _PyreQuery2Annotations(cst.CSTVisitor):
    _CALLABLE_RETTYPE_REGEX = re.compile(rf", ([^\]]+)\]$")

    METADATA_DEPENDENCIES = (
        metadata.TypeInferenceProvider,
        metadata.ScopeProvider,
        metadata.QualifiedNameProvider,
    )

    def __init__(self, modpkg: helpers.ModuleNameAndPackage) -> None:
        super().__init__()
        self.modpkg = modpkg
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

        qname, functy = inferred
        match self.get_metadata(metadata.ScopeProvider, target):
            case metadata.ClassScope():
                self.annotations.append((TypeCollectionCategory.CLASS_ATTR, qname, functy))

            case _:
                self.annotations.append((TypeCollectionCategory.VARIABLE, qname, functy))

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        if (inferred := self._infer_rettype(node)) is None:
            return None

        qname, functy = inferred
        self.annotations.append((TypeCollectionCategory.CALLABLE_RETURN, qname, functy))

    def visit_Param(self, node: cst.Param) -> bool | None:
        if (inferred := self._infer_type(node.name)) is None:
            return None

        qname, functy = inferred
        self.annotations.append((TypeCollectionCategory.CALLABLE_PARAMETER, qname, functy))

    def _infer_type(self, node: cst.CSTNode) -> tuple[str, str] | None:
        if (anno := self.get_metadata(metadata.TypeInferenceProvider, node, None)) is None:
            return None
        if not (qname := self.get_metadata(metadata.QualifiedNameProvider, node)):
            return None

        qname_s = next(iter(qname)).name.replace(".<locals>.", ".")
        return qname_s, anno.removeprefix(self.modpkg.name + ".")

    def _infer_rettype(self, node: cst.FunctionDef) -> tuple[str, str] | None:
        if (anno := self.get_metadata(metadata.TypeInferenceProvider, node.name, None)) is None:
            return None
        if not (qname := self.get_metadata(metadata.QualifiedNameProvider, node.name)):
            return None

        functy = re.findall(_PyreQuery2Annotations._CALLABLE_RETTYPE_REGEX, anno)
        assert len(functy) <= 1
        functy = functy[0] if functy else missing.NA

        qname_s = next(iter(qname)).name.replace(".<locals>.", ".")
        return qname_s, functy.removeprefix(self.modpkg.name + ".") if pd.notna(functy) else functy
