import os
import pathlib
import re
import shutil

import pandas._libs.missing as missing
import pandera.typing as pt

import libcst
from libcst import metadata
from libcst import matchers as m
from libcst import helpers
from libcst.codemod import _cli as cstcli

from pyre_check.client.commands import start, stop
from pyre_check.client import configuration, command_arguments

import pandas as pd

from common import ast_helper, visitors
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
        df = ast_helper.generate_qname_ssas_for_file(df)

        return df.assign(method=self.method, topn=0).pipe(pt.DataFrame[InferredSchema])


class _PyreQuery2Annotations(
    visitors.HintableDeclarationVisitor,
    visitors.HintableParameterVisitor,
    visitors.HintableReturnVisitor,
    visitors.ScopeAwareVisitor,
):
    _CALLABLE_RETTYPE_REGEX = re.compile(rf", ([^\]]+)\]$")

    METADATA_DEPENDENCIES = (
        metadata.TypeInferenceProvider,
        metadata.ScopeProvider,
    )

    def __init__(self, modpkg: helpers.ModuleNameAndPackage) -> None:
        super().__init__()
        self.modpkg = modpkg
        self.annotations: list[tuple[TypeCollectionCategory, str, str]] = []

    def annotated_assignment(self, target: libcst.Name | libcst.Attribute, _: libcst.Annotation) -> None:
        assgnty = self._infer_type(target)
        qname = self.qualified_name(target)
        self.annotations.append((TypeCollectionCategory.VARIABLE, qname, assgnty))

    def instance_attribute_hint(self, target: libcst.Name, _: libcst.Annotation | None) -> None:
        assgnty = self._infer_type(target)
        qname = self.qualified_name(target)
        self.annotations.append((TypeCollectionCategory.INSTANCE_ATTR, qname, assgnty))

    def unannotated_target(self, target: libcst.Name | libcst.Attribute) -> None:
        assgnty = self._infer_type(target)
        qname = self.qualified_name(target)
        self.annotations.append((TypeCollectionCategory.VARIABLE, qname, assgnty))

    # Ignore?
    def annotated_hint(self, _1: libcst.Name | libcst.Attribute, _2: libcst.Annotation) -> None:
        ...

    def annotated_param(self, param: libcst.Param, _: libcst.Annotation) -> None:
        qname = self.qualified_name(param.name.value)
        functy = self._infer_type(param)
        self.annotations.append((TypeCollectionCategory.CALLABLE_PARAMETER, qname, functy))

    def unannotated_param(self, param: libcst.Param) -> None:
        qname = self.qualified_name(param.name.value)
        functy = self._infer_type(param)
        self.annotations.append((TypeCollectionCategory.CALLABLE_PARAMETER, qname, functy))

    def annotated_function(self, function: libcst.FunctionDef, _: libcst.Annotation) -> None:
        qname = self.qualified_name(function.name.value)
        functy = self._infer_rettype(function)
        self.annotations.append((TypeCollectionCategory.CALLABLE_RETURN, qname, functy))

    def unannotated_function(self, function: libcst.FunctionDef) -> None:
        qname = self.qualified_name(function.name.value)
        functy = self._infer_rettype(function)
        self.annotations.append((TypeCollectionCategory.CALLABLE_RETURN, qname, functy))

    def _infer_type(self, node: libcst.Name | libcst.Attribute | libcst.Param) -> str | missing.NAType:
        if (anno := self.get_metadata(metadata.TypeInferenceProvider, node, None)) is None:
            return missing.NA

        return anno.removeprefix(self.modpkg.name + ".")

    def _infer_rettype(self, node: libcst.FunctionDef) -> str | missing.NAType:
        if (anno := self.get_metadata(metadata.TypeInferenceProvider, node.name, None)) is None:
            return missing.NA

        functy = re.findall(_PyreQuery2Annotations._CALLABLE_RETTYPE_REGEX, anno)
        functy = functy[0] if functy else missing.NA

        return functy.removeprefix(self.modpkg.name + ".") if pd.notna(functy) else functy
