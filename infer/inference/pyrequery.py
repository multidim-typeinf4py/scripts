import os
import pathlib
import re
import shutil
from typing import Union, Optional

from pandas._libs import missing
import pandera.typing as pt

import libcst
from libcst import metadata
from libcst import helpers
from libcst import codemod

from libsa4py import pyre

from pyre_check.client.commands import start, stop, initialize, ExitCode
from pyre_check.client import configuration, command_arguments

import pandas as pd
from pyre_check.client.identifiers import PyreFlavor

from common import ast_helper, visitors
from common.schemas import InferredSchema, TypeCollectionCategory, TypeCollectionSchema

from ._base import PerFileInference
import utils


class PyreQuery(PerFileInference):
    method = "pyre-query"

    def __init__(self, cache: Optional[pathlib.Path]):
        super().__init__(cache)
        self.repo_manager: Optional[metadata.FullRepoManager] = None

    def infer(
        self,
        mutable: pathlib.Path,
        readonly: pathlib.Path,
        subset: Optional[set[pathlib.Path]] = None,
    ) -> None:
        with utils.working_dir(wd=mutable):
            try:
                pyre.clean_pyre_config(project_path=str(mutable))
                cmd_args = command_arguments.CommandArguments(
                    source_directories=[str(mutable)]
                )
                config = configuration.create_configuration(
                    arguments=cmd_args,
                    base_directory=mutable,
                )

                initialize.write_configuration(
                    configuration=dict(
                        site_package_search_strategy="pep561",
                        source_directories=["."],
                    ),
                    configuration_path=mutable / ".pyre_configuration",
                )

                assert (
                    start.run(
                        configuration=config,
                        start_arguments=command_arguments.StartArguments.create(
                            command_argument=cmd_args, no_watchman=True
                        ),
                    )
                    == ExitCode.SUCCESS
                )

                paths = codemod.gather_files([str(mutable)])
                relpaths = [os.path.relpath(path, str(mutable)) for path in paths]
                self.repo_manager = metadata.FullRepoManager(
                    repo_root_dir=str(mutable),
                    paths=relpaths,
                    providers=[metadata.TypeInferenceProvider],
                    timeout=60,
                )

                super().infer(mutable=mutable, readonly=readonly, subset=subset)

            finally:
                stop.run(config, flavor=PyreFlavor.CLASSIC)

        if (dotdir := mutable / ".pyre").is_dir():
            shutil.rmtree(str(dotdir))

    def _infer_file(
        self, root: pathlib.Path, relative: pathlib.Path
    ) -> pt.DataFrame[InferredSchema]:
        assert self.repo_manager is not None
        fullpath = str(root / relative)
        module = self.repo_manager.get_metadata_wrapper_for_path(str(relative))

        modpkg = helpers.calculate_module_and_package(
            repo_root=str(root), filename=fullpath
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

        return df.assign(method=self.method, topn=1).pipe(pt.DataFrame[InferredSchema])


class _PyreQuery2Annotations(
    visitors.HintableDeclarationVisitor,
    visitors.HintableParameterVisitor,
    visitors.HintableReturnVisitor,
    visitors.ScopeAwareVisitor,
):
    _CALLABLE_RETTYPE_REGEX = re.compile(rf", ([^]]+)]$")

    METADATA_DEPENDENCIES = (
        metadata.TypeInferenceProvider,
        metadata.ScopeProvider,
    )

    def __init__(self, modpkg: helpers.ModuleNameAndPackage) -> None:
        super().__init__()
        self.modpkg = modpkg
        self.annotations: list[tuple[TypeCollectionCategory, str, str]] = []

    def libsa4py_hint(
        self, _: Union[libcst.Assign, libcst.AnnAssign], target: libcst.Name
    ) -> None:
        self._instance_attribute(target)

    def instance_attribute_hint(
        self, original_node: libcst.AnnAssign, target: libcst.Name
    ) -> None:
        self._instance_attribute(target)

    def annotated_param(self, param: libcst.Param, _: libcst.Annotation) -> None:
        self._parameter(param)

    def unannotated_param(self, param: libcst.Param) -> None:
        self._parameter(param)

    def annotated_function(
        self, function: libcst.FunctionDef, _: libcst.Annotation
    ) -> None:
        self._ret(function)

    def unannotated_function(self, function: libcst.FunctionDef) -> None:
        self._ret(function)

    def annotated_hint(
        self,
        original_node: libcst.AnnAssign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        self._variable(target)

    def annotated_assignment(
        self,
        original_node: libcst.AnnAssign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        self._variable(target)

    def unannotated_assign_single_target(
        self, original_node: libcst.Assign, target: Union[libcst.Name, libcst.Attribute]
    ) -> None:
        self._variable(target)

    def unannotated_assign_multiple_targets(
        self,
        original_node: Union[libcst.Assign, libcst.AugAssign],
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        self._variable(target)

    def for_target(
        self, original_node: libcst.For, target: Union[libcst.Name, libcst.Attribute]
    ) -> None:
        self._variable(target)

    def withitem_target(
        self, original_node: libcst.With, target: Union[libcst.Name, libcst.Attribute]
    ) -> None:
        self._variable(target)

    def _instance_attribute(self, target: libcst.Name) -> None:
        qname = self.qualified_name(target)
        functy = self._infer_type(target)
        self.annotations.append((TypeCollectionCategory.VARIABLE, qname, functy))

    def _variable(self, target: Union[libcst.Name, libcst.Attribute]) -> None:
        qname = self.qualified_name(target)
        functy = self._infer_type(target)
        self.annotations.append((TypeCollectionCategory.VARIABLE, qname, functy))

    def _parameter(self, param: libcst.Param) -> None:
        qname = self.qualified_name(param.name.value)
        functy = self._infer_type(param.name)
        self.annotations.append(
            (TypeCollectionCategory.CALLABLE_PARAMETER, qname, functy)
        )

    def _ret(self, function: libcst.FunctionDef) -> None:
        qname = self.qualified_name(function.name.value)
        functy = self._infer_rettype(function)
        self.annotations.append((TypeCollectionCategory.CALLABLE_RETURN, qname, functy))

    def global_target(
        self,
        original_node: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign],
        target: libcst.Name,
    ) -> None:
        pass

    def nonlocal_target(
        self,
        original_node: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign],
        target: libcst.Name,
    ) -> None:
        pass

    def _infer_type(self, node: libcst.Name) -> Union[str, missing.NAType]:
        if (
            anno := self.get_metadata(metadata.TypeInferenceProvider, node, None)
        ) is None:
            return missing.NA

        return anno.removeprefix(self.modpkg.name + ".")

    def _infer_rettype(self, node: libcst.FunctionDef) -> Union[str, missing.NAType]:
        if (
            anno := self.get_metadata(metadata.TypeInferenceProvider, node.name, None)
        ) is None:
            return missing.NA

        functy = re.findall(_PyreQuery2Annotations._CALLABLE_RETTYPE_REGEX, anno)
        functy = functy[0] if functy else missing.NA

        return (
            functy.removeprefix(self.modpkg.name + ".") if pd.notna(functy) else functy
        )
