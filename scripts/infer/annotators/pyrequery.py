import contextlib
import pathlib
import shutil
import time
import typing

import libcst
from libcst import codemod, metadata, matchers as m, Module
from libcst.metadata.type_inference_provider import PyreData
from libsa4py import pyre
from pyre_check.client.command_arguments import CommandArguments, StartArguments
from pyre_check.client.commands import initialize, start, ExitCode, stop
from pyre_check.client.configuration import configuration, Configuration
from pyre_check.client.identifiers import PyreFlavor

from .tool_annotator import ParallelTopNAnnotator
from .normalisation import Normalisation

from scripts import  utils
from scripts.common import transformers as t


class PyreQueryProjectApplier(
    ParallelTopNAnnotator[set[pathlib.Path], PyreData | None]
):
    @classmethod
    @contextlib.contextmanager
    def context(cls, project_location: pathlib.Path) -> None:
        with utils.working_dir(wd=project_location):
            pyre.clean_pyre_config(project_path=str(project_location))

            src_dirs = [str(project_location)]
            cmd_args = CommandArguments(source_directories=src_dirs)

            config = configuration.create_configuration(
                arguments=cmd_args,
                base_directory=project_location,
            )

            initialize.write_configuration(
                configuration=dict(
                    site_package_search_strategy="pep561",
                    source_directories=["."],
                ),
                configuration_path=project_location / ".pyre_configuration",
            )

            start_args = StartArguments.create(
                command_argument=cmd_args, no_watchman=True,
            )
            assert start.run(configuration=config, start_arguments=start_args) == ExitCode.SUCCESS

            yield

            assert stop.run(config, flavor=PyreFlavor.CLASSIC) == ExitCode.SUCCESS
            if (dotdir := project_location / ".pyre").is_dir():
                shutil.rmtree(str(dotdir))

    def extract_predictions_for_file(
        self, path2topn: set[pathlib.Path], path: pathlib.Path, topn: int
    ) -> PyreData | None:
        queried = pyre.pyre_query_types(
            project_path=str(self.context.metadata_manager.root_path),
            file_path=str(self.context.metadata_manager.root_path / path),
        )
        return queried

    def annotator(self, annotations: PyreData | None) -> codemod.Codemod:
        if not annotations:
            return _DummyApplier(context=self.context)
        return PyreQueryFileApplier(context=self.context, pyre_data=annotations)

    def normalisation(self) -> Normalisation:
        return Normalisation(
            bad_generics=True,
            lowercase_aliases=True,
            normalise_union_ts=True,
            typing_text_to_str=True,
        )


class _DummyApplier(codemod.Codemod):
    def transform_module_impl(self, tree: Module) -> Module:
        return tree

class PyreQueryFileApplier(
    t.HintableDeclarationTransformer,
    t.HintableParameterTransformer,
    t.HintableReturnTransformer,
):
    
    METADATA_DEPENDENCIES = (metadata.TypeInferenceProvider,)
    
    def __init__(self, context: codemod.CodemodContext) -> None:
        super().__init__(context)

    def annotated_param(
        self, param: libcst.Param, annotation: libcst.Annotation
    ) -> libcst.Param:
        return param.with_changes(annotation=self._infer_param_type(param))

    def unannotated_param(self, param: libcst.Param) -> libcst.Param:
        return param.with_changes(annotation=self._infer_param_type(param))

    def annotated_function(
        self, function: libcst.FunctionDef, annotation: libcst.Annotation
    ) -> libcst.FunctionDef:
        return function.with_changes(returns=self._infer_ret_type(function))

    def unannotated_function(self, function: libcst.FunctionDef) -> libcst.FunctionDef:
        return function.with_changes(returns=self._infer_ret_type(function))

    def instance_attribute_hint(
        self, original_node: libcst.AnnAssign, target: libcst.Name
    ) -> t.Actions:
        return self._annotate_annassign(original_node, target)

    def annotated_assignment(
        self,
        original_node: libcst.AnnAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> t.Actions:
        return self._annotate_annassign(original_node, target)

    def annotated_hint(
        self,
        original_node: libcst.AnnAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> t.Actions:
        return self._annotate_annassign(original_node, target)

    def assign_single_target(
        self,
        original_node: libcst.Assign,
        target: libcst.Name | libcst.Attribute,
    ) -> t.Actions:
        return self._annotate_single_assign(original_node, target)

    def assign_multiple_targets_or_augassign(
        self,
        original_node: typing.Union[libcst.Assign, libcst.AugAssign],
        target: libcst.Name | libcst.Attribute,
    ) -> t.Actions:
        return self._annotate_multiple_assign(original_node, target)

    def for_target(
        self,
        original_node: libcst.For,
        target: libcst.Name | libcst.Attribute,
    ) -> t.Actions:
        return self._annotate_compound_statement(original_node, target)

    def withitem_target(
        self,
        original_node: libcst.With,
        target: libcst.Name | libcst.Attribute,
    ) -> t.Actions:
        return self._annotate_compound_statement(original_node, target)

    def global_target(
        self,
        original_node: libcst.Assign | libcst.AnnAssign | libcst.AugAssign,
        target: libcst.Name,
    ) -> t.Actions:
        ...

    def nonlocal_target(
        self,
        original_node: libcst.Assign | libcst.AnnAssign | libcst.AugAssign,
        target: libcst.Name,
    ) -> t.Actions:
        ...

    def _infer_param_type(self, node: libcst.Param) -> libcst.Annotation | None:
        inferred = self.get_metadata(
            metadata.TypeInferenceProvider, node.name, None
        )
        if inferred:
            return libcst.Annotation(libcst.parse_expression(inferred))

        return None

    def _infer_ret_type(self, function: libcst.FunctionDef) -> libcst.Annotation | None:
        inferred = self.get_metadata(
            metadata.TypeInferenceProvider, function.name, None
        )
        if inferred:
            return libcst.Annotation(libcst.parse_expression(inferred))

        return None

    def _annotate_annassign(
        self,
        original_node: libcst.AnnAssign,
        target: libcst.Name,
    ) -> t.Actions:
        if vartype := self._infer_var_type(target):
            action = t.Replace(
                matcher=m.Annotation(original_node.annotation.annotation),
                replacement=vartype,
            )
        else:
            action = t.Untouched()

        return [action]

    def _annotate_single_assign(
        self, original_node: libcst.Assign, target: libcst.Name | libcst.Attribute
    ) -> t.Actions:
        if vartype := self._infer_var_type(target):
            action = t.Replace(
                matcher=m.Assign(),
                replacement=libcst.AnnAssign(
                    target=target, annotation=vartype, value=original_node.value
                ),
            )
        else:
            action = t.Untouched()

        return [action]

    def _annotate_multiple_assign(
        self,
        original_node: libcst.Assign | libcst.AugAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> t.Actions:
        if vartype := self._infer_var_type(target):
            action = t.Prepend(node=libcst.AnnAssign(target, annotation=vartype))
        else:
            action = t.Untouched()

        return [action]

    def _annotate_compound_statement(
        self,
        original_node: libcst.BaseCompoundStatement,
        target: libcst.Name | libcst.Attribute,
    ) -> t.Actions:
        if vartype := self._infer_var_type(target):
            action = t.Prepend(node=libcst.AnnAssign(target, annotation=vartype))
        else:
            action = t.Untouched()

        return [action]

    def _infer_var_type(
        self, node: libcst.Name | libcst.Attribute
    ) -> libcst.Annotation | None:
        inferred = self.get_metadata(metadata.TypeInferenceProvider, node, None)
        if inferred:
            return libcst.Annotation(libcst.parse_expression(inferred))

        return None
