import typing

import libcst
from libcst import codemod, metadata, matchers as m

from scripts.common import transformers as t


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
        if inferred and (cleaned := self.try_to_cleanup_type(inferred)):
            return libcst.Annotation(libcst.parse_expression(cleaned))

        return None


    def try_to_cleanup_type(self, type_annotation: str) -> str | None:
        if type_annotation.startswith("typing_extensions.Literal[b'"):
            return "typing_extensions.Literal[b'']"

        if type_annotation.startswith("typing_extensions.Literal['"):
            return "typing_extensions.Literal['']"

        # Indicates assignment to method or function, give up on these, no clue what they are
        if type_annotation.startswith("BoundMethod[typing.Callable"):
            return None

        return type_annotation