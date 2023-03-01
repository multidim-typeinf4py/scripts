import abc
import pathlib

import libcst
from libcst import codemod, metadata
from libcst.codemod.visitors._add_imports import AddImportsVisitor

import pandera.typing as pt

from common.annotations import ApplyTypeAnnotationsVisitor, TypeAnnotationRemover
from common.schemas import TypeCollectionSchema
from common.storage import TypeCollection
from common import transformers as t

from symbols.collector import TypeCollectorVistor


class TypeAnnotationApplierTransformer(codemod.ContextAwareTransformer):
    def __init__(
        self,
        context: codemod.CodemodContext,
        annotations: pt.DataFrame[TypeCollectionSchema],
    ) -> None:
        super().__init__(context)
        self.annotations = annotations

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        assert self.context.filename is not None
        assert self.context.metadata_manager is not None

        relative = pathlib.Path(self.context.filename).relative_to(
            self.context.metadata_manager.root_path
        )

        module_tycol = self.annotations[
            self.annotations[TypeCollectionSchema.file] == str(relative)
        ].pipe(pt.DataFrame[TypeCollectionSchema])

        AddImportsVisitor.add_needed_import(self.context, "typing")
        # AddImportsVisitor.add_needed_import(self.context, "typing", "*")

        removed = tree.visit(TypeAnnotationRemover())

        symbol_collector = TypeCollectorVistor.strict(context=self.context)
        symbol_collector.transform_module(removed)
        annotations = TypeCollection.to_libcst_annotations(
            module_tycol, symbol_collector.collection.df
        )

        with_ssa_qnames = QName2SSATransformer(
            context=self.context, annotations=module_tycol
        ).transform_module(removed)

        hinted = ApplyTypeAnnotationsVisitor(
            context=self.context,
            annotations=annotations,
            overwrite_existing_annotations=True,
            use_future_annotations=True,
            handle_function_bodies=True,
            create_class_attributes=True,
        ).transform_module(with_ssa_qnames)

        with_qnames = SSA2QNameTransformer(
            context=self.context, annotations=module_tycol
        ).transform_module(hinted)
        return with_qnames


class _SSATransformer(t.HintableDeclarationTransformer, t.ScopeAwareTransformer, abc.ABC):
    def instance_attribute_hint(
        self, _1: libcst.AnnAssign, target: libcst.Name, _2: libcst.Annotation
    ) -> t.Actions:
        return self.transform_target(target)

    def transform_target(self, target: libcst.Name | libcst.Attribute) -> t.Actions:
        if (new_target := self.lookup(target)) is not None:
            action = t.Replace(target, libcst.parse_expression(new_target))
        else:
            action = t.Untouched()

        return t.Actions((action,))

    @abc.abstractmethod
    def lookup(self, target: libcst.Name | libcst.Attribute) -> str | None:
        ...

    # variations of INSTANCE_ATTR, globals and nonlocals, annotation hints remain untouched
    def instance_attribute_hint(
        self, _1: libcst.AnnAssign, _2: libcst.Name, _3: libcst.Annotation
    ) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def libsa4py_hint(self, _1: libcst.Assign, _2: libcst.Name) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def global_target(
        self, _: libcst.Assign | libcst.AnnAssign | libcst.AugAssign, _2: libcst.Name
    ) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def nonlocal_target(
        self, _1: libcst.Assign | libcst.AnnAssign | libcst.AugAssign, _2: libcst.Name
    ) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def annotated_hint(
        self,
        _1: libcst.AnnAssign,
        target: libcst.Name | libcst.Attribute,
        _2: libcst.Annotation,
    ) -> t.Actions:
        return t.Actions((t.Untouched(),))

    # actual assignments; simply rename targets
    def annotated_assignment(
        self,
        _1: libcst.AnnAssign,
        target: libcst.Name | libcst.Attribute,
        _2: libcst.Annotation,
    ) -> t.Actions:
        return self.transform_target(target)

    def unannotated_assign_single_target(
        self, _1: libcst.Assign | libcst.AugAssign, target: libcst.Name | libcst.Attribute
    ) -> t.Actions:
        return self.transform_target(target)

    def unannotated_assign_multiple_targets(
        self, _1: libcst.Assign | libcst.AugAssign, target: libcst.Name | libcst.Attribute
    ) -> t.Actions:
        return self.transform_target(target)

    def for_target(self, _1: libcst.For, target: libcst.Name | libcst.Attribute) -> t.Actions:
        return self.transform_target(target)

    def compfor_target(
        self, _1: libcst.CompFor, target: libcst.Name | libcst.Attribute
    ) -> t.Actions:
        return self.transform_target(target)

    def withitem_target(
        self, _1: libcst.With, _2: libcst.WithItem, target: libcst.Name | libcst.Attribute
    ) -> t.Actions:
        return self.transform_target(target)


class QName2SSATransformer(_SSATransformer):
    def __init__(
        self, context: codemod.CodemodContext, annotations: pt.DataFrame[TypeCollectionSchema]
    ):
        super().__init__(context)
        self.annotations = annotations.assign(consumed=0)

    def lookup(self, target: libcst.Name | libcst.Attribute) -> t.Actions:
        scope = self.qualified_scope()
        qname = self.qualified_name(target)

        cand_mask = (self.annotations[TypeCollectionSchema.qname] == qname) & (
            self.annotations["consumed"] != 1
        )
        candidates = self.annotations.loc[cand_mask]
        assert not candidates.empty, f"Unable to lookup: {qname}"

        qname_ssa_ser = candidates[TypeCollectionSchema.qname_ssa]
        qname_ssa = str(qname_ssa_ser.iloc[0])
        self.annotations.at[qname_ssa_ser.index[0], "consumed"] = 1

        if scope:
            qname_ssa = qname_ssa.removeprefix(".".join(scope) + ".")
        return qname_ssa


class SSA2QNameTransformer(_SSATransformer):
    def __init__(
        self, context: codemod.CodemodContext, annotations: pt.DataFrame[TypeCollectionSchema]
    ):
        super().__init__(context)
        self.annotations = annotations

    def lookup(self, target: libcst.Name | libcst.Attribute) -> str:
        scope = self.qualified_scope()
        qname_ssa = self.qualified_name(target)

        cand_mask = self.annotations[TypeCollectionSchema.qname] == qname_ssa
        candidates = self.annotations.loc[cand_mask]
        assert not candidates.empty, f"Unable to lookup: {qname_ssa}"

        qname_ser = candidates[TypeCollectionSchema.qname]
        qname = str(qname_ser.iloc[0])
        self.annotations.at[qname_ser.index[0], "consumed"] = 1

        if scope:
            qname = qname.removeprefix(".".join(scope) + ".")
        return qname
