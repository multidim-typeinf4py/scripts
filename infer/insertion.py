import abc
import pathlib

import libcst

import pandera.typing as pt
from libcst import codemod, helpers as h, matchers as m, metadata

from common import transformers as t
from common.metadata.anno4inst import Annotation4InstanceProvider
from common.annotations import ApplyTypeAnnotationsVisitor
from common.schemas import TypeCollectionSchema
from common.storage import TypeCollection
from symbols.collector import TypeCollectorVisitor


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

        # AddImportsVisitor.add_needed_import(self.context, "typing")
        # AddImportsVisitor.add_needed_import(self.context, "typing", "*")

        # removed = tree.visit(TypeAnnotationRemover(context=self.context))

        symbol_collector = TypeCollectorVisitor.strict(context=self.context)
        tree.visit(symbol_collector)
        annotations = TypeCollection.to_libcst_annotations(
            module_tycol, symbol_collector.collection.df
        )

        with_ssa_qnames = QName2SSATransformer(
            context=self.context, annotations=module_tycol
        ).transform_module(tree)

        hinted = ApplyTypeAnnotationsVisitor(
            context=self.context,
            annotations=annotations,
            overwrite_existing_annotations=False,
            use_future_annotations=True,
        ).transform_module(with_ssa_qnames)

        with_qnames = SSA2QNameTransformer(
            context=self.context, annotations=module_tycol
        ).transform_module(hinted)
        return with_qnames


class _SSATransformer(
    t.HintableDeclarationTransformer, t.ScopeAwareTransformer, abc.ABC
):
    METADATA_DEPENDENCIES = (Annotation4InstanceProvider, metadata.ParentNodeProvider)

    def transform_target(self, target: libcst.Name | libcst.Attribute) -> t.Actions:
        scope = self.qualified_scope()
        qname = self.qualified_name(target)

        if (new_target := self.lookup(target, scope, qname)) is not None:
            assert isinstance(
                replacement := libcst.parse_expression(new_target),
                libcst.Name | libcst.Attribute,
            )

            if isinstance(target, libcst.Name):
                matcher = m.Name(target.value)

            else:
                instance, attr = h.get_full_name_for_node_or_raise(target).split(".")
                matcher = m.Attribute(value=m.Name(instance), attr=m.Name(attr))

            action = t.Replace(matcher, replacement)

            parent = self.get_metadata(metadata.ParentNodeProvider, target)
            if not m.matches(parent, m.AnnAssign(value=None)):
                if (anno := self.get_metadata(Annotation4InstanceProvider, target)) is not None:
                    hint = t.Prepend(libcst.AnnAssign(target=replacement, annotation=anno))
                    return t.Actions((hint, action))

            return t.Actions((action,))

        else:
            return t.Actions((t.Untouched(),))

    @abc.abstractmethod
    def lookup(
        self,
        target: libcst.Name | libcst.Attribute,
        scope: tuple[str],
        qname: str,
    ) -> str | None:
        ...

    # variations of INSTANCE_ATTR, globals and nonlocals, annotation hints remain untouched
    def instance_attribute_hint(
        self,
        _1: libcst.AnnAssign,
        _2: libcst.Name,
    ) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def annotated_hint(
        self, _1: libcst.AnnAssign, _2: libcst.Name | libcst.Attribute
    ) -> t.Actions:
        return t.Actions((t.Remove(),))

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

    # actual assignments; simply rename targets
    def annotated_assignment(
        self,
        annassign: libcst.AnnAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> t.Actions:
        return self.transform_target(target)

    def unannotated_assign_single_target(
        self,
        assign: libcst.Assign | libcst.AugAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> t.Actions:
        return self.transform_target(target)

    def unannotated_assign_multiple_targets(
        self,
        assign: libcst.Assign | libcst.AugAssign,
        target: libcst.Name | libcst.Attribute,
    ) -> t.Actions:
        return self.transform_target(target)

    def for_target(
        self, forloop: libcst.For, target: libcst.Name | libcst.Attribute
    ) -> t.Actions:
        return self.transform_target(target)

    # def compfor_target(
    #    self, _1: libcst.CompFor, target: libcst.Name | libcst.Attribute
    # ) -> t.Actions:
    #    return self.transform_target(target)

    def withitem_target(
        self, withstmt: libcst.With, target: libcst.Name | libcst.Attribute
    ) -> t.Actions:
        return self.transform_target(target)


class QName2SSATransformer(_SSATransformer):
    def __init__(
        self,
        context: codemod.CodemodContext,
        annotations: pt.DataFrame[TypeCollectionSchema],
    ):
        super().__init__(context)
        self.annotations = annotations.assign(consumed=0)

    def lookup(
        self,
        target: libcst.Name | libcst.Attribute,
        scope: tuple[str],
        qname: str,
    ) -> t.Actions:
        cand_mask = (self.annotations[TypeCollectionSchema.qname] == qname) & (
            self.annotations["consumed"] != 1
        )
        candidates = self.annotations.loc[cand_mask]
        assert (
            not candidates.empty
        ), f"Unable to lookup: {qname} for {h.get_full_name_for_node_or_raise(target)}"

        qname_ssa_ser = candidates[TypeCollectionSchema.qname_ssa]
        qname_ssa = str(qname_ssa_ser.iloc[0])

        # annotations hints do not consume, but must still be renamed
        # so that it is visible that a target is implicitly annotated
        self.annotations.at[qname_ssa_ser.index[0], "consumed"] = 1

        if scope:
            qname_ssa = qname_ssa.removeprefix(".".join(scope) + ".")
        return qname_ssa


class SSA2QNameTransformer(_SSATransformer):
    def __init__(
        self,
        context: codemod.CodemodContext,
        annotations: pt.DataFrame[TypeCollectionSchema],
    ):
        super().__init__(context)
        self.annotations = annotations

    def lookup(
        self,
        target: libcst.Name | libcst.Attribute,
        scope: tuple[str],
        qname: str,
    ) -> str:
        cand_mask = self.annotations[TypeCollectionSchema.qname_ssa] == qname
        candidates = self.annotations.loc[cand_mask]
        assert (
            not candidates.empty
        ), f"Unable to lookup: {qname} for {h.get_full_name_for_node_or_raise(target)}"

        qname_ser = candidates[TypeCollectionSchema.qname]
        qname = str(qname_ser.iloc[0])

        if scope:
            qname = qname.removeprefix(".".join(scope) + ".")
        return qname
