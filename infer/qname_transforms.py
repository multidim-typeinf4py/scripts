import abc

import libcst
import pandera.typing as pt
from libcst import codemod as c, helpers as h, matchers as m

from common import transformers as t
from common.schemas import TypeCollectionSchema


class _SSATransformer(
    t.HintableDeclarationTransformer, t.ScopeAwareTransformer, abc.ABC
):
    def __init__(self, context: c.CodemodContext) -> None:
        super().__init__(context)

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
        context: c.CodemodContext,
        annotations: pt.DataFrame[TypeCollectionSchema],
    ):
        super().__init__(context)
        self.annotations = annotations.assign(consumed=0)

    def annotated_hint(
        self, annassign: libcst.AnnAssign, target: libcst.Name | libcst.Attribute
    ) -> t.Actions:
        new_target = self._lookup(
            target,
            scope=self.qualified_scope(),
            qname=self.qualified_name(target),
            consume=False,
        )
        assert isinstance(
            replacement_target := libcst.parse_expression(new_target),
            libcst.Name | libcst.Attribute,
        )

        matcher = m.AnnAssign(
            target=target, annotation=annassign.annotation, value=annassign.value
        )
        replacement = libcst.AnnAssign(
            target=replacement_target,
            annotation=annassign.annotation,
            value=annassign.value,
        )

        action = t.Replace(matcher, replacement)
        return t.Actions((action,))

    def lookup(
        self,
        target: libcst.Name | libcst.Attribute,
        scope: tuple[str],
        qname: str,
    ) -> str:
        return self._lookup(target, scope, qname, consume=True)

    def _lookup(
        self,
        target: libcst.Name | libcst.Attribute,
        scope: tuple[str],
        qname: str,
        consume: bool,
    ) -> str:
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
        if consume:
            self.annotations.at[qname_ssa_ser.index[0], "consumed"] = 1

        if scope:
            qname_ssa = qname_ssa.removeprefix(".".join(scope) + ".")
        return qname_ssa


class SSA2QNameTransformer(_SSATransformer):
    def __init__(
        self,
        context: c.CodemodContext,
        annotations: pt.DataFrame[TypeCollectionSchema],
    ):
        super().__init__(context)
        self.annotations = annotations

    def annotated_hint(
        self, annassign: libcst.AnnAssign, target: libcst.Name | libcst.Attribute
    ) -> t.Actions:
        return self.transform_target(target)

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
