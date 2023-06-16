import abc
from typing import Union, Optional

import libcst
import pandera.typing as pt
from libcst import codemod as c, helpers as h, matchers as m

from scripts.common import transformers as t
from scripts.common.schemas import TypeCollectionSchema


class _SSATransformer(
    t.HintableDeclarationTransformer, t.ScopeAwareTransformer, abc.ABC
):
    def __init__(self, context: c.CodemodContext) -> None:
        super().__init__(context)

    def transform_target(self, target: Union[libcst.Name, libcst.Attribute], consume: bool) -> t.Actions:
        scope = self.qualified_scope()
        qname = self.qualified_name(target)

        if (new_target := self.lookup(target, scope, qname, consume)) is not None:
            assert isinstance(
                replacement := libcst.parse_expression(new_target),
                (libcst.Name, libcst.Attribute),
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
        target: Union[libcst.Name, libcst.Attribute],
        scope: tuple[str],
        qname: str,
        consume: bool
    ) -> Optional[str]:
        ...

    # globals and nonlocals remain untouched
    def global_target(
        self, _: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign], _2: libcst.Name
    ) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def nonlocal_target(
        self, _1: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign], _2: libcst.Name
    ) -> t.Actions:
        return t.Actions((t.Untouched(),))

    # transform but do not consume
    def instance_attribute_hint(
        self,
        _1: libcst.AnnAssign,
        target: libcst.Name,
    ) -> t.Actions:
        return self.transform_target(target, consume=False)

    #
    def annotated_hint(
        self,
        original_node: libcst.AnnAssign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> t.Actions:
        return self.transform_target(target, consume=False)

    # actual assignments; simply rename targets
    def annotated_assignment(
        self,
        annassign: libcst.AnnAssign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> t.Actions:
        return self.transform_target(target, consume=True)

    def assign_single_target(
        self,
        assign: libcst.Assign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> t.Actions:
        return self.transform_target(target, consume=True)

    def assign_multiple_targets_or_augassign(
        self,
        assign: Union[libcst.Assign, libcst.AugAssign],
        target: Union[libcst.Name, libcst.Attribute],
    ) -> t.Actions:
        return self.transform_target(target, consume=True)

    def for_target(
        self, forloop: libcst.For, target: Union[libcst.Name, libcst.Attribute]
    ) -> t.Actions:
        return self.transform_target(target, consume=True)

    # def compfor_target(
    #    self, _1: libcst.CompFor, target: libcst.Name | libcst.Attribute
    # ) -> t.Actions:
    #    return self.transform_target(target)

    def withitem_target(
        self, withstmt: libcst.With, target: Union[libcst.Name, libcst.Attribute]
    ) -> t.Actions:
        return self.transform_target(target, consume=True)


class QName2SSATransformer(_SSATransformer):
    def __init__(
        self,
        context: c.CodemodContext,
        annotations: pt.DataFrame[TypeCollectionSchema],
    ):
        super().__init__(context)
        self.annotations = annotations.assign(consumed=0)

    def lookup(
        self,
        target: Union[libcst.Name, libcst.Attribute],
        scope: tuple[str],
        qname: str,
        consume: bool
    ) -> str:
        return self._lookup(target, scope, qname, consume=consume)

    def _lookup(
        self,
        target: Union[libcst.Name, libcst.Attribute],
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

    def lookup(
        self,
        target: Union[libcst.Name, libcst.Attribute],
        scope: tuple[str],
        qname: str,
        consume: bool,
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
