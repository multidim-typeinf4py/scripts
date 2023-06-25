import pathlib
import typing

import libcst
from libcst import codemod, matchers
from typet5.experiments import utils as typet5_utils
from typet5.static_analysis import SignatureMap, ProjectPath, _VisitKind, FunctionSignature, VariableSignature, \
    is_type_rhs

from scripts.infer.annotators import ParallelTopNAnnotator
from scripts.infer.annotators.normalisation import Normalisation


class HiTyperProjectApplier(
    ParallelTopNAnnotator[typing.Mapping[pathlib.Path, list[SignatureMap]], SignatureMap]
):
    def extract_predictions_for_file(
        self,
        path2topn: typing.Mapping[pathlib.Path, list[SignatureMap]],
        path: pathlib.Path,
        topn: int,
    ) -> SignatureMap:
        topns = path2topn.get(path, [{}])
        return topns[topn]


    def annotator(self, sigmap: SignatureMap) -> codemod.Codemod:
        return HiTyperFileApplier(context=self.context, sigmap=sigmap)

    def normalisation(self) -> Normalisation:
        return Normalisation.default()


class HiTyperFileApplier(codemod.Codemod):
    def __init__(
        self, context: codemod.CodemodContext, sigmap: SignatureMap
    ) -> None:
        super().__init__(context)
        self.sigmap = sigmap

    def transform_module_impl(self, tree: libcst.Module) -> libcst.Module:
        typet5_annotated = apply_sigmap(
            m=tree,
            sigmap=self.sigmap,
            module_name=self.context.full_module_name,
        )

        return typet5_annotated
    
    
# Adapted from TypeT5 implementation
def apply_sigmap(
    m: libcst.Module,
    sigmap: SignatureMap,
    module_name: str,
    add_default_imports=True,
) -> libcst.Module:
    """
    Apply the signature map to the module.
    """

    class Rewriter(libcst.CSTTransformer):
        def __init__(self):
            super().__init__()
            self.path_stack = [ProjectPath(module_name, "")]
            self.visit_stack = [_VisitKind.Root]

        @property
        def current_path(self) -> ProjectPath:
            return self.path_stack[-1]

        @property
        def current_visit_kind(self) -> _VisitKind:
            return self.visit_stack[-1]

        def enter_(self, name: str, kind: _VisitKind):
            self.path_stack.append(self.current_path.append(name))
            self.visit_stack.append(kind)

        def exit_(self):
            self.path_stack.pop()
            self.visit_stack.pop()

        def visit_FunctionDef(self, node: libcst.FunctionDef):
            self.enter_(node.name.value, _VisitKind.Function)

        def leave_FunctionDef(self, node, updated: libcst.FunctionDef):
            if isinstance(sig := sigmap.get(self.current_path), FunctionSignature):
                try:
                    updated = sig.apply(updated)
                except LookupError:
                    pass
            self.exit_()
            return updated

        def visit_ClassDef(self, node: "libcst.ClassDef") -> bool | None:
            self.enter_(node.name.value, _VisitKind.Class)

        def leave_ClassDef(self, node, updated: libcst.ClassDef):
            self.exit_()
            return updated

        def leave_AnnAssign(self, node, updated: libcst.AnnAssign):
            target = None
            match updated.target:
                case libcst.Name(name):
                    target = name

                case libcst.Attribute():
                    if matchers.matches(node, matchers.Attribute(matchers.Name("self"), matchers.Name())):
                        target = f"self.{node.attr.value}"
            if (
                target is not None
                and isinstance(
                    sig := sigmap.get(self.current_path.append(target)),
                    VariableSignature,
                )
                and sig.annot is not None
            ):
                updated = updated.with_changes(annotation=sig.annot)
            return updated

        def leave_Assign(self, node, updated: libcst.Assign):
            target = None
            match updated.targets:
                case [libcst.AssignTarget(target=libcst.Name(name))]:
                    target = name

                case [libcst.AssignTarget(target=libcst.Attribute())]:
                    node = typing.cast(libcst.Attribute, updated.targets[0].target)
                    if matchers.matches(node, matchers.Attribute(matchers.Name("self"), matchers.Name())):
                        target = f"self.{node.attr.value}"
            if (
                target is not None
                and isinstance(
                    sig := sigmap.get(self.current_path.append(target)),
                    VariableSignature,
                )
                and sig.annot is not None
                and not (
                    self.current_visit_kind == _VisitKind.Root
                    and is_type_rhs(updated.value)
                )  # skip annotating type aliases
            ):
                return libcst.AnnAssign(updated.targets[0].target, sig.annot, updated.value)
            return updated

        def leave_Module(self, node, updated: libcst.Module):
            if add_default_imports:
                return updated.with_changes(
                    body=[typet5_utils._DefaultImport] + list(updated.body),
                )
            return updated

    return m.visit(Rewriter())