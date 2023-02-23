# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import functools

import libcst as cst
import libcst.matchers as m

from libcst.codemod._context import CodemodContext
from libcst.codemod._visitor import ContextAwareTransformer
from libcst.codemod.visitors._add_imports import AddImportsVisitor
from libcst.codemod.visitors._gather_global_names import GatherGlobalNamesVisitor
from libcst.codemod.visitors._gather_imports import GatherImportsVisitor
from libcst.codemod.visitors._imports import ImportItem
from libcst.helpers import get_full_name_for_node, get_full_name_for_node_or_raise
from libcst.metadata import PositionProvider, QualifiedNameProvider, ScopeProvider, ClassScope

from libcst.codemod.visitors._apply_type_annotations import (
    NameOrAttribute,
    NAME_OR_ATTRIBUTE,
    StarParamType,
    _module_and_target,
    _get_unique_qualified_name,
    _get_import_alias_names,
    _get_imported_names,
    _is_non_sentinel,
    _get_string_value,
    _find_generic_base,
    FunctionKey,
    FunctionAnnotation,
    ImportedSymbol,
    ImportedSymbolCollector,
    TypeCollector,
    AnnotationCounts,
)


@dataclass
class MultiVarAnnotations:
    """
    FORK OF LIBCST'S ANNOTATIONS: attributes is now a defaultdict with lists for values!

    Represents all of the annotation information we might add to
    a class:
    - All data is keyed on the qualified name relative to the module root
    - The ``functions`` field also keys on the signature so that we
      do not apply stub types where the signature is incompatible.

    The idea is that
    - ``functions`` contains all function and method type
      information from the stub, and the qualifier for a method includes
      the containing class names (e.g. "Cat.meow")
    - ``attributes`` similarly contains all globals
      and class-level attribute type information.
    - The ``class_definitions`` field contains all of the classes
      defined in the stub. Most of these classes will be ignored in
      downstream logic (it is *not* used to annotate attributes or
      method), but there are some cases like TypedDict where a
      typing-only class needs to be injected.
    - The field ``typevars`` contains the assign statement for all
      type variables in the stub, and ``names`` tracks
      all of the names used in annotations; together these fields
      tell us which typevars should be included in the codemod
      (all typevars that appear in annotations.)
    """

    # TODO: consider simplifying this in a few ways:
    # - We could probably just inject all typevars, used or not.
    #   It doesn't seem to me that our codemod needs to act like
    #   a linter checking for unused names.
    # - We could probably decide which classes are typing-only
    #   in the visitor rather than the codemod, which would make
    #   it easier to reason locally about (and document) how the
    #   class_definitions field works.

    functions: Dict[FunctionKey, FunctionAnnotation]
    attributes: collections.defaultdict[str, list[cst.Annotation | None]]
    class_definitions: Dict[str, cst.ClassDef]
    typevars: Dict[str, cst.Assign]
    names: Set[str]

    @classmethod
    def empty(cls) -> "MultiVarAnnotations":
        return MultiVarAnnotations({}, collections.defaultdict(list), {}, {}, set())

    def update(self, other: "MultiVarAnnotations") -> None:
        self.functions.update(other.functions)
        self.attributes.update(other.attributes)
        self.class_definitions.update(other.class_definitions)
        self.typevars.update(other.typevars)
        self.names.update(other.names)

    def finish(self) -> None:
        self.typevars = {k: v for k, v in self.typevars.items() if k in self.names}


class MultiVarTypeCollector(m.MatcherDecoratableVisitor):
    """
    Collect type annotations from a stub module.
    """

    METADATA_DEPENDENCIES = (
        ScopeProvider,
        PositionProvider,
        QualifiedNameProvider,
    )

    annotations: MultiVarAnnotations

    def __init__(
        self,
        existing_imports: Set[str],
        module_imports: Dict[str, ImportItem],
        context: CodemodContext,
        handle_function_bodies: bool = False,
        create_class_attributes: bool = False,
        track_unannotated: bool = False,
    ) -> None:
        super().__init__()
        self.context = context
        # Existing imports, determined by looking at the target module.
        # Used to help us determine when a type in a stub will require new imports.
        #
        # The contents of this are fully-qualified names of types in scope
        # as well as module names, although downstream we effectively ignore
        # the module names as of the current implementation.
        self.existing_imports: Set[str] = existing_imports
        # Module imports, gathered by prescanning the stub file to determine
        # which modules need to be imported directly to qualify their symbols.
        self.module_imports: Dict[str, ImportItem] = module_imports
        # Fields that help us track temporary state as we recurse
        self.qualifier: List[str] = []
        self.current_assign: Optional[cst.Assign] = None  # used to collect typevars
        # Store the annotations.
        self.annotations = MultiVarAnnotations.empty()

        self._cst_annassign_hinting: dict[str, cst.Annotation] = {}

        self.handle_function_bodies = handle_function_bodies
        self.create_class_attributes = create_class_attributes
        self.track_unannotated = track_unannotated

    def visit_ClassDef(
        self,
        node: cst.ClassDef,
    ) -> None:
        self.qualifier.append(node.name.value)
        new_bases = []
        for base in node.bases:
            value = base.value
            if isinstance(value, NAME_OR_ATTRIBUTE):
                new_value = self._handle_NameOrAttribute(value)
            elif isinstance(value, cst.Subscript):
                new_value = self._handle_Subscript(value)
            else:
                start = self.get_metadata(PositionProvider, node).start
                raise ValueError(
                    "Invalid type used as base class in stub file at "
                    + f"{start.line}:{start.column}. Only subscripts, names, and "
                    + "attributes are valid base classes for static typing."
                )
            new_bases.append(base.with_changes(value=new_value))

        if self.create_class_attributes:
            # Match exactly one AnnAssign per line without a value
            matcher = m.SimpleStatementLine(
                body=[m.AnnAssign(target=m.Name(), annotation=m.Annotation(), value=None)]
            )
            hints: list[cst.AnnAssign] = [ssl for ssl in node.body.body if m.matches(ssl, matcher)]

        else:
            hints = []

        self.annotations.class_definitions[node.name.value] = node.with_changes(
            bases=new_bases, body=cst.IndentedBlock(body=hints)
        )

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
    ) -> None:
        self.qualifier.pop()

    def visit_FunctionDef(
        self,
        node: cst.FunctionDef,
    ) -> bool:
        self.qualifier.append(node.name.value)
        returns = node.returns
        return_annotation = (
            self._handle_Annotation(annotation=returns) if returns is not None else None
        )
        parameter_annotations = self._handle_Parameters(node.params)
        name = ".".join(self.qualifier)
        key = FunctionKey.make(name, node.params)
        self.annotations.functions[key] = FunctionAnnotation(
            parameters=parameter_annotations, returns=return_annotation
        )

        # pyi files don't support inner functions, return False to stop the traversal.
        return self.handle_function_bodies

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
    ) -> None:
        self.qualifier.pop()

    # def visit_AnnAssign(
    #    self,
    #    node: cst.AnnAssign,
    # ) -> bool:
    #    return True

    def leave_AnnAssign(
        self,
        original_node: cst.AnnAssign,
    ) -> None:
        return None

    def visit_Assign(
        self,
        node: cst.Assign,
    ) -> None:
        self.current_assign = node
        return self.track_unannotated

    def leave_Assign(
        self,
        original_node: cst.Assign,
    ) -> None:
        self.current_assign = None

    @m.call_if_inside(m.AnnAssign(target=m.Name() | m.Attribute(value=m.Name("self"))))
    def visit_AnnAssign(self, node: cst.AnnAssign):
        if type(self.get_metadata(ScopeProvider, node)) is not ClassScope:
            name = get_full_name_for_node_or_raise(node.target)
            annotation_value = self._handle_Annotation(annotation=node.annotation)

            self.qualifier.append(name)
            full_qual = ".".join(self.qualifier)
            self.qualifier.pop()

            # Hinting is used in an assignment; track
            if node.value is not None:
                self.annotations.attributes[full_qual].append(annotation_value)

                # If hint was given, drop it
                if full_qual in self._cst_annassign_hinting:
                    self._cst_annassign_hinting.pop(full_qual)

            # Otherwise AnnAssign is used purely for hinting; track this,
            # but do not store hint in attributes
            else:
                self._cst_annassign_hinting[full_qual] = annotation_value

    @m.call_if_inside(m.AssignTarget(target=m.Name() | m.Attribute(value=m.Name("self"))))
    def visit_AssignTarget(self, node: cst.AssignTarget) -> bool | None:
        self._visit_unannotated_target(node.target)
        return self.track_unannotated

    @m.call_if_inside(m.AugAssign(target=m.Name() | m.Attribute(value=m.Name("self"))))
    def visit_AugAssign(self, node: cst.AugAssign) -> bool | None:
        self._visit_unannotated_target(node.target)
        return self.track_unannotated

    @m.call_if_inside(m.AssignTarget() | m.AugAssign())
    def visit_Tuple(self, node: cst.Tuple) -> bool | None:
        self._visit_unpackable(node.elements)
        return self.track_unannotated

    @m.call_if_inside(m.AssignTarget() | m.AugAssign())
    def visit_List(self, node: cst.List) -> bool | None:
        self._visit_unpackable(node.elements)
        return self.track_unannotated

    def _visit_unpackable(self, elements: list[cst.BaseElement]) -> bool | None:
        targets = map(lambda e: e.value, elements)
        for target in filter(lambda e: not isinstance(e, (cst.Tuple, cst.List)), targets):
            self._visit_unannotated_target(target)

    def _visit_unannotated_target(self, target: cst.CSTNode) -> bool | None:
        if self.track_unannotated and m.matches(target, m.Name() | m.Attribute(value=m.Name("self"))):
            name = get_full_name_for_node_or_raise(target)

            self.qualifier.append(name)
            fullqual = ".".join(self.qualifier)
            self.qualifier.pop()

            # Consume stored hint if present
            hint = self._cst_annassign_hinting.pop(fullqual, None)
            self.annotations.attributes[fullqual].append(hint)

    @m.call_if_inside(m.Assign())
    @m.visit(m.Call(func=m.Name("TypeVar")))
    def record_typevar(
        self,
        node: cst.Call,
    ) -> None:
        # pyre-ignore current_assign is never None here
        name = get_full_name_for_node(self.current_assign.targets[0].target)
        if name is not None:
            # pyre-ignore current_assign is never None here
            self.annotations.typevars[name] = self.current_assign
            self._handle_qualification_and_should_qualify("typing.TypeVar")
            self.current_assign = None

    def leave_Module(
        self,
        original_node: cst.Module,
    ) -> None:
        self.annotations.finish()

    def _module_and_target(
        self,
        qualified_name: str,
    ) -> Tuple[str, str]:
        relative_prefix = ""
        while qualified_name.startswith("."):
            relative_prefix += "."
            qualified_name = qualified_name[1:]
        split = qualified_name.rsplit(".", 1)
        if len(split) == 1:
            qualifier, target = "", split[0]
        else:
            qualifier, target = split
        return (relative_prefix + qualifier, target)

    def _handle_qualification_and_should_qualify(
        self, qualified_name: str, node: Optional[cst.CSTNode] = None
    ) -> bool:
        """
        Based on a qualified name and the existing module imports, record that
        we need to add an import if necessary and return whether or not we
        should use the qualified name due to a preexisting import.
        """
        module, target = self._module_and_target(qualified_name)
        if module in ("", "builtins"):
            return False
        elif qualified_name not in self.existing_imports:
            if module in self.existing_imports:
                return True
            elif module in self.module_imports:
                m = self.module_imports[module]
                if m.obj_name is None:
                    asname = m.alias
                else:
                    asname = None
                AddImportsVisitor.add_needed_import(self.context, m.module_name, asname=asname)
                return True
            else:
                if node and isinstance(node, cst.Name) and node.value != target:
                    asname = node.value
                else:
                    asname = None
                AddImportsVisitor.add_needed_import(
                    self.context,
                    module,
                    target,
                    asname=asname,
                )
                return False
        return False

    # Handler functions.
    #
    # Each of these does one of two things, possibly recursively, over some
    # valid CST node for a static type:
    #  - process the qualified name and ensure we will add necessary imports
    #  - dequalify the node

    def _handle_NameOrAttribute(
        self,
        node: NameOrAttribute,
    ) -> Union[cst.Name, cst.Attribute]:
        qualified_name = _get_unique_qualified_name(self, node)
        should_qualify = self._handle_qualification_and_should_qualify(qualified_name, node)
        self.annotations.names.add(qualified_name)
        if should_qualify:
            qualified_node = (
                cst.parse_module(qualified_name) if isinstance(node, cst.Name) else node
            )
            return qualified_node  # pyre-ignore[7]
        else:
            dequalified_node = node.attr if isinstance(node, cst.Attribute) else node
            return dequalified_node

    def _handle_Index(
        self,
        slice: cst.Index,
    ) -> cst.Index:
        value = slice.value
        if isinstance(value, cst.Subscript):
            return slice.with_changes(value=self._handle_Subscript(value))
        elif isinstance(value, cst.Attribute):
            return slice.with_changes(value=self._handle_NameOrAttribute(value))
        else:
            if isinstance(value, cst.SimpleString):
                self.annotations.names.add(_get_string_value(value))
            return slice

    def _handle_Subscript(
        self,
        node: cst.Subscript,
    ) -> cst.Subscript:
        value = node.value
        if isinstance(value, NAME_OR_ATTRIBUTE):
            new_node = node.with_changes(value=self._handle_NameOrAttribute(value))
        else:
            raise ValueError("Expected any indexed type to have")
        if _get_unique_qualified_name(self, node) in ("Type", "typing.Type"):
            # Note: we are intentionally not handling qualification of
            # anything inside `Type` because it's common to have nested
            # classes, which we cannot currently distinguish from classes
            # coming from other modules, appear here.
            return new_node
        slice = node.slice
        if isinstance(slice, tuple):
            new_slice = []
            for item in slice:
                value = item.slice.value
                if isinstance(value, NAME_OR_ATTRIBUTE):
                    name = self._handle_NameOrAttribute(item.slice.value)
                    new_index = item.slice.with_changes(value=name)
                    new_slice.append(item.with_changes(slice=new_index))
                else:
                    if isinstance(item.slice, cst.Index):
                        new_index = item.slice.with_changes(value=self._handle_Index(item.slice))
                        item = item.with_changes(slice=new_index)
                    new_slice.append(item)
            return new_node.with_changes(slice=tuple(new_slice))
        elif isinstance(slice, cst.Index):
            new_slice = self._handle_Index(slice)
            return new_node.with_changes(slice=new_slice)
        else:
            return new_node

    def _handle_Annotation(
        self,
        annotation: cst.Annotation,
    ) -> cst.Annotation:
        node = annotation.annotation
        if isinstance(node, cst.SimpleString):
            self.annotations.names.add(_get_string_value(node))
            return annotation
        elif isinstance(node, cst.Subscript):
            return cst.Annotation(annotation=self._handle_Subscript(node))
        elif isinstance(node, NAME_OR_ATTRIBUTE):
            return cst.Annotation(annotation=self._handle_NameOrAttribute(node))
        else:
            raise ValueError(f"Unexpected annotation node: {node}")

    def _handle_Parameters(
        self,
        parameters: cst.Parameters,
    ) -> cst.Parameters:
        def update_annotations(
            parameters: Sequence[cst.Param],
        ) -> List[cst.Param]:
            updated_parameters = []
            for parameter in list(parameters):
                annotation = parameter.annotation
                if annotation is not None:
                    parameter = parameter.with_changes(
                        annotation=self._handle_Annotation(annotation=annotation)
                    )
                updated_parameters.append(parameter)
            return updated_parameters

        return parameters.with_changes(params=update_annotations(parameters.params))
