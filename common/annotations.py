# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import functools

import libcst
import libcst.matchers as m
from libcst import metadata

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

from common.visitors import (
    HintableDeclarationVisitor,
    HintableParameterVisitor,
    HintableReturnVisitor,
    ScopeAwareVisitor,
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
    attributes: collections.defaultdict[str, list[libcst.Annotation | None]]
    class_definitions: Dict[str, libcst.ClassDef]
    typevars: Dict[str, libcst.Assign]
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


class MultiVarTypeCollector(
    HintableDeclarationVisitor, HintableParameterVisitor, HintableReturnVisitor, ScopeAwareVisitor
):
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
        self.current_assign: Optional[libcst.Assign] = None  # used to collect typevars
        # Store the annotations.
        self.annotations = MultiVarAnnotations.empty()

        self._cst_annassign_hinting: dict[str, libcst.Annotation] = {}

    def annotated_function(self, function: libcst.FunctionDef, _: libcst.Annotation) -> None:
        self._function(function)

    def unannotated_function(self, function: libcst.FunctionDef) -> None:
        self._function(function)

    def _function(
        self,
        node: libcst.FunctionDef,
    ) -> bool:
        returns = node.returns
        return_annotation = (
            self._handle_Annotation(annotation=returns) if returns is not None else None
        )
        parameter_annotations = self._handle_Parameters(node.params)
        name = self.qualified_name(node.name.value)
        key = FunctionKey.make(name, node.params)
        self.annotations.functions[key] = FunctionAnnotation(
            parameters=parameter_annotations, returns=return_annotation
        )

    # no-op: handle parameters and function in one go
    def annotated_param(self, _1: libcst.Param, _2: libcst.Annotation) -> None:
        ...

    def unannotated_param(self, _: libcst.Param) -> None:
        ...

    def visit_Assign(
        self,
        node: libcst.Assign,
    ) -> None:
        self.current_assign = node

    def leave_Assign(
        self,
        original_node: libcst.Assign,
    ) -> None:
        self.current_assign = None

    def annotated_assignment(
        self, target: libcst.Name | libcst.Attribute, annotation: libcst.Annotation
    ) -> None:
        full_qual = self.qualified_name(target)
        annotation_value = self._handle_Annotation(annotation=annotation)

        # Track annotation
        self.annotations.attributes[full_qual].append(annotation_value)

        # Delete hinting if previously given
        self._cst_annassign_hinting.pop(full_qual, None)

        # NOTE: Debatedly, propagation can be achieved using this
        # NOTE: However, we expect tools to annotate "aggressively"
        # NOTE: i.e. to annotate as many symbols as they can

        # Track hint for unannotated variables used thereafter
        # = propagation of type hint through scope
        # self._cst_annassign_hinting[full_qual] = annotation_value

        # # Hinting is used in an assignment; track
        # if node.value is not None:
        #     self.annotations.attributes[full_qual].append(annotation_value)

        #     # If hint was given, drop it
        #     if full_qual in self._cst_annassign_hinting:
        #         self._cst_annassign_hinting.pop(full_qual)

        # # Otherwise AnnAssign is used purely for hinting; track this,
        # # but do not store hint in attributes
        # else:
        #     self._cst_annassign_hinting[full_qual] = annotation_value

    def annotated_hint(
        self, target: libcst.Name | libcst.Attribute, annotation: libcst.Annotation
    ) -> None:
        full_qual = self.qualified_name(target)
        annotation_value = self._handle_Annotation(annotation=annotation)

        # Track hint for unannotated variables used thereafter
        # = propagation of type hint through scope
        self._cst_annassign_hinting[full_qual] = annotation_value

    def instance_attribute_hint(
        self, target: libcst.Name, annotation: libcst.Annotation | None
    ) -> None:
        if annotation is not None:
            annotation_value = self._handle_Annotation(annotation=annotation)
        else:
            annotation_value = None

        # Mark as an instance attribute
        scope = self.qualified_scope()
        key = ".".join(scope)
        *_, classname = scope

        classdef = self.annotations.class_definitions.get(
            key, libcst.ClassDef(name=libcst.Name(classname), body=libcst.IndentedBlock(body=[]))
        )

        if annotation_value is not None:
            classdef.body.body.append(
                libcst.AnnAssign(
                    target=target,
                    annotation=annotation_value,
                    value=None,
                )
            )
        else:
            classdef.body.body.append(
                libcst.Assign(targets=[libcst.AssignTarget(target=target)], value=libcst.Ellipsis())
            )

        self.annotations.class_definitions[key] = classdef

    def unannotated_target(self, target: libcst.Name | libcst.Attribute) -> None:
        # Reference stored hint if present
        full_qual = self.qualified_name(target)
        hint = self._cst_annassign_hinting.get(full_qual, None)

        self.annotations.attributes[full_qual].append(hint)

    # no-op, as these cannot ever be annotated
    def scope_overwritten_target(self, target: libcst.Name) -> None:
        ...

    @m.call_if_inside(m.Assign())
    @m.visit(m.Call(func=m.Name("TypeVar")))
    def record_typevar(
        self,
        node: libcst.Call,
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
        original_node: libcst.Module,
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
        self, qualified_name: str, node: Optional[libcst.CSTNode] = None
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
                if node and isinstance(node, libcst.Name) and node.value != target:
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
    ) -> Union[libcst.Name, libcst.Attribute]:
        qualified_name = _get_unique_qualified_name(self, node)
        should_qualify = self._handle_qualification_and_should_qualify(qualified_name, node)
        self.annotations.names.add(qualified_name)
        if should_qualify:
            qualified_node = (
                libcst.parse_module(qualified_name) if isinstance(node, libcst.Name) else node
            )
            return qualified_node  # pyre-ignore[7]
        else:
            dequalified_node = node.attr if isinstance(node, libcst.Attribute) else node
            return dequalified_node

    def _handle_Index(
        self,
        slice: libcst.Index,
    ) -> libcst.Index:
        value = slice.value
        if isinstance(value, libcst.Subscript):
            return slice.with_changes(value=self._handle_Subscript(value))
        elif isinstance(value, libcst.Attribute):
            return slice.with_changes(value=self._handle_NameOrAttribute(value))
        else:
            if isinstance(value, libcst.SimpleString):
                self.annotations.names.add(_get_string_value(value))
            return slice

    def _handle_Subscript(
        self,
        node: libcst.Subscript,
    ) -> libcst.Subscript:
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
                    if isinstance(item.slice, libcst.Index):
                        new_index = item.slice.with_changes(value=self._handle_Index(item.slice))
                        item = item.with_changes(slice=new_index)
                    new_slice.append(item)
            return new_node.with_changes(slice=tuple(new_slice))
        elif isinstance(slice, libcst.Index):
            new_slice = self._handle_Index(slice)
            return new_node.with_changes(slice=new_slice)
        else:
            return new_node

    def _handle_Annotation(
        self,
        annotation: libcst.Annotation,
    ) -> libcst.Annotation:
        node = annotation.annotation
        if isinstance(node, libcst.SimpleString):
            self.annotations.names.add(_get_string_value(node))
            return annotation
        elif isinstance(node, libcst.Subscript):
            return libcst.Annotation(annotation=self._handle_Subscript(node))
        elif isinstance(node, NAME_OR_ATTRIBUTE):
            return libcst.Annotation(annotation=self._handle_NameOrAttribute(node))
        elif isinstance(node, libcst.BinaryOperation):
            return libcst.Annotation(
                annotation=libcst.BinaryOperation(
                    left=self._handle_Annotation(libcst.Annotation(node.left)).annotation,
                    operator=node.operator,
                    right=self._handle_Annotation(libcst.Annotation(node.right)).annotation,
                )
            )
        else:
            raise ValueError(f"Unexpected annotation node: {node}")

    def _handle_Parameters(
        self,
        parameters: libcst.Parameters,
    ) -> libcst.Parameters:
        def update_annotations(
            parameters: Sequence[libcst.Param],
        ) -> List[libcst.Param]:
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


from libcst.codemod.visitors._apply_type_annotations import Annotations


class ApplyTypeAnnotationsVisitor(ContextAwareTransformer):
    """
    Apply type annotations to a source module using the given stub mdules.
    You can also pass in explicit annotations for functions and attributes and
    pass in new class definitions that need to be added to the source module.

    This is one of the transforms that is available automatically to you when
    running a codemod. To use it in this manner, import
    :class:`~libcst.codemod.visitors.ApplyTypeAnnotationsVisitor` and then call
    the static
    :meth:`~libcst.codemod.visitors.ApplyTypeAnnotationsVisitor.store_stub_in_context`
    method, giving it the current context (found as ``self.context`` for all
    subclasses of :class:`~libcst.codemod.Codemod`), the stub module from which
    you wish to add annotations.

    For example, you can store the type annotation ``int`` for ``x`` using::

        stub_module = parse_module("x: int = ...")

        ApplyTypeAnnotationsVisitor.store_stub_in_context(self.context, stub_module)

    You can apply the type annotation using::

        source_module = parse_module("x = 1")
        ApplyTypeAnnotationsVisitor.transform_module(source_module)

    This will produce the following code::

        x: int = 1

    If the function or attribute already has a type annotation, it will not be
    overwritten.

    To overwrite existing annotations when applying annotations from a stub,
    use the keyword argument ``overwrite_existing_annotations=True`` when
    constructing the codemod or when calling ``store_stub_in_context``.
    """

    CONTEXT_KEY = "ApplyTypeAnnotationsVisitor"

    def __init__(
        self,
        context: CodemodContext,
        annotations: Optional[Annotations] = None,
        overwrite_existing_annotations: bool = False,
        use_future_annotations: bool = False,
        strict_posargs_matching: bool = True,
        strict_annotation_matching: bool = False,
        always_qualify_annotations: bool = False,
        handle_function_bodies: bool = False,
        create_class_attributes: bool = False,
    ) -> None:
        super().__init__(context)
        # Qualifier for storing the canonical name of the current function.
        self.qualifier: List[str] = []
        self.annotations: Annotations = Annotations.empty() if annotations is None else annotations
        self.toplevel_annotations: Dict[str, libcst.Annotation] = {}
        self.visited_classes: Set[str] = set()
        self.overwrite_existing_annotations = overwrite_existing_annotations
        self.use_future_annotations = use_future_annotations
        self.strict_posargs_matching = strict_posargs_matching
        self.strict_annotation_matching = strict_annotation_matching
        self.always_qualify_annotations = always_qualify_annotations
        self.handle_function_bodies = handle_function_bodies
        self.create_class_attributes = create_class_attributes

        # We use this to determine the end of the import block so that we can
        # insert top-level annotations.
        self.import_statements: List[libcst.ImportFrom] = []

        # We use this to report annotations added, as well as to determine
        # whether to abandon the codemod in edge cases where we may have
        # only made changes to the imports.
        self.annotation_counts: AnnotationCounts = AnnotationCounts()

        # We use this to collect typevars, to avoid importing existing ones from the pyi file
        self.current_assign: Optional[libcst.Assign] = None
        self.typevars: Dict[str, libcst.Assign] = {}

        # Global variables and classes defined on the toplevel of the target module.
        # Used to help determine which names we need to check are in scope, and add
        # quotations to avoid undefined forward references in type annotations.
        self.global_names: Set[str] = set()

    @staticmethod
    def store_stub_in_context(
        context: CodemodContext,
        stub: libcst.Module,
        overwrite_existing_annotations: bool = False,
        use_future_annotations: bool = False,
        strict_posargs_matching: bool = True,
        strict_annotation_matching: bool = False,
        always_qualify_annotations: bool = False,
    ) -> None:
        """
        Store a stub module in the :class:`~libcst.codemod.CodemodContext` so
        that type annotations from the stub can be applied in a later
        invocation of this class.

        If the ``overwrite_existing_annotations`` flag is ``True``, the
        codemod will overwrite any existing annotations.

        If you call this function multiple times, only the last values of
        ``stub`` and ``overwrite_existing_annotations`` will take effect.
        """
        context.scratch[ApplyTypeAnnotationsVisitor.CONTEXT_KEY] = (
            stub,
            overwrite_existing_annotations,
            use_future_annotations,
            strict_posargs_matching,
            strict_annotation_matching,
            always_qualify_annotations,
        )

    def transform_module_impl(
        self,
        tree: libcst.Module,
    ) -> libcst.Module:
        """
        Collect type annotations from all stubs and apply them to ``tree``.

        Gather existing imports from ``tree`` so that we don't add duplicate imports.

        Gather global names from ``tree`` so forward references are quoted.
        """
        import_gatherer = GatherImportsVisitor(CodemodContext())
        tree.visit(import_gatherer)
        existing_import_names = _get_imported_names(import_gatherer.all_imports)

        global_names_gatherer = GatherGlobalNamesVisitor(CodemodContext())
        tree.visit(global_names_gatherer)
        self.global_names = global_names_gatherer.global_names.union(
            global_names_gatherer.class_names
        )

        context_contents = self.context.scratch.get(ApplyTypeAnnotationsVisitor.CONTEXT_KEY)
        if context_contents is not None:
            (
                stub,
                overwrite_existing_annotations,
                use_future_annotations,
                strict_posargs_matching,
                strict_annotation_matching,
                always_qualify_annotations,
            ) = context_contents
            self.overwrite_existing_annotations = (
                self.overwrite_existing_annotations or overwrite_existing_annotations
            )
            self.use_future_annotations = self.use_future_annotations or use_future_annotations
            self.strict_posargs_matching = self.strict_posargs_matching and strict_posargs_matching
            self.strict_annotation_matching = (
                self.strict_annotation_matching or strict_annotation_matching
            )
            self.always_qualify_annotations = (
                self.always_qualify_annotations or always_qualify_annotations
            )
            module_imports = self._get_module_imports(stub, import_gatherer)
            visitor = TypeCollector(
                existing_import_names,
                module_imports,
                self.context,
            )
            libcst.MetadataWrapper(stub).visit(visitor)
            self.annotations.update(visitor.annotations)

        if self.use_future_annotations:
            AddImportsVisitor.add_needed_import(self.context, "__future__", "annotations")

        tree_with_imports = AddImportsVisitor(self.context).transform_module(tree)
        tree_with_changes = tree_with_imports.visit(self)

        # don't modify the imports if we didn't actually add any type information
        if self.annotation_counts.any_changes_applied():
            return tree_with_changes
        else:
            return tree

    # helpers for collecting type information from the stub files

    def _get_module_imports(  # noqa: C901: too complex
        self, stub: libcst.Module, existing_import_gatherer: GatherImportsVisitor
    ) -> Dict[str, ImportItem]:
        """Returns a dict of modules that need to be imported to qualify symbols."""
        # We correlate all imported symbols, e.g. foo.bar.Baz, with a list of module
        # and from imports. If the same unqualified symbol is used from different
        # modules, we give preference to an explicit from-import if any, and qualify
        # everything else by importing the module.
        #
        # e.g. the following stub:
        #   import foo as quux
        #   from bar import Baz as X
        #   def f(x: X) -> quux.X: ...
        # will return {'foo': ImportItem("foo", "quux")}. When the apply type
        # annotation visitor hits `quux.X` it will retrieve the canonical name
        # `foo.X` and then note that `foo` is in the module imports map, so it will
        # leave the symbol qualified.
        import_gatherer = GatherImportsVisitor(CodemodContext())
        stub.visit(import_gatherer)
        symbol_map = import_gatherer.symbol_mapping
        existing_import_names = _get_imported_names(existing_import_gatherer.all_imports)
        symbol_collector = ImportedSymbolCollector(existing_import_names, self.context)
        libcst.MetadataWrapper(stub).visit(symbol_collector)
        module_imports = {}
        for sym, imported_symbols in symbol_collector.imported_symbols.items():
            existing = existing_import_gatherer.symbol_mapping.get(sym)
            if existing and any(s.module_name != existing.module_name for s in imported_symbols):
                # If a symbol is imported in the main file, we have to qualify
                # it when imported from a different module in the stub file.
                used = True
            elif len(imported_symbols) == 1 and not self.always_qualify_annotations:
                # If we have a single use of a new symbol we can from-import it
                continue
            else:
                # There are multiple occurrences in the stub file and none in
                # the main file. At least one can be from-imported.
                used = False
            for imp_sym in imported_symbols:
                if not imp_sym.symbol:
                    continue
                imp = symbol_map.get(imp_sym.symbol)
                if self.always_qualify_annotations and sym not in existing_import_names:
                    # Override 'always qualify' if this is a typing import, or
                    # the main file explicitly from-imports a symbol.
                    if imp and imp.module_name != "typing":
                        module_imports[imp.module_name] = imp
                    else:
                        imp = symbol_map.get(imp_sym.module_symbol)
                        if imp:
                            module_imports[imp.module_name] = imp
                elif not used and imp and imp.module_name == imp_sym.module_name:
                    # We can only import a symbol directly once.
                    used = True
                elif sym in existing_import_names:
                    if imp:
                        module_imports[imp.module_name] = imp
                else:
                    imp = symbol_map.get(imp_sym.module_symbol)
                    if imp:
                        # imp will be None in corner cases like
                        #   import foo.bar as Baz
                        #   x: Baz
                        # which is technically valid python but nonsensical as a
                        # type annotation. Dropping it on the floor for now.
                        module_imports[imp.module_name] = imp
        return module_imports

    # helpers for processing annotation nodes
    def _quote_future_annotations(self, annotation: libcst.Annotation) -> libcst.Annotation:
        # TODO: We probably want to make sure references to classes defined in the current
        # module come to us fully qualified - so we can do the dequalification here and
        # know to look for what is in-scope without also catching builtins like "None" in the
        # quoting. This should probably also be extended to handle what imports are in scope,
        # as well as subscriptable types.
        # Note: We are collecting all imports and passing this to the type collector grabbing
        # annotations from the stub file; should consolidate import handling somewhere too.
        node = annotation.annotation
        if (
            isinstance(node, libcst.Name)
            and (node.value in self.global_names)
            and not (node.value in self.visited_classes)
        ):
            return annotation.with_changes(annotation=libcst.SimpleString(value=f'"{node.value}"'))
        return annotation

    # smart constructors: all applied annotations happen via one of these

    def _apply_annotation_to_attribute_or_global(
        self,
        name: str,
        annotation: libcst.Annotation,
        value: Optional[libcst.BaseExpression],
    ) -> libcst.AnnAssign:
        if len(self.qualifier) == 0:
            self.annotation_counts.global_annotations += 1
        else:
            self.annotation_counts.attribute_annotations += 1

        return libcst.AnnAssign(
            libcst.parse_expression(name),
            self._quote_future_annotations(annotation),
            value,
        )

    def _apply_annotation_to_parameter(
        self,
        parameter: libcst.Param,
        annotation: libcst.Annotation,
    ) -> libcst.Param:
        self.annotation_counts.parameter_annotations += 1
        return parameter.with_changes(
            annotation=self._quote_future_annotations(annotation),
        )

    def _apply_annotation_to_return(
        self,
        function_def: libcst.FunctionDef,
        annotation: libcst.Annotation,
    ) -> libcst.FunctionDef:
        self.annotation_counts.return_annotations += 1
        return function_def.with_changes(
            returns=self._quote_future_annotations(annotation),
        )

    def _add_annotated_attr_to_class_body(
        self,
        body: libcst.BaseSuite,
        hint: libcst.AnnAssign,
    ) -> libcst.BaseSuite:
        for member in body.body:
            if m.matches(
                member,
                m.AnnAssign(value=m.Name(hint.target.value)),
            ):
                if self.overwrite_existing_annotations:
                    member = member.with_changes(annotation=hint.annotation)
                    self.annotation_counts.attribute_annotations += 1
                return body

        self.annotation_counts.attribute_annotations += 1
        return body.with_changes(body=(libcst.SimpleStatementLine(body=[hint]), *body.body))

    # private methods used in the visit and leave methods

    def _qualifier_name(self) -> str:
        return ".".join(self.qualifier)

    def _annotate_single_target(
        self,
        node: libcst.Assign,
        updated_node: libcst.Assign,
    ) -> Union[libcst.Assign, libcst.AnnAssign]:
        only_target = node.targets[0].target
        if isinstance(only_target, (libcst.Tuple, libcst.List)):
            outside_annos: list[libcst.AnnAssign] = []
            for element in only_target.elements:
                value = element.value
                name = get_full_name_for_node(value)
                if name is not None and name != "_":

                    self.qualifier.append(name)
                    if self._qualifier_name() in self.annotations.attributes:
                        annotation = self.annotations.attributes[self._qualifier_name()]
                        outside_annos.append(
                            libcst.AnnAssign(
                                target=libcst.Name(name), annotation=annotation, value=None
                            )
                        )
                    self.qualifier.pop()

            updated_node = libcst.FlattenSentinel((*outside_annos, updated_node))

        elif isinstance(only_target, (libcst.Subscript)):
            pass
        else:
            name = get_full_name_for_node(only_target)
            if name is not None:
                self.qualifier.append(name)
                if self._qualifier_name() in self.annotations.attributes and not isinstance(
                    only_target, libcst.Subscript
                ):
                    annotation = self.annotations.attributes[self._qualifier_name()]
                    self.qualifier.pop()
                    return self._apply_annotation_to_attribute_or_global(
                        name=name,
                        annotation=annotation,
                        value=node.value,
                    )
                else:
                    self.qualifier.pop()
        return updated_node

    def _split_module(
        self,
        module: libcst.Module,
        updated_module: libcst.Module,
    ) -> Tuple[
        List[Union[libcst.SimpleStatementLine, libcst.BaseCompoundStatement]],
        List[Union[libcst.SimpleStatementLine, libcst.BaseCompoundStatement]],
    ]:
        import_add_location = 0
        # This works under the principle that while we might modify node contents,
        # we have yet to modify the number of statements. So we can match on the
        # original tree but break up the statements of the modified tree. If we
        # change this assumption in this visitor, we will have to change this code.
        for i, statement in enumerate(module.body):
            if isinstance(statement, libcst.SimpleStatementLine):
                for possible_import in statement.body:
                    for last_import in self.import_statements:
                        if possible_import is last_import:
                            import_add_location = i + 1
                            break

        return (
            list(updated_module.body[:import_add_location]),
            list(updated_module.body[import_add_location:]),
        )

    def _update_parameters(
        self,
        annotations: FunctionAnnotation,
        updated_node: libcst.FunctionDef,
    ) -> libcst.Parameters:
        # Update params and default params with annotations
        # Don't override existing annotations or default values unless asked
        # to overwrite existing annotations.
        def update_annotation(
            parameters: Sequence[libcst.Param],
            annotations: Sequence[libcst.Param],
            positional: bool,
        ) -> List[libcst.Param]:
            parameter_annotations = {}
            annotated_parameters = []
            positional = positional and not self.strict_posargs_matching
            for i, parameter in enumerate(annotations):
                key = i if positional else parameter.name.value
                if parameter.annotation:
                    parameter_annotations[key] = parameter.annotation.with_changes(
                        whitespace_before_indicator=libcst.SimpleWhitespace(value="")
                    )
            for i, parameter in enumerate(parameters):
                key = i if positional else parameter.name.value
                if key in parameter_annotations and (
                    self.overwrite_existing_annotations or not parameter.annotation
                ):
                    parameter = self._apply_annotation_to_parameter(
                        parameter=parameter,
                        annotation=parameter_annotations[key],
                    )
                annotated_parameters.append(parameter)
            return annotated_parameters

        return updated_node.params.with_changes(
            params=update_annotation(
                updated_node.params.params,
                annotations.parameters.params,
                positional=True,
            ),
            kwonly_params=update_annotation(
                updated_node.params.kwonly_params,
                annotations.parameters.kwonly_params,
                positional=False,
            ),
            posonly_params=update_annotation(
                updated_node.params.posonly_params,
                annotations.parameters.posonly_params,
                positional=True,
            ),
        )

    def _insert_empty_line(
        self,
        statements: List[Union[libcst.SimpleStatementLine, libcst.BaseCompoundStatement]],
    ) -> List[Union[libcst.SimpleStatementLine, libcst.BaseCompoundStatement]]:
        if len(statements) < 1:
            # No statements, nothing to add to
            return statements
        if len(statements[0].leading_lines) == 0:
            # Statement has no leading lines, add one!
            return [
                statements[0].with_changes(leading_lines=(libcst.EmptyLine(),)),
                *statements[1:],
            ]
        if statements[0].leading_lines[0].comment is None:
            # First line is empty, so its safe to leave as-is
            return statements
        # Statement has a comment first line, so lets add one more empty line
        return [
            statements[0].with_changes(
                leading_lines=(libcst.EmptyLine(), *statements[0].leading_lines)
            ),
            *statements[1:],
        ]

    def _match_signatures(  # noqa: C901: Too complex
        self,
        function: libcst.FunctionDef,
        annotations: FunctionAnnotation,
    ) -> bool:
        """Check that function annotations on both signatures are compatible."""

        def compatible(
            p: Optional[libcst.Annotation],
            q: Optional[libcst.Annotation],
        ) -> bool:
            if (
                self.overwrite_existing_annotations
                or not _is_non_sentinel(p)
                or not _is_non_sentinel(q)
            ):
                return True
            if not self.strict_annotation_matching:
                # We will not overwrite clashing annotations, but the signature as a
                # whole will be marked compatible so that holes can be filled in.
                return True
            return p.annotation.deep_equals(q.annotation)  # pyre-ignore[16]

        def match_posargs(
            ps: Sequence[libcst.Param],
            qs: Sequence[libcst.Param],
        ) -> bool:
            if len(ps) != len(qs):
                return False
            for p, q in zip(ps, qs):
                if self.strict_posargs_matching and not p.name.value == q.name.value:
                    return False
                if not compatible(p.annotation, q.annotation):
                    return False
            return True

        def match_kwargs(
            ps: Sequence[libcst.Param],
            qs: Sequence[libcst.Param],
        ) -> bool:
            ps_dict = {x.name.value: x for x in ps}
            qs_dict = {x.name.value: x for x in qs}
            if set(ps_dict.keys()) != set(qs_dict.keys()):
                return False
            for k in ps_dict.keys():
                if not compatible(ps_dict[k].annotation, qs_dict[k].annotation):
                    return False
            return True

        def match_star(
            p: StarParamType,
            q: StarParamType,
        ) -> bool:
            return _is_non_sentinel(p) == _is_non_sentinel(q)

        def match_params(
            f: libcst.FunctionDef,
            g: FunctionAnnotation,
        ) -> bool:
            p, q = f.params, g.parameters
            return (
                match_posargs(p.params, q.params)
                and match_posargs(p.posonly_params, q.posonly_params)
                and match_kwargs(p.kwonly_params, q.kwonly_params)
                and match_star(p.star_arg, q.star_arg)
                and match_star(p.star_kwarg, q.star_kwarg)
            )

        def match_return(
            f: libcst.FunctionDef,
            g: FunctionAnnotation,
        ) -> bool:
            return compatible(f.returns, g.returns)

        return match_params(function, annotations) and match_return(function, annotations)

    # transform API methods

    def visit_ClassDef(
        self,
        node: libcst.ClassDef,
    ) -> None:
        self.qualifier.append(node.name.value)

    def leave_ClassDef(
        self,
        original_node: libcst.ClassDef,
        updated_node: libcst.ClassDef,
    ) -> libcst.ClassDef:
        self.visited_classes.add(original_node.name.value)
        cls_name = ".".join(self.qualifier)
        self.qualifier.pop()
        definition = self.annotations.class_definitions.get(cls_name)
        if definition:
            b1 = _find_generic_base(definition)
            b2 = _find_generic_base(updated_node)
            if b1 and not b2:
                new_bases = list(updated_node.bases) + [b1]
                self.annotation_counts.typevars_and_generics_added += 1
                updated_node = updated_node.with_changes(bases=new_bases)

            if self.create_class_attributes:
                hint_matcher = m.AnnAssign(target=m.Name())
                flt_stmts = (b for body in definition.body.body for b in body.body)
                attr_hints = list(
                    filter(
                        lambda attr: m.matches(attr, hint_matcher),
                        flt_stmts,
                    )
                )

                updated_node = updated_node.with_changes(
                    body=functools.reduce(
                        self._add_annotated_attr_to_class_body,
                        attr_hints,
                        updated_node.body,
                    )
                )

        return updated_node

    def visit_FunctionDef(
        self,
        node: libcst.FunctionDef,
    ) -> bool:
        self.qualifier.append(node.name.value)
        return self.handle_function_bodies

    def leave_FunctionDef(
        self,
        original_node: libcst.FunctionDef,
        updated_node: libcst.FunctionDef,
    ) -> libcst.FunctionDef:
        key = FunctionKey.make(self._qualifier_name(), updated_node.params)
        self.qualifier.pop()
        if key in self.annotations.functions:
            function_annotation = self.annotations.functions[key]
            # Only add new annotation if:
            # * we have matching function signatures and
            # * we are explicitly told to overwrite existing annotations or
            # * there is no existing annotation
            if not self._match_signatures(updated_node, function_annotation):
                return updated_node
            set_return_annotation = (
                self.overwrite_existing_annotations or updated_node.returns is None
            )
            if set_return_annotation and function_annotation.returns is not None:
                updated_node = self._apply_annotation_to_return(
                    function_def=updated_node,
                    annotation=function_annotation.returns,
                )
            # Don't override default values when annotating functions
            new_parameters = self._update_parameters(function_annotation, updated_node)
            return updated_node.with_changes(params=new_parameters)
        return updated_node

    def visit_Assign(
        self,
        node: libcst.Assign,
    ) -> None:
        self.current_assign = node

    @m.call_if_inside(m.Assign())
    @m.visit(m.Call(func=m.Name("TypeVar")))
    def record_typevar(
        self,
        node: libcst.Call,
    ) -> None:
        # pyre-ignore current_assign is never None here
        name = get_full_name_for_node(self.current_assign.targets[0].target)
        if name is not None:
            # Preserve the whole node, even though we currently just use the
            # name, so that we can match bounds and variance at some point and
            # determine if two typevars with the same name are indeed the same.

            # pyre-ignore current_assign is never None here
            self.typevars[name] = self.current_assign
            self.current_assign = None

    def leave_AugAssign(
        self, original_node: libcst.AugAssign, updated_node: libcst.AugAssign
    ) -> libcst.FlattenSentinel[libcst.BaseSmallStatement]:

        outside_annos: list[libcst.AnnAssign] = []
        target = updated_node.target

        if isinstance(target, (libcst.Name, libcst.Attribute)):
            name = get_full_name_for_node(target)

            if name is not None and name != "_":
                # Add separate top-level annotations for `a = b = 1`
                # as `a: int` and `b: int`.
                self.qualifier.append(name)

                if self._qualifier_name() in self.annotations.attributes:
                    annotation = self.annotations.attributes[self._qualifier_name()]
                    outside_annos.append(
                        libcst.AnnAssign(
                            target=libcst.Name(name),
                            annotation=annotation,
                            value=None,
                        )
                    )
                self.qualifier.pop()
        return libcst.FlattenSentinel((*outside_annos, updated_node))

    def leave_Assign(
        self,
        original_node: libcst.Assign,
        updated_node: libcst.Assign,
    ) -> Union[libcst.Assign, libcst.AnnAssign, libcst.FlattenSentinel[libcst.BaseSmallStatement]]:

        self.current_assign = None

        if len(original_node.targets) > 1:
            outside_annos: list[libcst.AnnAssign] = []
            for assign in original_node.targets:
                target = assign.target
                if isinstance(target, (libcst.Name, libcst.Attribute)):
                    name = get_full_name_for_node(target)
                    if name is not None and name != "_":
                        # Add separate top-level annotations for `a = b = 1`
                        # as `a: int` and `b: int`.
                        self.qualifier.append(name)
                        if self._qualifier_name() in self.annotations.attributes:
                            annotation = self.annotations.attributes[self._qualifier_name()]
                            outside_annos.append(
                                libcst.AnnAssign(
                                    target=libcst.Name(name),
                                    annotation=annotation,
                                    value=None,
                                )
                            )
                        self.qualifier.pop()
            return libcst.FlattenSentinel((*outside_annos, updated_node))
        else:
            return self._annotate_single_target(original_node, updated_node)

    def leave_ImportFrom(
        self,
        original_node: libcst.ImportFrom,
        updated_node: libcst.ImportFrom,
    ) -> libcst.ImportFrom:
        self.import_statements.append(original_node)
        return updated_node

    def leave_Module(
        self,
        original_node: libcst.Module,
        updated_node: libcst.Module,
    ) -> libcst.Module:
        fresh_class_definitions = [
            definition
            for name, definition in self.annotations.class_definitions.items()
            if name not in self.visited_classes
        ]

        # NOTE: The entire change will also be abandoned if
        # self.annotation_counts is all 0s, so if adding any new category make
        # sure to record it there.
        if not (self.toplevel_annotations or fresh_class_definitions or self.annotations.typevars):
            return updated_node

        toplevel_statements = []
        # First, find the insertion point for imports
        statements_before_imports, statements_after_imports = self._split_module(
            original_node, updated_node
        )

        # Make sure there's at least one empty line before the first non-import
        statements_after_imports = self._insert_empty_line(statements_after_imports)

        for name, annotation in self.toplevel_annotations.items():
            annotated_assign = self._apply_annotation_to_attribute_or_global(
                name=name,
                annotation=annotation,
                value=None,
            )
            toplevel_statements.append(libcst.SimpleStatementLine([annotated_assign]))

        # TypeVar definitions could be scattered through the file, so do not
        # attempt to put new ones with existing ones, just add them at the top.
        typevars = {k: v for k, v in self.annotations.typevars.items() if k not in self.typevars}
        if typevars:
            for var, stmt in typevars.items():
                toplevel_statements.append(libcst.Newline())
                toplevel_statements.append(stmt)
                self.annotation_counts.typevars_and_generics_added += 1
            toplevel_statements.append(libcst.Newline())

        self.annotation_counts.classes_added = len(fresh_class_definitions)
        toplevel_statements.extend(fresh_class_definitions)

        return updated_node.with_changes(
            body=[
                *statements_before_imports,
                *toplevel_statements,
                *statements_after_imports,
            ]
        )


class TypeAnnotationRemover(libcst.CSTTransformer):
    """
    Configurable type annotation removal.
    Based on LibSA4Py implementation
    """

    def __init__(self, variables: bool = True, parameters: bool = True, rets: bool = True) -> None:
        super().__init__()

        self.variables = variables
        self.parameters = parameters
        self.rets = rets

    def leave_FunctionDef(
        self, original_node: libcst.FunctionDef, updated_node: libcst.FunctionDef
    ) -> Union[libcst.BaseStatement, libcst.RemovalSentinel]:
        if not self.rets:
            return updated_node
        return (
            updated_node.with_changes(returns=None)
            if original_node.returns is not None
            else updated_node
        )

    def leave_Param(
        self, original_node: libcst.Param, updated_node: libcst.Param
    ) -> Union[libcst.Param, libcst.MaybeSentinel, libcst.RemovalSentinel]:
        if not self.parameters:
            return updated_node
        return (
            updated_node.with_changes(annotation=None)
            if original_node.annotation is not None
            else updated_node
        )

    def leave_AnnAssign(
        self, original_node: libcst.AnnAssign, updated_node: libcst.AnnAssign
    ) -> Union[libcst.BaseSmallStatement, libcst.RemovalSentinel]:
        if not self.variables:
            return updated_node

        # Remove hinting like 'a: int' and 'self.foo: str' if outside of a class' body;
        # If it is a hint in a class, i.e. an INSTANCE_ATTR, ; replace it by 'a = ...'
        if m.matches(
            original_node,
            m.AnnAssign(
                target=m.Name() | m.Attribute(value=m.Name("self"), attr=m.Name()),
                annotation=m.Annotation(),
                value=None,
            ),
        ):
            if isinstance(
                self.get_metadata(metadata.ScopeProvider, updated_node), metadata.ClassScope
            ):
                updated_node = libcst.Assign(
                    targets=[libcst.AssignTarget(target=original_node.target)],
                    value=libcst.Ellipsis(),
                )
            else:
                updated_node = libcst.RemoveFromParent()
        else:
            updated_node = libcst.Assign(
                targets=[libcst.AssignTarget(target=original_node.target)],
                value=original_node.value,
            )
        return updated_node
