# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import libcst
from libcst import codemod as c, matchers as m, helpers as h
from libcst.codemod.visitors._add_imports import AddImportsVisitor
from libcst.codemod.visitors._apply_type_annotations import (
    NameOrAttribute,
    NAME_OR_ATTRIBUTE,
    StarParamType,
    _get_imported_names,
    _is_non_sentinel,
    _get_string_value,
    FunctionKey,
    FunctionAnnotation,
    ImportedSymbolCollector,
    TypeCollector,
)
from libcst.codemod.visitors._gather_global_names import GatherGlobalNamesVisitor
from libcst.codemod.visitors._gather_imports import GatherImportsVisitor
from libcst.codemod.visitors._imports import ImportItem
from libcst.helpers import get_full_name_for_node
from libcst.metadata import (
    PositionProvider,
    ScopeProvider,
    FullyQualifiedNameProvider,
    ClassScope,
)

from scripts.common import transformers as t
from scripts.common.metadata.anno4inst import Annotation4InstanceProvider
from scripts.common.visitors import (
    HintableDeclarationVisitor,
    HintableParameterVisitor,
    HintableReturnVisitor,
    ScopeAwareVisitor,
)


@dataclass
class MultiVarAnnotations:
    """
    FORK OF LIBCST'S ANNOTATIONS: attributes is now a defaultdict with lists for values!

    Represents all the annotation information we might add to
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
    attributes: collections.defaultdict[str, list[Optional[libcst.Annotation]]]
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
    HintableDeclarationVisitor,
    HintableParameterVisitor,
    HintableReturnVisitor,
    ScopeAwareVisitor,
):
    """
    Collect type annotations from a stub module.
    """

    METADATA_DEPENDENCIES = (
        ScopeProvider,
        PositionProvider,
        FullyQualifiedNameProvider,
        Annotation4InstanceProvider,
    )

    annotations: MultiVarAnnotations

    def __init__(
        self,
        existing_imports: Set[str],
        module_imports: Dict[str, ImportItem],
        context: c.CodemodContext,
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

        self.logger = logging.getLogger(type(self).__qualname__)

    def annotated_function(
        self, function: libcst.FunctionDef, _: libcst.Annotation
    ) -> None:
        self._function(function)

    def unannotated_function(self, function: libcst.FunctionDef) -> None:
        self._function(function)

    def _function(
        self,
        node: libcst.FunctionDef,
    ) -> None:
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
    def annotated_param(
        self, param: libcst.Param, annotation: libcst.Annotation
    ) -> None:
        ...

    def unannotated_param(self, param: libcst.Param) -> None:
        ...

    def visit_Assign(
        self,
        node: libcst.Assign,
    ) -> None:
        self.current_assign = node

    def leave_Assign(
        self,
        _: libcst.Assign,
    ) -> None:
        self.current_assign = None

    def instance_attribute_hint(
        self, original_node: libcst.AnnAssign, target: libcst.Name
    ) -> None:
        self._track_attribute_target(target)

    def annotated_hint(
        self,
        original_node: libcst.AnnAssign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        ...

    def annotated_assignment(
        self,
        original_node: libcst.AnnAssign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        self._track_attribute_target(target)

    def assign_single_target(
        self,
        original_node: libcst.Assign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        self._track_attribute_target(target)

    def assign_multiple_targets_or_augassign(
        self,
        original_node: Union[libcst.Assign, libcst.AugAssign],
        target: Union[libcst.Name, libcst.Attribute],
    ) -> None:
        self._track_attribute_target(target)

    def for_target(
        self, original_node: libcst.For, target: Union[libcst.Name, libcst.Attribute]
    ) -> None:
        self._track_attribute_target(target)

    def withitem_target(
        self, original_node: libcst.With, target: Union[libcst.Name, libcst.Attribute]
    ) -> None:
        self._track_attribute_target(target)

    def _track_attribute_target(
        self, target: Union[libcst.Name, libcst.Attribute]
    ) -> None:
        # Reference stored hint if present
        full_qual = self.qualified_name(target)
        if hint := self.get_metadata(Annotation4InstanceProvider, target).labelled:
            hint = self._handle_Annotation(annotation=hint)

        self.annotations.attributes[full_qual].append(hint)

    # no-op, as these cannot ever be annotated
    def global_target(
        self,
        _1: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign],
        _2: libcst.Name,
    ) -> None:
        ...

    def nonlocal_target(
        self,
        _1: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign],
        _2: libcst.Name,
    ) -> None:
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
        _: libcst.Module,
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
                AddImportsVisitor.add_needed_import(
                    self.context, m.module_name, asname=asname
                )
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
    #  - dequalify the node if the node is builtin

    def _handle_NameOrAttribute(
        self,
        node: NameOrAttribute,
    ) -> Union[libcst.Name, libcst.Attribute]:
        if m.matches(node, m.Name("None")):
            return node

        qualified_name = self._get_unique_qualified_name(node)
        # _ = self._handle_qualification_and_should_qualify(qualified_name, node)
        self.annotations.names.add(qualified_name)

        qualified_node = libcst.parse_expression(qualified_name)
        assert isinstance(qualified_node, libcst.Name | libcst.Attribute), f"Cannot parse {qualified_name} into Name or Attribute, got {type(qualified_node)} instead"
        return qualified_node  # pyre-ignore[7]
        # else:
        #    dequalified_node = node.attr if isinstance(node, libcst.Attribute) else node
        #    return dequalified_node

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
        if self._get_unique_qualified_name(node) in ("Type", "typing.Type"):
            # Note: we are intentionally not handling qualification of
            # anything inside `Type` because it's common to have nested
            # classes, which we cannot currently distinguish from classes
            # coming from other modules, appear here.
            return new_node
        slice = node.slice
        if isinstance(slice, (tuple, list)):
            new_slice = []
            for item in slice:
                value = item.slice.value
                if isinstance(value, NAME_OR_ATTRIBUTE):
                    name = self._handle_NameOrAttribute(item.slice.value)
                    new_index = item.slice.with_changes(value=name)
                    new_slice.append(item.with_changes(slice=new_index))
                else:
                    if isinstance(item.slice, libcst.Index):
                        new_index = item.slice.with_changes(
                            value=self._handle_Index(item.slice)
                        )
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
                    left=self._handle_Annotation(
                        libcst.Annotation(node.left)
                    ).annotation,
                    operator=node.operator,
                    right=self._handle_Annotation(
                        libcst.Annotation(node.right)
                    ).annotation,
                )
            )

        # Note: this is primarily meant to support pydantic style annotations
        # which have HIGHLY dynamic properties, e.g. pydantic.constr
        elif m.matches(node, m.Call(m.Name() | m.Attribute())):
            return libcst.Annotation(annotation=self._handle_NameOrAttribute(node.func))

        else:
            code = libcst.Module([]).code_for_node(node)
            msg = f"{self.context.filename}: Unhandled annotation {code}"

            self.logger.error(msg)
            raise ValueError(msg)

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

    def _get_unique_qualified_name(self, node: libcst.CSTNode) -> str:
        name = None
        names = [q for q in self.get_metadata(FullyQualifiedNameProvider, node)]
        if len(names) == 0:
            # we hit this branch if the stub is directly using a fully
            # qualified name, which is not technically valid python but is
            # convenient to allow.
            name = h.get_full_name_for_node_or_raise(node).replace(".<locals>.", ".")
        elif len(names) >= 1:
            n = next(
                filter(lambda qname: not qname.name.startswith("."), names), names[0]
            )
            name = n.name.replace(".<locals>.", ".")

        if name is None:
            start = self.get_metadata(PositionProvider, node).start
            raise ValueError(
                "Could not resolve a unique qualified name for type "
                + f"{get_full_name_for_node(node)} at {start.line}:{start.column}. "
                + f"Candidate names were: {names!r}"
            )
        return name


from libcst.codemod.visitors._apply_type_annotations import Annotations


class ApplyTypeAnnotationsVisitor(
    t.HintableParameterTransformer,
    t.HintableReturnTransformer,
    t.HintableDeclarationTransformer,
    t.ScopeAwareTransformer,
):
    """
    Apply type annotations to a source module using the given stub modules.
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
        context: c.CodemodContext,
        annotations: Optional[Annotations] = None,
        overwrite_existing_annotations: bool = False,
        use_future_annotations: bool = False,
        strict_posargs_matching: bool = True,
        strict_annotation_matching: bool = False,
        always_qualify_annotations: bool = False,
    ) -> None:
        super().__init__(context)
        self.annotations: Annotations = (
            Annotations.empty() if annotations is None else annotations
        )
        self.visited_classes: Set[str] = set()
        self.overwrite_existing_annotations = overwrite_existing_annotations
        self.use_future_annotations = use_future_annotations
        self.strict_posargs_matching = strict_posargs_matching
        self.strict_annotation_matching = strict_annotation_matching
        self.always_qualify_annotations = always_qualify_annotations

        # We use this to determine the end of the import block so that we can
        # insert top-level annotations.
        self.import_statements: List[libcst.ImportFrom] = []

        # We use this to collect typevars, to avoid importing existing ones from the pyi file
        self.current_assign: Optional[libcst.Assign] = None
        self.typevars: Dict[str, libcst.Assign] = {}

        # Global variables and classes defined on the toplevel of the target module.
        # Used to help determine which names we need to check are in scope, and add
        # quotations to avoid undefined forward references in type annotations.
        self.global_names: Set[str] = set()

    @staticmethod
    def store_stub_in_context(
        context: c.CodemodContext,
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
        import_gatherer = GatherImportsVisitor(c.CodemodContext())
        tree.visit(import_gatherer)
        existing_import_names = _get_imported_names(import_gatherer.all_imports)

        global_names_gatherer = GatherGlobalNamesVisitor(c.CodemodContext())
        tree.visit(global_names_gatherer)
        self.global_names = global_names_gatherer.global_names.union(
            global_names_gatherer.class_names
        )

        context_contents = self.context.scratch.get(
            ApplyTypeAnnotationsVisitor.CONTEXT_KEY
        )
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
            self.use_future_annotations = (
                self.use_future_annotations or use_future_annotations
            )
            self.strict_posargs_matching = (
                self.strict_posargs_matching and strict_posargs_matching
            )
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
            AddImportsVisitor.add_needed_import(
                self.context, "__future__", "annotations"
            )

        tree_with_imports = AddImportsVisitor(self.context).transform_module(tree)
        tree_with_changes = libcst.MetadataWrapper(tree_with_imports).visit(self)

        # don't modify the imports if we didn't actually add any type information
        return tree_with_changes

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
        import_gatherer = GatherImportsVisitor(c.CodemodContext())
        stub.visit(import_gatherer)
        symbol_map = import_gatherer.symbol_mapping
        existing_import_names = _get_imported_names(
            existing_import_gatherer.all_imports
        )
        symbol_collector = ImportedSymbolCollector(existing_import_names, self.context)
        libcst.MetadataWrapper(stub).visit(symbol_collector)
        module_imports = {}
        for sym, imported_symbols in symbol_collector.imported_symbols.items():
            existing = existing_import_gatherer.symbol_mapping.get(sym)
            if existing and any(
                s.module_name != existing.module_name for s in imported_symbols
            ):
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
    def _quote_future_annotations(
        self, annotation: libcst.Annotation
    ) -> libcst.Annotation:
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
            return annotation.with_changes(
                annotation=libcst.SimpleString(value=f'"{node.value}"')
            )
        return annotation

    # smart constructors: all applied annotations happen via one of these

    def _apply_annotation_to_parameter(
        self,
        parameter: libcst.Param,
        annotation: libcst.Annotation,
    ) -> libcst.Param:
        return parameter.with_changes(
            annotation=self._quote_future_annotations(annotation),
        )

    def _apply_annotation_to_return(
        self,
        function_def: libcst.FunctionDef,
        annotation: libcst.Annotation,
    ) -> libcst.FunctionDef:
        return function_def.with_changes(
            returns=self._quote_future_annotations(annotation),
        )

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

    def unannotated_param(
        self, param: libcst.Param
    ) -> Union[
        libcst.Param,
        libcst.MaybeSentinel,
        libcst.FlattenSentinel[libcst.Param],
        libcst.RemovalSentinel,
    ]:
        return param

    def annotated_param(
        self, param: libcst.Param, annotation: libcst.Annotation
    ) -> Union[
        libcst.Param,
        libcst.MaybeSentinel,
        libcst.FlattenSentinel[libcst.Param],
        libcst.RemovalSentinel,
    ]:
        return param

    def annotated_function(
        self, function: libcst.FunctionDef, _: libcst.Annotation
    ) -> Union[
        libcst.BaseStatement,
        libcst.FlattenSentinel[libcst.BaseStatement],
        libcst.RemovalSentinel,
    ]:
        return self._handle_function(function)

    def unannotated_function(
        self, function: libcst.FunctionDef
    ) -> Union[
        libcst.BaseStatement,
        libcst.FlattenSentinel[libcst.BaseStatement],
        libcst.RemovalSentinel,
    ]:
        return self._handle_function(function)

    def _handle_function(
        self,
        updated_node: libcst.FunctionDef,
    ) -> libcst.FunctionDef:
        key = FunctionKey.make(".".join(self.qualified_scope()), updated_node.params)
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
        statements: List[
            Union[libcst.SimpleStatementLine, libcst.BaseCompoundStatement]
        ],
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

        return match_params(function, annotations) and match_return(
            function, annotations
        )

    # transform API methods

    def instance_attribute_hint(
        self, updated_node: libcst.AnnAssign, target: libcst.Name
    ) -> t.Actions:
        return self._handle_annotated_target(updated_node, target)

    @m.call_if_inside(m.Assign())
    @m.visit(m.Assign())
    def _visit_Assign(
        self,
        node: libcst.Assign,
    ) -> None:
        self.current_assign = node

    @m.call_if_inside(m.Assign())
    @m.visit(m.Call(func=m.Name("TypeVar")))
    def _record_typevar(
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

    def annotated_assignment(
        self,
        updated_node: libcst.AnnAssign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> t.Actions:
        return self._handle_annotated_target(updated_node, target)

    def annotated_hint(
        self,
        updated_node: libcst.AnnAssign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> t.Actions:
        return self._handle_annotated_target(updated_node, target)

    def _handle_annotated_target(
        self, annassign: libcst.AnnAssign, target: Union[libcst.Name, libcst.Attribute]
    ) -> t.Actions:
        if not self.overwrite_existing_annotations:
            return t.Actions((t.Untouched(),))

        if annotation := self.annotations.attributes.get(self.qualified_name(target)):
            matcher = m.Annotation(annotation=annassign.annotation.annotation)
            return t.Actions((t.Replace(matcher=matcher, replacement=annotation),))
        return t.Actions((t.Untouched(),))

    def assign_single_target(
        self,
        updated_node: libcst.Assign,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> t.Actions:
        annotation = self.annotations.attributes.get(self.qualified_name(target))
        if annotation is None:
            return t.Actions((t.Untouched(),))

        return t.Actions(
            (
                t.Replace(
                    matcher=m.Assign(updated_node.targets, value=updated_node.value),
                    replacement=libcst.AnnAssign(
                        target=target, annotation=annotation, value=updated_node.value
                    ),
                ),
            )
        )

    def assign_multiple_targets_or_augassign(
        self,
        updated_node: Union[libcst.Assign, libcst.AugAssign],
        target: Union[libcst.Name, libcst.Attribute],
    ) -> t.Actions:
        return self._hint_as_prepend(updated_node, target)

    def withitem_target(
        self,
        updated_node: libcst.With,
        target: Union[libcst.Name, libcst.Attribute],
    ) -> t.Actions:
        return self._hint_as_prepend(updated_node, target)

    def for_target(
        self, updated_node: libcst.For, target: Union[libcst.Name, libcst.Attribute]
    ) -> t.Actions:
        return self._hint_as_prepend(updated_node, target)

    def _hint_as_prepend(
        self, _: libcst.CSTNode, target: Union[libcst.Name, libcst.Attribute]
    ) -> t.Actions:
        annotation = self.annotations.attributes.get(self.qualified_name(target))
        if annotation is not None:
            action = t.Prepend(libcst.AnnAssign(target=target, annotation=annotation))
        else:
            action = t.Untouched()

        return t.Actions((action,))

    # no-op; do not annotate these targets
    def global_target(
        self,
        _1: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign],
        _2: libcst.Name,
    ) -> t.Actions:
        return t.Actions((t.Untouched(),))

    def nonlocal_target(
        self,
        _1: Union[libcst.Assign, libcst.AnnAssign, libcst.AugAssign],
        _2: libcst.Name,
    ) -> t.Actions:
        return t.Actions((t.Untouched(),))

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
        toplevel_statements = []
        # First, find the insertion point for imports
        statements_before_imports, statements_after_imports = self._split_module(
            original_node, updated_node
        )

        # Make sure there's at least one empty line before the first non-import
        statements_after_imports = self._insert_empty_line(statements_after_imports)

        # TypeVar definitions could be scattered through the file, so do not
        # attempt to put new ones with existing ones, just add them at the top.
        typevars = {
            k: v for k, v in self.annotations.typevars.items() if k not in self.typevars
        }
        if typevars:
            for var, stmt in typevars.items():
                toplevel_statements.append(libcst.Newline())
                toplevel_statements.append(stmt)
            toplevel_statements.append(libcst.Newline())

        # toplevel_statements.extend(fresh_class_definitions)

        return updated_node.with_changes(
            body=[
                *statements_before_imports,
                *toplevel_statements,
                *statements_after_imports,
            ]
        )