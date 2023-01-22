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
from libcst.helpers import get_full_name_for_node
from libcst.metadata import PositionProvider, QualifiedNameProvider

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
    attributes: collections.defaultdict[str, list[cst.Annotation]]
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
