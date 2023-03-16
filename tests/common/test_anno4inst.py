import textwrap
from typing import Mapping

import libcst
from libcst import metadata, matchers as m


from common.metadata import anno4inst


def test_instance_attribute():
    code = textwrap.dedent(
        """
        class C:
            foo = ...
            foo2: int = ...
        """
    )
    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    foo, foo2 = m.findall(module, m.Name("foo") | m.Name("foo2"))

    assert anno4insts[foo] is None

    assert m.matches(anno4insts[foo2].annotation, m.Annotation(m.Name("int")))
    assert anno4insts[foo2].lowered is False


def test_annotated_assignment():
    code = textwrap.dedent(
        """
        a: int = 5
        b: amod.A = 20
        """
    )
    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    a, b = m.findall(module, m.Name("a") | m.Name("b"))

    assert m.matches(anno4insts[a].annotation, m.Annotation(m.Name("int")))
    assert anno4insts[a].lowered is False

    assert m.matches(
        anno4insts[b].annotation, m.Annotation(m.Attribute(m.Name("amod"), m.Name("A")))
    )
    assert anno4insts[b].lowered is False


def test_unannotated_target():
    code = textwrap.dedent(
        """
        b = 5
        """
    )
    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    (b,) = m.findall(module, m.Name("b"))

    assert anno4insts[b] is None


def test_annotated_hint():
    code = textwrap.dedent(
        """
    a: int"""
    )

    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    (a,) = m.findall(module, m.Name("a"))

    assert a not in anno4insts


def test_hint_usage():
    code = textwrap.dedent(
        """
        a: int
        a = 5
        a = "Hello World"
        """
    )

    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    a1, a2, a3 = m.findall(module, m.Name("a"))

    assert a1 not in anno4insts

    assert m.matches(anno4insts[a2].annotation, m.Annotation(m.Name("int")))
    assert anno4insts[a2].lowered is False

    assert m.matches(anno4insts[a3].annotation, m.Annotation(m.Name("int")))
    assert anno4insts[a3].lowered is False


def test_if_else_branching():
    code = textwrap.dedent(
        """
        a: int | None
        if cond:
            a = 5
        else:
            a = None
            
        a = "Hello World"
        """
    )

    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    _, a1, a2, a3 = m.findall(module, m.Name("a"))
    union_ty = libcst.parse_expression("int | None")

    assert m.matches(anno4insts[a1].annotation, m.Annotation(union_ty))
    assert anno4insts[a1].lowered is True

    assert m.matches(anno4insts[a2].annotation, m.Annotation(union_ty))
    assert anno4insts[a2].lowered is True

    assert m.matches(anno4insts[a3].annotation, m.Annotation(union_ty))
    assert anno4insts[a3].lowered is False


def test_only_if_branching():
    code = textwrap.dedent(
        """
        a: int | None
        if cond:
            a = 5
            
        a = "Hello World"
        """
    )

    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    _, a1, a2 = m.findall(module, m.Name("a"))
    union_ty = libcst.parse_expression("int | None")

    assert m.matches(anno4insts[a1].annotation, m.Annotation(union_ty))
    assert anno4insts[a1].lowered is True

    assert m.matches(anno4insts[a2].annotation, m.Annotation(union_ty))
    assert anno4insts[a2].lowered is False


def test_if_elif_branching():
    code = textwrap.dedent(
        """
        a: int | None
        if cond:
            a = 5
        elif cond2:
            a = None

        a = "Hello World"
        """
    )

    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    _, a1, a2, a3 = m.findall(module, m.Name("a"))
    union_ty = libcst.parse_expression("int | None")

    assert m.matches(anno4insts[a1].annotation, m.Annotation(union_ty))
    assert anno4insts[a1].lowered is True

    assert m.matches(anno4insts[a2].annotation, m.Annotation(union_ty))
    assert anno4insts[a2].lowered is True

    assert m.matches(anno4insts[a3].annotation, m.Annotation(union_ty))
    assert anno4insts[a3].lowered is False


def test_hint_branching():
    code = textwrap.dedent(
        """
        a: int | str | None
        if cond:
            a = 5

            if another_cond:
                a: str
                a = "Hello World"

                a = "Another Word"

            a = 10
            
            a: None
        else:
            a = None
            
        a = 20
        """
    )

    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    _, a1, _, a2, a22, a3, _, a4, a5 = m.findall(module, m.Name("a"))

    union_ty = libcst.parse_expression("int | str | None")

    assert m.matches(anno4insts[a1].annotation, m.Annotation(union_ty))
    assert anno4insts[a1].lowered is True

    assert m.matches(anno4insts[a2].annotation, m.Annotation(m.Name("str")))
    assert anno4insts[a2].lowered is False

    assert m.matches(anno4insts[a22].annotation, m.Annotation(m.Name("str")))
    assert anno4insts[a22].lowered is False

    assert m.matches(anno4insts[a3].annotation, m.Annotation(union_ty))
    assert anno4insts[a3].lowered is True

    assert m.matches(anno4insts[a4].annotation, m.Annotation(union_ty))
    assert anno4insts[a4].lowered is True

    assert m.matches(anno4insts[a5].annotation, m.Annotation(union_ty))
    assert anno4insts[a5].lowered is False



def test_retain_unused_through_branching():
    code = textwrap.dedent(
        """
    a: int | None
    if cond:
        b = 10
    else:
        b = None
    a = 20
    """
    )

    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    _, a1 = m.findall(module, m.Name("a"))
    assert m.matches(anno4insts[a1].annotation, m.Annotation(libcst.parse_expression("int | None")))
    assert anno4insts[a1].lowered is False


def test_narrowing():
    code = textwrap.dedent(
        """
    a: int | None
    if cond:
        a: int = 5
    else:
        a: None = None

    a = None
    """
    )

    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    _, a1, a2, a3 = m.findall(module, m.Name("a"))
    
    assert m.matches(anno4insts[a1].annotation, m.Annotation(libcst.parse_expression("int")))
    assert anno4insts[a1].lowered is False
    
    assert m.matches(anno4insts[a2].annotation, m.Annotation(libcst.parse_expression("None")))
    assert anno4insts[a2].lowered is False
    
    assert m.matches(anno4insts[a3].annotation, m.Annotation(libcst.parse_expression("int | None")))
    assert anno4insts[a3].lowered is False
