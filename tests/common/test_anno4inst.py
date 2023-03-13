import textwrap
from typing import Mapping

import libcst
from libcst import metadata, matchers as m


from common.metadata import anno4inst


def retrieve_metadata(code: libcst.Module) -> Mapping[libcst.CSTNode, libcst.Annotation | None]:
    return metadata.MetadataWrapper(code).resolve(anno4inst.Annotation4InstanceProvider)


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
    assert m.matches(anno4insts[foo2], m.Annotation(m.Name("int")))


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

    assert m.matches(anno4insts[a], m.Annotation(m.Name("int")))
    assert m.matches(anno4insts[b], m.Annotation(m.Attribute(m.Name("amod"), m.Name("A"))))


def test_unannotated_target():
    code = textwrap.dedent(
        """
        b = 5
        """
    )
    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    b, = m.findall(module, m.Name("b"))

    assert anno4insts[b] is None


def test_annotated_hint():
    code = textwrap.dedent(
        """
    a: int"""
    )

    module = metadata.MetadataWrapper(libcst.parse_module(code))
    anno4insts = module.resolve(anno4inst.Annotation4InstanceProvider)

    a, = m.findall(module, m.Name("a"))

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
    assert m.matches(anno4insts[a2], m.Annotation(m.Name("int")))
    assert anno4insts[a3] is None