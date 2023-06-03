import textwrap
from typing import Optional
from unittest import TestCase

import libcst
from libcst import metadata, matchers as m

from scripts.common.metadata.anno4inst import Lowered, TrackedAnnotation, Annotation4InstanceProvider


class LabelTesting(TestCase):
    def assertLabelled(
        self,
        meta: TrackedAnnotation,
        labelled: libcst.Annotation,
        lowerage: Optional[Lowered] = None,
    ):
        assert m.matches(meta.labelled, m.Annotation(labelled.annotation))
        assert m.matches(meta.inferred, m.Annotation(labelled.annotation))

        if lowerage is not None:
            assert meta.lowered is lowerage

    def assertInferred(
        self,
        meta: TrackedAnnotation,
        inferred: libcst.Annotation,
        lowerage: Optional[Lowered] = None,
    ):
        assert meta.labelled is None
        assert m.matches(meta.inferred, m.Annotation(inferred.annotation))

        if lowerage is not None:
            assert meta.lowered is lowerage

    def assertUnannotated(
        self,
        meta: TrackedAnnotation,
    ):
        assert meta.labelled is meta.inferred is None


class Individual(LabelTesting):
    def test_instance_attribute_hint(self) -> None:
        code = textwrap.dedent(
            """
        class C:
            foo: int
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        (foo,) = m.findall(module, m.Name("foo"))
        self.assertLabelled(
            anno4insts[foo],
            labelled=libcst.Annotation(libcst.parse_expression("int")),
        )

    def test_libsa4py_hint(self) -> None:
        code = textwrap.dedent(
            """
        class C:
            foo = ...
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        (foo,) = m.findall(module, m.Name("foo"))
        self.assertUnannotated(anno4insts[foo])

    def test_annotated_assignment(self) -> None:
        code = textwrap.dedent("b: bmod.B = 20")

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        (b,) = m.findall(module, m.Name("b"))
        self.assertLabelled(
            anno4insts[b],
            labelled=libcst.Annotation(libcst.parse_expression("bmod.B")),
        )

    def test_annotated_hint(self) -> None:
        code = textwrap.dedent(
            """
        b: int
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        (hint,) = m.findall(module, m.Name("b"))
        assert hint not in anno4insts

    def test_unannotated_assign_single_target(self) -> None:
        code = textwrap.dedent(
            """
        b: int
        b = 20
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        _, b1 = m.findall(module, m.Name("b"))

        self.assertInferred(
            anno4insts[b1],
            inferred=libcst.Annotation(libcst.parse_expression("int")),
        )

    def test_unannotated_assign_multiple_targets(self) -> None:
        code = textwrap.dedent(
            """
        a: int
        b: int
        a, b = 10, 20
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        _, a = m.findall(module, m.Name("a"))
        _, b = m.findall(module, m.Name("b"))

        self.assertLabelled(
            anno4insts[a],
            labelled=libcst.Annotation(libcst.parse_expression("int")),
        )
        self.assertLabelled(
            anno4insts[b],
            labelled=libcst.Annotation(libcst.parse_expression("int")),
        )

    def test_for_target(self) -> None:
        code = textwrap.dedent(
            """
        a: int
        b: str
        
        for a, b in zip([1, 2, 3], "abc"):
            ...
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        _, a = m.findall(module, m.Name("a"))
        _, b = m.findall(module, m.Name("b"))

        self.assertLabelled(
            anno4insts[a],
            labelled=libcst.Annotation(libcst.parse_expression("int")),
        )
        self.assertLabelled(
            anno4insts[b],
            labelled=libcst.Annotation(libcst.parse_expression("str")),
        )

    def test_withitem_target(self) -> None:
        code = textwrap.dedent(
            """
        f: _io.TextWrapper
        with p.open() as f:
            ...
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        _, f = m.findall(module, m.Name("f"))

        self.assertLabelled(
            anno4insts[f],
            labelled=libcst.Annotation(libcst.parse_expression("_io.TextWrapper")),
        )


class Consumption(LabelTesting):
    def test_annotated_assignment(self) -> None:
        code = textwrap.dedent(
            """
        a: int = 10
        a = 20
        a = 30
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        a1, a2, a3 = m.findall(module, m.Name("a"))

        annotation = libcst.Annotation(libcst.parse_expression("int"))
        self.assertLabelled(anno4insts[a1], labelled=annotation)
        self.assertInferred(anno4insts[a2], inferred=annotation)
        self.assertInferred(anno4insts[a3], inferred=annotation)

    def test_annotated_hint(self) -> None:
        code = textwrap.dedent(
            """
        a: int
        a = 20
        a = 30
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        a1, a2, a3 = m.findall(module, m.Name("a"))

        annotation = libcst.Annotation(libcst.parse_expression("int"))
        assert a1 not in anno4insts
        self.assertInferred(anno4insts[a2], inferred=annotation)
        self.assertInferred(anno4insts[a3], inferred=annotation)

    def test_unannotated_assign_single_target(self) -> None:
        code = textwrap.dedent(
            """
        a = 20

        a: int
        a = 30
        a = 20
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        a1, a2, a3, a4 = m.findall(module, m.Name("a"))

        annotation = libcst.Annotation(libcst.parse_expression("int"))
        self.assertUnannotated(anno4insts[a1])
        assert a2 not in anno4insts
        self.assertInferred(anno4insts[a3], inferred=annotation)
        self.assertInferred(anno4insts[a4], inferred=annotation)

    def test_unannotated_assign_multiple_targets(self) -> None:
        code = textwrap.dedent(
            """
        a = 20

        a: int
        c = 10

        a, b = 30, "Hello World"
        a, _ = a, b
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        a1, a2, a3, a4, _ = m.findall(module, m.Name("a"))
        (b, _) = m.findall(module, m.Name("b"))
        (c,) = m.findall(module, m.Name("c"))

        annotation = libcst.Annotation(libcst.parse_expression("int"))
        self.assertUnannotated(anno4insts[a1])

        assert a2 not in anno4insts
        self.assertUnannotated(anno4insts[c])

        self.assertLabelled(anno4insts[a3], labelled=annotation)
        self.assertUnannotated(anno4insts[b])
        self.assertInferred(anno4insts[a4], inferred=annotation)

    def test_for_target(self) -> None:
        code = textwrap.dedent(
            """
        for a, b in zip([1, 2, 3], "abc"):
            ...

        a: int
        b: str
        
        for a, b in zip([1, 2, 3], "abc"):
            ...

        for a, b in zip([1, 2, 3], "abc"):
            ...
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        a1, ahint, a3, a4 = m.findall(module, m.Name("a"))
        intanno = libcst.Annotation(libcst.parse_expression("int"))

        self.assertUnannotated(anno4insts[a1])
        assert ahint not in anno4insts
        self.assertLabelled(anno4insts[a3], labelled=intanno)
        self.assertInferred(anno4insts[a4], inferred=intanno)

        b1, bhint, b3, b4 = m.findall(module, m.Name("b"))
        stranno = libcst.Annotation(libcst.parse_expression("str"))

        self.assertUnannotated(anno4insts[b1])
        assert bhint not in anno4insts
        self.assertLabelled(anno4insts[b3], labelled=stranno)
        self.assertInferred(anno4insts[b4], inferred=stranno)

    def test_withitem_target(self) -> None:
        code = textwrap.dedent(
            """
        with p.open() as f:
            ...

        f: _io.TextWrapper
        with p.open() as f:
            ...

        with p.open() as f:
            ...
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        f1, fhint, f2, f3 = m.findall(module, m.Name("f"))
        annotation = libcst.Annotation(libcst.parse_expression("_io.TextWrapper"))

        self.assertUnannotated(anno4insts[f1])
        assert fhint not in anno4insts
        self.assertLabelled(anno4insts[f2], labelled=annotation)
        self.assertInferred(anno4insts[f3], inferred=annotation)


class Lowering(LabelTesting):
    def test_if_branching(self):
        code = textwrap.dedent(
            """
            a: int | str = 10
            if cond:
                a: str = "Hello World"
            
            a = 20
            """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        a1, a2, a3 = m.findall(module, m.Name("a"))

        self.assertLabelled(
            anno4insts[a1],
            labelled=libcst.Annotation(libcst.parse_expression("int | str")),
            lowerage=Lowered.UNALTERED,
        )
        self.assertLabelled(
            anno4insts[a2],
            labelled=libcst.Annotation(libcst.parse_expression("str")),
            lowerage=Lowered.UNALTERED,
        )
        self.assertInferred(
            anno4insts[a3],
            inferred=libcst.Annotation(libcst.parse_expression("int | str")),
            lowerage=Lowered.UNALTERED,
        )

    def test_if_else_branching(self):
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
        anno4insts = module.resolve(Annotation4InstanceProvider)

        _, a1, a2, a3 = m.findall(module, m.Name("a"))
        union_ty = libcst.Annotation(libcst.parse_expression("int | None"))

        self.assertInferred(anno4insts[a1], inferred=union_ty, lowerage=Lowered.ALTERED)
        self.assertInferred(anno4insts[a2], inferred=union_ty, lowerage=Lowered.ALTERED)
        self.assertInferred(anno4insts[a3], inferred=union_ty, lowerage=Lowered.UNALTERED)

    def test_hint_branching(self):
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
        anno4insts = module.resolve(Annotation4InstanceProvider)

        _, a1, _, a2, a22, a3, _, a4, a5 = m.findall(module, m.Name("a"))

        union_ty = libcst.Annotation(libcst.parse_expression("int | str | None"))
        stranno = libcst.Annotation(libcst.parse_expression("str"))

        self.assertInferred(anno4insts[a1], inferred=union_ty, lowerage=Lowered.ALTERED)
        self.assertInferred(anno4insts[a2], inferred=stranno, lowerage=Lowered.UNALTERED)
        self.assertInferred(anno4insts[a22], inferred=stranno, lowerage=Lowered.UNALTERED)
        self.assertInferred(anno4insts[a3], inferred=union_ty, lowerage=Lowered.ALTERED)
        self.assertInferred(anno4insts[a4], inferred=union_ty, lowerage=Lowered.ALTERED)
        self.assertInferred(anno4insts[a5], inferred=union_ty, lowerage=Lowered.UNALTERED)

    def test_retain_unused_through_branching(self):
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
        anno4insts = module.resolve(Annotation4InstanceProvider)

        (_, a) = m.findall(module, m.Name("a"))
        b1, b2 = m.findall(module, m.Name("b"))

        self.assertUnannotated(anno4insts[b1])
        self.assertUnannotated(anno4insts[b2])

        self.assertInferred(
            anno4insts[a],
            libcst.Annotation(libcst.parse_expression("int | None")),
            Lowered.UNALTERED,
        )

    def test_narrowing(self):
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
        anno4insts = module.resolve(Annotation4InstanceProvider)

        _, a1, a2, a3 = m.findall(module, m.Name("a"))

        self.assertLabelled(
            anno4insts[a1],
            libcst.Annotation(libcst.parse_expression("int")),
            Lowered.UNALTERED,
        )
        self.assertLabelled(
            anno4insts[a2],
            libcst.Annotation(libcst.parse_expression("None")),
            Lowered.UNALTERED,
        )
        self.assertInferred(
            anno4insts[a3],
            libcst.Annotation(libcst.parse_expression("int | None")),
            Lowered.UNALTERED,
        )

    def test_hints_only_inside_body(self):
        code = textwrap.dedent(
            """
            if cond:
                a: int = 5
            else:
                a: None = None
                
            a += 1
            """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        a1, a2, a3 = m.findall(module, m.Name("a"))

        self.assertLabelled(
            anno4insts[a1],
            libcst.Annotation(libcst.parse_expression("int")),
            Lowered.UNALTERED,
        )
        self.assertLabelled(
            anno4insts[a2],
            libcst.Annotation(libcst.parse_expression("None")),
            Lowered.UNALTERED,
        )
        self.assertUnannotated(anno4insts[a3])

    def test_hinting_unannotatable(self):
        code = textwrap.dedent(
            """
        a: int | None
        if cond:
            a, = 10,
        else:
            a, = None,
        """
        )

        module = metadata.MetadataWrapper(libcst.parse_module(code))
        anno4insts = module.resolve(Annotation4InstanceProvider)

        _, a2, a3 = m.findall(module, m.Name("a"))
        annotation = libcst.Annotation(libcst.parse_expression("int | None"))

        self.assertInferred(
            anno4insts[a2],
            annotation,
            Lowered.ALTERED,
        )
        self.assertInferred(
            anno4insts[a3],
            annotation,
            Lowered.ALTERED,
        )
