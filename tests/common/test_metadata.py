import textwrap

import libcst
from libcst import metadata
from libcst import matchers as m

from common.metadata import KeywordModifiedScopeProvider, KeywordContext


def test_nonlocal():
    code = textwrap.dedent(
        """
    x = 0
    def outer():
        x = 1
        def inner():
            nonlocal x
            x = 2
            print("inner:", x)

        inner()
        print("outer:", x)
        x = 10

    outer()
    print("global:", x)

    x += 5

    x: int = 10
    """
    )

    wrapper = metadata.MetadataWrapper(libcst.parse_module(code))
    mapping = wrapper.resolve(KeywordModifiedScopeProvider)

    xs = m.findall(
        wrapper,
        m.Assign(targets=[m.AssignTarget(m.Name(value="x"))])
        | m.AugAssign(target=m.Name(value="x"))
        | m.AnnAssign(target=m.Name(value="x")),
    )

    assert mapping[xs[0].targets[0].target] is KeywordContext.UNCHANGED
    assert mapping[xs[1].targets[0].target] is KeywordContext.UNCHANGED
    assert mapping[xs[2].targets[0].target] is KeywordContext.NONLOCAL
    assert mapping[xs[3].targets[0].target] is KeywordContext.UNCHANGED

    assert mapping[xs[4].target] is KeywordContext.UNCHANGED
    assert mapping[xs[5].target] is KeywordContext.UNCHANGED


def test_global():
    code = textwrap.dedent(
        """
    x = 0
    def outer():
        x = 1
        def inner():
            global x
            x = 2
            print("inner:", x)

        inner()
        print("outer:", x)
        x = 10

    outer()
    print("global:", x)

    x += 5

    x: int = 10
    """
    )

    wrapper = metadata.MetadataWrapper(libcst.parse_module(code))
    mapping = wrapper.resolve(KeywordModifiedScopeProvider)

    xs = m.findall(
        wrapper,
        m.Assign(targets=[m.AssignTarget(m.Name(value="x"))])
        | m.AugAssign(target=m.Name(value="x"))
        | m.AnnAssign(target=m.Name(value="x")),
    )

    assert mapping[xs[0].targets[0].target] is KeywordContext.UNCHANGED
    assert mapping[xs[1].targets[0].target] is KeywordContext.UNCHANGED
    assert mapping[xs[2].targets[0].target] is KeywordContext.GLOBAL
    assert mapping[xs[3].targets[0].target] is KeywordContext.UNCHANGED

    assert mapping[xs[4].target] is KeywordContext.UNCHANGED
    assert mapping[xs[5].target] is KeywordContext.UNCHANGED



def test_wat():
    from libcst.metadata import ScopeProvider

    code = textwrap.dedent(
        """
        a = 10
        b, _ = 10, None
        c += "Hello"
    """
    )

    module = libcst.parse_module(code)
    wrapper = metadata.MetadataWrapper(module)
    mapping = wrapper.resolve(ScopeProvider)

    xs = m.findall(
        wrapper,
        m.Name(),
    )

    assert xs[0] in mapping
    assert xs[1] in mapping
    assert xs[2] in mapping


    wrapper2 = metadata.MetadataWrapper(module)
    mapping2 = wrapper2.resolve(ScopeProvider)

    xs2 = m.findall(
        wrapper2,
        m.Name(),
    )

    assert xs2[0] in mapping2
    assert xs2[1] in mapping2
    assert xs2[2] in mapping2
