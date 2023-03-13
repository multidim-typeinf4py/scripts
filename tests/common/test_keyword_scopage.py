import textwrap

import libcst
from libcst import metadata
from libcst import matchers as m

from common.metadata.keyword_scopage import KeywordModifiedScopeProvider, KeywordContext


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