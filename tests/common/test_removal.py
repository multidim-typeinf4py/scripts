from libcst import codemod

from scripts.common import TypeAnnotationRemover


class Test_Removal(codemod.CodemodTest):
    TRANSFORM = TypeAnnotationRemover

    def test_remove_parameter_hints(self):
        self.assertCodemod(
            before="""
            def f(a: int, b, c: typing.Callable) -> None:
                x: int = 10
                
            class C:
                def g(self: typing.Self, d: lmao, e: more) -> int:
                    return e + 20
            """,
            after="""
            def f(a, b, c) -> None:
                x: int = 10
                
            class C:
                def g(self, d, e) -> int:
                    return e + 20
            """,
            parameters=True,
            rets=False,
            variables=False,
        )

    def test_remove_return_hints(self):
        self.assertCodemod(
            before="""
            def f(a: int, b, c: typing.Callable) -> None:
                x: int = 10

            class C:
                a: int = 10
                def g(self: typing.Self, d: lmao, e: more) -> int:
                    return e + 20
            """,
            after="""
            def f(a: int, b, c: typing.Callable):
                x: int = 10

            class C:
                a: int = 10
                def g(self: typing.Self, d: lmao, e: more):
                    return e + 20
            """,
            parameters=False,
            rets=True,
            variables=False,
        )

    def test_remove_variable_hints(self):
        self.assertCodemod(
            before="""
            def f(a: int, b, c: typing.Callable) -> None:
                x = 10

            class C:
                a: int
                def g(self: typing.Self, d: lmao, e: more) -> int:
                    return e + 20
            """,
            after="""
            def f(a: int, b, c: typing.Callable) -> None:
                x = 10

            class C:
                a = ...
                def g(self: typing.Self, d: lmao, e: more) -> int:
                    return e + 20
            """,
            parameters=False,
            rets=False,
            variables=True,
        )