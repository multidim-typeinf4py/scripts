from __future__ import annotations
from proj1.amod import A
from proj1.bmod import B
from proj1.cmod import C
from typing import TYPE_CHECKING, Tuple


def function(a: int, b: str, c: int) -> int:
    v = a + b + c
    return v


def function_with_multiline_parameters(a: str, b: int, c: str) -> int:
    v = a + b + c
    return v


class Clazz(dict):
    a: int = ...

    def __init__(self, a: int) -> None:
        self.a: int = a

    # NOTE: Check that "None" is inferred
    # NOTE: "None" != None
    def method(self, a: int, b: str, c: int) -> Tuple[int, str, int]:
        return a, b, c

    def multiline_method(
        self,
        a: str,
        b: int,
        c: int,  # NOTE: Check that this parameter does not receive a type!
    ) -> Tuple[str, int, int]:
        return a, b, c

    # NOTE: Check that the parameter's hints symbols are fully qualified!
    def function(self, a: A, b: B, c: C) -> int:
        v = a + b + c  # NOTE: Not typed on purpose!
        return v


a: int = 5


def outer() -> int:
    def nested(a: int) -> str:
        result: str = str(a)
        return result

    return int(nested(10))

class Outer:
    class Inner:
        def __init__(self) -> None:
            self.x = 10

print("Hello World!")
Clazz(10).method(1, "Hello", 3)
Clazz(10).multiline_method("World", 4, 6)
