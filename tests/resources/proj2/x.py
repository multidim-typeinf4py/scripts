# Based on proj1/x.py, minus type hints for Clazz.function and a

from a import A
from b import B
from c import C


def function(a: int, b: str, c: int) -> int:
    v: str = f"{a}{b}{c}"
    return int(v)


def function_with_multiline_parameters(a: str, b: int, c: str) -> int:
    v: str = f"{a}{b}{c}"
    return int(v)


class Clazz(dict):
    a: int

    def __init__(self, a: int) -> None:
        self.a: int = a

    # NOTE: Check that "None" is inferred
    # NOTE: "None" != None
    def method(self, a: int, b: str, c: int):
        return a, b, c

    def multiline_method(
        self,
        a: str,
        b: int,
        c,  # NOTE: Check that this parameter does not receive a type!
    ) -> tuple:
        return a, b, c

    # NOTE: Not typed on purpose!
    def function(self, a, b, c):
        v = f"{a}{b}{c}"
        return int(v)


a = 5


def outer() -> int:
    def nested(a: int) -> str:
        result: str = str(a)
        return result

    return int(nested(10))
