from a import A
from b import B
from c import C

def function(a: int, b: str, c: int) -> int:
    v: str = f'{a}{b}{c}'
    return int(v)

def function_with_multiline_parameters(
    a: str,
    b: int,
    c: str
) -> int:
    v: str = f'{a}{b}{c}'
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
        c # NOTE: Check that this parameter does not receive a type!
    ) -> tuple:
        return a, b, c

    # NOTE: Check that the parameter's hints symbols are fully qualified!
    def function(self, a: A, b: B, c: C) -> int:
        v = f'{a}{b}{c}' # NOTE: Not typed on purpose!
        return int(v)

a: int = 5

def outer() -> int:
    def nested(a: int) -> str:
        result: str = str(a)
        return result

    return int(nested(10))