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

class Clazz:
    def __init__(self, a: int) -> None:
        self.a: int = a

    def method(self, a: int, b: str, c: int) -> tuple:
        return a, b, c

    def multiline_method(
        self, 
        a: str, 
        b: int, 
        c: str
    ) -> tuple:
        return a, b, c

    def function(self, a: A, b: B, c: C) -> int:
        v: str = f'{a}{b}{c}'
        return int(v)

a: int = 5

def outer() -> int:
    def nested(a: int) -> str:
        result = str(a)
        return result

    return int(nested(10))