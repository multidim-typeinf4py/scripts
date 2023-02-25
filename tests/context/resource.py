def looping():
    # For loop loopage
    for _ in range(10):
        x = 20

    # While loop loopage
    while x not in range(10, 20):
        x *= 2

    # Loop in loopage
    for _ in range(20):
        a = x
        while a not in range(0, 2):
            a /= 2


class UserDefinedClass:
    ...


def userdeffed():
    abc: int = 10
    efg: float = 5.0
    udc: UserDefinedClass = UserDefinedClass()


def local_reassign():
    c = "Hello World"
    c = 50
    print(c)


# Global variable
a = 10


def f():
    # Local variable; NOT a reassignment of the global
    a = 5

    def g():
        # NOT a reassignment!
        a = "Hello World"

    g()
    print(type(a))


def g():
    # Reassignment of global variable; do NOT mark as reassigned as the assignment is contained within this scope?
    global a
    a = "magic"

    # Repeated reassignment of global variable
    a = bytes([1, 2, 3])


def parammed(p: int | None):
    p: int = p or 10

def branching():
    # This x is outside of the branching, ignore
    x = 10

    if x < 10:
        b = True
    elif x > 10:
        b = None
    else:
        b = False

    for e in range(10):
        a = e
    else:
        a = 20

if __name__ == "__main__":
    local_reassign()