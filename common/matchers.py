from libcst import matchers as m

NAME = m.Name()
INSTANCE_ATTR = m.Attribute(m.Name("self"), m.Name())

TUPLE = m.Tuple()
LIST = m.List()

UNPACKABLE_ELEMENT = (m.StarredElement | m.Element)(NAME | INSTANCE_ATTR)