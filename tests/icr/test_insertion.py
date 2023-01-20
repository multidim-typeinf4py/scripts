import pathlib
import tempfile
import textwrap


from libcst import codemod

from common._helper import generate_qname_ssas_for_file
from common.schemas import TypeCollectionCategory, TypeCollectionSchema

from icr.insertion import FromQName2SSAQNameTransformer, FromSSAQName2QnameTransformer
from symbols.collector import build_type_collection

import pandas as pd
import pandas._libs.missing as missing
import pandera.typing as pt


class QNameTransformation(codemod.CodemodTest):
    QNAME = textwrap.dedent(
        """
        a = 10
        a = "Hello World"

        def f(a, b, c): ...
        
        class C:
            def __init__(self):
                self.x = 0
                default = self.x or "10"
                self.x = self.x or "10"
        """
    )

    QNAME_SSA = textwrap.dedent(
        """
        aλ1 = 10
        aλ2 = "Hello World"

        def f(a, b, c): ...
        
        class C:
            def __init__(self):
                self.xλ1 = 0
                defaultλ1 = self.x or "10"
                self.xλ2 = self.x or "10"
        """
    )

    ANNOS = pd.DataFrame(
        {
            "file": ["x.py"] * 5,
            "category": [TypeCollectionCategory.VARIABLE] * 5,
            "qname": ["a"] * 2 + [f"C.__init__.{v}" for v in ("self.x", "default", "self.x")],
            "anno": [missing.NA] * 5,
        }
    ).pipe(generate_qname_ssas_for_file).pipe(pt.DataFrame[TypeCollectionSchema])


class Test_QName2QNameSSA(QNameTransformation):
    TRANSFORM = FromQName2SSAQNameTransformer

    def test_qnames_transformed(self):
        self.assertCodemod(
            Test_QName2QNameSSA.QNAME,
            Test_QName2QNameSSA.QNAME_SSA,
            annotations=Test_QName2QNameSSA.ANNOS,
        )


class Test_QNameSSA2QName(QNameTransformation):
    TRANSFORM = FromSSAQName2QnameTransformer

    def test_qname_ssas_transformed(self):
        self.assertCodemod(
            Test_QName2QNameSSA.QNAME_SSA,
            Test_QName2QNameSSA.QNAME,
            annotations=Test_QName2QNameSSA.ANNOS,
        )
