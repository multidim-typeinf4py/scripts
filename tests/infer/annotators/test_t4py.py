import json
import textwrap
import libcst
import requests
from libcst import codemod, metadata
from libsa4py.cst_transformers import TypeAnnotationRemover, TypeApplier

from scripts.common.schemas import TypeCollectionCategory
from scripts.infer.preprocessers import t4py as t4py_proc
from scripts.infer.annotators import t4py as t4py_annot


class Test_Type4PyArtifacts(codemod.CodemodTest):
    TRANSFORM = t4py_annot.Type4PyFileApplier

    def test_libsa4py_artifact_removed(self):
        code = textwrap.dedent(
            """
        class Interface:
            x: int
        x: int
        """
        )
        annotated = libcst.parse_module(code)
        print(annotated.code)

        unannotated = t4py_proc.Type4PyAnnotationRemover(
            context=codemod.CodemodContext(), task=TypeCollectionCategory.VARIABLE
        ).transform_module(annotated)
        print(unannotated.code)

        predictions = json.loads(
            textwrap.dedent(
                """{
  "classes": [
    {
      "cls_lc": [
        [
          1,
          0
        ],
        [
          2,
          19
        ]
      ],
      "cls_var_ln": {
        "x": [
          [
            2,
            4
          ],
          [
            2,
            5
          ]
        ]
      },
      "cls_var_occur": {
        "x": []
      },
      "funcs": [],
      "name": "Interface",
      "q_name": "Interface",
      "variables": {
        "x": "builtins.int"
      },
      "variables_p": {
        "x": [
          [
            "int",
            0.4
          ]
        ]
      }
    }
  ],
  "funcs": [],
  "imports": [],
  "mod_var_ln": {},
  "mod_var_occur": {},
  "no_types_annot": {
    "D": 1,
    "I": 0,
    "U": 0
  },
  "session_id": "LWG2gorzExov8kYpz0tkF0sKUmwGmVa316ddfero_5g",
  "set": null,
  "tc": [
    false,
    null
  ],
  "type_annot_cove": 1,
  "typed_seq": "",
  "untyped_seq": "",
  "variables": {"x": "builtins.int"},
  "variables_p": {"x": [
          [
            "int",
            0.4
          ]
        ]}
} """
            )
        )

        self.assertCodemod(
            before=unannotated.code,
            after="""
            import builtins
            class Interface:
                x: builtins.int
            x: builtins.int
            """,
            predictions=predictions,
        )
