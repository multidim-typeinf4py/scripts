import pathlib

from libsa4py.cst_extractor import Extractor
from type4py.deploy.infer import (
    get_dps_single_file,
    get_type_preds_single_file,
)

from scripts.infer.inference.t4py import PTType4Py
from ._hityper import ModelAdaptor, HiTyper


class Type4PyAdaptor(ModelAdaptor):
    def __init__(self, model_path: pathlib.Path, topn: int) -> None:
        super().__init__(model_path)
        self.type4py = PTType4Py(pre_trained_model_path=model_path, topn=topn)

    def topn(self) -> int:
        return self.type4py.topn

    def predict(
        self, project: pathlib.Path, subset: set[pathlib.Path]
    ) -> ModelAdaptor.ProjectPredictions:
        r = ModelAdaptor.ProjectPredictions(__root__=dict())

        for file in subset:
            with (project / file).open() as f:
                src_f_read = f.read()
            type_hints = Extractor.extract(src_f_read, include_seq2seq=False).to_dict()

            (
                all_type_slots,
                vars_type_hints,
                params_type_hints,
                rets_type_hints,
            ) = get_dps_single_file(type_hints)

            if not any(h for h in (vars_type_hints, params_type_hints, rets_type_hints)):
                continue

            p = get_type_preds_single_file(
                type_hints,
                all_type_slots,
                (vars_type_hints, params_type_hints, rets_type_hints),
                self.type4py,
                filter_pred_types=False,
            )

            parsed = ModelAdaptor.FilePredictions.parse_obj(p)
            r.__root__[str(project.resolve() / file)] = parsed

        return r


class _HiType4PyTopN(HiTyper):
    def __init__(self, topn: int) -> None:
        super().__init__(Type4PyAdaptor(model_path=pathlib.Path("models") / "type4py", topn=topn))

    def method(self) -> str:
        return f"HiType4PyN{self.adaptor.topn()}"


class HiType4PyTop1(_HiType4PyTopN):
    def __init__(self) -> None:
        super().__init__(topn=1)


class HiType4Py3(_HiType4PyTopN):
    def __init__(self) -> None:
        super().__init__(topn=3)


class HiType4Py5(_HiType4PyTopN):
    def __init__(self) -> None:
        super().__init__(topn=5)


class HiType4PyTop10(_HiType4PyTopN):
    def __init__(self) -> None:
        super().__init__(topn=10)
