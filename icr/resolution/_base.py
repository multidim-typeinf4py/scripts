import abc
from common.schemas import InferredSchema

import pandera.typing as pt


class ConflictResolution(abc.ABC):
    @abc.abstractmethod
    def forward(
        self,
        inferences: dict[str, pt.DataFrame[InferredSchema]]
    ) -> pt.DataFrame[InferredSchema]:
        ...
