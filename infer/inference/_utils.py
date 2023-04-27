from io import BytesIO

import torch
import pickle


class BetterUnpickler(pickle.Unpickler):
    def __init__(self, *args, map_location="cpu", **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(BytesIO(b), map_location=self._map_location)
        else:
            return super().find_class(module, name)
