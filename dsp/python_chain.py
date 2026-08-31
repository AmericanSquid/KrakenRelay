import numpy as np


class PythonDSPChain:
    def __init__(self):
        self.stages = []

    def add(self, stage):
        self.stages.append(stage)

    def reset(self):
        for stage in self.stages:
            reset = getattr(stage, "reset", None)
            if callable(reset):
                reset()

    def process_int16_to_int16(self, samples: np.ndarray) -> np.ndarray:
        out = samples
        for stage in self.stages:
            out = stage.process_int16_to_int16(out)
        return out
