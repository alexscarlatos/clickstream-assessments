import random
from numpy.lib.function_base import copy
import torch

class CopyBaseline:
    def __init__(self):
        pass

    def __call__(self, batch):
        correct = 0.0
        predictions = []
        for sequence, sequence_len in zip(batch["event_types"], batch["sequence_lengths"]):
            for idx in range(sequence_len):
                # Prediction is the preceeding event (except at idx 0)
                copy_idx = idx + 1 if idx == 0 else idx - 1
                predictions.append(sequence[copy_idx])
                if sequence[copy_idx] == sequence[idx]:
                    correct += 1
        return torch.tensor([0]), correct / sum(batch["sequence_lengths"]), predictions
