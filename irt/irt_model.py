from typing import Union
import torch
from model import LSTMModel
from ckt_model import CKTPredictor

class IRT(torch.nn.Module):
    def __init__(self, num_students: int, num_questions: int, behavior_model: Union[LSTMModel, CKTPredictor]):
        super().__init__()
        self.ability = torch.nn.parameter.Parameter(torch.normal(0.0, 0.1, (num_students,)))
        self.difficulty = torch.nn.parameter.Parameter(torch.normal(0.0, 0.1, (num_questions,)))
        self.behavior_model = behavior_model

    def forward(self, batch):
        softplus = torch.nn.Softplus() # Ensure that ability and difficulty are always treated as positive values
        ability = softplus(self.ability[batch["student_ids"]])
        difficulty = softplus(self.difficulty[batch["question_ids_collapsed"]])
        predictions = ability - difficulty
        if self.behavior_model:
            behavior = self.behavior_model(batch)
            predictions += behavior
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        avg_loss = loss_fn(predictions, batch["labels"])
        return avg_loss, predictions.detach().cpu().numpy()
