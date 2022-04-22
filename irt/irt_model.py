from typing import Union
import torch
from model import LSTMModel
from ckt_model import CKTJoint
from constants import Mode

class IRT(torch.nn.Module):
    def __init__(self, mode: Mode, num_students: int, num_questions: int, behavior_model: Union[LSTMModel, CKTJoint]):
        super().__init__()
        self.mode = mode
        self.ability = torch.nn.parameter.Parameter(torch.normal(0.0, 0.1, (num_students,)))
        self.difficulty = torch.nn.parameter.Parameter(torch.normal(0.0, 0.1, (num_questions,)))
        self.behavior_model = behavior_model

    def forward(self, batch):
        softplus = torch.nn.Softplus() # Ensure that ability and difficulty are always treated as positive values
        ability = softplus(self.ability[batch["student_ids"]])
        difficulty = softplus(self.difficulty[batch["question_ids_collapsed"]])
        predictions = ability - difficulty
        if self.behavior_model:
            if self.mode == Mode.CLUSTER:
                behavior_pred_state, behavior = self.behavior_model(batch)
                predictions += behavior
                return behavior_pred_state, behavior, predictions
            else:
                behavior = self.behavior_model(batch)
                predictions += behavior
        else:
            predictions = ability - difficulty

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        avg_loss = loss_fn(predictions, batch["labels"])
        return avg_loss, predictions.detach().cpu().numpy()
