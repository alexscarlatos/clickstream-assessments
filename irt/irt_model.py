import torch

class IRT(torch.nn.Module):
    def __init__(self, num_students: int, num_questions: int):
        super().__init__()
        self.ability = torch.nn.parameter.Parameter(torch.normal(0.0, 0.1, (num_students,)))
        self.difficulty = torch.nn.parameter.Parameter(torch.normal(0.0, 0.1, (num_questions,)))

    def forward(self, batch):
        softplus = torch.nn.Softplus() # Ensure that ability and difficulty are always treated as positive values
        ability = softplus(self.ability[batch["student_ids"]])
        difficulty = softplus(self.difficulty[batch["question_ids"]])
        predictions = ability - difficulty
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        avg_loss = loss_fn(predictions, batch["labels"])
        return avg_loss, predictions
