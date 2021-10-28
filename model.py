from typing import Dict
from enum import Enum
import torch
from torch import nn

question_embedding_size = 16
question_type_embedding_size = 128
event_type_embedding_size = 128
hidden_size = 128

class Mode(Enum):
    PRE_TRAIN = 1
    PREDICT = 2

class Direction(Enum):
    FWD = 1
    BACK = 2
    BI = 3

class PredictionState(Enum):
    LAST = 1
    FIRST = 2
    BOTH_SUM = 3
    BOTH_CONCAT = 4
    AVG = 5

class TrainOptions:
    def __init__(self, options: dict):
        self.lstm_dir: Direction = options.get("lstm_dir", Direction.BI)
        self.use_pretrained_weights: bool = options.get("pretrained_model", True)
        self.use_pretrained_embeddings: bool = options.get("pretrained_emb", True)
        self.freeze_model: bool = options.get("freeze_model", True)
        self.freeze_embeddings: bool = options.get("freeze_emb", True)
        self.prediction_state: PredictionState = options.get("pred_state", PredictionState.AVG)
        self.attention: bool = options.get("attention", True)
        self.dropout: float = options.get("dropout", 0.25)

class LSTMModel(nn.Module):
    def __init__(self, mode: Mode, type_mappings: Dict[str, list], options: TrainOptions):
        super().__init__()
        self.options = options
        self.mode = mode
        num_questions = len(type_mappings["question_ids"])
        num_question_types = len(type_mappings["question_types"])
        self.num_event_types = len(type_mappings["event_types"])
        # self.question_embeddings = nn.Embedding(num_questions, question_embedding_size)
        self.question_type_embeddings = nn.Embedding(num_question_types, question_type_embedding_size)
        self.event_type_embeddings = nn.Embedding(self.num_event_types, event_type_embedding_size)
        self.dropout = nn.Dropout(options.dropout)
        # [question; question type; event type; time delta]
        self.input_size = question_type_embedding_size + event_type_embedding_size + 1
        # self.input_size = question_embedding_size + question_type_embedding_size + event_type_embedding_size + 1
        # TODO: use GRU instead?
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, batch_first=True, bidirectional=options.lstm_dir in (Direction.BACK, Direction.BI))
        if mode == Mode.PRE_TRAIN:
            self.output_layer = nn.Linear(hidden_size * (2 if options.lstm_dir == Direction.BI else 1), self.num_event_types)
        if mode == Mode.PREDICT:
            # TODO: try multiple layers here
            self.pred_output_layer = nn.Linear(hidden_size * (2 if options.prediction_state in (PredictionState.BOTH_CONCAT, PredictionState.AVG) else 1), 1)

    def forward(self, batch):
        # import pdb; pdb.set_trace()
        # TODO: account for device to move to GPU
        batch_size = batch["event_types"].shape[0]
        # questions = self.question_embeddings(batch["question_ids"])
        question_types = self.question_type_embeddings(batch["question_types"])
        event_types = self.event_type_embeddings(batch["event_types"])
        time_deltas = batch["time_deltas"].unsqueeze(2) # Add a third dimension to be able to concat with embeddings

        lstm_input = torch.cat([question_types, event_types, time_deltas], dim=-1)
        # lstm_input = torch.cat([questions, question_types, event_types, time_deltas], dim=-1)
        # lstm_input = self.dropout(lstm_input)

        packed_lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input, lengths=batch["sequence_lengths"], batch_first=True, enforce_sorted=False)
        packed_lstm_output, (hidden, _) = self.lstm(packed_lstm_input)
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm_output, batch_first=True)

        if self.mode == Mode.PRE_TRAIN:
            # At each hidden state h_i, h_i_fwd contains info from inputs [x_0...x_i], and h_i_back contains info from inputs [x_i...x_n]
            # We would like to predict input x_i using info from [x_0...x_i-1] and [x_i+1...x_n]
            # So input x_i should be predicted by h_i-1_fwd and h_i+1_back
            # So the prediction matrix for each batch should look like [ [0; h_1_back], [h_0_fwd; h_2_back], ..., [h_n-1_fwd; 0] ]
            forward_output = torch.cat([torch.zeros(batch_size, 1, hidden_size), lstm_output[:, :-1, :hidden_size]], dim=1)
            if self.options.lstm_dir in (Direction.BACK, Direction.BI):
                backward_output = torch.cat([lstm_output[:, 1:, hidden_size:], torch.zeros(batch_size, 1, hidden_size)], dim=1)
                if self.options.lstm_dir == Direction.BACK:
                    full_output = backward_output
                if self.options.lstm_dir == Direction.BI:
                    full_output = torch.cat([forward_output, backward_output], dim=2)
            else:
                full_output = forward_output
            full_output *= batch["mask"].unsqueeze(2) # Mask to prevent info leakage at end of sequence before padding
            predictions = self.output_layer(full_output)
            # TODO: can mask predictions by only allowing event types that are allowed with corresponding question type
            event_types_1d = batch["event_types"].view(-1)
            mask = batch["mask"].view(-1)

            # Get cross-entropy loss of predictions with labels, note that this automatically performs the softmax step
            loss_fn = nn.CrossEntropyLoss(reduction="none")
            # Loss function expects 2d matrix, so compute with all sequences from all batches in single array
            loss = loss_fn(predictions.view(-1, self.num_event_types), event_types_1d)
            loss = loss * mask # Don't count loss for indices within the padding of the sequences
            avg_loss = loss.mean()

            # Get collapsed predictions
            predicted_event_types = torch.max(predictions, dim=-1)[1].view(-1) # Get indices of max values of predicted event vectors

            return avg_loss, predicted_event_types.detach().cpu().numpy()

        if self.mode == Mode.PREDICT:
            pred_state = None
            if self.options.prediction_state == PredictionState.AVG:
                pred_state = lstm_output.mean(dim=1)
            else:
                final_fwd_state = hidden[0]
                if self.options.prediction_state == PredictionState.LAST:
                    pred_state = final_fwd_state
                else:
                    final_back_state = hidden[1]
                    if self.options.prediction_state == PredictionState.FIRST:
                        pred_state = final_back_state
                    if self.options.prediction_state == PredictionState.BOTH_SUM:
                        pred_state = final_fwd_state + final_back_state
                    if self.options.prediction_state == PredictionState.BOTH_CONCAT:
                        pred_state = torch.cat([final_fwd_state, final_back_state], dim=-1)

                if self.options.attention:
                    pass # TODO

            predictions = self.pred_output_layer(pred_state).view(-1)
            labels = batch["labels"]

            # Get cross entropy loss of predictions with labels, note that this automatically performs the sigmoid step
            loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
            avg_loss = loss_fn(predictions, labels)

            return avg_loss, predictions.detach().cpu().numpy()
