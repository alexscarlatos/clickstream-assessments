from typing import Dict
import torch
from torch import nn
from utils import device
from constants import Mode, TrainOptions
from model import LSTMModel, hidden_size

class JointModel(nn.Module):
    """
    Encodes questions using a joint LSTM model
    Then passes representations through MLP for predictions
    """
    pred_mlp_hidden_size = 100
    rnn_hidden_size = 100
    irt_mlp_hidden_size = 50

    def __init__(self, mode: Mode, type_mappings: Dict[str, list], options: TrainOptions, num_labels: int = 1, num_input_qids: int = None):
        super().__init__()
        self.mode = mode
        self.num_labels = num_labels
        question_ids = sorted(type_mappings["question_ids"].values())
        self.encoding_size = hidden_size * 2
        self.concat_visits = options.concat_visits
        self.encoder = LSTMModel(mode, type_mappings, options)

        if self.mode in (Mode.PREDICT, Mode.CLUSTER):
            if self.concat_visits:
                # Construct multi-layer network that takes concatenated encodings of each question type in sequence
                self.predictor = nn.Sequential(
                    nn.Linear(num_input_qids * self.encoding_size, self.pred_mlp_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(self.pred_mlp_hidden_size, num_labels)
                )
            else:
                # Construct RNN that takes encoding and question id of each visit
                # num_questions = len(question_ids)
                # self.question_embeddings = torch.eye(num_questions).to(device)
                # input_size = self.encoding_size + num_questions
                input_size = self.encoding_size
                self.rnn = nn.GRU(input_size=input_size, hidden_size=self.rnn_hidden_size, batch_first=True)
                self.pred_layer = nn.Sequential(
                    nn.Dropout(0.25), nn.ReLU(), nn.Linear(self.rnn_hidden_size, num_labels)
                )

    def load_params(self, pretrained_state_dict: dict):
        """
        Load the params from the given state dict
        Works for just loading a pre-trained encoder or for loading a full predictor model
        """
        state_dict = self.state_dict()
        for param_name, param_val in pretrained_state_dict.items():
            if param_name in state_dict:
                state_dict[param_name] = param_val
        self.load_state_dict(state_dict)

    def forward(self, batch):
        batch_size = len(batch["sequence_lengths"])

        # When training encoder, just return encoder network's output
        # Assume batched by data_class and that data_class holds question_id
        if self.mode == Mode.PRE_TRAIN:
            return self.encoder(batch)

        max_seq_len = max(batch["sequence_lengths"])
        all_encodings = torch.zeros((batch_size * max_seq_len, self.encoding_size)).to(device)
        # Run each question-specific encoder on all sub-sequences of that question
        for qid, sub_batch in batch["questions"].items():
            encodings = self.encoder(sub_batch)
            # Assign resulting encodings to their corresponding places in the encoding matrix
            all_encodings[sub_batch["target_idxs"]] = encodings

        pred_state = all_encodings.view(batch_size, max_seq_len * self.encoding_size)

        # Generate predictions from encodings
        if self.concat_visits:
            predictions = self.predictor(pred_state)
        else:
            # qids = self.question_embeddings[batch["question_ids"]]
            # rnn_input = torch.cat([all_encodings.view(batch_size, max_seq_len, self.encoding_size), qids], dim=2)
            rnn_input = all_encodings.view(batch_size, max_seq_len, self.encoding_size)
            packed_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(
                rnn_input, lengths=batch["sequence_lengths"], batch_first=True, enforce_sorted=False)
            _, rnn_output = self.rnn(packed_rnn_input)
            predictions = self.pred_layer(rnn_output)

        if self.num_labels == 1:
            predictions = predictions.view(-1)
        else:
            predictions = predictions.view(-1, self.num_labels)

        if self.mode == Mode.PREDICT:
            avg_loss = nn.BCEWithLogitsLoss(reduction="mean")(predictions, batch["labels"])
            return avg_loss, predictions.detach().cpu().numpy()

        if self.mode == Mode.CLUSTER:
            return pred_state, predictions
