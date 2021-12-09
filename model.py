from typing import Dict
from enum import Enum
import torch
from torch import nn
from constants import ASSISTIVE_EVENT_IDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

question_embedding_size = 16
question_type_embedding_size = 128
event_type_embedding_size = 128
hidden_size = 128

class Mode(Enum):
    PRE_TRAIN = 1
    PREDICT = 2
    CLUSTER = 3

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
    ATTN = 6

class TrainOptions:
    def __init__(self, options: dict):
        self.lr: float = options.get("lr", 1e-4)
        self.weight_decay: bool = options.get("weight_decay", 1e-6)
        self.split_data: bool = options.get("split_data", False)

        self.lstm_dir: Direction = options.get("lstm_dir", Direction.BI)
        self.use_pretrained_weights: bool = options.get("pretrained_model", True)
        self.use_pretrained_embeddings: bool = options.get("pretrained_emb", True)
        self.use_pretrained_head: bool = options.get("pretrained_head", False)
        self.freeze_model: bool = options.get("freeze_model", True)
        self.freeze_embeddings: bool = options.get("freeze_emb", True)
        self.prediction_state: PredictionState = options.get("pred_state", PredictionState.BOTH_CONCAT)
        self.dropout: float = options.get("dropout", 0.25)
        self.hidden_ff_layer: bool = options.get("hidden_ff_layer", False)
        self.engineered_features: bool = options.get("eng_feat", False)
        self.multi_head: bool = options.get("multi_head", False)

class LSTMModel(nn.Module):
    def __init__(self, mode: Mode, type_mappings: Dict[str, list], options: TrainOptions):
        super().__init__()
        self.options = options
        self.mode = mode
        num_questions = len(type_mappings["question_ids"])
        num_question_types = len(type_mappings["question_types"])
        self.num_event_types = len(type_mappings["event_types"])
        self.question_embeddings = nn.Embedding(num_questions, question_embedding_size)
        self.question_type_embeddings = nn.Embedding(num_question_types, question_type_embedding_size)
        self.event_type_embeddings = nn.Embedding(self.num_event_types, event_type_embedding_size)
        self.dropout = nn.Dropout(options.dropout)
        # [question; question type; event type; time delta]
        # self.input_size = question_type_embedding_size + event_type_embedding_size + 1
        self.input_size = question_embedding_size + question_type_embedding_size + event_type_embedding_size + 1
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=options.lstm_dir in (Direction.BACK, Direction.BI)
        )
        if mode == Mode.PRE_TRAIN:
            self.output_layer = nn.Linear(hidden_size * (2 if options.lstm_dir == Direction.BI else 1), self.num_event_types)
        if mode in (Mode.PREDICT, Mode.CLUSTER):
            output_size = hidden_size * (2 if options.prediction_state in (PredictionState.BOTH_CONCAT, PredictionState.ATTN, PredictionState.AVG) else 1)
            final_layer_size = output_size + (3 + len(ASSISTIVE_EVENT_IDS) if options.engineered_features else 0)
            if options.multi_head:
                self.attention = nn.ModuleDict()
                self.hidden_layers = nn.ModuleDict()
                self.pred_output_layer = nn.ModuleDict()
                for data_class in ["10", "20", "30"]:
                    self.attention[data_class] = nn.Linear(output_size, 1)
                    self.hidden_layers[data_class] = nn.Sequential(
                        nn.Dropout(options.dropout), nn.ReLU(), nn.Linear(output_size, output_size))
                    self.pred_output_layer[data_class] = nn.Sequential(
                        nn.Dropout(options.dropout), nn.ReLU(), nn.Linear(final_layer_size, 1))
            else:
                self.attention = nn.Linear(output_size, 1)
                self.hidden_layers = nn.Sequential(
                    nn.Dropout(options.dropout), nn.ReLU(), nn.Linear(output_size, output_size))
                self.pred_output_layer = nn.Sequential(
                    nn.Dropout(options.dropout), nn.ReLU(), nn.Linear(final_layer_size, 1))

    def forward(self, batch):
        # import pdb; pdb.set_trace()
        batch_size = batch["event_types"].shape[0]
        questions = self.question_embeddings(batch["question_ids"])
        question_types = self.question_type_embeddings(batch["question_types"])
        event_types = self.event_type_embeddings(batch["event_types"])
        time_deltas = batch["time_deltas"].unsqueeze(2) # Add a third dimension to be able to concat with embeddings

        # lstm_input = torch.cat([question_types, event_types, time_deltas], dim=-1)
        lstm_input = torch.cat([questions, question_types, event_types, time_deltas], dim=-1)
        # TODO: is this the right condition to not do dropout?
        # if not self.options.freeze_model:
        #     lstm_input = self.dropout(lstm_input)

        packed_lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input, lengths=batch["sequence_lengths"], batch_first=True, enforce_sorted=False)
        packed_lstm_output, (hidden, _) = self.lstm(packed_lstm_input)
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm_output, batch_first=True)

        if self.mode == Mode.PRE_TRAIN:
            # At each hidden state h_i, h_i_fwd contains info from inputs [x_0...x_i], and h_i_back contains info from inputs [x_i...x_n]
            # We would like to predict input x_i using info from [x_0...x_i-1] and [x_i+1...x_n]
            # So input x_i should be predicted by h_i-1_fwd and h_i+1_back
            # So the prediction matrix for each batch should look like [ [0; h_1_back], [h_0_fwd; h_2_back], ..., [h_n-1_fwd; 0] ]
            forward_output = torch.cat([torch.zeros(batch_size, 1, hidden_size).to(device), lstm_output[:, :-1, :hidden_size]], dim=1)
            if self.options.lstm_dir in (Direction.BACK, Direction.BI):
                backward_output = torch.cat([lstm_output[:, 1:, hidden_size:], torch.zeros(batch_size, 1, hidden_size).to(device)], dim=1)
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

        if self.mode in (Mode.PREDICT, Mode.CLUSTER):
            data_class = batch["data_class"]
            attention = self.attention[data_class] if self.options.multi_head else self.attention
            hidden_layers = self.hidden_layers[data_class] if self.options.multi_head else self.hidden_layers
            pred_output_layer = self.pred_output_layer[data_class] if self.options.multi_head else self.pred_output_layer

            pred_state = None
            if self.options.prediction_state == PredictionState.ATTN:
                # Multiply each output vector with learnable attention vector to get attention activations at each timestep
                activations = attention(lstm_output).squeeze(2) # batch_size x max_seq_len
                # Apply mask so that output in padding regions gets 0 probability after softmax
                activations[batch["mask"] == 0] = -torch.inf
                # Apply softmax to get distribution across timesteps of each sequence
                attention_weights = nn.Softmax(dim=1)(activations) # batch_size x max_seq_len
                # Multiply each output vector with its corresponding attention weight
                weighted_output = lstm_output * attention_weights.unsqueeze(2)
                # Add weighted output vectors along each sequence in the batch
                pred_state = torch.sum(weighted_output, dim=1)
            elif self.options.prediction_state == PredictionState.AVG:
                pred_state = lstm_output.mean(dim=1) # Average output vectors along each sequence in the batch
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

            # Pass output through hidden layer if needed
            if self.options.hidden_ff_layer:
                pred_state = hidden_layers(pred_state)
            # Append engineered features to latent state if needed (note that we don't want this )
            if self.options.engineered_features:
                pred_state = torch.cat([pred_state, batch["engineered_features"]], dim=1)
            predictions = pred_output_layer(pred_state).view(-1)

            if self.mode == Mode.PREDICT:
                # Get cross entropy loss of predictions with labels, note that this automatically performs the sigmoid step
                loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
                avg_loss = loss_fn(predictions, batch["labels"])

                return avg_loss, predictions.detach().cpu().numpy()

            if self.mode == Mode.CLUSTER:
                return pred_state, predictions
