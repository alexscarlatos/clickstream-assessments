from typing import Dict
from enum import Enum
import torch
from torch import nn
from constants import ASSISTIVE_EVENT_IDS
from utils import device

question_embedding_size = 32
question_type_embedding_size = 32
event_type_embedding_size = 32
hidden_size = 64

num_correctness_states = 4

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
        self.task: str = options.get("task", "comp")

        self.lr: float = options.get("lr", 1e-4)
        self.weight_decay: bool = options.get("weight_decay", 1e-6)
        self.epochs: int = options.get("epochs", 100)
        self.mixed_time: bool = options.get("mixed_time", False)
        self.random_trim: bool = options.get("random_trim", False)

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
    def __init__(self, mode: Mode, type_mappings: Dict[str, list], options: TrainOptions, available_qids: torch.BoolTensor = None):
        super().__init__()
        self.options = options
        self.mode = mode
        self.num_questions = len(type_mappings["question_ids"])
        self.available_qids = available_qids
        # num_question_types = len(type_mappings["question_types"])
        self.num_event_types = len(type_mappings["event_types"])
        self.question_embeddings = nn.Embedding(self.num_questions, question_embedding_size)
        # Question type gives redundant information when we have question ID
        # self.question_type_embeddings = nn.Embedding(num_question_types, question_type_embedding_size)
        self.event_type_embeddings = nn.Embedding(self.num_event_types, event_type_embedding_size)
        self.correctness_embeddings = torch.eye(num_correctness_states).to(device) # One-hot embedding for correctness states
        self.dropout = nn.Dropout(options.dropout)
        input_size = question_embedding_size + event_type_embedding_size + num_correctness_states + 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=options.lstm_dir in (Direction.BACK, Direction.BI)
        )
        if mode == Mode.PRE_TRAIN:
            output_size = hidden_size * (2 if options.lstm_dir == Direction.BI else 1)
            self.event_pred_layer = nn.Linear(output_size, self.num_event_types)
            self.time_pred_layer = nn.Linear(output_size, 1)
            self.qid_pred_layer = nn.Linear(output_size, self.num_questions)
            self.correctness_pred_layer = nn.Linear(output_size, 3)
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
        batch_size = batch["event_types"].shape[0]
        questions = self.question_embeddings(batch["question_ids"])
        # question_types = self.question_type_embeddings(batch["question_types"])
        event_types = self.event_type_embeddings(batch["event_types"])
        correctness = self.correctness_embeddings[batch["correctness"]]
        time_deltas = batch["time_deltas"].unsqueeze(2) # Add a third dimension to be able to concat with embeddings

        lstm_input = torch.cat([questions, event_types, correctness, time_deltas], dim=-1)
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
                elif self.options.lstm_dir == Direction.BI:
                    full_output = torch.cat([forward_output, backward_output], dim=2)
            else:
                full_output = forward_output
            full_output *= batch["mask"].unsqueeze(2) # Mask to prevent info leakage at end of sequence before padding

            # TODO: can mask predictions by only allowing event types that are allowed with corresponding question type
            event_predictions = self.event_pred_layer(full_output)
            # Get cross-entropy loss of predictions with labels, note that this automatically performs the softmax step
            event_loss_fn = nn.CrossEntropyLoss(reduction="none")
            # Loss function expects 2d matrix, so compute with all sequences from all batches in single array
            event_loss = event_loss_fn(event_predictions.view(-1, self.num_event_types), batch["event_types"].view(-1))

            time_predictions = self.time_pred_layer(full_output)
            # Get cross-entropy loss of time predictions with time interpolation at each step, sigmoid performed implicitly
            time_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            # All sequences unrolled into single array for loss calculation
            time_loss = time_loss_fn(time_predictions.view(-1), batch["time_ratios"].view(-1))

            # Get event-level prediction loss
            loss = event_loss + time_loss
            loss = loss * batch["mask"].view(-1) # Don't count loss for indices within the padding of the sequences
            avg_loss = loss.mean()

            # import pdb; pdb.set_trace()
            # Visit-level pretraining objectives - predict question id and correctness of each visit to each question
            # Follows similar process to event-level predictions above, so code is condensed
            final_idxs = batch["visits"]["idxs"][:, :-1] # The index of the last event in each visit, last index of last visit not used as prediction state so removed
            # The fwd state at the last index of the first visit is used to predict the second visit, and so on. Left pad since first visit has no preceeding info.
            fwd_states_at_final_idxs = torch.take_along_dim(lstm_output[:, :, :hidden_size], dim=1, indices=final_idxs.unsqueeze(2))
            visit_fwd_pred_states = torch.cat([torch.zeros(batch_size, 1, hidden_size).to(device), fwd_states_at_final_idxs], dim=1)
            if self.options.lstm_dir in (Direction.BACK, Direction.BI):
                # The back state at the first index of the second visit is used to predict the first visit, and so on. Right pad since last visit has no proceeding info.
                visit_start_idxs = torch.clamp(final_idxs + 1, max=lstm_output.shape[1] - 1) # Clamp to avoid overflow on last idx, which gets thrown out later anwyay
                back_states_at_first_idxs = torch.take_along_dim(lstm_output[:, :, hidden_size:], dim=1, indices=visit_start_idxs.unsqueeze(2))
                # Explicitly mask the back state at the final visit of each sequence to remove noise from index copy above
                back_states_at_first_idxs *= batch["visits"]["mask"][:, 1:].unsqueeze(2)
                visit_back_pred_states = torch.cat([back_states_at_first_idxs, torch.zeros(batch_size, 1, hidden_size).to(device)], dim=1)
                if self.options.lstm_dir == Direction.BACK:
                    visit_pred_states = visit_back_pred_states
                elif self.options.lstm_dir == Direction.BI:
                    visit_pred_states = torch.cat([visit_fwd_pred_states, visit_back_pred_states], dim=2)
            else:
                visit_pred_states = visit_fwd_pred_states
            visit_pred_states *= batch["visits"]["mask"].unsqueeze(2) # Mask out copied states in padded regions
            qid_predictions = self.qid_pred_layer(visit_pred_states)
            qid_predictions[:, :, self.available_qids == False] = -torch.inf # Don't assign probability to qids that aren't available
            qid_loss = nn.CrossEntropyLoss(reduction="none")(qid_predictions.view(-1, self.num_questions), batch["visits"]["qids"].view(-1))
            correctness_predictions = self.correctness_pred_layer(visit_pred_states)
            correctness_loss = nn.CrossEntropyLoss(reduction="none")(correctness_predictions.view(-1, 3), batch["visits"]["correctness"].view(-1))
            visit_loss = qid_loss + correctness_loss
            visit_loss = visit_loss * batch["visits"]["mask"].view(-1) # Don't count loss for padded regions
            avg_visit_loss = visit_loss.mean()

            # Get final loss
            final_avg_loss = avg_loss + avg_visit_loss

            # Get collapsed predictions
            predicted_event_types = torch.max(event_predictions, dim=-1)[1].view(-1).detach().cpu().numpy() # Get indices of max values of predicted event vectors
            predicted_qids = torch.max(qid_predictions, dim=-1)[1].view(-1).detach().cpu().numpy()
            predicted_correctness = torch.max(correctness_predictions, dim=-1)[1].view(-1).detach().cpu().numpy()

            return final_avg_loss, (predicted_event_types, predicted_qids, predicted_correctness)

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
