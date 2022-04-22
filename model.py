from typing import Dict
import torch
from torch import nn
from constants import Mode, Direction, PredictionState, TrainOptions, ASSISTIVE_EVENT_IDS, NUM_CORRECTNESS_STATES
from utils import device

question_embedding_size = 16
event_type_embedding_size = 16
hidden_size = 100

class LSTMModel(nn.Module):
    def __init__(self, mode: Mode, type_mappings: Dict[str, list], options: TrainOptions, available_qids: torch.BoolTensor = None,
                 num_labels: int = 1, pred_classes: list = None, use_qid: bool = True):
        super().__init__()
        self.options = options
        self.mode = mode
        self.use_qid = use_qid
        self.num_questions = len(type_mappings["question_ids"])
        self.available_qids = available_qids
        self.num_event_types = len(type_mappings["event_types"])
        self.question_embeddings = nn.Embedding(self.num_questions, question_embedding_size)
        self.event_type_embeddings = nn.Embedding(self.num_event_types, event_type_embedding_size)

        self.use_correctness = options.use_correctness
        input_size = event_type_embedding_size + 1
        if self.use_qid:
            input_size += question_embedding_size
        if self.use_correctness:
            self.correctness_embeddings = torch.eye(NUM_CORRECTNESS_STATES).to(device) # One-hot embedding for correctness states
            input_size += NUM_CORRECTNESS_STATES
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=options.lstm_dir in (Direction.BACK, Direction.BI)
        )
        if mode == Mode.PRE_TRAIN:
            # Linear projections to predict input
            output_size = hidden_size * (2 if options.lstm_dir == Direction.BI else 1)
            self.event_pred_layer = nn.Linear(output_size, self.num_event_types)
            self.time_pred_layer = nn.Linear(output_size, 1)
            self.qid_pred_layer = nn.Linear(output_size, self.num_questions)
            self.correctness_pred_layer = nn.Linear(output_size, 3)
        if mode in (Mode.PREDICT, Mode.CLUSTER, Mode.IRT):
            self.num_labels = num_labels
            output_size = hidden_size * (2 if options.prediction_state in (PredictionState.BOTH_CONCAT, PredictionState.ATTN, PredictionState.AVG) else 1)
            final_layer_size = output_size + (7 + len(ASSISTIVE_EVENT_IDS) if options.engineered_features else 0)

            if options.multi_head: # Create separate output network for each data class
                self.attention = nn.ModuleDict()
                self.hidden_layers = nn.ModuleDict()
                self.pred_output_layer = nn.ModuleDict()
                all_classes = pred_classes or ["10", "20", "30"]
                for data_class in all_classes:
                    self.attention[data_class] = nn.Linear(output_size, 1)
                    self.hidden_layers[data_class] = nn.Sequential(
                        nn.Dropout(options.dropout), nn.ReLU(), nn.Linear(output_size, output_size))
                    self.pred_output_layer[data_class] = nn.Sequential(
                        nn.Dropout(options.dropout), nn.Linear(final_layer_size, num_labels))
            else:
                self.attention = nn.Linear(output_size, 1)
                self.hidden_layers = nn.Sequential(
                    nn.Dropout(options.dropout), nn.ReLU(), nn.Linear(output_size, output_size))
                self.pred_output_layer = nn.Sequential(
                    nn.Dropout(options.dropout), nn.Linear(final_layer_size, num_labels))

    def forward(self, batch):
        batch_size = batch["event_types"].shape[0]

        # Construct input and run through LSTM
        event_types = self.event_type_embeddings(batch["event_types"])
        time_deltas = batch["time_deltas"].unsqueeze(2) # Add a third dimension to be able to concat with embeddings
        input_tensors = [event_types, time_deltas]
        if self.use_qid:
            questions = self.question_embeddings(batch["question_ids"])
            input_tensors.append(questions)
        if self.use_correctness:
            correctness = self.correctness_embeddings[batch["correctness"]]
            input_tensors.append(correctness)
        lstm_input = torch.cat(input_tensors, dim=-1)
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

            # Prediction loss at each time step
            loss = torch.zeros(batch_size * full_output.shape[1]).to(device)

            event_predictions = None
            if self.options.predict_event_type:
                event_predictions = self.event_pred_layer(full_output)
                # Get cross-entropy loss of predictions with labels, note that this automatically performs the softmax step
                event_loss_fn = nn.CrossEntropyLoss(reduction="none")
                # Loss function expects 2d matrix, so compute with all sequences from all batches in single array
                event_loss = event_loss_fn(event_predictions.view(-1, self.num_event_types), batch["event_types"].view(-1))
                loss += event_loss

            time_predictions = None
            if self.options.predict_time:
                time_predictions = self.time_pred_layer(full_output)
                # Get cross-entropy loss of time predictions with time interpolation at each step, sigmoid performed implicitly
                time_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
                # All sequences unrolled into single array for loss calculation
                time_loss = time_loss_fn(time_predictions.view(-1), batch["time_ratios"].view(-1))
                loss += time_loss

            # For per-question architecture, predict visit-level features at each time step
            qid_predictions = None
            correctness_predictions = None
            if self.options.per_q_arch:
                if self.options.predict_qid:
                    qid_predictions = self.qid_pred_layer(full_output)
                    qid_loss = nn.CrossEntropyLoss(reduction="none")(qid_predictions.view(-1, self.num_questions), batch["question_ids"].view(-1))
                    loss += qid_loss
                if self.use_correctness and self.options.predict_correctness:
                    correctness_predictions = self.correctness_pred_layer(full_output)
                    correctness_loss = nn.CrossEntropyLoss(reduction="none")(correctness_predictions.view(-1, 3), batch["correctness"].view(-1))
                    loss += correctness_loss

            # Get event-level prediction loss
            loss *= batch["mask"].view(-1) # Don't count loss for indices within the padding of the sequences
            avg_loss = loss.mean()

            # Visit-level pretraining objectives - predict question id and correctness of each visit to each question
            # Follows similar process to event-level predictions above, so code is condensed
            if self.options.use_visit_pt_objs:
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
                if self.use_correctness:
                    correctness_predictions = self.correctness_pred_layer(visit_pred_states)
                    correctness_loss = nn.CrossEntropyLoss(reduction="none")(correctness_predictions.view(-1, 3), batch["visits"]["correctness"].view(-1))
                    visit_loss = qid_loss + correctness_loss
                else:
                    visit_loss = qid_loss
                visit_loss = visit_loss * batch["visits"]["mask"].view(-1) # Don't count loss for padded regions
                avg_visit_loss = visit_loss.mean()
                avg_loss += avg_visit_loss

            # Get collapsed predictions, take index of maximally predicted class for each sequence
            predicted_event_types = torch.max(event_predictions, dim=-1)[1].view(-1).detach().cpu().numpy() if event_predictions is not None else None
            predicted_qids = torch.max(qid_predictions, dim=-1)[1].view(-1).detach().cpu().numpy() if qid_predictions is not None else None
            predicted_correctness = torch.max(correctness_predictions, dim=-1)[1].view(-1).detach().cpu().numpy() if correctness_predictions is not None else None

            return avg_loss, (predicted_event_types, predicted_qids, predicted_correctness)

        if self.mode in (Mode.PREDICT, Mode.CLUSTER, Mode.IRT):
            data_class = batch.get("data_class")
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

            # For per-question architecture, return prediction state for use by higher level model
            if self.options.per_q_arch:
                return pred_state

            predictions = pred_output_layer(pred_state)
            if self.num_labels == 1:
                predictions = predictions.view(-1)
            else:
                predictions = predictions.view(-1, self.num_labels)

            if self.mode == Mode.IRT:
                return predictions

            if self.mode == Mode.PREDICT:
                # Get cross entropy loss of predictions with labels, note that this automatically performs the sigmoid step
                loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
                avg_loss = loss_fn(predictions, batch["labels"])

                return avg_loss, predictions.detach().cpu().numpy()

            if self.mode == Mode.CLUSTER:
                return pred_state, predictions
