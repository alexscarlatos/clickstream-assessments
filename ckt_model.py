from typing import Dict
import torch
from torch import nn
from utils import device
from constants import Mode, NUM_CORRECTNESS_STATES

encoding_size = 50

class CKTEncoder(nn.Module):
    """
    Model based on the encoding section from the Clickstream Knowledge Tracing paper
    Will train an encoder and decoder, given sequences from a single question across multiple students
    """

    hidden_size = 100

    def __init__(self, type_mappings: Dict[str, list], train_mode: bool, use_correctness: bool, available_event_types: torch.BoolTensor = None):
        super().__init__()
        self.train_mode = train_mode
        self.use_correctness = use_correctness
        self.num_event_types = len(type_mappings["event_types"])
        self.available_event_types = available_event_types
        self.event_embeddings = torch.eye(self.num_event_types).to(device)
        self.representation_size = self.num_event_types + 1
        if self.use_correctness:
            self.correctness_embeddings = torch.eye(NUM_CORRECTNESS_STATES).to(device)
            self.representation_size += NUM_CORRECTNESS_STATES
        self.encoder = nn.GRU(input_size=self.representation_size, hidden_size=self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.encoder_to_c = nn.Sequential(nn.Tanh(), nn.Linear(self.hidden_size, encoding_size))
        self.c_to_decoder = nn.Linear(encoding_size, self.hidden_size)
        self.decoder = nn.GRU(input_size=self.representation_size, hidden_size=self.hidden_size, batch_first=True)
        self.event_pred_layer = nn.Linear(self.hidden_size, self.num_event_types)
        self.time_pred_layer = nn.Linear(self.hidden_size, 1)
        if self.use_correctness:
            self.correctness_pred_layer = nn.Linear(self.hidden_size, 3)

    def forward(self, batch):
        batch_size = batch["event_types"].shape[0]
        event_types = self.event_embeddings[batch["event_types"]]
        time_deltas = batch["time_deltas"].unsqueeze(2)
        if self.use_correctness:
            correctness = self.correctness_embeddings[batch["correctness"]]
            rnn_input = torch.cat([event_types, correctness, time_deltas], dim=2)
        else:
            rnn_input = torch.cat([event_types, time_deltas], dim=2)
        packed_encoder_input = torch.nn.utils.rnn.pack_padded_sequence(
            rnn_input, lengths=batch["sequence_lengths"], batch_first=True, enforce_sorted=False)
        _, encoder_output = self.encoder(packed_encoder_input)
        if self.train_mode:
            encoder_output = self.dropout(encoder_output)
        encodings = self.encoder_to_c(encoder_output)

        if not self.train_mode:
            return encodings.view(-1, encoding_size)

        # Start with 0 vector so first output can predict the first event in the sequence
        decoder_input = torch.cat([torch.zeros(batch_size, 1, self.representation_size).to(device), rnn_input], dim=1)
        # By passing original sequence_lengths, will discard final idx output, but we don't use it for predictions anyways
        packed_decoder_input = torch.nn.utils.rnn.pack_padded_sequence(
            decoder_input, lengths=batch["sequence_lengths"], batch_first=True, enforce_sorted=False)
        decoder_start_state = self.c_to_decoder(encodings)
        packed_decoder_output, _ = self.decoder(packed_decoder_input, decoder_start_state)
        decoder_output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_decoder_output, batch_first=True)

        event_preds = self.event_pred_layer(decoder_output)
        event_preds[:, :, self.available_event_types == False] = -torch.inf # Don't assign probability to impossible event types
        time_preds = self.time_pred_layer(decoder_output)
        event_loss = nn.CrossEntropyLoss(reduction="none")(event_preds.view(-1, self.num_event_types), batch["event_types"].view(-1))
        time_loss = nn.MSELoss(reduction="none")(time_preds.view(-1), batch["time_deltas"].view(-1))
        final_loss = event_loss + time_loss
        if self.use_correctness:
            correctness_preds = self.correctness_pred_layer(decoder_output)
            correctness_loss = nn.CrossEntropyLoss(reduction="none")(correctness_preds.view(-1, 3), batch["correctness"].view(-1))
            final_loss += correctness_loss
        final_loss *= batch["mask"].view(-1) # Don't count loss for indices within the padding of the sequences
        avg_loss = final_loss.mean()

        predicted_event_types = torch.max(event_preds, dim=-1)[1].view(-1).detach().cpu().numpy() # Get indices of max values of predicted event vectors
        predicted_correctness = torch.max(correctness_preds, dim=-1)[1].view(-1).detach().cpu().numpy() if self.use_correctness else None

        return avg_loss, (predicted_event_types, None, predicted_correctness)

class CKTJoint(nn.Module):
    """
    Contains CKT-based encoder/decoder for each question
    For per-student labels - Predictor network that operate on arrays of sub-sequences
        Multi-layer network for concatenated visits, or RNN for sequential visits
    For IRT - Predictor network that operates on one question at a time
    """

    pred_mlp_hidden_size = 100
    rnn_hidden_size = 100
    irt_mlp_hidden_size = 50

    def __init__(self, mode: Mode, type_mappings: Dict[str, list], use_correctness: bool, available_event_types: Dict[int, torch.BoolTensor],
                 concat_visits: bool = False, num_labels: int = 1, num_input_qids: int = 0):
        super().__init__()
        self.mode = mode
        self.num_labels = num_labels
        question_ids = sorted(type_mappings["question_ids"].values())

        # Create an encoder for each question
        self.encoder = nn.ModuleDict()
        for qid in question_ids:
            self.encoder[str(qid)] = CKTEncoder(type_mappings, self.mode == Mode.CKT_ENCODE, use_correctness, available_event_types[qid])

        if self.mode == Mode.PREDICT:
            self.concat_visits = concat_visits
            if self.concat_visits:
                # Construct multi-layer network that takes concatenated encodings of each question type in sequence
                self.predictor = nn.Sequential(
                    nn.Linear(num_input_qids * encoding_size, self.pred_mlp_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(self.pred_mlp_hidden_size, num_labels)
                )
            else:
                # Construct RNN that takes encoding and question id of each visit
                num_questions = len(question_ids)
                self.question_embeddings = torch.eye(num_questions).to(device)
                input_size = encoding_size + num_questions
                self.rnn = nn.GRU(input_size=input_size, hidden_size=self.rnn_hidden_size, batch_first=True)
                self.pred_layer = nn.Sequential(
                    nn.Dropout(0.25), nn.ReLU(), nn.Linear(self.rnn_hidden_size, num_labels)
                )

        if self.mode == Mode.IRT:
            # Construct multi-layer network that takes single question encoding plus one-hot question indicator
            self.question_encodings = torch.eye(len(question_ids)).to(device)
            self.predictor = nn.Sequential(
                nn.Linear(encoding_size + len(question_ids), self.irt_mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(self.irt_mlp_hidden_size, 1)
            )

    def load_params(self, pretrained_state_dict: dict):
        """
        Load the params from the given state dict
        Works for just loading a pre-trained encoder or for loading a full predictor model
        """
        state_dict = self.state_dict()
        for param_name, param_val in pretrained_state_dict.items():
            state_dict[param_name] = param_val
        self.load_state_dict(state_dict)

    def forward(self, batch):
        batch_size = len(batch["sequence_lengths"])

        # When training encoder, just return encoder network's output
        if self.mode == Mode.CKT_ENCODE:
            qid = batch["data_class"]
            return self.encoder[qid](batch)

        if self.mode == Mode.PREDICT:
            max_seq_len = max(batch["sequence_lengths"])
            all_encodings = torch.zeros((batch_size * max_seq_len, encoding_size)).to(device)
            # Run each question-specific encoder on all sub-sequences of that question
            for qid, sub_batch in batch["questions"].items():
                encodings = self.encoder[qid](sub_batch)
                # Assign resulting encodings to their corresponding places in the encoding matrix
                all_encodings[sub_batch["target_idxs"]] = encodings

            # Generate predictions from encodings
            if self.concat_visits:
                predictions = self.predictor(all_encodings.view(batch_size, max_seq_len * encoding_size))
            else:
                qids = self.question_embeddings[batch["question_ids"]]
                rnn_input = torch.cat([all_encodings.view(batch_size, max_seq_len, encoding_size), qids], dim=2)
                packed_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(
                    rnn_input, lengths=batch["sequence_lengths"], batch_first=True, enforce_sorted=False)
                _, rnn_output = self.rnn(packed_rnn_input)
                predictions = self.pred_layer(rnn_output)

            # Get final predictions and loss
            if self.num_labels == 1:
                predictions = predictions.view(-1)
            else:
                predictions = predictions.view(-1, self.num_labels)
            avg_loss = nn.BCEWithLogitsLoss(reduction="mean")(predictions, batch["labels"])
            return avg_loss, predictions.detach().cpu().numpy()

        if self.mode == Mode.IRT:
            qid = batch["data_class"]
            # Get sequence encodings
            encodings = self.encoder[qid](batch)
            # Get question encoding for class and repeat for each sequence in batch, note that expand does not allocate memory
            question_indicators = self.question_encodings[int(qid)].expand(batch_size, self.question_encodings.shape[1])
            # Pass input through predictor network
            pred_input = torch.cat([encodings, question_indicators], dim=1)
            predictions = self.predictor(pred_input).view(-1)
            return predictions
