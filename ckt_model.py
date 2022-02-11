from typing import Dict
import torch
from torch import nn
from utils import device
from constants import Mode

# TODO: way to restructure for fine-tuning
# single model class, where encoder/decoder params are param dicts from question id

encoding_size = 50

class CKTEncoder(nn.Module):
    """
    Model based on the encoding section from the Clickstream Knowledge Tracing paper
    Will train an encoder and decoder, given sequences from a single question across multiple students
    """

    hidden_size = 100

    def __init__(self, type_mappings: Dict[str, list], train_mode: bool, available_event_types: torch.BoolTensor = None):
        super().__init__()
        self.train_mode = train_mode
        self.num_event_types = len(type_mappings["event_types"])
        self.available_event_types = available_event_types
        self.event_embeddings = torch.eye(self.num_event_types).to(device)
        # TODO: partial score, aloc embeddings and answer state embeddings
        self.representation_size = self.num_event_types + 1
        self.encoder = nn.GRU(input_size=self.representation_size, hidden_size=self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.encoder_to_c = nn.Sequential(nn.Tanh(), nn.Linear(self.hidden_size, encoding_size))
        self.c_to_decoder = nn.Linear(encoding_size, self.hidden_size)
        self.decoder = nn.GRU(input_size=self.representation_size, hidden_size=self.hidden_size, batch_first=True)
        self.event_pred_layer = nn.Linear(self.hidden_size, self.num_event_types)
        self.time_pred_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, batch):
        batch_size = batch["event_types"].shape[0]
        event_types = self.event_embeddings[batch["event_types"]]
        time_deltas = batch["time_deltas"].unsqueeze(2)
        # TODO: aloc embeddings and answer state embeddings
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
        # By passing original sequence_lengths, will disregard final idx output, but we don't use it for predictions anyways
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
        final_loss *= batch["mask"].view(-1) # Don't count loss for indices within the padding of the sequences
        avg_loss = final_loss.mean()

        predicted_event_types = torch.max(event_preds, dim=-1)[1].view(-1).detach().cpu().numpy() # Get indices of max values of predicted event vectors

        return avg_loss, predicted_event_types

class CKTPredictor(nn.Module):
    """
    Train a multi-class predictor on top of encoded sequences
    """

    mlp_hidden_size = 100
    rnn_hidden_size = 100

    def __init__(self, concat_visits: bool, num_labels: int, type_mappings: dict, block_a_qids: list):
        super().__init__()
        self.concat_visits = concat_visits
        self.num_labels = num_labels
        if concat_visits:
            # Construct multi-layer perceptron network that takes concatenated encodings of each question type in sequence
            self.num_questions = len(block_a_qids)
            self.mlp = nn.Sequential(
                nn.Linear(self.num_questions * encoding_size, self.mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(self.mlp_hidden_size, num_labels)
            )
        else:
            # Construct RNN that takes encoding and question id of each visit
            self.num_questions = len(type_mappings["question_ids"])
            self.question_embeddings = torch.eye(self.num_questions).to(device)
            input_size = encoding_size + self.num_questions
            self.rnn = nn.GRU(input_size=input_size, hidden_size=self.rnn_hidden_size, batch_first=True)
            self.pred_layer = nn.Sequential(
                nn.Dropout(0.25), nn.ReLU(), nn.Linear(self.rnn_hidden_size, num_labels)
            )

    def forward(self, batch):
        if self.concat_visits:
            # Pass encodings through MLP
            mlp_input = batch["encodings"].view(-1, self.num_questions * encoding_size) # Unroll list of question encodings into single vector per student
            predictions = self.mlp(mlp_input)
        else:
            # Pass encodings through RNN
            qids = self.question_embeddings[batch["question_ids"]]
            rnn_input = torch.cat([batch["encodings"], qids], dim=2)
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

class CKTJoint(nn.Module):
    """
    A CKT-based encoder that represents all questions
    """

    mlp_hidden_size = 100
    rnn_hidden_size = 100

    def __init__(self, mode: Mode):
        super().__init__()
        self.mode = mode
        # TODO: (maybe, worth it?) have single encoder/decoder model that processes all questions, can append ont-hot qid to encoded vector for reconstruction
        # TODO: then, have map of qid to separate encoder/decoders, what we'll do is use the batch_sampler from before to send only one qid per batch

    def forward(self, batch):
        pass
