import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from constants import Mode, NUM_CORRECTNESS_STATES
from utils import device

num_heads = 1
num_layers = 2
hidden_dim = 200
event_type_embedding_size = 32
question_embedding_size = 31

class ClickBERT(nn.Module):
    def __init__(self, mode: Mode, type_mappings: dict, num_labels: int = 1):
        super().__init__()
        self.mode = mode
        self.num_labels = num_labels
        self.num_event_types = len(type_mappings["event_types"])
        self.num_questions = len(type_mappings["question_ids"])
        self.event_type_embeddings = nn.Embedding(self.num_event_types, event_type_embedding_size)
        self.question_embeddings = nn.Embedding(self.num_questions, question_embedding_size)
        # self.correctness_encoding = torch.eye(NUM_CORRECTNESS_STATES).to(device)
        # self.cls_embedding = nn.Embedding(1, event_type_embedding_size + question_embedding_size + NUM_CORRECTNESS_STATES)
        # self.mask_embedding = nn.Embedding(1, event_type_embedding_size + question_embedding_size + NUM_CORRECTNESS_STATES)
        # self.input_size = event_type_embedding_size + question_embedding_size + NUM_CORRECTNESS_STATES + 1
        self.cls_embedding = nn.Embedding(1, event_type_embedding_size + question_embedding_size)
        self.mask_embedding = nn.Embedding(1, event_type_embedding_size + question_embedding_size)
        self.input_size = event_type_embedding_size + question_embedding_size + 1

        self.pos_encoder = PositionalEncoding(self.input_size)
        encoder_layer = TransformerEncoderLayer(self.input_size, num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.event_pred_layer = nn.Linear(self.input_size, self.num_event_types)
        self.time_pred_layer = nn.Linear(self.input_size, 1)
        self.pred_layer = nn.Linear(self.input_size, self.num_labels)

    def forward(self, batch):
        # import pdb; pdb.set_trace()
        batch_size = batch["event_types"].shape[0]
        emb_zero_idx = torch.LongTensor([0]).to(device)

        # Construct input
        event_types = self.event_type_embeddings(batch["event_types"]) * math.sqrt(event_type_embedding_size)
        questions = self.question_embeddings(batch["question_ids"]) * math.sqrt(question_embedding_size)
        time_deltas = batch["time_deltas"].unsqueeze(2)
        tokens = torch.concat([event_types, questions, time_deltas], dim=2)

        # Apply mask tokens for pretraining
        if self.mode == Mode.PRE_TRAIN:
            mask_token = nn.functional.pad(self.mask_embedding(emb_zero_idx), (0,1))
            tokens[batch["pretrain_mask"]] = mask_token

        # Apply positional encoding
        tokens = self.pos_encoder(tokens)

        # Prepend CLS token to input
        cls_token = nn.functional.pad(self.cls_embedding(emb_zero_idx), (0,1)).expand(batch_size, 1, self.input_size)
        tokens = torch.concat([cls_token, tokens], dim=1)

        # Sequence padding, add one on left to account for CLS token
        mask = torch.concat([torch.ones((batch_size, 1), dtype=bool).to(device), batch["mask"]], dim=1)

        # Forward pass, negate mask (API doesn't compute attention where mask=True)
        output = self.transformer_encoder(tokens, src_key_padding_mask=~mask)

        if self.mode == Mode.PRE_TRAIN:
            # Run output through predictive layers for events and time
            # Calculate cross-entropy loss for all masked timesteps
            pretrain_mask = batch["pretrain_mask"].view(-1)
            event_predictions = self.event_pred_layer(output[:, 1:]).view(-1, self.num_event_types)
            event_loss = nn.CrossEntropyLoss()(event_predictions[pretrain_mask], batch["event_types"].view(-1)[pretrain_mask])
            time_predictions = self.time_pred_layer(output[:, 1:]).view(-1)
            time_loss = nn.BCEWithLogitsLoss()(time_predictions[pretrain_mask], batch["time_ratios"].view(-1)[pretrain_mask])
            final_loss = event_loss + time_loss
            predicted_event_types = torch.max(event_predictions, dim=-1)[1].detach().cpu().numpy() # Get indices of max values of predicted event vectors
            return final_loss, (predicted_event_types, None, None)
        else:
            # Run the CLS vector for each sequence through the predictive layer
            cls_output = output[:, 0]
            predictions = self.pred_layer(cls_output)
            if self.num_labels == 1:
                predictions = predictions.view(-1)
            else:
                predictions = predictions.view(-1, self.num_labels)
            # TODO: handle IRT and CLUSTER modes
            loss = nn.BCEWithLogitsLoss()(predictions, batch["labels"])
            return loss, predictions.detach().cpu().numpy()

class PositionalEncoding(nn.Module):
    """
    Taken directly from pytorch Transformer tutorial https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Modified to be batch-first
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
