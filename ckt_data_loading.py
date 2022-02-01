from typing import Dict, List
import torch
import numpy as np
from ckt_model import encoding_size
from data_loading import add_visit_level_features_and_correctness
from utils import device

def get_sub_sequences(sequence, question_ids: set, concat_visits = False):
    # Split a sequence into a list of subsequences by visit
    add_visit_level_features_and_correctness(sequence)
    start_idx = 0
    sub_sequences = []
    for last_idx in sequence["visits"]["idxs"]:
        qid = sequence["question_ids"][start_idx]
        if qid in question_ids:
            sub_sequence = {
                "student_id": sequence["student_id"],
                "question_id": qid,
                "event_types": sequence["event_types"][start_idx : last_idx + 1],
                # Convert to log2 as per CKT paper, add 1 to avoid log(0)
                "time_deltas": np.log2(np.array(sequence["time_deltas"][start_idx : last_idx + 1]) + 1)
            }
            sub_sequences.append(sub_sequence)
        start_idx = last_idx + 1

    # If requested, concatenate visits per qid
    if concat_visits:
        qid_to_sub_sequences = {}
        for sub_seq in sub_sequences:
            if sub_seq["question_id"] not in qid_to_sub_sequences:
                qid_to_sub_sequences[sub_seq["question_id"]] = sub_seq
            else:
                qid_sub_seqs = qid_to_sub_sequences[sub_seq["question_id"]]
                qid_sub_seqs["event_types"] += sub_seq["event_types"]
                qid_sub_seqs["time_deltas"] = np.concatenate([qid_sub_seqs["time_deltas"], sub_seq["time_deltas"]])
        sub_sequences = list(qid_to_sub_sequences.values())

    return sub_sequences

class CKTEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict[str, list]], question_id: int, concat_visits: bool):
        self.data = []
        for sequence in data:
            self.data += get_sub_sequences(sequence, {question_id}, concat_visits)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class CKTEncoderCollator:
    def __init__(self):
        pass

    def __call__(self, batch: List[dict]):
        event_type_batches = []
        time_delta_batches = []
        mask = []

        for sequence in batch:
            event_type_batches.append(torch.LongTensor(sequence["event_types"]))
            time_delta_batches.append(torch.from_numpy(sequence["time_deltas"]).type(torch.float32))
            mask.append(torch.ones(len(sequence["event_types"])))

        return {
            "event_types": torch.nn.utils.rnn.pad_sequence(event_type_batches, batch_first=True).to(device),
            "time_deltas": torch.nn.utils.rnn.pad_sequence(time_delta_batches, batch_first=True).to(device),
            "mask": torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).to(device),
            "sequence_lengths": torch.LongTensor([seq.shape[0] for seq in event_type_batches]) # Must be on CPU
        }

class CKTPredictorDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict[str, list]], models: dict, labels: Dict[str, bool], allowed_qids: set, concat_visits: bool):
        sub_seq_collator = CKTEncoderCollator()
        self.data = []
        print("Encoding sub-sequences...")
        with torch.no_grad():
            for sequence in data:
                encodings = []
                question_ids = []
                sub_seqs = get_sub_sequences(sequence, allowed_qids, concat_visits)
                if concat_visits: # Ensure order with concat_visits since encodings will be fed through linear NN layer
                    sub_seqs.sort(key=lambda sub_seq: sub_seq["question_id"])
                for sub_seq in sub_seqs:
                    collated_batch = sub_seq_collator([sub_seq]) # Collate data into encoder model's expected format
                    encodings.append(models[sub_seq["question_id"]](collated_batch).view(encoding_size)) # Run model on subsequence for encoding
                    question_ids.append(sub_seq["question_id"])

                # Insert blank encodings for missing questions
                if concat_visits:
                    for q_idx, qid in enumerate(sorted(allowed_qids)):
                        if len(question_ids) <= q_idx or question_ids[q_idx] != qid:
                            encodings.insert(q_idx, torch.zeros(encoding_size).to(device))
                            question_ids.insert(q_idx, qid)

                self.data.append({
                    "encodings": encodings,
                    "question_ids": question_ids,
                    "label": labels[str(sequence["student_id"])]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class CKTPredictorCollator:
    def __init__(self):
        pass

    def __call__(self, batch: List[dict]):
        encoding_batches = []
        question_id_batches = []
        labels = []

        for sequence in batch:
            encoding_batches.append(torch.vstack(sequence["encodings"]))
            question_id_batches.append(torch.LongTensor(sequence["question_ids"]))
            labels.append(sequence["label"])
        
        return {
            "encodings": torch.nn.utils.rnn.pad_sequence(encoding_batches, batch_first=True).to(device),
            "question_ids": torch.nn.utils.rnn.pad_sequence(question_id_batches, batch_first=True).to(device),
            "labels": torch.Tensor(labels).to(device),
            "sequence_lengths": [encodings.shape[0] for encodings in encoding_batches]
        }
