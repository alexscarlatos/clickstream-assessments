from typing import Dict, List, Set
import torch
from data_loading import get_sub_sequences, add_visit_level_features_and_correctness
from utils import device

class CKTPredictorDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict[str, list]], labels: Dict[str, bool], allowed_qids: set, concat_visits: bool, correctness_seq: bool):
        self.data = []
        for sequence in data:
            add_visit_level_features_and_correctness(sequence)
            sub_seqs = get_sub_sequences(sequence, allowed_qids, concat_visits, True, False, correctness_seq)
            if concat_visits: # Ensure order with concat_visits since encodings will be fed through linear NN layer
                sub_seqs.sort(key=lambda sub_seq: sub_seq["question_id"])
            question_ids = [sub_seq["question_id"] for sub_seq in sub_seqs]

            # Insert null sub-sequences for missing questions
            if concat_visits:
                for q_idx, qid in enumerate(sorted(allowed_qids)):
                    if len(question_ids) <= q_idx or question_ids[q_idx] != qid:
                        sub_seqs.insert(q_idx, None)
                        question_ids.insert(q_idx, qid)

            self.data.append({
                "sub_seqs": sub_seqs,
                "question_ids": question_ids,
                "label": labels[str(sequence["student_id"])]
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class CKTPredictorCollator:
    def __init__(self, available_qids: Set[int]):
        self.available_qids = available_qids

    def __call__(self, batch: List[dict]):
        # Batch info for each question
        question_batches = {
            qid: {"event_types": [], "time_deltas": [], "correctness": [], "target_idxs": []}
            for qid in self.available_qids
        }
        question_id_batches = []
        labels = []

        # Each sequence in batch is contains array of sub-sequences, either one per question or per visit, depending on setting
        # Pick apart sequences to group questions across all sequences together for batch processing by encoders
        # Target index is maintained so after processing, resulting encodings can be mapped back to respective sequences
        # (refers to index in unrolled and padded sequence x sub-sequence matrix)
        sequence_lengths = [len(seq["sub_seqs"]) for seq in batch]
        max_seq_len = max(sequence_lengths)
        for seq_idx, sequence in enumerate(batch):
            for sub_seq_idx, sub_seq in enumerate(sequence["sub_seqs"]):
                if not sub_seq: # Sub-sequence can be None when concat_visits=True to indicate the student didn't visit that question
                    continue
                question_batch = question_batches[sub_seq["question_id"]]
                question_batch["event_types"].append(torch.LongTensor(sub_seq["event_types"]))
                question_batch["time_deltas"].append(torch.from_numpy(sub_seq["time_deltas"]).type(torch.float32))
                if "correctness" in sub_seq:
                    question_batch["correctness"].append(torch.LongTensor(sub_seq["correctness"]))
                question_batch["target_idxs"].append(seq_idx * max_seq_len + sub_seq_idx)
            question_id_batches.append(torch.LongTensor(sequence["question_ids"]))
            labels.append(sequence["label"])

        return {
            "questions": {
                str(qid): {
                    "event_types": torch.nn.utils.rnn.pad_sequence(question_batch["event_types"], batch_first=True).to(device),
                    "time_deltas": torch.nn.utils.rnn.pad_sequence(question_batch["time_deltas"], batch_first=True).to(device),
                    "correctness": torch.nn.utils.rnn.pad_sequence(question_batch["correctness"], batch_first=True).to(device) if question_batch["correctness"] else None,
                    "target_idxs": torch.LongTensor(question_batch["target_idxs"]).to(device),
                    "sequence_lengths": [event_types.shape[0] for event_types in question_batch["event_types"]]
                }
                for qid, question_batch in question_batches.items()
            },
            "question_ids": torch.nn.utils.rnn.pad_sequence(question_id_batches, batch_first=True).to(device),
            "labels": torch.Tensor(labels).to(device),
            "sequence_lengths": sequence_lengths
        }
