from typing import Dict, List, Optional
import random
import torch
import numpy as np
from constants import Correctness
from utils import device

def add_engineered_features(sequence):
    time_taken = np.array([question["time"] for question in sequence["q_stats"].values()])
    visits = np.array([question["visits"] for question in sequence["q_stats"].values()])
    correctness = np.array([question["correct"] == Correctness.CORRECT.value for question in sequence["q_stats"].values()])
    sequence["engineered_features"] = [
        # np.mean(time_taken),
        np.std(time_taken),
        np.max(time_taken),
        np.min(time_taken),
        # np.mean(visits),
        np.std(visits),
        np.max(visits),
        np.min(visits),
        np.mean(correctness),
        *sequence["assistive_uses"].values()
    ]

def add_visit_level_features_and_correctness(sequence):
    """
    Calculate visits - for the last event of each visit to a question - the index, question id, correctness, and timestamp
    Additionally, apply WORKING correctness state to all events that are not the last in a visit
    """

    seq_len = len(sequence["question_ids"])
    visits = sequence["visits"] = {
        "idxs": [],
        "qids": [],
        "correctness": [],
        "time_deltas": [],
        "mask": None
    }
    cur_qid = sequence["question_ids"][0]
    for idx, qid in enumerate(sequence["question_ids"]):
        # Create visit for previous problem when a new question ID is encountered
        if qid != cur_qid:
            visits["idxs"].append(idx - 1)
            visits["qids"].append(cur_qid)
            visits["correctness"].append(sequence["correctness"][idx - 1])
            visits["time_deltas"].append(sequence["time_deltas"][idx - 1])
            cur_qid = qid
        # Correctness should always be "working" except at last timestep of each visit
        elif idx > 0:
            sequence["correctness"][idx - 1] = Correctness.WORKING.value
    # Create final visit
    visits["idxs"].append(seq_len - 1)
    visits["qids"].append(cur_qid)
    visits["correctness"].append(sequence["correctness"][seq_len - 1])
    visits["time_deltas"].append(sequence["time_deltas"][seq_len - 1])
    # Add mask for collator and model
    visits["mask"] = torch.ones(len(visits["time_deltas"]), dtype=torch.bool)

def add_time_ratios(sequence):
    """
    For each event, calculate the ratio between the previous and following timesteps
    For timestep t, previous timestep t1 and following timestep t2, ratio = (t - t1)/(t2 - t1)
    Value does not exist for events 0 and M, so use values 0 and 1 respectively
    """
    td = sequence["time_deltas"]
    if len(td) == 1:
        # TODO: deal with data issue, example is VH098810 and HELPMAT8 for student 2333126380
        sequence["time_ratios"] = np.array([0])
    else:
        time_steps = td[1:] - td[:-1]
        time_spans = td[2:] - td[:-2]
        time_spans[time_spans == 0] = 1 # Sometimes we have the same timestamp for multiple events, so avoid division by 0
        ratio = time_steps[:-1] / time_spans
        sequence["time_ratios"] = np.concatenate([[0], ratio, [1]])

def get_sub_sequences(sequence, question_ids: set = None, concat_visits = True, log_time: bool = False, qid_seq: bool = False):
    # Split a sequence into a list of subsequences by visit
    start_idx = 0
    sub_sequences = []
    for last_idx in sequence["visits"]["idxs"]:
        qid = sequence["question_ids"][start_idx]
        if not question_ids or qid in question_ids:
            sub_seq = {
                "student_id": sequence["student_id"],
                "data_class": sequence["data_class"],
                "question_id": qid,
                "event_types": sequence["event_types"][start_idx : last_idx + 1],
                "complete": sequence["correctness"][last_idx] != Correctness.INCOMPLETE.value,
                "correct": sequence["correctness"][last_idx] == Correctness.CORRECT.value
            }
            if qid_seq:
                sub_seq["question_ids"] = sequence["question_ids"][start_idx : last_idx + 1]
            if log_time:
                # Convert to log2 as per CKT paper, add 1 to avoid log(0)
                sub_seq["time_deltas"] = np.log2(sequence["time_deltas"][start_idx : last_idx + 1]) + 1
            else:
                sub_seq["time_deltas"] = sequence["time_deltas"][start_idx : last_idx + 1]
            sub_sequences.append(sub_seq)
        start_idx = last_idx + 1

    # If requested, concatenate visits per qid
    if concat_visits:
        qid_to_sub_sequences = {}
        for sub_seq in sub_sequences:
            if sub_seq["question_id"] not in qid_to_sub_sequences:
                qid_to_sub_sequences[sub_seq["question_id"]] = sub_seq
            else:
                qid_sub_seqs = qid_to_sub_sequences[sub_seq["question_id"]]
                qid_sub_seqs["question_ids"] += sub_seq["question_ids"]
                qid_sub_seqs["event_types"] += sub_seq["event_types"]
                # TODO: concat is really slow
                qid_sub_seqs["time_deltas"] = np.concatenate([qid_sub_seqs["time_deltas"], sub_seq["time_deltas"]])
                # Take correctness of most recent visit
                qid_sub_seqs["complete"] = sub_seq["complete"]
                qid_sub_seqs["correct"] = sub_seq["correct"]
        sub_sequences = list(qid_to_sub_sequences.values())

    return sub_sequences

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict[str, list]], type_mappings: dict, labels: Dict[str, bool] = None, correct_as_label: bool = False,
                 engineered_features: bool = False, qids_for_subseq_split: set = None, concat_visits: bool = True, time_ratios: bool = False,
                 log_time: bool = False, qid_seq: bool = True):

        self.data = []
        total_positive = 0
        bin_labels = labels and isinstance(next(iter(labels.values())), bool)

        # Process a single sequence and add it to the dataset
        def add_sequence(sequence):
            nonlocal total_positive

            sequence["sid"] = type_mappings["student_ids"][str(sequence["student_id"])]
            sequence["time_deltas"] = np.array(sequence["time_deltas"], dtype=np.float32)
            sequence["mask"] = torch.ones(len(sequence["time_deltas"]), dtype=torch.bool)

            if correct_as_label:
                sequence["label"] = sequence["correct"]
            elif labels:
                sequence["label"] = labels[str(sequence["student_id"])]
                if bin_labels:
                    total_positive += 1 if sequence["label"] else 0

            if time_ratios:
                add_time_ratios(sequence)

            if engineered_features:
                add_engineered_features(sequence)

            self.data.append(sequence)

        # Iterate over given data list and process each sequence
        for sequence in data:
            add_visit_level_features_and_correctness(sequence)

            if qids_for_subseq_split:
                for sub_seq in get_sub_sequences(sequence, qids_for_subseq_split, concat_visits, log_time, qid_seq):
                    add_sequence(sub_seq)
            else:
                add_sequence(sequence)

        if bin_labels:
            print(f"Positive rate: {total_positive / len(self.data):.3f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class Sampler(torch.utils.data.Sampler):
    def __init__(self, chunk_sizes: List[int], batch_size: Optional[int] = None):
        self.chunk_sizes = chunk_sizes
        self.batch_size = batch_size
        self.len = sum(int(chunk_size / self.batch_size) if self.batch_size else 1 for chunk_size in self.chunk_sizes)

    def __iter__(self):
        """
        This function will shuffle the indices of each chunk, and shuffle the order in which batches are drawn from each chunk
        In effect, yielding will return a random batch from a random chunk
        Assumption: initial data ordering will be contiguous chunks
        """
        # Shuffle sample indices within each chunk
        chunk_idx_shuffles = [np.array(random.sample(list(range(chunk_size)), chunk_size)) for chunk_size in self.chunk_sizes]
        # Shuffle order from which chunks are drawn, resulting in array with size of total number of batches
        batches_per_chunk = [int(chunk_size / self.batch_size) if self.batch_size else 1 for chunk_size in self.chunk_sizes]
        chunk_draws = [chunk_num for chunk_num, batches_in_chunk in enumerate(batches_per_chunk) for _ in range(batches_in_chunk)]
        random.shuffle(chunk_draws)
        # Keep track of current batch index for each chunk
        chunk_batch_idx = [0] * len(self.chunk_sizes)
        # Iterate over shuffle chunk draw order
        for chunk_num in chunk_draws:
            batch_size = self.batch_size or self.chunk_sizes[chunk_num]
            # Get and increase current batch index for current chunk
            batch_start_idx = chunk_batch_idx[chunk_num] * batch_size
            chunk_batch_idx[chunk_num] += 1
            # Get corresponding shuffled data indices for batch
            # idxs_in_chunk = chunk_idx_shuffles[chunk_num][np.arange(batch_start_idx, batch_start_idx + batch_size)]
            idxs_in_chunk = chunk_idx_shuffles[chunk_num][batch_start_idx : batch_start_idx + batch_size]
            idxs = idxs_in_chunk + sum(self.chunk_sizes[:chunk_num])
            yield idxs

    def __len__(self):
        return self.len


class Collator:
    def __init__(self, random_trim=False):
        self.random_trim = random_trim

    def __call__(self, batch: List[Dict]):
        question_id_batches = []
        # question_type_batches = []
        event_type_batches = []
        time_delta_batches = []
        time_ratio_batches = []
        correctness_batches = []
        visit_idx_batches = []
        visit_qid_batches = []
        visit_correctness_batches = []
        mask_batches = []
        visit_mask_batches = []
        student_ids = []
        question_ids_collapsed = []
        engineered_features = []
        labels = []

        if self.random_trim:
            trim_length = 5 * 60
            trim_max = 30 * 60
            trim_at = random.randint(1, trim_max / trim_length) * trim_length

        for sequence in batch:
            # Convert data structures to torch tensors
            time_deltas = torch.from_numpy(sequence["time_deltas"])
            if len(time_deltas) < 2: # A single-event sequence messes with time_ratios calculations, so is unusable
                print(f"Skipping student {sequence['student_id']}, sequence has length {len(time_deltas)}")
                continue
            if "question_ids" in sequence:
                qids = torch.LongTensor(sequence["question_ids"])
            eids = torch.LongTensor(sequence["event_types"])
            mask = sequence["mask"]
            if "correctness" in sequence:
                correctness = torch.LongTensor(sequence["correctness"])
            if "time_ratios" in sequence:
                time_ratios = torch.from_numpy(sequence["time_ratios"])
            if "visits" in sequence:
                visit_time_deltas = torch.FloatTensor(sequence["visits"]["time_deltas"])
                visit_idxs = torch.LongTensor(sequence["visits"]["idxs"])
                visit_qids = torch.LongTensor(sequence["visits"]["qids"])
                visit_correctness = torch.LongTensor(sequence["visits"]["correctness"])
                visit_mask = sequence["visits"]["mask"]

            # Apply trimming if specified
            if self.random_trim:
                time_mask = time_deltas <= trim_at
                time_deltas = time_deltas[time_mask]
                if "question_ids" in sequence:
                    qids = qids[time_mask]
                eids = eids[time_mask]
                mask = mask[time_mask]
                if "correctness" in sequence:
                    correctness = correctness[time_mask]
                if "time_ratios" in sequence:
                    time_ratios = time_ratios[time_mask]
                if "visits" in sequence:
                    visit_time_mask = visit_time_deltas <= trim_at
                    visit_idxs = visit_idxs[visit_time_mask]
                    visit_qids = visit_qids[visit_time_mask]
                    visit_correctness = visit_correctness[visit_time_mask]
                    visit_mask = visit_mask[visit_time_mask]

            # Accumulate tensor for the batch
            time_delta_batches.append(time_deltas)
            if "question_ids" in sequence:
                question_id_batches.append(qids)
            event_type_batches.append(eids)
            mask_batches.append(mask)
            if "correctness" in sequence:
                correctness_batches.append(correctness)
            if "time_ratios" in sequence:
                time_ratio_batches.append(time_ratios)
            if "visits" in sequence:
                visit_idx_batches.append(visit_idxs)
                visit_qid_batches.append(visit_qids)
                visit_correctness_batches.append(visit_correctness)
                visit_mask_batches.append(visit_mask)
            if "engineered_features" in sequence:
                engineered_features.append(sequence["engineered_features"])
            if "label" in sequence:
                labels.append(sequence["label"])
            student_ids.append(sequence["sid"])
            if "question_id" in sequence:
                question_ids_collapsed.append(sequence["question_id"])

        return {
            "question_ids": torch.nn.utils.rnn.pad_sequence(question_id_batches, batch_first=True).to(device) if question_id_batches else None,
            # "question_types": torch.nn.utils.rnn.pad_sequence(question_type_batches, batch_first=True).to(device),
            "event_types": torch.nn.utils.rnn.pad_sequence(event_type_batches, batch_first=True).to(device),
            "time_deltas": torch.nn.utils.rnn.pad_sequence(time_delta_batches, batch_first=True).to(device),
            "correctness": torch.nn.utils.rnn.pad_sequence(correctness_batches, batch_first=True).to(device) if correctness_batches else None,
            "mask": torch.nn.utils.rnn.pad_sequence(mask_batches, batch_first=True).to(device),
            "time_ratios": torch.nn.utils.rnn.pad_sequence(time_ratio_batches, batch_first=True).to(device) if time_ratio_batches else None,
            "visits": {
                "idxs": torch.nn.utils.rnn.pad_sequence(visit_idx_batches, batch_first=True).to(device),
                "qids": torch.nn.utils.rnn.pad_sequence(visit_qid_batches, batch_first=True).to(device),
                "correctness": torch.nn.utils.rnn.pad_sequence(visit_correctness_batches, batch_first=True).to(device),
                "mask": torch.nn.utils.rnn.pad_sequence(visit_mask_batches, batch_first=True).to(device),
            } if visit_idx_batches else None,
            "student_ids": torch.LongTensor(student_ids).to(device), # For IRT
            "question_ids_collapsed": torch.LongTensor(question_ids_collapsed).to(device), # For IRT
            "engineered_features": torch.Tensor(engineered_features).to(device),
            "labels": torch.Tensor(labels).to(device),
            "data_class": batch[0]["data_class"], # Sampler ensures that each batch is drawn from a single class
            "sequence_lengths": torch.LongTensor([seq.shape[0] for seq in event_type_batches]) # Must be on CPU
        }
