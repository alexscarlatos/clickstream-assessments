from typing import Dict, List, Optional
import random
import torch
import numpy as np
from model import Mode
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
    # Calculate visits - the last event of each visit to a question, the index, question id, correctness, and timestamp
    # Also refine sequence-level correctness

    seq_len = len(sequence["question_ids"])
    visits = sequence["visits"] = {
        "idxs": [],
        "qids": [],
        "correctness": [],
        "time_deltas": []
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

def add_time_ratios(sequence):
    # For each event, calculate the ratio between the previous and following timesteps
    # For timestep t, previous timestep t1 and following timestep t2, ratio = (t - t1)/(t2 - t1)
    # Value does not exist for events 0 and M, so use values 0 and 1 respectively
    td = np.array(sequence["time_deltas"])
    time_steps = td[1:] - td[:-1]
    time_spans = td[2:] - td[:-2]
    time_spans[time_spans == 0] = 1 # Sometimes we have the same timestamp for multiple events, so avoid division by 0
    ratio = time_steps[:-1] / time_spans
    sequence["time_ratios"] = np.concatenate([[0], ratio, [1]])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict[str, list]], mode: Mode, labels: Dict[str, bool] = None, engineered_features: bool = False):
        self.data = data

        total_positive = 0
        for sequence in self.data:
            add_visit_level_features_and_correctness(sequence)

            if engineered_features:
                add_engineered_features(sequence)

            if labels:
                sequence["label"] = 1 if labels[str(sequence["student_id"])] else 0
                total_positive += sequence["label"]

            if mode == Mode.PRE_TRAIN:
                add_time_ratios(sequence)

        print(f"Data size: {len(self.data)}")
        if labels:
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
        question_type_batches = []
        event_type_batches = []
        time_delta_batches = []
        time_ratio_batches = []
        correctness_batches = []
        visit_idx_batches = []
        visit_qid_batches = []
        visit_correctness_batches = []
        mask = []
        visit_mask = []
        engineered_features = []
        labels = []

        if self.random_trim:
            trim_length = 5 * 60
            trim_max = 30 * 60
            trim_at = random.randint(1, trim_max / trim_length) * trim_length

        for sequence in batch:
            time_deltas = torch.FloatTensor(sequence["time_deltas"])
            if len(time_deltas) < 2: # A single-event sequence messes with time_deltas calculations, so is unusable
                print(f"Skipping student {sequence['student_id']}, sequence has length {len(time_deltas)}")
                continue
            time_mask = time_deltas <= trim_at if self.random_trim else torch.ones(time_deltas.shape[0]).type(torch.bool)
            question_id_batches.append(torch.LongTensor(sequence["question_ids"])[time_mask])
            question_type_batches.append(torch.LongTensor(sequence["question_types"])[time_mask])
            event_type_batches.append(torch.LongTensor(sequence["event_types"])[time_mask])
            time_delta_batches.append(time_deltas[time_mask])
            correctness_batches.append(torch.LongTensor(sequence["correctness"])[time_mask])
            mask.append(torch.ones(time_deltas.shape[0])[time_mask])
            if "time_ratios" in sequence:
                time_ratio_batches.append(torch.from_numpy(sequence["time_ratios"])[time_mask])
            if "visits" in sequence:
                visit_time_deltas = torch.FloatTensor(sequence["visits"]["time_deltas"])
                visit_time_mask = visit_time_deltas <= trim_at if self.random_trim else torch.ones(visit_time_deltas.shape[0]).type(torch.bool)
                visit_idx_batches.append(torch.LongTensor(sequence["visits"]["idxs"])[visit_time_mask])
                visit_qid_batches.append(torch.LongTensor(sequence["visits"]["qids"])[visit_time_mask])
                visit_correctness_batches.append(torch.LongTensor(sequence["visits"]["correctness"])[visit_time_mask])
                visit_mask.append(torch.ones(visit_time_deltas.shape[0])[visit_time_mask])
            if "engineered_features" in sequence:
                engineered_features.append(sequence["engineered_features"])
            if "label" in sequence:
                labels.append(sequence["label"])

        return {
            "question_ids": torch.nn.utils.rnn.pad_sequence(question_id_batches, batch_first=True).to(device),
            "question_types": torch.nn.utils.rnn.pad_sequence(question_type_batches, batch_first=True).to(device),
            "event_types": torch.nn.utils.rnn.pad_sequence(event_type_batches, batch_first=True).to(device),
            "time_deltas": torch.nn.utils.rnn.pad_sequence(time_delta_batches, batch_first=True).to(device),
            "correctness": torch.nn.utils.rnn.pad_sequence(correctness_batches, batch_first=True).to(device),
            "mask": torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).to(device),
            "time_ratios": torch.nn.utils.rnn.pad_sequence(time_ratio_batches, batch_first=True).to(device) if time_ratio_batches else None,
            "visits": {
                "idxs": torch.nn.utils.rnn.pad_sequence(visit_idx_batches, batch_first=True).to(device),
                "qids": torch.nn.utils.rnn.pad_sequence(visit_qid_batches, batch_first=True).to(device),
                "correctness": torch.nn.utils.rnn.pad_sequence(visit_correctness_batches, batch_first=True).to(device),
                "mask": torch.nn.utils.rnn.pad_sequence(visit_mask, batch_first=True).to(device),
            } if visit_idx_batches else None,
            "engineered_features": torch.Tensor(engineered_features).to(device),
            "labels": torch.Tensor(labels).to(device),
            "data_class": batch[0]["data_class"], # Sampler ensures that each batch is drawn from a single class
            "sequence_lengths": torch.LongTensor([seq.shape[0] for seq in event_type_batches]) # Must be on CPU
        }
