import json
from typing import Dict, List, Optional
import random
import pandas
import torch
import numpy as np
from constants import ASSISTIVE_EVENT_IDS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_type_mappings(data_file):
    src_data = pandas.read_csv(data_file)
    unique_questions = {question: idx for idx, question in enumerate(src_data["AccessionNumber"].unique())}
    unique_question_types = {question_type: idx for idx, question_type in enumerate(src_data["ItemType"].unique())}
    unique_event_types = {event_type: idx for idx, event_type in enumerate(src_data["Observable"].unique())}

    with open("data/types.json", "w") as types_file:
        json.dump({
            "question_ids": unique_questions,
            "question_types": unique_question_types,
            "event_types": unique_event_types
        }, types_file)

def load_type_mappings():
    with open("data/types.json") as type_mapping_file:
        return json.load(type_mapping_file)

def convert_raw_data_to_json(data_filenames: List[str], output_filename: str, trim_after: List[float] = None, data_classes: List[str] = None):
    if trim_after:
        assert len(trim_after) == len(data_filenames)
    if data_classes:
        assert len(data_classes) == len(data_filenames)

    student_to_sequences: Dict[int, dict] = {}
    type_mappings = load_type_mappings()

    # Process data set - each sequence is list of events per student
    for file_idx, data_file in enumerate(data_filenames):
        print("Processing", data_file)
        src_data = pandas.read_csv(data_file, parse_dates=["EventTime"]).sort_values(["STUDENTID", "EventTime"])
        for _, event in src_data.iterrows():
            # Skip entries with no timestamp
            if pandas.isnull(event["EventTime"]):
                print("Skipping event, no timestamp", event)
                continue

            sequence: Dict[str, list] = student_to_sequences.setdefault(event["STUDENTID"], {
                "student_id": event["STUDENTID"],
                "question_ids": [],
                "question_types": [],
                "event_types": [],
                "time_deltas": [],
                "data_class": data_classes[file_idx] if data_classes else None
            })
            if not sequence["event_types"]:
                start_time = event["EventTime"]
            time_delta = (event["EventTime"] - start_time).total_seconds()
            if trim_after and time_delta > trim_after[file_idx]:
                continue

            sequence["question_ids"].append(type_mappings["question_ids"][event["AccessionNumber"]])
            sequence["question_types"].append(type_mappings["question_types"][event["ItemType"]])
            sequence["event_types"].append(type_mappings["event_types"][event["Observable"]])
            sequence["time_deltas"].append(time_delta)

    with open(output_filename, "w") as output_file:
        json.dump(list(student_to_sequences.values()), output_file)

def convert_raw_labels_to_json(data_file, output_filename):
    src_data = pandas.read_csv(data_file)

    student_to_label = {}
    for _, event in src_data.iterrows():
        student_to_label[event["STUDENTID"]] = event["EfficientlyCompletedBlockB"]

    with open(output_filename, "w") as output_file:
        json.dump(student_to_label, output_file)

def add_engineered_features(data):
    for sequence in data:
        total_assistives = {eid: 0 for eid in ASSISTIVE_EVENT_IDS}

        # Get details on the visits to each question in the sequence
        qid_to_visits = {}
        cur_question_id = None
        for event, qid, timestamp in zip(sequence["event_types"], sequence["question_ids"], sequence["time_deltas"]):
            # List of [start_time, end_time] for each visit to this question
            q_visits = qid_to_visits.setdefault(qid, [])

            # If we went to a new question, start a new visit
            if qid != cur_question_id:
                q_visits.append([timestamp, timestamp])
                cur_question_id = qid
            else:
                # Update end_time of current visit
                q_visits[-1][1] = timestamp

            # Update number of times assistives used
            if event in total_assistives:
                total_assistives[event] += 1

        # Calculate final features for questions in this sequence
        time_taken = np.array([sum(end_time - start_time for start_time, end_time in q_visits) for q_visits in qid_to_visits.values()])
        visits = np.array([len(q_visits) for q_visits in qid_to_visits.values()])
        sequence["engineered_features"] = [
            # np.mean(time_taken),
            np.std(time_taken),
            np.max(time_taken),
            np.min(time_taken),
            # np.mean(visits),
            # np.std(visits),
            # np.max(visits),
            # np.min(visits),
            *total_assistives.values()
        ]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict[str, list]], labels: Dict[str, bool] = None, engineered_features: bool = False):
        self.data = data
        if engineered_features:
            add_engineered_features(self.data)
        if labels:
            for sequence in self.data:
                sequence["label"] = 1 if labels[str(sequence["student_id"])] else 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class Sampler:
    def __init__(self, chunk_sizes: List[int], batch_size: Optional[int] = None):
        self.chunk_sizes = chunk_sizes
        self.batch_size = batch_size

    def __iter__(self):
        """
        This function will shuffle the indices of each chunk, and shuffle the order in which batches are drawn from each chunk
        In effect, yielding will return a random batch from a random chunk
        Assumption: data ordering will be contiguous chunks
        """
        # import pdb; pdb.set_trace()
        # Shuffle sample indices within each chunk
        chunk_idx_shuffles = [np.array([idx for idx in random.sample(list(range(chunk_size)), chunk_size)]) for chunk_size in self.chunk_sizes]
        # Shuffle order from which chunks are drawn, size of array is total number of batches
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
            idxs_in_chunk = chunk_idx_shuffles[chunk_num][np.arange(batch_start_idx, batch_start_idx + batch_size)]
            idxs = idxs_in_chunk + sum(self.chunk_sizes[:chunk_num])
            yield idxs

class Collator:
    def __init__(self):
        pass

    def __call__(self, batch: List[Dict]):
        question_id_batches = []
        question_type_batches = []
        event_type_batches = []
        time_delta_batches = []
        mask = []
        engineered_features = []
        labels = []
        for sequence in batch:
            question_id_batches.append(torch.LongTensor(sequence["question_ids"]))
            question_type_batches.append(torch.LongTensor(sequence["question_types"]))
            event_type_batches.append(torch.LongTensor(sequence["event_types"]))
            time_delta_batches.append(torch.FloatTensor(sequence["time_deltas"]))
            mask.append(torch.ones(len(sequence["event_types"])))
            if "engineered_features" in sequence:
                engineered_features.append(sequence["engineered_features"])
            if "label" in sequence:
                labels.append(sequence["label"])

        return {
            "question_ids": torch.nn.utils.rnn.pad_sequence(question_id_batches, batch_first=True).to(device),
            "question_types": torch.nn.utils.rnn.pad_sequence(question_type_batches, batch_first=True).to(device),
            "event_types": torch.nn.utils.rnn.pad_sequence(event_type_batches, batch_first=True).to(device),
            "time_deltas": torch.nn.utils.rnn.pad_sequence(time_delta_batches, batch_first=True).to(device),
            "mask": torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).to(device),
            "engineered_features": torch.Tensor(engineered_features).to(device),
            "labels": torch.Tensor(labels).to(device),
            "data_class": batch[0]["data_class"], # Sampler ensures that each batch is drawn from a single class
            "sequence_lengths": torch.LongTensor([len(sequence["event_types"]) for sequence in batch]) # Must be on CPU
        }
