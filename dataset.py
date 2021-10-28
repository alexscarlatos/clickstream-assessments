import json
from typing import Dict, List
import pandas
import torch

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

def convert_raw_data_to_json(data_filenames, output_filename, trim_after=None):
    student_to_sequences: Dict[int, dict] = {}
    type_mappings = load_type_mappings()

    # Process data set - each sequence is list of events per student
    for data_file in data_filenames:
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
                "time_deltas": []
            })
            if not sequence["event_types"]:
                start_time = event["EventTime"]
            time_delta = (event["EventTime"] - start_time).total_seconds()
            if trim_after and time_delta > trim_after:
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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict[str, list]], labels: Dict[str, bool] = None):
        self.data = data
        if labels:
            for sequence in self.data:
                sequence["label"] = 1 if labels[str(sequence["student_id"])] else 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class Collator:
    def __init__(self):
        pass

    def __call__(self, batch: List[Dict[str, list]]):
        question_id_batches = []
        question_type_batches = []
        event_type_batches = []
        time_delta_batches = []
        mask = []
        labels = []
        for sequence in batch:
            question_id_batches.append(torch.LongTensor(sequence["question_ids"]))
            question_type_batches.append(torch.LongTensor(sequence["question_types"]))
            event_type_batches.append(torch.LongTensor(sequence["event_types"]))
            time_delta_batches.append(torch.FloatTensor(sequence["time_deltas"]))
            mask.append(torch.ones(len(sequence["event_types"])))
            if "label" in sequence:
                labels.append(sequence["label"])

        return {
            "question_ids": torch.nn.utils.rnn.pad_sequence(question_id_batches, batch_first=True).to(device),
            "question_types": torch.nn.utils.rnn.pad_sequence(question_type_batches, batch_first=True).to(device),
            "event_types": torch.nn.utils.rnn.pad_sequence(event_type_batches, batch_first=True).to(device),
            "time_deltas": torch.nn.utils.rnn.pad_sequence(time_delta_batches, batch_first=True).to(device),
            "mask": torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).to(device),
            "labels": torch.Tensor(labels).to(device),
            "sequence_lengths": torch.LongTensor([len(sequence["event_types"]) for sequence in batch]) # Must be on CPU
        }
