from typing import List
import torch
from data_processing import get_problem_qids
from data_loading import get_sub_sequences
from utils import device

def prep_data_for_irt(data: List[dict], type_mappings: dict, concat_visits: bool):
    problem_qids = {qid for _, qid in get_problem_qids("A", type_mappings) + get_problem_qids("B", type_mappings)}
    sub_seqs = []
    for sequence in data:
        # Note that if a student didn't visit a particular question, there will be no entry for that pair
        sub_seqs += get_sub_sequences(sequence, question_ids=problem_qids, concat_visits=concat_visits)
    return sub_seqs

class IRTDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[dict], type_mappings: dict):
        self.data = data
        for seq in self.data:
            seq["student_id"] = type_mappings["student_ids"][str(seq["student_id"])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class IRTCollator:
    def __init__(self):
        pass

    def __call__(self, batch: List[dict]):
        return {
            "student_ids": torch.LongTensor([seq["student_id"] for seq in batch]).to(device),
            "question_ids": torch.LongTensor([seq["question_id"] for seq in batch]).to(device),
            "labels": torch.Tensor([seq["correct"] for seq in batch]).to(device)
        }
