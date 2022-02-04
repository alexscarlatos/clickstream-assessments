"""
Goal: train a_i and d_j for all students and questions to optimize BCE on P(Y_ij = 1) = sigmoid(a_i - d_j)

Complications:
- Some students don't have any events for a question
- Some students have events for a question but don't complete it,
   and this could be trivially deducted by the model to be incomplete or incorrect,
   so we should evaluate both with and without these questions

Masking: remove data permanently or reserve for later
- Implementation: Flatten student/question matrix, create index for target student/question pairs, and target using vectorized ops
- Steps:
    - First, remove all untouched questions
    - Second, reserve test set (stratified on question type)
    - Third, with cross-validation (stratified on question type), reserve validation set

Model training:
- Batches taken from flattened matrix
- We index into the parameter vectors using torch's vectorized operations
- Softplus is used as a parameter transformation to ensure that ability and difficulty are always positive
"""

import torch
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from data_processing import load_type_mappings
from training import get_data, train, evaluate_model, BATCH_SIZE
from irt.irt_model import IRT
from irt.irt_data_loading import prep_data_for_irt, IRTDataset, IRTCollator
from constants import Mode, TrainOptions, Correctness
from utils import device

# TODO: will have to retrain our model on per-question sequences
# TODO: may want to try out using one-hot encoding for question ids on our model

def irt(model_name: str, data_file: str, options: TrainOptions):
    type_mappings = load_type_mappings()
    src_data = get_data(data_file)
    entries = np.array(prep_data_for_irt(src_data, type_mappings, options.concat_visits))
    # Try to balance each student fairly in the train/val/test sets
    # This should effectively balance questions as well from the randomness, since |students| >> |questions|
    stratify_labels = np.array([entry["student_id"] for entry in entries])

    # Make sure correct states are all, well, correct
    if True:
        student_to_stats = {seq["student_id"]: seq["q_stats"] for seq in src_data}
        rev_qid_map = {qid: qid_str for qid_str, qid in type_mappings["question_ids"].items()}
        for entry in entries:
            assert entry["correct"] == (student_to_stats[entry["student_id"]][rev_qid_map[entry["question_id"]]]["correct"] == Correctness.CORRECT.value)

    # Do stratified train/test split
    test_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=221)
    train_all_idx, test_idx = next(test_skf.split(entries, stratify_labels))
    train_data_all = entries[train_all_idx]
    train_stratify_labels_all = stratify_labels[train_all_idx]
    test_data = entries[test_idx]
    test_loader = torch.utils.data.DataLoader(
        IRTDataset(test_data, type_mappings),
        collate_fn=IRTCollator(),
        batch_size=BATCH_SIZE
    )

    def data_stats(prefix: str, data: list):
        student_counts = Counter(entry["student_id"] for entry in data).most_common()
        qid_counts = Counter(entry["question_id"] for entry in data).most_common()
        print(f"{prefix} Students: Most: {student_counts[0]}, Least: {student_counts[-1]}; Questions: {qid_counts[0]}, Least: {qid_counts[-1]}")

    data_stats("All:", entries)
    data_stats("Train Full:", train_data_all)
    data_stats("Test:", test_data)

    # Do cross-validation training for IRT model
    val_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=221)
    for k, (train_idx, val_idx) in enumerate(val_skf.split(train_data_all, train_stratify_labels_all)):
        # Get train/validation datasets
        train_data = train_data_all[train_idx]
        val_data = train_data_all[val_idx]
        train_loader = torch.utils.data.DataLoader(
            IRTDataset(train_data, type_mappings),
            collate_fn=IRTCollator(),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            IRTDataset(val_data, type_mappings),
            collate_fn=IRTCollator(),
            batch_size=BATCH_SIZE
        )

        data_stats("Train:", train_data)
        data_stats("Val:", val_data)

        # Train model
        model = IRT(len(type_mappings["student_ids"]), len(type_mappings["question_ids"]))
        train(model, Mode.PREDICT, model_name, train_loader, val_loader, options.lr, options.weight_decay, options.epochs)

        # Test model
        loss, accuracy, auc, kappa, aggregated = evaluate_model(model, test_loader, Mode.PREDICT)
        print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, Adj AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")

        break # TODO: just one fold for now

def test_irt(model_name: str, data_file: str, options: TrainOptions):
    type_mappings = load_type_mappings()
    src_data = get_data(data_file)

    model = IRT(len(type_mappings["student_ids"]), len(type_mappings["question_ids"]))
    model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))

    # Plot learned ability vs. student score over both blocks
    student_idxs = [type_mappings["student_ids"][str(seq["student_id"])] for seq in src_data]
    abilities = torch.nn.Softplus()(model.ability).detach().cpu().numpy()[student_idxs]
    scores = [seq["block_a_score"] + seq["block_b_score"] for seq in src_data]
    plt.plot(abilities, scores, "bo")
    plt.xlabel("Learned Ability")
    plt.ylabel("Total Score")
    plt.show()

    # Plot question difficulty vs. avg correctness
    entries = np.array(prep_data_for_irt(src_data, type_mappings, options.concat_visits))
    qid_to_score = {}
    qid_to_num = {}
    for entry in entries:
        qid = entry["question_id"]
        qid_to_score.setdefault(qid, 0)
        qid_to_score[qid] += 1 if entry["correct"] else 0
        qid_to_num.setdefault(qid, 0)
        qid_to_num[qid] += 1
    difficulties = torch.nn.Softplus()(model.difficulty).detach().cpu().numpy()[list(qid_to_score.keys())]
    avg_score = [qid_to_score[qid] / qid_to_num[qid] for qid in qid_to_score]
    plt.plot(difficulties, avg_score, "ro")
    plt.xlabel("Learned Difficulty")
    plt.ylabel("Average Score")
    plt.show()
