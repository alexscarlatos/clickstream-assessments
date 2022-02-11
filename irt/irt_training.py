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

from typing import Tuple
import torch
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from data_processing import load_type_mappings, get_problem_qids
from data_loading import Dataset, Collator
from training import get_data, train, evaluate_model, create_predictor_model, BATCH_SIZE
from model import LSTMModel
from irt.irt_model import IRT
from constants import Mode, TrainOptions, Correctness
from utils import device, initialize_seeds


def irt(pretrained_name: str, model_name: str, data_file: str, ckt: bool, options: TrainOptions):
    # Not using correctness info to verify that additional performance come strictly from behavioral data
    options.use_correctness = False
    # Since representing a single question at a time, task switching cannot be represented
    options.use_visit_pt_objs = False

    type_mappings = load_type_mappings()
    src_data = get_data(data_file)

    problem_qids = {qid for _, qid in get_problem_qids("A", type_mappings) + get_problem_qids("B", type_mappings)}
    full_dataset = Dataset(src_data, type_mappings, correct_as_label=True, qids_for_subseq_split=problem_qids,
                           concat_visits=options.concat_visits, time_ratios=not ckt, log_time=ckt, qid_seq=not ckt)

    # Try to balance each student fairly in the train/val/test sets
    # This should effectively balance questions as well from the randomness, since |students| >> |questions|
    stratify_labels = np.array([entry["student_id"] for entry in full_dataset])

    # Make sure correct states are all, well, correct
    if True:
        student_to_stats = {seq["student_id"]: seq["q_stats"] for seq in src_data}
        rev_qid_map = {qid: qid_str for qid_str, qid in type_mappings["question_ids"].items()}
        for entry in full_dataset:
            assert entry["correct"] == (student_to_stats[entry["student_id"]][rev_qid_map[entry["question_id"]]]["correct"] == Correctness.CORRECT.value)

    # Do stratified train/test split
    test_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=221)
    train_all_idx, test_idx = next(test_skf.split(full_dataset, stratify_labels))
    train_data_all = torch.utils.data.Subset(full_dataset, train_all_idx)
    train_stratify_labels_all = stratify_labels[train_all_idx]
    test_data = torch.utils.data.Subset(full_dataset, test_idx)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        collate_fn=Collator(),
        batch_size=BATCH_SIZE
    )

    def data_stats(prefix: str, data: list):
        student_counts = Counter(entry["student_id"] for entry in data).most_common()
        qid_counts = Counter(entry["question_id"] for entry in data).most_common()
        print(f"{prefix} Students: Most: {student_counts[0]}, Least: {student_counts[-1]}; Questions: {qid_counts[0]}, Least: {qid_counts[-1]}")

    data_stats("All:", full_dataset)
    data_stats("Train Full:", train_data_all)
    data_stats("Test:", test_data)

    # Do cross-validation training for IRT model
    val_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=221)
    for k, (train_idx, val_idx) in enumerate(val_skf.split(train_data_all, train_stratify_labels_all)):
        initialize_seeds(221)

        print(f"\n----- Iteration {k +1} -----")
        cur_model_name = f"{model_name}_{k + 1}"
        train_data = torch.utils.data.Subset(train_data_all, train_idx)
        val_data = torch.utils.data.Subset(train_data_all, val_idx)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            collate_fn=Collator(),
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data,
            collate_fn=Collator(),
            batch_size=BATCH_SIZE
        )

        data_stats("Train:", train_data)
        data_stats("Val:", val_data)

        use_behavior_model = True
        if use_behavior_model:
            # Train behavior model on train data split
            print("\nPretraining Behavior Model")
            pretrained_behavior_model_name = f"{pretrained_name}_{k + 1}"
            if ckt:
                # TODO: pretrain CKT encoder model on subsequences
                # TODO: create CKTPredictor model with IRT mode set
                pass
            else:
                behavior_model_pt = LSTMModel(Mode.PRE_TRAIN, type_mappings, options, available_qids=problem_qids).to(device)
                # train(behavior_model_pt, Mode.PRE_TRAIN, pretrained_behavior_model_name, train_loader, val_loader,
                #       lr=1e-3, weight_decay=1e-6, epochs=20)
                options.freeze_model = True
                options.freeze_embeddings = True
                options.use_pretrained_weights = True
                options.use_pretrained_embeddings = True
                behavior_model = create_predictor_model(pretrained_behavior_model_name, Mode.IRT, type_mappings, options)
        else:
            behavior_model = None

        # Train model
        print("\nTraining IRT Model")
        model = IRT(len(type_mappings["student_ids"]), len(type_mappings["question_ids"]), behavior_model).to(device)
        train(model, Mode.PREDICT, cur_model_name, train_loader, val_loader, lr=5e-3, weight_decay=1e-6, epochs=25)
        model.load_state_dict(torch.load(f"{cur_model_name}.pt", map_location=device)) # Load from best epoch
        model.eval()

        # Test model
        loss, accuracy, auc, kappa, aggregated = evaluate_model(model, test_loader, Mode.PREDICT)
        print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, Adj AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")

        # Fine-tune model
        if use_behavior_model:
            print("\nFine-Tuning IRT Model")
            cur_model_ft_name = f"{model_name}_ft_{k + 1}"

            # TODO: special CKT handling needed

            # Unfreeze LSTM model and embeddings
            for comp in [model.behavior_model.question_embeddings, model.behavior_model.event_type_embeddings, model.behavior_model.lstm]:
                for param in comp.parameters():
                    param.requires_grad = True

            train(model, Mode.PREDICT, cur_model_ft_name, train_loader, val_loader, lr=1e-4, weight_decay=1e-6, epochs=20)
            model.load_state_dict(torch.load(f"{cur_model_ft_name}.pt", map_location=device)) # Load from best epoch
            model.eval()

            # Test model
            loss, accuracy, auc, kappa, aggregated = evaluate_model(model, test_loader, Mode.PREDICT)
            print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, Adj AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")

        break # TODO: just one fold for now

def test_irt(model_name: str, data_file: str, ckt: bool, options: TrainOptions):
    options.use_correctness = False
    type_mappings = load_type_mappings()
    src_data = get_data(data_file)

    # Load dataset
    problem_qids = {qid for _, qid in get_problem_qids("A", type_mappings) + get_problem_qids("B", type_mappings)}
    full_dataset = Dataset(src_data, type_mappings, correct_as_label=True, qids_for_subseq_split=problem_qids,
                           concat_visits=options.concat_visits, time_ratios=not ckt, log_time=ckt, qid_seq=not ckt)

    # Load model
    use_behavior_model = True
    if use_behavior_model:
        behavior_model = LSTMModel(Mode.IRT, type_mappings, options, available_qids=problem_qids).to(device)
    else:
        behavior_model = None
    model = IRT(len(type_mappings["student_ids"]), len(type_mappings["question_ids"]), behavior_model).to(device)
    model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
    model.eval()

    # Evaluate on test set
    stratify_labels = np.array([entry["student_id"] for entry in full_dataset])
    test_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=221)
    _, test_idx = next(test_skf.split(full_dataset, stratify_labels))
    test_data = torch.utils.data.Subset(full_dataset, test_idx)
    full_test_loader = torch.utils.data.DataLoader(
        test_data,
        collate_fn=Collator(),
        batch_size=BATCH_SIZE
    )
    loss, accuracy, auc, kappa, aggregated = evaluate_model(model, full_test_loader, Mode.PREDICT)
    print(f"All: Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, Adj AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")
    complete_idxs = [idx for idx, entry in enumerate(test_data) if entry["complete"]]
    complete_test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_data, complete_idxs),
        collate_fn=Collator(),
        batch_size=BATCH_SIZE
    )
    loss, accuracy, auc, kappa, aggregated = evaluate_model(model, complete_test_loader, Mode.PREDICT)
    print(f"Only Complete: Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, Adj AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")

    # Plot learned ability vs. student score over both blocks
    student_idxs = [type_mappings["student_ids"][str(seq["student_id"])] for seq in src_data]
    abilities = torch.nn.Softplus()(model.ability).detach().cpu().numpy()[student_idxs]
    scores = [seq["block_a_score"] + seq["block_b_score"] for seq in src_data]
    plt.plot(abilities, scores, "bo")
    plt.xlabel("Learned Ability")
    plt.ylabel("Total Score")
    plt.show()

    # Plot question difficulty vs. avg correctness
    qid_to_score = {}
    qid_to_num = {}
    for entry in full_dataset:
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
