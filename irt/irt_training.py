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
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from data_processing import load_type_mappings, get_problem_qids
from data_loading import Dataset, Collator, Sampler
from training import get_data, train, evaluate_model, create_predictor_model, get_event_types_by_qid, get_chunk_sizes, BATCH_SIZE
from model import LSTMModel
from ckt_model import CKTJoint
from irt.irt_model import IRT
from constants import Mode, TrainOptions, Correctness
from utils import device, initialize_seeds


def get_all_problem_qids(type_mappings: dict):
    return {qid for _, qid in get_problem_qids("A", type_mappings) + get_problem_qids("B", type_mappings)}

def get_processed_dataset(src_data: list, type_mappings: dict, ckt: bool, problem_qids: set, options: TrainOptions):
    full_dataset = Dataset(src_data, type_mappings, correct_as_label=True, qids_for_subseq_split=problem_qids,
                           concat_visits=options.concat_visits, time_ratios=not ckt, log_time=ckt, qid_seq=not ckt, correctness_seq=False)
    full_dataset.shuffle(221) # Randomly arrange the data
    return full_dataset

def irt(pretrained_name: str, model_name: str, data_file: str, use_behavior_model: bool, ckt: bool, options: TrainOptions):
    # Not using correctness info to verify that additional performance come strictly from behavioral data
    options.use_correctness = False
    # Since representing a single question at a time, task switching cannot be represented
    options.use_visit_pt_objs = False

    # Get dataset
    type_mappings = load_type_mappings()
    src_data = get_data(data_file)
    problem_qids = get_all_problem_qids(type_mappings)
    full_dataset = get_processed_dataset(src_data, type_mappings, ckt, problem_qids, options)

    # Decide if we need to batch the data by some classification
    batch_by_class = False
    if options.multi_head or ckt:
        batch_by_class = True
        # For CKT, we need to batch by question_id to run the encoders in parallel for efficiency
        # Otherwise, if doing multi-head prediction for our model, batch by type to have a separate prediction head for each type
        full_dataset.set_data_class("question_id" if ckt else "question_type")
        full_dataset.sort_by_data_class() # Sorted data is prerequisite for chunk calculation and batch sampler

    # Gather metadata used throughout function
    student_ids = [seq["student_id"] for seq in src_data]
    event_types_by_qid = get_event_types_by_qid(type_mappings)
    pred_classes = list({str(seq["data_class"]) for seq in full_dataset})

    # Make sure correct states are all, well, correct
    if False:
        student_to_stats = {seq["student_id"]: seq["q_stats"] for seq in src_data}
        rev_qid_map = {qid: qid_str for qid_str, qid in type_mappings["question_ids"].items()}
        for entry in full_dataset:
            assert entry["correct"] == (student_to_stats[entry["student_id"]][rev_qid_map[entry["question_id"]]]["correct"] == Correctness.CORRECT.value)

    # Do stratified train/test split
    # Balance students and questions as evenly as possible to full train/test each of their unique parameters
    test_skf = MultilabelStratifiedKFold(n_splits=5, shuffle=False) # Not shuffling to preserve order from data_class sort
    stratify_labels = np.array([[entry["student_id"], entry["question_id"]] for entry in full_dataset])
    train_all_idx, test_idx = next(test_skf.split(full_dataset, stratify_labels))
    train_data_all = torch.utils.data.Subset(full_dataset, train_all_idx)
    train_stratify_labels_all = stratify_labels[train_all_idx]
    test_data = torch.utils.data.Subset(full_dataset, test_idx)
    test_chunk_sizes = get_chunk_sizes(test_data)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        collate_fn=Collator(),
        **({"batch_sampler": Sampler(test_chunk_sizes)} if batch_by_class else {"batch_size": BATCH_SIZE})
    )

    def data_stats(prefix: str, data: list):
        student_counter = Counter(entry["student_id"] for entry in data)
        qid_counter = Counter(entry["question_id"] for entry in data)
        for sid in student_ids:
            if sid not in student_counter:
                student_counter[sid] = 0
                print("Missing student", sid)
        for qid in problem_qids:
            if qid not in qid_counter:
                qid_counter[qid] = 0
                print("Missing question", qid)
        student_counts = student_counter.most_common()
        qid_counts = qid_counter.most_common()
        print(f"{prefix} Students: Most: {student_counts[0]}, Least: {student_counts[-1]}; Questions: {qid_counts[0]}, Least: {qid_counts[-1]}")

    data_stats("All:", full_dataset)
    data_stats("Train Full:", train_data_all)
    data_stats("Test:", test_data)

    # Do cross-validation training for IRT model
    val_skf = MultilabelStratifiedKFold(n_splits=5, shuffle=False) # Not shuffling to preserve order from data_class sort
    for k, (train_idx, val_idx) in enumerate(val_skf.split(train_data_all, train_stratify_labels_all)):
        initialize_seeds(221)

        print(f"\n----- Iteration {k +1} -----")
        cur_model_name = f"{model_name}_{k + 1}"
        train_data = torch.utils.data.Subset(train_data_all, train_idx)
        train_chunk_sizes = get_chunk_sizes(train_data)
        val_data = torch.utils.data.Subset(train_data_all, val_idx)
        val_chunk_sizes = get_chunk_sizes(val_data)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            collate_fn=Collator(),
            # TODO: see if this works without drop_last
            **({"batch_sampler": Sampler(train_chunk_sizes, BATCH_SIZE)} if batch_by_class else {"batch_size": BATCH_SIZE, "shuffle": True, "drop_last": True})
        )
        val_loader = torch.utils.data.DataLoader(
            val_data,
            collate_fn=Collator(),
            **({"batch_sampler": Sampler(val_chunk_sizes)} if batch_by_class else {"batch_size": BATCH_SIZE})
        )

        data_stats("Train:", train_data)
        data_stats("Val:", val_data)

        do_pretrain = True
        if use_behavior_model:
            # Train behavior model on train data split
            print("\nPretraining Behavior Model")
            pretrained_behavior_model_name = f"{pretrained_name}_{k + 1}"
            if ckt:
                if do_pretrain:
                    behavior_model_pt = CKTJoint(Mode.CKT_ENCODE, type_mappings, False, event_types_by_qid).to(device)
                    train(behavior_model_pt, Mode.CKT_ENCODE, pretrained_behavior_model_name, train_loader, val_loader,
                        lr=1e-3, weight_decay=1e-6, epochs=20)
                behavior_model = CKTJoint(Mode.IRT, type_mappings, False, event_types_by_qid).to(device)
                behavior_model.load_params(torch.load(f"{pretrained_behavior_model_name}.pt", map_location=device))
                for param in behavior_model.encoder.parameters():
                    param.requires_grad = False
            else:
                if do_pretrain:
                    behavior_model_pt = LSTMModel(Mode.PRE_TRAIN, type_mappings, options, available_qids=problem_qids).to(device)
                    train(behavior_model_pt, Mode.PRE_TRAIN, pretrained_behavior_model_name, train_loader, val_loader,
                        lr=1e-3, weight_decay=1e-6, epochs=20)
                options.freeze_model = True
                options.freeze_embeddings = True
                options.use_pretrained_weights = True
                options.use_pretrained_embeddings = True
                behavior_model = create_predictor_model(pretrained_behavior_model_name, Mode.IRT, type_mappings, options, pred_classes=pred_classes)
        else:
            behavior_model = None

        initialize_seeds(221) # Re-initialize seeds so that the following will be exactly the same whether or not we skip pretraing

        # Train model
        print("\nTraining IRT Model")
        model = IRT(Mode.IRT, len(type_mappings["student_ids"]), len(type_mappings["question_ids"]), behavior_model).to(device)
        # TODO: probably need to increase number of epochs, check for use_behavior_model
        train(model, Mode.PREDICT, cur_model_name, train_loader, val_loader, lr=5e-3, weight_decay=1e-6, epochs=25)
        model.load_state_dict(torch.load(f"{cur_model_name}.pt", map_location=device)) # Load from best epoch
        model.eval()

        # Test model
        loss, accuracy, auc, kappa, aggregated = evaluate_model(model, test_loader, Mode.PREDICT)
        print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")

        # Fine-tune model
        if use_behavior_model:
            print("\nFine-Tuning IRT Model")
            cur_model_ft_name = f"{model_name}_ft_{k + 1}"

            # Unfreeze encoder/pretrained model parameters
            for param in model.behavior_model.parameters():
                param.requires_grad = True

            train(model, Mode.PREDICT, cur_model_ft_name, train_loader, val_loader, lr=1e-4, weight_decay=1e-6, epochs=20)
            model.load_state_dict(torch.load(f"{cur_model_ft_name}.pt", map_location=device)) # Load from best epoch
            model.eval()

            # Test model
            loss, accuracy, auc, kappa, aggregated = evaluate_model(model, test_loader, Mode.PREDICT)
            print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")

        break # TODO: just one fold for now

def get_model_for_testing(mode: Mode, model_name: str, type_mappings: dict, use_behavior_model: dict, ckt: bool,
                          event_types_by_qid: dict, pred_classes: list, options: TrainOptions):
    options.use_correctness = False
    if use_behavior_model:
        if ckt:
            behavior_model = CKTJoint(mode, type_mappings, False, event_types_by_qid).to(device)
        else:
            behavior_model = LSTMModel(mode, type_mappings, options, pred_clases=pred_classes).to(device)
    else:
        behavior_model = None
    model = IRT(mode, len(type_mappings["student_ids"]), len(type_mappings["question_ids"]), behavior_model).to(device)
    model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
    model.eval()
    return model

def test_irt(model_name: str, data_file: str, use_behavior_model: bool, ckt: bool, options: TrainOptions):
    type_mappings = load_type_mappings()
    event_types_by_qid = get_event_types_by_qid(type_mappings)
    src_data = get_data(data_file)

    # Load dataset
    problem_qids = get_all_problem_qids(type_mappings)
    full_dataset = get_processed_dataset(src_data, type_mappings, ckt, problem_qids, options)

    # Decide if we need to batch the data by some classification
    batch_by_class = False
    if options.multi_head or ckt:
        batch_by_class = True
        # For CKT, we need to batch by question_id to run the encoders in parallel for efficiency
        # Otherwise, if doing multi-head prediction for our model, batch by type to have a separate prediction head for each type
        full_dataset.set_data_class("question_id" if ckt else "question_type")
        full_dataset.sort_by_data_class() # Sorted data is prerequisite for chunk calculation and batch sampler
    pred_classes = list({str(seq["data_class"]) for seq in full_dataset})

    # Load model
    model = get_model_for_testing(Mode.IRT, model_name, type_mappings, use_behavior_model, ckt, event_types_by_qid, pred_classes, options)

    # Evaluate on test set
    test_skf = MultilabelStratifiedKFold(n_splits=5, shuffle=False)
    stratify_labels = np.array([[entry["student_id"], entry["question_id"]] for entry in full_dataset])
    _, test_idx = next(test_skf.split(full_dataset, stratify_labels))
    test_data = torch.utils.data.Subset(full_dataset, test_idx)
    test_chunk_sizes = get_chunk_sizes(test_data)
    full_test_loader = torch.utils.data.DataLoader(
        test_data,
        collate_fn=Collator(),
        **({"batch_sampler": Sampler(test_chunk_sizes)} if batch_by_class else {"batch_size": BATCH_SIZE})
    )
    loss, accuracy, auc, kappa, aggregated = evaluate_model(model, full_test_loader, Mode.PREDICT)
    print(f"All: Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")
    complete_idxs = [idx for idx, entry in enumerate(test_data) if entry["complete"]]
    complete_test_data = torch.utils.data.Subset(test_data, complete_idxs)
    complete_test_chunk_sizes = get_chunk_sizes(complete_test_data)
    complete_test_loader = torch.utils.data.DataLoader(
        complete_test_data,
        collate_fn=Collator(),
        **({"batch_sampler": Sampler(complete_test_chunk_sizes)} if batch_by_class else {"batch_size": BATCH_SIZE})
    )
    loss, accuracy, auc, kappa, aggregated = evaluate_model(model, complete_test_loader, Mode.PREDICT)
    print(f"Only Complete: Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")
