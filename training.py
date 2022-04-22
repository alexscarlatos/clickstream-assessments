import json
import time
from typing import Dict, List
import torch
import numpy as np
from sklearn import metrics
from data_processing import load_type_mappings, load_question_info, load_event_types_per_question, get_problem_qids
from data_loading import Dataset, Collator, Sampler
from per_question_data_loading import PerQuestionDataset, PerQuestionCollator
from model import LSTMModel
from joint_model import JointModel
from ckt_model import CKTJoint
from baseline import CopyBaseline
from utils import device
from constants import Mode, TrainOptions

def get_data(data_filename: str, partition: float = None, three_way_split: bool = False) -> List[dict]:
    print("Loading data")
    with open(data_filename) as data_file:
        data: List[dict] = json.load(data_file)
    data_len = len(data)

    if partition:
        if three_way_split:
            data.sort(key=lambda seq: (seq["data_class"], seq["student_id"]))
            res = [[],[]]
            chunk_size = int(data_len / 3)
            for i in range(0, data_len, chunk_size):
                chunk = data[i:i + chunk_size]
                res[0] += chunk[:int(partition * chunk_size)]
                res[1] += chunk[int(partition * chunk_size):]
        else:
            res = [
                data[:int(partition * data_len)],
                data[int(partition * data_len):],
            ]
        # Ensure no overlap between partitions
        assert not any(vd["student_id"] == td["student_id"] for vd in res[1] for td in res[0])
        print(f"Data loaded; Train size: {len(res[0])}, Val size: {len(res[1])}")
        return res
    else:
        print(f"Data size: {data_len}")
        return data

def get_labels(task: str, train: bool):
    if task == "comp":
        if train:
            label_filename = "data/train_labels.json"
        else:
            label_filename = "data/test_labels.json"
    elif task == "score":
        label_filename = "data/label_score.json"
    elif task == "q_stats":
        label_filename = "data/label_q_stats.json"
    else:
        raise Exception(f"Invalid task {task}")

    with open(label_filename) as label_file:
        return json.load(label_file)

def get_block_a_qids(type_mappings: Dict[str, Dict[str, int]]) -> torch.BoolTensor:
    question_info = load_question_info()
    block_a_qids = [False] * len(type_mappings["question_ids"])
    for q_str, qid in type_mappings["question_ids"].items():
        if question_info[q_str]["block"] in ("A", "any"):
            block_a_qids[qid] = True
    return torch.BoolTensor(block_a_qids)

def get_event_types_by_qid(type_mappings: Dict[int, Dict[str, int]]) -> Dict[int, torch.BoolTensor]:
    qid_to_event_types = load_event_types_per_question()
    qid_to_event_bool_tensors = {}
    for qid, event_types in qid_to_event_types.items():
        qid_int = int(qid)
        qid_to_event_bool_tensors[qid_int] = torch.zeros(len(type_mappings["event_types"])).type(torch.bool)
        qid_to_event_bool_tensors[qid_int][event_types] = True
    return qid_to_event_bool_tensors

def get_block_a_problem_qids(type_mappings):
    question_info = load_question_info()
    for qid_str, info in question_info.items():
        if info["block"] != "A" or info["answer"] == "na":
            continue
        yield type_mappings["question_ids"][qid_str], qid_str

def get_chunk_sizes(dataset):
    """
    Given a dataset sorted by data_class
    Return the number of elements of each class, in sorted order
    """
    chunk_sizes = []
    cur_chunk_size = 0
    cur_class = dataset[0]["data_class"]
    allocated_classes = set()
    for seq in dataset:
        assert seq["data_class"] not in allocated_classes, "Data not sorted by class"
        if seq["data_class"] != cur_class:
            allocated_classes.add(cur_class)
            cur_class = seq["data_class"]
            chunk_sizes.append(cur_chunk_size)
            cur_chunk_size = 1
        else:
            cur_chunk_size += 1
    chunk_sizes.append(cur_chunk_size) # Add last chunk since no class change will occur at end
    assert sum(chunk_sizes) == len(dataset)
    return chunk_sizes

def evaluate_model(model, validation_loader: torch.utils.data.DataLoader, mode: Mode):
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in validation_loader:
            loss, predictions = model(batch)
            if mode == Mode.PRE_TRAIN or mode == Mode.CKT_ENCODE:
                has_events = predictions[0] is not None
                has_qids = predictions[1] is not None
                has_correctness = predictions[2] is not None
                event_types = batch["event_types"].view(-1).detach().cpu().numpy()
                per_event_qids = has_qids and predictions[1].shape == event_types.shape
                if per_event_qids:
                    qids = batch["question_ids"].view(-1).detach().cpu().numpy() if has_qids else None
                else:
                    qids = batch["visits"]["qids"].view(-1).detach().cpu().numpy() if has_qids else None
                per_event_correctness = has_correctness and predictions[2].shape == event_types.shape
                if per_event_correctness:
                    correctness = batch["correctness"].view(-1).detach().cpu().numpy() if has_correctness else None
                else:
                    correctness = batch["visits"]["correctness"].view(-1).detach().cpu().numpy() if has_correctness else None
                if "pretrain_mask" in batch:
                    mask = batch["pretrain_mask"].view(-1).detach().cpu().numpy()
                else:
                    mask = batch["mask"].view(-1).detach().cpu().numpy()
                if batch.get("visits"):
                    visit_mask = batch["visits"]["mask"].view(-1).detach().cpu().numpy()

                all_predictions.append([
                    predictions[0][mask] if has_events else [],
                    predictions[1][mask if per_event_qids else visit_mask] if has_qids else [],
                    predictions[2][mask if per_event_correctness else visit_mask] if has_correctness else [],
                ])
                all_labels.append([
                    event_types[mask] if has_events else [],
                    qids[mask if per_event_qids else visit_mask] if has_qids else [],
                    correctness[mask if per_event_correctness else visit_mask] if has_correctness else [],
                ])
            if mode == Mode.PREDICT:
                all_predictions.append(predictions)
                all_labels.append(batch["labels"].detach().cpu().numpy())
            total_loss += float(loss.detach().cpu().numpy())
            num_batches += 1

    if mode == Mode.PRE_TRAIN or mode == Mode.CKT_ENCODE:
        event_preds = np.concatenate([preds[0] for preds in all_predictions])
        event_labels = np.concatenate([labels[0] for labels in all_labels])
        event_accuracy = metrics.accuracy_score(event_labels, event_preds) if event_preds.size else 0
        qid_preds = np.concatenate([preds[1] for preds in all_predictions])
        qid_labels = np.concatenate([labels[1] for labels in all_labels])
        qid_accuracy = metrics.accuracy_score(qid_labels, qid_preds) if qid_preds.size else 0
        correctness_preds = np.concatenate([preds[2] for preds in all_predictions])
        correctness_labels = np.concatenate([labels[2] for labels in all_labels])
        correctness_accuracy = metrics.accuracy_score(correctness_labels, correctness_preds) if correctness_preds.size else 0
        return total_loss / num_batches, event_accuracy, qid_accuracy, correctness_accuracy
    if mode == Mode.PREDICT:
        all_preds_np = np.concatenate(all_predictions, axis=0)
        all_labels_np = np.concatenate(all_labels, axis=0)
        # AUC is area under ROC curve, and is calculated on non-collapsed predictions
        # In multi-label case, will return the average AUC across all labels
        auc = metrics.roc_auc_score(all_labels_np, all_preds_np)
        adj_auc = 2 * (auc - .5)
        # Collapse predictions to calculate accuracy and kappa
        all_preds_np[all_preds_np < 0] = 0
        all_preds_np[all_preds_np > 0] = 1
        # Flatten to handle the multi-label case
        all_preds_np = all_preds_np.flatten()
        all_labels_np = all_labels_np.flatten()
        accuracy = metrics.accuracy_score(all_labels_np, all_preds_np)
        kappa = metrics.cohen_kappa_score(all_labels_np, all_preds_np)
        agg = adj_auc + kappa
        return total_loss / num_batches, accuracy, auc, kappa, agg

def train(model, mode: Mode, model_name: str, train_loader, validation_loader, lr=1e-4, weight_decay=1e-6, epochs=200, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    torch.autograd.set_detect_anomaly(True) # Pause exectuion and get stack trace if something weird happens (ex: NaN grads)
    best_metric = None
    best_stats = None
    cur_stats = None
    best_epoch = 0
    for epoch in range(epochs):
        start_time = time.time()
        model.train() # Set model to training mode
        train_loss = 0
        num_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _ = model(batch)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu().numpy())
            num_batches += 1

        model.eval() # Set model to evaluation mode
        if mode == Mode.PRE_TRAIN or mode == Mode.CKT_ENCODE:
            train_loss, train_evt_acc, train_qid_acc, train_crct_acc = evaluate_model(model, train_loader, mode)
            val_loss, val_evt_acc, val_qid_acc, val_crct_acc = evaluate_model(model, validation_loader, mode) if validation_loader else (0, 0, 0, 0)
            cur_stats = [epoch, val_loss, val_evt_acc, val_qid_acc, val_crct_acc]
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Accuracy: Event: {train_evt_acc:.3f}, QID: {train_qid_acc:.3f}, Correctness: {train_crct_acc:.3f}, "
                f"Val Loss: {val_loss:.3f}, Accuracy: Event: {val_evt_acc:.3f}, QID: {val_qid_acc:.3f}, Correctness: {val_crct_acc:.3f}, "
                f"Time: {time.time() - start_time:.2f}")
        if mode == Mode.PREDICT:
            train_loss, train_accuracy, train_auc, train_kappa, train_agg = evaluate_model(model, train_loader, mode)
            val_loss, val_accuracy, val_auc, val_kappa, val_agg = evaluate_model(model, validation_loader, mode) if validation_loader else (0, 0, 0, 0, 0)
            cur_stats = [epoch, val_loss, val_accuracy, val_auc, val_kappa, val_agg]
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Acc: {train_accuracy:.3f}, AUC: {train_auc:.3f}, Kappa: {train_kappa:.3f}, Agg: {train_agg:.3f}, "
                f"Val Loss: {val_loss:.3f}, Acc: {val_accuracy:.3f}, AUC: {val_auc:.3f}, Kappa: {val_kappa:.3f}, Agg: {val_agg:.3f}, "
                f"Time: {time.time() - start_time:.2f}")

        # Save model for best validation metric
        # if not best_metric or (val_auc > best_metric if mode == Mode.PREDICT else val_loss < best_metric):
        if not best_metric or val_loss < best_metric:
            # best_metric = val_auc if mode == Mode.PREDICT else val_loss
            best_metric = val_loss
            best_epoch = epoch
            best_stats = cur_stats
            print("Saving model")
            torch.save(model.state_dict(), f"{model_name}.pt")

        # Stop training if we haven't improved in a while
        # if epoch - best_epoch >= patience:
        #     print("Early stopping")
        #     break

    return best_stats

def pretrain_and_split_data(model_name: str, data_file: str, options: TrainOptions):
    train_data, val_data = get_data(data_file or "data/train_data.json", .8)
    return pretrain(model_name, train_data, val_data, options)

def pretrain(model_name: str, train_data: List[dict], val_data: List[dict], options: TrainOptions):
    type_mappings = load_type_mappings()

    if options.per_q_arch:
        block_a_qids = {qid for _, qid in get_problem_qids("A", type_mappings)}
        train_dataset = Dataset(train_data, type_mappings, qids_for_subseq_split=block_a_qids, concat_visits=options.concat_visits,
                                time_ratios=True, log_time=False, qid_seq=True, correctness_seq=options.use_correctness)
        train_dataset.set_data_class("question_id")
        train_dataset.sort_by_data_class()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=Collator(options.random_trim),
            batch_sampler=Sampler(get_chunk_sizes(train_dataset), options.batch_size)
        )
        val_dataset = Dataset(val_data, type_mappings, qids_for_subseq_split=block_a_qids, concat_visits=options.concat_visits,
                                time_ratios=True, log_time=False, qid_seq=True, correctness_seq=options.use_correctness)
        val_dataset.set_data_class("question_id")
        val_dataset.sort_by_data_class()
        validation_loader = torch.utils.data.DataLoader(
            val_dataset,
            collate_fn=Collator(),
            batch_sampler=Sampler(get_chunk_sizes(val_dataset))
        ) if val_data is not None else None

        options.use_visit_pt_objs = False
        model = JointModel(Mode.PRE_TRAIN, type_mappings, options).to(device)
        train(model, Mode.PRE_TRAIN, model_name, train_loader, validation_loader, lr=options.lr, weight_decay=options.weight_decay, epochs=options.epochs, patience=10)
    else:
        train_loader = torch.utils.data.DataLoader(
        Dataset(train_data, type_mappings, time_ratios=True),
            collate_fn=Collator(options.random_trim),
            batch_size=options.batch_size,
            shuffle=True,
            drop_last=True
        )
        validation_loader = torch.utils.data.DataLoader(
            Dataset(val_data, type_mappings, time_ratios=True),
            collate_fn=Collator(),
            batch_size=len(val_data)
        ) if val_data is not None else None

        model = LSTMModel(Mode.PRE_TRAIN, type_mappings, options, available_qids=get_block_a_qids(type_mappings)).to(device)
        train(model, Mode.PRE_TRAIN, model_name, train_loader, validation_loader, lr=options.lr, weight_decay=options.weight_decay, epochs=options.epochs, patience=10)

def test_pretrain(model_name: str, data_file: str, options: TrainOptions):
    type_mappings = load_type_mappings()
    # Load test data
    test_data = get_data(data_file or "data/test_data.json")
    test_loader = torch.utils.data.DataLoader(
        Dataset(test_data, type_mappings, time_ratios=True),
        collate_fn=Collator(),
        batch_size=len(test_data)
    )

    # Load model
    model_type = "lstm"
    if model_type == "lstm":
        type_mappings = load_type_mappings()
        model = LSTMModel(Mode.PRE_TRAIN, type_mappings, options, available_qids=get_block_a_qids(type_mappings)).to(device)
        model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
        model.eval()
    elif model_type == "baseline":
        model = CopyBaseline()

    # Test model
    loss, event_accuracy, qid_accuracy, correctness_accuracy = evaluate_model(model, test_loader, Mode.PRE_TRAIN)
    print(f"Loss: {loss:.3f}, Accuracy: Events: {event_accuracy:.3f}, QIDs: {qid_accuracy:.3f}, Correctness: {correctness_accuracy:.3f}")

def create_predictor_model(pretrain_model_name: str, mode: Mode, type_mappings: dict, options: TrainOptions, num_labels: int = 1, pred_classes: list = None):
    model = LSTMModel(mode, type_mappings, options, num_labels=num_labels, pred_classes=pred_classes)

    # Copy pretrained parameters based on settings
    states_to_copy = []
    if options.use_pretrained_embeddings:
        states_to_copy += ["question_embeddings", "event_type_embeddings"]
    if options.use_pretrained_weights:
        states_to_copy += ["lstm"]
    if options.use_pretrained_head:
        states_to_copy += ["attention", "hidden_layers", "pred_output_layer"]
    if states_to_copy:
        state_dict = model.state_dict()
        pretrained_state = torch.load(f"{pretrain_model_name}.pt", map_location=device)
        for high_level_state in states_to_copy:
            for state in state_dict:
                if state.startswith(high_level_state):
                    print("Copying", state)
                    state_dict[state] = pretrained_state[state]
        model.load_state_dict(state_dict)

    # Freeze parameters based on settings
    components_to_freeze = []
    if options.freeze_embeddings:
        components_to_freeze += [model.question_embeddings, model.event_type_embeddings]
    if options.freeze_model:
        components_to_freeze += [model.lstm]
    for component in components_to_freeze:
        for param in component.parameters():
            param.requires_grad = False

    model = model.to(device)
    return model

def train_predictor_and_split_data(pretrain_model_name: str, model_name: str, data_file: str, options: TrainOptions):
    train_data, val_data = get_data(data_file or "data/train_data.json", .8, options.mixed_time)
    train_labels = get_labels(options.task, True)
    return train_predictor(pretrain_model_name, model_name, train_data, val_data, train_labels, options)

def train_predictor(pretrain_model_name: str, model_name: str, train_data: List[dict], val_data: List[dict], labels: dict, options: TrainOptions):
    type_mappings = load_type_mappings()

    if options.per_q_arch:
        block_a_qids = {qid for _, qid in get_problem_qids("A", type_mappings)}
        train_loader = torch.utils.data.DataLoader(
            PerQuestionDataset(train_data, labels, block_a_qids, False, True, options.concat_visits, options.use_correctness),
            collate_fn=PerQuestionCollator(block_a_qids),
            batch_size=options.batch_size,
            shuffle=True,
            drop_last=True
        )
        validation_loader = torch.utils.data.DataLoader(
            PerQuestionDataset(val_data, labels, block_a_qids, False, True, options.concat_visits, options.use_correctness),
            collate_fn=PerQuestionCollator(block_a_qids),
            batch_size=options.batch_size
        ) if val_data is not None else None

        num_labels = len(get_problem_qids("B", type_mappings)) if options.task == "q_stats" else 1
        model = JointModel(Mode.PREDICT, type_mappings, options, num_labels=num_labels, num_input_qids=len(block_a_qids)).to(device)
        if options.use_pretrained_weights:
            model.load_params(torch.load(f"{pretrain_model_name}.pt", map_location=device))
        if options.freeze_model:
            for param in model.encoder.parameters():
                param.requires_grad = False
        return train(model, Mode.PREDICT, model_name, train_loader, validation_loader, lr=options.lr, weight_decay=options.weight_decay, epochs=options.epochs)
    else:
        train_chunk_sizes = [int(len(train_data) / 3)] * 3 if options.mixed_time else [len(train_data)]
        val_chunk_sizes = ([int(len(val_data) / 3)] * 3 if options.mixed_time else [len(val_data)]) if val_data is not None else None
        train_loader = torch.utils.data.DataLoader(
            Dataset(train_data, type_mappings, labels=labels, engineered_features=options.engineered_features),
            collate_fn=Collator(),
            batch_sampler=Sampler(train_chunk_sizes, options.batch_size)
        )
        validation_loader = torch.utils.data.DataLoader(
            Dataset(val_data,type_mappings, labels=labels, engineered_features=options.engineered_features),
            collate_fn=Collator(),
            batch_sampler=Sampler(val_chunk_sizes)
        ) if val_data is not None else None

        num_labels = len(get_problem_qids("B", type_mappings)) if options.task == "q_stats" else 1
        model = create_predictor_model(pretrain_model_name, Mode.PREDICT, type_mappings, options, num_labels=num_labels)
        return train(model, Mode.PREDICT, model_name, train_loader, validation_loader, lr=options.lr, weight_decay=options.weight_decay, epochs=options.epochs, patience=15)

def test_predictor(model_name: str, data_file: str, options: TrainOptions):
    test_data = get_data(data_file or "data/test_data.json")
    return test_predictor_with_data(model_name, test_data, options)

def test_predictor_with_data(model_name: str, test_data: List[dict], options: TrainOptions):
    type_mappings = load_type_mappings()

    if options.per_q_arch:
        block_a_qids = {qid for _, qid in get_problem_qids("A", type_mappings)}
        labels = get_labels(options.task, False)
        test_loader = torch.utils.data.DataLoader(
            PerQuestionDataset(test_data, labels, block_a_qids, False, True, options.concat_visits, options.use_correctness),
            collate_fn=PerQuestionCollator(block_a_qids),
            batch_size=options.batch_size
        )

        num_labels = len(get_problem_qids("B", type_mappings)) if options.task == "q_stats" else 1
        model = JointModel(Mode.PREDICT, type_mappings, options, num_labels=num_labels, num_input_qids=len(block_a_qids)).to(device)
        model.load_params(torch.load(f"{model_name}.pt", map_location=device))
        model.eval()

        # Test model
        loss, accuracy, auc, kappa, aggregated = evaluate_model(model, test_loader, Mode.PREDICT)
        print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")
        return [loss, accuracy, auc, kappa, aggregated]
    else:
        # Load test data
        chunk_sizes = [len([seq for seq in test_data if seq["data_class"] == data_class]) for data_class in ["10", "20", "30"]]
        chunk_sizes = [chunk_size for chunk_size in chunk_sizes if chunk_size] # Filter out empty chunks
        test_loader = torch.utils.data.DataLoader(
            Dataset(test_data, type_mappings, labels=get_labels(options.task, False), engineered_features=options.engineered_features),
            collate_fn=Collator(),
            batch_sampler=Sampler(chunk_sizes)
        )

        # Load model
        model_type = "lstm"
        if model_type == "lstm":
            num_labels = len(get_problem_qids("B", type_mappings)) if options.task == "q_stats" else 1
            model = LSTMModel(Mode.PREDICT, type_mappings, options, num_labels=num_labels).to(device)
            model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
            model.eval()
        elif model_type == "baseline":
            model = CopyBaseline()

        # Test model
        loss, accuracy, auc, kappa, aggregated = evaluate_model(model, test_loader, Mode.PREDICT)
        print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")
        return [loss, accuracy, auc, kappa, aggregated]

def train_ckt_encoder_and_split_data(model_name: str, data_file: str, options: TrainOptions):
    train_data, val_data = get_data(data_file or "data/train_data.json", .8)
    return train_ckt_encoder(model_name, train_data, val_data, options)

def train_ckt_encoder(model_name: str, train_data: List[dict], val_data: List[dict], options: TrainOptions):
    type_mappings = load_type_mappings()
    qid_to_event_types = get_event_types_by_qid(type_mappings)
    block_a_qids = {qid for _, qid in get_problem_qids("A", type_mappings)}

    train_dataset = Dataset(train_data, type_mappings, qids_for_subseq_split=block_a_qids, concat_visits=options.concat_visits,
                            log_time=True, qid_seq=False, correctness_seq=options.use_correctness)
    train_dataset.set_data_class("question_id")
    train_dataset.sort_by_data_class()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=Collator(),
        batch_sampler=Sampler(get_chunk_sizes(train_dataset), options.batch_size)
    )
    val_dataset = Dataset(val_data, type_mappings, qids_for_subseq_split=block_a_qids, concat_visits=options.concat_visits,
                          log_time=True, qid_seq=False, correctness_seq=options.use_correctness)
    val_dataset.set_data_class("question_id")
    val_dataset.sort_by_data_class()
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        collate_fn=Collator(),
        batch_sampler=Sampler(get_chunk_sizes(val_dataset))
    ) if val_data is not None else None

    model = CKTJoint(Mode.CKT_ENCODE, type_mappings, options.use_correctness, qid_to_event_types).to(device)
    train(model, Mode.CKT_ENCODE, model_name, train_loader, validation_loader, lr=options.lr, weight_decay=options.weight_decay, epochs=options.epochs)

def test_ckt_encoder(model_name: str, data_file: str, options: TrainOptions):
    type_mappings = load_type_mappings()
    qid_to_event_types = get_event_types_by_qid(type_mappings)
    block_a_qids = {qid for _, qid in get_problem_qids("A", type_mappings)}

    test_data = get_data(data_file or "data/test_data.json")
    test_dataset = Dataset(test_data, type_mappings, qids_for_subseq_split=block_a_qids, concat_visits=options.concat_visits,
                           log_time=True, qid_seq=False, correctness_seq=options.use_correctness)
    test_dataset.set_data_class("question_id")
    test_dataset.sort_by_data_class()
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=Collator(),
        batch_sampler=Sampler(get_chunk_sizes(test_dataset))
    )
    model = CKTJoint(Mode.CKT_ENCODE, type_mappings, options.use_correctness, qid_to_event_types).to(device)
    model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
    model.eval()
    loss, event_accuracy, _, correctness_accuracy = evaluate_model(model, test_loader, Mode.CKT_ENCODE)
    print(f"Loss: {loss:.3f}, Accuracy: Event: {event_accuracy:.3f}, Correctness: {correctness_accuracy:.3f}")

def train_ckt_predictor_and_split_data(encoder_model_name: str, model_name: str, data_file: str, options: TrainOptions):
    train_data, val_data = get_data(data_file or "data/train_data.json", .8)
    train_labels = get_labels(options.task, True)
    return train_ckt_predictor(encoder_model_name, model_name, train_data, val_data, train_labels, options)

def train_ckt_predictor(encoder_model_name: str, model_name: str, train_data: List[dict], val_data: List[dict], labels: dict, options: TrainOptions):
    type_mappings = load_type_mappings()
    qid_to_event_types = get_event_types_by_qid(type_mappings)
    block_a_qids = {qid for _, qid in get_problem_qids("A", type_mappings)}

    # Load data
    train_loader = torch.utils.data.DataLoader(
        PerQuestionDataset(train_data, labels, block_a_qids, True, False, options.concat_visits, options.use_correctness),
        collate_fn=PerQuestionCollator(block_a_qids),
        batch_size=options.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        PerQuestionDataset(val_data, labels, block_a_qids, True, False, options.concat_visits, options.use_correctness),
        collate_fn=PerQuestionCollator(block_a_qids),
        batch_size=options.batch_size
    )

    # Create and train model
    num_labels = len(get_problem_qids("B", type_mappings)) if options.task == "q_stats" else 1
    model = CKTJoint(Mode.PREDICT, type_mappings, options.use_correctness, qid_to_event_types,
                     concat_visits=options.concat_visits, num_labels=num_labels, num_input_qids=len(block_a_qids)).to(device)
    # If the option is set will load all parameters of pretrained model, could just be encoder but also could be prediction head
    if options.use_pretrained_weights:
        model.load_params(torch.load(f"{encoder_model_name}.pt", map_location=device))
    if options.freeze_model:
        for param in model.encoder.parameters():
            param.requires_grad = False
    return train(model, Mode.PREDICT, model_name, train_loader, val_loader, options.lr, options.weight_decay, options.epochs)

def test_ckt_predictor(model_name: str, data_file: str, options: TrainOptions):
    test_data = get_data(data_file or "data/test_data.json")
    return test_ckt_predictor_with_data(model_name, test_data, options)

def test_ckt_predictor_with_data(model_name: str, test_data: List[dict], options: TrainOptions):
    type_mappings = load_type_mappings()
    qid_to_event_types = get_event_types_by_qid(type_mappings)
    block_a_qids = {qid for _, qid in get_problem_qids("A", type_mappings)}

    # Load data
    labels = get_labels(options.task, False)
    test_loader = torch.utils.data.DataLoader(
        PerQuestionDataset(test_data, labels, block_a_qids, True, False, options.concat_visits, options.use_correctness),
        collate_fn=PerQuestionCollator(block_a_qids),
        batch_size=options.batch_size
    )

    # Load model
    num_labels = len(get_problem_qids("B", type_mappings)) if options.task == "q_stats" else 1
    model = CKTJoint(Mode.PREDICT, type_mappings, options.use_correctness, qid_to_event_types,
                     concat_visits=options.concat_visits, num_labels=num_labels, num_input_qids=len(block_a_qids)).to(device)
    model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
    model.eval()

    # Test model
    loss, accuracy, auc, kappa, aggregated = evaluate_model(model, test_loader, Mode.PREDICT)
    print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")
    return [loss, accuracy, auc, kappa, aggregated]
