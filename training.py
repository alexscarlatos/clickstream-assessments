import json
import time
from typing import List
import torch
import numpy as np
from sklearn import metrics, manifold, decomposition
import matplotlib.pyplot as plt
from data_processing import load_type_mappings
from data_loading import Dataset, Sampler, Collator
from model import LSTMModel, Mode, TrainOptions
from baseline import CopyBaseline
from utils import device

BATCH_SIZE = 32

def get_data(data_filename, partition=None, three_way_split=False):
    print("Loading data")
    with open(data_filename) as data_file:
        data = json.load(data_file)
    data_len = len(data)

    if partition:
        if three_way_split:
            data.sort(key=lambda sequence: sequence["data_class"])
            res = [[],[]]
            chunk_size = int(data_len / 3)
            for i in range(0, data_len, chunk_size):
                chunk = data[i:i + chunk_size]
                # TODO: shuffle chunks without breaking it
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

def evaluate_model(model, validation_loader: torch.utils.data.DataLoader, mode: Mode):
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in validation_loader:
            loss, predictions = model(batch)
            if mode == Mode.PRE_TRAIN:
                event_types = batch["event_types"].view(-1).detach().cpu().numpy()
                mask = batch["mask"].view(-1).detach().cpu().numpy()
                all_predictions.append(predictions[mask == 1])
                all_labels.append(event_types[mask == 1])
            if mode == Mode.PREDICT:
                all_predictions.append(predictions)
                all_labels.append(batch["labels"].detach().cpu().numpy())
            total_loss += float(loss.detach().cpu().numpy())
            num_batches += 1

    all_labels_np = np.concatenate(all_labels, axis=0)
    all_preds_np = np.concatenate(all_predictions, axis=0)

    if mode == Mode.PRE_TRAIN:
        accuracy = metrics.accuracy_score(all_labels_np, all_preds_np)
        return total_loss / num_batches, accuracy, 0, 0, 0
    if mode == Mode.PREDICT:
        # AUC is area under ROC curve, and is calculated on non-collapsed predictions
        auc = metrics.roc_auc_score(all_labels_np, all_preds_np)
        adj_auc = 2 * (auc - .5)
        # Collapse predictions to calculate accuracy and kappa
        all_preds_np[all_preds_np < 0] = 0
        all_preds_np[all_preds_np > 0] = 1
        accuracy = metrics.accuracy_score(all_labels_np, all_preds_np)
        kappa = metrics.cohen_kappa_score(all_labels_np, all_preds_np)
        agg = adj_auc + kappa
        return total_loss / num_batches, accuracy, adj_auc, kappa, agg

def train(model, mode: Mode, model_name: str, train_loader, validation_loader, lr=1e-4, weight_decay=1e-6, epochs=200, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    torch.autograd.set_detect_anomaly(True) # Pause exectuion and get stack trace if something weird happens (ex: NaN grads)
    best_metric = None
    best_stats = None
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
        train_loss, train_accuracy, train_auc, train_kappa, train_agg = evaluate_model(model, train_loader, mode)
        val_loss, val_accuracy, val_auc, val_kappa, val_agg = evaluate_model(model, validation_loader, mode) if validation_loader else (0, 0, 0, 0, 0)
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Acc: {train_accuracy:.3f}, AUC: {train_auc:.3f}, Kappa: {train_kappa:.3f}, Agg: {train_agg:.3f}, "
              f"Val Loss: {val_loss:.3f}, Acc: {val_accuracy:.3f}, AUC: {val_auc:.3f}, Kappa: {val_kappa:.3f}, Agg: {val_agg:.3f}, "
              f"Time: {time.time() - start_time:.2f}")

        # Save model for best validation metric
        # if not best_metric or (val_agg > best_metric if mode == Mode.PREDICT else val_loss < best_metric):
        if not best_metric or val_loss < best_metric:
            # best_metric = val_agg if mode == Mode.PREDICT else val_loss
            best_metric = val_loss
            best_epoch = epoch
            best_stats = [epoch, val_loss, val_accuracy, val_auc, val_kappa, val_agg]
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
    train_loader = torch.utils.data.DataLoader(
        Dataset(train_data),
        collate_fn=Collator(options.random_trim),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    validation_loader = torch.utils.data.DataLoader(
        Dataset(val_data),
        collate_fn=Collator(),
        batch_size=len(val_data)
    ) if val_data is not None else None

    model = LSTMModel(Mode.PRE_TRAIN, load_type_mappings(), options).to(device)
    train(model, Mode.PRE_TRAIN, model_name, train_loader, validation_loader, lr=options.lr, weight_decay=options.weight_decay, epochs=options.epochs, patience=10)

def test_pretrain(model_name: str, data_file: str, options: TrainOptions):
    # Load test data
    test_data = get_data(data_file or "data/test_data.json")
    test_loader = torch.utils.data.DataLoader(
        Dataset(test_data),
        collate_fn=Collator(),
        batch_size=len(test_data)
    )

    # Load model
    model_type = "lstm"
    if model_type == "lstm":
        model = LSTMModel(Mode.PRE_TRAIN, load_type_mappings(), options).to(device)
        model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
        model.eval()
    elif model_type == "baseline":
        model = CopyBaseline()

    # Test model
    loss, accuracy, _, _, _ = evaluate_model(model, test_loader, Mode.PRE_TRAIN)
    print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")

def train_predictor_and_split_data(pretrain_model_name: str, model_name: str, data_file: str, options: TrainOptions):
    train_data, val_data = get_data(data_file or "data/train_data.json", .8, options.split_data)
    train_labels = get_labels(options.task, True)
    return train_predictor(pretrain_model_name, model_name, train_data, val_data, train_labels, options)

def train_predictor(pretrain_model_name: str, model_name: str, train_data: List[dict], val_data: List[dict], labels: dict, options: TrainOptions):
    # train_chunk_sizes = [int(len(train_data) / 3)] * 3 if options.split_data else [len(train_data)]
    # val_chunk_sizes = ([int(len(val_data) / 3)] * 3 if options.split_data else [len(val_data)]) if val_data else None
    train_loader = torch.utils.data.DataLoader(
        Dataset(train_data, labels, options.engineered_features),
        collate_fn=Collator(options.random_trim),
        # batch_sampler=Sampler(train_chunk_sizes, BATCH_SIZE)
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    validation_loader = torch.utils.data.DataLoader(
        Dataset(val_data, labels, options.engineered_features),
        collate_fn=Collator(),
        # batch_sampler=Sampler(val_chunk_sizes)
        batch_size=len(val_data)
    ) if val_data is not None else None
    model = LSTMModel(Mode.PREDICT, load_type_mappings(), options)

    # Copy pretrained parameters based on settings
    states_to_copy = []
    if options.use_pretrained_embeddings:
        states_to_copy += ["question_embeddings", "question_type_embeddings", "event_type_embeddings"]
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
        components_to_freeze += [model.question_embeddings, model.question_type_embeddings, model.event_type_embeddings]
    if options.freeze_model:
        components_to_freeze += [model.lstm]
    for component in components_to_freeze:
        for param in component.parameters():
            param.requires_grad = False

    model = model.to(device)
    return train(model, Mode.PREDICT, model_name, train_loader, validation_loader, lr=options.lr, weight_decay=options.weight_decay, epochs=options.epochs, patience=15)

def test_predictor(model_name: str, data_file: str, options: TrainOptions):
    # Load test data
    test_data = get_data(data_file or "data/test_data.json")
    chunk_sizes = [len([seq for seq in test_data if seq["data_class"] == data_class]) for data_class in ["10", "20", "30"]]
    chunk_sizes = [chunk_size for chunk_size in chunk_sizes if chunk_size]
    test_loader = torch.utils.data.DataLoader(
        Dataset(test_data, get_labels(options.task, False), options.engineered_features),
        collate_fn=Collator(),
        # batch_sampler=Sampler(chunk_sizes)
        batch_size=len(test_data)
    )

    # Load model
    model_type = "lstm"
    if model_type == "lstm":
        model = LSTMModel(Mode.PREDICT, load_type_mappings(), options).to(device)
        model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
        model.eval()
    elif model_type == "baseline":
        model = CopyBaseline()

    # Test model
    loss, accuracy, auc, kappa, aggregated = evaluate_model(model, test_loader, Mode.PREDICT)
    print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, Adj AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")
    return [loss, accuracy, auc, kappa, aggregated]

def cluster(model_name: str, data_file: str, options: TrainOptions):
    # Load data
    data = get_data(data_file or "data/train_data.json")
    data_loader = torch.utils.data.DataLoader(
        Dataset(data, get_labels(options.task, True)),
        collate_fn=Collator(),
        batch_size=len(data),
        shuffle=True,
        drop_last=True
    )

    # Load model
    model = LSTMModel(Mode.CLUSTER, load_type_mappings(), options).to(device)
    model.load_state_dict(torch.load(f"{model_name}.pt", map_location=device))
    model.eval()

    # Extract latent state for each sequence in the dataset
    print("Extracting latent states")
    with torch.no_grad():
        for batch in data_loader:
            latent_states, predictions = model(batch)
            latent_states = latent_states.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()

    # Represent latent states in 2D space
    print("Performing Dimension Reduction")
    transformer = decomposition.PCA(2)
    reduced_states = transformer.fit_transform(latent_states)
    true_states = reduced_states[labels == 1]
    false_states = reduced_states[labels == 0]
    plt.plot(true_states[:,0], true_states[:,1], 'bo', label="NAEP EDM Label = 1")
    plt.plot(false_states[:,0], false_states[:,1], 'ro', label="NAEP EDM Label = 0")
    # true_states = reduced_states[predictions > 0]
    # false_states = reduced_states[predictions <= 0]
    # plt.plot(true_states[:,0], true_states[:,1], 'bo', label="NAEP EDM Prediction = 1")
    # plt.plot(false_states[:,0], false_states[:,1], 'ro', label="NAEP EDM Prediction = 0")
    plt.title("Dimension-Reduced Latent State Vectors")
    plt.legend()
    plt.show()
