import json
import time
import torch
from sklearn import metrics
import numpy as np
from dataset import Dataset, Collator, load_type_mappings
from model import LSTMModel, Mode, TrainOptions
from baseline import CopyBaseline

BATCH_SIZE = 16
NUM_WORKERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data(data_filename, partition=None):
    print("Loading data")
    with open(data_filename) as data_file:
        data = json.load(data_file)
    data_len = len(data)
    if partition:
        res = [
            data[:int(partition * data_len)],
            data[int(partition * data_len):],
        ]
        print(f"Data loaded; Train size: {len(res[0])}, Val size: {len(res[1])}")
        return res
    else:
        print(f"Data size: {data_len}")
        return data

def get_labels(label_filename):
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
        return total_loss / num_batches, accuracy, 0, 0
    if mode == Mode.PREDICT:
        # AUC is area under ROC curve, and is calculated on non-collapsed predictions
        auc = metrics.roc_auc_score(all_labels_np, all_preds_np)
        # Collapse predictions to calculate accuracy and kappa
        all_preds_np[all_preds_np < 0] = 0
        all_preds_np[all_preds_np > 0] = 1
        accuracy = metrics.accuracy_score(all_labels_np, all_preds_np)
        kappa = metrics.cohen_kappa_score(all_labels_np, all_preds_np)
        return total_loss / num_batches, accuracy, 2 * (auc - .5), kappa

def pretrain(model_name: str, options: TrainOptions):
    # import pdb; pdb.set_trace()
    train_data, validation_data = get_data("data/train_data.json", .8)
    train_loader = torch.utils.data.DataLoader(
        Dataset(train_data),
        collate_fn=Collator(),
        batch_size=BATCH_SIZE,
        # num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )
    validation_loader = torch.utils.data.DataLoader(
        Dataset(validation_data),
        collate_fn=Collator(),
        batch_size=len(validation_data),
        # num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )
    model = LSTMModel(Mode.PRE_TRAIN, load_type_mappings(), options).to(device)
    # TODO: look into Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    torch.autograd.set_detect_anomaly(True) # Pause exectuion and get stack trace if something weird happens (ex: NaN grads)
    for epoch in range(5):
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
            if num_batches % 20 == 0:
                print(f"Batch: {num_batches}, Average Loss: {train_loss / num_batches}")

        model.eval() # Set model to evaluation mode
        train_loss, train_accuracy, _, _ = evaluate_model(model, train_loader, Mode.PRE_TRAIN)
        val_loss, val_accuracy, _, _ = evaluate_model(model, validation_loader, Mode.PRE_TRAIN)
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}, "
              f"Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.3f}, Time: {time.time() - start_time:.2f}")
        torch.save(model.state_dict(), f"{model_name}.pt")

def test_pretrain(model_name: str, options: TrainOptions):
    # Load validation data
    test_data = get_data("data/test_data.json")
    test_loader = torch.utils.data.DataLoader(
        Dataset(test_data),
        collate_fn=Collator(),
        batch_size=len(test_data),
        #num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
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
    loss, accuracy, _, _ = evaluate_model(model, test_loader, Mode.PRE_TRAIN)
    print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")

def train(pretrain_model_name: str, model_name: str, options: TrainOptions):
    train_data, validation_data = get_data("data/train_data.json", .8)
    train_labels = get_labels("data/train_labels.json")
    train_loader = torch.utils.data.DataLoader(
        Dataset(train_data, train_labels),
        collate_fn=Collator(),
        batch_size=BATCH_SIZE,
        # num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )
    validation_loader = torch.utils.data.DataLoader(
        Dataset(validation_data, train_labels),
        collate_fn=Collator(),
        batch_size=len(validation_data), # TODO: try using batch size
        # num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
    )
    model = LSTMModel(Mode.PREDICT, load_type_mappings(), options)

    # Copy pretrained parameters based on settings
    states_to_copy = []
    if options.use_pretrained_embeddings:
        states_to_copy += ["question_type_embeddings", "event_type_embeddings"]
    if options.use_pretrained_weights:
        states_to_copy += ["lstm"]
    if states_to_copy:
        state_dict = model.state_dict()
        pretrained_state = torch.load(f"{pretrain_model_name}.pt")
        for high_level_state in states_to_copy:
            for state in state_dict:
                if state.startswith(high_level_state):
                    state_dict[state] = pretrained_state[state]
                    break
        model.load_state_dict(state_dict)

    # Freeze parameters based on settings
    components_to_freeze = []
    if options.freeze_embeddings:
        components_to_freeze += [model.question_type_embeddings, model.event_type_embeddings]
    if options.freeze_model:
        components_to_freeze += [model.lstm]
    for component in components_to_freeze:
        for param in component.parameters():
            param.requires_grad = False

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    torch.autograd.set_detect_anomaly(True) # Pause exectuion and get stack trace if something weird happens (ex: NaN grads)
    best_loss = None
    best_epoch = 0
    for epoch in range(50):
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
            if num_batches % 20 == 0:
                print(f"Batch: {num_batches}, Average Loss: {train_loss / num_batches:.4f}")

        model.eval() # Set model to evaluation mode
        train_loss, train_accuracy, train_auc, _ = evaluate_model(model, train_loader, Mode.PREDICT)
        val_loss, val_accuracy, val_auc, _ = evaluate_model(model, validation_loader, Mode.PREDICT)
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}, Train AUC: {train_auc:.3f}, "
              f"Val Loss: {val_loss:.3f}, Val Accuracy: {val_accuracy:.3f}, Val AUC: {val_auc:.3f}, Time: {time.time() - start_time:.2f}")

        torch.save(model.state_dict(), f"{model_name}.pt")

        # Analyze validation loss curve
        if not best_loss or val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
        # Stop training if we haven't improved in a while
        if epoch - best_epoch >= 10:
            print("Early stopping")
            break

def test_predictor(model_name: str, options: TrainOptions):
    # Load validation data
    test_data = get_data("data/test_data.json")
    test_loader = torch.utils.data.DataLoader(
        Dataset(test_data, get_labels("data/test_labels.json")),
        collate_fn=Collator(),
        batch_size=len(test_data),
        #num_workers=NUM_WORKERS,
        shuffle=True,
        drop_last=True
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
    loss, accuracy, auc, kappa = evaluate_model(model, test_loader, Mode.PREDICT)
    aggregated = auc + kappa
    print(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}, Adj AUC: {auc:.3f}, Kappa: {kappa:.3f}, Agg: {aggregated:.3f}")
